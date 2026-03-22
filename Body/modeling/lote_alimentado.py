import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.integrate import solve_ivp

PAGE_ID = 'fedbatch'


# ---------------------------------------------------------------------------
# Kinetic models
# ---------------------------------------------------------------------------

def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)


def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * (S ** n) / (Ks ** n + S ** n)


def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    S = max(0, S)
    O2 = max(0, O2)
    P = max(0, P)
    inhibition_P = (1 - P / KP) if KP > 0 and P < KP else 0
    inhibition_P = max(0, inhibition_P)
    mu = mumax * (S / (Ks + S)) * (O2 / (KO + O2)) * inhibition_P
    return max(0, mu)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _slider(label, id_suffix, min_val, max_val, default, step=None):
    marks = None
    kwargs = dict(
        id=f'{PAGE_ID}-{id_suffix}',
        min=min_val,
        max=max_val,
        value=default,
        tooltip={"placement": "bottom", "always_visible": False},
        className="mb-3",
    )
    if step is not None:
        kwargs['step'] = step
    return html.Div([
        html.Label(label, className="text-white-50 small"),
        dcc.Slider(**kwargs),
    ])


def _number(label, id_suffix, min_val, max_val, default, step=None):
    kwargs = dict(
        id=f'{PAGE_ID}-{id_suffix}',
        type='number',
        min=min_val,
        max=max_val,
        value=default,
        style={"width": "100%"},
        className="mb-2",
    )
    if step is not None:
        kwargs['step'] = step
    return html.Div([
        html.Label(label, className="text-white-50 small"),
        dcc.Input(**kwargs),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_params_layout():
    return html.Div(
        style={"background": "#2c3e50", "padding": "15px", "borderRadius": "8px"},
        children=[
            html.H5("Model Parameters", className="text-white mb-3"),

            html.Label("Kinetic Model", className="text-white-50 small"),
            dcc.Dropdown(
                id=f'{PAGE_ID}-tipo-mu',
                options=[
                    {"label": "Simple Monod",           "value": "Simple Monod"},
                    {"label": "Sigmoidal Monod",         "value": "Sigmoidal Monod"},
                    {"label": "Monod with restrictions", "value": "Monod with restrictions"},
                ],
                value="Simple Monod",
                clearable=False,
                className="mb-3",
            ),

            _slider("μmax [1/h]",     "mumax", 0.1,  1.0,   0.4),
            _slider("Ks [g/L]",       "Ks",    0.01, 2.0,   0.5),

            # Conditional: Sigmoidal exponent
            html.Div(
                id=f'{PAGE_ID}-sigmoidal-params',
                style={"display": "none"},
                children=[_slider("Sigmoidal exponent (n)", "n", 1, 5, 2, step=1)],
            ),

            # Conditional: O2 / product inhibition constants
            html.Div(
                id=f'{PAGE_ID}-restrictions-params',
                style={"display": "none"},
                children=[
                    _slider("O2 saturation constant KO [mg/L]", "KO", 0.1, 5.0,  0.5),
                    _slider("Product inhibition constant KP [g/L]", "KP", 0.1, 10.0, 5.0),
                ],
            ),

            _slider("Yxs [g/g]",            "Yxs", 0.1,  1.0,   0.6),
            _slider("Ypx [g/g]",            "Ypx", 0.0,  1.0,   0.3),
            _slider("Yxo [g/g]",            "Yxo", 0.1,  1.0,   0.2),
            _slider("kLa [1/h]",            "Kla", 1.0,  200.0, 50.0),
            _slider("Saturated O2 [mg/L]",  "Cs",  5.0,  15.0,  8.0),
            _slider("Substrate in Feed [g/L]", "Sin", 50.0, 300.0, 150.0),
            _slider("S Maintenance [g/g/h]","ms",  0.0,  0.1,   0.001),
            _slider("X Decay [1/h]",        "Kd",  0.0,  0.1,   0.02),
            _slider("O2 Maintenance [g/g/h]","mo", 0.0,  0.1,   0.01),

            html.Hr(style={"borderColor": "#4a6278"}),
            html.H6("Feeding Strategy", className="text-white mb-2"),

            html.Label("Type", className="text-white-50 small"),
            dcc.Dropdown(
                id=f'{PAGE_ID}-estrategia',
                options=[
                    {"label": "Constant",    "value": "Constant"},
                    {"label": "Exponential", "value": "Exponential"},
                    {"label": "Step",        "value": "Step"},
                    {"label": "Linear",      "value": "Linear"},
                ],
                value="Constant",
                clearable=False,
                className="mb-3",
            ),

            _slider("Base Flow (or Initial for Linear) [L/h]", "F-base",         0.01, 5.0,  0.5),
            _slider("Start Feeding [h]",                       "t-alim-inicio",   0.0,  24.0, 2.0,  step=0.5),
            _slider("End Feeding [h]",                         "t-alim-fin",      0.1,  48.0, 24.0, step=0.5),

            # Conditional: Linear final flow
            html.Div(
                id=f'{PAGE_ID}-linear-params',
                style={"display": "none"},
                children=[_slider("Final Flow (Linear) [L/h]", "F-lineal-fin", 0.01, 10.0, 1.0)],
            ),

            html.Hr(style={"borderColor": "#4a6278"}),
            html.H6("Initial Conditions", className="text-white mb-2"),

            _number("Initial Volume [L]",    "V0", 1.0,  100.0, 3.0),
            _number("Initial Biomass [g/L]", "X0", 0.1,  50.0,  1.0),
            _number("Initial Substrate [g/L]","S0", 0.1,  100.0, 30.0),
            _number("Initial Product [g/L]", "P0", 0.0,  50.0,  0.0),
            _number("Initial O2 [mg/L]",     "O0", 0.0,  15.0,  8.0),

            _slider("Simulation Time [h]", "t-final", 10.0, 200.0, 48.0, step=1.0),

            _number("Absolute Tolerance (atol)", "atol", 1e-10, 1e-2, 1e-6, step=1e-8),
            _number("Relative Tolerance (rtol)", "rtol", 1e-10, 1e-2, 1e-6, step=1e-8),

            html.Button(
                "Run Simulation",
                id=f'{PAGE_ID}-run-btn',
                n_clicks=0,
                className='btn btn-success w-100 mt-3',
            ),
        ],
    )


def get_content_layout():
    return html.Div([
        html.H2("Operation Mode: Fedbatch"),
        html.Div(id=f'{PAGE_ID}-kinetics-desc', className="mb-3"),
        dcc.Graph(id=f'{PAGE_ID}-graph', figure=go.Figure()),
        html.Div(id=f'{PAGE_ID}-results-div', className="mt-3"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):

    # 1. Toggle conditional kinetic parameter panels
    @app.callback(
        Output(f'{PAGE_ID}-sigmoidal-params',    'style'),
        Output(f'{PAGE_ID}-restrictions-params', 'style'),
        Input(f'{PAGE_ID}-tipo-mu', 'value'),
    )
    def toggle_kinetic_params(tipo_mu):
        show   = {"display": "block"}
        hidden = {"display": "none"}
        return (
            show   if tipo_mu == "Sigmoidal Monod"         else hidden,
            show   if tipo_mu == "Monod with restrictions" else hidden,
        )

    # 2. Toggle linear feeding parameter panel
    @app.callback(
        Output(f'{PAGE_ID}-linear-params', 'style'),
        Input(f'{PAGE_ID}-estrategia', 'value'),
    )
    def toggle_linear_params(estrategia):
        return {"display": "block"} if estrategia == "Linear" else {"display": "none"}

    # 3. Update kinetics description
    @app.callback(
        Output(f'{PAGE_ID}-kinetics-desc', 'children'),
        Input(f'{PAGE_ID}-tipo-mu', 'value'),
    )
    def update_kinetics_desc(tipo_mu):
        if tipo_mu == "Simple Monod":
            return html.Div([
                html.H4("Simple Monod Kinetics"),
                html.P(
                    "The Simple Monod model describes the relationship between the specific "
                    "growth rate of microorganisms (μ) and the substrate concentration (S):"
                ),
                dcc.Markdown(
                    r"$$\mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S}$$",
                    mathjax=True,
                ),
                dcc.Markdown(
                    "**Where:** μ — specific growth rate (1/h) · μ_max — maximum specific growth rate (1/h) "
                    "· S — substrate concentration (g/L) · K_s — saturation constant (g/L)"
                ),
            ])
        elif tipo_mu == "Sigmoidal Monod":
            return html.Div([
                html.H4("Sigmoidal Monod Kinetics"),
                html.P(
                    "An extension of the simple Monod model using a sigmoidal (Hill) function:"
                ),
                dcc.Markdown(
                    r"$$\mu = \mu_{\text{max}} \cdot \frac{S^n}{K_s^n + S^n}$$",
                    mathjax=True,
                ),
                dcc.Markdown(
                    "**Where:** μ — specific growth rate (1/h) · μ_max — maximum specific growth rate (1/h) "
                    "· S — substrate (g/L) · K_s — saturation constant (g/L) · n — Hill coefficient"
                ),
            ])
        else:  # Monod with restrictions
            return html.Div([
                html.H4("Monod Kinetics with Restrictions"),
                html.P(
                    "Considers the combined effect of substrate, dissolved oxygen, and product inhibition:"
                ),
                dcc.Markdown(
                    r"$$\mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S} \cdot \frac{O_2}{K_O + O_2} "
                    r"\cdot \left(1 - \frac{P}{K_P}\right), \quad P < K_P$$",
                    mathjax=True,
                ),
                dcc.Markdown(
                    "**Where:** S — substrate (g/L) · K_s — saturation constant (g/L) · "
                    "O₂ — dissolved oxygen (mg/L) · K_O — oxygen saturation constant (mg/L) · "
                    "P — product (g/L) · K_P — product inhibition constant (g/L)"
                ),
            ])

    # 4. Main simulation callback
    @app.callback(
        Output(f'{PAGE_ID}-graph',       'figure'),
        Output(f'{PAGE_ID}-results-div', 'children'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-tipo-mu',      'value'),
        State(f'{PAGE_ID}-mumax',        'value'),
        State(f'{PAGE_ID}-Ks',           'value'),
        State(f'{PAGE_ID}-n',            'value'),
        State(f'{PAGE_ID}-KO',           'value'),
        State(f'{PAGE_ID}-KP',           'value'),
        State(f'{PAGE_ID}-Yxs',         'value'),
        State(f'{PAGE_ID}-Ypx',         'value'),
        State(f'{PAGE_ID}-Yxo',         'value'),
        State(f'{PAGE_ID}-Kla',         'value'),
        State(f'{PAGE_ID}-Cs',          'value'),
        State(f'{PAGE_ID}-Sin',         'value'),
        State(f'{PAGE_ID}-ms',          'value'),
        State(f'{PAGE_ID}-Kd',          'value'),
        State(f'{PAGE_ID}-mo',          'value'),
        State(f'{PAGE_ID}-estrategia',  'value'),
        State(f'{PAGE_ID}-F-base',      'value'),
        State(f'{PAGE_ID}-t-alim-inicio','value'),
        State(f'{PAGE_ID}-t-alim-fin',  'value'),
        State(f'{PAGE_ID}-F-lineal-fin','value'),
        State(f'{PAGE_ID}-V0',          'value'),
        State(f'{PAGE_ID}-X0',          'value'),
        State(f'{PAGE_ID}-S0',          'value'),
        State(f'{PAGE_ID}-P0',          'value'),
        State(f'{PAGE_ID}-O0',          'value'),
        State(f'{PAGE_ID}-t-final',     'value'),
        State(f'{PAGE_ID}-atol',        'value'),
        State(f'{PAGE_ID}-rtol',        'value'),
        prevent_initial_call=True,
    )
    def run_simulation(
        n_clicks,
        tipo_mu, mumax, Ks, n, KO, KP,
        Yxs, Ypx, Yxo, Kla, Cs, Sin, ms, Kd, mo,
        estrategia, F_base, t_alim_inicio, t_alim_fin, F_lineal_fin,
        V0, X0, S0, P0, O0, t_final, atol, rtol,
    ):
        # --- Safe defaults for optional states ---
        mumax         = mumax         or 0.4
        Ks            = Ks            or 0.5
        n             = n             or 2
        KO            = KO            or 0.5
        KP            = KP            or 5.0
        Yxs           = Yxs           or 0.6
        Ypx           = Ypx           or 0.3
        Yxo           = Yxo           or 0.2
        Kla           = Kla           or 50.0
        Cs            = Cs            or 8.0
        Sin           = Sin           or 150.0
        ms            = ms            if ms  is not None else 0.001
        Kd            = Kd            if Kd  is not None else 0.02
        mo            = mo            if mo  is not None else 0.01
        F_base        = F_base        or 0.5
        t_alim_inicio = t_alim_inicio if t_alim_inicio is not None else 2.0
        t_alim_fin    = t_alim_fin    if t_alim_fin    is not None else 24.0
        F_lineal_fin  = F_lineal_fin  if F_lineal_fin  is not None else F_base
        V0            = V0            or 3.0
        X0            = X0            or 1.0
        S0            = S0            or 30.0
        P0            = P0            if P0 is not None else 0.0
        O0            = O0            if O0 is not None else Cs
        t_final       = t_final       or 48.0
        atol          = atol          if atol is not None else 1e-6
        rtol          = rtol          if rtol is not None else 1e-6

        def calcular_flujo(t):
            if t_alim_inicio <= t <= t_alim_fin:
                if estrategia == "Constant":
                    return F_base
                elif estrategia == "Exponential":
                    exponent = 0.15 * (t - t_alim_inicio)
                    try:
                        return min(F_base * np.exp(exponent), F_base * 1000)
                    except OverflowError:
                        return F_base * 1000
                elif estrategia == "Step":
                    t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2
                    return F_base * 2 if t > t_medio else F_base
                elif estrategia == "Linear":
                    delta_t = t_alim_fin - t_alim_inicio
                    if delta_t > 0:
                        slope = (F_lineal_fin - F_base) / delta_t
                        return F_base + slope * (t - t_alim_inicio)
                    return F_base
            return 0.0

        def modelo_fedbatch(t, y):
            X, S, P, O2, V = y
            X  = max(0, X);  S  = max(0, S)
            P  = max(0, P);  O2 = max(0, O2)
            V  = max(1e-6, V)

            if tipo_mu == "Simple Monod":
                mu = mu_monod(S, mumax, Ks)
            elif tipo_mu == "Sigmoidal Monod":
                mu = mu_sigmoidal(S, mumax, Ks, n)
            elif tipo_mu == "Monod with restrictions":
                mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
            else:
                mu = 0
            mu = max(0, mu)

            F = calcular_flujo(t)
            dXdt = (mu - Kd) * X - (F / V) * X
            dSdt = -(mu / Yxs + ms) * X + (F / V) * (Sin - S)
            dPdt = Ypx * mu * X - (F / V) * P
            consumo_o2 = (mu / Yxo + mo) * X
            dOdt = Kla * (Cs - O2) - consumo_o2 - (F / V) * O2
            dVdt = F
            return [dXdt, dSdt, dPdt, dOdt, dVdt]

        y0     = [X0, S0, P0, O0, V0]
        t_span = [0, t_final]
        t_eval = np.linspace(t_span[0], t_span[1], 500)

        sol = solve_ivp(
            modelo_fedbatch, t_span, y0,
            t_eval=t_eval, method='RK45', atol=atol, rtol=rtol,
        )

        if not sol.success:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Integration failed: {sol.message}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color="red"),
            )
            error_fig.update_layout(paper_bgcolor="white")
            return error_fig, dbc.Alert(f"Integration failed: {sol.message}", color="danger")

        flujo_sim = np.array([calcular_flujo(t) for t in sol.t])

        # --- Build subplot figure ---
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2, "secondary_y": True}, None],
                [{}, {}],
                [{}, {}],
            ],
            subplot_titles=(
                "Feeding & Volume Profile",
                "",
                "Biomass (X) [g/L]", "Substrate (S) [g/L]",
                "Product (P) [g/L]", "Dissolved O\u2082 [mg/L]",
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        # Row 1: Feed flow (primary y) + Volume (secondary y)
        fig.add_trace(
            go.Scatter(x=sol.t, y=flujo_sim, name="Feed Flow [L/h]",
                       line=dict(color="red")),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=sol.t, y=sol.y[4], name="Volume [L]",
                       line=dict(color="blue", dash="dash")),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="Flow [L/h]",   row=1, col=1, secondary_y=False,
                         color="red",  gridcolor="lightgray")
        fig.update_yaxes(title_text="Volume [L]",   row=1, col=1, secondary_y=True,
                         color="blue", gridcolor="lightgray")

        # Row 2: Biomass / Substrate
        fig.add_trace(
            go.Scatter(x=sol.t, y=sol.y[0], name="Biomass X [g/L]",
                       line=dict(color="green")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sol.t, y=sol.y[1], name="Substrate S [g/L]",
                       line=dict(color="magenta")),
            row=2, col=2,
        )

        # Row 3: Product / O2
        fig.add_trace(
            go.Scatter(x=sol.t, y=sol.y[2], name="Product P [g/L]",
                       line=dict(color="black")),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sol.t, y=sol.y[3], name="O\u2082 [mg/L]",
                       line=dict(color="cyan")),
            row=3, col=2,
        )

        fig.update_xaxes(title_text="Time [h]", gridcolor="lightgray")
        fig.update_yaxes(gridcolor="lightgray")
        fig.update_layout(
            height=850,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.08),
        )

        # --- Build results metrics ---
        t_end   = sol.t[-1]
        Xf      = sol.y[0, -1]
        Sf      = sol.y[1, -1]
        Pf      = sol.y[2, -1]
        Vf      = sol.y[4, -1]

        prod_P  = Pf / t_end if t_end > 0 else 0
        prod_X  = Xf / t_end if t_end > 0 else 0
        denom   = S0 * V0 + np.trapz(flujo_sim * Sin, sol.t) - Sf * Vf
        yield_PS = (Pf * Vf - P0 * V0) / denom if denom > 1e-6 else 0

        def _metric(label, value):
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.P(label, className="text-muted small mb-1"),
                        html.H5(value, className="mb-0 fw-bold"),
                    ]),
                    className="text-center shadow-sm",
                ),
                xs=12, sm=6, md=4, className="mb-3",
            )

        results_children = html.Div([
            html.H5(f"Final Results (t = {t_end:.1f} h)", className="mt-2 mb-3"),
            dbc.Row([
                _metric("Final Volume [L]",            f"{Vf:.2f}"),
                _metric("Final Biomass [g/L]",         f"{Xf:.2f}"),
                _metric("Final Product [g/L]",         f"{Pf:.2f}"),
                _metric("Productivity Vol. P [g/L/h]", f"{prod_P:.3f}"),
                _metric("Productivity Vol. X [g/L/h]", f"{prod_X:.3f}"),
                _metric("Total P/S Yield [g/g]",       f"{yield_PS:.3f}"),
            ]),
        ])

        return fig, results_children