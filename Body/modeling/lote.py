import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

PAGE_ID = 'lote'

_KINETIC_OPTIONS = ["Simple Monod", "Sigmoidal Monod", "Monod with restrictions"]

_DESC_MONOD = html.Div([
    dcc.Markdown(r"""
## Simple Monod Kinetics
The Simple Monod model describes the relationship between the specific
growth rate of microorganisms (μ) and the substrate concentration (S),
through the following equation:
""", mathjax=True),
    dcc.Markdown(r"$$\mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S}$$", mathjax=True),
    dcc.Markdown(r"""
Where:
- $\mu$ is the specific growth rate (1/h)
- $\mu_{\text{max}}$ is the maximum specific growth rate (1/h)
- $S$ is the substrate concentration (g/L)
- $K_s$ is the saturation constant (g/L)
""", mathjax=True),
])

_DESC_SIGMOIDAL = html.Div([
    dcc.Markdown(r"""
## Sigmoidal Monod Kinetics
The sigmoidal Monod model is an extension of the simple Monod model,
which describes the specific growth rate of microorganisms (μ) as a function
of substrate concentration (S) using a sigmoidal function:
""", mathjax=True),
    dcc.Markdown(r"$$\mu = \mu_{\text{max}} \cdot \frac{S^n}{K_s^n + S^n}$$", mathjax=True),
    dcc.Markdown(r"""
Where:
- $\mu$ is the specific growth rate (1/h)
- $\mu_{\text{max}}$ is the maximum specific growth rate (1/h)
- $S$ is the substrate concentration (g/L)
- $K_s$ is the saturation constant (g/L)
- $n$ is the Hill coefficient
""", mathjax=True),
])

_DESC_COMPLETA = html.Div([
    dcc.Markdown(r"""
## Monod Kinetics with Restrictions
The Monod model with restrictions considers the effect of product (P)
and dissolved oxygen (O2) on the specific growth rate (μ):
""", mathjax=True),
    dcc.Markdown(r"$$\mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S} \cdot \frac{O_2}{K_O + O_2} \cdot \frac{K_P}{K_P + P}$$", mathjax=True),
    dcc.Markdown(r"""
Where:
- $\mu$ is the specific growth rate (1/h)
- $\mu_{\text{max}}$ is the maximum specific growth rate (1/h)
- $S$ is the substrate concentration (g/L)
- $K_s$ is the saturation constant (g/L)
- $P$ is the product concentration (g/L)
- $K_P$ is the product inhibition constant (g/L)
- $O_2$ is the dissolved oxygen concentration (mg/L)
- $K_O$ is the oxygen inhibition constant (mg/L)
""", mathjax=True),
])

_KINETIC_DESC_MAP = {
    "Simple Monod": _DESC_MONOD,
    "Sigmoidal Monod": _DESC_SIGMOIDAL,
    "Monod with restrictions": _DESC_COMPLETA,
}


def get_params_layout():
    return html.Div([
        html.Label("Kinetic model", className="text-white-50 small"),
        dcc.Dropdown(
            id=f'{PAGE_ID}-dropdown-kinetic',
            options=[{"label": o, "value": o} for o in _KINETIC_OPTIONS],
            value=_KINETIC_OPTIONS[0],
            clearable=False,
            style={"marginBottom": "10px"},
        ),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.Label("μ_max", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-mumax', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("K_s", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-ks', min=0.01, max=1.0, value=0.1, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Yxs", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-yxs', min=0.1, max=1.0, value=0.5, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Ypx", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-ypx', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Yxo", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-yxo', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("kLa", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-kla', min=0.1, max=100.0, value=20.0, step=0.1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Saturated Oxygen (Cs)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-cs', min=0.1, max=10.0, value=8.0, step=0.1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Bioreactor Volume (L)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-v', min=0.5, max=10.0, value=2.0, step=0.1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Maintenance (ms)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-ms', min=0.0, max=0.5, value=0.005, step=0.001,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Decay (Kd)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-kd', min=0.0, max=0.5, value=0.005, step=0.001,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("O2 Maintenance (mo)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-mo', min=0.0, max=0.5, value=0.05, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.Label("Initial Biomass (g/L)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-x0', type='number', min=0.1, max=10.0, value=0.5, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial Substrate (g/L)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-s0', type='number', min=0.1, max=100.0, value=20.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial Product (g/L)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-p0', type='number', min=0.0, max=50.0, value=0.0, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial dissolved O2 (mg/L)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-o0', type='number', min=0.0, max=10.0, value=5.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.Label("Final time (h)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-tfinal', min=1, max=100, value=30, step=1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Absolute tolerance (atol)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-atol', type='number', min=1e-10, max=1e-2, value=1e-6, step=1e-8,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Relative tolerance (rtol)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-rtol', type='number', min=1e-10, max=1e-2, value=1e-6, step=1e-8,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Button("▶ Run Simulation", id=f'{PAGE_ID}-run-btn', n_clicks=0,
                    className='btn btn-success w-100 mt-3'),
    ], style={"padding": "10px"})


def get_content_layout():
    return html.Div([
        html.H2("Operation mode: Batch"),
        html.Div(id=f'{PAGE_ID}-kinetics-desc', children=_DESC_MONOD),
        html.Hr(),
        html.H4("Simulation Results"),
        dcc.Markdown("The following graphs show the evolution of biomass (X), substrate (S), "
                     "product (P), and dissolved oxygen (O2) over time."),
        dcc.Graph(id=f'{PAGE_ID}-graph', figure=go.Figure()),
    ])


def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-kinetics-desc', 'children'),
        Input(f'{PAGE_ID}-dropdown-kinetic', 'value'),
    )
    def update_kinetics_desc(tipo_mu):
        return _KINETIC_DESC_MAP.get(tipo_mu, _DESC_MONOD)

    @app.callback(
        Output(f'{PAGE_ID}-graph', 'figure'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-dropdown-kinetic', 'value'),
        State(f'{PAGE_ID}-slider-mumax', 'value'),
        State(f'{PAGE_ID}-slider-ks', 'value'),
        State(f'{PAGE_ID}-slider-yxs', 'value'),
        State(f'{PAGE_ID}-slider-ypx', 'value'),
        State(f'{PAGE_ID}-slider-yxo', 'value'),
        State(f'{PAGE_ID}-slider-kla', 'value'),
        State(f'{PAGE_ID}-slider-cs', 'value'),
        State(f'{PAGE_ID}-slider-ms', 'value'),
        State(f'{PAGE_ID}-slider-kd', 'value'),
        State(f'{PAGE_ID}-slider-mo', 'value'),
        State(f'{PAGE_ID}-input-x0', 'value'),
        State(f'{PAGE_ID}-input-s0', 'value'),
        State(f'{PAGE_ID}-input-p0', 'value'),
        State(f'{PAGE_ID}-input-o0', 'value'),
        State(f'{PAGE_ID}-slider-tfinal', 'value'),
        State(f'{PAGE_ID}-input-atol', 'value'),
        State(f'{PAGE_ID}-input-rtol', 'value'),
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, tipo_mu, mumax, Ks, Yxs, Ypx, Yxo,
                       Kla, Cs, ms, Kd, mo, X0, S0, P0, O0,
                       t_final, atol, rtol):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        atol = atol or 1e-6
        rtol = rtol or 1e-6
        t_eval = np.linspace(0, t_final, 300)

        def modelo_lote(t, y):
            X, S, P, O2 = y
            S = max(S, 0.0)
            if tipo_mu == "Simple Monod":
                mu = mu_monod(S, mumax, Ks)
            elif tipo_mu == "Sigmoidal Monod":
                mu = mu_sigmoidal(S, mumax, Ks, n=2)
            else:
                mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)
            dXdt = mu * X - Kd * X
            dSdt = 0.0 if S <= 0 else (-1.0 / Yxs * mu * X - ms * X)
            dPdt = Ypx * mu * X
            dOdt = Kla * (Cs - O2) - (1.0 / Yxo) * mu * X - mo * X
            return [dXdt, dSdt, dPdt, dOdt]

        y0 = [X0, S0, P0, O0]
        sol = solve_ivp(modelo_lote, [0, t_final], y0,
                        t_eval=t_eval, atol=atol, rtol=rtol)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Biomass (X)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Substrate (S)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Product (P)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Dissolved Oxygen (O2)'))
        fig.update_xaxes(title_text="Time (h)", showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Concentration (g/L or mg/L)", showgrid=True, gridcolor='lightgray')
        fig.update_layout(
            title_text="Batch Simulation Results",
            plot_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig