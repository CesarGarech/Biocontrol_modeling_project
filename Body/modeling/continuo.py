import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

PAGE_ID = 'continuo'

_KINETIC_MODELS = ["Simple Monod", "Sigmoidal Monod", "Monod with restrictions"]


def get_params_layout():
    return html.Div([
        html.H6("Model Parameters", className="text-white fw-bold mb-3"),

        html.Label("Kinetic model", className="text-white-50 small"),
        dcc.Dropdown(
            id=f'{PAGE_ID}-tipo-mu',
            options=[{"label": o, "value": o} for o in _KINETIC_MODELS],
            value=_KINETIC_MODELS[0],
            clearable=False,
            className="mb-2",
        ),

        html.Label("μmax", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-mumax', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Ks", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-ks', min=0.01, max=1.0, value=0.1, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Yxs", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-yxs', min=0.1, max=1.0, value=0.5, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Ypx", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-ypx', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Yxo", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-yxo', min=0.1, max=1.0, value=0.3, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("kLa", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-kla', min=0.1, max=100.0, value=20.0, step=0.1,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Saturated Oxygen (Cs)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-cs', min=0.1, max=10.0, value=8.0, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Maintenance (ms)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-ms', min=0.0, max=0.5, value=0.005, step=0.001,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Decay (Kd)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-kd', min=0.0, max=0.5, value=0.005, step=0.001,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("O2 Maintenance (mo)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-mo', min=0.0, max=0.5, value=0.05, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Substrate in Feed (Sin)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-sin', min=0.0, max=100.0, value=50.0, step=0.1,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Dilution Rate D (1/h)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-d', min=0.0, max=1.0, value=0.01, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Hr(className="border-secondary"),
        html.Label("Initial Biomass (g/L)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-x0', type='number', min=0.1, max=10.0, value=0.5,
                  step=0.01, style={"width": "100%"}),

        html.Label("Initial Substrate (g/L)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-s0', type='number', min=0.1, max=100.0, value=20.0,
                  step=0.01, style={"width": "100%"}),

        html.Label("Initial Product (g/L)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-p0', type='number', min=0.0, max=50.0, value=0.0,
                  step=0.01, style={"width": "100%"}),

        html.Label("Initial dissolved O2 (mg/L)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-o0', type='number', min=0.0, max=10.0, value=5.0,
                  step=0.01, style={"width": "100%"}),

        html.Hr(className="border-secondary"),
        html.Label("n value (Sigmoidal Monod)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-n-sigmoidal', min=0.0, max=5.0, value=2.0, step=0.01,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Final time (h)", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-t-final', min=1, max=100, value=30, step=1,
                   tooltip={"placement": "bottom", "always_visible": True}),

        html.Label("Absolute tolerance (atol)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-atol', type='number', min=1e-10, max=1e-2, value=1e-6,
                  step=1e-7, style={"width": "100%"}),

        html.Label("Relative tolerance (rtol)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-rtol', type='number', min=1e-10, max=1e-2, value=1e-6,
                  step=1e-7, style={"width": "100%"}),

        html.Button("Run Simulation", id=f'{PAGE_ID}-run-btn', n_clicks=0,
                    className='btn btn-success w-100 mt-3'),
    ])


def get_content_layout():
    return html.Div([
        html.H2("Operation mode: Continuous (Chemostat)"),
        html.H4("Simulation Results"),
        dcc.Graph(id=f'{PAGE_ID}-graph', figure=go.Figure()),
        html.Div(id=f'{PAGE_ID}-results-div'),
    ])


def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-graph', 'figure'),
        Output(f'{PAGE_ID}-results-div', 'children'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-tipo-mu', 'value'),
        State(f'{PAGE_ID}-mumax', 'value'),
        State(f'{PAGE_ID}-ks', 'value'),
        State(f'{PAGE_ID}-yxs', 'value'),
        State(f'{PAGE_ID}-ypx', 'value'),
        State(f'{PAGE_ID}-yxo', 'value'),
        State(f'{PAGE_ID}-kla', 'value'),
        State(f'{PAGE_ID}-cs', 'value'),
        State(f'{PAGE_ID}-ms', 'value'),
        State(f'{PAGE_ID}-kd', 'value'),
        State(f'{PAGE_ID}-mo', 'value'),
        State(f'{PAGE_ID}-sin', 'value'),
        State(f'{PAGE_ID}-d', 'value'),
        State(f'{PAGE_ID}-x0', 'value'),
        State(f'{PAGE_ID}-s0', 'value'),
        State(f'{PAGE_ID}-p0', 'value'),
        State(f'{PAGE_ID}-o0', 'value'),
        State(f'{PAGE_ID}-n-sigmoidal', 'value'),
        State(f'{PAGE_ID}-t-final', 'value'),
        State(f'{PAGE_ID}-atol', 'value'),
        State(f'{PAGE_ID}-rtol', 'value'),
        prevent_initial_call=True,
    )
    def update_simulation(
        n_clicks,
        tipo_mu, mumax, Ks, Yxs, Ypx, Yxo, Kla, Cs, ms, Kd, mo, Sin, D,
        X0, S0, P0, O0, n_sigmoidal, t_final, atol, rtol,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        # Fall back to defaults for any missing numeric inputs
        mumax = mumax or 0.3
        Ks = Ks or 0.1
        Yxs = Yxs or 0.5
        Ypx = Ypx or 0.3
        Yxo = Yxo or 0.3
        Kla = Kla or 20.0
        Cs = Cs or 8.0
        ms = ms if ms is not None else 0.005
        Kd = Kd if Kd is not None else 0.005
        mo = mo if mo is not None else 0.05
        Sin = Sin if Sin is not None else 50.0
        D = D if D is not None else 0.01
        X0 = X0 or 0.5
        S0 = S0 or 20.0
        P0 = P0 if P0 is not None else 0.0
        O0 = O0 if O0 is not None else 5.0
        n_sigmoidal = n_sigmoidal or 2.0
        t_final = t_final or 30
        atol = atol or 1e-6
        rtol = rtol or 1e-6

        t_eval = np.linspace(0, t_final, 300)

        def modelo_continuo(t, y):
            X, S, P, O2 = y
            if tipo_mu == "Simple Monod":
                mu = mu_monod(S, mumax, Ks)
            elif tipo_mu == "Sigmoidal Monod":
                mu = mu_sigmoidal(S, mumax, Ks, n=n_sigmoidal)
            else:  # "Monod with restrictions"
                mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)

            dXdt = mu * X - Kd * X - D * X
            dSdt = -1 / Yxs * mu * X - ms * X + D * (Sin - S)
            dPdt = Ypx * mu * X - D * P
            dOdt = Kla * (Cs - O2) - 1 / Yxo * mu * X - mo * X - D * O2
            return [dXdt, dSdt, dPdt, dOdt]

        y0 = [X0, S0, P0, O0]
        sol = solve_ivp(modelo_continuo, [0, t_final], y0, t_eval=t_eval,
                        atol=atol, rtol=rtol)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name='Biomass (X)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name='Substrate (S)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name='Product (P)'))
        fig.add_trace(go.Scatter(x=sol.t, y=sol.y[3], name='Dissolved Oxygen (O2)'))
        fig.update_layout(
            xaxis_title="Time (h)",
            yaxis_title="Concentration (g/L or mg/L)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="lightgray"),
            yaxis=dict(gridcolor="lightgray"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig, html.Div()