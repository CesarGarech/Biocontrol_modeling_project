import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PAGE_ID = 'analysis'


def get_params_layout():
    return html.Div([
        html.H5("⚙️ Analysis Configuration", className="text-white"),
        html.Label("Key parameter", className="text-white-50 small"),
        dcc.Dropdown(
            id=f'{PAGE_ID}-dropdown-param',
            options=[{"label": o, "value": o} for o in ["μ_max", "K_s", "Yxs", "Kd"]],
            value="μ_max",
            clearable=False,
            style={"marginBottom": "10px"},
        ),
        html.Label("Percentage variation (% from base)", className="text-white-50 small"),
        dcc.RangeSlider(
            id=f'{PAGE_ID}-slider-rango',
            min=-50, max=200, value=[0, 100], step=1,
            marks={-50: '-50%', 0: '0%', 100: '100%', 200: '200%'},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Label("Number of simulations", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-nsim', min=2, max=50, value=5, step=1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("🔬 Base parameters", className="text-white"),
        html.Label("μ_max base [1/h]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-mumax', type='number', min=0.1, max=2.0, value=0.5, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("K_s base [g/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ks', type='number', min=0.01, max=5.0, value=0.2, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Yxs base [g/g]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-yxs', type='number', min=0.1, max=1.0, value=0.5, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Kd base [1/h]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kd', type='number', min=0.0, max=0.5, value=0.01, step=0.001,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("🔧 Fixed parameters", className="text-white"),
        html.Label("Ypx [g/g]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ypx', type='number', min=0.1, max=1.0, value=0.3, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("kLa [1/h]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kla', type='number', min=0.1, max=100.0, value=20.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Saturated oxygen [mg/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-cs', type='number', min=0.1, max=10.0, value=8.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("O2 maintenance [g/g/h]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-mo', type='number', min=0.0, max=0.5, value=0.05, step=0.001,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("🎚 Initial conditions", className="text-white"),
        html.Label("Initial Biomass [g/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-x0', type='number', min=0.1, max=10.0, value=1.0, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial Substrate [g/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-s0', type='number', min=0.1, max=100.0, value=20.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial Product [g/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-p0', type='number', min=0.0, max=50.0, value=0.0, step=0.01,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Label("Initial O2 [mg/L]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-o0', type='number', min=0.0, max=10.0, value=5.0, step=0.1,
                  style={"width": "100%", "marginBottom": "6px"}),
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("⏳ Simulation Time", className="text-white"),
        html.Label("Duration [h]", className="text-white-50 small"),
        dcc.Slider(id=f'{PAGE_ID}-slider-tfinal', min=1, max=100, value=24, step=1,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Button("🚀 Run Analysis", id=f'{PAGE_ID}-run-btn', n_clicks=0,
                    className='btn btn-success w-100 mt-3'),
    ], style={"padding": "10px"})


def get_content_layout():
    return html.Div([
        html.H2("📈 Sensitivity Analysis - Batch Model"),
        dcc.Graph(id=f'{PAGE_ID}-graph-sim', figure=go.Figure()),
        html.Div(id=f'{PAGE_ID}-results-section', children=[]),
    ])


def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-graph-sim', 'figure'),
        Output(f'{PAGE_ID}-results-section', 'children'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-dropdown-param', 'value'),
        State(f'{PAGE_ID}-slider-rango', 'value'),
        State(f'{PAGE_ID}-slider-nsim', 'value'),
        State(f'{PAGE_ID}-input-mumax', 'value'),
        State(f'{PAGE_ID}-input-ks', 'value'),
        State(f'{PAGE_ID}-input-yxs', 'value'),
        State(f'{PAGE_ID}-input-kd', 'value'),
        State(f'{PAGE_ID}-input-ypx', 'value'),
        State(f'{PAGE_ID}-input-kla', 'value'),
        State(f'{PAGE_ID}-input-cs', 'value'),
        State(f'{PAGE_ID}-input-mo', 'value'),
        State(f'{PAGE_ID}-input-x0', 'value'),
        State(f'{PAGE_ID}-input-s0', 'value'),
        State(f'{PAGE_ID}-input-p0', 'value'),
        State(f'{PAGE_ID}-input-o0', 'value'),
        State(f'{PAGE_ID}-slider-tfinal', 'value'),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, parametro, rango, n_sim,
                     mumax_base, Ks_base, Yxs_base, Kd_base,
                     Ypx, Kla, Cs, mo,
                     X0, S0, P0, O0, t_final):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        y0 = [X0, S0, P0, O0]
        t_eval = np.linspace(0, t_final, 100)
        valores = np.linspace(1 + rango[0] / 100, 1 + rango[1] / 100, n_sim)

        # Build viridis color sequence
        viridis = [
            f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)'
            for r, g, b, _ in [
                __import__('matplotlib').cm.viridis(v) for v in np.linspace(0, 1, n_sim)
            ]
        ]

        def modelo_lote_b(t, y, mumax, Ks, Yxs, Kd, Ypx, Kla, Cs, mo):
            X, S, P, O2 = y
            mu = mumax * S / (Ks + S)
            dXdt = mu * X - Kd * X
            dSdt = (-mu / Yxs) * X
            dPdt = Ypx * mu * X
            dOdt = Kla * (Cs - O2) - (mu / Yxs) * X - mo * X
            return [dXdt, dSdt, dPdt, dOdt]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=('Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]'))
        resultados = []

        for i, factor in enumerate(valores):
            if parametro == "μ_max":
                params = [mumax_base * factor, Ks_base, Yxs_base, Kd_base]
            elif parametro == "K_s":
                params = [mumax_base, Ks_base * factor, Yxs_base, Kd_base]
            elif parametro == "Yxs":
                params = [mumax_base, Ks_base, Yxs_base * factor, Kd_base]
            else:
                params = [mumax_base, Ks_base, Yxs_base, Kd_base * factor]

            sol = solve_ivp(modelo_lote_b, [0, t_final], y0,
                            args=(*params, Ypx, Kla, Cs, mo),
                            t_eval=t_eval)

            resultados.append({
                'Variation (%)': round((factor - 1) * 100, 1),
                'Parameter Value': round(factor, 2),
                'Max Biomass': round(float(sol.y[0].max()), 2),
                'Min Substrate': round(float(sol.y[1].min()), 2),
                'Max Product': round(float(sol.y[2].max()), 2),
                'Peak Time': round(float(sol.t[np.argmax(sol.y[0])]), 1),
            })

            show_legend = (i == 0)
            label = f'Factor {factor:.2f}×'
            fig.add_trace(go.Scatter(x=sol.t, y=sol.y[0], name=label,
                                     line=dict(color=viridis[i]),
                                     showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=sol.t, y=sol.y[1], name=label,
                                     line=dict(color=viridis[i]),
                                     showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=sol.t, y=sol.y[2], name=label,
                                     line=dict(color=viridis[i]),
                                     showlegend=show_legend), row=3, col=1)

        # Colorbar via a dummy scatter trace
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale='viridis',
                cmin=float(valores.min()),
                cmax=float(valores.max()),
                color=[float(valores.min())],
                colorbar=dict(title=f'Factor de {parametro}', thickness=15, x=1.02),
                showscale=True,
            ),
            showlegend=False,
        ))

        fig.update_xaxes(title_text="Time [h]", row=3, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')
        fig.update_layout(
            height=700,
            plot_bgcolor='white',
            title_text=f"Sensitivity Analysis: {parametro}",
        )

        # Quantitative results table
        df = pd.DataFrame(resultados)
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
            ],
        )

        # Global sensitivity bar chart
        sensibilidad = df[['Max Biomass', 'Min Substrate', 'Max Product']].std() / df[['Max Biomass', 'Min Substrate', 'Max Product']].mean().abs()
        fig2 = go.Figure(go.Bar(
            x=list(sensibilidad.index),
            y=list(sensibilidad.values),
            marker_color=['#4c72b0', '#55a868', '#c44e52'],
            text=[f'{v:.3f}' for v in sensibilidad.values],
            textposition='outside',
        ))
        fig2.update_layout(
            title_text="Variation Coefficient (σ/μ)",
            yaxis_title="Relative Sensitivity",
            plot_bgcolor='white',
        )
        fig2.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig2.update_yaxes(showgrid=True, gridcolor='lightgray')

        results_section = html.Div([
            html.H4("📊 Quantitative Results"),
            table,
            html.H4("📐 Global Sensitivity", className="mt-4"),
            dcc.Graph(figure=fig2),
        ])

        return fig, results_section