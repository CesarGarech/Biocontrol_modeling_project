# reg_feed_onoff.py - On-Off Substrate Control (Dash Version)
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

PAGE_ID = 'reg_feed_onoff'

#==========================================================================
# MODELO ODE FED-BATCH CON CONTROL ON-OFF (V VARIABLE, VENTANA TIEMPO)
#==========================================================================
def sustrato_onoff_fedbatch_py(t, y, mu_max, Ks, Y_XS, Sin, V0, Fmax, S_min, S_max_ref, t_onoff_start, t_onoff_end):
    """
    Define las ecuaciones diferenciales para Fed-Batch con control On-Off.
    El control On-Off solo está activo entre t_onoff_start y t_onoff_end.
    """
    X, S, V = y
    X = max(1e-9, X); S = max(0.0, S); V = max(1e-6, V)
    mu = mu_max * (S / (Ks + S)) if (Ks + S) > 1e-9 else 0; mu = max(0, mu)

    # Lógica de Control On-Off CON VENTANA DE TIEMPO
    F = 0.0
    if t >= t_onoff_start and t < t_onoff_end:
        if S <= S_min:
            F = Fmax
    F = max(0, F)

    D = F / V
    dXdt = mu * X - D * X
    dSdt = - (mu / Y_XS) * X + D * (Sin - S)
    dVdt = F
    return [dXdt, dSdt, dVdt]

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("Simulation Parameters", className="mb-3"),
        
        # 1. Kinetic and Stoichiometric
        html.H6("1. Kinetic and Stoichiometric", className="mt-3"),
        html.Label("μ_max [1/h]:"),
        dcc.Input(id=f'{PAGE_ID}-mu_max', type='number', value=0.4, min=0.01, max=2.0, step=0.001, className="form-control mb-2"),
        html.Label("Ks [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-ks', type='number', value=0.1, min=0.01, max=5.0, step=0.001, className="form-control mb-2"),
        html.Label("Y_XS [gX/gS]:"),
        dcc.Input(id=f'{PAGE_ID}-yxs', type='number', value=0.5, min=0.1, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("Sin [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-sin', type='number', value=10.0, min=1.0, max=100.0, step=0.1, className="form-control mb-3"),
        
        # 2. Operational
        html.H6("2. Operational", className="mt-3"),
        html.Label("Fmax [L/h]:"),
        dcc.Input(id=f'{PAGE_ID}-fmax', type='number', value=0.2, min=0.01, max=1.0, step=0.001, className="form-control mb-3"),
        
        # 3. On-Off Control
        html.H6("3. On-Off Control", className="mt-3"),
        html.Label("S_min [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-smin', type='number', value=1.5, min=0.1, max=10.0, step=0.01, className="form-control mb-2"),
        html.Label("S_max (Visual Reference) [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-smax', type='number', value=2.5, min=0.1, max=15.0, step=0.01, className="form-control mb-2"),
        html.Label("Start of Control [h]:"),
        dcc.Input(id=f'{PAGE_ID}-t_start', type='number', value=5.0, min=0.0, max=100.0, step=0.5, className="form-control mb-2"),
        html.Label("End of Control [h]:"),
        dcc.Input(id=f'{PAGE_ID}-t_end', type='number', value=8.0, min=0.0, max=100.0, step=0.5, className="form-control mb-3"),
        
        # 4. Initials and Simulation
        html.H6("4. Initials and Simulation", className="mt-3"),
        html.Label("X0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-x0', type='number', value=0.1, min=0.01, max=10.0, step=0.001, className="form-control mb-2"),
        html.Label("S0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-s0', type='number', value=3.0, min=0.0, max=50.0, step=0.1, className="form-control mb-2"),
        html.Label("V0 [L]:"),
        dcc.Input(id=f'{PAGE_ID}-v0', type='number', value=1.0, min=0.1, max=100.0, step=0.1, className="form-control mb-2"),
        html.Label("Final Time [h]:"),
        dcc.Input(id=f'{PAGE_ID}-tfinal', type='number', value=40.0, min=10.0, max=200.0, step=10.0, className="form-control mb-2"),
        html.Label("rtol Solver:"),
        dcc.Input(id=f'{PAGE_ID}-rtol', type='number', value=1e-5, min=1e-7, max=1e-3, step=1e-6, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("atol Solver:"),
        dcc.Input(id=f'{PAGE_ID}-atol', type='number', value=1e-8, min=1e-10, max=1e-5, step=1e-9, className="form-control mb-3", style={'fontSize': '12px'}),
        
        dbc.Button("▶️ Simulate On-Off Control", id=f'{PAGE_ID}-btn-run', color="primary", className="w-100 mt-3")
    ])

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("⛽ On-Off Substrate Control Simulation (Fed-Batch with Window)", className="mb-3"),
        dcc.Markdown("""
        This page simulates a Fed-Batch process with On-Off substrate control active **only during a specified time interval** ($t_{start}$ to $t_{end}$).
        Outside this window, the feed rate ($F$) is zero.

        * **Biological Model:** Monod.
        * **On-Off Control (Time Window):**
            - If $t_{start} \\le t < t_{end}$ AND $S \\le S_{min}$, then $F = F_{max}$.
            - In all other cases, $F = 0$.
        * **Physical Model:** Fed-batch with variable volume.
        
        **Equations:**
        
        $$\\frac{dX}{dt} = \\mu X - D X$$
        
        $$\\frac{dS}{dt} = -\\frac{\\mu}{Y_{XS}} X + D (S_{in} - S)$$
        
        $$\\frac{dV}{dt} = F$$
        
        $$D = F / V$$
        
        $$\\mu = \\mu_{max} \\frac{S}{K_s + S}$$
        
        $$F = \\begin{cases} F_{max} & \\text{if } t_{start} \\le t < t_{end} \\text{ and } S \\le S_{min} \\\\ 0 & \\text{otherwise} \\end{cases}$$
        """, mathjax=True),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("Set the parameters and click 'Simulate On-Off Control' to run the simulation.", color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-output', 'children'),
        Input(f'{PAGE_ID}-btn-run', 'n_clicks'),
        [State(f'{PAGE_ID}-mu_max', 'value'),
         State(f'{PAGE_ID}-ks', 'value'),
         State(f'{PAGE_ID}-yxs', 'value'),
         State(f'{PAGE_ID}-sin', 'value'),
         State(f'{PAGE_ID}-fmax', 'value'),
         State(f'{PAGE_ID}-smin', 'value'),
         State(f'{PAGE_ID}-smax', 'value'),
         State(f'{PAGE_ID}-t_start', 'value'),
         State(f'{PAGE_ID}-t_end', 'value'),
         State(f'{PAGE_ID}-x0', 'value'),
         State(f'{PAGE_ID}-s0', 'value'),
         State(f'{PAGE_ID}-v0', 'value'),
         State(f'{PAGE_ID}-tfinal', 'value'),
         State(f'{PAGE_ID}-rtol', 'value'),
         State(f'{PAGE_ID}-atol', 'value')],
        prevent_initial_call=True
    )
    def simulate_onoff(n_clicks, mu_max, Ks, Y_XS, Sin, Fmax, S_min, S_max, 
                       t_onoff_start, t_onoff_end, X0, S0, V0, t_final, rtol, atol):
        if not n_clicks:
            return dbc.Alert("Click the simulate button to start.", color="info")
        
        try:
            # Validate inputs
            if t_onoff_end <= t_onoff_start:
                return dbc.Alert("Error: End of Control must be greater than Start of Control.", color="danger")
            
            # 1. Preparar simulación
            y0 = [X0, S0, V0]
            t_span = [0, t_final]
            t_eval = np.linspace(t_span[0], t_span[1], int(t_final * 50) + 1)
            ode_args = (mu_max, Ks, Y_XS, Sin, V0, Fmax, S_min, S_max, t_onoff_start, t_onoff_end)

            # 2. Simular usando solve_ivp
            sol = solve_ivp(sustrato_onoff_fedbatch_py, t_span, y0, args=ode_args, 
                           t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol)
            if not sol.success:
                sol = solve_ivp(sustrato_onoff_fedbatch_py, t_span, y0, args=ode_args, 
                               t_eval=t_eval, method='BDF', rtol=rtol, atol=atol)
                if not sol.success:
                    return dbc.Alert(f"Simulation failed: {sol.message}", color="danger")

            t_sim = sol.t
            X_sim = np.maximum(0, sol.y[0, :])
            S_sim = np.maximum(0, sol.y[1, :])
            V_sim = np.maximum(1e-6, sol.y[2, :])

            # 3. Reconstruir Flujo F post-simulación
            F_sim = np.zeros_like(t_sim)
            for i in range(len(t_sim)):
                if t_sim[i] >= t_onoff_start and t_sim[i] < t_onoff_end and S_sim[i] <= S_min:
                    F_sim[i] = Fmax

            # 4. Crear gráficas con Plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Biomass Evolution', 'Substrate Evolution (On-Off Control)', 
                               'Feed Flow Profile (On-Off)', 'Volume Evolution'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Biomass
            fig.add_trace(go.Scatter(x=t_sim, y=X_sim, mode='lines', name='Biomass', 
                                    line=dict(color='blue', width=2)), row=1, col=1)
            fig.add_vline(x=t_onoff_start, line=dict(color='limegreen', dash='dash'), row=1, col=1)
            fig.add_vline(x=t_onoff_end, line=dict(color='tomato', dash='dash'), row=1, col=1)
            
            # Substrate
            fig.add_trace(go.Scatter(x=t_sim, y=S_sim, mode='lines', name='Substrate', 
                                    line=dict(color='red', width=2)), row=1, col=2)
            fig.add_hline(y=S_min, line=dict(color='gray', dash='dash'), annotation_text=f'S_min ({S_min:.2f})', row=1, col=2)
            fig.add_hline(y=S_max, line=dict(color='gray', dash='dot'), annotation_text=f'S_max Ref ({S_max:.2f})', row=1, col=2)
            fig.add_vline(x=t_onoff_start, line=dict(color='limegreen', dash='dash'), row=1, col=2)
            fig.add_vline(x=t_onoff_end, line=dict(color='tomato', dash='dash'), row=1, col=2)
            
            # Flow
            fig.add_trace(go.Scatter(x=t_sim, y=F_sim, mode='lines', name='Flow', 
                                    line=dict(color='black', width=1.5, shape='hv')), row=2, col=1)
            fig.add_vline(x=t_onoff_start, line=dict(color='limegreen', dash='dash'), row=2, col=1)
            fig.add_vline(x=t_onoff_end, line=dict(color='tomato', dash='dash'), row=2, col=1)
            
            # Volume
            fig.add_trace(go.Scatter(x=t_sim, y=V_sim, mode='lines', name='Volume', 
                                    line=dict(color='purple', width=2)), row=2, col=2)
            fig.add_vline(x=t_onoff_start, line=dict(color='limegreen', dash='dash'), row=2, col=1)
            fig.add_vline(x=t_onoff_end, line=dict(color='tomato', dash='dash'), row=2, col=2)

            # Update axes
            fig.update_xaxes(title_text="Time (h)", row=1, col=2)
            fig.update_xaxes(title_text="Time (h)", row=2, col=1)
            fig.update_xaxes(title_text="Time (h)", row=2, col=2)
            fig.update_yaxes(title_text="Biomass (g/L)", row=1, col=1)
            fig.update_yaxes(title_text="Substrate (g/L)", row=1, col=2)
            fig.update_yaxes(title_text="Feed Flow (L/h)", row=2, col=1)
            fig.update_yaxes(title_text="Volume (L)", row=2, col=2)

            fig.update_layout(height=700, showlegend=False, title_text="Simulation Results")

            # 5. Crear tabla de datos
            df_results = pd.DataFrame({
                'Time (h)': t_sim,
                'Biomass (g/L)': X_sim,
                'Substrate (g/L)': S_sim,
                'Volume (L)': V_sim,
                'Flow (L/h)': F_sim
            })

            return html.Div([
                dbc.Alert("Simulation completed successfully!", color="success", className="mb-3"),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.H5("Simulation Data", className="mt-4"),
                html.Div([
                    html.P(f"Showing {min(100, len(df_results))} of {len(df_results)} data points"),
                    dcc.Markdown(df_results.head(100).to_markdown(index=False, floatfmt='.4f'))
                ], style={'maxHeight': '400px', 'overflowY': 'scroll', 'fontSize': '12px'})
            ])
            
        except Exception as e:
            return dbc.Alert(f"An error occurred during simulation: {str(e)}", color="danger")