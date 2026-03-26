import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
try:
    import control
except ImportError:
    control = None

PAGE_ID = 'reg_temp'


def get_params_layout():
    """Parameters layout for sidebar"""
    return html.Div([
        html.H6("1. Process Parameters (Bioreactor)", className="text-white-50 mt-3"),
        html.Label("Process Gain (Kp)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp', type='number', min=0.1, value=2.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Process Time Constant (τp) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-taup', type='number', min=1.0, value=50.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("2. Valve Parameters", className="text-white-50"),
        html.Label("Valve Gain (Kv)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kv', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Valve Time Constant (τv) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-tauv', type='number', min=0.1, value=5.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("3. Sensor Parameters", className="text-white-50"),
        html.Label("Sensor Gain (Ks)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ks', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Sensor Time Constant (τs) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-taus', type='number', min=0.1, value=2.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("4. PID Controller Parameters", className="text-white-50"),
        html.Label("Proportional Gain (Kp_pid)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-pid', type='number', min=0.0, value=5.1, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Integral Gain (Ki_pid)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ki-pid', type='number', min=0.0, value=0.0, step=0.01,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Small("Set 0 for P or PD control", className="text-muted"),
        html.Label("Derivative Gain (Kd_pid)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-input-kd-pid', type='number', min=0.0, value=0.0, step=0.5,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Small("Set 0 for P or PI control", className="text-muted"),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("5. Simulation Configuration", className="text-white-50"),
        html.Label("Final Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-tfinal', type='number', min=50.0, value=500.0, step=50.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Initial Setpoint [°C]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-initial', type='number', value=0.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Final Setpoint [°C]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-final', type='number', value=30.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Setpoint Change Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-step', type='number', min=0.0, value=150.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Button("▶ Simulate", id=f'{PAGE_ID}-run-btn', n_clicks=0,
                    className='btn btn-success w-100 mt-3'),
    ], style={"padding": "10px"})


def get_content_layout():
    """Main content layout"""
    if control is None:
        return html.Div([
            dbc.Alert("The 'python-control' library is not installed. Please install it: pip install control",
                      color="danger")
        ])
    
    return html.Div([
        html.H2("🌡️ Temperature Regulatory Control"),
        dcc.Markdown("""
        This page simulates a closed-loop temperature control system for a bioreactor. The system consists of:
        * **Process (Bioreactor):** Modeled as a first-order system.
        * **Final Control Element (Valve):** Modeled as a first-order system.
        * **Temperature Sensor:** Modeled as a first-order system.
        * **Controller:** A Proportional-Integral-Derivative (PID) controller.
        
        You can modify the parameters of each component and the controller to observe how they affect 
        the system's response to a change in the temperature setpoint.
        """),
        html.Hr(),
        
        html.H4("System Transfer Functions"),
        dbc.Row([
            dbc.Col([
                dcc.Markdown(r"$$G_p(s) = \frac{K_p}{\tau_p s + 1}$$", mathjax=True),
                html.P("Process/Bioreactor", className="text-muted small"),
                dcc.Markdown(r"$$G_v(s) = \frac{K_v}{\tau_v s + 1}$$", mathjax=True),
                html.P("Valve", className="text-muted small"),
            ], md=6),
            dbc.Col([
                dcc.Markdown(r"$$G_s(s) = \frac{K_s}{\tau_s s + 1}$$", mathjax=True),
                html.P("Sensor", className="text-muted small"),
                dcc.Markdown(r"$$C(s) = K_{p_{pid}} + \frac{K_{i_{pid}}}{s} + K_{d_{pid}} s$$", mathjax=True),
                html.P("PID Controller", className="text-muted small"),
            ], md=6),
        ]),
        dcc.Markdown(r"$$G_{total}(s) = G_p(s) G_v(s) G_s(s)$$", mathjax=True),
        html.P("Full Plant in Open-Loop", className="text-muted small"),
        dcc.Markdown(r"$$T(s) = \frac{C(s)G_{total}(s)}{1 + C(s)G_{total}(s)}$$", mathjax=True),
        html.P("Closed-Loop System (with unit feedback after the sensor)", className="text-muted small"),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output-info'),
        dcc.Loading(
            id=f'{PAGE_ID}-loading',
            type="default",
            children=[
                dcc.Graph(id=f'{PAGE_ID}-graph'),
                html.Div(id=f'{PAGE_ID}-table-container')
            ]
        ),
    ])


def register_callbacks(app):
    """Register Dash callbacks"""
    
    @app.callback(
        [Output(f'{PAGE_ID}-output-info', 'children'),
         Output(f'{PAGE_ID}-graph', 'figure'),
         Output(f'{PAGE_ID}-table-container', 'children')],
        [Input(f'{PAGE_ID}-run-btn', 'n_clicks')],
        [State(f'{PAGE_ID}-input-kp', 'value'),
         State(f'{PAGE_ID}-input-taup', 'value'),
         State(f'{PAGE_ID}-input-kv', 'value'),
         State(f'{PAGE_ID}-input-tauv', 'value'),
         State(f'{PAGE_ID}-input-ks', 'value'),
         State(f'{PAGE_ID}-input-taus', 'value'),
         State(f'{PAGE_ID}-input-kp-pid', 'value'),
         State(f'{PAGE_ID}-input-ki-pid', 'value'),
         State(f'{PAGE_ID}-input-kd-pid', 'value'),
         State(f'{PAGE_ID}-input-tfinal', 'value'),
         State(f'{PAGE_ID}-input-sp-initial', 'value'),
         State(f'{PAGE_ID}-input-sp-final', 'value'),
         State(f'{PAGE_ID}-input-t-step', 'value')],
        prevent_initial_call=True
    )
    def simulate_temperature_control(n_clicks, Kp, tau_p, Kv, tau_v, Ks_sens, tau_s,
                                     Kp_pid, Ki_pid, Kd_pid, t_final, sp_initial, sp_final, t_step_value):
        if n_clicks == 0 or control is None:
            return "", {}, ""
        
        try:
            # 1. Define transfer functions
            s = control.tf('s')
            Gp = Kp / (tau_p * s + 1)
            Gv = Kv / (tau_v * s + 1)
            Gs = Ks_sens / (tau_s * s + 1)
            G_total = Gs * Gp * Gv
            
            # 2. Define PID controller
            C_pid = control.tf([Kd_pid, Kp_pid, Ki_pid], [1, 0])
            if Ki_pid == 0 and Kd_pid == 0:
                C_pid = Kp_pid
            
            # 3. Calculate closed loop
            T_cerrado = control.feedback(C_pid * G_total, 1)
            
            # 4. Prepare simulation
            num_points = int(t_final * 5) + 1
            t = np.linspace(0, t_final, num_points)
            setpoint = np.ones_like(t) * sp_initial
            setpoint[t >= t_step_value] = sp_final
            
            # 5. Simulate response
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            
            # 6. Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T, y=setpoint, mode='lines', name='Setpoint (°C)',
                                    line=dict(color='red', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=T, y=yout, mode='lines', name='Controlled Temperature (°C)',
                                    line=dict(color='blue', width=2)))
            
            fig.update_layout(
                title='Bioreactor Temperature Response with PID Control',
                xaxis_title='Time (s)',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            # 7. Create data table
            df_results = pd.DataFrame({
                'Time (s)': T[::50],  # Subsample for display
                'Setpoint (°C)': setpoint[::50],
                'Temperature (°C)': yout[::50]
            })
            
            table = html.Div([
                html.H5("Simulation Data (sampled)", className="mt-4"),
                dash_table.DataTable(
                    data=df_results.round(3).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_results.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '8px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                )
            ])
            
            info = dbc.Alert([
                html.H5("Simulation Complete", className="alert-heading"),
                html.P(f"Controller: C(s) = {str(C_pid)[:100]}..."),
                html.P(f"Closed-Loop: T(s) calculated successfully"),
            ], color="success")
            
            return info, fig, table
            
        except Exception as e:
            error_msg = dbc.Alert([
                html.H5("Simulation Error", className="alert-heading"),
                html.P(str(e)),
            ], color="danger")
            return error_msg, {}, ""