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

PAGE_ID = 'reg_oxigeno'


def get_params_layout():
    """Parameters layout for sidebar"""
    return html.Div([
        html.H6("1. Process Parameters", className="text-white-50 mt-3"),
        html.Label("Process Gain (Kp_proc) [%DO/RPM unit]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-proc', type='number', min=0.001, value=0.05, step=0.005,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Process Time Constant (T_proc) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-proc', type='number', min=1.0, value=10.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("2. Actuator Parameters", className="text-white-50"),
        html.Label("Actuator Gain (Kp_valv)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-valv', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Actuator Time Constant (T_valv) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-valv', type='number', min=0.1, value=2.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("3. Sensor Parameters", className="text-white-50"),
        html.Label("Sensor Gain (Kp_sensor)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-sensor', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Sensor Time Constant (T_sensor) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-sensor', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("4. PID Controller Parameters", className="text-white-50"),
        html.Label("Proportional Gain (Kp)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-pid', type='number', min=0.0, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Integral Gain (Ki)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ki-pid', type='number', min=0.0, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Derivative Gain (Kd)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kd-pid', type='number', min=0.0, value=0.1, step=0.01,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("5. Simulation Configuration", className="text-white-50"),
        html.Label("Final Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-tfinal', type='number', min=50.0, value=500.0, step=50.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Initial Setpoint [% DO]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-initial', type='number', min=0.0, max=100.0, value=10.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Final Setpoint [% DO]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-final', type='number', min=0.0, max=100.0, value=30.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Setpoint Change Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-step', type='number', min=0.0, value=250.0, step=10.0,
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
        html.H2("💨 Dissolved Oxygen Regulatory Control"),
        dcc.Markdown("""
        This page simulates a closed-loop Dissolved Oxygen (DO) control system.
        The manipulated variable used to control DO is the **stirring speed (RPM)**, 
        whose effect is modeled through the process transfer function.
        
        * **Process:** Simplified relationship between the stirring signal and DO.
        * **Actuator:** Dynamics associated with stirring changes (motor).
        * **Sensor:** Dynamics of the DO sensor.
        * **PID Controller:** Calculates the signal for the stirring actuator.
        """),
        html.Hr(),
        
        html.H4("System Transfer Functions"),
        dbc.Row([
            dbc.Col([
                dcc.Markdown(r"$$G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}$$", mathjax=True),
                html.P("Process (DO / Stirring Signal)", className="text-muted small"),
                dcc.Markdown(r"$$G_{valv}(s) = \frac{K_{p_{valv}}}{T_{valv} s + 1}$$", mathjax=True),
                html.P("Stirring Actuator", className="text-muted small"),
            ], md=6),
            dbc.Col([
                dcc.Markdown(r"$$G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}$$", mathjax=True),
                html.P("DO Sensor", className="text-muted small"),
                dcc.Markdown(r"$$C(s) = K_p + \frac{K_i}{s} + K_d s$$", mathjax=True),
                html.P("PID Controller", className="text-muted small"),
            ], md=6),
        ]),
        dcc.Markdown(r"$$G_{open}(s) = G_{proc}(s) G_{valv}(s) G_{sensor}(s)$$", mathjax=True),
        dcc.Markdown(r"$$T(s) = \frac{C(s)G_{open}(s)}{1 + C(s)G_{open}(s)}$$", mathjax=True),
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
        [State(f'{PAGE_ID}-input-kp-proc', 'value'),
         State(f'{PAGE_ID}-input-t-proc', 'value'),
         State(f'{PAGE_ID}-input-kp-valv', 'value'),
         State(f'{PAGE_ID}-input-t-valv', 'value'),
         State(f'{PAGE_ID}-input-kp-sensor', 'value'),
         State(f'{PAGE_ID}-input-t-sensor', 'value'),
         State(f'{PAGE_ID}-input-kp-pid', 'value'),
         State(f'{PAGE_ID}-input-ki-pid', 'value'),
         State(f'{PAGE_ID}-input-kd-pid', 'value'),
         State(f'{PAGE_ID}-input-tfinal', 'value'),
         State(f'{PAGE_ID}-input-sp-initial', 'value'),
         State(f'{PAGE_ID}-input-sp-final', 'value'),
         State(f'{PAGE_ID}-input-t-step', 'value')],
        prevent_initial_call=True
    )
    def simulate_oxygen_control(n_clicks, Kp_proc, T_proc, Kp_valv, T_valv, Kp_sensor, T_sensor,
                                Kp_pid, Ki_pid, Kd_pid, t_final, sp_initial, sp_final, t_step_value):
        if n_clicks == 0 or control is None:
            return "", {}, ""
        
        try:
            # 1. Define transfer functions
            s = control.tf('s')
            G_proc = Kp_proc / (T_proc * s + 1)
            G_valv = Kp_valv / (T_valv * s + 1)
            G_sensor = Kp_sensor / (T_sensor * s + 1)
            G_open = G_proc * G_valv * G_sensor
            
            # 2. Define PID controller
            C_pid = control.tf([Kd_pid, Kp_pid, Ki_pid], [1, 0])
            if Ki_pid == 0 and Kd_pid == 0:
                C_pid = Kp_pid
            
            # 3. Calculate closed loop
            T_cerrado = control.feedback(C_pid * G_open, 1)
            
            # 4. Prepare simulation
            num_points = int(t_final * 5) + 1
            t = np.linspace(0, t_final, num_points)
            setpoint = np.ones_like(t) * sp_initial
            setpoint[t >= t_step_value] = sp_final
            
            # 5. Simulate response
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            
            # 6. Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T, y=setpoint, mode='lines', name='Setpoint (% DO)',
                                    line=dict(color='red', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=T, y=yout, mode='lines', name='Controlled Oxygen (% DO)',
                                    line=dict(color='blue', width=2)))
            
            fig.update_layout(
                title='Dissolved Oxygen PID Control Response',
                xaxis_title='Time (s)',
                yaxis_title='Dissolved Oxygen (% Saturation)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            # 7. Create data table
            df_results = pd.DataFrame({
                'Time (s)': T[::50],
                'Setpoint (%DO)': setpoint[::50],
                'Oxygen (%DO)': yout[::50]
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
            
            info = dbc.Alert("Simulation completed successfully!", color="success")
            
            return info, fig, table
            
        except Exception as e:
            error_msg = dbc.Alert([
                html.H5("Simulation Error", className="alert-heading"),
                html.P(str(e)),
            ], color="danger")
            return error_msg, {}, ""
