import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
try:
    import control
except ImportError:
    control = None

PAGE_ID = 'reg_ph'



def get_params_layout():
    """Parameters layout for sidebar"""
    return html.Div([
        html.H6("1. Process Parameters", className="text-white-50 mt-3"),
        html.Label("Process Gain (Kp_proc) [pH/(flow unit)]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-proc', type='number', min=0.01, value=0.1, step=0.01,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Process Time Constant (T_proc) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-proc', type='number', min=1.0, value=20.0, step=1.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("2. Pumps Parameters", className="text-white-50"),
        html.Label("Acid Pump Gain (Kp_acid)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-acid', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Acid Pump Time Constant (T_acid) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-acid', type='number', min=0.1, value=5.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Base Pump Gain (Kp_base)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-input-kp-base', type='number', min=0.1, value=1.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Base Pump Time Constant (T_base) [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-base', type='number', min=0.1, value=5.0, step=0.1,
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
        dcc.Input(id=f'{PAGE_ID}-input-kp-pid', type='number', min=0.0, value=2.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Integral Gain (Ki)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-ki-pid', type='number', min=0.0, value=1.2, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Small("Set 0 for P or PD control", className="text-muted"),
        html.Label("Derivative Gain (Kd)", className="text-white-50 small mt-2"),
        dcc.Input(id=f'{PAGE_ID}-input-kd-pid', type='number', min=0.0, value=0.0, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Small("Set 0 for P or PI control", className="text-muted"),
        
        html.Hr(style={"borderColor": "#4a6278"}),
        html.H6("5. Simulation Configuration", className="text-white-50"),
        html.Label("Final Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-tfinal', type='number', min=100.0, value=1000.0, step=50.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Initial Setpoint [pH]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-initial', type='number', min=0.0, max=14.0, value=8.1, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Final Setpoint [pH]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-sp-final', type='number', min=0.0, max=14.0, value=4.5, step=0.1,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Setpoint Change Time [s]", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-t-step', type='number', min=0.0, value=450.0, step=10.0,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Label("Initial Offset pH (vs initial SP)", className="text-white-50 small"),
        dcc.Input(id=f'{PAGE_ID}-input-y0-offset', type='number', value=0.1, step=0.05,
                  style={"width": "100%", "marginBottom": "8px"}),
        html.Small("Positive activates base, negative activates acid", className="text-muted"),
        
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
        html.H2("💧 pH Regulatory Control (Split-Range)"),
        dcc.Markdown("""
        This page simulates a closed-loop pH control system for a bioreactor using a **split-range** strategy.
        Two final control elements (acid and base pumps) are operated by a single PID controller.
        
        * **Process:** The relationship between acid/base flow and reactor pH.
        * **Pumps (Acid/Base):** The dynamics of each dosing pump.
        * **Sensor:** The dynamics of the pH sensor.
        * **PID Controller:** Calculates the necessary control action.
        * **Split-Range Logic:** If the measured pH is above the setpoint, the acid pump is activated. 
          If it is below, the base pump is activated.
        """),
        html.Hr(),
        
        html.H4("System Transfer Functions"),
        dbc.Row([
            dbc.Col([
                dcc.Markdown(r"$$G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}$$", mathjax=True),
                html.P("Process (pH vs Flow)", className="text-muted small"),
                dcc.Markdown(r"$$G_{acid}(s) = \frac{K_{p_{acid}}}{T_{acid} s + 1}$$", mathjax=True),
                html.P("Acid Pump", className="text-muted small"),
                dcc.Markdown(r"$$G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}$$", mathjax=True),
                html.P("pH Sensor", className="text-muted small"),
            ], md=6),
            dbc.Col([
                dcc.Markdown(r"$$G_{base}(s) = \frac{K_{p_{base}}}{T_{base} s + 1}$$", mathjax=True),
                html.P("Base Pump", className="text-muted small"),
                dcc.Markdown(r"$$C(s) = K_p + \frac{K_i}{s} + K_d s$$", mathjax=True),
                html.P("PID Controller", className="text-muted small"),
            ], md=6),
        ]),
        html.P("Open-loops (simplified for decoupled simulation):"),
        dcc.Markdown(r"$$G_{open, acid}(s) = G_{proc}(s) G_{acid}(s) G_{sensor}(s)$$", mathjax=True),
        dcc.Markdown(r"$$G_{open, base}(s) = G_{proc}(s) G_{base}(s) G_{sensor}(s)$$", mathjax=True),
        html.P("Note: The closed-loops for acid and base are simulated independently and then combined according to the split-range logic.", 
               className="text-muted small"),
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
         State(f'{PAGE_ID}-input-kp-acid', 'value'),
         State(f'{PAGE_ID}-input-t-acid', 'value'),
         State(f'{PAGE_ID}-input-kp-base', 'value'),
         State(f'{PAGE_ID}-input-t-base', 'value'),
         State(f'{PAGE_ID}-input-kp-sensor', 'value'),
         State(f'{PAGE_ID}-input-t-sensor', 'value'),
         State(f'{PAGE_ID}-input-kp-pid', 'value'),
         State(f'{PAGE_ID}-input-ki-pid', 'value'),
         State(f'{PAGE_ID}-input-kd-pid', 'value'),
         State(f'{PAGE_ID}-input-tfinal', 'value'),
         State(f'{PAGE_ID}-input-sp-initial', 'value'),
         State(f'{PAGE_ID}-input-sp-final', 'value'),
         State(f'{PAGE_ID}-input-t-step', 'value'),
         State(f'{PAGE_ID}-input-y0-offset', 'value')],
        prevent_initial_call=True
    )
    def simulate_ph_control(n_clicks, Kp_proc, T_proc, Kp_acid, T_acid, Kp_base, T_base,
                           Kp_sensor, T_sensor, Kp_pid, Ki_pid, Kd_pid, t_final_ph, 
                           sp_initial_ph, sp_final_ph, t_step_ph_value, y0_offset):
        if n_clicks == 0 or control is None:
            return "", {}, ""
        
        try:
            # 1. Define transfer functions
            s = control.tf('s')
            G_proc = abs(Kp_proc) / (T_proc * s + 1)
            G_acid = Kp_acid / (T_acid * s + 1)
            G_base = Kp_base / (T_base * s + 1)
            G_sensor = Kp_sensor / (T_sensor * s + 1)
            
            # 2. Define PID controller
            C_pid = control.tf([Kd_pid, Kp_pid, Ki_pid], [1, 0])
            if Ki_pid == 0 and Kd_pid == 0:
                C_pid = Kp_pid
            
            # 3. Open and closed loops
            G_open_acid = G_proc * G_acid * G_sensor
            G_open_base = G_proc * G_base * G_sensor
            G_closed_acid = control.feedback(C_pid * G_open_acid, 1)
            G_closed_base = control.feedback(C_pid * G_open_base, 1)
            
            # 4. Prepare simulation
            num_points_ph = int(t_final_ph * 2) + 1
            t = np.linspace(0, t_final_ph, num_points_ph)
            setpoint = np.ones_like(t) * sp_initial_ph
            setpoint[t >= t_step_ph_value] = sp_final_ph
            
            # 5. Simulate independent responses
            T_sim, y_acid_resp = control.forced_response(G_closed_acid, T=t, U=setpoint)
            _, y_base_resp = control.forced_response(G_closed_base, T=t, U=setpoint)
            
            # 6. Apply split-range logic
            y_combined = np.zeros_like(t)
            pump_active = np.zeros_like(t)
            y_combined[0] = sp_initial_ph + y0_offset
            
            for i in range(1, len(t)):
                error_ph = setpoint[i] - y_combined[i-1]
                if error_ph < 0:  # pH > Setpoint --> Need ACID
                    y_combined[i] = y_acid_resp[i]
                    pump_active[i] = -1
                else:  # pH <= Setpoint --> Need BASE
                    y_combined[i] = y_base_resp[i]
                    pump_active[i] = 1
            
            y_combined = np.clip(y_combined, 0, 14)
            
            # 7. Create figure with subplots
            fig = make_subplots(rows=2, cols=1, subplot_titles=("pH Response", "Control Action (Split-Range)"),
                               vertical_spacing=0.12)
            
            # pH plot
            fig.add_trace(go.Scatter(x=t, y=setpoint, mode='lines', name='pH Setpoint',
                                    line=dict(color='red', dash='dash', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=y_combined, mode='lines', name='Controlled pH',
                                    line=dict(color='blue', width=1.5)), row=1, col=1)
            
            # Pump activity plot
            acid_signal = np.where(pump_active == -1, 1, 0)
            base_signal = np.where(pump_active == 1, 1, 0)
            fig.add_trace(go.Scatter(x=t, y=acid_signal, mode='lines', name='Acid Pump Active',
                                    line=dict(color='green', width=1.5, shape='hv')), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=base_signal, mode='lines', name='Base Pump Active',
                                    line=dict(color='magenta', width=1.5, shape='hv')), row=2, col=1)
            
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="pH", row=1, col=1)
            fig.update_yaxes(title_text="Pump Status", row=2, col=1, tickvals=[0, 1], ticktext=['Inactive', 'Active'])
            
            fig.update_layout(height=700, hovermode='x unified', template='plotly_white', showlegend=True)
            
            # 8. Create data table
            df_results = pd.DataFrame({
                'Time (s)': t[::100],
                'pH Setpoint': setpoint[::100],
                'Simulated pH': y_combined[::100],
                'Active Pump': np.where(pump_active[::100]==-1, 'Acid', np.where(pump_active[::100]==1, 'Base', 'None'))
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
            
            info = dbc.Alert("Simulation completed successfully with split-range control!", color="success")
            
            return info, fig, table
            
        except Exception as e:
            error_msg = dbc.Alert([
                html.H5("Simulation Error", className="alert-heading"),
                html.P(str(e)),
            ], color="danger")
            return error_msg, {}, ""