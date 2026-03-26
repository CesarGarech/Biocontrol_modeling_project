# reg_cascade_oxigen.py - DO Cascade Control (Dash Version)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

PAGE_ID = 'reg_cascade_oxigen'

# --- PID Controller Class ---
class PIDController:
    """A simple discrete PID controller class."""

    def __init__(self, Kp, Ki, Kd, Ts, setpoint=0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_meas = 0.0
        self.min_output, self.max_output = output_limits

    def compute(self, measured_value):
        """Calculates the controller output."""
        error = self.setpoint - measured_value
        proportional = self.Kp * error
        self.integral += self.Ki * error * self.Ts
        derivative = 0.0
        if self.Ts > 1e-9:
            derivative = self.Kd * (measured_value - self.prev_meas) / self.Ts
        output = proportional + self.integral - derivative
        self.prev_error = error
        self.prev_meas = measured_value
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)
        return output

    def update_setpoint(self, new_setpoint):
        """Updates the controller's setpoint."""
        self.setpoint = new_setpoint

    def reset(self, initial_measurement=0):
        """Resets the integral and previous error/measurement."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_meas = initial_measurement

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("Simulation Parameters", className="mb-3"),
        
        html.H6("1. Simulation Timing", className="mt-3"),
        html.Label("Sample Time (Ts) [s]:"),
        dcc.Input(id=f'{PAGE_ID}-ts', type='number', value=1.0, min=0.1, max=10.0, step=0.1, className="form-control mb-2"),
        html.Label("Final Simulation Time [s]:"),
        dcc.Input(id=f'{PAGE_ID}-tfinal', type='number', value=1000.0, min=100.0, max=5000.0, step=50.0, className="form-control mb-3"),
        
        html.H6("2. Process Models (First-Order)", className="mt-3"),
        html.Label("Motor Gain (K_motor) [RPM/%Power]:"),
        dcc.Input(id=f'{PAGE_ID}-k_motor', type='number', value=10.0, min=1.0, max=20.0, step=0.5, className="form-control mb-2"),
        html.Label("Motor Time Constant (τ_motor) [s]:"),
        dcc.Input(id=f'{PAGE_ID}-tau_motor', type='number', value=10.0, min=1.0, max=50.0, step=1.0, className="form-control mb-2"),
        html.Label("DO Process Gain (K_do) [%DO/RPM]:"),
        dcc.Input(id=f'{PAGE_ID}-k_do', type='number', value=0.1, min=0.01, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("DO Time Constant (τ_do) [s]:"),
        dcc.Input(id=f'{PAGE_ID}-tau_do', type='number', value=80.0, min=10.0, max=500.0, step=10.0, className="form-control mb-3"),
        
        html.H6("3. Sensor Models (First-Order)", className="mt-3"),
        html.Label("RPM Sensor τ [s]:"),
        dcc.Input(id=f'{PAGE_ID}-tau_sens_rpm', type='number', value=2.0, min=0.1, max=20.0, step=0.5, className="form-control mb-2"),
        html.Label("DO Sensor τ [s]:"),
        dcc.Input(id=f'{PAGE_ID}-tau_sens_do', type='number', value=15.0, min=1.0, max=60.0, step=1.0, className="form-control mb-3"),
        
        html.H6("4. PID Controllers", className="mt-3"),
        html.Label("Outer Loop (DO → RPM_sp):"),
        dcc.Input(id=f'{PAGE_ID}-kp_do', type='number', value=4.0, min=0.0, max=100.0, step=0.1, className="form-control mb-1", placeholder="Kp_DO"),
        dcc.Input(id=f'{PAGE_ID}-ki_do', type='number', value=0.05, min=0.0, max=10.0, step=0.01, className="form-control mb-1", placeholder="Ki_DO"),
        dcc.Input(id=f'{PAGE_ID}-kd_do', type='number', value=0.1, min=0.0, max=5.0, step=0.01, className="form-control mb-2", placeholder="Kd_DO"),
        html.Label("RPM Output Limits:"),
        dcc.Input(id=f'{PAGE_ID}-rpm_min', type='number', value=0.0, min=0.0, max=500.0, step=10.0, className="form-control mb-1", placeholder="Min RPM"),
        dcc.Input(id=f'{PAGE_ID}-rpm_max', type='number', value=1000.0, min=100.0, max=1000.0, step=50.0, className="form-control mb-2", placeholder="Max RPM"),
        
        html.Label("Inner Loop (RPM → Power):"),
        dcc.Input(id=f'{PAGE_ID}-kp_rpm', type='number', value=0.8, min=0.0, max=10.0, step=0.1, className="form-control mb-1", placeholder="Kp_RPM"),
        dcc.Input(id=f'{PAGE_ID}-ki_rpm', type='number', value=0.2, min=0.0, max=5.0, step=0.05, className="form-control mb-1", placeholder="Ki_RPM"),
        dcc.Input(id=f'{PAGE_ID}-kd_rpm', type='number', value=0.0, min=0.0, max=2.0, step=0.01, className="form-control mb-3", placeholder="Kd_RPM"),
        
        html.H6("5. Setpoint & Disturbance", className="mt-3"),
        html.Label("Initial DO Setpoint [% Sat]:"),
        dcc.Input(id=f'{PAGE_ID}-sp_init', type='number', value=10.0, min=0.0, max=100.0, step=1.0, className="form-control mb-2"),
        html.Label("Final DO Setpoint [% Sat]:"),
        dcc.Input(id=f'{PAGE_ID}-sp_final', type='number', value=30.0, min=0.0, max=100.0, step=1.0, className="form-control mb-2"),
        html.Label("Setpoint Change Time [s]:"),
        dcc.Input(id=f'{PAGE_ID}-t_step', type='number', value=50.0, min=0.0, max=5000.0, step=10.0, className="form-control mb-2"),
        html.Label("Disturbance Magnitude [% Power]:"),
        dcc.Input(id=f'{PAGE_ID}-dist_mag', type='number', value=-20.0, min=-50.0, max=50.0, step=5.0, className="form-control mb-2"),
        html.Label("Disturbance Start Time [s]:"),
        dcc.Input(id=f'{PAGE_ID}-t_dist', type='number', value=700.0, min=0.0, max=5000.0, step=50.0, className="form-control mb-3"),
        
        dbc.Button("▶️ Run Cascade Simulation", id=f'{PAGE_ID}-btn-run', color="primary", className="w-100 mt-3")
    ])

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("💧 Dissolved Oxygen Cascade Control Simulation", className="mb-3"),
        dcc.Markdown("""
        This page simulates a **cascade control** strategy for Dissolved Oxygen (DO) in a bioreactor.
        Cascade control is useful when the primary variable (DO) is slow, but its manipulated
        variable (Agitation RPM) is faster and subject to its own disturbances.

        **Concept:**
        1. **Primary (Outer) Loop:** Controls the main objective, **DO**. It measures DO and compares it to the DO setpoint. 
           Its output is the **setpoint for the Agitation (RPM)**. This loop is typically tuned slower.
        2. **Secondary (Inner) Loop:** Controls the **Agitation (RPM)**. It measures RPM and compares it to the RPM setpoint 
           received from the outer loop. Its output is the **motor power**. This loop is tuned faster to quickly correct RPM 
           deviations before they significantly affect DO.

        **Advantages:** The inner loop quickly rejects disturbances affecting the secondary variable (RPM), improving the 
        stability and performance of the primary loop (DO).
        
        **Mathematical Model:**
        
        Discrete first-order systems: $x[k+1] = \\alpha x[k] + (1-\\alpha) u[k]$ where $\\alpha = e^{-T_s/\\tau}$
        
        **Cascade Structure:**
        - Outer loop: $RPM_{sp} = PID_{DO}(DO_{sp} - DO_{meas})$
        - Inner loop: $Power = PID_{RPM}(RPM_{sp} - RPM_{meas})$
        """, mathjax=True),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("Set parameters and click 'Run Cascade Simulation' to start.", color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-output', 'children'),
        Input(f'{PAGE_ID}-btn-run', 'n_clicks'),
        [State(f'{PAGE_ID}-ts', 'value'),
         State(f'{PAGE_ID}-tfinal', 'value'),
         State(f'{PAGE_ID}-k_motor', 'value'),
         State(f'{PAGE_ID}-tau_motor', 'value'),
         State(f'{PAGE_ID}-k_do', 'value'),
         State(f'{PAGE_ID}-tau_do', 'value'),
         State(f'{PAGE_ID}-tau_sens_rpm', 'value'),
         State(f'{PAGE_ID}-tau_sens_do', 'value'),
         State(f'{PAGE_ID}-kp_do', 'value'),
         State(f'{PAGE_ID}-ki_do', 'value'),
         State(f'{PAGE_ID}-kd_do', 'value'),
         State(f'{PAGE_ID}-rpm_min', 'value'),
         State(f'{PAGE_ID}-rpm_max', 'value'),
         State(f'{PAGE_ID}-kp_rpm', 'value'),
         State(f'{PAGE_ID}-ki_rpm', 'value'),
         State(f'{PAGE_ID}-kd_rpm', 'value'),
         State(f'{PAGE_ID}-sp_init', 'value'),
         State(f'{PAGE_ID}-sp_final', 'value'),
         State(f'{PAGE_ID}-t_step', 'value'),
         State(f'{PAGE_ID}-dist_mag', 'value'),
         State(f'{PAGE_ID}-t_dist', 'value')],
        prevent_initial_call=True
    )
    def simulate_cascade(n_clicks, Ts, t_final_sim, K_motor, tau_motor, K_do, tau_do,
                        tau_sensor_rpm, tau_sensor_do, Kp_DO, Ki_DO, Kd_DO, rpm_min, rpm_max,
                        Kp_RPM, Ki_RPM, Kd_RPM, sp_initial_do, sp_final_do, t_step_do, 
                        dist_mag, t_dist_start):
        if not n_clicks:
            return dbc.Alert("Click the simulate button to start.", color="info")
        
        try:
            # --- Simulation Setup ---
            t = np.arange(0, t_final_sim + Ts, Ts)
            N = len(t)

            # --- Discrete Process Models ---
            alpha_motor = np.exp(-Ts / max(1e-6, tau_motor))
            alpha_do = np.exp(-Ts / max(1e-6, tau_do))
            alpha_sensor_rpm = np.exp(-Ts / max(1e-6, tau_sensor_rpm))
            alpha_sensor_do = np.exp(-Ts / max(1e-6, tau_sensor_do))

            # --- Controller Initialization ---
            power_min, power_max = 0.0, 100.0
            pid_DO = PIDController(Kp_DO, Ki_DO, Kd_DO, Ts, output_limits=(rpm_min, rpm_max))
            pid_RPM = PIDController(Kp_RPM, Ki_RPM, Kd_RPM, Ts, output_limits=(power_min, power_max))

            # --- Initialization of Simulation Arrays ---
            DO_sp_vec = np.zeros(N)
            DO_actual = np.zeros(N)
            DO_meas = np.zeros(N)
            RPM_sp_vec = np.zeros(N)
            RPM_actual = np.zeros(N)
            RPM_meas = np.zeros(N)
            Power = np.zeros(N)
            Power_dist_vec = np.zeros(N)

            # --- Set Initial Conditions ---
            DO_actual[0] = sp_initial_do
            DO_meas[0] = DO_actual[0]
            RPM_actual[0] = DO_actual[0] / max(1e-6, K_do) if K_do != 0 else rpm_min
            RPM_actual[0] = np.clip(RPM_actual[0], rpm_min, rpm_max)
            RPM_meas[0] = RPM_actual[0]
            RPM_sp_vec[0] = RPM_actual[0]
            Power[0] = RPM_actual[0] / max(1e-6, K_motor) if K_motor != 0 else power_min
            Power[0] = np.clip(Power[0], power_min, power_max)

            pid_DO.reset(initial_measurement=DO_meas[0])
            pid_RPM.reset(initial_measurement=RPM_meas[0])

            # --- Setpoint and Disturbance Profiles ---
            DO_sp_vec[:] = sp_initial_do
            DO_sp_vec[t >= t_step_do] = sp_final_do
            Power_dist_vec[t >= t_dist_start] = dist_mag

            # --- Main Simulation Loop ---
            for k in range(1, N):
                pid_DO.update_setpoint(DO_sp_vec[k])
                RPM_sp_vec[k] = pid_DO.compute(DO_meas[k-1])
                
                pid_RPM.update_setpoint(RPM_sp_vec[k])
                Power[k] = pid_RPM.compute(RPM_meas[k-1])
                
                power_with_disturbance = Power[k] + Power_dist_vec[k]
                power_with_disturbance = np.clip(power_with_disturbance, power_min, power_max)
                
                RPM_actual[k] = alpha_motor * RPM_actual[k-1] + \
                                (1 - alpha_motor) * K_motor * power_with_disturbance
                
                DO_actual[k] = alpha_do * DO_actual[k-1] + \
                               (1 - alpha_do) * K_do * RPM_actual[k-1]
                
                RPM_meas[k] = alpha_sensor_rpm * RPM_meas[k-1] + \
                              (1 - alpha_sensor_rpm) * RPM_actual[k] + np.random.randn() * 0.5
                
                DO_meas[k] = alpha_sensor_do * DO_meas[k-1] + \
                             (1 - alpha_sensor_do) * DO_actual[k] + np.random.randn() * 0.1

            # --- Create Plotly Figure ---
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Dissolved Oxygen (Primary Loop)', 'Agitation (Secondary Loop)', 
                               'Motor Power (Final Control Output)'),
                vertical_spacing=0.1
            )

            # Plot 1: DO
            fig.add_trace(go.Scatter(x=t, y=DO_actual, mode='lines', name='DO Actual', 
                                    line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=DO_meas, mode='lines', name='DO Measured', 
                                    line=dict(color='magenta', dash='dash'), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=DO_sp_vec, mode='lines', name='DO Setpoint', 
                                    line=dict(color='red', dash='dot', width=2)), row=1, col=1)

            # Plot 2: RPM
            fig.add_trace(go.Scatter(x=t, y=RPM_actual, mode='lines', name='RPM Actual', 
                                    line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=RPM_meas, mode='lines', name='RPM Measured', 
                                    line=dict(color='magenta', dash='dash'), opacity=0.7), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=RPM_sp_vec, mode='lines', name='RPM Setpoint (from DO)', 
                                    line=dict(color='green', dash='dot', width=2)), row=2, col=1)

            # Plot 3: Power
            fig.add_trace(go.Scatter(x=t, y=Power, mode='lines', name='PID Power Output', 
                                    line=dict(color='black')), row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=Power_dist_vec, mode='lines', name='Disturbance', 
                                    line=dict(color='red', dash='dash')), row=3, col=1)

            fig.update_xaxes(title_text="Time (s)", row=3, col=1)
            fig.update_yaxes(title_text="% Saturation", row=1, col=1)
            fig.update_yaxes(title_text="RPM", row=2, col=1)
            fig.update_yaxes(title_text="Power (%)", row=3, col=1)
            fig.update_layout(height=900, showlegend=True, title_text="Cascade Control Simulation Results")

            # --- Data Table ---
            df_results = pd.DataFrame({
                'Time (s)': t,
                'DO Setpoint (%)': DO_sp_vec,
                'DO Actual (%)': DO_actual,
                'DO Measured (%)': DO_meas,
                'RPM Setpoint': RPM_sp_vec,
                'RPM Actual': RPM_actual,
                'RPM Measured': RPM_meas,
                'Power Cmd (%)': Power,
                'Power Dist (%)': Power_dist_vec
            })

            return html.Div([
                dbc.Alert("Simulation completed successfully!", color="success", className="mb-3"),
                dcc.Graph(figure=fig),
                html.Hr(),
                html.H5("Simulation Data (first 100 points)", className="mt-4"),
                html.Div([
                    dcc.Markdown(df_results.head(100).to_markdown(index=False, floatfmt='.2f'))
                ], style={'maxHeight': '400px', 'overflowY': 'scroll', 'fontSize': '12px'})
            ])
            
        except Exception as e:
            return dbc.Alert(f"An error occurred during simulation: {str(e)}", color="danger")