# ekf.py - Extended Kalman Filter (Dash Version)
import numpy as np
import casadi as ca
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

PAGE_ID = 'ekf'

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("EKF Parameters", className="mb-3"),
        
        html.Label("Final time (h):"),
        dcc.Slider(id=f'{PAGE_ID}-tfinal', min=5, max=50, step=5, value=20,
                   marks={5: '5', 20: '20', 50: '50'}, tooltip={"placement": "bottom", "always_visible": True}),
        
        html.H6("EKF Initial Conditions", className="mt-3"),
        html.Label("Estimated initial X (g/L):"),
        dcc.Input(id=f'{PAGE_ID}-x0_est', type='number', value=0.05, min=0.01, max=5.0, step=0.01, className="form-control mb-2"),
        html.Label("Estimated initial S (g/L):"),
        dcc.Input(id=f'{PAGE_ID}-s0_est', type='number', value=5.0, min=0.1, max=50.0, step=0.1, className="form-control mb-2"),
        html.Label("Estimated initial P (g/L):"),
        dcc.Input(id=f'{PAGE_ID}-p0_est', type='number', value=0.01, min=0.0, max=10.0, step=0.01, className="form-control mb-2"),
        html.Label("Estimated initial μmax (1/h):"),
        dcc.Input(id=f'{PAGE_ID}-mu0_est', type='number', value=0.40, min=0.1, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("Estimated initial Yxs (g/g):"),
        dcc.Input(id=f'{PAGE_ID}-yxs0_est', type='number', value=0.50, min=0.1, max=1.0, step=0.01, className="form-control mb-3"),
        
        html.H6("Initial Uncertainty P₀ (Diagonals)", className="mt-3"),
        html.Label("P0 - X:"),
        dcc.Input(id=f'{PAGE_ID}-p0_x', type='number', value=0.01, min=1e-4, max=1.0, step=0.001, className="form-control mb-2"),
        html.Label("P0 - S:"),
        dcc.Input(id=f'{PAGE_ID}-p0_s', type='number', value=0.01, min=1e-4, max=1.0, step=0.001, className="form-control mb-2"),
        html.Label("P0 - P:"),
        dcc.Input(id=f'{PAGE_ID}-p0_p', type='number', value=0.03, min=1e-4, max=1.0, step=0.001, className="form-control mb-2"),
        html.Label("P0 - μmax:"),
        dcc.Input(id=f'{PAGE_ID}-p0_mu', type='number', value=0.01, min=1e-4, max=1.0, step=0.001, className="form-control mb-2"),
        html.Label("P0 - Yxs:"),
        dcc.Input(id=f'{PAGE_ID}-p0_yxs', type='number', value=0.01, min=1e-4, max=1.0, step=0.001, className="form-control mb-3"),
        
        html.H6("Process Noise Q (Diagonals)", className="mt-3"),
        html.Label("Q - X:"),
        dcc.Input(id=f'{PAGE_ID}-q_x', type='number', value=1e-5, min=1e-8, max=1e-2, step=1e-6, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("Q - S:"),
        dcc.Input(id=f'{PAGE_ID}-q_s', type='number', value=1e-8, min=1e-10, max=1e-2, step=1e-9, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("Q - P:"),
        dcc.Input(id=f'{PAGE_ID}-q_p', type='number', value=1e-5, min=1e-8, max=1e-2, step=1e-6, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("Q - μmax:"),
        dcc.Input(id=f'{PAGE_ID}-q_mu', type='number', value=1e-6, min=1e-8, max=1e-2, step=1e-7, className="form-control mb-2", style={'fontSize': '12px'}),
        html.Label("Q - Yxs:"),
        dcc.Input(id=f'{PAGE_ID}-q_yxs', type='number', value=1e-6, min=1e-8, max=1e-2, step=1e-7, className="form-control mb-3", style={'fontSize': '12px'}),
        
        html.H6("Measurement Noise R (Diagonals)", className="mt-3"),
        html.Label("R - DO:"),
        dcc.Input(id=f'{PAGE_ID}-r_od', type='number', value=0.05, min=1e-4, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("R - pH:"),
        dcc.Input(id=f'{PAGE_ID}-r_ph', type='number', value=0.02, min=1e-4, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("R - Temp:"),
        dcc.Input(id=f'{PAGE_ID}-r_t', type='number', value=0.5, min=1e-2, max=5.0, step=0.1, className="form-control mb-3"),
        
        dbc.Button("▶️ Run EKF Simulation", id=f'{PAGE_ID}-btn-run', color="primary", className="w-100 mt-3")
    ], style={'maxHeight': '80vh', 'overflowY': 'scroll'})

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("Estimation of States and Parameters with Extended Kalman Filter (EKF)", className="mb-3"),
        dcc.Markdown("""
        This section simulates a batch bioprocess and uses an EKF to estimate the concentrations
        of Biomass (X), Substrate (S), Product (P), and two kinetic parameters
        ($\\mu_{max}$, $Y_{X/S}$) based on simulated and noisy measurements of
        Dissolved Oxygen (DO), pH and Temperature (T).

        **You can adjust:**
        * The **initial conditions** assumed by the EKF (the "initial guess").
        * The **initial uncertainty** about that guess (matrix $P_0$).
        * The **noise** levels assumed by the filter for the process ($Q$) and the measurements ($R$).
        
        Observe how these adjustments affect the EKF's ability to track actual values.
        
        **Mathematical Formulation:**
        
        **State-Space Model (5 states):**
        
        $$\\frac{dX}{dt} = \\mu X, \\quad \\mu = \\mu_{max} \\frac{S}{K_s + S}$$
        
        $$\\frac{dS}{dt} = -\\frac{1}{Y_{XS}} \\frac{dX}{dt}$$
        
        $$\\frac{dP}{dt} = \\alpha \\frac{dX}{dt}$$
        
        $$\\dot{\\mu}_{max} = 0, \\quad \\dot{Y}_{XS} = 0$$ (parameters assumed constant)
        
        **Measurement Model:**
        
        $$DO = DO_{sat} - k_{OUR} X + noise$$
        
        $$pH = pH_0 - k_{acid}(P - P_{ref}) + noise$$
        
        $$T = T_{set} + k_{Temp}(X \\cdot S) + noise$$
        
        **EKF Algorithm:**
        1. **Prediction:** $\\hat{x}_{k|k-1} = f(x_{k-1})$, $P_{k|k-1} = F_k P_{k-1} F_k^T + Q$
        2. **Update:** $K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1}$
        3. **Correction:** $\\hat{x}_k = \\hat{x}_{k|k-1} + K_k(z_k - h(\\hat{x}_{k|k-1}))$
        """, mathjax=True),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("Set parameters in the sidebar and click 'Run EKF Simulation' to start.", color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-output', 'children'),
        Input(f'{PAGE_ID}-btn-run', 'n_clicks'),
        [State(f'{PAGE_ID}-tfinal', 'value'),
         State(f'{PAGE_ID}-x0_est', 'value'), State(f'{PAGE_ID}-s0_est', 'value'),
         State(f'{PAGE_ID}-p0_est', 'value'), State(f'{PAGE_ID}-mu0_est', 'value'),
         State(f'{PAGE_ID}-yxs0_est', 'value'),
         State(f'{PAGE_ID}-p0_x', 'value'), State(f'{PAGE_ID}-p0_s', 'value'),
         State(f'{PAGE_ID}-p0_p', 'value'), State(f'{PAGE_ID}-p0_mu', 'value'),
         State(f'{PAGE_ID}-p0_yxs', 'value'),
         State(f'{PAGE_ID}-q_x', 'value'), State(f'{PAGE_ID}-q_s', 'value'),
         State(f'{PAGE_ID}-q_p', 'value'), State(f'{PAGE_ID}-q_mu', 'value'),
         State(f'{PAGE_ID}-q_yxs', 'value'),
         State(f'{PAGE_ID}-r_od', 'value'), State(f'{PAGE_ID}-r_ph', 'value'),
         State(f'{PAGE_ID}-r_t', 'value')],
        prevent_initial_call=True
    )
    def run_ekf(n_clicks, t_final, X0_est, S0_est, P0_est, mu0_est, yxs0_est,
                p0_X, p0_S, p0_P, p0_mu, p0_yxs,
                q_X, q_S, q_P, q_mu, q_yxs,
                r_OD, r_pH, r_T):
        if not n_clicks:
            return dbc.Alert("Click the button to start.", color="info")
        
        try:
            # Fixed parameters
            mu_max_real = 0.4
            Yxs_real = 0.5
            Ks = 0.1
            alpha = 0.1
            OD_sat = 8.0
            k_OUR = 0.5
            pH0 = 7.0
            P0_meas_ref = 0.0
            k_acid = 0.2
            Tset = 30
            k_Temp = 0.02
            dt_ekf = 0.1
            
            # --- CasADi Definitions ---
            n_states_ekf = 5
            n_meas_ekf = 3
            x_sym_ekf = ca.SX.sym('x', n_states_ekf)
            X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym_ekf)

            mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
            dX = mu_sym * X_sym
            dS = - (1 / Yxs_sym) * dX
            dP = alpha * dX
            dMu_max = 0
            dYxs = 0
            x_next_sym = x_sym_ekf + dt_ekf * ca.vertcat(dX, dS, dP, dMu_max, dYxs)
            f_func_ekf = ca.Function('f', [x_sym_ekf], [x_next_sym], ['x_k'], ['x_k_plus_1'])

            OD_val_sym = OD_sat - k_OUR * X_sym
            pH_val_sym = pH0 - k_acid * (P_sym - P0_meas_ref)
            T_val_sym = Tset + k_Temp * (X_sym * S_sym)
            z_sym_ekf = ca.vertcat(OD_val_sym, pH_val_sym, T_val_sym)
            h_func_ekf = ca.Function('h', [x_sym_ekf], [z_sym_ekf], ['x'], ['z'])

            F_sym_ekf = ca.jacobian(x_next_sym, x_sym_ekf)
            H_sym_ekf = ca.jacobian(z_sym_ekf, x_sym_ekf)
            F_func_ekf = ca.Function('F', [x_sym_ekf], [F_sym_ekf], ['x'], ['Fk'])
            H_func_ekf = ca.Function('H', [x_sym_ekf], [H_sym_ekf], ['x'], ['Hk'])

            # --- Simulation Setup ---
            time_vec_ekf = np.arange(0, t_final + dt_ekf, dt_ekf)
            N_ekf = len(time_vec_ekf)

            Q_ekf = np.diag([q_X, q_S, q_P, q_mu, q_yxs])
            R_ekf = np.diag([r_OD, r_pH, r_T])

            X0_real = 0.1
            S0_real = 5.0
            P0_real = 0.0
            x_real_ekf = np.array([[X0_real], [S0_real], [P0_real], [mu_max_real], [Yxs_real]])

            x_est_ekf = np.array([[X0_est], [S0_est], [P0_est], [mu0_est], [yxs0_est]])
            P_est_ekf = np.diag([p0_X, p0_S, p0_P, p0_mu, p0_yxs])

            x_real_hist = np.zeros((n_states_ekf, N_ekf))
            x_est_hist = np.zeros((n_states_ekf, N_ekf))
            z_meas_hist = np.zeros((n_meas_ekf, N_ekf))

            # --- EKF Simulation Loop ---
            for k in range(N_ekf):
                x_real_hist[:, k] = x_real_ekf.flatten()
                x_est_hist[:, k] = x_est_ekf.flatten()

                z_noisefree_dm = h_func_ekf(x_real_ekf)
                z_noisefree = z_noisefree_dm.full()
                noise_meas = np.random.multivariate_normal(np.zeros(n_meas_ekf), R_ekf).reshape(-1, 1)
                z_k = z_noisefree + noise_meas
                z_meas_hist[:, k] = z_k.flatten()

                if k < N_ekf - 1:
                    x_pred_dm = f_func_ekf(x_est_ekf)
                    x_pred = x_pred_dm.full()
                    Fk_dm = F_func_ekf(x_est_ekf)
                    Fk = Fk_dm.full()
                    P_pred = Fk @ P_est_ekf @ Fk.T + Q_ekf

                    Hk_dm = H_func_ekf(x_pred)
                    Hk = Hk_dm.full()
                    h_pred_dm = h_func_ekf(x_pred)
                    h_pred = h_pred_dm.full()
                    Sk = Hk @ P_pred @ Hk.T + R_ekf
                    Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk)
                    y_k = z_k - h_pred
                    x_upd = x_pred + Kk @ y_k
                    P_upd = (np.eye(n_states_ekf) - Kk @ Hk) @ P_pred

                    x_est_ekf = x_upd
                    x_est_ekf[0:3] = np.maximum(x_est_ekf[0:3], 0)
                    x_est_ekf[3:] = np.maximum(x_est_ekf[3:], 1e-6)
                    P_est_ekf = P_upd

                    x_real_next_no_noise_dm = f_func_ekf(x_real_ekf)
                    x_real_next_no_noise = x_real_next_no_noise_dm.full()
                    noise_proc = np.random.multivariate_normal(np.zeros(n_states_ekf), Q_ekf).reshape(-1, 1)
                    x_real_ekf = x_real_next_no_noise + noise_proc
                    x_real_ekf[0:3] = np.maximum(x_real_ekf[0:3], 0)

            # --- Create Plotly Figures ---
            fig1 = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Biomass (X)', 'DO Measurement', 'Substrate (S)', 
                               'pH Measurement', 'Product (P)', 'Temperature Measurement'),
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )

            # Biomass
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_real_hist[0, :], mode='lines', name='X real', line=dict(color='blue')), row=1, col=1)
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_est_hist[0, :], mode='lines', name='X estimated', line=dict(color='red', dash='dash')), row=1, col=1)

            # DO Measurement
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=z_meas_hist[0, :], mode='markers', name='DO measured', marker=dict(color='black', size=3)), row=1, col=2)

            # Substrate
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_real_hist[1, :], mode='lines', name='S real', line=dict(color='blue'), showlegend=False), row=2, col=1)
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_est_hist[1, :], mode='lines', name='S estimated', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)

            # pH Measurement
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=z_meas_hist[1, :], mode='markers', name='pH measured', marker=dict(color='black', size=3), showlegend=False), row=2, col=2)

            # Product
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_real_hist[2, :], mode='lines', name='P real', line=dict(color='blue'), showlegend=False), row=3, col=1)
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=x_est_hist[2, :], mode='lines', name='P estimated', line=dict(color='red', dash='dash'), showlegend=False), row=3, col=1)

            # Temperature Measurement
            fig1.add_trace(go.Scatter(x=time_vec_ekf, y=z_meas_hist[2, :], mode='markers', name='T measured', marker=dict(color='black', size=3), showlegend=False), row=3, col=2)

            fig1.update_xaxes(title_text="Time (h)", row=3)
            fig1.update_yaxes(title_text="Biomass (g/L)", row=1, col=1)
            fig1.update_yaxes(title_text="DO (mg/L)", row=1, col=2)
            fig1.update_yaxes(title_text="Substrate (g/L)", row=2, col=1)
            fig1.update_yaxes(title_text="pH", row=2, col=2)
            fig1.update_yaxes(title_text="Product (g/L)", row=3, col=1)
            fig1.update_yaxes(title_text="Temperature (°C)", row=3, col=2)
            fig1.update_layout(height=800, title_text="Estimation of States and Measurements (EKF)")

            # Figure 2: Parameters
            fig2 = make_subplots(rows=2, cols=1, subplot_titles=('μmax Estimation', 'Yxs Estimation'),
                                vertical_spacing=0.15)

            fig2.add_trace(go.Scatter(x=time_vec_ekf, y=x_real_hist[3, :], mode='lines', name='μmax real', line=dict(color='blue')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=time_vec_ekf, y=x_est_hist[3, :], mode='lines', name='μmax estimated', line=dict(color='red', dash='dash')), row=1, col=1)

            fig2.add_trace(go.Scatter(x=time_vec_ekf, y=x_real_hist[4, :], mode='lines', name='Yxs real', line=dict(color='blue'), showlegend=False), row=2, col=1)
            fig2.add_trace(go.Scatter(x=time_vec_ekf, y=x_est_hist[4, :], mode='lines', name='Yxs estimated', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=1)

            fig2.update_xaxes(title_text="Time (h)", row=2, col=1)
            fig2.update_yaxes(title_text="μmax (1/h)", row=1, col=1)
            fig2.update_yaxes(title_text="Y_X/S (gX/gS)", row=2, col=1)
            fig2.update_layout(height=600, title_text="Parameter Estimation (EKF)")

            # Final values table
            result_table = html.Div([
                html.H5("Final Values:"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Actual"),
                        html.P(f"X: {x_real_ekf[0,0]:.3f} g/L"),
                        html.P(f"S: {x_real_ekf[1,0]:.3f} g/L"),
                        html.P(f"P: {x_real_ekf[2,0]:.3f} g/L"),
                        html.P(f"μmax: {x_real_ekf[3,0]:.3f} 1/h"),
                        html.P(f"Yxs: {x_real_ekf[4,0]:.3f} g/g"),
                    ], width=6),
                    dbc.Col([
                        html.H6("Estimated"),
                        html.P(f"X est.: {x_est_ekf[0,0]:.3f} g/L"),
                        html.P(f"S est.: {x_est_ekf[1,0]:.3f} g/L"),
                        html.P(f"P est.: {x_est_ekf[2,0]:.3f} g/L"),
                        html.P(f"μmax est.: {x_est_ekf[3,0]:.3f} 1/h"),
                        html.P(f"Yxs est.: {x_est_ekf[4,0]:.3f} g/g"),
                    ], width=6)
                ])
            ])

            return html.Div([
                dbc.Alert("EKF simulation completed successfully!", color="success", className="mb-3"),
                dcc.Graph(figure=fig1),
                html.Hr(),
                dcc.Graph(figure=fig2),
                html.Hr(),
                result_table
            ])
            
        except Exception as e:
            return dbc.Alert(f"An error occurred: {str(e)}", color="danger")
