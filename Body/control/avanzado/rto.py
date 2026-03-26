# rto.py - RTO Feed Profile Optimization (Dash Version)
import casadi as ca
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

PAGE_ID = 'rto'

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("RTO Configuration", className="mb-3"),
        
        html.H6("Model Parameters", className="mt-3"),
        html.Label("μmax [1/h]:"),
        dcc.Input(id=f'{PAGE_ID}-mu_max', type='number', value=0.6, min=0.01, step=0.01, className="form-control mb-2"),
        html.Label("Ks [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-Ks', type='number', value=0.2, min=0.01, step=0.01, className="form-control mb-2"),
        html.Label("KO [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-Ko', type='number', value=0.01, min=0.001, step=0.001, className="form-control mb-2"),
        html.Label("KP [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-KP', type='number', value=0.1, min=0.001, step=0.01, className="form-control mb-2"),
        html.Label("Yxs [g/g]:"),
        dcc.Input(id=f'{PAGE_ID}-Yxs', type='number', value=0.5, min=0.1, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("Yxo [g/g]:"),
        dcc.Input(id=f'{PAGE_ID}-Yxo', type='number', value=0.1, min=0.01, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("Yps [g/g]:"),
        dcc.Input(id=f'{PAGE_ID}-Yps', type='number', value=0.3, min=0.1, max=1.0, step=0.01, className="form-control mb-2"),
        html.Label("Feed concentration Sf [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-Sf', type='number', value=500.0, min=1, className="form-control mb-2"),
        html.Label("Max reactor volume [L]:"),
        dcc.Input(id=f'{PAGE_ID}-Vmax', type='number', value=2.0, min=0.1, step=0.1, className="form-control mb-3"),
        
        html.Hr(),
        html.H6("Initial Conditions", className="mt-3"),
        html.Label("X0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-X0', type='number', value=1.0, min=0, step=0.1, className="form-control mb-2"),
        html.Label("S0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-S0', type='number', value=20.0, min=0, className="form-control mb-2"),
        html.Label("P0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-P0', type='number', value=0.0, min=0, className="form-control mb-2"),
        html.Label("O0 [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-O0', type='number', value=0.08, min=0, step=0.01, className="form-control mb-2"),
        html.Label("V0 [L]:"),
        dcc.Input(id=f'{PAGE_ID}-V0', type='number', value=0.2, min=0.01, step=0.1, className="form-control mb-3"),
        
        html.Hr(),
        html.H6("Time Configuration", className="mt-3"),
        html.Label("Batch time [h]:"),
        dcc.Input(id=f'{PAGE_ID}-tbatch', type='number', value=5.0, min=0, className="form-control mb-2"),
        html.Label("Total process time [h]:"),
        dcc.Input(id=f'{PAGE_ID}-ttotal', type='number', value=24.0, min=6, className="form-control mb-3"),
        
        html.Hr(),
        html.H6("Operating Restrictions", className="mt-3"),
        html.Label("Min Flow [L/h]:"),
        dcc.Input(id=f'{PAGE_ID}-Fmin', type='number', value=0.0, min=0, step=0.01, className="form-control mb-2"),
        html.Label("Max Flow [L/h]:"),
        dcc.Input(id=f'{PAGE_ID}-Fmax', type='number', value=0.3, min=0, step=0.01, className="form-control mb-2"),
        html.Label("Max Substrate [g/L]:"),
        dcc.Input(id=f'{PAGE_ID}-Smax', type='number', value=30.0, min=0, className="form-control mb-3"),
        
        html.Hr(),
        html.H6("Kinetic Model", className="mt-3"),
        dcc.Dropdown(id=f'{PAGE_ID}-kinetic_model', options=[
            {'label': 'Monod', 'value': 'Monod'},
            {'label': 'Sigmoidal', 'value': 'Sigmoidal'},
            {'label': 'Complete', 'value': 'Complete'}
        ], value='Monod', clearable=False, className="mb-2"),
        html.Div(id=f'{PAGE_ID}-n_sigmoidal_div', children=[
            html.Label("n (Sigmoidal parameter):"),
            dcc.Input(id=f'{PAGE_ID}-n_sigmoidal', type='number', value=2.0, min=1.0, step=0.1, className="form-control")
        ], className="mb-3", style={'display': 'none'}),
        
        dbc.Button("🚀 Run RTO Optimization", id=f'{PAGE_ID}-btn-run', color="primary", className="w-100 mt-3")
    ], style={'maxHeight': '80vh', 'overflowY': 'scroll'})

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("🧠 RTO Control - Feed Profile Optimization", className="mb-3"),
        dcc.Markdown("""
        This module implements **Real-Time Optimization (RTO)** for fed-batch bioreactor operation using 
        orthogonal collocation on finite elements. The objective is to maximize the **total product mass** 
        (P × V) at the end of the process.

        **Process Model:**
        
        The bioreactor operates in two phases:
        1. **Batch phase** (0 to t_batch): F = 0, initial growth
        2. **Fed-batch phase** (t_batch to t_total): F optimized using collocation
        
        **State equations:**
        
        $$\\frac{dX}{dt} = \\mu X - DX, \\quad \\frac{dS}{dt} = -\\frac{\\mu X}{Y_{XS}} + D(S_f - S), \\quad \\frac{dP}{dt} = Y_{PS}\\mu X - DP$$
        
        $$\\frac{dV}{dt} = F, \\quad D = \\frac{F}{V}$$
        
        **Optimization:**
        - Objective: $\\max(P_{final} \\times V_{final})$
        - Collocation method: Radau IIA (degree 2)
        - Solver: IPOPT via CasADi
        - Constraints: $F_{min} \\leq F \\leq F_{max}$, $S \\leq S_{max}$, $V \\leq V_{max}$
        """, mathjax=True),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("Configure parameters in the sidebar and click 'Run RTO Optimization'.", color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    # Show/hide sigmoidal parameter input
    @app.callback(
        Output(f'{PAGE_ID}-n_sigmoidal_div', 'style'),
        Input(f'{PAGE_ID}-kinetic_model', 'value')
    )
    def toggle_sigmoidal(kinetic_model):
        if kinetic_model == 'Sigmoidal':
            return {'display': 'block'}
        return {'display': 'none'}
    
    # Main RTO optimization callback
    @app.callback(
        Output(f'{PAGE_ID}-output', 'children'),
        Input(f'{PAGE_ID}-btn-run', 'n_clicks'),
        [State(f'{PAGE_ID}-mu_max', 'value'),
         State(f'{PAGE_ID}-Ks', 'value'),
         State(f'{PAGE_ID}-Ko', 'value'),
         State(f'{PAGE_ID}-KP', 'value'),
         State(f'{PAGE_ID}-Yxs', 'value'),
         State(f'{PAGE_ID}-Yxo', 'value'),
         State(f'{PAGE_ID}-Yps', 'value'),
         State(f'{PAGE_ID}-Sf', 'value'),
         State(f'{PAGE_ID}-Vmax', 'value'),
         State(f'{PAGE_ID}-X0', 'value'),
         State(f'{PAGE_ID}-S0', 'value'),
         State(f'{PAGE_ID}-P0', 'value'),
         State(f'{PAGE_ID}-O0', 'value'),
         State(f'{PAGE_ID}-V0', 'value'),
         State(f'{PAGE_ID}-tbatch', 'value'),
         State(f'{PAGE_ID}-ttotal', 'value'),
         State(f'{PAGE_ID}-Fmin', 'value'),
         State(f'{PAGE_ID}-Fmax', 'value'),
         State(f'{PAGE_ID}-Smax', 'value'),
         State(f'{PAGE_ID}-kinetic_model', 'value'),
         State(f'{PAGE_ID}-n_sigmoidal', 'value')],
        prevent_initial_call=True
    )
    def run_rto(n_clicks, mu_max, Ks, Ko, KP, Yxs, Yxo, Yps, Sf_input, V_max_input,
                X0, S0, P0, O0, V0, t_batch, t_total, F_min, F_max, S_max, kinetic_model, n_sigmoidal):
        if not n_clicks:
            return dbc.Alert("Click the button to run optimization.", color="info")
        
        try:
            def radau_coefficients(d):
                """Radau IIA collocation coefficients for d=2"""
                if d == 2:
                    C_mat = np.array([[-2.0, 2.0], [1.5, -4.5], [0.5, 2.5]])
                    D_vec = np.array([0.0, 0.0, 1.0])
                    return C_mat, D_vec
                else:
                    raise NotImplementedError("Only d=2 implemented.")
            
            def odefun(x, u):
                """Fed-batch ODE model"""
                X_, S_, P_, O_, V_ = x[0], x[1], x[2], x[3], x[4]
                
                if kinetic_model == "Monod":
                    mu = mu_monod(S_, mu_max, Ks) * (O_ / (Ko + O_))
                elif kinetic_model == "Sigmoidal":
                    mu = mu_sigmoidal(S_, mu_max, Ks, n_sigmoidal) * (O_ / (Ko + O_))
                elif kinetic_model == "Complete":
                    mu = mu_completa(S_, O_, P_, mu_max, Ks, Ko, KP)
                else:
                    raise ValueError("Invalid kinetic model.")
                
                D = u / V_
                dX = mu * X_ - D * X_
                dS = -mu * X_ / Yxs + D * (Sf_input - S_)
                dP = Yps * mu * X_ - D * P_
                dO = 0.0
                dV = u
                return ca.vertcat(dX, dS, dP, dO, dV)
            
            # Batch phase integration
            n_fb_intervals = int(t_total - t_batch)
            dt_fb = (t_total - t_batch) / n_fb_intervals if n_fb_intervals > 0 else 0.0
            
            x_sym = ca.MX.sym("x", 5)
            u_sym = ca.MX.sym("u")
            ode_expr = odefun(x_sym, u_sym)
            
            batch_integrator = ca.integrator("batch_int", "idas",
                                            {"x": x_sym, "p": u_sym, "ode": ode_expr},
                                            {"t0": 0, "tf": t_batch})
            
            x0_np = np.array([X0, S0, P0, O0, V0])
            res_batch = batch_integrator(x0=x0_np, p=0.0)
            x_after_batch = np.array(res_batch['xf']).flatten()
            
            # Fed-batch phase with collocation
            opti = ca.Opti()
            d = 2
            C_radau, D_radau = radau_coefficients(d)
            nx = 5
            
            X_col = []
            F_col = []
            
            for k in range(n_fb_intervals):
                row_states = []
                for j in range(d + 1):
                    if (k == 0 and j == 0):
                        xk0_param = opti.parameter(nx)
                        opti.set_value(xk0_param, x_after_batch)
                        row_states.append(xk0_param)
                    else:
                        xk_j = opti.variable(nx)
                        row_states.append(xk_j)
                        opti.subject_to(xk_j >= 0)
                        opti.subject_to(xk_j[1] <= S_max)
                        opti.subject_to(xk_j[4] <= V_max_input)
                X_col.append(row_states)
                
                Fk = opti.variable()
                F_col.append(Fk)
                opti.subject_to(Fk >= F_min)
                opti.subject_to(Fk <= F_max)
            
            # Collocation equations
            h = dt_fb
            for k in range(n_fb_intervals):
                for j in range(1, d + 1):
                    xp_j = 0
                    for m in range(d + 1):
                        xp_j += C_radau[m, j - 1] * X_col[k][m]
                    fkj = odefun(X_col[k][j], F_col[k])
                    coll_eq = h * fkj - xp_j
                    opti.subject_to(coll_eq == 0)
                
                Xk_end = 0
                for m in range(d + 1):
                    Xk_end += D_radau[m] * X_col[k][m]
                
                if k < n_fb_intervals - 1:
                    for i_ in range(nx):
                        opti.subject_to(Xk_end[i_] == X_col[k + 1][0][i_])
            
            X_final = X_col[-1][-1]
            P_final = X_final[2]
            V_final = X_final[4]
            
            opti.minimize(-(P_final * V_final))
            
            # Initial guesses
            for k in range(n_fb_intervals):
                opti.set_initial(F_col[k], 0.1)
                for j in range(d + 1):
                    if not (k == 0 and j == 0):
                        opti.set_initial(X_col[k][j], x_after_batch)
            
            p_opts = {}
            s_opts = {"max_iter": 2000, "print_level": 0, "sb": 'yes', "mu_strategy": "adaptive"}
            opti.solver("ipopt", p_opts, s_opts)
            
            sol = opti.solve()
            
            F_opt = [sol.value(fk) for fk in F_col]
            X_fin_val = sol.value(X_final)
            P_fin_val = X_fin_val[2]
            V_fin_val = X_fin_val[4]
            
            # Reconstruct trajectories for plotting
            N_batch_plot = 50
            t_batch_plot = np.linspace(0, t_batch, N_batch_plot)
            dt_b = t_batch_plot[1] - t_batch_plot[0] if N_batch_plot > 1 else t_batch
            
            batch_plot_int = ca.integrator("batch_plot", "idas",
                                          {"x": x_sym, "p": u_sym, "ode": ode_expr},
                                          {"t0": 0, "tf": dt_b})
            
            xbatch_traj = [x0_np]
            xk_ = x0_np.copy()
            for _ in range(N_batch_plot - 1):
                res_ = batch_plot_int(x0=xk_, p=0.0)
                xk_ = np.array(res_["xf"]).flatten()
                xbatch_traj.append(xk_)
            xbatch_traj = np.array(xbatch_traj)
            
            t_fb_plot = np.linspace(t_batch, t_total, 400)
            dt_fb_plot = t_fb_plot[1] - t_fb_plot[0] if len(t_fb_plot) > 1 else (t_total - t_batch)
            
            fb_plot_int = ca.integrator("fb_plot", "idas",
                                       {"x": x_sym, "p": u_sym, "ode": ode_expr},
                                       {"t0": 0, "tf": dt_fb_plot})
            
            xfb_traj = []
            xk_ = xbatch_traj[-1].copy()
            for i, t_ in enumerate(t_fb_plot):
                xfb_traj.append(xk_)
                if i == len(t_fb_plot) - 1:
                    break
                kk_ = int((t_ - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, min(n_fb_intervals - 1, kk_))
                F_now = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                if xk_[4] >= V_max_input:
                    F_now = 0.0
                res_ = fb_plot_int(x0=xk_, p=F_now)
                xk_ = np.array(res_["xf"]).flatten()
            
            xfb_traj = np.array(xfb_traj)
            
            t_full = np.concatenate([t_batch_plot, t_fb_plot])
            x_full = np.vstack([xbatch_traj, xfb_traj])
            
            X_full = x_full[:, 0]
            S_full = x_full[:, 1]
            P_full = x_full[:, 2]
            O_full = x_full[:, 3]
            V_full = x_full[:, 4]
            
            F_batch_plot = np.zeros_like(t_batch_plot)
            F_fb_plot = []
            for i, tt in enumerate(t_fb_plot):
                kk_ = int((tt - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, min(n_fb_intervals - 1, kk_))
                valF = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                if xfb_traj[i, 4] >= V_max_input:
                    valF = 0.0
                F_fb_plot.append(valF)
            F_fb_plot = np.array(F_fb_plot)
            
            F_plot = np.concatenate([F_batch_plot, F_fb_plot])
            
            # Create plots
            fig = make_subplots(rows=2, cols=3,
                               subplot_titles=('Feed Flow F(t)', 'Biomass X(t)', 'Substrate S(t)',
                                             'Product P(t)', 'Oxygen O(t)', 'Volume V(t)'),
                               vertical_spacing=0.12, horizontal_spacing=0.08)
            
            fig.add_trace(go.Scatter(x=t_full, y=F_plot, mode='lines', line=dict(color='blue', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_full, y=X_full, mode='lines', line=dict(color='green', width=2)), row=1, col=2)
            fig.add_trace(go.Scatter(x=t_full, y=S_full, mode='lines', line=dict(color='orange', width=2)), row=1, col=3)
            fig.add_trace(go.Scatter(x=[0, t_total], y=[S_max, S_max], mode='lines',
                                    line=dict(color='red', dash='dash'), name='S_max'), row=1, col=3)
            fig.add_trace(go.Scatter(x=t_full, y=P_full, mode='lines', line=dict(color='purple', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=t_full, y=O_full, mode='lines', line=dict(color='brown', width=2)), row=2, col=2)
            fig.add_trace(go.Scatter(x=t_full, y=V_full, mode='lines', line=dict(color='darkblue', width=2)), row=2, col=3)
            fig.add_trace(go.Scatter(x=[0, t_total], y=[V_max_input, V_max_input], mode='lines',
                                    line=dict(color='red', dash='dash'), name='V_max'), row=2, col=3)
            
            fig.update_xaxes(title_text="Time (h)")
            fig.update_yaxes(title_text="F (L/h)", row=1, col=1)
            fig.update_yaxes(title_text="X (g/L)", row=1, col=2)
            fig.update_yaxes(title_text="S (g/L)", row=1, col=3)
            fig.update_yaxes(title_text="P (g/L)", row=2, col=1)
            fig.update_yaxes(title_text="O (g/L)", row=2, col=2)
            fig.update_yaxes(title_text="V (L)", row=2, col=3)
            fig.update_layout(height=700, showlegend=False, title_text="RTO Optimization Results")
            
            s_in_total = Sf_input * (V_fin_val - V0)
            rend = (P_fin_val * V_fin_val) / s_in_total if s_in_total > 1e-9 else 0
            
            return html.Div([
                dbc.Alert("✓ Optimization completed successfully!", color="success", className="mb-3"),
                dcc.Graph(figure=fig),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Total Accumulated Product", className="card-title"),
                                html.H3(f"{P_fin_val * V_fin_val:.2f} g", className="text-primary")
                            ])
                        ], className="mb-3")
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Product/Substrate Yield", className="card-title"),
                                html.H3(f"{rend:.3f} g/g", className="text-success")
                            ])
                        ], className="mb-3")
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Total Process Time", className="card-title"),
                                html.H3(f"{t_total:.2f} h", className="text-info")
                            ])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Final Volume", className="card-title"),
                                html.H3(f"{V_fin_val:.2f} L", className="text-warning")
                            ])
                        ])
                    ])
                ])
            ])
            
        except Exception as e:
            return dbc.Alert(f"Error in optimization: {str(e)}", color="danger")