# reg_ident.py - System Identification for pH Bioprocess (Dash Version)
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import TransferFunction, lsim
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

PAGE_ID = 'reg_ident'

# --- Process Model ---
def modelo_ph_planta(t, y, param):
    """Non-linear model of the fermentation bioprocess to simulate the real plant."""
    X, S, P, pH = y[0], y[1], y[2], y[3]
    if t < param['t_step']:
        D = param['D_base']
    else:
        D = param['D_base'] * (1 + param['step_percent'] / 100.0)
    
    f_pH = np.exp(-((pH - param['pH_opt'])**2) / (2 * param['pH_tol']**2))
    mu = param['mu_max'] * (S / (param['Ks'] + S)) * f_pH
    
    dXdt = (mu - D) * X
    dSdt = D * (param['S_in'] - S) - (1/param['Y_XS']) * mu * X
    dPdt = D * (param['P_in'] - P) + param['Y_PX'] * mu * X
    dpHdt = -param['alpha'] * P + D * (param['pH_in'] - pH)
    
    return [dXdt, dSdt, dPdt, dpHdt]

# --- Functions for System Identification ---
def transfer_function_response(t, u, params, n_poles, has_delay):
    """Calculates the response of a transfer function model."""
    K = params[0]
    tau = params[1]
    
    if n_poles == 2:
        zeta = params[2]
        den = [tau**2, 2*zeta*tau, 1]
    else:
        den = [tau, 1]
    
    num = [K]
    delay = params[-1] if has_delay else 0.0

    sys = TransferFunction(num, den)
    t_delayed = np.maximum(0, t - delay)
    u_delayed = np.interp(t_delayed, t, u)
    _, y_sim, _ = lsim(sys, U=u_delayed, T=t, interp=False)
    return y_sim

def error_function(params, t, u, y_meas, n_poles, has_delay):
    """Error function for least squares optimization."""
    y_sim = transfer_function_response(t, u, params, n_poles, has_delay)
    return y_sim - y_meas

def format_tf_latex(params, n_poles, has_delay):
    """Formats the identified transfer function into a LaTeX string."""
    K = params[0]
    tau = params[1]
    delay = params[-1] if has_delay else 0.0

    if n_poles == 2:
        zeta = params[2]
        den_str = f"{tau**2:.4f}s^2 + {2*zeta*tau:.4f}s + 1"
    else:
        den_str = f"{tau:.4f}s + 1"
    
    num_str = f"{K:.4f}"
    delay_str = f"e^{{-{delay:.2f}s}}" if has_delay and delay > 0.01 else ""
    
    return f"G(s) = \\frac{{{num_str}}}{{{den_str}}} {delay_str}"

#==========================================================================
# DASH LAYOUTS
#==========================================================================
def get_params_layout():
    """Parameters sidebar layout"""
    return html.Div([
        html.H4("Configuration", className="mb-3"),
        
        html.H6("1. Simulation Parameters", className="mt-3"),
        html.Label("Base Dilution Rate (D) [1/h]:"),
        dcc.Slider(id=f'{PAGE_ID}-d_base', min=0.01, max=0.2, step=0.001, value=0.065, 
                   marks={0.01: '0.01', 0.1: '0.1', 0.2: '0.2'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Time of Step Change (h):", className="mt-3"),
        dcc.Slider(id=f'{PAGE_ID}-t_step', min=50, max=250, step=10, value=150,
                   marks={50: '50', 150: '150', 250: '250'}, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Step Change Magnitude (%):", className="mt-3"),
        dcc.Slider(id=f'{PAGE_ID}-step_pct', min=5, max=50, step=1, value=20,
                   marks={5: '5', 25: '25', 50: '50'}, tooltip={"placement": "bottom", "always_visible": True}),
        
        html.H6("2. System Identification Settings", className="mt-4"),
        html.Label("Model Order (Poles):"),
        dcc.Dropdown(id=f'{PAGE_ID}-n_poles', options=[
            {'label': '1st Order', 'value': 1},
            {'label': '2nd Order', 'value': 2}
        ], value=2, clearable=False, className="mb-2"),
        dcc.Checklist(id=f'{PAGE_ID}-has_delay', options=[
            {'label': ' Estimate Time Delay', 'value': 'yes'}
        ], value=['yes'], className="mb-3"),
        
        dbc.Button("▶️ Run Simulation & Identification", id=f'{PAGE_ID}-btn-run', color="primary", className="w-100 mt-3")
    ])

def get_content_layout():
    """Main content layout"""
    return html.Div([
        html.H2("🧪 System Identification for a pH Bioprocess", className="mb-3"),
        dcc.Markdown("""
        This application simulates a non-linear bioprocess model and then uses the generated data 
        to identify a linear model (transfer function) that describes the pH dynamics.
        
        **Process:**
        1. **Non-linear Simulation:** A detailed bioprocess model with Monod kinetics and pH dependency
        2. **Data Extraction:** Deviation variables from steady-state after step change
        3. **Transfer Function Identification:** Least-squares optimization to fit 1st or 2nd order model
        4. **Validation:** R² goodness of fit metric
        
        **Model Equations:**
        
        $$f_{pH} = \\exp\\left(-\\frac{(pH - pH_{opt})^2}{2\\sigma_{pH}^2}\\right)$$
        
        $$\\mu = \\mu_{max} \\frac{S}{K_s + S} f_{pH}$$
        
        $$\\frac{dX}{dt} = (\\mu - D) X, \\quad \\frac{dS}{dt} = D(S_{in} - S) - \\frac{\\mu}{Y_{XS}} X$$
        
        $$\\frac{dpH}{dt} = -\\alpha P + D(pH_{in} - pH)$$
        """, mathjax=True),
        html.Hr(),
        
        html.Div(id=f'{PAGE_ID}-output', children=[
            dbc.Alert("Configure parameters and click 'Run Simulation & Identification' to start.", color="info")
        ])
    ])

#==========================================================================
# DASH CALLBACKS
#==========================================================================
def register_callbacks(app):
    @app.callback(
        Output(f'{PAGE_ID}-output',  'children'),
        Input(f'{PAGE_ID}-btn-run', 'n_clicks'),
        [State(f'{PAGE_ID}-d_base', 'value'),
         State(f'{PAGE_ID}-t_step', 'value'),
         State(f'{PAGE_ID}-step_pct', 'value'),
         State(f'{PAGE_ID}-n_poles', 'value'),
         State(f'{PAGE_ID}-has_delay', 'value')],
        prevent_initial_call=True
    )
    def run_identification(n_clicks, d_base, t_step, step_pct, n_poles, has_delay_list):
        if not n_clicks:
            return dbc.Alert("Click the button to start.", color="info")
        
        try:
            has_delay = 'yes' in (has_delay_list or [])
            
            # --- 1. Process Simulation ---
            param = {
                'D_base': d_base,
                't_step': t_step,
                'step_percent': step_pct,
                't_final': 300,
                'mu_max': 0.5, 'Ks': 1.0, 'pH_opt': 5.5, 'pH_tol': 0.5,
                'Y_XS': 0.5, 'Y_PX': 0.3, 'alpha': 0.1,
                'S_in': 10.0, 'P_in': 0.0, 'pH_in': 7.0
            }
            
            y0 = [0.5, 5.0, 0.0, 6.0]
            t_span = [0, param['t_final']]
            t_eval = np.linspace(0, param['t_final'], 1000)
            sol = solve_ivp(lambda t, y: modelo_ph_planta(t, y, param), t_span, y0, t_eval=t_eval, method='LSODA')
            t, y = sol.t, sol.y.T

            # Create simulation plot
            fig1 = make_subplots(rows=2, cols=2, subplot_titles=('Biomass (X)', 'Substrate (S)', 'Product (P)', 'pH'),
                                vertical_spacing=0.12)
            fig1.add_trace(go.Scatter(x=t, y=y[:, 0], mode='lines', name='X', line=dict(color='blue')), row=1, col=1)
            fig1.add_trace(go.Scatter(x=t, y=y[:, 1], mode='lines', name='S', line=dict(color='red')), row=1, col=2)
            fig1.add_trace(go.Scatter(x=t, y=y[:, 2], mode='lines', name='P', line=dict(color='green')), row=2, col=1)
            fig1.add_trace(go.Scatter(x=t, y=y[:, 3], mode='lines', name='pH', line=dict(color='magenta')), row=2, col=2)
            fig1.update_xaxes(title_text="Time (h)")
            fig1.update_layout(height=600, showlegend=False, title_text="Full Non-linear Simulation")

            # --- 2. Data Preparation ---
            idx_start = np.where(t >= param['t_step'])[0][0]
            y_ss = y[idx_start - 1, 3]
            u_ss = param['D_base']
            t_ident = t[idx_start:] - t[idx_start]
            ph_slice = y[idx_start:, 3]
            d_input_full = np.ones_like(t) * param['D_base']
            d_input_full[t >= param['t_step']] = param['D_base'] * (1 + param['step_percent'] / 100.0)
            d_slice = d_input_full[idx_start:]
            y_ident = ph_slice - y_ss
            u_ident = d_slice - u_ss

            # Create data extraction plot
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Input Data (Change in D)', 'Output Data (Change in pH)'))
            fig2.add_trace(go.Scatter(x=t_ident, y=u_ident, mode='lines', line=dict(color='black')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=t_ident, y=y_ident, mode='lines', line=dict(color='magenta')), row=1, col=2)
            fig2.update_xaxes(title_text="Time (h)")
            fig2.update_layout(height=400, showlegend=False)

            # --- 3. Model Identification ---
            gain_guess = y_ident[-1] / u_ident[-1] if u_ident[-1] != 0 else 0
            
            if n_poles == 2:
                p0, lb, ub = [gain_guess, 10.0, 1.0], [0, 0.1, 0.1], [np.inf, 100, 10]
            else:
                p0, lb, ub = [gain_guess, 10.0], [0, 0.1], [np.inf, 100]

            if has_delay:
                p0.append(1.0)
                lb.append(0.0)
                ub.append(t_ident[-1] / 2)
            
            res = least_squares(
                fun=error_function, x0=p0, bounds=(lb, ub),
                args=(t_ident, u_ident, y_ident, n_poles, has_delay),
                method='trf'
            )
            params_opt = res.x

            # --- 4. Validation ---
            tf_latex = format_tf_latex(params_opt, n_poles, has_delay)
            y_sim_opt = transfer_function_response(t_ident, u_ident, params_opt, n_poles, has_delay)
            r2 = r2_score(y_ident, y_sim_opt)

            # Create validation plot
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=t_ident, y=y_ident, mode='lines', name='Real pH Response', line=dict(color='magenta', width=2)))
            fig3.add_trace(go.Scatter(x=t_ident, y=y_sim_opt, mode='lines', name='Identified Model', line=dict(color='black', dash='dash', width=2)))
            fig3.update_layout(title=f"Model Validation (R² = {r2:.4f})", xaxis_title="Time (h)", 
                             yaxis_title="Change in pH", height=500, legend=dict(x=0.7, y=0.95))

            return html.Div([
                dbc.Alert("Identification completed successfully!", color="success", className="mb-3"),
                
                html.H4("1. Process Simulation Results", className="mt-3"),
                dcc.Graph(figure=fig1),
                
                html.H4("2. Data for System Identification", className="mt-4"),
                html.P("Data extracted after perturbation, adjusted to represent deviations from steady-state."),
                dcc.Graph(figure=fig2),
                
                html.H4("3. Identification Results", className="mt-4"),
                html.H5("Identified Transfer Function:"),
                dcc.Markdown(f"$${tf_latex}$$", mathjax=True, className="mb-3"),
                
                html.H5("Model Validation:"),
                dcc.Graph(figure=fig3),
                html.Div([
                    html.H5(f"Goodness of Fit (R²): {r2:.4f}", className="text-center"),
                    html.P("An R-squared value close to 1.0 indicates an excellent model fit.", 
                          className="text-muted text-center")
                ], className="mt-3")
            ])
            
        except Exception as e:
            return dbc.Alert(f"An error occurred: {str(e)}", color="danger")