import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
import base64
import io
import openpyxl
import traceback

PAGE_ID = 'ajuste_fedbatch'

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def calculate_feed_rate(t, strategy, t_start, t_end, F_min, F_max,
                        V0=None, mu_set=None, Xs_f_ratio=None, Sf=None,
                        step_data=None, X_current=None, V_current=None):
    t = float(t)
    t_start = float(t_start)
    t_end = float(t_end)

    if t < t_start - 1e-9 or t > t_end + 1e-9:
        return 0.0

    F_min = float(F_min)
    F_max = float(F_max)

    if strategy == 'Constant':
        return F_max

    elif strategy == 'Linear':
        if abs(t_end - t_start) < 1e-9:
            return F_max if abs(t - t_start) < 1e-9 else 0.0
        F = F_min + (F_max - F_min) * (t - t_start) / (t_end - t_start)
        return max(F_min, min(F, F_max))

    elif strategy == 'Exponential (Simple)':
        if abs(t_end - t_start) < 1e-9 or F_min <= 1e-9 or F_max < F_min:
            if abs(t_end - t_start) < 1e-9:
                return F_max if abs(t - t_start) < 1e-9 else 0.0
            else:
                return F_min if F_min > 1e-9 else 0.0
        k = np.log(F_max / F_min) / (t_end - t_start)
        F = F_min * np.exp(k * (t - t_start))
        return max(F_min, min(F, F_max))

    elif strategy == 'Exponential (constant mu)':
        if (mu_set is None or X_current is None or V_current is None or
                Sf is None or Xs_f_ratio is None or Sf <= 1e-9 or
                Xs_f_ratio <= 1e-9 or mu_set < 0):
            return 0.0
        Yxs_est = float(Xs_f_ratio)
        if Yxs_est <= 1e-9 or Sf <= 1e-9 or X_current < 0 or V_current < 0:
            return 0.0
        F = (mu_set / Yxs_est) * X_current * V_current / Sf
        return max(F_min, min(F, F_max))

    elif strategy == 'Step':
        if not step_data:
            return 0.0
        step_data_sorted = sorted(step_data, key=lambda item: item[0])
        current_flow = 0.0
        last_step_time = t_start
        flow_in_interval = 0.0
        found_interval = False
        for step_time, flow_value in step_data_sorted:
            step_time = float(step_time)
            flow_value = float(flow_value)
            if t >= last_step_time - 1e-9 and t < step_time - 1e-9:
                current_flow = flow_in_interval
                found_interval = True
                break
            if step_time >= t_start - 1e-9:
                flow_in_interval = flow_value
            last_step_time = step_time
        if not found_interval and t >= last_step_time - 1e-9:
            current_flow = flow_in_interval
        if t < t_start - 1e-9 or t > t_end + 1e-9:
            return 0.0
        else:
            return max(F_min, min(current_flow, F_max))
    else:
        return 0.0


def modelo_ode_fedbatch(t, y, params, feed_params):
    try:
        X, S, P, O2, V = y
        if len(params) != 6:
            raise ValueError(
                f"6 parameters were expected [mumax, Ks, Yxs, Kd, Ypx, Ksi], "
                f"but only {len(params)} were received"
            )
        mumax, Ks, Yxs, Kd, Ypx, Ksi = params

        strategy = feed_params['strategy']
        t_start = feed_params['t_start']
        t_end = feed_params['t_end']
        F_min = feed_params['F_min']
        F_max = feed_params['F_max']
        Sf = feed_params['Sf']
        Xf = feed_params['Xf']
        Pf = feed_params['Pf']
        step_data = feed_params.get('step_data', None)
        mu_set = feed_params.get('mu_set', None)
        V0 = feed_params.get('V0', None)
        Yxs_param_for_F = max(Yxs, 1e-9)

        F = calculate_feed_rate(t, strategy, t_start, t_end, F_min, F_max,
                                V0=V0, mu_set=mu_set, Xs_f_ratio=Yxs_param_for_F,
                                Sf=Sf, step_data=step_data,
                                X_current=X, V_current=V)

        X_safe = max(X, 0.0)
        S_safe = max(S, 0.0)
        V_safe = max(V, 1e-9)
        D = F / V_safe

        Ks_safe = max(Ks, 1e-9)
        Yxs_safe = max(Yxs, 1e-9)
        Kd_safe = max(Kd, 0.0)
        Ypx_safe = max(Ypx, 0.0)
        mumax_safe = max(mumax, 0.0)
        Ksi_safe = max(Ksi, 1e-6)

        denominator = Ks_safe + S_safe + (S_safe ** 2 / Ksi_safe)
        if denominator <= 1e-12:
            mu = 0.0
        else:
            mu = mumax_safe * S_safe / denominator
        mu = max(mu, 0.0)

        dXdt = mu * X_safe - Kd_safe * X_safe - D * X + D * Xf
        dSdt = -(mu / Yxs_safe) * X_safe + D * (Sf - S)
        dPdt = Ypx_safe * mu * X_safe - D * P + D * Pf
        dO2dt = 0
        dVdt = F

        return [dXdt, dSdt, dPdt, dO2dt, dVdt]

    except Exception as e:
        raise


def compute_jacobian_fedbatch(params_opt, t_exp, y0_fit, feed_params, atol, rtol):
    delta = 1e-7
    n_params = len(params_opt)
    if n_params != 6:
        return None

    n_times = len(t_exp)
    n_vars_measured = 3
    y_nominal_flat = np.zeros(n_times * n_vars_measured)

    try:
        sol_nominal = solve_ivp(
            modelo_ode_fedbatch, [t_exp[0], t_exp[-1]], y0_fit,
            args=(params_opt, feed_params),
            t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA'
        )
        if sol_nominal.status != 0:
            return np.full((n_times * n_vars_measured, n_params), np.nan)
        y_nominal_flat = sol_nominal.y[0:n_vars_measured, :].flatten()
    except Exception:
        return np.full((n_times * n_vars_measured, n_params), np.nan)

    jac = np.zeros((n_times * n_vars_measured, n_params))

    for i in range(n_params):
        params_perturbed = np.array(params_opt, dtype=float)
        h = delta * abs(params_perturbed[i]) if abs(params_perturbed[i]) > 1e-8 else delta
        if h == 0:
            jac[:, i] = 0.0
            continue
        params_perturbed[i] += h

        try:
            sol_perturbed = solve_ivp(
                modelo_ode_fedbatch, [t_exp[0], t_exp[-1]], y0_fit,
                args=(params_perturbed, feed_params),
                t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA'
            )
            if sol_perturbed.status != 0:
                jac[:, i] = np.nan
                continue
            y_perturbed_flat = sol_perturbed.y[0:n_vars_measured, :].flatten()
            jac[:, i] = (y_perturbed_flat - y_nominal_flat) / h
        except Exception:
            jac[:, i] = np.nan

    return jac


def objetivo_fedbatch(params, t_exp, y_exp_stacked, y0_fit, feed_params, atol, rtol):
    if len(params) != 6:
        return 1e18

    try:
        sol = solve_ivp(
            modelo_ode_fedbatch, [t_exp[0], t_exp[-1]], y0_fit,
            args=(params, feed_params),
            t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA'
        )

        if sol.status != 0:
            return 1e6 + np.sum(np.abs(params))

        y_pred = sol.y[0:3, :]
        y_pred_stacked = y_pred.flatten()
        if y_exp_stacked.shape != y_pred_stacked.shape:
            return 1e11
        mask = ~np.isnan(y_exp_stacked)
        if np.sum(mask) == 0:
            return 1e12
        sse = np.sum((y_pred_stacked[mask] - y_exp_stacked[mask]) ** 2)
        rmse = np.sqrt(sse / np.sum(mask))

        if np.any(sol.y[0:3, :] < -1e-3):
            neg_penalty = np.sum(np.abs(sol.y[0:3, :][sol.y[0:3, :] < -1e-3]))
            rmse += neg_penalty * 1e3
        if np.any(sol.y[4, :] < 0):
            rmse += 1e5
        if np.isnan(rmse) or np.isinf(rmse):
            return 1e15

        return rmse

    except Exception:
        return 1e15


# --------------------------------------------------------------------------
# Layout Functions
# --------------------------------------------------------------------------

def get_params_layout():
    return html.Div([
        html.H6("📤 Upload experimental data", className='text-white mt-2'),
        dcc.Upload(
            id=f'{PAGE_ID}-upload',
            children=html.Div(['Drag & Drop or ', html.A('Select Excel File (.xlsx)')]),
            style={
                'width': '100%', 'height': '50px', 'lineHeight': '50px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'color': 'white', 'marginBottom': '8px'
            },
            accept='.xlsx'
        ),

        html.H6("⚙️ Initial Conditions", className='text-white mt-2'),
        html.Label("Initial Biomass X0 [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-x0', type='number', value=0.1, min=0.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Initial Substrate S0 [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-s0', type='number', value=10.0, min=0.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Initial Product P0 [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-p0', type='number', value=0.0, min=0.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Initial O2 [mg/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-o0', type='number', value=8.0, min=0.0, max=20.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Initial Volume V0 [L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-v0', type='number', value=1.0, min=1e-3, step=0.001,
                  className='form-control form-control-sm mb-2'),

        html.H6("Feeding Configuration", className='text-white mt-2'),
        html.Label("Feeding Strategy", className='text-white small'),
        dcc.Dropdown(
            id=f'{PAGE_ID}-feed-strategy',
            options=['Constant', 'Linear', 'Exponential (Simple)', 'Exponential (constant mu)', 'Step'],
            value='Constant',
            className='mb-2'
        ),
        html.Label("Start Feeding Time [h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-t-start-feed', type='number', value=0.0, step=0.1,
                  className='form-control form-control-sm mb-2'),
        html.Label("End Feeding Time [h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-t-end-feed', type='number', value=100.0, step=0.1,
                  className='form-control form-control-sm mb-2'),
        html.Label("F_min [L/h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-f-min', type='number', value=0.0, min=0.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("F_max [L/h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-f-max', type='number', value=0.1, min=0.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Substrate Feed Conc. Sf [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-sf', type='number', value=100.0, min=0.0, step=0.1,
                  className='form-control form-control-sm mb-2'),
        html.Label("Biomass Feed Conc. Xf [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-xf', type='number', value=0.0, min=0.0, step=0.01,
                  className='form-control form-control-sm mb-2'),
        html.Label("Product Feed Conc. Pf [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-pf', type='number', value=0.0, min=0.0, step=0.01,
                  className='form-control form-control-sm mb-2'),

        html.Div(id=f'{PAGE_ID}-step-data-section', style={'display': 'none'}, children=[
            html.Label("Step Data (time,flow per line)", className='text-white small'),
            dcc.Textarea(
                id=f'{PAGE_ID}-step-data-text',
                placeholder="Format: time,flow per line\n2, 0.05\n5, 0.1",
                style={'width': '100%', 'height': '100px'}
            )
        ]),
        html.Div(id=f'{PAGE_ID}-mu-set-section', style={'display': 'none'}, children=[
            html.Label("Desired μ_set [1/h]", className='text-white small'),
            dcc.Input(id=f'{PAGE_ID}-mu-set', type='number', value=0.1, min=0.0, max=2.0, step=0.001,
                      className='form-control form-control-sm mb-2')
        ]),

        html.H6("Kinetic Parameters (6)", className='text-white mt-2'),
        html.Label("μmax [1/h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-mumax', type='number', value=0.5, min=0.001, max=5.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Ks [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-ks', type='number', value=0.2, min=1e-4, max=20.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Yxs [g/g]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-yxs', type='number', value=0.5, min=0.01, max=3.0, step=0.01,
                  className='form-control form-control-sm mb-2'),
        html.Label("Kd [1/h]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-kd', type='number', value=0.01, min=0.0, max=1.0, step=0.001,
                  className='form-control form-control-sm mb-2'),
        html.Label("Ypx [g/g]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-ypx', type='number', value=0.3, min=0.0, max=10.0, step=0.01,
                  className='form-control form-control-sm mb-2'),
        html.Label("Ksi (Substrate Inhib.) [g/L]", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-ksi', type='number', value=100.0, min=1.0, max=1000.0, step=1.0,
                  className='form-control form-control-sm mb-2'),

        html.H6("ODE Solver Tolerances", className='text-white mt-2'),
        html.Label("atol", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-atol', type='number', value=1e-8, step=1e-9,
                  className='form-control form-control-sm mb-2'),
        html.Label("rtol", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-rtol', type='number', value=1e-8, step=1e-9,
                  className='form-control form-control-sm mb-2'),

        html.H6("Optimization Options", className='text-white mt-2'),
        html.Label("Method", className='text-white small'),
        dcc.Dropdown(
            id=f'{PAGE_ID}-method',
            options=['L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell', 'differential_evolution'],
            value='L-BFGS-B',
            className='mb-2'
        ),
        html.Label("Max Iterations", className='text-white small'),
        dcc.Input(id=f'{PAGE_ID}-maxiter', type='number', value=500, min=10, max=10000, step=10,
                  className='form-control form-control-sm mb-2'),

        html.Button(
            "🚀 Run Adjustment (Fed-Batch)",
            id=f'{PAGE_ID}-run-btn',
            n_clicks=0,
            className='btn btn-success w-100 mt-2'
        ),
    ])


def get_content_layout():
    return html.Div([
        dcc.Store(id=f'{PAGE_ID}-data-store'),
        dcc.Store(id=f'{PAGE_ID}-step-data-store'),
        html.H4("🔧 Adjustment of Kinetic Parameters (Fed-Batch with Substrate Inhibition)"),
        html.Div(id=f'{PAGE_ID}-preview'),
        html.Div(id=f'{PAGE_ID}-results-div'),
    ])


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------

def register_callbacks(app):

    @app.callback(
        Output(f'{PAGE_ID}-step-data-section', 'style'),
        Output(f'{PAGE_ID}-mu-set-section', 'style'),
        Input(f'{PAGE_ID}-feed-strategy', 'value'),
    )
    def show_hide_strategy_sections(strategy):
        step_style = {'display': 'block'} if strategy == 'Step' else {'display': 'none'}
        mu_style = {'display': 'block'} if strategy == 'Exponential (constant mu)' else {'display': 'none'}
        return step_style, mu_style

    @app.callback(
        Output(f'{PAGE_ID}-data-store', 'data'),
        Output(f'{PAGE_ID}-preview', 'children'),
        Input(f'{PAGE_ID}-upload', 'contents'),
        State(f'{PAGE_ID}-upload', 'filename'),
    )
    def parse_upload(contents, filename):
        if contents is None:
            return None, html.Div()

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')

            required_cols = ['time', 'biomass', 'substrate', 'product']
            if not all(col in df.columns for col in required_cols):
                return None, dbc.Alert(
                    f"File must contain columns: {', '.join(required_cols)}", color='danger'
                )

            df = df.sort_values(by='time').reset_index(drop=True)
            df[['biomass', 'substrate', 'product']] = df[['biomass', 'substrate', 'product']].apply(
                pd.to_numeric, errors='coerce'
            )

            preview = html.Div([
                dbc.Alert(f"✅ File '{filename}' loaded ({len(df)} rows).", color='success'),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'fontSize': '12px'},
                    page_size=5,
                )
            ])
            return df.to_dict('records'), preview

        except Exception as e:
            return None, dbc.Alert(f"Error reading file: {e}", color='danger')

    @app.callback(
        Output(f'{PAGE_ID}-step-data-store', 'data'),
        Input(f'{PAGE_ID}-step-data-text', 'value'),
    )
    def parse_step_data(text):
        if not text:
            return []
        result = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) == 2:
                    result.append([float(parts[0].strip()), float(parts[1].strip())])
            except ValueError:
                continue
        return result

    @app.callback(
        Output(f'{PAGE_ID}-results-div', 'children'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-x0', 'value'),
        State(f'{PAGE_ID}-s0', 'value'),
        State(f'{PAGE_ID}-p0', 'value'),
        State(f'{PAGE_ID}-o0', 'value'),
        State(f'{PAGE_ID}-v0', 'value'),
        State(f'{PAGE_ID}-feed-strategy', 'value'),
        State(f'{PAGE_ID}-t-start-feed', 'value'),
        State(f'{PAGE_ID}-t-end-feed', 'value'),
        State(f'{PAGE_ID}-f-min', 'value'),
        State(f'{PAGE_ID}-f-max', 'value'),
        State(f'{PAGE_ID}-sf', 'value'),
        State(f'{PAGE_ID}-xf', 'value'),
        State(f'{PAGE_ID}-pf', 'value'),
        State(f'{PAGE_ID}-step-data-store', 'data'),
        State(f'{PAGE_ID}-mu-set', 'value'),
        State(f'{PAGE_ID}-mumax', 'value'),
        State(f'{PAGE_ID}-ks', 'value'),
        State(f'{PAGE_ID}-yxs', 'value'),
        State(f'{PAGE_ID}-kd', 'value'),
        State(f'{PAGE_ID}-ypx', 'value'),
        State(f'{PAGE_ID}-ksi', 'value'),
        State(f'{PAGE_ID}-atol', 'value'),
        State(f'{PAGE_ID}-rtol', 'value'),
        State(f'{PAGE_ID}-method', 'value'),
        State(f'{PAGE_ID}-maxiter', 'value'),
        State(f'{PAGE_ID}-data-store', 'data'),
        prevent_initial_call=True,
    )
    def run_fit(n_clicks, x0, s0, p0, o0, v0,
                feed_strategy, t_start_feed, t_end_feed, f_min, f_max,
                sf, xf, pf, step_data_stored, mu_set,
                mumax, ks, yxs, kd, ypx, ksi,
                atol, rtol, method, maxiter, data_store):

        if not data_store:
            return dbc.Alert("No experimental data loaded. Please upload a file first.", color='warning')

        # Reconstruct DataFrame
        df_exp = pd.DataFrame(data_store)
        t_exp = df_exp['time'].values
        y_exp = df_exp[['biomass', 'substrate', 'product']].values.T
        y_exp_stacked = y_exp.flatten()

        # Initial conditions
        X0 = float(x0 or 0.1)
        S0 = float(s0 or 10.0)
        P0 = float(p0 or 0.0)
        O0 = float(o0 or 8.0)
        V0 = float(v0 or 1.0)
        y0 = [X0, S0, P0, O0, V0]

        # Feed params – convert stored list-of-lists to list of tuples
        step_data = None
        if step_data_stored:
            step_data = [tuple(pair) for pair in step_data_stored]

        feed_params = {
            'strategy': feed_strategy or 'Constant',
            't_start': float(t_start_feed or 0.0),
            't_end': float(t_end_feed or 100.0),
            'F_min': float(f_min or 0.0),
            'F_max': float(f_max or 0.1),
            'Sf': float(sf or 100.0),
            'Xf': float(xf or 0.0),
            'Pf': float(pf or 0.0),
            'step_data': step_data,
            'mu_set': float(mu_set) if mu_set is not None else None,
            'V0': V0,
        }

        initial_guess = [
            float(mumax or 0.5), float(ks or 0.2), float(yxs or 0.5),
            float(kd or 0.01), float(ypx or 0.3), float(ksi or 100.0)
        ]
        bounds = [(1e-3, 5.0), (1e-4, 20.0), (0.01, 3.0), (0.0, 1.0), (0.0, 10.0), (1.0, 1000.0)]
        atol_v = float(atol or 1e-8)
        rtol_v = float(rtol or 1e-8)
        max_iter = int(maxiter or 500)

        # Run optimization
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    objetivo_fedbatch, bounds,
                    args=(t_exp, y_exp_stacked, y0, feed_params, atol_v, rtol_v),
                    maxiter=max_iter, tol=1e-7, mutation=(0.5, 1.5),
                    recombination=0.8, strategy='best1bin',
                    updating='deferred', workers=-1, seed=42
                )
            else:
                minimizer_kwargs = {
                    'args': (t_exp, y_exp_stacked, y0, feed_params, atol_v, rtol_v),
                    'method': method,
                    'bounds': bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                    'options': {'maxiter': max_iter, 'disp': False}
                }
                if method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                    minimizer_kwargs['options']['ftol'] = 1e-9
                    minimizer_kwargs['options']['gtol'] = 1e-7
                elif method == 'Nelder-Mead':
                    minimizer_kwargs['options']['xatol'] = 1e-7
                    minimizer_kwargs['options']['fatol'] = 1e-9
                result = minimize(objetivo_fedbatch, initial_guess, **minimizer_kwargs)

        except Exception as e:
            return dbc.Alert(f"Fatal error during optimization: {e}\n{traceback.format_exc()}", color='danger')

        if result is None or not hasattr(result, 'x'):
            return dbc.Alert("Optimization failed to return a result.", color='danger')

        params_opt = result.x

        # Final simulation
        try:
            sol = solve_ivp(
                modelo_ode_fedbatch, [t_exp[0], t_exp[-1]], y0,
                args=(params_opt, feed_params),
                t_eval=t_exp, atol=atol_v, rtol=rtol_v, method='LSODA'
            )
        except Exception as e:
            return dbc.Alert(f"Error in final simulation: {e}", color='danger')

        if sol.status != 0:
            return dbc.Alert(f"Final simulation failed (status {sol.status}): {sol.message}", color='danger')

        y_pred_final = sol.y[0:3, :]

        # Metrics
        param_names = ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx', 'Ksi']
        units = ['1/h', 'g/L', 'g/g', '1/h', 'g/g', 'g/L']

        params_table_data = [
            {'Parameter': param_names[i], 'Optimized Value': f"{params_opt[i]:.5f}", 'Units': units[i]}
            for i in range(6)
        ]

        metricas_list = []
        for i, var in enumerate(['Biomass', 'Substrate', 'Product']):
            exp_data = y_exp[i]
            pred_data = y_pred_final[i]
            valid_mask = ~np.isnan(exp_data)
            if np.sum(valid_mask) > 1:
                exp_v = exp_data[valid_mask]
                pred_v = pred_data[valid_mask]
                try:
                    r2 = r2_score(exp_v, pred_v)
                except ValueError:
                    r2 = np.nan
                rmse_val = np.sqrt(mean_squared_error(exp_v, pred_v))
                metricas_list.append({'Variable': var, 'R²': f"{r2:.4f}", 'RMSE': f"{rmse_val:.4f}"})
            else:
                metricas_list.append({'Variable': var, 'R²': 'N/A', 'RMSE': 'N/A'})

        # Comparison graph
        fig_cmp = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=['Biomass', 'Substrate', 'Product'])
        colors_exp = ['blue', 'orange', 'green']
        colors_mod = ['#1f77b4', '#ff7f0e', '#2ca02c']
        var_labels = ['Biomass [g/L]', 'Substrate [g/L]', 'Product [g/L]']
        for i in range(3):
            fig_cmp.add_trace(
                go.Scatter(x=t_exp, y=y_exp[i], mode='markers',
                           name=f'{["Biomass","Substrate","Product"][i]} Exp.',
                           marker=dict(color=colors_exp[i], size=6)),
                row=i + 1, col=1
            )
            fig_cmp.add_trace(
                go.Scatter(x=sol.t, y=sol.y[i], mode='lines',
                           name=f'{["Biomass","Substrate","Product"][i]} Model',
                           line=dict(color=colors_mod[i], dash='dash')),
                row=i + 1, col=1
            )
            fig_cmp.update_yaxes(title_text=var_labels[i], row=i + 1, col=1)
        fig_cmp.update_xaxes(title_text='Time [h]', row=3, col=1)
        fig_cmp.update_layout(
            title="Model vs Experimental Data (Fed-Batch with Substrate Inhibition)",
            height=700
        )

        # Volume / Flow graph
        yxs_opt = params_opt[2]
        interp_x = np.interp(sol.t, sol.t, sol.y[0])
        interp_v = np.interp(sol.t, sol.t, sol.y[4])
        F_t_values = np.array([
            calculate_feed_rate(
                ti, feed_params['strategy'], feed_params['t_start'], feed_params['t_end'],
                feed_params['F_min'], feed_params['F_max'],
                V0=feed_params['V0'], mu_set=feed_params['mu_set'],
                Xs_f_ratio=yxs_opt, Sf=feed_params['Sf'],
                step_data=feed_params['step_data'],
                X_current=interp_x[idx], V_current=interp_v[idx]
            )
            for idx, ti in enumerate(sol.t)
        ])
        F_t_times = sol.t

        fig_vf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vf.add_trace(
            go.Scatter(x=sol.t, y=sol.y[4], name='Volume [L]', line=dict(color='red')),
            secondary_y=False
        )
        flow_line_shape = 'hv' if feed_params['strategy'] in ['Constant', 'Step'] else 'linear'
        fig_vf.add_trace(
            go.Scatter(x=F_t_times, y=F_t_values, name='Feed Flow [L/h]',
                       line=dict(color='blue', dash='dash', shape=flow_line_shape)),
            secondary_y=True
        )
        fig_vf.update_yaxes(title_text="Volume [L]", secondary_y=False)
        fig_vf.update_yaxes(title_text="Feed Flow [L/h]", secondary_y=True)
        fig_vf.update_xaxes(title_text="Time [h]")
        fig_vf.update_layout(title="Volume and Feed Flow Evolution")

        # Confidence Intervals
        ci_section = []
        cov_matrix = None
        parametros_df = pd.DataFrame({
            'Parameter': param_names,
            'Optimized Value': params_opt,
            'Units': units
        })

        try:
            residuals = y_exp - y_pred_final
            residuals_flat = residuals.flatten()
            residuals_flat_clean = residuals_flat[~np.isnan(residuals_flat)]
            n_obs_clean = len(residuals_flat_clean)
            n_params_ci = len(params_opt)
            dof = n_obs_clean - n_params_ci

            if dof <= 0:
                ci_section.append(dbc.Alert(
                    f"Not enough data points ({n_obs_clean}) for CI (need > {n_params_ci}).",
                    color='warning'
                ))
            else:
                jac = compute_jacobian_fedbatch(params_opt, t_exp, y0, feed_params, atol_v, rtol_v)

                if jac is None or np.isnan(jac).any() or np.isinf(jac).any():
                    ci_section.append(dbc.Alert("Invalid Jacobian – CI cannot be computed.", color='warning'))
                else:
                    mse = np.sum(residuals_flat_clean ** 2) / dof
                    jtj = jac.T @ jac
                    cov_matrix = np.linalg.pinv(jtj) * mse
                    diag_cov = np.diag(cov_matrix)
                    valid_variance = diag_cov > 1e-15
                    std_errors = np.full_like(diag_cov, np.nan)
                    std_errors[valid_variance] = np.sqrt(diag_cov[valid_variance])
                    alpha = 0.05
                    t_val = t.ppf(1.0 - alpha / 2.0, df=dof)
                    intervals = t_val * std_errors

                    parametros_df['Standard Error'] = std_errors
                    parametros_df['Interval ± (95%)'] = intervals
                    parametros_df['95% CI Lower'] = np.where(
                        np.isnan(intervals), np.nan, parametros_df['Optimized Value'] - intervals
                    )
                    parametros_df['95% CI Upper'] = np.where(
                        np.isnan(intervals), np.nan, parametros_df['Optimized Value'] + intervals
                    )

                    ci_table_cols = ['Parameter', 'Optimized Value', 'Units',
                                     'Standard Error', 'Interval ± (95%)', '95% CI Lower', '95% CI Upper']
                    ci_table_data = []
                    for _, row in parametros_df.iterrows():
                        ci_table_data.append({
                            col: (f"{row[col]:.5f}" if isinstance(row[col], float) else row[col])
                            for col in ci_table_cols
                        })

                    ci_section.append(dbc.Alert("✅ Confidence Intervals Calculated.", color='success'))
                    ci_section.append(html.H6("Optimized Parameters and 95% Confidence Intervals"))
                    ci_section.append(dash_table.DataTable(
                        data=ci_table_data,
                        columns=[{'name': c, 'id': c} for c in ci_table_cols],
                        style_table={'overflowX': 'auto'},
                        style_cell={'fontSize': '12px'},
                    ))

                    # CI bar chart
                    errors_for_plot = parametros_df['Interval ± (95%)'].copy().fillna(0).values
                    fig_ci = go.Figure(go.Bar(
                        y=param_names,
                        x=parametros_df['Optimized Value'].fillna(0).values,
                        error_x=dict(type='data', array=errors_for_plot),
                        orientation='h',
                        marker_color='#1f77b4'
                    ))
                    fig_ci.update_layout(
                        title='95% Confidence Intervals for Parameters',
                        xaxis_title='Parameter Value',
                    )
                    ci_section.append(dcc.Graph(figure=fig_ci))

        except Exception as e:
            ci_section.append(dbc.Alert(
                f"Error calculating CI: {e}", color='warning'
            ))

        # Residuals histograms
        fig_hist = make_subplots(rows=1, cols=3,
                                 subplot_titles=['Biomass Residuals', 'Substrate Residuals', 'Product Residuals'])
        hist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(3):
            res = y_exp[i] - y_pred_final[i]
            res_clean = res[~np.isnan(res)]
            if len(res_clean) > 1:
                fig_hist.add_trace(
                    go.Histogram(x=res_clean, name=['Biomass', 'Substrate', 'Product'][i],
                                 marker_color=hist_colors[i], opacity=0.75),
                    row=1, col=i + 1
                )
        fig_hist.update_layout(title='Residuals Distribution', showlegend=False)

        # Correlation heatmap
        heatmap_section = []
        if cov_matrix is not None and not np.isnan(cov_matrix).all():
            try:
                std_devs = np.sqrt(np.diag(cov_matrix))
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_matrix_calc = cov_matrix / np.outer(std_devs, std_devs)
                corr_matrix_calc[~np.isfinite(corr_matrix_calc)] = np.nan
                np.fill_diagonal(corr_matrix_calc, 1.0)
                corr_matrix_calc = np.clip(corr_matrix_calc, -1.0, 1.0)

                fig_heatmap = go.Figure(go.Heatmap(
                    z=corr_matrix_calc,
                    x=param_names, y=param_names,
                    colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                    text=np.round(corr_matrix_calc, 2),
                    texttemplate='%{text}',
                ))
                fig_heatmap.update_layout(title='Parameter Correlation Matrix')
                heatmap_section = [html.H6("📌 Parameter Correlation Matrix"), dcc.Graph(figure=fig_heatmap)]
            except Exception:
                pass

        # Assemble results
        return html.Div([
            dbc.Alert(f"✅ Optimization complete. Final RMSE: {result.fun:.6f}", color='success'),

            html.H5("📋 Optimized Parameters"),
            dash_table.DataTable(
                data=params_table_data,
                columns=[{'name': c, 'id': c} for c in ['Parameter', 'Optimized Value', 'Units']],
                style_table={'overflowX': 'auto'},
                style_cell={'fontSize': '13px'},
            ),

            html.H5("📊 Fit Metrics", className='mt-3'),
            dash_table.DataTable(
                data=metricas_list,
                columns=[{'name': c, 'id': c} for c in ['Variable', 'R²', 'RMSE']],
                style_table={'overflowX': 'auto'},
                style_cell={'fontSize': '13px'},
            ),

            html.H5("📈 Model vs Experimental", className='mt-3'),
            dcc.Graph(figure=fig_cmp),

            html.H5("💧 Volume and Feed Flow", className='mt-3'),
            dcc.Graph(figure=fig_vf),

            html.H5("📐 Statistical Analysis", className='mt-3'),
            *ci_section,

            html.H5("📉 Residuals Analysis", className='mt-3'),
            dcc.Graph(figure=fig_hist),

            *heatmap_section,
        ])
