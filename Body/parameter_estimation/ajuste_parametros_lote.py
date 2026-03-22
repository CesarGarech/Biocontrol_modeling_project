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

PAGE_ID = 'ajuste_lote'


def modelo_ode(t, y, params):
    X, S, P, O2 = y
    mumax, Ks, Yxs, Kd, Ypx = params
    mu = mumax * S / (Ks + S)
    dXdt = mu * X - Kd * X
    dSdt = -(mu / Yxs) * X
    dPdt = Ypx * mu * X
    dO2dt = 0
    return [dXdt, dSdt, dPdt, dO2dt]


def compute_jacobian(params_opt, t_exp, y_exp, X0, S0, P0, O0, atol, rtol):
    delta = 1e-6
    jac = []
    sol_nominal = solve_ivp(
        modelo_ode, [0, t_exp[-1]], [X0, S0, P0, O0],
        args=(params_opt,), t_eval=t_exp, atol=atol, rtol=rtol
    )
    y_nominal = np.vstack([sol_nominal.y[0], sol_nominal.y[1], sol_nominal.y[2]])
    for i in range(len(params_opt)):
        params_perturbed = np.array(params_opt, dtype=float)
        params_perturbed[i] += delta
        sol_perturbed = solve_ivp(
            modelo_ode, [0, t_exp[-1]], [X0, S0, P0, O0],
            args=(params_perturbed,), t_eval=t_exp, atol=atol, rtol=rtol
        )
        y_perturbed = np.vstack([sol_perturbed.y[0], sol_perturbed.y[1], sol_perturbed.y[2]])
        derivative = (y_perturbed - y_nominal) / delta
        jac.append(derivative.flatten())
    return np.array(jac).T


def get_params_layout():
    def _label(text):
        return html.Label(text, className='text-white small')

    def _input(id_suffix, value, step=0.01, min_val=None, max_val=None):
        props = {
            'id': f'{PAGE_ID}-{id_suffix}',
            'type': 'number',
            'value': value,
            'step': step,
            'className': 'form-control form-control-sm mb-2',
        }
        if min_val is not None:
            props['min'] = min_val
        if max_val is not None:
            props['max'] = max_val
        return dcc.Input(**props)

    return html.Div([
        html.P("⚙️ Adjustment Settings", className='text-white fw-bold mb-2'),

        html.P("Parameters to Optimize", className='text-white small fw-bold mb-1'),
        _label("Initial μmax [1/h]"),
        _input('mumax', 0.5, min_val=0.01, max_val=2.0),
        _label("Initial Ks [g/L]"),
        _input('ks', 0.2, min_val=0.01, max_val=5.0),
        _label("Initial Yxs [g/g]"),
        _input('yxs', 0.5, min_val=0.1, max_val=1.0),
        _label("Initial Kd [1/h]"),
        _input('kd', 0.01, min_val=0.0, max_val=0.5),
        _label("Initial Ypx [g/g]"),
        _input('ypx', 0.3, min_val=0.1, max_val=1.0),

        html.Hr(className='border-secondary'),
        html.P("Initial Conditions", className='text-white small fw-bold mb-1'),
        _label("Initial Biomass [g/L]"),
        _input('x0', 1.0, min_val=0.1, max_val=10.0),
        _label("Initial Substrate [g/L]"),
        _input('s0', 20.0, min_val=0.1, max_val=100.0),
        _label("Initial Product [g/L]"),
        _input('p0', 0.0, min_val=0.0, max_val=50.0),
        _label("Initial O2 [mg/L]"),
        _input('o0', 8.0, min_val=0.0, max_val=10.0),

        html.Hr(className='border-secondary'),
        html.P("Solver Tolerances", className='text-white small fw-bold mb-1'),
        _label("Absolute Tolerance (atol)"),
        _input('atol', 1e-6, step=1e-8, min_val=1e-10, max_val=1e-2),
        _label("Relative Tolerance (rtol)"),
        _input('rtol', 1e-6, step=1e-8, min_val=1e-10, max_val=1e-2),

        html.Hr(className='border-secondary'),
        html.P("Optimization Options", className='text-white small fw-bold mb-1'),
        _label("Optimization Method"),
        dcc.Dropdown(
            id=f'{PAGE_ID}-method',
            options=[
                {'label': 'L-BFGS-B', 'value': 'L-BFGS-B'},
                {'label': 'Nelder-Mead', 'value': 'Nelder-Mead'},
                {'label': 'Differential Evolution', 'value': 'differential_evolution'},
            ],
            value='L-BFGS-B',
            clearable=False,
            className='mb-2',
        ),
        _label("Maximum Iterations"),
        _input('max-iter', 100, step=10, min_val=10, max_val=1000),

        html.Hr(className='border-secondary'),
        html.Button(
            "🚀 Run Adjustment",
            id=f'{PAGE_ID}-run-btn',
            n_clicks=0,
            className='btn btn-success w-100',
        ),
    ])


def get_content_layout():
    return html.Div([
        dcc.Store(id=f'{PAGE_ID}-data-store'),
        html.H4("🔧 Kinetic Parameter Adjustment - Batch", className='mb-3'),
        dbc.Card([
            dbc.CardBody([
                html.H6("📤 Load Experimental Data", className='card-title'),
                dcc.Upload(
                    id=f'{PAGE_ID}-upload',
                    children=html.Div([
                        "Drag and Drop or ",
                        html.A("Select an Excel file (.xlsx)"),
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'marginBottom': '10px',
                    },
                    accept='.xlsx',
                    multiple=False,
                ),
                html.Div(id=f'{PAGE_ID}-preview'),
            ])
        ], className='mb-3'),
        html.Div(id=f'{PAGE_ID}-results-div'),
    ])


def register_callbacks(app):

    @app.callback(
        Output(f'{PAGE_ID}-data-store', 'data'),
        Output(f'{PAGE_ID}-preview', 'children'),
        Input(f'{PAGE_ID}-upload', 'contents'),
        State(f'{PAGE_ID}-upload', 'filename'),
        prevent_initial_call=True,
    )
    def parse_upload(contents, filename):
        if contents is None:
            return None, dbc.Alert("⏳ Please upload a data file to begin the adjustment", color='warning')

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        except Exception as exc:
            return None, dbc.Alert(f"Error reading file: {exc}", color='danger')

        required_cols = ['time', 'biomass', 'substrate', 'product']
        if not all(col in df.columns for col in required_cols):
            return None, dbc.Alert(
                f"The file must contain the column names: {', '.join(required_cols)}",
                color='danger',
            )

        preview = html.Div([
            html.P(f"✅ Loaded: {filename}", className='text-success small mb-1'),
            html.P("Data Preview (first 5 rows):", className='small mb-1'),
            dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': c, 'id': c} for c in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'fontSize': '12px', 'padding': '4px'},
                page_size=5,
            ),
        ])
        return df.to_dict('records'), preview

    @app.callback(
        Output(f'{PAGE_ID}-results-div', 'children'),
        Input(f'{PAGE_ID}-run-btn', 'n_clicks'),
        State(f'{PAGE_ID}-mumax', 'value'),
        State(f'{PAGE_ID}-ks', 'value'),
        State(f'{PAGE_ID}-yxs', 'value'),
        State(f'{PAGE_ID}-kd', 'value'),
        State(f'{PAGE_ID}-ypx', 'value'),
        State(f'{PAGE_ID}-x0', 'value'),
        State(f'{PAGE_ID}-s0', 'value'),
        State(f'{PAGE_ID}-p0', 'value'),
        State(f'{PAGE_ID}-o0', 'value'),
        State(f'{PAGE_ID}-atol', 'value'),
        State(f'{PAGE_ID}-rtol', 'value'),
        State(f'{PAGE_ID}-method', 'value'),
        State(f'{PAGE_ID}-max-iter', 'value'),
        State(f'{PAGE_ID}-data-store', 'data'),
        prevent_initial_call=True,
    )
    def run_fit(n_clicks, mumax_guess, ks_guess, yxs_guess, kd_guess, ypx_guess,
                x0, s0, p0, o0, atol, rtol, method, max_iter, store_data):

        if not store_data:
            return dbc.Alert("⏳ Please upload a data file before running the adjustment.", color='warning')

        # Coerce inputs to float with safe defaults
        mumax_guess = float(mumax_guess or 0.5)
        ks_guess    = float(ks_guess    or 0.2)
        yxs_guess   = float(yxs_guess   or 0.5)
        kd_guess    = float(kd_guess    or 0.01)
        ypx_guess   = float(ypx_guess   or 0.3)
        X0  = float(x0   or 1.0)
        S0  = float(s0   or 20.0)
        P0  = float(p0   or 0.0)
        O0  = float(o0   or 8.0)
        atol     = float(atol     or 1e-6)
        rtol     = float(rtol     or 1e-6)
        max_iter = int(max_iter   or 100)

        df_exp = pd.DataFrame(store_data)
        t_exp  = df_exp['time'].values
        y_exp  = df_exp[['biomass', 'substrate', 'product']].values.T

        def objetivo(params, t_exp, y_exp):
            try:
                sol = solve_ivp(
                    modelo_ode, [0, t_exp[-1]], [X0, S0, P0, O0],
                    args=(params,), t_eval=t_exp, atol=atol, rtol=rtol
                )
                y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])
                return np.sqrt(np.nanmean((y_pred - y_exp) ** 2))
            except Exception:
                return 1e6

        bounds        = [(0.01, 2), (0.01, 5), (0.1, 1), (0, 0.5), (0.1, 1)]
        initial_guess = [mumax_guess, ks_guess, yxs_guess, kd_guess, ypx_guess]

        try:
            if method == 'differential_evolution':
                result = differential_evolution(objetivo, bounds, args=(t_exp, y_exp))
            else:
                result = minimize(
                    objetivo, initial_guess, args=(t_exp, y_exp),
                    method=method, bounds=bounds, options={'maxiter': max_iter}
                )
        except Exception as exc:
            return dbc.Alert(f"Optimization failed: {exc}", color='danger')

        params_opt = result.x

        # Final simulation
        sol = solve_ivp(
            modelo_ode, [0, t_exp[-1]], [X0, S0, P0, O0],
            args=(params_opt,), t_eval=t_exp, atol=atol, rtol=rtol
        )
        y_pred = np.vstack([sol.y[0], sol.y[1], sol.y[2]])

        # --- Parameters table ---
        param_names = ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx']
        param_units = ['1/h', 'g/L', 'g/g', '1/h', 'g/g']
        df_params = pd.DataFrame({
            'Parameter': param_names,
            'Value': [round(v, 4) for v in params_opt],
            'Units': param_units,
        })

        # --- Metrics table ---
        variables = ['Biomass', 'Substrate', 'Product']
        df_metrics = pd.DataFrame({
            'Variable': variables,
            'R²':   [round(r2_score(y_exp[i], y_pred[i]), 4) for i in range(3)],
            'RMSE': [round(float(np.sqrt(mean_squared_error(y_exp[i], y_pred[i]))), 4) for i in range(3)],
        })

        # --- Confidence intervals ---
        residuals      = y_exp - y_pred
        residuals_flat = residuals.flatten()
        jac = compute_jacobian(params_opt, t_exp, y_exp, X0, S0, P0, O0, atol, rtol)
        try:
            cov_matrix = (
                np.linalg.pinv(jac.T @ jac)
                * (residuals_flat @ residuals_flat)
                / (len(residuals_flat) - len(params_opt))
            )
            std_errors = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            std_errors = np.full(len(params_opt), np.nan)

        t_val     = t.ppf(0.975, df=max(len(residuals_flat) - len(params_opt), 1))
        intervals = t_val * std_errors

        df_ci = pd.DataFrame({
            'Parameter':         param_names,
            'Value':             [round(v, 4) for v in params_opt],
            'Interval ±':        [round(v, 4) for v in intervals],
            '95% CI Lower':      [round(v - ci, 4) for v, ci in zip(params_opt, intervals)],
            '95% CI Upper':      [round(v + ci, 4) for v, ci in zip(params_opt, intervals)],
        })

        # ── Plot 1: Comparison (3 stacked subplots) ──────────────────────────
        fig_compare = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=variables,
            vertical_spacing=0.08,
        )
        colors_exp = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (var, color) in enumerate(zip(variables, colors_exp), start=1):
            fig_compare.add_trace(
                go.Scatter(x=t_exp, y=y_exp[i - 1], mode='markers',
                           name=f'{var} Exp', marker=dict(color=color)),
                row=i, col=1,
            )
            fig_compare.add_trace(
                go.Scatter(x=sol.t, y=sol.y[i - 1], mode='lines',
                           name=f'{var} Model', line=dict(color=color, dash='dash')),
                row=i, col=1,
            )
        fig_compare.update_layout(
            height=700, title_text="Model vs Experimental Data", showlegend=True
        )

        # ── Plot 2: CI horizontal bar chart ──────────────────────────────────
        fig_ci = go.Figure(go.Bar(
            x=params_opt,
            y=param_names,
            orientation='h',
            error_x=dict(type='data', array=intervals, visible=True, color='#ff7f0e'),
            marker_color='#1f77b4',
        ))
        fig_ci.update_layout(
            title='95% Confidence Intervals',
            xaxis_title='Parameter Value',
            height=350,
        )

        # ── Plot 3: Residual histograms (1×3) ────────────────────────────────
        hist_colors = ['#2ca02c', '#9467bd', '#d62728']
        fig_hist = make_subplots(rows=1, cols=3, subplot_titles=variables)
        for i, (var, color) in enumerate(zip(variables, hist_colors), start=1):
            res = y_exp[i - 1] - y_pred[i - 1]
            fig_hist.add_trace(
                go.Histogram(x=res, name=f'Residuals {var}', marker_color=color,
                             opacity=0.75),
                row=1, col=i,
            )
        fig_hist.update_layout(
            title_text='Residual Distributions', height=350, showlegend=False
        )

        # ── Plot 4: Correlation heatmap ───────────────────────────────────────
        corr_matrix = pd.DataFrame(
            jac, columns=param_names
        ).corr().values
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix,
            x=param_names,
            y=param_names,
            colorscale='RdBu',
            zmid=0,
            text=[[f'{v:.2f}' for v in row] for row in corr_matrix],
            texttemplate='%{text}',
        ))
        fig_corr.update_layout(title='Parameter Correlation Matrix', height=400)

        # ── Assemble output ───────────────────────────────────────────────────
        def _datatable(df, id_suffix):
            return dash_table.DataTable(
                id=f'{PAGE_ID}-{id_suffix}',
                data=df.to_dict('records'),
                columns=[{'name': c, 'id': c} for c in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'fontSize': '13px', 'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                page_size=10,
            )

        return html.Div([
            dbc.Alert(f"✅ Final RMSE: {result.fun:.4f}", color='success'),

            html.H5("📊 Optimized Parameters"),
            _datatable(df_params, 'params-table'),

            html.H5("📈 Goodness-of-Fit Metrics", className='mt-3'),
            _datatable(df_metrics, 'metrics-table'),

            html.H5("📐 Confidence Intervals (95%)", className='mt-3'),
            _datatable(df_ci, 'ci-table'),

            html.H5("📉 Model vs Experimental", className='mt-3'),
            dcc.Graph(id=f'{PAGE_ID}-compare-graph', figure=fig_compare),

            html.H5("📐 Parameter Confidence Intervals", className='mt-3'),
            dcc.Graph(id=f'{PAGE_ID}-ci-graph', figure=fig_ci),

            html.H5("📉 Residual Distributions", className='mt-3'),
            dcc.Graph(id=f'{PAGE_ID}-hist-graph', figure=fig_hist),

            html.H5("📌 Parameter Correlation Matrix", className='mt-3'),
            dcc.Graph(id=f'{PAGE_ID}-corr-graph', figure=fig_corr),
        ])