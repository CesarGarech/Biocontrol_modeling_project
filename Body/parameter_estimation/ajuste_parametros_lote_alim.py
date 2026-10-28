import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
from .ci_band import model_confidence_band, draw_band
import openpyxl 
import seaborn as sns
import traceback 

#--------------------------------------------------------------------------
# Funciones Auxiliares para Estrategias de Alimentación
#--------------------------------------------------------------------------
def calculate_feed_rate(t, strategy, t_start, t_end, F_min, F_max, V0=None, mu_set=None, Xs_f_ratio=None, Sf=None, step_data=None, X_current=None, V_current=None):
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
                 return F_max if abs(t-t_start) < 1e-9 else 0.0
             else:
                 return F_min if F_min > 1e-9 else 0.0
        k = np.log(F_max / F_min) / (t_end - t_start)
        F = F_min * np.exp(k * (t - t_start))
        return max(F_min, min(F, F_max))
    elif strategy == 'Exponential (constant mu)':
        if mu_set is None or X_current is None or V_current is None or Sf is None or Xs_f_ratio is None or Sf <= 1e-9 or Xs_f_ratio <= 1e-9 or mu_set < 0:
             return 0.0
        Yxs_est = float(Xs_f_ratio)
        if Yxs_est <= 1e-9 or Sf <= 1e-9 or X_current < 0 or V_current < 0:
            return 0.0
        F = (mu_set / Yxs_est) * X_current * V_current / Sf
        return max(F_min, min(F, F_max))
    elif strategy == 'Step':
        if not step_data:
            return 0.0
        step_data.sort(key=lambda item: item[0])
        current_flow = 0.0
        last_step_time = t_start
        flow_in_interval = 0.0
        found_interval = False
        for i, (step_time, flow_value) in enumerate(step_data):
            step_time = float(step_time)
            flow_value = float(flow_value)
            if t >= last_step_time - 1e-9 and t < step_time - 1e-9:
                 current_flow = flow_in_interval
                 found_interval = True
                 break
            if step_time >= t_start - 1e-9:
                 flow_in_interval = flow_value
            last_step_time = step_time
        if not found_interval and t >= last_step_time - 1e-9 :
             current_flow = flow_in_interval
        if t < t_start - 1e-9 or t > t_end + 1e-9:
            return 0.0
        else:
            return max(F_min, min(current_flow, F_max))
    else:
        st.error(f"Unknown feeding strategy: {strategy}")
        return 0.0

#--------------------------------------------------------------------------
# Modelo ODE para Lote Alimentado
#--------------------------------------------------------------------------
def modelo_ode_fedbatch(t, y, params, feed_params):
    try:
        X, S, P, O2, V = y
        if len(params) != 6:
             raise ValueError(f"6 parameters were expected, only {len(params)} received")
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
                                V0=V0, mu_set=mu_set, Xs_f_ratio=Yxs_param_for_F, Sf=Sf,
                                step_data=step_data, X_current=X, V_current=V)

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

        denominator = Ks_safe + S_safe + (S_safe**2 / Ksi_safe)
        if denominator <= 1e-12: mu = 0.0
        else: mu = mumax_safe * S_safe / denominator
        mu = max(mu, 0.0) 

        dXdt = mu * X_safe - Kd_safe * X_safe - D * X + D * Xf
        dSdt = - (mu / Yxs_safe) * X_safe + D * (Sf - S)
        dPdt = Ypx_safe * mu * X_safe - D * P + D * Pf
        dO2dt = 0 
        dVdt = F

        return [dXdt, dSdt, dPdt, dO2dt, dVdt]

    except Exception as e:
        st.error(f"Error in modelo_ode_fedbatch: {e}")
        raise

#--------------------------------------------------------------------------
# Jacobian function
#--------------------------------------------------------------------------
def compute_jacobian_fedbatch(params_opt, t_exp, y0_fit, feed_params, atol, rtol):
    delta = 1e-7
    n_params = len(params_opt)
    if n_params != 6:
         st.error(f"Error in Jacobian: 6 parameters were expected")
         return None 

    n_times = len(t_exp)
    n_vars_measured = 3
    y_nominal_flat = np.zeros(n_times * n_vars_measured)

    try:
        sol_nominal = solve_ivp(modelo_ode_fedbatch, [t_exp[0], t_exp[-1]],
                                y0_fit, args=(params_opt, feed_params), 
                                t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
        if sol_nominal.status != 0:
             return np.full((n_times * n_vars_measured, n_params), np.nan)
        y_nominal_flat = sol_nominal.y[0:n_vars_measured, :].flatten()
    except Exception as e:
         return np.full((n_times * n_vars_measured, n_params), np.nan)

    jac = np.zeros((n_times * n_vars_measured, n_params))
    for i in range(n_params): 
        params_perturbed = np.array(params_opt, dtype=float)
        h = delta * abs(params_perturbed[i]) if abs(params_perturbed[i]) > 1e-8 else delta
        if h == 0:
            jac[:, i] = 0.0; continue
        params_perturbed[i] += h

        try:
            sol_perturbed = solve_ivp(modelo_ode_fedbatch, [t_exp[0], t_exp[-1]],
                                      y0_fit, args=(params_perturbed, feed_params), 
                                      t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
            if sol_perturbed.status != 0:
                jac[:, i] = np.nan; continue
            y_perturbed_flat = sol_perturbed.y[0:n_vars_measured, :].flatten()
            derivative = (y_perturbed_flat - y_nominal_flat) / h
            jac[:, i] = derivative
        except Exception as e:
            jac[:, i] = np.nan

    return jac

#--------------------------------------------------------------------------
# Función Objetivo
#--------------------------------------------------------------------------
def objetivo_fedbatch(params, t_exp, y_exp_stacked, y0_fit, feed_params, atol, rtol):
    if len(params) != 6: return 1e18 
    try:
        sol = solve_ivp(modelo_ode_fedbatch, [t_exp[0], t_exp[-1]], y0_fit,
                        args=(params, feed_params), t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
        if sol.status != 0: return 1e6 + np.sum(np.abs(params))

        y_pred = sol.y[0:3, :]
        y_pred_stacked = y_pred.flatten()
        if y_exp_stacked.shape != y_pred_stacked.shape: return 1e11
        
        mask = ~np.isnan(y_exp_stacked)
        if np.sum(mask) == 0: return 1e12
        sse = np.sum((y_pred_stacked[mask] - y_exp_stacked[mask])**2)
        rmse = np.sqrt(sse / np.sum(mask))

        if np.any(sol.y[0:3,:] < -1e-3):
             neg_penalty = np.sum(np.abs(sol.y[0:3,:][sol.y[0:3,:] < -1e-3]))
             rmse += neg_penalty * 1e3
        if np.any(sol.y[4,:] < 0): rmse += 1e5
        if np.isnan(rmse) or np.isinf(rmse): return 1e15
        return rmse
    except Exception as e:
        return 1e15

#--------------------------------------------------------------------------
# Página de Streamlit
#--------------------------------------------------------------------------
def ajuste_parametros_fedbatch_page():
    st.header("🔧 Adjustment of Kinetic Parameters (Fed-Batch with Substrate Inhibition)")

    if 'params_opt_fedbatch' not in st.session_state: st.session_state.params_opt_fedbatch = None 
    if 'result_fedbatch' not in st.session_state: st.session_state.result_fedbatch = None
    if 'parametros_df_fedbatch' not in st.session_state: st.session_state.parametros_df_fedbatch = None
    if 'sol_fedbatch' not in st.session_state: st.session_state.sol_fedbatch = None
    if 'df_exp_fedbatch' not in st.session_state: st.session_state.df_exp_fedbatch = None
    if 'y0_fit_fedbatch' not in st.session_state: st.session_state.y0_fit_fedbatch = None
    if 'feed_params_fedbatch' not in st.session_state: st.session_state.feed_params_fedbatch = {}
    if 't_exp_fedbatch' not in st.session_state: st.session_state.t_exp_fedbatch = None
    if 'y_exp_fedbatch' not in st.session_state: st.session_state.y_exp_fedbatch = None
    if 'y_exp_stacked_fedbatch' not in st.session_state: st.session_state.y_exp_stacked_fedbatch = None
    # --- MODIFICACIÓN: Estado de variables estándar ---
    if 'y_exp_std_fedbatch' not in st.session_state: st.session_state.y_exp_std_fedbatch = None
    # ---------------------------------------------------
    if 'run_complete_fedbatch' not in st.session_state: st.session_state.run_complete_fedbatch = False
    if 'last_uploaded_filename' not in st.session_state: st.session_state.last_uploaded_filename = None

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload experimental data")
        uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], key="fedbatch_uploader_inhib") 

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_uploaded_filename:
                 st.session_state.params_opt_fedbatch = None
                 st.session_state.result_fedbatch = None
                 st.session_state.run_complete_fedbatch = False
                 st.session_state.last_uploaded_filename = uploaded_file.name
            try:
                df_exp = pd.read_excel(uploaded_file, engine='openpyxl')
                required_cols = ['time', 'biomass', 'substrate', 'product']
                if not all(col in df_exp.columns for col in required_cols):
                    st.error(f"File must contain the column names: {', '.join(required_cols)}")
                    st.session_state.df_exp_fedbatch = None
                    st.stop()
                else:
                    df_exp = df_exp.sort_values(by='time').reset_index(drop=True)
                    df_exp[['biomass', 'substrate', 'product']] = df_exp[['biomass', 'substrate', 'product']].apply(pd.to_numeric, errors='coerce')
                    st.session_state.df_exp_fedbatch = df_exp
                    st.session_state.t_exp_fedbatch = df_exp['time'].values
                    st.session_state.y_exp_fedbatch = df_exp[['biomass', 'substrate', 'product']].values.T
                    st.session_state.y_exp_stacked_fedbatch = st.session_state.y_exp_fedbatch.flatten()
                    
                    # --- MODIFICACIÓN: Extraer desviaciones estándar de fed-batch ---
                    std_cols = ['biomass_std', 'substrate_std', 'product_std']
                    y_exp_std = []
                    for col in std_cols:
                        if col in df_exp.columns:
                            y_exp_std.append(df_exp[col].fillna(0).values)
                        else:
                            y_exp_std.append(np.zeros(len(df_exp)))
                    st.session_state.y_exp_std_fedbatch = np.array(y_exp_std)
                    # ----------------------------------------------------------------
                    
                    st.dataframe(df_exp.head()) 
            except Exception as e:
                st.error(f"Error reading the Excel file: {e}")
                st.session_state.df_exp_fedbatch = None 
                uploaded_file = None
        elif st.session_state.df_exp_fedbatch is not None:
             st.info(f"Using the previously uploaded data from '{st.session_state.last_uploaded_filename}'.")

        if st.session_state.df_exp_fedbatch is not None:
            df_exp = st.session_state.df_exp_fedbatch
            t_exp = st.session_state.t_exp_fedbatch
            st.subheader("⚙️ Process Configuration")

            st.markdown("##### Reactor Initial Conditions")
            X0_default = df_exp['biomass'].iloc[0] if pd.notna(df_exp['biomass'].iloc[0]) else 0.1
            S0_default = df_exp['substrate'].iloc[0] if pd.notna(df_exp['substrate'].iloc[0]) else 10.0
            P0_default = df_exp['product'].iloc[0] if pd.notna(df_exp['product'].iloc[0]) else 0.0
            O0_default = 8.0; V0_default = 1.0
            X0_fit = st.number_input("Initial Biomass (X0) [g/L]", min_value=0.0, value=float(X0_default), format="%.4f")
            S0_fit = st.number_input("Initial Substrate (S0) [g/L]", min_value=0.0, value=float(S0_default), format="%.4f")
            P0_fit = st.number_input("Initial Product (P0) [g/L]", min_value=0.0, value=float(P0_default), format="%.4f")
            O0_fit = st.number_input("Initial O2 [mg/L]", min_value=0.0, max_value=20.0, value=float(O0_default), format="%.4f")
            V0_fit = st.number_input("Initial Volume (V0) [L]", min_value=1e-3, value=float(V0_default), format="%.4f")
            st.session_state.y0_fit_fedbatch = [X0_fit, S0_fit, P0_fit, O0_fit, V0_fit]

            st.markdown("##### Feeding Configuration")
            feed_strategy = st.selectbox("Feeding Strategy", ['Constant', 'Linear', 'Exponential (Simple)', 'Exponential (constant mu)', 'Step'], key="feed_strategy_inhib")
            t_max_exp = float(t_exp[-1]) if t_exp is not None and len(t_exp) > 0 else 100.0
            t_min_exp = float(t_exp[0]) if t_exp is not None and len(t_exp) > 0 else 0.0
            t_start_feed = st.number_input("Start Feeding Time [h]", min_value=t_min_exp, max_value=t_max_exp, value=max(t_min_exp, 0.0), step = 0.1, format="%.2f")
            t_end_feed_default = max(float(t_start_feed), t_max_exp)
            t_end_feed = st.number_input("End Feeding Time [h]", min_value=float(t_start_feed), max_value=t_max_exp + 1.0, value=t_end_feed_default, step = 0.1, format="%.2f")
            F_min_feed = st.number_input("Minimum Flow (F_min) [L/h]", min_value=0.0, max_value=10.0, value=0.0, step=0.001, format="%.4f")
            F_max_default = max(float(F_min_feed), 0.1)
            F_max_feed = st.number_input("Maximum Flow (F_max) [L/h]", min_value=float(F_min_feed), max_value=10.0, value=F_max_default, step=0.001, format="%.4f")
            Sf_feed = st.number_input("Substrate Feed Concentration (Sf) [g/L]", min_value=0.0, max_value=1000.0, value=100.0, format="%.2f")
            Xf_feed = st.number_input("Biomass Feed Concentration (Xf) [g/L]", min_value=0.0, max_value=50.0, value=0.0, format="%.2f")
            Pf_feed = st.number_input("Product Feed Concentration (Pf) [g/L]", min_value=0.0, max_value=50.0, value=0.0, format="%.2f")

            step_data_input_list = None; mu_set_feed = None
            if feed_strategy == 'Step':
                step_data_input_str = st.text_area("Format: one 'time, flow' pair per line.", height=100, key="step_data_text_inhib")
                step_data_input_list = []
                if step_data_input_str:
                    lines = step_data_input_str.strip().split('\n')
                    for i, line in enumerate(lines):
                         try:
                             parts = line.split(',')
                             if len(parts) == 2: step_data_input_list.append((float(parts[0].strip()), float(parts[1].strip())))
                         except ValueError: pass
            elif feed_strategy == 'Exponential (constant mu)':
                mu_set_feed = st.number_input("Desired Specific Growth Rate (μ_set) [1/h]", min_value=0.0, max_value=2.0, value=0.1, format="%.4f")

            st.session_state.feed_params_fedbatch = {'strategy': feed_strategy, 't_start': t_start_feed, 't_end': t_end_feed, 'F_min': F_min_feed, 'F_max': F_max_feed, 'Sf': Sf_feed, 'Xf': Xf_feed, 'Pf': Pf_feed, 'step_data': step_data_input_list, 'mu_set': mu_set_feed, 'V0': V0_fit}

            st.subheader("⚙️ Adjustment Configuration")
            p_opt = st.session_state.params_opt_fedbatch 
            mumax_guess_val = float(p_opt[0]) if p_opt is not None and len(p_opt)>0 else 0.5
            Ks_guess_val = float(p_opt[1]) if p_opt is not None and len(p_opt)>1 else 0.2
            Yxs_guess_val = float(p_opt[2]) if p_opt is not None and len(p_opt)>2 else 0.5
            Kd_guess_val = float(p_opt[3]) if p_opt is not None and len(p_opt)>3 else 0.01
            Ypx_guess_val = float(p_opt[4]) if p_opt is not None and len(p_opt)>4 else 0.3
            Ksi_guess_val = float(p_opt[5]) if p_opt is not None and len(p_opt)>5 else 100.0 

            mumax_guess = st.number_input("μmax [1/h]", 0.0, 5.0, mumax_guess_val, format="%.4f")
            Ks_guess = st.number_input("Ks [g/L]", 1e-4, 20.0, Ks_guess_val, format="%.4f")
            Yxs_guess = st.number_input("Yxs [g/g]", 0.01, 3.0, Yxs_guess_val, format="%.4f")
            Kd_guess = st.number_input("Kd [1/h]", 0.0, 1.0, Kd_guess_val, format="%.4f")
            Ypx_guess = st.number_input("Ypx [g/g]", 0.0, 10.0, Ypx_guess_val, format="%.4f")
            Ksi_guess = st.number_input("Ksi (Substrate Inhib.) [g/L]", 1.0, 1000.0, Ksi_guess_val, format="%.2f")

            bounds = [(1e-3, 5.0), (1e-4, 20.0), (0.01, 3.0), (0.0, 1.0), (0.0, 10.0), (1.0, 1000.0)]
            atol = st.number_input("Absolute tolerance (atol)", min_value=1e-12, max_value=1e-3, value=1e-8, format="%e")
            rtol = st.number_input("Relative tolerance (rtol)", min_value=1e-12, max_value=1e-3, value=1e-8, format="%e")
            metodo = st.selectbox("Optimization Methods", ['L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell', 'differential_evolution'], index=0, key="opt_method_fedbatch_inhib")
            max_iter = st.number_input("Maximum iterations", 10, 10000, 500)

            if st.button("🚀 Run Adjustment (with Substrate Inhib.)", key="run_fedbatch_fit_inhib"):
                if st.session_state.y_exp_stacked_fedbatch is None or st.session_state.t_exp_fedbatch is None:
                    st.error("The adjustment cannot be executed. Experimental data are missing.")
                else:
                    with st.spinner("Optimizing parameters (model with inhibition)..."):
                        initial_guess = [mumax_guess, Ks_guess, Yxs_guess, Kd_guess, Ypx_guess, Ksi_guess]
                        y0_run = st.session_state.y0_fit_fedbatch
                        feed_params_run = st.session_state.feed_params_fedbatch
                        t_exp_run = st.session_state.t_exp_fedbatch
                        y_exp_stacked_run = st.session_state.y_exp_stacked_fedbatch
                        result = None; st.session_state.run_complete_fedbatch = False

                        try:
                            if metodo == 'differential_evolution':
                                result = differential_evolution(objetivo_fedbatch, bounds, args=(t_exp_run, y_exp_stacked_run, y0_run, feed_params_run, atol, rtol), maxiter=max_iter, tol=1e-7, mutation=(0.5, 1.5), recombination=0.8, strategy='best1bin', updating='deferred', workers=-1, seed=42)
                            else:
                                 minimizer_kwargs = {"args": (t_exp_run, y_exp_stacked_run, y0_run, feed_params_run, atol, rtol), "method": metodo, "bounds": bounds if metodo in ['L-BFGS-B', 'TNC', 'SLSQP'] else None, "options": {'maxiter': max_iter, 'disp': False}}
                                 result = minimize(objetivo_fedbatch, initial_guess, **minimizer_kwargs)

                            if result and hasattr(result, 'success') and result.success:
                                st.session_state.params_opt_fedbatch = result.x; st.session_state.result_fedbatch = result; st.session_state.run_complete_fedbatch = True; st.success(f"Optimization ({metodo}) completed with success.")
                            elif result and hasattr(result, 'x'):
                                 st.session_state.params_opt_fedbatch = result.x; st.session_state.result_fedbatch = result; st.session_state.run_complete_fedbatch = True
                            else: st.error(f"The optimization ({metodo}) failed.")
                        except Exception as e:
                            st.session_state.params_opt_fedbatch = None; st.session_state.result_fedbatch = None; st.session_state.run_complete_fedbatch = False
                        st.rerun()

    with col2:
        if st.session_state.run_complete_fedbatch and st.session_state.result_fedbatch is not None:
            st.subheader("📊 Adjustment Results (Model with Substrate Inhibition)")

            result = st.session_state.result_fedbatch
            params_opt = st.session_state.params_opt_fedbatch 
            y0_res = st.session_state.y0_fit_fedbatch
            feed_params_res = st.session_state.feed_params_fedbatch
            df_exp_res = st.session_state.df_exp_fedbatch
            t_exp_res = st.session_state.t_exp_fedbatch
            y_exp_res = st.session_state.y_exp_fedbatch
            
            # --- Recuperar las desviaciones estándar de fedbatch de la memoria ---
            y_exp_std_res = st.session_state.y_exp_std_fedbatch
            
            atol_res = atol; rtol_res = rtol 

            if params_opt is None or len(params_opt) != 6:
                 st.session_state.params_opt_fedbatch = None; st.session_state.result_fedbatch = None; st.session_state.run_complete_fedbatch = False; st.stop() 

            final_rmse = getattr(result, 'fun', np.nan)
            st.write(f"**Final Objective Function (RMSE):** {final_rmse:.6f}")

            param_names = ['μ_0', 'Ks', 'Yxs', 'Kd', 'Ypx', 'Ksi'] 
            param_units = ['1/h', 'g/L', 'g/g', '1/h', 'g/g', 'g/L'] 
            parametros_df = pd.DataFrame({'Parameter': param_names, 'Optimized Value': params_opt, 'Units': param_units})
            st.dataframe(parametros_df.style.format({'Optimized Value': '{:.5f}'}))
            st.session_state.parametros_df_fedbatch = parametros_df 

            st.subheader("📈 Final Simulation and Metrics")
            try:
                sol = solve_ivp(modelo_ode_fedbatch, [t_exp_res[0], t_exp_res[-1]], y0_res, args=(params_opt, feed_params_res), t_eval=t_exp_res, atol=atol_res, rtol=rtol_res, method='LSODA')
                if sol.status != 0: st.session_state.sol_fedbatch = None
                else:
                     st.session_state.sol_fedbatch = sol
                     y_pred_final = sol.y[0:3, :]
                     metricas_list = []
                     for i in range(3):
                         variable_name = ['Biomass', 'Substrate', 'Product'][i]
                         exp_data = y_exp_res[i]; pred_data = y_pred_final[i]
                         valid_mask = ~np.isnan(exp_data)
                         if np.sum(valid_mask) > 1:
                             exp_valid = exp_data[valid_mask]; pred_valid = pred_data[valid_mask]
                             try: r2 = r2_score(exp_valid, pred_valid)
                             except ValueError: r2 = np.nan
                             rmse = np.sqrt(mean_squared_error(exp_valid, pred_valid))
                             metricas_list.append({'Variable': variable_name, 'R²': r2, 'RMSE': rmse})
                     st.dataframe(pd.DataFrame(metricas_list).style.format({'R²': '{:.4f}', 'RMSE': '{:.4f}'}))
            except Exception as e: st.session_state.sol_fedbatch = None

            if st.session_state.sol_fedbatch is not None:
                st.subheader("📉 Comparative Graphs")
                sol = st.session_state.sol_fedbatch
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                variables = ['Biomass', 'Substrate', 'Product']; unidades = ['g/L', 'g/L', 'g/L']; colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                def _simulate_ci(p):
                    s_ = solve_ivp(modelo_ode_fedbatch, [t_exp_res[0], t_exp_res[-1]], y0_res, args=(p, feed_params_res), t_eval=t_exp_res, atol=atol_res, rtol=rtol_res, method='LSODA')
                    return s_.y[0:3, :]
                band = model_confidence_band(_simulate_ci, params_opt, y_exp_res[0:3, :], sol.y[0:3, :])
                
                for i in range(3):
                    # --- MODIFICACIÓN: Graficar con errorbar si existe el desvío estándar ---
                    if y_exp_std_res is not None and np.any(y_exp_std_res[i] > 0):
                        axes[i].errorbar(t_exp_res, y_exp_res[i], yerr=y_exp_std_res[i], fmt='o', markersize=5, alpha=0.7, label=f'{variables[i]} Exp.', color=colors[i], capsize=3)
                    else:
                        axes[i].plot(t_exp_res, y_exp_res[i], 'o', markersize=5, alpha=0.7, label=f'{variables[i]} Exp.', color=colors[i])
                    # ------------------------------------------------------------------------
                    axes[i].plot(sol.t, sol.y[i], '-', linewidth=2, label=f'{variables[i]} Mod.', color=colors[i])
                    if band is not None: draw_band(axes[i], sol.t, band[0][i], band[1][i], color=colors[i])
                    axes[i].set_ylabel(f"{variables[i]} [{unidades[i]}]"); axes[i].legend(); axes[i].grid(True, linestyle='--', alpha=0.6)
                
                axes[-1].set_xlabel("Time [h]")
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                st.pyplot(fig)

                st.subheader("💧 Volume and Feed Flow")
                try:
                    interp_x = np.interp(sol.t, sol.t, sol.y[0]); interp_v = np.interp(sol.t, sol.t, sol.y[4])
                    F_t = np.array([calculate_feed_rate(ti, feed_params_res['strategy'], feed_params_res['t_start'], feed_params_res['t_end'], feed_params_res['F_min'], feed_params_res['F_max'], V0=feed_params_res['V0'], mu_set=feed_params_res['mu_set'], Xs_f_ratio=params_opt[2], Sf=feed_params_res['Sf'], step_data=feed_params_res['step_data'], X_current=interp_x[idx], V_current=interp_v[idx]) for idx, ti in enumerate(sol.t)])
                    fig_vol_feed, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.set_xlabel('Time [h]'); ax1.set_ylabel('Volume [L]', color='tab:red'); ax1.plot(sol.t, sol.y[4], color='tab:red', linestyle='-', linewidth=2, label='Volume (Model)'); ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.grid(True, which='major', linestyle='--', alpha=0.7); ax1.legend(loc='upper left')
                    ax2 = ax1.twinx(); ax2.set_ylabel('Feed Flow [L/h]', color='tab:blue')
                    if feed_params_res['strategy'] in ['Constant', 'Step']: ax2.step(sol.t, F_t, where='post', color='tab:blue', linestyle='--', label=f'Flow ({feed_params_res["strategy"]})')
                    else: ax2.plot(sol.t, F_t, color='tab:blue', linestyle='--', label=f'Flow ({feed_params_res["strategy"]})')
                    ax2.tick_params(axis='y', labelcolor='tab:blue'); ax2.legend(loc='upper right'); fig_vol_feed.tight_layout(); st.pyplot(fig_vol_feed)
                except Exception: pass

                st.subheader("📈 Statistical Analysis")
                with st.spinner("Calculating confidence intervals..."):
                    try:
                        residuals = (y_exp_res - y_pred_final).flatten()
                        residuals_flat_clean = residuals[~np.isnan(residuals)]
                        dof = len(residuals_flat_clean) - len(params_opt) 
                        if dof > 0:
                            jac = compute_jacobian_fedbatch(params_opt, t_exp_res, y0_res, feed_params_res, atol_res, rtol_res)
                            if jac is not None and not np.isnan(jac).any():
                                 cov_matrix = np.linalg.pinv(jac.T @ jac) * (np.sum(residuals_flat_clean**2) / dof)
                                 std_errors = np.full_like(np.diag(cov_matrix), np.nan)
                                 std_errors[np.diag(cov_matrix) > 1e-15] = np.sqrt(np.diag(cov_matrix)[np.diag(cov_matrix) > 1e-15])
                                 intervals = t.ppf(1.0 - 0.05 / 2.0, df=dof) * std_errors
                                 parametros_df['Standard Error'] = std_errors; parametros_df['Interval ± (95%)'] = intervals
                                 parametros_df['95% CI Lower bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] - intervals)
                                 parametros_df['95% CI Upper bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] + intervals)
                    except Exception: pass
                    st.dataframe(parametros_df.style.format({'Optimized Value': '{:.5f}', 'Standard Error': '{:.5f}', 'Interval ± (95%)': '{:.5f}', '95% CI - Lower bound': '{:.5f}', '95% CI - Upper bound': '{:.5f}'}, na_rep='N/A'))
                    st.session_state.parametros_df_fedbatch = parametros_df

                    if 'Interval ± (95%)' in parametros_df.columns and parametros_df['Interval ± (95%)'].notna().any():
                        st.subheader("📐 Confidence Intervals for Parameters")
                        fig_ci, ax = plt.subplots(figsize=(10, max(4, len(parametros_df) * 0.6)))
                        y_pos = np.arange(len(parametros_df)); errors_for_plot = parametros_df['Interval ± (95%)'].fillna(0).values
                        ax.barh(y_pos, parametros_df['Optimized Value'].fillna(0), xerr=errors_for_plot, align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5, alpha=0.8)
                        ax.set_yticks(y_pos); ax.set_yticklabels(parametros_df['Parameter']); ax.invert_yaxis(); ax.grid(True, axis='x', linestyle='--', alpha=0.6)
                        plt.tight_layout(); st.pyplot(fig_ci)

                    st.subheader("📉 Residuals Analysis")
                    fig_hist, axs = plt.subplots(1, 3, figsize=(15, 5))
                    for i, (var, color) in enumerate(zip(['Biomass', 'Substrate', 'Product'], ['#1f77b4', '#ff7f0e', '#2ca02c'])):
                        res_clean = (y_exp_res[i] - y_pred_final[i])[~np.isnan(y_exp_res[i] - y_pred_final[i])]
                        if len(res_clean) > 1:
                            sns.histplot(res_clean, kde=True, color=color, ax=axs[i], bins='auto')
                            axs[i].axvline(0, color='k', linestyle='--')
                    plt.tight_layout(); st.pyplot(fig_hist)

                    if 'cov_matrix' in locals() and cov_matrix is not None:
                        st.subheader("📌 Parameter Correlation Matrix")
                        try:
                            std_devs = np.sqrt(np.diag(cov_matrix))
                            with np.errstate(divide='ignore', invalid='ignore'): corr_matrix_calc = cov_matrix / np.outer(std_devs, std_devs)
                            corr_matrix_calc[~np.isfinite(corr_matrix_calc)] = np.nan; np.fill_diagonal(corr_matrix_calc, 1.0); corr_matrix_calc = np.clip(corr_matrix_calc, -1.0, 1.0)
                            fig_corr, ax = plt.subplots(figsize=(8, 7)) 
                            sns.heatmap(pd.DataFrame(corr_matrix_calc, index=param_names, columns=param_names), annot=True, cmap='coolwarm', fmt=".2f", center=0, ax=ax, vmin=-1, vmax=1) 
                            plt.tight_layout(); st.pyplot(fig_corr)
                        except Exception: pass
        elif not st.session_state.run_complete_fedbatch: st.info("⬅️ Upload data, set parameters and run adjustment.")

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Fed-Batch Adjustment (S Inhibition)")
    ajuste_parametros_fedbatch_page()