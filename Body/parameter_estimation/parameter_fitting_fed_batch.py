import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
import openpyxl # Necessary para leer el input
import seaborn as sns
# import io # Ya no es necesario sin descarga
import traceback # Para imprimir stack trace en errores

#--------------------------------------------------------------------------
# Funciones Auxiliares para Estrategias de Alimentación
# (Sin cambios en esta sección)
#--------------------------------------------------------------------------
def calculate_feed_rate(t, strategy, t_start, t_end, F_min, F_max, V0=None, mu_set=None, Xs_f_ratio=None, Sf=None, step_data=None, X_current=None, V_current=None):
    """
    Calcula el flujo de alimentación F(t) basado en la estrategia seleccionada.
    (Código idéntico a la versión anterior - omitido por brevedad)
    """
    # Ensure que los tiempos sean flotantes para comparación segura
    t = float(t)
    t_start = float(t_start)
    t_end = float(t_end)

    # Condición inicial: fuera del intervalo de alimentación
    # Usar una pequeña tolerancia numérica por si acaso
    if t < t_start - 1e-9 or t > t_end + 1e-9:
        return 0.0

    # Ensure que F_min y F_max sean flotantes
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
# Model ODE para Lote Alimentado - ¡MODIFICADO!
#--------------------------------------------------------------------------
def modelo_ode_fedbatch(t, y, params, feed_params):
    """
    Define las ecuaciones diferenciales para el modelo de bioproceso en lote alimentado,
    INCLUYENDO INHIBICIÓN POR SUSTRATO.

    Args:
        t (float): Tiempo actual.
        y (list/array): Vector de estado [X, S, P, O2, V].
        params (list/array): Parámetros cinéticos [mumax, Ks, Yxs, Kd, Ypx, Ksi]. ¡Ahora 6!
        feed_params (dict): Diccionario con parámetros de alimentación.

    Returns:
        list: Derivadas [dXdt, dSdt, dPdt, dO2dt, dVdt].
    """
    try:
        X, S, P, O2, V = y
        # --- ¡AHORA SE DESEMPAQUETAN 6 PARÁMETROS! ---
        if len(params) != 6:
             raise ValueError(f"6 parameters were expected [mumax, Ks, Yxs, Kd, Ypx, Ksi], but only {len(params)} were received")
        mumax, Ks, Yxs, Kd, Ypx, Ksi = params

        # Extraer parámetros de alimentación
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

        # Calculate F(t)
        F = calculate_feed_rate(t, strategy, t_start, t_end, F_min, F_max,
                                V0=V0, mu_set=mu_set, Xs_f_ratio=Yxs_param_for_F, Sf=Sf,
                                step_data=step_data, X_current=X, V_current=V)

        # Valores seguros para cálculos
        X_safe = max(X, 0.0)
        S_safe = max(S, 0.0)
        V_safe = max(V, 1e-9)
        D = F / V_safe # Tasa de dilución

        # Parameters cinéticos seguros
        Ks_safe = max(Ks, 1e-9)
        Yxs_safe = max(Yxs, 1e-9)
        Kd_safe = max(Kd, 0.0)
        Ypx_safe = max(Ypx, 0.0)
        mumax_safe = max(mumax, 0.0)
        Ksi_safe = max(Ksi, 1e-6) # Ksi debe ser positivo y no cero

        # --- ¡NUEVO CÁLCULO DE MU CON INHIBICIÓN POR SUSTRATO! ---
        denominator = Ks_safe + S_safe + (S_safe**2 / Ksi_safe)
        if denominator <= 1e-12:
            mu = 0.0
        else:
            mu = mumax_safe * S_safe / denominator
        mu = max(mu, 0.0) # Ensure que mu no sea negativo

        # Ecuaciones diferenciales Fed-Batch (iguales, pero mu es diferente)
        dXdt = mu * X_safe - Kd_safe * X_safe - D * X + D * Xf
        dSdt = - (mu / Yxs_safe) * X_safe + D * (Sf - S)
        dPdt = Ypx_safe * mu * X_safe - D * P + D * Pf
        dO2dt = 0 # Simplified
        dVdt = F

        return [dXdt, dSdt, dPdt, dO2dt, dVdt]

    except Exception as e:
        st.error(f"Error in modelo_ode_fedbatch on t={t} with y={y}, params={params}: {e}")
        raise

#--------------------------------------------------------------------------
# Function para calcular Jacobiano (Adaptada para Fed-Batch)
# (Sin cambios en la lógica, pero operará sobre 6 parámetros)
#--------------------------------------------------------------------------
def compute_jacobian_fedbatch(params_opt, t_exp, y0_fit, feed_params, atol, rtol):
    """
    Calcula el Jacobiano numéricamente para el modelo fed-batch.
    Ahora opera sobre 6 parámetros [mumax, Ks, Yxs, Kd, Ypx, Ksi].
    La salida del Jacobiano es (n_puntos_tiempo * n_variables_medidas) x 6.
    """
    delta = 1e-7
    # --- ¡n_params ahora será 6 si params_opt tiene 6 elementos! ---
    n_params = len(params_opt)
    if n_params != 6:
         st.error(f"Error in Jacobian: 6 parameters were expected, only {n_params} were received")
         return None # O manejar el error de otra forma

    n_times = len(t_exp)
    n_vars_measured = 3

    y_nominal_flat = np.zeros(n_times * n_vars_measured)

    # Simulation Nominal
    try:
        sol_nominal = solve_ivp(modelo_ode_fedbatch, [t_exp[0], t_exp[-1]],
                                y0_fit,
                                args=(params_opt, feed_params), # params_opt debe tener 6 elementos
                                t_eval=t_exp, atol=atol, rtol=rtol,
                                method='LSODA')
        if sol_nominal.status != 0:
             st.warning(f"Jacobian: Nominal Solver Failed (status {sol_nominal.status}): {sol_nominal.message}")
             return np.full((n_times * n_vars_measured, n_params), np.nan)
        y_nominal_flat = sol_nominal.y[0:n_vars_measured, :].flatten()
    except Exception as e:
         st.error(f"Fatal error in nominal simulation for the Jacobian: {e}")
         st.text(traceback.format_exc())
         return np.full((n_times * n_vars_measured, n_params), np.nan)

    # Calculation de Derivadas Numéricas
    jac = np.zeros((n_times * n_vars_measured, n_params))

    for i in range(n_params): # El bucle ahora itera de 0 a 5
        params_perturbed = np.array(params_opt, dtype=float)
        h = delta * abs(params_perturbed[i]) if abs(params_perturbed[i]) > 1e-8 else delta
        if h == 0:
            jac[:, i] = 0.0
            continue
        params_perturbed[i] += h

        try:
            sol_perturbed = solve_ivp(modelo_ode_fedbatch, [t_exp[0], t_exp[-1]],
                                      y0_fit,
                                      args=(params_perturbed, feed_params), # params_perturbed tiene 6 elem.
                                      t_eval=t_exp, atol=atol, rtol=rtol,
                                      method='LSODA')
            if sol_perturbed.status != 0:
                st.warning(f"Jacobian: Perturbed solver (param {i}, status {sol_perturbed.status}) failed: {sol_perturbed.message}")
                jac[:, i] = np.nan
                continue
            y_perturbed_flat = sol_perturbed.y[0:n_vars_measured, :].flatten()
            derivative = (y_perturbed_flat - y_nominal_flat) / h
            jac[:, i] = derivative
        except Exception as e:
            st.error(f"Fatal error in perturbed simulation (parameter {i}) for the Jacobian: {e}")
            st.text(traceback.format_exc())
            jac[:, i] = np.nan

    return jac


#--------------------------------------------------------------------------
# Function Objetivo (Adaptada para Fed-Batch)
# (Sin cambios en la lógica, pero recibirá 6 parámetros)
#--------------------------------------------------------------------------
def objetivo_fedbatch(params, t_exp, y_exp_stacked, y0_fit, feed_params, atol, rtol):
    """
    Función objetivo para minimizar (RMSE) usando el modelo fed-batch.
    Ahora recibe 6 parámetros [mumax, Ks, Yxs, Kd, Ypx, Ksi].
    y_exp_stacked es el array aplanado de datos experimentales (X, S, P).
    """
    # Validar número de parámetros recibidos podría ser útil aquí también
    if len(params) != 6:
         st.warning(f"Objective function received {len(params)} parameters, it was expected to receive 6.")
         return 1e18 # Penalización muy alta

    try:
        sol = solve_ivp(modelo_ode_fedbatch, # Esta función ahora usa el modelo con Ksi
                        [t_exp[0], t_exp[-1]],
                        y0_fit,
                        args=(params, feed_params), # Pasa los 6 parámetros
                        t_eval=t_exp, atol=atol, rtol=rtol,
                        method='LSODA')

        # Manejo de Fallos del Solver (igual que antes)
        if sol.status != 0:
             return 1e6 + np.sum(np.abs(params))

        # Calculation del Error (igual que antes)
        y_pred = sol.y[0:3, :]
        y_pred_stacked = y_pred.flatten()
        if y_exp_stacked.shape != y_pred_stacked.shape:
             st.error(f"Shape discrepancy in objective: Exp {y_exp_stacked.shape}, Pred {y_pred_stacked.shape}")
             return 1e11
        mask = ~np.isnan(y_exp_stacked)
        if np.sum(mask) == 0:
            return 1e12
        sse = np.sum((y_pred_stacked[mask] - y_exp_stacked[mask])**2)
        rmse = np.sqrt(sse / np.sum(mask))

        # Penalizaciones Adicionales (igual que antes)
        if np.any(sol.y[0:3,:] < -1e-3):
             neg_penalty = np.sum(np.abs(sol.y[0:3,:][sol.y[0:3,:] < -1e-3]))
             rmse += neg_penalty * 1e3
        if np.any(sol.y[4,:] < 0):
             rmse += 1e5
        if np.isnan(rmse) or np.isinf(rmse):
            return 1e15

        return rmse

    except Exception as e:
        # st.error(f"Excepción en objetivo con params {params}: {str(e)}")
        return 1e15


#--------------------------------------------------------------------------
# Página de Streamlit - ¡MODIFICADA!
#--------------------------------------------------------------------------
def parameter_fitting_fedbatch_page():
    st.header("🔧 Adjustment of Kinetic Parameters (Fed-Batch with Substrate Inhibition)")

    # --- Inicialización del Estado de Sesión (igual que antes) ---
    if 'params_opt_fedbatch' not in st.session_state:
        st.session_state.params_opt_fedbatch = None # Ahora tendrá 6 elementos si es exitoso
    if 'result_fedbatch' not in st.session_state:
        st.session_state.result_fedbatch = None
    if 'parametros_df_fedbatch' not in st.session_state:
        st.session_state.parametros_df_fedbatch = None
    if 'sol_fedbatch' not in st.session_state:
        st.session_state.sol_fedbatch = None
    if 'df_exp_fedbatch' not in st.session_state:
        st.session_state.df_exp_fedbatch = None
    if 'y0_fit_fedbatch' not in st.session_state:
        st.session_state.y0_fit_fedbatch = None
    if 'feed_params_fedbatch' not in st.session_state:
        st.session_state.feed_params_fedbatch = {}
    if 't_exp_fedbatch' not in st.session_state:
        st.session_state.t_exp_fedbatch = None
    if 'y_exp_fedbatch' not in st.session_state:
        st.session_state.y_exp_fedbatch = None
    if 'y_exp_stacked_fedbatch' not in st.session_state:
        st.session_state.y_exp_stacked_fedbatch = None
    if 'run_complete_fedbatch' not in st.session_state:
        st.session_state.run_complete_fedbatch = False
    if 'last_uploaded_filename' not in st.session_state:
        st.session_state.last_uploaded_filename = None


    # --- Columna Izquierda: Carga de Datos y Configuración ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload experimental data")
        uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], key="fedbatch_uploader_inhib") # Cambiar key por si acaso

        # Procesamiento de archivo (igual que antes)
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_uploaded_filename:
                 st.session_state.params_opt_fedbatch = None
                 st.session_state.result_fedbatch = None
                 st.session_state.run_complete_fedbatch = False
                 st.session_state.last_uploaded_filename = uploaded_file.name
                 st.info(f"New file '{uploaded_file.name}' uploaded. Previous results reset.")
            try:
                df_exp = pd.read_excel(uploaded_file, engine='openpyxl')
                # (Validación de columnas y procesamiento igual que antes)
                required_cols = ['time', 'biomass', 'substrate', 'product']
                if not all(col in df_exp.columns for col in required_cols):
                    st.error(f"File must contain the column names: {', '.join(required_cols)}")
                    st.session_state.df_exp_fedbatch = None # Resetear
                    st.stop()
                else:
                    df_exp = df_exp.sort_values(by='time').reset_index(drop=True)
                    df_exp[['biomass', 'substrate', 'product']] = df_exp[['biomass', 'substrate', 'product']].apply(pd.to_numeric, errors='coerce')
                    st.session_state.df_exp_fedbatch = df_exp
                    st.session_state.t_exp_fedbatch = df_exp['time'].values
                    st.session_state.y_exp_fedbatch = df_exp[['biomass', 'substrate', 'product']].values.T
                    st.session_state.y_exp_stacked_fedbatch = st.session_state.y_exp_fedbatch.flatten()
                    st.dataframe(df_exp.head()) # Show preview después de procesar
            except Exception as e:
                st.error(f"Error reading the Excel file: {e}")
                st.session_state.df_exp_fedbatch = None # Resetear
                uploaded_file = None
        elif st.session_state.df_exp_fedbatch is not None:
             st.info(f"Using the previously uploaded data from '{st.session_state.last_uploaded_filename}'.")


        # --- Configuración del Proceso (Solo si hay datos cargados) ---
        if st.session_state.df_exp_fedbatch is not None:
            df_exp = st.session_state.df_exp_fedbatch
            t_exp = st.session_state.t_exp_fedbatch

            st.subheader("⚙️ Process Configuration")

            # Condiciones Iniciales (igual que antes)
            st.markdown("##### Reactor Initial Conditions")
            X0_default = df_exp['biomass'].iloc[0] if pd.notna(df_exp['biomass'].iloc[0]) else 0.1
            S0_default = df_exp['substrate'].iloc[0] if pd.notna(df_exp['substrate'].iloc[0]) else 10.0
            P0_default = df_exp['product'].iloc[0] if pd.notna(df_exp['product'].iloc[0]) else 0.0
            O0_default = 8.0
            V0_default = 1.0
            X0_fit = st.number_input("Initial Biomass (X0) [g/L]", min_value=0.0, value=float(X0_default), format="%.4f")
            S0_fit = st.number_input("Initial Substrate (S0) [g/L]", min_value=0.0, value=float(S0_default), format="%.4f")
            P0_fit = st.number_input("Initial Product (P0) [g/L]", min_value=0.0, value=float(P0_default), format="%.4f")
            O0_fit = st.number_input("Initial O2 [mg/L]", min_value=0.0, max_value=20.0, value=float(O0_default), format="%.4f")
            V0_fit = st.number_input("Initial Volume (V0) [L]", min_value=1e-3, value=float(V0_default), format="%.4f")
            st.session_state.y0_fit_fedbatch = [X0_fit, S0_fit, P0_fit, O0_fit, V0_fit]

            # Parameters de Alimentación (igual que antes)
            st.markdown("##### Feeding Configuration")
            feed_strategy = st.selectbox("Feeding Strategy",
                                          ['Constant', 'Linear', 'Exponential (Simple)', 'Exponential (constant mu)', 'Step'],
                                          key="feed_strategy_inhib")
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

            # Parameters Específicos por Estrategia (igual que antes)
            step_data_input_list = None
            mu_set_feed = None
            if feed_strategy == 'Step':
                # (Código de parseo de escalones igual que antes - omitido por brevedad)
                st.write(f"Define steps (time, flow) in [{t_start_feed:.2f}h - {t_end_feed:.2f}h]:")
                step_data_input_str = st.text_area("Format: one 'time, flow' pair per line.\nEj: 2, 0.05\\n5, 0.1\\n10, 0.08", height=100, key="step_data_text_inhib")
                step_data_input_list = []
                if step_data_input_str:
                    lines = step_data_input_str.strip().split('\n')
                    valid_steps = True
                    for i, line in enumerate(lines):
                         line = line.strip();
                         if not line: continue
                         try:
                             parts = line.split(',');
                             if len(parts) == 2:
                                 time = float(parts[0].strip()); flow = float(parts[1].strip())
                                 if time < t_start_feed - 1e-6 or time > t_end_feed + 1e-6: continue
                                 if flow < F_min_feed - 1e-6 or flow > F_max_feed + 1e-6: continue
                                 step_data_input_list.append((time, flow))
                             else: valid_steps = False; break
                         except ValueError: valid_steps = False; break
                    if valid_steps and step_data_input_list: step_data_input_list.sort(key=lambda x: x[0]); st.success(f"{len(step_data_input_list)} escalones válidos.")
                    elif valid_steps and not step_data_input_list: st.info("No valid steps were defined.")
                    else: step_data_input_list = None; st.error("Error en steps format.")
                if not step_data_input_list: st.warning("Selected strategies without valid steps.")

            elif feed_strategy == 'Exponential (constant mu)':
                mu_set_feed = st.number_input("Desired Specific Growth Rate (μ_set) [1/h]", min_value=0.0, max_value=2.0, value=0.1, format="%.4f")

            # Guardar feed_params (igual que antes)
            st.session_state.feed_params_fedbatch = {
                'strategy': feed_strategy, 't_start': t_start_feed, 't_end': t_end_feed,
                'F_min': F_min_feed, 'F_max': F_max_feed, 'Sf': Sf_feed, 'Xf': Xf_feed,
                'Pf': Pf_feed, 'step_data': step_data_input_list, 'mu_set': mu_set_feed,
                'V0': V0_fit
            }

            # --- Configuración del Ajuste - ¡MODIFICADA! ---
            st.subheader("⚙️ Adjustment Configuration")

            # Parameters a ajustar (estimaciones iniciales) - ¡Añadido Ksi!
            st.markdown("##### Kinetic Parameters to Optimize (Initial Estimates)")
            p_opt = st.session_state.params_opt_fedbatch # Puede tener 5 o 6 elementos de ejecuciones anteriores
            mumax_guess_val = float(p_opt[0]) if p_opt is not None and len(p_opt)>0 else 0.5
            Ks_guess_val = float(p_opt[1]) if p_opt is not None and len(p_opt)>1 else 0.2
            Yxs_guess_val = float(p_opt[2]) if p_opt is not None and len(p_opt)>2 else 0.5
            Kd_guess_val = float(p_opt[3]) if p_opt is not None and len(p_opt)>3 else 0.01
            Ypx_guess_val = float(p_opt[4]) if p_opt is not None and len(p_opt)>4 else 0.3
            # --- NUEVO PARÁMETRO Ksi ---
            Ksi_guess_val = float(p_opt[5]) if p_opt is not None and len(p_opt)>5 else 100.0 # Valor inicial para Ksi

            mumax_guess = st.number_input("μmax [1/h]", 0.0, 5.0, mumax_guess_val, format="%.4f")
            Ks_guess = st.number_input("Ks [g/L]", 1e-4, 20.0, Ks_guess_val, format="%.4f")
            Yxs_guess = st.number_input("Yxs [g/g]", 0.01, 3.0, Yxs_guess_val, format="%.4f")
            Kd_guess = st.number_input("Kd [1/h]", 0.0, 1.0, Kd_guess_val, format="%.4f")
            Ypx_guess = st.number_input("Ypx [g/g]", 0.0, 10.0, Ypx_guess_val, format="%.4f")
            # --- INPUT PARA Ksi ---
            Ksi_guess = st.number_input("Ksi (Substrate Inhib.) [g/L]", 1.0, 1000.0, Ksi_guess_val, format="%.2f")

            # Límites estrictos para la optimización - ¡Añadido Ksi! (6 parámetros)
            bounds = [(1e-3, 5.0), (1e-4, 20.0), (0.01, 3.0), (0.0, 1.0), (0.0, 10.0), (1.0, 1000.0)]

            # Tolerances del solver ODE (igual que antes)
            st.markdown("##### ODE Solver Tolerances")
            atol = st.number_input("Absolute tolerance (atol)", min_value=1e-12, max_value=1e-3, value=1e-8, format="%e")
            rtol = st.number_input("Relative tolerance (rtol)", min_value=1e-12, max_value=1e-3, value=1e-8, format="%e")

            # Options de optimización (igual que antes)
            st.markdown("##### Optimization Options")
            metodo = st.selectbox("Optimization Methods",
                                  ['L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell', 'differential_evolution'],
                                  index=0, key="opt_method_fedbatch_inhib")
            max_iter = st.number_input("Maximum iterations", 10, 10000, 500)


            # --- Botón de Ejecución - ¡MODIFICADO! ---
            if st.button("🚀 Run Adjustment (with Substrate Inhib.)", key="run_fedbatch_fit_inhib"):
                if st.session_state.y_exp_stacked_fedbatch is None or st.session_state.t_exp_fedbatch is None:
                    st.error("The adjustment cannot be executed. Experimental data are missing.")
                elif feed_strategy == 'Escalonada' and not step_data_input_list:
                     st.error("Step strategy selected, but no valid steps were defined.")
                else:
                    with st.spinner("Optimizing parameters (model with inhibition)..."):
                        # --- ¡AHORA 6 PARÁMETROS INICIALES! ---
                        initial_guess = [mumax_guess, Ks_guess, Yxs_guess, Kd_guess, Ypx_guess, Ksi_guess]

                        # Referenciar datos y configuración de sesión (igual que antes)
                        y0_run = st.session_state.y0_fit_fedbatch
                        feed_params_run = st.session_state.feed_params_fedbatch
                        t_exp_run = st.session_state.t_exp_fedbatch
                        y_exp_stacked_run = st.session_state.y_exp_stacked_fedbatch

                        result = None
                        st.session_state.run_complete_fedbatch = False

                        try:
                            # --- ¡OPTIMIZACIÓN CON 6 PARÁMETROS! ---
                            if metodo == 'differential_evolution':
                                result = differential_evolution(objetivo_fedbatch, bounds, # bounds ahora tiene 6 rangos
                                                                args=(t_exp_run, y_exp_stacked_run, y0_run, feed_params_run, atol, rtol),
                                                                maxiter=max_iter, tol=1e-7, mutation=(0.5, 1.5),
                                                                recombination=0.8, strategy='best1bin',
                                                                updating='deferred', workers=-1, seed=42)
                            else:
                                 minimizer_kwargs = {
                                     "args": (t_exp_run, y_exp_stacked_run, y0_run, feed_params_run, atol, rtol),
                                     "method": metodo,
                                     "bounds": bounds if metodo in ['L-BFGS-B', 'TNC', 'SLSQP'] else None, # bounds tiene 6 rangos
                                     "options": {'maxiter': max_iter, 'disp': False}
                                 }
                                 if metodo in ['L-BFGS-B', 'TNC', 'SLSQP']:
                                     minimizer_kwargs['options']['ftol'] = 1e-9
                                     minimizer_kwargs['options']['gtol'] = 1e-7
                                 elif metodo == 'Nelder-Mead':
                                     minimizer_kwargs['options']['xatol'] = 1e-7
                                     minimizer_kwargs['options']['fatol'] = 1e-9
                                 # ¡initial_guess ahora tiene 6 elementos!
                                 result = minimize(objetivo_fedbatch, initial_guess, **minimizer_kwargs)


                            # Procesamiento de resultados (igual que antes)
                            if result and hasattr(result, 'success') and result.success:
                                st.session_state.params_opt_fedbatch = result.x # Guardará 6 parámetros
                                st.session_state.result_fedbatch = result
                                st.session_state.run_complete_fedbatch = True
                                st.success(f"Optimization ({metodo}) completed with success.")
                            elif result and hasattr(result, 'x'):
                                 st.session_state.params_opt_fedbatch = result.x # Guardará 6 parámetros
                                 st.session_state.result_fedbatch = result
                                 st.session_state.run_complete_fedbatch = True
                                 st.warning(f"The optimization ({metodo}) finished but it was unsuccessful: {getattr(result, 'message', 'No message')}")
                            else:
                                 st.error(f"The optimization ({metodo})  failed or did not return a valid result.")

                        except Exception as e:
                            st.error(f"Fatal error during ({metodo}) optimization: {e}")
                            st.text(traceback.format_exc())
                            st.session_state.params_opt_fedbatch = None
                            st.session_state.result_fedbatch = None
                            st.session_state.run_complete_fedbatch = False

                        st.rerun()


        elif st.session_state.df_exp_fedbatch is None:
            st.warning("⏳ Please, upload an experimental data file to get started.")


    # --- Columna Derecha: Resultados - ¡MODIFICADA! ---
    with col2:
        if st.session_state.run_complete_fedbatch and st.session_state.result_fedbatch is not None:
            st.subheader("📊 Adjustment Results (Model with Substrate Inhibition)")

            result = st.session_state.result_fedbatch
            params_opt = st.session_state.params_opt_fedbatch # Ahora debería tener 6 elementos
            y0_res = st.session_state.y0_fit_fedbatch
            feed_params_res = st.session_state.feed_params_fedbatch
            df_exp_res = st.session_state.df_exp_fedbatch
            t_exp_res = st.session_state.t_exp_fedbatch
            y_exp_res = st.session_state.y_exp_fedbatch
            atol_res = atol # Usar valor actual del widget
            rtol_res = rtol # Usar valor actual del widget

            # Verify si params_opt tiene 6 elementos
            if params_opt is None or len(params_opt) != 6:
                 st.error("Optimized parameters are not available or do not have the expected length (6). Detailed results cannot be shown.")
                 # Resetear estado problemático
                 st.session_state.params_opt_fedbatch = None
                 st.session_state.result_fedbatch = None
                 st.session_state.run_complete_fedbatch = False
                 st.stop() # Detener para evitar más errores

            final_rmse = getattr(result, 'fun', np.nan)
            if pd.notna(final_rmse):
                 st.write(f"**Final Objective Function (RMSE):** {final_rmse:.6f}")
            else:
                 st.write("**Final Objective Function (RMSE):** Not available")

            # --- Tabla de parámetros optimizados - ¡Añadido Ksi! ---
            param_names = ['μmax', 'Ks', 'Yxs', 'Kd', 'Ypx', 'Ksi'] # 6 nombres
            param_units = ['1/h', 'g/L', 'g/g', '1/h', 'g/g', 'g/L'] # 6 unidades
            parametros_df = pd.DataFrame({
                'Parameter': param_names,
                'Optimized Value': params_opt, # params_opt tiene 6 elementos
                'Units': param_units
            })
            st.dataframe(parametros_df.style.format({'Optimized Value': '{:.5f}'}))
            st.session_state.parametros_df_fedbatch = parametros_df # Guardar df con 6 params

            # --- Simulación Final y Métricas (Usa el nuevo modelo) ---
            st.subheader("📈 Final Simulation and Metrics")
            try:
                # solve_ivp llamará a modelo_ode_fedbatch que ahora incluye inhibición
                sol = solve_ivp(modelo_ode_fedbatch, [t_exp_res[0], t_exp_res[-1]],
                                y0_res,
                                args=(params_opt, feed_params_res), # Pasa los 6 params
                                t_eval=t_exp_res, atol=atol_res, rtol=rtol_res,
                                method='LSODA')

                if sol.status != 0:
                     st.error(f"The final simulation with the optimized parameters failed (status {sol.status}): {sol.message}")
                     st.session_state.sol_fedbatch = None
                else:
                     st.session_state.sol_fedbatch = sol
                     y_pred_final = sol.y[0:3, :]
                     # Calculation de métricas (igual que antes)
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
                         else:
                             metricas_list.append({'Variable': variable_name, 'R²': np.nan, 'RMSE': np.nan})
                     metricas_df = pd.DataFrame(metricas_list)
                     st.dataframe(metricas_df.style.format({'R²': '{:.4f}', 'RMSE': '{:.4f}'}))

            except Exception as e:
                 st.error(f"Error during final simulation: {e}")
                 st.text(traceback.format_exc())
                 st.session_state.sol_fedbatch = None


            # --- Gráficos Comparativos (igual que antes, pero con nueva simulación) ---
            if st.session_state.sol_fedbatch is not None:
                st.subheader("📉 Comparative Graphs")
                sol = st.session_state.sol_fedbatch
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                variables = ['Biomass', 'Substrate', 'Product']
                unidades = ['g/L', 'g/L', 'g/L']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                for i in range(3):
                    axes[i].plot(t_exp_res, y_exp_res[i], 'o', markersize=5, alpha=0.7, label=f'{variables[i]} Exp.')
                    axes[i].plot(sol.t, sol.y[i], '-', linewidth=2, label=f'{variables[i]} Mod.')
                    axes[i].set_ylabel(f"{variables[i]} [{unidades[i]}]")
                    axes[i].legend()
                    axes[i].grid(True, linestyle='--', alpha=0.6)
                axes[-1].set_xlabel("Time [h]")
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                fig.suptitle("Model Comparation (Substrate Inhibition) vs Experimental data", fontsize=16)
                st.pyplot(fig)

                # Plot Adicional: Volumen y Flujo (igual que antes)
                st.subheader("💧 Volume and Feed Flow")
                try:
                    interp_x = np.interp(sol.t, sol.t, sol.y[0])
                    interp_v = np.interp(sol.t, sol.t, sol.y[4])
                    yxs_opt = params_opt[2] # Yxs optimizado
                    F_t = np.array([calculate_feed_rate(ti, feed_params_res['strategy'], feed_params_res['t_start'], feed_params_res['t_end'],
                                                         feed_params_res['F_min'], feed_params_res['F_max'], V0=feed_params_res['V0'],
                                                         mu_set=feed_params_res['mu_set'], Xs_f_ratio=yxs_opt, Sf=feed_params_res['Sf'],
                                                         step_data=feed_params_res['step_data'], X_current=interp_x[idx],
                                                         V_current=interp_v[idx]) for idx, ti in enumerate(sol.t)])
                    fig_vol_feed, ax1 = plt.subplots(figsize=(10, 5))
                    color_vol = 'tab:red'; ax1.set_xlabel('Time [h]'); ax1.set_ylabel('Volume [L]', color=color_vol)
                    ax1.plot(sol.t, sol.y[4], color=color_vol, linestyle='-', linewidth=2, label='Volume (Model)')
                    ax1.tick_params(axis='y', labelcolor=color_vol); ax1.grid(True, which='major', linestyle='--', alpha=0.7)
                    ax1.grid(True, which='minor', linestyle=':', alpha=0.4); ax1.minorticks_on(); ax1.legend(loc='upper left')
                    ax2 = ax1.twinx(); color_feed = 'tab:blue'; ax2.set_ylabel('Feed Flow [L/h]', color=color_feed)
                    if feed_params_res['strategy'] in ['Constant', 'Step']:
                        ax2.step(sol.t, F_t, where='post', color=color_feed, linestyle='--', label=f'Flow ({feed_params_res["strategy"]})')
                    else:
                         ax2.plot(sol.t, F_t, color=color_feed, linestyle='--', label=f'Flow ({feed_params_res["strategy"]})')
                    ax2.tick_params(axis='y', labelcolor=color_feed); ax2.legend(loc='upper right')
                    ax2.set_ylim(bottom=max(-0.01, F_min_feed * 0.9), top=F_max_feed * 1.1 + 0.01)
                    fig_vol_feed.tight_layout(); plt.title("Volume and Feed Flow Evolution"); st.pyplot(fig_vol_feed)
                except Exception as e:
                    st.warning(f"Volume/Flow graph could not be generated: {e}")


                # --- Análisis Estadístico - ¡MODIFICADO! (n_params = 6) ---
                st.subheader("📈 Statistical Analysis")
                with st.spinner("Calculating confidence intervals..."):
                    try:
                        residuals = y_exp_res - y_pred_final
                        residuals_flat = residuals.flatten()
                        residuals_flat_clean = residuals_flat[~np.isnan(residuals_flat)]
                        n_obs_clean = len(residuals_flat_clean)
                        # --- ¡AHORA n_params = 6! ---
                        n_params = len(params_opt) # Should be 6
                        dof = n_obs_clean - n_params # Grados de libertad ajustados

                        if dof <= 0:
                             st.warning(f"Not enough data points ({n_obs_clean}) to calculate CI (needed > {n_params}).")
                             for col in ['Standard Error', 'Interval ± (95%)', '95% CI Lower bound', '95% CI Upper bound']:
                                 if col not in parametros_df.columns: parametros_df[col] = np.nan
                        else:
                            # Jacobian ahora tendrá 6 columnas
                            jac = compute_jacobian_fedbatch(params_opt, t_exp_res, y0_res, feed_params_res, atol_res, rtol_res)

                            if jac is None or np.isnan(jac).any() or np.isinf(jac).any():
                                st.warning("Invalid Jacobian. CI cannot be calculated.")
                                for col in ['Standard Error', 'Interval ± (95%)', '95% CI Lower bound', '95% CI Upper bound']:
                                    if col not in parametros_df.columns: parametros_df[col] = np.nan
                            else:
                                 # Calculation de covarianza e IC (igual que antes, pero con jac de 6 cols)
                                 mse = np.sum(residuals_flat_clean**2) / dof
                                 jtj = jac.T @ jac
                                 try: cov_matrix = np.linalg.pinv(jtj) * mse
                                 except np.linalg.LinAlgError: cov_matrix = np.linalg.pinv(jtj) * mse # Fallback
                                 diag_cov = np.diag(cov_matrix)
                                 valid_variance = diag_cov > 1e-15
                                 std_errors = np.full_like(diag_cov, np.nan)
                                 std_errors[valid_variance] = np.sqrt(diag_cov[valid_variance])
                                 if np.any(~valid_variance): st.warning("Non-positive variances found.")
                                 alpha = 0.05; t_val = t.ppf(1.0 - alpha / 2.0, df=dof)
                                 intervals = t_val * std_errors
                                 parametros_df['Standard Error'] = std_errors
                                 parametros_df['Interval ± (95%)'] = intervals
                                 parametros_df['95% CI Lower bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] - intervals)
                                 parametros_df['95% CI Upper bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] + intervals)
                                 st.success("Confidence Intervals Calculated.")

                    except Exception as e:
                        st.error(f"Error calculating Confidence Intervals: {e}")
                        st.text(traceback.format_exc())
                        for col in ['Standard Error', 'Interval ± (95%)', '95% CI - Lower bound', '95% CI - Upper bound']:
                            if col not in parametros_df.columns: parametros_df[col] = np.nan

                    # Show tabla de parámetros con IC (ahora con 6 filas)
                    st.write("Optimized Parameters and Confidence Intervals (95%):")
                    st.dataframe(parametros_df.style.format({
                       'Optimized Value': '{:.5f}', 'Standard Error': '{:.5f}',
                       'Interval ± (95%)': '{:.5f}', '95% CI - Lower bound': '{:.5f}',
                       '95% CI - Upper bound': '{:.5f}'}, na_rep='N/A'))
                    st.session_state.parametros_df_fedbatch = parametros_df

                    # Plot de IC (ahora con 6 barras)
                    if 'Interval ± (95%)' in parametros_df.columns and parametros_df['Interval ± (95%)'].notna().any():
                        st.subheader("📐 Confidence Intervals for Parameters")
                        fig_ci, ax = plt.subplots(figsize=(10, max(4, len(parametros_df) * 0.6)))
                        y_pos = np.arange(len(parametros_df)); errors_for_plot = parametros_df['Interval ± (95%)'].copy()
                        mask_nan_interval = errors_for_plot.isna() & parametros_df['Standard Error'].notna()
                        errors_for_plot[mask_nan_interval] = 1.96 * parametros_df['Standard Error'][mask_nan_interval]
                        errors_for_plot = errors_for_plot.fillna(0).values
                        bars = ax.barh(y_pos, parametros_df['Optimized Value'].fillna(0), xerr=errors_for_plot,
                                align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5, alpha=0.8)
                        ax.set_yticks(y_pos); ax.set_yticklabels(parametros_df['Parameter']) # Ahora incluye Ksi
                        ax.invert_yaxis(); ax.set_xlabel('Parameter')
                        ax.set_title('95% Confidence Intervals(or ~1.96*SE if exact CI fails)'); ax.grid(True, axis='x', linestyle='--', alpha=0.6)
                        plt.tight_layout(); st.pyplot(fig_ci)

                    # Analysis de Residuales (igual que antes)
                    st.subheader("📉 Residuals Analysis")
                    fig_hist, axs = plt.subplots(1, 3, figsize=(15, 5))
                    variables_res = ['Biomass', 'Substrate', 'Product']; colors_res = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    for i, (var, color) in enumerate(zip(variables_res, colors_res)):
                        res = y_exp_res[i] - y_pred_final[i]; res_clean = res[~np.isnan(res)]
                        if len(res_clean) > 1:
                            sns.histplot(res_clean, kde=True, color=color, ax=axs[i], bins='auto')
                            axs[i].set_title(f'Residuals {var} (N={len(res_clean)})'); axs[i].set_xlabel('Error (Experimental - Model)')
                            axs[i].set_ylabel('Frecuence / Density'); axs[i].axvline(0, color='k', linestyle='--'); axs[i].grid(True, linestyle='--', alpha=0.3)
                            mean_res = np.mean(res_clean); std_res = np.std(res_clean)
                            axs[i].text(0.05, 0.95, f'Mean={mean_res:.2f}\nStd={std_res:.2f}', transform=axs[i].transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
                        else:
                            axs[i].set_title(f'Residuals {var} (Insufficient data)'); axs[i].text(0.5, 0.5, 'Not enough valid data', ha='center', va='center', transform=axs[i].transAxes)
                    plt.tight_layout(); st.pyplot(fig_hist)

                    # Matriz de Correlación (ahora 6x6)
                    if 'cov_matrix' in locals() and cov_matrix is not None and not np.isnan(cov_matrix).all():
                        st.subheader("📌 Parameter Correlation Matrix")
                        try:
                            std_devs = np.sqrt(np.diag(cov_matrix))
                            with np.errstate(divide='ignore', invalid='ignore'): corr_matrix_calc = cov_matrix / np.outer(std_devs, std_devs)
                            corr_matrix_calc[~np.isfinite(corr_matrix_calc)] = np.nan
                            np.fill_diagonal(corr_matrix_calc, 1.0); corr_matrix_calc = np.clip(corr_matrix_calc, -1.0, 1.0)
                            corr_df = pd.DataFrame(corr_matrix_calc, index=param_names, columns=param_names) # Usa los 6 nombres
                            fig_corr, ax = plt.subplots(figsize=(8, 7)) # Ligeramente más grande
                            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", center=0, linewidths=.5, linecolor='black', ax=ax, vmin=-1, vmax=1, annot_kws={"size": 9}) # Letra más pequeña
                            ax.set_title('Parameters Estimated Correlation Matrix'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                            plt.tight_layout(); st.pyplot(fig_corr)
                        except Exception as e: st.warning(f"Correlation Matrix could not be plotted/calculated: {e}")
                    else: st.info("Correlation Matrix not available.")


            else: # Si la simulación falló
                st.info("Complete successfull adjustment and final simulation to see all the analysis")

        # Mensaje inicial
        elif not st.session_state.run_complete_fedbatch:
             st.info("⬅️ Upload data, set parameters and run adjustment.")


# --- Ejecución Principal ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Fed-Batch Adjustment (S Inhibition)")
    ajuste_parametros_fedbatch_page()