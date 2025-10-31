# adjustment_parametros_ferm.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import t
import openpyxl
import seaborn as sns
import traceback
try:
    from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion
except ImportError:
    st.error("Could not import 'Utils.kinetics'. Using dummy functions.")
    def mu_monod(S, mumax, Ks): return mumax * S / (Ks + S) if (Ks + S) > 0 else 0
    def mu_sigmoidal(S, mumax, Ks, n): return mumax * (S**n) / (Ks**n + S**n) if (Ks**n + S**n) > 0 else 0
    def mu_completa(S, O2, P, mumax, Ks, KO, KP):
        term_s = S / (Ks + S) if (Ks + S) > 0 else 0; term_o = O2 / (KO + O2) if (KO + O2) > 0 else 0
        term_p = (1 - P/KP) if KP > 0 and P < KP else 0; return mumax * term_s * term_o * term_p
    def mu_fermentacion(S, P, O2, mumax_a, Ks_a, KO_a, mumax_an, Ks_an, KiS_an, KP_an, n_p, KO_inhib_an):
        mu_aerob = mumax_a * (S / (Ks_a + S)) * (O2 / (KO_a + O2)) if (Ks_a + S) > 0 and (KO_a + O2) > 0 else 0
        inhib_s = (1 - S/KiS_an) if KiS_an > 0 and S < KiS_an else 0; inhib_p = (1 - P/KP_an)**n_p if KP_an > 0 and P < KP_an else 0
        inhib_o = (1 - O2/KO_inhib_an) if KO_inhib_an > 0 and O2 < KO_inhib_an else 0
        mu_anaerob = mumax_an * (S / (Ks_an + S)) * inhib_s * inhib_p * inhib_o if (Ks_an + S) > 0 else 0
        return max(0, mu_aerob) + max(0, mu_anaerob)

#==========================================================================
# MODELO ODE FERMENTACION
#==========================================================================
# (Sin cambios respecto a la versión anterior - usa Kla fijo)
def modelo_fermentacion(t, y, params):
    """ ODE Fermentation Model. Uses fixed Kla parameters. """
    X, S, P, O2, V = y
    X = max(1e-9, X); S = max(0, S); P = max(0, P); O2 = max(0, O2); V = max(1e-6, V)
    try:
        tipo_mu = params.get("tipo_mu", "Simple Monod")
        Ks = params.get("Ks", 1.0); Yxs = params.get("Yxs", 0.1); Yps = params.get("Yps", 0.45); Yxo = params.get("Yxo", 0.8)
        alpha_lp = params.get("alpha_lp", 2.2); beta_lp = params.get("beta_lp", 0.05); ms = params.get("ms", 0.02); mo = params.get("mo", 0.01); Kd = params.get("Kd", 0.01)
        Kla = params.get("Kla", 100.0); Cs = params.get("Cs", 7.5); Sin = params.get("Sin", 400.0) # Usa Kla único
        t_batch_inicial_fin = params.get("t_batch_inicial_fin", 10.0); t_alim_inicio = params.get("t_alim_inicio", 10.1); t_alim_fin = params.get("t_alim_fin", 34.1)
        estrategia = params.get("estrategia", "Constant"); F_base = params.get("F_base", 0.1); F_lineal_fin_val = params.get("F_lineal_fin", F_base * 2); k_exp_val = params.get("k_exp", 0.1)
    except KeyError as e: raise KeyError(f"Missing operational/fixed parameter '{e}'.") from e

    F = 0.0
    if t_alim_inicio <= t <= t_alim_fin:
        if estrategia == "Constant": F = F_base
        elif estrategia == "Exponential":
             try: F = min(F_base * np.exp(k_exp_val * (t - t_alim_inicio)), F_base * 100)
             except OverflowError: F = F_base * 100
        elif estrategia == "Step": t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2; F = F_base * 2 if t > t_medio else F_base
        elif estrategia == "Linear":
             delta_t = t_alim_fin - t_alim_inicio
             if delta_t > 1e-6: slope = (F_lineal_fin_val - F_base) / delta_t; F = max(0, F_base + slope * (t - t_alim_inicio))
             else: F = F_base
    F = max(0, F)
    es_lote_inicial = (t < t_batch_inicial_fin); O2_controlado = params.get("O2_controlado", 0.08)

    mu = 0.0
    try:
        if tipo_mu == "Fermentaction":
            mu = mu_fermentacion(S, P, O2, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params["n_p"], params["KO_inhib_anaerob"])
        # elif tipo_mu == "Fermentación Conmutada": ...
        elif tipo_mu == "Simple Monod": mu = mu_monod(S, params["mumax"], Ks)
        elif tipo_mu == "Sigmoidal Monod": mu = mu_sigmoidal(S, params["mumax"], Ks, params["n_sig"])
        elif tipo_mu == "Monod with restrictions": mu = mu_completa(S, O2, P, params["mumax"], Ks, params["KO"], params["KP_gen"])
        else: mu = mu_monod(S, params.get("mumax", 0.0), Ks)
    except KeyError as e: raise KeyError(f"Missing kinetic parameter'{e}' for '{tipo_mu}' model.") from e
    except Exception as e: raise RuntimeError(f"Error calculating mu ({tipo_mu}): {e}") from e
    mu = max(0, mu)

    Yxs=max(1e-6, Yxs); Yps=max(1e-6, Yps); Yxo=max(1e-6, Yxo)
    qP = alpha_lp * mu + beta_lp; rate_P = qP * X
    consumo_S_X = (mu / Yxs) * X; consumo_S_P_correcto = rate_P / Yps; consumo_S_maint = ms * X
    rate_S = consumo_S_X + consumo_S_P_correcto + consumo_S_maint
    consumo_O2_X = (mu / Yxo) * X; consumo_O2_maint = mo * X
    OUR_g = consumo_O2_X + consumo_O2_maint; OUR_mg = OUR_g * 1000.0

    dXdt = (mu - Kd) * X - (F / V) * X; dSdt = -rate_S + (F / V) * (Sin - S)
    dPdt = rate_P - (F / V) * P; dVdt = F
    if es_lote_inicial: dOdt = 0.0
    else: dOdt = Kla * (Cs - O2) - OUR_mg - (F / V) * O2
    return [dXdt, dSdt, dPdt, dOdt, dVdt]

#==========================================================================
# FUNCIONES DE OPTIMIZACIÓN
#==========================================================================

# --- Función Objetivo (CON ESCALADO Y PONDERACIÓN!) ---
def objetivo_ferm(params_trial, param_names_opt, t_exp, y_exp_data, y0_fit, fixed_params, weights, atol, rtol): # Añadido 'weights'
    """
    Función objetivo con residuos escalados Y PONDERADOS.
    weights: lista o array [w_X, w_S, w_P, w_O2]
    y_exp_data tiene shape (4, n_times) -> [X_exp, S_exp, P_exp, O2_exp]
    Devuelve la suma ponderada de errores cuadráticos escalados (SSE_wp).
    """
    params_full = fixed_params.copy()
    if len(params_trial) != len(param_names_opt): st.error(f"Length discrepancy between params_trial/names"); return 1e20
    for name, value in zip(param_names_opt, params_trial): params_full[name] = value

    try:
        sol = solve_ivp(modelo_fermentacion, [t_exp[0], t_exp[-1]], y0_fit, args=(params_full,), t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
        if sol.status != 0: return 1e6 + np.sum(np.abs(params_trial))

        y_pred = sol.y[0:4, :]
        if y_exp_data.shape != y_pred.shape: st.error(f"Shape discrepancy between target data"); return 1e11

        sse_weighted_scaled = 0.0; valid_points_total = 0
        if len(weights) != 4: weights = [1.0, 1.0, 1.0, 1.0] # Fallback

        for i in range(4): # Iterar sobre X, S, P, O2
            exp_data_i = y_exp_data[i, :]; pred_data_i = y_pred[i, :]; mask_i = ~np.isnan(exp_data_i)
            if np.sum(mask_i) > 0:
                scale_factor = np.nanmax(np.abs(exp_data_i[mask_i])); scale_factor = 1.0 if scale_factor < 1e-6 else scale_factor
                weighted_scaled_residuals_sq = (weights[i] * ((pred_data_i[mask_i] - exp_data_i[mask_i]) / scale_factor)**2)
                sse_weighted_scaled += np.sum(weighted_scaled_residuals_sq)
                valid_points_total += np.sum(mask_i)

        if valid_points_total == 0: return 1e12
        objective_value = sse_weighted_scaled # Minimiza SSE ponderado y escalado

        # Penalizaciones
        if np.any(sol.y[0:4,:] < -1e-3): neg_penalty = np.sum(np.abs(sol.y[0:4,:][sol.y[0:4,:] < -1e-3])); objective_value += neg_penalty * 1e3
        if np.any(sol.y[4,:] < 0): objective_value += 1e5
        if np.isnan(objective_value) or np.isinf(objective_value): return 1e15
        return objective_value
    except KeyError as e: st.warning(f"Key Error: {e} is missing. Penalizing."); return 1e16 + np.sum(np.abs(params_trial))
    except Exception as e: return 1e15

# --- Cálculo de Jacobiano ---
# (Sin cambios respecto a la versión anterior)
def compute_jacobian_ferm(params_opt_trial, param_names_opt, t_exp, y0_fit, fixed_params, atol, rtol):
    """ Jacobiano numérico para modelo fermentación. """
    delta = 1e-7; n_params_opt = len(params_opt_trial); n_times = len(t_exp); n_vars_measured = 4
    y_nominal_flat = np.zeros(n_times * n_vars_measured)
    try:
        params_full_nominal = fixed_params.copy();
        for name, value in zip(param_names_opt, params_opt_trial): params_full_nominal[name] = value
        sol_nominal = solve_ivp(modelo_fermentacion, [t_exp[0], t_exp[-1]], y0_fit, args=(params_full_nominal,), t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
        if sol_nominal.status != 0: st.warning(f"Jacobian Nominal Ferm: {sol_nominal.message}"); return np.full((n_times * n_vars_measured, n_params_opt), np.nan)
        y_nominal_flat = sol_nominal.y[0:n_vars_measured, :].flatten()
    except Exception as e: st.error(f"Jacobian Nominal Ferm Error: {e}"); st.text(traceback.format_exc()); return np.full((n_times * n_vars_measured, n_params_opt), np.nan)
    jac = np.zeros((n_times * n_vars_measured, n_params_opt))
    for i in range(n_params_opt):
        params_perturbed_trial = np.array(params_opt_trial, dtype=float); h = delta * abs(params_perturbed_trial[i]) if abs(params_perturbed_trial[i]) > 1e-8 else delta
        if h == 0: jac[:, i] = 0.0; continue
        params_perturbed_trial[i] += h
        params_full_perturbed = fixed_params.copy()
        for name, value in zip(param_names_opt, params_perturbed_trial): params_full_perturbed[name] = value
        try:
            sol_perturbed = solve_ivp(modelo_fermentacion, [t_exp[0], t_exp[-1]], y0_fit, args=(params_full_perturbed,), t_eval=t_exp, atol=atol, rtol=rtol, method='LSODA')
            if sol_perturbed.status != 0: st.warning(f"Jacobian Disturbed Ferm (p{i}): {sol_perturbed.message}"); jac[:, i] = np.nan; continue
            y_perturbed_flat = sol_perturbed.y[0:n_vars_measured, :].flatten()
            derivative = (y_perturbed_flat - y_nominal_flat) / h
            jac[:, i] = derivative
        except Exception as e: st.error(f"Jacobian Disturbed Ferm Error (p{i}): {e}"); st.text(traceback.format_exc()); jac[:, i] = np.nan
    return jac

# --- Función auxiliar para calcular flujo post-simulación ---
# (Sin cambios respecto a la versión anterior)
def calcular_flujo_post_sim(t, fixed_params):
    """ Calcula F(t) usando los parámetros fijos guardados """
    t_alim_inicio = fixed_params.get("t_alim_inicio", 10.1); t_alim_fin = fixed_params.get("t_alim_fin", 34.1)
    estrategia = fixed_params.get("estrategia", "Constant"); F_base = fixed_params.get("F_base", 0.1)
    F_lineal_fin_val = fixed_params.get("F_lineal_fin", F_base * 2); k_exp_val = fixed_params.get("k_exp", 0.1)
    F = 0.0
    if t_alim_inicio <= t <= t_alim_fin:
        if estrategia == "Constant": F = F_base
        elif estrategia == "Exponential":
             try: F = min(F_base * np.exp(k_exp_val * (t - t_alim_inicio)), F_base * 100)
             except OverflowError: F = F_base * 100
        elif estrategia == "Step": t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2; F = F_base * 2 if t > t_medio else F_base
        elif estrategia == "Linear":
             delta_t = t_alim_fin - t_alim_inicio
             if delta_t > 1e-6: slope = (F_lineal_fin_val - F_base) / delta_t; F = max(0, F_base + slope * (t - t_alim_inicio))
             else: F = F_base
    return max(0, F)

#==========================================================================
# PÁGINA STREAMLIT PARA AJUSTE (¡MODIFICADA!)
#==========================================================================
def parameter_fitting_ferm_page():
    st.header("🔧 Parameter Adjustment - Alcoholic Fermentation Model")

    # --- Inicialización del Estado de Sesión ---
    if 'params_opt_ferm' not in st.session_state: st.session_state.params_opt_ferm = None
    if 'result_ferm' not in st.session_state: st.session_state.result_ferm = None
    if 'parametros_df_ferm' not in st.session_state: st.session_state.parametros_df_ferm = None
    if 'sol_ferm' not in st.session_state: st.session_state.sol_ferm = None
    if 'df_exp_ferm' not in st.session_state: st.session_state.df_exp_ferm = None
    if 'y0_fit_ferm' not in st.session_state: st.session_state.y0_fit_ferm = None
    if 'fixed_params_ferm' not in st.session_state: st.session_state.fixed_params_ferm = {}
    if 'param_config_ferm' not in st.session_state: st.session_state.param_config_ferm = {'names': [], 'initial_guess': [], 'bounds': [], 'units': {}}
    if 't_exp_ferm' not in st.session_state: st.session_state.t_exp_ferm = None
    if 'y_exp_ferm' not in st.session_state: st.session_state.y_exp_ferm = None
    if 'run_complete_ferm' not in st.session_state: st.session_state.run_complete_ferm = False
    if 'last_uploaded_filename_ferm' not in st.session_state: st.session_state.last_uploaded_filename_ferm = None

    # --- Columna Izquierda: Carga de Datos y Configuración ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload Experimental Data")
        uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"], key="ferm_uploader")
        # (Código de carga sin cambios)
        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_uploaded_filename_ferm:
                 st.session_state.params_opt_ferm = None; st.session_state.result_ferm = None
                 st.session_state.run_complete_ferm = False; st.session_state.last_uploaded_filename_ferm = uploaded_file.name
                 st.info(f"New file '{uploaded_file.name}' uploaded.")
            try:
                df_exp = pd.read_excel(uploaded_file, engine='openpyxl')
                required_cols = ['time', 'biomass', 'substrate', 'product', 'oxygen']
                if not all(col in df_exp.columns for col in required_cols):
                    st.error(f"The file must contain the column names: {', '.join(required_cols)}"); st.session_state.df_exp_ferm = None; st.stop()
                else:
                    df_exp = df_exp.sort_values(by='time').reset_index(drop=True)
                    df_exp[required_cols[1:]] = df_exp[required_cols[1:]].apply(pd.to_numeric, errors='coerce')
                    st.session_state.df_exp_ferm = df_exp; st.session_state.t_exp_ferm = df_exp['time'].values
                    st.session_state.y_exp_ferm = df_exp[required_cols[1:]].values.T
                    st.write("Preview:"); st.dataframe(df_exp.head())
            except Exception as e: st.error(f"Error reading file: {e}"); st.session_state.df_exp_ferm = None
        elif st.session_state.df_exp_ferm is not None:
             st.info(f"Using '{st.session_state.last_uploaded_filename_ferm}' data.")

        # --- Configuración (Solo si hay datos) ---
        if st.session_state.df_exp_ferm is not None:
            df_exp = st.session_state.df_exp_ferm
            t_exp = st.session_state.t_exp_ferm
            y_exp_run = st.session_state.y_exp_ferm

            st.subheader("⚙️ Adjustment Settings")
            tipo_mu = st.selectbox("Kinetic Model to Fit", ["Fermentation", "Simple Monod", "Sigmoidal Monod", "Monod with restrictions", "Switched Fermentation"], key="tipo_mu_fit")

            # --- Parámetros Fijos ---
            st.markdown("##### Fixed Parameters (Not optimized)")
            with st.expander("View/Edit Fixed Parameters", expanded=False):
                 # (Inputs con correcciones de tipo float y clamping)
                t_exp_max_val = float(t_exp[-1]) if t_exp is not None and len(t_exp) > 0 else 100.0
                t_exp_min_val = float(t_exp[0]) if t_exp is not None and len(t_exp) > 0 else 0.0
                t_batch_max_val = t_exp_max_val; t_batch_min_val = t_exp_min_val
                t_batch_default_val = max(t_batch_min_val, min(5.0, t_batch_max_val))
                t_batch_inicial_fin_f = st.slider("End Initial Batch Phase [h]", float(t_batch_min_val), float(t_batch_max_val), float(t_batch_default_val), 0.5, key="t_batch_fin_f")
                t_alim_ini_min_val = float(t_batch_inicial_fin_f); t_alim_ini_max_val = t_exp_max_val
                t_alim_ini_default_val = max(t_alim_ini_min_val, min(t_alim_ini_min_val + 0.01, t_alim_ini_max_val))
                t_alim_inicio_f = st.slider("Start Feeding [h]", float(t_alim_ini_min_val), float(t_alim_ini_max_val), float(t_alim_ini_default_val), 0.5, key="t_alim_ini_f")
                t_alim_fin_min_val = float(t_alim_inicio_f) + 0.1; t_alim_fin_max_val = t_exp_max_val + 2.0
                t_alim_fin_default_val = max(t_alim_fin_min_val, min(t_alim_fin_min_val + 24.0, t_alim_fin_max_val))
                t_alim_fin_f = st.slider("End Feeding [h]", float(t_alim_fin_min_val), float(t_alim_fin_max_val), float(t_alim_fin_default_val), 0.05, key="t_alim_fin_f")
                estrategia_f = st.selectbox("Fixed Feeding Strategy", ["Constant", "Exponential", "Linear", "Step"], key="strat_f")
                Sin_f = st.slider("Fixed Sin [g/L]", 10.0, 700.0, 250.0, 10.0, key="sin_f")
                F_base_f = st.slider("Fixed F_base [L/h]", 0.0, 5.0, 0.01, 0.01, key="fbase_f")
                F_lineal_fin_val_f = F_base_f * 2.0; k_exp_val_f = 0.1
                if estrategia_f == "Linear": F_lineal_fin_val_f = st.slider("Fixed F_lineal_fin [L/h]", float(F_base_f), 10.0, max(float(F_base_f), F_base_f * 2.0), 0.01, key="ffin_lin_f")
                elif estrategia_f == "Exponential": k_exp_val_f = st.slider("Fixed k_exp [1/h]", 0.0, 0.5, 0.1, 0.01, key="kexp_f")
                Cs_f = st.slider("Fixed Cs [mg/L]", 0.1, 15.0, 7.5, 0.1, key="cs_f")
                Kla_f = st.slider("Fixed kLa [1/h]", 1.0, 400.0, 100.0, 10.0, key="kla_f")
                O2_controlado_f = st.slider("Fixed Controlled O2 [mg/L]", 0.0, float(Cs_f), 0.08, 0.01, key="o2_control_f")
                st.markdown("--- Other Fixed (Example) ---")
                alpha_lp_f = st.number_input("Fixed alpha_lp [g/g]", 0.0, 10.0, 2.2, format="%.2f", key="alpha_f")
                beta_lp_f = st.number_input("Fixed beta_lp [g/g/h]", 0.0, 1.0, 0.05, format="%.3f", key="beta_f")
                ms_f = st.number_input("Fixed ms [g/g/h]", 0.0, 0.5, 0.02, format="%.3f", key="ms_f")
                mo_f = st.number_input("Fixed mo [g/g/h]", 0.0, 0.2, 0.01, format="%.4f", key="mo_f")
                Kd_f = st.number_input("Fixed Kd [1/h]", 0.0, 0.2, 0.01, format="%.4f", key="kd_f")

            fixed_params = {
                "tipo_mu": tipo_mu, "t_batch_inicial_fin": t_batch_inicial_fin_f, "t_alim_inicio": t_alim_inicio_f,
                "t_alim_fin": t_alim_fin_f, "estrategia": estrategia_f, "Sin": Sin_f, "F_base": F_base_f,
                "F_lineal_fin": F_lineal_fin_val_f, "k_exp": k_exp_val_f, "Kla": Kla_f, "Cs": Cs_f,
                "O2_controlado": O2_controlado_f, "alpha_lp": alpha_lp_f, "beta_lp": beta_lp_f,
                "ms": ms_f, "mo": mo_f, "Kd": Kd_f
            }
            st.session_state.fixed_params_ferm = fixed_params

            # --- Parámetros a Optimizar (¡CON CLAMPING!) ---
            st.markdown("##### Parameters to Optimize")
            param_config = {'names': [], 'initial_guess': [], 'bounds': [], 'units': {}}
            p_opt = st.session_state.params_opt_ferm

            if tipo_mu == "Simple Monod":
                param_config['names'] = ["mumax", "Ks", "Yxs", "Yps", "Yxo"]
                param_config['units'] = {"mumax":"1/h", "Ks":"g/L", "Yxs":"g/g", "Yps":"g/g", "Yxo":"g/g"}
                guesses = []
                params_info_ms = [("μmax_g", 0.01, 2.0, 0.4, "mumax", 0), ("Ks_g", 0.01, 20.0, 1.0, "ks", 1), ("Yxs_g", 0.01, 0.8, 0.1, "yxs", 2), ("Yps_g", 0.1, 0.6, 0.45, "yps", 3), ("Yxo_g", 0.1, 2.0, 0.8, "yxo", 4)]
                for label, min_v, max_v, default_v, key_sfx, idx in params_info_ms:
                    raw_value = float(p_opt[idx]) if p_opt is not None and len(p_opt) > idx else default_v
                    clamped_value = max(min_v, min(raw_value, max_v))
                    guess = st.number_input(label, min_value=min_v, max_value=max_v, value=clamped_value, key=f"g_{key_sfx}")
                    guesses.append(guess)
                param_config['initial_guess'] = guesses
                param_config['bounds'] = [(0.01, 2.0), (0.01, 20.0), (0.01, 0.8), (0.1, 0.6), (0.1, 2.0)]

            elif tipo_mu == "Fermentation":
                param_config['names'] = ["mumax_aerob", "Ks_aerob", "KO_aerob", "mumax_anaerob", "Ks_anaerob", "KiS_anaerob", "KP_anaerob", "n_p", "KO_inhib_anaerob", "Yxs", "Yps", "Yxo"]
                param_config['units'] = {"mumax_aerob": "1/h", "Ks_aerob": "g/L", "KO_aerob": "mg/L", "mumax_anaerob": "1/h", "Ks_anaerob": "g/L", "KiS_anaerob": "g/L", "KP_anaerob": "g/L", "n_p": "-", "KO_inhib_anaerob":"mg/L", "Yxs": "g/g", "Yps": "g/g", "Yxo": "g/g"}
                guesses = []
                params_info = [("μmax_aerob_g", 0.1, 1.0, 0.4, "ma_f", 0), ("Ks_aerob_g", 0.01, 10.0, 0.5, "ksa_f", 1), ("KO_aerob_g", 0.01, 5.0, 0.2, "koa_f", 2), ("μmax_anaerob_g", 0.05, 0.8, 0.15, "man_f", 3), ("Ks_anaerob_g", 0.1, 20.0, 1.0, "ksan_f", 4), ("KiS_anaerob_g", 50.0, 500.0, 150.0, "kisan_f", 5), ("KP_anaerob_g", 20.0, 150.0, 80.0, "kpan_f", 6), ("n_p_g", 0.5, 3.0, 1.0, "np_f", 7), ("KO_inhib_anaerob_g", 0.01, 5.0, 0.1, "koian_f", 8), ("Yxs_g", 0.01, 0.8, 0.1, "yxs_f", 9), ("Yps_g", 0.1, 0.6, 0.45, "yps_f", 10), ("Yxo_g", 0.1, 2.0, 0.8, "yxo_f", 11)]
                for label, min_v, max_v, default_v, key_sfx, idx in params_info:
                    raw_value = float(p_opt[idx]) if p_opt is not None and len(p_opt) > idx else default_v
                    clamped_value = max(min_v, min(raw_value, max_v))
                    guess = st.number_input(label, min_value=min_v, max_value=max_v, value=clamped_value, key=f"g_{key_sfx}")
                    guesses.append(guess)
                param_config['initial_guess'] = guesses
                param_config['bounds'] = [(0.1, 1.0), (0.01, 10.0), (0.01, 5.0), (0.05, 0.8), (0.1, 20.0), (50.0, 500.0), (20.0, 150.0), (0.5, 3.0), (0.01, 5.0), (0.01, 0.8), (0.1, 0.6), (0.1, 2.0)]
            # --- AÑADIR ELIF PARA OTROS MODELOS (aplicando la misma corrección de clamping) ---
            else: st.warning(f"Adjustment for '{tipo_mu}' not implemented."); param_config = {'names': [], 'initial_guess': [], 'bounds': [], 'units': {}}
            st.session_state.param_config_ferm = param_config

            # --- Condiciones Iniciales Experimento ---
            st.markdown("##### Initial Experiment Conditions")
            X0_exp = df_exp['biomass'].iloc[0] if pd.notna(df_exp['biomass'].iloc[0]) else 0.1
            S0_exp = df_exp['substrate'].iloc[0] if pd.notna(df_exp['substrate'].iloc[0]) else 100.0
            P0_exp = df_exp['product'].iloc[0] if pd.notna(df_exp['product'].iloc[0]) else 0.0
            O0_exp = df_exp['oxygen'].iloc[0] if pd.notna(df_exp['oxygen'].iloc[0]) else fixed_params["O2_controlado"]
            V0_exp = fixed_params.get("V0", 0.25);
            if "V0" not in fixed_params: V0_exp = st.number_input("Initial Volume (V0) [L]", 0.01, 100.0, value=V0_exp, key="v0_fit")
            st.write(f"(Usando V0 = {V0_exp:.2f} L)")
            y0_fit = [X0_exp, S0_exp, P0_exp, O0_exp, V0_exp]
            st.session_state.y0_fit_ferm = y0_fit

            # --- Configuración Optimizador ---
            st.markdown("##### Optimization Options")
            metodo_opt = st.selectbox("Optimization Method", ['L-BFGS-B', 'Nelder-Mead', 'differential_evolution'], key="opt_method_ferm")
            max_iter_opt = st.number_input("Maximum Iterations", 50, 10000, 500, key="max_iter_ferm")
            # --- Inputs para Pesos ---
            st.markdown("##### Weights for Objetive Function (Scaled)")
            st.caption("Higher weight = More importance to adjust that variable.")
            cols_w = st.columns(4)
            w_X = cols_w[0].number_input("Biomass Weight (w_X)", 0.1, 100.0, value=1.0, step=0.5, key="w_x") # Increased max y step
            w_S = cols_w[1].number_input("Substrate Weight (w_S)", 0.1, 100.0, value=1.0, step=0.5, key="w_s")
            w_P = cols_w[2].number_input("Product Weight (w_P)", 0.1, 100.0, value=1.0, step=0.5, key="w_p")
            w_O2 = cols_w[3].number_input("Oxygen Weight (w_O2)", 0.1, 100.0, value=1.0, step=0.5, key="w_o2")
            weights_run = [w_X, w_S, w_P, w_O2] # Lista para pasar a la función objetivo
            # Tolerances Solver
            atol_solver = st.number_input("Absolute Tolerance Solver (atol)", 1e-9, 1e-3, 1e-6, format="%e", key="atol_ferm")
            rtol_solver = st.number_input("Relative Tolerance Solver (rtol)", 1e-9, 1e-3, 1e-6, format="%e", key="rtol_ferm")

            # --- Botón de Ejecución ---
            if st.button("🚀 Run Parameters Adjustment", key="run_ferm_fit"):
                if y_exp_run is None: st.error("Upload experimental data.")
                elif not st.session_state.param_config_ferm['names']: st.error("There are no defined parameters to optimize.")
                else:
                    with st.spinner(f"Optimizing {len(param_config['names'])} parameters with {metodo_opt}..."):
                        param_names_to_opt = st.session_state.param_config_ferm['names']
                        initial_guess_to_opt = st.session_state.param_config_ferm['initial_guess']
                        bounds_to_opt = st.session_state.param_config_ferm['bounds']
                        fixed_params_run = st.session_state.fixed_params_ferm
                        y0_run = st.session_state.y0_fit_ferm
                        t_exp_run = st.session_state.t_exp_ferm
                        y_exp_data_run = y_exp_run # Pasar datos (4,n)
                        result = None; st.session_state.run_complete_ferm = False
                        try:
                            if metodo_opt == 'differential_evolution':
                                # Pasar weights_run
                                result = differential_evolution(objetivo_ferm, bounds_to_opt, args=(param_names_to_opt, t_exp_run, y_exp_data_run, y0_run, fixed_params_run, weights_run, atol_solver, rtol_solver), maxiter=max_iter_opt, tol=1e-6, updating='deferred', workers=-1, seed=42, init='latinhypercube')
                            else:
                                 # Pasar weights_run
                                 minimizer_kwargs = {"args": (param_names_to_opt, t_exp_run, y_exp_data_run, y0_run, fixed_params_run, weights_run, atol_solver, rtol_solver), "method": metodo_opt, "bounds": bounds_to_opt if metodo_opt in ['L-BFGS-B', 'TNC', 'SLSQP'] else None, "options": {'maxiter': max_iter_opt, 'disp': False}}
                                 if metodo_opt in ['L-BFGS-B', 'TNC', 'SLSQP']: minimizer_kwargs['options']['ftol'] = 1e-8; minimizer_kwargs['options']['gtol'] = 1e-7
                                 elif metodo_opt == 'Nelder-Mead': minimizer_kwargs['options']['xatol'] = 1e-6; minimizer_kwargs['options']['fatol'] = 1e-8
                                 result = minimize(objetivo_ferm, initial_guess_to_opt, **minimizer_kwargs)
                            if result and hasattr(result, 'success') and result.success: st.session_state.params_opt_ferm = result.x; st.session_state.result_ferm = result; st.session_state.run_complete_ferm = True; st.success(f"Optimization ({metodo_opt}) completed.")
                            elif result and hasattr(result, 'x'): st.session_state.params_opt_ferm = result.x; st.session_state.result_ferm = result; st.session_state.run_complete_ferm = True; st.warning(f"Optimization ({metodo_opt}) finished but it was unsuccessful: {getattr(result, 'message', 'N/A')}")
                            else: st.error(f"Optimization ({metodo_opt}) failed.")
                        except Exception as e: st.error(f"Optimization Error: {e}"); st.text(traceback.format_exc()); st.session_state.params_opt_ferm = None; st.session_state.result_ferm = None; st.session_state.run_complete_ferm = False
                        # st.rerun() # Eliminado

        elif st.session_state.df_exp_ferm is None: st.warning("⏳ Upload experimental data.")

    # --- Columna Derecha: Resultados ---
    with col2:
        if st.session_state.run_complete_ferm and st.session_state.result_ferm is not None:
            st.subheader(f"📊 Adjustment Results (Model: {st.session_state.fixed_params_ferm.get('tipo_mu','N/A')})")
            # (Código de visualización de resultados sin cambios respecto a la versión anterior)
            result = st.session_state.result_ferm; params_opt = st.session_state.params_opt_ferm
            param_names_res = st.session_state.param_config_ferm['names']; param_units_res = st.session_state.param_config_ferm['units']
            fixed_params_res = st.session_state.fixed_params_ferm; y0_res = st.session_state.y0_fit_ferm
            df_exp_res = st.session_state.df_exp_ferm; t_exp_res = st.session_state.t_exp_ferm
            y_exp_res = st.session_state.y_exp_ferm
            # Usar tolerancias actuales de los widgets (idealmente se guardarían las usadas)
            atol_res = atol_solver; rtol_res = rtol_solver

            if params_opt is not None and len(params_opt) == len(param_names_res):
                final_objective_value = getattr(result, 'fun', np.nan)
                st.write(f"**Final Objective Value:** {final_objective_value:.6f}" if pd.notna(final_objective_value) else "**Final Objective Value:** N/A")
                st.caption("(Weighted sum of scaled quadratic errors)")
                parametros_opt_list = [{'Parameter': name, 'Optimized Value': value, 'Units': param_units_res.get(name, 'N/A')} for name, value in zip(param_names_res, params_opt)]
                parametros_df = pd.DataFrame(parametros_opt_list)
                st.dataframe(parametros_df.style.format({'Optimized Value': '{:.5f}'}))
                st.session_state.parametros_df_ferm = parametros_df
            else: st.error("Invalid Optimized Parameters."); st.stop()

            st.subheader("📈 Final Simulation and Metrics (RMSE y R² not-scaled)")
            params_full_res = fixed_params_res.copy()
            for name, value in zip(param_names_res, params_opt): params_full_res[name] = value
            try:
                sol = solve_ivp(modelo_fermentacion, [t_exp_res[0], t_exp_res[-1]], y0_res, args=(params_full_res,), t_eval=t_exp_res, atol=atol_res, rtol=rtol_res, method='LSODA')
                if sol.status != 0: st.error(f"Final Simulation Failed: {sol.message}"); st.session_state.sol_ferm = None
                else:
                     st.session_state.sol_ferm = sol; y_pred_final = sol.y[0:4, :]
                     variables_medidas = ['Biomass', 'Substrate', 'Product', 'Oxygen']; metricas_list = []
                     for i in range(4):
                         exp_data = y_exp_res[i]; pred_data = y_pred_final[i]; valid_mask = ~np.isnan(exp_data)
                         if np.sum(valid_mask) > 1:
                             exp_valid = exp_data[valid_mask]; pred_valid = pred_data[valid_mask]
                             try: r2 = r2_score(exp_valid, pred_valid)
                             except ValueError: r2 = np.nan
                             rmse_var = np.sqrt(mean_squared_error(exp_valid, pred_valid))
                             metricas_list.append({'Variable': variables_medidas[i], 'R²': r2, 'RMSE': rmse_var})
                         else: metricas_list.append({'Variable': variables_medidas[i], 'R²': np.nan, 'RMSE': np.nan})
                     metricas_df = pd.DataFrame(metricas_list)
                     st.dataframe(metricas_df.style.format({'R²': '{:.4f}', 'RMSE': '{:.4f}'}))
            except Exception as e: st.error(f"Final Simulation Error: {e}"); st.text(traceback.format_exc()); st.session_state.sol_ferm = None

            if st.session_state.sol_ferm is not None:
                sol = st.session_state.sol_ferm
                st.subheader("📉 Comparative Data")
                fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True); axes = axes.flatten()
                variables = ['Biomass', 'Substrat', 'Product', 'Oxygen']; unidades = ['g/L', 'g/L', 'g/L', 'mg/L']; colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for i in range(4):
                    ax = axes[i]; ax.plot(t_exp_res, y_exp_res[i], 'o', markersize=5, alpha=0.7, label=f'{variables[i]} Exp.', color=colors[i])
                    ax.plot(sol.t, sol.y[i], '-', linewidth=2, label=f'{variables[i]} Mod.', color=colors[i])
                    ax.set_ylabel(f"{variables[i]} [{unidades[i]}]"); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
                    if i >= 2: ax.set_xlabel("Time [h]")
                    ax.axvline(fixed_params_res["t_batch_inicial_fin"], color='gray', linestyle=':', lw=1, alpha=0.8)
                    ax.axvline(fixed_params_res["t_alim_inicio"], color='orange', linestyle=':', lw=1, alpha=0.8)
                    ax.axvline(fixed_params_res["t_alim_fin"], color='purple', linestyle=':', lw=1, alpha=0.8)
                plt.tight_layout(rect=[0, 0.03, 1, 0.97]); fig.suptitle(f"Comparing Model ({fixed_params_res['tipo_mu']}) vs Experimental Data", fontsize=16); st.pyplot(fig)

                st.subheader("💧 Volume and Feed Flow")
                try:
                    F_t = np.array([calcular_flujo_post_sim(ti, fixed_params_res) for ti in sol.t])
                    fig_vol_feed, ax1 = plt.subplots(figsize=(10, 5))
                    color_vol = 'tab:red'; ax1.set_xlabel('Time [h]'); ax1.set_ylabel('Volume [L]', color=color_vol); ax1.plot(sol.t, sol.y[4], color=color_vol, linestyle='-', linewidth=2, label='Volume (Model)')
                    ax1.tick_params(axis='y', labelcolor=color_vol); ax1.grid(True, which='major', linestyle='--', alpha=0.7); ax1.grid(True, which='minor', linestyle=':', alpha=0.4); ax1.minorticks_on(); ax1.legend(loc='upper left')
                    ax2 = ax1.twinx(); color_feed = 'tab:blue'; ax2.set_ylabel('Feed Flow [L/h]', color=color_feed)
                    if fixed_params_res['estrategia'] in ['Constant', 'Step']: ax2.step(sol.t, F_t, where='post', color=color_feed, linestyle='--', label=f"Flow ({fixed_params_res['estrategia']})")
                    else: ax2.plot(sol.t, F_t, color=color_feed, linestyle='--', label=f"Flow ({fixed_params_res['estrategia']})")
                    ax2.tick_params(axis='y', labelcolor=color_feed); ax2.legend(loc='upper right')
                    fmin_plot = fixed_params_res.get("F_base", 0.0) if fixed_params_res['estrategia'] != "Linear" else min(fixed_params_res.get("F_base", 0.0), fixed_params_res.get("F_lineal_fin", 0.0))
                    fmax_plot = fixed_params_res.get("F_base", 0.1) if fixed_params_res['estrategia'] != "Linear" else max(fixed_params_res.get("F_base", 0.1), fixed_params_res.get("F_lineal_fin", 0.1))
                    ax2.set_ylim(bottom=max(-0.01, fmin_plot * 0.9), top=fmax_plot * 1.1 + 0.01)
                    ax1.axvline(fixed_params_res["t_batch_inicial_fin"], color='gray', linestyle=':', lw=1, alpha=0.8, label='_nolegend_')
                    ax1.axvline(fixed_params_res["t_alim_inicio"], color='orange', linestyle=':', lw=1, alpha=0.8, label='_nolegend_')
                    ax1.axvline(fixed_params_res["t_alim_fin"], color='purple', linestyle=':', lw=1, alpha=0.8, label='_nolegend_')
                    fig_vol_feed.tight_layout(); plt.title("Volume and Feed Flow Evolution"); st.pyplot(fig_vol_feed)
                except Exception as e: st.warning(f"Volume/Flow graph could not be generated: {e}")

                st.subheader("📈 Statistical Analysis")
                with st.spinner("Calculating confidence intervals..."):
                     try:
                         residuals = y_exp_res[0:4, :] - y_pred_final; residuals_flat = residuals.flatten()
                         residuals_flat_clean = residuals_flat[~np.isnan(residuals_flat)]
                         n_obs_clean = len(residuals_flat_clean); n_params_opt_res = len(params_opt); dof = n_obs_clean - n_params_opt_res
                         if dof <= 0:
                              st.warning(f"Not enough data points ({n_obs_clean}) to calculate CI (needed > {n_params_opt_res}).")
                              for col in ['Standard Error', 'Interval ± (95%)', '95% CI Lower bound', '95% CI Upper bound']:
                                   if col not in parametros_df.columns: parametros_df[col] = np.nan
                         else:
                              jac = compute_jacobian_ferm(params_opt, param_names_res, t_exp_res, y0_res, fixed_params_res, atol_res, rtol_res)
                              if jac is None or np.isnan(jac).any() or np.isinf(jac).any():
                                   st.warning("Invalid Jacobian. CI cannot be calculated.")
                                   for col in ['Standard Error', 'Interval ± (95%)', '95% CI Lower bound', '95% CI Upper bound']:
                                        if col not in parametros_df.columns: parametros_df[col] = np.nan
                              else:
                                   mse = np.sum(residuals_flat_clean**2) / dof; jtj = jac.T @ jac
                                   try: cov_matrix = np.linalg.pinv(jtj) * mse
                                   except np.linalg.LinAlgError: cov_matrix = np.linalg.pinv(jtj) * mse
                                   diag_cov = np.diag(cov_matrix); valid_variance = diag_cov > 1e-15
                                   std_errors = np.full_like(diag_cov, np.nan); std_errors[valid_variance] = np.sqrt(diag_cov[valid_variance])
                                   if np.any(~valid_variance): st.warning("Non-positive variances found.")
                                   alpha = 0.05; t_val = t.ppf(1.0 - alpha / 2.0, df=dof); intervals = t_val * std_errors
                                   parametros_df['Standard Error'] = std_errors; parametros_df['Interval ± (95%)'] = intervals
                                   parametros_df['95% CI Lower bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] - intervals)
                                   parametros_df['95% CI Upper bound'] = np.where(np.isnan(intervals), np.nan, parametros_df['Optimized Value'] + intervals)
                                   st.success("Confidence Intervals Calculated.")
                     except Exception as e:
                          st.error(f"Error calculating CI: {e}"); st.text(traceback.format_exc())
                          for col in ['Standard Error', 'Interval ± (95%)', '95% CI - Lower bound', '95% CI - Upper bound']:
                              if col not in parametros_df.columns: parametros_df[col] = np.nan
                     st.write("Optimized Parameters and Confidence Intervals (95%):")
                     st.dataframe(parametros_df.style.format({'Optimized Value': '{:.5f}', 'Standard Error': '{:.5f}', 'Interval ± (95%)': '{:.5f}', '95% CI - Lower bound': '{:.5f}', '95% CI - Upper bound': '{:.5f}'}, na_rep='N/A'))
                     st.session_state.parametros_df_ferm = parametros_df

                     if 'Interval ± (95%)' in parametros_df.columns and parametros_df['Interval ± (95%)'].notna().any():
                          st.subheader("📐 Confidence Intervals for Parameters")
                          fig_ci, ax = plt.subplots(figsize=(10, max(4, len(parametros_df) * 0.6)))
                          y_pos = np.arange(len(parametros_df)); errors_for_plot = parametros_df['Interval ± (95%)'].copy()
                          mask_nan_interval = errors_for_plot.isna() & parametros_df['Standard Error'].notna()
                          errors_for_plot[mask_nan_interval] = 1.96 * parametros_df['Standard Error'][mask_nan_interval]
                          errors_for_plot = errors_for_plot.fillna(0).values
                          bars = ax.barh(y_pos, parametros_df['Optimized Value'].fillna(0), xerr=errors_for_plot, align='center', color='#1f77b4', ecolor='#ff7f0e', capsize=5, alpha=0.8)
                          ax.set_yticks(y_pos); ax.set_yticklabels(parametros_df['Parameter']); ax.invert_yaxis()
                          ax.set_xlabel('Parameter Value'); ax.set_title('95% Confidence Intervals (o ~1.96*SE)'); ax.grid(True, axis='x', linestyle='--', alpha=0.6)
                          plt.tight_layout(); st.pyplot(fig_ci)

                     st.subheader("📉 Residuals Analysis")
                     fig_hist, axs = plt.subplots(2, 2, figsize=(12, 8)); axs = axs.flatten()
                     variables_res = ['Biomass', 'Substrate', 'Product', 'Oxygen']; colors_res = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                     for i, (var, color) in enumerate(zip(variables_res, colors_res)):
                         ax = axs[i]; res = y_exp_res[i] - y_pred_final[i]; res_clean = res[~np.isnan(res)]
                         if len(res_clean) > 1:
                             sns.histplot(res_clean, kde=True, color=color, ax=ax, bins='auto')
                             ax.set_title(f'Residuals {var} (N={len(res_clean)})'); ax.set_xlabel('Error (Exp - Mod)'); ax.set_ylabel('Frecuence/Density')
                             ax.axvline(0, color='k', linestyle='--'); ax.grid(True, linestyle='--', alpha=0.3)
                             mean_res = np.mean(res_clean); std_res = np.std(res_clean)
                             ax.text(0.05, 0.95, f'Mean={mean_res:.2f}\nStd={std_res:.2f}', transform=ax.transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
                         else: ax.set_title(f'Residuals {var} (Insuf.)'); ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                     plt.tight_layout(); st.pyplot(fig_hist)

                     if 'cov_matrix' in locals() and cov_matrix is not None and not np.isnan(cov_matrix).all():
                          st.subheader("📌 Parameter Correlation Matrix")
                          try:
                              n_p_corr = cov_matrix.shape[0]; param_names_corr = parametros_df['Parameter'].tolist()
                              if len(param_names_corr) == n_p_corr:
                                   std_devs = np.sqrt(np.diag(cov_matrix));
                                   with np.errstate(divide='ignore', invalid='ignore'): corr_matrix_calc = cov_matrix / np.outer(std_devs, std_devs)
                                   corr_matrix_calc[~np.isfinite(corr_matrix_calc)] = np.nan; np.fill_diagonal(corr_matrix_calc, 1.0); corr_matrix_calc = np.clip(corr_matrix_calc, -1.0, 1.0)
                                   corr_df = pd.DataFrame(corr_matrix_calc, index=param_names_corr, columns=param_names_corr)
                                   fig_corr, ax = plt.subplots(figsize=(max(6, n_p_corr*0.8), max(5, n_p_corr*0.7)))
                                   sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", center=0, linewidths=.5, linecolor='black', ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
                                   ax.set_title('Estimated Correlation Matrix'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                                   plt.tight_layout(); st.pyplot(fig_corr)
                              else: st.warning("Covariance discrepancy between Names/matrix.")
                          except Exception as e: st.warning(f"Correlation Matrix could not be plotted: {e}")
                     else: st.info("Correlation Matrix not available.")

            else: # Si simulación falló
                st.info("Complete adjustment and final simulation to see the analysis.")

        elif not st.session_state.run_complete_ferm: # Mensaje inicial
             st.info("⬅️ Please upload data file, configure adjustment and run.")

# --- Ejecución Principal ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Fermentation Model Fit")
    ajuste_parametros_ferm_page()