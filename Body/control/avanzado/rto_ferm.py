# -*- coding: utf-8 -*-
import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D # Para leyendas personalizadas
import traceback # Para mostrar errores detallados
import time # Para medir tiempos
from io import BytesIO

# Import CasADi-compatible kinetic functions from Utils
from Utils.kinetics import (
    mu_monod_rto as mu_monod_ca,
    mu_sigmoidal_rto as mu_sigmoidal_ca,
    mu_completa_rto as mu_completa_ca,
    mu_fermentacion_rto as mu_fermentacion_ca
)

# ====================================================
# --- Página de Streamlit ---
# ====================================================
def rto_fermentation_page():
     st.header("🧠 RTO Control - Alcoholic Fermentation")
     st.markdown("""
     Optimization of the feed profile ($F(t)$) to maximize $P_{final} V_{final}$.
     **Method:** Orthogonal Collocation (Radau, d=2). Slacks O2+S. N=12.
     **Correction:** Modified qP/dSdt balances.
     """)

     with st.sidebar:
          # --- Configuración Sidebar ---
          # (Idéntico a la versión anterior, colapsado)
          st.subheader("1. Kinetic Model and Parameters")
          tipo_mu = st.selectbox("Kinetic Model (μ)", ["Switched Fermentation", "Fermentation", "Simple Monod", "Sigmoidal Monod", "Monod with Restrictions"], index=1)
          Ks_base = st.number_input("Base Ks (default) [g/L]", 0.01, 10.0, 1.0, 0.1, key="ks_base")
          params_cineticos = {}
          params_cineticos['tipo_mu'] = tipo_mu
          if tipo_mu == "Simple Monod":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_simple")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_simple")
               params_cineticos.update({k: 1e9 for k in ['KO', 'KP_gen', 'n_sig', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Sigmoidal Monod":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_sig")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_sig")
               params_cineticos['n_sig'] = st.slider("sigmoidal Power (n)", 1.0, 5.0, 2.0, 0.1, key="n_sig")
               params_cineticos.update({k: 1e9 for k in ['KO', 'KP_gen', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Monod with Restrictions":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_restr")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_restr")
               params_cineticos['KO'] = st.slider("KO (O2 - restriction) [g/L]", 0.0001, 0.05, 0.002, 0.0001, format="%.4f", key="ko_restr")
               params_cineticos['KP_gen'] = st.slider("KP (Generic product Inhib.) [g/L]", 1.0, 100.0, 50.0, 1.0, key="kp_gen")
               params_cineticos.update({k: 1e9 for k in ['n_sig', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Switched Fermentation":
               st.info("Switched Model: Phase 1 uses aerobic kinetics, Phases 2 and 3 use anaerobic kinetics.")
               params_cineticos['mumax_aerob'] = st.slider("μmax (Aerobic) [1/h]", 0.1, 1.0, 0.45, 0.05, key="mumax_aero_c")
               params_cineticos['Ks_aerob'] = st.slider("Ks (Aerobic) [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aero_c")
               params_cineticos['KO_aerob'] = st.slider("KO (O2 Affinity - Aerobic) [g/L]", 0.0001, 0.05, 0.002, 0.0001, format="%.4f", key="ko_aero_c")
               params_cineticos['mumax_anaerob'] = st.slider("μmax (Anaerobic) [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaero_c")
               params_cineticos['Ks_anaerob'] = st.slider("Ks (Anaerobic) [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaero_c")
               params_cineticos['KiS_anaerob'] = st.slider("KiS (Inhib. Sustrato - Anaerobic) [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaero_c")
               params_cineticos['KP_anaerob'] = st.slider("KP (Inhib. Ethanol - Anaerobic) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaero_c")
               params_cineticos['n_p'] = st.slider("Ethanol Inhib.Exponent (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_anaero_c")
               params_cineticos['KO_inhib_anaerob'] = st.slider("KO_inhib (O2 Inhib. in Anaerobic μ) [g/L]", 1e-6, 0.01, 0.0005, 1e-6, format="%.6f", key="ko_inhib_anaero_c")
               params_cineticos.update({k: 1e9 for k in ['Ks', 'KO', 'KP_gen', 'n_sig','mumax']})
          elif tipo_mu == "Fermentation":
               st.info("Mixed model: mu = mu_aerobic + mu_anaerobic.")
               with st.expander("mu Parameters (Aerobic)", expanded=True):
                    ko_aerob_val = st.slider("KO_aerob (O2 affinity) [g/L]", 0.0001, 0.05, 0.0002, 0.0001, format="%.4f", key="ko_aerob_m")
                    params_cineticos['mumax_aerob'] = st.slider("μmax_aerob [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_aerob_m")
                    params_cineticos['Ks_aerob'] = st.slider("Ks_aerob [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aerob_m")
                    params_cineticos['KO_aerob'] = ko_aerob_val
               with st.expander("mu Parameterss (Anaerobic/Fermentative)", expanded=True):
                    ko_inhib_anaerob_val = st.slider("KO_inhib_anaerob (O2 Inhib. in Anaerobic μ) [g/L]", 1e-6, 0.01, 0.0005, 1e-6, format="%.6f", key="ko_inhib_m")
                    params_cineticos['mumax_anaerob'] = st.slider("μmax_anaerob [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaerob_m")
                    params_cineticos['Ks_anaerob'] = st.slider("Ks_anaerob [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaerob_m")
                    params_cineticos['KiS_anaerob'] = st.slider("KiS_anaerob [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaerob_m")
                    params_cineticos['KP_anaerob'] = st.slider("KP_anaerob (Inhib. Ethanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaerob_m")
                    params_cineticos['n_p'] = st.slider("Ethanol Inhib. Exponent (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_m")
                    params_cineticos['KO_inhib_anaerob'] = ko_inhib_anaerob_val
                    if abs(params_cineticos['KO_aerob'] - params_cineticos['KO_inhib_anaerob']) < 0.0002: st.warning(f"KO_aerob and KO_inhib_anaerob nearby.")
               params_cineticos.update({k: 1e9 for k in ['Ks', 'KO', 'KP_gen', 'n_sig', 'mumax']})
          st.subheader("2. Stoichiometric Parameters and Others")
          params_esteq = {}
          params_esteq['Yxs'] = st.slider("Yxs [g/g]", 0.05, 0.6, 0.1, 0.01, key="yxs"); params_esteq['Yps'] = st.slider("Yps [g/g]", 0.1, 0.51, 0.45, 0.01, key="yps"); params_esteq['Yxo'] = st.slider("Yxo [gX/gO2]", 0.1, 2.0, 0.8, 0.1, key="yxo"); params_esteq['alpha_lp'] = st.slider("α [gP/gX]", 0.0, 5.0, 2.2, 0.1, key="alpha"); params_esteq['beta_lp'] = st.slider("β [gP/gX/h]", 0.0, 0.5, 0.05, 0.01, key="beta"); params_esteq['ms'] = st.slider("ms [gS/gX/h]", 0.0, 0.2, 0.02, 0.01, key="ms"); params_esteq['mo'] = st.slider("mo [gO2/gX/h]", 0.0, 0.1, 0.01, 0.005, key="mo"); params_esteq['Kd'] = st.slider("Kd [1/h]", 0.0, 0.1, 0.01, 0.005, key="kd"); ko_inhib_p_mgL = st.slider("KO_inhib_prod [mg/L]", 0.001, 1.0, 0.05, 0.005, key="ko_inhib_p_mgL"); params_esteq['KO_inhib_prod'] = ko_inhib_p_mgL / 1000.0
          st.subheader("3. Oxygen Transfer")
          params_transfer = {}; params_transfer['Kla1'] = st.slider("kLa Phase 1 [1/h]", 10.0, 800.0, 100.0, 10.0, key="kla1"); params_transfer['Kla2'] = st.slider("kLa Phase 2/3 [1/h]", 0.0, 50.0, 15.0, 0.1, key="kla2"); Cs_mgL = st.slider("Cs [mg/L]", 1.0, 15.0, 7.5, 0.1, key="cs_mgL"); params_transfer['Cs'] = Cs_mgL / 1000.0
          st.subheader("4. Feeding and Reactor")
          params_reactor = {}; params_reactor['Sin'] = st.number_input("Sin [g/L]", 10.0, 700.0, 250.0, 10.0, key="sin_conc"); params_reactor['Vmax'] = st.number_input("Vmax [L]", value=0.5, min_value=0.1, step=0.5, key="vmax_reactor")
          st.subheader("5. Temporary Configuration and Collocation")
          params_tiempo = {}
          t_aerobic_batch_val = st.number_input("End Phase 1 [h]", value=4.0, min_value=0.1, step=0.5, key="t_aerobic_end")
          params_tiempo['t_aerobic_batch'] = t_aerobic_batch_val
          t_feed_end_val = st.number_input("End Phase 2 [h]", value=9.0, min_value=params_tiempo['t_aerobic_batch'] + 0.1, step=0.5, key="t_feed_end_rto")
          params_tiempo['t_feed_end'] = t_feed_end_val
          t_total_val = st.number_input("Total Time [h]", value=16.0, min_value=params_tiempo['t_feed_end'], step=0.5, key="t_total_rto")
          params_tiempo['t_total'] = t_total_val
          feed_duration = params_tiempo['t_feed_end'] - params_tiempo['t_aerobic_batch']
          n_intervals_val = st.number_input("Finite Intervals (N)", value=12, min_value=1, max_value=100, step=1, key="n_intervals_rto_coll", help=f"Number of intervals where F is constant. Duration Phase 2: {feed_duration:.1f} h")
          params_tiempo['n_intervals'] = n_intervals_val
          degree_val = st.number_input("Collocation Degree (d)", value=4, min_value=1, max_value=5, step=1, key="degree_coll", help="Degree of the polynomial used in each interval. d=2 or d=3 is usually a good compromise.")
          params_tiempo['degree'] = degree_val
          collocation_scheme = st.selectbox("Collocation Scheme", ["radau", "legendre"], index=0, key="scheme_coll", help="'radau' es generally good for DAEs stiff.")
          params_tiempo['scheme'] = collocation_scheme
          st.subheader("6. Initial Conditions (t=0)")
          cond_iniciales = {}; cond_iniciales['X0'] = st.number_input("X0 [g/L]", 0.01, 10.0, 1.20, step=0.01, key="x0_init"); cond_iniciales['S0'] = st.number_input("S0 [g/L]", 1.0, 200.0, 10.17, step=1.0, key="s0_init"); cond_iniciales['P0'] = st.number_input("P0 [g/L]", 0.0, 50.0, 0.0, step=0.1, key="p0_init"); o0_default_mgL = min(params_transfer['Cs'] * 1000 * 0.95, 0.08); O0_mgL = st.number_input("O0 [mg/L]", min_value=0.0, max_value=Cs_mgL, value=o0_default_mgL, step=0.01, key="o0_mgL", help=f"Maximum: {Cs_mgL:.2f} mg/L"); cond_iniciales['O0'] = O0_mgL / 1000.0; cond_iniciales['V0'] = st.number_input("V0 [L]", value=0.25, min_value=0.05, step=0.1, key="v0_init")
          st.subheader("7. RTO Constraints and Penalty")
          params_rto = {}; fmin_val = st.number_input("Fmin [L/h]", value=0.01, min_value=0.0, format="%.4f", key="fmin_rto"); params_rto['Fmin'] = fmin_val; params_rto['Fmax'] = st.number_input("Fmax [L/h]", value=0.26, min_value=params_rto['Fmin'], step=0.01, key="fmax_rto");
          params_rto['Smax_constraint'] = st.number_input("Smax [g/L]", value=30.0, min_value=0.1, step=1.0, key="smax_const_rto", help="Upper limit for S during feeding.")
          default_pmax_rto = 100.0; kp_to_use = None
          if tipo_mu in ["Fermentation", "Switched Fermentation"]: kp_to_use = params_cineticos.get('KP_anaerob', None)
          elif tipo_mu == "Monod with Restrictions": kp_to_use = params_cineticos.get('KP_gen', None)
          if kp_to_use is not None and isinstance(kp_to_use, (int, float)) and kp_to_use > 1e-6 and kp_to_use < 1e8: default_pmax_rto = max(10.0, kp_to_use * 0.95)
          params_rto['Pmax_constraint'] = st.number_input("Pmax [g/L]", value=default_pmax_rto, min_value=1.0, step=1.0, key="pmax_const_rto", help=f"Suggested P. Limit: ~95% KP ({kp_to_use:.1f} g/L if applicable)."); w_Smax_penalty_val = st.number_input("Weight Penalty Smax", value=1000.0, min_value=0.0, key="w_smax_rto", help="Set to 0 to disable."); params_rto['w_Smax_penalty'] = w_Smax_penalty_val
          params_rto['w_O2_slack'] = st.number_input("Slack O2 Weight", value=1e6, min_value=0.0, format="%.2e", key="w_o2_slack", help="High weight to penalize violations O2>=0.")
          params_rto['w_S_slack'] = st.number_input("Slack S Weight", value=1e6, min_value=0.0, format="%.2e", key="w_s_slack", help="High weight to penalize violations S>=0.")

          all_params = {**params_cineticos, **params_esteq, **params_transfer, **params_reactor, **params_tiempo, **params_rto}

     # --- Factores de Escalado ---
     # (Sin cambios)
     X_scale = max(1.0, cond_iniciales['X0'] * 10, 50.0); S_scale = max(1.0, cond_iniciales['S0'], params_reactor['Sin'], params_rto['Smax_constraint']); P_scale = max(1.0, params_rto.get('Pmax_constraint', 100.0)); O2_scale = max(1e-5, params_transfer['Cs']); V_scale = max(0.1, params_reactor['Vmax'], cond_iniciales['V0']); F_scale = max(1e-3, params_rto['Fmax']) if params_rto['Fmax'] > 0 else 0.1; x_scales = ca.vertcat(X_scale, S_scale, P_scale, O2_scale, V_scale); u_scale = F_scale

     # --- Función DAE/ODE ESCALADO (para usar con Colocación) ---
     def create_ode_function_scaled(params, scales):
          # Define variables simbólicas
          nx = 5
          x_scaled_sym = ca.MX.sym("x_scaled", nx)
          u_scaled_sym = ca.MX.sym("u_scaled")
          Kla_sym = ca.MX.sym("Kla_phase")
          considerar_O2_sym = ca.MX.sym("considerar_O2_flag") # Siempre presente, valdrá 0 si no es conmutado
          p_ode_sym = ca.vertcat(Kla_sym, considerar_O2_sym) # Parámetros para la ODE

          # Desescalar
          x_sc = scales['x']; u_sc = scales['u']
          x_orig_sym = x_scaled_sym * x_sc
          u_orig_sym = u_scaled_sym * u_sc
          X, S, P, O2, V = x_orig_sym[0], x_orig_sym[1], x_orig_sym[2], x_orig_sym[3], x_orig_sym[4]
          F = u_orig_sym
          safe_X = ca.fmax(1e-9, X); safe_S = ca.fmax(0.0, S); safe_P = ca.fmax(0.0, P); safe_O2 = ca.fmax(0.0, O2); safe_V = ca.fmax(1e-6, V)
          D = F / safe_V

          # --- Cálculo de mu (total), mu_aer y mu_anaer ---
          mu = 0.0
          mu_aer = 0.0
          mu_anaer = 0.0
          tipo = params['tipo_mu']

          # Calcular componentes aerobio y anaerobio si el modelo los usa
          if tipo in ["Fermentation", "Switched Fermentation"]:
               # Componente Aerobio
               mu_aer = mu_fermentacion_ca(safe_S, safe_P, safe_O2, params['mumax_aerob'], params['Ks_aerob'], params['KO_aerob'], 0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
               # Componente Anaerobio
               mu_anaer = mu_fermentacion_ca(safe_S, safe_P, safe_O2, 0, 1, float('inf'), params['mumax_anaerob'], params['Ks_anaerob'], params['KiS_anaerob'], params['KP_anaerob'], params.get('n_p', 1.0), params['KO_inhib_anaerob'], considerar_O2=False)
               # Combinar según el tipo de modelo
               if tipo == "Fermentation":
                    mu = mu_aer + mu_anaer
               elif tipo == "Switched Fermentation":
                    # El flag p_ode_sym[1] decide cuál usar
                    mu = ca.if_else(p_ode_sym[1] > 0.5, mu_aer, mu_anaer)
                    # Asegurar que mu_aer o mu_anaer sea cero si no es la fase activa
                    # (Importante para los balances corregidos)
                    mu_aer = ca.if_else(p_ode_sym[1] > 0.5, mu_aer, 0.0)
                    mu_anaer = ca.if_else(p_ode_sym[1] < 0.5, mu_anaer, 0.0)

          elif tipo == "Simple Monod":
               mu = mu_monod_ca(safe_S, params['mumax'], params['Ks'])
               mu_aer = mu # Asumir todo aerobio
               mu_anaer = 0.0
          elif tipo == "Sigmoidal Monod":
               mu = mu_sigmoidal_ca(safe_S, params['mumax'], params['Ks'], params['n_sig'])
               mu_aer = mu # Asumir todo aerobio
               mu_anaer = 0.0
          elif tipo == "Monod with Restrictions":
               mu = mu_completa_ca(safe_S, safe_O2, safe_P, params['mumax'], params['Ks'], params['KO'], params['KP_gen'])
               mu_aer = mu # Asumir todo aerobio
               mu_anaer = 0.0

          # Tasa de crecimiento neta (considerando muerte/declinación)
          mu_net = ca.fmax(0.0, mu) - params['Kd']

          # --- Calcular Tasa Específica de Producción de Producto (qP) ---
          # <<< CORRECCIÓN qP: Usar SOLO mu_anaer para el término asociado a crecimiento >>>
          # <<< CORRECCIÓN qP: Eliminar inhibición por O2 (KO_inhib_prod) >>>
          qP = params["alpha_lp"] * mu_anaer + params["beta_lp"]
          qP = ca.fmax(0.0, qP) # Asegurar no negatividad

          # --- Calcular Tasa Específica de Consumo de Sustrato (qS) ---
          # <<< CORRECCIÓN dSdt: Usar SOLO mu_aer para el consumo por crecimiento >>>
          consumo_S_X = (mu_aer / ca.fmax(params['Yxs'], 1e-9)) # Usa mu_aer
          consumo_S_P = (qP / ca.fmax(params['Yps'], 1e-9)) # Usa qP (que depende de mu_anaer)
          consumo_S_maint = params['ms']
          qS = consumo_S_X + consumo_S_P + consumo_S_maint
          qS = ca.fmax(0.0, qS)

          # --- Calcular Tasa Específica de Consumo de Oxígeno (qO) ---
          # <<< CORRECCIÓN dOdt: qO solo depende de mu_aer >>>
          consumo_O2_X = (mu_aer / ca.fmax(params['Yxo'], 1e-9))
          consumo_O2_maint = params['mo']
          qO = consumo_O2_X + consumo_O2_maint
          qO = ca.fmax(0.0, qO)

          # Calcular Tasas Volumétricas
          Rate_X = mu_net * safe_X
          Rate_S = -qS * safe_X
          Rate_P = qP * safe_X # rate_P corregido
          OUR = qO * safe_X # OUR corregido

          # Calcular Tasa de Transferencia de Oxígeno (OTR)
          current_Kla = p_ode_sym[0] # Usa Kla del parámetro
          OTR = current_Kla * (params['Cs'] - safe_O2)

          # Calcular Tasa Neta de Cambio de Oxígeno Disuelto
          Rate_O2 = OTR - OUR

          # --- Ecuaciones Diferenciales (Balances de Masa) ---
          dXdt = Rate_X - D * safe_X
          dSdt = Rate_S + D * (params['Sin'] - safe_S) # rate_S corregido
          dPdt = Rate_P - D * safe_P # rate_P corregido
          dOdt = Rate_O2 - D * safe_O2 # OUR corregido
          dVdt = F

          # --- Expresiones ODE Escaladas ---
          ode_expr_scaled = ca.vertcat(
               dXdt / x_sc[0],
               dSdt / x_sc[1],
               dPdt / x_sc[2],
               dOdt / x_sc[3],
               dVdt / x_sc[4]
          )

          # Crear y devolver la función CasADi
          ode_func = ca.Function('ode_func_scaled', [x_scaled_sym, u_scaled_sym, p_ode_sym], [ode_expr_scaled],
                                   ['x', 'u', 'p'], ['dxdt'])
          return ode_func

     # --- Ejecución de la Optimización RTO ---
     if st.button("🚀 Run RTO Optimization"):
          st.info("Starting RTO Optimization with Orthogonal Collocation...")
          start_time_rto = time.time()

          # --- Validaciones y Tiempos ---
          t_fase1 = all_params['t_aerobic_batch']; t_fase2_end = all_params['t_feed_end']; t_fase3_end = all_params['t_total'];
          n_intervals = all_params['n_intervals'] # Número de intervalos finitos
          degree = all_params['degree']           # Grado de colocación
          T_feed_duration = t_fase2_end - t_fase1; T_post_feed_duration = t_fase3_end - t_fase2_end
          if T_feed_duration <= 1e-6 or n_intervals <= 0: st.error(f"Invalid temporary Conf. Phase 2."); st.stop()
          if T_post_feed_duration < -1e-6: st.error(f"Invalid temporary Conf. Phase 3."); st.stop()
          h_interval = T_feed_duration / n_intervals # Duración de cada intervalo finito
          if u_scale <= 1e-9: st.warning(f"F_scale bajo ({u_scale:.2e}). Usando 1.0."); u_scale = 1.0

          # --- Crear Función ODE Escalada ---
          try:
               ode_func_scaled = create_ode_function_scaled(all_params, {'x': x_scales, 'u': u_scale})
               st.success("Scaled ODE function for created collocation.")
          except Exception as e: st.error(f"Error creating ODE function: {e}"); st.error(traceback.format_exc()); st.stop()

          # --- Definir Parámetros (p) para cada Fase ---
          # Parámetros para la función ODE [Kla, considerar_O2_flag]
          p_list_phase1_ode = [all_params['Kla1']]
          p_list_phase2_ode = [all_params['Kla2']]
          p_list_phase3_ode = [all_params['Kla2']]
          if all_params['tipo_mu'] == "Switched Fermentation":
               p_list_phase1_ode.append(1.0); p_list_phase2_ode.append(0.0); p_list_phase3_ode.append(0.0)
          else:
               p_list_phase1_ode.append(0.0); p_list_phase2_ode.append(0.0); p_list_phase3_ode.append(0.0)
          p_phase1_ode = ca.vertcat(*p_list_phase1_ode)
          p_phase2_ode = ca.vertcat(*p_list_phase2_ode)
          p_phase3_ode = ca.vertcat(*p_list_phase3_ode)

          # --- Simulación Fase 1 (Batch Aerobio) ---
          # (Sin cambios)
          st.info(f"[PHASE 1] Simulating aerobic batch up to t={t_fase1:.1f} h...")
          try:
               x_sym_f1 = ca.MX.sym("x", 5); p_sym_f1 = ca.MX.sym("p", 2)
               ode_expr_f1 = ode_func_scaled(x_sym_f1, 0.0, p_sym_f1)
               dae_dict_f1 = {'x': x_sym_f1, 'p': p_sym_f1, 'ode': ode_expr_f1}
               integrator_opts_sim = {"t0": 0, "tf": t_fase1, "reltol": 1e-6, "abstol": 1e-8}
               integrator_phase1 = ca.integrator("integrator_p1", "idas", dae_dict_f1, integrator_opts_sim)
               x0_np_orig = np.array([cond_iniciales['X0'], cond_iniciales['S0'], cond_iniciales['P0'], cond_iniciales['O0'], cond_iniciales['V0']])
               x0_np_scaled = x0_np_orig / np.array(x_scales).flatten()
               res_phase1_scaled = integrator_phase1(x0=x0_np_scaled, p=p_phase1_ode)
               x_end_phase1_scaled = np.array(res_phase1_scaled['xf']).flatten()
               if any(np.isnan(x_end_phase1_scaled)): raise ValueError("NaN Phase 1.")
               x_end_phase1_scaled = np.maximum(x_end_phase1_scaled, 0.0)
               x_end_phase1 = x_end_phase1_scaled * np.array(x_scales).flatten()
               st.success(f"[PHASE 1] Completed. Final: X={x_end_phase1[0]:.3f}, S={x_end_phase1[1]:.3f}, P={x_end_phase1[2]:.3f}, O2={x_end_phase1[3]*1000:.4f}, V={x_end_phase1[4]:.3f}")
          except Exception as e: st.error(f"Error Phase 1: {e}"); st.error(traceback.format_exc()); st.stop()

          # --- Formulación del Problema de Optimización por Colocación (Fase 2) ---
          st.info(f"[PHASE 2] Formulating RTO with Collocation (N={n_intervals}, d={degree})...")
          opti = ca.Opti()
          nx = 5 # Número de estados
          collocation_method = all_params['scheme'] # 'radau' o 'legendre'

          # Variables de estado en puntos de colocación
          X_phase2_coll = [] # Lista de variables de estado para cada intervalo
          # Variables de control (una por intervalo)
          F_phase2_scaled = opti.variable(n_intervals)
          # Variables de holgura para O2 >= 0
          Slack_O2_phase2 = opti.variable(n_intervals, degree + 1)
          # Variables de holgura para S >= 0
          Slack_S_phase2 = opti.variable(n_intervals, degree + 1)

          # Parámetro para el estado inicial de Fase 2
          x_start_phase2_param = opti.parameter(nx)
          opti.set_value(x_start_phase2_param, x_end_phase1_scaled)

          # Bucle sobre los intervalos finitos
          Xk_end_prev = x_start_phase2_param # Estado al final del intervalo anterior (inicializado)
          for k in range(n_intervals):
               # --- Variables de estado para el intervalo k ---
               Xk = opti.variable(nx, degree + 1)
               X_phase2_coll.append(Xk)
               # --- Variable de holgura para este intervalo ---
               slack_o2_k = Slack_O2_phase2[k, :]
               slack_s_k = Slack_S_phase2[k, :] # Slack para S

               # --- Restricción de continuidad ---
               opti.subject_to(Xk[:, 0] == Xk_end_prev)

               # --- Restricciones de colocación ---
               tau = ca.collocation_points(degree, collocation_method)
               C, D, B = ca.collocation_coeff(tau)
               ode_at_coll = ode_func_scaled(Xk[:, 1:], F_phase2_scaled[k], p_phase2_ode)
               for j in range(degree):
                    xp_kj = 0
                    for r in range(degree + 1):
                         xp_kj += Xk[:, r] * C[r, j]
                    opti.subject_to(h_interval * ode_at_coll[:, j] == xp_kj)

               # --- Estado al final del intervalo k ---
               Xk_end_current = 0
               for r in range(degree + 1):
                    Xk_end_current += Xk[:, r] * D[r]
               Xk_end_prev = Xk_end_current

               # --- Restricciones adicionales en el intervalo k ---
               Fmin_sc = all_params['Fmin'] / u_scale if u_scale > 1e-9 else 0.0
               Fmax_sc = all_params['Fmax'] / u_scale if u_scale > 1e-9 else 1.0
               opti.subject_to(F_phase2_scaled[k] >= Fmin_sc)
               opti.subject_to(F_phase2_scaled[k] <= Fmax_sc)

               # Restricciones en los estados en todos los puntos (inicio + colocación)
               Vmax_sc = all_params['Vmax'] / V_scale
               for j in range(degree + 1): # Incluye el punto inicial Xk[:,0]
                    # No negatividad para X, P, V (índices 0, 2, 4)
                    opti.subject_to(Xk[[0,2,4], j] >= -1e-12)
                    # Restricción S con holgura
                    opti.subject_to(Xk[1, j] >= -slack_s_k[j]) # S_scaled >= -slack_s
                    opti.subject_to(slack_s_k[j] >= 0)        # Slack_s >= 0
                    # Restricción O2 con holgura
                    opti.subject_to(Xk[3, j] >= -slack_o2_k[j]) # O2_scaled >= -slack_o2
                    opti.subject_to(slack_o2_k[j] >= 0)       # Slack_o2 >= 0

               # Restricción Vmax solo al final del intervalo
               opti.subject_to(Xk_end_current[4] <= Vmax_sc + 1e-6)

          # El estado final de la Fase 2 es Xk_end_prev del último intervalo
          X_end_feed_scaled = Xk_end_prev

          # --- Penalización Suave para Smax (aplicada en puntos finales de intervalo) ---
          penalty_smax_total = ca.MX(0.0)
          if all_params['w_Smax_penalty'] > 1e-9:
               Smax_scaled = all_params['Smax_constraint'] / S_scale
               interval_ends_S = []
               Xk_end_k = x_start_phase2_param
               _, D_coeff, _ = ca.collocation_coeff(ca.collocation_points(degree, collocation_method))
               for k in range(n_intervals):
                    Xk_end_k = 0
                    for r in range(degree + 1):
                         Xk_end_k += X_phase2_coll[k][:, r] * D_coeff[r]
                    interval_ends_S.append(Xk_end_k[1])
               violation_vector = ca.fmax(0, ca.vertcat(*interval_ends_S) - Smax_scaled)
               sum_sq_violation = ca.dot(violation_vector, violation_vector)
               penalty_smax_total = all_params['w_Smax_penalty'] * sum_sq_violation

          # --- Simulación Fase 3 (Batch Final) - Integración Simbólica ---
          st.info("[PHASE 3] Adding final integration (if applicable)...")
          slack_O2_f3 = opti.variable()
          slack_S_f3 = opti.variable() # Slack para S al final
          opti.subject_to(slack_O2_f3 >= 0)
          opti.subject_to(slack_S_f3 >= 0) # Slack S >= 0

          if T_post_feed_duration > 1e-6:
               x_sym_f3 = ca.MX.sym("x", 5); p_sym_f3 = ca.MX.sym("p", 2)
               ode_expr_f3 = ode_func_scaled(x_sym_f3, 0.0, p_sym_f3)
               dae_dict_f3 = {'x': x_sym_f3, 'p': p_sym_f3, 'ode': ode_expr_f3}
               integrator_opts_opti_f3 = {"t0": 0, "tf": T_post_feed_duration, "reltol": 1e-7, "abstol": 1e-9}
               integrator_phase3_scaled = ca.integrator("integrator_p3_scaled", "idas", dae_dict_f3, integrator_opts_opti_f3)
               res_phase3_sym_scaled = integrator_phase3_scaled(x0=X_end_feed_scaled, p=p_phase3_ode)
               X_final_total_scaled = res_phase3_sym_scaled['xf']
               # No negatividad para X, P, V al final
               opti.subject_to(X_final_total_scaled[[0,2,4]] >= -1e-12)
               # Restricciones con holgura para S y O2 al final
               opti.subject_to(X_final_total_scaled[1] >= -slack_S_f3)
               opti.subject_to(X_final_total_scaled[3] >= -slack_O2_f3)
          else:
               X_final_total_scaled = X_end_feed_scaled
               # Asegurar que la restricción de holgura se aplique incluso si no hay Fase 3
               opti.subject_to(X_final_total_scaled[1] >= -slack_S_f3)
               opti.subject_to(X_final_total_scaled[3] >= -slack_O2_f3)


          # --- Función Objetivo ---
          P_final_unsc = X_final_total_scaled[2] * P_scale
          V_final_unsc = X_final_total_scaled[4] * V_scale
          objective_PV = -(P_final_unsc * V_final_unsc)
          # *** NUEVO: Añadir penalización por holgura de S y O2 ***
          penalty_o2_slack = all_params['w_O2_slack'] * (ca.sumsqr(Slack_O2_phase2) + ca.sumsqr(slack_O2_f3))
          penalty_s_slack = all_params['w_S_slack'] * (ca.sumsqr(Slack_S_phase2) + ca.sumsqr(slack_S_f3))
          objective_total = objective_PV + penalty_smax_total + penalty_o2_slack + penalty_s_slack
          opti.minimize(objective_total)

          # --- Inicialización (Guess) ---
          st.info("Establishing initial guesses...")
          F_guess_val = (all_params['Fmax'] + all_params['Fmin']) / 2.0 * 0.8
          F_guess_scaled = F_guess_val / u_scale if u_scale > 1e-9 else 0.0
          opti.set_initial(F_phase2_scaled, F_guess_scaled)
          # Guess para slacks (idealmente 0)
          opti.set_initial(Slack_O2_phase2, 1e-9)
          opti.set_initial(slack_O2_f3, 1e-9)
          opti.set_initial(Slack_S_phase2, 1e-9) # Guess para slack S
          opti.set_initial(slack_S_f3, 1e-9)   # Guess para slack S final
          # Guess para estados
          for k in range(n_intervals):
               for j in range(degree + 1):
                    if not (k == 0 and j == 0):
                         opti.set_initial(X_phase2_coll[k][:, j], x_end_phase1_scaled)

          # --- Opciones del Solver (IPOPT) ---
          st.info("Configuring IPOPT solver...")
          p_opts = {"expand": True}
          s_opts = {
                    "max_iter": 5000, # <-- Aumentado
                    "print_level": 0, "sb": 'yes',
                    "tol": 1e-6, # <-- Tolerancia principal restaurada
                    "constr_viol_tol": 1e-6,
                    "acceptable_tol": 1e-4,
                    "acceptable_constr_viol_tol": 1e-4,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive"}
          opti.solver("ipopt", p_opts, s_opts)

          # --- Resolución del Problema ---
          try:
               st.info("🚀 Solving RTO with IPOPT (Collocation + Slacks O2+S)...") #Colocación
               solve_start_time = time.time()
               sol = opti.solve()
               solve_end_time = time.time()
               st.success(f"[OPTIMIZATION] ¡Solution found in {solve_end_time - solve_start_time:.2f} seconds!")

               # --- Extracción y Presentación de Resultados ---
               F_opt_phase2_scaled = sol.value(F_phase2_scaled)
               X_final_total_scaled_opt = sol.value(X_final_total_scaled)
               F_opt_phase2 = F_opt_phase2_scaled * u_scale; F_opt_phase2 = np.maximum(0.0, F_opt_phase2)
               X_final_total_opt = X_final_total_scaled_opt * np.array(x_scales).flatten();
               # Forzar no negatividad explícita en el resultado final
               X_final_total_opt = np.maximum(0.0, X_final_total_opt)

               P_final_opt = X_final_total_opt[2]; V_final_opt = X_final_total_opt[4]; O2_final_opt_mgL = X_final_total_opt[3] * 1000.0
               st.metric("Optimal Total Product (P*V)", f"{P_final_opt * V_final_opt:.4f} g")
               col1, col2, col3 = st.columns(3); col1.metric("Final Ethanol Conc.", f"{P_final_opt:.3f} g/L"); col2.metric("Final Volume", f"{V_final_opt:.3f} L"); col3.metric("Final O2", f"{O2_final_opt_mgL:.4f} mg/L")
               try: Smax_penalty_value=sol.value(penalty_smax_total); col1.metric("Smax Penalty", f"{Smax_penalty_value:.4g}")
               except: col1.metric("Smax Penalty", "N/A")
               # Mostrar valores de las penalizaciones de slack
               try:
                    o2_slack_penalty_value = sol.value(penalty_o2_slack)
                    s_slack_penalty_value = sol.value(penalty_s_slack) # Penalización slack S
                    o2_slack_f2_max = np.max(sol.value(Slack_O2_phase2)) if n_intervals > 0 else 0
                    s_slack_f2_max = np.max(sol.value(Slack_S_phase2)) if n_intervals > 0 else 0 # Max slack S
                    o2_slack_f3_val = sol.value(slack_O2_f3)
                    s_slack_f3_val = sol.value(slack_S_f3) # Slack S final
                    col2.metric("Slack Penalty (O2, S)", f"{o2_slack_penalty_value:.3g}, {s_slack_penalty_value:.3g}")
                    col3.metric("Max Slack (O2, S)", f"{o2_slack_f2_max:.2e}, {s_slack_f2_max:.2e}")
               except:
                    col2.metric("Slack Penalty", "N/A")


               st.write("Optimal Flow Profile (Phase 2):"); t_feed_points=np.linspace(t_fase1,t_fase2_end,n_intervals+1)
               if len(F_opt_phase2)==n_intervals:
                    df_flow=pd.DataFrame({'Start T (h)':t_feed_points[:-1],'End T (h)': t_feed_points[1:],'F Opt (L/h)': F_opt_phase2}); st.dataframe(df_flow.style.format({"F Opt (L/h)": "{:.4f}"}))
                    fig_flow,ax_flow=plt.subplots(figsize=(10,3)); ax_flow.step(t_feed_points[:-1],F_opt_phase2,where='post',label='$F_{opt}$');
                    if len(F_opt_phase2)>0: ax_flow.plot([t_feed_points[-2],t_feed_points[-1]],[F_opt_phase2[-1],F_opt_phase2[-1]],'-')
                    ax_flow.set_xlabel("T [h]"); ax_flow.set_ylabel("F [L/h]"); ax_flow.set_title("Optimal Flow Profile (Phase 2)"); ax_flow.set_xlim(t_fase1,t_fase2_end); f_min_plot=-0.05*all_params['Fmax'] if all_params['Fmax'] > 0 else -0.01; ax_flow.set_ylim(bottom=f_min_plot); ax_flow.grid(True,axis='y',ls=':'); ax_flow.legend(); st.pyplot(fig_flow)
               else: st.warning(f"Length F_opt ({len(F_opt_phase2)}) != n_intervals ({n_intervals}).")

          except RuntimeError as e:
               st.error(f"[ERROR] IPOPT Solver failed: {e}");
               try:
                    st.warning("Debug values (can be from the last failed attempt):")
                    st.write(f"Total Objetive: {opti.debug.value(objective_total):.4g}")
                    try:
                         f_vals = opti.debug.value(F_phase2_scaled)
                         # Convertir a lista de floats para imprimir
                         f_vals_list = [float(fv) for fv in np.array(f_vals).flatten()]
                         st.write(f"  Scaled Flows (Fk_scaled): {[f'{v:.4g}' for v in f_vals_list]}")
                    except Exception as e_f:
                         st.write(f"  Scaled Flows (Fk_scaled): Error getting value - {e_f}")
                    st.write(f"  Scaled Total Final State (X_final_total_scaled): {[f'{v:.4g}' for v in opti.debug.value(X_final_total_scaled)]}")
                    # Mostrar valores de los slacks
                    try:
                         st.write(f"  Slack O2 Phase 2 (max): {np.max(opti.debug.value(Slack_O2_phase2)):.4g}")
                         st.write(f"  Slack O2 Phase 3: {opti.debug.value(slack_O2_f3):.4g}")
                         st.write(f"  Slack S Phase 2 (max): {np.max(opti.debug.value(Slack_S_phase2)):.4g}")
                         st.write(f"  Slack S Phase 3: {opti.debug.value(slack_S_f3):.4g}")
                    except Exception as e_s:
                         st.write(f"  Slacks: Error getting value - {e_s}")

               except Exception as de: st.error(f"Error getting debug values: {de}")
               st.stop()
          except Exception as e: st.error(f"Error Opt/Results: {e}"); st.error(traceback.format_exc()); st.stop()

          # --- Reconstrucción de la Trayectoria Completa y Gráficas ---
          # (Código idéntico a la versión anterior, se colapsa por brevedad)
          st.info("Reconstructing the complete trajectory with the optimal profile...")
          F_opt_phase2_scaled_again = F_opt_phase2 / u_scale if u_scale > 1e-9 else np.zeros_like(F_opt_phase2)
          t_plot_full = []; x_plot_full_scaled = []; f_plot_full_used = []
          plot_ok = True; start_time_sim = time.time()
          # Fase 1
          N_plot_p1 = max(20, int(t_fase1 * 10)); t_eval_p1 = np.linspace(0, t_fase1, N_plot_p1); dt_p1 = t_eval_p1[1] - t_eval_p1[0] if N_plot_p1 > 1 else t_fase1
               
          try:
               x_sym_f1_plot = ca.MX.sym("x", 5); p_sym_f1_plot = ca.MX.sym("p", 2)
               ode_expr_f1_plot = ode_func_scaled(x_sym_f1_plot, 0.0, p_sym_f1_plot)
               dae_dict_f1_plot = {'x': x_sym_f1_plot, 'p': p_sym_f1_plot, 'ode': ode_expr_f1_plot}
               integrator_p1_plot = ca.integrator("int_p1_plot", "idas", dae_dict_f1_plot, {"t0": 0, "tf": dt_p1, "reltol": 1e-7, "abstol": 1e-9})
          except Exception as e: st.error(f"Error creating integrator for plot Phase 1: {e}"); st.stop()
          xk_plot_scaled = x0_np_scaled.copy(); t_plot_full.append(0.0); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
          for i in range(N_plot_p1 - 1):
               try:
                    res_plot_scaled = integrator_p1_plot(x0=xk_plot_scaled, p=p_phase1_ode)
                    xk_plot_scaled = np.array(res_plot_scaled['xf']).flatten()
                    if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F1 plot")
                    xk_plot_scaled = np.maximum(xk_plot_scaled, 0.0)
                    t_plot_full.append(t_eval_p1[i + 1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
               except Exception as plot_e: st.error(f"Failure plot F1: {plot_e}"); plot_ok = False; break
          if not plot_ok: st.stop()
          # Fase 2
          N_plot_p2_per_interval = max(2, int(1.0 / h_interval * 2) if h_interval > 1e-6 else 5); N_plot_p2 = n_intervals * N_plot_p2_per_interval
          t_eval_p2 = np.linspace(t_fase1, t_fase2_end, N_plot_p2 + 1); dt_p2_fine = (t_fase2_end - t_fase1) / N_plot_p2 if N_plot_p2 > 0 else 0
          try:
               x_sym_f2_plot = ca.MX.sym("x", 5); u_sym_f2_plot = ca.MX.sym("u"); p_sym_f2_plot = ca.MX.sym("p", 2)
               ode_expr_f2_plot = ode_func_scaled(x_sym_f2_plot, u_sym_f2_plot, p_sym_f2_plot)
               dae_dict_f2_plot = {'x': x_sym_f2_plot, 'p': p_sym_f2_plot, 'u': u_sym_f2_plot, 'ode': ode_expr_f2_plot}
               integrator_p2_plot = ca.integrator("int_p2_plot", "idas", dae_dict_f2_plot, {"t0": 0, "tf": dt_p2_fine, "reltol": 1e-7, "abstol": 1e-9})
          except Exception as e: st.error(f"Error creating integrator for plot Phase 2: {e}"); st.stop()
          for i in range(N_plot_p2):
               t_now = t_eval_p2[i]; k_interval = int((t_now - t_fase1) / h_interval + 1e-9) if h_interval > 1e-9 else 0; k_interval = max(0, min(k_interval, n_intervals - 1))
               F_now_scaled = F_opt_phase2_scaled[k_interval]; F_now_unscaled = F_opt_phase2[k_interval]
               V_current_unscaled = xk_plot_scaled[4] * V_scale;
               if V_current_unscaled >= all_params['Vmax'] - 1e-6: F_now_scaled = 0.0; F_now_unscaled = 0.0
               try:
                    res_plot_scaled = integrator_p2_plot(x0=xk_plot_scaled, p=p_phase2_ode, u=F_now_scaled)
                    xk_plot_scaled = np.array(res_plot_scaled['xf']).flatten()
                    if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F2 plot")
                    xk_plot_scaled = np.maximum(xk_plot_scaled, 0.0)
                    t_plot_full.append(t_eval_p2[i + 1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(F_now_unscaled)
               except Exception as plot_e: st.error(f"Failure plot F2: {plot_e}"); plot_ok = False; break
          if not plot_ok: st.stop()
          # Fase 3
          if T_post_feed_duration > 1e-6:
               N_plot_p3 = max(20, int(T_post_feed_duration * 10)); t_eval_p3 = np.linspace(t_fase2_end, t_fase3_end, N_plot_p3 + 1); dt_p3_fine = T_post_feed_duration / N_plot_p3 if N_plot_p3 > 0 else 0
               try:
                    x_sym_f3_plot = ca.MX.sym("x", 5); p_sym_f3_plot = ca.MX.sym("p", 2)
                    ode_expr_f3_plot = ode_func_scaled(x_sym_f3_plot, 0.0, p_sym_f3_plot)
                    dae_dict_f3_plot = {'x': x_sym_f3_plot, 'p': p_sym_f3_plot, 'ode': ode_expr_f3_plot}
                    integrator_p3_plot = ca.integrator("int_p3_plot", "idas", dae_dict_f3_plot, {"t0": 0, "tf": dt_p3_fine, "reltol": 1e-7, "abstol": 1e-9})
               except Exception as e: st.error(f"Error creating integrator for plot Phase 3: {e}"); st.stop()
               for i in range(N_plot_p3):
                    try:
                         res_plot_scaled = integrator_p3_plot(x0=xk_plot_scaled, p=p_phase3_ode)
                         xk_plot_scaled = np.array(res_plot_scaled['xf']).flatten()
                         if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F3 plot")
                         xk_plot_scaled = np.maximum(xk_plot_scaled, 0.0)
                         t_plot_full.append(t_eval_p3[i + 1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
                    except Exception as plot_e: st.error(f"Failure plot F3: {plot_e}"); plot_ok = False; break
               if not plot_ok: st.stop()
          else:
               if f_plot_full_used: f_plot_full_used[-1] = 0.0
          end_time_sim = time.time(); st.success(f"Detailed Simulation Completed ({end_time_sim - start_time_sim:.2f} s).")

          # --- Procesamiento para Gráficas ---
          t_sim = np.array(t_plot_full); x_sim_scaled = np.array(x_plot_full_scaled); f_sim = np.array(f_plot_full_used)
          x_sim = x_sim_scaled * np.array(x_scales).flatten()[np.newaxis, :]
          X_sim, S_sim, P_sim, O2_sim, V_sim = [x_sim[:, i] for i in range(nx)]; O2_sim_mgL = O2_sim * 1000.0

          # --- Cálculo de Tasas y Gráficas Detalladas ---
          if plot_ok:
               st.info("📊 Calculating rates and generating detailed graphs...")
               # (Código de cálculo de tasas y ploteo idéntico a la versión anterior)
               mu_sim_calc = []; qP_sim_calc = []; qS_sim_calc = []; qO_sim_calc = []
               OTR_sim_calc = []; OUR_sim_calc = []; inhib_factor_sim_calc = []
               ca_params = all_params.copy()
               for k, v in ca_params.items():
                    if isinstance(v, list): ca_params[k] = ca.DM(v)
               sym_S = ca.SX.sym('S'); sym_P = ca.SX.sym('P'); sym_O2 = ca.SX.sym('O2')
               sym_X = ca.SX.sym('X'); sym_Kla = ca.SX.sym('Kla'); sym_consider_O2 = ca.SX.sym('consider_O2_flag')
               mu_eval = 0.0; tipo = ca_params['tipo_mu']
               if tipo == "Simple Monod": mu_eval = mu_monod_ca(sym_S, ca_params['mumax'], ca_params['Ks'])
               elif tipo == "Sigmoidal Monod": mu_eval = mu_sigmoidal_ca(sym_S, ca_params['mumax'], ca_params['Ks'], ca_params['n_sig'])
               elif tipo == "Monod with Restrictions": mu_eval = mu_completa_ca(sym_S, sym_O2, sym_P, ca_params['mumax'], ca_params['Ks'], ca_params['KO'], ca_params['KP_gen'])
               elif tipo == "Fermentation": mu_eval = mu_fermentacion_ca(sym_S, sym_P, sym_O2, ca_params['mumax_aerob'], ca_params['Ks_aerob'], ca_params['KO_aerob'], ca_params['mumax_anaerob'], ca_params['Ks_anaerob'], ca_params['KiS_anaerob'], ca_params['KP_anaerob'], ca_params.get('n_p', 1.0), ca_params['KO_inhib_anaerob'], considerar_O2=None)
               elif tipo == "Switched Fermentation": mu_eval = mu_fermentacion_ca(sym_S, sym_P, sym_O2, ca_params['mumax_aerob'], ca_params['Ks_aerob'], ca_params['KO_aerob'], ca_params['mumax_anaerob'], ca_params['Ks_anaerob'], ca_params['KiS_anaerob'], ca_params['KP_anaerob'], ca_params.get('n_p', 1.0), ca_params['KO_inhib_anaerob'], considerar_O2=sym_consider_O2)
               inputs_mu = [sym_S, sym_P, sym_O2]; inputs_qP = [sym_S, sym_P, sym_O2]; inputs_qS = [sym_S, sym_P, sym_O2]
               if tipo == "Switched Fermentation": inputs_mu.append(sym_consider_O2); inputs_qP.append(sym_consider_O2); inputs_qS.append(sym_consider_O2)
               f_mu = ca.Function('f_mu', inputs_mu, [mu_eval])
               qP_base_eval = ca_params['alpha_lp'] * ca.fmax(0.0, mu_eval) + ca_params['beta_lp']; safe_O2_inhib_eval = ca.fmax(1e-9, sym_O2); inhib_fact_eval = ca.fmax(ca_params['KO_inhib_prod'], 1e-9) / ca.fmax(ca_params['KO_inhib_prod'] + safe_O2_inhib_eval, 1e-9); qP_eval = qP_base_eval * inhib_fact_eval; qP_eval = ca.fmax(0.0, qP_eval)
               f_qP = ca.Function('f_qP', inputs_qP, [qP_eval, inhib_fact_eval])
               cSX_eval = (ca.fmax(0.0, mu_eval) / ca.fmax(ca_params['Yxs'], 1e-9)); cSP_eval = (qP_eval / ca.fmax(ca_params['Yps'], 1e-9)); cSm_eval = ca_params['ms']; qS_eval = cSX_eval + cSP_eval + cSm_eval; qS_eval = ca.fmax(0.0, qS_eval);
               f_qS = ca.Function('f_qS', inputs_qS, [qS_eval])
               safe_O2_qO_eval = ca.fmax(1e-9, sym_O2); mu_aer_only_eval = ca_params['mumax_aerob'] * (sym_S / ca.fmax(ca_params['Ks_aerob'] + sym_S, 1e-9)) * (safe_O2_qO_eval / ca.fmax(ca_params['KO_aerob'] + safe_O2_qO_eval, 1e-9)); mu_aer_only_eval = ca.fmax(0.0, mu_aer_only_eval); cOXa_eval = (mu_aer_only_eval / ca.fmax(ca_params['Yxo'], 1e-9)); cOm_eval = ca_params['mo']; qO_eval = cOXa_eval + cOm_eval; qO_eval = ca.fmax(0.0, qO_eval);
               f_qO = ca.Function('f_qO', [sym_S, sym_O2], [qO_eval])
               OTR_eval = sym_Kla * (ca_params['Cs'] - ca.fmax(0.0, sym_O2)); OUR_eval = qO_eval * ca.fmax(1e-9, sym_X);
               f_OTR_OUR = ca.Function('f_OTR_OUR', [sym_S, sym_O2, sym_X, sym_Kla], [OTR_eval, OUR_eval])
               for i in range(len(t_sim)):
                    ti, Xi, Si, Pi, O2i, Vi = t_sim[i], X_sim[i], S_sim[i], P_sim[i], O2_sim[i], V_sim[i]; safe_Xi = max(1e-9, Xi)
                    fase = 1; kla_now = ca_params['Kla1']; flag_o2 = 1.0
                    if ti >= t_fase1 - 1e-6: fase = 2; kla_now = ca_params['Kla2']; flag_o2 = 0.0
                    if ti >= t_fase2_end - 1e-6: fase = 3; kla_now = ca_params['Kla2']; flag_o2 = 0.0
                    mu_args = [Si, Pi, O2i]; qP_args = [Si, Pi, O2i]; qS_args = [Si, Pi, O2i]
                    if ca_params['tipo_mu'] == "Switched Fermentation": mu_args.append(flag_o2); qP_args.append(flag_o2); qS_args.append(flag_o2)
                    mu_i = float(f_mu(*mu_args)[0]); qP_i, inhib_factor_i = f_qP(*qP_args); qP_i=float(qP_i); inhib_factor_i=float(inhib_factor_i); qS_i = float(f_qS(*qS_args)[0]); qO_i = float(f_qO(Si, O2i)[0])
                    OTR_i, OUR_i = f_OTR_OUR(Si, O2i, Xi, kla_now); OTR_i = float(OTR_i); OUR_i = float(OUR_i)
                    mu_sim_calc.append(mu_i); qP_sim_calc.append(qP_i); qS_sim_calc.append(qS_i); qO_sim_calc.append(qO_i); inhib_factor_sim_calc.append(inhib_factor_i); OTR_sim_calc.append(OTR_i * 1000.0); OUR_sim_calc.append(OUR_i * 1000.0)
               mu_sim=np.array(mu_sim_calc); qP_sim=np.array(qP_sim_calc); qS_sim=np.array(qS_sim_calc); qO_sim=np.array(qO_sim_calc)*1000; OTR_sim=np.array(OTR_sim_calc); OUR_sim=np.array(OUR_sim_calc); inhib_factor_sim=np.array(inhib_factor_sim_calc)

               # --- Configurações Globais de Fonte (Times New Roman, Tamanho 22) ---
               NEW_FONT_SIZE = 22
               plt.rcParams.update({
               'font.size': NEW_FONT_SIZE,
               'font.family': 'serif',
               'font.serif': ['Times New Roman'],
               'axes.titlesize': NEW_FONT_SIZE,
               'axes.labelsize': NEW_FONT_SIZE,
               'xtick.labelsize': NEW_FONT_SIZE * 0.9,
               'ytick.labelsize': NEW_FONT_SIZE * 0.9,
               'legend.fontsize': NEW_FONT_SIZE * 0.9 # Mantido caso precise no futuro
               })

               # --- Supondo que as variáveis e funções JÁ EXISTEM ---
               # t_sim, f_sim, V_sim, X_sim, S_sim, P_sim, O2_sim_mgL, inhib_factor_sim, mu_sim,
               # qS_sim, qP_sim, qO_sim, OTR_sim, OUR_sim, Yps_inst, Yxs_inst
               # t_fase1, t_fase2_end, t_fase3_end
               # all_params (dicionário), cond_iniciales (dicionário)

               # --- Graficação RTO (PT-BR, Layout 3x3, Fonte 22, Sem Legenda) ---
               st.subheader("📊 Detailes RTO graphs") # Traduzido

               # Ajuste o figsize para o layout 3x3 (largura, altura) - pode precisar ajustar
               fig = plt.figure(figsize=(15, 10)) # Ex: 18x18 para 3x3 com fontes grandes

               # --- Função auxiliar para adicionar linhas de fase ---
               # (Mantendo a lógica original do usuário para cálculo de y_lim, se necessário)
               def add_phase_lines(ax, t1, t2):
                    current_ymin, current_ymax = ax.get_ylim()
                    # Lógica complexa para ajustar y_lim se não for finito ou muito pequeno
                    if not np.isfinite(current_ymin) or not np.isfinite(current_ymax) or current_ymax <= current_ymin + 1e-6:
                         y_data=[]
                         [y_data.extend(line.get_ydata()) for line in ax.get_lines()]
                         valid_data = [y for y in y_data if np.isfinite(y)]
                         if valid_data:
                              data_range = abs(np.max(valid_data)-np.min(valid_data))
                              if data_range > 1e-6:
                                   current_ymin = np.min(valid_data) - 0.1 * data_range
                                   current_ymax = np.max(valid_data) + 0.1 * data_range
                              else:
                                   current_ymin = np.min(valid_data) - 0.1
                                   current_ymax = np.max(valid_data) + 0.1
                         else: # Sem dados válidos
                              current_ymin, current_ymax = (0, 1)

                         # Garante que o range não seja zero ou negativo
                         current_ymax = current_ymin + 1.0 if current_ymax <= current_ymin + 1e-6 else current_ymax
                         ax.set_ylim(current_ymin, current_ymax) # Aplica o ylim recalculado

                    # Desenha as linhas verticais usando os limites y calculados/atuais
                    ax.vlines([t1, t2], current_ymin, current_ymax, colors=['orange', 'purple'], ls='--', lw=1.5, zorder=10)
                    ax.set_ylim(current_ymin, current_ymax) # Reaplicar ylim pode ser necessário após vlines

               # --- Gráficos no Layout 3x3 ---

               # 1. Vazão e Volume (Posição 1: Linha 1, Coluna 1)
               ax1 = plt.subplot(2, 3, 1)
               color = 'tab:red'; ax1.step(t_sim, f_sim, where='post', color=color) # label removido
               ax1.plot([t_sim[-1]], [f_sim[-1]], marker='o', ls='', color=color)
               ax1.set_ylabel('Flow [L/h]', color=color) # Traduzido
               ax1.tick_params(axis='y', labelcolor=color)
               ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t_sim, V_sim, color=color, ls='-') # label removido
               ax1b.set_ylabel('Volume [L]', color=color) # Traduzido
               ax1b.tick_params(axis='y', labelcolor=color)
               ax1.set_xlabel('Time [h]') # Traduzido
               ax1.grid(True)
               ax1.set_title('Optimized Feeding and Volume') # Traduzido
               add_phase_lines(ax1, t_fase1, t_fase2_end) # Chamada mantida
               # Linhas de criação de legenda manual (leg_el_phases) e ax1b.legend() removidas
               f_min_plot = -0.05 * all_params.get('Fmax', 0) # Usa .get para segurança
               ax1.set_ylim(bottom=f_min_plot)

               # 2. Biomassa (X) (Posição 2: Linha 1, Coluna 2)
               ax2 = plt.subplot(2, 3, 2)
               ax2.plot(t_sim, X_sim, 'g-')
               ax2.set_title('Biomass (X)') # Traduzido
               ax2.set_ylabel('[g/L]')
               ax2.set_xlabel('Time [h]') # Traduzido
               ax2.grid(True); ax2.set_ylim(bottom=0)
               add_phase_lines(ax2, t_fase1, t_fase2_end)

               # 3. Substrato (S) (Posição 3: Linha 1, Coluna 3)
               ax3 = plt.subplot(2, 3, 3)
               ax3.plot(t_sim, S_sim, 'm-')
               ax3.set_title('Substrate (S)') # Traduzido
               ax3.set_ylabel('[g/L]')
               ax3.set_xlabel('Time [h]') # Traduzido
               ax3.grid(True); ax3.set_ylim(bottom=0)
               # Linha de restrição (sem label)
               smax_val = all_params.get('Smax_constraint', np.nan)
               if not np.isnan(smax_val):
                    ax3.axhline(smax_val, color='red', ls=':', lw=1.5) # label removido
               # ax3.legend() removido
               add_phase_lines(ax3, t_fase1, t_fase2_end)

               # 4. Etanol (P) (Posição 4: Linha 2, Coluna 1)
               ax4 = plt.subplot(2, 3, 4)
               ax4.plot(t_sim, P_sim, 'k-')
               ax4.set_title('Ethanol (P)') # Título já estava ok
               ax4.set_ylabel('[g/L]')
               ax4.set_xlabel('Time [h]') # Traduzido
               ax4.grid(True); ax4.set_ylim(bottom=0)
               # Linha de restrição (sem label)
               pmax_val = all_params.get('Pmax_constraint', np.nan)
               if not np.isnan(pmax_val):
                    ax4.axhline(pmax_val, color='red', ls=':', lw=1.5) # label removido
               # ax4.legend() removido
               add_phase_lines(ax4, t_fase1, t_fase2_end)

               # 5. O2 e Fator Inibição P (Posição 5: Linha 2, Coluna 2)
               ax5 = plt.subplot(2, 3, 5)
               ax5.plot(t_sim, O2_sim_mgL, 'c-') # label removido
               ax5.set_title(r'$O_2$ and Inhibition Factor $P$') # Traduzido (usando LaTeX)
               ax5.set_ylabel(r'$O_2$ [mg/L]', color='c') # Usando LaTeX
               ax5.set_xlabel('Time [h]') # Traduzido
               ax5.grid(True)
               o2_top = max(all_params.get('Cs',0)*1000*1.1, np.max(O2_sim_mgL)*1.1 if len(O2_sim_mgL[np.isfinite(O2_sim_mgL)])>0 else all_params.get('Cs',0)*1000*1.1)
               o2_top = max(o2_top, 0.1)
               ax5.set_ylim(bottom=0.0, top=o2_top)
               ax5.tick_params(axis='y', labelcolor='c')
               ax5b = ax5.twinx()
               ax5b.plot(t_sim, inhib_factor_sim, color='darkorange', ls='--') # label removido
               ax5b.set_ylabel('Prod. Inhib. Factor[-]', color='darkorange') # Traduzido
               ax5b.tick_params(axis='y', labelcolor='darkorange')
               ax5b.set_ylim(bottom=-0.05, top=1.05)
               # ax5b.legend() removido
               add_phase_lines(ax5, t_fase1, t_fase2_end)

               # 6. Taxa Específica de Crescimento (mu) (Posição 6: Linha 2, Coluna 3)
               ax6 = plt.subplot(2, 3, 6)
               ax6.plot(t_sim, mu_sim, 'y-')
               ax6.set_title(r'Specific Growth Rate ($\mu$)') # Traduzido e LaTeX
               ax6.set_ylabel('[1/h]')
               ax6.set_xlabel('Tempo [h]') # Traduzido
               ax6.grid(True); ax6.set_ylim(bottom=0)
               add_phase_lines(ax6, t_fase1, t_fase2_end)


               # Ajusta o layout
               plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 1]) # Mantido rect original se necessário
               st.pyplot(fig)

               # --- Métricas Finais RTO (Tradução PT-BR) ---
               st.subheader("📈 Final Metrics (RTO Simulation)") # Traduzido

               # Código de cálculo de métricas idêntico ao original, mas com labels traduzidos
               # Adicionando checagens de segurança para arrays vazios antes de indexar [-1]
               col_m1, col_m2, col_m3 = st.columns(3)

               vol_final_sim = V_sim[-1] if len(V_sim) > 0 else cond_iniciales.get('V0', 0)
               et_final_conc = P_sim[-1] if len(P_sim) > 0 else cond_iniciales.get('P0', 0)
               bio_final_conc = X_sim[-1] if len(X_sim) > 0 else cond_iniciales.get('X0', 0)
               S_final = S_sim[-1] if len(S_sim) > 0 else cond_iniciales.get('S0', 0)

               S_ini_tot = cond_iniciales.get('S0', 0) * cond_iniciales.get('V0', 0)
               # Usa .get para Sin e checa se t_sim tem mais de 1 ponto para trapz
               S_alim_tot = np.trapz(f_sim * all_params.get('Sin', 0), t_sim) if len(t_sim) > 1 and len(f_sim) == len(t_sim) else 0
               S_fin_tot = S_final * vol_final_sim
               S_cons_tot = max(1e-9, S_ini_tot + S_alim_tot - S_fin_tot)
               P_ini_tot = cond_iniciales.get('P0', 0) * cond_iniciales.get('V0', 0)
               et_fin_tot = et_final_conc * vol_final_sim
               et_prod_tot = max(0, et_fin_tot - P_ini_tot)

               col_m1.metric("Final Volume [L]", f"{vol_final_sim:.3f}") # Traduzido
               col_m2.metric("Final Ethanol [g/L]", f"{et_final_conc:.3f}") # Traduzido
               col_m3.metric("Final Biomass [g/L]", f"{bio_final_conc:.3f}") # Traduzido

               # Usa t_fase3_end se definido e maior que 0, senão usa t_sim[-1]
               tempo_final_prod = t_fase3_end if 't_fase3_end' in locals() and t_fase3_end > 0 else (t_sim[-1] if len(t_sim)>0 else 0)

               prod_vol_et = et_prod_tot / vol_final_sim / tempo_final_prod if tempo_final_prod > 0 and vol_final_sim > 1e-6 else 0
               col_m1.metric("Ethanol Volume Productivity [g/L/h]", f"{prod_vol_et:.4f}") # Traduzido

               rend_glob_et = et_prod_tot / S_cons_tot if S_cons_tot > 1e-9 else 0 # Evita divisão por zero
               col_m2.metric("Global Yield P/S [g/g]", f"{rend_glob_et:.4f}") # Traduzido

               try:
                    # Checa se t_sim e X_sim/V_sim têm pontos suficientes para integrar
                    if len(t_sim) > 1 and len(X_sim) == len(t_sim) and len(V_sim) == len(t_sim):
                         X_V_int = np.trapz(X_sim * V_sim, t_sim)
                    else: X_V_int = 0
                    prod_esp_media = et_prod_tot / X_V_int if X_V_int > 1e-9 and tempo_final_prod > 0 else 0
                    col_m3.metric("Avg. Specific Production [gP/gX/h]", f"{prod_esp_media:.5f}" if prod_esp_media > 0 else "N/A") # Traduzido
               except Exception as me:
                    print(f"Avg. Specific Prod. Calculation Error: {me}")
                    col_m3.metric("Avg. Specific Production [gP/gX/h]", "Erro") # Traduzido

               try:
                    if len(P_sim) > 0: # Checa se P_sim não está vazio
                         p_max_idx = np.argmax(P_sim)
                         # Checa se t_sim correspondente existe
                         tempo_p_max = t_sim[p_max_idx] if p_max_idx < len(t_sim) else np.nan
                         col_m1.metric("Max Ethanol [g/L]", f"{P_sim[p_max_idx]:.3f} (t={tempo_p_max:.1f} h)") # Traduzido
                    else:
                         col_m1.metric("Max Ethanol [g/L]", "N/A") # Traduzido
               except (ValueError, IndexError) as e:
                    print(f"Max Ethanol Calculation Error: {e}")
                    col_m1.metric("Max Ethanol [g/L]", "N/A") # Traduzido

               col_m2.metric("Residual Substrate [g/L]", f"{S_final:.3f}") # Traduzido

               data_simulation = {
                    'Time [h]': t_sim,
                    'Flow [L/h]': f_sim,
                    'Volume [L]': V_sim,
                    'Biomass (X) [g/L]': X_sim,
                    'Substrate (S) [g/L]': S_sim,
                    'Ethanol (P) [g/L]': P_sim,
                    'Dissolved Oxygen (O2) [g/L]': O2_sim_mgL,
                    'Specific Growth Rate (mu) [1/h]': mu_sim
               }

               df_data_simulation = pd.DataFrame(data_simulation)

               buffer = BytesIO()
               with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_data_simulation.to_excel(writer, index=False, sheet_name='Dados')

                    # Inserir o gráfico na planilha
                    workbook = writer.book
                    worksheet = writer.sheets['Dados']
                    img_stream = BytesIO()
                    fig.savefig(img_stream, format='png')
                    img_stream.seek(0)
                    worksheet.insert_image('E2', 'grafico.png', {'image_data': img_stream})

               buffer.seek(0)

               st.download_button(
                    label="Download Simulation Data as Excel",
                    data=buffer,
                    file_name="dados_rto.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
               )

          else: # Si plot_ok es False
            st.error("Detailed plots could not be generated due to a simulation error.")


# --- Punto de Entrada Principal ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="RTO Detailed Fermentation")
    try:
        # Ejecutar la función principal de la página Streamlit
        rto_fermentation_page()
    except Exception as main_e:
        # Capturar errores inesperados a nivel global
        st.error(f"Unexpected application error: {main_e}")
        st.error(traceback.format_exc())

