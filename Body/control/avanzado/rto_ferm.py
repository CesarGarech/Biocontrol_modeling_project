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

# ... (Definiciones cinéticas sin cambios) ...
def mu_monod_ca(S, mumax, Ks):
    safe_S = ca.fmax(0.0, S); safe_den = ca.fmax(Ks + S, 1e-9)
    mu = mumax * safe_S / safe_den; return ca.fmax(mu, 0.0)
def mu_sigmoidal_ca(S, mumax, Ks, n):
    safe_S = ca.fmax(0.0, S); safe_S_n = safe_S**n; safe_Ks_n = ca.fmax(Ks**n, 1e-9)
    safe_den = ca.fmax(safe_Ks_n + safe_S_n, 1e-9); mu = mumax * safe_S_n / safe_den
    return ca.fmax(mu, 0.0)
def mu_completa_ca(S, O2, P, mumax, Ks, KO, KP_gen):
    safe_S = ca.fmax(0.0, S); safe_O2 = ca.fmax(0.0, O2); safe_P = ca.fmax(0.0, P)
    term_S = safe_S / ca.fmax(Ks + safe_S, 1e-9); term_O2 = safe_O2 / ca.fmax(KO + safe_O2, 1e-9)
    term_P = ca.fmax(KP_gen / ca.fmax(KP_gen + safe_P, 1e-9), 0.0)
    mu = mumax * term_S * term_O2 * term_P; return ca.fmax(mu, 0.0)
def mu_fermentacion_ca(S, P, O2, mumax_aerob, Ks_aerob, KO_aerob, mumax_anaerob, Ks_anaerob, KiS_anaerob, KP_anaerob, n_p, KO_inhib_anaerob, considerar_O2=None):
    safe_S = ca.fmax(0.0, S); safe_P = ca.fmax(0.0, P); safe_O2 = ca.fmax(1e-9, O2)
    mu_aer = mumax_aerob * (safe_S / ca.fmax(Ks_aerob + safe_S, 1e-9)) * (safe_O2 / ca.fmax(KO_aerob + safe_O2, 1e-9))
    mu_aer = ca.fmax(0.0, mu_aer)
    large_kis = 1e9
    den_S_an = Ks_anaerob + safe_S + ca.if_else(KiS_anaerob < large_kis, safe_S**2 / ca.fmax(KiS_anaerob, 1e-9), 0.0)
    safe_den_S_an = ca.fmax(den_S_an, 1e-9); term_S_an = safe_S / safe_den_S_an
    safe_KP_an = ca.fmax(KP_anaerob, 1e-9); term_P_base = 1.0 - (safe_P / safe_KP_an)
    safe_term_P_base = ca.fmax(0.0, term_P_base); safe_n_p = ca.fmax(n_p, 1e-6)
    term_P_an = safe_term_P_base**safe_n_p
    safe_KO_inhib_an = ca.fmax(KO_inhib_anaerob, 1e-9)
    term_O2_inhib_an = safe_KO_inhib_an / ca.fmax(safe_KO_inhib_an + safe_O2, 1e-9)
    mu_anaer = mumax_anaerob * term_S_an * term_P_an * term_O2_inhib_an
    mu_anaer = ca.fmax(0.0, mu_anaer)
    if isinstance(considerar_O2, (ca.MX, ca.SX)): mu = ca.if_else(considerar_O2 > 0.5, mu_aer, mu_anaer)
    elif considerar_O2 is None: mu = mu_aer + mu_anaer
    elif considerar_O2: mu = mu_aer
    else: mu = mu_anaer
    return ca.fmax(mu, 0.0)

# ====================================================
# --- Página de Streamlit ---
# ====================================================
def rto_fermentation_page():
     st.header("🧠 Control RTO - Fermentación Alcohólica")
     st.markdown("""
     Optimización del perfil de alimentación ($F(t)$) para maximizar $P_{final} V_{final}$.
     **Nota:** Aumentado kLa2 por defecto para intentar resolver inviabilidad.
     """)

     with st.sidebar:
          # --- Configuración Sidebar ---
          st.subheader("1. Modelo Cinético y Parámetros")
          tipo_mu = st.selectbox("Modelo Cinético (μ)", ["Fermentación Conmutada", "Fermentación", "Monod simple", "Monod sigmoidal", "Monod con restricciones"], index=1)
          Ks_base = st.number_input("Ks base (default) [g/L]", 0.01, 10.0, 1.0, 0.1, key="ks_base")
          params_cineticos = {}
          params_cineticos['tipo_mu'] = tipo_mu
          # ... (Definiciones de parámetros para todos los modelos, como antes) ...
          if tipo_mu == "Monod simple":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_simple")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_simple")
               params_cineticos.update({k: 1e9 for k in ['KO', 'KP_gen', 'n_sig', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Monod sigmoidal":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_sig")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_sig")
               params_cineticos['n_sig'] = st.slider("Exponente sigmoidal (n)", 1.0, 5.0, 2.0, 0.1, key="n_sig")
               params_cineticos.update({k: 1e9 for k in ['KO', 'KP_gen', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Monod con restricciones":
               params_cineticos['mumax'] = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_restr")
               params_cineticos['Ks'] = st.slider("Ks [g/L]", 0.01, 10.0, Ks_base, 0.1, key="ks_restr")
               params_cineticos['KO'] = st.slider("KO (O2 - restricción) [g/L]", 0.0001, 0.05, 0.002, 0.0001, format="%.4f", key="ko_restr")
               params_cineticos['KP_gen'] = st.slider("KP (Inhib. Producto genérico) [g/L]", 1.0, 100.0, 50.0, 1.0, key="kp_gen")
               params_cineticos.update({k: 1e9 for k in ['n_sig', 'mumax_aerob', 'Ks_aerob', 'KO_aerob', 'mumax_anaerob', 'Ks_anaerob', 'KiS_anaerob', 'KP_anaerob', 'n_p', 'KO_inhib_anaerob']})
          elif tipo_mu == "Fermentación Conmutada":
               st.info("Modelo Conmutado: Fase 1 usa cinética aerobia, Fases 2 y 3 usan cinética anaerobia.")
               params_cineticos['mumax_aerob'] = st.slider("μmax (Aerobio) [1/h]", 0.1, 1.0, 0.45, 0.05, key="mumax_aero_c")
               params_cineticos['Ks_aerob'] = st.slider("Ks (Aerobio) [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aero_c")
               params_cineticos['KO_aerob'] = st.slider("KO (Afinidad O2 - Aerobio) [g/L]", 0.0001, 0.05, 0.002, 0.0001, format="%.4f", key="ko_aero_c")
               params_cineticos['mumax_anaerob'] = st.slider("μmax (Anaerobio) [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaero_c")
               params_cineticos['Ks_anaerob'] = st.slider("Ks (Anaerobio) [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaero_c")
               params_cineticos['KiS_anaerob'] = st.slider("KiS (Inhib. Sustrato - Anaerobio) [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaero_c")
               params_cineticos['KP_anaerob'] = st.slider("KP (Inhib. Etanol - Anaerobio) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaero_c")
               params_cineticos['n_p'] = st.slider("Exponente Inhib. Etanol (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_anaero_c")
               params_cineticos['KO_inhib_anaerob'] = st.slider("KO_inhib (Inhib. O2 en μ Anaerobio) [g/L]", 1e-6, 0.01, 0.0005, 1e-6, format="%.6f", key="ko_inhib_anaero_c")
               params_cineticos.update({k: 1e9 for k in ['Ks', 'KO', 'KP_gen', 'n_sig','mumax']})
          elif tipo_mu == "Fermentación":
               st.info("Modelo Mixto: μ = μ_aerobio + μ_anaerobio.")
               with st.expander("Parámetros mu (Aerobio)", expanded=True):
                    ko_aerob_val = st.slider("KO_aerob (afinidad O2) [g/L]", 0.0001, 0.05, 0.0002, 0.0001, format="%.4f", key="ko_aerob_m")
                    params_cineticos['mumax_aerob'] = st.slider("μmax_aerob [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_aerob_m")
                    params_cineticos['Ks_aerob'] = st.slider("Ks_aerob [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aerob_m")
                    params_cineticos['KO_aerob'] = ko_aerob_val
               with st.expander("Parámetros mu (Anaerobio/Fermentativo)", expanded=True):
                    ko_inhib_anaerob_val = st.slider("KO_inhib_anaerob (Inhib. O2 en μ Anaerobio) [g/L]", 1e-6, 0.01, 0.0005, 1e-6, format="%.6f", key="ko_inhib_m")
                    params_cineticos['mumax_anaerob'] = st.slider("μmax_anaerob [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaerob_m")
                    params_cineticos['Ks_anaerob'] = st.slider("Ks_anaerob [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaerob_m")
                    params_cineticos['KiS_anaerob'] = st.slider("KiS_anaerob [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaerob_m")
                    params_cineticos['KP_anaerob'] = st.slider("KP_anaerob (Inhib. Etanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaerob_m")
                    params_cineticos['n_p'] = st.slider("Exponente Inhib. Etanol (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_m")
                    params_cineticos['KO_inhib_anaerob'] = ko_inhib_anaerob_val
                    if abs(params_cineticos['KO_aerob'] - params_cineticos['KO_inhib_anaerob']) < 0.0002: st.warning(f"KO_aerob y KO_inhib_anaerob cercanos.")
               params_cineticos.update({k: 1e9 for k in ['Ks', 'KO', 'KP_gen', 'n_sig', 'mumax']})

          # Parámetros Estequiométricos, Transferencia, Reactor
          st.subheader("2. Parámetros Estequiométricos y Otros")
          params_esteq = {}
          params_esteq['Yxs']=st.slider("Yxs [g/g]", 0.05, 0.6, 0.1, 0.01, key="yxs"); params_esteq['Yps']=st.slider("Yps [g/g]", 0.1, 0.51, 0.45, 0.01, key="yps"); params_esteq['Yxo']=st.slider("Yxo [gX/gO2]", 0.1, 2.0, 0.8, 0.1, key="yxo"); params_esteq['alpha_lp']=st.slider("α [gP/gX]", 0.0, 5.0, 2.2, 0.1, key="alpha"); params_esteq['beta_lp']=st.slider("β [gP/gX/h]", 0.0, 0.5, 0.05, 0.01, key="beta"); params_esteq['ms']=st.slider("ms [gS/gX/h]", 0.0, 0.2, 0.02, 0.01, key="ms"); params_esteq['mo']=st.slider("mo [gO2/gX/h]", 0.0, 0.1, 0.01, 0.005, key="mo"); params_esteq['Kd']=st.slider("Kd [1/h]", 0.0, 0.1, 0.01, 0.005, key="kd"); ko_inhib_p_mgL=st.slider("KO_inhib_prod [mg/L]", 0.001, 1.0, 0.05, 0.005, key="ko_inhib_p_mgL"); params_esteq['KO_inhib_prod']=ko_inhib_p_mgL / 1000.0
          st.subheader("3. Transferencia de Oxígeno")
          params_transfer = {}
          params_transfer['Kla1']=st.slider("kLa Fase 1 [1/h]", 10.0, 800.0, 100.0, 10.0, key="kla1")
          # *** CAMBIO 1: Aumentar kLa2 por defecto ***
          params_transfer['Kla2']=st.slider("kLa Fase 2/3 [1/h]", 0.0, 50.0, 15.0, 0.1, key="kla2") # <-- AUMENTADO DEFAULT a 15.0
          Cs_mgL=st.slider("Cs [mg/L]", 1.0, 15.0, 7.5, 0.1, key="cs_mgL"); params_transfer['Cs']=Cs_mgL / 1000.0
          st.subheader("4. Alimentación y Reactor")
          params_reactor = {}
          params_reactor['Sin']=st.number_input("Sin [g/L]", 10.0, 700.0, 400.0, 10.0, key="sin_conc"); params_reactor['Vmax']=st.number_input("Vmax [L]", value=10.0, min_value=0.1, step=0.5, key="vmax_reactor")

          # Configuración Temporal (n_intervals=12)
          st.subheader("5. Configuración Temporal")
          params_tiempo = {}
          t_aerobic_batch_val = st.number_input("Fin Fase 1 [h]", value=10.0, min_value=0.1, step=0.5, key="t_aerobic_end")
          params_tiempo['t_aerobic_batch'] = t_aerobic_batch_val
          t_feed_end_val = st.number_input("Fin Fase 2 [h]", value=34.0, min_value=params_tiempo['t_aerobic_batch'] + 0.1, step=0.5, key="t_feed_end_rto")
          params_tiempo['t_feed_end'] = t_feed_end_val
          t_total_val = st.number_input("Tiempo total [h]", value=39.0, min_value=params_tiempo['t_feed_end'] + 0.1, step=0.5, key="t_total_rto")
          params_tiempo['t_total'] = t_total_val
          n_intervals_val = st.number_input("Intervalos Control Fase 2", value=12, min_value=1, max_value=100, step=1, key="n_intervals_rto", help=f"Duración Fase 2: {params_tiempo['t_feed_end'] - params_tiempo['t_aerobic_batch']:.1f} h")
          params_tiempo['n_intervals'] = n_intervals_val

          # Condiciones Iniciales y Restricciones
          st.subheader("6. Condiciones Iniciales (t=0)")
          cond_iniciales = {}
          cond_iniciales['X0']=st.number_input("X0 [g/L]", 0.01, 10.0, 0.1, step=0.01, key="x0_init"); cond_iniciales['S0']=st.number_input("S0 [g/L]", 1.0, 200.0, 100.0, step=1.0, key="s0_init"); cond_iniciales['P0']=st.number_input("P0 [g/L]", 0.0, 50.0, 0.0, step=0.1, key="p0_init"); o0_default_mgL = min(0.08, Cs_mgL); O0_mgL=st.number_input("O0 [mg/L]", min_value=0.0, max_value=Cs_mgL, value=o0_default_mgL, step=0.01, key="o0_mgL"); cond_iniciales['O0']=O0_mgL / 1000.0; cond_iniciales['V0']=st.number_input("V0 [L]", value=5.0, min_value=0.05, step=0.1, key="v0_init")
          st.subheader("7. Restricciones y Penalización RTO")
          params_rto = {}
          fmin_val=st.number_input("Fmin [L/h]", value=0.0, min_value=0.0, format="%.4f", key="fmin_rto"); params_rto['Fmin']=fmin_val; params_rto['Fmax']=st.number_input("Fmax [L/h]", value=0.2, min_value=params_rto['Fmin'], step=0.01, key="fmax_rto"); params_rto['Smax_constraint']=st.number_input("Smax [g/L]", value=50.0, min_value=0.1, step=1.0, key="smax_const_rto"); default_pmax_rto=100.0; kp_to_use = None
          if tipo_mu in ["Fermentación", "Fermentación Conmutada"]: kp_to_use = params_cineticos.get('KP_anaerob', None)
          elif tipo_mu == "Monod con restricciones": kp_to_use = params_cineticos.get('KP_gen', None)
          if kp_to_use is not None and isinstance(kp_to_use, (int, float)) and kp_to_use > 1e-6 and kp_to_use < 1e8: default_pmax_rto = max(10.0, kp_to_use * 0.95)
          params_rto['Pmax_constraint']=st.number_input("Pmax [g/L]", value=default_pmax_rto, min_value=1.0, step=1.0, key="pmax_const_rto", help=f"Límite P. Sugerido: ~95% KP ({kp_to_use:.1f} g/L si aplica).");
          w_Smax_penalty_val = st.number_input("Peso Pen. Smax", value=100.0, min_value=0.0, key="w_smax_rto", help="Poner a 0 para desactivar") # Permitir desactivar
          params_rto['w_Smax_penalty'] = w_Smax_penalty_val

          all_params = {**params_cineticos, **params_esteq, **params_transfer, **params_reactor, **params_tiempo, **params_rto}

     # --- Factores de Escalado ---
     X_scale=max(1.0,cond_iniciales['X0']*10,50.0); S_scale=max(1.0,cond_iniciales['S0'],params_reactor['Sin'],params_rto['Smax_constraint']); P_scale=max(1.0,params_rto.get('Pmax_constraint',100.0)); O2_scale=max(1e-5,params_transfer['Cs']); V_scale=max(0.1,params_reactor['Vmax'],cond_iniciales['V0']); F_scale=max(1e-3,params_rto['Fmax']) if params_rto['Fmax'] > 0 else 0.1; x_scales=ca.vertcat(X_scale,S_scale,P_scale,O2_scale,V_scale); u_scale=F_scale

     # --- Función DAE/ODE ESCALADO (CON CORRECCIÓN qO) ---
     def create_unified_dae_dict_scaled(params, scales):
          # ... (Código interno idéntico a la versión anterior, con qO corregido) ...
          nx=5; x_scaled_sym=ca.MX.sym("x_scaled",nx); u_scaled_sym=ca.MX.sym("u_scaled"); Kla_sym=ca.MX.sym("Kla_phase"); params_sym_list=[Kla_sym]; param_dict={'Kla_sym': Kla_sym}; tipo=params['tipo_mu']
          if tipo=="Fermentación Conmutada": considerar_O2_sym=ca.MX.sym("considerar_O2_flag"); params_sym_list.append(considerar_O2_sym); param_dict['considerar_O2_sym']=considerar_O2_sym
          else: placeholder_sym=ca.MX.sym("placeholder"); params_sym_list.append(placeholder_sym); param_dict['considerar_O2_sym']=None
          p_sym=ca.vertcat(*params_sym_list); x_sc=scales['x']; u_sc=scales['u']; x_orig_sym=x_scaled_sym*x_sc; u_orig_sym=u_scaled_sym*u_sc; X,S,P,O2,V=x_orig_sym[0],x_orig_sym[1],x_orig_sym[2],x_orig_sym[3],x_orig_sym[4]; F=u_orig_sym; safe_X=ca.fmax(1e-9,X); safe_S=ca.fmax(0.0,S); safe_P=ca.fmax(0.0,P); safe_O2=ca.fmax(0.0,O2); safe_V=ca.fmax(1e-6,V); D=F/safe_V; mu=0.0
          if tipo=="Monod simple": mu=mu_monod_ca(safe_S, params['mumax'], params['Ks'])
          elif tipo=="Monod sigmoidal": mu=mu_sigmoidal_ca(safe_S, params['mumax'], params['Ks'], params['n_sig'])
          elif tipo=="Monod con restricciones": mu=mu_completa_ca(safe_S, safe_O2, safe_P, params['mumax'], params['Ks'], params['KO'], params['KP_gen'])
          elif tipo=="Fermentación": mu=mu_fermentacion_ca(safe_S, safe_P, safe_O2, params['mumax_aerob'], params['Ks_aerob'], params['KO_aerob'], params['mumax_anaerob'], params['Ks_anaerob'], params['KiS_anaerob'], params['KP_anaerob'], params.get('n_p', 1.0), params['KO_inhib_anaerob'], considerar_O2=None)
          elif tipo=="Fermentación Conmutada": considerar_O2_flag_sym = param_dict['considerar_O2_sym']; mu = mu_fermentacion_ca(safe_S, safe_P, safe_O2, params['mumax_aerob'], params['Ks_aerob'], params['KO_aerob'], params['mumax_anaerob'], params['Ks_anaerob'], params['KiS_anaerob'], params['KP_anaerob'], params.get('n_p', 1.0), params['KO_inhib_anaerob'], considerar_O2=considerar_O2_flag_sym )
          mu_net=ca.fmax(0.0, mu)-params['Kd']; qP_base=params['alpha_lp']*ca.fmax(0.0, mu)+params['beta_lp']; safe_O2_inhib=ca.fmax(1e-9, safe_O2); inhib_factor_O2_prod=ca.fmax(params['KO_inhib_prod'], 1e-9)/ca.fmax(params['KO_inhib_prod']+safe_O2_inhib, 1e-9); qP=qP_base*inhib_factor_O2_prod; qP=ca.fmax(0.0, qP); consumo_S_X=(ca.fmax(0.0, mu)/ca.fmax(params['Yxs'], 1e-9)); consumo_S_P=(qP/ca.fmax(params['Yps'], 1e-9)); consumo_S_maint=params['ms']; qS=consumo_S_X+consumo_S_P+consumo_S_maint; qS=ca.fmax(0.0, qS);
          safe_O2_for_mu_aer = ca.fmax(1e-9, safe_O2); mu_aer_only = params['mumax_aerob'] * (safe_S / ca.fmax(params['Ks_aerob'] + safe_S, 1e-9)) * (safe_O2_for_mu_aer / ca.fmax(params['KO_aerob'] + safe_O2_for_mu_aer, 1e-9)); mu_aer_only = ca.fmax(0.0, mu_aer_only); consumo_O2_X_aerob = (mu_aer_only / ca.fmax(params['Yxo'], 1e-9)); consumo_O2_maint = params['mo']; qO = consumo_O2_X_aerob + consumo_O2_maint; qO = ca.fmax(0.0, qO); # qO Corregido
          Rate_X=mu_net*safe_X; Rate_S=-qS*safe_X; Rate_P=qP*safe_X; OUR=qO*safe_X; current_Kla=param_dict['Kla_sym']; OTR=current_Kla*(params['Cs']-safe_O2); Rate_O2=OTR-OUR; dXdt=Rate_X-D*safe_X; dSdt=Rate_S+D*(params['Sin']-safe_S); dPdt=Rate_P-D*safe_P; dOdt=Rate_O2-D*safe_O2; dVdt=F; ode_expr_scaled=ca.vertcat(dXdt/x_sc[0], dSdt/x_sc[1], dPdt/x_sc[2], dOdt/x_sc[3], dVdt/x_sc[4]); dae={'x':x_scaled_sym,'p':p_sym,'u':u_scaled_sym,'ode':ode_expr_scaled}; return dae

     # --- Ejecución de la Optimización RTO ---
     if st.button("🚀 Ejecutar Optimización RTO"):
          st.info("Iniciando RTO...")
          start_time_rto = time.time()

          # Validaciones y tiempos
          t_fase1=all_params['t_aerobic_batch']; t_fase2_end=all_params['t_feed_end']; t_fase3_end=all_params['t_total']; n_ctrl_intervals=all_params['n_intervals']; T_feed_duration=t_fase2_end-t_fase1; T_post_feed_duration=t_fase3_end-t_fase2_end
          if T_feed_duration <= 1e-6 or n_ctrl_intervals <= 0 or T_post_feed_duration < -1e-6: st.error("Conf. temporal inválida."); st.stop()
          dt_fb = T_feed_duration / n_ctrl_intervals;
          if u_scale <= 1e-9: st.warning("F_scale bajo."); u_scale=1.0

          # Crear DAE
          try:
               scales_dict = {'x': x_scales, 'u': u_scale}; dae_system_scaled = create_unified_dae_dict_scaled(all_params, scales_dict); st.success("DAE ESCALADO creado.")
          except Exception as e: st.error(f"Error DAE: {e}"); st.error(traceback.format_exc()); st.stop()

          # Parámetros p
          p_list_phase1=[all_params['Kla1']]; p_list_phase2=[all_params['Kla2']]; p_list_phase3=[all_params['Kla2']];
          if all_params['tipo_mu']=="Fermentación Conmutada": p_list_phase1.append(1.0); p_list_phase2.append(0.0); p_list_phase3.append(0.0)
          else: p_list_phase1.append(0.0); p_list_phase2.append(0.0); p_list_phase3.append(0.0)
          p_phase1=ca.vertcat(*p_list_phase1); p_phase2=ca.vertcat(*p_list_phase2); p_phase3=ca.vertcat(*p_list_phase3)

          # Simulación Fase 1
          st.info(f"[FASE 1] Simulando...")
          try:
               integrator_opts_sim = {"t0":0, "tf":t_fase1, "reltol":1e-7, "abstol":1e-9}
               integrator_phase1 = ca.integrator("integrator_p1", "idas", dae_system_scaled, integrator_opts_sim)
               x0_np_orig=np.array([cond_iniciales['X0'],cond_iniciales['S0'],cond_iniciales['P0'],cond_iniciales['O0'],cond_iniciales['V0']])
               x0_np_scaled = x0_np_orig / np.array(x_scales).flatten()
               res_phase1_scaled = integrator_phase1(x0=x0_np_scaled, p=p_phase1, u=0.0)
               x_end_phase1_scaled = np.maximum(np.array(res_phase1_scaled['xf']).flatten(), 0.0)
               if any(np.isnan(x_end_phase1_scaled)): raise ValueError("NaN Fase 1")
               x_end_phase1_scaled[3] = max(0.0, x_end_phase1_scaled[3]) # Forzar O2 >= 0
               x_end_phase1 = x_end_phase1_scaled * np.array(x_scales).flatten()
               st.success(f"[FASE 1] Estado final: X={x_end_phase1[0]:.3f}, S={x_end_phase1[1]:.3f}, P={x_end_phase1[2]:.3f}, O2={x_end_phase1[3]*1000:.4f}, V={x_end_phase1[4]:.3f}")
          except Exception as e: st.error(f"Error Fase 1: {e}"); st.error(traceback.format_exc()); st.stop()

          # Simulación Guess
          st.info("[GUESS] Simulando trayectoria inicial...")
          x_guess_traj_scaled=[x_end_phase1_scaled]; xk_guess=x_end_phase1_scaled.copy(); F_const_guess=(all_params['Fmax']+all_params['Fmin'])/2.0*0.8; u_const_guess_scaled=F_const_guess/u_scale if u_scale>1e-9 else 0.0; integrator_opts_guess={"t0":0,"tf":dt_fb,"reltol":1e-6,"abstol":1e-8}; integrator_guess_f2=ca.integrator("int_guess_f2","idas",dae_system_scaled,integrator_opts_guess); integrator_opts_guess_f3={"t0":0,"tf":T_post_feed_duration,"reltol":1e-6,"abstol":1e-8}; integrator_guess_f3=ca.integrator("int_guess_f3","idas",dae_system_scaled,integrator_opts_guess_f3)
          try:
               for k in range(n_ctrl_intervals): V_curr_unsc=xk_guess[4]*V_scale; F_act_guess=F_const_guess if V_curr_unsc < all_params['Vmax']-1e-6 else 0.0; u_act_guess_sc=F_act_guess/u_scale if u_scale>1e-9 else 0.0; res_guess_k=integrator_guess_f2(x0=xk_guess,p=p_phase2,u=u_act_guess_sc); xk_guess=np.maximum(np.array(res_guess_k['xf']).flatten(),0.0); x_guess_traj_scaled.append(xk_guess)
               if T_post_feed_duration > 1e-6: res_guess_f3 = integrator_guess_f3(x0=xk_guess, p=p_phase3, u=0.0)
               st.success("[GUESS] Simulación inicial OK.")
          except Exception as e: st.warning(f"Fallo guess: {e}. Usando guess simple."); x_guess_traj_scaled=[x_end_phase1_scaled]*(n_ctrl_intervals+1)

          # Formulación Optimización
          st.info(f"[FASE 2] Formulando RTO...")
          opti = ca.Opti()
          nx=5; X_scaled_vars=[]; F_scaled_vars=[]
          x_start_phase2_scaled_param = opti.parameter(nx)
          opti.set_value(x_start_phase2_scaled_param, x_end_phase1_scaled)
          Xk_scaled = x_start_phase2_scaled_param
          # Tolerancias estrictas para integrador en Opti
          integrator_opts_opti = {"t0": 0, "tf": dt_fb, "reltol": 1e-7, "abstol": 1e-9}
          integrator_interval_scaled = ca.integrator("int_interval_scaled", "idas", dae_system_scaled, integrator_opts_opti)

          for k in range(n_ctrl_intervals):
               Fk_scaled = opti.variable()
               Fmin_sc=all_params['Fmin']/u_scale if u_scale>1e-9 else 0.0; Fmax_sc=all_params['Fmax']/u_scale if u_scale>1e-9 else 1.0
               opti.subject_to(Fk_scaled >= Fmin_sc); opti.subject_to(Fk_scaled <= Fmax_sc); F_scaled_vars.append(Fk_scaled)
               res_k_scaled = integrator_interval_scaled(x0=Xk_scaled, p=p_phase2, u=Fk_scaled); Xk_end_scaled = res_k_scaled['xf']
               # Restricción no-negatividad estricta
               opti.subject_to(Xk_end_scaled >= 0.0)
               # Restricciones Smax, Pmax, Vmax
               Smax_sc=all_params['Smax_constraint']/S_scale; Pmax_sc=all_params['Pmax_constraint']/P_scale; Vmax_sc=all_params['Vmax']/V_scale
               # opti.subject_to(Xk_end_scaled[1] <= Smax_sc + 1e-6) # Smax (Descomentar si necesario)
               # opti.subject_to(Xk_end_scaled[2] <= Pmax_sc + 1e-6) # Pmax (Descomentar si necesario)
               opti.subject_to(Xk_end_scaled[4] <= Vmax_sc + 1e-6) # Vmax activa
               if len(x_guess_traj_scaled)>k+1: F_guess_k=F_const_guess; V_guess_k_unsc=x_guess_traj_scaled[k][4]*V_scale; u_guess_k_sc= (F_guess_k if V_guess_k_unsc<all_params['Vmax']-1e-6 else 0.0)/u_scale if u_scale>1e-9 else 0.0; opti.set_initial(Fk_scaled, u_guess_k_sc)
               else: opti.set_initial(Fk_scaled, u_const_guess_scaled)
               X_scaled_vars.append(Xk_end_scaled); Xk_scaled = Xk_end_scaled
          X_end_feed_scaled = Xk_scaled

          # Penalización Smax (Corregido)
          penalty_smax_total = ca.MX(0.0)
          if all_params['w_Smax_penalty'] > 1e-9:
               Smax_scaled = all_params['Smax_constraint'] / S_scale
               violation_vector = ca.fmax(0, ca.vertcat(*[xk[1] for xk in X_scaled_vars]) - Smax_scaled)
               sum_sq_violation = ca.dot(violation_vector, violation_vector)
               penalty_smax_total = all_params['w_Smax_penalty'] * sum_sq_violation

          # Simulación Fase 3
          st.info("[FASE 3 - Integración en Opti]...")
          if T_post_feed_duration > 1e-6:
               integrator_opts_opti_f3 = {"t0":0, "tf":T_post_feed_duration, "reltol":1e-7, "abstol":1e-9} # Tolerancias estrictas
               integrator_phase3_scaled = ca.integrator("integrator_p3_scaled", "idas", dae_system_scaled, integrator_opts_opti_f3)
               res_phase3_sym_scaled = integrator_phase3_scaled(x0=X_end_feed_scaled, p=p_phase3, u=0.0); X_final_total_scaled = res_phase3_sym_scaled['xf']
          else: X_final_total_scaled = X_end_feed_scaled

          # Función Objetivo
          P_final_unsc=X_final_total_scaled[2]*P_scale; V_final_unsc=X_final_total_scaled[4]*V_scale; objective_PV = -(P_final_unsc*V_final_unsc); objective_total = objective_PV + penalty_smax_total
          opti.minimize(objective_total)

          # --- Opciones del Solver ---
          st.info("Configurando solver IPOPT...")
          p_opts = {"expand": True}
          s_opts = {
               "max_iter": 3000, "print_level": 0, "sb": 'yes',
               "tol": 1e-6, "constr_viol_tol": 1e-6,
               "acceptable_tol": 1e-4, "acceptable_constr_viol_tol": 1e-4,
               "hessian_approximation": "limited-memory",
          }
          opti.solver("ipopt", p_opts, s_opts)

          # --- Solución y Resultados ---
          try:
               st.info("🚀 Resolviendo RTO...")
               solve_start_time = time.time(); sol = opti.solve(); solve_end_time = time.time()
               st.success(f"[OPTIMIZACIÓN] ¡Solución encontrada en {solve_end_time - solve_start_time:.2f} segundos!")

               # Extracción, desescalado y reporte
               F_opt_phase2_scaled=np.array([sol.value(fk) for fk in F_scaled_vars]); X_final_total_scaled_opt=sol.value(X_final_total_scaled)
               F_opt_phase2=F_opt_phase2_scaled*u_scale; X_final_total_opt=X_final_total_scaled_opt*np.array(x_scales).flatten()
               P_final_opt=X_final_total_opt[2]; V_final_opt=X_final_total_opt[4]; O2_final_opt_mgL=X_final_total_opt[3]*1000.0
               st.metric("Producto Total Óptimo (P*V)", f"{P_final_opt * V_final_opt:.4f} g")
               col1, col2, col3 = st.columns(3); col1.metric("Conc. Final Etanol", f"{P_final_opt:.3f} g/L"); col2.metric("Volumen Final", f"{V_final_opt:.3f} L"); col3.metric("O2 Final", f"{O2_final_opt_mgL:.4f} mg/L")
               try: Smax_penalty_value=sol.value(penalty_smax_total); col1.metric("Penalización Smax", f"{Smax_penalty_value:.4g}")
               except: col1.metric("Penalización Smax", "N/A")
               st.write("Perfil óptimo de flujo (Fase 2):"); t_feed_points=np.linspace(t_fase1,t_fase2_end,n_ctrl_intervals+1)
               if len(F_opt_phase2)==n_ctrl_intervals:
                    df_flow=pd.DataFrame({'T Inicio (h)':t_feed_points[:-1],'T Fin (h)': t_feed_points[1:],'F Opt (L/h)': F_opt_phase2}); st.dataframe(df_flow.style.format({"F Opt (L/h)": "{:.4f}"}))
                    fig_flow,ax_flow=plt.subplots(figsize=(10,3)); ax_flow.step(t_feed_points[:-1],F_opt_phase2,where='post',label='$F_{opt}$'); ax_flow.plot([t_feed_points[-2],t_feed_points[-1]],[F_opt_phase2[-1],F_opt_phase2[-1]],'-'); ax_flow.set_xlabel("T [h]"); ax_flow.set_ylabel("F [L/h]"); ax_flow.set_title("Perfil Flujo Óptimo"); ax_flow.set_xlim(t_fase1,t_fase2_end); ax_flow.set_ylim(bottom=-0.05*all_params['Fmax']); ax_flow.grid(True,axis='y',ls=':'); ax_flow.legend(); st.pyplot(fig_flow)
               else: st.warning("Error F_opt len")

          except RuntimeError as e:
               st.error(f"[ERROR] Solver IPOPT: {e}");
               try: st.warning("Debug:"); st.write(f"Obj:{opti.debug.value(objective_total):.4g}"); st.write(f"PV:{opti.debug.value(objective_PV):.4g}"); st.write(f"Pen:{opti.debug.value(penalty_smax_total):.4g}"); st.write("F sc:",[f"{opti.debug.value(fv_s):.4g}" for fv_s in F_scaled_vars]); st.write(f"Xf_F2sc:{[f'{v:.4g}' for v in opti.debug.value(X_end_feed_scaled)]}"); st.write(f"Xf_F3sc:{[f'{v:.4g}' for v in opti.debug.value(X_final_total_scaled)]}")
               except Exception as de: st.error(f"Err debug: {de}")
               st.stop()
          except Exception as e: st.error(f"Error Opt: {e}"); st.error(traceback.format_exc()); st.stop()

          # --- Reconstrucción y Gráficas ---
          # ... (Código idéntico al anterior) ...
          F_opt_phase2_scaled_again = F_opt_phase2 / u_scale if u_scale > 1e-9 else np.zeros_like(F_opt_phase2)
          st.info("Reconstruyendo trayectoria completa...")
          t_plot_full = []; x_plot_full_scaled = []; f_plot_full_used = []
          plot_ok = True; start_time_sim = time.time()
          N_plot_p1 = max(20, int(t_fase1*10)); t_eval_p1 = np.linspace(0, t_fase1, N_plot_p1); dt_p1 = t_eval_p1[1]-t_eval_p1[0] if N_plot_p1 > 1 else t_fase1
          integrator_p1_plot = ca.integrator("int_p1_plot", "idas", dae_system_scaled, {"t0":0, "tf":dt_p1, "reltol":1e-7, "abstol":1e-9})
          xk_plot_scaled = x0_np_scaled.copy(); t_plot_full.append(0.0); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
          for i in range(N_plot_p1 - 1):
               try:
                    res_plot_scaled = integrator_p1_plot(x0=xk_plot_scaled, p=p_phase1, u=0.0)
                    xk_plot_scaled = np.maximum(np.array(res_plot_scaled['xf']).flatten(), 0.0);
                    if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F1 plot")
                    t_plot_full.append(t_eval_p1[i+1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
               except Exception as plot_e: st.error(f"Fallo plot F1: {plot_e}"); plot_ok = False; break
          if not plot_ok: st.stop()
          N_plot_p2_per_interval = max(2, int(1.0/dt_fb*2) if dt_fb > 1e-6 else 5); N_plot_p2 = n_ctrl_intervals*N_plot_p2_per_interval
          t_eval_p2 = np.linspace(t_fase1, t_fase2_end, N_plot_p2 + 1); dt_p2_fine = (t_fase2_end - t_fase1)/N_plot_p2 if N_plot_p2 > 0 else 0
          integrator_p2_plot = ca.integrator("int_p2_plot", "idas", dae_system_scaled, {"t0":0, "tf":dt_p2_fine, "reltol":1e-7, "abstol":1e-9})
          for i in range(N_plot_p2):
               t_now = t_eval_p2[i]; k_interval = int((t_now - t_fase1)/dt_fb + 1e-9) if dt_fb > 1e-9 else 0; k_interval = max(0, min(k_interval, n_ctrl_intervals - 1))
               F_now_scaled = F_opt_phase2_scaled_again[k_interval]; F_now_unscaled = F_opt_phase2[k_interval]
               V_current_unscaled = xk_plot_scaled[4]*V_scale;
               if V_current_unscaled >= all_params['Vmax'] - 1e-6: F_now_scaled = 0.0; F_now_unscaled = 0.0
               try:
                    res_plot_scaled = integrator_p2_plot(x0=xk_plot_scaled, p=p_phase2, u=F_now_scaled)
                    xk_plot_scaled = np.maximum(np.array(res_plot_scaled['xf']).flatten(), 0.0)
                    if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F2 plot")
                    t_plot_full.append(t_eval_p2[i+1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(F_now_unscaled)
               except Exception as plot_e: st.error(f"Fallo plot F2 (t={t_now:.2f}): {plot_e}"); plot_ok = False; break
          if not plot_ok: st.stop()
          if T_post_feed_duration > 1e-6:
               N_plot_p3 = max(20, int(T_post_feed_duration*10)); t_eval_p3 = np.linspace(t_fase2_end, t_fase3_end, N_plot_p3 + 1); dt_p3_fine = T_post_feed_duration/N_plot_p3 if N_plot_p3 > 0 else 0
               integrator_p3_plot = ca.integrator("int_p3_plot", "idas", dae_system_scaled, {"t0":0, "tf":dt_p3_fine, "reltol":1e-7, "abstol":1e-9})
               for i in range(N_plot_p3):
                    try:
                         res_plot_scaled = integrator_p3_plot(x0=xk_plot_scaled, p=p_phase3, u=0.0)
                         xk_plot_scaled = np.maximum(np.array(res_plot_scaled['xf']).flatten(), 0.0)
                         if any(np.isnan(xk_plot_scaled)): raise ValueError("NaN F3 plot")
                         t_plot_full.append(t_eval_p3[i+1]); x_plot_full_scaled.append(xk_plot_scaled); f_plot_full_used.append(0.0)
                    except Exception as plot_e: st.error(f"Fallo plot F3: {plot_e}"); plot_ok = False; break
               if not plot_ok: st.stop()
          else:
               if f_plot_full_used: f_plot_full_used[-1] = 0.0
          end_time_sim = time.time(); st.success(f"Simulación detallada OK ({end_time_sim - start_time_sim:.2f} s).")
          t_sim=np.array(t_plot_full); x_sim_scaled=np.array(x_plot_full_scaled); f_sim=np.array(f_plot_full_used)
          x_sim = x_sim_scaled * np.array(x_scales).flatten()[np.newaxis, :]
          X_sim, S_sim, P_sim, O2_sim, V_sim = [x_sim[:, i] for i in range(nx)]; O2_sim_mgL = O2_sim*1000.0
          if plot_ok:
               st.info("📊 Calculando tasas y graficando...")
               mu_sim_calc=[]; qP_sim_calc=[]; qS_sim_calc=[]; qO_sim_calc=[]
               OTR_sim_calc=[]; OUR_sim_calc=[]; inhib_factor_sim_calc=[]
               ca_params = all_params.copy()
               for k, v in ca_params.items():
                    if isinstance(v, list): ca_params[k] = ca.DM(v)
               sym_S = ca.SX.sym('S'); sym_P = ca.SX.sym('P'); sym_O2 = ca.SX.sym('O2')
               mu_eval = mu_fermentacion_ca(sym_S, sym_P, sym_O2, ca_params['mumax_aerob'], ca_params['Ks_aerob'], ca_params['KO_aerob'], ca_params['mumax_anaerob'], ca_params['Ks_anaerob'], ca_params['KiS_anaerob'], ca_params['KP_anaerob'], ca_params.get('n_p', 1.0), ca_params['KO_inhib_anaerob'], considerar_O2=None)
               f_mu = ca.Function('f_mu', [sym_S, sym_P, sym_O2], [mu_eval])
               qP_base_eval = ca_params['alpha_lp'] * ca.fmax(0.0, mu_eval) + ca_params['beta_lp']
               safe_O2_inhib_eval = ca.fmax(1e-9, sym_O2); inhib_fact_eval = ca.fmax(ca_params['KO_inhib_prod'], 1e-9) / ca.fmax(ca_params['KO_inhib_prod'] + safe_O2_inhib_eval, 1e-9); qP_eval = qP_base_eval * inhib_fact_eval; qP_eval = ca.fmax(0.0, qP_eval)
               f_qP = ca.Function('f_qP', [sym_S, sym_P, sym_O2], [qP_eval, inhib_fact_eval])
               cSX_eval = (ca.fmax(0.0, mu_eval) / ca.fmax(ca_params['Yxs'], 1e-9)); cSP_eval = (qP_eval / ca.fmax(ca_params['Yps'], 1e-9)); cSm_eval = ca_params['ms']; qS_eval = cSX_eval + cSP_eval + cSm_eval; qS_eval = ca.fmax(0.0, qS_eval); f_qS = ca.Function('f_qS', [sym_S, sym_P, sym_O2], [qS_eval])
               safe_O2_qO_eval = ca.fmax(1e-9, sym_O2); mu_aer_only_eval = ca_params['mumax_aerob'] * (sym_S / ca.fmax(ca_params['Ks_aerob'] + sym_S, 1e-9)) * (safe_O2_qO_eval / ca.fmax(ca_params['KO_aerob'] + safe_O2_qO_eval, 1e-9)); mu_aer_only_eval = ca.fmax(0.0, mu_aer_only_eval); cOXa_eval = (mu_aer_only_eval / ca.fmax(ca_params['Yxo'], 1e-9)); cOm_eval = ca_params['mo']; qO_eval = cOXa_eval + cOm_eval; qO_eval = ca.fmax(0.0, qO_eval); f_qO = ca.Function('f_qO', [sym_S, sym_O2], [qO_eval])
               for i in range(len(t_sim)):
                    ti, Xi, Si, Pi, O2i, Vi = t_sim[i], X_sim[i], S_sim[i], P_sim[i], O2_sim[i], V_sim[i]
                    safe_Xi=max(1e-9,Xi); fase=1; kla_now=ca_params['Kla1']
                    if ti >= t_fase1 - 1e-6: fase=2; kla_now=ca_params['Kla2'];
                    if ti >= t_fase2_end - 1e-6: fase=3
                    mu_i = float(f_mu(Si, Pi, O2i)[0]); qP_i, inhib_factor_i = f_qP(Si, Pi, O2i); qP_i=float(qP_i); inhib_factor_i=float(inhib_factor_i); qS_i = float(f_qS(Si, Pi, O2i)[0]); qO_i = float(f_qO(Si, O2i)[0])
                    mu_sim_calc.append(mu_i); qP_sim_calc.append(qP_i); qS_sim_calc.append(qS_i); qO_sim_calc.append(qO_i); inhib_factor_sim_calc.append(inhib_factor_i); OTR_i = kla_now * (ca_params['Cs'] - O2i); OUR_i = qO_i * safe_Xi; OTR_sim_calc.append(OTR_i * 1000.0); OUR_sim_calc.append(OUR_i * 1000.0)
               mu_sim=np.array(mu_sim_calc);qP_sim=np.array(qP_sim_calc);qS_sim=np.array(qS_sim_calc);qO_sim=np.array(qO_sim_calc)*1000;OTR_sim=np.array(OTR_sim_calc);OUR_sim=np.array(OUR_sim_calc);inhib_factor_sim=np.array(inhib_factor_sim_calc)
               
               
               
               """
               st.subheader("📊 Gráficas Detalladas")
               fig = plt.figure(figsize=(14, 22))
               # ... (Código de graficación idéntico) ...
               def add_phase_lines(ax, t1, t2):
                    current_ymin, current_ymax = ax.get_ylim()
                    if not np.isfinite(current_ymin) or not np.isfinite(current_ymax) or current_ymax <= current_ymin + 1e-6: y_data=[]; [y_data.extend(line.get_ydata()) for line in ax.get_lines()]; valid_data = [y for y in y_data if np.isfinite(y)]; current_ymin, current_ymax = (np.min(valid_data) - 0.1*abs(np.max(valid_data)-np.min(valid_data)), np.max(valid_data) + 0.1*abs(np.max(valid_data)-np.min(valid_data))) if valid_data and abs(np.max(valid_data)-np.min(valid_data)) > 1e-6 else (np.min(valid_data)-0.1, np.max(valid_data)+0.1) if valid_data else (0,1); current_ymax = current_ymin + 1.0 if current_ymax <= current_ymin + 1e-6 else current_ymax; ax.set_ylim(current_ymin, current_ymax)
                    ax.vlines([t1, t2], current_ymin, current_ymax, colors=['gray', 'purple'], ls='--', lw=1.5, zorder=10); ax.set_ylim(current_ymin, current_ymax)
               ax1=plt.subplot(5,2,1); color='tab:red'; ax1.step(t_sim,f_sim,where='post',color=color,label='$F_{opt}$'); ax1.plot([t_sim[-1]],[f_sim[-1]],marker='o',ls='',color=color); ax1.set_ylabel('Flujo [L/h]',color=color); ax1.tick_params(axis='y',labelcolor=color); ax1b=ax1.twinx(); color='tab:blue'; ax1b.plot(t_sim,V_sim,color=color,ls='-',label='$V$'); ax1b.set_ylabel('Volumen [L]',color=color); ax1b.tick_params(axis='y',labelcolor=color); ax1.set_xlabel('Tiempo [h]'); ax1.grid(True); ax1.set_title('Alimentación Optimizada y Volumen'); lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax1b.get_legend_handles_labels(); leg_el_phases=[Line2D([0],[0],color='gray',ls='--',lw=1.5,label=f'Fin F1 ({t_fase1:.1f}h)'),Line2D([0],[0],color='purple',ls='--',lw=1.5,label=f'Fin F2 ({t_fase2_end:.1f}h)')]; ax1b.legend(lines+lines2+leg_el_phases,labels+labels2+[le.get_label() for le in leg_el_phases],loc='best'); add_phase_lines(ax1,t_fase1,t_fase2_end); f_min_plot=-0.05*all_params['Fmax']; ax1.set_ylim(bottom=f_min_plot)
               ax2=plt.subplot(5,2,3); ax2.plot(t_sim,X_sim,'g-'); ax2.set_title('Biomasa (X)'); ax2.set_ylabel('[g/L]'); ax2.set_xlabel('Tiempo [h]'); ax2.grid(True); ax2.set_ylim(bottom=0); add_phase_lines(ax2,t_fase1,t_fase2_end)
               ax3=plt.subplot(5,2,4); ax3.plot(t_sim,S_sim,'m-'); ax3.set_title('Sustrato (S)'); ax3.set_ylabel('[g/L]'); ax3.set_xlabel('Tiempo [h]'); ax3.grid(True); ax3.set_ylim(bottom=0); ax3.axhline(all_params['Smax_constraint'],color='red',ls=':',lw=1.5,label=f'$S_{{max}}$ Lim ({all_params["Smax_constraint"]:.1f})'); ax3.legend(loc='best'); add_phase_lines(ax3,t_fase1,t_fase2_end)
               ax4=plt.subplot(5,2,5); ax4.plot(t_sim,P_sim,'k-'); ax4.set_title('Etanol (P)'); ax4.set_ylabel('[g/L]'); ax4.set_xlabel('Tiempo [h]'); ax4.grid(True); ax4.set_ylim(bottom=0); ax4.axhline(all_params['Pmax_constraint'],color='red',ls=':',lw=1.5,label=f'$P_{{max}}$ Lim ({all_params["Pmax_constraint"]:.1f})'); ax4.legend(loc='best'); add_phase_lines(ax4,t_fase1,t_fase2_end)
               ax5=plt.subplot(5,2,6); ax5.plot(t_sim,O2_sim_mgL,'c-',label='$O_2$'); ax5.set_title('$O_2$ y Factor Inhibición $P$'); ax5.set_ylabel('$O_2$ [mg/L]',color='c'); ax5.set_xlabel('Tiempo [h]'); ax5.grid(True); o2_top=max(all_params['Cs']*1000*1.1,np.max(O2_sim_mgL)*1.1 if len(O2_sim_mgL[np.isfinite(O2_sim_mgL)])>0 else all_params['Cs']*1000*1.1); o2_top=max(o2_top,0.1); ax5.set_ylim(bottom=0.0,top=o2_top); ax5.tick_params(axis='y',labelcolor='c'); ax5b=ax5.twinx(); ax5b.plot(t_sim,inhib_factor_sim,color='darkorange',ls='--',label=r'Inhib $O_2 \to P$'); ax5b.set_ylabel('Factor Inhib. Prod. [-]',color='darkorange'); ax5b.tick_params(axis='y',labelcolor='darkorange'); ax5b.set_ylim(bottom=-0.05,top=1.05); lines5,labels5=ax5.get_legend_handles_labels(); lines5b,labels5b=ax5b.get_legend_handles_labels(); ax5b.legend(lines5+lines5b,labels5+labels5b,loc='center right'); add_phase_lines(ax5,t_fase1,t_fase2_end)
               ax6=plt.subplot(5,2,7); ax6.plot(t_sim,mu_sim,'y-'); ax6.set_title('Tasa Crecimiento (μ)'); ax6.set_ylabel('[1/h]'); ax6.set_xlabel('Tiempo [h]'); ax6.grid(True); ax6.set_ylim(bottom=0); add_phase_lines(ax6,t_fase1,t_fase2_end)
               ax7=plt.subplot(5,2,8); ax7.plot(t_sim,qS_sim,'r-',label=r'$q_S$'); ax7.plot(t_sim,qP_sim,'k-',label=r'$q_P$'); ax7.set_title('Tasas Específicas ($q_S, q_P$)'); ax7.set_ylabel('[g/gX/h]'); ax7.set_xlabel('Tiempo [h]'); ax7.grid(True); ax7.legend(loc='upper left'); ax7b=ax7.twinx(); ax7b.plot(t_sim,qO_sim,'c--',label=r'$q_{O2}$'); ax7b.set_ylabel('[mg O2/gX/h]',color='c'); ax7b.tick_params(axis='y',labelcolor='c'); ax7b.legend(loc='upper right'); add_phase_lines(ax7,t_fase1,t_fase2_end); valid_qs=qS_sim[np.isfinite(qS_sim)]; valid_qp=qP_sim[np.isfinite(qP_sim)]; valid_qo=qO_sim[np.isfinite(qO_sim)]; q_min=min(0,np.min(valid_qs)*1.1 if len(valid_qs)>0 else 0,np.min(valid_qp)*1.1 if len(valid_qp)>0 else 0); qO_min=min(0,np.min(valid_qo)*1.1 if len(valid_qo)>0 else 0); q_max=max(0.1,np.max(valid_qs)*1.1 if len(valid_qs)>0 else 0.1,np.max(valid_qp)*1.1 if len(valid_qp)>0 else 0.1); qO_max=max(0.1,np.max(valid_qo)*1.1 if len(valid_qo)>0 else 0.1); ax7.set_ylim(bottom=q_min,top=q_max); ax7b.set_ylim(bottom=qO_min,top=qO_max)
               ax8=plt.subplot(5,2,9); ax8.plot(t_sim,OTR_sim,'b-',label='OTR'); ax8.plot(t_sim,OUR_sim,'g--',label='OUR'); ax8.set_title('Tasas Volumétricas O2'); ax8.set_ylabel('[mg O2/L/h]'); ax8.set_xlabel('Tiempo [h]'); ax8.grid(True); ax8.legend(loc='best'); add_phase_lines(ax8,t_fase1,t_fase2_end); valid_otr=OTR_sim[np.isfinite(OTR_sim)]; valid_our=OUR_sim[np.isfinite(OUR_sim)]; otr_our_min=min(0,np.min(valid_otr)*1.1 if len(valid_otr)>0 else 0,np.min(valid_our)*1.1 if len(valid_our)>0 else 0); otr_our_max=max(0.1,np.max(valid_otr)*1.1 if len(valid_otr)>0 else 0.1,np.max(valid_our)*1.1 if len(valid_our)>0 else 0.1); ax8.set_ylim(bottom=otr_our_min,top=otr_our_max)
               ax9=plt.subplot(5,2,10); qS_nozero=np.where(np.abs(qS_sim)>1e-7,qS_sim,1e-7); mu_bruto_sim=mu_sim; Yps_inst=np.maximum(0,qP_sim/qS_nozero); Yxs_inst=np.maximum(0,mu_bruto_sim/qS_nozero); Yps_inst=np.clip(Yps_inst,0,1.0); Yxs_inst=np.clip(Yxs_inst,0,1.0); ax9.plot(t_sim,Yps_inst,'k-',label=r'$Y_{P/S}$'); ax9.plot(t_sim,Yxs_inst,'g--',label=r'$Y_{X/S}$'); ax9.set_title('Rendimientos Instantáneos'); ax9.set_ylabel('[g/g]'); ax9.set_xlabel('Tiempo [h]'); ax9.grid(True); ax9.legend(loc='best'); add_phase_lines(ax9,t_fase1,t_fase2_end); valid_yps=Yps_inst[np.isfinite(Yps_inst)]; valid_yxs=Yxs_inst[np.isfinite(Yxs_inst)]; y_top=max(0.6,np.max(valid_yps)*1.1 if len(valid_yps)>0 else 0.6,np.max(valid_yxs)*1.1 if len(valid_yxs)>0 else 0.6); ax9.set_ylim(bottom=0,top=min(y_top,1.0))
               plt.tight_layout(pad=2.0, rect=[0,0.03,1,1]); st.pyplot(fig)

               st.subheader("📈 Métricas Finales (Simulación)")
               # ... (código de cálculo de métricas idéntico) ...
               col_m1,col_m2,col_m3=st.columns(3); vol_final_sim=V_sim[-1]; et_final_conc=P_sim[-1]; bio_final_conc=X_sim[-1]; S_ini_tot=cond_iniciales['S0']*cond_iniciales['V0']; S_alim_tot=np.trapz(f_sim*all_params['Sin'], t_sim) if len(t_sim)>1 else 0; S_fin_tot=S_sim[-1]*vol_final_sim; S_cons_tot=max(1e-9, S_ini_tot+S_alim_tot-S_fin_tot); P_ini_tot=cond_iniciales['P0']*cond_iniciales['V0']; et_fin_tot=et_final_conc*vol_final_sim; et_prod_tot=max(0, et_fin_tot-P_ini_tot); col_m1.metric("Vol Final [L]", f"{vol_final_sim:.3f}"); col_m2.metric("Etanol Fin [g/L]", f"{et_final_conc:.3f}"); col_m3.metric("Biomasa Fin [g/L]", f"{bio_final_conc:.3f}"); prod_vol_et=et_prod_tot/vol_final_sim/t_fase3_end if t_fase3_end>0 and vol_final_sim>1e-6 else 0; col_m1.metric("Prod Vol Etanol [g/L/h]", f"{prod_vol_et:.4f}"); rend_glob_et=et_prod_tot/S_cons_tot; col_m2.metric("Rend Global P/S [g/g]", f"{rend_glob_et:.4f}");
               try: X_V_int=np.trapz(X_sim*V_sim,t_sim) if len(t_sim)>1 else 0; prod_esp_media=et_prod_tot/X_V_int if X_V_int>1e-9 and t_fase3_end>0 else 0; col_m3.metric("Prod Esp Media [gP/gX/h]", f"{prod_esp_media:.5f}" if prod_esp_media>0 else "N/A")
               except Exception as me: col_m3.metric("Prod Esp Media [gP/gX/h]", "Error")
               try: p_max_idx=np.argmax(P_sim); col_m1.metric("Etanol Máx [g/L]", f"{P_sim[p_max_idx]:.3f} (t={t_sim[p_max_idx]:.1f} h)")
               except ValueError: col_m1.metric("Etanol Máx [g/L]", "N/A")
               col_m2.metric("Sustrato Res [g/L]", f"{S_sim[-1]:.3f}")
               """

               # Importar streamlit se ainda não foi feito no início do seu script
               # import streamlit as st
               # from matplotlib.lines import Line2D # Não será mais necessário se removermos a legenda manual

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
               st.subheader("📊 Gráficos Detalhados do RTO") # Traduzido

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
               ax1.set_ylabel('Vazão [L/h]', color=color) # Traduzido
               ax1.tick_params(axis='y', labelcolor=color)
               ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t_sim, V_sim, color=color, ls='-') # label removido
               ax1b.set_ylabel('Volume [L]', color=color) # Traduzido
               ax1b.tick_params(axis='y', labelcolor=color)
               ax1.set_xlabel('Tempo [h]') # Traduzido
               ax1.grid(True)
               ax1.set_title('Alimentação Otimizada e Volume') # Traduzido
               add_phase_lines(ax1, t_fase1, t_fase2_end) # Chamada mantida
               # Linhas de criação de legenda manual (leg_el_phases) e ax1b.legend() removidas
               f_min_plot = -0.05 * all_params.get('Fmax', 0) # Usa .get para segurança
               ax1.set_ylim(bottom=f_min_plot)

               # 2. Biomassa (X) (Posição 2: Linha 1, Coluna 2)
               ax2 = plt.subplot(2, 3, 2)
               ax2.plot(t_sim, X_sim, 'g-')
               ax2.set_title('Biomassa (X)') # Traduzido
               ax2.set_ylabel('[g/L]')
               ax2.set_xlabel('Tempo [h]') # Traduzido
               ax2.grid(True); ax2.set_ylim(bottom=0)
               add_phase_lines(ax2, t_fase1, t_fase2_end)

               # 3. Substrato (S) (Posição 3: Linha 1, Coluna 3)
               ax3 = plt.subplot(2, 3, 3)
               ax3.plot(t_sim, S_sim, 'm-')
               ax3.set_title('Substrato (S)') # Traduzido
               ax3.set_ylabel('[g/L]')
               ax3.set_xlabel('Tempo [h]') # Traduzido
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
               ax4.set_title('Etanol (P)') # Título já estava ok
               ax4.set_ylabel('[g/L]')
               ax4.set_xlabel('Tempo [h]') # Traduzido
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
               ax5.set_title(r'$O_2$ e Fator Inibição $P$') # Traduzido (usando LaTeX)
               ax5.set_ylabel(r'$O_2$ [mg/L]', color='c') # Usando LaTeX
               ax5.set_xlabel('Tempo [h]') # Traduzido
               ax5.grid(True)
               o2_top = max(all_params.get('Cs',0)*1000*1.1, np.max(O2_sim_mgL)*1.1 if len(O2_sim_mgL[np.isfinite(O2_sim_mgL)])>0 else all_params.get('Cs',0)*1000*1.1)
               o2_top = max(o2_top, 0.1)
               ax5.set_ylim(bottom=0.0, top=o2_top)
               ax5.tick_params(axis='y', labelcolor='c')
               ax5b = ax5.twinx()
               ax5b.plot(t_sim, inhib_factor_sim, color='darkorange', ls='--') # label removido
               ax5b.set_ylabel('Fator Inib. Prod. [-]', color='darkorange') # Traduzido
               ax5b.tick_params(axis='y', labelcolor='darkorange')
               ax5b.set_ylim(bottom=-0.05, top=1.05)
               # ax5b.legend() removido
               add_phase_lines(ax5, t_fase1, t_fase2_end)

               # 6. Taxa Específica de Crescimento (mu) (Posição 6: Linha 2, Coluna 3)
               ax6 = plt.subplot(2, 3, 6)
               ax6.plot(t_sim, mu_sim, 'y-')
               ax6.set_title(r'Taxa Específica de Crescimento ($\mu$)') # Traduzido e LaTeX
               ax6.set_ylabel('[1/h]')
               ax6.set_xlabel('Tempo [h]') # Traduzido
               ax6.grid(True); ax6.set_ylim(bottom=0)
               add_phase_lines(ax6, t_fase1, t_fase2_end)


               # Ajusta o layout
               plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 1]) # Mantido rect original se necessário
               st.pyplot(fig)

               # --- Métricas Finais RTO (Tradução PT-BR) ---
               st.subheader("📈 Métricas Finais (Simulação RTO)") # Traduzido

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

               col_m1.metric("Vol Final [L]", f"{vol_final_sim:.3f}") # Traduzido
               col_m2.metric("Etanol Fin [g/L]", f"{et_final_conc:.3f}") # Traduzido
               col_m3.metric("Biomassa Fin [g/L]", f"{bio_final_conc:.3f}") # Traduzido

               # Usa t_fase3_end se definido e maior que 0, senão usa t_sim[-1]
               tempo_final_prod = t_fase3_end if 't_fase3_end' in locals() and t_fase3_end > 0 else (t_sim[-1] if len(t_sim)>0 else 0)

               prod_vol_et = et_prod_tot / vol_final_sim / tempo_final_prod if tempo_final_prod > 0 and vol_final_sim > 1e-6 else 0
               col_m1.metric("Prod Vol Etanol [g/L/h]", f"{prod_vol_et:.4f}") # Traduzido

               rend_glob_et = et_prod_tot / S_cons_tot if S_cons_tot > 1e-9 else 0 # Evita divisão por zero
               col_m2.metric("Rend Global P/S [g/g]", f"{rend_glob_et:.4f}") # Traduzido

               try:
                    # Checa se t_sim e X_sim/V_sim têm pontos suficientes para integrar
                    if len(t_sim) > 1 and len(X_sim) == len(t_sim) and len(V_sim) == len(t_sim):
                         X_V_int = np.trapz(X_sim * V_sim, t_sim)
                    else: X_V_int = 0
                    prod_esp_media = et_prod_tot / X_V_int if X_V_int > 1e-9 and tempo_final_prod > 0 else 0
                    col_m3.metric("Prod Esp Média [gP/gX/h]", f"{prod_esp_media:.5f}" if prod_esp_media > 0 else "N/A") # Traduzido
               except Exception as me:
                    print(f"Erro cálculo Prod Esp Média: {me}")
                    col_m3.metric("Prod Esp Média [gP/gX/h]", "Erro") # Traduzido

               try:
                    if len(P_sim) > 0: # Checa se P_sim não está vazio
                         p_max_idx = np.argmax(P_sim)
                         # Checa se t_sim correspondente existe
                         tempo_p_max = t_sim[p_max_idx] if p_max_idx < len(t_sim) else np.nan
                         col_m1.metric("Etanol Máx [g/L]", f"{P_sim[p_max_idx]:.3f} (t={tempo_p_max:.1f} h)") # Traduzido
                    else:
                         col_m1.metric("Etanol Máx [g/L]", "N/A") # Traduzido
               except (ValueError, IndexError) as e:
                    print(f"Erro cálculo Etanol Máx: {e}")
                    col_m1.metric("Etanol Máx [g/L]", "N/A") # Traduzido

               col_m2.metric("Substrato Res [g/L]", f"{S_final:.3f}") # Traduzido


          else: st.error("Error en simulación detallada.")

# --- Ejecución ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="RTO Fermentación Detallada")
    try:
        rto_fermentation_page()
    except Exception as main_e:
        st.error(f"Error inesperado: {main_e}")
        st.error(traceback.format_exc())