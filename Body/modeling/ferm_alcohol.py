# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import traceback # Para mostrar errores detallados
import time # Para medir tiempos

#libraries for saving the data
import pandas as pd
from io import BytesIO
import xlsxwriter

# --- INICIO: Definiciones dummy de funciones cinéticas (si no tienes Utils.kinetics) ---
# Reemplaza esto con tus importaciones reales si están disponibles
def mu_monod(S, mumax, Ks):
    # Asegurar S no negativo y evitar división por cero
    S = max(0.0, S)
    Ks = max(1e-9, Ks)
    # Evitar división por cero en el denominador
    den = Ks + S
    if den < 1e-9: return 0.0
    return mumax * S / den

def mu_sigmoidal(S, mumax, Ks, n):
    S = max(0.0, S)
    Ks = max(1e-9, Ks)
    n = max(1e-6, n) # Evitar n=0
    try:
        S_n = S**n
        Ks_n = Ks**n
        den = Ks_n + S_n
        if den < 1e-9: return 0.0
        return mumax * S_n / den
    except OverflowError: # Manejar posible overflow con n grande
        return mumax if S > Ks else 0.0


def mu_completa(S, O2, P, mumax, Ks, KO, KP_gen):
    S = max(0.0, S)
    O2_mgL = max(0.0, O2) # O2 en mg/L
    P = max(0.0, P)
    Ks = max(1e-9, Ks)
    KO_mgL = max(1e-9, KO) # KO en mg/L
    KP_gen = max(1e-9, KP_gen)

    term_S = S / (Ks + S)
    term_O2 = O2_mgL / (KO_mgL + O2_mgL) # KO y O2 ambos en mg/L
    term_P = KP_gen / (KP_gen + P) # Asumiendo inhibición tipo no competitivo

    mu = mumax * term_S * term_O2 * term_P
    return max(0.0, mu)


def mu_fermentacion(S, P, O2, mumax_aerob, Ks_aerob, KO_aerob, mumax_anaerob, Ks_anaerob, KiS_anaerob, KP_anaerob, n_p, KO_inhib_anaerob, considerar_O2=None):
    # Asegurar valores no negativos y evitar división por cero
    S = max(0.0, S)
    P = max(0.0, P)
    O2_mgL = max(0.0, O2) # O2 en mg/L

    Ks_aerob = max(1e-9, Ks_aerob)
    KO_aerob_mgL = max(1e-9, KO_aerob) # KO_aerob en mg/L
    Ks_anaerob = max(1e-9, Ks_anaerob)
    KiS_anaerob = max(1e-9, KiS_anaerob)
    KP_anaerob = max(1e-9, KP_anaerob)
    n_p = max(1e-6, n_p)
    KO_inhib_anaerob_mgL = max(1e-9, KO_inhib_anaerob) # KO_inhib_anaerob en mg/L

    # --- Crecimiento Aerobio ---
    den_O2_aer = KO_aerob_mgL + O2_mgL
    term_O2_aer = O2_mgL / den_O2_aer if den_O2_aer > 1e-9 else 0.0
    mu_aer = mumax_aerob * (S / (Ks_aerob + S)) * term_O2_aer
    mu_aer = max(0.0, mu_aer)

    # --- Crecimiento Anaerobio ---
    den_S_an = Ks_anaerob + S + (S**2 / KiS_anaerob)
    term_S_an = S / max(1e-9, den_S_an)

    # Asegurar que la base de la potencia no sea negativa
    term_P_base = max(0.0, 1.0 - (P / KP_anaerob))
    term_P_an = term_P_base**n_p

    den_O2_inhib = KO_inhib_anaerob_mgL + O2_mgL
    term_O2_inhib_an = KO_inhib_anaerob_mgL / den_O2_inhib if den_O2_inhib > 1e-9 else 0.0

    mu_anaer = mumax_anaerob * term_S_an * term_P_an * term_O2_inhib_an
    mu_anaer = max(0.0, mu_anaer)

    # --- Combinación de tasas (dependiendo del contexto de llamada) ---
    if considerar_O2 is True: # Llamado desde Conmutada - Fase Aerobia
        return mu_aer
    elif considerar_O2 is False: # Llamado desde Conmutada - Fase Anaerobia
        return mu_anaer
    else: # Llamado desde modelo "Fermentación" (mixto)
        return mu_aer + mu_anaer
# --- FIN: Definiciones dummy ---


def fermentacion_alcoholica_page():
    st.header("Fed-Batch Alcoholic Fermentation Simulation")
    st.markdown("""
    This model simulates an alcoholic fermenttion that begins in **batch (aerobic phase)**,
    continues in **fed-batch (transition/anaerobic phase)**, and ends in
    **batch (anaerbic phase)** to deplete the substrate. Select the kinetic model.
    
    """)

    with st.sidebar:
        st.subheader("1. Kinetic Model and Parameters")
        tipo_mu = st.selectbox("Kinetic Model", ["Fermentation", "Switched Fermentation", "Simple Monod", "Sigmoidal Monod", "Monod with restrictions"])

        Ks = st.slider("Base Ks [g/L]", 0.01, 10.0, 1.0, 0.1)

        # --- Parámetros específicos de cada modelo cinético ---
        if tipo_mu == "Simple Monod":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
        elif tipo_mu == "Sigmoidal Monod":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
            n_sig = st.slider("Sigmoidal power (n)", 1, 5, 2)
        elif tipo_mu == "Monod with restrictions":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
            KO_restr = st.slider("KO (O2 - restriction) [mg/L]", 0.01, 5.0, 0.1, 0.01)
            KP_gen = st.slider("KP (Generic product inhibition) [g/L]", 1.0, 100.0, 50.0)
        elif tipo_mu == "Switched Fermentation":
            st.info("Switched model: Uses mu_fermentation with mumax_aero/anaero and optional O2.")
            mumax_aero_c = st.slider("μmax (Aerobic Phase) [1/h]", 0.1, 1.0, 0.45, 0.05, key="mumax_aero_c")
            mumax_anaero_c = st.slider("μmax (Anaerobic Phase) [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaero_c")
            KiS_c = st.slider("KiS (Substrate Inhibition) [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_c")
            KP_c = st.slider("KP (Ethanol Inhibition) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_c")
            n_p_c = st.slider("Ethanol Inhib. Exponent (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_c")
            KO_ferm_c = st.slider("KO (O2 - aerobic affinity) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_ferm_c")
            KO_inhib_anaerob_c = st.slider("KO_inhib_anaerob (O2 inhibition on anaerobic μ) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_inhib_c") # Añadido para consistencia
        elif tipo_mu == "Fermentation":
            st.info("Mixed Model: mu = mu1(aerobic) + mu2(anaerobic).")
            st.markdown("**mu1 parameters (Aerobic):**")
            mumax_aerob_m = st.slider("μmax_aerob [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_aerob_m")
            Ks_aerob_m = st.slider("Ks_aerob [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aerob_m")
            KO_aerob_m = st.slider("KO_aerob (O2 affinity) [mg/L]", 0.01, 5.0, 0.2, 0.01, key="ko_aerob_m")
            st.markdown("**mu2 parameters (Anaerobic/Fermentative):**")
            mumax_anaerob_m = st.slider("μmax_anaerob [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaerob_m")
            Ks_anaerob_m = st.slider("Ks_anaerob [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaerob_m")
            KiS_anaerob_m = st.slider("KiS_anaerob [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaerob_m")
            KP_anaerob_m = st.slider("KP_anaerob (Inhib. Etanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaerob_m")
            n_p_m = st.slider("Ethanol Inhib. Exponent (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_m")
            KO_inhib_anaerob_m = st.slider("KO_inhib_anaerob (Inhib. O2) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_inhib_m")

        # --- Resto de parámetros ---
        st.subheader("2. Stoichiometric and Maintenance Parameters")
        Yxs = st.slider("Yxs (Biomass/Substrate) [g/g]", 0.05, 0.6, 0.1, 0.01, key="yxs")
        Yps = st.slider("Yps (Ethanol/Substrate) [g/g]", 0.1, 0.51, 0.45, 0.01, key="yps")
        Yxo = st.slider("Yxo (Biomass/O2) [gX/gO2]", 0.1, 2.0, 0.8, 0.1, key="yxo")
        alpha_lp = st.slider("α (Associated to growth) [g P / g X]", 0.0, 10.0, 4.5, 0.1, key="alpha")
        beta_lp = st.slider("β (Not associated to growth) [g P / g X / h]", 0.0, 1.5, 0.40, 0.01, key="beta")
        ms = st.slider("ms (Substrate Maintenance) [g S / g X / h]", 0.0, 0.2, 0.02, 0.01, key="ms")
        mo = st.slider("mo (O2 Maintenance) [gO2/gX/h]", 0.0, 0.1, 0.01, 0.005, key="mo")
        Kd = st.slider("Kd (Biomass Decay) [1/h]", 0.0, 0.1, 0.01, 0.005, key="kd")
        KO_inhib_prod = st.slider("KO_inhib_prod (O2 Inhib. in Ethanol Prod.) [mg/L]", 0.001, 1.0, 0.05, 0.005, key="ko_inhib_p", help="No longer used in rate_P. O2 concentration that reduces ethanol production by half.")

        st.subheader("3. Oxygen Transfer")
        Kla = st.slider("kLa [1/h]", 10.0, 400.0, 100.0, 10.0, key="kla")
        Cs = st.slider("Saturated O2 (Cs) [mg/L]", 0.01, 15.0, 0.09, 0.01, key="cs")

        st.subheader("4. Feeding and Operation Phases")
        t_batch_inicial_fin = st.slider("End Initial Batch Phase [h]", 1.0, 30.0, 4.0, 1.0, key="t_batch_fin")
        t_alim_inicio = st.slider("Start Feeding [h]", t_batch_inicial_fin, t_batch_inicial_fin + 24.0, t_batch_inicial_fin + 0.01, 0.5, key="t_alim_ini")
        t_alim_fin = st.slider("End Feeding (Start Final Batch) [h]", t_alim_inicio + 1.0, 40.0, t_alim_inicio + 5.0, 1.0, key="t_alim_fin")
        t_final = st.slider("Total Simulation Time [h]", t_alim_fin + 1.0, 100.0, t_alim_fin + 7.0, 1.0, key="t_total")
        O2_controlado = st.slider("O2 Objetive/Ref Level (Initial Batch Phase) [mg/L]", 0.01, Cs, 0.08, 0.01, key="o2_control")

        estrategia = st.selectbox("Feeding Strategy", ["Linear", "Exponential","Constant", "Step"], key="strat")
        Sin = st.slider("Substrate in Feeding (Sin) [g/L]", 10.0, 700.0, 250.0, 10.0, key="sin")
        F_base = st.slider("Base Flow (or Initial) [L/h]", 0.01, 5.0, 0.01, 0.01, key="fbase")
        if estrategia == "Linear":
            F_lineal_fin = st.slider("Final Flow (Lineat) [L/h]", F_base, 10.0, F_base * 11, 0.01, key="ffin_lin")
        elif estrategia == "Exponential":
            k_exp = st.slider("Constant Growth Exp. (k_exp) [1/h]", 0.01, 0.5, 0.1, 0.01, key="kexp")

        st.subheader("5. Initial Conditions")
        V0 = st.number_input("Initial Volume [L]", 0.1, 100.0, 0.25, key="v0")
        X0 = st.number_input("Initial Biomass [g/L]", 0.05, 10.0, 1.20, key="x0")
        S0 = st.number_input("Initial Substrate [g/L]", 10.0, 200.0, 20.0, key="s0")
        P0 = st.number_input("Initial Ethanol [g/L]", 0.0, 50.0, 0.0, key="p0")
        O0 = st.number_input("Initial O2 [mg/L]", min_value=0.0, max_value=Cs, value=0.08, step=0.01, key="o0")

        st.subheader("6. Solver Parameters")
        atol = st.number_input("Absolute Tolerance (atol)", min_value=1e-9, max_value=1e-3, value=1e-6, format="%e", key="atol")
        rtol = st.number_input("Relative Tolerance (rtol)", min_value=1e-9, max_value=1e-3, value=1e-6, format="%e", key="rtol")

    # ------- Funciones de Cálculo Auxiliares -------
    F_lineal_fin_val = F_base * 2
    k_exp_val = 0.1
    if estrategia == "Linear": F_lineal_fin_val = F_lineal_fin
    elif estrategia == "Exponential": k_exp_val = k_exp

    def calcular_flujo(t):
        # (Sin cambios aquí)
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constant": return F_base
            elif estrategia == "Exponential":
                try: return min(F_base * np.exp(k_exp_val * (t - t_alim_inicio)), F_base * 100) # Limitar flujo exp
                except OverflowError: return F_base * 100
            elif estrategia == "Step":
                t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2
                return F_base * 2 if t > t_medio else F_base
            elif estrategia == "Linear":
                delta_t = t_alim_fin - t_alim_inicio
                if delta_t > 1e-6:
                    slope = (F_lineal_fin_val - F_base) / delta_t
                    return max(0, F_base + slope * (t - t_alim_inicio))
                else: return F_base
        return 0.0

    # Empaquetar parámetros para pasar al ODE
    params = {
        "tipo_mu": tipo_mu, "Ks": Ks, "Yxs": Yxs, "Yps": Yps, "Yxo": Yxo,
        "alpha_lp": alpha_lp, "beta_lp": beta_lp, "ms": ms, "mo": mo, "Kd": Kd,
        "Kla": Kla, "Cs": Cs, "Sin": Sin,
        "t_batch_inicial_fin": t_batch_inicial_fin, # Tiempo crítico
        "KO_inhib_prod": KO_inhib_prod, # Se mantiene por si se usa en otro lado, pero no en rate_P
    }
    if tipo_mu == "Simple Monod": params["mumax"] = mumax
    elif tipo_mu == "Sigmoidal Monod": params["mumax"] = mumax; params["n_sig"] = n_sig
    elif tipo_mu == "Monod with Restrictions": params["mumax"] = mumax; params["KO"] = KO_restr; params["KP_gen"] = KP_gen
    elif tipo_mu == "Switched Fermentation":
        params["mumax_aero"] = mumax_aero_c; params["mumax_anaero"] = mumax_anaero_c
        params["Ks_aerob"] = Ks # Usar Ks base como Ks_aerob
        params["KO_aerob"] = KO_ferm_c # Usar KO_ferm_c como KO_aerob (mg/L)
        params["Ks_anaerob"] = Ks # Usar Ks base como Ks_anaerob
        params["KiS_anaerob"] = KiS_c # Usar KiS_c como KiS_anaerob
        params["KP_anaerob"] = KP_c # Usar KP_c como KP_anaerob
        params["n_p"] = n_p_c
        params["KO_inhib_anaerob"] = KO_inhib_anaerob_c # Usar el nuevo slider (mg/L)
        params["O2_controlado"] = O2_controlado
    elif tipo_mu == "Fermentation":
        params["mumax_aerob"] = mumax_aerob_m; params["Ks_aerob"] = Ks_aerob_m; params["KO_aerob"] = KO_aerob_m
        params["mumax_anaerob"] = mumax_anaerob_m; params["Ks_anaerob"] = Ks_anaerob_m; params["KiS_anaerob"] = KiS_anaerob_m
        params["KP_anaerob"] = KP_anaerob_m; params["n_p"] = n_p_m; params["KO_inhib_anaerob"] = KO_inhib_anaerob_m

    # ------- Modelo ODE -------
    def modelo_fermentacion(t, y, params):
        X, S, P, O2, V = y
        # Asegurar valores no negativos y evitar división por cero
        X = max(1e-9, X); S = max(0, S); P = max(0, P); O2 = max(0, O2); V = max(1e-6, V)

        es_lote_inicial = (t < params["t_batch_inicial_fin"])

        # --- Cálculo de mu (total), mu_aer y mu_anaer ---
        mu = 0.0
        mu_aer = 0.0
        mu_anaer = 0.0

        if params["tipo_mu"] == "Fermentation":
            # Calcular ambos componentes por separado
            mu_aer = mu_fermentacion(S, P, O2, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], 0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
            mu_anaer = mu_fermentacion(S, P, O2, 0, 1, float('inf'), params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
            mu = mu_aer + mu_anaer # mu total es la suma
        elif params["tipo_mu"] == "Switched Fermentation":
            if es_lote_inicial:
                current_O2_for_mu = params.get("O2_controlado", O2)
                mu_aer = mu_fermentacion(S, P, current_O2_for_mu, params["mumax_aero"], params["Ks_aerob"], params["KO_aerob"], 0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                mu = mu_aer # En fase 1, mu_anaer es 0
            else:
                mu_anaer = mu_fermentacion(S, P, O2, 0, 1, float('inf'), params["mumax_anaero"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
                mu = mu_anaer # En fase 2/3, mu_aer es 0
        elif params["tipo_mu"] == "Simple Monod":
             mu = mu_monod(S, params.get("mumax", 0.0), params["Ks"])
             # Asumir que todo el crecimiento es aerobio para el balance de S
             mu_aer = mu
        elif params["tipo_mu"] == "Sigmoidal Monod":
             mu = mu_sigmoidal(S, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
             # Asumir que todo el crecimiento es aerobio para el balance de S
             mu_aer = mu
        elif params["tipo_mu"] == "Monod with Restrictions":
             mu = mu_completa(S, O2, P, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))
             # Asumir que todo el crecimiento es aerobio para el balance de S
             mu_aer = mu

        mu = max(0, mu)
        mu_aer = max(0, mu_aer)
        mu_anaer = max(0, mu_anaer)
        mu_net = mu - params["Kd"] # Tasa neta total

        # --- Cálculo de Flujo y Tasas de Consumo/Producción ---
        F = calcular_flujo(t)

        # <<< CORRECCIÓN qP >>>
        # Tasa específica de producción de Etanol (Luedeking-Piret)
        # Usar SOLO mu_anaer para el término asociado a crecimiento
        qP = params["alpha_lp"] * mu_anaer + params["beta_lp"]
        qP = max(0.0, qP) # Asegurar no negatividad

        # Tasa volumétrica de producción de Etanol (SIN inhibición por O2)
        rate_P = qP * X

        # <<< CORRECCIÓN dSdt >>>
        # Consumo de Sustrato para crecimiento: usar SOLO mu_aer
        consumo_S_X = (mu_aer / params["Yxs"]) * X if params["Yxs"] > 1e-6 else 0
        # Consumo de Sustrato para producto (usa qP calculado con mu_anaer)
        consumo_S_P = (qP / params["Yps"]) * X if params["Yps"] > 1e-6 else 0
        consumo_S_maint = params["ms"] * X
        rate_S = consumo_S_X + consumo_S_P + consumo_S_maint

        # Consumos de Oxígeno (usa mu_aer para crecimiento)
        consumo_O2_X = (mu_aer / params["Yxo"]) * X if params["Yxo"] > 1e-6 else 0
        consumo_O2_maint = params["mo"] * X
        OUR_g = consumo_O2_X + consumo_O2_maint # [g O2 / L / h]
        OUR_mg = OUR_g * 1000.0 # Convertir a [mg O2 / L / h]

        # Ecuaciones Diferenciales
        dXdt = mu_net * X - (F / V) * X # Usa mu_net total
        dSdt = -rate_S + (F / V) * (params["Sin"] - S) # rate_S ahora usa mu_aer para crecimiento
        dPdt = rate_P - (F / V) * P # rate_P ahora usa mu_anaer para crecimiento
        dVdt = F

        # --- Cálculo de dOdt (sin cambios en la lógica de fase) ---
        if es_lote_inicial:
            dOdt = 0.0 # Mantiene O2 constante en la fase inicial
        else:
            OTR = params["Kla"] * (params["Cs"] - O2) # [mg O2 / L / h]
            # <<< CORRECCIÓN SYNTAX ERROR (anterior) >>>
            dOdt = OTR - OUR_mg - (F / V) * O2 # [mg/L/h]

        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    # ------- Simulación y Resultados -------
    y0 = [X0, S0, P0, O0, V0]
    t_span = [0, t_final]
    num_puntos = max(500, int(t_final * 25) + 1)
    t_eval = np.linspace(t_span[0], t_span[1], num_puntos)

    try:
        # --- Bloque solve_ivp (sin cambios) ---
        sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval, method='RK45', atol=atol, rtol=rtol, args=(params,), max_step=0.5)
        if not sol.success:
            st.warning("RK45 failed, trying with BDF...")
            sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval, method='BDF', atol=atol, rtol=rtol, args=(params,))
            if not sol.success:
                st.error(f"Integration failed with both methods: {sol.message}")
                st.stop()
            else: st.success("Integration completed with BDF.")

        t = sol.t
        X, S, P, O2, V = sol.y
        # Corregir posibles valores negativos muy pequeños por errores numéricos
        X = np.maximum(X, 0); S = np.maximum(S, 0); P = np.maximum(P, 0); O2 = np.maximum(O2, 0); V = np.maximum(V, 1e-6)

        flujo_sim = np.array([calcular_flujo(ti) for ti in t])

        # --- Recalcular Tasas Específicas y Volumétricas Post-Simulación ---
        mu_sim = []
        mu_aer_sim = [] # Guardar mu_aer para recálculo
        mu_anaer_sim = [] # Guardar mu_anaer para recálculo
        qP_sim = []
        qS_sim = []
        qO_sim = []
        OTR_sim = []
        # inhib_factor_sim = [] # Ya no se usa para rate_P

        for i in range(len(t)):
            # Recalcular mu usando los valores simulados
            xi, si, pi, o2i, vi = X[i], S[i], P[i], O2[i], V[i]
            ti = t[i]
            es_lote_inicial_i = (ti < params["t_batch_inicial_fin"])

            # Re-calcular mu, mu_aer, mu_anaer exactamente como dentro del solver
            mu_i = 0.0
            mu_aer_i = 0.0
            mu_anaer_i = 0.0
            if params["tipo_mu"] == "Fermentation":
                mu_aer_i = mu_fermentacion(si, pi, o2i, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], 0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                mu_anaer_i = mu_fermentacion(si, pi, o2i, 0, 1, float('inf'), params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
                mu_i = mu_aer_i + mu_anaer_i
            elif params["tipo_mu"] == "Switched Fermentation":
                if es_lote_inicial_i:
                    current_O2_for_mu_i = params.get("O2_controlado", o2i)
                    mu_aer_i = mu_fermentacion(si, pi, current_O2_for_mu_i, params["mumax_aero"], params["Ks_aerob"], params["KO_aerob"], 0, 1, float('inf'), float('inf'), 1, float('inf'), considerar_O2=True)
                    mu_i = mu_aer_i
                else:
                    mu_anaer_i = mu_fermentacion(si, pi, o2i, 0, 1, float('inf'), params["mumax_anaero"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"], considerar_O2=False)
                    mu_i = mu_anaer_i
            elif params["tipo_mu"] == "Simple Monod":
                 mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
                 mu_aer_i = mu_i
            elif params["tipo_mu"] == "Sigmoidal Monod":
                 mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
                 mu_aer_i = mu_i
            elif params["tipo_mu"] == "Monod with Restrictions":
                 mu_i = mu_completa(si, o2i, pi, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))
                 mu_aer_i = mu_i

            mu_i = max(0, mu_i)
            mu_aer_i = max(0, mu_aer_i)
            mu_anaer_i = max(0, mu_anaer_i)
            mu_sim.append(mu_i)
            mu_aer_sim.append(mu_aer_i)
            mu_anaer_sim.append(mu_anaer_i)

            # Recalcular qP, qS, qO, OTR usando la misma lógica que el solver
            # <<< CORRECCIÓN qP >>>
            qP_i = params["alpha_lp"] * mu_anaer_i + params["beta_lp"] # Usa mu_anaer_i
            qP_i = max(0.0, qP_i)
            qP_sim.append(qP_i)

            # <<< CORRECCIÓN dSdt >>>
            consumo_S_X_i = (mu_aer_i / params["Yxs"]) if params["Yxs"] > 1e-6 else 0 # Usa mu_aer_i
            consumo_S_P_i = (qP_i / params["Yps"]) if params["Yps"] > 1e-6 else 0
            qS_i = consumo_S_X_i + consumo_S_P_i + params["ms"]
            qS_sim.append(qS_i)

            # qO usa mu_aer_i
            consumo_O2_X_i = (mu_aer_i / params["Yxo"]) if params["Yxo"] > 1e-6 else 0
            qO_i = consumo_O2_X_i + params["mo"] # gO2/gX/h
            qO_sim.append(qO_i)

            OTR_i = params["Kla"] * (params["Cs"] - o2i) # mg/L/h
            OTR_sim.append(OTR_i)
            # inhib_factor_sim ya no es necesario

        # Convertir listas a arrays numpy
        mu_sim = np.array(mu_sim)
        qP_sim = np.array(qP_sim)
        qS_sim = np.array(qS_sim)
        qO_sim = np.array(qO_sim) * 1000 # Convertir a mgO2/gX/h
        OTR_sim = np.array(OTR_sim)
        OUR_sim = qO_sim * X # mgO2/L/h
        #inhib_factor_sim = np.array(inhib_factor_sim)


         # Importar streamlit se ainda não foi feito no início do seu script
        # import streamlit as st

        # --- Configurações Globais de Fonte (Times New Roman, Tamanho Maior) ---
        # Defina o tamanho da fonte desejado (ex: 12, 14)

        NEW_FONT_SIZE = 22
        plt.rcParams.update({
            'font.size': NEW_FONT_SIZE,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.titlesize': NEW_FONT_SIZE,
            'axes.labelsize': NEW_FONT_SIZE,
            'xtick.labelsize': NEW_FONT_SIZE * 0.9, # Ligeiramente menor para ticks
            'ytick.labelsize': NEW_FONT_SIZE * 0.9,
            'legend.fontsize': NEW_FONT_SIZE * 0.9 # Mantido caso precise no futuro
        })

        # --- Supondo que as variáveis e funções abaixo JÁ EXISTEM ---
        # t, flujo_sim, V, X, S, P, O2
        # t_batch_inicial_fin, t_alim_inicio, t_alim_fin, Cs, Sin, V0, S0, P0, X0, O0
        # params (dicionário com parâmetros)
        # Funções: mu_fermentacion, mu_monod, mu_sigmoidal, mu_completa

        # --- Graficação (PT-BR, Layout 2x3, Fonte Maior, Sem Legenda) ---
        # Usa st.subheader do código original, traduzido
        st.subheader("Simulation results")

        # Ajuste o figsize para o layout 2x3 (largura, altura) - pode precisar ajustar
        fig = plt.figure(figsize=(15, 10)) # Ex: 15 de largura, 10 de altura

        # --- Função auxiliar para adicionar linhas de fase (sem labels para legenda) ---
        # (Função original, apenas removendo os labels condicionais)
        def add_phase_lines(ax, t_batch, t_feed_start, t_feed_end):
            ax.axvline(t_batch, color='gray', linestyle='--', lw=1.5) # Label removido
            ax.axvline(t_feed_start, color='orange', linestyle='--', lw=1.5) # Label removido
            ax.axvline(t_feed_end, color='purple', linestyle='--', lw=1.5) # Label removido

        # --- Gráficos no Layout 2x3 ---

        # 1. Vazão e Volume (Posição 1: Linha 1, Coluna 1)
        # Usa plt.subplot(2, 3, 1) para o novo layout
        ax1 = plt.subplot(2, 3, 1) # Label 'ax1' removido, não é mais necessário para a legenda
        color = 'tab:red'; ax1.plot(t, flujo_sim, color=color) # Label 'Vazão de Alimentação' removido
        ax1.set_ylabel('Flow [L/h]', color=color) # Traduzido de 'Flujo'
        ax1.tick_params(axis='y', labelcolor=color)
        ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t, V, color=color, linestyle='-') # Label 'Volumen' removido
        ax1b.set_ylabel('Volume [L]', color=color) # Traduzido
        ax1b.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Time [h]') # Traduzido
        ax1.grid(True)
        ax1.set_title('Feeding and Volume') # Traduzido
        add_phase_lines(ax1, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)
        # Código da legenda combinada removido

        # 2. Biomassa (X) (Posição 2: Linha 1, Coluna 2)
        # Usa plt.subplot(2, 3, 2)
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(t, X, 'g-')
        ax2.set_title('Biomass (X)') # Traduzido
        ax2.set_ylabel('[g/L]')
        ax2.set_xlabel('Time [h]') # Traduzido
        ax2.grid(True)
        add_phase_lines(ax2, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 3. Substrato (S) (Posição 3: Linha 1, Coluna 3)
        # Usa plt.subplot(2, 3, 3)
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(t, S, 'm-')
        ax3.set_title('Substrate (S)') # Traduzido
        ax3.set_ylabel('[g/L]')
        ax3.set_xlabel('Time [h]') # Traduzido
        ax3.grid(True); ax3.set_ylim(bottom=0)
        add_phase_lines(ax3, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 4. Etanol (P) (Posição 4: Linha 2, Coluna 1)
        # Usa plt.subplot(2, 3, 4)
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(t, P, 'k-')
        ax4.set_title('Ethanol (P)') # Título já estava ok
        ax4.set_ylabel('[g/L]')
        ax4.set_xlabel('Time [h]') # Traduzido
        ax4.grid(True); ax4.set_ylim(bottom=0)
        add_phase_lines(ax4, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 5. Oxigênio Dissolvido (O2) (Posição 5: Linha 2, Coluna 2)
        # Usa plt.subplot(2, 3, 5)
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(t, O2, 'c-')
        ax5.set_title('Dissolved Oxygen (O2)') # Traduzido
        ax5.set_ylabel('[mg/L]')
        ax5.set_xlabel('Time [h]') # Traduzido
        ax5.grid(True); ax5.set_ylim(bottom=-0.1, top=Cs*1.1 if 'Cs' in locals() and Cs is not None else 8.0) # Checagem adicional para Cs
        add_phase_lines(ax5, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)
        # Linha comentada ax5.legend removida

        # 6. Tasa de Crecimiento Específica (mu) - RECALCULAR ANTES DE PLOTAR
        # **** INÍCIO DO BLOCO DE CÁLCULO DE mu_sim (ORIGINAL PRESERVADO) ****
        mu_sim = []
        # Garante que todas as listas/arrays de entrada tenham o mesmo comprimento
        min_len = len(t)
        if len(X) < min_len: min_len = len(X)
        if len(S) < min_len: min_len = len(S)
        if len(P) < min_len: min_len = len(P)
        if len(O2) < min_len: min_len = len(O2)
        if len(V) < min_len: min_len = len(V)

        for i in range(min_len): # Itera até o comprimento mínimo para evitar IndexError
            ti, xi, si, pi, o2i, vi = t[i], X[i], S[i], P[i], O2[i], V[i]
            # Determina a fase baseado no tempo atual vs t_batch_inicial_fin
            fase_i = "inicial_batch" if ti < params.get("t_batch_inicial_fin", float('inf')) else "fed_or_final_batch"
            mu_i = 0.0
            # Calcula mu_i baseado no tipo definido em params
            # (Lógica original mantida)
            if params["tipo_mu"] == "Fermentation":
                mu_i = mu_fermentacion(si, pi, o2i, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params["n_p"], params["KO_inhib_anaerob"])
            elif params["tipo_mu"] == "Switched Fermentation":
                if fase_i == "inicial_batch":
                    current_mumax_i = params.get("mumax_aero", 0.0); current_O2_for_mu_i = params.get("O2_controlado", o2i)
                    mu_i = mu_fermentacion(si, pi, current_O2_for_mu_i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=True)
                else: # fed_or_final_batch
                    current_mumax_i = params.get("mumax_anaero", 0.0)
                    mu_i = mu_fermentacion(si, pi, o2i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=False)
            elif params["tipo_mu"] == "Simple Monod":
                mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
            elif params["tipo_mu"] == "Sigmoidal Monod":
                mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
            elif params["tipo_mu"] == "Monod with Restrictions":
                current_O2_for_mu_i = o2i
                mu_i = mu_completa(si, current_O2_for_mu_i, pi, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))

            mu_sim.append(max(0, mu_i)) # Garante que mu não seja negativo
        # **** FIM DO BLOCO DE CÁLCULO DE mu_sim ****

        # Plotar mu_sim (Posição 6: Linha 2, Coluna 3)
        # Usa plt.subplot(2, 3, 6)
        ax6 = plt.subplot(2, 3, 6)
        # Verifica se mu_sim tem o mesmo tamanho que t (após o loop)
        if len(mu_sim) == len(t):
            ax6.plot(t, mu_sim, 'y-')
        elif len(mu_sim) == min_len: # Se o loop usou min_len
            ax6.plot(t[:min_len], mu_sim, 'y-') # Plota contra a parte correspondente de t
        else:
            print(f"Warning: Discrepancy in the size of t ({len(t)}) and mu_sim ({len(mu_sim)}). Plot may be incomplete.")
            # Tenta plotar mesmo assim se mu_sim não estiver vazio
            if mu_sim:
                ax6.plot(t[:len(mu_sim)], mu_sim, 'y-')


        # Usando LaTeX para a letra grega mu para melhor renderização
        ax6.set_title(r'Specific Growth Rate ($\mu$)') # Traduzido e com LaTeX
        ax6.set_ylabel('[1/h]')
        ax6.set_xlabel('Time [h]') # Traduzido
        ax6.grid(True); ax6.set_ylim(bottom=0)
        add_phase_lines(ax6, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # Remover o subplot ax7 (Marcadores de Fase) que estava em (4, 2, 8)

        # Ajusta o layout para evitar sobreposições
        plt.tight_layout(pad=3.0) # Padding ligeiramente aumentado
        st.pyplot(fig)

        # --- Métricas Clave ---
        st.subheader("Final Metrics (t = {:.1f} h)".format(t[-1]))
        col1, col2, col3 = st.columns(3)
        vol_final = V[-1]; etanol_final_conc = P[-1]; biomasa_final_conc = X[-1]
        S_inicial_total = S0 * V0
        S_alimentado_total = np.trapz(flujo_sim * Sin, t) if len(t)>1 else 0
        S_final_total = S[-1] * V[-1]
        S_consumido_total = max(1e-9, S_inicial_total + S_alimentado_total - S_final_total)
        P_inicial_total = P0 * V0
        etanol_final_total = etanol_final_conc * vol_final
        etanol_producido_total = etanol_final_total - P_inicial_total

        col1.metric("Final Volume [L]", f"{vol_final:.2f}")
        col2.metric("Final Etanol [g/L]", f"{etanol_final_conc:.2f}")
        col3.metric("Final Biomass [g/L]", f"{biomasa_final_conc:.2f}")

        prod_vol_etanol = etanol_producido_total / vol_final / t[-1] if t[-1] > 0 and vol_final > 1e-6 else 0
        col1.metric("Ethanol Volume Productivity [g/L/h]", f"{prod_vol_etanol:.3f}")

        rend_global_etanol = etanol_producido_total / S_consumido_total if S_consumido_total > 1e-9 else 0
        col2.metric("Global Yield P/S [g/g]", f"{rend_global_etanol:.3f}")

        try:
            X_V_int = np.trapz(X*V, t) if t[-1] > 0 else 0 # Integral de X*V
            if X_V_int > 1e-9 and t[-1] > 0:
                 prod_esp_etanol_media = etanol_producido_total / X_V_int
                 col3.metric("Avg. Specific Ethanol Production [g/gXh]", f"{prod_esp_etanol_media:.4f}")
            else: col3.metric("Avg. Specific Ethanol Production [g/gXh]", "N/A")
        except Exception: col3.metric("Avg. Specific Ethanol Production [g/gXh]", "Error")


        try:
            p_max_idx = np.argmax(P)
            col1.metric("Max Ethanol [g/L]", f"{P[p_max_idx]:.2f} (a t={t[p_max_idx]:.1f} h)")
        except ValueError: col1.metric("Max Ethanol [g/L]", "N/A")

        col2.metric("Residual Substrate [g/L]", f"{S[-1]:.2f}")

        #Adição do código para salvar as variáveis

        data_simulation = {
            'Time [h]': t,
            'Flow [L/h]': flujo_sim,
            'Volume [L]': V,
			'Biomass (X) [g/L]': X,
			'Substrate (S) [g/L]': S,
			'Ethanol (P) [g/L]': P,
			'Dissolved Oxygen (O2) [g/L]': O2,
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
            worksheet.insert_image('E2', 'graphic.png', {'image_data': img_stream})

        buffer.seek(0)


        st.download_button(
            label="Download Simulation Data as Excel",
            data=buffer,
            file_name="Simulation_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    except Exception as e:
        st.error(f"An error occurred during simulation or results processing: {e}")
        st.error(traceback.format_exc())

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Alcoholic Fermentation Simulator")
    fermentacion_alcoholica_page()
