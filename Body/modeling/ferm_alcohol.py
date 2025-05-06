# ferm_alcohol.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Importar TODAS las funciones cinéticas (asegúrate que estén disponibles)
# from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion
# --- INICIO: Definiciones dummy de funciones cinéticas (si no tienes Utils.kinetics) ---
# Reemplaza esto con tus importaciones reales si están disponibles
def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)
def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * (S**n) / (Ks**n + S**n)
def mu_completa(S, O2, P, mumax, Ks, KO, KP_gen):
    mu = mumax * (S / (Ks + S)) * (O2 / (KO + O2)) * (KP_gen / (KP_gen + P))
    return mu
def mu_fermentacion(S, P, O2, mumax_aerob, Ks_aerob, KO_aerob, mumax_anaerob, Ks_anaerob, KiS_anaerob, KP_anaerob, n_p, KO_inhib_anaerob, considerar_O2=None):
    # Modelo mixto simplificado: suma ponderada por O2
    mu_aer = mumax_aerob * (S / (Ks_aerob + S)) * (O2 / (KO_aerob + O2))
    mu_anaer = mumax_anaerob * (S / (Ks_anaerob + S + S**2/KiS_anaerob)) * (KP_anaerob**n_p / (KP_anaerob**n_p + P**n_p)) * (KO_inhib_anaerob / (KO_inhib_anaerob + O2))
    # El argumento considerar_O2 no se usa en esta implementación mixta directa,
    # pero se mantiene por compatibilidad con la llamada en 'Fermentación Conmutada'.
    # Si 'Fermentación Conmutada' lo llama con considerar_O2=True, debería usar mu_aer.
    # Si lo llama con considerar_O2=False, debería usar mu_anaer.
    # Esta implementación simple suma ambas, lo cual es más para el tipo "Fermentación".
    # Para una implementación correcta de "Conmutada", la lógica debería estar más arriba.
    if considerar_O2 is True: # Llamado desde Conmutada - Fase Aerobia
        return mumax_aerob * (S / (Ks_aerob + S)) * (O2 / (KO_aerob + O2)) # Asume Ks_aerob = Ks, KO_aerob = KO
    elif considerar_O2 is False: # Llamado desde Conmutada - Fase Anaerobia
         return mumax_anaerob * (S / (Ks_anaerob + S + S**2/KiS_anaerob)) * (KP_anaerob**n_p / (KP_anaerob**n_p + P**n_p)) # Asume Ks_anaerob = Ks, etc.
    else: # Llamado desde modelo "Fermentación" (mixto)
        return mu_aer + mu_anaer
# --- FIN: Definiciones dummy ---


def fermentacion_alcoholica_page():
    st.header("Simulación de Fermentación Alcohólica en Lote Alimentado")
    st.markdown("""
    Este modelo simula una fermentación alcohólica que inicia en **lote (fase aeróbica)**,
    continúa en **lote alimentado (fase de transición/anaeróbica)**, y finaliza
    en **lote (fase anaeróbica)** para agotar el sustrato. Seleccione el modelo cinético.
    **NUEVO:** Se incluye inhibición de producción de etanol por Oxígeno ($K_{O,P}$).
    """)

    with st.sidebar:
        st.subheader("1. Modelo Cinético y Parámetros")
        tipo_mu = st.selectbox("Modelo Cinético", ["Fermentación", "Fermentación Conmutada", "Monod simple", "Monod sigmoidal", "Monod con restricciones"])

        Ks = st.slider("Ks base [g/L]", 0.01, 10.0, 1.0, 0.1)

        # --- Parámetros específicos de cada modelo cinético ---
        # (Sin cambios aquí respecto a tu código original)
        if tipo_mu == "Monod simple":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
        elif tipo_mu == "Monod sigmoidal":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
            n_sig = st.slider("Exponente sigmoidal (n)", 1, 5, 2)
        elif tipo_mu == "Monod con restricciones":
            mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4, 0.05)
            KO_restr = st.slider("KO (O2 - restricción) [mg/L]", 0.01, 5.0, 0.1, 0.01)
            KP_gen = st.slider("KP (Inhib. Producto genérico) [g/L]", 1.0, 100.0, 50.0)
        elif tipo_mu == "Fermentación Conmutada":
            st.info("Modelo conmutado: Usa mu_fermentacion con mumax_aero/anaero y O2 opcional.")
            mumax_aero_c = st.slider("μmax (Fase Aeróbica) [1/h]", 0.1, 1.0, 0.45, 0.05, key="mumax_aero_c")
            mumax_anaero_c = st.slider("μmax (Fase Anaeróbica) [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaero_c")
            KiS_c = st.slider("KiS (Inhib. Sustrato) [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_c")
            KP_c = st.slider("KP (Inhib. Etanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_c")
            n_p_c = st.slider("Exponente Inhib. Etanol (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_c")
            KO_ferm_c = st.slider("KO (O2 - afinidad aerobia) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_ferm_c")
        elif tipo_mu == "Fermentación":
            st.info("Modelo mixto: mu = mu1(aerobio) + mu2(anaerobio).")
            st.markdown("**Parámetros mu1 (Aerobio):**")
            mumax_aerob_m = st.slider("μmax_aerob [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_aerob_m")
            Ks_aerob_m = st.slider("Ks_aerob [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aerob_m")
            KO_aerob_m = st.slider("KO_aerob (afinidad O2) [mg/L]", 0.01, 5.0, 0.2, 0.01, key="ko_aerob_m")
            st.markdown("**Parámetros mu2 (Anaerobio/Fermentativo):**")
            mumax_anaerob_m = st.slider("μmax_anaerob [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaerob_m")
            Ks_anaerob_m = st.slider("Ks_anaerob [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaerob_m")
            KiS_anaerob_m = st.slider("KiS_anaerob [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaerob_m")
            KP_anaerob_m = st.slider("KP_anaerob (Inhib. Etanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaerob_m")
            n_p_m = st.slider("Exponente Inhib. Etanol (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_m")
            KO_inhib_anaerob_m = st.slider("KO_inhib_anaerob (Inhib. O2) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_inhib_m")

        # --- Resto de parámetros ---
        st.subheader("2. Parámetros Estequiométricos y de Mantenimiento")
        Yxs = st.slider("Yxs (Biomasa/Sustrato) [g/g]", 0.05, 0.6, 0.1, 0.01, key="yxs")
        Yps = st.slider("Yps (Etanol/Sustrato) [g/g]", 0.1, 0.51, 0.45, 0.01, key="yps")
        Yxo = st.slider("Yxo (Biomasa/O2) [gX/gO2]", 0.1, 2.0, 0.8, 0.1, key="yxo")
        alpha_lp = st.slider("α (Asociado a crecimiento) [g P / g X]", 0.0, 5.0, 2.2, 0.1, key="alpha")
        beta_lp = st.slider("β (No asociado a crecimiento) [g P / g X / h]", 0.0, 0.5, 0.05, 0.01, key="beta")
        ms = st.slider("ms (Mantenimiento Sustrato) [g S / g X / h]", 0.0, 0.2, 0.02, 0.01, key="ms")
        mo = st.slider("mo (Mantenimiento O2) [gO2/gX/h]", 0.0, 0.1, 0.01, 0.005, key="mo")
        Kd = st.slider("Kd (Decaimiento Biomasa) [1/h]", 0.0, 0.1, 0.01, 0.005, key="kd")
        # <<< NUEVO: Parámetro de inhibición de producción de etanol por O2 >>>
        KO_inhib_prod = st.slider("KO_inhib_prod (Inhib. O2 en Prod. Etanol) [mg/L]", 0.001, 1.0, 0.05, 0.005, key="ko_inhib_p", help="Concentración de O2 que reduce a la mitad la producción de etanol. Valor bajo = fuerte inhibición.")

        st.subheader("3. Transferencia de Oxígeno")
        Kla = st.slider("kLa [1/h]", 10.0, 400.0, 100.0, 10.0, key="kla")
        Cs = st.slider("O2 Saturado (Cs) [mg/L]", 0.01, 15.0, 7.5, 0.01, key="cs") # <<< MODIFICADO: Valor mínimo a 0.01 para evitar problemas

        st.subheader("4. Fases de Operación y Alimentación")
        t_batch_inicial_fin = st.slider("Fin Fase Lote Inicial [h]", 1.0, 30.0, 10.0, 1.0, key="t_batch_fin")
        t_alim_inicio = st.slider("Inicio Alimentación [h]", t_batch_inicial_fin, t_batch_inicial_fin + 24.0, t_batch_inicial_fin + 0.1, 0.5, key="t_alim_ini")
        t_alim_fin = st.slider("Fin Alimentación (Inicio Lote Final) [h]", t_alim_inicio + 1.0, 40.0, t_alim_inicio + 24.0, 1.0, key="t_alim_fin")
        t_final = st.slider("Tiempo Total de Simulación [h]", t_alim_fin + 1.0, 100.0, t_alim_fin + 5.0, 1.0, key="t_total")
        O2_controlado = st.slider("Nivel O2 Objetivo/Ref (Fase Lote Inicial) [mg/L]", 0.01, Cs, 0.08, 0.01, key="o2_control") # Valor 0.08

        estrategia = st.selectbox("Estrategia Alimentación", ["Constante", "Exponencial", "Lineal", "Escalon"], key="strat")
        Sin = st.slider("Sustrato en Alimentación (Sin) [g/L]", 10.0, 700.0, 400.0, 10.0, key="sin")
        F_base = st.slider("Flujo Base (o Inicial) [L/h]", 0.01, 5.0, 0.1, 0.01, key="fbase")
        if estrategia == "Lineal":
            F_lineal_fin = st.slider("Flujo Final (Lineal) [L/h]", F_base, 10.0, F_base * 2, 0.01, key="ffin_lin")
        elif estrategia == "Exponencial":
            k_exp = st.slider("Constante Crecimiento Exp. (k_exp) [1/h]", 0.01, 0.5, 0.1, 0.01, key="kexp")

        st.subheader("5. Condiciones Iniciales")
        V0 = st.number_input("Volumen Inicial [L]", 0.1, 100.0, 5.0, key="v0")
        X0 = st.number_input("Biomasa Inicial [g/L]", 0.05, 10.0, 0.1, key="x0")
        S0 = st.number_input("Sustrato Inicial [g/L]", 10.0, 200.0, 100.0, key="s0")
        P0 = st.number_input("Etanol Inicial [g/L]", 0.0, 50.0, 0.0, key="p0")
        O0 = st.number_input("O2 Inicial [mg/L]", min_value=0.0, max_value=Cs, value=0.08, step=0.01, key="o0") # Valor 0.08

        st.subheader("6. Parámetros del Solver")
        atol = st.number_input("Tolerancia absoluta (atol)", min_value=1e-9, max_value=1e-3, value=1e-6, format="%e", key="atol")
        rtol = st.number_input("Tolerancia relativa (rtol)", min_value=1e-9, max_value=1e-3, value=1e-6, format="%e", key="rtol")

    # ------- Funciones de Cálculo Auxiliares -------
    F_lineal_fin_val = F_base * 2
    k_exp_val = 0.1
    if estrategia == "Lineal": F_lineal_fin_val = F_lineal_fin
    elif estrategia == "Exponencial": k_exp_val = k_exp

    def calcular_flujo(t):
        # (Sin cambios aquí)
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constante": return F_base
            elif estrategia == "Exponencial":
                try: return min(F_base * np.exp(k_exp_val * (t - t_alim_inicio)), F_base * 100) # Limitar flujo exp
                except OverflowError: return F_base * 100
            elif estrategia == "Escalon":
                t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2
                return F_base * 2 if t > t_medio else F_base
            elif estrategia == "Lineal":
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
        "KO_inhib_prod": KO_inhib_prod, # <<< NUEVO: Añadir el parámetro al diccionario
    }
    if tipo_mu == "Monod simple": params["mumax"] = mumax
    elif tipo_mu == "Monod sigmoidal": params["mumax"] = mumax; params["n_sig"] = n_sig
    elif tipo_mu == "Monod con restricciones": params["mumax"] = mumax; params["KO"] = KO_restr; params["KP_gen"] = KP_gen
    elif tipo_mu == "Fermentación Conmutada":
        params["mumax_aero"] = mumax_aero_c; params["mumax_anaero"] = mumax_anaero_c
        params["KiS"] = KiS_c; params["KP"] = KP_c; params["n_p"] = n_p_c; params["KO"] = KO_ferm_c
        params["O2_controlado"] = O2_controlado # Necesario para este modelo
         # <<< MODIFICADO: Pasar parámetros necesarios a mu_fermentacion incluso en modo conmutado >>>
        params["Ks_aerob"] = Ks # Usar Ks base como Ks_aerob
        params["KO_aerob"] = KO_ferm_c # Usar KO_ferm_c como KO_aerob
        params["Ks_anaerob"] = Ks # Usar Ks base como Ks_anaerob
        params["KiS_anaerob"] = KiS_c # Usar KiS_c como KiS_anaerob
        params["KP_anaerob"] = KP_c # Usar KP_c como KP_anaerob
        # n_p ya está ("n_p")
        params["KO_inhib_anaerob"] = 1e-6 # Asumir inhibición muy fuerte por O2 en anaerobiosis si no se define explícitamente
    elif tipo_mu == "Fermentación":
        params["mumax_aerob"] = mumax_aerob_m; params["Ks_aerob"] = Ks_aerob_m; params["KO_aerob"] = KO_aerob_m
        params["mumax_anaerob"] = mumax_anaerob_m; params["Ks_anaerob"] = Ks_anaerob_m; params["KiS_anaerob"] = KiS_anaerob_m
        params["KP_anaerob"] = KP_anaerob_m; params["n_p"] = n_p_m; params["KO_inhib_anaerob"] = KO_inhib_anaerob_m

    # ------- Modelo ODE -------
    def modelo_fermentacion(t, y, params):
        X, S, P, O2, V = y
        # Asegurar valores no negativos y evitar división por cero
        X = max(1e-9, X); S = max(0, S); P = max(0, P); O2 = max(0, O2); V = max(1e-6, V)

        es_lote_inicial = (t < params["t_batch_inicial_fin"])

        # --- Cálculo de mu ---
        mu = 0.0
        # <<< MODIFICADO: Lógica de cálculo de mu para "Fermentación Conmutada" y "Fermentación" >>>
        if params["tipo_mu"] == "Fermentación":
             # Llama a mu_fermentacion que ahora debería sumar componentes aerobio y anaerobio
             mu = mu_fermentacion(S, P, O2, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"],
                                  params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"],
                                  params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"],
                                  considerar_O2=None) # None indica modo mixto
        elif params["tipo_mu"] == "Fermentación Conmutada":
             # Llama a mu_fermentacion pasando flag para seleccionar la cinética correcta
             if es_lote_inicial:
                 current_mumax = params.get("mumax_aero", 0.0)
                 current_O2_for_mu = params.get("O2_controlado", O2) # Ojo: Usa O2 controlado para el cálculo de mu en fase 1
                 mu = mu_fermentacion(S, P, current_O2_for_mu, current_mumax, params["Ks"], # Usa Ks base
                                      params.get("KO", 0.1), # Usa KO_ferm_c (como KO_aerob)
                                      0, params.get("Ks", 1.0), # Placeholder para anaerobios
                                      float('inf'), float('inf'), 1.0, 1e-6, # Placeholder para anaerobios
                                      considerar_O2=True) # Indica fase aerobia
             else:
                 current_mumax = params.get("mumax_anaero", 0.0)
                 # Usa O2 real simulado para la cinética anaerobia (inhibición por O2)
                 mu = mu_fermentacion(S, P, O2, 0, params.get("Ks", 1.0), 1e-6, # Placeholder para aerobios
                                      current_mumax, params["Ks"], # Usa Ks base
                                      params.get("KiS", float('inf')), params.get("KP", float('inf')),
                                      params.get("n_p", 1.0), params.get("KO_inhib_anaerob", 1e-6), # Necesita KO_inhib_anaerob
                                      considerar_O2=False) # Indica fase anaerobia
        elif params["tipo_mu"] == "Monod simple": mu = mu_monod(S, params.get("mumax", 0.0), params["Ks"])
        elif params["tipo_mu"] == "Monod sigmoidal": mu = mu_sigmoidal(S, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
        elif params["tipo_mu"] == "Monod con restricciones":
             mu = mu_completa(S, O2, P, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))

        mu = max(0, mu) # Asegurar que mu no sea negativo

        # --- Cálculo de Flujo y Tasas de Consumo/Producción ---
        F = calcular_flujo(t)

        # Tasa específica de producción de Etanol (Luedeking-Piret base)
        qP_base = params["alpha_lp"] * mu + params["beta_lp"]

        # <<< NUEVO: Factor de inhibición por O2 en la producción de Etanol >>>
        ko_inhib_p = params.get("KO_inhib_prod", 0.05) # Obtener K_O,P del diccionario
        # Evitar división por cero si ko_inhib_p es muy pequeño o O2 es negativo
        inhib_factor_O2_prod = ko_inhib_p / (ko_inhib_p + max(1e-9, O2))

        # Tasa volumétrica de producción de Etanol (MODIFICADA por inhibición de O2)
        rate_P = qP_base * X * inhib_factor_O2_prod
        # <<< FIN NUEVO >>>

        # Consumos de Sustrato (sin cambios aquí)
        consumo_S_X = (mu / params["Yxs"]) * X if params["Yxs"] > 1e-6 else 0
        # <<< MODIFICADO: Consumo de sustrato para producto debe usar la rate_P efectiva >>>
        consumo_S_P = (rate_P / params["Yps"]) if params["Yps"] > 1e-6 else 0
        consumo_S_maint = params["ms"] * X
        rate_S = consumo_S_X + consumo_S_P + consumo_S_maint

        # Consumos de Oxígeno (sin cambios aquí)
        consumo_O2_X = (mu / params["Yxo"]) * X if params["Yxo"] > 1e-6 else 0
        consumo_O2_maint = params["mo"] * X
        OUR_g = consumo_O2_X + consumo_O2_maint # [g O2 / L / h]
        OUR_mg = OUR_g * 1000.0 # Convertir a [mg O2 / L / h] para balance de O2

        # Ecuaciones Diferenciales
        dXdt = (mu - params["Kd"]) * X - (F / V) * X
        dSdt = -rate_S + (F / V) * (params["Sin"] - S)
        # dPdt usa la rate_P modificada que incluye la inhibición por O2
        dPdt = rate_P - (F / V) * P
        dVdt = F

        # --- Cálculo de dOdt (sin cambios en la lógica de fase) ---
        if es_lote_inicial:
             # En la fase inicial, si O2 está controlado, asumimos que dOdt es tal que mantiene O2 en el nivel objetivo.
             # Para la simulación, fijar O2 puede ser complejo. Fijar dOdt=0 es una simplificación
             # que mantiene O2 en su valor inicial si O0 = O2_controlado.
             # Si O0 != O2_controlado, O2 permanecerá en O0 durante esta fase.
             # Una mejor aproximación sería calcular el OTR necesario y ajustar Kla, pero es más complejo.
             # Manteniendo tu lógica original:
             dOdt = 0.0 # Fuerza derivada a cero en la fase inicial batch
             # <<< NOTA: Si O0 es diferente de O2_controlado, O2 se mantendrá en O0.
             # El O2 usado en el cálculo de mu para "Fermentación Conmutada" sí usa O2_controlado >>>
        else:
             # Calcular normalmente en fases posteriores (fed-batch y final batch)
             OTR = params["Kla"] * (params["Cs"] - O2) # [mg O2 / L / h]
             dOdt = OTR - OUR_mg - (F / V) * O2     # [mg/L/h]

        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    # ------- Simulación y Resultados -------
    y0 = [X0, S0, P0, O0, V0]
    t_span = [0, t_final]
    # Aumentar el número de puntos para capturar mejor la dinámica
    num_puntos = max(500, int(t_final * 25) + 1)
    t_eval = np.linspace(t_span[0], t_span[1], num_puntos)

    try:
        # --- Bloque solve_ivp (sin cambios) ---
        sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval, method='RK45', atol=atol, rtol=rtol, args=(params,), max_step=0.5)
        if not sol.success:
            st.warning("RK45 falló, intentando con BDF...")
            sol = solve_ivp(modelo_fermentacion, t_span, y0, t_eval=t_eval, method='BDF', atol=atol, rtol=rtol, args=(params,))
            if not sol.success:
                st.error(f"Integración falló con ambos métodos: {sol.message}")
                st.stop()
            else: st.success("Integración completada con BDF.")

        t = sol.t
        X, S, P, O2, V = sol.y
        # Corregir posibles valores negativos muy pequeños por errores numéricos
        X = np.maximum(X, 0); S = np.maximum(S, 0); P = np.maximum(P, 0); O2 = np.maximum(O2, 0); V = np.maximum(V, 1e-6)

        # <<< MODIFICADO: Corregir O2 en la fase inicial si dOdt=0 fue usado >>>
        # Si se usó dOdt=0, O2 debería ser constante = O0 en esa fase.
        # O si se usó O2_controlado en mu, quizás sea mejor setear O2 a ese valor en la fase 1 para consistencia gráfica.
        # Por simplicidad, mantendremos la salida del solver, pero ten en cuenta esta posible inconsistencia.
        # Si quieres forzar O2 = O2_controlado en fase 1 post-simulación:
        # indices_fase1 = np.where(t < t_batch_inicial_fin)[0]
        # if len(indices_fase1) > 0:
        #    O2[indices_fase1] = O2_controlado # O podrías usar O0

        flujo_sim = np.array([calcular_flujo(ti) for ti in t])

        # --- Recalcular Tasas Específicas y Volumétricas Post-Simulación ---
        mu_sim = []
        qP_sim = []
        qS_sim = []
        qO_sim = []
        OTR_sim = []
        inhib_factor_sim = []

        for i in range(len(t)):
            # Recalcular mu usando los valores simulados
            xi, si, pi, o2i, vi = X[i], S[i], P[i], O2[i], V[i]
            ti = t[i]
            es_lote_inicial_i = (ti < params["t_batch_inicial_fin"])

            # Re-calcular mu exactamente como dentro del solver
            mu_i = 0.0
            if params["tipo_mu"] == "Fermentación":
                 mu_i = mu_fermentacion(si, pi, o2i, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"],
                                      params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"],
                                      params["KP_anaerob"], params.get("n_p", 1.0), params["KO_inhib_anaerob"],
                                      considerar_O2=None)
            elif params["tipo_mu"] == "Fermentación Conmutada":
                 if es_lote_inicial_i:
                     current_mumax_i = params.get("mumax_aero", 0.0)
                     # Usar O2 simulado aquí para ver el mu real bajo esa condición,
                     # incluso si el solver usó O2_controlado internamente para la fase 1.
                     # O usar O2_controlado si quieres ver el mu 'objetivo' de esa fase. Usemos el simulado.
                     current_O2_for_mu_i = o2i
                     mu_i = mu_fermentacion(si, pi, current_O2_for_mu_i, current_mumax_i, params["Ks"],
                                          params.get("KO", 0.1), 0, params.get("Ks", 1.0),
                                          float('inf'), float('inf'), 1.0, 1e-6,
                                          considerar_O2=True)
                 else:
                     current_mumax_i = params.get("mumax_anaero", 0.0)
                     mu_i = mu_fermentacion(si, pi, o2i, 0, params.get("Ks", 1.0), 1e-6,
                                          current_mumax_i, params["Ks"], params.get("KiS", float('inf')),
                                          params.get("KP", float('inf')), params.get("n_p", 1.0),
                                          params.get("KO_inhib_anaerob", 1e-6),
                                          considerar_O2=False)
            elif params["tipo_mu"] == "Monod simple": mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
            elif params["tipo_mu"] == "Monod sigmoidal": mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
            elif params["tipo_mu"] == "Monod con restricciones":
                 mu_i = mu_completa(si, o2i, pi, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))

            mu_i = max(0, mu_i)
            mu_sim.append(mu_i)

            # Recalcular qP, qS, qO, OTR
            qP_base_i = params["alpha_lp"] * mu_i + params["beta_lp"]
            ko_inhib_p_i = params.get("KO_inhib_prod", 0.05)
            inhib_factor_i = ko_inhib_p_i / (ko_inhib_p_i + max(1e-9, o2i))
            inhib_factor_sim.append(inhib_factor_i)
            qP_i = qP_base_i * inhib_factor_i
            qP_sim.append(qP_i)

            consumo_S_X_i = (mu_i / params["Yxs"]) if params["Yxs"] > 1e-6 else 0
            consumo_S_P_i = (qP_i / params["Yps"]) if params["Yps"] > 1e-6 else 0 # Usa qP efectivo
            qS_i = consumo_S_X_i + consumo_S_P_i + params["ms"]
            qS_sim.append(qS_i)

            consumo_O2_X_i = (mu_i / params["Yxo"]) if params["Yxo"] > 1e-6 else 0
            qO_i = consumo_O2_X_i + params["mo"] # gO2/gX/h
            qO_sim.append(qO_i)

            OTR_i = params["Kla"] * (params["Cs"] - o2i) # mg/L/h
            OTR_sim.append(OTR_i)

        # Convertir listas a arrays numpy
        mu_sim = np.array(mu_sim)
        qP_sim = np.array(qP_sim)
        qS_sim = np.array(qS_sim)
        qO_sim = np.array(qO_sim) * 1000 # Convertir a mgO2/gX/h
        OTR_sim = np.array(OTR_sim)
        OUR_sim = qO_sim * X # mgO2/L/h
        inhib_factor_sim = np.array(inhib_factor_sim)


        # --- Graficación (Añadir gráficas de tasas y factor de inhibición) ---
        st.subheader("Resultados de la Simulación")
        # Aumentar el número de filas para las nuevas gráficas
        fig = plt.figure(figsize=(14, 24)) # Ajustar tamaño si es necesario

        # Función auxiliar para añadir líneas de fase (sin cambios)
        def add_phase_lines(ax, t_batch, t_feed_start, t_feed_end):
            ymin, ymax = ax.get_ylim()
            ax.vlines([t_batch, t_feed_start, t_feed_end], ymin, ymax,
                      colors=['gray', 'orange', 'purple'], linestyles='--', lw=1.5)
            ax.set_ylim(ymin, ymax) # Restablecer límites Y después de vlines

        # Añadir etiquetas solo una vez (por ejemplo, en la primera gráfica)
        add_labels = True

        # 1. Flujo y Volumen
        ax1 = plt.subplot(6, 2, 1) # Cambiado a 6 filas
        color = 'tab:red'; ax1.plot(t, flujo_sim, color=color, label='Flujo Alimentación')
        ax1.set_ylabel('Flujo [L/h]', color=color); ax1.tick_params(axis='y', labelcolor=color)
        ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t, V, color=color, linestyle='-', label='Volumen')
        ax1b.set_ylabel('Volumen [L]', color=color); ax1b.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Tiempo [h]'); ax1.grid(True); ax1.set_title('Alimentación y Volumen');
        # Combinar leyendas y añadir líneas de fase con etiquetas
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ymin1, ymax1 = ax1.get_ylim(); ymin2, ymax2 = ax1b.get_ylim()
        vh = ax1.vlines([t_batch_inicial_fin, t_alim_inicio, t_alim_fin], ymin1, ymax1,
                        colors=['gray', 'orange', 'purple'], linestyles='--', lw=1.5,
                        label='_nolegend_') # Evitar leyenda automática de vlines
        # Crear handles para la leyenda de las líneas de fase manualmente
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label=f'Fin Lote Inicial ({t_batch_inicial_fin:.1f}h)'),
            Line2D([0], [0], color='orange', linestyle='--', lw=1.5, label=f'Inicio Aliment. ({t_alim_inicio:.1f}h)'),
            Line2D([0], [0], color='purple', linestyle='--', lw=1.5, label=f'Fin Aliment. ({t_alim_fin:.1f}h)')
        ]
        ax1b.legend(lines + lines2 + legend_elements, labels + labels2 + [le.get_label() for le in legend_elements], loc='best')
        ax1.set_ylim(ymin1, ymax1); ax1b.set_ylim(ymin2, ymax2) # Restaurar límites
        add_labels = False # Ya se añadieron las etiquetas


        # 2. Biomasa (X)
        ax2 = plt.subplot(6, 2, 3); ax2.plot(t, X, 'g-'); ax2.set_title('Biomasa (X)'); ax2.set_ylabel('[g/L]'); ax2.set_xlabel('Tiempo [h]'); ax2.grid(True); ax2.set_ylim(bottom=0)
        add_phase_lines(ax2, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 3. Sustrato (S)
        ax3 = plt.subplot(6, 2, 4); ax3.plot(t, S, 'm-'); ax3.set_title('Sustrato (S)'); ax3.set_ylabel('[g/L]'); ax3.set_xlabel('Tiempo [h]'); ax3.grid(True); ax3.set_ylim(bottom=0)
        add_phase_lines(ax3, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 4. Etanol (P)
        ax4 = plt.subplot(6, 2, 5); ax4.plot(t, P, 'k-'); ax4.set_title('Etanol (P)'); ax4.set_ylabel('[g/L]'); ax4.set_xlabel('Tiempo [h]'); ax4.grid(True); ax4.set_ylim(bottom=0)
        add_phase_lines(ax4, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 5. Oxígeno Disuelto (O2)
        ax5 = plt.subplot(6, 2, 6); ax5.plot(t, O2, 'c-', label='O2 Disuelto'); ax5.set_title('Oxígeno Disuelto (O2) y Factor Inhibición Etanol'); ax5.set_ylabel('[mg/L]', color='c'); ax5.set_xlabel('Tiempo [h]'); ax5.grid(True); ax5.set_ylim(bottom=-0.05, top=max(Cs*1.1, np.max(O2)*1.1)); ax5.tick_params(axis='y', labelcolor='c')
        # Añadir línea de O2 controlado como referencia
        ax5.axhline(O2_controlado, color='lightcoral', linestyle=':', lw=1.5, label=f'O2 Ref Fase 1 ({O2_controlado:.2f})')
        # Graficar el factor de inhibición en eje Y secundario
        ax5b = ax5.twinx()
        ax5b.plot(t, inhib_factor_sim, color='darkorange', linestyle='--', label=r'Factor Inhibición $O_2$ ($K_{O,P}/(K_{O,P}+O_2)$)')
        ax5b.set_ylabel('Factor Inhibición Prod. Etanol [-]', color='darkorange')
        ax5b.tick_params(axis='y', labelcolor='darkorange')
        ax5b.set_ylim(bottom=-0.05, top=1.05)
        lines5, labels5 = ax5.get_legend_handles_labels()
        lines5b, labels5b = ax5b.get_legend_handles_labels()
        ax5b.legend(lines5 + lines5b, labels5 + labels5b, loc='center right')
        add_phase_lines(ax5, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)


        # 6. Tasa de Crecimiento Específica (mu)
        ax6 = plt.subplot(6, 2, 7); ax6.plot(t, mu_sim, 'y-'); ax6.set_title('Tasa Crecimiento Específica (μ)'); ax6.set_ylabel('[1/h]'); ax6.set_xlabel('Tiempo [h]'); ax6.grid(True); ax6.set_ylim(bottom=0)
        add_phase_lines(ax6, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 7. Tasas Específicas (qS, qP, qO)
        ax7 = plt.subplot(6, 2, 8);
        ax7.plot(t, qS_sim, 'r-', label=r'$q_S$ (Sustrato)')
        ax7.plot(t, qP_sim, 'k-', label=r'$q_P$ (Etanol)')
        ax7.set_title('Tasas Específicas (qS, qP)')
        ax7.set_ylabel('[g/gX/h]')
        ax7.set_xlabel('Tiempo [h]')
        ax7.grid(True)
        ax7.legend(loc='upper left')
        ax7b = ax7.twinx()
        ax7b.plot(t, qO_sim, 'c--', label=r'$q_{O2}$ (Oxígeno)')
        ax7b.set_ylabel('[mg O2/gX/h]', color='c')
        ax7b.tick_params(axis='y', labelcolor='c')
        ax7b.legend(loc='upper right')
        ax7.set_ylim(bottom=min(0, np.min(qS_sim)*1.1, np.min(qP_sim)*1.1) ) # Ajustar límites Y
        ax7b.set_ylim(bottom=min(0, np.min(qO_sim)*1.1) )
        add_phase_lines(ax7, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 8. Tasas Volumétricas (OUR, OTR)
        ax8 = plt.subplot(6, 2, 9);
        ax8.plot(t, OTR_sim, 'b-', label='OTR (Transferencia O2)')
        ax8.plot(t, OUR_sim, 'g--', label='OUR (Consumo O2)')
        ax8.set_title('Tasas Volumétricas de Oxígeno (OTR, OUR)')
        ax8.set_ylabel('[mg O2/L/h]')
        ax8.set_xlabel('Tiempo [h]')
        ax8.grid(True)
        ax8.legend(loc='best')
        ax8.set_ylim(bottom=0)
        add_phase_lines(ax8, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)


        # 9. Rendimientos (Yps, Yxs) Instantáneos (Opcional, puede ser ruidoso)
        ax9 = plt.subplot(6, 2, 10);
        # Calcular Yps instantáneo: qP / qS (cuidado con división por cero)
        # Calcular Yxs instantáneo: mu / qS
        qS_nozero = np.where(np.abs(qS_sim) > 1e-6, qS_sim, 1e-6) # Evitar división por cero
        Yps_inst = np.maximum(0, qP_sim / qS_nozero)
        Yxs_inst = np.maximum(0, mu_sim / qS_nozero)
        ax9.plot(t, Yps_inst, 'k-', label=r'$Y_{P/S}$ instantáneo ($q_P/q_S$)')
        ax9.plot(t, Yxs_inst, 'g--', label=r'$Y_{X/S}$ instantáneo ($\mu/q_S$)')
        ax9.set_title('Rendimientos Instantáneos Aparentes')
        ax9.set_ylabel('[g/g]')
        ax9.set_xlabel('Tiempo [h]')
        ax9.grid(True)
        ax9.legend(loc='best')
        ax9.set_ylim(bottom=0, top=max(0.6, np.max(Yps_inst)*1.1, np.max(Yxs_inst)*1.1)) # Ajustar límite superior
        add_phase_lines(ax9, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 10. Marcadores de Fase (Actualizado para 6 filas)
        ax10 = plt.subplot(6, 2, 12); ax10.axis('off'); ax10.set_title('Información Adicional')
        # (Texto sin cambios, solo posición)
        ax10.text(0.05, 0.8, f"1. Lote Inicial: 0 - {t_batch_inicial_fin:.1f} h", transform=ax10.transAxes)
        ax10.text(0.05, 0.6, f"2. Lote Alimentado: {t_alim_inicio:.1f} - {t_alim_fin:.1f} h", transform=ax10.transAxes)
        ax10.text(0.05, 0.4, f"3. Lote Final: > {t_alim_fin:.1f} h", transform=ax10.transAxes)
        ax10.text(0.05, 0.2, f"O2 Ref Fase 1: {O2_controlado:.2f} mg/L", transform=ax10.transAxes)
        ax10.text(0.05, 0.0, f"$K_{{O,P}}$ (Inhib. Prod): {KO_inhib_prod:.3f} mg/L", transform=ax10.transAxes)


        plt.tight_layout(pad=2.0)
        st.pyplot(fig)

        # --- Métricas Clave ---
        st.subheader("Métricas Finales (t = {:.1f} h)".format(t[-1]))
        col1, col2, col3 = st.columns(3)
        vol_final = V[-1]; etanol_final_conc = P[-1]; biomasa_final_conc = X[-1]
        S_inicial_total = S0 * V0
        # Calcular sustrato alimentado integrando F*Sin
        S_alimentado_total = np.trapz(flujo_sim * Sin, t) if len(t)>1 else 0
        S_final_total = S[-1] * V[-1]
        S_consumido_total = max(1e-9, S_inicial_total + S_alimentado_total - S_final_total) # Evitar división por cero
        P_inicial_total = P0 * V0
        etanol_final_total = etanol_final_conc * vol_final
        etanol_producido_total = etanol_final_total - P_inicial_total

        col1.metric("Volumen Final [L]", f"{vol_final:.2f}")
        col2.metric("Etanol Final [g/L]", f"{etanol_final_conc:.2f}")
        col3.metric("Biomasa Final [g/L]", f"{biomasa_final_conc:.2f}")

        prod_vol_etanol = etanol_producido_total / vol_final / t[-1] if t[-1] > 0 else 0
        col1.metric("Productividad Vol. Etanol [g/L/h]", f"{prod_vol_etanol:.3f}")

        rend_global_etanol = etanol_producido_total / S_consumido_total
        col2.metric("Rendimiento Global P/S [g/g]", f"{rend_global_etanol:.3f}")

        try: # Calcular productividad específica media de etanol
            X_medio_integral = np.trapz(X*V, t) / t[-1] if t[-1] > 0 else X0*V0 # Biomasa*Volumen medio integral
            if X_medio_integral > 1e-6 and t[-1] > 0:
                 prod_esp_etanol_media = etanol_producido_total / X_medio_integral / t[-1]
                 col3.metric("Prod. Esp. Etanol Media [g/gX/h]", f"{prod_esp_etanol_media:.4f}")
            else: col3.metric("Prod. Esp. Etanol Media [g/gX/h]", "N/A")
        except Exception: col3.metric("Prod. Esp. Etanol Media [g/gX/h]", "Error")

        # Mostrar Etanol Máximo alcanzado
        try:
            p_max_idx = np.argmax(P)
            col1.metric("Etanol Máx [g/L]", f"{P[p_max_idx]:.2f} (a t={t[p_max_idx]:.1f} h)")
        except ValueError: col1.metric("Etanol Máx [g/L]", "N/A")

        # Mostrar Sustrato Residual
        col2.metric("Sustrato Residual [g/L]", f"{S[-1]:.2f}")


    except Exception as e:
        st.error(f"Ocurrió un error durante la simulación o el procesamiento de resultados: {e}")
        import traceback
        st.error(traceback.format_exc())

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Simulador Fermentación Alcohólica")
    fermentacion_alcoholica_page()