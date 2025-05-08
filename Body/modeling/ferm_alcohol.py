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
        st.subheader("Resultados da Simulação")

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
        ax1.set_ylabel('Vazão [L/h]', color=color) # Traduzido de 'Flujo'
        ax1.tick_params(axis='y', labelcolor=color)
        ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t, V, color=color, linestyle='-') # Label 'Volumen' removido
        ax1b.set_ylabel('Volume [L]', color=color) # Traduzido
        ax1b.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Tempo [h]') # Traduzido
        ax1.grid(True)
        ax1.set_title('Alimentação e Volume') # Traduzido
        add_phase_lines(ax1, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)
        # Código da legenda combinada removido

        # 2. Biomassa (X) (Posição 2: Linha 1, Coluna 2)
        # Usa plt.subplot(2, 3, 2)
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(t, X, 'g-')
        ax2.set_title('Biomassa (X)') # Traduzido
        ax2.set_ylabel('[g/L]')
        ax2.set_xlabel('Tempo [h]') # Traduzido
        ax2.grid(True)
        add_phase_lines(ax2, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 3. Substrato (S) (Posição 3: Linha 1, Coluna 3)
        # Usa plt.subplot(2, 3, 3)
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(t, S, 'm-')
        ax3.set_title('Substrato (S)') # Traduzido
        ax3.set_ylabel('[g/L]')
        ax3.set_xlabel('Tempo [h]') # Traduzido
        ax3.grid(True); ax3.set_ylim(bottom=0)
        add_phase_lines(ax3, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 4. Etanol (P) (Posição 4: Linha 2, Coluna 1)
        # Usa plt.subplot(2, 3, 4)
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(t, P, 'k-')
        ax4.set_title('Etanol (P)') # Título já estava ok
        ax4.set_ylabel('[g/L]')
        ax4.set_xlabel('Tempo [h]') # Traduzido
        ax4.grid(True); ax4.set_ylim(bottom=0)
        add_phase_lines(ax4, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 5. Oxigênio Dissolvido (O2) (Posição 5: Linha 2, Coluna 2)
        # Usa plt.subplot(2, 3, 5)
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(t, O2, 'c-')
        ax5.set_title('Oxigênio Dissolvido (O2)') # Traduzido
        ax5.set_ylabel('[mg/L]')
        ax5.set_xlabel('Tempo [h]') # Traduzido
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
            if params["tipo_mu"] == "Fermentación":
                mu_i = mu_fermentacion(si, pi, o2i, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params["n_p"], params["KO_inhib_anaerob"])
            elif params["tipo_mu"] == "Fermentación Conmutada":
                if fase_i == "inicial_batch":
                    current_mumax_i = params.get("mumax_aero", 0.0); current_O2_for_mu_i = params.get("O2_controlado", o2i)
                    mu_i = mu_fermentacion(si, pi, current_O2_for_mu_i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=True)
                else: # fed_or_final_batch
                    current_mumax_i = params.get("mumax_anaero", 0.0)
                    mu_i = mu_fermentacion(si, pi, o2i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=False)
            elif params["tipo_mu"] == "Monod simple":
                mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
            elif params["tipo_mu"] == "Monod sigmoidal":
                mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
            elif params["tipo_mu"] == "Monod con restricciones":
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
            print(f"Aviso: Discrepância no tamanho de t ({len(t)}) e mu_sim ({len(mu_sim)}). Plot pode estar incompleto.")
            # Tenta plotar mesmo assim se mu_sim não estiver vazio
            if mu_sim:
                ax6.plot(t[:len(mu_sim)], mu_sim, 'y-')


        # Usando LaTeX para a letra grega mu para melhor renderização
        ax6.set_title(r'Taxa Específica de Crescimento ($\mu$)') # Traduzido e com LaTeX
        ax6.set_ylabel('[1/h]')
        ax6.set_xlabel('Tempo [h]') # Traduzido
        ax6.grid(True); ax6.set_ylim(bottom=0)
        add_phase_lines(ax6, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # Remover o subplot ax7 (Marcadores de Fase) que estava em (4, 2, 8)

        # Ajusta o layout para evitar sobreposições
        plt.tight_layout(pad=3.0) # Padding ligeiramente aumentado

        # --- COMANDO ESSENCIAL PARA EXIBIR O GRÁFICO NO STREAMLIT ---
        st.pyplot(fig) # <--- Esta linha mostra o gráfico 'fig' na interface do Streamlit

        # --- Métricas Finais (Tradução PT-BR e usando Streamlit) ---
        # Mantém a estrutura original com st.subheader e st.columns/st.metric
        st.subheader("Métricas Finais (t = {:.1f} h)".format(t[-1] if len(t) > 0 else 0))

        # Adiciona checagens para evitar erros se as listas/arrays estiverem vazios
        col1, col2, col3 = st.columns(3)
        vol_final = V[-1] if V.size > 0 else V0
        etanol_final_conc = P[-1] if len(P) > 0 else P0
        biomasa_final_conc = X[-1] if len(X) > 0 else X0
        S_final_total = (S[-1] * V[-1]) if len(S) > 0 and len(V) > 0 else (S0 * V0)
        
        etanol_final_total = etanol_final_conc * vol_final
        
        

        # Recalcula S_alimentado_total com checagem de tamanho e existência de Sin
        S_alimentado_total = 0
        if len(t) > 0 and len(flujo_sim) > 0 and 'Sin' in locals():
            # Garante que fluxo_sim e Sin tenham o mesmo tamanho que t para trapz
            len_t = len(t)
            if len(flujo_sim) == len_t:
                flujo_sim_corr = flujo_sim
            else:
                # Trata caso de tamanho incorreto (pode precisar de ajuste específico)
                print(f"Aviso: Tamanho de flujo_sim ({len(flujo_sim)}) diferente de t ({len_t}).")
                flujo_sim_corr = flujo_sim[:len_t] if len(flujo_sim) > len_t else np.pad(flujo_sim, (0, len_t - len(flujo_sim)))


            if isinstance(Sin, (int, float)): # Se Sin for um número
                S_alimentado_total = Sin * np.trapz(flujo_sim_corr, t)
            elif isinstance(Sin, (list, np.ndarray)): # Se Sin for array/lista
                if len(Sin) == len_t:
                    S_alimentado_total = np.trapz(np.array(flujo_sim_corr) * np.array(Sin), t)
                else:
                    print(f"Aviso: Tamanho de Sin ({len(Sin)}) diferente de t ({len_t}). Cálculo de S_alimentado_total pode estar incorreto.")
                    # Tenta usar Sin escalar se possível ou os primeiros elementos
                    if len(Sin) > 0:
                        S_alimentado_total = np.trapz(np.array(flujo_sim_corr) * Sin[0], t) # Exemplo: usa o primeiro valor de Sin

        S_final_total = (S[-1] * V[-1]) if len(S) > 0 and len(V) > 0 else (S0 * V0)
        S_inicial_total = S0 * V0
        S_consumido_total = S_inicial_total + S_alimentado_total - S_final_total

        col1.metric("Volume Final [L]", f"{vol_final:.2f}") # Traduzido
        col2.metric("Etanol Final [g/L]", f"{etanol_final_conc:.2f}") # Traduzido
        col3.metric("Biomassa Final [g/L]", f"{biomasa_final_conc:.2f}") # Traduzido

        prod_vol_etanol = etanol_final_conc / t[-1] if len(t) > 0 and t[-1] > 0 else 0
        col1.metric("Produtividade Vol. Etanol [g/L/h]", f"{prod_vol_etanol:.3f}") # Traduzido
        rend_global_etanol = (etanol_final_total - P0 * V0) / S_consumido_total if S_consumido_total > 1e-6 else 0
        col2.metric("Rendimento Global P/S [g/g]", f"{rend_global_etanol:.3f}") # Traduzido
        try:
            if len(P) > 0:  # Verifica se P não está vazio
                p_max_idx = np.argmax(P)
                col3.metric("Etanol Máx [g/L]", f"{P[p_max_idx]:.2f} (em t={t[p_max_idx]:.1f} h)") # Traduzido
            else:
                col3.metric("Etanol Máx [g/L]", "N/A") # Traduzido
        except (ValueError, IndexError) as e:  # Captura possíveis erros de índice também
            print(f"Erro ao calcular Etanol Máx: {e}")
            col3.metric("Etanol Máx [g/L]", "N/A") # Traduzido


    except Exception as e:
        st.error(f"Ocurrió un error durante la simulación o el procesamiento de resultados: {e}")
        import traceback
        st.error(traceback.format_exc())

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Simulador Fermentación Alcohólica")
    fermentacion_alcoholica_page()