# ferm_alcohol.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Importar TODAS las funciones cinéticas, incluyendo la nueva
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion, mu_fermentacion

def fermentacion_alcoholica_page():
    st.header("Simulación de Fermentación Alcohólica en Lote Alimentado")
    st.markdown("""
    Este modelo simula una fermentación alcohólica que inicia en **lote (fase aeróbica)**,
    continúa en **lote alimentado (fase de transición/anaeróbica)**, y finaliza
    en **lote (fase anaeróbica)** para agotar el sustrato. Seleccione el modelo cinético.
    """)

    with st.sidebar:
        st.subheader("1. Modelo Cinético y Parámetros")
        tipo_mu = st.selectbox("Modelo Cinético", ["Fermentación",  "Monod simple", "Monod sigmoidal", "Monod con restricciones"])

        Ks = st.slider("Ks base [g/L]", 0.01, 10.0, 1.0, 0.1)

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

        st.subheader("3. Transferencia de Oxígeno")
        Kla = st.slider("kLa [1/h]", 10.0, 400.0, 100.0, 10.0, key="kla")
        Cs = st.slider("O2 Saturado (Cs) [mg/L]", 0.01, 15.0, 7.5, 0.01, key="cs")

        st.subheader("4. Fases de Operación y Alimentación")
        t_batch_inicial_fin = st.slider("Fin Fase Lote Inicial [h]", 1.0, 48.0, 10.0, 1.0, key="t_batch_fin")
        t_alim_inicio = st.slider("Inicio Alimentación [h]", t_batch_inicial_fin, t_batch_inicial_fin + 24.0, t_batch_inicial_fin + 0.1, 0.5, key="t_alim_ini")
        t_alim_fin = st.slider("Fin Alimentación (Inicio Lote Final) [h]", t_alim_inicio + 1.0, 96.0, t_alim_inicio + 24.0, 1.0, key="t_alim_fin")
        t_final = st.slider("Tiempo Total de Simulación [h]", t_alim_fin + 1.0, 150.0, t_alim_fin + 12.0, 1.0, key="t_total")
        # CAMBIO: Valor por defecto de O2_controlado
        O2_controlado = st.slider("Nivel O2 Objetivo/Ref (Fase Lote Inicial) [mg/L]", 0.01, Cs, 0.08, 0.01, key="o2_control") # Valor 0.08

        estrategia = st.selectbox("Estrategia Alimentación", ["Constante", "Exponencial", "Lineal", "Escalon"], key="strat")
        Sin = st.slider("Sustrato en Alimentación (Sin) [g/L]", 50.0, 700.0, 400.0, 10.0, key="sin")
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
        # CAMBIO: Valor por defecto de O0
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
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constante": return F_base
            elif estrategia == "Exponencial":
                try: return min(F_base * np.exp(k_exp_val * (t - t_alim_inicio)), F_base * 100)
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
    }
    if tipo_mu == "Monod simple": params["mumax"] = mumax
    elif tipo_mu == "Monod sigmoidal": params["mumax"] = mumax; params["n_sig"] = n_sig
    elif tipo_mu == "Monod con restricciones": params["mumax"] = mumax; params["KO"] = KO_restr; params["KP_gen"] = KP_gen
    elif tipo_mu == "Fermentación Conmutada":
        params["mumax_aero"] = mumax_aero_c; params["mumax_anaero"] = mumax_anaero_c
        params["KiS"] = KiS_c; params["KP"] = KP_c; params["n_p"] = n_p_c; params["KO"] = KO_ferm_c
        params["O2_controlado"] = O2_controlado # Necesario para este modelo
    elif tipo_mu == "Fermentación":
        params["mumax_aerob"] = mumax_aerob_m; params["Ks_aerob"] = Ks_aerob_m; params["KO_aerob"] = KO_aerob_m
        params["mumax_anaerob"] = mumax_anaerob_m; params["Ks_anaerob"] = Ks_anaerob_m; params["KiS_anaerob"] = KiS_anaerob_m
        params["KP_anaerob"] = KP_anaerob_m; params["n_p"] = n_p_m; params["KO_inhib_anaerob"] = KO_inhib_anaerob_m

    # ------- Modelo ODE -------
    def modelo_fermentacion(t, y, params):
        X, S, P, O2, V = y
        X = max(1e-9, X); S = max(0, S); P = max(0, P); O2 = max(0, O2); V = max(1e-6, V)

        # Determinar la fase actual
        fase = "inicial_batch" if t < params["t_batch_inicial_fin"] else "fed_batch" if t < t_alim_fin else "final_batch"
        # O más simple para la lógica de O2:
        es_lote_inicial = (t < params["t_batch_inicial_fin"])

        # --- Cálculo de mu ---
        mu = 0.0
        # (Lógica de cálculo de mu sin cambios respecto a la versión anterior)
        if params["tipo_mu"] == "Fermentación":
             mu = mu_fermentacion(S, P, O2, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params["n_p"], params["KO_inhib_anaerob"])
        elif params["tipo_mu"] == "Fermentación Conmutada":
             if es_lote_inicial:
                 current_mumax = params.get("mumax_aero", 0.0)
                 current_O2_for_mu = params.get("O2_controlado", O2)
                 mu = mu_fermentacion(S, P, current_O2_for_mu, current_mumax, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=True)
             else:
                 current_mumax = params.get("mumax_anaero", 0.0)
                 mu = mu_fermentacion(S, P, O2, current_mumax, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=False)
        elif params["tipo_mu"] == "Monod simple": mu = mu_monod(S, params.get("mumax", 0.0), params["Ks"])
        elif params["tipo_mu"] == "Monod sigmoidal": mu = mu_sigmoidal(S, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
        elif params["tipo_mu"] == "Monod con restricciones":
             current_O2_for_mu = O2 # Usar O2 simulado
             mu = mu_completa(S, current_O2_for_mu, P, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))
        mu = max(0, mu)

        # --- Cálculo de Flujo y Derivadas ---
        F = calcular_flujo(t)
        qP = params["alpha_lp"] * mu + params["beta_lp"]
        rate_P = qP * X
        consumo_S_X = (mu / params["Yxs"]) * X if params["Yxs"] > 1e-6 else 0
        consumo_S_P = (rate_P / params["Yps"]) if params["Yps"] > 1e-6 else 0
        consumo_S_maint = params["ms"] * X
        rate_S = consumo_S_X + consumo_S_P + consumo_S_maint
        consumo_O2_X = (mu / params["Yxo"]) * X if params["Yxo"] > 1e-6 else 0
        consumo_O2_maint = params["mo"] * X
        OUR_g = consumo_O2_X + consumo_O2_maint
        OUR_mg = OUR_g * 1000.0

        # Ecuaciones Diferenciales
        dXdt = (mu - params["Kd"]) * X - (F / V) * X
        dSdt = -rate_S + (F / V) * (params["Sin"] - S)
        dPdt = rate_P - (F / V) * P
        dVdt = F

        # --- Cálculo de dOdt (MODIFICADO) ---
        if es_lote_inicial:
            dOdt = 0.0 # Forzar derivada a cero en la fase inicial
        else:
            # Calcular normalmente en fases posteriores
            OTR = params["Kla"] * (params["Cs"] - O2) # [mg O2 / L / h]
            dOdt = OTR - OUR_mg - (F / V) * O2      # [mg/L/h]

        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    # ------- Simulación y Resultados -------
    y0 = [X0, S0, P0, O0, V0]
    t_span = [0, t_final]
    t_eval = np.linspace(t_span[0], t_span[1], int(t_final * 20)+1)

    try:
        # (Bloque de solve_ivp con intento de BDF si falla RK45 - sin cambios)
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
        O2 = np.maximum(O2, 0)
        flujo_sim = np.array([calcular_flujo(ti) for ti in t])

        # --- Graficación (AÑADIR LÍNEAS DE FASE) ---
        st.subheader("Resultados de la Simulación")
        fig = plt.figure(figsize=(14, 16))

        # Función auxiliar para añadir líneas de fase
        def add_phase_lines(ax, t_batch, t_feed_start, t_feed_end):
            ax.axvline(t_batch, color='gray', linestyle='--', lw=1.5, label='Fin Lote Inicial' if ax.get_label() == 'ax1' else "") # Label solo una vez
            ax.axvline(t_feed_start, color='orange', linestyle='--', lw=1.5, label='Inicio Alimentación' if ax.get_label() == 'ax1' else "")
            ax.axvline(t_feed_end, color='purple', linestyle='--', lw=1.5, label='Fin Alimentación' if ax.get_label() == 'ax1' else "")

        # 1. Flujo y Volumen
        ax1 = plt.subplot(4, 2, 1, label='ax1') # Añadir label para la leyenda de líneas
        color = 'tab:red'; ax1.plot(t, flujo_sim, color=color, label='Flujo Alimentación'); ax1.set_ylabel('Flujo [L/h]', color=color); ax1.tick_params(axis='y', labelcolor=color)
        ax1b = ax1.twinx(); color = 'tab:blue'; ax1b.plot(t, V, color=color, linestyle='-', label='Volumen'); ax1b.set_ylabel('Volumen [L]', color=color); ax1b.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Tiempo [h]'); ax1.grid(True); ax1.set_title('Alimentación y Volumen');
        add_phase_lines(ax1, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)
        # Combinar leyendas
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        # Añadir labels de axvline (tomados de ax1 donde se definieron)
        line_labels = [l for l in ax1.lines if l.get_linestyle() == '--'] # Obtener las vlines
        label_vlines = [l.get_label() for l in line_labels if l.get_label()] # Obtener sus labels no vacíos
        ax1b.legend(lines + lines2 + line_labels, labels + labels2 + label_vlines, loc='best')


        # 2. Biomasa (X)
        ax2 = plt.subplot(4, 2, 3); ax2.plot(t, X, 'g-'); ax2.set_title('Biomasa (X)'); ax2.set_ylabel('[g/L]'); ax2.set_xlabel('Tiempo [h]'); ax2.grid(True)
        add_phase_lines(ax2, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 3. Sustrato (S)
        ax3 = plt.subplot(4, 2, 4); ax3.plot(t, S, 'm-'); ax3.set_title('Sustrato (S)'); ax3.set_ylabel('[g/L]'); ax3.set_xlabel('Tiempo [h]'); ax3.grid(True); ax3.set_ylim(bottom=0)
        add_phase_lines(ax3, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 4. Etanol (P)
        ax4 = plt.subplot(4, 2, 5); ax4.plot(t, P, 'k-'); ax4.set_title('Etanol (P)'); ax4.set_ylabel('[g/L]'); ax4.set_xlabel('Tiempo [h]'); ax4.grid(True); ax4.set_ylim(bottom=0)
        add_phase_lines(ax4, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 5. Oxígeno Disuelto (O2)
        ax5 = plt.subplot(4, 2, 6); ax5.plot(t, O2, 'c-'); ax5.set_title('Oxígeno Disuelto (O2)'); ax5.set_ylabel('[mg/L]'); ax5.set_xlabel('Tiempo [h]'); ax5.grid(True); ax5.set_ylim(bottom=-0.1, top=Cs*1.1);
        # ax5.axhline(O2_controlado, color='r', linestyle=':', lw=1, label=f'O2 Obj/Ref ({O2_controlado} mg/L)') # Opcional mostrar referencia
        add_phase_lines(ax5, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)
        #ax5.legend(loc='best')

        # 6. Tasa de Crecimiento Específica (mu) - Recalcular post-simulación
        # (Lógica de cálculo de mu_sim sin cambios)
        mu_sim = []
        for i in range(len(t)):
            ti, xi, si, pi, o2i, vi = t[i], X[i], S[i], P[i], O2[i], V[i]
            fase_i = "inicial_batch" if ti < params.get("t_batch_inicial_fin", float('inf')) else "fed_or_final_batch"
            mu_i = 0.0
            if params["tipo_mu"] == "Fermentación":
                 mu_i = mu_fermentacion(si, pi, o2i, params["mumax_aerob"], params["Ks_aerob"], params["KO_aerob"], params["mumax_anaerob"], params["Ks_anaerob"], params["KiS_anaerob"], params["KP_anaerob"], params["n_p"], params["KO_inhib_anaerob"])
            elif params["tipo_mu"] == "Fermentación Conmutada":
                 if fase_i == "inicial_batch":
                     current_mumax_i = params.get("mumax_aero", 0.0); current_O2_for_mu_i = params.get("O2_controlado", o2i)
                     mu_i = mu_fermentacion(si, pi, current_O2_for_mu_i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=True)
                 else:
                     current_mumax_i = params.get("mumax_anaero", 0.0)
                     mu_i = mu_fermentacion(si, pi, o2i, current_mumax_i, params["Ks"], params.get("KiS", float('inf')), params.get("KP", float('inf')), params.get("n_p", 1.0), params.get("KO", 0.1), considerar_O2=False)
            elif params["tipo_mu"] == "Monod simple": mu_i = mu_monod(si, params.get("mumax", 0.0), params["Ks"])
            elif params["tipo_mu"] == "Monod sigmoidal": mu_i = mu_sigmoidal(si, params.get("mumax", 0.0), params["Ks"], params.get("n_sig", 1))
            elif params["tipo_mu"] == "Monod con restricciones":
                 current_O2_for_mu_i = o2i
                 mu_i = mu_completa(si, current_O2_for_mu_i, pi, params.get("mumax", 0.0), params["Ks"], params.get("KO", 0.1), params.get("KP_gen", 50.0))
            mu_sim.append(max(0, mu_i))

        ax6 = plt.subplot(4, 2, 7); ax6.plot(t, mu_sim, 'y-'); ax6.set_title('Tasa Crecimiento Específica (μ)'); ax6.set_ylabel('[1/h]'); ax6.set_xlabel('Tiempo [h]'); ax6.grid(True); ax6.set_ylim(bottom=0)
        add_phase_lines(ax6, t_batch_inicial_fin, t_alim_inicio, t_alim_fin)

        # 7. Marcadores de Fase (Actualizado)
        ax7 = plt.subplot(4, 2, 8); ax7.axis('off'); ax7.set_title('Fases Operativas')
        ax7.text(0.05, 0.8, f"1. Lote Inicial: 0 - {t_batch_inicial_fin:.1f} h (hasta línea gris)", transform=ax7.transAxes)
        ax7.text(0.05, 0.6, f"2. Lote Alimentado: {t_alim_inicio:.1f} - {t_alim_fin:.1f} h (naranja a púrpura)", transform=ax7.transAxes)
        ax7.text(0.05, 0.4, f"3. Lote Final: > {t_alim_fin:.1f} h (después de línea púrpura)", transform=ax7.transAxes)
        ax7.text(0.05, 0.2, f"O2 Inicial: {O0:.2f} mg/L (dOdt=0 en Fase 1)", transform=ax7.transAxes)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig)

        # --- Métricas Clave (sin cambios funcionales) ---
        st.subheader("Métricas Finales (t = {:.1f} h)".format(t[-1]))
        # ... (código métricas) ...
        col1, col2, col3 = st.columns(3)
        vol_final = V[-1] if len(V)>0 else V0; etanol_final_conc = P[-1] if len(P)>0 else P0
        etanol_final_total = etanol_final_conc * vol_final; biomasa_final_conc = X[-1] if len(X)>0 else X0
        S_inicial_total = S0 * V0
        if len(t) > 1: S_alimentado_total = np.trapz(flujo_sim * Sin, t)
        else: S_alimentado_total = 0
        S_final_total = (S[-1] * V[-1]) if len(S)>0 else (S0*V0); S_consumido_total = S_inicial_total + S_alimentado_total - S_final_total
        col1.metric("Volumen Final [L]", f"{vol_final:.2f}"); col2.metric("Etanol Final [g/L]", f"{etanol_final_conc:.2f}"); col3.metric("Biomasa Final [g/L]", f"{biomasa_final_conc:.2f}")
        prod_vol_etanol = etanol_final_conc / t[-1] if len(t)>0 and t[-1] > 0 else 0; col1.metric("Productividad Vol. Etanol [g/L/h]", f"{prod_vol_etanol:.3f}")
        rend_global_etanol = (etanol_final_total - P0*V0) / S_consumido_total if S_consumido_total > 1e-6 else 0; col2.metric("Rendimiento Global P/S [g/g]", f"{rend_global_etanol:.3f}")
        try:
             p_max_idx = np.argmax(P) if len(P)>0 else -1
             if p_max_idx >= 0: col3.metric("Etanol Máx [g/L]", f"{P[p_max_idx]:.2f} (a t={t[p_max_idx]:.1f} h)")
             else: col3.metric("Etanol Máx [g/L]", "N/A")
        except ValueError: col3.metric("Etanol Máx [g/L]", "N/A")


    except Exception as e:
        st.error(f"Ocurrió un error durante la simulación o el procesamiento de resultados: {e}")
        import traceback
        st.error(traceback.format_exc())

# --- Ejecución de la Aplicación ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Simulador Fermentación Alcohólica")
    fermentacion_alcoholica_page()