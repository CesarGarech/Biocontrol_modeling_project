# -*- coding: utf-8 -*-
import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch # Asegúrate que Patch está importado

# ====================================================
# --- Definiciones Cinéticas (SIN CAMBIOS desde la versión anterior) ---
# ====================================================
def mu_aerobic(S, O2, mu_max_aer, Ks_aerob, KO_aerob):
    safe_den_S = ca.fmax(Ks_aerob + S, 1e-9)
    safe_den_O2 = ca.fmax(KO_aerob + O2, 1e-9)
    term_S = S / safe_den_S
    term_O2 = O2 / safe_den_O2
    mu = mu_max_aer * term_S * term_O2
    return ca.fmax(mu, 0.0)

def mu_anaerobic(S, P, O2, mu_max_an, Ks_an, KiS_an, KP_an, n_p, KO_inhib_an):
    if isinstance(KiS_an, (int, float)) and KiS_an < 1e-9:
         den_S = Ks_an + S
    else:
        safe_KiS_an = ca.fmax(KiS_an, 1e-9)
        den_S = Ks_an + S + (S**2 / safe_KiS_an)
    safe_den_S = ca.fmax(den_S, 1e-9)
    term_S = S / safe_den_S
    safe_KP_an = ca.fmax(KP_an, 1e-9)
    term_P_base = 1.0 - (P / safe_KP_an)
    safe_term_P_base = ca.fmax(0.0, term_P_base)
    term_P = ca.if_else(P < KP_an, safe_term_P_base**n_p, 0.0)
    term_P = ca.fmax(0.0, term_P)
    safe_den_O2_inhib = ca.fmax(KO_inhib_an + O2, 1e-9)
    term_O2_inhib = KO_inhib_an / safe_den_O2_inhib
    mu = mu_max_an * term_S * term_P * term_O2_inhib
    return ca.fmax(mu, 0.0)

# --- Página de Streamlit ---
def rto_fermentation_page ():
    st.header("🧠 Control RTO - Fermentación Alcohólica (3 Fases con Modelos Simples)")
    st.markdown("""
    Optimización del perfil de alimentación para maximizar la **productividad volumétrica final ($P_{final} V_{final}$)**,
    con penalización por alta concentración de sustrato ($S > S_{max}$) durante la alimentación.
    Se usan modelos cinéticos simplificados y balance de oxígeno dinámico en fase anaerobia.
    Fases:
    1. Fase Aerobia Batch (Crecimiento $\mu_1$)
    2. Fase Anaerobia Fed-Batch (Control Opt., Crecimiento $\mu_2$, consumo O2 residual)
    3. Fase Anaerobia Batch (Post-Feed, Crecimiento $\mu_2$, consumo O2 residual)
    """)

    with st.sidebar:
        st.subheader("📌 Parámetros Cinéticos Aerobios ($\mu_1$)")
        mu_max_aer = st.number_input("μmax (Aerobio) [1/h]", value=0.4, min_value=0.01, key="mu_max_aer")
        Ks_aerob = st.number_input("Ks (Aerobio) [g/L]", value=0.5, min_value=0.01, key="ks_aerob")
        KO_aerob = st.number_input("KO (Afinidad O2 - Aerobio) [g/L]", value=0.002, min_value=0.0001, format="%.4f", key="ko_aerob")
        Yxs_aer = st.number_input("Yxs (Aerobio) [g/g]", value=0.5, min_value=0.1, max_value=1.0, key="yxs_aer")
        Yos_aer = st.number_input("Yos (Consumo O2/Sustrato - Aerobio) [g O2/g S]", value=0.8, min_value=0.0, key="yos_aer")

        st.subheader("📌 Parámetros Cinéticos Anaerobios ($\mu_2$)")
        mu_max_an = st.number_input("μmax (Anaerobio) [1/h]", value=0.3, min_value=0.01, key="mu_max_an")
        Ks_anaerob = st.number_input("Ks (Anaerobio) [g/L]", value=1.0, min_value=0.01, key="ks_anaerob")
        KiS_anaerob = st.number_input("KiS (Inhibición Sustrato - Anaerobio) [g/L]", value=150.0, min_value=1.0, key="kis_anaerob")
        KP_anaerob = st.number_input("KP (Inhibición Producto - Anaerobio) [g/L]", value=90.0, min_value=10.0, key="kp_anaerob")
        n_p = st.number_input("n_p (Exponente Inhibición Producto)", value=1.0, min_value=0.1, max_value=3.0, key="n_p")
        KO_inhib_anaerob = st.number_input("KO_inhib (Inhibición O2 - Anaerobio) [g/L]", value=0.001, min_value=1e-6, format="%.6f", key="ko_inhib_anaerob")
        Yxs_an = st.number_input("Yxs (Anaerobio) [g/g]", value=0.1, min_value=0.01, max_value=1.0, key="yxs_an")
        Yps_an = st.number_input("Yps (Anaerobio - Etanol) [g/g]", value=0.45, min_value=0.1, max_value=0.51, key="yps_an")
        Yos_an = st.number_input("Yos (Consumo O2/Sustrato - Anaerobio) [g O2/g S]", value=0.01, min_value=0.0, key="yos_an")

        st.subheader("💨 Transferencia de Oxígeno")
        kla = st.number_input("KLa (Aerobio - Fase 1) [1/h]", value=550.0, min_value=0.0, key="kla")
        kla_an = st.number_input("KLa (Anaerobio - Fase 2 y 3) [1/h]", value=0.001, min_value=0.000,format="%.4f", key="kla_an")
        O_sat = st.number_input("O2 Saturación [g/L]", value=0.08, min_value=0.01, max_value=0.1, format="%.4f", key="o_sat")

        st.subheader("💧 Alimentación y Reactor")
        Sf_input = st.number_input("Concentración del alimentado Sf [g/L]", value=250.0, key="sf")
        V_max_input = st.number_input("Volumen máximo del reactor [L]", value=0.5, key="vmax")

        st.subheader("🎚 Condiciones Iniciales (t=0)")
        X0 = st.number_input("X0 (Biomasa) [g/L]", value=1.16, key="x0")
        S0 = st.number_input("S0 (Sustrato) [g/L]", value=10.17, key="s0")
        P0 = st.number_input("P0 (Etanol) [g/L]", value=0.0, key="p0")
        O0 = st.number_input("O0 (Oxígeno disuelto inicial) [g/L]", value=0.08, min_value=0.0, max_value=O_sat, format="%.4f", key="o0")
        V0 = st.number_input("V0 (Volumen inicial) [L]", value=0.25, key="v0")

        st.subheader("⏳ Configuración Temporal")
        t_aerobic_batch = st.number_input("Tiempo Fase Aerobia Batch [h]", value=3.0, min_value=1.0, key="t_aerobic")
        t_anaerobic_feed_end = st.number_input("Tiempo Fin Alimentación Anaerobia [h]", value=8.0, min_value=t_aerobic_batch + 1.0, key="t_feed_end")
        t_total = st.number_input("Tiempo total del proceso [h]", value=10.0, min_value=t_anaerobic_feed_end + 0.1, key="t_total")
        n_fb_intervals = st.number_input("Número de Intervalos de Control (Fase 2)", value=12, min_value=1, key="n_intervals", help=f"Duración Fase 2: {t_anaerobic_feed_end - t_aerobic_batch:.1f} h")

        st.subheader("🔧 Restricciones y Penalización")
        F_min = st.number_input("Flujo mínimo [L/h]", value=0.01, min_value=0.0, key="fmin")
        F_max = st.number_input("Flujo máximo [L/h]", value=0.26, min_value=F_min, key="fmax")
        S_max_constraint = st.number_input("Sustrato máximo (Restricción Dura) [g/L]", value=50.0, key="smax_const")
        pmax_constraint_default = max(10.0, KP_anaerob - 1.0) if KP_anaerob > 11.0 else KP_anaerob * 0.9
        P_max_constraint = st.number_input("Producto máximo (Restricción Dura) [g/L]", value=pmax_constraint_default, key="pmax_const", help="Debe ser menor que KP de inhibición")
        w_penalty_smax = st.number_input("Peso Penalización S > Smax", value=10.0, min_value=0.0, key="w_smax", format="%.2f", help="Penaliza S por encima de Smax (restricción dura) en Fase 2. 0 para desactivar.")


    if st.button("🚀 Ejecutar Optimización RTO (Estabilizado)"):
        st.info("Optimizando perfil de alimentación para fermentación...")

        T_feed_duration = t_anaerobic_feed_end - t_aerobic_batch
        if T_feed_duration <= 0: st.error("Tiempo fin feed < tiempo aerobio"); st.stop()
        if n_fb_intervals <= 0: st.error("Intervalos deben ser > 0"); st.stop()
        dt_fb = T_feed_duration / n_fb_intervals
        T_post_feed_duration = t_total - t_anaerobic_feed_end
        if T_post_feed_duration < -1e-6: st.error("Tiempo total < tiempo fin feed"); st.stop()

        nx = 5 # X, S, P, O2, V
        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u") # Flujo F
        X_, S_, P_, O2_, V_ = x_sym[0], x_sym[1], x_sym[2], x_sym[3], x_sym[4]
        F_ = u_sym

        # ====================================================
        # 1) Definición de las funciones ODE (MODIFICADO dO2_an)
        # ====================================================

        # --- ODE Fase 1: Aerobia Batch ---
        mu1 = mu_aerobic(S_, O2_, mu_max_aer, Ks_aerob, KO_aerob)
        qS_aer = mu1 / Yxs_aer if Yxs_aer > 1e-9 else 0
        dX_aer = mu1 * X_
        dS_aer = -qS_aer * X_
        dP_aer = ca.MX(0.0)
        OUR_aer = (Yos_aer * qS_aer) * X_
        OTR_aer = kla * (O_sat - O2_) # Usa kla aerobio
        dO2_aer = OTR_aer - OUR_aer
        dV_aer = ca.MX(0.0)
        ode_expr_aerobic = ca.vertcat(dX_aer, dS_aer, dP_aer, dO2_aer, dV_aer)
        odefun_aerobic = ca.Function('odefun_aerobic', [x_sym], [ode_expr_aerobic], ['x'], ['dxdt'])

        # --- ODE Fase 2 y 3: Anaerobia (Fed-Batch / Batch) ---
        mu2 = mu_anaerobic(S_, P_, O2_, mu_max_an, Ks_anaerob, KiS_anaerob, KP_anaerob, n_p, KO_inhib_anaerob)
        qS_an = mu2 / Yxs_an if Yxs_an > 1e-9 else 0
        qP_an = Yps_an * qS_an
        D = F_ / ca.fmax(V_, 1e-6)
        dX_an = mu2 * X_ - D * X_
        dS_an = -qS_an * X_ + D * (Sf_input - S_)
        dP_an = qP_an * X_ - D * P_

        # *** Balance O2 Anaerobio MODIFICADO con Estabilización ***
        OUR_an = (Yos_an * qS_an) * X_
        # Protección numérica para OTR
        safe_O2_for_OTR = ca.fmax(1e-9, O2_) # Evita usar exactamente 0 o negativo en OTR
        OTR_an = kla_an * (O_sat - safe_O2_for_OTR) # Usa kla_an
        # Protección numérica para OUR (Opción 2: Monod en OUR para O2 bajo)
        Ko_our = 1e-5 # Constante de afinidad (muy pequeña) para el consumo residual de O2
        term_O2_our = O2_ / ca.fmax(1e-9, (Ko_our + O2_)) # Proteger denominador interno
        effective_OUR_an = OUR_an * ca.fmax(0.0, term_O2_our) # OUR disminuye si O2 es muy bajo
        # Cálculo final de dO2/dt
        dO2_an = OTR_an - effective_OUR_an - D * O2_

        dV_an = F_
        ode_expr_anaerobic = ca.vertcat(dX_an, dS_an, dP_an, dO2_an, dV_an)
        odefun_anaerobic = ca.Function('odefun_anaerobic', [x_sym, u_sym], [ode_expr_anaerobic], ['x', 'u'], ['dxdt'])

        # ====================================================
        # 2) Simulación Fase 1: Aerobia Batch (SIN CAMBIOS)
        # ====================================================
        st.info(f"[FASE 1] Simulando fase aerobia batch hasta t={t_aerobic_batch} h...")
        try:
            batch_integrator_aerobic = ca.integrator("batch_integrator_aerobic", "idas",
                                                     {"x": x_sym, "ode": ode_expr_aerobic},
                                                     {"t0": 0, "tf": t_aerobic_batch, "reltol": 1e-7, "abstol": 1e-9})
            x0_np = np.array([X0, S0, P0, O0, V0])
            res_batch_aerobic = batch_integrator_aerobic(x0=x0_np)
            x_end_aerobic = np.array(res_batch_aerobic['xf']).flatten()
            st.success(f"[FASE 1] Estado final: X={x_end_aerobic[0]:.2f}, S={x_end_aerobic[1]:.2f}, P={x_end_aerobic[2]:.2f}, O2={x_end_aerobic[3]:.4f}, V={x_end_aerobic[4]:.2f}")
            if any(np.isnan(x_end_aerobic)) or any(x_end_aerobic < -1e-9): st.error(f"Estado inválido Fase 1: {x_end_aerobic}."); st.stop()
            x_end_aerobic = np.maximum(x_end_aerobic, 0.0)
        except Exception as e:
            st.error(f"Error simulación Fase 1: {e}"); st.exception(e); st.stop()

        # ====================================================
        # 3) Formulación Optimización Fase 2 (SIN CAMBIOS ESTRUCTURALES)
        # ====================================================
        st.info(f"[FASE 2] Formulando problema RTO (t={t_aerobic_batch}h a t={t_anaerobic_feed_end}h)...")
        opti = ca.Opti()
        d = 2
        C_radau = np.array([[-2.0, 2.0], [1.5, -4.5], [0.5, 2.5]])
        D_radau = np.array([0.0, 0.0, 1.0])

        X_col_phase2 = []
        F_col_phase2 = []
        x_start_phase2_param = opti.parameter(nx)
        opti.set_value(x_start_phase2_param, x_end_aerobic)

        for k in range(n_fb_intervals):
            row_states = []
            for j in range(d + 1):
                if k == 0 and j == 0:
                    row_states.append(x_start_phase2_param)
                else:
                    xk_j = opti.variable(nx)
                    row_states.append(xk_j)
                    opti.subject_to(xk_j >= -1e-9) # Permitir ligeras negatividades numéricas
                    opti.subject_to(xk_j[1] <= S_max_constraint)
                    opti.subject_to(xk_j[2] <= P_max_constraint)
                    opti.subject_to(xk_j[4] <= V_max_input)
                    # Podríamos añadir opti.subject_to(xk_j[3] >= 0.0) explícitamente si la estabilización no basta
            X_col_phase2.append(row_states)

            Fk = opti.variable()
            F_col_phase2.append(Fk)
            opti.subject_to(Fk >= F_min)
            opti.subject_to(Fk <= F_max)

        h = dt_fb
        penalty_smax_total = ca.MX(0.0)
        Xk_end = None

        for k in range(n_fb_intervals):
            for j in range(1, d + 1):
                xp_kj = 0
                for m in range(d + 1): xp_kj += C_radau[m, j - 1] * X_col_phase2[k][m]
                fkj = odefun_anaerobic(X_col_phase2[k][j], F_col_phase2[k])
                opti.subject_to((h * fkj - xp_kj) == 0)

            Xk_end = 0
            for m in range(d + 1): Xk_end += D_radau[m] * X_col_phase2[k][m]

            if k < n_fb_intervals - 1:
                opti.subject_to(Xk_end == X_col_phase2[k + 1][0])

            if w_penalty_smax > 1e-9:
                s_k_end = Xk_end[1]
                violation_k = ca.fmax(0, s_k_end - S_max_constraint)
                penalty_smax_total += violation_k**2

        if w_penalty_smax > 1e-9:
             penalty_smax_total = w_penalty_smax * penalty_smax_total

        if Xk_end is None and n_fb_intervals > 0: # Asegurar que Xk_end se asigna si hay bucle
             # Esto no debería ocurrir si n_fb_intervals>0, pero por seguridad
              Xk_end = X_col_phase2[n_fb_intervals-1][d] # Usar el último punto como referencia
        elif n_fb_intervals == 0:
              Xk_end = x_start_phase2_param

        X_end_feed = Xk_end


        # ====================================================
        # 4) Simulación Fase 3 (dentro de Opti) (SIN CAMBIOS ESTRUCTURALES)
        # ====================================================
        st.info("[FASE 3 - Integración en Opti] Definiendo simulación post-alimentación...")
        if T_post_feed_duration > 1e-6:
            phase3_integrator = ca.integrator("phase3_integrator", "idas",
                                              {"x": x_sym, "p": u_sym, "ode": ode_expr_anaerobic},
                                              {"t0": 0, "tf": T_post_feed_duration, "reltol": 1e-7, "abstol": 1e-9})
            # Asegurarse que X_end_feed es un estado válido
            res_phase3_sym = phase3_integrator(x0=X_end_feed, p=0.0)
            X_final_total = res_phase3_sym['xf']
        else:
            X_final_total = X_end_feed

        # ====================================================
        # 5) Función Objetivo y Resolución (SIN CAMBIOS ESTRUCTURALES)
        # ====================================================
        P_final_total = X_final_total[2]
        V_final_total = X_final_total[4]
        objective_PV = -(P_final_total * V_final_total)
        objective_total = objective_PV + penalty_smax_total
        opti.minimize(objective_total)

        st.info("Estableciendo guesses iniciales...")
        F_guess = (F_max + F_min) / 2.0 * 0.5
        for k in range(n_fb_intervals): opti.set_initial(F_col_phase2[k], F_guess)
        x_guess = x_end_aerobic.copy()
        v_guess_mid = min(V_max_input, x_guess[4] + F_guess * T_feed_duration / 2)
        x_guess[4] = v_guess_mid
        x_guess[3] = max(1e-7, x_guess[3]) # Asegurar O2 inicial guess no sea 0
        for k in range(n_fb_intervals):
            start_j = 1 if k == 0 else 0
            for j in range(start_j, d + 1): opti.set_initial(X_col_phase2[k][j], x_guess)

        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000, "print_level": 0, "sb": 'yes', "tol": 1e-6, "constr_viol_tol": 1e-6}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            st.info("🚀 Resolviendo el problema de optimización...")
            sol = opti.solve()
            st.success("[OPTIMIZACIÓN] ¡Solución encontrada!")

            F_opt_phase2 = np.array([sol.value(fk) for fk in F_col_phase2])
            X_end_feed_opt = sol.value(X_end_feed)
            X_final_total_opt = sol.value(X_final_total)
            P_final_opt = X_final_total_opt[2]
            V_final_opt = X_final_total_opt[4]
            O2_final_opt = X_final_total_opt[3]
            Smax_penalty_value = sol.value(penalty_smax_total)

            st.metric("Producto Total Final (P*V)", f"{P_final_opt * V_final_opt:.3f} g")
            st.metric("Concentración Final Etanol", f"{P_final_opt:.3f} g/L")
            st.metric("Volumen Final", f"{V_final_opt:.3f} L")
            st.metric("O2 Final", f"{O2_final_opt:.5f} g/L")
            st.metric("Valor Penalización Smax", f"{Smax_penalty_value:.4f}", delta=None, delta_color="off")

            st.write("Perfil óptimo de flujo (Fase 2):")
            t_feed_points = np.linspace(t_aerobic_batch, t_anaerobic_feed_end, n_fb_intervals + 1)
            df_flow = pd.DataFrame({'Tiempo (h)': t_feed_points[:-1], 'Flujo (L/h)': F_opt_phase2})
            st.line_chart(df_flow.set_index('Tiempo (h)'))

        except RuntimeError as e:
            st.error(f"[ERROR] Solver: {e}")
            try:
                st.warning("Debug info:")
                st.write(f"Objective: {opti.debug.value(objective_total):.4f}")
                st.write(f"PV Term: {opti.debug.value(objective_PV):.4f}")
                st.write(f"Smax Penalty: {opti.debug.value(penalty_smax_total):.4f}")
            except Exception as debug_e: st.error(f"Error debug: {debug_e}")
            st.stop()
        except Exception as e:
            st.error(f"Error optimización: {e}"); st.exception(e); st.stop()

        # ====================================================
        # 6) Reconstrucción de la Trayectoria Completa (SIN CAMBIOS ESTRUCTURALES)
        # ====================================================
        st.info("Reconstruyendo trayectoria completa con perfil óptimo...")
        # --- a) Simulación fina Fase 1 ---
        N_plot_phase1 = 50
        t_plot_phase1 = np.linspace(0, t_aerobic_batch, N_plot_phase1)
        dt_plot_p1 = t_plot_phase1[1] - t_plot_phase1[0] if N_plot_phase1 > 1 else t_aerobic_batch
        integrator_p1_plot = ca.integrator("int_p1_plot", "idas", {"x":x_sym, "ode":ode_expr_aerobic}, {"t0":0, "tf":dt_plot_p1})
        x_traj_p1 = [x0_np]
        xk_ = x0_np.copy()
        for i in range(N_plot_phase1 - 1):
            try:
                res_ = integrator_p1_plot(x0=xk_); xk_ = np.array(res_["xf"]).flatten()
                xk_ = np.maximum(xk_, 0.0) # Asegurar no negatividad
                x_traj_p1.append(xk_)
            except Exception as int_e:
                st.error(f"Fallo integrador Fase 1 (plot) en paso {i}: {int_e}")
                st.write(f"Estado previo: {x_traj_p1[-1]}")
                # Rellenar con el último valor bueno para intentar continuar el gráfico
                for _ in range(i, N_plot_phase1 - 1): x_traj_p1.append(x_traj_p1[-1])
                break # Salir del bucle de simulación
        x_traj_p1 = np.array(x_traj_p1)


        # --- b) Simulación fina Fase 2 ---
        N_plot_phase2 = n_fb_intervals * 10
        t_plot_phase2 = np.linspace(t_aerobic_batch, t_anaerobic_feed_end, N_plot_phase2)
        dt_plot_p2 = t_plot_phase2[1] - t_plot_phase2[0] if N_plot_phase2 > 1 else T_feed_duration
        integrator_p2_plot = ca.integrator("int_p2_plot", "idas", {"x":x_sym, "p":u_sym, "ode":ode_expr_anaerobic}, {"t0":0, "tf":dt_plot_p2})
        x_traj_p2 = []
        # Estado inicial de Fase 2 es el final de Fase 1 simulada finamente
        xk_ = x_traj_p1[-1].copy()
        F_plot_phase2 = []
        plot_p2_ok = True
        for i, t_now in enumerate(t_plot_phase2):
            x_traj_p2.append(xk_)
            if i == len(t_plot_phase2) - 1: break
            k_interval = int((t_now - t_aerobic_batch) / dt_fb) if dt_fb > 1e-9 else 0
            k_interval = max(0, min(k_interval, n_fb_intervals - 1))
            # Asegurarse que F_opt_phase2 tiene datos
            if k_interval < len(F_opt_phase2):
                F_now = F_opt_phase2[k_interval]
            else: # Si hay discrepancia de índices, usar flujo cero
                F_now = 0.0
            if xk_[4] >= V_max_input - 1e-6: F_now = 0.0
            F_plot_phase2.append(F_now)
            try:
                res_ = integrator_p2_plot(x0=xk_, p=F_now); xk_ = np.array(res_["xf"]).flatten()
                xk_ = np.maximum(xk_, 0.0) # Asegurar no negatividad
            except Exception as int_e:
                 st.error(f"Fallo integrador Fase 2 (plot) en paso {i} (t={t_now:.2f}): {int_e}")
                 st.write(f"Estado previo: {x_traj_p2[-1]}")
                 st.write(f"Flujo aplicado: {F_now}")
                 # Rellenar con el último valor bueno
                 for _ in range(i, N_plot_phase2 - 1): x_traj_p2.append(x_traj_p2[-1])
                 plot_p2_ok = False
                 break
        x_traj_p2 = np.array(x_traj_p2)
        if F_plot_phase2: F_plot_phase2.append(F_plot_phase2[-1])
        else: F_plot_phase2.append(0.0)
        F_plot_phase2 = np.array(F_plot_phase2)


        # --- c) Simulación fina Fase 3 ---
        if T_post_feed_duration > 1e-6 and plot_p2_ok: # Solo simular si Fase 2 fue OK
            N_plot_phase3 = 50
            t_plot_phase3 = np.linspace(t_anaerobic_feed_end, t_total, N_plot_phase3)
            dt_plot_p3 = t_plot_phase3[1] - t_plot_phase3[0] if N_plot_phase3 > 1 else T_post_feed_duration
            integrator_p3_plot = ca.integrator("int_p3_plot", "idas", {"x":x_sym, "p":u_sym, "ode":ode_expr_anaerobic}, {"t0":0, "tf":dt_plot_p3})
            x_traj_p3 = []
            xk_ = x_traj_p2[-1].copy()
            plot_p3_ok = True
            for i in range(N_plot_phase3):
                 x_traj_p3.append(xk_)
                 if i == N_plot_phase3 -1: break
                 try:
                     res_ = integrator_p3_plot(x0=xk_, p=0.0); xk_ = np.array(res_["xf"]).flatten()
                     xk_ = np.maximum(xk_, 0.0)
                 except Exception as int_e:
                     st.error(f"Fallo integrador Fase 3 (plot) en paso {i}: {int_e}")
                     st.write(f"Estado previo: {x_traj_p3[-1]}")
                     # Rellenar con el último valor bueno
                     for _ in range(i, N_plot_phase3 - 1): x_traj_p3.append(x_traj_p3[-1])
                     plot_p3_ok = False
                     break
            x_traj_p3 = np.array(x_traj_p3)
        elif not plot_p2_ok: # Si Fase 2 falló, crear datos vacíos o placeholder para Fase 3
            st.warning("Saltando simulación Fase 3 debido a fallo en Fase 2.")
            t_plot_phase3 = np.array([t_anaerobic_feed_end])
            x_traj_p3 = np.array([x_traj_p2[-1]]) # Solo el último punto (fallido) de la fase 2
        else: # Si no hay duración de fase 3
            t_plot_phase3 = np.array([t_anaerobic_feed_end])
            x_traj_p3 = np.array([x_traj_p2[-1]])

        # --- d) Unir Trayectorias y Flujo ---
        try:
            t_full = np.concatenate([t_plot_phase1[:-1], t_plot_phase2[:-1], t_plot_phase3])
            x_full = np.vstack([x_traj_p1[:-1, :], x_traj_p2[:-1, :], x_traj_p3])
            F_plot_phase1 = np.zeros(len(t_plot_phase1) - 1)
            F_plot_phase2_aligned = F_plot_phase2[:len(t_plot_phase2)-1]
            F_plot_phase3 = np.zeros(len(t_plot_phase3))
            F_full_steps = np.concatenate([F_plot_phase1, F_plot_phase2_aligned, F_plot_phase3[:-1]])
            if len(F_full_steps) == 0: # Caso extremo sin pasos simulados
                 F_full_points = np.array([0.0] * len(t_full)) if len(t_full)>0 else np.array([0.0])
            else:
                 F_full_points = np.append(F_full_steps, F_full_steps[-1])
                 # Asegurar longitud correcta si t_full tiene longitud 1
                 if len(t_full) == 1 and len(F_full_points) == 0 : F_full_points = np.array([0.0])
                 elif len(F_full_points) > len(t_full): F_full_points = F_full_points[:len(t_full)]


            X_full, S_full, P_full, O2_full, V_full = [x_full[:, i] for i in range(nx)]
            plot_data_ok = True
        except ValueError as concat_e:
            st.error(f"Error al concatenar datos para graficar: {concat_e}")
            st.write(f"Longitudes: t1={len(t_plot_phase1)}, t2={len(t_plot_phase2)}, t3={len(t_plot_phase3)}")
            st.write(f"Longitudes x: x1={len(x_traj_p1)}, x2={len(x_traj_p2)}, x3={len(x_traj_p3)}")
            plot_data_ok = False


        # ====================================================
        # 7) Gráficas
        # ====================================================
        if plot_data_ok:
            st.info("📊 Generando gráficas del proceso optimizado con fases marcadas...")
            fig, axs = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True, sharex=True)
            axs = axs.ravel()

            def add_phase_shading(ax, t1, t2, t3):
                ax.axvspan(0, t1, facecolor='#A6C3D8', alpha=0.3)
                ax.axvspan(t1, t2, facecolor='#A8D8A6', alpha=0.3)
                if t3 > t2 + 1e-6 : ax.axvspan(t2, t3, facecolor='#D8A6A6', alpha=0.3)

            for ax in axs: add_phase_shading(ax, t_aerobic_batch, t_anaerobic_feed_end, t_total)

            axs[0].step(t_full, F_full_points, where='post', linewidth=2, color='black')
            axs[0].set_title("Flujo de Alimentación $F(t)$")
            axs[0].set_ylabel("$F$ (L/h)")
            axs[0].grid(True, axis='y', linestyle=':')
            axs[0].set_ylim(bottom=-0.001)

            axs[1].plot(t_full, X_full, linewidth=2, color='green')
            axs[1].set_title("Biomasa $X(t)$")
            axs[1].set_ylabel("$X$ (g/L)")
            axs[1].grid(True, axis='y', linestyle=':')

            axs[2].plot(t_full, S_full, linewidth=2, color='blue')
            axs[2].axhline(S_max_constraint, color='red', linestyle='--', lw=1, label=f"$S_{{max}}$ (lim)")
            axs[2].set_title("Sustrato $S(t)$")
            axs[2].set_ylabel("$S$ (g/L)")
            axs[2].grid(True, axis='y', linestyle=':')

            axs[3].plot(t_full, P_full, linewidth=2, color='purple')
            axs[3].axhline(P_max_constraint, color='red', linestyle='--', lw=1, label=f"$P_{{max}}$ (lim)")
            axs[3].set_title("Producto (Etanol) $P(t)$")
            axs[3].set_ylabel("$P$ (g/L)")
            axs[3].grid(True, axis='y', linestyle=':')

            axs[4].plot(t_full, O2_full, linewidth=2, color='cyan')
            axs[4].set_title("Oxígeno Disuelto $O_2(t)$")
            axs[4].set_ylabel("$O_2$ (g/L)")
            axs[4].grid(True, axis='y', linestyle=':')
            o2_upper_lim = max(O_sat, np.max(O2_full) if len(O2_full)>0 else O_sat) * 1.1
            axs[4].set_ylim(-0.0005, o2_upper_lim if o2_upper_lim > 1e-6 else 0.001) # Evitar límite superior cero
            axs[4].ticklabel_format(style='sci', axis='y', scilimits=(-3,4))

            axs[5].plot(t_full, V_full, linewidth=2, color='orange')
            axs[5].axhline(V_max_input, color='red', linestyle='--', lw=1, label=f"$V_{{max}}$ (lim)")
            axs[5].set_title("Volumen $V(t)$")
            axs[5].set_ylabel("$V$ (L)")
            axs[5].grid(True, axis='y', linestyle=':')

            for ax in axs: ax.set_xlabel("Tiempo (h)"); ax.margins(x=0.01)

            legend_elements = [Patch(facecolor='#A6C3D8', alpha=0.5, label='Fase 1: Aerobia'),
                               Patch(facecolor='#A8D8A6', alpha=0.5, label='Fase 2: Anaerobia Feed'),
                               Patch(facecolor='#D8A6A6', alpha=0.5, label='Fase 3: Anaerobia Batch')]
            h2, l2 = axs[2].get_legend_handles_labels()
            h3, l3 = axs[3].get_legend_handles_labels()
            h5, l5 = axs[5].get_legend_handles_labels()
            fig.legend(handles=legend_elements + h2 + h3 + h5,
                       labels=['Fase 1', 'Fase 2', 'Fase 3'] + l2 + l3 + l5,
                       loc='outside lower center', ncol=4, title="Fases y Límites")

            plt.tight_layout(rect=[0, 0.08, 1, 1])
            st.pyplot(fig)
        else:
            st.error("No se pudieron generar las gráficas debido a errores en la simulación o manejo de datos.")


        # --- Métricas Finales ---
        if plot_data_ok and len(P_full)>0: # Calcular solo si hay datos
            st.subheader("📈 Métricas Finales del Proceso Simulado con Perfil Óptimo")
            col1, col2, col3 = st.columns(3)
            P_fin_sim = P_full[-1]; V_fin_sim = V_full[-1]
            col1.metric("Producto Total Acumulado (Sim)", f"{P_fin_sim * V_fin_sim:.3f} g")
            # Asegurarse que F_opt_phase2 existe y es array numpy
            if isinstance(F_opt_phase2, (list, np.ndarray)) and len(F_opt_phase2) == n_fb_intervals:
                 S_total_fed = sum(F_opt_phase2 * dt_fb * Sf_input)
            else:
                 S_total_fed = 0 # No se pudo calcular
                 st.warning("No se pudo calcular S_total_fed.")

            if len(S_full)>0 and len(V_full)>0 :
                S_consumed = (S0 * V0 + S_total_fed - S_full[-1] * V_full[-1])
            else:
                S_consumed = 0

            Global_Yield_P_S = (P_fin_sim * V_fin_sim - P0*V0) / S_consumed if S_consumed > 1e-6 else 0
            col2.metric("Rendimiento Global (P/S)", f"{Global_Yield_P_S:.3f} g/g")
            Productivity = (P_fin_sim * V_fin_sim) / t_total if t_total > 0 else 0
            col3.metric("Productividad Vol. Media", f"{Productivity:.3f} g/h")
            col1.metric("Concentración Final Etanol (Sim)", f"{P_fin_sim:.3f} g/L")
            col2.metric("Volumen Final (Sim)", f"{V_fin_sim:.3f} L")
            col3.metric("Tiempo Total", f"{t_total:.1f} h")
            if len(O2_full)>0:
                 col1.metric("O2 Final (Sim)", f"{O2_full[-1]:.5f} g/L")
        else:
             st.warning("No se calcularon métricas finales por falta de datos.")


# --- Ejecución ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="RTO Fermentación")
    # Añadir manejo de errores global básico
    try:
        rto_fermentation_page()
    except Exception as main_e:
        st.error(f"Error inesperado en la ejecución principal: {main_e}")
        st.exception(main_e)