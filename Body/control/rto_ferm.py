import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Definiciones Cinéticas (Igual que antes) ---
def mu_aerobic(S, O, mu_max_aer, Ks, Ko):
    mu = mu_max_aer * (S / (Ks + S + S**2/1000)) * (O / (Ko + O)) # Added S inhibition term example
    return ca.fmax(mu, 0.0)

def mu_anaerobic_inhib(S, P, mu_max_an, Ks, Pmax, Kpi):
    K_SI = 200.0 # Ejemplo
    inhibition_S = (Ks + S + S**2 / K_SI)
    inhibition_P = (1 - P / Pmax)**1.0 if Pmax > 0 else 1.0
    mu = mu_max_an * (S / inhibition_S) * ca.fmax(0.0, inhibition_P)
    return ca.fmax(mu, 0.0)

# --- Página de Streamlit ---
def rto_fermentation_page (): # Renombrado para claridad
    st.header("🧠 Control RTO - Fermentación Alcohólica (3 Fases con Penalización S)")
    st.markdown("""
    Optimización del perfil de alimentación para maximizar la **productividad volumétrica final ($P_{final} V_{final}$)**,
    con penalización por alta concentración de sustrato ($S > S_{max}$) durante la alimentación.
    Fases:
    1. Fase Aerobia Batch
    2. Fase Anaerobia Fed-Batch (Control Opt.)
    3. Fase Anaerobia Batch (Post-Feed)
    """)

    with st.sidebar:
        st.subheader("📌 Parámetros Cinéticos")
        # (Parámetros cinéticos sin cambios respecto a la versión anterior)
        mu_max_aer = st.number_input("μmax (Aerobio) [1/h]", value=0.4, min_value=0.01, key="mu_max_aer")
        Yxs_aer = st.number_input("Yxs (Aerobio) [g/g]", value=0.5, min_value=0.1, max_value=1.0, key="yxs_aer")
        Yps_aer = st.number_input("Yps (Aerobio - Etanol?) [g/g]", value=0.05, min_value=0.0, max_value=1.0, key="yps_aer", help="Producción de etanol bajo aerobiosis suele ser baja (Crabtree)")
        mu_max_an = st.number_input("μmax (Anaerobio) [1/h]", value=0.3, min_value=0.01, key="mu_max_an")
        Yxs_an = st.number_input("Yxs (Anaerobio) [g/g]", value=0.1, min_value=0.01, max_value=1.0, key="yxs_an")
        Yps_an = st.number_input("Yps (Anaerobio - Etanol) [g/g]", value=0.45, min_value=0.1, max_value=0.5, key="yps_an", help="Rendimiento teórico máx ~0.51")
        Ks = st.number_input("Ks [g/L]", value=0.5, min_value=0.01, key="ks")
        Ko = st.number_input("KO (afinidad O2) [g/L]", value=0.002, min_value=0.0001, format="%.4f", key="ko")
        P_max_inhib = st.number_input("Pmax (Inhibición Etanol) [g/L]", value=90.0, min_value=10.0, key="pmax")

        st.subheader("💧 Alimentación y Reactor")
        Sf_input = st.number_input("Concentración del alimentado Sf [g/L]", value=250.0, key="sf")
        V_max_input = st.number_input("Volumen máximo del reactor [L]", value=0.5, key="vmax")

        st.subheader("🎚 Condiciones Iniciales (t=0)")
        X0 = st.number_input("X0 (Biomasa) [g/L]", value=1.16, key="x0")
        S0 = st.number_input("S0 (Sustrato) [g/L]", value=10.17, key="s0")
        P0 = st.number_input("P0 (Etanol) [g/L]", value=0.0, key="p0")
        O0 = st.number_input("O0 (Oxígeno disuelto) [g/L]", value=0.007, min_value=0.0, max_value=0.01, format="%.4f", key="o0", help="Saturación aire ~0.008 g/L. Mantener constante en Fase 1.")
        V0 = st.number_input("V0 (Volumen inicial) [L]", value=0.25, key="v0")

        st.subheader("⏳ Configuración Temporal")
        t_aerobic_batch = st.number_input("Tiempo Fase Aerobia Batch [h]", value=3.0, min_value=1.0, key="t_aerobic")
        t_anaerobic_feed_end = st.number_input("Tiempo Fin Alimentación Anaerobia [h]", value=8.0, min_value=t_aerobic_batch + 1.0, key="t_feed_end")
        t_total = st.number_input("Tiempo total del proceso [h]", value=10.0, min_value=t_anaerobic_feed_end + 0.1, key="t_total") # Permitir fase 3 corta
        n_fb_intervals = st.number_input("Número de Intervalos de Control (Fase 2)", value=12, min_value=1, key="n_intervals", help=f"Duración Fase 2: {t_anaerobic_feed_end - t_aerobic_batch:.1f} h")

        st.subheader("🔧 Restricciones y Penalización")
        F_min = st.number_input("Flujo mínimo [L/h]", value=0.01, min_value=0.0, key="fmin")
        F_max = st.number_input("Flujo máximo [L/h]", value=0.26, min_value=F_min, key="fmax")
        S_max_constraint = st.number_input("Sustrato máximo (Restricción Dura) [g/L]", value=50.0, key="smax_const")
        P_max_constraint = st.number_input("Producto máximo (Restricción Dura) [g/L]", value=P_max_inhib - 1.0 , key="pmax_const", help="Debe ser menor que Pmax de inhibición")
        # Nuevo: Peso para la penalización de Smax
        w_penalty_smax = st.number_input("Peso Penalización S > Smax", value=10.0, min_value=0.0, key="w_smax", format="%.2f",
                                         help="Penaliza S por encima de Smax (restricción dura) en Fase 2. 0 para desactivar.")


    if st.button("🚀 Ejecutar Optimización RTO (con Penalización S)"):
        st.info("Optimizando perfil de alimentación para fermentación...")

        # (Cálculos iniciales de tiempo y dt_fb igual que antes)
        T_feed_duration = t_anaerobic_feed_end - t_aerobic_batch
        if T_feed_duration <= 0:
            st.error("El tiempo de fin de alimentación debe ser mayor que el tiempo de fase aerobia.")
            st.stop()
        if n_fb_intervals <= 0:
             st.error("El número de intervalos de control debe ser positivo.")
             st.stop()
        dt_fb = T_feed_duration / n_fb_intervals

        T_post_feed_duration = t_total - t_anaerobic_feed_end
        if T_post_feed_duration < -1e-6: # Permitir duración cero, pero no negativa
             st.error("El tiempo total debe ser mayor o igual al tiempo de fin de alimentación.")
             st.stop()

        nx = 5 # X, S, P, O, V
        x_sym = ca.MX.sym("x", nx)
        u_sym = ca.MX.sym("u")
        X_, S_, P_, O_, V_ = x_sym[0], x_sym[1], x_sym[2], x_sym[3], x_sym[4]
        F_ = u_sym

        # ====================================================
        # 1) Definición de las funciones ODE (Sin cambios)
        # ====================================================
        mu_aer = mu_aerobic(S_, O_, mu_max_aer, Ks, Ko)
        YXS_AER = Yxs_aer
        YPS_AER = Yps_aer
        qS_aer = mu_aer / YXS_AER if YXS_AER > 1e-9 else 0
        qP_aer = YPS_AER * mu_aer
        dX_aer = mu_aer * X_
        dS_aer = -qS_aer * X_
        dP_aer = qP_aer * X_
        dO_aer = ca.MX(0.0)
        dV_aer = ca.MX(0.0)
        ode_expr_aerobic = ca.vertcat(dX_aer, dS_aer, dP_aer, dO_aer, dV_aer)
        odefun_aerobic = ca.Function('odefun_aerobic', [x_sym, u_sym], [ode_expr_aerobic], ['x', 'u'], ['dxdt'])

        mu_an = mu_anaerobic_inhib(S_, P_, mu_max_an, Ks, P_max_inhib, 0)
        YXS_AN = Yxs_an
        YPS_AN = Yps_an
        qS_an_simple = mu_an / YXS_AN if YXS_AN > 1e-9 else 0
        qP_an = YPS_AN * mu_an
        D = F_ / ca.fmax(V_, 1e-6) # Evitar división por cero si V es variable Opti
        dX_an = mu_an * X_ - D * X_
        dS_an = -qS_an_simple * X_ + D * (Sf_input - S_)
        dP_an = qP_an * X_ - D * P_
        dO_an = ca.MX(0.0)
        dV_an = F_
        ode_expr_anaerobic = ca.vertcat(dX_an, dS_an, dP_an, dO_an, dV_an)
        odefun_anaerobic = ca.Function('odefun_anaerobic', [x_sym, u_sym], [ode_expr_anaerobic], ['x', 'u'], ['dxdt'])

        # ====================================================
        # 2) Simulación Fase 1: Aerobia Batch (Sin cambios)
        # ====================================================
        st.info(f"[FASE 1] Simulando fase aerobia batch hasta t={t_aerobic_batch} h...")
        try:
            batch_integrator_aerobic = ca.integrator(
                "batch_integrator_aerobic", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr_aerobic},
                {"t0": 0, "tf": t_aerobic_batch, "reltol": 1e-7, "abstol": 1e-9}
            )
            x0_np = np.array([X0, S0, P0, O0, V0])
            res_batch_aerobic = batch_integrator_aerobic(x0=x0_np, p=0.0)
            x_end_aerobic = np.array(res_batch_aerobic['xf']).flatten()
            st.success(f"[FASE 1] Estado final: X={x_end_aerobic[0]:.2f}, S={x_end_aerobic[1]:.2f}, P={x_end_aerobic[2]:.2f}, O={x_end_aerobic[3]:.4f}, V={x_end_aerobic[4]:.2f}")
            if any(np.isnan(x_end_aerobic)) or any(x_end_aerobic < -1e-6):
                 st.error(f"Estado inválido después de Fase 1: {x_end_aerobic}.")
                 st.stop()
            x_end_aerobic = np.maximum(x_end_aerobic, 0.0)
        except Exception as e:
            st.error(f"Error durante la simulación de la Fase 1: {e}")
            st.stop()

        # ====================================================
        # 3) Formulación Optimización Fase 2: Anaerobia Fed-Batch
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
                    opti.subject_to(xk_j >= -1e-9)
                    opti.subject_to(xk_j[1] <= S_max_constraint) # Restricción dura S
                    opti.subject_to(xk_j[2] <= P_max_constraint) # Restricción dura P
                    opti.subject_to(xk_j[4] <= V_max_input)      # Restricción dura V
            X_col_phase2.append(row_states)

            Fk = opti.variable()
            F_col_phase2.append(Fk)
            opti.subject_to(Fk >= F_min)
            opti.subject_to(Fk <= F_max)

        # Ecuaciones Colocación y Continuidad + Cálculo Penalización Smax
        h = dt_fb
        penalty_smax_total = ca.MX(0.0) # Inicializar penalización acumulada

        for k in range(n_fb_intervals):
            # Ecuaciones en puntos interiores j=1..d
            for j in range(1, d + 1):
                xp_kj = 0
                for m in range(d + 1):
                    xp_kj += C_radau[m, j - 1] * X_col_phase2[k][m]
                fkj = odefun_anaerobic(X_col_phase2[k][j], F_col_phase2[k])
                opti.subject_to((h * fkj - xp_kj) == 0)

                # --- Añadir Penalización Smax en puntos de colocación interiores ---
                # Opcional: podrías penalizar solo al final del intervalo (ver abajo)
                # s_k_j = X_col_phase2[k][j][1] # Sustrato en el punto k,j
                # violation_j = ca.fmax(0, s_k_j - S_max_constraint)
                # Aquí necesitarías pesos de cuadratura para integrar correctamente
                # penalty_smax_total += w_penalty_smax * violation_j**2 * (h / d) # Aproximación simple

            # Continuidad y Estado al final del intervalo k
            Xk_end = 0
            for m in range(d + 1):
                Xk_end += D_radau[m] * X_col_phase2[k][m]

            if k < n_fb_intervals - 1:
                opti.subject_to(Xk_end == X_col_phase2[k + 1][0])

            # --- Añadir Penalización Smax al final de cada intervalo k ---
            # Esta es una forma más simple de implementar la penalización
            if w_penalty_smax > 1e-9: # Solo si el peso es significativo
                s_k_end = Xk_end[1] # Sustrato al final del intervalo k
                violation_k = ca.fmax(0, s_k_end - S_max_constraint)
                # Penalización cuadrática, ponderada por duración del intervalo h
                penalty_smax_total += violation_k**2

        # Multiplicar la suma de penalizaciones por el peso y la duración (aproxima integral)
        # Si penalizas en cada punto interior, ajusta esto. Si penalizas al final como arriba:
        if w_penalty_smax > 1e-9:
             penalty_smax_total = w_penalty_smax * penalty_smax_total * h


        # Estado al final de la Fase 2 (Fed-Batch)
        X_end_feed = Xk_end # Último Xk_end calculado

        # ====================================================
        # 4) Simulación Fase 3 (dentro de Opti) - (Sin cambios)
        # ====================================================
        st.info("[FASE 3 - Integración en Opti] Definiendo simulación post-alimentación...")
        if T_post_feed_duration > 1e-6:
             phase3_integrator = ca.integrator(
                 "phase3_integrator", "idas",
                 {"x": x_sym, "p": u_sym, "ode": ode_expr_anaerobic},
                 {"t0": 0, "tf": T_post_feed_duration, "reltol": 1e-7, "abstol": 1e-9}
             )
             res_phase3_sym = phase3_integrator(x0=X_end_feed, p=0.0)
             X_final_total = res_phase3_sym['xf']
        else:
             X_final_total = X_end_feed

        # ====================================================
        # 5) Función Objetivo (Modificada) y Resolución
        # ====================================================
        P_final_total = X_final_total[2]
        V_final_total = X_final_total[4]
        objective_PV = -(P_final_total * V_final_total) # Maximizar PV final

        # Objetivo Total = Maximizar PV + Penalización Smax
        objective_total = objective_PV + penalty_smax_total
        opti.minimize(objective_total)

        # Guesses iniciales (Sin cambios)
        st.info("Estableciendo guesses iniciales...")
        F_guess = (F_max + F_min) / 2.0 * 0.5
        for k in range(n_fb_intervals):
            opti.set_initial(F_col_phase2[k], F_guess)
        x_guess = x_end_aerobic.copy()
        x_guess[4] = min(V_max_input, x_guess[4] + F_guess * T_feed_duration / 2)
        for k in range(n_fb_intervals):
            start_j = 1 if k == 0 else 0
            for j in range(start_j, d + 1):
                 opti.set_initial(X_col_phase2[k][j], x_guess)

        # Configurar Solver (Sin cambios)
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 3000,
            "print_level": 5,
            "sb": 'yes',
            "tol": 1e-6,
            "constr_viol_tol": 1e-6
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            st.info("🚀 Resolviendo el problema de optimización (con penalización S)...")
            sol = opti.solve()
            st.success("[OPTIMIZACIÓN] ¡Solución encontrada!")

            # Extraer resultados (Sin cambios)
            F_opt_phase2 = np.array([sol.value(fk) for fk in F_col_phase2])
            X_end_feed_opt = sol.value(X_end_feed)
            X_final_total_opt = sol.value(X_final_total)
            P_final_opt = X_final_total_opt[2]
            V_final_opt = X_final_total_opt[4]
            Smax_penalty_value = sol.value(penalty_smax_total) # Valor de la penalización

            st.metric("Producto Total Final (P*V)", f"{P_final_opt * V_final_opt:.3f} g")
            st.metric("Concentración Final Etanol", f"{P_final_opt:.3f} g/L")
            st.metric("Volumen Final", f"{V_final_opt:.3f} L")
            st.metric("Valor Penalización Smax", f"{Smax_penalty_value:.4f}", delta=None, delta_color="off")


            st.write("Perfil óptimo de flujo (Fase 2):")
            st.line_chart(F_opt_phase2)

        except RuntimeError as e:
            # (Manejo de errores sin cambios)
            st.error(f"[ERROR] El solver no pudo encontrar una solución: {e}")
            try:
                st.warning("Mostrando posibles puntos de infactibilidad (debug info):")
                st.write(f"Valor objetivo: {opti.debug.value(objective_total)}")
                st.write(f"Valor PV: {opti.debug.value(objective_PV)}")
                st.write(f"Valor Penaliz Smax: {opti.debug.value(penalty_smax_total)}")
                st.write("Valores de restricciones (g(x)):")
                st.text(opti.debug.value(opti.g)) # Mostrar valores de restricciones
            except Exception as debug_e:
                 st.error(f"Error al obtener información de debug: {debug_e}")
            st.stop()
        except Exception as e:
             st.error(f"Ocurrió un error inesperado durante la optimización: {e}")
             st.stop()


        # ====================================================
        # 6) Reconstrucción de la Trayectoria Completa (Sin cambios en la lógica, solo en gráficos)
        # ====================================================
        st.info("Reconstruyendo trayectoria completa con perfil óptimo...")
        # (Simulación fina Fase 1, 2, 3 igual que antes)
        # --- a) Simulación fina Fase 1 ---
        N_plot_phase1 = 50
        t_plot_phase1 = np.linspace(0, t_aerobic_batch, N_plot_phase1)
        dt_plot_p1 = t_plot_phase1[1] - t_plot_phase1[0] if N_plot_phase1 > 1 else t_aerobic_batch
        integrator_p1_plot = ca.integrator("int_p1", "idas", {"x":x_sym, "p":u_sym, "ode":ode_expr_aerobic}, {"t0":0, "tf":dt_plot_p1})
        x_traj_p1 = [x0_np]
        xk_ = x0_np.copy()
        for _ in range(N_plot_phase1 - 1):
            res_ = integrator_p1_plot(x0=xk_, p=0.0); xk_ = np.array(res_["xf"]).flatten()
            x_traj_p1.append(xk_)
        x_traj_p1 = np.array(x_traj_p1)

        # --- b) Simulación fina Fase 2 ---
        N_plot_phase2 = n_fb_intervals * 10
        t_plot_phase2 = np.linspace(t_aerobic_batch, t_anaerobic_feed_end, N_plot_phase2)
        dt_plot_p2 = t_plot_phase2[1] - t_plot_phase2[0] if N_plot_phase2 > 1 else T_feed_duration
        integrator_p2_plot = ca.integrator("int_p2", "idas", {"x":x_sym, "p":u_sym, "ode":ode_expr_anaerobic}, {"t0":0, "tf":dt_plot_p2})
        x_traj_p2 = []
        xk_ = x_traj_p1[-1].copy()
        F_plot_phase2 = []
        for i, t_now in enumerate(t_plot_phase2):
            x_traj_p2.append(xk_)
            if i == len(t_plot_phase2) - 1: break
            k_interval = int((t_now - t_aerobic_batch) / dt_fb) if dt_fb > 1e-9 else 0
            k_interval = max(0, min(k_interval, n_fb_intervals - 1))
            F_now = F_opt_phase2[k_interval]
            if xk_[4] >= V_max_input - 1e-6: F_now = 0.0
            F_plot_phase2.append(F_now)
            res_ = integrator_p2_plot(x0=xk_, p=F_now); xk_ = np.array(res_["xf"]).flatten()
            xk_ = np.maximum(xk_, 0.0)
        x_traj_p2 = np.array(x_traj_p2)
        if F_plot_phase2: F_plot_phase2.append(F_plot_phase2[-1])
        else: F_plot_phase2.append(0.0) # Handle case N_plot_phase2=1
        F_plot_phase2 = np.array(F_plot_phase2)

        # --- c) Simulación fina Fase 3 ---
        if T_post_feed_duration > 1e-6:
            N_plot_phase3 = 50
            t_plot_phase3 = np.linspace(t_anaerobic_feed_end, t_total, N_plot_phase3)
            dt_plot_p3 = t_plot_phase3[1] - t_plot_phase3[0] if N_plot_phase3 > 1 else T_post_feed_duration
            integrator_p3_plot = ca.integrator("int_p3", "idas", {"x":x_sym, "p":u_sym, "ode":ode_expr_anaerobic}, {"t0":0, "tf":dt_plot_p3})
            x_traj_p3 = []
            xk_ = x_traj_p2[-1].copy()
            for _ in range(N_plot_phase3):
                 x_traj_p3.append(xk_)
                 res_ = integrator_p3_plot(x0=xk_, p=0.0); xk_ = np.array(res_["xf"]).flatten()
                 xk_ = np.maximum(xk_, 0.0)
            x_traj_p3 = np.array(x_traj_p3)
        else:
            t_plot_phase3 = np.array([t_anaerobic_feed_end])
            x_traj_p3 = np.array([x_traj_p2[-1]])

        # --- d) Unir Trayectorias y Flujo ---
        t_full = np.concatenate([t_plot_phase1[:-1], t_plot_phase2[:-1], t_plot_phase3])
        x_full = np.vstack([x_traj_p1[:-1, :], x_traj_p2[:-1, :], x_traj_p3])
        F_plot_phase1 = np.zeros(len(t_plot_phase1) -1)
        F_plot_phase2_aligned = F_plot_phase2[:-1] # Align F with time points
        F_plot_phase3 = np.zeros(len(t_plot_phase3))
        F_full = np.concatenate([F_plot_phase1, F_plot_phase2_aligned, F_plot_phase3])
        X_full, S_full, P_full, O_full, V_full = [x_full[:, i] for i in range(nx)]

        # ====================================================
        # 7) Gráficas (Con Marcado de Fases)
        # ====================================================
        st.info("📊 Generando gráficas del proceso optimizado con fases marcadas...")
        fig, axs = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True, sharex=True) # Share x-axis
        axs = axs.ravel()

        # Función para sombrear fases
        def add_phase_shading(ax, t1, t2, t3):
            ax.axvspan(0, t1, facecolor='#A6C3D8', alpha=0.3, label='_nolegend_') # Azul claro
            ax.axvspan(t1, t2, facecolor='#A8D8A6', alpha=0.3, label='_nolegend_') # Verde claro
            if t3 > t2 + 1e-6 : # Solo sombrear fase 3 si existe
                 ax.axvspan(t2, t3, facecolor='#D8A6A6', alpha=0.3, label='_nolegend_') # Rojo claro

        # Aplicar sombreado a todos los ejes
        for ax in axs:
             add_phase_shading(ax, t_aerobic_batch, t_anaerobic_feed_end, t_total)

        # Flujo F(t)
        axs[0].plot(t_full, F_full, linewidth=2, drawstyle='steps-post', color='black')
        axs[0].set_title("Flujo de Alimentación $F(t)$")
        axs[0].set_ylabel("$F$ (L/h)")
        axs[0].grid(True, axis='y', linestyle=':')
        axs[0].set_ylim(bottom=-0.001)

        # Biomasa X(t)
        axs[1].plot(t_full, X_full, linewidth=2, color='green')
        axs[1].set_title("Biomasa $X(t)$")
        axs[1].set_ylabel("$X$ (g/L)")
        axs[1].grid(True, axis='y', linestyle=':')

        # Sustrato S(t)
        axs[2].plot(t_full, S_full, linewidth=2, color='blue')
        axs[2].axhline(S_max_constraint, color='red', linestyle='--', lw=1, label=f"$S_{{max}}$ (lim)")
        axs[2].set_title("Sustrato $S(t)$")
        axs[2].set_ylabel("$S$ (g/L)")
        axs[2].grid(True, axis='y', linestyle=':')
        #axs[2].legend(loc='upper right')

        # Producto P(t) - Etanol
        axs[3].plot(t_full, P_full, linewidth=2, color='purple')
        axs[3].axhline(P_max_constraint, color='red', linestyle='--', lw=1, label=f"$P_{{max}}$ (lim)")
        axs[3].set_title("Producto (Etanol) $P(t)$")
        axs[3].set_ylabel("$P$ (g/L)")
        axs[3].grid(True, axis='y', linestyle=':')
        #axs[3].legend(loc='lower right')

        # Oxígeno O(t)
        axs[4].plot(t_full, O_full, linewidth=2, color='cyan')
        axs[4].set_title("Oxígeno Disuelto $O(t)$")
        axs[4].set_ylabel("$O_2$ (g/L)")
        axs[4].grid(True, axis='y', linestyle=':')
        if O0 < 0.01: axs[4].set_ylim(-0.0005, O0 * 1.5)

        # Volumen V(t)
        axs[5].plot(t_full, V_full, linewidth=2, color='orange')
        axs[5].axhline(V_max_input, color='red', linestyle='--', lw=1, label=f"$V_{{max}}$ (lim)")
        axs[5].set_title("Volumen $V(t)$")
        axs[5].set_ylabel("$V$ (L)")
        axs[5].grid(True, axis='y', linestyle=':')
        #axs[5].legend(loc='lower right')

        # Añadir xlabel común y leyenda de fases
        for ax in axs:
             ax.set_xlabel("Tiempo (h)")
             ax.margins(x=0.01) # Pequeño margen

        # Crear leyenda para las fases manualmente fuera de los ejes
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#A6C3D8', alpha=0.5, label='Fase 1: Aerobia Batch'),
                           Patch(facecolor='#A8D8A6', alpha=0.5, label='Fase 2: Anaerobia Fed-Batch'),
                           Patch(facecolor='#D8A6A6', alpha=0.5, label='Fase 3: Anaerobia Batch (Post)')]
        # Añadir leyendas de restricciones si existen
        h2, l2 = axs[2].get_legend_handles_labels()
        h3, l3 = axs[3].get_legend_handles_labels()
        h5, l5 = axs[5].get_legend_handles_labels()
        fig.legend(handles=legend_elements + h2 + h3 + h5, labels=['Fase 1', 'Fase 2', 'Fase 3'] + l2 + l3 + l5,
                   loc='outside lower center', ncol=3, title="Fases y Límites")

        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Ajustar layout para dejar espacio a la leyenda inferior

        st.pyplot(fig)

        # (Métricas finales sin cambios)
        st.subheader("📈 Métricas Finales del Proceso Optimizado")
        col1, col2, col3 = st.columns(3)
        P_fin_sim = P_full[-1]
        V_fin_sim = V_full[-1]
        col1.metric("Producto Total Acumulado", f"{P_fin_sim * V_fin_sim:.3f} g")
        S_total_fed = 0
        t_interval_starts = np.linspace(t_aerobic_batch, t_anaerobic_feed_end - dt_fb, n_fb_intervals)
        for k in range(n_fb_intervals): S_total_fed += F_opt_phase2[k] * dt_fb * Sf_input
        S_consumed = (S0 * V0 + S_total_fed - S_full[-1] * V_full[-1])
        Global_Yield_P_S = (P_fin_sim * V_fin_sim - P0*V0) / S_consumed if S_consumed > 1e-6 else 0
        col2.metric("Rendimiento Global (P/S)", f"{Global_Yield_P_S:.3f} g/g")
        Productivity = (P_fin_sim * V_fin_sim) / t_total if t_total > 0 else 0
        col3.metric("Productividad Vol. Media", f"{Productivity:.3f} g/h")
        col1.metric("Concentración Final Etanol", f"{P_fin_sim:.3f} g/L")
        col2.metric("Volumen Final", f"{V_fin_sim:.3f} L")
        col3.metric("Tiempo Total", f"{t_total:.1f} h")


# --- Ejecución ---
if __name__ == '__main__':
    st.set_page_config(layout="wide")
    rto_fermentation_page () # Llamar a la nueva función