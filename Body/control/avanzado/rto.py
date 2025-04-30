import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

def rto_page():
    st.header("🧠 Control RTO - Optimización del perfil de alimentación")

    with st.sidebar:
        st.subheader("📌 Parámetros del modelo")
        mu_max = st.number_input("μmax [1/h]", value=0.6, min_value=0.01)
        Ks = st.number_input("Ks [g/L]", value=0.2, min_value=0.01)
        Ko = st.number_input("KO [g/L]", value=0.01, min_value=0.001)
        KP = st.number_input("KP [g/L]", value=0.1, min_value=0.001)
        Yxs = st.number_input("Yxs [g/g]", value=0.5, min_value=0.1, max_value=1.0)
        Yxo = st.number_input("Yxo [g/g]", value=0.1, min_value=0.01, max_value=1.0)
        Yps = st.number_input("Yps [g/g]", value=0.3, min_value=0.1, max_value=1.0)
        Sf_input = st.number_input("Concentración del alimentado Sf [g/L]", value=500.0)
        V_max_input = st.number_input("Volumen máximo del reactor [L]", value=2.0)

        st.subheader("🎚 Condiciones Iniciales")
        X0 = st.number_input("X0 (Biomasa) [g/L]", value=1.0)
        S0 = st.number_input("S0 (Sustrato) [g/L]", value=20.0)
        P0 = st.number_input("P0 (Producto) [g/L]", value=0.0)
        O0 = st.number_input("O0 (Oxígeno) [g/L]", value=0.08)
        V0 = st.number_input("V0 (Volumen inicial) [L]", value=0.2)

        st.subheader("⏳ Configuración temporal")
        t_batch = st.number_input("Tiempo de lote (t_batch) [h]", value=5.0, min_value=0.0)
        t_total = st.number_input("Tiempo total del proceso [h]", value=24.0, min_value=t_batch + 1.0)

        st.subheader("🔧 Restricciones de operación")
        F_min = st.number_input("Flujo mínimo [L/h]", value=0.0, min_value=0.0)
        F_max = st.number_input("Flujo máximo [L/h]", value=0.3, min_value=F_min)
        S_max = st.number_input("Sustrato máximo permitido [g/L]", value=30.0)

        st.subheader("🔬 Selección del modelo cinético")
        kinetic_model = st.selectbox("Modelo cinético", ["Monod", "Sigmoidal", "Completa"])
        if kinetic_model == "Sigmoidal":
            n_sigmoidal = st.number_input("n (para Monod Sigmoidal)", value=2.0, min_value=1.0)

    if st.button("🚀 Ejecutar Optimización RTO"):
        st.info("Optimizando perfil de alimentación...")

        try:
            def radau_coefficients(d):
                """
                Retorna C_mat (shape (d+1, d)) y D_vec (shape d+1)
                para la colocación de Radau IIA con grado d=2.
                Estos valores son los correctos para Radau IIA order 3.
                """
                if d == 2:
                    C_mat = np.array([
                        [-2.0,   2.0],
                        [ 1.5,  -4.5],
                        [ 0.5,   2.5]
                    ])
                    D_vec = np.array([0.0, 0.0, 1.0])
                    return C_mat, D_vec
                else:
                    raise NotImplementedError("Solo implementado para d=2.")
            # ====================================================
            # 1) Definición de la función ODE BIO
            # ====================================================
            def odefun(x, u):
                """
                Ecuaciones diferenciales Fed-Batch con O=constante.
                - Divisiones con fmax(V, epsilon) para evitar 1/0.
                """
                # Parámetros
                mu_max_local = mu_max
                Ks_local = Ks
                Ko_local = Ko
                KP_local = KP
                Yxs_local = Yxs
                Yxo_local = Yxo
                Yps_local = Yps
                Sf_local = Sf_input
                V_max_local = V_max_input

                # Extraer variables de estado (evitar desempacado iterativo)
                X_ = x[0]
                S_ = x[1]
                P_ = x[2]
                O_ = x[3]
                V_ = x[4]

                # Tasa de crecimiento
                if kinetic_model == "Monod":
                    mu = mu_monod(S_, mu_max_local, Ks_local) * (O_ / (Ko_local + O_)) # Assuming oxygen dependence
                elif kinetic_model == "Sigmoidal":
                    mu = mu_sigmoidal(S_, mu_max_local, Ks_local, n_sigmoidal) * (O_ / (Ko_local + O_)) # Assuming oxygen dependence
                elif kinetic_model == "Completa":
                    mu = mu_completa(S_, O_, P_, mu_max_local, Ks_local, Ko_local, KP_local)
                else:
                    raise ValueError("Modelo cinético no seleccionado correctamente.")

                # Tasa de dilución
                D = u / V_

                dX = mu * X_ - D * X_
                dS = -mu * X_ / Yxs_local + D * (Sf_local - S_)
                dP = Yps_local * mu * X_ - D * P_
                dO = 0.0   # asumiendo oxígeno constante
                dV = u

                return ca.vertcat(dX, dS, dP, dO, dV)

            # ====================================================
            # 3) Parámetros del proceso y condiciones iniciales
            # ====================================================
            n_fb_intervals = int((t_total - t_batch))
            dt_fb = (t_total - t_batch) / n_fb_intervals if n_fb_intervals > 0 else 0.0

            # ====================================================
            # 4) Fase BATCH con F=0 (integración)
            # ====================================================
            x_sym = ca.MX.sym("x", 5)
            u_sym = ca.MX.sym("u")
            ode_expr = odefun(x_sym, u_sym)

            batch_integrator = ca.integrator(
                "batch_integrator", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": t_batch}
            )

            x0_np = np.array([X0, S0, P0, O0, V0])
            res_batch = batch_integrator(x0=x0_np, p=0.0)
            x_after_batch = np.array(res_batch['xf']).flatten()
            st.info(f"[INFO] Estado tras fase batch: {x_after_batch}")

            # ====================================================
            # 5) Formulación de la fase Fed-Batch con colocación
            # ====================================================
            opti = ca.Opti()

            d = 2
            C_radau, D_radau = radau_coefficients(d)
            nx = 5

            # Variables de estado y control
            X_col = []
            F_col = []

            for k in range(n_fb_intervals):
                row_states = []
                for j in range(d + 1):
                    if (k == 0 and j == 0):
                        # Fijar el estado inicial del primer intervalo
                        # con un "parameter" (no es variable)
                        xk0_param = opti.parameter(nx)
                        opti.set_value(xk0_param, x_after_batch)
                        row_states.append(xk0_param)
                    else:
                        # variable
                        xk_j = opti.variable(nx)
                        row_states.append(xk_j)
                        # Restricciones
                        # no-negatividad:
                        opti.subject_to(xk_j >= 0)
                        # S <= S_max
                        opti.subject_to(xk_j[1] <= S_max)
                        # V <= V_max
                        opti.subject_to(xk_j[4] <= V_max_input)
                X_col.append(row_states)

                # Variable de control en cada intervalo
                Fk = opti.variable()
                F_col.append(Fk)
                opti.subject_to(Fk >= F_min)
                opti.subject_to(Fk <= F_max)

            # ====================================================
            # 6) Ecuaciones de Colocación
            # ====================================================
            h = dt_fb
            for k in range(n_fb_intervals):
                for j in range(1, d + 1):
                    # xp_j = sum_{m=0..d} C_radau[m, j-1]* X_col[k][m]
                    xp_j = 0
                    for m in range(d + 1):
                        xp_j += C_radau[m, j - 1] * X_col[k][m]

                    # f(Xk_j, Fk)
                    fkj = odefun(X_col[k][j], F_col[k])
                    # Restricción => h*f - xp_j = 0
                    coll_eq = h * fkj - xp_j
                    opti.subject_to(coll_eq == 0)

                # Continuidad al final del subintervalo
                Xk_end = 0
                for m in range(d + 1):
                    Xk_end += D_radau[m] * X_col[k][m]

                if k < n_fb_intervals - 1:
                    # Xk_end = X_{k+1}[0]
                    for i_ in range(nx):
                        opti.subject_to(Xk_end[i_] == X_col[k + 1][0][i_])

            # Estado final global => X_final
            X_final = X_col[-1][-1]

            P_final = X_final[2]
            V_final = X_final[4]

            # ====================================================
            # 7) Función objetivo => maximizar (P_final*V_final)
            # ====================================================
            opti.minimize(-(P_final * V_final))

            # ====================================================
            # 8) Guesses iniciales (importante para evitar NaNs)
            # ====================================================
            for k in range(n_fb_intervals):
                opti.set_initial(F_col[k], 0.1)
                for j in range(d + 1):
                    # Si no es el primer "parameter"
                    if not (k == 0 and j == 0):
                        # Como guess, usemos el estado final de batch (o algo similar)
                        opti.set_initial(X_col[k][j], x_after_batch)

            # ====================================================
            # 9) Configurar y resolver
            # ====================================================
            p_opts = {}
            s_opts = {
                "max_iter": 2000,
                "print_level": 0,
                "sb": 'yes',
                "mu_strategy": "adaptive"
            }
            opti.solver("ipopt", p_opts, s_opts)

            try:
                sol = opti.solve()
                st.success("[INFO] ¡Solución encontrada!")
            except RuntimeError as e:
                st.error(f"[ERROR] No se encontró solución: {e}")
                try:
                    # Mostrar infeasibilidades
                    opti.debug.show_infeasibilities()
                except:
                    pass
                st.stop()

            F_opt = [sol.value(fk) for fk in F_col]
            X_fin_val = sol.value(X_final)
            P_fin_val = X_fin_val[2]
            V_fin_val = X_fin_val[4]

            st.info(f"Flujo óptimo de alimentación (F_opt): {F_opt}")
            st.info(f"Estado final del reactor: {X_fin_val}")
            st.info(f"Concentración final de Producto (P_final): {P_fin_val:.4f} g/L")
            st.info(f"Volumen final del reactor (V_final): {V_fin_val:.4f} L")
            st.info(f"Producto total final: {(P_fin_val * V_fin_val):.4f} g")

            # ====================================================
            # 10) Reconstruir y graficar trayectoria
            #     (batch + fed-batch)
            # ====================================================
            # a) Fase batch: con dt pequeño
            N_batch_plot = 50
            t_batch_plot = np.linspace(0, t_batch, N_batch_plot)
            dt_b = t_batch_plot[1] - t_batch_plot[0]

            batch_plot_int = ca.integrator(
                "batch_plot_int", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": dt_b}
            )

            xbatch_traj = [x0_np]
            xk_ = x0_np.copy()
            for _ in range(N_batch_plot - 1):
                res_ = batch_plot_int(x0=xk_, p=0.0)
                xk_ = np.array(res_["xf"]).flatten()
                xbatch_traj.append(xk_)
            xbatch_traj = np.array(xbatch_traj)

            # b) Fase fed-batch: integrando de 5 a 24 h con dt fino
            t_fb_plot = np.linspace(t_batch, t_total, 400)   # algo denso
            dt_fb_plot = t_fb_plot[1] - t_fb_plot[0]

            fb_plot_int = ca.integrator(
                "fb_plot_int", "idas",
                {"x": x_sym, "p": u_sym, "ode": ode_expr},
                {"t0": 0, "tf": dt_fb_plot}
            )

            xfb_traj = []
            xk_ = xbatch_traj[-1].copy()
            for i, t_ in enumerate(t_fb_plot):
                xfb_traj.append(xk_)
                if i == len(t_fb_plot) - 1:
                    break
                # Determinar en qué subintervalo k estamos
                kk_ = int((t_ - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, kk_)
                kk_ = min(n_fb_intervals - 1, kk_)
                # Tomar F correspondiente
                F_now = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                # Apagar F si V>=Vmax
                if xk_[4] >= V_max_input:
                    F_now = 0.0
                # Integrar
                res_ = fb_plot_int(x0=xk_, p=F_now)
                xk_ = np.array(res_["xf"]).flatten()

            xfb_traj = np.array(xfb_traj)

            # Unimos
            t_full = np.concatenate([t_batch_plot, t_fb_plot])
            x_full = np.vstack([xbatch_traj, xfb_traj])

            X_full = x_full[:, 0]
            S_full = x_full[:, 1]
            P_full = x_full[:, 2]
            O_full = x_full[:, 3]
            V_full = x_full[:, 4]

            # Construir F para graficar
            F_batch_plot = np.zeros_like(t_batch_plot)
            F_fb_plot = []
            for i, tt in enumerate(t_fb_plot):
                kk_ = int((tt - t_batch) // dt_fb) if dt_fb > 0 else 0
                kk_ = max(0, kk_)
                kk_ = min(n_fb_intervals - 1, kk_)
                valF = sol.value(F_col[kk_]) if n_fb_intervals > 0 else 0.0
                if xfb_traj[i, 4] >= V_max_input:
                    valF = 0.0
                F_fb_plot.append(valF)
            F_fb_plot = np.array(F_fb_plot)

            F_plot = np.concatenate([F_batch_plot, F_fb_plot])

            # ====================================================
            # 11) Gráficas
            # ====================================================
            fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            axs = axs.ravel()

            # F
            axs[0].plot(t_full, F_plot, linewidth=2)
            axs[0].set_title("Flujo de alimentación F(t)")
            axs[0].set_xlabel("Tiempo (h)")
            axs[0].set_ylabel("F (L/h)")
            axs[0].grid(True)

            # X
            axs[1].plot(t_full, X_full, linewidth=2)
            axs[1].set_title("Biomasa X(t)")
            axs[1].set_xlabel("Tiempo (h)")
            axs[1].set_ylabel("X (g/L)")
            axs[1].grid(True)

            # S
            axs[2].plot(t_full, S_full, linewidth=2)
            axs[2].axhline(S_max, color='r', linestyle='--', label="S_max")
            axs[2].set_title("Sustrato S(t)")
            axs[2].set_xlabel("Tiempo (h)")
            axs[2].set_ylabel("S (g/L)")
            axs[2].legend()
            axs[2].grid(True)

            # P
            axs[3].plot(t_full, P_full, linewidth=2)
            axs[3].set_title("Producto P(t)")
            axs[3].set_xlabel("Tiempo (h)")
            axs[3].set_ylabel("P (g/L)")
            axs[3].grid(True)

            # O
            axs[4].plot(t_full, O_full, linewidth=2)
            axs[4].set_title("Oxígeno disuelto O(t) (constante)")
            axs[4].set_xlabel("Tiempo (h)")
            axs[4].set_ylabel("O (g/L)")
            axs[4].grid(True)

            # V
            axs[5].plot(t_full, V_full, linewidth=2)
            axs[5].axhline(V_max_input, color='r', linestyle='--', label="V_max")
            axs[5].set_title("Volumen V(t)")
            axs[5].set_xlabel("Tiempo (h)")
            axs[5].set_ylabel("V (L)")
            axs[5].legend()
            axs[5].grid(True)

            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Producto total acumulado", f"{P_fin_val * V_fin_val:.2f} g")
                s_in_total = Sf_input * (V_fin_val - V0)
                rend = (P_fin_val * V_fin_val) / s_in_total if s_in_total > 1e-9 else 0
                st.metric("Rendimiento Producto/Sustrato", f"{rend:.3f} g/g")
            with col2:
                st.metric("Tiempo total del proceso", f"{t_total:.2f} h")
                st.metric("Volumen final", f"{V_fin_val:.2f} L")

        except Exception as e:
            st.error(f"Error en la optimización: {str(e)}")
            st.stop()

if __name__ == '__main__':
    rto_page()