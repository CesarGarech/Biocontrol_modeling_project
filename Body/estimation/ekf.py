import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def ekf_page():
    st.header("Estimation of States and Parameters with Extended Kalman Filter (EKF)")
    st.markdown("""
    This section simulates a batch bioprocess and uses an EKF to estimate the concentrations
    of Biomass (X), Substrate (S), Product (P), and two kinetic parameters
    ($\mu_{max}$, $Y_{X/S}$) based on simulated and noisy measurements of
    Dissolved Oxygen (DO), pH y Temperature (T).

    **You can adjusts:**
    * The **initial conditions** assumed by the EKF (the "initial guess").
    * The **initial uncertainty** about that guess (matrix $P_0$).
    * The **noise** levels assumed by the filter for the process ($Q$) and the measurements ($R$).
    Observe how these adjustments affect the EKF's ability to track actual values.
    """)

    st.sidebar.subheader("EKF Parameters and Simulation")

    # --- Parámetros Fijos del Modelo "Real" (No ajustables por usuario aquí) ---
    #     (Podrían ponerse en un expander si se quiere verlos)
    mu_max_real = 0.4     # (1/h)
    Yxs_real    = 0.5     # (gX/gS)
    Ks          = 0.1     # (g/L)
    alpha       = 0.1     # (gP/gX) - Relacionado con Ypx
    OD_sat      = 8.0     # mg/L
    k_OUR       = 0.5     # mgO2/(L*gX)
    pH0         = 7.0
    P0_meas_ref = 0.0     # P de referencia para cálculo de pH
    k_acid      = 0.2
    Tset        = 30      # (°C)
    k_Temp      = 0.02

    # --- Parámetros Ajustables por el Usuario ---
    t_final_ekf = st.sidebar.slider("Final time (h)", 5, 50, 20, key="ekf_tf")
    dt_ekf      = 0.1 # Fijo para esta simulación EKF

    st.sidebar.markdown("**EKF Initial Conditions**")
    X0_est  = st.sidebar.number_input("Estimated initial X (g/L)", 0.01, 5.0, 0.05, format="%.2f", key="ekf_x0e")
    S0_est  = st.sidebar.number_input("Estimated initial S (g/L)", 0.1, 50.0, 5.0, format="%.1f", key="ekf_s0e")
    P0_est  = st.sidebar.number_input("Estimated initial P (g/L)", 0.0, 10.0, 0.01, format="%.2f", key="ekf_p0e")
    mu0_est = st.sidebar.number_input("Estimated initial μmax (1/h)", 0.1, 1.0, 0.40, format="%.2f", key="ekf_mu0e")
    yxs0_est= st.sidebar.number_input("Estimated initial Yxs (g/g)", 0.1, 1.0, 0.50, format="%.2f", key="ekf_yxs0e")

    st.sidebar.markdown("**Initial Uncertainty $P_0$ (Diagonals)**")
    p0_X   = st.sidebar.number_input("P0 - X", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0x")
    p0_S   = st.sidebar.number_input("P0 - S", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0s")
    p0_P   = st.sidebar.number_input("P0 - P", 1e-4, 1.0, 0.03, format="%.4f", key="ekf_p0p")
    p0_mu  = st.sidebar.number_input("P0 - μmax", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0mu")
    p0_yxs = st.sidebar.number_input("P0 - Yxs", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_p0yxs")

    st.sidebar.markdown("**Process Noise $Q$ (Diagonals)**")
    q_X   = st.sidebar.number_input("Q - X", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_qx")
    q_S   = st.sidebar.number_input("Q - S", 1e-10, 1e-2, 1e-8, format="%.2e", key="ekf_qs")
    q_P   = st.sidebar.number_input("Q - P", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_qp")
    q_mu  = st.sidebar.number_input("Q - μmax", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_qmu")
    q_yxs = st.sidebar.number_input("Q - Yxs", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_qyxs")

    st.sidebar.markdown("**Measurement Noise $R$ (Diagonals)**")
    r_OD = st.sidebar.number_input("R - DO", 1e-4, 1.0, 0.05, format="%.4f", key="ekf_rod")
    r_pH = st.sidebar.number_input("R - pH", 1e-4, 1.0, 0.02, format="%.4f", key="ekf_rph")
    r_T  = st.sidebar.number_input("R - Temp", 1e-2, 5.0, 0.5, format="%.2f", key="ekf_rtemp")

    # Botón para ejecutar la simulación
    run_ekf = st.sidebar.button("Run EKF Simulation")

    # --- Definiciones CasADi (Fuera del botón para no redefinir) ---
    n_states_ekf = 5
    n_meas_ekf   = 3
    x_sym_ekf = ca.SX.sym('x', n_states_ekf)
    X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym_ekf)

    mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
    dX = mu_sym * X_sym
    dS = - (1 / Yxs_sym) * dX
    dP = alpha * dX
    dMu_max = 0
    dYxs = 0
    x_next_sym = x_sym_ekf + dt_ekf * ca.vertcat(dX, dS, dP, dMu_max, dYxs)
    f_func_ekf = ca.Function('f', [x_sym_ekf], [x_next_sym], ['x_k'], ['x_k_plus_1'])

    OD_val_sym = OD_sat - k_OUR * X_sym
    pH_val_sym = pH0 - k_acid * (P_sym - P0_meas_ref)
    T_val_sym  = Tset + k_Temp * (X_sym * S_sym)
    z_sym_ekf = ca.vertcat(OD_val_sym, pH_val_sym, T_val_sym)
    h_func_ekf = ca.Function('h', [x_sym_ekf], [z_sym_ekf], ['x'], ['z'])

    F_sym_ekf = ca.jacobian(x_next_sym, x_sym_ekf)
    H_sym_ekf = ca.jacobian(z_sym_ekf, x_sym_ekf)
    F_func_ekf = ca.Function('F', [x_sym_ekf], [F_sym_ekf], ['x'], ['Fk'])
    H_func_ekf = ca.Function('H', [x_sym_ekf], [H_sym_ekf], ['x'], ['Hk'])
    # --- Fin Definiciones CasADi ---


    if run_ekf:
        st.subheader("EKF Estimation Results")

        # --- Preparación basada en Inputs del Usuario ---
        time_vec_ekf = np.arange(0, t_final_ekf + dt_ekf, dt_ekf)
        N_ekf = len(time_vec_ekf)

        # Covarianzas de ruido desde sliders
        Q_ekf = np.diag([q_X, q_S, q_P, q_mu, q_yxs])
        R_ekf = np.diag([r_OD, r_pH, r_T])

        # Condiciones iniciales "reales" (fijas en este ejemplo)
        X0_real = 0.1
        S0_real = 5.0
        P0_real = 0.0
        x_real_ekf = np.array([[X0_real], [S0_real], [P0_real], [mu_max_real], [Yxs_real]])

        # Estimación inicial EKF desde sliders
        x_est_ekf = np.array([[X0_est], [S0_est], [P0_est], [mu0_est], [yxs0_est]])
        P_est_ekf = np.diag([p0_X, p0_S, p0_P, p0_mu, p0_yxs])

        # Arrays para guardar resultados
        x_real_hist = np.zeros((n_states_ekf, N_ekf))
        x_est_hist  = np.zeros((n_states_ekf, N_ekf))
        z_meas_hist = np.zeros((n_meas_ekf, N_ekf))

        # --- Bucle de Simulación EKF ---
        for k in range(N_ekf):
            # Guardar valores actuales
            x_real_hist[:, k] = x_real_ekf.flatten()
            x_est_hist[:, k]  = x_est_ekf.flatten()

            # (A) Generar medición "real"
            z_noisefree_dm = h_func_ekf(x_real_ekf)
            z_noisefree = z_noisefree_dm.full()
            noise_meas = np.random.multivariate_normal(np.zeros(n_meas_ekf), R_ekf).reshape(-1, 1)
            z_k = z_noisefree + noise_meas
            z_meas_hist[:, k] = z_k.flatten()

            if k < N_ekf - 1:
                # (B) Predicción EKF
                x_pred_dm = f_func_ekf(x_est_ekf)
                x_pred = x_pred_dm.full()
                Fk_dm = F_func_ekf(x_est_ekf)
                Fk = Fk_dm.full()
                P_pred = Fk @ P_est_ekf @ Fk.T + Q_ekf

                # (C) Corrección EKF
                Hk_dm = H_func_ekf(x_pred)
                Hk = Hk_dm.full()
                h_pred_dm = h_func_ekf(x_pred)
                h_pred = h_pred_dm.full()
                Sk = Hk @ P_pred @ Hk.T + R_ekf
                Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk) # Usar pinv
                y_k = z_k - h_pred
                x_upd = x_pred + Kk @ y_k
                P_upd = (np.eye(n_states_ekf) - Kk @ Hk) @ P_pred

                # Actualizar
                x_est_ekf = x_upd
                 # Forzar no-negatividad en estados estimados si se desea
                x_est_ekf[0:3] = np.maximum(x_est_ekf[0:3], 0)
                # Forzar positividad en parámetros estimados si se desea (cuidado con mu_max=0)
                x_est_ekf[3:] = np.maximum(x_est_ekf[3:], 1e-6)

                P_est_ekf = P_upd

                # (D) Avance del proceso real
                x_real_next_no_noise_dm = f_func_ekf(x_real_ekf)
                x_real_next_no_noise = x_real_next_no_noise_dm.full()
                noise_proc = np.random.multivariate_normal(np.zeros(n_states_ekf), Q_ekf).reshape(-1, 1)
                x_real_ekf = x_real_next_no_noise + noise_proc
                # Forzar no-negatividad estados reales físicos
                x_real_ekf[0:3] = np.maximum(x_real_ekf[0:3], 0)


        # --- Gráficas de Resultados EKF ---
        # (Usando los arrays x_real_hist, x_est_hist, z_meas_hist)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Figura 1: Estados y Mediciones
        fig1_ekf, axs1_ekf = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        fig1_ekf.suptitle('Estimation of States and Measurements (EKF)', fontsize=14)

        # Biomasa
        axs1_ekf[0, 0].plot(time_vec_ekf, x_real_hist[0, :], 'b-', label='X real')
        axs1_ekf[0, 0].plot(time_vec_ekf, x_est_hist[0, :], 'r--', label='X estimada')
        axs1_ekf[0, 0].set_ylabel('Biomass (g/L)')
        axs1_ekf[0, 0].legend()
        axs1_ekf[0, 0].grid(True)

        # Medición OD
        axs1_ekf[0, 1].plot(time_vec_ekf, z_meas_hist[0, :], 'k.-', markersize=3, linewidth=1, label='OD medido')
        axs1_ekf[0, 1].set_ylabel('DO (mg/L)')
        axs1_ekf[0, 1].set_title('DO measurement')
        axs1_ekf[0, 1].legend()
        axs1_ekf[0, 1].grid(True)

        # Sustrato
        axs1_ekf[1, 0].plot(time_vec_ekf, x_real_hist[1, :], 'b-', label='S real')
        axs1_ekf[1, 0].plot(time_vec_ekf, x_est_hist[1, :], 'r--', label='S estimada')
        axs1_ekf[1, 0].set_ylabel('Substrate (g/L)')
        axs1_ekf[1, 0].legend()
        axs1_ekf[1, 0].grid(True)

        # Medición pH
        axs1_ekf[1, 1].plot(time_vec_ekf, z_meas_hist[1, :], 'k.-', markersize=3, linewidth=1, label='pH medido')
        axs1_ekf[1, 1].set_ylabel('pH')
        axs1_ekf[1, 1].set_title('pH measurement')
        axs1_ekf[1, 1].legend()
        axs1_ekf[1, 1].grid(True)

        # Producto
        axs1_ekf[2, 0].plot(time_vec_ekf, x_real_hist[2, :], 'b-', label='P real')
        axs1_ekf[2, 0].plot(time_vec_ekf, x_est_hist[2, :], 'r--', label='P estimada')
        axs1_ekf[2, 0].set_xlabel('Time (h)')
        axs1_ekf[2, 0].set_ylabel('Product (g/L)')
        axs1_ekf[2, 0].legend()
        axs1_ekf[2, 0].grid(True)

        # Medición Temperatura
        axs1_ekf[2, 1].plot(time_vec_ekf, z_meas_hist[2, :], 'k.-', markersize=3, linewidth=1, label='T medida')
        axs1_ekf[2, 1].set_xlabel('Time (h)')
        axs1_ekf[2, 1].set_ylabel('Temperature (°C)')
        axs1_ekf[2, 1].set_title('Temperature Measurement')
        axs1_ekf[2, 1].legend()
        axs1_ekf[2, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        st.pyplot(fig1_ekf)


        # Figura 2: Parámetros Estimados
        fig2_ekf, axs2_ekf = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        fig2_ekf.suptitle('Parameter Estimation (EKF)', fontsize=14)

        # mu_max
        axs2_ekf[0].plot(time_vec_ekf, x_real_hist[3, :], 'b-', label=r'$\mu_{max}$ real')
        axs2_ekf[0].plot(time_vec_ekf, x_est_hist[3, :], 'r--', label=r'$\mu_{max}$ estimada')
        axs2_ekf[0].set_ylabel(r'$\mu_{max}$ (1/h)')
        axs2_ekf[0].legend()
        axs2_ekf[0].grid(True)

        # Yxs
        axs2_ekf[1].plot(time_vec_ekf, x_real_hist[4, :], 'b-', label=r'$Y_{X/S}$ real')
        axs2_ekf[1].plot(time_vec_ekf, x_est_hist[4, :], 'r--', label=r'$Y_{X/S}$ estimada')
        axs2_ekf[1].set_xlabel('Time (h)')
        axs2_ekf[1].set_ylabel(r'$Y_{X/S}$ (gX/gS)')
        axs2_ekf[1].legend()
        axs2_ekf[1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        st.pyplot(fig2_ekf)

        st.write("Final Values:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Actual**")
            st.metric("X", f"{x_real_ekf[0,0]:.3f} g/L")
            st.metric("S", f"{x_real_ekf[1,0]:.3f} g/L")
            st.metric("P", f"{x_real_ekf[2,0]:.3f} g/L")
            st.metric("μmax", f"{x_real_ekf[3,0]:.3f} 1/h")
            st.metric("Yxs", f"{x_real_ekf[4,0]:.3f} g/g")
        with col2:
            st.write("**Estimated**")
            st.metric("X est.", f"{x_est_ekf[0,0]:.3f} g/L")
            st.metric("S est.", f"{x_est_ekf[1,0]:.3f} g/L")
            st.metric("P est.", f"{x_est_ekf[2,0]:.3f} g/L")
            st.metric("μmax est.", f"{x_est_ekf[3,0]:.3f} 1/h")
            st.metric("Yxs est.", f"{x_est_ekf[4,0]:.3f} g/g")

    else:
        st.info("Set the parameters in the sidebar and click on 'Run EKF Simulation'.")

    if __name__ == '__main__':
        ekf_page()