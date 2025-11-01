import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

def ekf_nmpc_page():
    st.header("EKF-NMPC: Extended Kalman Filter with Nonlinear Model Predictive Control")
    st.markdown("""
    This section integrates an **Extended Kalman Filter (EKF)** with **Nonlinear Model Predictive Control (NMPC)**
    for a bioreactor system.
    
    **Process Flow:**
    1. **EKF** estimates states (Biomass X, Substrate S, Product P) and parameters (μ_max, Y_XS) from noisy measurements (DO, pH, T)
    2. **NMPC** uses the EKF-estimated states to compute optimal control actions (F_S, Q_j)
    3. Control actions are applied to the real plant with process noise
    4. New noisy measurements are generated and fed back to the EKF
    
    **Model States:** X (biomass), S (substrate), P (product), μ_max, Y_XS  
    **Measurements:** DO (dissolved oxygen), pH, Temperature  
    **Control Inputs:** F_S (substrate flow), Q_j (thermal load)  
    **Control Objectives:** Track setpoints for X and T
    """)

    # Fixed model parameters (not adjustable)
    Ks = 0.1          # Monod constant (g/L)
    alpha = 0.1       # Product yield (gP/gX)
    DO_sat = 8.0      # Saturated DO (mg/L)
    k_OUR = 0.5       # Oxygen uptake rate coefficient (mgO2/(L*gX))
    pH0 = 7.0         # Nominal pH
    P0_ref = 0.0      # Reference product for pH
    k_acid = 0.2      # pH acidification coefficient
    Tset = 303.15     # Temperature setpoint base (K) - 30°C
    k_Temp = 0.02     # Temperature coupling coefficient
    
    # NMPC model parameters
    Y_QX = 15000.0    # Heat/Biomass yield (J/g)
    S_in = 10.0       # Inlet substrate concentration (g/L)
    V = 1.0           # Reactor volume (L)
    rho = 1000.0      # Density (g/L)
    Cp = 4184.0       # Heat capacity (J/(g*K))
    T_in = 298.15     # Inlet temperature (K) - 25°C
    F_const = 0.0     # Additional constant flow (L/h)

    # --- Sidebar Configuration ---
    st.sidebar.subheader("Simulation Parameters")
    
    # Simulation time
    t_final = st.sidebar.slider("Total Simulation Time (h)", 5, 50, 24, key="ekf_nmpc_tf")
    dt = st.sidebar.number_input("Time Step (h)", 0.01, 0.5, 0.1, step=0.01, format="%.2f", key="ekf_nmpc_dt")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("EKF Configuration")
    
    # Real initial conditions (hidden from user, represents actual plant)
    X0_real = 1.0
    S0_real = 8.0
    P0_real = 0.0
    mu_max_real = 0.4
    Yxs_real = 0.5
    
    # EKF initial estimates
    st.sidebar.markdown("**Initial State Estimates**")
    X0_est = st.sidebar.number_input("Est. initial X (g/L)", 0.01, 5.0, 0.5, format="%.2f", key="ekf_nmpc_x0e")
    S0_est = st.sidebar.number_input("Est. initial S (g/L)", 0.1, 20.0, 5.0, format="%.1f", key="ekf_nmpc_s0e")
    P0_est = st.sidebar.number_input("Est. initial P (g/L)", 0.0, 5.0, 0.01, format="%.2f", key="ekf_nmpc_p0e")
    mu0_est = st.sidebar.number_input("Est. initial μ_max (1/h)", 0.1, 1.0, 0.35, format="%.2f", key="ekf_nmpc_mu0e")
    yxs0_est = st.sidebar.number_input("Est. initial Y_XS (g/g)", 0.1, 1.0, 0.45, format="%.2f", key="ekf_nmpc_yxs0e")
    
    st.sidebar.markdown("**EKF Covariance Matrices**")
    # Initial uncertainty P0
    p0_X = st.sidebar.number_input("P0 - X", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_nmpc_p0x")
    p0_S = st.sidebar.number_input("P0 - S", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_nmpc_p0s")
    p0_P = st.sidebar.number_input("P0 - P", 1e-4, 1.0, 0.03, format="%.4f", key="ekf_nmpc_p0p")
    p0_mu = st.sidebar.number_input("P0 - μ_max", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_nmpc_p0mu")
    p0_yxs = st.sidebar.number_input("P0 - Y_XS", 1e-4, 1.0, 0.01, format="%.4f", key="ekf_nmpc_p0yxs")
    
    # Process noise Q
    q_X = st.sidebar.number_input("Q - X", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_nmpc_qx")
    q_S = st.sidebar.number_input("Q - S", 1e-10, 1e-2, 1e-7, format="%.2e", key="ekf_nmpc_qs")
    q_P = st.sidebar.number_input("Q - P", 1e-8, 1e-2, 1e-5, format="%.2e", key="ekf_nmpc_qp")
    q_mu = st.sidebar.number_input("Q - μ_max", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_nmpc_qmu")
    q_yxs = st.sidebar.number_input("Q - Y_XS", 1e-8, 1e-2, 1e-6, format="%.2e", key="ekf_nmpc_qyxs")
    
    # Measurement noise R
    r_DO = st.sidebar.number_input("R - DO", 1e-4, 1.0, 0.05, format="%.4f", key="ekf_nmpc_rdo")
    r_pH = st.sidebar.number_input("R - pH", 1e-4, 1.0, 0.02, format="%.4f", key="ekf_nmpc_rph")
    r_T = st.sidebar.number_input("R - Temp", 1e-2, 5.0, 0.5, format="%.2f", key="ekf_nmpc_rt")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("NMPC Configuration")
    
    N = st.sidebar.number_input("Prediction Horizon (N)", 1, 30, 10, key="ekf_nmpc_N")
    M = st.sidebar.number_input("Control Horizon (M)", 1, 20, 4, key="ekf_nmpc_M")
    
    st.sidebar.markdown("**NMPC Weights**")
    Q_X_weight = st.sidebar.number_input("Weight Q - X tracking", 0.1, 100.0, 10.0, key="ekf_nmpc_qx_weight")
    Q_T_weight = st.sidebar.number_input("Weight Q - T tracking", 0.001, 10.0, 1.0, key="ekf_nmpc_qt_weight")
    W_FS_weight = st.sidebar.number_input("Weight W - ΔF_S", 0.001, 10.0, 0.1, key="ekf_nmpc_wfs")
    W_Qj_weight = st.sidebar.number_input("Weight W - ΔQ_j (×10⁻⁸)", 0.1, 100.0, 1.0, format="%.1f", key="ekf_nmpc_wqj")
    W_Qj_weight = W_Qj_weight * 1e-8  # Scale to reasonable value
    
    st.sidebar.markdown("**Control Input Limits**")
    min_FS = st.sidebar.number_input("Min F_S (L/h)", 0.0, 1.0, 0.0, key="ekf_nmpc_min_fs")
    max_FS = st.sidebar.number_input("Max F_S (L/h)", 0.5, 5.0, 1.5, key="ekf_nmpc_max_fs")
    min_Qj = st.sidebar.number_input("Min Q_j (W)", -20000.0, 0.0, -10000.0, format="%.0f", key="ekf_nmpc_min_qj")
    max_Qj = st.sidebar.number_input("Max Q_j (W)", 0.0, 20000.0, 10000.0, format="%.0f", key="ekf_nmpc_max_qj")
    
    st.sidebar.markdown("**Control Input Rate Limits**")
    delta_FS_max = st.sidebar.number_input("Max |ΔF_S| (L/h)", 0.01, 1.0, 0.2, key="ekf_nmpc_dfs")
    delta_Qj_max = st.sidebar.number_input("Max |ΔQ_j| (W)", 100.0, 10000.0, 5000.0, format="%.0f", key="ekf_nmpc_dqj")
    
    st.sidebar.markdown("**State Limits**")
    min_X_opt = st.sidebar.number_input("Min X (g/L)", 0.0, 1.0, 0.0, key="ekf_nmpc_min_x")
    max_X_opt = st.sidebar.number_input("Max X (g/L)", 3.0, 10.0, 5.0, key="ekf_nmpc_max_x")
    min_S_opt = st.sidebar.number_input("Min S (g/L)", 0.0, 1.0, 0.0, key="ekf_nmpc_min_s")
    max_S_opt = st.sidebar.number_input("Max S (g/L)", 5.0, 20.0, 15.0, key="ekf_nmpc_max_s")
    min_T_opt = st.sidebar.number_input("Min T (K)", 290.0, 300.0, 295.0, key="ekf_nmpc_min_t")
    max_T_opt = st.sidebar.number_input("Max T (K)", 310.0, 320.0, 315.0, key="ekf_nmpc_max_t")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Control Setpoints")
    
    # Initial setpoints
    sp_X_initial = st.sidebar.number_input("Initial X setpoint (g/L)", 0.5, 5.0, 1.0, key="ekf_nmpc_sp_x0")
    sp_T_initial = st.sidebar.number_input("Initial T setpoint (K)", 295.0, 315.0, 305.0, key="ekf_nmpc_sp_t0")
    
    # Setpoint changes
    t_sp1 = st.sidebar.number_input("1st change time (h)", 0.0, 50.0, 8.0, key="ekf_nmpc_t_sp1")
    sp_X_t1 = st.sidebar.number_input("X setpoint at t1 (g/L)", 0.5, 5.0, 2.5, key="ekf_nmpc_sp_x1")
    sp_T_t1 = st.sidebar.number_input("T setpoint at t1 (K)", 295.0, 315.0, 308.0, key="ekf_nmpc_sp_t1")
    
    t_sp2 = st.sidebar.number_input("2nd change time (h)", 0.0, 50.0, 16.0, key="ekf_nmpc_t_sp2")
    sp_X_t2 = st.sidebar.number_input("X setpoint at t2 (g/L)", 0.5, 5.0, 1.5, key="ekf_nmpc_sp_x2")
    sp_T_t2 = st.sidebar.number_input("T setpoint at t2 (K)", 295.0, 315.0, 303.0, key="ekf_nmpc_sp_t2")
    
    # Initial control inputs
    st.sidebar.markdown("**Initial Control Inputs**")
    initial_FS = st.sidebar.number_input("Initial F_S (L/h)", 0.0, 2.0, 0.1, key="ekf_nmpc_fs0")
    initial_Qj = st.sidebar.number_input("Initial Q_j (W)", -5000.0, 5000.0, 0.0, key="ekf_nmpc_qj0")
    
    # Run button
    run_simulation = st.sidebar.button("Run EKF-NMPC Simulation")
    
    # --- CasADi Model Definitions (shared by EKF and NMPC) ---
    # EKF model: 5 states [X, S, P, μ_max, Y_XS]
    n_states_ekf = 5
    n_meas = 3
    x_ekf_sym = ca.SX.sym('x_ekf', n_states_ekf)
    X_ekf, S_ekf, P_ekf, mu_max_ekf, Yxs_ekf = ca.vertsplit(x_ekf_sym)
    
    # Control inputs for dynamics
    u_sym = ca.SX.sym('u', 2)  # [F_S, Q_j]
    F_S_sym = u_sym[0]
    Q_j_sym = u_sym[1]
    
    # EKF dynamics (with control inputs)
    mu_ekf = mu_max_ekf * (S_ekf / (Ks + S_ekf))
    F_total = F_S_sym + F_const
    D = F_total / V
    
    dX_ekf = mu_ekf * X_ekf - D * X_ekf
    dS_ekf = D * (S_in - S_ekf) - (1.0 / Yxs_ekf) * mu_ekf * X_ekf
    dP_ekf = alpha * mu_ekf * X_ekf - D * P_ekf
    dMu_max = 0
    dYxs = 0
    
    x_ekf_dot = ca.vertcat(dX_ekf, dS_ekf, dP_ekf, dMu_max, dYxs)
    
    # Discrete-time EKF prediction (Euler)
    dt_sym = ca.SX.sym('dt')
    x_ekf_next = x_ekf_sym + dt_sym * x_ekf_dot
    f_ekf_func = ca.Function('f_ekf', [x_ekf_sym, u_sym, dt_sym], [x_ekf_next])
    
    # EKF measurement model
    DO_val = DO_sat - k_OUR * X_ekf
    pH_val = pH0 - k_acid * (P_ekf - P0_ref)
    T_val = Tset + k_Temp * (X_ekf * S_ekf)
    z_ekf = ca.vertcat(DO_val, pH_val, T_val)
    h_ekf_func = ca.Function('h_ekf', [x_ekf_sym], [z_ekf])
    
    # EKF Jacobians
    F_ekf_jac = ca.jacobian(x_ekf_next, x_ekf_sym)
    H_ekf_jac = ca.jacobian(z_ekf, x_ekf_sym)
    F_ekf_func = ca.Function('F_ekf', [x_ekf_sym, u_sym, dt_sym], [F_ekf_jac])
    H_ekf_func = ca.Function('H_ekf', [x_ekf_sym], [H_ekf_jac])
    
    # NMPC model: 3 states [X, S, T] (using estimated parameters from EKF)
    x_nmpc_sym = ca.MX.sym('x_nmpc', 3)  # [X, S, T]
    X_nmpc, S_nmpc, T_nmpc = ca.vertsplit(x_nmpc_sym)
    
    u_nmpc_sym = ca.MX.sym('u_nmpc', 2)  # [F_S, Q_j]
    F_S_nmpc = u_nmpc_sym[0]
    Q_j_nmpc = u_nmpc_sym[1]
    
    # Parameters from EKF estimates
    mu_max_param = ca.MX.sym('mu_max_param')
    Yxs_param = ca.MX.sym('Yxs_param')
    
    # NMPC dynamics
    mu_nmpc = mu_max_param * S_nmpc / (Ks + S_nmpc)
    F_total_nmpc = F_S_nmpc + F_const
    D_nmpc = F_total_nmpc / V
    
    dX_nmpc = mu_nmpc * X_nmpc - D_nmpc * X_nmpc
    dS_nmpc = D_nmpc * (S_in - S_nmpc) - (mu_nmpc / Yxs_param) * X_nmpc
    
    # Energy balance
    Q_rem = -Q_j_nmpc * 3600.0  # W to J/h
    Q_gen = Y_QX * mu_nmpc * X_nmpc * V
    Q_flow = F_total_nmpc * rho * Cp * (T_in - T_nmpc)
    dT_nmpc = (Q_gen + Q_rem + Q_flow) / (rho * Cp * V)
    
    dx_nmpc = ca.vertcat(dX_nmpc, dS_nmpc, dT_nmpc)
    
    # NMPC output (controlled variables)
    c_nmpc = ca.vertcat(X_nmpc, T_nmpc)
    
    model_ode_nmpc = ca.Function('model_ode_nmpc', 
                                   [x_nmpc_sym, u_nmpc_sym, mu_max_param, Yxs_param], 
                                   [dx_nmpc])
    output_func_nmpc = ca.Function('output_func_nmpc', [x_nmpc_sym], [c_nmpc])
    
    # --- NMPC Class ---
    class NMPCBioreactor:
        def __init__(self, dt, N, M, Q, W, model_ode, output_func, 
                     lbx, ubx, lbu, ubu, lbdu, ubdu, m=3):
            self.dt = dt
            self.N = N
            self.M = M
            self.Q = np.diag(Q)
            self.W = np.diag(W)
            self.model_ode = model_ode
            self.output_func = output_func
            self.nx = 3  # [X, S, T]
            self.nu = 2  # [F_S, Q_j]
            self.nc = 2  # [X, T]
            self.lbx = lbx
            self.ubx = ubx
            self.lbu = lbu
            self.ubu = ubu
            self.lbdu = lbdu
            self.ubdu = ubdu
            self.m = m
            
            # Collocation setup
            self.tau_root = np.append(0, ca.collocation_points(self.m, 'legendre'))
            self.C = np.zeros((self.m + 1, self.m + 1))
            self.D = np.zeros(self.m + 1)
            EPSILON = 1e-10  # Small value to prevent division by zero in polynomial construction
            for j in range(self.m + 1):
                p = np.poly1d([1])
                for r in range(self.m + 1):
                    if r != j:
                        p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j] - self.tau_root[r] + EPSILON)
                p_der = np.polyder(p)
                for i in range(self.m + 1):
                    self.C[j, i] = np.polyval(p_der, self.tau_root[i])
                p_int = np.polyint(p)
                self.D[j] = np.polyval(p_int, 1.0)
            
            self._build_nlp()
            self.w0 = np.zeros(self.dim_w)
        
        def _build_nlp(self):
            # Constraint tolerance for equality constraints (numerical stability)
            CONSTRAINT_TOL = 1e-9
            
            # Collocation step function
            Xk_step = ca.MX.sym('Xk_step', self.nx)
            Xc_step = ca.MX.sym('Xc_step', self.nx, self.m)
            Uk_step = ca.MX.sym('Uk_step', self.nu)
            mu_param = ca.MX.sym('mu_param')
            Yxs_param = ca.MX.sym('Yxs_param')
            
            X_all_coll = ca.horzcat(Xk_step, Xc_step)
            ode_at_coll = []
            for j in range(1, self.m + 1):
                ode_at_coll.append(self.model_ode(X_all_coll[:, j], Uk_step, mu_param, Yxs_param))
            
            coll_eqs = []
            for j in range(1, self.m + 1):
                xp_coll = sum(self.C[r, j] * X_all_coll[:, r] for r in range(self.m + 1))
                coll_eqs.append(xp_coll - self.dt * ode_at_coll[j-1])
            
            Xk_end = Xk_step + self.dt * sum(self.D[j] * ode_at_coll[j-1] for j in range(1, self.m + 1))
            
            self.F_coll = ca.Function('F_coll', 
                                       [Xk_step, Xc_step, Uk_step, mu_param, Yxs_param],
                                       [Xk_end, ca.vertcat(*coll_eqs)])
            
            # NLP variables
            self.w = []
            self.lbw = []
            self.ubw = []
            self.g = []
            self.lbg = []
            self.ubg = []
            
            # Parameters
            self.x0_sym = ca.MX.sym('x0', self.nx)
            self.sp_sym = ca.MX.sym('sp', self.nc, self.N)
            self.uprev_sym = ca.MX.sym('uprev', self.nu)
            self.mu_max_param_sym = ca.MX.sym('mu_max_param')
            self.Yxs_param_sym = ca.MX.sym('Yxs_param')
            
            p_nlp = ca.vertcat(self.x0_sym, ca.vec(self.sp_sym), self.uprev_sym,
                               self.mu_max_param_sym, self.Yxs_param_sym)
            
            J = 0
            Uk_prev = self.uprev_sym
            Xk_iter = self.x0_sym
            
            for k in range(self.N):
                # Control input
                Uk_k = ca.MX.sym(f'U_{k}', self.nu)
                self.w.append(Uk_k)
                self.lbw.extend(self.lbu)
                self.ubw.extend(self.ubu)
                
                # Delta u constraints
                delta_u = Uk_k - Uk_prev
                self.g.append(delta_u)
                self.lbg.extend(self.lbdu)
                self.ubg.extend(self.ubdu)
                
                # Control horizon constraint
                if k >= self.M:
                    self.g.append(Uk_k - Uk_prev_M)
                    self.lbg.extend([-CONSTRAINT_TOL] * self.nu)
                    self.ubg.extend([+CONSTRAINT_TOL] * self.nu)
                
                # Collocation states
                Xc_k = ca.MX.sym(f'Xc_{k}', self.nx, self.m)
                self.w.append(ca.vec(Xc_k))
                self.lbw.extend(self.lbx * self.m)
                self.ubw.extend(self.ubx * self.m)
                
                # Apply collocation
                Xk_end, coll_eqs = self.F_coll(Xk_iter, Xc_k, Uk_k, 
                                                self.mu_max_param_sym, self.Yxs_param_sym)
                
                # Collocation equations
                self.g.append(coll_eqs)
                self.lbg.extend([-CONSTRAINT_TOL] * self.nx * self.m)
                self.ubg.extend([+CONSTRAINT_TOL] * self.nx * self.m)
                
                # Next state
                Xk_next = ca.MX.sym(f'X_{k+1}', self.nx)
                self.w.append(Xk_next)
                self.lbw.extend(self.lbx)
                self.ubw.extend(self.ubx)
                
                # Continuity
                self.g.append(Xk_end - Xk_next)
                self.lbg.extend([-CONSTRAINT_TOL] * self.nx)
                self.ubg.extend([+CONSTRAINT_TOL] * self.nx)
                
                # Cost
                Ck_next = self.output_func(Xk_next)
                sp_k = self.sp_sym[:, k]
                J += ca.mtimes([(Ck_next - sp_k).T, self.Q, (Ck_next - sp_k)])
                J += ca.mtimes([delta_u.T, self.W, delta_u])
                
                # Update
                Xk_iter = Xk_next
                Uk_prev = Uk_k
                if k == self.M - 1:
                    Uk_prev_M = Uk_k
            
            # Create solver
            nlp_dict = {
                'f': J,
                'x': ca.vertcat(*self.w),
                'g': ca.vertcat(*self.g),
                'p': p_nlp
            }
            opts = {
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.max_iter': 150,
                'ipopt.tol': 1e-6,
                'ipopt.warm_start_init_point': 'yes'
            }
            self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)
            
            self._prepare_indices()
        
        def _prepare_indices(self):
            self.indices = {'X': [], 'U': [], 'Xc': []}
            offset = 0
            for k in range(self.N):
                self.indices['U'].append(offset)
                offset += self.nu
                self.indices['Xc'].append(offset)
                offset += self.nx * self.m
                self.indices['X'].append(offset)
                offset += self.nx
            self.dim_w = offset
        
        def solve(self, x_current, sp_trajectory, u_previous, mu_max_est, Yxs_est):
            # Prepare parameters
            p_val = np.concatenate([
                x_current,
                sp_trajectory.flatten('F'),
                u_previous,
                [mu_max_est, Yxs_est]
            ])
            
            # Warm start
            if len(self.w0) != self.dim_w or np.all(np.abs(self.w0) < 1e-9):
                # Initialize guess
                w0_guess = []
                x_guess = np.array(x_current)
                u_guess = np.array(u_previous)
                for k in range(self.N):
                    w0_guess.extend(np.clip(u_guess, self.lbu, self.ubu))
                    w0_guess.extend(np.tile(np.clip(x_guess, self.lbx, self.ubx), self.m))
                    # Simple prediction
                    try:
                        dx = self.model_ode(x_guess, u_guess, mu_max_est, Yxs_est).full().flatten()
                        x_guess = x_guess + dx * self.dt
                    except:
                        pass
                    w0_guess.extend(np.clip(x_guess, self.lbx, self.ubx))
                
                if len(w0_guess) == self.dim_w:
                    self.w0 = np.array(w0_guess)
                else:
                    self.w0 = np.zeros(self.dim_w)
            
            # Solve
            try:
                sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw,
                                  lbg=self.lbg, ubg=self.ubg, p=p_val)
                w_opt = sol['x'].full().flatten()
                sol_stats = self.solver.stats()
                
                if not sol_stats['success']:
                    return u_previous, sol_stats
                
                self.w0 = w_opt
                u_apply = w_opt[self.indices['U'][0]:self.indices['U'][0] + self.nu]
                
                return u_apply, sol_stats
            
            except Exception as e:
                return u_previous, {'success': False, 'error': str(e)}
    
    # --- Main Simulation ---
    if run_simulation:
        st.subheader("EKF-NMPC Simulation Results")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Setup
        N_steps = int(t_final / dt)
        time_vec = np.arange(0, t_final + dt, dt)
        N_steps = len(time_vec) - 1
        
        # Covariance matrices
        Q_ekf = np.diag([q_X, q_S, q_P, q_mu, q_yxs])
        R_ekf = np.diag([r_DO, r_pH, r_T])
        
        # Real plant state (5 states for EKF model)
        x_real = np.array([[X0_real], [S0_real], [P0_real], [mu_max_real], [Yxs_real]])
        
        # EKF estimates
        x_est = np.array([[X0_est], [S0_est], [P0_est], [mu0_est], [yxs0_est]])
        P_est = np.diag([p0_X, p0_S, p0_P, p0_mu, p0_yxs])
        
        # Control inputs
        u_current = np.array([initial_FS, initial_Qj])
        
        # Current setpoint
        current_sp = np.array([sp_X_initial, sp_T_initial])
        
        # Setpoint schedule
        sp_schedule = {
            t_sp1: np.array([sp_X_t1, sp_T_t1]),
            t_sp2: np.array([sp_X_t2, sp_T_t2])
        }
        
        # History arrays
        x_real_hist = np.zeros((n_states_ekf, N_steps + 1))
        x_est_hist = np.zeros((n_states_ekf, N_steps + 1))
        z_meas_hist = np.zeros((n_meas, N_steps + 1))
        u_hist = np.zeros((2, N_steps))
        sp_hist = np.zeros((2, N_steps + 1))
        
        # NMPC limits
        lbx_nmpc = [min_X_opt, min_S_opt, min_T_opt]
        ubx_nmpc = [max_X_opt, max_S_opt, max_T_opt]
        lbu_nmpc = [min_FS, min_Qj]
        ubu_nmpc = [max_FS, max_Qj]
        lbdu_nmpc = [-delta_FS_max, -delta_Qj_max]
        ubdu_nmpc = [delta_FS_max, delta_Qj_max]
        Q_weights = [Q_X_weight, Q_T_weight]
        W_weights = [W_FS_weight, W_Qj_weight]
        
        # Initialize NMPC
        try:
            nmpc = NMPCBioreactor(dt, N, M, Q_weights, W_weights,
                                  model_ode_nmpc, output_func_nmpc,
                                  lbx_nmpc, ubx_nmpc, lbu_nmpc, ubu_nmpc,
                                  lbdu_nmpc, ubdu_nmpc)
        except Exception as e:
            st.error(f"Error initializing NMPC: {e}")
            st.stop()
        
        # Initial values
        x_real_hist[:, 0] = x_real.flatten()
        x_est_hist[:, 0] = x_est.flatten()
        sp_hist[:, 0] = current_sp
        
        # Generate initial measurement
        z_true = h_ekf_func(x_real).full()
        z_meas = z_true + np.random.multivariate_normal(np.zeros(n_meas), R_ekf).reshape(-1, 1)
        z_meas_hist[:, 0] = z_meas.flatten()
        
        # Simulation loop
        for k in range(N_steps):
            t_current = k * dt
            status_text.text(f"Simulating step {k+1}/{N_steps} (t={t_current:.2f}h)...")
            
            # Check for setpoint changes
            for sp_time, new_sp in sorted(sp_schedule.items()):
                if t_current < sp_time <= t_current + dt + 1e-9:
                    current_sp = new_sp
                    break
            
            sp_hist[:, k] = current_sp
            
            # (1) EKF Prediction
            x_pred_dm = f_ekf_func(x_est, u_current, dt)
            x_pred = x_pred_dm.full()
            
            Fk_dm = F_ekf_func(x_est, u_current, dt)
            Fk = Fk_dm.full()
            P_pred = Fk @ P_est @ Fk.T + Q_ekf
            
            # (2) EKF Update
            Hk_dm = H_ekf_func(x_pred)
            Hk = Hk_dm.full()
            
            h_pred_dm = h_ekf_func(x_pred)
            h_pred = h_pred_dm.full()
            
            Sk = Hk @ P_pred @ Hk.T + R_ekf
            Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk)
            
            innovation = z_meas - h_pred
            x_upd = x_pred + Kk @ innovation
            P_upd = (np.eye(n_states_ekf) - Kk @ Hk) @ P_pred
            
            # Update estimates with constraints
            x_est = x_upd
            x_est[0:3] = np.maximum(x_est[0:3], 0)  # Non-negative states
            x_est[3:] = np.maximum(x_est[3:], 1e-6)  # Positive parameters
            P_est = P_upd
            
            x_est_hist[:, k+1] = x_est.flatten()
            
            # (3) NMPC Control - use EKF estimates
            # Extract state estimates for NMPC [X, S, T from measurement model]
            x_nmpc_current = np.array([x_est[0, 0], x_est[1, 0], 
                                       z_meas[2, 0]])  # Use measured T
            mu_max_est = x_est[3, 0]
            Yxs_est = x_est[4, 0]
            
            # Create setpoint trajectory
            sp_traj = np.tile(current_sp, (N, 1)).T
            
            # Solve NMPC
            u_optimal, stats = nmpc.solve(x_nmpc_current, sp_traj, u_current,
                                          mu_max_est, Yxs_est)
            
            if not stats['success']:
                u_apply = u_current  # Keep previous input on failure
            else:
                u_apply = u_optimal
                # Apply rate limits
                delta_u = u_apply - u_current
                delta_u = np.clip(delta_u, lbdu_nmpc, ubdu_nmpc)
                u_apply = u_current + delta_u
                u_apply = np.clip(u_apply, lbu_nmpc, ubu_nmpc)
            
            u_hist[:, k] = u_apply
            
            # (4) Apply control to real plant
            x_real_next_dm = f_ekf_func(x_real, u_apply, dt)
            x_real_next = x_real_next_dm.full()
            
            # Add process noise
            proc_noise = np.random.multivariate_normal(np.zeros(n_states_ekf), Q_ekf).reshape(-1, 1)
            x_real = x_real_next + proc_noise
            x_real[0:3] = np.maximum(x_real[0:3], 0)  # Physical constraints
            
            x_real_hist[:, k+1] = x_real.flatten()
            
            # (5) Generate new measurement
            z_true = h_ekf_func(x_real).full()
            meas_noise = np.random.multivariate_normal(np.zeros(n_meas), R_ekf).reshape(-1, 1)
            z_meas = z_true + meas_noise
            z_meas_hist[:, k+1] = z_meas.flatten()
            
            # Update control
            u_current = u_apply
            
            # Update progress
            progress_bar.progress((k + 1) / N_steps)
        
        # Final setpoint
        sp_hist[:, -1] = current_sp
        
        status_text.text("Simulation completed!")
        st.success("EKF-NMPC simulation finished successfully!")
        
        # --- Plotting Results ---
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Figure 1: EKF Estimation Performance
        fig1, axs1 = plt.subplots(3, 2, figsize=(14, 10))
        fig1.suptitle('EKF State Estimation Performance', fontsize=14, fontweight='bold')
        
        # Biomass
        axs1[0, 0].plot(time_vec, x_real_hist[0, :], 'b-', linewidth=2, label='Real X')
        axs1[0, 0].plot(time_vec, x_est_hist[0, :], 'r--', linewidth=2, label='Estimated X')
        axs1[0, 0].set_ylabel('Biomass (g/L)', fontsize=10)
        axs1[0, 0].legend(fontsize=9)
        axs1[0, 0].grid(True, alpha=0.3)
        
        # Substrate
        axs1[1, 0].plot(time_vec, x_real_hist[1, :], 'b-', linewidth=2, label='Real S')
        axs1[1, 0].plot(time_vec, x_est_hist[1, :], 'r--', linewidth=2, label='Estimated S')
        axs1[1, 0].set_ylabel('Substrate (g/L)', fontsize=10)
        axs1[1, 0].legend(fontsize=9)
        axs1[1, 0].grid(True, alpha=0.3)
        
        # Product
        axs1[2, 0].plot(time_vec, x_real_hist[2, :], 'b-', linewidth=2, label='Real P')
        axs1[2, 0].plot(time_vec, x_est_hist[2, :], 'r--', linewidth=2, label='Estimated P')
        axs1[2, 0].set_xlabel('Time (h)', fontsize=10)
        axs1[2, 0].set_ylabel('Product (g/L)', fontsize=10)
        axs1[2, 0].legend(fontsize=9)
        axs1[2, 0].grid(True, alpha=0.3)
        
        # DO measurement
        axs1[0, 1].plot(time_vec, z_meas_hist[0, :], 'k.', markersize=2, label='DO measured')
        axs1[0, 1].set_ylabel('DO (mg/L)', fontsize=10)
        axs1[0, 1].set_title('Dissolved Oxygen', fontsize=10)
        axs1[0, 1].legend(fontsize=9)
        axs1[0, 1].grid(True, alpha=0.3)
        
        # pH measurement
        axs1[1, 1].plot(time_vec, z_meas_hist[1, :], 'k.', markersize=2, label='pH measured')
        axs1[1, 1].set_ylabel('pH', fontsize=10)
        axs1[1, 1].set_title('pH', fontsize=10)
        axs1[1, 1].legend(fontsize=9)
        axs1[1, 1].grid(True, alpha=0.3)
        
        # Temperature measurement
        axs1[2, 1].plot(time_vec, z_meas_hist[2, :], 'k.', markersize=2, label='T measured')
        axs1[2, 1].set_xlabel('Time (h)', fontsize=10)
        axs1[2, 1].set_ylabel('Temperature (K)', fontsize=10)
        axs1[2, 1].set_title('Temperature', fontsize=10)
        axs1[2, 1].legend(fontsize=9)
        axs1[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Figure 2: Parameter Estimation
        fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6))
        fig2.suptitle('EKF Parameter Estimation', fontsize=14, fontweight='bold')
        
        # mu_max
        axs2[0].plot(time_vec, x_real_hist[3, :], 'b-', linewidth=2, label=r'Real $\mu_{max}$')
        axs2[0].plot(time_vec, x_est_hist[3, :], 'r--', linewidth=2, label=r'Estimated $\mu_{max}$')
        axs2[0].set_ylabel(r'$\mu_{max}$ (1/h)', fontsize=10)
        axs2[0].legend(fontsize=9)
        axs2[0].grid(True, alpha=0.3)
        
        # Y_XS
        axs2[1].plot(time_vec, x_real_hist[4, :], 'b-', linewidth=2, label=r'Real $Y_{X/S}$')
        axs2[1].plot(time_vec, x_est_hist[4, :], 'r--', linewidth=2, label=r'Estimated $Y_{X/S}$')
        axs2[1].set_xlabel('Time (h)', fontsize=10)
        axs2[1].set_ylabel(r'$Y_{X/S}$ (g/g)', fontsize=10)
        axs2[1].legend(fontsize=9)
        axs2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Figure 3: NMPC Control Performance
        fig3, axs3 = plt.subplots(2, 2, figsize=(14, 8))
        fig3.suptitle('NMPC Control Performance', fontsize=14, fontweight='bold')
        
        # Biomass tracking
        axs3[0, 0].plot(time_vec, x_real_hist[0, :], 'b-', linewidth=2, label='Biomass X')
        axs3[0, 0].plot(time_vec, sp_hist[0, :], 'r--', linewidth=2, label='X Setpoint')
        axs3[0, 0].set_ylabel('Biomass (g/L)', fontsize=10)
        axs3[0, 0].legend(fontsize=9)
        axs3[0, 0].grid(True, alpha=0.3)
        
        # Temperature tracking
        axs3[1, 0].plot(time_vec, z_meas_hist[2, :], 'b-', linewidth=2, label='Temperature')
        axs3[1, 0].plot(time_vec, sp_hist[1, :], 'r--', linewidth=2, label='T Setpoint')
        axs3[1, 0].set_xlabel('Time (h)', fontsize=10)
        axs3[1, 0].set_ylabel('Temperature (K)', fontsize=10)
        axs3[1, 0].legend(fontsize=9)
        axs3[1, 0].grid(True, alpha=0.3)
        
        # Substrate flow
        time_u = time_vec[:-1]
        axs3[0, 1].step(time_u, u_hist[0, :], 'g-', linewidth=2, where='post', label='F_S')
        axs3[0, 1].axhline(min_FS, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axs3[0, 1].axhline(max_FS, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axs3[0, 1].set_ylabel('F_S (L/h)', fontsize=10)
        axs3[0, 1].legend(fontsize=9)
        axs3[0, 1].grid(True, alpha=0.3)
        
        # Thermal load
        axs3[1, 1].step(time_u, u_hist[1, :], 'm-', linewidth=2, where='post', label='Q_j')
        axs3[1, 1].axhline(min_Qj, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axs3[1, 1].axhline(max_Qj, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axs3[1, 1].set_xlabel('Time (h)', fontsize=10)
        axs3[1, 1].set_ylabel('Q_j (W)', fontsize=10)
        axs3[1, 1].legend(fontsize=9)
        axs3[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Display final metrics
        st.subheader("Final Values")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Real States**")
            st.metric("X", f"{x_real_hist[0, -1]:.3f} g/L")
            st.metric("S", f"{x_real_hist[1, -1]:.3f} g/L")
            st.metric("P", f"{x_real_hist[2, -1]:.3f} g/L")
            st.metric("μ_max", f"{x_real_hist[3, -1]:.3f} 1/h")
            st.metric("Y_XS", f"{x_real_hist[4, -1]:.3f} g/g")
        
        with col2:
            st.write("**EKF Estimates**")
            st.metric("X est.", f"{x_est_hist[0, -1]:.3f} g/L")
            st.metric("S est.", f"{x_est_hist[1, -1]:.3f} g/L")
            st.metric("P est.", f"{x_est_hist[2, -1]:.3f} g/L")
            st.metric("μ_max est.", f"{x_est_hist[3, -1]:.3f} 1/h")
            st.metric("Y_XS est.", f"{x_est_hist[4, -1]:.3f} g/g")
        
        with col3:
            st.write("**Control & Setpoints**")
            st.metric("F_S", f"{u_hist[0, -1]:.3f} L/h")
            st.metric("Q_j", f"{u_hist[1, -1]:.1f} W")
            st.metric("X setpoint", f"{sp_hist[0, -1]:.3f} g/L")
            st.metric("T setpoint", f"{sp_hist[1, -1]:.2f} K")
            
            # Tracking error
            X_error = abs(x_real_hist[0, -1] - sp_hist[0, -1])
            T_error = abs(z_meas_hist[2, -1] - sp_hist[1, -1])
            st.metric("X tracking error", f"{X_error:.3f} g/L")
            st.metric("T tracking error", f"{T_error:.2f} K")
    
    else:
        st.info("Configure the parameters in the sidebar and click 'Run EKF-NMPC Simulation' to start.")

if __name__ == '__main__':
    ekf_nmpc_page()
