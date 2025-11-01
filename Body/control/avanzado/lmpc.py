import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal
from scipy.linalg import solve_discrete_are
import control as ct
import time

# Define la página/función principal de Streamlit
def lmpc_page():
    st.header("Linear Model Predictive Control (LMPC) of the Bioreactor")
    st.markdown("""
    This application simulates the LMPC control of a bioreactor using linearized transfer functions.
    The system is linearized around an operating point to obtain a 2x2 MIMO transfer function model.
    
    **System Structure (2x2):**
    - **Inputs (u):** `[F_S, Q_j]` (Substrate feed flow [L/h] and jacket heat rate [W])
    - **Outputs (y):** `[X, T]` (Biomass concentration [g/L] and Temperature [K])
    
    **Transfer Function Model:**
    The system is linearized around a steady-state operating point to obtain:
    ```
    Y(s) = G(s) * U(s)
    where G(s) is a 2x2 transfer function matrix
    ```
    """)

    # --- Sidebar para configuración ---
    with st.sidebar:
        st.subheader("Operating Point for Linearization")
        X_op = st.number_input("Biomass (X_op) [g/L]", value=2.0, key="X_op")
        S_op = st.number_input("Substrate (S_op) [g/L]", value=1.0, key="S_op")
        T_op = st.number_input("Temperature (T_op) [K]", value=305.0, key="T_op")
        FS_op = st.number_input("Substrate Flow (F_S,op) [L/h]", value=0.2, key="FS_op")
        Qj_op = st.number_input("Thermal Load (Q_j,op) [W]", value=500.0, key="Qj_op")
        
        st.subheader("Model Parameters")
        mu_max_input = st.number_input("Maximum Growth Rate (μ_max) [1/h]", value=0.4, key="mu_max")
        K_S_input = st.number_input("Monod Constant (K_S) [g/L]", value=0.05, key="K_S")
        Y_XS_input = st.number_input("Biomass/Substrate Yield (Y_XS) [g/g]", value=0.5, key="Y_XS")
        Y_QX_input = st.number_input("Heat/Biomass Yield (Y_QX) [J/g]", value=15000.0, key="Y_QX", format="%e")
        S_in_input = st.number_input("Inlet Substrate Concentration (S_in) [g/L]", value=10.0, key="S_in")
        V_input = st.number_input("Reactor Volume (V) [L]", value=1.0, key="V")
        rho_input = st.number_input("Average Density (ρ) [g/L]", value=1000.0, key="rho", format="%e")
        Cp_input = st.number_input("Heat Capacity (Cp) [J/(g*K)]", value=4184.0, key="Cp", format="%e")
        T_in_input = st.number_input("Inlet Temperature (T_in) [K]", value=298.15, key="T_in")
        F_const_input = st.number_input("Additional Constant Flow (F_const) [L/h]", value=0.0, key="F_const")

        st.subheader("LMPC Configuration")
        N_input = st.number_input("Prediction Horizon (N)", min_value=1, value=20, key="N")
        M_input = st.number_input("Control Horizon (M)", min_value=1, value=5, key="M")
        dt_lmpc_input = st.number_input("LMPC Sampling Time (dt) [h]", min_value=0.01, value=0.1, step=0.01, format="%.2f", key="dt_lmpc")
        simulation_time = st.number_input("Total Simulation Time [h]", min_value=1.0, value=24.0, key="sim_time")

        st.subheader("Input Limits (u)")
        min_FS = st.number_input("Minimum Substrate Flow (F_S) [L/h]", value=0.0, key="min_FS")
        max_FS = st.number_input("Maximum Substrate Flow (F_S) [L/h]", value=1.5, key="max_FS")
        min_Qj = st.number_input("Minimum Thermal Load (Q_j) [W]", value=-10000.0, key="min_Qj", format="%e")
        max_Qj = st.number_input("Maximum Thermal Load (Q_j) [W]", value=10000.0, key="max_Qj", format="%e")

        st.subheader("Inputs Change Rate Limits (Δu)")
        delta_FS_max = st.number_input("Maximum |ΔF_S| change per step [L/h]", value=0.1, min_value=0.0, key="dFS_max")
        delta_Qj_max = st.number_input("Maximum |ΔQ_j| change per step [W]", value=5000.0, min_value=0.0, key="dQj_max", format="%e")

        st.subheader("LMPC Weights")
        Q_X_weight = st.number_input("Weight Q - Biomass Error (X)", value=10.0, key="Q_X")
        Q_T_weight = st.number_input("Weight Q - Temperature Error (T)", value=1.0, key="Q_T")
        R_FS_weight = st.number_input("Weight R - Substrate Flow Change (ΔF_S)", value=0.1, key="R_FS")
        R_Qj_weight = st.number_input("Weight R - Heat Load Change (ΔQ_j) [1/W^2]", value=1e-8, format="%e", key="R_Qj")

        st.subheader("Initial Conditions and Setpoints")
        initial_X = st.number_input("Initial Biomass (X0) [g/L]", value=1.5, key="X0")
        initial_S = st.number_input("Initial Substrate (S0) [g/L]", value=9.0, key="S0")
        initial_T = st.number_input("Initial Temperature (T0) [K]", value=305.0, key="T0")
        initial_FS = st.number_input("Initial Substrate Flow (F_S,0) [L/h]", value=0.1, key="Fs0")
        initial_Qj = st.number_input("Initial Thermal Load (Q_j,0) [W]", value=0.0, key="Qj0")

        # Setpoints
        t_sp1 = 5.0
        t_sp2 = 12.0
        st.markdown(f"**Setpoints at t={t_sp1}h:**")
        setpoint_X_t1 = st.number_input(f"Biomass Setpoint (X) at t={t_sp1}h", value=2.0, key="sp_X1")
        setpoint_T_t1 = st.number_input(f"Temperature Setpoint (T) at t={t_sp1}h [K]", value=308.0, key="sp_T1")
        st.markdown(f"**Setpoints at t={t_sp2}h:**")
        setpoint_X_t2 = st.number_input(f"Biomass Setpoint (X) at t={t_sp2}h", value=1.0, key="sp_X2")
        setpoint_T_t2 = st.number_input(f"Temperature Setpoint (T) at t={t_sp2}h [K]", value=303.0, key="sp_T2")

        compute_tf = st.button("Compute Transfer Functions")
        start_simulation = st.button("Start LMPC Simulation")

    # --- Helper Functions ---
    def get_bioreactor_model(params_input):
        """Define the symbolic bioreactor model using CasADi."""
        params = params_input

        # Symbolic variables
        X = ca.MX.sym('X')  # Biomass concentration (g/L)
        S = ca.MX.sym('S')  # Substrate concentration (g/L)
        T = ca.MX.sym('T')  # Reactor temperature (K)
        x = ca.vertcat(X, S, T)

        # Symbolic inputs
        u = ca.MX.sym('u', 2)
        F_S = u[0]  # Substrate flow (L/h)
        Q_j = u[1]  # Jacket heat rate (W)

        # Model equations
        mu = params['mu_max'] * S / (params['K_S'] + S + 1e-9)
        F_total = F_S + params['F_const']
        D = F_total / params['V']

        # Material balances
        dX_dt = mu * X - D * X
        dS_dt = D * (params['S_in'] - S) - (mu / params['Y_XS']) * X

        # Energy balance
        Q_rem = -Q_j * 3600.0  # Heat removed by jacket (J/h)
        Q_gen = params['Y_QX'] * mu * X * params['V']  # Heat generated (J/h)
        Q_flow = F_total * params['rho'] * params['Cp'] * (params['T_in'] - T)  # Heat by flow (J/h)
        dT_dt = (Q_gen + Q_rem + Q_flow) / (params['rho'] * params['Cp'] * params['V'] + 1e-9)

        dx = ca.vertcat(dX_dt, dS_dt, dT_dt)

        # Outputs
        c = ca.vertcat(X, T)

        # Create CasADi functions
        model_ode = ca.Function('model_ode', [x, u], [dx], ['x', 'u'], ['dx'])
        output_func = ca.Function('output_func', [x], [c], ['x'], ['c'])

        return model_ode, output_func, x, u, c, dx, params

    def linearize_bioreactor(model_ode, x_op, u_op):
        """
        Linearize the bioreactor model around an operating point.
        Returns the state-space matrices A, B, C, D for deviation variables.
        """
        nx = 3
        nu = 2
        ny = 2

        # Create symbolic variables for Jacobian computation
        x_sym = ca.MX.sym('x', nx)
        u_sym = ca.MX.sym('u', nu)
        
        # Get the dynamics
        dx = model_ode(x_sym, u_sym)
        
        # Compute Jacobians
        A_sym = ca.jacobian(dx, x_sym)
        B_sym = ca.jacobian(dx, u_sym)
        
        # Create functions to evaluate Jacobians
        A_func = ca.Function('A_func', [x_sym, u_sym], [A_sym])
        B_func = ca.Function('B_func', [x_sym, u_sym], [B_sym])
        
        # Evaluate at operating point
        A = A_func(x_op, u_op).full()
        B = B_func(x_op, u_op).full()
        
        # Output matrix C (we measure X and T, which are states 0 and 2)
        C = np.array([[1, 0, 0],
                      [0, 0, 1]])
        
        # Feedthrough matrix D (no direct feedthrough)
        D = np.zeros((ny, nu))
        
        return A, B, C, D

    def ss_to_tf(A, B, C, D, dt):
        """
        Convert continuous state-space to discrete transfer function matrix.
        Returns a 2x2 matrix of transfer functions.
        """
        # Discretize the state-space model
        sys_c = ct.ss(A, B, C, D)
        sys_d = ct.c2d(sys_c, dt, method='zoh')
        
        # Get discrete matrices
        Ad, Bd, Cd, Dd = sys_d.A, sys_d.B, sys_d.C, sys_d.D
        
        # Create a dictionary to store transfer functions
        tf_matrix = {}
        
        for i in range(2):  # outputs
            for j in range(2):  # inputs
                # Extract SISO system from MIMO
                # Create a state-space with single input and output
                C_siso = Cd[i:i+1, :]
                B_siso = Bd[:, j:j+1]
                D_siso = Dd[i:i+1, j:j+1]
                
                sys_siso = ct.ss(Ad, B_siso, C_siso, D_siso, dt)
                tf_siso = ct.tf(sys_siso)
                
                tf_matrix[(i, j)] = tf_siso
        
        return tf_matrix, Ad, Bd, Cd, Dd

    # --- LMPC Controller Class ---
    class LMPCController:
        def __init__(self, Ad, Bd, Cd, Dd, N, M, Q, R, dt, u_min, u_max, du_max):
            """
            Linear MPC controller using discrete state-space model.
            
            Args:
                Ad, Bd, Cd, Dd: Discrete state-space matrices
                N: Prediction horizon
                M: Control horizon
                Q: Output error weight matrix (ny x ny)
                R: Input change weight matrix (nu x nu)
                dt: Sampling time
                u_min, u_max: Input limits
                du_max: Input rate of change limits
            """
            self.Ad = Ad
            self.Bd = Bd
            self.Cd = Cd
            self.Dd = Dd
            self.N = N
            self.M = M
            self.Q = Q
            self.R = R
            self.dt = dt
            self.nx = Ad.shape[0]
            self.nu = Bd.shape[1]
            self.ny = Cd.shape[0]
            self.u_min = u_min
            self.u_max = u_max
            self.du_max = du_max
            
            # Build QP matrices
            self._build_qp_matrices()
        
        def _build_qp_matrices(self):
            """Build the QP formulation matrices for LMPC."""
            nx, nu, ny = self.nx, self.nu, self.ny
            N, M = self.N, self.M
            
            # Build prediction matrices
            # State prediction: X = Sx*x0 + Su*U
            Sx = np.zeros((nx * (N + 1), nx))
            Su = np.zeros((nx * (N + 1), nu * M))
            
            Sx[0:nx, :] = np.eye(nx)
            
            for k in range(N):
                # State matrix
                Sx[(k+1)*nx:(k+2)*nx, :] = self.Ad @ Sx[k*nx:(k+1)*nx, :]
                
                # Input matrix
                for j in range(min(k+1, M)):
                    if j == 0:
                        Su[(k+1)*nx:(k+2)*nx, j*nu:(j+1)*nu] = self.Bd
                    else:
                        Su[(k+1)*nx:(k+2)*nx, j*nu:(j+1)*nu] = \
                            self.Ad @ Su[k*nx:(k+1)*nx, (j-1)*nu:j*nu]
                
                # For steps beyond M, repeat last control
                for j in range(min(k+1, M), k+1):
                    Su[(k+1)*nx:(k+2)*nx, (M-1)*nu:M*nu] += \
                        np.linalg.matrix_power(self.Ad, k+1-j) @ self.Bd
            
            # Output prediction: Y = Cy*X = Cy*Sx*x0 + Cy*Su*U
            Cy_ext = np.zeros((ny * (N + 1), nx * (N + 1)))
            for k in range(N + 1):
                Cy_ext[k*ny:(k+1)*ny, k*nx:(k+1)*nx] = self.Cd
            
            self.F = Cy_ext @ Sx  # Output prediction from initial state
            self.Phi = Cy_ext @ Su  # Output prediction from control inputs
            
            # Build cost function matrices
            # Cost: J = (Y - R)^T Q_bar (Y - R) + dU^T R_bar dU
            Q_bar = np.kron(np.eye(N + 1), self.Q)
            R_bar = np.kron(np.eye(M), self.R)
            
            # QP cost: 0.5 * dU^T H dU + f^T dU
            self.H = 2 * (self.Phi.T @ Q_bar @ self.Phi + R_bar)
            self.Phi_Q = 2 * self.Phi.T @ Q_bar
            
        def solve(self, x_current, x_op, u_previous, u_op, setpoint_trajectory):
            """
            Solve the LMPC optimization problem.
            
            Args:
                x_current: Current state (absolute values)
                x_op: Operating point state
                u_previous: Previous input (absolute values)
                u_op: Operating point input
                setpoint_trajectory: Setpoint trajectory (ny x (N+1)) in absolute values
                
            Returns:
                u_optimal: Optimal control action (absolute values)
            """
            # Convert to deviation variables
            x_dev = x_current - x_op
            u_prev_dev = u_previous - u_op
            
            # Convert setpoint to output deviations
            y_sp_dev = np.zeros((self.ny * (self.N + 1)))
            for k in range(self.N + 1):
                y_sp = setpoint_trajectory[:, k]
                # Output at operating point
                y_op = self.Cd @ x_op
                y_dev = y_sp - y_op
                y_sp_dev[k*self.ny:(k+1)*self.ny] = y_dev
            
            # Compute linear term in QP
            f = self.Phi_Q @ (self.F @ x_dev - y_sp_dev)
            
            # Constraint matrices for du
            # du_min <= du <= du_max
            # u_min - u_prev <= cumsum(du) <= u_max - u_prev
            
            # Build constraint matrices
            # Inequality constraints: A_ineq @ dU <= b_ineq
            
            # 1. Input rate constraints
            A_rate = np.vstack([np.eye(self.nu * self.M), -np.eye(self.nu * self.M)])
            b_rate = np.hstack([np.tile(self.du_max, self.M), np.tile(self.du_max, self.M)])
            
            # 2. Input magnitude constraints (cumulative sum)
            # Build cumulative sum matrix
            T = np.zeros((self.nu * self.M, self.nu * self.M))
            for i in range(self.M):
                for j in range(i + 1):
                    T[i*self.nu:(i+1)*self.nu, j*self.nu:(j+1)*self.nu] = np.eye(self.nu)
            
            # u = u_prev + T @ dU
            # u_min <= u_prev + T @ dU <= u_max
            # T @ dU <= u_max - u_prev
            # -T @ dU <= -(u_min - u_prev) = u_prev - u_min
            
            u_prev_rep = np.tile(u_prev_dev, self.M)
            u_max_dev = (self.u_max - u_op) - u_prev_dev
            u_min_dev = (self.u_min - u_op) - u_prev_dev
            
            A_mag = np.vstack([T, -T])
            b_mag = np.hstack([np.tile(u_max_dev, self.M), np.tile(-u_min_dev, self.M)])
            
            # Combine constraints
            A_ineq = np.vstack([A_rate, A_mag])
            b_ineq = np.hstack([b_rate, b_mag])
            
            # Solve QP using quadprog-like formulation
            # We'll use a simple approach with scipy.optimize.minimize
            from scipy.optimize import minimize, Bounds, LinearConstraint
            
            # Initial guess
            dU0 = np.zeros(self.nu * self.M)
            
            # Objective function
            def objective(dU):
                return 0.5 * dU @ self.H @ dU + f @ dU
            
            def gradient(dU):
                return self.H @ dU + f
            
            # Bounds on dU
            bounds = Bounds(-np.tile(self.du_max, self.M), np.tile(self.du_max, self.M))
            
            # Linear constraints for magnitude
            linear_constraint = LinearConstraint(T, 
                                                 np.tile(u_min_dev, self.M), 
                                                 np.tile(u_max_dev, self.M))
            
            # Solve
            result = minimize(objective, dU0, method='SLSQP', jac=gradient,
                            bounds=bounds, constraints=[linear_constraint],
                            options={'maxiter': 200, 'ftol': 1e-6})
            
            if not result.success:
                st.warning(f"LMPC optimization did not converge: {result.message}")
            
            dU_opt = result.x
            
            # Extract first control move
            du_apply = dU_opt[0:self.nu]
            u_apply_dev = u_prev_dev + du_apply
            
            # Convert back to absolute values
            u_apply = u_apply_dev + u_op
            
            # Clip to bounds
            u_apply = np.clip(u_apply, self.u_min, self.u_max)
            
            return u_apply

    # --- Plant Simulation ---
    def simulate_plant_step(x_current, u_applied, dt_sim, model_ode_func):
        """Simulates a step of the real plant."""
        def ode_sys(t, x, u):
            return model_ode_func(x, u).full().flatten()
        
        t_span = [0, dt_sim]
        sol = solve_ivp(ode_sys, t_span, x_current, args=(u_applied,), 
                       method='RK45', rtol=1e-5, atol=1e-8)
        return sol.y[:, -1]

    # --- Compute Transfer Functions ---
    if compute_tf:
        st.info("Computing transfer functions from linearization...")
        
        # Gather parameters
        params_dict = {
            'mu_max': mu_max_input, 'K_S': K_S_input, 'Y_XS': Y_XS_input,
            'Y_QX': Y_QX_input, 'S_in': S_in_input, 'V': V_input,
            'rho': rho_input, 'Cp': Cp_input, 'T_in': T_in_input,
            'F_const': F_const_input
        }
        
        # Get model
        model_ode, output_func, x_sym, u_sym, c_sym, dx_sym, params = get_bioreactor_model(params_dict)
        
        # Operating point
        x_op_array = np.array([X_op, S_op, T_op])
        u_op_array = np.array([FS_op, Qj_op])
        
        # Linearize
        A, B, C, D = linearize_bioreactor(model_ode, x_op_array, u_op_array)
        
        st.subheader("Linearized State-Space Model")
        st.write("**State-space matrices (continuous-time):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**A matrix:**")
            st.write(A)
            st.write("**B matrix:**")
            st.write(B)
        with col2:
            st.write("**C matrix:**")
            st.write(C)
            st.write("**D matrix:**")
            st.write(D)
        
        # Convert to transfer functions
        tf_matrix, Ad, Bd, Cd, Dd = ss_to_tf(A, B, C, D, dt_lmpc_input)
        
        st.subheader("Transfer Function Matrix G(z)")
        st.write("Discrete-time transfer functions (Z-domain):")
        
        # Display transfer functions
        for i in range(2):
            for j in range(2):
                output_name = ['X', 'T'][i]
                input_name = ['F_S', 'Q_j'][j]
                st.write(f"**G_{{{output_name},{input_name}}}(z)**: {output_name} response to {input_name}")
                st.code(str(tf_matrix[(i, j)]))
        
        st.success("Transfer functions computed successfully!")

    # --- Main Simulation Loop ---
    if start_simulation:
        st.info("Starting LMPC Simulation...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Gather parameters
        params_dict = {
            'mu_max': mu_max_input, 'K_S': K_S_input, 'Y_XS': Y_XS_input,
            'Y_QX': Y_QX_input, 'S_in': S_in_input, 'V': V_input,
            'rho': rho_input, 'Cp': Cp_input, 'T_in': T_in_input,
            'F_const': F_const_input
        }

        # Get model
        model_ode, output_func, x_sym, u_sym, c_sym, dx_sym, params = get_bioreactor_model(params_dict)

        # Operating point
        x_op = np.array([X_op, S_op, T_op])
        u_op = np.array([FS_op, Qj_op])

        # Linearize
        A, B, C, D = linearize_bioreactor(model_ode, x_op, u_op)
        
        # Discretize
        tf_matrix, Ad, Bd, Cd, Dd = ss_to_tf(A, B, C, D, dt_lmpc_input)

        # LMPC weights
        Q = np.diag([Q_X_weight, Q_T_weight])
        R = np.diag([R_FS_weight, R_Qj_weight])

        # Input constraints
        u_min = np.array([min_FS, min_Qj])
        u_max = np.array([max_FS, max_Qj])
        du_max = np.array([delta_FS_max, delta_Qj_max])

        # Create LMPC controller
        try:
            lmpc = LMPCController(Ad, Bd, Cd, Dd, N_input, M_input, Q, R, 
                                dt_lmpc_input, u_min, u_max, du_max)
        except Exception as e:
            st.error(f"Error initializing LMPC: {e}")
            st.stop()

        # Simulation setup
        t_final = simulation_time
        dt_sim = dt_lmpc_input
        n_steps = int(t_final / dt_sim)

        # Initial conditions
        x_current = np.array([initial_X, initial_S, initial_T])
        u_previous = np.array([initial_FS, initial_Qj])

        # Setpoints
        initial_sp = np.array([initial_X, initial_T])
        current_setpoint = initial_sp.copy()

        setpoint_changes = {
            t_sp1: np.array([setpoint_X_t1, setpoint_T_t1]),
            t_sp2: np.array([setpoint_X_t2, setpoint_T_t2]),
        }

        # History
        t_history = np.linspace(0, t_final, n_steps + 1)
        x_history = np.zeros((3, n_steps + 1))
        u_history = np.zeros((2, n_steps))
        c_history = np.zeros((2, n_steps + 1))
        sp_history = np.zeros((2, n_steps + 1))

        # Save initial state
        x_history[:, 0] = x_current
        c_history[:, 0] = output_func(x_current).full().flatten()
        sp_history[:, 0] = current_setpoint

        start_time_sim = time.time()

        # Simulation loop
        for k in range(n_steps):
            t_current = k * dt_sim
            status_text.text(f"Simulating step {k+1}/{n_steps} (t={t_current:.2f}h)...")

            # Update setpoint
            next_step_time = (k + 1) * dt_sim
            for sp_time, new_sp in sorted(setpoint_changes.items()):
                if t_current < sp_time <= next_step_time + 1e-9:
                    st.write(f"-> Setpoint change at t ≈ {sp_time:.2f}h to: X={new_sp[0]}, T={new_sp[1]}K")
                    current_setpoint = new_sp
                    break

            # Create setpoint trajectory
            sp_traj = np.tile(current_setpoint, (N_input + 1, 1)).T

            # Solve LMPC
            u_optimal = lmpc.solve(x_current, x_op, u_previous, u_op, sp_traj)

            # Simulate plant
            try:
                x_next = simulate_plant_step(x_current, u_optimal, dt_sim, model_ode)
            except Exception as e:
                st.error(f"Error simulating plant at step {k+1}: {e}")
                st.warning("Stopping simulation.")
                break

            # Update
            x_current = x_next
            u_previous = u_optimal

            # Save history
            x_history[:, k+1] = x_current
            u_history[:, k] = u_optimal
            c_history[:, k+1] = output_func(x_current).full().flatten()
            sp_history[:, k+1] = current_setpoint

            # Update progress
            progress_bar.progress((k + 1) / n_steps)

        end_time_sim = time.time()
        total_sim_time = end_time_sim - start_time_sim
        status_text.text(f"Simulation completed in {total_sim_time:.2f} seconds.")
        st.success("LMPC simulation completed.")

        # Plot results
        st.subheader("LMPC Simulation Results")

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        fig.suptitle("LMPC Bioreactor Simulation Results", fontsize=16)

        # States and Setpoints
        axes[0, 0].plot(t_history, x_history[0, :], label='Biomass (X)')
        axes[0, 0].plot(t_history, sp_history[0, :], 'r--', label='X Setpoint')
        axes[0, 0].set_ylabel('Biomass (g/L)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[1, 0].plot(t_history, x_history[1, :], label='Substrate (S)')
        axes[1, 0].set_ylabel('Substrate (g/L)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[2, 0].plot(t_history, x_history[2, :], label='Temperature (T)')
        axes[2, 0].plot(t_history, sp_history[1, :], 'r--', label='T Setpoint')
        axes[2, 0].set_ylabel('Temperature (K)')
        axes[2, 0].set_xlabel('Time (h)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # Inputs
        axes[0, 1].step(t_history, np.append(u_history[0, :], u_history[0, -1]), 
                       where='post', label='Substrate Flow (F_S)')
        axes[0, 1].hlines([min_FS, max_FS], t_history[0], t_history[-1], 
                         colors='gray', linestyles='--', label='F_S Limits')
        axes[0, 1].set_ylabel('F_S (L/h)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 1].step(t_history, np.append(u_history[1, :], u_history[1, -1]), 
                       where='post', label='Thermal Load (Q_j)')
        axes[1, 1].hlines([min_Qj, max_Qj], t_history[0], t_history[-1], 
                         colors='gray', linestyles='--', label='Q_j Limits')
        axes[1, 1].set_ylabel('Q_j (W)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[2, 1].axis('off')
        axes[2, 1].set_xlabel('Time (h)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

# --- Run the application ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    lmpc_page()
