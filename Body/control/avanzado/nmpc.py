import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time # Para medir tiempo si es necesario

# Define la página/función principal de Streamlit
def nmpc_page():
    st.header("Nonlinear Model Predictive Control (NMPC) of the Bioreactor")
    st.markdown("""
    This application simulates the NMPC control of a simple bioreactor.
    Set the model parameters, NMPC settings, limits and initisl conditions
    in the sidebar, and then click on "Start NMPC Simulation".

    **Simplified Model:**
    It considers a bioreactor with material balances for biomass (X) and substrate (S),
    and an energy balance for temperature (T).
    - **States (x):** `[X, S, T]`
    - **Manipulated Inputs (u):** `[F_S, Q_j]` (Substrate feed flow and jacket heat rate)
    - **Controlled Outputs (c):** `[X, T]`
    """)

    # --- Sidebar para configuración ---
    with st.sidebar:
        st.subheader("Model Configuration")
        # Usamos claves únicas para los number_input si se repiten nombres
        mu_max_input = st.number_input("Maximum Growth Rate (μ_max) [1/h]", value=0.4, key="mu_max")
        K_S_input = st.number_input("Monod Constant (K_S) [g/L]", value=0.05, key="K_S")
        Y_XS_input = st.number_input("Biomass/Substrate Yield (Y_XS) [g/g]", value=0.5, key="Y_XS")
        Y_QX_input = st.number_input("Heat/Biomass Yield (Y_QX) [J/g]", value=15000.0, key="Y_QX", format="%e")
        S_in_input = st.number_input("Inlet Substrate Concentration (S_in) [g/L]", value=10.0, key="S_in")
        V_input = st.number_input("Reactor Volume (V) [L]", value=1.0, key="V")
        rho_input = st.number_input("Average Density (ρ) [g/L]", value=1000.0, key="rho", format="%e")
        Cp_input = st.number_input("Heat Capacity (Cp) [J/(g*K)]", value=4184.0, key="Cp", format="%e")
        T_in_input = st.number_input("Inlet Temperature (T_in) [K]", value=298.15, key="T_in")
        F_const_input = st.number_input("Aditional Constant Flow (F_const) [L/h]", value=0.0, key="F_const")

        st.subheader("NMPC Configuration")
        N_input = st.number_input("Prediction Horizon (N)", min_value=1, value=10, key="N")
        M_input = st.number_input("Control Horizon (M)", min_value=1, value=4, key="M")
        dt_nmpc_input = st.number_input("NMPC Sampling Time (dt) [h]", min_value=0.01, value=0.1, step=0.01, format="%.2f", key="dt_nmpc")
        simulation_time = st.number_input("Total Simulation Time [h]", min_value=1.0, value=24.0, key="sim_time")

        st.subheader("Input Limits (u)")
        min_FS = st.number_input("Minimum Substrate Flow (F_S) [L/h]", value=0.0, key="min_FS")
        max_FS = st.number_input("Maximum Substrate Flow (F_S) [L/h]", value=1.5, key="max_FS")
        min_Qj = st.number_input("Minimum Thermal Load (Q_j) [W]", value=-10000.0, key="min_Qj", format="%e") # Watts
        max_Qj = st.number_input("Maximum Thermal Load (Q_j) [W]", value=10000.0, key="max_Qj", format="%e") # Watts

        st.subheader("Inputs Change Rate Limits (Δu)")
        # Por defecto, permitir cambios grandes, ajustar según necesidad
        delta_FS_max = st.number_input("Maximum |ΔF_S| change per step [L/h]", value=0.1, min_value=0.0, key="dFS_max")
        delta_Qj_max = st.number_input("Maximum |ΔQ_j| change per step [W]", value=5000.0, min_value=0.0, key="dQj_max", format="%e")

        st.subheader("State Limits (x) - For optimization")
        # Estos son límites suaves o de referencia para el optimizador
        min_X_opt = st.number_input("Min X (Optimizer)", value=0.0, key="min_X_opt")
        max_X_opt = st.number_input("Max X (Optimizer)", value=5.0, key="max_X_opt")
        min_S_opt = st.number_input("Min S (Optimizer)", value=0.0, key="min_S_opt")
        max_S_opt = st.number_input("Max S (Optimizer)", value=10.0, key="max_S_opt")
        min_T_opt = st.number_input("Min T (Optimizer) [K]", value=290.0, key="min_T_opt")
        max_T_opt = st.number_input("Max T (Optimizer) [K]", value=315.0, key="max_T_opt")


        st.subheader("NMPC Weights")
        Q_X_weight = st.number_input("Weight Q - Biomass Error (X)", value=1.0, key="Q_X")
        Q_T_weight = st.number_input("Weight Q - Temperature Error (T)", value=0.01, key="Q_T")
        W_FS_weight = st.number_input("Weight W - Substrate Flow Change (ΔF_S)", value=0.1, key="W_FS")
        # Ajustar el peso de Qj según la escala (Watts vs L/h)
        W_Qj_weight = st.number_input("Weight W - Heat Load Change (ΔQ_j) [1/W^2]", value=1e-8, format="%e", key="W_Qj")

        st.subheader("Initial Conditions and Setpoints")
        initial_X = st.number_input("Initial Biomass (X0) [g/L]", value=1.5, key="X0")
        initial_S = st.number_input("Initial Substrate (S0) [g/L]", value=9.0, key="S0")
        initial_T = st.number_input("Initial Temperature (T0) [K]", value=305.0, key="T0")
        initial_FS = st.number_input("Initial Substrate Flow (F_S,0) [L/h]", value=0.1, key="Fs0")
        initial_Qj = st.number_input("Initial Thermal Load (Q_j,0) [W]", value=0.0, key="Qj0")

        # Setpoints Escalonados (Ejemplo fijo en el tiempo, valores configurables)
        t_sp1 = 5.0
        t_sp2 = 12.0
        st.markdown(f"**Setpoints in t={t_sp1}h:**")
        setpoint_X_t1 = st.number_input(f"Biomass Setpoint (X) in t={t_sp1}h", value=2.0, key="sp_X1")
        setpoint_T_t1 = st.number_input(f"Temperature Setpoint (T) in t={t_sp1}h [K]", value=308.0, key="sp_T1")
        st.markdown(f"**Setpoints in t={t_sp2}h:**")
        setpoint_X_t2 = st.number_input(f"Biomass Setpoint (X) in t={t_sp2}h", value=1.0, key="sp_X2")
        setpoint_T_t2 = st.number_input(f"Temperature Setpoint (T) in t={t_sp2}h [K]", value=303.0, key="sp_T2")

        start_simulation = st.button("Start NMPC Simulation")

    # ---------------------------------------------------
    # 1. Modelo del Biorreactor (Simbólico con CasADi)
    # ---------------------------------------------------
    # @st.cache_data # Podría cachearse si los params no cambian mucho
    def get_bioreactor_model(params_input):
        """Define el modelo simbólico del biorreactor usando CasADi."""
        params = params_input # Usar los parámetros de la entrada

        # Variables simbólicas
        X = ca.MX.sym('X')  # Concentración de biomasa (g/L)
        S = ca.MX.sym('S')  # Concentración de sustrato (g/L)
        T = ca.MX.sym('T')  # Temperatura del reactor (K)
        x = ca.vertcat(X, S, T)

        # Entradas simbólicas
        # u[0]: F_S (L/h)
        # u[1]: Q_j (W = J/s) - ¡La entrada del NMPC será Qj directamente en Watts!
        u = ca.MX.sym('u', 2)
        F_S = u[0]
        Q_j = u[1] # Directamente en Watts

        # --- Ecuaciones del modelo ---
        mu = params['mu_max'] * S / (params['K_S'] + S + 1e-9) # Tasa crecimiento Monod (+epsilon)

        F_total = F_S + params['F_const']
        D = F_total / params['V'] # Tasa de dilución (1/h)

        # Balances de materia
        dX_dt = mu * X - D * X
        dS_dt = D * (params['S_in'] - S) - (mu / params['Y_XS']) * X

        # Balance de energía
        # Convertir Q_j de Watts (J/s) a J/h para consistencia de unidades
        Q_rem = - Q_j * 3600.0 # Calor removido por la chaqueta (J/h)
        Q_gen = params['Y_QX'] * mu * X * params['V'] # Calor generado (J/h)
        Q_flow = F_total * params['rho'] * params['Cp'] * (params['T_in'] - T) # Calor por flujo (J/h)

        # dT/dt en K/h
        dT_dt = (Q_gen + Q_rem + Q_flow) / (params['rho'] * params['Cp'] * params['V'] + 1e-9) # (K/h)

        # Vector de derivadas
        dx = ca.vertcat(dX_dt, dS_dt, dT_dt)

        # Variables controladas (salidas) y = h(x)
        c = ca.vertcat(X, T)

        # Crear funciones CasADi
        model_ode = ca.Function('model_ode', [x, u], [dx], ['x', 'u'], ['dx'])
        output_func = ca.Function('output_func', [x], [c], ['x'], ['c'])

        return model_ode, output_func, x, u, c, dx, params

    # ---------------------------------------------------
    # 2. Clase NMPC
    # ---------------------------------------------------
    class NMPCBioreactor:
        def __init__(self, dt, N, M, Q, W, model_ode, output_func, x_sym, u_sym, c_sym, params,
                     lbx, ubx, lbu, ubu, lbdu, ubdu, m=3, pol='legendre'):
            """
            Inicializa el controlador NMPC.
             Args:
                 Q: Vector/Lista de pesos para error de salida (CV) -> Se convierte a diag
                 W: Vector/Lista de pesos para movimiento de entrada (MV) -> Se convierte a diag
                 lbu, ubu: Límites inferiores/superiores para entradas u [F_S (L/h), Q_j (W)]
                 lbdu, ubdu: Límites inferiores/superiores para tasa de cambio de entradas du [ΔF_S, ΔQ_j]
            """
            self.dt = dt
            self.N = N
            self.M = M
            self.Q = np.diag(Q) # Matriz diagonal de pesos de salida
            self.W = np.diag(W) # Matriz diagonal de pesos de entrada (tasa de cambio)
            self.model_ode = model_ode
            self.output_func = output_func
            self.params = params
            self.nx = x_sym.shape[0]
            self.nu = u_sym.shape[0]
            self.nc = c_sym.shape[0]
            self.lbx = lbx # Límites para estados (en el NLP)
            self.ubx = ubx
            self.lbu = lbu # Límites para entradas u
            self.ubu = ubu
            self.lbdu = lbdu # Límites para tasa de cambio delta_u
            self.ubdu = ubdu
            self.m = m
            self.pol = pol

            # --- Preparar Colocación Ortogonal ---
            self.tau_root = np.append(0, ca.collocation_points(self.m, self.pol))
            self.C = np.zeros((self.m + 1, self.m + 1))
            self.D = np.zeros(self.m + 1)
            for j in range(self.m + 1):
                p = np.poly1d([1])
                for r in range(self.m + 1):
                    if r != j:
                        p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j] - self.tau_root[r] + 1e-10)
                p_der = np.polyder(p)
                for i in range(self.m + 1):
                    self.C[j, i] = np.polyval(p_der, self.tau_root[i])
                p_int = np.polyint(p)
                self.D[j] = np.polyval(p_int, 1.0)

            # --- Construir el NLP ---
            self._build_nlp()
            # Inicializar w0 (vector de ceros con la dimensión correcta)
            self.w0 = np.zeros(self.dim_w) # Usar dim_w calculado en _prepare_indices

        def _build_nlp(self):
            """Construye el problema NLP de optimización."""
            # Función de colocación para un paso
            Xk_step = ca.MX.sym('Xk_step', self.nx)
            Xc_step = ca.MX.sym('Xc_step', self.nx, self.m)
            Uk_step = ca.MX.sym('Uk_step', self.nu)
            X_all_coll_step = ca.horzcat(Xk_step, Xc_step)
            ode_at_coll_step = []
            for j in range(1, self.m + 1):
                 ode_at_coll_step.append(self.model_ode(X_all_coll_step[:, j], Uk_step))
            coll_eqs_step = []
            for j in range(1, self.m + 1):
                xp_coll_j = 0
                for r in range(self.m + 1):
                    xp_coll_j += self.C[r, j] * X_all_coll_step[:, r]
                coll_eqs_step.append(xp_coll_j - (self.dt * ode_at_coll_step[j-1]))
            Xk_end_step = Xk_step
            for j in range(1, self.m + 1):
                Xk_end_step += self.dt * self.D[j] * ode_at_coll_step[j-1]
            self.F_coll = ca.Function('F_coll', [Xk_step, Xc_step, Uk_step],
                                       [Xk_end_step, ca.vertcat(*coll_eqs_step)],
                                       ['Xk_step', 'Xc_step', 'Uk_step'], ['Xk_end', 'coll_eqs'])

            # --- Variables de decisión del NMPC ---
            self.w = []
            self.lbw = []
            self.ubw = []
            self.g = []
            self.lbg = []
            self.ubg = []

            # Parámetros del NLP
            self.x0_sym = ca.MX.sym('x0', self.nx)
            self.sp_sym = ca.MX.sym('sp', self.nc, self.N) # Trayectoria de SP para N pasos
            self.uprev_sym = ca.MX.sym('uprev', self.nu)
            p_nlp = ca.vertcat(self.x0_sym, ca.vec(self.sp_sym), self.uprev_sym)

            J = 0 # Función de costo
            Uk_prev = self.uprev_sym
            Xk_iter = self.x0_sym # Estado inicial es parámetro

            # Bucle sobre el horizonte de predicción N
            for k in range(self.N):
                # Variable de entrada Uk
                Uk_k = ca.MX.sym(f'U_{k}', self.nu)
                self.w.append(Uk_k)
                self.lbw.extend(self.lbu)
                self.ubw.extend(self.ubu)

                # Restricciones delta_u
                delta_u = Uk_k - Uk_prev
                self.g.append(delta_u)
                self.lbg.extend(self.lbdu) # Límites inferiores para delta_u
                self.ubg.extend(self.ubdu) # Límites superiores para delta_u

                # Restricciones de horizonte de control M
                if k >= self.M:
                    # Uk = U_{M-1} para k >= M
                    self.g.append(Uk_k - Uk_prev_control_horizon)
                    self.lbg.extend([-1e-9] * self.nu)
                    self.ubg.extend([+1e-9] * self.nu)

                # Variables de estado de colocación Xc_k
                Xc_k = ca.MX.sym(f'Xc_{k}', self.nx, self.m)
                self.w.append(ca.vec(Xc_k))
                self.lbw.extend(self.lbx * self.m) # Límites para estados de colocación
                self.ubw.extend(self.ubx * self.m)

                # Aplicar el paso de colocación
                Xk_end_k, coll_eqs_k = self.F_coll(Xk_iter, Xc_k, Uk_k)

                # Añadir restricciones de colocación (igualdad a cero)
                self.g.append(coll_eqs_k)
                self.lbg.extend([-1e-9] * self.nx * self.m)
                self.ubg.extend([+1e-9] * self.nx * self.m)

                # Variable para el estado al final del intervalo X_{k+1}
                Xk_next = ca.MX.sym(f'X_{k+1}', self.nx)
                self.w.append(Xk_next)
                self.lbw.extend(self.lbx) # Límites para Xk_next
                self.ubw.extend(self.ubx)

                # Restricción de continuidad (disparo) Xk_end_k == Xk_next
                self.g.append(Xk_end_k - Xk_next)
                self.lbg.extend([-1e-9] * self.nx)
                self.ubg.extend([+1e-9] * self.nx)

                # Calcular costo del paso k+1
                Ck_next = self.output_func(Xk_next) # Salida predicha en k+1
                sp_k = self.sp_sym[:, k]           # Setpoint en el paso k (para el estado k+1)
                # Penalizar desviación de salida respecto al setpoint
                J += ca.mtimes([(Ck_next - sp_k).T, self.Q, (Ck_next - sp_k)])
                # Penalizar cambio en la entrada
                J += ca.mtimes([delta_u.T, self.W, delta_u])

                # Actualizar para el siguiente paso
                Xk_iter = Xk_next # El estado inicial del siguiente paso es Xk_next
                Uk_prev = Uk_k
                if k == self.M - 1:
                    Uk_prev_control_horizon = Uk_k # Guardar U_{M-1}

            # --- Crear el solver NLP ---
            nlp_dict = {
                'f': J,
                'x': ca.vertcat(*self.w),
                'g': ca.vertcat(*self.g),
                'p': p_nlp
            }
            opts = {
                'ipopt.print_level': 0, # 0 silencia la salida de IPOPT
                'print_time': 0,
                'ipopt.max_iter': 150,
                'ipopt.tol': 1e-6,
                'ipopt.acceptable_tol': 1e-5,
                'ipopt.warm_start_init_point': 'yes',
                # Ajustes opcionales para robustez / velocidad
                'ipopt.warm_start_bound_push': 1e-9,
                'ipopt.warm_start_mult_bound_push': 1e-9,
                # 'ipopt.mu_strategy': 'adaptive',
                # 'ipopt.hessian_approximation': 'limited-memory' # Si el cálculo Hessiano es lento
            }
            self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

            # Guardar índices para extraer resultados fácilmente
            self._prepare_indices()

        def _prepare_indices(self):
            """Calcula los índices de inicio para cada tipo de variable en w."""
            self.indices = {'X': [], 'U': [], 'Xc': []}
            offset = 0
            # X0 NO está en w (es parámetro)
            for k in range(self.N):
                # Uk
                self.indices['U'].append(offset)
                offset += self.nu
                # Xc_k (m*nx variables)
                self.indices['Xc'].append(offset)
                offset += self.nx * self.m
                # X_{k+1}
                self.indices['X'].append(offset) # Índice para X_{k+1}
                offset += self.nx
            self.dim_w = offset # Dimensión total del vector w

        def solve(self, x_current, sp_trajectory, u_previous):
            """Resuelve el problema NMPC."""
            # Establecer valores de parámetros
            # sp_trajectory debe ser [nc x N]
            p_val = np.concatenate([
                x_current,
                sp_trajectory.flatten('F'), # 'F' for Fortran order (column-major)
                u_previous
            ])

            # Preparar punto inicial (warm start)
            current_w0 = self.w0 # Usa la solución anterior

            # Si es la primera llamada o w0 es inválido, intentar inicializar
            if np.all(np.abs(current_w0) < 1e-9) or len(current_w0) != self.dim_w:
                # print("Generando guess inicial para w0...")
                w0_guess = []
                x_guess = np.array(x_current) # Copiar
                u_guess = np.array(u_previous) # Copiar
                for k in range(self.N):
                    w0_guess.extend(np.clip(u_guess, self.lbu, self.ubu)) # U_k (clip a límites)
                    # Inicializar Xc con el estado actual repetido
                    w0_guess.extend(np.tile(np.clip(x_guess, self.lbx, self.ubx), self.m)) # Xc_k (clip)
                    # Predecir X_{k+1} (simple Euler, podría ser mejor)
                    try:
                        dx_guess = self.model_ode(x_guess, u_guess).full().flatten()
                        x_guess = x_guess + dx_guess * self.dt
                    except:
                         # Si falla la evaluación, mantener el guess anterior
                        pass

                    w0_guess.extend(np.clip(x_guess, self.lbx, self.ubx)) # X_{k+1} (clip)
                # Asegurarse de que la dimensión sea correcta
                if len(w0_guess) == self.dim_w:
                    current_w0 = np.array(w0_guess)
                else:
                    # Fallback a ceros si la construcción del guess falló
                    # print(f"Warning: Dimension mismatch in w0 guess: {len(w0_guess)} vs {self.dim_w}. Usando ceros.")
                    current_w0 = np.zeros(self.dim_w)

            # Resolver el NLP
            try:
                sol = self.solver(
                    x0=current_w0,
                    lbx=self.lbw,
                    ubx=self.ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=p_val
                )
                w_opt = sol['x'].full().flatten()
                sol_stats = self.solver.stats()

                # Verificar éxito del solver
                if not sol_stats['success']:
                    st.warning(f"NMPC Solver did not converged! Status: {sol_stats.get('return_status', 'Unknown')}")
                    # No actualizar w0 si falló, reusar el anterior o resetear
                    # self.w0 = np.zeros(self.dim_w) # Opción: resetear en fallo
                    return u_previous, sol_stats, None, None # Devolver entrada anterior

                # Actualizar w0 para warm start de la PRÓXIMA llamada
                self.w0 = w_opt

                # --- Extraer la secuencia de control óptima y predicciones ---
                u_optimal_sequence = np.zeros((self.nu, self.N))
                x_predicted_sequence = np.zeros((self.nx, self.N + 1))
                x_predicted_sequence[:, 0] = x_current # El primer estado es el actual

                for k in range(self.N):
                    u_optimal_sequence[:, k] = w_opt[self.indices['U'][k] : self.indices['U'][k] + self.nu]
                    x_predicted_sequence[:, k+1] = w_opt[self.indices['X'][k] : self.indices['X'][k] + self.nx]

                u_apply = u_optimal_sequence[:, 0] # Primera acción de control

                return u_apply, sol_stats, x_predicted_sequence, u_optimal_sequence

            except Exception as e:
                st.error(f"Error during the NMPC optimization: {e}")
                # Fallback: devolver la entrada anterior
                return u_previous, {'success': False, 'return_status': 'SolverError'}, None, None

    # ---------------------------------------------------
    # 3. Simulación del Sistema (Planta) - Función separada
    # ---------------------------------------------------
    # @st.cache_data # No cachear si depende de u que cambia
    def simulate_plant_step(x_current, u_applied, dt_sim, model_ode_func):
        """Simulates a step of the real plant (integrating model)."""
        # Define la función de EDO para solve_ivp (compatible con NumPy)
        def ode_sys(t, x, u):
            # Llama a la función CasADi y convierte a numpy array plano
            return model_ode_func(x, u).full().flatten()

        # Integra usando solve_ivp
        t_span = [0, dt_sim]
        sol = solve_ivp(ode_sys, t_span, x_current, args=(u_applied,), method='RK45', rtol=1e-5, atol=1e-8)
        # Devuelve el estado al final del intervalo
        return sol.y[:, -1]

    # ---------------------------------------------------
    # 4. Bucle Principal de Simulación (dentro del if button)
    # ---------------------------------------------------
    if start_simulation:
        st.info("Starting NMPC Simulation ...")
        progress_bar = st.progress(0)
        status_text = st.empty() # Para mostrar el progreso

        # Recuperar TODOS los parámetros de la barra lateral
        params_dict = {
            'mu_max': mu_max_input, 'K_S': K_S_input, 'Y_XS': Y_XS_input,
            'Y_QX': Y_QX_input, 'S_in': S_in_input, 'V': V_input,
            'rho': rho_input, 'Cp': Cp_input, 'T_in': T_in_input,
            'F_const': F_const_input
        }

        # Obtener modelo con parámetros actuales
        model_ode, output_func, x_sym, u_sym, c_sym, dx_sym, params = get_bioreactor_model(params_dict)
        nx = x_sym.shape[0]
        nu = u_sym.shape[0]
        nc = c_sym.shape[0]

        # Pesos NMPC
        Q_weights = [Q_X_weight, Q_T_weight]
        W_weights = [W_FS_weight, W_Qj_weight] # W para [delta_Fs, delta_Qj]

        # Límites para el NLP (inputs ya están en las unidades correctas [L/h, W])
        lbx_opt = [min_X_opt, min_S_opt, min_T_opt]
        ubx_opt = [max_X_opt, max_S_opt, max_T_opt]
        # Límites de entrada u = [F_S, Q_j]
        lbu_nmpc = [min_FS, min_Qj]
        ubu_nmpc = [max_FS, max_Qj]
        # Límites tasa de cambio delta_u = [delta_F_S, delta_Q_j]
        lbdu_nmpc = [-delta_FS_max, -delta_Qj_max]
        ubdu_nmpc = [ delta_FS_max,  delta_Qj_max]

        # Crear instancia NMPC con configuración actual
        try:
            nmpc = NMPCBioreactor(dt_nmpc_input, N_input, M_input, Q_weights, W_weights,
                                  model_ode, output_func, x_sym, u_sym, c_sym, params,
                                  lbx_opt, ubx_opt, lbu_nmpc, ubu_nmpc, lbdu_nmpc, ubdu_nmpc)
        except Exception as e:
            st.error(f"Error initializing NMPC: {e}")
            st.stop() # Detener ejecución si falla la inicialización

        # --- Preparar Simulación ---
        t_final = simulation_time
        dt_sim = dt_nmpc_input # Usar el mismo dt para simulación y NMPC
        n_steps = int(t_final / dt_sim)

        # Condiciones iniciales
        x_current = np.array([initial_X, initial_S, initial_T])
        u_previous = np.array([initial_FS, initial_Qj]) # [F_S (L/h), Q_j (W)]

        # Setpoints (usar los valores iniciales como SP inicial)
        # El SP inicial debe tener dimension nc
        initial_sp = np.array([initial_X, initial_T])
        current_setpoint = initial_sp.copy()

        # Definir los cambios de setpoint y los tiempos
        setpoint_changes = {
            t_sp1: np.array([setpoint_X_t1, setpoint_T_t1]),
            t_sp2: np.array([setpoint_X_t2, setpoint_T_t2]),
        }

        # Historiales
        t_history = np.linspace(0, t_final, n_steps + 1)
        x_history = np.zeros((nx, n_steps + 1))
        u_history = np.zeros((nu, n_steps))
        c_history = np.zeros((nc, n_steps + 1)) # Salidas controladas
        sp_history = np.zeros((nc, n_steps + 1)) # Setpoint efectivo

        # Guardar estado inicial
        x_history[:, 0] = x_current
        c_history[:, 0] = output_func(x_current).full().flatten()
        sp_history[:, 0] = current_setpoint

        start_time_sim = time.time()

        # --- Bucle de Simulación ---
        for k in range(n_steps):
            t_current = k * dt_sim
            status_text.text(f"Simulating step {k+1}/{n_steps} (t={t_current:.2f}h)...")

            # Actualizar setpoint si es necesario
            # Comprobar si el *siguiente* paso cruzará un tiempo de cambio de SP
            next_step_time = (k + 1) * dt_sim
            for sp_time, new_sp in sorted(setpoint_changes.items()):
                # Si el tiempo del SP está dentro del intervalo [t_current, next_step_time)
                if t_current < sp_time <= next_step_time + 1e-9: # Usar tolerancia pequeña
                    st.write(f"-> Setpoint change detected in t ≈ {sp_time:.2f}h to: X={new_sp[0]}, T={new_sp[1]}K")
                    current_setpoint = new_sp
                    break # Aplicar solo el primer cambio encontrado en el intervalo

            # Crear trayectoria de setpoint para el NMPC (constante por ahora)
            # sp_traj shape: [nc x N]
            sp_traj = np.tile(current_setpoint, (N_input, 1)).T

            # Resolver NMPC
            u_optimal, stats, _, _ = nmpc.solve(x_current, sp_traj, u_previous)

            # Aplicar lógica si el solver falló
            if not stats['success']:
                # Mantener la entrada anterior si NMPC falla
                u_apply = u_previous
                st.warning(f"Step {k+1}: NMPC did not converged, using u_prev = [{u_apply[0]:.3f}, {u_apply[1]:.1f}]")
            else:
                u_apply = u_optimal
                # Aplicar restricciones de límites y tasa de cambio a la acción calculada
                # 1. Restringir delta_u
                delta_u_applied = u_apply - u_previous
                delta_u_applied = np.clip(delta_u_applied, lbdu_nmpc, ubdu_nmpc)
                # 2. Calcular u_apply basado en delta_u restringido
                u_apply = u_previous + delta_u_applied
                # 3. Restringir u_apply a sus límites absolutos
                u_apply = np.clip(u_apply, lbu_nmpc, ubu_nmpc)

            # Simular la planta para el siguiente paso
            try:
                x_next = simulate_plant_step(x_current, u_apply, dt_sim, model_ode)
            except Exception as e:
                st.error(f"Error when simulating the plant in step {k+1}: {e}")
                st.warning("Stopping simulation.")
                break # Salir del bucle si la simulación de la planta falla

            # Actualizar estado y entrada para el siguiente ciclo
            x_current = x_next
            u_previous = u_apply

            # Guardar historial
            x_history[:, k+1] = x_current
            u_history[:, k] = u_apply
            c_history[:, k+1] = output_func(x_current).full().flatten()
            sp_history[:, k+1] = current_setpoint # Guardar SP usado en este paso

            # Actualizar barra de progreso
            progress_bar.progress((k + 1) / n_steps)

        end_time_sim = time.time()
        total_sim_time = end_time_sim - start_time_sim
        status_text.text(f"Simulation completed in {total_sim_time:.2f} seconds.")
        st.success("NMPC simulation completed.")

        # --- Graficar Resultados ---
        st.subheader("NMPC Simulation Results")

        # Crear figura Matplotlib
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        fig.suptitle("NMPC Bioreactor Simulation Results", fontsize=16)

        # Estados y Setpoints
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

        # Entradas (acción de control)
        # Usar drawstyle='steps-post' para visualizar mejor acción ZOH
        axes[0, 1].step(t_history, np.append(u_history[0, :], u_history[0, -1]), where='post', label=f'Substrate Flow (F_S)')
        axes[0, 1].hlines([min_FS, max_FS], t_history[0], t_history[-1], colors='gray', linestyles='--', label='Fs Limits')
        axes[0, 1].set_ylabel('F_S (L/h)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 1].step(t_history, np.append(u_history[1, :], u_history[1, -1]), where='post', label=f'Thermal Load (Q_j)')
        axes[1, 1].hlines([min_Qj, max_Qj], t_history[0], t_history[-1], colors='gray', linestyles='--', label='Qj Limits')
        axes[1, 1].set_ylabel('Q_j (W)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Gráfico vacío para alinear
        axes[2, 1].axis('off')
        axes[2, 1].set_xlabel('Time (h)')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo

        # Mostrar la figura en Streamlit
        st.pyplot(fig)

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    st.set_page_config(layout="wide") # Usar layout ancho para más espacio
    nmpc_page()