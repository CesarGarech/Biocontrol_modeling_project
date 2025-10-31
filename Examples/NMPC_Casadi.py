import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------
# 1. Modelo del Biorreactor (Simbólico con CasADi)
# ---------------------------------------------------
def get_bioreactor_model():
    """Define el modelo simbólico del biorreactor usando CasADi."""
    # Parameters del modelo (ejemplo - ¡ajusta a tu sistema!)
    params = {
        'mu_max': 0.4,     # Tasa máx crecimiento (1/h)
        'K_S': 0.05,       # Constante de Monod (g/L)
        'Y_XS': 0.5,       # Rendimiento Biomasa/Substrato (g/g)
        'Y_QX': 15000,     # Rendimiento Calor/Biomasa (J/g) - ¡Ajustar!
        'S_in': 10.0,      # Concentración de sustrato entrante (g/L)
        'V': 1.0,          # Volumen del reactor (L)
        'rho': 1000.0,     # Densidad del medio (kg/m^3 -> g/L)
        'Cp': 4184.0,      # Capacidad calorífica (J/(kg*K) -> J/(g*K))
        'T_in': 298.15,    # Temperatura de entrada (K)
        'F_const': 0.0,    # Flujo constante adicional (L/h) - si existe
    }

    # Variables simbólicas
    X = ca.MX.sym('X')  # Concentración de biomasa (g/L)
    S = ca.MX.sym('S')  # Concentración de sustrato (g/L)
    T = ca.MX.sym('T')  # Temperatura del reactor (K)
    x = ca.vertcat(X, S, T) # x es puramente simbólico (vertcat de MX.sym)

    # Define 'u' as a base symbol of 2 elements
    u = ca.MX.sym('u', 2)
    # u[0] representará F_S (L/h)
    # u[1] representará Q_j / 3600.0 (Carga térmica en J/h) - ¡Comentario indica J/h pero división sugiere Watts!

    # --- Ecuaciones del modelo ---
    mu = params['mu_max'] * S / (params['K_S'] + S + 1e-9) # Tasa crecimiento Monod (+epsilon)

    # Extraer F_S simbólico de u
    F_S = u[0]
    F_total = F_S + params['F_const']
    D = F_total / params['V'] # Tasa de dilución (1/h)

    # Balances de materia
    dX_dt = mu * X - D * X
    dS_dt = D * (params['S_in'] - S) - (mu / params['Y_XS']) * X

    # Balance de energía
    Q_gen = params['Y_QX'] * mu * X * params['V'] # Calor generado (J/h)
    Q_flow = F_total * params['rho'] * params['Cp'] * (params['T_in'] - T) # Calor por flujo (J/h)
    # Asumimos que u[1] es directamente el calor añadido/quitado por la chaqueta en J/h
    # ¡¡¡ POSIBLE INCONSISTENCIA DE UNIDADES: u[1] parece estar en Watts (J/s) por la división !!!
    # Si u[1] está en Watts, Q_rem debería ser - u[1] * 3600.0 para estar en J/h
    Q_rem = - u[1] * 3600.0 # Calor removido (J/h) - Corregido asumiendo u[1] es Q_j/3600 (Watts)

    dT_dt = (Q_gen + Q_rem + Q_flow) / (params['rho'] * params['Cp'] * params['V']) # (K/h)

    # Vector de derivadas
    dx = ca.vertcat(dX_dt, dS_dt, dT_dt)

    # Controlled variables (outputs)
    c = ca.vertcat(X, T)

    # Create CasADi functions
    # Ahora los inputs [x, u] son ambos símbolos base o vertcat de símbolos base.
    model_ode = ca.Function('model_ode', [x, u], [dx], ['x', 'u'], ['dx'])
    output_func = ca.Function('output_func', [x], [c], ['x'], ['c'])

    # Devuelve el símbolo 'u' que ahora es MX.sym('u', 2)
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
            dt: Tiempo de muestreo (h)
            N: Horizonte de predicción
            M: Horizonte de control
            Q: Matriz de peso para error de salida (CV)
            W: Matriz de peso para movimiento de entrada (MV)
            model_ode: Función CasADi para las EDOs dx = f(x, u)
            output_func: Función CasADi para las salidas c = h(x)
            x_sym, u_sym, c_sym: Variables simbólicas CasADi
            params: Diccionario de parámetros del modelo
            lbx, ubx: Límites inferiores/superiores para estados x
            lbu, ubu: Límites inferiores/superiores para entradas u
            lbdu, ubdu: Límites inferiores/superiores para tasa de cambio de entradas du
            m: Grado del polinomio de colocación
            pol: Tipo de polinomio ('legendre' o 'radau')
        """
        self.dt = dt
        self.N = N
        self.M = M
        self.Q = np.diag(Q) # Ensure que sean matrices diagonales
        self.W = np.diag(W)
        self.model_ode = model_ode
        self.output_func = output_func
        self.params = params
        self.nx = x_sym.shape[0]
        self.nu = u_sym.shape[0]
        self.nc = c_sym.shape[0]
        self.lbx = lbx
        self.ubx = ubx
        self.lbu = lbu
        self.ubu = ubu
        self.lbdu = lbdu
        self.ubdu = ubdu
        self.m = m
        self.pol = pol

        # --- Preparar Colocación Ortogonal ---
        # Puntos de colocación (tau_root[0] = 0)
        self.tau_root = np.append(0, ca.collocation_points(self.m, self.pol))

        # Matriz de coeficientes para derivadas en puntos de colocación
        self.C = np.zeros((self.m + 1, self.m + 1))
        # Matriz de coeficientes para la integral (estado final)
        self.D = np.zeros(self.m + 1)

        # Construir C y D
        for j in range(self.m + 1):
            # Construir polinomio de Lagrange j
            p = np.poly1d([1])
            for r in range(self.m + 1):
                if r != j:
                    p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j] - self.tau_root[r] + 1e-10) # Avoid div por cero

            # Evaluar la derivada del polinomio en los puntos tau
            p_der = np.polyder(p)
            for i in range(self.m + 1):
                self.C[j, i] = np.polyval(p_der, self.tau_root[i])

            # Evaluar la integral del polinomio de 0 a 1
            p_int = np.polyint(p)
            self.D[j] = np.polyval(p_int, 1.0)

        # --- Construir el NLP ---
        self._build_nlp()

    def _build_nlp(self):
        """Construye el problema NLP de optimización."""
        # Create an instance of the collocation function once
        # Symbolic variables for an integration step
        Xk_step = ca.MX.sym('Xk_step', self.nx)
        Xc_step = ca.MX.sym('Xc_step', self.nx, self.m) # Estados en puntos interiores k,1...k,m
        Uk_step = ca.MX.sym('Uk_step', self.nu)

        # Calculate ecuaciones de colocación y estado final para un paso
        X_all_coll_step = ca.horzcat(Xk_step, Xc_step) # Estados en k,0(=Xk_step), k,1, ..., k,m
        ode_at_coll_step = []
        for j in range(1, self.m + 1): # Calculate ODE en puntos interiores 1..m
            ode_at_coll_step.append(self.model_ode(X_all_coll_step[:, j], Uk_step))
        # ode_at_coll_step es ahora una lista de m vectores columna [nx x 1]

        # Restricciones de colocación: Xk_coll_j - (Xk + dt * sum(Aij*f(Xk_coll_i, uk))) = 0
        # Usaremos una forma diferente: dx/dt = f(x,u) => dt*f(x_coll, u) = sum(Cij*x_coll_j)
        # donde x_coll incluye Xk_step
        coll_eqs_step = []
        for j in range(1, self.m + 1): # Para cada punto de colocación interior j = 1..m
            xp_coll_j = 0 # Suma Polinomios Derivados * Estados Colocacion
            for r in range(self.m + 1):
                xp_coll_j += self.C[r, j] * X_all_coll_step[:, r]
            # Ecuación: Derivada estimada = dt * Modelo evaluado en el punto
            coll_eqs_step.append(xp_coll_j - (self.dt * ode_at_coll_step[j-1])) # ode_at_coll_step[j-1] es f(X_coll_j, Uk)

        # Estado al final del intervalo (usando coeficientes D)
        Xk_end_step = Xk_step # Empezar con Xk
        for j in range(1, self.m + 1): # Sumar contribuciones de los puntos interiores
            Xk_end_step += self.dt * self.D[j] * ode_at_coll_step[j-1]

        # Create the CasADi function for one step
        self.F_coll = ca.Function('F_coll', [Xk_step, Xc_step, Uk_step],
                                    [Xk_end_step, ca.vertcat(*coll_eqs_step)],
                                    ['Xk_step', 'Xc_step', 'Uk_step'], ['Xk_end', 'coll_eqs'])

        # --- NMPC decision variables ---
        self.w = []        # Vector de variables de decisión
        self.w0_init = [] # Estimación inicial (basada en parámetros)
        self.lbw = []      # Límite inferior
        self.ubw = []      # Límite superior

        self.g = []        # Vector de restricciones
        self.lbg = []      # Límite inferior
        self.ubg = []      # Límite superior

        # Parameters del NLP (estado inicial, setpoints, entrada anterior)
        self.x0_sym = ca.MX.sym('x0', self.nx)
        self.sp_sym = ca.MX.sym('sp', self.nc, self.N)
        self.uprev_sym = ca.MX.sym('uprev', self.nu)
        p_nlp = ca.vertcat(self.x0_sym, ca.vec(self.sp_sym), self.uprev_sym)

        # Function de costo
        J = 0
        Uk_prev = self.uprev_sym # Entrada anterior para calcular delta_u

        # El estado al inicio del primer intervalo es el PARÁMETRO x0_sym
        Xk_iter = self.x0_sym

        # Loop over prediction horizon N
        for k in range(self.N):
            # Input variables Uk
            Uk_k = ca.MX.sym(f'U_{k}', self.nu)
            self.w.append(Uk_k)
            # Inicializar entradas con u_previous
            self.w0_init.extend(np.zeros(self.nu)) # Placeholder, se usará uprev_sym numérico después
            self.lbw.extend(self.lbu)
            self.ubw.extend(self.ubu)

            # Restricciones delta_u
            delta_u = Uk_k - Uk_prev
            self.g.append(delta_u)
            self.lbg.extend(self.lbdu)
            self.ubg.extend(self.ubdu)

            # Restricciones de horizonte de control M
            if k >= self.M:
                self.g.append(Uk_k - Uk_prev_control_horizon) # Uk = U_{M-1}
                self.lbg.extend([-1e-9] * self.nu) # Permitir tolerancia numérica
                self.ubg.extend([+1e-9] * self.nu)

            # Collocation state variables Xc_k (only interior points 1..m)
            Xc_k = ca.MX.sym(f'Xc_{k}', self.nx, self.m)
            self.w.append(ca.vec(Xc_k)) # Vectorizar para añadir a w
            # Inicializar estados de colocación con x0_sym
            self.w0_init.extend(np.zeros(self.nx * self.m)) # Placeholder
            self.lbw.extend(self.lbx * self.m)
            self.ubw.extend(self.ubx * self.m)

            # Aplicar el paso de colocación usando Xk_iter (que es x0_sym para k=0)
            Xk_end_k, coll_eqs_k = self.F_coll(Xk_iter, Xc_k, Uk_k)

            # Añadir restricciones de colocación
            self.g.append(coll_eqs_k)
            self.lbg.extend([-1e-9] * self.nx * self.m) # Igualdad con tolerancia numérica (m ecuaciones)
            self.ubg.extend([+1e-9] * self.nx * self.m)

            # Variable para el estado al final del intervalo X_{k+1}
            Xk_next = ca.MX.sym(f'X_{k+1}', self.nx)
            self.w.append(Xk_next)
            # Inicializar Xk_next con x0_sym
            self.w0_init.extend(np.zeros(self.nx)) # Placeholder
            self.lbw.extend(self.lbx)
            self.ubw.extend(self.ubx)

            # Restricción de continuidad (disparo)
            self.g.append(Xk_end_k - Xk_next)
            self.lbg.extend([-1e-9] * self.nx) # Permitir pequeña tolerancia numérica
            self.ubg.extend([+1e-9] * self.nx)

            # Calculate costo del paso k
            Ck = self.output_func(Xk_next) # Salida al final del intervalo k+1
            sp_k = self.sp_sym[:, k]      # Setpoint en el paso k
            J += (Ck - sp_k).T @ self.Q @ (Ck - sp_k) # Costo de salida
            J += delta_u.T @ self.W @ delta_u          # Costo de movimiento de entrada

            # Actualizar para el siguiente paso
            Xk_iter = Xk_next # Ahora Xk_iter es la VARIABLE DE DECISIÓN X_{k+1}
            Uk_prev = Uk_k
            if k == self.M - 1:
                Uk_prev_control_horizon = Uk_k # Guardar la última entrada del horizonte M

        # --- Create the NLP solver ---
        nlp_dict = {
            'f': J,
            'x': ca.vertcat(*self.w), # Variables: U0, Xc0, X1, U1, Xc1, X2, ...
            'g': ca.vertcat(*self.g), # Restricciones
            'p': p_nlp                 # Parameters: x0, sp, uprev
        }

        # Options del solver (IPOPT)
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 150,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-9,
            'ipopt.warm_start_mult_bound_push': 1e-9,
            # 'ipopt.mu_strategy': 'adaptive',
            # 'ipopt.hessian_approximation': 'limited-memory'
        }

        # --- DEBUGGING START ---
        print("--- nlp_dict ---")
        for key, value in nlp_dict.items():
            print(f"Key: {key}, Type: {type(value)}")
            if isinstance(value, ca.MX):
                print(f"  Shape: {value.shape}")
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if value:
                    print(f"  Type of first element: {type(value[0])}")
        print("------------------")
        # --- DEBUGGING END ---

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

        # Guardar índices para extraer resultados fácilmente
        self._prepare_indices()

        # Inicializar w0 (vector de ceros con la dimensión correcta)
        # Se usará numéricamente en solve() para la primera iteración
        self.w0 = np.zeros(ca.vertcat(*self.w).shape[0]) # CORRECCIÓN AQUÍ


    def _prepare_indices(self):
        """Calcula los índices de inicio para cada tipo de variable en w."""
        self.indices = {'X': [], 'U': [], 'Xc': []}
        offset = 0
        # X0 NO está en w
        for k in range(self.N):
            # Uk
            self.indices['U'].append(offset)
            offset += self.nu
            # Xc_k (m*nx variables)
            self.indices['Xc'].append(offset)
            offset += self.nx * self.m
            # X_{k+1}
            self.indices['X'].append(offset) # Este índice corresponde a X_{k+1}
            offset += self.nx
        self.dim_w = offset


    def solve(self, x_current, sp_trajectory, u_previous):
        """
        Resuelve el problema NMPC para un estado inicial y trayectoria de setpoint.
        Args:
            x_current: Estado actual del sistema (vector numpy)
            sp_trajectory: Trayectoria de setpoints para el horizonte N (numpy array [nc x N])
            u_previous: Acción de control aplicada en el paso anterior (vector numpy)
        Returns:
            u_optimal: La primera acción de control óptima a aplicar (vector numpy)
            sol_stats: Estadísticas de la solución del solver
            predicted_x: Trayectoria de estados predicha (numpy array [nx x N+1])
            predicted_u: Secuencia de entradas predicha (numpy array [nu x N])
        """
        # Establecer valores de parámetros
        p_val = np.concatenate([x_current, sp_trajectory.flatten('F'), u_previous])

        # Usar la última solución como punto de partida si está disponible y tiene la dimensión correcta
        current_w0 = self.w0 if len(self.w0) == ca.vertcat(*self.w).shape[0] else np.zeros(ca.vertcat(*self.w).shape[0])

        # Si es la primera llamada, intentar inicializar w0 de forma más inteligente
        if np.all(current_w0 == 0):
            w0_guess = []
            x_guess = x_current
            u_guess = u_previous
            for k in range(self.N):
                w0_guess.extend(u_guess) # U_k
                # Inicializar Xc con el estado actual repetido
                w0_guess.extend(np.tile(x_guess, self.m)) # Xc_k
                # Predecir X_{k+1} (simplificado - podrías integrar un paso)
                # dx_guess = self.model_ode(x_guess, u_guess).full().flatten()
                # x_guess = x_guess + dx_guess * self.dt
                w0_guess.extend(x_guess) # X_{k+1}
            if len(w0_guess) == ca.vertcat(*self.w).shape[0]:
                current_w0 = np.array(w0_guess)
            #else:
                # print(f"Warning: Dimension mismatch in w0 guess: {len(w0_guess)} vs {self.solver.n_x()}")
                # pass # Keep current_w0 as zeros


        # Resolver el NLP
        try:
            sol = self.solver(
                x0=current_w0,     # Estimación inicial (warm start)
                lbx=self.lbw,      # Usar límites originales definidos
                ubx=self.ubw,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p_val
            )
        except RuntimeError as e:
            print(f"Error en la llamada al solver: {e}")
            # Intentar resolver sin warm start (x0=[0]) como último recurso
            try:
                print("Intentando resolver sin warm start...")
                sol = self.solver(
                    x0=np.zeros(ca.vertcat(*self.w).shape[0]), # Intentar con ceros
                    lbx=self.lbw, ubx=self.ubx,
                    lbg=self.lbg, ubg=self.ubg,
                    p=p_val
                )
            except RuntimeError as e2:
                print(f"Segundo error en la llamada al solver: {e2}")
                return u_previous, {'success': False, 'return_status': 'SolverError'}, None, None


        # Extraer la solución
        w_opt = sol['x'].full().flatten()
        sol_stats = self.solver.stats()

        # Verify éxito del solver
        if not sol_stats['success']:
            # Imprimir más detalles si está disponible
            print(f"¡ADVERTENCIA: El solver NMPC no convergió! Estado: {sol_stats.get('return_status', 'Desconocido')}")
            # print(f"Iteraciones: {sol_stats.get('iter_count', 'N/A')}")
            # Considerar resetear w0 si falla repetidamente
            # self.w0 = np.zeros(self.solver.n_x())
            return u_previous, sol_stats, None, None

        # Actualizar w0 para warm start para la PRÓXIMA llamada
        self.w0 = w_opt

        # --- Extraer la secuencia de control óptima y predicciones ---
        u_optimal_sequence = np.zeros((self.nu, self.N))
        x_predicted_sequence = np.zeros((self.nx, self.N + 1))

        # El primer estado de la predicción es el estado actual real
        x_predicted_sequence[:, 0] = x_current

        # Extraer el resto de la trayectoria de w_opt
        for k in range(self.N):
            u_optimal_sequence[:, k] = w_opt[self.indices['U'][k] : self.indices['U'][k] + self.nu]
            # self.indices['X'][k] apunta al índice de inicio de X_{k+1} en w_opt
            x_predicted_sequence[:, k+1] = w_opt[self.indices['X'][k] : self.indices['X'][k] + self.nx]

        # Devolver la primera acción de control
        u_apply = u_optimal_sequence[:, 0]

        return u_apply, sol_stats, x_predicted_sequence, u_optimal_sequence


# ---------------------------------------------------
# 3. Simulación del Sistema (Planta)
# ---------------------------------------------------
def simulate_plant(x0, u, dt, model_ode):
    """Simula un paso de la planta real (integrando el modelo)."""
    # Ensure que u esté en formato correcto para la EDO
    # u[0] es F_S (L/h), u[1] es Q_j/3600.0 (J/h)
    u_val = u

    # Define la función de EDO para solve_ivp
    def ode_sys(t, x):
        # Llama a la función CasADi y convierte a numpy array plano
        return model_ode(x, u_val).full().flatten()

    # Integra
    t_span = [0, dt]
    sol = solve_ivp(ode_sys, t_span, x0, method='RK45', rtol=1e-5, atol=1e-8) # Puedes usar 'BDF' para sistemas stiff

    # Devuelve el estado al final del intervalo
    return sol.y[:, -1]

# ---------------------------------------------------
# 4. Main Simulation Loop
# ---------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    t_final = 24.0      # Time final de simulación (h)
    dt_nmpc = 0.1       # Time de muestreo del NMPC (h)
    n_steps = int(t_final / dt_nmpc)

    # Obtener modelo
    model_ode, output_func, x_sym, u_sym, c_sym, dx_sym, params = get_bioreactor_model()
    nx = x_sym.shape[0]
    nu = u_sym.shape[0]
    nc = c_sym.shape[0]

    # Parameters NMPC
    N = 10          # Horizonte de predicción
    M = 4           # Horizonte de control
    # Pesos (¡AJUSTAR ESTOS!) - Mayor Q[0] penaliza error en X, Mayor Q[1] error en T
    # Mayor W[0] penaliza cambios en F_S, Mayor W[1] cambios en Q_j
    Q_weights = [1.0, 0.01]    # Pesos para [X, T]
    W_weights = [0.1, 1e-8]  # Pesos para cambios en [F_S, Q_j/3600] (Q_j suele ser grande)

    # Límites (Ejemplo - ¡AJUSTAR!)
    lbx = [0.0, 0.0, 290.0]    # [X_min, S_min, T_min(K)]
    ubx = [5.0, 10.0, 315.0]   # [X_max, S_max, T_max(K)]
    # Límites para u: [F_S (L/h), Q_j/3600 (J/h)]
    lbu = [0.0, -10000.0] # [F_S_min, Q_j_min/3600] (Puede enfriar)
    ubu = [1.5,   10000.0] # [F_S_max, Q_j_max/3600] (Puede calentar)
    # Límites para la tasa de cambio (delta_u)
    lbdu = [-0.1, -5000.0] # Máximo decremento por paso [dF_S, d(Q_j/3600)]
    ubdu = [ 0.1,   5000.0] # Máximo incremento por paso

    # Crear instancia NMPC
    print("Construyendo NMPC...")
    nmpc = NMPCBioreactor(dt_nmpc, N, M, Q_weights, W_weights, model_ode, output_func,
                            x_sym, u_sym, c_sym, params,
                            lbx, ubx, lbu, ubu, lbdu, ubdu)
    print("NMPC construido.")

    # --- Simulación ---
    # Estado inicial de la planta
    x_current = np.array([1.5, 9.0, 305.0]) # [X0, S0, T0]
    # Entrada inicial u = [F_S, Q_j/3600.0]
    u_previous = np.array([0.1, 0.0])

    # Definir los cambios de setpoint y los tiempos en que ocurren
    setpoint_changes = {
        5.0: np.array([2.0, 308.0]),  # En t=5h, cambiar a X=2.0, T=308.0
        10.0: np.array([1.0, 303.0]) # En t=10h, cambiar a X=1.0, T=303.0
    }
    current_setpoint = np.array([1.5, 305.0]) # Setpoint inicial

    # Historial para guardar resultados
    t_history = np.linspace(0, t_final, n_steps + 1)
    x_history = np.zeros((nx, n_steps + 1))
    u_history = np.zeros((nu, n_steps)) # Guardará [F_S, Q_j/3600]
    c_history = np.zeros((nc, n_steps + 1))
    sp_history = np.zeros((nc, n_steps + 1))

    x_history[:, 0] = x_current
    c_history[:, 0] = output_func(x_current).full().flatten()
    sp_history[:, 0] = current_setpoint

    print("Iniciando simulación NMPC...")
    for k in range(n_steps):
        t_current = k * dt_nmpc

        # Verify si hay un cambio de setpoint en el tiempo actual
        for time, new_sp in setpoint_changes.items():
            if abs(t_current - time) < 1e-9 or (t_current < time and abs((k + 1) * dt_nmpc - time) < 1e-9):
                print(f"-> Cambio de setpoint en t={time} h a: {new_sp}")
                current_setpoint = new_sp
                break # Solo aplicar el primer cambio que ocurra en este paso

        # Crear trayectoria de setpoints para el horizonte N con el setpoint actual
        sp_traj = np.tile(current_setpoint, (N, 1)).T # Array [nc x N]

        # 1. Resolver NMPC
        u_optimal, stats, _, _ = nmpc.solve(x_current, sp_traj, u_previous)

        if not stats['success']:
            print(f"-> Usando entrada anterior en paso {k}.")
            u_apply = u_previous
        else:
            u_apply = u_optimal
            # Aplicar saturación por si acaso (solver debería respetarlo)
            u_apply = np.clip(u_apply, lbu, ubu)
            # Aplicar restricción delta_u también por si acaso
            delta_u_applied = u_apply - u_previous
            delta_u_applied = np.clip(delta_u_applied, lbdu, ubdu)
            u_apply = u_previous + delta_u_applied
            # Re-aplicar saturación absoluta
            u_apply = np.clip(u_apply, lbu, ubu)


        # 2. Simular la planta
        x_next = simulate_plant(x_current, u_apply, dt_nmpc, model_ode)

        # 3. Actualizar estado y guardar historial
        x_current = x_next
        u_previous = u_apply # Actualizar entrada anterior

        x_history[:, k+1] = x_current
        u_history[:, k] = u_apply # Guardar u = [F_S, Q_j/3600]
        c_history[:, k+1] = output_func(x_current).full().flatten()
        sp_history[:, k+1] = current_setpoint # Guardar SP actual para graficar

        if (k + 1) % 10 == 0 or k == 0:
            print(f"Paso {k+1}/{n_steps} | t={t_current + dt_nmpc:.1f}h | X={x_current[0]:.3f} T={x_current[2]:.2f}K | Fs={u_apply[0]:.3f} Qj={u_apply[1]*3600:.1f}W | SP={current_setpoint}")

    print("Simulación completada.")

    # --- Graficar Resultados ---
    plt.figure(figsize=(12, 10))
    plt.suptitle("Simulación NMPC Biorreactor con Setpoints Escalonados (2x2)", fontsize=16)

    # Estados
    plt.subplot(3, 2, 1)
    plt.plot(t_history, x_history[0, :], label='Biomasa (X)')
    plt.plot(t_history, sp_history[0, :], 'r--', label='Setpoint X')
    plt.ylabel('Biomasa (g/L)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t_history, x_history[1, :], label='Substrato (S)')
    plt.ylabel('Substrato (g/L)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t_history, x_history[2, :], label='Temperatura (T)')
    plt.plot(t_history, sp_history[1, :], 'r--', label='Setpoint T')
    plt.ylabel('Temperatura (K)')
    plt.xlabel('Tiempo (h)')
    plt.legend()
    plt.grid(True)

    # Entradas (acción de control)
    plt.subplot(3, 2, 2)
    plt.step(t_history[:-1], u_history[0, :], where='post', label='Flujo Sustrato (F_S)')
    plt.ylabel('F_S (L/h)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=min(lbu[0] - 0.01, np.min(u_history[0,:]) - 0.01),
             top=max(ubu[0] + 0.01, np.max(u_history[0,:]) + 0.01)) # Ajustar límites eje Y


    plt.subplot(3, 2, 4)
    # Graficar Qj en Watts (u[1] * 3600)
    plt.step(t_history[:-1], u_history[1, :] * 3600.0, where='post', label='Carga Térmica (Q_j)')
    plt.ylabel('Q_j (W)')
    plt.xlabel('Tiempo (h)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=min(lbu[1]*3600 - 100, np.min(u_history[1,:]*3600) - 100),
             top=max(ubu[1]*3600 + 100, np.max(u_history[1,:]*3600) + 100)) # Ajustar límites eje Y

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo
    plt.show()