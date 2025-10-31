# NMPC_Casadi_unified.py
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from unified_bioreactor_model import get_unified_model, x0_unified, lbx, ubx, lbu, ubu, lbdu, ubdu # Importar modelo, x0 y límites

# ---------------------------------------------------
# 1. Get Unified Model and Parameters
# ---------------------------------------------------
# Usar el modelo completo con dinámica de oxígeno
ode_func, output_func_nmpc, x_sym, u_sym, c_sym, params, x_names, u_names, c_names = get_unified_model(use_oxygen_dynamics=True)
nx = x_sym.shape[0] # 6
nu = u_sym.shape[0] # 2: [F_S, Q_j]
nc = c_sym.shape[0] # 2: [F_S_meas, T_meas] - Salidas a controlar

print("[INFO] NMPC usando Modelo Unificado:")
print(f"  Estados (nx={nx}): {x_names}")
print(f"  Entradas (nu={nu}): {u_names}")
print(f"  Salidas Controladas (nc={nc}): {c_names}")


# ---------------------------------------------------
# 2. Clase NMPC (Modificada para el sistema 2x2 y modelo unificado)
# ---------------------------------------------------
class NMPCBioreactorUnified:
    def __init__(self, dt, N, M, Q, W, ode_func, output_func_nmpc,
                 x_sym, u_sym, c_sym, params,
                 lbx, ubx, lbu, ubu, lbdu, ubdu, m=3, pol='legendre'):
        """
        Inicializa el controlador NMPC para el modelo unificado.
        Args:
            dt, N, M: Parámetros NMPC
            Q: Matriz de peso para error de salida CV=[F_S, T] (tamaño nc x nc)
            W: Matriz de peso para movimiento de entrada MV=[F_S, Q_j] (tamaño nu x nu)
            ode_func: Función CasADi ODE dx = f(x, u)
            output_func_nmpc: Función CasADi Salidas c = h(x, u) -> [F_S, T]
            x_sym, u_sym, c_sym: Variables simbólicas CasADi
            params: Diccionario de parámetros del modelo
            lbx, ubx: Límites para estados x (tamaño nx)
            lbu, ubu: Límites para entradas u = [F_S, Q_j] (tamaño nu)
            lbdu, ubdu: Límites para tasa de cambio de entradas du (tamaño nu)
            m, pol: Parámetros colocación
        """
        self.dt = dt
        self.N = N
        self.M = M
        # Ensure Q y W sean matrices numpy (si vienen como listas)
        self.Q = np.diag(np.array(Q)) if isinstance(Q, (list, tuple)) else np.array(Q)
        self.W = np.diag(np.array(W)) if isinstance(W, (list, tuple)) else np.array(W)
        if self.Q.shape != (nc, nc): raise ValueError(f"Q debe ser {nc}x{nc}")
        if self.W.shape != (nu, nu): raise ValueError(f"W debe ser {nu}x{nu}")

        self.ode_func = ode_func
        self.output_func_nmpc = output_func_nmpc # c = h(x, u)
        self.params = params
        self.nx = nx
        self.nu = nu
        self.nc = nc
        self.lbx = lbx
        self.ubx = ubx
        self.lbu = lbu
        self.ubu = ubu
        self.lbdu = lbdu
        self.ubdu = ubdu
        self.m = m
        self.pol = pol

        # --- Preparar Colocación Ortogonal (sin cambios) ---
        self.tau_root = np.append(0, ca.collocation_points(self.m, self.pol))
        self.C = np.zeros((self.m + 1, self.m + 1))
        self.D = np.zeros(self.m + 1)
        for j in range(self.m + 1):
            p = np.poly1d([1])
            for r in range(self.m + 1):
                if r != j:
                    p *= np.poly1d([1, -self.tau_root[r]]) / (self.tau_root[j] - self.tau_root[r] + 1e-10)
            p_der = np.polyder(p)
            for i in range(self.m + 1): self.C[j, i] = np.polyval(p_der, self.tau_root[i])
            p_int = np.polyint(p)
            self.D[j] = np.polyval(p_int, 1.0)

        # --- Construir el NLP ---
        self._build_nlp()

    def _build_nlp(self):
        """Construye el problema NLP de optimización."""
        # --- Función de Colocación para un paso ---
        Xk_step = ca.MX.sym('Xk_step', self.nx)
        Xc_step = ca.MX.sym('Xc_step', self.nx, self.m) # Estados en puntos interiores k,1...k,m
        Uk_step = ca.MX.sym('Uk_step', self.nu)         # Entradas [F_S, Q_j] en intervalo k

        X_all_coll_step = ca.horzcat(Xk_step, Xc_step)
        ode_at_coll_step = []
        for j in range(1, self.m + 1):
             # Evalúa la ODE en el punto de colocación j con la entrada Uk constante en el intervalo
            ode_at_coll_step.append(self.ode_func(X_all_coll_step[:, j], Uk_step))

        coll_eqs_step = []
        for j in range(1, self.m + 1):
            xp_coll_j = ca.mtimes(X_all_coll_step, self.C[:, j]) # Derivative estimada
            coll_eqs_step.append(xp_coll_j - (self.dt * ode_at_coll_step[j-1])) # Restricción

        # Estado al final del intervalo usando coeficientes D y ODEs evaluadas
        Xk_end_step = ca.mtimes(X_all_coll_step, self.D) # Method directo con D

        self.F_coll = ca.Function('F_coll', [Xk_step, Xc_step, Uk_step],
                                    [Xk_end_step, ca.vertcat(*coll_eqs_step)],
                                    ['Xk_step', 'Xc_step', 'Uk_step'], ['Xk_end', 'coll_eqs'])

        # --- Variables de decisión del NMPC (w) ---
        self.w = []
        self.lbw = []
        self.ubw = []
        # --- Restricciones del NMPC (g) ---
        self.g = []
        self.lbg = []
        self.ubg = []

        # --- Parámetros del NLP (p) ---
        self.x0_sym = ca.MX.sym('x0', self.nx)
        # Setpoints para F_S y T para todo el horizonte N: [nc x N] -> [2 x N]
        self.sp_sym = ca.MX.sym('sp', self.nc, self.N)
        self.uprev_sym = ca.MX.sym('uprev', self.nu) # u_{k-1} = [F_S, Q_j]_{k-1}
        p_nlp = ca.vertcat(self.x0_sym, ca.vec(self.sp_sym), self.uprev_sym)

        # --- Construcción del NLP ---
        J = 0 # Costo
        Uk_prev = self.uprev_sym # Para calcular delta_u
        Xk_iter = self.x0_sym    # Estado inicial para el primer intervalo

        U_vars = [] # Guardar Uk para M>N

        for k in range(self.N):
            # Variables de entrada Uk = [F_S_k, Q_j_k]
            Uk_k = ca.MX.sym(f'U_{k}', self.nu)
            self.w.append(Uk_k)
            self.lbw.extend(self.lbu)
            self.ubw.extend(self.ubu)
            U_vars.append(Uk_k)

            # Restricciones delta_u
            delta_u = Uk_k - Uk_prev
            self.g.append(delta_u)
            self.lbg.extend(self.lbdu)
            self.ubg.extend(self.ubdu)

            # Restricciones de horizonte de control M
            if k >= self.M:
                # Uk = U_{M-1}
                self.g.append(Uk_k - U_vars[self.M-1])
                self.lbg.extend([-1e-9] * self.nu) # Igualdad numérica
                self.ubg.extend([+1e-9] * self.nu)

            # Variables de estado de colocación Xc_k
            Xc_k = ca.MX.sym(f'Xc_{k}', self.nx, self.m)
            self.w.append(ca.vec(Xc_k))
            self.lbw.extend(self.lbx * self.m) # Repetir límites para puntos de colocación
            self.ubw.extend(self.ubx * self.m)

            # Aplicar el paso de colocación F_coll
            Xk_end_k, coll_eqs_k = self.F_coll(Xk_iter, Xc_k, Uk_k)

            # Añadir restricciones de colocación g==0
            self.g.append(coll_eqs_k)
            self.lbg.extend([0.0] * self.nx * self.m) # Igualdad exacta (o con tolerancia)
            self.ubg.extend([0.0] * self.nx * self.m)

            # Variable para el estado al final del intervalo X_{k+1}
            Xk_next = ca.MX.sym(f'X_{k+1}', self.nx)
            self.w.append(Xk_next)
            self.lbw.extend(self.lbx) # Límites para el estado X_{k+1}
            self.ubw.extend(self.ubx)

            # Restricción de continuidad (disparo) Xk_end_k = Xk_next
            self.g.append(Xk_end_k - Xk_next)
            self.lbg.extend([-1e-9] * self.nx) # Igualdad numérica
            self.ubg.extend([+1e-9] * self.nx)

            # --- Calcular costo del paso k ---
            # Salida Ck = [F_S_k, T_{k+1}] = h(X_{k+1}, U_k)
            # OJO: La salida depende de Xk_next y Uk_k
            Ck = self.output_func_nmpc(Xk_next, Uk_k)
            sp_k = self.sp_sym[:, k]      # Setpoint [F_S_sp, T_sp] en paso k
            error_k = Ck - sp_k
            J += ca.mtimes([error_k.T, self.Q, error_k]) # Costo de error de salida (stage cost)
            J += ca.mtimes([delta_u.T, self.W, delta_u]) # Costo de movimiento de entrada

            # Actualizar para el siguiente paso
            Xk_iter = Xk_next # El estado inicial del prox intervalo es la variable X_{k+1}
            Uk_prev = Uk_k

        # --- Crear el solver NLP ---
        nlp_dict = {
            'f': J,
            'x': ca.vertcat(*self.w), # Vector de variables de decisión
            'g': ca.vertcat(*self.g), # Vector de restricciones
            'p': p_nlp                # Vector de parámetros (x0, sp_traj, uprev)
        }

        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6, 'ipopt.acceptable_tol': 1e-5,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

        # Guardar índices para extraer resultados
        self._prepare_indices()
        # Inicializar w0 (se llenará mejor en la primera llamada a solve)
        self.w0 = np.zeros(ca.vertcat(*self.w).shape[0])

    def _prepare_indices(self):
        """Calcula los índices de inicio para U, Xc, X en w."""
        self.indices = {'U': [], 'Xc': [], 'X': []}
        offset = 0
        for k in range(self.N):
            # Uk
            self.indices['U'].append(offset)
            offset += self.nu
            # Xc_k
            self.indices['Xc'].append(offset)
            offset += self.nx * self.m
            # X_{k+1}
            self.indices['X'].append(offset)
            offset += self.nx
        self.dim_w = offset
        print(f"[NMPC Info] Dimensión variables NLP (w): {self.dim_w}")


    def solve(self, x_current, sp_trajectory, u_previous):
        """Resuelve el NMPC para un estado y setpoints dados."""
        # Preparar parámetros p = [x0, vec(sp_traj), uprev]
        p_val = np.concatenate([x_current, sp_trajectory.flatten('F'), u_previous])

        # Estimación inicial w0 (usa solución anterior o heurística simple)
        if np.all(self.w0 == 0) or len(self.w0) != self.dim_w: # Si es la primera vez o hubo error
             # Heurística: mantener u constante, simular x linealmente?
             # O simplemente dejar que IPOPT empiece de cero la primera vez.
             # Vamos a inicializar con u_previous y x_current repetidos
             w0_guess = []
             x_guess = x_current
             u_guess = u_previous
             for k in range(self.N):
                 w0_guess.extend(u_guess) # U_k
                 w0_guess.extend(np.tile(x_guess, self.m)) # Xc_k
                 # Simplificación: X_{k+1} = Xk (se ajustará por solver)
                 w0_guess.extend(x_guess) # X_{k+1}
             # Validar longitud antes de asignar
             if len(w0_guess) == self.dim_w:
                 current_w0 = np.array(w0_guess)
                 print("[NMPC Info] Usando guess heurístico para w0.")
             else:
                 print(f"[NMPC Warning] Dimensiones de guess w0 ({len(w0_guess)}) no coinciden con dim_w ({self.dim_w}). Usando ceros.")
                 current_w0 = np.zeros(self.dim_w)
        else:
             print("[NMPC Info] Usando warm start (w0 de la iteración anterior).")
             current_w0 = self.w0 # Usar solución anterior

        # Resolver NLP
        try:
            sol = self.solver(x0=current_w0, lbx=self.lbw, ubx=self.ubw,
                              lbg=self.lbg, ubg=self.ubg, p=p_val)
            w_opt = sol['x'].full().flatten()
            sol_stats = self.solver.stats()

            if not sol_stats['success']:
                print(f"¡ADVERTENCIA NMPC: Solver no convergió! Estado: {sol_stats.get('return_status', 'N/A')}")
                 # Intentar resolver desde cero como fallback?
                try:
                    print("[NMPC Warning] Intentando resolver desde cero...")
                    sol = self.solver(x0=np.zeros(self.dim_w), lbx=self.lbw, ubx=self.ubw,
                                      lbg=self.lbg, ubg=self.ubg, p=p_val)
                    w_opt = sol['x'].full().flatten()
                    sol_stats = self.solver.stats()
                    if not sol_stats['success']:
                         print(f"[NMPC ERROR] Solver falló de nuevo desde cero. Estado: {sol_stats.get('return_status', 'N/A')}")
                         return u_previous, sol_stats, None, None # Devolver u anterior
                except Exception as e_fallback:
                    print(f"[NMPC ERROR] Excepción al intentar resolver desde cero: {e_fallback}")
                    return u_previous, {'success': False, 'return_status': 'FallbackError'}, None, None

            # Actualizar w0 para el próximo warm start
            self.w0 = w_opt

            # Extraer resultados
            u_optimal_sequence = np.zeros((self.nu, self.N))
            x_predicted_sequence = np.zeros((self.nx, self.N + 1))
            x_predicted_sequence[:, 0] = x_current # El primer estado es el real

            for k in range(self.N):
                u_optimal_sequence[:, k] = w_opt[self.indices['U'][k] : self.indices['U'][k] + self.nu]
                x_predicted_sequence[:, k+1] = w_opt[self.indices['X'][k] : self.indices['X'][k] + self.nx]

            u_apply = u_optimal_sequence[:, 0] # Primera acción de control

            return u_apply, sol_stats, x_predicted_sequence, u_optimal_sequence

        except Exception as e:
            print(f"[NMPC ERROR] Excepción durante la llamada al solver: {e}")
            # Resetear w0 si hay error grave
            self.w0 = np.zeros(self.dim_w)
            return u_previous, {'success': False, 'return_status': f'Exception: {e}'}, None, None

# ---------------------------------------------------
# 3. Simulación Planta (Usando ODE unificada)
# ---------------------------------------------------
def simulate_plant_unified(x0, u, dt, ode_func):
    """Simula un paso de la planta real (integrando el modelo unificado)."""
    # u = [F_S, Q_j]
    ode_sys = lambda t, x: ode_func(x, u).full().flatten()
    sol = solve_ivp(ode_sys, [0, dt], x0, method='RK45', rtol=1e-6, atol=1e-9)
    return sol.y[:, -1]

# ---------------------------------------------------
# 4. Bucle Principal de Simulación NMPC
# ---------------------------------------------------
if __name__ == "__main__":
    # --- Configuración Simulación ---
    t_start_nmpc = 5.0 # Empezar NMPC después de la fase batch RTO
    t_final = 24.0
    dt_nmpc = 0.05      # Time de muestreo NMPC (h)
    n_steps = int((t_final - t_start_nmpc) / dt_nmpc)

    # --- Cargar Perfil RTO ---
    try:
        rto_data = np.load("rto_optimal_feed_profile.npz")
        t_rto = rto_data['t_profile']
        F_S_rto_profile = rto_data['F_S_profile']
        print("[INFO] Perfil RTO cargado correctamente.")
        # Crear función interpoladora para F_S_ref(t)
        # Usar interpolación 'zero' o 'previous' para mantener valor constante entre puntos RTO
        F_S_ref_interp = interp1d(t_rto, F_S_rto_profile, kind='previous',
                                  bounds_error=False, fill_value=(F_S_rto_profile[0], F_S_rto_profile[-1]))
    except FileNotFoundError:
        print("[ERROR] Archivo 'rto_optimal_feed_profile.npz' no encontrado. Ejecuta RTO primero.")
        exit()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar o interpolar el perfil RTO: {e}")
        exit()

    # --- Parámetros NMPC ---
    N = 10          # Horizonte de predicción
    M = 4           # Horizonte de control
    # Pesos Q para error en [F_S, T] y W para delta en [F_S, Q_j]
    # Dar más peso al seguimiento de F_S si es crítico. Ajustar según necesidad.
    # Q_FS: Penaliza (F_S_medido - F_S_ref_RTO)^2
    # Q_T: Penaliza (T_medido - T_ref)^2
    # W_dFS: Penaliza (F_S_k - F_S_{k-1})^2
    # W_dQj: Penaliza (Q_j_k - Q_j_{k-1})^2
    Q_weights = [100.0, 0.0]   # Peso mayor en seguir F_S_ref [Q_FS, Q_T]
    W_weights = [1.0, 1e-8]   # Penalizar cambios en F_S, menos en Q_j [W_dFS, W_dQj]

    # --- Crear Instancia NMPC ---
    print("Construyendo NMPC con modelo unificado...")
    nmpc = NMPCBioreactorUnified(dt_nmpc, N, M, Q_weights, W_weights, ode_func, output_func_nmpc,
                                 x_sym, u_sym, c_sym, params,
                                 lbx, ubx, lbu, ubu, lbdu, ubdu)
    print("NMPC construido.")

    # --- Simulación ---
    # Estado inicial: Usar el estado final del batch RTO
    # (Asumiendo que RTO se ejecutó justo antes y guardó x_after_batch)
    # Necesitamos recalcular x_after_batch aquí o cargarlo si RTO no se corre en el mismo script
    print("Calculando estado inicial NMPC (final de fase batch RTO)...")
    # Re-ejecutar integración batch RTO para obtener x0_nmpc consistente
    # Usar la misma función ODE que usa RTO (Qj=0)
    ode_rto_internal_lambda = lambda x, u_rto: ode_func(x, ca.vertcat(u_rto, 0.0))
    ode_rto_func_temp = ca.Function('ode_rto_temp', [x_sym, u_sym[0]],
                                     [ode_rto_internal_lambda(x_sym, u_sym[0])],
                                     ['x', 'u_rto'], ['dx'])
    batch_integrator_nmpc = ca.integrator(
        "batch_integrator_nmpc", "idas",
        {"x": x_sym, "p": u_sym[0], "ode": ode_rto_func_temp(x_sym, u_sym[0])},
        {"t0": 0, "tf": t_start_nmpc} # t_batch = t_start_nmpc
    )
    res_batch_nmpc = batch_integrator_nmpc(x0=x0_unified, p=0.0)
    x_current = res_batch_nmpc['xf'].full().flatten()
    print(f"Estado inicial NMPC (t={t_start_nmpc}): {x_current}")
    print(f"   ({', '.join([f'{n}={v:.3f}' for n, v in zip(x_names, x_current)])})")


    # Entrada inicial u = [F_S, Q_j]
    u_previous = np.array([0.0, 0.0]) # Empezar con cero

    # Setpoint de Temperatura (ejemplo: constante)
    T_setpoint_const = 308.0 # K

    # Historial
    t_history_nmpc = np.linspace(t_start_nmpc, t_final, n_steps + 1)
    x_history = np.zeros((nx, n_steps + 1))
    u_history = np.zeros((nu, n_steps)) # Guarda [F_S_cmd, Q_j]
    c_history = np.zeros((nc, n_steps + 1)) # Guarda [F_S_med, T_med]
    sp_history = np.zeros((nc, n_steps + 1)) # Guarda [F_S_ref, T_ref]

    x_history[:, 0] = x_current
    # Calculate salida inicial c = h(x0, u_prev)
    c_initial = output_func_nmpc(x_current, u_previous).full().flatten()
    c_history[:, 0] = c_initial
    # SP inicial
    F_S_ref_0 = F_S_ref_interp(t_start_nmpc)
    sp_history[:, 0] = [F_S_ref_0, T_setpoint_const]


    print("\nIniciando simulación NMPC...")
    for k in range(n_steps):
        t_current_loop = t_history_nmpc[k]

        # 1. Construir trayectoria de Setpoints para el horizonte N
        sp_traj = np.zeros((nc, N))
        for i in range(N):
            t_pred = t_current_loop + i * dt_nmpc
            sp_traj[0, i] = F_S_ref_interp(t_pred) # F_S_ref(t) del RTO interpolado
            sp_traj[1, i] = T_setpoint_const       # T_ref constante

        # Guardar el primer SP para graficar
        if k == 0: sp_history[:, 1] = sp_traj[:, 0] # Correction índice

        # 2. Resolver NMPC
        u_optimal, stats, x_pred_opt, u_pred_opt = nmpc.solve(x_current, sp_traj, u_previous)

        # 3. Aplicar la acción de control (con chequeo de éxito)
        if not stats['success'] or u_optimal is None:
            print(f"-> NMPC Falló en paso {k}. Usando entrada anterior.")
            u_apply = u_previous
        else:
            u_apply = u_optimal
            # Aplicar saturación y límites delta_u (aunque el solver debería hacerlo)
            u_apply = np.clip(u_apply, lbu, ubu)
            delta_u_applied = u_apply - u_previous
            delta_u_applied = np.clip(delta_u_applied, lbdu, ubdu)
            u_apply = u_previous + delta_u_applied
            u_apply = np.clip(u_apply, lbu, ubu) # Re-aplicar saturación

        # 4. Simular la planta con u_apply = [F_S_cmd, Q_j]
        x_next = simulate_plant_unified(x_current, u_apply, dt_nmpc, ode_func)

        # 5. Actualizar estado y guardar historial
        x_current = x_next
        u_previous = u_apply # Actualizar entrada anterior para delta_u

        x_history[:, k+1] = x_current
        u_history[:, k] = u_apply
        # Calculate salida medida c = h(x_k+1, u_k)
        c_measured = output_func_nmpc(x_current, u_apply).full().flatten()
        c_history[:, k+1] = c_measured
        # Guardar SP del siguiente paso (primer elemento de la trayectoria usada)
        sp_history[:, k+1] = sp_traj[:, 0]

        if (k + 1) % 10 == 0:
            print(f"Paso {k+1}/{n_steps} | t={t_history_nmpc[k+1]:.1f}h | "
                  f"X={x_current[0]:.3f} T={x_current[5]:.2f}K | "
                  f"F_S={u_apply[0]:.3f}(Ref:{sp_traj[0,0]:.3f}) Qj={u_apply[1]:.1f} J/h")

    print("Simulación NMPC completada.")

    # --- Graficar Resultados NMPC ---
    plt.figure(figsize=(14, 12))
    plt.suptitle("Simulación NMPC con Modelo Unificado (Tracking F_S de RTO y T)", fontsize=16)

    # Estados (seleccionados)
    plt.subplot(4, 2, 1)
    plt.plot(t_history_nmpc, x_history[x_names.index('X'), :], label='X (Biomasa)')
    plt.ylabel('g/L')
    plt.legend()
    plt.grid(True)
    plt.title('Biomasa')

    plt.subplot(4, 2, 3)
    plt.plot(t_history_nmpc, x_history[x_names.index('S'), :], label='S (Sustrato)')
    plt.ylabel('g/L')
    plt.legend()
    plt.grid(True)
    plt.title('Sustrato')

    plt.subplot(4, 2, 5)
    plt.plot(t_history_nmpc, x_history[x_names.index('P'), :], label='P (Producto)')
    plt.ylabel('g/L')
    plt.legend()
    plt.grid(True)
    plt.title('Producto')

    plt.subplot(4, 2, 7)
    plt.plot(t_history_nmpc, x_history[x_names.index('V'), :], label='V (Volumen)')
    plt.ylabel('L')
    plt.xlabel('Tiempo (h)')
    plt.legend()
    plt.grid(True)
    plt.title('Volumen')


    # Entradas (MVs) y Salidas Controladas (CVs)
    plt.subplot(4, 2, 2)
    plt.step(t_history_nmpc[:-1], u_history[0, :], where='post', label='$F_S$ Comando (MV)')
    plt.plot(t_history_nmpc, sp_history[0, :], 'r--', label='$F_S$ Referencia (RTO)')
    # plt.plot(t_history_nmpc, c_history[0, :], 'g-.', label='F_S Medido (CV)') # En este caso MV=CV
    plt.ylabel('F_S (L/h)')
    plt.legend()
    plt.grid(True)
    plt.title('Control Flujo Alimentación')

    plt.subplot(4, 2, 4)
    plt.plot(t_history_nmpc, c_history[1, :], label='T Medido (CV)')
    plt.plot(t_history_nmpc, sp_history[1, :], 'r--', label='T Referencia')
    plt.ylabel('Temperatura (K)')
    plt.legend()
    plt.grid(True)
    plt.title('Control Temperatura')


    plt.subplot(4, 2, 6)
    # Graficar Qj en Watts (dividir por 3600) para comparar con NMPC original
    plt.step(t_history_nmpc[:-1], u_history[1, :] / 3600.0, where='post', label='$Q_j$ (MV)')
    plt.ylabel('$Q_j$ (W)')
    plt.legend()
    plt.grid(True)
    plt.title('Entrada Calor')


    plt.subplot(4, 2, 8)
    plt.plot(t_history_nmpc, x_history[x_names.index('O'), :], label='O (Oxígeno)')
    plt.ylabel('O (g/L)')
    plt.xlabel('Tiempo (h)')
    plt.legend()
    plt.grid(True)
    plt.title('Oxígeno Disuelto')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()