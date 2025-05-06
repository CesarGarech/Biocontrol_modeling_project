import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # Para interpolar la trayectoria de referencia

# ====================================================
# 1) Definición de la función ODE BIO (Igual que en RTO)
# ====================================================
def odefun(x, u):
    """
    Ecuaciones diferenciales Fed-Batch con O=constante.
    - Divisiones con fmax(V, epsilon) para evitar 1/0.
    """
    # Parámetros (iguales al RTO)
    mu_max = 0.6
    Ks     = 0.2
    Ko     = 0.01
    Yxs    = 0.5
    Yxo    = 0.1
    Yps    = 0.3
    kLa    = 180.0   # (no relevante si dO/dt=0)
    O_sat  = 0.18    # (no relevante si dO/dt=0)
    Sf     = 500.0
    V_max  = 2.0     # Restricción importante

    # Extraer variables de estado
    X_ = x[0]
    S_ = x[1]
    P_ = x[2]
    O_ = x[3]
    V_ = x[4]

    # Salvaguarda para V (evitar división por 0 si V es muy pequeño)
    eps = 1e-8
    V_eff = ca.fmax(V_, eps)

    # Saturación de flujo si V >= V_max (importante en NMPC)
    # NOTA: Esto es una restricción física. El NMPC también debe tener
    # esta restricción en sus predicciones.
    u_sat = ca.if_else(V_ >= V_max, 0.0, u)

    # Tasa de crecimiento
    mu = mu_max * (S_/(Ks + S_)) * (O_/(Ko + O_))
    # Tasa de dilución
    D = u_sat / V_eff # Usar V_eff y u_sat

    dX = mu*X_ - D*X_
    dS = -mu*X_/Yxs + D*(Sf - S_)
    dP = Yps*mu*X_ - D*P_
    dO = 0.0 # Oxígeno constante
    dV = u_sat # El volumen aumenta según el flujo real

    return ca.vertcat(dX, dS, dP, dO, dV)

# ====================================================
# 2) Parámetros del proceso y NMPC
# ====================================================
# Tiempos del proceso original
t_batch = 5.0
t_total = 24.0
tf_fb = t_total - t_batch # Duración de la fase fed-batch

# Condiciones iniciales (iguales al RTO)
X0, S0, P0, O0, V0 = 1.0, 20.0, 0.0, 0.08, 0.2
x0_np = np.array([X0, S0, P0, O0, V0])

# Restricciones (iguales al RTO)
F_min = 0.0
F_max = 0.3
S_max = 30.0
V_max = 2.0
nx = 5 # Número de estados
nu = 1 # Número de controles (F)

# Parámetros del NMPC
T_nmpc = 0.25 # Tiempo de muestreo del NMPC (h) - ¡Más corto que dt_fb del RTO!
N_p = 10      # Horizonte de predicción (número de pasos de T_nmpc)
N_sim_steps = int(tf_fb / T_nmpc) # Número de pasos de simulación NMPC

print(f"[INFO] NMPC configurado con:")
print(f"  - Tiempo de muestreo (T_nmpc): {T_nmpc} h")
print(f"  - Horizonte de predicción (N_p): {N_p} pasos ({N_p*T_nmpc} h)")
print(f"  - Pasos de simulación: {N_sim_steps}")

# ====================================================
# 3) Fase BATCH (Simulación inicial - igual que en RTO)
# ====================================================
x_sym = ca.MX.sym("x", nx)
u_sym = ca.MX.sym("u", nu)
ode_expr = odefun(x_sym, u_sym)

batch_integrator = ca.integrator(
    "batch_integrator","idas",
    {"x":x_sym, "p":u_sym, "ode":ode_expr},
    {"t0":0, "tf":t_batch}
)

res_batch = batch_integrator(x0=x0_np, p=0.0)
x_start_nmpc = np.array(res_batch['xf']).flatten()
print(f"[INFO] Estado tras fase batch (inicio NMPC en t={t_batch}): {x_start_nmpc}")

# ====================================================
# 4) Cargar y preparar la trayectoria de referencia (del RTO)
# ====================================================
try:
    rto_data = np.load("rto_original_feed_profile.npz")
    t_rto_profile = rto_data['t_profile'] # Tiempos donde F cambia
    F_rto_profile = rto_data['F_S_profile'] # Valores de F óptimos
    print("[INFO] Perfil de referencia RTO cargado desde 'rto_original_feed_profile.npz'")

    # Necesitamos simular la trayectoria de ESTADO del RTO para usarla como referencia
    # Reutilizamos el código de simulación del script RTO

    # Integrador para simulación fina (similar al paso 10b del RTO)
    dt_fine_sim = 0.05 # dt pequeño para simulación precisa de referencia
    # Integrador para simulación fina (similar al paso 10b del RTO)
    dt_fine_sim = 0.05 # dt pequeño para simulación precisa de referencia
    sim_integrator = ca.integrator(
        "sim_integrator", "rk",
        {"x": x_sym, "p": u_sym, "ode": ode_expr},
        {"tf": dt_fine_sim} # Especificar el tamaño del paso tf
    )
    # Crear perfil de F completo para la simulación fina
    # Necesitamos un valor de F para cada dt_fine_sim
    t_sim_ref = np.arange(t_batch, t_total + dt_fine_sim/2, dt_fine_sim) # Tiempos finos
    F_sim_ref_func = interp1d(t_rto_profile, F_rto_profile, kind='previous', # 'previous' para mantener F constante en el intervalo
                              bounds_error=False, fill_value=(F_rto_profile[0], F_rto_profile[-1]))
    F_sim_ref = F_sim_ref_func(t_sim_ref)

    # Simular la trayectoria de estado de referencia
    x_ref_traj = [x_start_nmpc]
    xk_ref = x_start_nmpc.copy()
    for i in range(len(t_sim_ref) - 1):
        F_now = F_sim_ref[i]
        # Aplicar restricción de volumen también en la simulación de referencia
        if xk_ref[4] >= V_max:
            F_now = 0.0
        res_ref = sim_integrator(x0=xk_ref, p=F_now)
        xk_ref = np.array(res_ref["xf"]).flatten()
        x_ref_traj.append(xk_ref)
    x_ref_traj = np.array(x_ref_traj)

    # Crear funciones de interpolación para la referencia (estados y control)
    interp_x_ref = interp1d(t_sim_ref, x_ref_traj, axis=0, kind='linear', bounds_error=False, fill_value=(x_ref_traj[0,:], x_ref_traj[-1,:]))
    interp_F_ref = interp1d(t_sim_ref, F_sim_ref, kind='previous', bounds_error=False, fill_value=(F_sim_ref[0], F_sim_ref[-1])) # Usa el mismo F que la simulación

    print("[INFO] Trayectoria de referencia (estados y control) simulada e interpoladores creados.")

except FileNotFoundError:
    print("[ERROR] No se encontró el archivo 'rto_original_feed_profile.npz'.")
    print("        Asegúrate de ejecutar primero el script dRTO para generarlo.")
    exit()
except Exception as e:
    print(f"[ERROR] Ocurrió un error al cargar o procesar el perfil RTO: {e}")
    exit()


# ====================================================
# 5) Configuración del problema de optimización NMPC
# ====================================================
opti_nmpc = ca.Opti()

# ---- Variables de decisión ----
U_nmpc = opti_nmpc.variable(nu, N_p) # Secuencia de control F[k], F[k+1], ..., F[k+Np-1]
X_pred = opti_nmpc.variable(nx, N_p + 1) # Estados predichos X[k|k]...X[k+Np|k]

# ---- Parámetros (cambian en cada paso de NMPC) ----
X0_nmpc = opti_nmpc.parameter(nx)     # Estado actual medido/estimado x[k]
X_ref_nmpc = opti_nmpc.parameter(nx, N_p + 1) # Trayectoria de estado de referencia para el horizonte
F_ref_nmpc = opti_nmpc.parameter(nu, N_p)     # Trayectoria de control de referencia

# ---- Función objetivo ----
# Ponderaciones (¡IMPORTANTE - requiere ajuste!)
Q_mat = np.diag([0.5, 0.5, 5.0, 0.0, 1.0]) # Ponderación estados (X, S, P, O, V) - Mayor peso en P y V
R_mat = np.diag([0.1])                     # Ponderación control F
# S_mat = np.diag([0.01])                    # Ponderación cambio de control dF (opcional)

objective = 0
for k in range(N_p):
    # Error de seguimiento del estado
    state_error = X_pred[:, k+1] - X_ref_nmpc[:, k+1]
    objective += ca.mtimes([state_error.T, Q_mat, state_error])

    # Error de seguimiento del control (opcional pero útil)
    control_error = U_nmpc[:, k] - F_ref_nmpc[:, k]
    objective += ca.mtimes([control_error.T, R_mat, control_error])

    # Penalización del cambio de control (opcional, para suavizar)
    # if k < N_p - 1:
    #    control_change = U_nmpc[:, k+1] - U_nmpc[:, k]
    #    objective += ca.mtimes([control_change.T, S_mat, control_change])

opti_nmpc.minimize(objective)

# ---- Restricciones ----
# 1) Dinámica del modelo (usando RK4 explícito para la predicción)
dt_nmpc_ca = opti_nmpc.parameter(1) # Pasar T_nmpc como parámetro
opti_nmpc.set_value(dt_nmpc_ca, T_nmpc)

for k in range(N_p):
    # Integración RK4 de un paso
    k1 = odefun(X_pred[:, k],       U_nmpc[:, k])
    k2 = odefun(X_pred[:, k] + dt_nmpc_ca / 2.0 * k1, U_nmpc[:, k])
    k3 = odefun(X_pred[:, k] + dt_nmpc_ca / 2.0 * k2, U_nmpc[:, k])
    k4 = odefun(X_pred[:, k] + dt_nmpc_ca * k3,     U_nmpc[:, k])
    x_next = X_pred[:, k] + dt_nmpc_ca / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti_nmpc.subject_to(X_pred[:, k+1] == x_next) # Restricción de igualdad del modelo

# 2) Restricción inicial (el primer estado predicho es el actual)
opti_nmpc.subject_to(X_pred[:, 0] == X0_nmpc)

# 3) Límites del control
opti_nmpc.subject_to(opti_nmpc.bounded(F_min, U_nmpc, F_max))

# 4) Límites de los estados predichos
for k in range(1, N_p + 1): # Empezar desde k=1 porque X_pred[:,0] es el estado actual
    opti_nmpc.subject_to(X_pred[0, k] >= 0) # X >= 0
    opti_nmpc.subject_to(X_pred[1, k] >= 0) # S >= 0
    opti_nmpc.subject_to(X_pred[1, k] <= S_max) # S <= S_max
    opti_nmpc.subject_to(X_pred[2, k] >= 0) # P >= 0
    opti_nmpc.subject_to(X_pred[4, k] >= 0) # V >= 0
    opti_nmpc.subject_to(X_pred[4, k] <= V_max) # V <= V_max

# ---- Configuración del Solver ----
p_opts_nmpc = {}
s_opts_nmpc = {
    "max_iter": 3000, # Puede necesitar ajuste
    "print_level": 0, # Silencioso dentro del bucle
    "sb": 'yes', # Evita mensaje de licencia
    "tol": 1e-6,              # Relajar tolerancia de optimalidad (default: ~1e-8)
    "constr_viol_tol": 1e-5,  # Relajar tolerancia de violación de restricciones (default: 1e-4)
    "warm_start_init_point": "yes", # Importante para velocidad
    # "mu_init": 1e-2, # Puede ayudar convergencia
    # "nlp_scaling_method": "gradient-based"
}
opti_nmpc.solver("ipopt", p_opts_nmpc, s_opts_nmpc)
print("[INFO] Optimizador NMPC ('ipopt') configurado.")

# ====================================================
# 6) Bucle de simulación NMPC
# ====================================================
# Historial para guardar resultados
t_history = [t_batch]
x_history = [x_start_nmpc]
u_history = [] # Guardará el control aplicado en cada paso

# Estado actual del sistema simulado
x_current = x_start_nmpc.copy()

# Guesses iniciales para el optimizador NMPC
u_guess = np.full((nu, N_p), F_min) # Empezar con flujo mín imo o un valor razonable
x_guess = np.tile(x_current, (N_p + 1, 1)).T # Replicar estado actual

# Integrador para simular la planta real entre pasos NMPC
plant_integrator = ca.integrator(
    "plant_integrator", "rk", # RK4 es suficiente para T_nmpc
    {"x": x_sym, "p": u_sym, "ode": ode_expr},
    {"tf": T_nmpc} # Especificar el tamaño del paso tf (T_nmpc)
)

print("\n[INFO] Iniciando simulación NMPC...")
for i in range(N_sim_steps):
    current_time = t_batch + i * T_nmpc
    print(f"  Sim NMPC - Paso {i+1}/{N_sim_steps}, Tiempo: {current_time:.2f} h", end='\r')

    # 1. Obtener la referencia para el horizonte actual
    t_horizon = current_time + np.arange(N_p + 1) * T_nmpc
    x_ref_horizon = interp_x_ref(t_horizon)
    f_ref_horizon = interp_F_ref(t_horizon[:-1]) # F es para los intervalos

    # 2. Establecer los parámetros del NMPC
    opti_nmpc.set_value(X0_nmpc, x_current)
    opti_nmpc.set_value(X_ref_nmpc, x_ref_horizon.T) # Transponer para shape (nx, Np+1)
    opti_nmpc.set_value(F_ref_nmpc, f_ref_horizon.reshape(nu, N_p)) # Shape (nu, Np)

    # 3. Establecer guesses iniciales (warm start)
    opti_nmpc.set_initial(X_pred, x_guess)
    opti_nmpc.set_initial(U_nmpc, u_guess)

    # 4. Resolver el problema NMPC
    try:
        sol_nmpc = opti_nmpc.solve()

        # Recuperar la secuencia de control óptima y los estados predichos
        u_optimal_sequence = sol_nmpc.value(U_nmpc)
        x_predicted_sequence = sol_nmpc.value(X_pred)

        # Aplicar SOLO el primer control de la secuencia (que ahora es 1D)
        u_apply = u_optimal_sequence[0]

        # Preparar guesses para la siguiente iteración (shift para array 1D)
        u_guess_1d = np.roll(u_optimal_sequence, -1) # Shift en 1D
        u_guess_1d[-1] = u_optimal_sequence[-1]     # Repetir el último elemento
        # Asegurar que u_guess tenga la forma (nu, N_p) para set_initial
        u_guess = u_guess_1d.reshape(nu, N_p)
        x_guess = np.roll(x_predicted_sequence, -1, axis=1)
        x_guess[:, -1] = x_predicted_sequence[:, -1] # Repetir el último

    except RuntimeError as e:
        print(f"\n[ERROR] Falló la optimización NMPC en el paso {i} (t={current_time:.2f}): {e}")
        print("         Aplicando control del paso anterior o F=0.")
        # Estrategia de fallback: usar control anterior o F=0
        if len(u_history) > 0:
            u_apply = u_history[-1]
        else:
            u_apply = 0.0
        # Mantener los guesses anteriores
        u_guess = np.roll(u_guess, -1, axis=1)
        u_guess[:, -1] = u_guess[:, -2]
        x_guess = np.roll(x_guess, -1, axis=1)
        x_guess[:, -1] = x_guess[:, -2]
        # Se podría intentar resolver de nuevo con otro guess o parar

    # Asegurar que el control aplicado respete V_max en el estado ACTUAL
    if x_current[4] >= V_max:
        u_apply = 0.0

    # 5. Aplicar el control al sistema simulado (planta)
    res_plant = plant_integrator(x0=x_current, p=u_apply)
    x_next = np.array(res_plant['xf']).flatten()

    # 6. Guardar historial
    t_history.append(current_time + T_nmpc)
    x_history.append(x_next)
    u_history.append(u_apply)

    # 7. Actualizar estado actual
    x_current = x_next

print("\n[INFO] Simulación NMPC completada.")

# Convertir historial a arrays numpy
t_history = np.array(t_history)
x_history = np.array(x_history)
u_history = np.array(u_history)

# Ajustar el tiempo del control para graficar (u_history[i] se aplica en t_history[i])
t_control_history = t_history[:-1]


# ====================================================
# 7) Graficar resultados NMPC vs Referencia RTO
# ====================================================
print("[INFO] Generando gráficas comparativas...")

# Recrear la trayectoria de referencia completa (batch + fed-batch) para comparación
t_ref_full = t_sim_ref # Tiempos de la simulación de referencia (ya incluye t_batch)
x_ref_full = x_ref_traj # Estados de la simulación de referencia
F_ref_full = interp_F_ref(t_ref_full) # Control de la simulación de referencia

# Añadir la fase batch a los resultados NMPC para graficar desde t=0
t_batch_plot = np.linspace(0, t_batch, 50) # Puntos para graficar la fase batch
batch_plot_int = ca.integrator("batch_plot_int","idas",
                              {"x":x_sym,"p":u_sym,"ode":ode_expr},
                              {"t0":0, "tf":t_batch/49.0})
xbatch_traj_plot = [x0_np]
xk_ = x0_np.copy()
for _ in range(49):
    res_ = batch_plot_int(x0=xk_, p=0.0)
    xk_ = np.array(res_["xf"]).flatten()
    xbatch_traj_plot.append(xk_)
xbatch_traj_plot = np.array(xbatch_traj_plot)

# Combinar batch y NMPC para la gráfica completa
t_plot_nmpc = np.concatenate([t_batch_plot, t_history[1:]]) # t_history[0] es t_batch
x_plot_nmpc = np.vstack([xbatch_traj_plot, x_history[1:,:]]) # x_history[0] es x_start_nmpc
u_plot_nmpc = np.concatenate([np.zeros(len(t_batch_plot)), u_history]) # F=0 en batch
t_u_plot_nmpc = np.concatenate([t_batch_plot, t_control_history]) # Tiempos para F

# Graficas
fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axs = axs.ravel()

# Flujo F(t)
axs[0].plot(t_ref_full, F_ref_full, '--', label='Referencia RTO', linewidth=2, drawstyle='steps-post')
axs[0].step(t_u_plot_nmpc, u_plot_nmpc, where='post', label='NMPC Aplicado', linewidth=2)
axs[0].set_ylabel("F (L/h)")
axs[0].set_title("Flujo de Alimentación F(t)")
axs[0].legend()
axs[0].grid(True)
axs[0].axhline(F_max, color='r', linestyle=':', linewidth=1, label='F_max')
axs[0].axhline(F_min, color='r', linestyle=':', linewidth=1, label='F_min')

# Biomasa X(t)
axs[1].plot(t_ref_full, x_ref_full[:, 0], '--', label='Referencia RTO', linewidth=2)
axs[1].plot(t_plot_nmpc, x_plot_nmpc[:, 0], label='NMPC', linewidth=2)
axs[1].set_ylabel("X (g/L)")
axs[1].set_title("Biomasa X(t)")
axs[1].legend()
axs[1].grid(True)

# Sustrato S(t)
axs[2].plot(t_ref_full, x_ref_full[:, 1], '--', label='Referencia RTO', linewidth=2)
axs[2].plot(t_plot_nmpc, x_plot_nmpc[:, 1], label='NMPC', linewidth=2)
axs[2].axhline(S_max, color='r', linestyle=':', linewidth=1, label="S_max")
axs[2].set_ylabel("S (g/L)")
axs[2].set_title("Sustrato S(t)")
axs[2].legend()
axs[2].grid(True)

# Producto P(t)
axs[3].plot(t_ref_full, x_ref_full[:, 2], '--', label='Referencia RTO', linewidth=2)
axs[3].plot(t_plot_nmpc, x_plot_nmpc[:, 2], label='NMPC', linewidth=2)
axs[3].set_ylabel("P (g/L)")
axs[3].set_title("Producto P(t)")
axs[3].legend()
axs[3].grid(True)

# Volumen V(t)
axs[4].plot(t_ref_full, x_ref_full[:, 4], '--', label='Referencia RTO', linewidth=2)
axs[4].plot(t_plot_nmpc, x_plot_nmpc[:, 4], label='NMPC', linewidth=2)
axs[4].axhline(V_max, color='r', linestyle=':', linewidth=1, label="V_max")
axs[4].set_ylabel("V (L)")
axs[4].set_title("Volumen V(t)")
axs[4].set_xlabel("Tiempo (h)")
axs[4].legend()
axs[4].grid(True)

# Producto Total P*V(t)
PV_ref = x_ref_full[:, 2] * x_ref_full[:, 4]
PV_nmpc = x_plot_nmpc[:, 2] * x_plot_nmpc[:, 4]
axs[5].plot(t_ref_full, PV_ref, '--', label='Referencia RTO', linewidth=2)
axs[5].plot(t_plot_nmpc, PV_nmpc, label='NMPC', linewidth=2)
axs[5].set_ylabel("P*V (g)")
axs[5].set_title("Producto Total P(t)V(t)")
axs[5].set_xlabel("Tiempo (h)")
axs[5].legend()
axs[5].grid(True)


plt.tight_layout()
plt.show()

# Calcular producto final
PV_final_rto = PV_ref[-1]
PV_final_nmpc = PV_nmpc[-1]
print(f"\nProducto Total Final (P*V):")
print(f"  - Referencia RTO: {PV_final_rto:.4f} g")
print(f"  - NMPC:           {PV_final_nmpc:.4f} g")
print(f"  - Diferencia:     {PV_final_nmpc - PV_final_rto:.4f} g ({(PV_final_nmpc - PV_final_rto)/PV_final_rto*100:.2f} %)")