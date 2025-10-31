# RTO_casadi_unified.py
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from unified_bioreactor_model import get_unified_model, x0_unified # Importar modelo y x0

# ====================================================
# 1) Get Unified Model and Parameters
# ====================================================
ode_func, _, x_sym_model, u_sym_model, _, params, x_names, _, _ = get_unified_model(use_oxygen_dynamics=True)
nx = x_sym_model.shape[0]
nu = u_sym_model.shape[0]

u_rto_sym = ca.MX.sym("u_rto")
u_full_for_ode = ca.vertcat(u_rto_sym, 0.0)
ode_rto_func = ca.Function('ode_rto', [x_sym_model, u_rto_sym],
                           [ode_func(x_sym_model, u_full_for_ode)],
                           ['x', 'u_rto'], ['dx'])

# ====================================================
# 2) Coeficientes de colocación Radau (d=2) - Sin cambios
# ====================================================
def radau_coefficients(d):
    # ... (código sin cambios)
    if d == 2:
        C_mat = np.array([[-2.0, 2.0], [1.5, -4.5], [0.5, 2.5]])
        D_vec = np.array([0.0, 0.0, 1.0])
        return C_mat, D_vec
    else:
        raise NotImplementedError("Solo implementado para d=2.")

# ====================================================
# 3) Process parameters and initial conditions
# ====================================================
t_batch = 5.0
t_total = 24.0
n_fb_intervals = 20
dt_fb = (t_total - t_batch) / n_fb_intervals

F_min = 0.0
F_max = 0.3 # Mantener el límite original o usar params['F_max'] si existe
S_max = 80.0
V_max = params['V_max']
T_min = 290.0
T_max = 315.0
O_min = 0.0 # Límite inferior oxígeno
O_max = params['O_sat'] * 1.1 # Límite superior oxígeno (un poco margen)


x0_batch = x0_unified.copy()
print("[INFO] Condiciones iniciales para RTO:", x0_batch)

# ====================================================
# 4) Fase BATCH con F=0 (integración)
# ====================================================
batch_integrator = ca.integrator(
    "batch_integrator", "idas",
    {"x": x_sym_model, "p": u_rto_sym, "ode": ode_rto_func(x_sym_model, u_rto_sym)},
    {"t0": 0, "tf": t_batch}
)

try:
    res_batch = batch_integrator(x0=x0_batch, p=0.0)
    x_after_batch = np.array(res_batch['xf']).flatten()
    print("[INFO] Estado tras fase batch:", x_after_batch)
    print(f"       ({', '.join([f'{n}={v:.3f}' for n, v in zip(x_names, x_after_batch)])})")
except Exception as e:
     print(f"[ERROR] Falla en la integración de la fase Batch: {e}")
     print("Verifica los parámetros y condiciones iniciales del modelo.")
     exit()


# ====================================================
# 5) Formulación de la fase Fed-Batch con colocación
# ====================================================
opti = ca.Opti()

d = 2
C_radau, D_radau = radau_coefficients(d)

X_col = []
F_col = []

# Límites de estado ahora incluyen T y O explícitos
lbx_rto = [0.0] * nx # No negatividad general
ubx_rto = [ca.inf] * nx
lbx_rto[x_names.index('V')] = 0.0
ubx_rto[x_names.index('V')] = V_max
lbx_rto[x_names.index('S')] = 0.0
# ubx_rto[x_names.index('S')] = S_max # <-- Restricción en S puede ser problemática, probar quitarla
ubx_rto[x_names.index('S')] = ca.inf # <-- Relajar S_max temporalmente
lbx_rto[x_names.index('T')] = T_min
ubx_rto[x_names.index('T')] = T_max
lbx_rto[x_names.index('O')] = O_min
ubx_rto[x_names.index('O')] = O_max


for k in range(n_fb_intervals):
    row_states = []
    for j in range(d + 1):
        xk_j = opti.variable(nx)
        opti.subject_to(opti.bounded(lbx_rto, xk_j, ubx_rto))
        if k == 0 and j == 0:
            opti.subject_to(xk_j == x_after_batch)
            X_start_fb = xk_j
        row_states.append(xk_j)
    X_col.append(row_states)

    Fk = opti.variable()
    F_col.append(Fk)
    opti.subject_to(opti.bounded(F_min, Fk, F_max))

    # *** Se eliminó la restricción if_else Fk vs Vmax ***

# ====================================================
# 6) Ecuaciones de Colocación
# ====================================================
h = dt_fb
for k in range(n_fb_intervals):
    # Agrupar puntos de colocación para eficiencia
    Xk_matrix = ca.horzcat(*X_col[k]) # Matriz [nx x (d+1)]

    for j in range(1, d + 1):
        # Derivative estimada usando coeficientes C
        xp_j = Xk_matrix @ C_radau[:, j - 1]
        # Evaluar f(Xk_j, Fk)
        fkj = ode_rto_func(X_col[k][j], F_col[k])
        # Restricción de Colocación
        opti.subject_to(h * fkj == xp_j)

    # Continuidad entre intervalos
    # Estado al final del intervalo k usando coeficientes D
    Xk_end = Xk_matrix @ D_radau
    if k < n_fb_intervals - 1:
        opti.subject_to(Xk_end == X_col[k + 1][0])

# Estado final global
X_final_matrix = ca.horzcat(*X_col[n_fb_intervals-1])
X_final = X_final_matrix @ D_radau

P_final_idx = x_names.index('P')
V_final_idx = x_names.index('V')
P_final = X_final[P_final_idx]
V_final = X_final[V_final_idx]

# ====================================================
# 7) Función objetivo => maximizar (P_final * V_final)
# ====================================================
opti.minimize(-(P_final * V_final))
print(f"[INFO] Objetivo: Maximizar Producto Total (P*V) = X[{P_final_idx}]*X[{V_final_idx}]")

# ====================================================
# 8) Guesses iniciales (Mejorados)
# ====================================================
# Create a simple integrator to propagate the guess
integrator_guess = ca.integrator('int_guess', 'cvodes',
                                 {"x": x_sym_model, "p": u_rto_sym, "ode": ode_rto_func(x_sym_model, u_rto_sym)},
                                 {"tf": dt_fb / (d+1)}) # Integrar pasos pequeños dentro del intervalo

F_guess_val = (F_min + F_max) / 10 # Guess constante para F
x_guess_k = x_after_batch.copy() # Estado inicial para el guess

print("[INFO] Estableciendo guesses iniciales mejorados...")
for k in range(n_fb_intervals):
    opti.set_initial(F_col[k], F_guess_val)
    x_guess_k_j = x_guess_k.copy() # Estado al inicio del intervalo k
    for j in range(d + 1):
        if k == 0 and j == 0:
             opti.set_initial(X_col[k][j], x_guess_k_j) # Ya fijado por constraint, pero bueno tener guess
        else:
            # Establecer guess para el punto de colocación
            opti.set_initial(X_col[k][j], x_guess_k_j)
            # Propagar el estado para el siguiente punto de colocación (si no es el último)
            if j < d:
                try:
                    res_guess = integrator_guess(x0=x_guess_k_j, p=F_guess_val)
                    x_guess_k_j = res_guess['xf'].full().flatten()
                except:
                    # Si falla la integración del guess, mantener el anterior
                    pass
    # Actualizar el estado inicial del siguiente intervalo para el guess
    # (usando el último estado propagado en el intervalo actual)
    x_guess_k = x_guess_k_j.copy()


# ====================================================
# 9) Configurar y resolver
# ====================================================
p_opts = {"expand": True}
s_opts = {
    "max_iter": 3000,
    "print_level": 5,
    "linear_solver": "mumps", # *** Cambiado a mumps ***
    "mu_strategy": "adaptive",
    "tol": 1e-6, # Un poco más relajado
    "acceptable_tol": 1e-5, # Un poco más relajado
    "warm_start_init_point": "yes",
    # Options adicionales que a veces ayudan con Restoration Failed:
    "required_infeasibility_reduction": 0.8, # Exigir menos reducción por iteración
    "max_resto_iter": 100 # Permitir más iteraciones en restauración
}
opti.solver("ipopt", p_opts, s_opts)

try:
    sol = opti.solve()
    print("[INFO] ¡Solución RTO encontrada!")

    # Extraer resultados
    F_opt_values = np.array([sol.value(fk) for fk in F_col])
    X_final_opt = sol.value(X_final)
    P_final_opt = X_final_opt[P_final_idx]
    V_final_opt = X_final_opt[V_final_idx]
    Producto_Total_opt = P_final_opt * V_final_opt

    print("\n--- Resultados RTO ---")
    print(f"F_opt (primeros 10): {F_opt_values[:10]}")
    print(f"Estado final óptimo: {X_final_opt}")
    print(f"   ({', '.join([f'{n}={v:.3f}' for n, v in zip(x_names, X_final_opt)])})")
    print(f"P_final = {P_final_opt:.4f} g/L")
    print(f"V_final = {V_final_opt:.4f} L")
    print(f"Producto Total Óptimo = {Producto_Total_opt:.4f} g")

    # Guardar el perfil de F_S para NMPC
    t_rto_intervals = np.linspace(t_batch, t_total, n_fb_intervals + 1)
    rto_output = {'t_profile': t_rto_intervals[:-1], 'F_S_profile': F_opt_values}
    np.savez("rto_optimal_feed_profile.npz", **rto_output)
    print("\n[INFO] Perfil óptimo de F_S guardado en 'rto_optimal_feed_profile.npz'")

except RuntimeError as e:
    print(f"[ERROR] RTO falló: {e}")
    print("\n--- Debug Info (Valores al Fallar) ---")
    try:
        print("Valores de F:")
        print([opti.debug.value(fk) for fk in F_col])
        print("\nValores de X (final de cada intervalo):")
        X_ends_debug = []
        for k in range(n_fb_intervals):
             Xk_mat_debug = ca.horzcat(*[opti.debug.value(x) for x in X_col[k]])
             Xk_end_debug = Xk_mat_debug @ D_radau
             X_ends_debug.append(Xk_end_debug)
        print(X_ends_debug)
        # Podrías añadir más variables al debug si es necesario
    except Exception as debug_e:
        print(f"No se pudieron obtener los valores de debug: {debug_e}")
    # raise # Opcional: volver a lanzar la excepción si quieres que el script se detenga

# ====================================================
# 10) Reconstruir y graficar trayectoria RTO (si tuvo éxito)
# ====================================================
if 'sol' in locals(): # Verify if solution exists
    # ... (Simulation and plotting code unchanged)...
    # a) Batch Simulation (more points for plot)
    N_batch_plot = 50
    t_batch_plot = np.linspace(0, t_batch, N_batch_plot)
    dt_b = t_batch / (N_batch_plot - 1)
    batch_plot_int = ca.integrator(
        "batch_plot_int", "idas",
        {"x": x_sym_model, "p": u_rto_sym, "ode": ode_rto_func(x_sym_model, u_rto_sym)},
        {"t0": 0, "tf": dt_b}
    )
    xbatch_traj = [x0_batch]
    xk_ = x0_batch.copy()
    for _ in range(N_batch_plot - 1):
        res_ = batch_plot_int(x0=xk_, p=0.0)
        xk_ = np.array(res_["xf"]).flatten()
        xbatch_traj.append(xk_)
    xbatch_traj = np.array(xbatch_traj)

    # b) Simulación Fed-batch con F_opt
    N_fb_plot = n_fb_intervals * 20 # Puntos por intervalo RTO
    t_fb_plot = np.linspace(t_batch, t_total, N_fb_plot)
    dt_fb_plot = (t_total - t_batch) / (N_fb_plot - 1)

    fb_plot_int = ca.integrator(
        "fb_plot_int", "idas",
         {"x": x_sym_model, "p": u_rto_sym, "ode": ode_rto_func(x_sym_model, u_rto_sym)},
        {"t0": 0, "tf": dt_fb_plot}
    )

    xfb_traj = [x_after_batch] # Empezar desde el final del batch
    xk_ = x_after_batch.copy()
    F_plot_fb = [] # Para graficar F usado en simulación

    for i in range(N_fb_plot - 1):
        t_current = t_batch + i * dt_fb_plot
        # Encontrar el intervalo RTO k correspondiente
        k_rto = int((t_current - t_batch) / dt_fb)
        k_rto = min(k_rto, n_fb_intervals - 1) # Ensure que esté en rango
        F_now = F_opt_values[k_rto]

        # *** APLICAR LÓGICA Vmax AQUÍ, POST-OPTIMIZACIÓN ***
        if xk_[V_final_idx] >= V_max - 1e-4:
            F_now = 0.0

        F_plot_fb.append(F_now) # Guardar F usado

        # Integrar un paso dt_fb_plot
        try:
            res_ = fb_plot_int(x0=xk_, p=F_now)
            xk_ = np.array(res_["xf"]).flatten()
        except Exception as sim_e:
            print(f"[ERROR] Falla en la simulación Fed-Batch en t={t_current:.2f}: {sim_e}")
            # Detener simulación si falla
            t_fb_plot = t_fb_plot[:i+1] # Acortar tiempo
            xfb_traj = np.array(xfb_traj)
            F_plot_fb.append(F_plot_fb[-1]) # Añadir último valor
            break # Salir del bucle de simulación
        xfb_traj.append(xk_)
    else: # Si el bucle terminó sin break
        xfb_traj = np.array(xfb_traj)
        F_plot_fb.append(F_plot_fb[-1]) # Añadir último valor de F para que coincida tamaño


    # Unir trayectorias
    t_full = np.concatenate([t_batch_plot, t_fb_plot])
    # Ensure que xfb_traj tenga datos antes de vstack
    if len(xfb_traj)>0:
        x_full = np.vstack([xbatch_traj, xfb_traj])
    else: # Si la simulación fb falló inmediatamente
        x_full = xbatch_traj

    # Construir perfil F completo para gráfico
    F_plot = np.concatenate([np.zeros(N_batch_plot), np.array(F_plot_fb)])


    # ====================================================
    # 11) Gráficas RTO
    # ====================================================
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.ravel()
    plt.suptitle("Simulación RTO con Modelo Unificado (Q_j=0) - Modificado", fontsize=16)

    # F(t)
    axs[0].plot(t_full, F_plot, label='F_S simulado (con Vmax check)')
    axs[0].step(t_rto_intervals[:-1], F_opt_values, where='post', linestyle='--', label='F_S óptimo (RTO)', alpha=0.7)
    axs[0].set_ylabel("$F_S$ (L/h)")
    axs[0].grid(True); axs[0].legend(); axs[0].set_title("Perfil de Alimentación")

    # V(t)
    axs[1].plot(t_full, x_full[:, x_names.index('V')])
    axs[1].axhline(V_max, color='r', linestyle='--', label="$V_{max}$")
    axs[1].set_ylabel("Volumen V (L)"); axs[1].grid(True); axs[1].legend(); axs[1].set_title("Volumen")

    # X(t)
    axs[2].plot(t_full, x_full[:, x_names.index('X')])
    axs[2].set_ylabel("Biomasa X (g/L)"); axs[2].grid(True); axs[2].set_title("Biomasa")

    # S(t)
    axs[3].plot(t_full, x_full[:, x_names.index('S')])
    # axs[3].axhline(S_max, color='r', linestyle='--', label="S_max (Relajado)") # Show si se relajó S_max
    axs[3].set_ylabel("Sustrato S (g/L)"); axs[3].grid(True); axs[3].legend(); axs[3].set_title("Sustrato")

    # P(t)
    axs[4].plot(t_full, x_full[:, x_names.index('P')])
    axs[4].set_ylabel("Producto P (g/L)"); axs[4].set_xlabel("Tiempo (h)"); axs[4].grid(True); axs[4].set_title("Producto")

    # T(t) y O(t)
    axT = axs[5]; axO = axT.twinx()
    p1, = axT.plot(t_full, x_full[:, x_names.index('T')], 'b-', label='Temperatura T (K)')
    p2, = axO.plot(t_full, x_full[:, x_names.index('O')], 'g-', label='Oxígeno O (g/L)')
    axT.set_xlabel("Tiempo (h)"); axT.set_ylabel("Temperatura T (K)", color='b'); axO.set_ylabel("Oxígeno O (g/L)", color='g')
    axT.tick_params(axis='y', labelcolor='b'); axO.tick_params(axis='y', labelcolor='g')
    axT.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
    axT.legend(handles=[p1, p2], loc='best'); axT.set_title("Temperatura y Oxígeno (con Q_j=0)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("\n[INFO] No se generaron gráficos porque la optimización RTO falló.")