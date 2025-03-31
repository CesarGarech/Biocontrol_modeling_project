import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parámetros del modelo (Alineados con MATLAB original)
# ----------------------------
mu_max = 0.6
Ks = 0.2
Ko = 0.01
Yxs = 0.5
Yxo = 0.1  # No relevante si dO/dt=0
Yps = 0.3
kLa = 180.0 # No relevante si dO/dt=0
O_sat = 0.18 # No relevante si dO/dt=0
Sf = 500.0      # <-- Alineado con MATLAB
V_max = 2.0       # <-- Alineado con MATLAB

# Condiciones iniciales (Alineadas con MATLAB original)
X0, S0, P0, O0, V0 = 1.0, 20.0, 0.0, 0.08, 0.2 # <-- S0 alineado

# Configuración temporal
t_batch = 5.0 # Asegurar flotante
t_total = 24.0
dt_control = 1.0 # Duración del intervalo de control
n_fed_batch_intervals = int((t_total - t_batch) / dt_control)
# t_grid_fed_batch = np.arange(n_fed_batch_intervals) * dt_control + t_batch # Tiempo solo para fase fed-batch

# Restricciones de alimentación
F_min = 0.0
F_max = 0.3
S_max = 30.0 # Límite para S

# ----------------------------
# Construcción del optimizador
# ----------------------------
opti = ca.Opti()

# --- Variables de Decisión ---
F_profile = opti.variable(n_fed_batch_intervals) # Flujos en la fase fed-batch
opti.set_initial(F_profile, 0.1) # Valor inicial para el optimizador

# --- Modelo del Sistema ---
# Estado inicial como MX para usarlo en la simulación
x_initial = ca.MX([X0, S0, P0, O0, V0])

# Definición simbólica para la función ODE
x_sym = ca.MX.sym('x', 5)
u_sym = ca.MX.sym('u') # Control (Flujo F)

def odefun(x, u):
    X, S, P, O, V = ca.vertsplit(x)
    # Safeguards (evitar valores negativos/cero que causen problemas)
    S = ca.fmax(S, 1e-9)
    O = ca.fmax(O, 1e-9) # Aunque O es constante, mantenemos por robustez simbólica
    V = ca.fmax(V, 1e-9)

    # Tasa de crecimiento (dependiente de S y O fijo)
    mu = mu_max * S / (Ks + S) * O / (Ko + O)
    # Tasa de dilución
    D = u / V

    # Ecuaciones diferenciales
    dX = mu * X - D * X
    dS = -mu * X / Yxs + D * (Sf - S)
    dP = Yps * mu * X - D * P
    # dO = kLa * (O_sat - O) - mu * X / Yxo - D * O # <-- Originalmente aquí
    dO = 0.0  # <-- CORRECCIÓN CLAVE: Replicar modelo MATLAB (O constante)
    dV = u  #ca.if_else(V < V_max, u, 0.0) # Detener llenado en V_max

    return ca.vertcat(dX, dS, dP, dO, dV)

# Crear la función ODE de CasADi
ode_casadi = {'x': x_sym, 'p': u_sym, 'ode': odefun(x_sym, u_sym)}

# --- Integradores ---
# Integrador para la fase batch (duración t_batch)
intg_batch = ca.integrator('intg_batch', 'idas', ode_casadi, 0, t_batch)

# Integrador para los intervalos de control fed-batch (duración dt_control)
intg_control = ca.integrator('intg_control', 'idas', ode_casadi, 0, dt_control)

# ----------------------------
# Simulación dentro de la Optimización
# ----------------------------

# 1. Simular Fase Batch (F=0)
res_batch = intg_batch(x0=x_initial, p=0.0) # p=0.0 significa F=0
x_after_batch = res_batch['xf']

# Guardar estados para posible análisis o restricciones futuras (opcional)
all_states_sym = [x_initial, x_after_batch]

# Restricciones al final del batch (si las hubiera)
# opti.subject_to(x_after_batch[1] <= S_max * 1.1) # Ejemplo: S no debe ser muy alto al final del batch

# Estado actual al inicio de la fase fed-batch
xk = x_after_batch

# 2. Loop de integración y restricciones (Fase Fed-Batch)
for i in range(n_fed_batch_intervals):
    Fi = F_profile[i] # Flujo para este intervalo

    # Integrar un intervalo de control
    res_interval = intg_control(x0=xk, p=Fi)
    xk_next = res_interval['xf'] # Estado al final del intervalo

    # Extraer estados simbólicos al final del intervalo (para restricciones)
    X_, S_, P_, O_, V_ = ca.vertsplit(xk_next)

    # --- Aplicar Restricciones ---
    # Límites del flujo de alimentación
    opti.subject_to(Fi >= F_min)
    opti.subject_to(Fi <= F_max)

    # Restricciones del estado al final del intervalo
    opti.subject_to(xk_next >= 0)    # No negatividad para todos los estados
    opti.subject_to(S_ <= S_max)     # Límite máximo de sustrato

    opti.subject_to(V_ <= V_max)

    # Actualizar estado para el siguiente intervalo
    xk = xk_next
    all_states_sym.append(xk) # Guardar estado simbólico

# Estado final simbólico después de toda la simulación
x_final_sym = xk
P_final_sym = x_final_sym[2]
V_final_sym = x_final_sym[4]

# ----------------------------
# Función Objetivo
# ----------------------------
# Maximizar Producto Total Final (P*V) -> Minimizar -(P*V)
J = -P_final_sym * V_final_sym

# Penalización suave para violaciones de S_max
# penalty_S = 0
# margen_tolerancia = 0.5  # Se penaliza solo si S > S_max + 0.5
# for k in range(1, len(all_states_sym)):
#     Sk = all_states_sym[k][1]
#     exceso = ca.fmax(0, Sk - (S_max + margen_tolerancia))
#     penalty_S += exceso**2

# J += 5e2 * penalty_S  # Peso menor para evitar dominancia

# penalty_V = 0
# margen_tolerancia = 0.5  # Penaliza solo si V > V_max + 0.5

# for k in range(1, len(all_states_sym)):
#     Vk = all_states_sym[k][4]  # Volumen
#     exceso = ca.fmax(0, Vk - (V_max + margen_tolerancia))
#     penalty_V += exceso**2

# J += 1e2 * penalty_V  # Peso moderado a la penalización



opti.minimize(J)

# ----------------------------
# Solver y Diagnóstico
# ----------------------------
solver_options = {
    "ipopt.print_level": 0, # 0: sin salida, 5: detallado
    "ipopt.max_iter": 500, # Reducido de 1000 a 500 como en MATLAB
    "print_time": False
}
opti.solver('ipopt', solver_options)

try:
    sol = opti.solve()
    print("✅ Optimización completada.")
except RuntimeError as e:
    print(f"❌ Error durante la optimización: {e}")
    # Intenta mostrar detalles de infactibilidad si falla
    try:
        opti.debug.show_infeasibilities()
    except Exception as debug_e:
        print(f"(No se pudieron mostrar detalles de infactibilidad: {debug_e})")
    raise e # Relanzar el error original

# ----------------------------
# Extracción y Simulación Final para Gráficas
# ----------------------------
F_opt = sol.value(F_profile)

# Simular de nuevo TODO el proceso con F_opt para obtener trayectoria completa
# (Necesario porque la optimización solo guarda los puntos finales de cada intervalo)
t_sim_plot = np.linspace(0, t_total, 201) # Más puntos para gráfica suave
dt_plot = t_sim_plot[1] - t_sim_plot[0]
intg_plot = ca.integrator('intg_plot', 'idas', ode_casadi, 0, dt_plot)

x_current_plot = np.array([X0, S0, P0, O0, V0]) # Estado inicial numérico
trajectory_plot = [x_current_plot]
F_values_plot = [] # Para graficar F usado en cada paso fino

current_F_plot = 0.0
fed_batch_interval_idx = 0
for k in range(len(t_sim_plot) - 1):
    t_now = t_sim_plot[k]

    # Determinar F para el tiempo actual t_now
    if t_now < t_batch:
        current_F_plot = 0.0
    else:
        # Encontrar el índice del intervalo de control correspondiente
        # restamos t_batch, dividimos por duración, tomamos parte entera
        fed_batch_interval_idx = min(int((t_now - t_batch) // dt_control), n_fed_batch_intervals - 1)
        current_F_plot = F_opt[fed_batch_interval_idx]

    F_values_plot.append(current_F_plot) # Guardar F usado

    # Integrar un paso pequeño dt_plot
    res_plot = intg_plot(x0=x_current_plot, p=current_F_plot)
    x_current_plot = res_plot['xf'].full().flatten()
    trajectory_plot.append(x_current_plot)

# Convertir resultados a arrays numpy
trajectory_plot = np.array(trajectory_plot)
X_plot = trajectory_plot[:, 0]
S_plot = trajectory_plot[:, 1]
P_plot = trajectory_plot[:, 2]
O_plot = trajectory_plot[:, 3]
V_plot = trajectory_plot[:, 4]

# Construir perfil F para gráfica 'stairs'
F_plot_stairs_vals = np.concatenate(([0.0], F_opt)) # F=0 al inicio, luego F_opt
t_plot_stairs_time = np.concatenate(([0.0, t_batch], t_batch + np.arange(1, n_fed_batch_intervals + 1) * dt_control))

# ----------------------------
# Resultados Finales Numéricos
# ----------------------------
P_final_val = P_plot[-1]
V_final_val = V_plot[-1]
producto_total_final = P_final_val * V_final_val

print(f"✅ Producto total final: {producto_total_final:.2f} g")

# Rendimiento (fórmula estándar: Producto / Substrato Consumido)
sustrato_alimentado_total = np.sum(F_opt * dt_control * Sf)
sustrato_inicial_total = S0 * V0
sustrato_final_remanente = S_plot[-1] * V_plot[-1]
sustrato_total_consumido = (sustrato_inicial_total + sustrato_alimentado_total) - sustrato_final_remanente
rendimiento_global_std = producto_total_final / sustrato_total_consumido if sustrato_total_consumido > 1e-9 else 0
print(f"📉 Rendimiento global (P/S consumido): {rendimiento_global_std:.2f} g/g")


# ----------------------------
# Gráficas tipo MATLAB
# ----------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
fig.suptitle('Simulación Fed-Batch Optimizada (Python/CasADi)', fontsize=16)

# Subplot 1: Perfil óptimo de alimentación F_opt
axs[0, 0].stairs(F_plot_stairs_vals, t_plot_stairs_time, color='r', linewidth=2, baseline=None)
axs[0, 0].set_title('Perfil óptimo de alimentación')
axs[0, 0].set_ylabel("F (L/h)")
axs[0, 0].set_xlim(0, t_total)
axs[0, 0].set_ylim(bottom=0)

# Subplot 2: Biomasa X
axs[0, 1].plot(t_sim_plot, X_plot, 'b', linewidth=2)
axs[0, 1].set_title("Biomasa")
axs[0, 1].set_ylabel("X (g/L)")
axs[0, 1].set_xlim(0, t_total)

# Subplot 3: Sustrato S
axs[0, 2].plot(t_sim_plot, S_plot, 'k', linewidth=2)
axs[0, 2].axhline(S_max, color='grey', linestyle='--', label=f'S_max={S_max}')
axs[0, 2].set_title("Sustrato")
axs[0, 2].set_ylabel("S (g/L)")
axs[0, 2].set_xlim(0, t_total)
axs[0, 2].legend()

# Subplot 4: Producto P
axs[1, 0].plot(t_sim_plot, P_plot, 'm', linewidth=2)
axs[1, 0].set_title("Producto")
axs[1, 0].set_ylabel("P (g/L)")
axs[1, 0].set_xlim(0, t_total)

# Subplot 5: Oxígeno disuelto O
axs[1, 1].plot(t_sim_plot, O_plot, 'g', linewidth=2)
axs[1, 1].set_title("Oxígeno disuelto (Constante)")
axs[1, 1].set_ylabel("O (g/L)")
axs[1, 1].set_xlim(0, t_total)
axs[1, 1].set_ylim(bottom=0, top=O0*1.2 if O0 > 1e-6 else 0.1) # Ajustar ylim para O

# Subplot 6: Volumen V
axs[1, 2].plot(t_sim_plot, V_plot, 'c', linewidth=2)
axs[1, 2].axhline(V_max, color='grey', linestyle='--', label=f'V_max={V_max}')
axs[1, 2].set_title("Volumen")
axs[1, 2].set_ylabel("V (L)")
axs[1, 2].set_xlim(0, t_total)
axs[1, 2].legend()

# Etiquetas y grid comunes
for ax in axs.flat:
    ax.set_xlabel("Tiempo (h)")
    ax.grid(True)

# plt.tight_layout() # constrained_layout=True suele ser mejor
plt.show()