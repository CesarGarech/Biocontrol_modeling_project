import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 1. Parámetros del modelo
mu_max = 0.6
Ks = 0.2
Ko = 0.01
Yxs = 0.5
Yxo = 0.1
Yps = 0.3
kLa = 180
O_sat = 0.18

Sf = 500  # g/L
V_max = 2  # L

# Iniciales
X0 = 1.0
S0 = 20.0
P0 = 0.0
O0 = 0.08
V0 = 0.2

t_batch = 5
t_total = 24
n_seg = t_total - t_batch

# Interpolador de perfil F
def interp_F(F_array):
    t_points = np.linspace(t_batch, t_total, len(F_array))
    def F_func(t):
        if t < t_batch:
            return 0.0
        return float(np.interp(t, t_points, F_array))
    return F_func

# Dinámica del biorreactor
def bioreactor_odes(t, y, F_func):
    X, S, P, O, V = y
    F = F_func(t)
    mu = mu_max * (S / (Ks + S)) * (O / (Ko + O))
    D = F / V if V > 0 else 0
    dX = mu * X - D * X
    dS = -mu * X / Yxs + D * (Sf - S)
    dP = Yps * mu * X - D * P
    dO = 0.0
    dV = F if V < V_max else 0.0
    return [dX, dS, dP, dO, dV]

# Simulación para evaluar función objetivo
def simulate_bioreactor(F_array):
    F_func = interp_F(F_array)
    y0 = [X0, S0, P0, O0, V0]
    t_eval = np.linspace(0, t_total, 200)
    sol = solve_ivp(bioreactor_odes, [0, t_total], y0, args=(F_func,), t_eval=t_eval, method='LSODA')
    return sol.t, sol.y.T

# Función objetivo
def objective(F_array):
    t, y = simulate_bioreactor(F_array)
    P_end = y[-1, 2]
    V_end = y[-1, 4]
    S_max = np.max(y[:, 1])
    penalty = 1e4 * max(0, S_max - 30) ** 2
    return -P_end * V_end + penalty

# Optimización estilo MATLAB (fmincon)
F0 = 0.1 * np.ones(n_seg)
F_bounds = [(0.01, 0.3) for _ in range(n_seg)]

result = minimize(objective, F0, method='SLSQP', bounds=F_bounds,
                  options={'disp': True, 'maxiter': 500, 'ftol': 1e-6})

F_opt = result.x

# Simulación final
T, Y = simulate_bioreactor(F_opt)

print(f"Producto final: {Y[-1, 2]:.2f} g")
S_total = S0 + Sf * (Y[-1, 4] - V0) / Y[-1, 4]
print(f"Rendimiento global (P/S): {Y[-1, 2]/S_total:.2f}")

# Gráficas
plt.figure(figsize=(12, 7))
plt.subplot(2, 3, 1)
plt.step(np.linspace(t_batch, t_total, n_seg), F_opt, where='post', color='r')
plt.title('Perfil óptimo de alimentación')
plt.xlabel('Tiempo (h)')
plt.ylabel('F (L/h)')
plt.grid(True)

labels = ['Biomasa (X)', 'Sustrato (S)', 'Producto (P)', 'Oxígeno (O)', 'Volumen (V)']
for i in range(5):
    plt.subplot(2, 3, i+2)
    plt.plot(T, Y[:, i], linewidth=2)
    plt.title(labels[i])
    plt.xlabel('Tiempo (h)')
    plt.ylabel(labels[i])
    plt.grid(True)

plt.tight_layout()
plt.show()