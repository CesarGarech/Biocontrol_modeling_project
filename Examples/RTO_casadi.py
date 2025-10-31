import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================================================
# 1) Definición de la función ODE BIO
# ====================================================
def odefun(x, u):
    """
    Ecuaciones diferenciales Fed-Batch con O=constante.
    - Divisiones con fmax(V, epsilon) para evitar 1/0.
    """
    # Parameters
    mu_max = 0.6
    Ks      = 0.2
    Ko      = 0.01
    Yxs     = 0.5
    Yxo     = 0.1
    Yps     = 0.3
    kLa     = 180.0   # (no relevante si dO/dt=0)
    O_sat   = 0.18    # (no relevante si dO/dt=0)
    Sf      = 500.0
    V_max   = 2.0

    # Extraer variables de estado (evitar desempacado iterativo)
    X_ = x[0]
    S_ = x[1]
    P_ = x[2]
    O_ = x[3]
    V_ = x[4]

    # Salvaguarda para V
    # eps = 1e-8
    # V_eff = ca.fmax(V_, eps)   # evita división por 0

    # Si V>=V_max => F=0
    # u_sat = ca.if_else(V_ < V_max, u, 0.0)

    # Tasa de crecimiento
    mu = mu_max * (S_/(Ks + S_)) * (O_/(Ko + O_))   # O_ se asume >0
    # Tasa de dilución
    D = u / V_

    dX = mu*X_ - D*X_
    dS = -mu*X_/Yxs + D*(Sf - S_)
    dP = Yps*mu*X_ - D*P_
    dO = 0.0   # asumiendo oxígeno constante
    dV = u

    return ca.vertcat(dX, dS, dP, dO, dV)

# ====================================================
# 2) Coeficientes de colocación Radau (d=2)
# ====================================================
def radau_coefficients(d):
    """
    Retorna C_mat (shape (d+1, d)) y D_vec (shape d+1)
    para la colocación de Radau IIA con grado d=2.
    Estos valores son los correctos para Radau IIA order 3.
    """
    if d == 2:
        C_mat = np.array([
            [-2.0,   2.0],
            [ 1.5,  -4.5],
            [ 0.5,   2.5]
        ])
        D_vec = np.array([0.0, 0.0, 1.0])
        return C_mat, D_vec
    else:
        raise NotImplementedError("Solo implementado para d=2.")

# ====================================================
# 3) Process parameters and initial conditions
# ====================================================
t_batch = 5.0
t_total = 24.0
n_fb_intervals = int((t_total - t_batch))  # Increased number of intervals
dt_fb = (t_total - t_batch)/n_fb_intervals

F_min = 0.0
F_max = 0.3
S_max = 30.0
V_max = 2.0

X0, S0, P0, O0, V0 = 1.0, 20.0, 0.0, 0.08, 0.2

# ====================================================
# 4) Fase BATCH con F=0 (integración)
# ====================================================
x_sym = ca.MX.sym("x",5)
u_sym = ca.MX.sym("u")
ode_expr = odefun(x_sym, u_sym)

batch_integrator = ca.integrator(
    "batch_integrator","idas",
    {"x":x_sym, "p":u_sym, "ode":ode_expr},
    {"t0":0, "tf":t_batch}
)

x0_np = np.array([X0, S0, P0, O0, V0])
res_batch = batch_integrator(x0=x0_np, p=0.0)
x_after_batch = np.array(res_batch['xf']).flatten()
print("[INFO] Estado tras fase batch:", x_after_batch)

# ====================================================
# 5) Formulación de la fase Fed-Batch con colocación
# ====================================================
opti = ca.Opti()

d = 2
C_radau, D_radau = radau_coefficients(d)
nx = 5

# State and control variables
X_col = []
F_col = []

for k in range(n_fb_intervals):
    row_states = []
    for j in range(d+1):
        if (k==0 and j==0):
            # Fijar el estado inicial del primer intervalo
            # con un "parameter" (no es variable)
            xk0_param = opti.parameter(nx)
            opti.set_value(xk0_param, x_after_batch)
            row_states.append(xk0_param)
        else:
            # variable
            xk_j = opti.variable(nx)
            row_states.append(xk_j)
            # Restricciones
            # no-negatividad:
            opti.subject_to(xk_j >= 0)
            # S <= S_max
            opti.subject_to(xk_j[1] <= S_max)
            # V <= V_max
            opti.subject_to(xk_j[4] <= V_max)
    X_col.append(row_states)

    # Variable de control en cada intervalo
    Fk = opti.variable()
    F_col.append(Fk)
    opti.subject_to(Fk >= F_min)
    opti.subject_to(Fk <= F_max)

# ====================================================
# 6) Ecuaciones de Colocación
# ====================================================
h = dt_fb
for k in range(n_fb_intervals):
    for j in range(1,d+1):
        # xp_j = sum_{m=0..d} C_radau[m, j-1]* X_col[k][m]
        xp_j = 0
        for m in range(d+1):
            xp_j += C_radau[m, j-1]* X_col[k][m]

        # f(Xk_j, Fk)
        fkj = odefun(X_col[k][j], F_col[k])
        # Restricción => h*f - xp_j = 0
        coll_eq = h*fkj - xp_j
        opti.subject_to(coll_eq == 0)

    # Continuidad al final del subintervalo
    Xk_end = 0
    for m in range(d+1):
        Xk_end += D_radau[m]* X_col[k][m]

    if k < n_fb_intervals-1:
        # Xk_end = X_{k+1}[0]
        for i_ in range(nx):
            opti.subject_to(Xk_end[i_] == X_col[k+1][0][i_])

# Estado final global => X_final
X_final = X_col[-1][-1] # The state at the end of the last interval

P_final = X_final[2]
V_final = X_final[4]

# ====================================================
# 7) Función objetivo => maximizar (P_final*V_final)
# ====================================================
opti.minimize(- (P_final*V_final))

# ====================================================
# 8) Guesses iniciales (importante para evitar NaNs)
# ====================================================
for k in range(n_fb_intervals):
    opti.set_initial(F_col[k], 0.1)   # Try a constant initial guess
    for j in range(d+1):
        # If not es el primer "parameter"
        if not (k==0 and j==0):
            # Como guess, usemos el estado final de batch (o algo similar)
            opti.set_initial(X_col[k][j], x_after_batch)

# ====================================================
# 9) Configurar y resolver
# ====================================================
p_opts = {}
s_opts = {
    "max_iter": 2000,  # Increased max_iter
    "print_level": 0,
    "sb": 'yes',   # debug
    "mu_strategy": "adaptive"
}
opti.solver("ipopt", p_opts, s_opts)

try:
    sol = opti.solve()
    print("[INFO] ¡Solución encontrada!")
except RuntimeError as e:
    print("[ERROR] No se encontró solución:", e)
    try:
        # Show infeasibilidades
        opti.debug.show_infeasibilities()
    except:
        pass
    raise

F_opt = [sol.value(fk) for fk in F_col]
X_fin_val = sol.value(X_final)
P_fin_val = X_fin_val[2]
V_fin_val = X_fin_val[4]

print("F_opt:", F_opt)
print("Estado final:", X_fin_val)
print("P_final =", P_fin_val)
print("V_final =", V_fin_val)
print("Producto total =", P_fin_val*V_fin_val)

# --- INICIO: GUARDAR PERFIL ---
print("\n[INFO] Guardando perfil óptimo de F_S para NMPC...")
# Necesitamos los tiempos donde cambia F (inicio de cada intervalo RTO)
t_rto_intervals = np.linspace(t_batch, t_total, n_fb_intervals + 1)
# Guardamos F_opt y los tiempos donde cambia (t_profile es el inicio de cada intervalo donde F_opt[k] aplica)
rto_original_output = {'t_profile': t_rto_intervals[:-1], 'F_S_profile': np.array(F_opt)}

# --- INICIO: Modificaciones para guardar en carpeta "Output" ---
output_dir = "Output"
file_name = "rto_original_feed_profile.npz"
# Construir la ruta completa al archivo
full_path = os.path.join(output_dir, file_name)

# Create "Output" folder if it does not exist
# exist_ok=True evita un error si la carpeta ya existe
os.makedirs(output_dir, exist_ok=True)

# Guardar el archivo usando la ruta completa
np.savez(full_path, **rto_original_output)
# Actualizar el mensaje de confirmación para mostrar la ruta correcta
print(f"[INFO] Perfil óptimo de F_S (original RTO) guardado en '{full_path}'")
# ====================================================
# 10) Reconstruir y graficar trayectoria
#     (batch + fed-batch)
# ====================================================
# a) Fase batch: con dt pequeño
N_batch_plot = 50
t_batch_plot = np.linspace(0, t_batch, N_batch_plot)
dt_b = t_batch_plot[1] - t_batch_plot[0]

batch_plot_int = ca.integrator(
    "batch_plot_int","idas",
    {"x":x_sym,"p":u_sym,"ode":ode_expr},
    {"t0":0, "tf":dt_b}
)

xbatch_traj = [x0_np]
xk_ = x0_np.copy()
for _ in range(N_batch_plot-1):
    res_ = batch_plot_int(x0=xk_, p=0.0)
    xk_ = np.array(res_["xf"]).flatten()
    xbatch_traj.append(xk_)
xbatch_traj = np.array(xbatch_traj)

# b) Fase fed-batch: integrando de 5 a 24 h con dt fino
t_fb_plot = np.linspace(t_batch, t_total, 400)   # algo denso
dt_fb_plot = t_fb_plot[1]-t_fb_plot[0]

fb_plot_int = ca.integrator(
    "fb_plot_int","idas",
    {"x":x_sym,"p":u_sym,"ode":ode_expr},
    {"t0":0,"tf":dt_fb_plot}
)

xfb_traj = []
xk_ = xbatch_traj[-1].copy()
for i, t_ in enumerate(t_fb_plot):
    xfb_traj.append(xk_)
    if i == len(t_fb_plot)-1:
        break
    # Determinar en qué subintervalo k estamos
    kk_ = int((t_ - t_batch)//dt_fb)
    kk_ = max(0, kk_)
    kk_ = min(n_fb_intervals-1, kk_)
    # Tomar F correspondiente
    F_now = sol.value(F_col[kk_])
    # Apagar F si V>=Vmax
    if xk_[4] >= V_max:
        F_now = 0.0
    # Integrar
    res_ = fb_plot_int(x0=xk_, p=F_now)
    xk_ = np.array(res_["xf"]).flatten()

xfb_traj = np.array(xfb_traj)

# Unimos
t_full = np.concatenate([t_batch_plot, t_fb_plot])
x_full = np.vstack([xbatch_traj, xfb_traj])

X_full = x_full[:,0]
S_full = x_full[:,1]
P_full = x_full[:,2]
O_full = x_full[:,3]
V_full = x_full[:,4]

# Construir F para graficar
F_batch_plot = np.zeros_like(t_batch_plot)
F_fb_plot = []
for i, tt in enumerate(t_fb_plot):
    kk_ = int((tt - t_batch)//dt_fb)
    kk_ = max(0, kk_)
    kk_ = min(n_fb_intervals-1, kk_)
    valF = sol.value(F_col[kk_])
    if xfb_traj[i,4]>=V_max:
        valF=0.0
    F_fb_plot.append(valF)
F_fb_plot = np.array(F_fb_plot)

F_plot = np.concatenate([F_batch_plot, F_fb_plot])

# ====================================================
# 11) Gráficas
# ====================================================
fig, axs = plt.subplots(2,3, figsize=(14,8))
axs=axs.ravel()

# F
axs[0].plot(t_full, F_plot, linewidth=2)
axs[0].set_title("Flujo de alimentación F(t)")
axs[0].set_xlabel("Tiempo (h)")
axs[0].set_ylabel("F (L/h)")
axs[0].grid(True)

# X
axs[1].plot(t_full, X_full, linewidth=2)
axs[1].set_title("Biomasa X(t)")
axs[1].set_xlabel("Tiempo (h)")
axs[1].set_ylabel("X (g/L)")
axs[1].grid(True)

# S
axs[2].plot(t_full, S_full, linewidth=2)
axs[2].axhline(S_max, color='r', linestyle='--', label="S_max")
axs[2].set_title("Sustrato S(t)")
axs[2].set_xlabel("Tiempo (h)")
axs[2].set_ylabel("S (g/L)")
axs[2].legend()
axs[2].grid(True)

# P
axs[3].plot(t_full, P_full, linewidth=2)
axs[3].set_title("Producto P(t)")
axs[3].set_xlabel("Tiempo (h)")
axs[3].set_ylabel("P (g/L)")
axs[3].grid(True)

# O
axs[4].plot(t_full, O_full, linewidth=2)
axs[4].set_title("Oxígeno disuelto O(t) (constante)")
axs[4].set_xlabel("Tiempo (h)")
axs[4].set_ylabel("O (g/L)")
axs[4].grid(True)

# V
axs[5].plot(t_full, V_full, linewidth=2)
axs[5].axhline(V_max, color='r', linestyle='--', label="V_max")
axs[5].set_title("Volumen V(t)")
axs[5].set_xlabel("Tiempo (h)")
axs[5].set_ylabel("V (L)")
axs[5].legend()
axs[5].grid(True)

plt.tight_layout()
plt.show()