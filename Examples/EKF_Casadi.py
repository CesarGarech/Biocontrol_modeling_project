import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# Para asegurar reproducibilidad del ruido (opcional)
# np.random.seed(42)

# === 1. Parámetros del modelo "real" ===
mu_max_real = 0.4     # (1/h)   Tasa max de crecimiento (valor real)
Yxs_real    = 0.5     # (gX/gS) Rendimiento biomasa/sustrato (valor real)
Ks          = 0.1     # (g/L)   Constante de semisaturación
alpha       = 0.1     # (gP/gX) Formación de producto asociada al crecimiento

# Parámetros "extra" para la medición
OD_sat      = 8.0     # mg/L   (Ejemplo saturación O2)
k_OUR       = 0.5     # mgO2/(L*gX) Factor de consumo O2
pH0         = 7.0     # Valor base de pH
P0_meas_ref = 0.0     # Producto base para cálculo pH (P0 en MATLAB)
k_acid      = 0.2     # Ajusta cómo el P afecta al pH
Tset        = 30      # (°C)   Temperatura base
k_Temp      = 0.02    # Ajusta la contribución exotérmica (X*S)

# === 2. Parámetros de simulación ===
dt        = 0.1   # h   Paso de integración
t_final   = 20    # h   Tiempo total
time_vec  = np.arange(0, t_final + dt, dt)
N         = len(time_vec)
n_states  = 5  # [X, S, P, mu_max, Yxs]
n_meas    = 3  # [OD, pH, T]

# Covarianzas de ruidos (Asegurarse que sean matrices NumPy)
Q = np.diag([1e-5, 1e-8, 1e-5, 1e-6, 1e-6])  # Ruido de proceso
R = np.diag([0.05, 0.02, 0.5])               # Ruido de medición (OD, pH, T)

# === 3. Definición Simbólica con CasADi ===

# Variables simbólicas
x_sym = ca.SX.sym('x', n_states) # Vector de estado simbólico [X, S, P, mu_max, Yxs]

# Desempaquetar estados simbólicos para claridad
X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym)

# --- Función de Actualización de Estado (f) ---
mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))

dX = mu_sym * X_sym
dS = - (1 / Yxs_sym) * dX
dP = alpha * dX
dMu_max = 0 # Modelo asume constante
dYxs = 0    # Modelo asume constante

# Discretización Euler explícito
x_next_sym = x_sym + dt * ca.vertcat(dX, dS, dP, dMu_max, dYxs)

# Crear función CasADi para f(x)
f_func = ca.Function('f', [x_sym], [x_next_sym], ['x_k'], ['x_k_plus_1'])

# --- Función de Medición (h) ---
OD_val_sym = OD_sat - k_OUR * X_sym
pH_val_sym = pH0 - k_acid * (P_sym - P0_meas_ref)
T_val_sym  = Tset + k_Temp * (X_sym * S_sym)

z_sym = ca.vertcat(OD_val_sym, pH_val_sym, T_val_sym)

# Crear función CasADi para h(x)
h_func = ca.Function('h', [x_sym], [z_sym], ['x'], ['z'])

# --- Jacobianos (Calculados automáticamente por CasADi) ---
F_sym = ca.jacobian(x_next_sym, x_sym)
H_sym = ca.jacobian(z_sym, x_sym)

# Crear funciones CasADi para los Jacobianos
F_func = ca.Function('F', [x_sym], [F_sym], ['x'], ['Fk'])
H_func = ca.Function('H', [x_sym], [H_sym], ['x'], ['Hk'])

# === 4. Condiciones iniciales ===
# 4.1. Estado "real" (como vector columna NumPy)
X0 = 0.1
S0 = 5.0
P0_real = 0.0
x_real = np.array([[X0], [S0], [P0_real], [mu_max_real], [Yxs_real]])

# 4.2. Estimación inicial (EKF) (como vector columna NumPy)
x_est = np.array([[0.05], [4.5], [0.0], [0.4], [0.5]]) # Conjetura inicial
P_est = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])      # Covarianza inicial

# Arreglos para guardar la evolución (usando listas de Python para facilidad)
X_real_list  = []
S_real_list  = []
P_real_list  = []
mu_real_list = []
Yxs_real_list= []

X_est_list   = []
S_est_list   = []
P_est_list   = []
mu_est_list  = []
Yxs_est_list = []

OD_meas_list   = []
pH_meas_list   = []
Temp_meas_list = []

# === 5. Bucle de simulación ===
for k in range(N):
    # --- Guardar valores actuales ---
    X_real_list.append(x_real[0, 0])
    S_real_list.append(x_real[1, 0])
    P_real_list.append(x_real[2, 0])
    mu_real_list.append(x_real[3, 0])
    Yxs_real_list.append(x_real[4, 0])

    X_est_list.append(x_est[0, 0])
    S_est_list.append(x_est[1, 0])
    P_est_list.append(x_est[2, 0])
    mu_est_list.append(x_est[3, 0])
    Yxs_est_list.append(x_est[4, 0])

    # --- (A) Generar medición "real" (z_k) ---
    # 1) Calcular salida real h(x_real) (Usando la función CasADi)
    z_noisefree_dm = h_func(x_real)
    z_noisefree = z_noisefree_dm.full() # Convertir de CasADi DM a NumPy array

    # 2) Agregar ruido de medición
    #    np.random.multivariate_normal necesita media 1D y covarianza 2D
    noise_meas = np.random.multivariate_normal(np.zeros(n_meas), R).reshape(-1, 1)
    z_k = z_noisefree + noise_meas

    OD_meas_list.append(z_k[0, 0])
    pH_meas_list.append(z_k[1, 0])
    Temp_meas_list.append(z_k[2, 0])

    if k < N - 1: # Evitar calcular predicción/corrección en el último paso
        # --- (B) EKF: Predicción ---
        # (1) f(x_est) -> Predicción del estado (Usando la función CasADi)
        x_pred_dm = f_func(x_est)
        x_pred = x_pred_dm.full()

        # (2) Calcular Jacobiano F (Usando la función CasADi)
        Fk_dm = F_func(x_est) # Evaluado en la estimación ANTERIOR x_est
        Fk = Fk_dm.full()

        # (3) Predicción de covarianza
        P_pred = Fk @ P_est @ Fk.T + Q

        # --- (C) EKF: Corrección ---
        # (4) Jacobiano de medición H (Usando la función CasADi)
        Hk_dm = H_func(x_pred) # Evaluado en el estado PREDICHO x_pred
        Hk = Hk_dm.full()

        # (5) Ganancia de Kalman
        Sk = Hk @ P_pred @ Hk.T + R
        # Usar pinv (pseudo-inversa) por estabilidad numérica, aunque inv podría funcionar
        Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk)

        # (6) Corrección de la estimación
        # Calcular la predicción de la medición h(x_pred)
        h_pred_dm = h_func(x_pred)
        h_pred = h_pred_dm.full()
        # Innovación (residuo)
        y_k = z_k - h_pred
        # Actualización del estado
        x_upd = x_pred + Kk @ y_k

        # (7) Corrección de la covarianza (forma estándar)
        P_upd = (np.eye(n_states) - Kk @ Hk) @ P_pred

        # Actualizar estimación para el siguiente paso
        x_est = x_upd
        P_est = P_upd

        # --- (D) Avance del "proceso real" ---
        # Calcular siguiente estado real (sin ruido primero)
        x_real_next_no_noise_dm = f_func(x_real)
        x_real_next_no_noise = x_real_next_no_noise_dm.full()
        # Agregar ruido de proceso
        noise_proc = np.random.multivariate_normal(np.zeros(n_states), Q).reshape(-1, 1)
        x_real = x_real_next_no_noise + noise_proc
        # Opcional: Asegurar no negatividad de estados si es físicamente requerido
        # x_real[0:3] = np.maximum(0, x_real[0:3]) # X, S, P >= 0

# Convertir listas a arrays NumPy para graficar
X_real_arr = np.array(X_real_list)
S_real_arr = np.array(S_real_list)
P_real_arr = np.array(P_real_list)
mu_real_arr= np.array(mu_real_list)
Yxs_real_arr= np.array(Yxs_real_list)

X_est_arr  = np.array(X_est_list)
S_est_arr  = np.array(S_est_list)
P_est_arr  = np.array(P_est_list)
mu_est_arr = np.array(mu_est_list)
Yxs_est_arr= np.array(Yxs_est_list)

OD_meas_arr   = np.array(OD_meas_list)
pH_meas_arr   = np.array(pH_meas_list)
Temp_meas_arr = np.array(Temp_meas_list)

# === 6. Gráficas de resultados ===
plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráfica

fig1, axs1 = plt.subplots(3, 2, figsize=(12, 10))
fig1.suptitle('EKF Results for Batch Bioprocess (Python/CasADi)', fontsize=14)

# Biomass
axs1[0, 0].plot(time_vec, X_real_arr, 'b-', label='X real')
axs1[0, 0].plot(time_vec, X_est_arr, 'r--', label='X estimated')
axs1[0, 0].set_xlabel('Time (h)')
axs1[0, 0].set_ylabel('Biomass (g/L)')
axs1[0, 0].set_title('Biomass')
axs1[0, 0].legend()
axs1[0, 0].grid(True)

# DO Measurement
axs1[0, 1].plot(time_vec, OD_meas_arr, 'k.-', markersize=3, linewidth=1, label='DO measured')
axs1[0, 1].set_xlabel('Time (h)')
axs1[0, 1].set_ylabel('DO (mg/L)')
axs1[0, 1].set_title('Dissolved Oxygen Measurement')
axs1[0, 1].legend()
axs1[0, 1].grid(True)

# Substrate
axs1[1, 0].plot(time_vec, S_real_arr, 'b-', label='S real')
axs1[1, 0].plot(time_vec, S_est_arr, 'r--', label='S estimated')
axs1[1, 0].set_xlabel('Time (h)')
axs1[1, 0].set_ylabel('Substrate (g/L)')
axs1[1, 0].set_title('Substrate')
axs1[1, 0].legend()
axs1[1, 0].grid(True)

# pH Measurement
axs1[1, 1].plot(time_vec, pH_meas_arr, 'k.-', markersize=3, linewidth=1, label='pH measured')
axs1[1, 1].set_xlabel('Time (h)')
axs1[1, 1].set_ylabel('pH')
axs1[1, 1].set_title('pH Measurement')
axs1[1, 1].legend()
axs1[1, 1].grid(True)

# Product
axs1[2, 0].plot(time_vec, P_real_arr, 'b-', label='P real')
axs1[2, 0].plot(time_vec, P_est_arr, 'r--', label='P estimated')
axs1[2, 0].set_xlabel('Time (h)')
axs1[2, 0].set_ylabel('Product (g/L)')
axs1[2, 0].set_title('Product')
axs1[2, 0].legend()
axs1[2, 0].grid(True)

# Temperature Measurement
axs1[2, 1].plot(time_vec, Temp_meas_arr, 'k.-', markersize=3, linewidth=1, label='T measured')
axs1[2, 1].set_xlabel('Time (h)')
axs1[2, 1].set_ylabel('Temperature (°C)')
axs1[2, 1].set_title('Temperature Measurement')
axs1[2, 1].legend()
axs1[2, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for title

# Figure for estimated parameters
fig2, axs2 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig2.suptitle('Parameter Estimation (Python/CasADi)', fontsize=14)

# mu_max
axs2[0].plot(time_vec, mu_real_arr, 'b-', label=r'$\mu_{max}$ real')
axs2[0].plot(time_vec, mu_est_arr, 'r--', label=r'$\mu_{max}$ estimated')
axs2[0].set_ylabel(r'$\mu_{max}$ (1/h)')
axs2[0].set_title(r'Estimation of $\mu_{max}$')
axs2[0].legend()
axs2[0].grid(True)

# Yxs
axs2[1].plot(time_vec, Yxs_real_arr, 'b-', label=r'$Y_{X/S}$ real')
axs2[1].plot(time_vec, Yxs_est_arr, 'r--', label=r'$Y_{X/S}$ estimated')
axs2[1].set_xlabel('Time (h)')
axs2[1].set_ylabel(r'$Y_{X/S}$ (gX/gS)')
axs2[1].set_title(r'Estimation of $Y_{X/S}$')
axs2[1].legend()
axs2[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()