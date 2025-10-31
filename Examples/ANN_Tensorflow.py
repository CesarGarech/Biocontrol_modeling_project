import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# === PARTE 1: GENERACIÓN DE DATOS USANDO EL CÓDIGO EKF PROPORCIONADO      ===
# === Este bloque es idéntico al código original para generar nuestros datos ===
# ==============================================================================

# Para asegurar reproducibilidad del ruido y del entrenamiento de la ANN
np.random.seed(42)
tf.random.set_seed(42)

# === 1. "Real" model parameters ===
mu_max_real = 0.4      # (1/h)   Tasa max de crecimiento (valor real)
Yxs_real    = 0.5      # (gX/gS) Rendimiento biomasa/sustrato (valor real)
Ks          = 0.1      # (g/L)   Constante de semisaturación
alpha       = 0.1      # (gP/gX) Formación de producto asociada al crecimiento

# Parameters "extra" para la medición
OD_sat      = 8.0      # mg/L   (Ejemplo saturación O2)
k_OUR       = 0.5      # mgO2/(L*gX) Factor de consumo O2
pH0         = 7.0      # Valor base de pH
P0_meas_ref = 0.0      # Producto base para cálculo pH (P0 en MATLAB)
k_acid      = 0.2      # Ajusta cómo el P afecta al pH
Tset        = 30       # (°C)   Temperatura base
k_Temp      = 0.02     # Ajusta la contribución exotérmica (X*S)

# === 2. Simulation parameters ===
dt        = 0.1   # h   Paso de integración
t_final   = 20    # h   Tiempo total
time_vec  = np.arange(0, t_final + dt, dt)
N         = len(time_vec)
n_states  = 5  # [X, S, P, mu_max, Yxs]
n_meas    = 3  # [OD, pH, T]

# Covarianzas de ruidos (Asegurarse que sean matrices NumPy)
Q = np.diag([1e-5, 1e-8, 1e-5, 1e-6, 1e-6])  # Ruido de proceso
R = np.diag([0.05, 0.02, 0.5])              # Ruido de medición (OD, pH, T)

# === 3. Definición Simbólica con CasADi ===
x_sym = ca.SX.sym('x', n_states)
X_sym, S_sym, P_sym, mu_max_sym, Yxs_sym = ca.vertsplit(x_sym)

mu_sym = mu_max_sym * (S_sym / (Ks + S_sym))
dX = mu_sym * X_sym
dS = - (1 / Yxs_sym) * dX
dP = alpha * dX
dMu_max = 0
dYxs = 0

x_next_sym = x_sym + dt * ca.vertcat(dX, dS, dP, dMu_max, dYxs)
f_func = ca.Function('f', [x_sym], [x_next_sym])

OD_val_sym = OD_sat - k_OUR * X_sym
pH_val_sym = pH0 - k_acid * (P_sym - P0_meas_ref)
T_val_sym  = Tset + k_Temp * (X_sym * S_sym)
z_sym = ca.vertcat(OD_val_sym, pH_val_sym, T_val_sym)
h_func = ca.Function('h', [x_sym], [z_sym])

F_sym = ca.jacobian(x_next_sym, x_sym)
H_sym = ca.jacobian(z_sym, x_sym)
F_func = ca.Function('F', [x_sym], [F_sym])
H_func = ca.Function('H', [x_sym], [H_sym])

# === 4. Condiciones iniciales ===
X0 = 0.1
S0 = 5.0
P0_real = 0.0
x_real = np.array([[X0], [S0], [P0_real], [mu_max_real], [Yxs_real]])
x_est = np.array([[0.05], [4.5], [0.0], [0.4], [0.5]])
P_est = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])

# Arrays to save evolution
X_real_list, S_real_list, P_real_list, mu_real_list, Yxs_real_list = [], [], [], [], []
X_est_list, S_est_list, P_est_list, mu_est_list, Yxs_est_list = [], [], [], [], []
OD_meas_list, pH_meas_list, Temp_meas_list = [], [], []

# === 5. Simulation loop ===
for k in range(N):
    X_real_list.append(x_real[0, 0]); S_real_list.append(x_real[1, 0]); P_real_list.append(x_real[2, 0])
    mu_real_list.append(x_real[3, 0]); Yxs_real_list.append(x_real[4, 0])
    X_est_list.append(x_est[0, 0]); S_est_list.append(x_est[1, 0]); P_est_list.append(x_est[2, 0])
    mu_est_list.append(x_est[3, 0]); Yxs_est_list.append(x_est[4, 0])

    z_noisefree = h_func(x_real).full()
    noise_meas = np.random.multivariate_normal(np.zeros(n_meas), R).reshape(-1, 1)
    z_k = z_noisefree + noise_meas
    OD_meas_list.append(z_k[0, 0]); pH_meas_list.append(z_k[1, 0]); Temp_meas_list.append(z_k[2, 0])

    if k < N - 1:
        x_pred = f_func(x_est).full()
        Fk = F_func(x_est).full()
        P_pred = Fk @ P_est @ Fk.T + Q

        Hk = H_func(x_pred).full()
        Sk = Hk @ P_pred @ Hk.T + R
        Kk = P_pred @ Hk.T @ np.linalg.pinv(Sk)
        
        h_pred = h_func(x_pred).full()
        y_k = z_k - h_pred
        x_upd = x_pred + Kk @ y_k
        P_upd = (np.eye(n_states) - Kk @ Hk) @ P_pred
        
        x_est = x_upd
        P_est = P_upd

        x_real_next_no_noise = f_func(x_real).full()
        noise_proc = np.random.multivariate_normal(np.zeros(n_states), Q).reshape(-1, 1)
        x_real = x_real_next_no_noise + noise_proc

# Convertir listas a arrays NumPy
X_real_arr, S_real_arr, P_real_arr = np.array(X_real_list), np.array(S_real_list), np.array(P_real_list)
mu_real_arr, Yxs_real_arr = np.array(mu_real_list), np.array(Yxs_real_list)
X_est_arr, S_est_arr, P_est_arr = np.array(X_est_list), np.array(S_est_list), np.array(P_est_list)
mu_est_arr, Yxs_est_arr = np.array(mu_est_list), np.array(Yxs_est_list)
OD_meas_arr, pH_meas_arr, Temp_meas_arr = np.array(OD_meas_list), np.array(pH_meas_list), np.array(Temp_meas_list)


# ==============================================================================
# === PARTE 2: DISEÑO, ENTRENAMIENTO Y PREDICCIÓN CON LA RED NEURONAL (ANN) ===
# ==============================================================================
print("\n--- Iniciando la fase de la Red Neuronal Artificial (ANN) ---")

# --- 7. Preparación de Datos para la ANN ---
# Entradas (X): Mediciones de los sensores
X_data = np.stack([OD_meas_arr, pH_meas_arr, Temp_meas_arr], axis=1)

# Salidas (Y): Estados y parámetros "reales" que queremos predecir
Y_data = np.stack([X_real_arr, S_real_arr, P_real_arr, mu_real_arr, Yxs_real_arr], axis=1)

print(f"Forma de los datos de entrada (mediciones): {X_data.shape}")
print(f"Forma de los datos de salida (estados/parámetros): {Y_data.shape}")

# --- 8. Escalado de Datos ---
# Es crucial para el buen rendimiento de la ANN
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_data)
Y_scaled = scaler_Y.fit_transform(Y_data)

# --- 9. División en Entrenamiento y Prueba ---
# 80% para entrenamiento, 20% para validación
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# --- 10. Construcción y Compilación del Modelo ANN ---
ann_model = Sequential([
    Input(shape=(n_meas,)), # Capa de entrada con 3 neuronas (OD, pH, T)
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(n_states, activation='linear') # Capa de salida con 5 neuronas (estados+params) y activación lineal
])

ann_model.compile(optimizer='adam', loss='mean_squared_error')
ann_model.summary()

# --- 11. Entrenamiento de la ANN ---
print("\nEntrenando el modelo ANN...")
history = ann_model.fit(
    X_train, y_train,
    epochs=150,  # Número de pasadas por el set de entrenamiento
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=0  # Poner en 1 para ver el progreso del entrenamiento por época
)
print("Entrenamiento completado.")

# --- 12. Predicción con la ANN y Des-escalado ---
# Predecir sobre todo el conjunto de datos para graficar la serie temporal completa
Y_pred_scaled = ann_model.predict(X_scaled)

# Revertir el escalado para obtener los valores en sus unidades originales
Y_pred_ann = scaler_Y.inverse_transform(Y_pred_scaled)

# Unpack las predicciones para graficar
X_ann_pred = Y_pred_ann[:, 0]
S_ann_pred = Y_pred_ann[:, 1]
P_ann_pred = Y_pred_ann[:, 2]
mu_ann_pred = Y_pred_ann[:, 3]
Yxs_ann_pred = Y_pred_ann[:, 4]

# ==============================================================================
# === PARTE 3: VISUALIZACIÓN Y COMPARACIÓN DE RESULTADOS                    ===
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- Gráfica de la pérdida durante el entrenamiento ---
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_title('ANN Learning Curve (MSE)')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Mean Squared Error (MSE)')
ax_loss.legend()
ax_loss.grid(True)
ax_loss.set_yscale('log') # Logarithmic scale is useful to see detail

# --- Comparative Plots of States ---
fig_states, axs_states = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig_states.suptitle('Comparison: Real vs. EKF vs. ANN - Process States', fontsize=16)

# Biomass
axs_states[0].plot(time_vec, X_real_arr, 'b-', label='X Real')
axs_states[0].plot(time_vec, X_est_arr, 'r--', label='X Estimated (EKF)')
axs_states[0].plot(time_vec, X_ann_pred, 'g:', linewidth=3, label='X Predicted (ANN)')
axs_states[0].set_ylabel('Biomass (g/L)')
axs_states[0].set_title('Biomass (X)')
axs_states[0].legend()
axs_states[0].grid(True)

# Substrate
axs_states[1].plot(time_vec, S_real_arr, 'b-', label='S Real')
axs_states[1].plot(time_vec, S_est_arr, 'r--', label='S Estimated (EKF)')
axs_states[1].plot(time_vec, S_ann_pred, 'g:', linewidth=3, label='S Predicted (ANN)')
axs_states[1].set_ylabel('Substrate (g/L)')
axs_states[1].set_title('Substrate (S)')
axs_states[1].legend()
axs_states[1].grid(True)

# Product
axs_states[2].plot(time_vec, P_real_arr, 'b-', label='P Real')
axs_states[2].plot(time_vec, P_est_arr, 'r--', label='P Estimated (EKF)')
axs_states[2].plot(time_vec, P_ann_pred, 'g:', linewidth=3, label='P Predicted (ANN)')
axs_states[2].set_xlabel('Time (h)')
axs_states[2].set_ylabel('Product (g/L)')
axs_states[2].set_title('Product (P)')
axs_states[2].legend()
axs_states[2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Comparative Plots of Parameters ---
fig_params, axs_params = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig_params.suptitle('Comparison: Real vs. EKF vs. ANN - Model Parameters', fontsize=16)

# mu_max
axs_params[0].plot(time_vec, mu_real_arr, 'b-', label=r'$\mu_{max}$ Real')
axs_params[0].plot(time_vec, mu_est_arr, 'r--', label=r'$\mu_{max}$ Estimated (EKF)')
axs_params[0].plot(time_vec, mu_ann_pred, 'g:', linewidth=3, label=r'$\mu_{max}$ Predicted (ANN)')
axs_params[0].set_ylabel(r'$\mu_{max}$ (1/h)')
axs_params[0].set_title(r'Maximum Growth Rate ($\mu_{max}$)')
axs_params[0].legend()
axs_params[0].grid(True)

# Yxs
axs_params[1].plot(time_vec, Yxs_real_arr, 'b-', label=r'$Y_{X/S}$ Real')
axs_params[1].plot(time_vec, Yxs_est_arr, 'r--', label=r'$Y_{X/S}$ Estimated (EKF)')
axs_params[1].plot(time_vec, Yxs_ann_pred, 'g:', linewidth=3, label=r'$Y_{X/S}$ Predicted (ANN)')
axs_params[1].set_xlabel('Time (h)')
axs_params[1].set_ylabel(r'$Y_{X/S}$ (gX/gS)')
axs_params[1].set_title(r'Biomass/Substrate Yield ($Y_{X/S}$)')
axs_params[1].legend()
axs_params[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()