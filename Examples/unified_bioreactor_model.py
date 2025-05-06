# unified_bioreactor_model.py
import casadi as ca
import numpy as np

def get_unified_model(use_oxygen_dynamics=True):
    """
    Define el modelo unificado del biorreactor fed-batch con CasADi.

    Args:
        use_oxygen_dynamics (bool): Si es True, usa dO/dt. Si False, O es constante.

    Returns:
        tuple: Contiene (ode_func, output_func, x_sym, u_sym, c_sym_for_nmpc, params, x_names, u_names, c_names)
    """
    # --- Parámetros Unificados ---
    # (Combinando y seleccionando de ambos archivos originales, ajusta según necesidad)
    params = {
        # Cinéticos / Estequiométricos
        'mu_max': 0.5,      # Tasa máx crecimiento (1/h) - Valor intermedio
        'K_S': 0.1,         # Constante Monod Sustrato (g/L) - Valor intermedio
        'K_O': 0.005,       # Constante Monod Oxígeno (g/L) - Valor RTO ajustado
        'Y_XS': 0.5,        # Rendimiento Biomasa/Substrato (g/g)
        'Y_XO': 0.15,       # Rendimiento Biomasa/Oxígeno (g/g) - Estimado
        'Y_PS': 0.2,        # Rendimiento Producto/Sustrato (via biomasa) (g/g) - RTO ajustado
        'm_S': 0.01,        # Mantenimiento Sustrato (g_S / (g_X * h)) - Añadido
        # Oxígeno
        'kLa': 200.0,       # Coef. Transf. Oxígeno (1/h) - RTO ajustado
        'O_sat': 0.008,     # Saturación Oxígeno (g/L) - Ajustado (T-dependiente en realidad)
        # Alimentación / Volumen
        'S_in': 200.0,      # Conc. Sustrato entrada (g/L) - RTO ajustado (era Sf)
        'O_in': 0.0,        # Conc. Oxígeno entrada (g/L)
        'V_max': 2.0,       # Volumen máximo reactor (L) - RTO
        # Termodinámicos
        'rho': 1000.0,      # Densidad medio (g/L) - NMPC
        'Cp': 4184.0,       # Capacidad calorífica (J/(g*K)) - NMPC (ajustado a g)
        'T_in': 298.15,     # Temperatura entrada (K) - NMPC
        'Y_QX': 2.0e7,      # Rendimiento Calor/Biomasa (J/g_X) - NMPC ajustado (era J/g?) verificar unidad
        # 'DeltaH_rxn_O': 1.4e7, # Alternativa: Calor por O2 (J/g_O)
        # Otros
        'epsilon': 1e-9     # Para evitar división por cero
    }

    # --- Variables Simbólicas ---
    X = ca.MX.sym('X')  # Biomasa (g/L)
    S = ca.MX.sym('S')  # Sustrato (g/L)
    P = ca.MX.sym('P')  # Producto (g/L)
    O = ca.MX.sym('O')  # Oxígeno Disuelto (g/L)
    V = ca.MX.sym('V')  # Volumen (L)
    T = ca.MX.sym('T')  # Temperatura (K)
    x = ca.vertcat(X, S, P, O, V, T)
    x_names = ['X', 'S', 'P', 'O', 'V', 'T']

    # Entradas: F_S (L/h), Q_j (J/h)
    # NOTA: NMPC original usaba Q_j/3600. Ahora usamos Q_j directamente en J/h.
    F_S = ca.MX.sym('F_S') # Flujo de alimentación (L/h)
    Q_j = ca.MX.sym('Q_j') # Calor añadido/removido (J/h) - Positivo si se añade calor
    u = ca.vertcat(F_S, Q_j)
    u_names = ['F_S', 'Q_j']

    # --- Ecuaciones del Modelo ---
    mu = params['mu_max'] * (S / (params['K_S'] + S + params['epsilon'])) \
                          * (O / (params['K_O'] + O + params['epsilon']))

    # Salvaguarda V > epsilon
    V_safe = ca.fmax(V, params['epsilon'])
    D = F_S / V_safe # Tasa de dilución (1/h)

    # Tasas específicas
    q_S = mu / params['Y_XS'] + params['m_S'] # Consumo sustrato
    q_O = mu / params['Y_XO']                 # Consumo Oxígeno (simplificado)
    q_P = params['Y_PS'] * mu                 # Producción Producto (asociado a crecimiento)

    # Balances de Materia
    dX_dt = mu * X - D * X
    dS_dt = -q_S * X + D * (params['S_in'] - S)
    dP_dt = q_P * X - D * P

    if use_oxygen_dynamics:
        OTR = params['kLa'] * (params['O_sat'] - O) # Transf O2
        OUR = q_O * X                          # Consumo O2
        dO_dt = OTR - OUR + D * (params['O_in'] - O)
    else:
        # Opción simplificada para RTO si se desea mantener O constante
        dO_dt = 0.0

    # Balance de Volumen (Fed-batch)
    dV_dt = F_S

    # Balance de Energía
    Q_gen = params['Y_QX'] * mu * X * V_safe # Generación calor por biomasa (J/h)
    # Q_gen = params['DeltaH_rxn_O'] * OUR * V_safe # Alternativa por O2
    Q_flow = F_S * params['rho'] * params['Cp'] * (params['T_in'] - T) # Calor por flujo entrada (J/h)
    # Q_j es la entrada manipulada (J/h)

    # Corrección: El denominador es rho*Cp*V, no solo rho*Cp
    dT_dt = (Q_flow + Q_gen - Q_j) / (params['rho'] * params['Cp'] * V_safe + params['epsilon']) # (K/h)

    # Vector de derivadas
    dx = ca.vertcat(dX_dt, dS_dt, dP_dt, dO_dt, dV_dt, dT_dt)

    # --- Salidas Controladas para NMPC (CVs) ---
    # Queremos controlar F_S (la propia entrada) y T (el estado)
    # La función de salida devolverá [F_S_actual, T_actual]
    # F_S_actual es simplemente la entrada u[0]
    # T_actual es el estado x[5]
    # Esta definición es para el NMPC. RTO no la usará directamente.
    c_nmpc = ca.vertcat(u[0], x[5])
    c_names = ['F_S_meas', 'T_meas']

    # --- Funciones CasADi ---
    ode_func = ca.Function('ode_func', [x, u], [dx], ['x', 'u'], ['dx'])
    # La función de salida para NMPC depende de 'x' y 'u'
    output_func_nmpc = ca.Function('output_func_nmpc', [x, u], [c_nmpc], ['x', 'u'], ['c_nmpc'])

    return ode_func, output_func_nmpc, x, u, c_nmpc, params, x_names, u_names, c_names

# --- Condiciones Iniciales y Límites Unificados ---
# (Ajusta estos valores según tu proceso específico)
x0_unified = np.array([
    1.0,    # X0 (g/L)
    15.0,   # S0 (g/L) - Intermedio
    0.0,    # P0 (g/L)
    0.007,  # O0 (g/L) - Cerca de O_sat, pero no exacto
    0.5,    # V0 (L) - Intermedio
    305.0   # T0 (K) - NMPC
])

# Límites para NMPC (ajustar)
# Estados: [X, S, P, O, V, T]
lbx = [0.0, 0.0, 0.0, 0.0, 0.1, 290.0]
ubx = [10.0, 200.0, 50.0, 0.01, 2.0, 315.0] # Vmax = 2.0

# Entradas: [F_S (L/h), Q_j (J/h)]
# Q_j limites: +/- 100 W => +/- 360000 J/h
lbu = [0.0, -3.6e5] # F_S_min, Q_j_min (enfriamiento max)
ubu = [0.3,  3.6e5] # F_S_max, Q_j_max (calentamiento max) - Fmax de RTO

# Cambios en entradas: [dF_S, dQ_j]
lbdu = [-0.1, -1.0e5] # Max decremento por paso
ubdu = [ 0.1,  1.0e5] # Max incremento por paso

if __name__ == '__main__':
    # Test rápido del modelo
    ode_func, output_func_nmpc, x, u, c_nmpc, params, x_names, u_names, c_names = get_unified_model()
    print("Modelo Unificado:")
    print("Estados (x):", x_names)
    print("Entradas (u):", u_names)
    print("Salidas NMPC (c):", c_names)
    print("Parámetros:", params.keys())
    print("Condiciones Iniciales (x0):", x0_unified)

    # Evaluar ODE en x0 con u=[0.1, 10000]
    u_test = np.array([0.1, 10000.0])
    dx0 = ode_func(x0_unified, u_test)
    print("\nDerivadas en x0 con u=", u_test, ":")
    print(dx0)

    # Evaluar Salidas en x0 con u_test
    c0 = output_func_nmpc(x0_unified, u_test)
    print("\nSalidas NMPC en x0 con u=", u_test, ":")
    print(c0)