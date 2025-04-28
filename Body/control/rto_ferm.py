# drto_anaerobic_page_simplified_optim.py
import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import traceback # Para imprimir errores detallados

# =============================================================================
# 1. MODELO Y CINÉTICA (Sin cambios en las definiciones base)
# =============================================================================

def mu_fermentacion(S, P, O2,
                    mumax_aerob, Ks_aerob, KO_aerob, # Params mu1 (aerobio)
                    mumax_anaerob, Ks_anaerob, KiS_anaerob, # Params mu2 (anaerobio) - Sustrato
                    KP_anaerob, n_p,                   # Params mu2 (anaerobio) - Producto
                    KO_inhib_anaerob):                 # Params mu2 (anaerobio) - O2 (Inhibición)
    """Calcula la tasa de crecimiento específica (mu)."""
    # ... (código idéntico al anterior) ...
    S = np.maximum(1e-9, S)
    P = np.maximum(0.0, P)
    O2 = np.maximum(0.0, O2)

    # mu1 (Aeróbica)
    term_S_aerob = S / (Ks_aerob + S) if (Ks_aerob + S) > 1e-9 else 0.0
    term_O2_aerob = O2 / (KO_aerob + O2) if (KO_aerob + O2) > 1e-9 else 0.0
    mu1 = mumax_aerob * term_S_aerob * term_O2_aerob

    # mu2 (Anaeróbica)
    denominador_S_anaerob = Ks_anaerob + S + (S**2 / KiS_anaerob) if KiS_anaerob > 1e-9 else Ks_anaerob + S
    term_S_anaerob = S / denominador_S_anaerob if denominador_S_anaerob > 1e-9 else 0.0
    base_P = 1.0 - (P / KP_anaerob)
    term_P_anaerob = np.power(np.maximum(0.0, base_P), n_p) if KP_anaerob > 1e-9 else 0.0
    term_P_anaerob = np.maximum(0.0, term_P_anaerob)
    term_O2_inhib_anaerob = KO_inhib_anaerob / (KO_inhib_anaerob + O2) if (KO_inhib_anaerob + O2) > 1e-9 else 1.0
    mu2 = mumax_anaerob * term_S_anaerob * term_P_anaerob * term_O2_inhib_anaerob

    mu_total = mu1 + mu2
    return np.maximum(0.0, mu_total)


def modelo_ferm_scipy(t, y, current_params, current_kla):
    """Modelo ODE para SciPy (simulación inicial batch - O2 dinámico)."""
    # ... (código idéntico al anterior) ...
    X, S, P, O2, V = y
    X = max(1e-9, X); S = max(0, S); P = max(0, P); O2 = max(0, O2); V = max(1e-6, V)
    F = 0.0 # Fase batch

    mu = mu_fermentacion(S, P, O2,
                         current_params["mumax_aerob"], current_params["Ks_aerob"], current_params["KO_aerob"],
                         current_params["mumax_anaerob"], current_params["Ks_anaerob"], current_params["KiS_anaerob"],
                         current_params["KP_anaerob"], current_params["n_p"], current_params["KO_inhib_anaerob"])
    mu = max(0, mu)

    qP = current_params["alpha_lp"] * mu + current_params["beta_lp"]
    rate_P = qP * X
    consumo_S_X = (mu / current_params["Yxs"]) * X if current_params["Yxs"] > 1e-9 else 0
    consumo_S_P = (rate_P / current_params["Yps"]) * X if current_params["Yps"] > 1e-9 else 0 # Corrección: Yps es gP/gS, debería ser rate_P / Yps
    consumo_S_P = (rate_P / current_params["Yps"]) if current_params["Yps"] > 1e-9 else 0
    consumo_S_maint = current_params["ms"] * X
    rate_S = consumo_S_X + consumo_S_P + consumo_S_maint
    consumo_O2_X = (mu / current_params["Yxo"]) * X if current_params["Yxo"] > 1e-9 else 0
    consumo_O2_maint = current_params["mo"] * X
    OUR_g = consumo_O2_X + consumo_O2_maint # [gO2/L/h]
    OUR_mg = OUR_g * 1000.0 # [mgO2/L/h]

    dXdt = (mu - current_params["Kd"]) * X - (F / V) * X
    dSdt = -rate_S + (F / V) * (current_params["Sin"] - S)
    dPdt = rate_P - (F / V) * P
    dVdt = F
    OTR = current_kla * (current_params["Cs"] - O2) # [mg O2 / L / h]
    dOdt = OTR - OUR_mg - (F / V) * O2      # [mg/L/h]

    return [dXdt, dSdt, dPdt, dOdt, dVdt]

def odefun_ferm_casadi_builder(params_dict, kla_value, constant_O2_value=None):
    """
    Construye la función ODE simbólica de CasADi.
    Si constant_O2_value no es None, fija dOdt=0 y usa ese valor para O2 en los cálculos.
    """
    nx = 5 # X, S, P, O2, V
    nu = 1 # F

    x_sym = ca.MX.sym('x', nx)
    u_sym = ca.MX.sym('u', nu)
    p_sym = {}
    for key, val in params_dict.items():
         p_sym[key] = val
    p_sym['Kla_optim_phase'] = kla_value

    # --- Lógica de la ODE usando CasADi ---
    X_ = x_sym[0]; S_ = x_sym[1]; P_ = x_sym[2]
    V_ = x_sym[4]
    F_ = u_sym[0]

    # --- Manejo de O2 ---
    if constant_O2_value is not None:
        O2_ = constant_O2_value # Usar valor constante
        dOdt = 0.0              # Fijar derivada a cero
    else:
        O2_ = x_sym[3]          # Usar estado dinámico O2

    V_safe = ca.fmax(V_, 1e-6)
    S_safe = ca.fmax(1e-9, S_); P_safe = ca.fmax(0.0, P_); O2_safe = ca.fmax(0.0, O2_)

    # --- Cálculos de mu (idénticos a la versión anterior) ---
    # mu1 (aeróbico)
    term_S_aerob = S_safe / (p_sym["Ks_aerob"] + S_safe)
    term_O2_aerob = O2_safe / (p_sym["KO_aerob"] + O2_safe)
    mu1 = p_sym["mumax_aerob"] * term_S_aerob * term_O2_aerob
    # mu2 (anaeróbico)
    den_S_anaerob = p_sym["Ks_anaerob"] + S_safe + (S_safe**2 / p_sym["KiS_anaerob"])
    term_S_anaerob = S_safe / den_S_anaerob
    base_P = 1.0 - (P_safe / p_sym["KP_anaerob"])
    term_P_anaerob = ca.power(ca.fmax(0.0, base_P), p_sym["n_p"])
    term_O2_inhib_anaerob = p_sym["KO_inhib_anaerob"] / (p_sym["KO_inhib_anaerob"] + O2_safe)
    mu2 = p_sym["mumax_anaerob"] * term_S_anaerob * term_P_anaerob * term_O2_inhib_anaerob
    mu = ca.fmax(0, mu1 + mu2)

    # --- Tasas (idénticas a la versión anterior) ---
    qP = p_sym["alpha_lp"] * mu + p_sym["beta_lp"]
    rate_P = qP * X_
    Yxs_safe = ca.fmax(p_sym["Yxs"], 1e-9); Yps_safe = ca.fmax(p_sym["Yps"], 1e-9); Yxo_safe = ca.fmax(p_sym["Yxo"], 1e-9)
    consumo_S_X = (mu / Yxs_safe) * X_
    consumo_S_P = (rate_P / Yps_safe)
    consumo_S_maint = p_sym["ms"] * X_
    rate_S = consumo_S_X + consumo_S_P + consumo_S_maint

    # --- Ecuaciones Diferenciales ---
    dXdt = (mu - p_sym["Kd"]) * X_ - (F_ / V_safe) * X_
    dSdt = -rate_S + (F_ / V_safe) * (p_sym["Sin"] - S_)
    dPdt = rate_P - (F_ / V_safe) * P_
    dVdt = F_

    # --- dOdt (solo si no es constante) ---
    if constant_O2_value is None:
        consumo_O2_X = (mu / Yxo_safe) * X_
        consumo_O2_maint = p_sym["mo"] * X_
        OUR_g = consumo_O2_X + consumo_O2_maint # [gO2/L/h]
        OUR_mg = OUR_g * 1000.0 # [mgO2/L/h]
        OTR = p_sym['Kla_optim_phase'] * (p_sym["Cs"] - O2_)
        dOdt = OTR - OUR_mg - (F_ / V_safe) * O2_

    dxdt = ca.vertcat(dXdt, dSdt, dPdt, dOdt, dVdt)

    # Crear función CasADi
    ode_casadi_func = ca.Function('ode_casadi', [x_sym, u_sym], [dxdt],
                                 ['x_in', 'u_in'], ['dxdt'])
    return ode_casadi_func

# =============================================================================
# 2. COEFICIENTES DE COLOCACIÓN (Sin cambios)
# =============================================================================
def get_hardcoded_radau_coeffs(d):
    # ... (código idéntico al anterior) ...
    if d == 2:
        C_mat = np.array([[-2.0, 2.0], [1.5, -4.5], [0.5, 2.5]])
        D_vec = np.array([0.0, 0.0, 1.0])
        return C_mat, D_vec
    else:
        raise NotImplementedError("Coeficientes hardcodeados solo para d=2.")

# =============================================================================
# 3. FUNCIONES DE PROCESO (Simulación, Optimización, Re-simulación)
#    (Modificaciones en run_optimization y resimulate)
# =============================================================================

def run_initial_simulation(y0, t_span, params, kla_aerobic, t_eval_hint):
    """Ejecuta la simulación de la fase inicial batch aeróbica (O2 dinámico)."""
    # ... (código idéntico al anterior, usa modelo_ferm_scipy) ...
    st.info("1. Ejecutando simulación de fase inicial (aeróbica)...")
    sol_initial = None
    x_final = None
    status = "Failure"
    message = ""

    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0]) * t_eval_hint) + 1)

    try:
        sol_initial = solve_ivp(modelo_ferm_scipy, t_span, y0,
                                method='BDF',
                                t_eval=t_eval,
                                args=(params, kla_aerobic),
                                atol=1e-8, rtol=1e-7)

        if sol_initial.success:
            x_final = sol_initial.y[:, -1]
            # --- IMPORTANTE: Asegurar que O2 no sea negativo ---
            x_final[3] = max(0.0, x_final[3])
            status = "Success"
            st.success("Simulación inicial completada.")
            st.write(f"Estado al inicio de la fase anaeróbica (t={t_span[1]:.2f} h):")
            st.code(f"X = {x_final[0]:.4f} g/L\nS = {x_final[1]:.4f} g/L\nP = {x_final[2]:.4f} g/L\nO2 = {x_final[3]:.4f} mg/L\nV = {x_final[4]:.4f} L")
        else:
            message = f"La simulación inicial falló: {sol_initial.message}"
            st.error(message)

    except Exception as e:
        message = f"Error durante la simulación inicial: {e}\n{traceback.format_exc()}"
        st.error(message)

    return sol_initial, x_final, status, message


def run_optimization(x_start, t_start, t_end, params, kla_anaerobic, N, d, constraints, solver_options):
    """
    Configura y resuelve el problema de optimización dinámica.
    *** USA O2 CONSTANTE durante la optimización ***
    """
    st.info(f"2. Configurando y resolviendo problema de optimización (t={t_start:.2f}h a t={t_end:.2f}h)...")
    st.warning("   -> Usando simplificación: O2 constante durante la optimización.")

    opti = ca.Opti()
    nx = 5 # X, S, P, O2, V
    nu = 1 # F
    sol = None
    results = None
    status = "Setup Failure"
    debug_info = {}

    try:
        T_optim = t_end - t_start
        if T_optim <= 0: raise ValueError("T_optim <= 0")
        dt_interval = T_optim / N
        h = dt_interval
        if d != 2: raise ValueError("d != 2")
        C_mat, D_vec = get_hardcoded_radau_coeffs(d)

        # --- Usar valor de O2 constante de la simulación inicial ---
        O2_const_optim = x_start[3]
        st.info(f"   -> Fijando O2 = {O2_const_optim:.4f} mg/L para la optimización.")
        ode_casadi_func = odefun_ferm_casadi_builder(params, kla_anaerobic,
                                                   constant_O2_value=O2_const_optim)

        # Variables de decisión y restricciones (más parecido al código funcional)
        X_col = []
        U_col = [] # Cambiado de F_col a U_col por convención
        x0_param = opti.parameter(nx)
        opti.set_value(x0_param, x_start)

        for k in range(N):
            states_k = []
            for j in range(d + 1):
                if k == 0 and j == 0:
                    states_k.append(x0_param)
                else:
                    xk_j = opti.variable(nx)
                    states_k.append(xk_j)
                    # Restricciones MÍNIMAS (como en el código funcional)
                    opti.subject_to(xk_j >= 0) # No negatividad general
                    # IMPORTANTE: No añadir xk_j[3] >= 0 si O2 es constante y positivo
                    if O2_const_optim < 0: # Chequeo por si acaso
                        st.error("O2 inicial para optimización es negativo!")
                        opti.subject_to(xk_j[3] >= 0) # Añadirla si O2 puede ser negativo
                    opti.subject_to(xk_j[1] <= constraints["S_max"]) # S <= S_max
                    opti.subject_to(xk_j[4] <= constraints["V_max"]) # V <= V_max
            X_col.append(states_k)

            Uk = opti.variable(nu)
            U_col.append(Uk)
            opti.subject_to(Uk >= constraints["F_min"])
            opti.subject_to(Uk <= constraints["F_max"])

        # Ecuaciones de Colocación y Continuidad (sin cambios)
        X_final = None
        for k in range(N):
            for j in range(1, d + 1):
                xp_kj = sum(C_mat[r, j-1] * X_col[k][r] for r in range(d + 1))
                f_kj = ode_casadi_func(X_col[k][j], U_col[k])
                opti.subject_to(h * f_kj == xp_kj)
            Xk_end = sum(D_vec[r] * X_col[k][r] for r in range(d + 1))
            if k < N - 1:
                # opti.subject_to(Xk_end == X_col[k+1][0]) # Forma concisa
                for i_ in range(nx): # Forma del código funcional
                    opti.subject_to(Xk_end[i_] == X_col[k + 1][0][i_])
            else:
                X_final = Xk_end
        if X_final is None: raise RuntimeError("X_final no determinado.")

        # Función Objetivo (sin cambios)
        P_final = X_final[2]
        V_final = X_final[4]
        opti.minimize(-(P_final * V_final))

        # Guesses iniciales (como en el código funcional)
        F_guess = 0.1 # Valor fijo del código funcional
        for k in range(N):
            opti.set_initial(U_col[k], F_guess)
            for j in range(d + 1):
                 if not (k == 0 and j == 0):
                     opti.set_initial(X_col[k][j], x_start) # Usar estado inicial como guess

        # Configurar Solver (como en el código funcional - MÁS SIMPLE)
        p_opts = {} # Sin "expand": True por defecto
        s_opts_simple = {
            "max_iter": 2000,       # Del código funcional
            "print_level": 0,
            "sb": 'yes',
            "mu_strategy": "adaptive" # Del código funcional
            # Quitar tol y constr_viol_tol explícitos, usar defaults de IPOPT
        }
        opti.solver("ipopt", p_opts, s_opts_simple) # Usar s_opts_simple
        status = "Solver Setup OK"

        # Resolver
        st.info("   -> Llamando a IPOPT (con O2 cte y config simple)...")
        sol = opti.solve()
        st.success("¡Optimización completada!")
        status = "Success"

        # Extraer resultados (sin cambios)
        F_opt_profile = np.array([sol.value(Uk) for Uk in U_col]).flatten()
        X_final_opt = sol.value(X_final)
        P_final_opt_val = X_final_opt[2]
        V_final_opt_val = X_final_opt[4]
        obj_val = P_final_opt_val * V_final_opt_val
        results = {'F_opt_profile': F_opt_profile, 'X_final_opt': X_final_opt,
                   'obj_val': obj_val, 'dt_interval': dt_interval}

    # Manejo de Errores (sin cambios respecto a la versión reestructurada)
    except RuntimeError as e:
        status = f"Solver Failure: {e}"
        st.error(status)
        if "Infeasible_Problem_Detected" in str(e):
            status = "Infeasible_Problem_Detected"
            st.warning("IPOPT detectó un problema infactible.")
            try:
                st.warning("Intentando mostrar infactibilidades:")
                debug_info['infeasibilities'] = opti.debug.show_infeasibilities(1e-5)
                st.warning("Intentando mostrar valores de variables:")
                debug_info['variables'] = opti.debug.value(opti.value_variables())
            except Exception as debug_e: st.error(f"Debug info failed: {debug_e}")
        else: # Otro runtime error
             debug_info['traceback'] = traceback.format_exc()
    except Exception as e:
        status = f"Optimization Error: {e}"
        debug_info['traceback'] = traceback.format_exc()
        st.error(status)

    return sol, status, results, debug_info


def resimulate_with_optimal_control(x_start, t_eval, F_opt_profile, dt_interval, t_anaerobic_start, params, kla_anaerobic):
    """
    Re-simula la trayectoria usando el control óptimo.
    *** USA O2 DINÁMICO para la re-simulación ***
    """
    st.info("4. Re-simulando trayectoria completa con perfil óptimo (O2 Dinámico)...")
    x_optim_traj = [x_start]
    xk_curr = x_start.copy()
    status = "Success"
    message = ""
    sim_integrator = None

    try:
        if len(t_eval) <= 1:
             st.warning("No hay puntos de tiempo suficientes para la re-simulación.")
             return np.array([x_start]), status, message

        dt_sim = t_eval[1] - t_eval[0]

        # --- Usar la ODE COMPLETA (O2 dinámico) para re-simulación ---
        ode_casadi_func_dynamic = odefun_ferm_casadi_builder(params, kla_anaerobic,
                                                            constant_O2_value=None)
        sim_integrator = ca.integrator('sim_integrator', 'idas',
                                     {'x': ode_casadi_func_dynamic.mx_in('x_in'),
                                      'p': ode_casadi_func_dynamic.mx_in('u_in'),
                                      'ode': ode_casadi_func_dynamic.mx_out('dxdt')},
                                     {'t0': 0, 'tf': dt_sim, 'reltol':1e-7, 'abstol':1e-8})

        N_intervals = len(F_opt_profile)

        for i, t_curr in enumerate(t_eval[:-1]):
            current_optim_time = t_curr - t_anaerobic_start
            k_interval = np.floor(current_optim_time / dt_interval).astype(int)
            k_interval = min(max(k_interval, 0), N_intervals - 1)
            F_current = F_opt_profile[k_interval]

             # --- APAGAR F si V >= Vmax (heurística del código funcional) ---
             # Esto puede ser útil si la optimización no respetó perfectamente Vmax
             # debido a la simplificación de O2 o las tolerancias.
            if xk_curr[4] >= params.get("V_max_input", constraints.get("V_max", float('inf'))): # Usar V_max real
                 if F_current > 0:
                     # st.write(f"Debug: V ({xk_curr[4]:.3f}) >= Vmax, forzando F=0 en t={t_curr:.2f}") # Opcional: para ver si se activa
                     F_current = 0.0

            # Simular un paso
            try:
                res_step = sim_integrator(x0=xk_curr, p=F_current)
                xk_curr = np.array(res_step['xf']).flatten()
                xk_curr = np.maximum(xk_curr, 0)
                xk_curr[0] = max(xk_curr[0], 1e-9); xk_curr[4] = max(xk_curr[4], 1e-6)
                x_optim_traj.append(xk_curr)
            except Exception as sim_step_err:
                 message += f"\nError en paso re-sim t={t_curr:.2f} F={F_current:.4f}: {sim_step_err}."
                 st.warning(message)
                 x_optim_traj.append(x_optim_traj[-1]) # Rellenar
                 # status = "Resimulation Step Failed"; break # Opcional: detener

    except Exception as e:
        status = "Resimulation Failed"
        message = f"Error durante la re-simulación: {e}\n{traceback.format_exc()}"
        st.error(message)
        return None, status, message

    return np.array(x_optim_traj), status, message


# =============================================================================
# 4. FUNCIONES AUXILIARES (Ploteo, Métricas) - Sin cambios
# =============================================================================
def plot_results(t_initial, y_initial, t_optim_phase, x_optim_traj,
                 F_opt_profile, dt_interval, t_anaerobic_start, constraints):
    # ... (código idéntico al anterior) ...
    st.subheader("📈 Resultados Gráficos")

    if x_optim_traj is None or len(x_optim_traj) == 0:
        st.warning("No hay trayectoria optimizada para graficar.")
        return None

    # Combinar trayectorias
    t_full_traj = np.concatenate([t_initial, t_optim_phase])
    y_full_traj = np.hstack([y_initial, x_optim_traj.T]) # Transponer optim traj

    # Construir perfil de flujo para plot escalonado
    t_f_steps = [t_anaerobic_start]
    F_optim_plot_steps = []
    N_intervals = len(F_opt_profile)
    t_final = t_optim_phase[-1]
    for k in range(N_intervals):
        t_start_k = t_anaerobic_start + k * dt_interval
        t_end_k = t_start_k + dt_interval
        if not np.isclose(t_f_steps[-1], t_start_k):
             t_f_steps.append(t_start_k)
             F_optim_plot_steps.append(F_opt_profile[k-1] if k>0 else F_opt_profile[0])
        t_f_steps.append(t_start_k)
        F_optim_plot_steps.append(F_opt_profile[k])
        t_f_steps.append(t_end_k)
        F_optim_plot_steps.append(F_opt_profile[k])
    if not np.isclose(t_f_steps[-1], t_final):
        t_f_steps.append(t_final)
        F_optim_plot_steps.append(F_opt_profile[-1])

    t_f_steps = np.array(t_f_steps)
    F_optim_plot_steps = np.array(F_optim_plot_steps)

    # Combinar con flujo cero inicial
    F_inicial_plot = np.zeros_like(t_initial)
    idx_insert = np.searchsorted(t_f_steps, t_initial[-1])
    t_full_F_plot = np.concatenate([t_initial, t_f_steps[idx_insert:]])
    F_full_F_plot = np.concatenate([F_inicial_plot, F_optim_plot_steps[idx_insert:]])
    first_optim_idx = np.searchsorted(t_full_F_plot, t_anaerobic_start)
    if first_optim_idx > 0: F_full_F_plot[first_optim_idx:] = F_optim_plot_steps

    # Crear Figura
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axs = axs.ravel()
    t_final_plot = t_full_traj[-1]

    # 0: Flujo
    axs[0].plot(t_f_steps, F_optim_plot_steps, label='F óptimo (step)', color='r', linewidth=2) # Plot directo de escalones
    axs[0].plot([0,t_anaerobic_start], [0,0], color='r', linewidth=2) # Fase inicial F=0
    axs[0].axhline(constraints["F_max"], color='gray', linestyle='--', label=f'F_max ({constraints["F_max"]:.2f})')
    axs[0].axhline(constraints["F_min"], color='gray', linestyle=':', label=f'F_min ({constraints["F_min"]:.2f})')
    axs[0].axvline(t_anaerobic_start, color='k', linestyle='--', label='Inicio Optim.')
    axs[0].set_title(r"Perfil Óptimo de Alimentación $F(t)$")
    axs[0].set_xlabel("Tiempo [h]")
    axs[0].set_ylabel("Flujo [L/h]")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(left=0, right=t_final_plot*1.02)
    axs[0].set_ylim(bottom=min(constraints["F_min"]*1.1, -0.01), top=constraints["F_max"]*1.1 + 0.01)

    # Plots 1-5 (Volumen, X, S, P, O2) idénticos
    # 1: Volumen
    axs[1].plot(t_full_traj, y_full_traj[4, :], label='Volumen', color='b', linewidth=2)
    axs[1].axhline(constraints["V_max"], color='r', linestyle='--', label=f'V_max ({constraints["V_max"]:.1f})')
    axs[1].axvline(t_anaerobic_start, color='k', linestyle='--')
    axs[1].set_title(r"Volumen $V(t)$")
    axs[1].set_xlabel("Tiempo [h]")
    axs[1].set_ylabel("Volumen [L]")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(left=0, right=t_final_plot*1.02)
    axs[1].set_ylim(bottom=0)

    # 2: Biomasa X
    axs[2].plot(t_full_traj, y_full_traj[0, :], label='Biomasa', color='g', linewidth=2)
    axs[2].axvline(t_anaerobic_start, color='k', linestyle='--')
    axs[2].set_title(r"Biomasa $X(t)$")
    axs[2].set_xlabel("Tiempo [h]")
    axs[2].set_ylabel("X [g/L]")
    axs[2].grid(True)
    axs[2].set_xlim(left=0, right=t_final_plot*1.02)
    axs[2].set_ylim(bottom=0)

    # 3: Sustrato S
    axs[3].plot(t_full_traj, y_full_traj[1, :], label='Sustrato', color='m', linewidth=2)
    axs[3].axhline(constraints["S_max"], color='r', linestyle='--', label=f'S_max ({constraints["S_max"]:.1f})')
    axs[3].axvline(t_anaerobic_start, color='k', linestyle='--')
    axs[3].set_title(r"Sustrato $S(t)$")
    axs[3].set_xlabel("Tiempo [h]")
    axs[3].set_ylabel("S [g/L]")
    axs[3].legend()
    axs[3].grid(True)
    axs[3].set_xlim(left=0, right=t_final_plot*1.02)
    axs[3].set_ylim(bottom=0)

    # 4: Producto P
    axs[4].plot(t_full_traj, y_full_traj[2, :], label='Etanol', color='k', linewidth=2)
    axs[4].axvline(t_anaerobic_start, color='k', linestyle='--')
    axs[4].set_title(r"Producto (Etanol) $P(t)$")
    axs[4].set_xlabel("Tiempo [h]")
    axs[4].set_ylabel("P [g/L]")
    axs[4].grid(True)
    axs[4].set_xlim(left=0, right=t_final_plot*1.02)
    axs[4].set_ylim(bottom=0)

    # 5: Oxígeno O2
    axs[5].plot(t_full_traj, y_full_traj[3, :], label='Oxígeno', color='c', linewidth=2)
    axs[5].axvline(t_anaerobic_start, color='k', linestyle='--')
    axs[5].set_title(r"Oxígeno Disuelto $O_2(t)$")
    axs[5].set_xlabel("Tiempo [h]")
    axs[5].set_ylabel("$O_2$ [mg/L]")
    axs[5].grid(True)
    axs[5].set_xlim(left=0, right=t_final_plot*1.02)
    axs[5].set_ylim(bottom=0)

    return fig

def display_final_metrics(y_full_traj, t_full_traj, x_start_optim, t_anaerobic_start):
    # ... (código idéntico al anterior) ...
    st.subheader("📊 Métricas Finales del Proceso Optimizado")
    col1, col2, col3 = st.columns(3)
    if y_full_traj is None or y_full_traj.shape[1] == 0:
        st.warning("No hay datos de trayectoria para calcular métricas.")
        return
    xf_val = y_full_traj[0, -1]; sf_val = y_full_traj[1, -1]; pf_val = y_full_traj[2, -1]
    vf_val = y_full_traj[4, -1]; tf_val = t_full_traj[-1]
    T_optim = tf_val - t_anaerobic_start
    prod_tot_final = pf_val * vf_val
    prod_vol_global = prod_tot_final / vf_val / tf_val if vf_val > 1e-6 and tf_val > 0 else 0
    pv_start_optim = x_start_optim[2] * x_start_optim[4]
    prod_vol_optim = (prod_tot_final - pv_start_optim) / vf_val / T_optim if vf_val > 1e-6 and T_optim > 0 else 0
    col1.metric("Producto Total (P*V) [g]", f"{prod_tot_final:.3f}")
    col2.metric("Concentración Final P [g/L]", f"{pf_val:.3f}")
    col3.metric("Volumen Final V [L]", f"{vf_val:.3f}")
    col1.metric("Prod. Vol. Global [g/L/h]", f"{prod_vol_global:.4f}")
    col2.metric("Concentración Final X [g/L]", f"{xf_val:.3f}")
    col3.metric("Concentración Final S [g/L]", f"{sf_val:.3f}")


# =============================================================================
# 5. APLICACIÓN STREAMLIT (Función Principal)
# =============================================================================

def drto_anaerobic_page():
    st.header("🧠 dRTO - Optimización Fase Anaeróbica (O2 Cte en Optim)")
    st.markdown(r"""
    Versión que simplifica la optimización asumiendo $O_2$ constante durante esa fase,
    basado en el valor al final de la simulación inicial aeróbica.
    La re-simulación final usa $O_2$ dinámico.
    """)

    # --- Sidebar: Entradas del Usuario (idéntico) ---
    with st.sidebar:
        # ... (copiar/pegar toda la sección de la sidebar de la versión anterior) ...
        st.subheader("1. Parámetros Cinéticos y Estequiométricos")
        st.markdown("**Parámetros Anaeróbicos (mu2):**")
        mumax_anaerob_m = st.slider("μmax_anaerob [1/h]", 0.05, 0.8, 0.15, 0.05, key="mumax_anaerob_m_rto")
        Ks_anaerob_m = st.slider("Ks_anaerob [g/L]", 0.1, 20.0, 1.0, 0.1, key="ks_anaerob_m_rto")
        KiS_anaerob_m = st.slider("KiS_anaerob [g/L]", 50.0, 500.0, 150.0, 10.0, key="kis_anaerob_m_rto")
        KP_anaerob_m = st.slider("KP_anaerob (Inhib. Etanol) [g/L]", 20.0, 150.0, 80.0, 5.0, key="kp_anaerob_m_rto")
        n_p_m = st.slider("Exponente Inhib. Etanol (n_p)", 0.5, 3.0, 1.0, 0.1, key="np_m_rto")
        KO_inhib_anaerob_m = st.slider("KO_inhib_anaerob (Inhib. O2) [mg/L]", 0.01, 5.0, 0.1, 0.01, key="ko_inhib_m_rto")
        st.markdown("**Parámetros Aeróbicos (mu1):**")
        mumax_aerob_m = st.slider("μmax_aerob [1/h]", 0.1, 1.0, 0.4, 0.05, key="mumax_aerob_m_rto")
        Ks_aerob_m = st.slider("Ks_aerob [g/L]", 0.01, 10.0, 0.5, 0.05, key="ks_aerob_m_rto")
        KO_aerob_m = st.slider("KO_aerob (afinidad O2) [mg/L]", 0.01, 5.0, 0.2, 0.01, key="ko_aerob_m_rto")
        st.markdown("**Otros Parámetros:**")
        Yxs = st.slider("Yxs [g/g]", 0.05, 0.6, 0.1, 0.01, key="yxs_rto")
        Yps = st.slider("Yps [g/g]", 0.1, 0.51, 0.45, 0.01, key="yps_rto")
        Yxo = st.slider("Yxo [gX/gO2]", 0.1, 2.0, 0.8, 0.1, key="yxo_rto")
        alpha_lp = st.slider("α [g P / g X]", 0.0, 5.0, 2.2, 0.1, key="alpha_rto")
        beta_lp = st.slider("β [g P / g X / h]", 0.0, 0.5, 0.05, 0.01, key="beta_rto")
        ms = st.slider("ms [g S / g X / h]", 0.0, 0.2, 0.02, 0.01, key="ms_rto")
        mo = st.slider("mo [gO2/gX/h]", 0.0, 0.1, 0.01, 0.005, key="mo_rto")
        Kd = st.slider("Kd [1/h]", 0.0, 0.1, 0.01, 0.005, key="kd_rto")

        st.subheader("2. Transferencia de Oxígeno")
        Kla_aerobic = st.slider("kLa (Fase Aeróbica Inicial) [1/h]", 10.0, 400.0, 100.0, 10.0, key="kla_aerobic_rto")
        Kla_anaerobic = st.slider("kLa (Fase Anaeróbica Optimizada) [1/h]", 0.0, 50.0, 1.0, 0.1, key="kla_anaerobic_rto")
        Cs = st.slider("O2 Saturado (Cs) [mg/L]", 0.01, 15.0, 7.5, 0.01, key="cs_rto")

        st.subheader("3. Configuración Temporal y Fases")
        t_anaerobic_start = st.slider("Inicio Fase Anaeróbica [h]", 1.0, 30.0, 10.0, 1.0, key="t_anaerobic_start_rto")
        t_final = st.slider("Tiempo Total [h]", t_anaerobic_start + 5.0, 100.0, t_anaerobic_start + 24.0, 1.0, key="t_final_rto")

        st.subheader("4. Configuración de la Optimización (Fase Anaeróbica)")
        N_intervals = st.number_input("Número de Intervalos (N)", min_value=5, max_value=100, value=20, step=1, key="N_intervals_rto")
        d_colloc = 2 # Hardcoded

        st.subheader("5. Alimentación y Restricciones Operativas")
        Sin = st.slider("Sin [g/L]", 10.0, 700.0, 400.0, 10.0, key="sin_rto")
        V_max = st.number_input("V_max [L]", min_value=0.1, value=10.0, step=0.1, format="%.1f", key="vmax_rto")
        S_max = st.number_input("S_max [g/L]", min_value=1.0, value=50.0, step=1.0, format="%.1f", key="smax_rto")
        F_min = st.number_input("F_min [L/h]", min_value=0.0, value=0.0, step=0.001, format="%.4f", key="fmin_rto")
        F_max = st.number_input("F_max [L/h]", min_value=F_min, value=0.5, step=0.01, format="%.4f", key="fmax_rto")

        st.subheader("6. Condiciones Iniciales (t=0)")
        V0 = st.number_input("V0 [L]", 0.1, 100.0, 5.0, key="v0_rto")
        X0 = st.number_input("X0 [g/L]", 0.05, 10.0, 0.1, key="x0_rto")
        S0 = st.number_input("S0 [g/L]", 10.0, 200.0, 100.0, key="s0_rto")
        P0 = st.number_input("P0 [g/L]", 0.0, 50.0, 0.0, key="p0_rto")
        O0 = st.number_input("O2 Inicial [mg/L]", min_value=0.0, max_value=Cs, value=Cs*0.5, step=0.01, key="o0_rto")

        st.subheader("7. Opciones del Solver (Simplificado)")
        max_iter_solver = st.number_input("Max Iteraciones IPOPT", min_value=100, max_value=10000, value=2000, step=100, key="max_iter_rto_simp")
        # Quitamos tol y constr_viol_tol explícitos para usar defaults
        # tol_solver = st.number_input("Tolerancia IPOPT (tol)", min_value=1e-9, max_value=1e-2, value=1e-6, step=1e-6, format="%.1e", key="tol_rto_simp")
        # constr_viol_tol_solver = st.number_input("Tolerancia Violación Restr. (constr_viol_tol)", min_value=1e-9, max_value=1e-2, value=1e-6, step=1e-6, format="%.1e", key="constr_viol_tol_rto_simp")


    # --- Empaquetar Entradas (idéntico) ---
    params = { # ... (copiar/pegar params de la versión anterior) ...
        "mumax_aerob": mumax_aerob_m, "Ks_aerob": Ks_aerob_m, "KO_aerob": KO_aerob_m,
        "mumax_anaerob": mumax_anaerob_m, "Ks_anaerob": Ks_anaerob_m, "KiS_anaerob": KiS_anaerob_m,
        "KP_anaerob": KP_anaerob_m, "n_p": n_p_m, "KO_inhib_anaerob": KO_inhib_anaerob_m,
        "Yxs": Yxs, "Yps": Yps, "Yxo": Yxo, "alpha_lp": alpha_lp, "beta_lp": beta_lp,
        "ms": ms, "mo": mo, "Kd": Kd, "Cs": Cs, "Sin": Sin,
        "V_max_input": V_max # Añadir V_max aquí para que esté disponible en re-simulación si es necesario
    }
    constraints = {"V_max": V_max, "S_max": S_max, "F_min": F_min, "F_max": F_max}
    solver_options_simple = { # Usar nombre diferente para claridad
        "max_iter": max_iter_solver, "print_level": 0, "sb": "yes", "mu_strategy": "adaptive"
    }
    y0_inicio = [X0, S0, P0, O0, V0]
    t_span_inicial = [0, t_anaerobic_start]

    # --- Botón de Ejecución y Flujo Principal (idéntico a la versión reestructurada) ---
    if st.button("🚀 Ejecutar Simulación y Optimización (O2 Cte en Optim)"):

        # 1. Simulación Inicial (O2 dinámico)
        sol_inicial, x_start_optim, sim_status, sim_msg = run_initial_simulation(
            y0_inicio, t_span_inicial, params, Kla_aerobic, t_eval_hint=10
        )
        if sim_status != "Success": st.stop()

        # 2. Optimización (O2 constante, config simple)
        optim_sol, optim_status, optim_results, debug_info = run_optimization(
            x_start=x_start_optim, t_start=t_anaerobic_start, t_end=t_final,
            params=params, kla_anaerobic=Kla_anaerobic, N=N_intervals, d=d_colloc,
            constraints=constraints, solver_options=solver_options_simple # Usar config simple
        )
        if optim_status != "Success":
            # ... (manejo de error de optimización como antes) ...
            st.error(f"Optimización falló: {optim_status}")
            if debug_info: st.warning("Información de depuración disponible (ver código/logs)")
            st.stop()

        # 3. Extraer Resultados y Mostrar F
        st.metric("Obj Val (Optim)", f"{optim_results['obj_val']:.3f} g")
        st.write("Perfil Óptimo F(t):")
        # ... (mostrar dataframe F_opt como antes) ...
        t_intervals_start = np.linspace(t_anaerobic_start, t_final - optim_results['dt_interval'], N_intervals)
        df_f_opt = pd.DataFrame({'T_inicio [h]': t_intervals_start, 'F_opt [L/h]': optim_results['F_opt_profile']})
        st.dataframe(df_f_opt.style.format({'T_inicio [h]': "{:.2f}", 'F_opt [L/h]': "{:.4f}"}))


        # 4. Re-simulación (O2 dinámico)
        t_optim_phase_eval = np.linspace(t_anaerobic_start, t_final, int((t_final - t_anaerobic_start) * 20) + 1)
        x_optim_traj, resim_status, resim_msg = resimulate_with_optimal_control(
            x_start=x_start_optim, t_eval=t_optim_phase_eval,
            F_opt_profile=optim_results['F_opt_profile'], dt_interval=optim_results['dt_interval'],
            t_anaerobic_start=t_anaerobic_start, params=params, kla_anaerobic=Kla_anaerobic
        )
        if resim_status != "Success": st.error(f"Re-simulación falló: {resim_msg}")

        # 5. Graficar Resultados
        fig = plot_results(
            t_initial=sol_inicial.t, y_initial=sol_inicial.y,
            t_optim_phase=t_optim_phase_eval[:len(x_optim_traj)], x_optim_traj=x_optim_traj,
            F_opt_profile=optim_results['F_opt_profile'], dt_interval=optim_results['dt_interval'],
            t_anaerobic_start=t_anaerobic_start, constraints=constraints
        )
        if fig: st.pyplot(fig)

        # 6. Mostrar Métricas Finales
        if resim_status == "Success" and x_optim_traj is not None and x_optim_traj.shape[0] > 0:
             t_full_traj = np.concatenate([sol_inicial.t, t_optim_phase_eval])
             y_full_traj = np.hstack([sol_inicial.y, x_optim_traj.T])
             display_final_metrics(y_full_traj, t_full_traj, x_start_optim, t_anaerobic_start)
        else: st.warning("No se muestran métricas finales.")


# --- Ejecución ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="dRTO Anaeróbico (O2 Cte Optim)")
    drto_anaerobic_page()