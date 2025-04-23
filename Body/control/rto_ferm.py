# rto_fermentacion_page.py (Completo con corrección CasADi Constraint)
import streamlit as st
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import io
import traceback

# --- Contador global para llamadas a ODE de Fase 1 (para debug) ---
ode_call_count = 0

# --- Funciones Cinéticas (Intenta importar, si no, usa dummy) ---
try:
    from Utils.kinetics import mu_fermentacion_rto, mu_fermentacion
    st.sidebar.success("Utils.kinetics importado.")
except ImportError:
    st.sidebar.error("WARN: No se pudo importar `Utils.kinetics`. Usando funciones dummy.")
    def mu_fermentacion_rto(S, P, O2, mumax_a, Ks_a, KO_a, mumax_an, Ks_an, KiS_an, KP_an, n_p, KO_inh_an):
        S = ca.fmax(S, 0.0); simple_mu = 0.1 * S / (0.5 + S); return simple_mu
    def mu_fermentacion(S, P, O2, mumax_a, Ks_a, KO_a, mumax_an, Ks_an, KiS_an, KP_an, n_p, KO_inh_an):
        S = max(S, 0.0); simple_mu = 0.1 * S / (0.5 + S); return simple_mu

# --- Función ODE para Fase 1 (NumPy/SciPy con Debugging) ---
def odefun_ferm_np(t, y, p):
    global ode_call_count
    ode_call_count += 1
    if ode_call_count % 200 == 0:
         print(f"[Fase 1 ODE] t={t:.3f}, y=[X:{y[0]:.3f}, S:{y[1]:.3f}, P:{y[2]:.3f}, O2:{y[3]:.3f}, V:{y[4]:.3f}]")
    X, S, P, O2_ignored, V = y
    X = max(1e-9, X); S = max(0.0, S); P = max(0.0, P); V = max(1e-6, V)
    O2_fijo = p["O2_controlado"]
    try:
        mu = mu_fermentacion(S, P, O2_fijo, p["mumax_aerob"], p["Ks_aerob"], p["KO_aerob"], p["mumax_anaerob"], p["Ks_anaerob"], p["KiS_anaerob"], p["KP_anaerob"], p["n_p"], p["KO_inhib_anaerob"])
    except Exception as e_mu:
        print(f"ERROR calculando mu en odefun_ferm_np: {e_mu}"); mu = 0
    qP = p["alpha_lp"] * mu + p["beta_lp"]; qP = max(0.0, qP)
    Yxs_safe = max(p["Yxs"], 1e-9); Yps_safe = max(p["Yps"], 1e-9)
    consumo_S_X = (mu / Yxs_safe) * X; consumo_S_P = (qP / Yps_safe) * X
    consumo_S_maint = p["ms"] * X; rate_S = consumo_S_X + consumo_S_P + consumo_S_maint
    dOdt = 0.0; F = 0.0; D = 0.0
    dXdt = (mu - p["Kd"]) * X - D * X; dSdt = -rate_S + D * (p["Sin"] - S)
    dPdt = qP * X - D * P; dVdt = F
    derivs = [dXdt, dSdt, dPdt, dOdt, dVdt]
    if any(np.isnan(d) or np.isinf(d) for d in derivs):
         print(f"!!! ERROR NUMÉRICO DETECTADO en odefun_ferm_np !!!"); print(f"t={t:.4f}"); print(f"y={np.array(y)}"); print(f"mu={mu}, qP={qP}"); print(f"derivs={np.array(derivs)}")
         raise ValueError(f"NaN o Inf detectado en las derivadas en t={t:.4f}")
    return derivs

# --- Página Principal Streamlit ---
def rto_fermentacion_page():
    st.header("🧠 RTO - Optimización Multi-Fase para Fermentación Alcohólica")
    # ... (markdown) ...
    st.markdown("""
    Optimiza el flujo de alimentación ($F(t)$) en la **Fase 2 (Fed-Batch)** para maximizar
    la cantidad total de Etanol ($P \\times V$) al final de la **Fase 3 (Lote Final)**,
    considerando una **Fase 1 (Lote Aerobio Inicial)** con $O_2$ controlado.
    """)

    # --- Parámetros en la Sidebar ---
    with st.sidebar:
        # ... (todos los inputs de la sidebar con keys _in) ...
        st.subheader("⏳ Configuración Temporal y Fases")
        t_batch_inicial_fin_in = st.number_input("Fin Fase 1 (Lote Inicial) [h]", value=10.0, min_value=0.1, step=0.5, key="t_p1_end_in")
        t_alim_fin_in = st.number_input("Fin Fase 2 (Alimentación) [h]", value=34.0, min_value=t_batch_inicial_fin_in + 0.1, step=1.0, key="t_p2_end_in")
        t_final_in = st.number_input("Fin Fase 3 (Tiempo Total) [h]", value=46.0, min_value=t_alim_fin_in + 0.1, step=1.0, key="t_p3_end_in")
        O2_controlado_in = st.number_input("Nivel O2 Controlado (Fase 1) [mg/L]", value=0.08, min_value=0.0, step=0.01, format="%.3f", key="o2_ctrl_in")
        st.subheader("🧬 Modelo Cinético (Fermentación Mixta RTO)")
        st.info("Usando mu_fermentacion_rto (CasADi)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**μ Aerobia:**")
            mumax_aerob_in = st.number_input("μmax_aerob [1/h]", value=0.45, min_value=0.01, step=0.01, format="%.3f", key="mumax_a_rto_in")
            Ks_aerob_in = st.number_input("Ks_aerob [g/L]", value=0.5, min_value=0.01, step=0.01, format="%.2f", key="ks_a_rto_in")
            KO_aerob_in = st.number_input("KO_aerob (afin.) [mg/L]", value=0.2, min_value=0.001, step=0.01, format="%.3f", key="ko_a_rto_in")
        with c2:
            st.markdown("**μ Anaerobia:**")
            mumax_anaerob_in = st.number_input("μmax_anaerob [1/h]", value=0.15, min_value=0.01, step=0.01, format="%.3f", key="mumax_an_rto_in")
            Ks_anaerob_in = st.number_input("Ks_anaerob [g/L]", value=1.0, min_value=0.01, step=0.1, format="%.2f", key="ks_an_rto_in")
            KiS_anaerob_in = st.number_input("KiS_anaerob [g/L]", value=150.0, min_value=10.0, step=5.0, format="%.1f", key="kis_an_rto_in")
            KP_anaerob_in = st.number_input("KP_anaerob (inhib.) [g/L]", value=80.0, min_value=10.0, step=5.0, format="%.1f", key="kp_an_rto_in")
            n_p_in = st.number_input("n_p (expon. inhib P)", value=1.0, min_value=0.1, step=0.1, format="%.1f", key="np_an_rto_in")
            KO_inhib_anaerob_in = st.number_input("KO_inhib_anaerob [mg/L]", value=0.1, min_value=0.001, step=0.01, format="%.3f", key="ko_inhib_an_rto_in")
        st.subheader("📊 Parámetros Estequiométricos y Otros")
        c1, c2, c3 = st.columns(3)
        with c1:
            Yxs_in = st.number_input("Yxs [gX/gS]", value=0.1, min_value=0.01, step=0.01, format="%.2f", key="yxs_rto_in")
            Yps_in = st.number_input("Yps [gP/gS]", value=0.45, min_value=0.01, max_value=0.51, step=0.01, format="%.2f", key="yps_rto_in")
            Yxo_in = st.number_input("Yxo [gX/gO2]", value=0.8, min_value=0.01, step=0.05, format="%.2f", key="yxo_rto_in")
        with c2:
            alpha_lp_in = st.number_input("α (L-P) [gP/gX]", value=2.2, min_value=0.0, step=0.1, format="%.1f", key="alpha_rto_in")
            beta_lp_in = st.number_input("β (L-P) [gP/gX/h]", value=0.05, min_value=0.0, step=0.01, format="%.2f", key="beta_rto_in")
            ms_in = st.number_input("ms [gS/gX/h]", value=0.02, min_value=0.0, step=0.005, format="%.3f", key="ms_rto_in")
        with c3:
            mo_in = st.number_input("mo [gO2/gX/h]", value=0.01, min_value=0.0, step=0.001, format="%.3f", key="mo_rto_in")
            Kd_in = st.number_input("Kd [1/h]", value=0.01, min_value=0.0, step=0.001, format="%.3f", key="kd_rto_in")
            Kla_in = st.number_input("kLa [1/h]", value=100.0, min_value=1.0, step=10.0, format="%.1f", key="kla_rto_in")
            Cs_in = st.number_input("Cs (O2 sat.) [mg/L]", value=7.5, min_value=0.1, step=0.1, format="%.2f", key="cs_rto_in")
        st.subheader("🧪 Alimentación y Reactor")
        Sin_input_in = st.number_input("Sustrato en Alimentación (Sin) [g/L]", value=400.0, min_value=10.0, step=10.0, key="sin_rto_in")
        V_max_input_in = st.number_input("Volumen máximo del reactor [L]", value=10.0, min_value=0.1, key="vmax_rto_in")
        st.subheader("🎚 Condiciones Iniciales")
        c1, c2, c3 = st.columns(3)
        with c1:
            X0_in = st.number_input("X0 [g/L]", value=0.1, min_value=0.01, format="%.2f", key="x0_rto_in")
            S0_in = st.number_input("S0 [g/L]", value=100.0, min_value=1.0, format="%.1f", key="s0_rto_in")
        with c2:
            P0_in = st.number_input("P0 [g/L]", value=0.0, min_value=0.0, format="%.1f", key="p0_rto_in")
            O0_user_in = st.number_input("O0 [mg/L]", value=O2_controlado_in, min_value=0.0, max_value=Cs_in, format="%.3f", key="o0_rto_in")
        with c3:
            V0_in = st.number_input("V0 [L]", value=5.00, min_value=0.10, format="%.2f", key="v0_rto_in")
        st.subheader("🔧 Restricciones de Operación (Fase 2 - Fed-Batch)")
        F_min_in = st.number_input("Flujo mínimo (Fase 2) [L/h]", value=0.0, min_value=0.0, format="%.3f", key="fmin_rto_in")
        F_max_in = st.number_input("Flujo máximo (Fase 2) [L/h]", value=0.5, min_value=F_min_in, format="%.3f", key="fmax_rto_in")
        S_max_in = st.number_input("Sustrato máximo permitido [g/L]", value=50.0, min_value=0.0, key="smax_rto_in")
        st.subheader("⚙️ Parámetros del Optimizador")
        n_intervals_in = st.number_input("Número Intervalos Totales (Fase 2+3)", value=40, min_value=2, step=1, key="nint_rto_in")
        d_in = 2 # Grado de colocación (Radau)
        st.subheader("🩺 Opciones de Diagnóstico")
        debug_mode_in = st.checkbox("Modo Debug (más información)", value=False, key="debug_rto_in")
        ipopt_print_level_in = st.select_slider("Nivel de detalle IPOPT (0-12)", options=list(range(13)), value=0 if not debug_mode_in else 5, key="ipopt_verb_rto_in")
        show_infeas_on_fail_in = st.checkbox("Intentar mostrar infactibilidades al fallar", value=True, key="show_inf_rto_in")
        relax_constraints_in = st.checkbox("Relajar V_max y S_max (para testeo)", value=False, key="relax_con_rto_in")
        relax_factor_in = 1.5

    # --- Asignar valores de la sidebar a variables locales ---
    t_p1_end = t_batch_inicial_fin_in; t_p2_end = t_alim_fin_in; t_p3_end = t_final_in
    O2_controlado = O2_controlado_in; mumax_aerob = mumax_aerob_in; Ks_aerob = Ks_aerob_in
    KO_aerob = KO_aerob_in; mumax_anaerob = mumax_anaerob_in; Ks_anaerob = Ks_anaerob_in
    KiS_anaerob = KiS_anaerob_in; KP_anaerob = KP_anaerob_in; n_p = n_p_in
    KO_inhib_anaerob = KO_inhib_anaerob_in; Yxs = Yxs_in; Yps = Yps_in; Yxo = Yxo_in
    alpha_lp = alpha_lp_in; beta_lp = beta_lp_in; ms = ms_in; mo = mo_in; Kd = Kd_in
    Kla = Kla_in; Cs = Cs_in; Sin_input = Sin_input_in; V_max_input = V_max_input_in
    X0 = X0_in; S0 = S0_in; P0 = P0_in; O0_user = O0_user_in; V0 = V0_in
    F_min = F_min_in; F_max = F_max_in; S_max = S_max_in
    n_intervals = n_intervals_in; d = d_in; debug_mode = debug_mode_in
    ipopt_print_level = ipopt_print_level_in; show_infeas_on_fail = show_infeas_on_fail_in
    relax_constraints = relax_constraints_in; relax_factor = relax_factor_in

    # --- Botón de Ejecución ---
    if st.button("🚀 Ejecutar Optimización RTO Multi-Fase"):
        # --- Validaciones Iniciales ---
        proceed = True
        if t_p1_end >= t_p2_end or t_p2_end >= t_p3_end:
             st.error(f"Error: Tiempos no secuenciales (F1={t_p1_end:.1f} < F2={t_p2_end:.1f} < F3={t_p3_end:.1f})"); proceed = False
        if V0 > V_max_input:
            st.error(f"Error: V0 ({V0:.2f} L) > V_max ({V_max_input:.2f} L)"); proceed = False
        O0 = O0_user
        if O0 > Cs:
             st.warning(f"Advertencia: O0 ({O0:.3f}) > Cs ({Cs:.2f}). Usando Cs como O0."); O0 = Cs
        if not proceed: st.stop()

        st.info("Iniciando proceso RTO Multi-Fase...")
        V_max_effective = V_max_input * relax_factor if relax_constraints else V_max_input
        S_max_effective = S_max * relax_factor if relax_constraints else S_max
        if relax_constraints: st.warning(f"⚠️ Restricciones Relajadas: V_max={V_max_effective:.2f} L, S_max={S_max_effective:.2f} g/L")

        params = { # Diccionario de parámetros
            "mumax_aerob": mumax_aerob, "Ks_aerob": Ks_aerob, "KO_aerob": KO_aerob,
            "mumax_anaerob": mumax_anaerob, "Ks_anaerob": Ks_anaerob, "KiS_anaerob": KiS_anaerob,
            "KP_anaerob": KP_anaerob, "n_p": n_p, "KO_inhib_anaerob": KO_inhib_anaerob,
            "Yxs": Yxs, "Yps": Yps, "Yxo": Yxo, "alpha_lp": alpha_lp, "beta_lp": beta_lp,
            "ms": ms, "mo": mo, "Kd": Kd, "Kla": Kla, "Cs": Cs, "Sin": Sin_input,
            "O2_controlado": O2_controlado
        }

        # --- Simulación Fase 1 ---
        st.write("---"); st.write(f"**1. Simulando Fase 1 (0 a {t_p1_end:.1f} h, O2={O2_controlado:.3f} mg/L)...**")
        global ode_call_count; ode_call_count = 0
        y0_fase1 = [X0, S0, P0, O0, V0]; t_eval_fase1 = np.linspace(0, t_p1_end, 100)
        try:
            st.write("...Preparando llamada a solve_ivp para Fase 1...")
            sol_fase1 = solve_ivp(odefun_ferm_np, [0, t_p1_end], y0_fase1, t_eval=t_eval_fase1, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
            st.write("...Llamada a solve_ivp para Fase 1 COMPLETADA.")
            if not sol_fase1.success:
                st.error(f"Fallo en la simulación de Fase 1: {sol_fase1.message}"); st.text_area("Detalles sol:", str(sol_fase1), height=150); st.stop()
            x_at_p1_end = sol_fase1.y[:, -1].copy(); x_at_p1_end[3] = O2_controlado
            st.success(f"Estado al final de Fase 1 (t={t_p1_end:.1f}h):")
            st.markdown(f"&nbsp;&nbsp;X={x_at_p1_end[0]:.3f}, S={x_at_p1_end[1]:.3f}, P={x_at_p1_end[2]:.3f}, O2={x_at_p1_end[3]:.3f}, V={x_at_p1_end[4]:.3f}")
            t_traj_p1 = sol_fase1.t; x_traj_p1 = sol_fase1.y.T.copy(); x_traj_p1[:, 3] = O2_controlado
            infeasible_start = False
            if x_at_p1_end[4] > V_max_effective + 1e-6: st.error(f"Error Crítico Post-F1: V ({x_at_p1_end[4]:.3f}) > V_max ({V_max_effective:.2f})"); infeasible_start = True
            elif x_at_p1_end[4] > V_max_effective * 0.99: st.warning(f"Advertencia Post-F1: V ({x_at_p1_end[4]:.3f}) cerca de V_max ({V_max_effective:.2f})")
            if x_at_p1_end[1] > S_max_effective + 1e-6: st.warning(f"Advertencia Mayor Post-F1: S ({x_at_p1_end[1]:.3f}) > S_max ({S_max_effective:.2f})")
            if infeasible_start: st.stop()
        except ValueError as e_val:
            st.error(f"Error Numérico DURANTE Fase 1: {e_val}"); st.error("Revisa la consola por NaNs/Infs."); st.error(traceback.format_exc()); st.stop()
        except Exception as e_fase1:
             st.error(f"Error General DURANTE Fase 1: {e_fase1}"); st.error(traceback.format_exc()); st.stop()

        # --- Configuración de Optimización (Fase 2 + Fase 3) ---
        st.write("---"); st.write(f"**2. Configurando Optimización (F2 [{t_p1_end:.1f}-{t_p2_end:.1f}h] + F3 [{t_p2_end:.1f}-{t_p3_end:.1f}h])...**")
        try:
            # Definición ODE CasADi
            x_sym = ca.SX.sym('x', 5); u_sym = ca.SX.sym('u'); p_sym_dict = {k: ca.SX.sym(k) for k in params}
            def odefun_casadi(x, u, p_dict):
                X_, S_, P_, O2_, V_ = x[0], x[1], x[2], x[3], x[4]
                X_ = ca.fmax(X_, 1e-9); S_ = ca.fmax(S_, 0.0); P_ = ca.fmax(P_, 0.0); O2_ = ca.fmax(O2_, 0.0); V_ = ca.fmax(V_, 1e-6)
                mu = mu_fermentacion_rto(S_, P_, O2_, p_dict["mumax_aerob"], p_dict["Ks_aerob"], p_dict["KO_aerob"], p_dict["mumax_anaerob"], p_dict["Ks_anaerob"], p_dict["KiS_anaerob"], p_dict["KP_anaerob"], p_dict["n_p"], p_dict["KO_inhib_anaerob"])
                qP = p_dict["alpha_lp"] * mu + p_dict["beta_lp"]; qP = ca.fmax(0.0, qP)
                consumo_S_X = (mu / ca.fmax(p_dict["Yxs"], 1e-9)) * X_; consumo_S_P = (qP / ca.fmax(p_dict["Yps"], 1e-9)) * X_
                consumo_S_maint = p_dict["ms"] * X_; rate_S = consumo_S_X + consumo_S_P + consumo_S_maint
                consumo_O2_X = (mu / ca.fmax(p_dict["Yxo"], 1e-9)) * X_; consumo_O2_maint = p_dict["mo"] * X_
                OUR_mg_per_L_h = (consumo_O2_X + consumo_O2_maint) * 1000.0
                OTR_mg_per_L_h = p_dict["Kla"] * (p_dict["Cs"] - O2_); D = u / V_
                dX = (mu - p_dict["Kd"]) * X_ - D * X_; dS = -rate_S + D * (p_dict["Sin"] - S_)
                dP = qP * X_ - D * P_; dO = OTR_mg_per_L_h - OUR_mg_per_L_h - D * O2_; dV = u
                return ca.vertcat(dX, dS, dP, dO, dV)

            # Coeficientes Radau
            def radau_coefficients(d_in):
                if d_in == 2: C_mat, D_vec = np.array([[-2.,2.],[1.5,-4.5],[.5,2.5]]), np.array([0.,0.,1.]); return C_mat, D_vec
                else: raise NotImplementedError(f"d={d_in} no implementado.")
            C_radau, D_radau = radau_coefficients(d); nx = 5

            # Configuración NLP
            opti = ca.Opti("conic"); t_opt_start = t_p1_end; t_opt_end = t_p3_end
            opt_duration = t_opt_end - t_opt_start
            if opt_duration <= 1e-6: st.error(f"Error: Duración optimización no positiva ({opt_duration:.3f})"); st.stop()
            n_opt_intervals = n_intervals; dt_opt = opt_duration / n_opt_intervals; h_coll = dt_opt
            t_p2_relative = t_p2_end - t_opt_start
            if abs(t_p2_relative % dt_opt)<1e-6*dt_opt : n_intervals_p2 = int(round(t_p2_relative / dt_opt))
            else: n_intervals_p2 = int(np.floor(t_p2_relative / dt_opt))
            n_intervals_p2 = min(n_opt_intervals, max(0, n_intervals_p2)); n_intervals_p3 = n_opt_intervals - n_intervals_p2
            st.info(f"Optimizando {n_opt_intervals} intervalos ({dt_opt:.3f} h/int): {n_intervals_p2} F2, {n_intervals_p3} F3.")
            if n_intervals_p2==0 and F_max>0 and t_p2_end>t_p1_end: st.warning("WARN: No hay intervalos en Fase 2.")
            if n_intervals_p3==0 and t_p3_end>t_p2_end: st.warning("WARN: No hay intervalos en Fase 3.")

            # Crear variables CasADi
            X_col = []; F_col = []; xk0_param = opti.parameter(nx); opti.set_value(xk0_param, x_at_p1_end)
            st.write("...Creando variables de optimización CasADi...")
            for k in range(n_opt_intervals):
                row_states = []
                for j in range(d + 1):
                    if j == 0:
                        if k == 0: row_states.append(xk0_param)
                        else: xk_j = opti.variable(nx); row_states.append(xk_j)
                    else: xk_j = opti.variable(nx); row_states.append(xk_j)
                    # Aplicar bounds básicos solo a variables (j>0 o k>0)
                    if not (k==0 and j==0):
                         opti.subject_to(xk_j[0] >= -1e-7); opti.subject_to(xk_j[1] >= -1e-7)
                         opti.subject_to(xk_j[2] >= -1e-7); opti.subject_to(xk_j[3] >= -1e-7)
                         opti.subject_to(xk_j[4] >= V0 * 0.9) # Evita V negativo/decreciente
                X_col.append(row_states)
                Fk = opti.variable()
                F_col.append(Fk)
                is_in_phase2 = (k < n_intervals_p2)
                if is_in_phase2: opti.subject_to(opti.bounded(F_min, Fk, F_max))
                else: opti.subject_to(Fk == 0.0)
            st.write("...Variables creadas.")

            # ===>>> INICIO BLOQUE CORREGIDO <<<===
            # Restricciones de Estado (S_max, V_max)
            st.write("...Aplicando restricciones de estado (S_max, V_max)...")
            for k in range(n_opt_intervals):
                for j in range(d + 1): # Iterar sobre todos los puntos definidos (0 a d)
                    # --- *** CORRECCIÓN AQUÍ *** ---
                    # No aplicar restricciones de trayectoria al parámetro inicial fijo (k=0, j=0)
                    if k == 0 and j == 0:
                        continue # Saltar a la siguiente iteración
                    # --- Aplicar restricciones a los demás puntos (que son variables) ---
                    opti.subject_to(X_col[k][j][1] <= S_max_effective + 1e-6) # S <= S_max
                    opti.subject_to(X_col[k][j][4] <= V_max_effective + 1e-6) # V <= V_max
            st.write("...Restricciones de estado aplicadas.")
            # ===>>> FIN BLOQUE CORREGIDO <<<===


            # Ecuaciones de Colocación y Continuidad
            st.write("...Aplicando ecuaciones de colocación y continuidad...")
            p_fixed_sx = ca.struct_SX([(k, p_sym_dict[k]) for k in params])
            p_fixed_val = ca.DM(list(params.values()))
            for k in range(n_opt_intervals):
                for j in range(1, d + 1):
                    xp_kj = sum(C_radau[r, j - 1] * X_col[k][r] for r in range(d + 1))
                    fkj = odefun_casadi(X_col[k][j], F_col[k], p_sym_dict)
                    coll_eq = h_coll * fkj - xp_kj
                    opti.subject_to(opti.fshow(coll_eq, p_fixed_sx) == 0)
                Xk_end = sum(D_radau[r] * X_col[k][r] for r in range(d + 1))
                if k < n_opt_intervals - 1:
                    for state_idx in range(nx): opti.subject_to(Xk_end[state_idx] == X_col[k+1][0][state_idx])
                else: X_final = Xk_end
            st.write("...Ecuaciones aplicadas.")

            # Función Objetivo
            P_final = X_final[2]; V_final = X_final[4]; objective = -(P_final * V_final)
            opti.minimize(objective)

            # Guesses Iniciales
            st.write("...Calculando initial guess para optimización...")
            # ... (código del guess sin cambios) ...
            x_guess = [x_at_p1_end.copy()]
            f_guess_val = (F_min + F_max) / 2.0
            xk_ = x_at_p1_end.copy()
            try:
                 ode_func_casadi_guess = ca.Function('ode_func_guess', [x_sym, u_sym, p_fixed_sx], [odefun_casadi(x_sym, u_sym, p_sym_dict)])
                 dae_guess = {'x': x_sym, 'p': ca.vertcat(u_sym, p_fixed_sx), 'ode': ode_func_casadi_guess(x_sym, u_sym, p_fixed_sx)}
                 integrator_guess = ca.integrator('int_guess', 'rk', dae_guess, {'tf': dt_opt, 'number_of_finite_elements': 2})
                 for k in range(n_opt_intervals):
                     current_f = f_guess_val if k < n_intervals_p2 else 0.0
                     if xk_[4] > V_max_effective * 0.95: current_f = 0.0
                     if xk_[1] > S_max_effective * 0.95 and k < n_intervals_p2 : current_f = F_min
                     p_sim_guess = ca.vertcat(current_f, p_fixed_val)
                     res_ = integrator_guess(x0=xk_, p=p_sim_guess)
                     xk_ = np.array(res_['xf']).flatten()
                     xk_[0] = max(xk_[0], 1e-7); xk_[1] = max(xk_[1], 0.0); xk_[2] = max(xk_[2], 0.0); xk_[3] = max(xk_[3], 0.0); xk_[4] = max(xk_[4], V0 * 0.9)
                     xk_[1] = min(xk_[1], S_max_effective*1.1); xk_[4] = min(xk_[4], V_max_effective*1.1)
                     x_guess.append(xk_.copy())
                 st.success("Initial guess basado en simulación simple generado.")
            except Exception as e_guess:
                 st.warning(f"Fallo al generar initial guess simulado: {e_guess}. Usando guess constante.")
                 x_guess = [x_at_p1_end] * (n_opt_intervals + 1)
            # Asignar guesses
            for k in range(n_opt_intervals):
                 is_in_phase2_guess = (k < n_intervals_p2)
                 F_guess_k = f_guess_val if is_in_phase2_guess else 0.0
                 if x_guess[k][4] > V_max_effective * 0.95 and is_in_phase2_guess: F_guess_k = F_min
                 opti.set_initial(F_col[k], F_guess_k)
                 x_start_interval = x_guess[k]; x_end_interval = x_guess[k+1]
                 opti.set_initial(X_col[k][0], x_start_interval)
                 for j in range(1, d + 1):
                     alpha = j / d; x_interp_guess = (1 - alpha) * x_start_interval + alpha * x_end_interval
                     opti.set_initial(X_col[k][j], x_interp_guess)
            st.write("...Initial guess asignado.")

        except Exception as e_setup:
             st.error(f"Error durante la configuración del problema de optimización CasADi: {e_setup}")
             st.error(traceback.format_exc())
             st.stop()

        # --- Resolver NLP ---
        # ... (resto del código: llamada a IPOPT, análisis de resultados, ploteo, métricas) ...
        # ... (sin cambios respecto a la versión anterior con diagnóstico mejorado de IPOPT) ...
        st.write("**3. Resolviendo el problema de optimización (Fase 2+3)...**")
        p_opts = {
            "expand": True, "ipopt.print_level": ipopt_print_level, "ipopt.sb": "yes",
            "ipopt.max_iter": 1500, "ipopt.tol": 1e-6, "ipopt.constr_viol_tol": 1e-6,
            "ipopt.ma57_automatic_scaling": "yes"
        }
        s_opts = {}
        solver_output = io.StringIO(); original_stdout = sys.stdout
        if debug_mode or ipopt_print_level > 0: sys.stdout = solver_output
        try:
            opti.solver("ipopt", p_opts, s_opts)
            sol = opti.solve()
            sys.stdout = original_stdout # Restaurar
            solver_log = solver_output.getvalue()
            if debug_mode or ipopt_print_level > 4: st.text_area("Log IPOPT:", solver_log, height=300)
            # Análisis detallado del resultado (igual que antes)
            stats = sol.stats(); success = stats.get('success', False); return_status = stats.get('return_status', 'N/A')
            if success and 'Solve_Succeeded' in return_status: st.success(f"Optimización finalizada con éxito. Estado: {return_status}")
            else:
                st.warning(f"IPOPT finalizó con estado: {return_status} (Éxito={success})")
                if 'Infeasible' in return_status:
                    st.error("🚨 PROBLEMA INFACTIBLE detectado."); st.subheader("Posibles Causas:")
                    msg = """* V_max: Eff={v_max_eff:.2f}, V_i={v_p1_end:.3f}\n* S_max: Eff={s_max_eff:.2f}, S_i={s_p1_end:.3f}\n* F:[{f_min:.3f}-{f_max:.3f}]\n* Otros: Tiempos, Modelo, Params, Numérica?"""
                    st.error(msg.format(v_max_eff=V_max_effective, v_p1_end=x_at_p1_end[4], s_max_eff=S_max_effective, s_p1_end=x_at_p1_end[1], f_min=F_min, f_max=F_max))
                    if show_infeas_on_fail:
                        try: st.text_area("Infactibilidades:", opti.debug.show_infeasibilities(tol=1e-4), height=200)
                        except Exception as dbg_e: st.warning(f"No se pudo mostrar infact.: {dbg_e}")
                    st.stop()
                elif 'Maximum_Iterations_Exceeded' in return_status: st.warning("IPOPT Max Iter. Solución puede no ser óptima.")
                elif 'Restoration_Failed' in return_status:
                    st.error("🚨 Falló Fase de Restauración IPOPT.");
                    if show_infeas_on_fail:
                        try: st.text_area("Infactibilidades:", opti.debug.show_infeasibilities(tol=1e-4), height=200)
                        except Exception as dbg_e: st.warning(f"No se pudo mostrar infact.: {dbg_e}")
                    st.stop()
                else:
                    st.error(f"Optimización falló o no convergió. Estado: {return_status}")
                    try: st.write(f"Debug: F[0]={opti.debug.value(F_col[0]):.4f}, Obj={opti.debug.value(objective):.4f}")
                    except Exception as dbg_e: st.warning(f"No se pudo obtener valor debug: {dbg_e}")
                    st.stop()
            # Extracción de Resultados
            st.write("Extrayendo resultados...")
            F_opt_vals = np.array([sol.value(fk) for fk in F_col]); F_opt_vals[n_intervals_p2:] = 0.0
            x_opt_sol = np.zeros((n_opt_intervals + 1, nx)); x_opt_sol[n_opt_intervals, :] = sol.value(X_final)
            for k in range(n_opt_intervals): x_opt_sol[k, :] = sol.value(X_col[k][0])
            X_final_opt = x_opt_sol[-1, :]; P_final_opt = X_final_opt[2]; V_final_opt = X_final_opt[4]; obj_val = P_final_opt * V_final_opt
            st.metric("Producto Total Final (P*V)", f"{obj_val:.4f} g")
            st.write(f"Estado final optimizado (t={t_p3_end:.1f}h):"); st.markdown(f"&nbsp;X={X_final_opt[0]:.3f}, S={X_final_opt[1]:.3f}, P={X_final_opt[2]:.3f}, O2={X_final_opt[3]:.3f}, V={V_final_opt:.3f}")
            if X_final_opt[4] > V_max_effective + p_opts["ipopt.constr_viol_tol"] * 1.1: st.warning(f"⚠️ Solución final viola V_max")
            if X_final_opt[1] > S_max_effective + p_opts["ipopt.constr_viol_tol"] * 1.1: st.warning(f"⚠️ Solución final viola S_max")

            # Reconstrucción y Ploteo
            st.write("---"); st.write("**4. Reconstruyendo trayectoria (Simulación Verificación)...**")
            # ... (Código de simulación/ploteo sin cambios) ...
            try:
                 ode_func_casadi_sim = ca.Function('ode_func_sim', [x_sym, u_sym, p_fixed_sx], [odefun_casadi(x_sym, u_sym, p_sym_dict)])
                 dae_sim = {'x': x_sym, 'p': ca.vertcat(u_sym, p_fixed_sx), 'ode': ode_func_casadi_sim(x_sym, u_sym, p_fixed_sx)}
                 integrator_sim = ca.integrator("integrator_sim", "idas", dae_sim, {"tf": dt_opt, "abstol": 1e-7, "reltol": 1e-7})
                 t_sim = np.linspace(t_opt_start, t_opt_end, n_opt_intervals + 1); x_sim_traj = [x_at_p1_end.copy()]; xk_sim = x_at_p1_end.copy(); F_sim_plot = []
                 for k in range(n_opt_intervals):
                     F_current = F_opt_vals[k];
                     if xk_sim[4] >= V_max_effective - 1e-5: F_current = 0.0
                     F_sim_plot.append(F_current); p_sim = ca.vertcat(F_current, p_fixed_val); res_sim = integrator_sim(x0=xk_sim, p=p_sim)
                     xk_sim = np.array(res_sim['xf']).flatten(); x_sim_traj.append(xk_sim.copy())
                 x_sim_traj = np.array(x_sim_traj); F_sim_plot = np.array(F_sim_plot)
                 t_full = np.concatenate([t_traj_p1[:-1], t_sim]); x_full = np.vstack([x_traj_p1[:-1, :], x_sim_traj])
                 F_plot_p1 = np.zeros(len(t_traj_p1)-1)
                 if len(F_sim_plot) < n_opt_intervals: F_sim_plot = np.append(F_sim_plot, [0.0]*(n_opt_intervals-len(F_sim_plot)))
                 F_plot_opt_adj = np.repeat(F_sim_plot, 1); F_plot = np.concatenate([F_plot_p1, F_plot_opt_adj])
                 if len(F_plot) < len(t_full): F_plot = np.append(F_plot, F_plot[-1])
                 elif len(F_plot) > len(t_full): F_plot = F_plot[:len(t_full)]
                 X_traj, S_traj, P_traj, O2_traj, V_traj = x_full.T
                 # Ploteo
                 st.subheader("📈 Resultados Simulación con Flujo Óptimo"); fig, axs = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True); axs = axs.ravel()
                 for ax in axs: ax.axvline(t_p1_end, color='grey', linestyle='--', lw=1.5); ax.axvline(t_p2_end, color='purple', linestyle='--', lw=1.5); ax.grid(True, which='both', linestyle=':', linewidth=0.5)
                 axs[0].plot(t_full, F_plot, 'b-', label='$F(t)$ Sim.'); axs[0].axhline(F_max, color='red', linestyle=':', lw=1.5, label=f'$F_{{max}}$'); axs[0].set_title("Perfil Flujo $F(t)$"); axs[0].set_ylabel("[L/h]"); axs[0].legend(); axs[0].set_ylim(bottom=F_min - 0.05*F_max if F_max > 0 else -0.01)
                 axs[1].plot(t_full, V_traj, 'k-', label='$V(t)$'); axs[1].axhline(V_max_effective, color='red', linestyle=':', lw=1.5, label=f'$V_{{max}}${"(R)" if relax_constraints else ""}'); axs[1].set_title("Volumen $V(t)$"); axs[1].set_ylabel("[L]"); axs[1].legend(); axs[1].set_ylim(bottom=0)
                 axs[2].plot(t_full, X_traj, 'g-', label='$X(t)$'); axs[2].set_title("Biomasa $X(t)$"); axs[2].set_ylabel("[g/L]"); axs[2].set_ylim(bottom=0)
                 axs[3].plot(t_full, S_traj, 'm-', label='$S(t)$'); axs[3].axhline(S_max_effective, color='red', linestyle=':', lw=1.5, label=f'$S_{{max}}${"(R)" if relax_constraints else ""}'); axs[3].set_title("Sustrato $S(t)$"); axs[3].set_ylabel("[g/L]"); axs[3].legend(); axs[3].set_ylim(bottom=-0.1)
                 axs[4].plot(t_full, P_traj, 'r-', label='$P(t)$'); axs[4].set_title("Producto $P(t)$"); axs[4].set_ylabel("[g/L]"); axs[4].set_ylim(bottom=0)
                 axs[5].plot(t_full, O2_traj, 'c-', label='$O_2(t)$'); axs[5].axhline(O2_controlado, color='orange', linestyle=':', lw=1, label=f'$O_2$ F1'); axs[5].set_title("Oxígeno Disuelto $O_2(t)$"); axs[5].set_ylabel("[mg/L]"); axs[5].legend(); axs[5].set_ylim(bottom=-0.01)
                 for ax in axs:
                    ax.set_xlabel("Tiempo [h]"); ylim = ax.get_ylim()
                    if ylim[1]>ylim[0]: ax.text(t_p1_end/2,ylim[1]*.95,"F1",ha='center',va='top',bbox=dict(fc='grey',alpha=.3)); ax.text((t_p1_end+t_p2_end)/2,ylim[1]*.95,"F2",ha='center',va='top',bbox=dict(fc='purple',alpha=.3)); ax.text((t_p2_end+t_p3_end)/2,ylim[1]*.95,"F3",ha='center',va='top',bbox=dict(fc='blue',alpha=.3))
                 st.pyplot(fig)
                 # Métricas
                 st.subheader("📊 Métricas Clave"); col1, col2, col3 = st.columns(3)
                 with col1: st.metric("Prod. Total Final", f"{obj_val:.3f} g"); st.metric("Vol. Final", f"{V_final_opt:.3f} L")
                 with col2: st.metric("Conc. Final P", f"{P_final_opt:.3f} g/L"); st.metric("Conc. Final X", f"{X_final_opt[0]:.3f} g/L")
                 with col3:
                     S_fed_total = 0; n_intervals_p2_actual = min(n_opt_intervals, max(0, int(np.floor((t_p2_end - t_p1_end) / dt_opt))))
                     for k in range(n_intervals_p2_actual): S_fed_total += F_opt_vals[k] * params['Sin'] * dt_opt
                     S_init_total = S0 * V0; S_consumed_total = (S_init_total + S_fed_total) - (X_final_opt[1] * V_final_opt)
                     rend_global_PS = obj_val / S_consumed_total if S_consumed_total > 1e-6 else 0
                     st.metric("Rendim. Global P/S", f"{rend_global_PS:.3f} g/g")
                     prod_vol = obj_val / V_final_opt / t_p3_end if V_final_opt > 1e-6 and t_p3_end > 0 else 0
                     st.metric("Productiv. Vol. Global", f"{prod_vol:.3f} g/L/h")
            except Exception as e_plot:
                st.error(f"Error durante reconstrucción/ploteo: {e_plot}"); st.error(traceback.format_exc())
        except RuntimeError as e_runtime:
            sys.stdout = original_stdout; solver_log = solver_output.getvalue(); st.error(f"Runtime Error Grave: {e_runtime}"); st.error("Problemas numéricos/memoria?")
            if solver_log: st.text_area("Log IPOPT (Previo Error):", solver_log, height=300)
            try: st.write(f"Debug F[0]={opti.debug.value(F_col[0]):.4f}")
            except Exception as dbg_e: st.warning(f"No se pudo obtener debug: {dbg_e}")
            st.stop()
        except Exception as e_main:
            sys.stdout = original_stdout; st.error(f"Error Inesperado Principal: {e_main}"); st.error(traceback.format_exc()); st.stop()

# --- Ejecución ---
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="RTO Fermentación Multi-Fase")
    # --- Bloque dummy Utils si no se encuentra el módulo real ---
    try: from Utils.kinetics import mu_fermentacion_rto, mu_fermentacion; print("Utils.kinetics importado en __main__")
    except ImportError:
        print("WARN: Utils.kinetics no encontrado en __main__. Creando/usando dummy.")
        try: from Utils.kinetics import mu_fermentacion_rto, mu_fermentacion
        except ImportError:
            if 'Utils' not in sys.modules:
                 from types import ModuleType; utils = ModuleType('Utils'); sys.modules['Utils'] = utils
                 utils_kinetics = ModuleType('Utils.kinetics'); sys.modules['Utils.kinetics'] = utils_kinetics
                 def mu_fermentacion_rto(S, P, O2, *args): S=ca.fmax(S,0); return 0.1*S/(0.5+S)
                 def mu_fermentacion(S, P, O2, *args): S=max(S,0); return 0.1*S/(0.5+S)
                 utils_kinetics.mu_fermentacion_rto = mu_fermentacion_rto; utils_kinetics.mu_fermentacion = mu_fermentacion
                 print("Funciones dummy para Utils.kinetics creadas.")
    # Ejecutar la página
    print("Ejecutando rto_fermentacion_page()..."); rto_fermentacion_page(); print("rto_fermentacion_page() finalizada.")