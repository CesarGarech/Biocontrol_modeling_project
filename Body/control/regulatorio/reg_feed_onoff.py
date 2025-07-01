# reg_feed_onoff.py (V3 - On-Off con Ventana de Tiempo)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import traceback

#==========================================================================
# MODELO ODE FED-BATCH CON CONTROL ON-OFF (V VARIABLE, VENTANA TIEMPO)
#==========================================================================
def sustrato_onoff_fedbatch_py(t, y, mu_max, Ks, Y_XS, Sin, V0, Fmax, S_min, S_max_ref, t_onoff_start, t_onoff_end): # Añadido t_onoff_start/end
    """
    Define las ecuaciones diferenciales para Fed-Batch con control On-Off.
    El control On-Off solo está activo entre t_onoff_start y t_onoff_end.
    """
    X, S, V = y # Desempaquetar las 3 variables de estado
    X = max(1e-9, X); S = max(0.0, S); V = max(1e-6, V) # Evitar división por cero en D
    mu = mu_max * (S / (Ks + S)) if (Ks + S) > 1e-9 else 0; mu = max(0, mu)

    # --- Lógica de Control On-Off CON VENTANA DE TIEMPO ---
    F = 0.0 # Flujo por defecto es CERO
    # Solo considera activar la bomba si estamos dentro de la ventana de tiempo
    if t >= t_onoff_start and t < t_onoff_end:
        # Dentro de la ventana, aplica la lógica On-Off basada en S_min
        if S <= S_min: # Usando <= por robustez numérica
            F = Fmax
    # Fuera de la ventana (t < t_onoff_start o t >= t_onoff_end), F siempre será 0
    # Si estamos dentro de la ventana pero S > S_min, F también será 0
    # ------------------------------------------------------
    F = max(0, F) # Asegurar F no negativo

    D = F / V # Tasa de Dilución
    dXdt = mu * X - D * X
    dSdt = - (mu / Y_XS) * X + D * (Sin - S)
    dVdt = F
    return [dXdt, dSdt, dVdt] # Devuelve las 3 derivadas

#==========================================================================
# PÁGINA STREAMLIT
#==========================================================================
def regulatorio_feed_onoff_page():
    st.header("⛽ On-Off Substrate Control Simulation (Fed-Batch with Window)")
    st.markdown("""
    This page simulates a Fed-Batch process with On-Off substrate control active **only during a specified time interval** ($t_{start}$ to $t_{end}$).
    Outside this window, the feed rate ($F$) is zero.

    * **Biological Model:** Monod.
    * **On-Off Control (Time Window):**
    - If $t_{start} \\le t < t_{end}$ AND $S \\le S_{min}$, then $F = F_{max}$.
        - In all other cases, $F = 0$.
    * **Physical Model:** Fed-batch with variable volume.
     **Equations:**
    """)
    st.latex(r'\frac{dX}{dt} = \mu X - D X')
    st.latex(r'\frac{dS}{dt} = -\frac{\mu}{Y_{XS}} X + D (S_{in} - S)')
    st.latex(r'\frac{dV}{dt} = F')
    st.latex(r'D = F / V')
    st.latex(r'\mu = \mu_{max} \frac{S}{K_s + S}')
    st.latex(r'F = \begin{cases} F_{max} & \text{if } t_{start} \le t < t_{end} \text{ and } S \le S_{min} \\ 0 & \text{in other cases} \end{cases}')
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Simulation Parameters")
        with st.expander("1. Kinetic and Stoichiometric", expanded=True):
             mu_max = st.number_input("μ_max [1/h]", 0.01, 2.0, 0.4, format="%.3f", key="onoff_mumax")
             Ks = st.number_input("Ks [g/L]", 0.01, 5.0, 0.1, format="%.3f", key="onoff_ks")
             Y_XS = st.number_input("Y_XS [gX/gS]", 0.1, 1.0, 0.5, format="%.2f", key="onoff_yxs")
             Sin = st.number_input("Sin [g/L]", 1.0, 100.0, 10.0, format="%.1f", key="onoff_sin")
        with st.expander("2. Operational", expanded=True):
             Fmax = st.number_input("Fmax [L/h]", 0.01, 1.0, 0.2, format="%.3f", key="onoff_fmax") # Ajustado a 0.2?

        # --- MODIFICADO: Control On-Off con Ventana ---
        with st.expander("3. On-Off Control", expanded=True):
            S_min = st.number_input("S_min [g/L]", 0.1, 10.0, 1.5, format="%.2f", key="onoff_smin")
            S_max = st.number_input("S_max (Visual Reference) [g/L]", S_min + 0.01, 15.0, 2.5, format="%.2f", key="onoff_smax")
            st.markdown("**Time Window for On-Off Control:**")
            # Input para inicio y fin de la ventana de control
            t_onoff_start = st.number_input("Start of Control [h]", 0.0, 100.0, 5.0, step=0.5, key="t_onoff_start", help="Time from which On-Off control can be activated.")
            t_onoff_end = st.number_input("End of Control [h]", t_onoff_start + 0.1, 100.0, 8.0, step=0.5, key="t_onoff_end", help="Time from which the feed rate will always be ZERO.")

        with st.expander("4. Initials and Simulation", expanded=True):
            X0 = st.number_input("X0 [g/L]", 0.01, 10.0, 0.1, format="%.3f", key="onoff_x0")
            S0 = st.number_input("S0 [g/L]", 0.0, 50.0, 3.0, format="%.2f", key="onoff_s0")
            V0 = st.number_input("V0 [L]", 0.1, 100.0, 1.0, format="%.2f", key="onoff_v0")
            # Quitar la advertencia sobre S0 vs S_min ya que el control ahora tiene ventana
            # if S0 <= S_min: st.warning(f"S0 ({S0:.2f}) <= S_min ({S_min:.2f}). Alimentación empezará ON.")
            t_final_onoff = st.number_input("Final Time Simulation [h]", 10.0, 200.0, 40.0, step=10.0, key="onoff_tfinal")
            # Opciones Solver
            rtol_sim = st.number_input("rtol Solver", 1e-7, 1e-3, 1e-5, format="%e", key="onoff_rtol")
            atol_sim = st.number_input("atol Solver", 1e-10, 1e-5, 1e-8, format="%e", key="onoff_atol")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("On-Off Control Simulation (Fed-Batch with Window)")

    if st.button("▶️ Simulate On-Off Control", key="run_onoff_sim_fed_window"):
        try:
            # 1. Preparar simulación con V0
            y0 = [X0, S0, V0]
            t_span = [0, t_final_onoff]
            t_eval = np.linspace(t_span[0], t_span[1], int(t_final_onoff * 50) + 1)
            # --- Pasar nuevos args (incluyendo t_onoff_start/end) a la ODE ---
            ode_args = (mu_max, Ks, Y_XS, Sin, V0, Fmax, S_min, S_max, t_onoff_start, t_onoff_end)

            # 2. Simular usando solve_ivp con la nueva ODE
            st.write(f"Simulating from t = 0 to {t_final_onoff} h...")
            sol = solve_ivp(sustrato_onoff_fedbatch_py, t_span, y0, args=ode_args, t_eval=t_eval, method='LSODA', rtol=rtol_sim, atol=atol_sim)
            if not sol.success:
                st.warning(f"Simulation with lsoda failed: {sol.message}. Trying with BDF...")
                sol = solve_ivp(sustrato_onoff_fedbatch_py, t_span, y0, args=ode_args, t_eval=t_eval, method='BDF', rtol=rtol_sim, atol=atol_sim)
                if not sol.success:
                    st.error(f"Simulation failed with both methods: {sol.message}"); st.stop()

            st.write("Simulation completed.")
            t_sim = sol.t; X_sim = np.maximum(0, sol.y[0, :]); S_sim = np.maximum(0, sol.y[1, :]); V_sim = np.maximum(1e-6, sol.y[2, :])

            # 3. Reconstruir Flujo F post-simulación (CON VENTANA DE TIEMPO)
            F_sim = np.zeros_like(t_sim)
            for i in range(len(t_sim)):
                # Lógica exacta de la ODE
                if t_sim[i] >= t_onoff_start and t_sim[i] < t_onoff_end and S_sim[i] <= S_min:
                    F_sim[i] = Fmax
                else:
                    F_sim[i] = 0

            # 4. Graficar resultados
            st.subheader("Simulation Results")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True); axes = axes.flatten()
            # Biomasa
            axes[0].plot(t_sim, X_sim, '-', color='b', linewidth=2); axes[0].set_ylabel('Biomass (g/L)'); axes[0].set_title('Biomass Evolution'); axes[0].grid(True); axes[0].set_ylim(bottom=0)
            # Sustrato
            axes[1].plot(t_sim, S_sim, '-', color='r', linewidth=2); axes[1].axhline(S_min, color='gray', linestyle='--', linewidth=1, label=f'S_min ({S_min:.2f})'); axes[1].axhline(S_max, color='gray', linestyle=':', linewidth=1, label=f'S_max Ref. ({S_max:.2f})'); axes[1].set_ylabel('Substrate (g/L)'); axes[1].set_title('Substrate Evolution (On-Off Control)'); axes[1].grid(True); axes[1].legend(loc='best'); axes[1].set_ylim(bottom=0); axes[1].set_xlabel('Time (h)')
            # Flujo
            axes[2].step(t_sim, F_sim, where='post', color='k', linewidth=1.5); axes[2].set_xlabel('Time (h)'); axes[2].set_ylabel('Feed Flow (L/h)'); axes[2].set_title('Feed Flow Profile (On-Off)'); axes[2].grid(True); axes[2].set_ylim(-0.01 * Fmax, Fmax * 1.1 + 0.01);
            # Volumen
            axes[3].plot(t_sim, V_sim, '-', color='purple', linewidth=2); axes[3].set_xlabel('Time (h)'); axes[3].set_ylabel('Volume (L)'); axes[3].set_title('Volume Evolution'); axes[3].grid(True); axes[3].set_ylim(bottom=max(0, V0*0.9))
            # Añadir líneas para ventana de control
            line_labels = {'Start Ctrl': t_onoff_start, 'End Ctrl': t_onoff_end}
            colors = {'Start Ctrl': 'limegreen', 'End Ctrl': 'tomato'}
            for i, ax in enumerate(axes):
                for label, time_val in line_labels.items():
                     ax.axvline(time_val, color=colors[label], linestyle='--', lw=1, alpha=0.9, label=label if i==0 else "_nolegend_") # Label solo en el primer eje
                ax.set_xlim(0, t_final_onoff)
            axes[0].legend(loc='best') # Para mostrar leyenda de líneas de control

            plt.tight_layout(); st.pyplot(fig)

            # 5. Mostrar tabla (opcional)
            df_results_onoff = pd.DataFrame({'Time (h)': t_sim, 'Biomass (g/L)': X_sim, 'Substrate (g/L)': S_sim, 'Volume (L)': V_sim, 'Flow (L/h)': F_sim})
            with st.expander("View simulation data"):
                st.dataframe(df_results_onoff.style.format({'Time (h)': '{:.2f}', 'Biomass (g/L)': '{:.4f}', 'Substrate (g/L)': '{:.4f}', 'Volume (L)': '{:.3f}', 'Flow (L/h)': '{:.3f}'}))
        except Exception as e: st.error(f"An error occurred during the On-Off simulation:"); st.exception(e)
    else: st.info("Set the parameters in the sidebar and click on 'Simulate On-Off Control'.")

# --- Punto de Entrada ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="On-Off Substrate Control (Fed-Batch)")
    regulatorio_feed_onoff_page()