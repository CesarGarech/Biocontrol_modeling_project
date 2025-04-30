# reg_feed_onoff.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import traceback

#==========================================================================
# MODELO ODE CON CONTROL ON-OFF (Python version)
#==========================================================================
# (Función ODE sin cambios)
def sustrato_onoff_py(t, y, mu_max, Ks, Y_XS, Sin, V, Fmax, S_min, S_max_ref):
    X, S = y
    X = max(1e-9, X); S = max(0.0, S)
    mu = mu_max * (S / (Ks + S)) if (Ks + S) > 1e-9 else 0; mu = max(0, mu)
    if S < S_min: F = Fmax
    else: F = 0
    F = max(0, F)
    # Asumiendo V constante para replicar MATLAB
    dXdt = mu * X - (F / V) * X
    dSdt = - (mu / Y_XS) * X + (F / V) * (Sin - S)
    return [dXdt, dSdt]

#==========================================================================
# PÁGINA STREAMLIT
#==========================================================================
def regulatorio_feed_onoff_page():
    st.header("⛽ Simulación de Control On-Off de Sustrato")
    st.markdown("""
    Esta página simula un biorreactor simple donde la concentración de sustrato
    se controla mediante una estrategia **On-Off** aplicada al flujo de alimentación ($F$).

    * **Modelo Biológico:** Crecimiento celular simple siguiendo la cinética de Monod.
    * **Control On-Off (Simple):**
        - Si el sustrato ($S$) cae por debajo de un umbral mínimo ($S_{min}$), la bomba de alimentación se enciende a un flujo máximo ($F_{max}$).
        - Si el sustrato ($S$) es mayor o igual a $S_{min}$, la bomba se apaga ($F=0$). (Nota: No hay banda muerta o histéresis en esta implementación simple).
    * **Modelo Físico:** Se asume un volumen constante $V$ y que $F$ representa un flujo neto que causa dilución (para replicar el ejemplo original). Un modelo Fed-Batch más realista tendría un volumen variable.

    **Ecuaciones:**
    """)
    st.latex(r'\frac{dX}{dt} = \mu X - \frac{F}{V} X')
    st.latex(r'\frac{dS}{dt} = -\frac{\mu}{Y_{XS}} X + \frac{F}{V} (S_{in} - S)')
    st.latex(r'\mu = \mu_{max} \frac{S}{K_s + S}')
    st.latex(r'F = \begin{cases} F_{max} & \text{si } S < S_{min} \\ 0 & \text{si } S \ge S_{min} \end{cases}')
    st.caption("Nota: Lógica On-Off sin histéresis. Modelo físico asume V constante.")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        # (Inputs de sidebar sin cambios)
        st.header("Parámetros de Simulación")
        with st.expander("1. Parámetros Cinéticos y Estequiométricos", expanded=True):
            mu_max = st.number_input("Tasa Máx. Crecimiento (μ_max) [1/h]", 0.01, 2.0, 0.4, format="%.3f", key="onoff_mumax")
            Ks = st.number_input("Constante Saturación (Ks) [g/L]", 0.01, 5.0, 0.1, format="%.3f", key="onoff_ks")
            Y_XS = st.number_input("Rendimiento (Y_XS) [gX/gS]", 0.1, 1.0, 0.5, format="%.2f", key="onoff_yxs")
            Sin = st.number_input("Sustrato Alimentación (Sin) [g/L]", 1.0, 100.0, 10.0, format="%.1f", key="onoff_sin")
        with st.expander("2. Parámetros Operacionales", expanded=True):
            V = st.number_input("Volumen Reactor (V) [L]", 0.1, 100.0, 1.0, format="%.2f", key="onoff_v", help="Asumido constante.")
            Fmax = st.number_input("Flujo Máx. Alimentación (Fmax) [L/h]", 0.01, 1.0, 0.1, format="%.3f", key="onoff_fmax")
        with st.expander("3. Parámetros Control On-Off", expanded=True):
            S_min = st.number_input("Umbral Inferior Sustrato (S_min) [g/L]", 0.1, 10.0, 1.5, format="%.2f", key="onoff_smin")
            S_max_ref = st.number_input("Umbral Superior Sustrato (S_max) [g/L]", S_min + 0.1, 15.0, 2.5, format="%.2f", key="onoff_smax", help="Referencia. Bomba apaga si S >= S_min.")
        with st.expander("4. Condiciones Iniciales y Simulación", expanded=True):
            X0 = st.number_input("Biomasa Inicial (X0) [g/L]", 0.01, 10.0, 0.1, format="%.3f", key="onoff_x0")
            S0 = st.number_input("Sustrato Inicial (S0) [g/L]", 0.0, 50.0, 3.0, format="%.2f", key="onoff_s0")
            if S0 < S_min: st.warning(f"S0 ({S0:.2f}) < S_min ({S_min:.2f}). Alimentación empezará ON.")
            t_final_onoff = st.number_input("Tiempo Final Simulación [h]", 10.0, 200.0, 40.0, step=10.0, key="onoff_tfinal")

    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Simulación del Control On-Off")

    if st.button("▶️ Simular Control On-Off", key="run_onoff_sim"):
        try:
            y0 = [X0, S0]; t_span = [0, t_final_onoff]
            t_eval = np.linspace(t_span[0], t_span[1], int(t_final_onoff * 20) + 1)
            ode_args = (mu_max, Ks, Y_XS, Sin, V, Fmax, S_min, S_max_ref)
            st.write(f"Simulando para t = 0 a {t_final_onoff} h...")
            sol = solve_ivp(sustrato_onoff_py, t_span, y0, args=ode_args, t_eval=t_eval, method='BDF', max_step=0.1)
            if not sol.success: st.error(f"Simulación falló: {sol.message}"); st.stop()
            st.write("Simulación completada.")
            t_sim = sol.t; X_sim = np.maximum(0, sol.y[0, :]); S_sim = np.maximum(0, sol.y[1, :])
            F_sim = np.zeros_like(t_sim)
            for i in range(len(t_sim)): F_sim[i] = Fmax if S_sim[i] < S_min else 0

            st.subheader("Resultados de la Simulación")
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            axes[0].plot(t_sim, X_sim, 'b-', linewidth=2); axes[0].set_ylabel('Biomasa (g/L)'); axes[0].set_title('Evolución Biomasa'); axes[0].grid(True); axes[0].set_ylim(bottom=0)
            axes[1].plot(t_sim, S_sim, 'r-', linewidth=2); axes[1].axhline(S_min, color='gray', linestyle='--', linewidth=1, label=f'S_min ({S_min:.2f})'); axes[1].axhline(S_max_ref, color='gray', linestyle=':', linewidth=1, label=f'S_max Ref. ({S_max_ref:.2f})'); axes[1].set_ylabel('Sustrato (g/L)'); axes[1].set_title('Evolución Sustrato (Control On-Off)'); axes[1].grid(True); axes[1].legend(loc='best'); axes[1].set_ylim(bottom=0)

            # --- CORRECCIÓN: Usar ax.step en lugar de ax.stairs ---
            axes[2].step(t_sim, F_sim, where='post', color='k', linewidth=1.5) # 'post' para que el escalón ocurra después del tiempo t
            # --- FIN CORRECCIÓN ---

            axes[2].set_xlabel('Tiempo (h)'); axes[2].set_ylabel('Flujo Alimentación (L/h)'); axes[2].set_title('Perfil Flujo Alimentación (On-Off)'); axes[2].grid(True); axes[2].set_ylim(-0.01 * Fmax, Fmax * 1.1 + 0.01); axes[2].set_xlim(0, t_final_onoff)
            plt.tight_layout(); st.pyplot(fig)

            df_results_onoff = pd.DataFrame({'Tiempo (h)': t_sim, 'Biomasa (g/L)': X_sim, 'Sustrato (g/L)': S_sim, 'Flujo (L/h)': F_sim})
            with st.expander("Ver datos de simulación"):
                st.dataframe(df_results_onoff.style.format({'Tiempo (h)': '{:.2f}', 'Biomasa (g/L)': '{:.4f}', 'Sustrato (g/L)': '{:.4f}', 'Flujo (L/h)': '{:.3f}'}))
        except Exception as e: st.error(f"Ocurrió un error durante la simulación On-Off:"); st.exception(e)
    else: st.info("Ajuste los parámetros y presione 'Simular Control On-Off'.")

# --- Punto de Entrada ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Control On-Off Sustrato")
    regulatorio_feed_onoff_page()