# reg_temperatura.py (o reg_temp.py)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    import control # Python Control Systems Library (necesita: pip install control)
except ImportError:
    st.error("The 'python-control'library is not installed. Please, install it by running: pip install control")
    st.stop() # Detener si falta la librería
import pandas as pd # Opcional: para mostrar parámetros o resultados en tabla
import traceback # Para mostrar errores detallados

# --- Función Principal de la Página ---
def regulatorio_temperatura_page():
    """
    Página de Streamlit para simular el control de temperatura de un biorreactor.
    """
    st.header("🌡️ Temperature Regulatory Control Simulation")
    st.markdown("""
    This page simulates a closed-loop temperature control system 
    for a bioreactor. The system consists of:
    * **Process (Bioreactor):** Modeled as a first-order system.
    * **Final Control Element (Valve):** Modeled as a first-order system.
    * **Temperature Sensor:** Modeled as a first-order system.
    * **Controller:** A Proportional-Integral-Derivative (PID) controller.

    You can modify the parameters of each component andthe controller
    to observe how they affect the system's response to a change
    in the temperature setpoint (desired value).
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("System Transfer Functions")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_p(s) = \frac{K_p}{\tau_p s + 1}')
        st.caption("Process/Bioreactor")
        st.latex(r'G_v(s) = \frac{K_v}{\tau_v s + 1}')
        st.caption("Valve")
    with col2:
        st.latex(r'G_s(s) = \frac{K_s}{\tau_s s + 1}')
        st.caption("Sensor")
        st.latex(r'C(s) = K_{p_{pid}} + \frac{K_{i_{pid}}}{s} + K_{d_{pid}} s')
        st.caption("PID Controller")

    st.latex(r'G_{total}(s) = G_p(s) G_v(s) G_s(s)')
    st.caption("Full Plant in Open-Loop (according to original example)")

    st.latex(r'T(s) = \frac{C(s)G_{total}(s)}{1 + C(s)G_{total}(s)}')
    st.caption("Closed-Loop System (with unit feedback after the sensor)")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Simulation Parameters")

        with st.expander("1. Process Parameters (Bioreactor)", expanded=True):
            Kp = st.number_input("Proces Gain (Kp)", min_value=0.1, value=2.0, step=0.1, format="%.2f", key="Kp")
            tau_p = st.number_input("Process Time Constant (τp) [s]", min_value=1.0, value=50.0, step=1.0, format="%.1f", key="tau_p")

        with st.expander("2. Valve Parameters", expanded=True):
            Kv = st.number_input("Valve Gain (Kv)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kv")
            tau_v = st.number_input("Valve Time Constant (τv) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="tau_v")

        with st.expander("3. Sensor Parameters", expanded=True):
            Ks_sens = st.number_input("Sensor Gain (Ks)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Ks_sens")
            tau_s = st.number_input("Sensor Time Constant (τs) [s]", min_value=0.1, value=2.0, step=0.1, format="%.1f", key="tau_s")

        with st.expander("4. PID Controller Parameters", expanded=True):
            Kp_pid = st.number_input("Proportional Gain (Kp_pid)", min_value=0.0, value=5.1, step=0.1, format="%.2f", key="Kp_pid")
            Ki_pid = st.number_input("Integral Gain (Ki_pid)", min_value=0.0, value=0.0, step=0.01, format="%.3f", help="Set 0 for P or PD control", key="Ki_pid")
            Kd_pid = st.number_input("Derivative Gain (Kd_pid)", min_value=0.0, value=0.0, step=0.5, format="%.2f", help="Set 0 for P or PI control", key="Kd_pid")

        with st.expander("5. Simulation Configuration", expanded=True):
            t_final = st.number_input("Final Time Simulation [s]", min_value=50.0, value=500.0, step=50.0, key="t_final")
            st.markdown("Setpoint Configuration (Step Input)")
            sp_initial = st.number_input("Initial Setpoint Value [°C]", value=0.0, step=1.0, key="sp_initial")
            sp_final = st.number_input("Final Setpoint Value [°C]", value=30.0, step=1.0, key="sp_final")
            t_step_value = st.number_input("Setpoint Change Time [s]", min_value=0.0, max_value=float(t_final), value=150.0, step=1.0, key="t_step")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Control Simulation")

    if st.button("▶️ Simulate Temperature Control", key="run_temp_sim"):
        try:
            # 1. Definir funciones de transferencia
            s = control.tf('s')
            Gp = Kp / (tau_p * s + 1)
            Gv = Kv / (tau_v * s + 1)
            Gs = Ks_sens / (tau_s * s + 1)
            G_total = Gs * Gp * Gv

            # 2. Definir controlador PID --- ¡CORRECCIÓN AQUÍ! ---
            # C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
            C_pid = control.tf([Kd_pid, Kp_pid, Ki_pid], [1, 0])
            # Simplificar la FT resultante si Ki y Kd son cero
            if Ki_pid == 0 and Kd_pid == 0:
                 C_pid = Kp_pid # Si es solo P, es una ganancia simple

            st.write("Controller C(s):")
            st.text(str(C_pid)) # Mostrar la FT del controlador

            # 3. Calcular lazo cerrado
            T_cerrado = control.feedback(C_pid * G_total, 1)
            st.write("Closed-Loop Transfer Function T(s):")
            st.text(str(T_cerrado))


            # 4. Preparar simulación
            num_points = int(t_final * 5) + 1
            t = np.linspace(0, t_final, num_points)
            setpoint = np.ones_like(t) * sp_initial
            setpoint[t >= t_step_value] = sp_final

            # 5. Simular respuesta
            st.write(f"Simulating response from t = 0 to {t_final} s...")
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            st.write("Simulation completed.")

            # 6. Graficar resultados
            st.subheader("System Response")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(T, setpoint, 'r--', linewidth=2, label='Setpoint (°C)')
            ax.plot(T, yout, 'b-', linewidth=2, label='Controlled Temperature (°C)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Bioreactor Response with PID Control')
            ax.legend(loc='best')
            ax.grid(True)
            min_y = min(sp_initial, sp_final, np.min(yout) if len(yout)>0 else 0)
            max_y = max(sp_initial, sp_final, np.max(yout) if len(yout)>0 else 30)
            range_y = max(1, max_y - min_y) # Evitar rango cero
            ax.set_ylim(bottom=min_y - range_y*0.1, top=max_y + range_y*0.1) # Ajuste dinámico de ylim
            ax.set_xlim(0, t_final)
            st.pyplot(fig)

            # 7. Mostrar tabla de resultados (opcional)
            df_results = pd.DataFrame({'Time (s)': T, 'Setpoint (°C)': setpoint, 'Temperature (°C)': yout})
            with st.expander("View simulation data"):
                st.dataframe(df_results.style.format({
                    'Time (s)': '{:.1f}',
                    'Setpoint (°C)': '{:.2f}',
                    'Temperature (°C)': '{:.3f}'
                }))

        except Exception as e:
            st.error(f"An error occurred during the temperature simulation:")
            st.exception(e)

    else:
        st.info("Set the parameters in the sidebar and click on 'Simulate Temperature Control'.")

# --- Punto de Entrada ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Temperature Control")
    regulatorio_temperatura_page()