# reg_oxigeno.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    # Intenta importar la librería control
    import control # Python Control Systems Library (necesita: pip install control)
except ImportError:
    # Muestra un error y detiene si la librería no está instalada
    st.error("The 'python-control'library is not installed. Please, install it by running: pip install control")
    st.stop() # Detiene la ejecución del script de esta página
import pandas as pd # Opcional: para mostrar parámetros o resultados en tabla
import traceback # Para mostrar errores detallados si ocurren

# --- Función Principal de la Página de Streamlit ---
def regulatorio_oxigeno_page():
    """
    Página de Streamlit para simular control regulatorio de Oxígeno Disuelto (OD).
    """
    st.header("💨 Dissolved Oxygen Regulatory Control Simulation")
    st.markdown("""
    This page simulates a closed-loop Dissolved Oxygen (DO) control system.
    The manipulated variable used to control DO is the **stirring speed (RPM)**,
    whose effect is modeled through the process transfer function.

    **Important:** This model uses a **simplified representation** with direct
    transfer functions. It does **not** explicitly model the oxygen transfer
    rate (OTR) based on $k_La$ nor the oxygen uptake rate (OUR) by the biomass.
    This is a didactic example of a control loop structure.

    * **Process:** Simplified relationship between the stirring signal (post-actuator) and DO.
    * **Actuator:** Dynamics associated with stirring changes (e.g., motor).
    * **Sensor:** Dynamics of the DO sensor.
    * **PID Controller:** Calculates the signal for the stirring actuator.
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("System Transfer Functions")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}')
        st.caption("Process (DO / Stirring Signal)")
        st.latex(r'G_{valv}(s) = \frac{K_{p_{valv}}}{T_{valv} s + 1}')
        st.caption("Stirring Actuator (Valve/Motor)")
    with col2:
        st.latex(r'G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}')
        st.caption("DO Sensor")
        st.latex(r'C(s) = K_p + \frac{K_i}{s} + K_d s')
        st.caption("PID Controller")

    st.latex(r'G_{open}(s) = G_{proc}(s) G_{valv}(s) G_{sensor}(s)')
    st.caption("Full Plant in Open-Loop (according to original example)")

    st.latex(r'T(s) = \frac{C(s)G_{open}(s)}{1 + C(s)G_{open}(s)}')
    st.caption("Closed-Loop System (with unit feedback after the sensor)")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Simulation Parameters")

        with st.expander("1. Process Parameters", expanded=True):
            # Valores por defecto del script MATLAB
            Kp_proc_o2 = st.number_input("Proces Gain (Kp_proc) [%DO/RPM unit]", min_value=0.001, value=0.05, step=0.005, format="%.3f", key="Kp_proc_o2", help="Change in %DO by change in RPM signal")
            T_proc_o2 = st.number_input("Process Time Constant (T_proc) [s]", min_value=1.0, value=10.0, step=1.0, format="%.1f", key="T_proc_o2")

        with st.expander("2. Actuator Parameters (Valve/Motor)", expanded=True):
            Kp_valv_o2 = st.number_input("Actuator Gain (Kp_valv)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_valv_o2")
            T_valv_o2 = st.number_input("Actuator Time Constant (T_valv) [s]", min_value=0.1, value=2.0, step=0.1, format="%.1f", key="T_valv_o2")

        with st.expander("3. Sensor Parameters", expanded=True):
            Kp_sensor_o2 = st.number_input("Sensor Gain (Kp_sensor)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_sensor_o2")
            T_sensor_o2 = st.number_input("Sensor Time Constant (T_sensor) [s]", min_value=0.1, value=1.0, step=0.1, format="%.1f", key="T_sensor_o2")

        with st.expander("4. PID Controller Parameters", expanded=True):
            Kp_pid_o2 = st.number_input("Proportional Gain (Kp)", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="Kp_pid_o2")
            Ki_pid_o2 = st.number_input("Integral Gain (Ki)", min_value=0.0, value=1.0, step=0.1, format="%.3f", key="Ki_pid_o2")
            Kd_pid_o2 = st.number_input("Derivative Gain (Kd)", min_value=0.0, value=0.1, step=0.01, format="%.3f", key="Kd_pid_o2")

        with st.expander("5. Simulation Configuration", expanded=True):
            t_final_o2 = st.number_input("Final Time Simulation [s]", min_value=50.0, value=500.0, step=50.0, key="t_final_o2")
            st.markdown("Setpoint Configuration (Step Input)")
            sp_initial_o2 = st.number_input("Initial Setpoint Value [% DO]", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="sp_initial_o2")
            sp_final_o2 = st.number_input("Final Setpoint Value [% DO]", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="sp_final_o2")
            t_step_o2_value = st.number_input("Setpoint Change Time [s]", min_value=0.0, max_value=float(t_final_o2), value=250.0, step=10.0, key="t_step_o2")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Dissolved Oxygen Control Simulation")

    if st.button("▶️ Simulate DO Control", key="run_o2_sim"):
        try:
            # 1. Definir funciones de transferencia
            s = control.tf('s')
            G_proc = Kp_proc_o2 / (T_proc_o2 * s + 1)
            G_valv = Kp_valv_o2 / (T_valv_o2 * s + 1)
            G_sensor = Kp_sensor_o2 / (T_sensor_o2 * s + 1)
            G_open = G_proc * G_valv * G_sensor # Lazo abierto como en MATLAB

            # 2. Definir controlador PID
            C_pid = control.tf([Kd_pid_o2, Kp_pid_o2, Ki_pid_o2], [1, 0])
            if Ki_pid_o2 == 0 and Kd_pid_o2 == 0: C_pid = Kp_pid_o2 # Simplificar si es P

            # 3. Calcular lazo cerrado
            T_cerrado = control.feedback(C_pid * G_open, 1) # Coincide con MATLAB

            # Mostrar FTs (opcional)
            with st.expander("View Transfer Functions Used"):
                st.text(f"G_proc(s): {G_proc}")
                st.text(f"G_valv(s): {G_valv}")
                st.text(f"G_sensor(s): {G_sensor}")
                st.text(f"C_pid(s): {C_pid}")
                st.text(f"T_closed(s): {T_cerrado}")


            # 4. Preparar simulación
            num_points_o2 = int(t_final_o2 * 5) + 1 # Suficientes puntos
            t = np.linspace(0, t_final_o2, num_points_o2)
            setpoint = np.ones_like(t) * sp_initial_o2
            setpoint[t >= t_step_o2_value] = sp_final_o2

            # 5. Simular respuesta
            st.write(f"Simulating response from t = 0 to {t_final_o2} s...")
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            st.write("Simulation completed.")

            # 6. Graficar resultados
            st.subheader("System Response")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(T, setpoint, 'r--', linewidth=2, label='Setpoint (% DO)')
            ax.plot(T, yout, 'b-', linewidth=2, label='Controlled Oxygen (% DO)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Dissolved Oxygen (% Saturation)')
            ax.set_title('Dissolved Oxygen PID Control Response')
            ax.legend(loc='best')
            ax.grid(True)
            # Ajuste dinámico de límites Y
            min_y_plot = min(0, sp_initial_o2, sp_final_o2, np.min(yout) if len(yout)>0 else 0)
            max_y_plot = max(100, sp_initial_o2, sp_final_o2, np.max(yout) if len(yout)>0 else 100)
            range_y_plot = max(5, max_y_plot - min_y_plot) # Mínimo rango de 5%
            ax.set_ylim(bottom=max(0, min_y_plot - range_y_plot * 0.1),
                        top=min(110, max_y_plot + range_y_plot * 0.1)) # Limitar a 110% máx
            ax.set_xlim(0, t_final_o2)
            st.pyplot(fig)

            # 7. Mostrar tabla de resultados (opcional)
            df_results_o2 = pd.DataFrame({'Time (s)': T, 'Setpoint (%DO)': setpoint, 'Oxygen (%DO)': yout})
            with st.expander("View simulation data"):
                st.dataframe(df_results_o2.style.format({
                    'Time (s)': '{:.1f}',
                    'Setpoint (%DO)': '{:.1f}',
                    'Oxygen (%DO)': '{:.2f}'
                }))

        except Exception as e:
            st.error(f"An error occurred during the DO simulation:")
            st.exception(e) # Muestra traceback en Streamlit

    else:
        st.info("Set the parameters in the sidebar and click on 'Simulate DO Control'.")

# --- Punto de Entrada ---
# Permite ejecutar el script directamente: python reg_oxigeno.py
# Si se llama desde main.py, esta parte no se ejecuta, solo se importa la función.
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Oxygen Control")
    regulatorio_oxigeno_page()