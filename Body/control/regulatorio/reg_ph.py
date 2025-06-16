# reg_ph.py (o como nombres el archivo)
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
def regulatorio_ph_page():
    """
    Página de Streamlit para simular control regulatorio de pH con gama partida.
    """
    st.header("💧 pH Regulatory Control Simulation (Split-Range)")
    st.markdown("""
    This page simulates a closed-loop pH control system for a
    bioreactor using a **split-range** strategy.
    Two final control elements (acid and base pumps)
    are operated by a single PID controller.
    
    * **Process:** The relationship between acid/base flow and reactor pH.
    * **Pumps (Acid/Base):** The dynamics of each dosing pump.
    * **Sensor:** The dynamics of the pH sensor.
    * **PID Controller:** Calculates the necessary control action.
    * **Split-Range Logic:** If the measured pH is above the setpoint, the
        acid pump is activated. If it is below, the base pump is activated.

    You can modify the parameters to observe the system response.
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("System Transfer Functions")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}')
        st.caption("Process (pH vs Flow)")
        st.latex(r'G_{acid}(s) = \frac{K_{p_{acid}}}{T_{acid} s + 1}')
        st.caption("Acid Pump")
        st.latex(r'G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}')
        st.caption("pH Sensor")
    with col2:
        st.latex(r'G_{base}(s) = \frac{K_{p_{base}}}{T_{base} s + 1}')
        st.caption("Base Pump")
        st.latex(r'C(s) = K_p + \frac{K_i}{s} + K_d s')
        st.caption("PID Controller")

    st.markdown("Open-loops (simplified for decoupled simulation):")
    st.latex(r'G_{open, acid}(s) = G_{proc}(s) G_{acid}(s) G_{sensor}(s)')
    st.latex(r'G_{open, base}(s) = G_{proc}(s) G_{base}(s) G_{sensor}(s)')
    st.caption("Note:The closed-loops for acid and base are simulated independently and then combined according to the split-range logic, similar to the original example.")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Simulation Parameters")

        with st.expander("1. Process Parameters", expanded=True):
            # Valores por defecto del script MATLAB
            Kp_proc = st.number_input("Proces Gain (Kp_proc) [pH/(flow unit)]", min_value=0.01, value=0.1, step=0.01, format="%.3f", key="Kp_proc", help="pH change per unit net flux (magnitude)")
            T_proc = st.number_input("Process Time Constant (T_proc) [s]", min_value=1.0, value=20.0, step=1.0, format="%.1f", key="T_proc")

        with st.expander("2. Pumps Parameters", expanded=True):
            Kp_acid = st.number_input("Acid Pump Gain (Kp_acid)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_acid")
            T_acid = st.number_input("Acid Pump Time Constant (T_acid) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="T_acid")
            st.divider() # Separador visual
            Kp_base = st.number_input("Base Pump Gain (Kp_base)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_base")
            T_base = st.number_input("Base Pump Time Constant (T_base) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="T_base")

        with st.expander("3. Sensor Parameters", expanded=True):
            Kp_sensor = st.number_input("Sensor Gain (Kp_sensor)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_sensor")
            T_sensor = st.number_input("Sensor Time Constant (T_sensor) [s]", min_value=0.1, value=1.0, step=0.1, format="%.1f", key="T_sensor")

        with st.expander("4. PID Controller Parameters", expanded=True):
            # Usar Kp, Ki, Kd como en el script MATLAB
            Kp_pid = st.number_input("Proportional Gain (Kp)", min_value=0.0, value=2.0, step=0.1, format="%.2f", key="Kp_pid_ph")
            Ki_pid = st.number_input("Integral Gain (Ki)", min_value=0.0, value=1.2, step=0.1, format="%.3f", help="Set 0 for P or PD control", key="Ki_pid_ph")
            Kd_pid = st.number_input("Derivative Gain (Kd)", min_value=0.0, value=0.0, step=0.1, format="%.2f", help="Set 0 for P or PI control", key="Kd_pid_ph")

        with st.expander("5. Simulation Configuration", expanded=True):
            t_final_ph = st.number_input("Final Time Simulation [s]", min_value=100.0, value=1000.0, step=50.0, key="t_final_ph")
            st.markdown("Setpoint Configuration (Step Input)")
            sp_initial_ph = st.number_input("Initial Setpoint Value [pH]", min_value=0.0, max_value=14.0, value=8.1, step=0.1, format="%.1f", key="sp_initial_ph")
            sp_final_ph = st.number_input("Final Setpoint Value [pH]", min_value=0.0, max_value=14.0, value=4.5, step=0.1, format="%.1f", key="sp_final_ph")
            # Asegurar que t_step no sea mayor que t_final
            t_step_ph_value = st.number_input("Setpoint Change Time [s]", min_value=0.0, max_value=float(t_final_ph), value=450.0, step=10.0, key="t_step_ph")
            # Offset inicial para decidir la primera acción de control
            y0_offset = st.number_input("Initial Offset pH (vs initial SP)", value=0.1, step=0.05, format="%.2f", help="Offset on initial SP to decide initial pump (positive activates base, negative activates acid).", key="y0_offset")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Control Simulation")

    if st.button("▶️ Simulate pH Control", key="run_ph_sim"):
        try:
            # 1. Definir funciones de transferencia
            s = control.tf('s')
            G_proc = abs(Kp_proc) / (T_proc * s + 1) # Usar magnitud para FT base
            G_acid = Kp_acid / (T_acid * s + 1)
            G_base = Kp_base / (T_base * s + 1)
            G_sensor = Kp_sensor / (T_sensor * s + 1)

            # 2. Definir controlador PID (forma TF corregida)
            C_pid = control.tf([Kd_pid, Kp_pid, Ki_pid], [1, 0])
            if Ki_pid == 0 and Kd_pid == 0: C_pid = Kp_pid # Simplificar si es P

            # 3. Lazos abiertos y cerrados (independientes para simulación)
            # El signo se maneja implícitamente en la lógica split-range
            G_open_acid = G_proc * G_acid * G_sensor
            G_open_base = G_proc * G_base * G_sensor

            # Calcular FTs en lazo cerrado para cada acción
            G_closed_acid = control.feedback(C_pid * G_open_acid, 1)
            G_closed_base = control.feedback(C_pid * G_open_base, 1)

            # 4. Preparar vectores de tiempo y setpoint
            num_points_ph = int(t_final_ph * 2) + 1 # Puntos suficientes para la dinámica
            t = np.linspace(0, t_final_ph, num_points_ph)
            setpoint = np.ones_like(t) * sp_initial_ph
            setpoint[t >= t_step_ph_value] = sp_final_ph

            # 5. Simular respuestas independientes
            st.write("Simulating independent loop responses...")
            T_sim, y_acid_resp = control.forced_response(G_closed_acid, T=t, U=setpoint)
            _, y_base_resp = control.forced_response(G_closed_base, T=t, U=setpoint) # T es el mismo
            st.write("Independent simulations completed.")

            # 6. Aplicar lógica de Gama Partida (Split-Range)
            st.write("Applying split-range logic...")
            y_combined = np.zeros_like(t)
            pump_active = np.zeros_like(t) # 0: None, 1: Base, -1: Acid

            # Inicializar y[0] con el offset para determinar la primera acción
            y_combined[0] = sp_initial_ph + y0_offset

            for i in range(1, len(t)):
                # Error basado en el pH combinado del paso anterior
                error_ph = setpoint[i] - y_combined[i-1]

                # Decisión de gama partida
                if error_ph < 0: # pH > Setpoint --> Necesita ÁCIDO
                    # Usar la respuesta simulada del lazo ácido
                    y_combined[i] = y_acid_resp[i]
                    pump_active[i] = -1 # Indicar bomba de ácido activa
                else: # pH <= Setpoint --> Necesita BASE
                    # Usar la respuesta simulada del lazo base
                    y_combined[i] = y_base_resp[i]
                    pump_active[i] = 1 # Indicar bomba de base activa

            # Forzar pH a estar entre 0 y 14 (límites físicos)
            y_combined = np.clip(y_combined, 0, 14)
            st.write("Split-range logic applied.")

            # 7. Graficar resultados
            st.subheader("System Response with Split-Range")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Gráfico 1: pH
            ax1.plot(t, setpoint, 'r--', linewidth=1.5, label='pH Setpoint')
            ax1.plot(t, y_combined, 'b-', linewidth=1.5, label='Controlled pH (Simulated)')
            ax1.set_ylabel('pH')
            ax1.set_title('System Response: pH')
            ax1.legend(loc='best')
            ax1.grid(True)
            # Ajuste dinámico de límites del eje Y para pH
            min_ph_plot = min(0, sp_initial_ph, sp_final_ph, np.min(y_combined) if len(y_combined)>0 else 0)
            max_ph_plot = max(14, sp_initial_ph, sp_final_ph, np.max(y_combined) if len(y_combined)>0 else 14)
            range_ph_plot = max(1, max_ph_plot - min_ph_plot) # Evitar rango cero
            ax1.set_ylim(bottom=max(0, min_ph_plot - range_ph_plot * 0.1),
                         top=min(14, max_ph_plot + range_ph_plot * 0.1))


            # Gráfico 2: Indicador de Bomba Activa
            acid_signal = np.where(pump_active == -1, 1, 0) # 1 si ácido activo
            base_signal = np.where(pump_active == 1, 1, 0) # 1 si base activo
            ax2.plot(t, acid_signal, 'g-', drawstyle='steps-post', linewidth=1.5, label='Active Acid Pump')
            ax2.plot(t, base_signal, 'm-', drawstyle='steps-post', linewidth=1.5, label='Active Base Pump')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Pump Status (Active=1)')
            ax2.set_title('Control Action (Split-Range)')
            ax2.legend(loc='best')
            ax2.grid(True)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Inactive', 'Active'])
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(0, t_final_ph) # Asegurar límite X correcto

            plt.tight_layout() # Ajustar espaciado
            st.pyplot(fig)

            # 8. Mostrar tabla de resultados (opcional)
            df_results_ph = pd.DataFrame({
                'Time (s)': t,
                'pH Setpoint': setpoint,
                'Simulated pH': y_combined,
                'Active Pump': np.where(pump_active==-1, 'Acid', np.where(pump_active==1, 'Base', 'None'))
             })
            with st.expander("View simulation data"):
                st.dataframe(df_results_ph.style.format({
                    'Time (s)': '{:.1f}',
                    'pH Setpoint': '{:.2f}',
                    'Simulated pH': '{:.3f}'
                }))

        except Exception as e:
            st.error(f"An error occurred during the pH simulation:")
            # Muestra el traceback completo en la página de Streamlit para depuración
            st.exception(e)

    else:
        # Mensaje inicial antes de presionar el botón
        st.info("Set the parameters in the sidebar and click on 'Simulate pH Control'.")

# --- Punto de Entrada (para ejecución directa del script) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="pH Control")
    regulatorio_ph_page()