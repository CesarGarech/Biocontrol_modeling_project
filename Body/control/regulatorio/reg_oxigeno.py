# reg_oxigeno.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    # Intenta importar la librería control
    import control # Python Control Systems Library (necesita: pip install control)
except ImportError:
    # Muestra un error y detiene si la librería no está instalada
    st.error("La librería 'python-control' no está instalada. Por favor, instálala ejecutando: pip install control")
    st.stop() # Detiene la ejecución del script de esta página
import pandas as pd # Opcional: para mostrar parámetros o resultados en tabla
import traceback # Para mostrar errores detallados si ocurren

# --- Función Principal de la Página de Streamlit ---
def regulatorio_oxigeno_page():
    """
    Página de Streamlit para simular control regulatorio de Oxígeno Disuelto (OD).
    """
    st.header("💨 Simulación de Control Regulatorio de Oxígeno Disuelto")
    st.markdown("""
    Esta página simula un sistema de control de Oxígeno Disuelto (OD) en lazo cerrado.
    La variable manipulada para controlar el OD es la **velocidad de agitación (RPM)**,
    cuyo efecto se modela a través de la función de transferencia del proceso.

    **Importante:** Este modelo utiliza una **representación simplificada** con funciones
    de transferencia directas. **No** modela explícitamente la tasa de transferencia
    de oxígeno (OTR) basada en $k_La$ o el consumo de oxígeno por la biomasa (OUR).
    Es un ejemplo didáctico de la estructura de un lazo de control.

    * **Proceso:** Relación simplificada entre la señal de agitación (post-actuador) y el OD.
    * **Actuador:** Dinámica asociada al cambio de agitación (ej. motor).
    * **Sensor:** Dinámica del sensor de OD.
    * **Controlador PID:** Calcula la señal para el actuador de agitación.
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("Funciones de Transferencia del Sistema")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}')
        st.caption("Proceso (OD / Señal Agitación)")
        st.latex(r'G_{valv}(s) = \frac{K_{p_{valv}}}{T_{valv} s + 1}')
        st.caption("Actuador Agitación (Válvula/Motor)")
    with col2:
        st.latex(r'G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}')
        st.caption("Sensor OD")
        st.latex(r'C(s) = K_p + \frac{K_i}{s} + K_d s')
        st.caption("Controlador PID")

    st.latex(r'G_{open}(s) = G_{proc}(s) G_{valv}(s) G_{sensor}(s)')
    st.caption("Planta Completa en Lazo Abierto (según ejemplo original)")

    st.latex(r'T(s) = \frac{C(s)G_{open}(s)}{1 + C(s)G_{open}(s)}')
    st.caption("Sistema en Lazo Cerrado (con realimentación unitaria después del sensor)")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Parámetros de Simulación")

        with st.expander("1. Parámetros del Proceso", expanded=True):
            # Valores por defecto del script MATLAB
            Kp_proc_o2 = st.number_input("Ganancia Proceso (Kp_proc) [%OD/unidad RPM]", min_value=0.001, value=0.05, step=0.005, format="%.3f", key="Kp_proc_o2", help="Cambio en %OD por cambio en señal de RPM")
            T_proc_o2 = st.number_input("Constante Tiempo Proceso (T_proc) [s]", min_value=1.0, value=10.0, step=1.0, format="%.1f", key="T_proc_o2")

        with st.expander("2. Parámetros del Actuador (Válvula/Motor)", expanded=True):
            Kp_valv_o2 = st.number_input("Ganancia Actuador (Kp_valv)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_valv_o2")
            T_valv_o2 = st.number_input("Constante Tiempo Actuador (T_valv) [s]", min_value=0.1, value=2.0, step=0.1, format="%.1f", key="T_valv_o2")

        with st.expander("3. Parámetros del Sensor", expanded=True):
            Kp_sensor_o2 = st.number_input("Ganancia Sensor (Kp_sensor)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_sensor_o2")
            T_sensor_o2 = st.number_input("Constante Tiempo Sensor (T_sensor) [s]", min_value=0.1, value=1.0, step=0.1, format="%.1f", key="T_sensor_o2")

        with st.expander("4. Parámetros del Controlador PID", expanded=True):
            Kp_pid_o2 = st.number_input("Ganancia Proporcional (Kp)", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="Kp_pid_o2")
            Ki_pid_o2 = st.number_input("Ganancia Integral (Ki)", min_value=0.0, value=1.0, step=0.1, format="%.3f", key="Ki_pid_o2")
            Kd_pid_o2 = st.number_input("Ganancia Derivativa (Kd)", min_value=0.0, value=0.1, step=0.01, format="%.3f", key="Kd_pid_o2")

        with st.expander("5. Configuración de Simulación", expanded=True):
            t_final_o2 = st.number_input("Tiempo Final Simulación [s]", min_value=50.0, value=500.0, step=50.0, key="t_final_o2")
            st.markdown("Configuración del Setpoint (Escalón)")
            sp_initial_o2 = st.number_input("Valor Inicial Setpoint [% OD]", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="sp_initial_o2")
            sp_final_o2 = st.number_input("Valor Final Setpoint [% OD]", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="sp_final_o2")
            t_step_o2_value = st.number_input("Tiempo Cambio Setpoint [s]", min_value=0.0, max_value=float(t_final_o2), value=250.0, step=10.0, key="t_step_o2")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Simulación del Control de Oxígeno Disuelto")

    if st.button("▶️ Simular Control OD", key="run_o2_sim"):
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
            with st.expander("Ver Funciones de Transferencia Usadas"):
                st.text(f"G_proc(s): {G_proc}")
                st.text(f"G_valv(s): {G_valv}")
                st.text(f"G_sensor(s): {G_sensor}")
                st.text(f"C_pid(s): {C_pid}")
                st.text(f"T_cerrado(s): {T_cerrado}")


            # 4. Preparar simulación
            num_points_o2 = int(t_final_o2 * 5) + 1 # Suficientes puntos
            t = np.linspace(0, t_final_o2, num_points_o2)
            setpoint = np.ones_like(t) * sp_initial_o2
            setpoint[t >= t_step_o2_value] = sp_final_o2

            # 5. Simular respuesta
            st.write(f"Simulando respuesta para t = 0 a {t_final_o2} s...")
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            st.write("Simulación completada.")

            # 6. Graficar resultados
            st.subheader("Respuesta del Sistema")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(T, setpoint, 'r--', linewidth=2, label='Setpoint (% OD)')
            ax.plot(T, yout, 'b-', linewidth=2, label='Oxígeno Controlado (% OD)')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Oxígeno Disuelto (% Saturación)')
            ax.set_title('Respuesta del Control PID de Oxígeno Disuelto')
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
            df_results_o2 = pd.DataFrame({'Tiempo (s)': T, 'Setpoint (%OD)': setpoint, 'Oxigeno (%OD)': yout})
            with st.expander("Ver datos de simulación"):
                st.dataframe(df_results_o2.style.format({
                    'Tiempo (s)': '{:.1f}',
                    'Setpoint (%OD)': '{:.1f}',
                    'Oxigeno (%OD)': '{:.2f}'
                }))

        except Exception as e:
            st.error(f"Ocurrió un error durante la simulación de OD:")
            st.exception(e) # Muestra traceback en Streamlit

    else:
        st.info("Ajuste los parámetros en la barra lateral y presione 'Simular Control OD'.")

# --- Punto de Entrada ---
# Permite ejecutar el script directamente: python reg_oxigeno.py
# Si se llama desde main.py, esta parte no se ejecuta, solo se importa la función.
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Control Oxígeno")
    regulatorio_oxigeno_page()