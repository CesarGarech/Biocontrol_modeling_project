# reg_ph.py (o como nombres el archivo)
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
def regulatorio_ph_page():
    """
    Página de Streamlit para simular control regulatorio de pH con gama partida.
    """
    st.header("💧 Simulación de Control Regulatorio de pH (Gama Partida)")
    st.markdown("""
    Esta página simula un sistema de control de pH en lazo cerrado para un
    biorreactor utilizando una estrategia de **gama partida (split-range)**.
    Se emplean dos elementos finales de control (bombas de ácido y base)
    accionados por un único controlador PID.

    * **Proceso:** La relación entre el flujo de ácido/base y el pH del reactor.
    * **Bombas (Ácido/Base):** La dinámica de cada bomba dosificadora.
    * **Sensor:** La dinámica del sensor de pH.
    * **Controlador PID:** Calcula la acción de control necesaria.
    * **Lógica Gama Partida:** Si el pH medido es mayor al setpoint, se activa la
        bomba de ácido. Si es menor, se activa la bomba de base.

    Puedes modificar los parámetros para observar la respuesta del sistema.
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("Funciones de Transferencia")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_{proc}(s) = \frac{K_{p_{proc}}}{T_{proc} s + 1}')
        st.caption("Proceso (pH vs Flujo)")
        st.latex(r'G_{acid}(s) = \frac{K_{p_{acid}}}{T_{acid} s + 1}')
        st.caption("Bomba Ácido")
        st.latex(r'G_{sensor}(s) = \frac{K_{p_{sensor}}}{T_{sensor} s + 1}')
        st.caption("Sensor pH")
    with col2:
        st.latex(r'G_{base}(s) = \frac{K_{p_{base}}}{T_{base} s + 1}')
        st.caption("Bomba Base")
        st.latex(r'C(s) = K_p + \frac{K_i}{s} + K_d s')
        st.caption("Controlador PID")

    st.markdown("Lazos abiertos (simplificados para simulación desacoplada):")
    st.latex(r'G_{open, acid}(s) = G_{proc}(s) G_{acid}(s) G_{sensor}(s)')
    st.latex(r'G_{open, base}(s) = G_{proc}(s) G_{base}(s) G_{sensor}(s)')
    st.caption("Nota: Se simulan los lazos cerrados para ácido y base de forma independiente y luego se combinan según la lógica de gama partida, similar al ejemplo original.")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Parámetros de Simulación")

        with st.expander("1. Parámetros del Proceso", expanded=True):
            # Valores por defecto del script MATLAB
            Kp_proc = st.number_input("Ganancia Proceso (Kp_proc) [pH/(unidad flujo)]", min_value=0.01, value=0.1, step=0.01, format="%.3f", key="Kp_proc", help="Cambio de pH por unidad de flujo neto (magnitud)")
            T_proc = st.number_input("Constante Tiempo Proceso (T_proc) [s]", min_value=1.0, value=20.0, step=1.0, format="%.1f", key="T_proc")

        with st.expander("2. Parámetros Bombas", expanded=True):
            Kp_acid = st.number_input("Ganancia Bomba Ácido (Kp_acid)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_acid")
            T_acid = st.number_input("Constante Tiempo Bomba Ácido (T_acid) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="T_acid")
            st.divider() # Separador visual
            Kp_base = st.number_input("Ganancia Bomba Base (Kp_base)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_base")
            T_base = st.number_input("Constante Tiempo Bomba Base (T_base) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="T_base")

        with st.expander("3. Parámetros del Sensor", expanded=True):
            Kp_sensor = st.number_input("Ganancia Sensor (Kp_sensor)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kp_sensor")
            T_sensor = st.number_input("Constante Tiempo Sensor (T_sensor) [s]", min_value=0.1, value=1.0, step=0.1, format="%.1f", key="T_sensor")

        with st.expander("4. Parámetros del Controlador PID", expanded=True):
            # Usar Kp, Ki, Kd como en el script MATLAB
            Kp_pid = st.number_input("Ganancia Proporcional (Kp)", min_value=0.0, value=2.0, step=0.1, format="%.2f", key="Kp_pid_ph")
            Ki_pid = st.number_input("Ganancia Integral (Ki)", min_value=0.0, value=1.2, step=0.1, format="%.3f", help="Poner 0 para control P o PD", key="Ki_pid_ph")
            Kd_pid = st.number_input("Ganancia Derivativa (Kd)", min_value=0.0, value=0.0, step=0.1, format="%.2f", help="Poner 0 para control P o PI", key="Kd_pid_ph")

        with st.expander("5. Configuración de Simulación", expanded=True):
            t_final_ph = st.number_input("Tiempo Final Simulación [s]", min_value=100.0, value=1000.0, step=50.0, key="t_final_ph")
            st.markdown("Configuración del Setpoint (Escalón)")
            sp_initial_ph = st.number_input("Valor Inicial Setpoint [pH]", min_value=0.0, max_value=14.0, value=8.1, step=0.1, format="%.1f", key="sp_initial_ph")
            sp_final_ph = st.number_input("Valor Final Setpoint [pH]", min_value=0.0, max_value=14.0, value=4.5, step=0.1, format="%.1f", key="sp_final_ph")
            # Asegurar que t_step no sea mayor que t_final
            t_step_ph_value = st.number_input("Tiempo Cambio Setpoint [s]", min_value=0.0, max_value=float(t_final_ph), value=450.0, step=10.0, key="t_step_ph")
            # Offset inicial para decidir la primera acción de control
            y0_offset = st.number_input("Offset Inicial pH (vs SP inicial)", value=0.1, step=0.05, format="%.2f", help="Offset sobre SP inicial para decidir bomba inicial (positivo activa base, negativo activa ácido).", key="y0_offset")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Simulación del Control")

    if st.button("▶️ Simular Control de pH", key="run_ph_sim"):
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
            st.write("Simulando respuestas de lazos independientes...")
            T_sim, y_acid_resp = control.forced_response(G_closed_acid, T=t, U=setpoint)
            _, y_base_resp = control.forced_response(G_closed_base, T=t, U=setpoint) # T es el mismo
            st.write("Simulaciones independientes completadas.")

            # 6. Aplicar lógica de Gama Partida (Split-Range)
            st.write("Aplicando lógica de gama partida...")
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
            st.write("Lógica de gama partida aplicada.")

            # 7. Graficar resultados
            st.subheader("Respuesta del Sistema con Gama Partida")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Gráfico 1: pH
            ax1.plot(t, setpoint, 'r--', linewidth=1.5, label='Setpoint pH')
            ax1.plot(t, y_combined, 'b-', linewidth=1.5, label='pH Controlado (Simulado)')
            ax1.set_ylabel('pH')
            ax1.set_title('Respuesta del Sistema: pH')
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
            ax2.plot(t, acid_signal, 'g-', drawstyle='steps-post', linewidth=1.5, label='Bomba Ácido Activa')
            ax2.plot(t, base_signal, 'm-', drawstyle='steps-post', linewidth=1.5, label='Bomba Base Activa')
            ax2.set_xlabel('Tiempo (s)')
            ax2.set_ylabel('Estado Bomba (Activa=1)')
            ax2.set_title('Acción de Control (Gama Partida)')
            ax2.legend(loc='best')
            ax2.grid(True)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Inactiva', 'Activa'])
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(0, t_final_ph) # Asegurar límite X correcto

            plt.tight_layout() # Ajustar espaciado
            st.pyplot(fig)

            # 8. Mostrar tabla de resultados (opcional)
            df_results_ph = pd.DataFrame({
                'Tiempo (s)': t,
                'Setpoint pH': setpoint,
                'pH Simulado': y_combined,
                'Bomba Activa': np.where(pump_active==-1, 'Ácido', np.where(pump_active==1, 'Base', 'Ninguna'))
             })
            with st.expander("Ver datos de simulación detallados"):
                st.dataframe(df_results_ph.style.format({
                    'Tiempo (s)': '{:.1f}',
                    'Setpoint pH': '{:.2f}',
                    'pH Simulado': '{:.3f}'
                }))

        except Exception as e:
            st.error(f"Ocurrió un error durante la simulación de pH:")
            # Muestra el traceback completo en la página de Streamlit para depuración
            st.exception(e)

    else:
        # Mensaje inicial antes de presionar el botón
        st.info("Ajuste los parámetros en la barra lateral y presione 'Simular Control de pH'.")

# --- Punto de Entrada (para ejecución directa del script) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Control pH")
    regulatorio_ph_page()