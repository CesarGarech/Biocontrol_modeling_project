# reg_temperatura.py (o reg_temp.py)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    import control # Python Control Systems Library (necesita: pip install control)
except ImportError:
    st.error("La librería 'python-control' no está instalada. Por favor, instálala ejecutando: pip install control")
    st.stop() # Detener si falta la librería
import pandas as pd # Opcional: para mostrar parámetros o resultados en tabla
import traceback # Para mostrar errores detallados

# --- Función Principal de la Página ---
def regulatorio_temperatura_page():
    """
    Página de Streamlit para simular el control de temperatura de un biorreactor.
    """
    st.header("🌡️ Simulación de Control Regulatorio de Temperatura")
    st.markdown("""
    Esta página simula un sistema de control de temperatura en lazo cerrado
    para un biorreactor. El sistema consta de:
    * **Proceso (Biorreactor):** Modelado como un sistema de primer orden.
    * **Elemento Final de Control (Válvula):** Modelado como primer orden.
    * **Sensor de Temperatura:** Modelado como primer orden.
    * **Controlador:** Un controlador Proporcional-Integral-Derivativo (PID).

    Puedes modificar los parámetros de cada componente y del controlador
    para observar cómo afectan la respuesta del sistema ante un cambio
    en el Setpoint (valor deseado) de temperatura.
    """)
    st.markdown("---")

    # --- Explicación de Funciones de Transferencia ---
    st.subheader("Funciones de Transferencia del Sistema")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'G_p(s) = \frac{K_p}{\tau_p s + 1}')
        st.caption("Proceso/Biorreactor")
        st.latex(r'G_v(s) = \frac{K_v}{\tau_v s + 1}')
        st.caption("Válvula")
    with col2:
        st.latex(r'G_s(s) = \frac{K_s}{\tau_s s + 1}')
        st.caption("Sensor")
        st.latex(r'C(s) = K_{p_{pid}} + \frac{K_{i_{pid}}}{s} + K_{d_{pid}} s')
        st.caption("Controlador PID")

    st.latex(r'G_{total}(s) = G_p(s) G_v(s) G_s(s)')
    st.caption("Planta Completa en Lazo Abierto (según ejemplo original)")

    st.latex(r'T(s) = \frac{C(s)G_{total}(s)}{1 + C(s)G_{total}(s)}')
    st.caption("Sistema en Lazo Cerrado (con realimentación unitaria después del sensor)")
    st.markdown("---")

    # --- Entradas del Usuario en la Barra Lateral ---
    with st.sidebar:
        st.header("Parámetros de Simulación")

        with st.expander("1. Parámetros del Proceso (Biorreactor)", expanded=True):
            Kp = st.number_input("Ganancia Proceso (Kp)", min_value=0.1, value=2.0, step=0.1, format="%.2f", key="Kp")
            tau_p = st.number_input("Constante Tiempo Proceso (τp) [s]", min_value=1.0, value=50.0, step=1.0, format="%.1f", key="tau_p")

        with st.expander("2. Parámetros de la Válvula", expanded=True):
            Kv = st.number_input("Ganancia Válvula (Kv)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Kv")
            tau_v = st.number_input("Constante Tiempo Válvula (τv) [s]", min_value=0.1, value=5.0, step=0.1, format="%.1f", key="tau_v")

        with st.expander("3. Parámetros del Sensor", expanded=True):
            Ks_sens = st.number_input("Ganancia Sensor (Ks)", min_value=0.1, value=1.0, step=0.1, format="%.2f", key="Ks_sens")
            tau_s = st.number_input("Constante Tiempo Sensor (τs) [s]", min_value=0.1, value=2.0, step=0.1, format="%.1f", key="tau_s")

        with st.expander("4. Parámetros del Controlador PID", expanded=True):
            Kp_pid = st.number_input("Ganancia Proporcional (Kp_pid)", min_value=0.0, value=5.1, step=0.1, format="%.2f", key="Kp_pid")
            Ki_pid = st.number_input("Ganancia Integral (Ki_pid)", min_value=0.0, value=0.0, step=0.01, format="%.3f", help="Poner 0 para control P o PD", key="Ki_pid")
            Kd_pid = st.number_input("Ganancia Derivativa (Kd_pid)", min_value=0.0, value=0.0, step=0.5, format="%.2f", help="Poner 0 para control P o PI", key="Kd_pid")

        with st.expander("5. Configuración de Simulación", expanded=True):
            t_final = st.number_input("Tiempo Final Simulación [s]", min_value=50.0, value=500.0, step=50.0, key="t_final")
            st.markdown("Configuración del Setpoint (Escalón)")
            sp_initial = st.number_input("Valor Inicial Setpoint [°C]", value=0.0, step=1.0, key="sp_initial")
            sp_final = st.number_input("Valor Final Setpoint [°C]", value=30.0, step=1.0, key="sp_final")
            t_step_value = st.number_input("Tiempo Cambio Setpoint [s]", min_value=0.0, max_value=float(t_final), value=150.0, step=1.0, key="t_step")


    # --- Simulación y Gráfica en el Área Principal ---
    st.subheader("Simulación del Control")

    if st.button("▶️ Simular Control de Temperatura", key="run_temp_sim"):
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

            st.write("Controlador C(s):")
            st.text(str(C_pid)) # Mostrar la FT del controlador

            # 3. Calcular lazo cerrado
            T_cerrado = control.feedback(C_pid * G_total, 1)
            st.write("Función de Transferencia en Lazo Cerrado T(s):")
            st.text(str(T_cerrado))


            # 4. Preparar simulación
            num_points = int(t_final * 5) + 1
            t = np.linspace(0, t_final, num_points)
            setpoint = np.ones_like(t) * sp_initial
            setpoint[t >= t_step_value] = sp_final

            # 5. Simular respuesta
            st.write(f"Simulando respuesta para t = 0 a {t_final} s...")
            T, yout = control.forced_response(T_cerrado, T=t, U=setpoint)
            st.write("Simulación completada.")

            # 6. Graficar resultados
            st.subheader("Respuesta del Sistema")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(T, setpoint, 'r--', linewidth=2, label='Setpoint (°C)')
            ax.plot(T, yout, 'b-', linewidth=2, label='Temperatura Controlada (°C)')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Temperatura (°C)')
            ax.set_title('Respuesta del Biorreactor con Control PID')
            ax.legend(loc='best')
            ax.grid(True)
            min_y = min(sp_initial, sp_final, np.min(yout) if len(yout)>0 else 0)
            max_y = max(sp_initial, sp_final, np.max(yout) if len(yout)>0 else 30)
            range_y = max(1, max_y - min_y) # Evitar rango cero
            ax.set_ylim(bottom=min_y - range_y*0.1, top=max_y + range_y*0.1) # Ajuste dinámico de ylim
            ax.set_xlim(0, t_final)
            st.pyplot(fig)

            # 7. Mostrar tabla de resultados (opcional)
            df_results = pd.DataFrame({'Tiempo (s)': T, 'Setpoint (°C)': setpoint, 'Temperatura (°C)': yout})
            with st.expander("Ver datos de simulación"):
                st.dataframe(df_results.style.format({
                    'Tiempo (s)': '{:.1f}',
                    'Setpoint (°C)': '{:.2f}',
                    'Temperatura (°C)': '{:.3f}'
                }))

        except Exception as e:
            st.error(f"Ocurrió un error durante la simulación:")
            st.exception(e)

    else:
        st.info("Ajuste los parámetros en la barra lateral y presione 'Simular Control de Temperatura'.")

# --- Punto de Entrada ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Control Temperatura")
    regulatorio_temperatura_page()