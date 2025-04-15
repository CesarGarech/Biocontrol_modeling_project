import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Asumiendo que tienes un archivo Utils/kinetics.py con las funciones mu
# Si no, puedes definirlas directamente aquí o ajustar la importación.
# Ejemplo de definiciones (si no tienes kinetics.py):
def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)
def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * (S**n) / (Ks**n + S**n)
def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    # Asegurarse que los valores no sean negativos para evitar errores matemáticos
    S = max(0, S)
    O2 = max(0, O2)
    P = max(0, P)
    # Término de inhibición por producto (modelo simple, ajustar si es necesario)
    # Evitar división por cero si KP es muy pequeño o P es grande
    inhibition_P = (1 - P / KP) if KP > 0 and P < KP else 0
    inhibition_P = max(0, inhibition_P) # Asegurar que no sea negativo

    # Monod para Sustrato y Oxígeno, con inhibición por Producto
    mu = mumax * (S / (Ks + S)) * (O2 / (KO + O2)) * inhibition_P
    return max(0, mu) # Asegurar que la tasa de crecimiento no sea negativa

# --- Fin definiciones de ejemplo ---

def lote_alimentado_page():
    st.header("Modo de operación: Lote Alimentado")

    with st.sidebar:
        st.subheader("Parámetros del Modelo")
        tipo_mu = st.selectbox("Modelo Cinético", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])
        mumax = st.slider("μmax [1/h]", 0.1, 1.0, 0.4)
        Ks = st.slider("Ks [g/L]", 0.01, 2.0, 0.5)

        if tipo_mu == "Monod sigmoidal":
            n = st.slider("Exponente sigmoidal (n)", 1, 5, 2)
        elif tipo_mu == "Monod con restricciones":
            KO = st.slider("Constante saturación O2 [mg/L]", 0.1, 5.0, 0.5)
            KP = st.slider("Constante inhibición producto [g/L]", 0.1, 10.0, 5.0) # Ajustado el valor por defecto

        Yxs = st.slider("Yxs [g/g]", 0.1, 1.0, 0.6)
        Ypx = st.slider("Ypx [g/g]", 0.0, 1.0, 0.3)
        Yxo = st.slider("Yxo [g/g]", 0.1, 1.0, 0.2)
        Kla = st.slider("kLa [1/h]", 1.0, 200.0, 50.0)
        Cs = st.slider("O2 Saturado [mg/L]", 5.0, 15.0, 8.0)
        Sin = st.slider("Sustrato en Alimentado [g/L]", 50.0, 300.0, 150.0)
        ms = st.slider("Mantenimiento S [g/g/h]", 0.0, 0.1, 0.001)
        Kd = st.slider("Decaimiento X [1/h]", 0.0, 0.1, 0.02) # Unidades corregidas
        mo = st.slider("Mantenimiento O2 [g/g/h]", 0.0, 0.1, 0.01)

        st.subheader("Estrategia de Alimentación")
        # Añadida opción "Lineal"
        estrategia = st.selectbox("Tipo", ["Constante", "Exponencial", "Escalon", "Lineal"])
        F_base = st.slider("Flujo Base (o Inicial para Lineal) [L/h]", 0.01, 5.0, 0.5)
        t_alim_inicio = st.slider("Inicio Alimentación [h]", 0.0, 24.0, 2.0, 0.5) # Ajustado a float
        t_alim_fin = st.slider("Fin Alimentación [h]", t_alim_inicio + 0.1, 48.0, 24.0, 0.5) # Asegura t_fin > t_inicio

        # Parámetro adicional solo para la estrategia Lineal
        F_lineal_fin = 0.0 # Inicializar
        if estrategia == "Lineal":
            F_lineal_fin = st.slider("Flujo Final (Lineal) [L/h]", F_base, 10.0, F_base * 2) # Asegura F_fin >= F_base

        st.subheader("Condiciones Iniciales")
        V0 = st.number_input("Volumen Inicial [L]", 1.0, 100.0, 3.0) # Rango aumentado
        X0 = st.number_input("Biomasa Inicial [g/L]", 0.1, 50.0, 1.0) # Rango aumentado
        S0 = st.number_input("Sustrato Inicial [g/L]", 0.1, 100.0, 30.0)
        P0 = st.number_input("Producto Inicial [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("O2 Inicial [mg/L]", 0.0, float(Cs), float(Cs)) # O2 inicial igual a saturado

        t_final = st.slider("Tiempo de Simulación [h]", max(10.0, t_alim_fin + 1), 200.0, 48.0, 1.0) # Rango aumentado y ajustado
        atol = st.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def calcular_flujo(t):
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constante":
                return F_base
            elif estrategia == "Exponencial":
                 # Asegurar que el exponente no cause overflow si t es grande
                exponent = 0.15 * (t - t_alim_inicio)
                try:
                    return F_base * np.exp(exponent)
                except OverflowError:
                    return float('inf') # O un valor máximo razonable
            elif estrategia == "Escalon":
                # Usar un punto medio claro entre inicio y fin
                t_medio = t_alim_inicio + (t_alim_fin - t_alim_inicio) / 2
                return F_base * 2 if t > t_medio else F_base
            elif estrategia == "Lineal":
                delta_t = t_alim_fin - t_alim_inicio
                if delta_t > 0:
                    slope = (F_lineal_fin - F_base) / delta_t
                    return F_base + slope * (t - t_alim_inicio)
                else:
                    # Si el intervalo es 0, devuelve el flujo inicial (o final, según se prefiera)
                    return F_base # O F_lineal_fin
        return 0.0

    def modelo_fedbatch(t, y):
        X, S, P, O2, V = y

        # Asegurar que las concentraciones no sean negativas (pueden ocurrir por errores numéricos)
        X = max(0, X)
        S = max(0, S)
        P = max(0, P)
        O2 = max(0, O2)
        V = max(1e-6, V) # Evitar división por cero si V se vuelve muy pequeño

        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
        else:
             mu = 0 # Caso por defecto o error

        mu = max(0, mu) # Asegurar que mu no sea negativo

        F = calcular_flujo(t)

        # Corrección en la ecuación de Biomasa: El término de decaimiento no se divide por V
        # dXdt = mu*X - Kd*X - (F/V)*X # Modelo estándar sin decaimiento por volumen
        # Corrección: La concentración cambia por dilución (F/V)*X y por crecimiento/decaimiento intrínseco
        dXdt = (mu - Kd) * X - (F / V) * X

        # dSdt: Consumo por crecimiento y mantenimiento, entrada por flujo, dilución por flujo
        dSdt = -(mu / Yxs + ms) * X + (F / V) * (Sin - S)

        # dPdt: Producción asociada al crecimiento, dilución por flujo
        dPdt = Ypx * mu * X - (F / V) * P

        # dOdt: Transferencia de O2, consumo por crecimiento y mantenimiento, dilución por flujo
        # Asegurar que el término de consumo no haga O2 negativo si X es alto y O2 bajo
        consumo_o2 = (mu / Yxo + mo) * X
        dOdt = Kla * (Cs - O2) - consumo_o2 - (F / V) * O2

        # dVdt: Cambio de volumen igual al flujo de entrada
        dVdt = F

        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    y0 = [X0, S0, P0, O0, V0]
    t_span = [0, t_final]
    t_eval = np.linspace(t_span[0], t_span[1], 500) # Aumentado número de puntos para suavidad

    # Usar un método más robusto si es necesario (e.g., 'Radau', 'BDF') para sistemas stiff
    sol = solve_ivp(modelo_fedbatch, t_span, y0, t_eval=t_eval, method='RK45', atol=atol, rtol=rtol)

    # Verificar si la solución fue exitosa
    if not sol.success:
        st.error(f"La integración falló: {sol.message}")
        st.stop() # Detener la ejecución si falla la integración

    # Calcular el flujo para los tiempos de la solución
    flujo_sim = np.array([calcular_flujo(t) for t in sol.t])

    # --- Graficación ---
    fig = plt.figure(figsize=(15, 12)) # Ajustado tamaño

    # Gráfico Flujo y Volumen
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(sol.t, flujo_sim, 'r-', label='Flujo Alimentación [L/h]')
    ax1.set_ylabel('Flujo [L/h]', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Tiempo [h]')
    ax1.set_title('Perfil de Alimentación y Volumen')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax1b = ax1.twinx()
    ax1b.plot(sol.t, sol.y[4], 'b--', label='Volumen [L]')
    ax1b.set_ylabel('Volumen [L]', color='b')
    ax1b.tick_params(axis='y', labelcolor='b')
    ax1b.legend(loc='upper right')

    # Gráfico Biomasa
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.plot(sol.t, sol.y[0], 'g-')
    ax2.set_title('Biomasa (X) [g/L]')
    ax2.set_ylabel('[g/L]')
    ax2.set_xlabel('Tiempo [h]')
    ax2.grid(True)

    # Gráfico Sustrato
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.plot(sol.t, sol.y[1], 'm-')
    ax3.set_title('Sustrato (S) [g/L]')
    ax3.set_ylabel('[g/L]')
    ax3.set_xlabel('Tiempo [h]')
    ax3.grid(True)
    ax3.set_ylim(bottom=0) # Asegurar que el eje Y no sea negativo

    # Gráfico Producto
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.plot(sol.t, sol.y[2], 'k-')
    ax4.set_title('Producto (P) [g/L]')
    ax4.set_ylabel('[g/L]')
    ax4.set_xlabel('Tiempo [h]')
    ax4.grid(True)
    ax4.set_ylim(bottom=0) # Asegurar que el eje Y no sea negativo

    # Gráfico Oxígeno Disuelto
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(sol.t, sol.y[3], 'c-')
    ax5.set_title('Oxígeno Disuelto (O2) [mg/L]')
    ax5.set_ylabel('[mg/L]')
    ax5.set_xlabel('Tiempo [h]')
    ax5.grid(True)
    ax5.set_ylim(bottom=0, top=Cs*1.1) # Limitar eje Y superior

    plt.tight_layout(pad=3.0) # Añadir padding
    st.pyplot(fig)

    # Mostrar resultados finales (opcional)
    st.subheader("Resultados Finales (t = {:.1f} h)".format(sol.t[-1]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Volumen Final [L]", "{:.2f}".format(sol.y[4, -1]))
    col2.metric("Biomasa Final [g/L]", "{:.2f}".format(sol.y[0, -1]))
    col3.metric("Producto Final [g/L]", "{:.2f}".format(sol.y[2, -1]))
    col1.metric("Productividad Vol. P [g/L/h]", "{:.3f}".format(sol.y[2, -1] / sol.t[-1]) if sol.t[-1] > 0 else 0)
    col2.metric("Productividad Vol. X [g/L/h]", "{:.3f}".format(sol.y[0, -1] / sol.t[-1]) if sol.t[-1] > 0 else 0)
    col3.metric("Rendimiento P/S Total [g/g]", "{:.3f}".format( (sol.y[2,-1]*sol.y[4,-1] - P0*V0) / (S0*V0 + np.trapz(flujo_sim * Sin, sol.t) - sol.y[1,-1]*sol.y[4,-1]) ) if (S0*V0 + np.trapz(flujo_sim * Sin, sol.t) - sol.y[1,-1]*sol.y[4,-1]) > 1e-6 else 0)


if __name__ == '__main__':
    # Configuración de la página de Streamlit (opcional pero recomendado)
    st.set_page_config(layout="wide", page_title="Simulador Fed-Batch")
    # Cargar las funciones de cinética si están en un archivo separado
    # from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa
    lote_alimentado_page()