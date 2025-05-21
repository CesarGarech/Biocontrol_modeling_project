import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, aiba  # Import kinetic functions

def lote_page():
    st.header("Modo de operación: Lote")
    st.sidebar.subheader("Parámetros del modelo")

    # Parámetros generales
    mumax = st.sidebar.slider("μmax", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider("Ks", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Oxígeno saturado (Cs)", 0.1, 10.0, 8.0)
    V = st.sidebar.slider("Volumen del biorreactor (L)", 0.5, 10.0, 2.0)
    ms = st.sidebar.slider("Mantenimiento (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decaimiento (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("Mantenimiento O2 (mo)", 0.0, 0.5, 0.05)

    I = st.sidebar.slider(" Intensidad de luz (W m^-2)", 0.0, 83.0, 40.0)
    Ki= st.sidebar.slider(" Constante de inhibición (W/m○^2)", 0.0, 1000.0, 959.2)
    KiL= st.sidebar.slider(" Constante de photoinhibición (m^2/W", 0.01, 1.0, 0.58)
    # Iniciales
    X0 = st.sidebar.number_input("Biomasa inicial (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Sustrato inicial (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Producto inicial (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("O2 disuelto inicial (mg/L)", 0.0, 10.0, 5.0)

    # Tipo de cinética
    tipo_mu = st.sidebar.selectbox("Tipo de cinética", ["Monod simple", "Monod sigmoidal", "Monod con restricciones","Aiba"])

    # Tiempo de simulación
    t_final = st.sidebar.slider("Tiempo final (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)

    # Tolerancias
    atol = st.sidebar.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def modelo_lote(t, y):
        X, S, P, O2 = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n=2)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)
        elif tipo_mu=="Aiba":
            mu = aiba(mumax,I,Ki,KiL)

        dXdt = mu * X - Kd * X
        dSdt = -1/Yxs * mu * X - ms * X
        dPdt = Ypx * mu * X
        dOdt = Kla * (Cs - O2) - (1/Yxo) * mu * X - mo * X
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(modelo_lote, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

    # Gráficas
    st.subheader("Resultados de simulación")
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='Biomasa (X)')
    ax.plot(sol.t, sol.y[1], label='Sustrato (S)')
    ax.plot(sol.t, sol.y[2], label='Producto (P)')
    ax.plot(sol.t, sol.y[3], label='Oxígeno disuelto (O2)')
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("Concentración (g/L o mg/L)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == '__main__':
    lote_page()