import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

def continuo_page():
    st.header("Modo de operación: Continuo (Chemostato)")
    st.sidebar.subheader("Parámetros del modelo")

    mumax = st.sidebar.slider("μmax", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider("Ks", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Oxígeno saturado (Cs)", 0.1, 10.0, 8.0)
    ms = st.sidebar.slider("Mantenimiento (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decaimiento (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("Mantenimiento O2 (mo)", 0.0, 0.5, 0.05)
    Sin = st.sidebar.slider("Sustrato en alimentación (Sin)", 0.0, 100.0, 50.0)
    D = st.sidebar.slider("Tasa de dilución D (1/h)", 0.0, 1.0, 0.01)

    X0 = st.sidebar.number_input("Biomasa inicial (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Sustrato inicial (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Producto inicial (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("O2 disuelto inicial (mg/L)", 0.0, 10.0, 5.0)

    tipo_mu = st.sidebar.selectbox("Tipo de cinética", ["Monod simple", "Monod sigmoidal", "Monod con restricciones"])
    n_sigmoidal = st.sidebar.slider("Valor de n (Monod sigmoidal)", 0.0, 5.0, 2.0)

    t_final = st.sidebar.slider("Tiempo final (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)
    atol = st.sidebar.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def modelo_continuo(t, y):
        X, S, P, O2 = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n=n_sigmoidal)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)

        dXdt = mu * X - Kd * X - D * X
        dSdt = -1/Yxs * mu * X - ms * X + D * (Sin - S)
        dPdt = Ypx * mu * X - D * P
        dOdt = Kla * (Cs - O2) - 1/Yxo * mu * X - mo * X - D * O2
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(modelo_continuo, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

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
    continuo_page()