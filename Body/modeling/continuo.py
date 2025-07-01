import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

def continuo_page():
    st.header("Operation mode: Continuous (Chemostat)")
    st.sidebar.subheader("Model Parameters")

    tipo_mu = st.sidebar.selectbox("Kinetic model", ["Simple Monod", "Sigmoidal Monod", "Monod with restrictions"])
    mumax = st.sidebar.slider("μmax", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider("Ks", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Saturated Oxygen (Cs)", 0.1, 10.0, 8.0)
    ms = st.sidebar.slider("Maintenance (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decay (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("O2 Maintenance (mo)", 0.0, 0.5, 0.05)
    Sin = st.sidebar.slider("Substrate in Feed (Sin)", 0.0, 100.0, 50.0)
    D = st.sidebar.slider("Dilution Rate D (1/h)", 0.0, 1.0, 0.01)

    X0 = st.sidebar.number_input("Initial Biomass  (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Initial Substrate (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Initial Product (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("Initial dissolved O2 (mg/L)", 0.0, 10.0, 5.0)

    n_sigmoidal = st.sidebar.slider("n value (Sigmoidal Monod)", 0.0, 5.0, 2.0)

    t_final = st.sidebar.slider("Final time (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)
    atol = st.sidebar.number_input("Absolute tolerance (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Relative tolerance (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def modelo_continuo(t, y):
        X, S, P, O2 = y
        if tipo_mu == "Simple Monod":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Sigmoidal Monod":
            mu = mu_sigmoidal(S, mumax, Ks, n=n_sigmoidal)
        elif tipo_mu == "Monod with restrictions":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)

        dXdt = mu * X - Kd * X - D * X
        dSdt = -1/Yxs * mu * X - ms * X + D * (Sin - S)
        dPdt = Ypx * mu * X - D * P
        dOdt = Kla * (Cs - O2) - 1/Yxo * mu * X - mo * X - D * O2
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(modelo_continuo, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

    st.subheader("Simulation Results")
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='Biomass (X)')
    ax.plot(sol.t, sol.y[1], label='Substrate (S)')
    ax.plot(sol.t, sol.y[2], label='Product (P)')
    ax.plot(sol.t, sol.y[3], label='Dissolved Oxygen (O2)')
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (g/L o mg/L)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == '__main__':
    continuo_page()