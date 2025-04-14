import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa

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
            KP = st.slider("Constante inhibición producto [g/L]", 0.1, 10.0, 0.5)

        Yxs = st.slider("Yxs [g/g]", 0.1, 1.0, 0.6)
        Ypx = st.slider("Ypx [g/g]", 0.0, 1.0, 0.3)
        Yxo = st.slider("Yxo [g/g]", 0.1, 1.0, 0.2)
        Kla = st.slider("kLa [1/h]", 1.0, 200.0, 50.0)
        Cs = st.slider("O2 Saturado [mg/L]", 5.0, 15.0, 8.0)
        Sin = st.slider("Sustrato en Alimentado [g/L]", 50.0, 300.0, 150.0)
        ms = st.slider("Mantenimiento S [g/g/h]", 0.0, 0.1, 0.001)
        Kd = st.slider("Decaimiento X [g/g/h]", 0.0, 0.1, 0.02)
        mo = st.slider("Mantenimiento O2 [g/g/h]", 0.0, 0.1, 0.01)

        st.subheader("Estrategia de Alimentación")
        estrategia = st.selectbox("Tipo", ["Constante", "Exponencial", "Escalon"])
        F_base = st.slider("Flujo Base [L/h]", 0.01, 5.0, 0.5)
        t_alim_inicio = st.slider("Inicio Alimentación [h]", 0, 24, 2)
        t_alim_fin = st.slider("Fin Alimentación [h]", 5, 48, 24)

        st.subheader("Condiciones Iniciales")
        V0 = st.number_input("Volumen Inicial [L]", 1.0, 10.0, 3.0)
        X0 = st.number_input("Biomasa Inicial [g/L]", 0.1, 10.0, 1.0)
        S0 = st.number_input("Sustrato Inicial [g/L]", 0.1, 100.0, 30.0)
        P0 = st.number_input("Producto Inicial [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("O2 Inicial [mg/L]", 0.0, 10.0, 8.0)

        t_final = st.slider("Tiempo de Simulación [h]", 10, 100, 48)
        atol = st.number_input("Tolerancia absoluta (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.number_input("Tolerancia relativa (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def calcular_flujo(t):
        if t_alim_inicio <= t <= t_alim_fin:
            if estrategia == "Constante":
                return F_base
            elif estrategia == "Exponencial":
                return F_base * np.exp(0.15 * (t - t_alim_inicio))
            elif estrategia == "Escalon":
                return F_base * 2 if t > (t_alim_inicio + t_alim_fin)/2 else F_base
        return 0.0

    def modelo_fedbatch(t, y):
        X, S, P, O2, V = y
        if tipo_mu == "Monod simple":
            mu = mu_monod(S, mumax, Ks)
        elif tipo_mu == "Monod sigmoidal":
            mu = mu_sigmoidal(S, mumax, Ks, n)
        elif tipo_mu == "Monod con restricciones":
            mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)

        F = calcular_flujo(t)
        dXdt = mu*X - Kd*X/V -  (F/V)*X
        dSdt = (-mu/Yxs - ms)*X + (F/V)*(Sin - S)
        dPdt = Ypx*mu*X - (F/V)*P
        dOdt = Kla*(Cs - O2) - (mu/Yxo + mo)*X - (F/V)*O2
        dVdt = F
        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    y0 = [X0, S0, P0, O0, V0]
    t_eval = np.linspace(0, t_final, 300)
    sol = solve_ivp(modelo_fedbatch, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)
    flujo_sim = [calcular_flujo(t) for t in sol.t]

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    ax1.plot(sol.t, flujo_sim, 'r-', label='Flujo de Alimentación')
    ax1.set_ylabel('Flujo [L/h]', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1b = ax1.twinx()
    ax1b.plot(sol.t, sol.y[4], 'b--', label='Volumen')
    ax1b.set_ylabel('Volumen [L]', color='b')
    ax1b.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Perfil de Alimentación y Volumen')
    ax1.grid(True)

    ax2.plot(sol.t, sol.y[0], 'g-')
    ax2.set_title('Biomasa [g/L]')
    ax2.grid(True)

    ax3.plot(sol.t, sol.y[1], 'm-')
    ax3.set_title('Sustrato [g/L]')
    ax3.grid(True)

    ax4.plot(sol.t, sol.y[2], 'k-')
    ax4.set_title('Producto [g/L]')
    ax4.grid(True)

    ax5.plot(sol.t, sol.y[3], 'c-')
    ax5.set_title('Oxígeno Disuelto [mg/L]')
    ax5.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == '__main__':
    lote_alimentado_page()