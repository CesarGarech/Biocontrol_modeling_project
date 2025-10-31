import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa  # Import kinetic functions


def batch_page():
    st.header("Operation mode: Batch")
    st.sidebar.subheader("Model Parameters")
    # Kinetic type
    kinetic_type = st.sidebar.selectbox("Kinetic model", ["Simple Monod", "Sigmoidal Monod", "Monod with restrictions"])
    if kinetic_type == "Simple Monod":
        st.markdown(""" 
        ## Simple Monod Kinetics
        The Simple Monod model describes the relationship between the specific 
        growth rate of microorganisms (μ) and the substrate concentration (S), 
        through the following equation:      
        """)
        
        st.latex(r"""
                    \mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S}
                    """)

        st.markdown("""
            Where:
            - $\\mu$ is the specific growth rate (1/h)
            - $\\mu_{\\text{max}}$ is the maximum specific growth rate (1/h)
            - $S$ is the substrate concentration (g/L)
            - $K_s$ is the saturation constant (g/L)
            """)
                
    elif kinetic_type == "Sigmoidal Monod":
        st.markdown(""" 
        ## Sigmoidal Monod Kinetics
        The sigmoidal Monod model is an extension of the simple Monod model, 
        which describes the specific growth rate of microorganisms (μ) as a function 
        of substrate concentration (S) using a sigmoidal function:
        """)
        
        st.latex(r"""
                    \mu = \mu_{\text{max}} \cdot \frac{S^n}{K_s^n + S^n}
                    """)

        st.markdown("""
            Where:
            - $\\mu$ is the specific growth rate (1/h)
            - $\\mu_{\\text{max}}$ is the maximum specific growth rate (1/h)
            - $S$ is the substrate concentration (g/L)
            - $K_s$ is the saturation constant (g/L)
            - $n$ is the Hill coefficient
            """)
    elif kinetic_type == "Monod with restrictions":
        st.markdown(""" 
        ## Monod Kinetics with Restrictions
        The Monod model with restrictions considers the effect of product (P) 
        and dissolved oxygen (O2) on the specific growth rate (μ):
        """)
        
        st.latex(r"""
                    \mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S} \cdot \frac{O_2}{K_O + O_2} \cdot \frac{K_P}{K_P + P}
                    """)

        st.markdown("""
            Where:
            - $\\mu$ is the specific growth rate (1/h)
            - $\\mu_{\\text{max}}$ is the maximum specific growth rate (1/h)
            - $S$ is the substrate concentration (g/L)
            - $K_s$ is the saturation constant (g/L)
            - $P$ is the product concentration (g/L)
            - $K_P$ is the product inhibition constant (g/L)
            - $O_2$ is the dissolved oxygen concentration (mg/L)
            - $K_O$ is the oxygen inhibition constant (mg/L)
            """)
    # General parameters
    
    mumax = st.sidebar.slider(r"$\mu_{\mathrm{max}}$", 0.1, 1.0, 0.3)
    Ks = st.sidebar.slider(r"$\ K_{\mathrm{s}}$", 0.01, 1.0, 0.1)
    Yxs = st.sidebar.slider("Yxs", 0.1, 1.0, 0.5)
    Ypx = st.sidebar.slider("Ypx", 0.1, 1.0, 0.3)
    Yxo = st.sidebar.slider("Yxo", 0.1, 1.0, 0.3)
    Kla = st.sidebar.slider("kLa", 0.1, 100.0, 20.0)
    Cs = st.sidebar.slider("Saturated Oxygen (Cs)", 0.1, 10.0, 8.0)
    V = st.sidebar.slider("Bioreactor Volumen (L)", 0.5, 10.0, 2.0)
    ms = st.sidebar.slider("Maintenance (ms)", 0.0, 0.5, 0.005)
    Kd = st.sidebar.slider("Decay (Kd)", 0.0, 0.5, 0.005)
    mo = st.sidebar.slider("O2 Maintenance (mo)", 0.0, 0.5, 0.05)

    # Initial conditions
    X0 = st.sidebar.number_input("Initial Biomass (g/L)", 0.1, 10.0, 0.5)
    S0 = st.sidebar.number_input("Initial Substrate (g/L)", 0.1, 100.0, 20.0)
    P0 = st.sidebar.number_input("Initial Product (g/L)", 0.0, 50.0, 0.0)
    O0 = st.sidebar.number_input("Initial dissolved O2 (mg/L)", 0.0, 10.0, 5.0)

    

    # Simulation time
    t_final = st.sidebar.slider("Final time (h)", 1, 100, 30)
    t_eval = np.linspace(0, t_final, 300)

    # Tolerances
    atol = st.sidebar.number_input("Absolute tolerance (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
    rtol = st.sidebar.number_input("Relative tolerance (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def batch_model(t, y):
        X, S, P, O2 = y
        if kinetic_type == "Simple Monod":
            mu = mu_monod(S, mumax, Ks)
        elif kinetic_type == "Sigmoidal Monod":
            mu = mu_sigmoidal(S, mumax, Ks, n=2)
            if S<=0:
                S=0
        elif kinetic_type == "Monod with restrictions":
            mu = mu_completa(S, O2, P, mumax, Ks, KO=0.5, KP=0.5)
        dXdt = mu * X - Kd * X
        dSdt = -1/Yxs * mu * X - ms * X
        if S<=0:
            dSdt=0
        dPdt = Ypx * mu * X
        dOdt = Kla * (Cs - O2) - (1/Yxo) * mu * X - mo * X
        return [dXdt, dSdt, dPdt, dOdt]

    y0 = [X0, S0, P0, O0]
    sol = solve_ivp(batch_model, [0, t_final], y0, t_eval=t_eval, atol=atol, rtol=rtol)

    # Plots
    st.subheader("Simulation Results")
    st.markdown("""
        The following graphs show the evolution of the concentrations of biomass (X), substrate (S), product (P), and dissolved oxygen (O2) over time.
    """)
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
    batch_page()