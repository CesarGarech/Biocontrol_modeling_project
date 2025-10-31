import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Assuming you have a Utils/kinetics.py file with the mu functions
# If not, you can define them directly here or adjust the import.
# Example definitions (if you don't have kinetics.py):
def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)
def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * (S**n) / (Ks**n + S**n)
def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    # Ensure values are not negative to avoid mathematical errors
    S = max(0, S)
    O2 = max(0, O2)
    P = max(0, P)
    # Product inhibition term (simple model, adjust if necessary)
    # Avoid division by zero if KP is very small or P is large
    inhibition_P = (1 - P / KP) if KP > 0 and P < KP else 0
    inhibition_P = max(0, inhibition_P) # Ensure it's not negative

    # Monod for Substrate and Oxygen, with Product inhibition
    mu = mumax * (S / (Ks + S)) * (O2 / (KO + O2)) * inhibition_P
    return max(0, mu) # Ensure growth rate is not negative

# --- End example definitions ---

def fed_batch_page():
    st.header("Operation Mode: Fed-Batch ") 
    st.sidebar.subheader("Model Parameters")
    kinetic_type = st.sidebar.selectbox("Kinetic Model", ["Simple Monod", "Sigmoidal Monod", "Monod with restrictions"])
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
                    \mu = \mu_{\text{max}} \cdot \frac{S}{K_s + S} \cdot \frac{O_2}{K_O + O_2} \cdot (1-\frac{K_P}{P})
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
    
   
    with st.sidebar:    
        # Definition of model parameters
        mumax = st.slider(r"$\mu_{\mathrm{max}}$", 0.1, 1.0, 0.4)
        Ks = st.slider(r"$\ K_{\mathrm{s}}$[g/L]", 0.01, 2.0, 0.5)

        if kinetic_type == "Sigmoidal Monod":
            n = st.slider("Sigmoidal exponent (n)", 1, 5, 2)
        elif kinetic_type == "Monod with restrictions":
            KO = st.slider("O2 saturation constant [mg/L]", 0.1, 5.0, 0.5)
            KP = st.slider("Product inhibition constant [g/L]", 0.1, 10.0, 5.0) # Adjusted default value

        Yxs = st.slider("Yxs [g/g]", 0.1, 1.0, 0.6)
        Ypx = st.slider("Ypx [g/g]", 0.0, 1.0, 0.3)
        Yxo = st.slider("Yxo [g/g]", 0.1, 1.0, 0.2)
        Kla = st.slider("kLa [1/h]", 1.0, 200.0, 50.0)
        Cs = st.slider("Saturated O2 [mg/L]", 5.0, 15.0, 8.0)
        Sin = st.slider("Substrate in Feed [g/L]", 50.0, 300.0, 150.0)
        ms = st.slider("S Maintenance [g/g/h]", 0.0, 0.1, 0.001)
        Kd = st.slider("X Decay [1/h]", 0.0, 0.1, 0.02) # Units corrected
        mo = st.slider("O2 Maintenance [g/g/h]", 0.0, 0.1, 0.01)

        st.subheader("Feeding Strategy")
        # Added "Linear" option
        feeding_strategy = st.selectbox("Type", ["Constant", "Exponential", "Step", "Linear"])
        F_base = st.slider("Base Flow (or Initial for Linear) [L/h]", 0.01, 5.0, 0.5)
        t_feed_start = st.slider("Start Feeding [h]", 0.0, 24.0, 2.0, 0.5) # Adjusted to float
        t_feed_end = st.slider("End Feeding [h]", t_feed_start + 0.1, 48.0, 24.0, 0.5) # Ensures t_end > t_start

        # Additional parameter only for Linear strategy
        F_linear_end = 0.0 # Initialize
        if feeding_strategy == "Linear":
            F_linear_end = st.slider("Final Flow (Linear) [L/h]", F_base, 10.0, F_base * 2) # Ensures F_end >= F_base

        st.subheader("Initial Conditions")
        V0 = st.number_input(" Initial Volume [L]", 1.0, 100.0, 3.0) # Increased range
        X0 = st.number_input("Initial Biomass [g/L]", 0.1, 50.0, 1.0) # Increased range
        S0 = st.number_input("Initial Substrate [g/L]", 0.1, 100.0, 30.0)
        P0 = st.number_input("Initial Product [g/L]", 0.0, 50.0, 0.0)
        O0 = st.number_input("Initial O2 [mg/L]", 0.0, float(Cs), float(Cs)) # Initial O2 equal to saturated

        t_final = st.slider("Simulation Time [h]", max(10.0, t_feed_end + 1), 200.0, 48.0, 1.0) # Increased and adjusted range
        atol = st.number_input("Absolute Tolerance (atol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")
        rtol = st.number_input("Relative Tolerance (rtol)", min_value=1e-10, max_value=1e-2, value=1e-6, format="%e")

    def calculate_flow(t):
        if t_feed_start <= t <= t_feed_end:
            if feeding_strategy == "Constant":
                return F_base
            elif feeding_strategy == "Exponential":
                 # Ensure the exponent doesn't cause overflow if t is large
                exponent = 0.15 * (t - t_feed_start)
                try:
                    return F_base * np.exp(exponent)
                except OverflowError:
                    return float('inf') # Or a reasonable maximum value
            elif feeding_strategy == "Step":
                # Use a clear midpoint between start and end
                t_mid = t_feed_start + (t_feed_end - t_feed_start) / 2
                return F_base * 2 if t > t_mid else F_base
            elif feeding_strategy == "Linear":
                delta_t = t_feed_end - t_feed_start
                if delta_t > 0:
                    slope = (F_linear_end - F_base) / delta_t
                    return F_base + slope * (t - t_feed_start)
                else:
                    # If interval is 0, return initial flow (or final, as preferred)
                    return F_base # Or F_linear_end
        return 0.0

    def fedbatch_model(t, y):
        X, S, P, O2, V = y

        # Ensure concentrations are not negative (can occur due to numerical errors)
        X = max(0, X)
        S = max(0, S)
        P = max(0, P)
        O2 = max(0, O2)
        V = max(1e-6, V) # Avoid division by zero if V becomes very small

        if kinetic_type == "Simple Monod":
            mu = mu_monod(S, mumax, Ks)
        elif kinetic_type == "Sigmoidal Monod":
            mu = mu_sigmoidal(S, mumax, Ks, n)
        elif kinetic_type == "Monod with restrictions":
            mu = mu_completa(S, O2, P, mumax, Ks, KO, KP)
        else:
             mu = 0 # Default case or error

        mu = max(0, mu) # Ensure mu is not negative

        F = calculate_flow(t)

        # Correction in Biomass equation: Decay term is not divided by V
        # dXdt = mu*X - Kd*X - (F/V)*X # Standard model without volume decay
        # Correction: Concentration changes by dilution (F/V)*X and by intrinsic growth/decay
        dXdt = (mu - Kd) * X - (F / V) * X

        # dSdt: Consumption by growth and maintenance, input by flow, dilution by flow
        dSdt = -(mu / Yxs + ms) * X + (F / V) * (Sin - S)

        # dPdt: Production associated with growth, dilution by flow
        dPdt = Ypx * mu * X - (F / V) * P

        # dOdt: O2 transfer, consumption by growth and maintenance, dilution by flow
        # Ensure consumption term doesn't make O2 negative if X is high and O2 low
        o2_consumption = (mu / Yxo + mo) * X
        dOdt = Kla * (Cs - O2) - o2_consumption - (F / V) * O2

        # dVdt: Volume change equal to input flow
        dVdt = F

        return [dXdt, dSdt, dPdt, dOdt, dVdt]

    y0 = [X0, S0, P0, O0, V0]
    t_span = [0, t_final]
    t_eval = np.linspace(t_span[0], t_span[1], 500) # Increased number of points for smoothness

    # Use a more robust method if necessary (e.g., 'Radau', 'BDF') for stiff systems
    sol = solve_ivp(fedbatch_model, t_span, y0, t_eval=t_eval, method='RK45', atol=atol, rtol=rtol)

    # Verify if solution was successful
    if not sol.success:
        st.error(f"Integration failed: {sol.message}")
        st.stop() # Stop execution if integration fails

    # Calculate flow for solution times
    flow_sim = np.array([calculate_flow(t) for t in sol.t])

    # Plots
    st.subheader("Simulation Results")
    st.markdown("""
        The following graphs show the evolution of the concentrations of biomass (X), substrate (S), product (P), and dissolved oxygen (O2) over time.
    """)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(15, 12)) # Adjusted size

    # Flow and Volume Plot
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(sol.t, flow_sim, 'r-', label='Feed Flow [L/h]')
    ax1.set_ylabel('Flow [L/h]', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel('Time [h]')
    ax1.set_title('Feeding and Volume Profile')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax1b = ax1.twinx()
    ax1b.plot(sol.t, sol.y[4], 'b--', label='Volume [L]')
    ax1b.set_ylabel('Volume [L]', color='b')
    ax1b.tick_params(axis='y', labelcolor='b')
    ax1b.legend(loc='upper right')

    # Biomass Plot
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.plot(sol.t, sol.y[0], 'g-')
    ax2.set_title('Biomass (X) [g/L]')
    ax2.set_ylabel('[g/L]')
    ax2.set_xlabel('Time [h]')
    ax2.grid(True)

    # Substrate Plot
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.plot(sol.t, sol.y[1], 'm-')
    ax3.set_title('Substrate (S) [g/L]')
    ax3.set_ylabel('[g/L]')
    ax3.set_xlabel('Time [h]')
    ax3.grid(True)
    ax3.set_ylim(bottom=0) # Ensure Y axis is not negative

    # Product Plot
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.plot(sol.t, sol.y[2], 'k-')
    ax4.set_title('Product (P) [g/L]')
    ax4.set_ylabel('[g/L]')
    ax4.set_xlabel('Time [h]')
    ax4.grid(True)
    ax4.set_ylim(bottom=0) # Ensure Y axis is not negative

    # Dissolved Oxygen Plot
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(sol.t, sol.y[3], 'c-')
    ax5.set_title('Dissolved Oxygen (O2) [mg/L]')
    ax5.set_ylabel('[mg/L]')
    ax5.set_xlabel('Time [h]')
    ax5.grid(True)
    ax5.set_ylim(bottom=0, top=Cs*1.1) # Limit upper Y axis

    plt.tight_layout(pad=3.0) # Add padding
    st.pyplot(fig)

    # Show final results (optional)
    st.subheader("Final Results (t = {:.1f} h)".format(sol.t[-1]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Volume [L]", "{:.2f}".format(sol.y[4, -1]))
    col2.metric("Final Biomass [g/L]", "{:.2f}".format(sol.y[0, -1]))
    col3.metric("Final Product [g/L]", "{:.2f}".format(sol.y[2, -1]))
    col1.metric("Productivity Vol. P [g/L/h]", "{:.3f}".format(sol.y[2, -1] / sol.t[-1]) if sol.t[-1] > 0 else 0)
    col2.metric("Productivity Vol. X [g/L/h]", "{:.3f}".format(sol.y[0, -1] / sol.t[-1]) if sol.t[-1] > 0 else 0)
    col3.metric("Total P/S Yield [g/g]", "{:.3f}".format( (sol.y[2,-1]*sol.y[4,-1] - P0*V0) / (S0*V0 + np.trapz(flow_sim * Sin, sol.t) - sol.y[1,-1]*sol.y[4,-1]) ) if (S0*V0 + np.trapz(flow_sim * Sin, sol.t) - sol.y[1,-1]*sol.y[4,-1]) > 1e-6 else 0)


if __name__ == '__main__':
    # Streamlit page configuration (optional but recommended)
    st.set_page_config(layout="wide", page_title="Fed-Batch Simulator")
    # Load kinetic functions if they are in a separate file
    # from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa
    fed_batch_page()