import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import TransferFunction, lsim
from sklearn.metrics import r2_score

# --- Process Model (Translation from MATLAB to Python) ---
def modelo_ph_planta(t, y, param):
    """
    Non-linear model of the fermentation bioprocess to simulate the real plant.
    """
    X, S, P, pH = y[0], y[1], y[2], y[3]
    if t < param['t_step']:
        D = param['D_base']
    else:
        D = param['D_base'] * (1 + param['step_percent'] / 100.0)
    
    f_pH = np.exp(-((pH - param['pH_opt'])**2) / (2 * param['pH_tol']**2))
    mu = param['mu_max'] * (S / (param['Ks'] + S)) * f_pH
    
    dXdt = (mu - D) * X
    dSdt = D * (param['S_in'] - S) - (1/param['Y_XS']) * mu * X
    dPdt = D * (param['P_in'] - P) + param['Y_PX'] * mu * X
    dpHdt = -param['alpha'] * P + D * (param['pH_in'] - pH)
    
    return [dXdt, dSdt, dPdt, dpHdt]

# --- Functions for System Identification (Robust Physical Parameterization) ---
def transfer_function_response(t, u, params, n_poles, has_delay):
    """
    Calculates the response of a transfer function model based on physical parameters.
    For n_poles=2: params = [K, tau, zeta, delay]
    For n_poles=1: params = [K, tau, delay]
    """
    K = params[0]
    tau = params[1]
    
    if n_poles == 2:
        zeta = params[2]
        den = [tau**2, 2*zeta*tau, 1]
    else: # n_poles == 1
        den = [tau, 1]
    
    num = [K]
    delay = params[-1] if has_delay else 0.0

    sys = TransferFunction(num, den)
    t_delayed = np.maximum(0, t - delay)
    u_delayed = np.interp(t_delayed, t, u)
    _, y_sim, _ = lsim(sys, U=u_delayed, T=t, interp=False)
    return y_sim

def error_function(params, t, u, y_meas, n_poles, has_delay):
    """
    Error function for the least squares optimization.
    """
    y_sim = transfer_function_response(t, u, params, n_poles, has_delay)
    return y_sim - y_meas

def format_tf_latex(params, n_poles, has_delay):
    """
    Formats the identified transfer function into a LaTeX string for display.
    """
    K = params[0]
    tau = params[1]
    delay = params[-1] if has_delay else 0.0

    if n_poles == 2:
        zeta = params[2]
        den_str = f"{tau**2:.4f}s^2 + {2*zeta*tau:.4f}s + 1"
    else: # n_poles == 1
        den_str = f"{tau:.4f}s + 1"
    
    num_str = f"{K:.4f}"
    delay_str = f"e^{{-{delay:.2f}s}}" if has_delay and delay > 0.01 else ""
    
    return f"G(s) = \\frac{{{num_str}}}{{{den_str}}} {delay_str}"

# --- Main Streamlit Page ---
def ph_identification_page():
    st.title("🧪 System Identification for a pH Bioprocess")
    st.markdown("""
    This application simulates a non-linear bioprocess model and then uses the generated data 
    to identify a linear model (transfer function) that describes the pH dynamics.
    """)

    st.sidebar.header("Configuration")
    
    with st.sidebar.expander("1. Simulation Parameters", expanded=True):
        param = {}
        param['D_base'] = st.slider("Base Dilution Rate (D)", 0.01, 0.2, 0.065, format="%.3f")
        param['t_step'] = st.slider("Time of Step Change in D (h)", 50, 250, 150)
        param['step_percent'] = st.slider("Step Change Magnitude (%)", 5, 50, 20)
        param['t_final'] = 300

    with st.sidebar.expander("2. System Identification Settings", expanded=True):
        n_poles = st.selectbox("Model Order (Poles)", [1, 2], index=1)
        has_delay = st.checkbox("Estimate Time Delay?", value=True)

    if st.sidebar.button("Run Simulation & Identification"):
        # --- 1. Process Simulation ---
        with st.spinner("Running non-linear process simulation..."):
            param.update({'mu_max': 0.5, 'Ks': 1.0, 'pH_opt': 5.5, 'pH_tol': 0.5, 'Y_XS': 0.5, 'Y_PX': 0.3, 'alpha': 0.1, 'S_in': 10.0, 'P_in': 0.0, 'pH_in': 7.0})
            y0 = [0.5, 5.0, 0.0, 6.0]
            t_span, t_eval = [0, param['t_final']], np.linspace(0, param['t_final'], 1000)
            sol = solve_ivp(lambda t, y: modelo_ph_planta(t, y, param), t_span, y0, t_eval=t_eval, method='LSODA')
            t, y = sol.t, sol.y.T

        st.header("1. Process Simulation Results")
        fig1, axs = plt.subplots(2, 2, figsize=(12, 8)); fig1.suptitle("Full Non-linear Simulation")
        axs[0, 0].plot(t, y[:, 0], 'b'); axs[0, 0].set_title('Biomass (X)'); axs[0,0].grid(True)
        axs[0, 1].plot(t, y[:, 1], 'r'); axs[0, 1].set_title('Substrate (S)'); axs[0,1].grid(True)
        axs[1, 0].plot(t, y[:, 2], 'g'); axs[1, 0].set_title('Product (P)'); axs[1,0].grid(True)
        axs[1, 1].plot(t, y[:, 3], 'm'); axs[1, 1].set_title('pH'); axs[1,1].grid(True)
        for ax in axs.flat: ax.set_xlabel('Time (h)')
        st.pyplot(fig1)

        # --- 2. Data Preparation (Corrected Logic) ---
        st.header("2. Data for System Identification")
        with st.spinner("Preparing data for identification..."):
            idx_start = np.where(t >= param['t_step'])[0][0]
            
            # Get steady-state values from just BEFORE the step
            y_ss = y[idx_start - 1, 3] # pH steady state
            u_ss = param['D_base']     # D steady state
            
            # Get data from the step change onwards
            t_ident = t[idx_start:] - t[idx_start]
            ph_slice = y[idx_start:, 3]
            
            d_input_full = np.ones_like(t) * param['D_base']
            d_input_full[t >= param['t_step']] = param['D_base'] * (1 + param['step_percent'] / 100.0)
            d_slice = d_input_full[idx_start:]

            # Calculate deviation from the pre-step steady state
            y_ident = ph_slice - y_ss
            u_ident = d_slice - u_ss

        st.markdown("Data is extracted after the perturbation and adjusted to represent deviations from the steady-state.")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t_ident, u_ident, 'k-'); ax1.set_title("Input Data (Change in D)"); ax1.grid(True)
        ax2.plot(t_ident, y_ident, 'm-'); ax2.set_title("Output Data (Change in pH)"); ax2.grid(True)
        st.pyplot(fig2)

        # --- 3. Model Identification ---
        st.header("3. Identification Results")
        with st.spinner(f"Identifying {n_poles}-pole model..."):
            gain_guess = y_ident[-1] / u_ident[-1] if u_ident[-1] != 0 else 0
            
            if n_poles == 2:
                p0, lb, ub = [gain_guess, 10.0, 1.0], [0, 0.1, 0.1], [np.inf, 100, 10]
            else:
                p0, lb, ub = [gain_guess, 10.0], [0, 0.1], [np.inf, 100]

            if has_delay:
                p0.append(1.0); lb.append(0.0); ub.append(t_ident[-1] / 2)
            
            res = least_squares(
                fun=error_function, x0=p0, bounds=(lb, ub),
                args=(t_ident, u_ident, y_ident, n_poles, has_delay),
                method='trf'
            )
            params_opt = res.x
        
        # --- 4. Display Results ---
        st.subheader("Identified Transfer Function")
        st.latex(format_tf_latex(params_opt, n_poles, has_delay))

        st.subheader("Model Validation")
        y_sim_opt = transfer_function_response(t_ident, u_ident, params_opt, n_poles, has_delay)
        r2 = r2_score(y_ident, y_sim_opt)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(t_ident, y_ident, 'm-', label='Real pH Response (from simulation)')
        ax3.plot(t_ident, y_sim_opt, 'k--', linewidth=2, label='Identified Model Response')
        ax3.set_title(f"Model Validation (R² = {r2:.4f})")
        ax3.set_xlabel("Time (h)"); ax3.set_ylabel("Change in pH")
        ax3.legend(); ax3.grid(True)
        st.pyplot(fig3)
        st.metric(label="Goodness of Fit (R-squared)", value=f"{r2:.4f}")
        st.caption("An R-squared value close to 1.0 indicates an excellent model fit.")

    else:
        st.info("Configure the parameters in the sidebar and click the button to start.")

if __name__ == '__main__':
    ph_identification_page()