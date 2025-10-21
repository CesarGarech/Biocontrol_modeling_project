# Body/control/regulatorio/reg_cascada_oxigeno.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import traceback

# --- PID Controller Class (copied from previous example) ---
# We use a class to create independent instances for each controller
class PIDController:
    """A simple discrete PID controller class."""

    def __init__(self, Kp, Ki, Kd, Ts, setpoint=0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.setpoint = setpoint

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_meas = 0.0 # Added for derivative on measurement

        self.min_output, self.max_output = output_limits

    def compute(self, measured_value):
        """Calculates the controller output."""
        error = self.setpoint - measured_value

        # Proportional term
        proportional = self.Kp * error

        # Integral term (with anti-windup implicitly handled by output limits later)
        self.integral += self.Ki * error * self.Ts

        # Derivative term (on measurement to reduce derivative kick)
        derivative = 0.0
        if self.Ts > 1e-9: # Avoid division by zero
            derivative = self.Kd * (measured_value - self.prev_meas) / self.Ts

        # Compute output
        output = proportional + self.integral - derivative # Note the minus for derivative on measurement

        # Store current error and measurement for next iteration
        self.prev_error = error
        self.prev_meas = measured_value

        # Apply output saturation (basic anti-windup)
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)

        # Optional: Clamp integral term if output is saturated (more robust anti-windup)
        # if output != (proportional + self.integral - derivative):
        #     self.integral -= self.Ki * error * self.Ts # Revert last integral step if saturated

        return output

    def update_setpoint(self, new_setpoint):
        """Updates the controller's setpoint."""
        self.setpoint = new_setpoint

    def reset(self, initial_measurement=0):
        """Resets the integral and previous error/measurement."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_meas = initial_measurement # Initialize prev_meas


# --- Main Streamlit Page Function ---
def regulatorio_cascade_oxigen_page():
    """
    Streamlit page for simulating cascade control of Dissolved Oxygen (DO).
    """
    st.header("💧 Dissolved Oxygen Cascade Control Simulation")
    st.markdown("""
    This page simulates a **cascade control** strategy for Dissolved Oxygen (DO) in a bioreactor.
    Cascade control is useful when the primary variable (DO) is slow, but its manipulated
    variable (Agitation RPM) is faster and subject to its own disturbances.

    **Concept:**
    1.  **Primary (Outer) Loop:** Controls the main objective, **DO**. It measures DO and compares it to the DO setpoint. Its output is *not* the final motor power, but the **setpoint for the Agitation (RPM)**. This loop is typically tuned slower.
    2.  **Secondary (Inner) Loop:** Controls the **Agitation (RPM)**. It measures RPM and compares it to the RPM setpoint received from the outer loop. Its output *is* the **motor power**. This loop is tuned faster to quickly correct RPM deviations before they significantly affect DO.

    **Advantages:** The inner loop quickly rejects disturbances affecting the secondary variable (RPM), improving the stability and performance of the primary loop (DO).
    """)
    st.markdown("---")

    # --- User Inputs in Sidebar ---
    with st.sidebar:
        st.header("Simulation Parameters")

        with st.expander("1. Simulation Timing", expanded=True):
            Ts_sim = st.number_input("Sample Time (Ts) [s]", 0.1, 10.0, 1.0, 0.1, key="cas_Ts")
            t_final_sim = st.number_input("Final Simulation Time [s]", 100.0, 5000.0, 1000.0, 50.0, key="cas_tf")

        with st.expander("2. Process Models (First-Order)", expanded=True):
            st.markdown("**Motor (Power -> RPM)**")
            K_motor = st.number_input("Motor Gain (K_motor) [RPM/%Power]", 1.0, 20.0, 10.0, 0.5, key="cas_kmot")
            tau_motor = st.number_input("Motor Time Constant (τ_motor) [s]", 1.0, 50.0, 10.0, 1.0, key="cas_tmot")
            st.markdown("**DO Process (RPM -> %DO)**")
            K_do = st.number_input("DO Process Gain (K_do) [%DO/RPM]", 0.01, 1.0, 0.1, 0.01, key="cas_kdo")
            tau_do = st.number_input("DO Time Constant (τ_do) [s]", 10.0, 500.0, 80.0, 10.0, key="cas_tdo")

        with st.expander("3. Sensor Models (First-Order)", expanded=True):
            tau_sensor_rpm = st.number_input("RPM Sensor Time Constant (τ_s_rpm) [s]", 0.1, 20.0, 2.0, 0.5, key="cas_tsrpm")
            tau_sensor_do = st.number_input("DO Sensor Time Constant (τ_s_do) [s]", 1.0, 60.0, 15.0, 1.0, key="cas_tsdo")

        with st.expander("4. PID Controllers", expanded=True):
            st.markdown("**Outer Loop (DO -> RPM_sp)**")
            Kp_DO = st.number_input("DO Kp", 0.0, 100.0, 4.0, 0.1, key="cas_kp_do")
            Ki_DO = st.number_input("DO Ki", 0.0, 10.0, 0.05, 0.01, key="cas_ki_do")
            Kd_DO = st.number_input("DO Kd", 0.0, 5.0, 0.1, 0.01, key="cas_kd_do")
            rpm_min = st.number_input("Min RPM Output", 0.0, 500.0, 0.0, 10.0, key="cas_rpmmin")
            rpm_max = st.number_input("Max RPM Output", 100.0, 1000.0, 1000.0, 50.0, key="cas_rpmmax")

            st.markdown("**Inner Loop (RPM -> Power)**")
            Kp_RPM = st.number_input("RPM Kp", 0.0, 10.0, 0.8, 0.1, key="cas_kp_rpm")
            Ki_RPM = st.number_input("RPM Ki", 0.0, 5.0, 0.2, 0.05, key="cas_ki_rpm")
            Kd_RPM = st.number_input("RPM Kd", 0.0, 2.0, 0.0, 0.01, key="cas_kd_rpm")
            power_min = 0.0
            power_max = 100.0
            st.caption(f"Power Output Limits: [{power_min:.1f}%, {power_max:.1f}%]")

        with st.expander("5. Setpoint & Disturbance", expanded=True):
            st.markdown("**DO Setpoint**")
            sp_initial_do = st.number_input("Initial DO Setpoint [% Sat]", 0.0, 100.0, 10.0, 1.0, key="cas_sp_init")
            sp_final_do = st.number_input("Final DO Setpoint [% Sat]", 0.0, 100.0, 30.0, 1.0, key="cas_sp_final")
            t_step_do = st.number_input("Setpoint Change Time [s]", 0.0, t_final_sim, 50.0, 10.0, key="cas_tstep")

            st.markdown("**Power Disturbance**")
            dist_mag = st.number_input("Disturbance Magnitude [% Power]", -50.0, 50.0, -20.0, 5.0, key="cas_dist_mag")
            t_dist_start = st.number_input("Disturbance Start Time [s]", 0.0, t_final_sim, t_final_sim * 0.7, 50.0, key="cas_tdist")

    # --- Simulation Area ---
    st.subheader("Simulation")

    if st.button("▶️ Run Cascade Simulation", key="run_cascade_sim"):
        try:
            # --- Simulation Setup ---
            Ts = Ts_sim
            t = np.arange(0, t_final_sim + Ts, Ts)
            N = len(t)

            # --- Discrete Process Models (First-Order Difference Equations) ---
            alpha_motor = np.exp(-Ts / max(1e-6, tau_motor))
            alpha_do = np.exp(-Ts / max(1e-6, tau_do))
            alpha_sensor_rpm = np.exp(-Ts / max(1e-6, tau_sensor_rpm))
            alpha_sensor_do = np.exp(-Ts / max(1e-6, tau_sensor_do))

            # --- Controller Initialization ---
            pid_DO = PIDController(Kp_DO, Ki_DO, Kd_DO, Ts, output_limits=(rpm_min, rpm_max))
            pid_RPM = PIDController(Kp_RPM, Ki_RPM, Kd_RPM, Ts, output_limits=(power_min, power_max))

            # --- Initialization of Simulation Arrays ---
            DO_sp_vec = np.zeros(N)
            DO_actual = np.zeros(N)
            DO_meas = np.zeros(N)
            RPM_sp_vec = np.zeros(N)
            RPM_actual = np.zeros(N)
            RPM_meas = np.zeros(N)
            Power = np.zeros(N)
            Power_dist_vec = np.zeros(N)

            # --- Set Initial Conditions ---
            DO_actual[0] = sp_initial_do # Start at initial setpoint
            DO_meas[0] = DO_actual[0]
            # Estimate initial RPM needed for this DO (steady state approx)
            RPM_actual[0] = DO_actual[0] / max(1e-6, K_do) if K_do != 0 else rpm_min
            RPM_actual[0] = np.clip(RPM_actual[0], rpm_min, rpm_max)
            RPM_meas[0] = RPM_actual[0]
            RPM_sp_vec[0] = RPM_actual[0]
            # Estimate initial Power needed for this RPM (steady state approx)
            Power[0] = RPM_actual[0] / max(1e-6, K_motor) if K_motor != 0 else power_min
            Power[0] = np.clip(Power[0], power_min, power_max)

            # Reset PIDs with initial measurements to avoid derivative kick
            pid_DO.reset(initial_measurement=DO_meas[0])
            pid_RPM.reset(initial_measurement=RPM_meas[0])

            # --- Setpoint and Disturbance Profiles ---
            DO_sp_vec[:] = sp_initial_do
            DO_sp_vec[t >= t_step_do] = sp_final_do
            Power_dist_vec[t >= t_dist_start] = dist_mag

            # --- Main Simulation Loop ---
            progress_bar = st.progress(0)
            status_text = st.empty()

            for k in range(1, N):
                # 1. Update Setpoints
                pid_DO.update_setpoint(DO_sp_vec[k])

                # 2. Outer Loop (DO) Calculation -> RPM Setpoint
                RPM_sp_vec[k] = pid_DO.compute(DO_meas[k-1])

                # 3. Inner Loop (RPM) Calculation -> Power Output
                pid_RPM.update_setpoint(RPM_sp_vec[k])
                Power[k] = pid_RPM.compute(RPM_meas[k-1])

                # 4. Process Simulation
                # Apply disturbance to power
                power_with_disturbance = Power[k] + Power_dist_vec[k]
                power_with_disturbance = np.clip(power_with_disturbance, power_min, power_max)

                # 4a. Motor (Inner Process)
                RPM_actual[k] = alpha_motor * RPM_actual[k-1] + \
                                (1 - alpha_motor) * K_motor * power_with_disturbance

                # 4b. DO Process (Outer Process)
                DO_actual[k] = alpha_do * DO_actual[k-1] + \
                               (1 - alpha_do) * K_do * RPM_actual[k-1] # Depends on previous RPM

                # 5. Sensor Simulation (with optional noise)
                # Adding small noise example: np.random.randn() * noise_std_dev
                RPM_meas[k] = alpha_sensor_rpm * RPM_meas[k-1] + \
                              (1 - alpha_sensor_rpm) * RPM_actual[k] + np.random.randn() * 0.5 # Example noise

                DO_meas[k] = alpha_sensor_do * DO_meas[k-1] + \
                             (1 - alpha_sensor_do) * DO_actual[k] + np.random.randn() * 0.1 # Example noise
                
                # Update progress
                progress = (k + 1) / N
                progress_bar.progress(progress)
                status_text.text(f"Simulation in progress: {progress * 100:.1f}%")
            
            status_text.text("Simulation completed.")


            # --- Plot Results ---
            st.subheader("Simulation Results")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

            # Plot 1: Dissolved Oxygen (Outer Loop)
            axes[0].plot(t, DO_actual, 'b-', label='DO Actual')
            axes[0].plot(t, DO_meas, 'm--', label='DO Measured', alpha=0.7)
            axes[0].plot(t, DO_sp_vec, 'r:', label='DO Setpoint', linewidth=2)
            axes[0].set_title('Cascade Control: Dissolved Oxygen (Primary Loop)', fontsize=14)
            axes[0].set_ylabel('% Saturation')
            axes[0].legend()
            axes[0].grid(True)

            # Plot 2: Agitation (Inner Loop)
            axes[1].plot(t, RPM_actual, 'b-', label='RPM Actual')
            axes[1].plot(t, RPM_meas, 'm--', label='RPM Measured', alpha=0.7)
            axes[1].plot(t, RPM_sp_vec, 'g:', label='RPM Setpoint (from DO)', linewidth=2)
            axes[1].set_title('Agitation (Secondary Loop)', fontsize=14)
            axes[1].set_ylabel('RPM')
            axes[1].legend()
            axes[1].grid(True)

            # Plot 3: Motor Power (Final Output)
            axes[2].plot(t, Power, 'k-', label='PID Power Output')
            axes[2].plot(t, Power_dist_vec, 'r--', label='Disturbance')
            axes[2].set_title('Motor Power (Final Control Output)', fontsize=14)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Power (%)')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            # Optional: Display data table
            df_results_cas = pd.DataFrame({
                'Time (s)': t,
                'DO Setpoint (%)': DO_sp_vec,
                'DO Actual (%)': DO_actual,
                'DO Measured (%)': DO_meas,
                'RPM Setpoint': RPM_sp_vec,
                'RPM Actual': RPM_actual,
                'RPM Measured': RPM_meas,
                'Power Cmd (%)': Power,
                'Power Dist (%)': Power_dist_vec
            })
            with st.expander("View simulation data"):
                 st.dataframe(df_results_cas.style.format({
                    'Time (s)': '{:.1f}', 'DO Setpoint (%)': '{:.1f}', 'DO Actual (%)': '{:.2f}',
                    'DO Measured (%)': '{:.2f}', 'RPM Setpoint': '{:.1f}', 'RPM Actual': '{:.1f}',
                    'RPM Measured': '{:.1f}', 'Power Cmd (%)': '{:.1f}', 'Power Dist (%)': '{:.1f}'
                 }))

        except Exception as e:
            st.error(f"An error occurred during the cascade simulation:")
            st.exception(e) # Show traceback

    else:
        st.info("Set parameters in the sidebar and click 'Run Cascade Simulation'.")


# --- Entry Point ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="DO Cascade Control")
    regulatorio_cascade_oxigen_page()