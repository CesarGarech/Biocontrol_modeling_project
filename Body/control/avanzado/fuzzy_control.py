"""
Fuzzy Logic Control System for Bioprocess Control

This module implements fuzzy logic controllers for pH, temperature, and substrate feeding
in bioprocesses based on substrate composition. The implementation follows classical fuzzy
control approaches using Mamdani inference.

The fuzzy control system consists of:
1. Fuzzification: Convert crisp inputs into fuzzy sets using membership functions
2. Rule Base: Define linguistic control rules based on expert knowledge
3. Inference Engine: Apply fuzzy rules using Mamdani min-max composition
4. Defuzzification: Convert fuzzy output back to crisp control action

References:
----------
- Zadeh, L. A. (1965). "Fuzzy sets." Information and Control, 8(3), 338-353.
- Mamdani, E. H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a 
  fuzzy logic controller." International Journal of Man-Machine Studies, 7(1), 1-13.
- Passino, K. M., & Yurkovich, S. (1998). "Fuzzy Control." Addison Wesley Longman.
- Wang, L. X. (1997). "A Course in Fuzzy Systems and Control." Prentice Hall PTR.
- Lee, C. C. (1990). "Fuzzy logic in control systems: fuzzy logic controller-Parts I and II."
  IEEE Transactions on Systems, Man, and Cybernetics, 20(2), 404-435.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def fuzzy_control_page():
    """
    Main Streamlit page for fuzzy control system simulation.
    """
    st.header("🔮 Fuzzy Logic Control System for Bioprocess")
    
    st.markdown("""
    ### Overview
    This page implements a **Fuzzy Logic Control System** for simultaneous control of:
    - **pH:** Controlled by acid/base addition based on substrate composition
    - **Temperature:** Controlled by heating/cooling based on substrate composition
    - **Substrate Feeding Rate:** Controlled based on current substrate concentration
    
    The fuzzy controller uses linguistic rules derived from bioprocess engineering expertise
    to maintain optimal conditions for microbial growth and product formation.
    """)
    
    # Theoretical background
    with st.expander("📚 Theoretical Background: Fuzzy Logic Control"):
        st.markdown(r"""
        #### Introduction to Fuzzy Logic Control
        
        Fuzzy logic control was introduced by **Lotfi A. Zadeh (1965)** as a way to handle 
        imprecision and uncertainty in control systems. Unlike conventional control methods 
        that require precise mathematical models, fuzzy control uses linguistic rules based 
        on expert knowledge.
        
        **Key Advantages for Bioprocess Control:**
        - Handles nonlinear and time-varying dynamics without detailed mathematical models
        - Incorporates expert knowledge through linguistic rules
        - Robust to measurement noise and process uncertainties
        - Effective for multivariable control with complex interactions
        
        #### Fuzzy Control System Architecture
        
        A fuzzy logic controller consists of four main components:
        
        1. **Fuzzification:**
           - Converts crisp input values to fuzzy membership degrees
           - Uses membership functions (triangular, trapezoidal, Gaussian, etc.)
           - Example: pH = 6.5 → "slightly acidic" (μ = 0.7), "neutral" (μ = 0.3)
        
        2. **Rule Base:**
           - Collection of IF-THEN rules expressing control strategy
           - Based on expert knowledge and process understanding
           - Example: "IF substrate is low AND pH is acidic THEN increase feed rate AND add base"
        
        3. **Inference Engine:**
           - Evaluates fuzzy rules and combines their outputs
           - Common methods: Mamdani (min-max), Sugeno (weighted average)
           - Aggregates all active rules to determine control action
        
        4. **Defuzzification:**
           - Converts fuzzy output to crisp control signal
           - Common methods: Centroid, Bisector, Mean of Maximum (MOM)
           - Produces actionable control value for actuators
        
        #### Membership Functions
        
        Membership functions define the degree of membership (0 to 1) of a value in a fuzzy set:
        
        **Common Types:**
        - **Triangular:** Simple, computationally efficient, peak at center
        - **Trapezoidal:** Flat top region for stable membership in range
        - **Gaussian:** Smooth, differentiable, natural linguistic interpretation
        - **Sigmoidal:** Asymmetric, useful for boundary representations
        
        For this implementation, we use **triangular and trapezoidal** functions for their
        simplicity and computational efficiency.
        
        #### Mamdani Inference
        
        The **Mamdani fuzzy inference system** (Mamdani & Assilian, 1975) is the most common
        approach for fuzzy control:
        
        1. **AND Operation:** Minimum (min) of membership degrees
        2. **OR Operation:** Maximum (max) of membership degrees
        3. **Implication:** Minimum between rule strength and output membership
        4. **Aggregation:** Maximum of all rule outputs
        5. **Defuzzification:** Centroid method (center of gravity)
        
        $$u_{crisp} = \frac{\int \mu(u) \cdot u \, du}{\int \mu(u) \, du}$$
        
        #### Application to Bioprocess Control
        
        Bioprocesses are inherently nonlinear, time-varying, and subject to:
        - Variable substrate composition and quality
        - Metabolic shifts during different growth phases
        - Environmental disturbances (temperature, contamination)
        - Measurement delays and sensor noise
        
        Fuzzy control is particularly suitable because:
        - **pH Control:** Nonlinear titration curves, variable buffering capacity
        - **Temperature Control:** Heat generation varies with metabolic activity
        - **Substrate Feeding:** Optimal rate depends on growth phase and substrate quality
        
        #### Design Considerations
        
        **Number of Membership Functions:**
        - Too few: Coarse control, poor performance
        - Too many: Increased computational cost, rule explosion
        - Typical: 3-7 functions per variable (Low, Medium, High, etc.)
        
        **Rule Base Design:**
        - Based on process understanding and operator experience
        - Should cover all possible input combinations
        - Conflicting rules should be avoided
        - Can be tuned using experimental data or optimization
        
        **Tuning Parameters:**
        - Membership function shapes and positions
        - Scaling factors for inputs and outputs
        - Defuzzification method selection
        - Rule weights (if using weighted inference)
        
        #### References
        
        **Foundational Papers:**
        - Zadeh, L. A. (1965). "Fuzzy sets." *Information and Control*, 8(3), 338-353.
        - Mamdani, E. H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a 
          fuzzy logic controller." *International Journal of Man-Machine Studies*, 7(1), 1-13.
        
        **Textbooks:**
        - Passino, K. M., & Yurkovich, S. (1998). *Fuzzy Control*. Addison Wesley Longman.
        - Wang, L. X. (1997). *A Course in Fuzzy Systems and Control*. Prentice Hall PTR.
        - Ross, T. J. (2010). *Fuzzy Logic with Engineering Applications* (3rd ed.). John Wiley & Sons.
        
        **Review Papers:**
        - Lee, C. C. (1990). "Fuzzy logic in control systems: fuzzy logic controller-Parts I and II."
          *IEEE Transactions on Systems, Man, and Cybernetics*, 20(2), 404-435.
        - Kaur, A., & Kaur, A. (2012). "Comparison of Mamdani-Type and Sugeno-Type fuzzy inference 
          systems for air conditioning system." *International Journal of Soft Computing and 
          Engineering*, 2(2), 323-325.
        
        **Bioprocess Applications:**
        - Chen, L., et al. (1996). "Fuzzy logic based control of dissolved oxygen in a bioprocess."
          *Computers & Chemical Engineering*, 20, S1337-S1342.
        - Von Stosch, M., et al. (2014). "Hybrid semi-parametric modeling in process systems 
          engineering: Past, present and future." *Computers & Chemical Engineering*, 60, 86-101.
        - Negiz, A., & Cinar, A. (1998). "Monitoring of multivariable dynamic processes and sensor 
          auditing." *Journal of Process Control*, 8(5-6), 375-380.
        """)
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Fuzzy Control Configuration")
        
        with st.expander("🔬 Process Parameters", expanded=True):
            st.subheader("Initial Conditions")
            pH_initial = st.number_input("Initial pH", min_value=3.0, max_value=11.0, 
                                        value=7.0, step=0.1, key="pH_init")
            temp_initial = st.number_input("Initial Temperature [°C]", min_value=20.0, 
                                          max_value=45.0, value=30.0, step=0.5, key="temp_init")
            substrate_initial = st.number_input("Initial Substrate Conc. [g/L]", 
                                               min_value=0.0, max_value=50.0, 
                                               value=10.0, step=0.5, key="sub_init")
            
            st.subheader("Setpoints")
            pH_setpoint = st.number_input("pH Setpoint", min_value=3.0, max_value=11.0, 
                                         value=6.5, step=0.1, key="pH_sp")
            temp_setpoint = st.number_input("Temperature Setpoint [°C]", 
                                           min_value=20.0, max_value=45.0, 
                                           value=35.0, step=0.5, key="temp_sp")
            substrate_setpoint = st.number_input("Substrate Target [g/L]", 
                                                min_value=0.0, max_value=50.0, 
                                                value=15.0, step=0.5, key="sub_sp")
        
        with st.expander("⏱️ Simulation Settings", expanded=True):
            sim_time = st.number_input("Simulation Time [hours]", min_value=1.0, 
                                      max_value=50.0, value=20.0, step=1.0, key="sim_time_fuzzy")
            dt = st.number_input("Sampling Time [hours]", min_value=0.01, 
                                max_value=1.0, value=0.1, step=0.01, key="dt_fuzzy")
        
        with st.expander("🔧 Disturbances (Optional)", expanded=False):
            enable_disturbances = st.checkbox("Enable disturbances", value=True, 
                                             key="enable_dist")
            if enable_disturbances:
                dist_time = st.number_input("Disturbance Time [hours]", min_value=0.0, 
                                           max_value=sim_time, value=10.0, key="dist_time")
                pH_dist = st.number_input("pH Disturbance", min_value=-2.0, 
                                         max_value=2.0, value=0.5, key="pH_dist")
                temp_dist = st.number_input("Temperature Disturbance [°C]", 
                                           min_value=-10.0, max_value=10.0, 
                                           value=5.0, key="temp_dist")
                substrate_dist = st.number_input("Substrate Disturbance [g/L]", 
                                                min_value=-10.0, max_value=10.0, 
                                                value=-3.0, key="sub_dist")
    
    # Main simulation button
    if st.button("🚀 Run Fuzzy Control Simulation", key="run_fuzzy_sim"):
        with st.spinner("Building fuzzy control systems..."):
            # Create fuzzy controllers
            ph_controller = create_ph_fuzzy_controller()
            temp_controller = create_temperature_fuzzy_controller()
            feed_controller = create_substrate_fuzzy_controller()
        
        with st.spinner("Running simulation..."):
            # Run simulation
            time_array = np.arange(0, sim_time, dt)
            n_steps = len(time_array)
            
            # Initialize arrays
            pH_array = np.zeros(n_steps)
            temp_array = np.zeros(n_steps)
            substrate_array = np.zeros(n_steps)
            
            acid_base_array = np.zeros(n_steps)
            heat_cool_array = np.zeros(n_steps)
            feed_rate_array = np.zeros(n_steps)
            
            # Initial conditions
            pH_array[0] = pH_initial
            temp_array[0] = temp_initial
            substrate_array[0] = substrate_initial
            
            # Simulation loop
            for i in range(1, n_steps):
                t = time_array[i]
                
                # Apply disturbances if enabled
                if enable_disturbances and t >= dist_time:
                    if i == np.argmin(np.abs(time_array - dist_time)):
                        pH_array[i-1] += pH_dist
                        temp_array[i-1] += temp_dist
                        substrate_array[i-1] += substrate_dist
                
                # Calculate errors
                pH_error = pH_setpoint - pH_array[i-1]
                temp_error = temp_setpoint - temp_array[i-1]
                substrate_error = substrate_setpoint - substrate_array[i-1]
                
                # Fuzzy control computations
                ph_controller.input['pH_error'] = pH_error
                ph_controller.input['substrate'] = substrate_array[i-1]
                ph_controller.compute()
                acid_base = ph_controller.output['acid_base']
                
                temp_controller.input['temp_error'] = temp_error
                temp_controller.input['substrate'] = substrate_array[i-1]
                temp_controller.compute()
                heat_cool = temp_controller.output['heat_cool']
                
                feed_controller.input['substrate_error'] = substrate_error
                feed_controller.input['substrate_level'] = substrate_array[i-1]
                feed_controller.compute()
                feed_rate = feed_controller.output['feed_rate']
                
                # Store control actions
                acid_base_array[i] = acid_base
                heat_cool_array[i] = heat_cool
                feed_rate_array[i] = feed_rate
                
                # Simple process dynamics (first-order system with external input)
                # pH dynamics: First-order response to acid/base addition with natural drift
                tau_pH = 2.0  # time constant [hours]
                K_pH = 0.05   # gain [pH / control unit]
                pH_drift = 0.02  # natural pH drift rate towards neutral (pH 7)
                pH_natural = 7.0  # natural equilibrium pH
                dpH = (K_pH * acid_base - (pH_array[i-1] - pH_natural) * pH_drift) / tau_pH
                pH_array[i] = pH_array[i-1] + dpH * dt
                
                # Temperature dynamics: First-order heat transfer with ambient temperature
                tau_T = 1.5   # time constant [hours]
                K_T = 0.1     # gain [°C / control unit]
                T_ambient = 25.0  # ambient temperature [°C]
                heat_loss_coeff = 0.1  # heat loss coefficient
                # Metabolic heat generation (increases with substrate)
                metabolic_heat = 0.02 * substrate_array[i-1] / (5.0 + substrate_array[i-1])
                dT = (K_T * heat_cool + metabolic_heat - (temp_array[i-1] - T_ambient) * heat_loss_coeff) / tau_T
                temp_array[i] = temp_array[i-1] + dT * dt
                
                # Substrate dynamics with consumption
                tau_S = 3.0   # time constant [hours]
                K_S = 0.15    # gain [g/L / control unit]
                # Monod-like consumption rate
                consumption_rate = 0.5 * substrate_array[i-1] / (2.0 + substrate_array[i-1])
                dS = K_S * feed_rate - consumption_rate
                substrate_array[i] = substrate_array[i-1] + dS * dt
                
                # Keep physical limits
                pH_array[i] = np.clip(pH_array[i], 3.0, 11.0)
                temp_array[i] = np.clip(temp_array[i], 20.0, 45.0)
                substrate_array[i] = np.clip(substrate_array[i], 0.0, 50.0)
        
        # Display results
        st.success("✅ Simulation completed successfully!")
        st.markdown("---")
        
        # Create plots
        st.subheader("📊 Simulation Results")
        
        # Plot 1: Process Variables
        fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
        
        # pH
        axes1[0].plot(time_array, pH_array, 'b-', linewidth=2, label='pH')
        axes1[0].axhline(y=pH_setpoint, color='r', linestyle='--', 
                        linewidth=1.5, label='Setpoint')
        axes1[0].set_ylabel('pH', fontsize=12, fontweight='bold')
        axes1[0].set_title('pH Control', fontsize=14, fontweight='bold')
        axes1[0].grid(True, alpha=0.3)
        axes1[0].legend(loc='best')
        axes1[0].set_xlim([0, sim_time])
        
        # Temperature
        axes1[1].plot(time_array, temp_array, 'orange', linewidth=2, label='Temperature')
        axes1[1].axhline(y=temp_setpoint, color='r', linestyle='--', 
                        linewidth=1.5, label='Setpoint')
        axes1[1].set_ylabel('Temperature [°C]', fontsize=12, fontweight='bold')
        axes1[1].set_title('Temperature Control', fontsize=14, fontweight='bold')
        axes1[1].grid(True, alpha=0.3)
        axes1[1].legend(loc='best')
        axes1[1].set_xlim([0, sim_time])
        
        # Substrate
        axes1[2].plot(time_array, substrate_array, 'g-', linewidth=2, label='Substrate')
        axes1[2].axhline(y=substrate_setpoint, color='r', linestyle='--', 
                        linewidth=1.5, label='Target')
        axes1[2].set_ylabel('Substrate [g/L]', fontsize=12, fontweight='bold')
        axes1[2].set_xlabel('Time [hours]', fontsize=12, fontweight='bold')
        axes1[2].set_title('Substrate Concentration', fontsize=14, fontweight='bold')
        axes1[2].grid(True, alpha=0.3)
        axes1[2].legend(loc='best')
        axes1[2].set_xlim([0, sim_time])
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Plot 2: Control Actions
        st.subheader("🎮 Control Actions")
        fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
        
        # Acid/Base addition
        axes2[0].plot(time_array, acid_base_array, 'm-', linewidth=2)
        axes2[0].set_ylabel('Acid/Base [-1 to 1]', fontsize=12, fontweight='bold')
        axes2[0].set_title('pH Control Action (Negative=Acid, Positive=Base)', 
                          fontsize=14, fontweight='bold')
        axes2[0].grid(True, alpha=0.3)
        axes2[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes2[0].set_xlim([0, sim_time])
        
        # Heating/Cooling
        axes2[1].plot(time_array, heat_cool_array, 'r-', linewidth=2)
        axes2[1].set_ylabel('Heat/Cool [-1 to 1]', fontsize=12, fontweight='bold')
        axes2[1].set_title('Temperature Control Action (Negative=Cool, Positive=Heat)', 
                          fontsize=14, fontweight='bold')
        axes2[1].grid(True, alpha=0.3)
        axes2[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes2[1].set_xlim([0, sim_time])
        
        # Feed rate
        axes2[2].plot(time_array, feed_rate_array, 'c-', linewidth=2)
        axes2[2].set_ylabel('Feed Rate [0 to 1]', fontsize=12, fontweight='bold')
        axes2[2].set_xlabel('Time [hours]', fontsize=12, fontweight='bold')
        axes2[2].set_title('Substrate Feed Rate', fontsize=14, fontweight='bold')
        axes2[2].grid(True, alpha=0.3)
        axes2[2].set_xlim([0, sim_time])
        axes2[2].set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("pH IAE", f"{np.sum(np.abs(pH_setpoint - pH_array) * dt):.3f}")
            st.metric("pH ISE", f"{np.sum((pH_setpoint - pH_array)**2 * dt):.3f}")
            st.metric("pH Final Error", f"{pH_setpoint - pH_array[-1]:.3f}")
        
        with col2:
            st.metric("Temp IAE [°C·h]", f"{np.sum(np.abs(temp_setpoint - temp_array) * dt):.3f}")
            st.metric("Temp ISE [°C²·h]", f"{np.sum((temp_setpoint - temp_array)**2 * dt):.3f}")
            st.metric("Temp Final Error [°C]", f"{temp_setpoint - temp_array[-1]:.3f}")
        
        with col3:
            st.metric("Substrate IAE [g·h/L]", 
                     f"{np.sum(np.abs(substrate_setpoint - substrate_array) * dt):.3f}")
            st.metric("Substrate ISE [(g/L)²·h]", 
                     f"{np.sum((substrate_setpoint - substrate_array)**2 * dt):.3f}")
            st.metric("Substrate Final Error [g/L]", 
                     f"{substrate_setpoint - substrate_array[-1]:.3f}")
        
        st.info("""
        **Performance Metrics:**
        - **IAE (Integral Absolute Error):** Measures total absolute deviation from setpoint
        - **ISE (Integral Squared Error):** Penalizes large errors more heavily
        - **Final Error:** Steady-state tracking error
        """)
    
    else:
        st.info("👆 Configure parameters in the sidebar and click '🚀 Run Fuzzy Control Simulation' to start")
        
        # Display membership functions
        st.subheader("🔍 Fuzzy System Visualization")
        st.markdown("Preview of membership functions for each controller:")
        
        with st.expander("pH Controller Membership Functions"):
            fig_ph = visualize_ph_controller()
            st.pyplot(fig_ph)
        
        with st.expander("Temperature Controller Membership Functions"):
            fig_temp = visualize_temperature_controller()
            st.pyplot(fig_temp)
        
        with st.expander("Substrate Feed Controller Membership Functions"):
            fig_feed = visualize_substrate_controller()
            st.pyplot(fig_feed)


def create_ph_fuzzy_controller():
    """
    Create fuzzy logic controller for pH control based on substrate composition.
    
    Inputs:
        - pH_error: Difference between setpoint and measured pH
        - substrate: Current substrate concentration
    
    Output:
        - acid_base: Control action (-1 = add acid, +1 = add base)
    """
    # Input variables
    pH_error = ctrl.Antecedent(np.linspace(-3, 3, 100), 'pH_error')
    substrate = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate')
    
    # Output variable
    acid_base = ctrl.Consequent(np.linspace(-1, 1, 100), 'acid_base')
    
    # Membership functions for pH error
    pH_error['very_low'] = fuzz.trapmf(pH_error.universe, [-3, -3, -2, -1])
    pH_error['low'] = fuzz.trimf(pH_error.universe, [-2, -1, 0])
    pH_error['optimal'] = fuzz.trimf(pH_error.universe, [-0.5, 0, 0.5])
    pH_error['high'] = fuzz.trimf(pH_error.universe, [0, 1, 2])
    pH_error['very_high'] = fuzz.trapmf(pH_error.universe, [1, 2, 3, 3])
    
    # Membership functions for substrate concentration
    substrate['very_low'] = fuzz.trapmf(substrate.universe, [0, 0, 5, 10])
    substrate['low'] = fuzz.trimf(substrate.universe, [5, 10, 15])
    substrate['medium'] = fuzz.trimf(substrate.universe, [10, 20, 30])
    substrate['high'] = fuzz.trimf(substrate.universe, [25, 35, 45])
    substrate['very_high'] = fuzz.trapmf(substrate.universe, [40, 45, 50, 50])
    
    # Membership functions for control action
    acid_base['strong_acid'] = fuzz.trapmf(acid_base.universe, [-1, -1, -0.7, -0.4])
    acid_base['mild_acid'] = fuzz.trimf(acid_base.universe, [-0.6, -0.3, 0])
    acid_base['neutral'] = fuzz.trimf(acid_base.universe, [-0.2, 0, 0.2])
    acid_base['mild_base'] = fuzz.trimf(acid_base.universe, [0, 0.3, 0.6])
    acid_base['strong_base'] = fuzz.trapmf(acid_base.universe, [0.4, 0.7, 1, 1])
    
    # Fuzzy rules for pH control based on pH error and substrate concentration
    # High substrate requires more careful pH control due to buffering capacity
    rules = [
        # When pH is very low (too acidic) → Add base
        ctrl.Rule(pH_error['very_low'] & substrate['very_low'], acid_base['strong_base']),
        ctrl.Rule(pH_error['very_low'] & substrate['low'], acid_base['strong_base']),
        ctrl.Rule(pH_error['very_low'] & substrate['medium'], acid_base['mild_base']),
        ctrl.Rule(pH_error['very_low'] & substrate['high'], acid_base['mild_base']),
        ctrl.Rule(pH_error['very_low'] & substrate['very_high'], acid_base['neutral']),
        
        # When pH is low (slightly acidic) → Add mild base
        ctrl.Rule(pH_error['low'] & substrate['very_low'], acid_base['mild_base']),
        ctrl.Rule(pH_error['low'] & substrate['low'], acid_base['mild_base']),
        ctrl.Rule(pH_error['low'] & substrate['medium'], acid_base['mild_base']),
        ctrl.Rule(pH_error['low'] & substrate['high'], acid_base['neutral']),
        ctrl.Rule(pH_error['low'] & substrate['very_high'], acid_base['neutral']),
        
        # When pH is optimal → No action needed
        ctrl.Rule(pH_error['optimal'], acid_base['neutral']),
        
        # When pH is high (slightly basic) → Add mild acid
        ctrl.Rule(pH_error['high'] & substrate['very_low'], acid_base['mild_acid']),
        ctrl.Rule(pH_error['high'] & substrate['low'], acid_base['mild_acid']),
        ctrl.Rule(pH_error['high'] & substrate['medium'], acid_base['mild_acid']),
        ctrl.Rule(pH_error['high'] & substrate['high'], acid_base['neutral']),
        ctrl.Rule(pH_error['high'] & substrate['very_high'], acid_base['neutral']),
        
        # When pH is very high (too basic) → Add acid
        ctrl.Rule(pH_error['very_high'] & substrate['very_low'], acid_base['strong_acid']),
        ctrl.Rule(pH_error['very_high'] & substrate['low'], acid_base['strong_acid']),
        ctrl.Rule(pH_error['very_high'] & substrate['medium'], acid_base['mild_acid']),
        ctrl.Rule(pH_error['very_high'] & substrate['high'], acid_base['mild_acid']),
        ctrl.Rule(pH_error['very_high'] & substrate['very_high'], acid_base['neutral']),
    ]
    
    # Create control system
    pH_ctrl = ctrl.ControlSystem(rules)
    pH_simulation = ctrl.ControlSystemSimulation(pH_ctrl)
    
    return pH_simulation


def create_temperature_fuzzy_controller():
    """
    Create fuzzy logic controller for temperature control based on substrate composition.
    
    Inputs:
        - temp_error: Difference between setpoint and measured temperature
        - substrate: Current substrate concentration
    
    Output:
        - heat_cool: Control action (-1 = cooling, +1 = heating)
    """
    # Input variables
    temp_error = ctrl.Antecedent(np.linspace(-15, 15, 100), 'temp_error')
    substrate = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate')
    
    # Output variable
    heat_cool = ctrl.Consequent(np.linspace(-1, 1, 100), 'heat_cool')
    
    # Membership functions for temperature error
    temp_error['very_cold'] = fuzz.trapmf(temp_error.universe, [-15, -15, -10, -5])
    temp_error['cold'] = fuzz.trimf(temp_error.universe, [-10, -5, 0])
    temp_error['optimal'] = fuzz.trimf(temp_error.universe, [-3, 0, 3])
    temp_error['hot'] = fuzz.trimf(temp_error.universe, [0, 5, 10])
    temp_error['very_hot'] = fuzz.trapmf(temp_error.universe, [5, 10, 15, 15])
    
    # Membership functions for substrate (affects metabolic heat generation)
    substrate['very_low'] = fuzz.trapmf(substrate.universe, [0, 0, 5, 10])
    substrate['low'] = fuzz.trimf(substrate.universe, [5, 10, 15])
    substrate['medium'] = fuzz.trimf(substrate.universe, [10, 20, 30])
    substrate['high'] = fuzz.trimf(substrate.universe, [25, 35, 45])
    substrate['very_high'] = fuzz.trapmf(substrate.universe, [40, 45, 50, 50])
    
    # Membership functions for control action
    heat_cool['strong_cooling'] = fuzz.trapmf(heat_cool.universe, [-1, -1, -0.7, -0.4])
    heat_cool['mild_cooling'] = fuzz.trimf(heat_cool.universe, [-0.6, -0.3, 0])
    heat_cool['neutral'] = fuzz.trimf(heat_cool.universe, [-0.2, 0, 0.2])
    heat_cool['mild_heating'] = fuzz.trimf(heat_cool.universe, [0, 0.3, 0.6])
    heat_cool['strong_heating'] = fuzz.trapmf(heat_cool.universe, [0.4, 0.7, 1, 1])
    
    # Fuzzy rules for temperature control
    # Higher substrate → more metabolic heat → may need more cooling
    rules = [
        # When temperature is very cold → Strong heating
        ctrl.Rule(temp_error['very_cold'] & substrate['very_low'], heat_cool['strong_heating']),
        ctrl.Rule(temp_error['very_cold'] & substrate['low'], heat_cool['strong_heating']),
        ctrl.Rule(temp_error['very_cold'] & substrate['medium'], heat_cool['mild_heating']),
        ctrl.Rule(temp_error['very_cold'] & substrate['high'], heat_cool['mild_heating']),
        ctrl.Rule(temp_error['very_cold'] & substrate['very_high'], heat_cool['neutral']),
        
        # When temperature is cold → Mild heating
        ctrl.Rule(temp_error['cold'] & substrate['very_low'], heat_cool['mild_heating']),
        ctrl.Rule(temp_error['cold'] & substrate['low'], heat_cool['mild_heating']),
        ctrl.Rule(temp_error['cold'] & substrate['medium'], heat_cool['mild_heating']),
        ctrl.Rule(temp_error['cold'] & substrate['high'], heat_cool['neutral']),
        ctrl.Rule(temp_error['cold'] & substrate['very_high'], heat_cool['neutral']),
        
        # When temperature is optimal → No action
        ctrl.Rule(temp_error['optimal'], heat_cool['neutral']),
        
        # When temperature is hot → Mild cooling
        ctrl.Rule(temp_error['hot'] & substrate['very_low'], heat_cool['mild_cooling']),
        ctrl.Rule(temp_error['hot'] & substrate['low'], heat_cool['mild_cooling']),
        ctrl.Rule(temp_error['hot'] & substrate['medium'], heat_cool['mild_cooling']),
        ctrl.Rule(temp_error['hot'] & substrate['high'], heat_cool['strong_cooling']),
        ctrl.Rule(temp_error['hot'] & substrate['very_high'], heat_cool['strong_cooling']),
        
        # When temperature is very hot → Strong cooling (especially with high substrate)
        ctrl.Rule(temp_error['very_hot'] & substrate['very_low'], heat_cool['mild_cooling']),
        ctrl.Rule(temp_error['very_hot'] & substrate['low'], heat_cool['strong_cooling']),
        ctrl.Rule(temp_error['very_hot'] & substrate['medium'], heat_cool['strong_cooling']),
        ctrl.Rule(temp_error['very_hot'] & substrate['high'], heat_cool['strong_cooling']),
        ctrl.Rule(temp_error['very_hot'] & substrate['very_high'], heat_cool['strong_cooling']),
    ]
    
    # Create control system
    temp_ctrl = ctrl.ControlSystem(rules)
    temp_simulation = ctrl.ControlSystemSimulation(temp_ctrl)
    
    return temp_simulation


def create_substrate_fuzzy_controller():
    """
    Create fuzzy logic controller for substrate feeding based on substrate level and error.
    
    Inputs:
        - substrate_error: Difference between target and current substrate
        - substrate_level: Current substrate concentration
    
    Output:
        - feed_rate: Substrate feed rate (0 to 1, normalized)
    """
    # Input variables
    substrate_error = ctrl.Antecedent(np.linspace(-30, 30, 100), 'substrate_error')
    substrate_level = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate_level')
    
    # Output variable
    feed_rate = ctrl.Consequent(np.linspace(0, 1, 100), 'feed_rate')
    
    # Membership functions for substrate error
    substrate_error['very_negative'] = fuzz.trapmf(substrate_error.universe, [-30, -30, -20, -10])
    substrate_error['negative'] = fuzz.trimf(substrate_error.universe, [-20, -10, 0])
    substrate_error['zero'] = fuzz.trimf(substrate_error.universe, [-5, 0, 5])
    substrate_error['positive'] = fuzz.trimf(substrate_error.universe, [0, 10, 20])
    substrate_error['very_positive'] = fuzz.trapmf(substrate_error.universe, [10, 20, 30, 30])
    
    # Membership functions for substrate level
    substrate_level['very_low'] = fuzz.trapmf(substrate_level.universe, [0, 0, 5, 10])
    substrate_level['low'] = fuzz.trimf(substrate_level.universe, [5, 10, 20])
    substrate_level['medium'] = fuzz.trimf(substrate_level.universe, [15, 25, 35])
    substrate_level['high'] = fuzz.trimf(substrate_level.universe, [30, 40, 50])
    substrate_level['very_high'] = fuzz.trapmf(substrate_level.universe, [45, 50, 50, 50])
    
    # Membership functions for feed rate
    feed_rate['none'] = fuzz.trimf(feed_rate.universe, [0, 0, 0.1])
    feed_rate['very_low'] = fuzz.trimf(feed_rate.universe, [0, 0.15, 0.3])
    feed_rate['low'] = fuzz.trimf(feed_rate.universe, [0.2, 0.35, 0.5])
    feed_rate['medium'] = fuzz.trimf(feed_rate.universe, [0.4, 0.55, 0.7])
    feed_rate['high'] = fuzz.trimf(feed_rate.universe, [0.6, 0.75, 0.9])
    feed_rate['very_high'] = fuzz.trimf(feed_rate.universe, [0.8, 0.95, 1.0])
    
    # Fuzzy rules for substrate feeding
    # Positive error → need more substrate → increase feed
    # Negative error → too much substrate → decrease feed
    # Current level also affects decision (avoid overfeeding)
    rules = [
        # Very positive error (need much more substrate)
        ctrl.Rule(substrate_error['very_positive'] & substrate_level['very_low'], feed_rate['very_high']),
        ctrl.Rule(substrate_error['very_positive'] & substrate_level['low'], feed_rate['high']),
        ctrl.Rule(substrate_error['very_positive'] & substrate_level['medium'], feed_rate['medium']),
        ctrl.Rule(substrate_error['very_positive'] & substrate_level['high'], feed_rate['low']),
        ctrl.Rule(substrate_error['very_positive'] & substrate_level['very_high'], feed_rate['none']),
        
        # Positive error (need more substrate)
        ctrl.Rule(substrate_error['positive'] & substrate_level['very_low'], feed_rate['high']),
        ctrl.Rule(substrate_error['positive'] & substrate_level['low'], feed_rate['medium']),
        ctrl.Rule(substrate_error['positive'] & substrate_level['medium'], feed_rate['low']),
        ctrl.Rule(substrate_error['positive'] & substrate_level['high'], feed_rate['very_low']),
        ctrl.Rule(substrate_error['positive'] & substrate_level['very_high'], feed_rate['none']),
        
        # Zero error (at target)
        ctrl.Rule(substrate_error['zero'] & substrate_level['very_low'], feed_rate['low']),
        ctrl.Rule(substrate_error['zero'] & substrate_level['low'], feed_rate['very_low']),
        ctrl.Rule(substrate_error['zero'] & substrate_level['medium'], feed_rate['very_low']),
        ctrl.Rule(substrate_error['zero'] & substrate_level['high'], feed_rate['none']),
        ctrl.Rule(substrate_error['zero'] & substrate_level['very_high'], feed_rate['none']),
        
        # Negative error (too much substrate)
        ctrl.Rule(substrate_error['negative'], feed_rate['none']),
        
        # Very negative error (way too much substrate)
        ctrl.Rule(substrate_error['very_negative'], feed_rate['none']),
    ]
    
    # Create control system
    feed_ctrl = ctrl.ControlSystem(rules)
    feed_simulation = ctrl.ControlSystemSimulation(feed_ctrl)
    
    return feed_simulation


def visualize_ph_controller():
    """Visualize pH controller membership functions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # pH error
    pH_error = np.linspace(-3, 3, 100)
    axes[0].plot(pH_error, fuzz.trapmf(pH_error, [-3, -3, -2, -1]), 'b', linewidth=2, label='Very Low')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [-2, -1, 0]), 'g', linewidth=2, label='Low')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [-0.5, 0, 0.5]), 'orange', linewidth=2, label='Optimal')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [0, 1, 2]), 'r', linewidth=2, label='High')
    axes[0].plot(pH_error, fuzz.trapmf(pH_error, [1, 2, 3, 3]), 'm', linewidth=2, label='Very High')
    axes[0].set_title('pH Error Membership Functions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('pH Error')
    axes[0].set_ylabel('Membership Degree')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Substrate
    substrate = np.linspace(0, 50, 100)
    axes[1].plot(substrate, fuzz.trapmf(substrate, [0, 0, 5, 10]), 'b', linewidth=2, label='Very Low')
    axes[1].plot(substrate, fuzz.trimf(substrate, [5, 10, 15]), 'g', linewidth=2, label='Low')
    axes[1].plot(substrate, fuzz.trimf(substrate, [10, 20, 30]), 'orange', linewidth=2, label='Medium')
    axes[1].plot(substrate, fuzz.trimf(substrate, [25, 35, 45]), 'r', linewidth=2, label='High')
    axes[1].plot(substrate, fuzz.trapmf(substrate, [40, 45, 50, 50]), 'm', linewidth=2, label='Very High')
    axes[1].set_title('Substrate Concentration Membership Functions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Substrate [g/L]')
    axes[1].set_ylabel('Membership Degree')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Control action
    acid_base = np.linspace(-1, 1, 100)
    axes[2].plot(acid_base, fuzz.trapmf(acid_base, [-1, -1, -0.7, -0.4]), 'b', linewidth=2, label='Strong Acid')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [-0.6, -0.3, 0]), 'c', linewidth=2, label='Mild Acid')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [-0.2, 0, 0.2]), 'orange', linewidth=2, label='Neutral')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [0, 0.3, 0.6]), 'g', linewidth=2, label='Mild Base')
    axes[2].plot(acid_base, fuzz.trapmf(acid_base, [0.4, 0.7, 1, 1]), 'm', linewidth=2, label='Strong Base')
    axes[2].set_title('pH Control Action Membership Functions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Acid/Base Action')
    axes[2].set_ylabel('Membership Degree')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_temperature_controller():
    """Visualize temperature controller membership functions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperature error
    temp_error = np.linspace(-15, 15, 100)
    axes[0].plot(temp_error, fuzz.trapmf(temp_error, [-15, -15, -10, -5]), 'b', linewidth=2, label='Very Cold')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [-10, -5, 0]), 'c', linewidth=2, label='Cold')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [-3, 0, 3]), 'g', linewidth=2, label='Optimal')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [0, 5, 10]), 'orange', linewidth=2, label='Hot')
    axes[0].plot(temp_error, fuzz.trapmf(temp_error, [5, 10, 15, 15]), 'r', linewidth=2, label='Very Hot')
    axes[0].set_title('Temperature Error Membership Functions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Temperature Error [°C]')
    axes[0].set_ylabel('Membership Degree')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Substrate
    substrate = np.linspace(0, 50, 100)
    axes[1].plot(substrate, fuzz.trapmf(substrate, [0, 0, 5, 10]), 'b', linewidth=2, label='Very Low')
    axes[1].plot(substrate, fuzz.trimf(substrate, [5, 10, 15]), 'g', linewidth=2, label='Low')
    axes[1].plot(substrate, fuzz.trimf(substrate, [10, 20, 30]), 'orange', linewidth=2, label='Medium')
    axes[1].plot(substrate, fuzz.trimf(substrate, [25, 35, 45]), 'r', linewidth=2, label='High')
    axes[1].plot(substrate, fuzz.trapmf(substrate, [40, 45, 50, 50]), 'm', linewidth=2, label='Very High')
    axes[1].set_title('Substrate Concentration Membership Functions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Substrate [g/L]')
    axes[1].set_ylabel('Membership Degree')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Control action
    heat_cool = np.linspace(-1, 1, 100)
    axes[2].plot(heat_cool, fuzz.trapmf(heat_cool, [-1, -1, -0.7, -0.4]), 'b', linewidth=2, label='Strong Cooling')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [-0.6, -0.3, 0]), 'c', linewidth=2, label='Mild Cooling')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [-0.2, 0, 0.2]), 'g', linewidth=2, label='Neutral')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [0, 0.3, 0.6]), 'orange', linewidth=2, label='Mild Heating')
    axes[2].plot(heat_cool, fuzz.trapmf(heat_cool, [0.4, 0.7, 1, 1]), 'r', linewidth=2, label='Strong Heating')
    axes[2].set_title('Temperature Control Action Membership Functions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Heat/Cool Action')
    axes[2].set_ylabel('Membership Degree')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_substrate_controller():
    """Visualize substrate feed controller membership functions."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Substrate error
    substrate_error = np.linspace(-30, 30, 100)
    axes[0].plot(substrate_error, fuzz.trapmf(substrate_error, [-30, -30, -20, -10]), 'b', linewidth=2, label='Very Negative')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [-20, -10, 0]), 'c', linewidth=2, label='Negative')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [-5, 0, 5]), 'g', linewidth=2, label='Zero')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [0, 10, 20]), 'orange', linewidth=2, label='Positive')
    axes[0].plot(substrate_error, fuzz.trapmf(substrate_error, [10, 20, 30, 30]), 'r', linewidth=2, label='Very Positive')
    axes[0].set_title('Substrate Error Membership Functions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Substrate Error [g/L]')
    axes[0].set_ylabel('Membership Degree')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Substrate level
    substrate_level = np.linspace(0, 50, 100)
    axes[1].plot(substrate_level, fuzz.trapmf(substrate_level, [0, 0, 5, 10]), 'b', linewidth=2, label='Very Low')
    axes[1].plot(substrate_level, fuzz.trimf(substrate_level, [5, 10, 20]), 'c', linewidth=2, label='Low')
    axes[1].plot(substrate_level, fuzz.trimf(substrate_level, [15, 25, 35]), 'g', linewidth=2, label='Medium')
    axes[1].plot(substrate_level, fuzz.trimf(substrate_level, [30, 40, 50]), 'orange', linewidth=2, label='High')
    axes[1].plot(substrate_level, fuzz.trapmf(substrate_level, [45, 50, 50, 50]), 'r', linewidth=2, label='Very High')
    axes[1].set_title('Substrate Level Membership Functions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Substrate Level [g/L]')
    axes[1].set_ylabel('Membership Degree')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Feed rate
    feed_rate = np.linspace(0, 1, 100)
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0, 0, 0.1]), 'b', linewidth=2, label='None')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0, 0.15, 0.3]), 'c', linewidth=2, label='Very Low')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0.2, 0.35, 0.5]), 'g', linewidth=2, label='Low')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0.4, 0.55, 0.7]), 'y', linewidth=2, label='Medium')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0.6, 0.75, 0.9]), 'orange', linewidth=2, label='High')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0.8, 0.95, 1.0]), 'r', linewidth=2, label='Very High')
    axes[2].set_title('Feed Rate Membership Functions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Feed Rate [normalized]')
    axes[2].set_ylabel('Membership Degree')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# For standalone testing
if __name__ == "__main__":
    fuzzy_control_page()
