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


def safe_trimf_points(a, b, c, max_val=None):
    """
    Ensure triangular membership function points are valid.
    Returns (a, b, c) ensuring a <= b <= c for valid trimf.
    
    Parameters:
    -----------
    a, b, c : float
        Triangle points (left, peak, right)
    max_val : float, optional
        Maximum value to clamp points to
        
    Returns:
    --------
    tuple or list : (a, b, c) with a <= b <= c
    """
    if max_val is not None:
        points = sorted([min(a, max_val), min(b, max_val), min(c, max_val)])
        return points
    else:
        a, b, c = sorted([a, b, c])
        # Ensure b is the middle point
        if a == c:  # Degenerate case
            c = a + 0.01
        return (a, b, c)


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
        
        # Fuzzy Controller Tuning
        with st.expander("🎯 Fuzzy Controller Tuning", expanded=False):
            st.markdown("### Membership Function Configuration")
            st.info("Adjust membership function parameters to tune controller behavior. "
                   "These parameters define the linguistic terms (Low, Medium, High, etc.) "
                   "used in the fuzzy rules.")
            
            # pH Controller MF Parameters
            st.markdown("#### pH Controller")
            col1, col2 = st.columns(2)
            with col1:
                pH_error_range = st.slider("pH Error Range", -5.0, 5.0, (-3.0, 3.0), 0.5, 
                                          key="pH_err_range", 
                                          help="Adjust the input range for pH error")
                pH_error_optimal_width = st.slider("pH Optimal Zone Width", 0.1, 2.0, 0.3, 0.1,
                                                   key="pH_opt_width",
                                                   help="Width of the optimal pH range (±)")
            with col2:
                pH_output_range = st.slider("pH Control Action Range", 0.1, 1.0, 1.0, 0.1,
                                           key="pH_out_range",
                                           help="Maximum control action magnitude")
                pH_control_aggressive = st.slider("pH Control Aggressiveness", 0.5, 2.0, 1.3, 0.1,
                                                 key="pH_aggr",
                                                 help="Multiplier for control action strength")
            
            # Temperature Controller MF Parameters
            st.markdown("#### Temperature Controller")
            col1, col2 = st.columns(2)
            with col1:
                temp_error_range = st.slider("Temp Error Range [°C]", -20.0, 20.0, (-15.0, 15.0), 1.0,
                                            key="temp_err_range",
                                            help="Adjust the input range for temperature error")
                temp_error_optimal_width = st.slider("Temp Optimal Zone Width [°C]", 0.5, 5.0, 2.0, 0.5,
                                                     key="temp_opt_width",
                                                     help="Width of the optimal temperature range (±)")
            with col2:
                temp_output_range = st.slider("Temp Control Action Range", 0.1, 1.0, 1.0, 0.1,
                                             key="temp_out_range",
                                             help="Maximum control action magnitude")
                temp_control_aggressive = st.slider("Temp Control Aggressiveness", 0.5, 2.0, 1.3, 0.1,
                                                   key="temp_aggr",
                                                   help="Multiplier for control action strength")
            
            # Substrate Feed Controller MF Parameters
            st.markdown("#### Substrate Feed Controller")
            col1, col2 = st.columns(2)
            with col1:
                sub_error_range = st.slider("Substrate Error Range [g/L]", -40.0, 40.0, (-30.0, 30.0), 5.0,
                                           key="sub_err_range",
                                           help="Adjust the input range for substrate error")
                sub_error_zero_width = st.slider("Substrate Zero Error Width [g/L]", 1.0, 10.0, 3.0, 1.0,
                                                key="sub_zero_width",
                                                help="Width of the zero error zone (±)")
            with col2:
                feed_rate_max = st.slider("Maximum Feed Rate", 0.5, 1.0, 1.0, 0.05,
                                         key="feed_max",
                                         help="Maximum normalized feed rate")
                feed_sensitivity = st.slider("Feed Rate Sensitivity", 0.5, 2.0, 1.2, 0.1,
                                            key="feed_sens",
                                            help="How quickly feed rate responds to substrate error")
            
            # Substrate influence on pH and Temperature
            st.markdown("#### Substrate Influence")
            substrate_influence_pH = st.slider("Substrate Influence on pH Control", 0.0, 1.0, 0.3, 0.1,
                                              key="sub_inf_pH",
                                              help="How much substrate level affects pH control (0=none, 1=high)")
            substrate_influence_temp = st.slider("Substrate Influence on Temp Control", 0.0, 1.0, 0.3, 0.1,
                                                key="sub_inf_temp",
                                                help="How much substrate level affects temperature control")
        
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
            # Create fuzzy controllers with user-configured parameters
            fuzzy_params = {
                'pH_error_range': pH_error_range,
                'pH_error_optimal_width': pH_error_optimal_width,
                'pH_output_range': pH_output_range,
                'pH_control_aggressive': pH_control_aggressive,
                'substrate_influence_pH': substrate_influence_pH,
                'temp_error_range': temp_error_range,
                'temp_error_optimal_width': temp_error_optimal_width,
                'temp_output_range': temp_output_range,
                'temp_control_aggressive': temp_control_aggressive,
                'substrate_influence_temp': substrate_influence_temp,
                'sub_error_range': sub_error_range,
                'sub_error_zero_width': sub_error_zero_width,
                'feed_rate_max': feed_rate_max,
                'feed_sensitivity': feed_sensitivity
            }
            
            ph_controller = create_ph_fuzzy_controller(fuzzy_params)
            temp_controller = create_temperature_fuzzy_controller(fuzzy_params)
            feed_controller = create_substrate_fuzzy_controller(fuzzy_params)
        
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
                
                # Improved process dynamics for better controllability
                # pH dynamics: First-order response to acid/base addition with natural drift
                tau_pH = 0.5  # time constant [hours] - faster response
                K_pH = 0.3    # gain [pH / control unit] - increased sensitivity
                pH_drift = 0.01  # natural pH drift rate towards neutral (pH 7) - reduced
                pH_natural = 7.0  # natural equilibrium pH
                dpH = (K_pH * acid_base - (pH_array[i-1] - pH_natural) * pH_drift) / tau_pH
                pH_array[i] = pH_array[i-1] + dpH * dt
                
                # Temperature dynamics: First-order heat transfer with ambient temperature
                tau_T = 0.8   # time constant [hours] - faster response
                K_T = 0.5     # gain [°C / control unit] - increased sensitivity
                T_ambient = 25.0  # ambient temperature [°C]
                heat_loss_coeff = 0.05  # heat loss coefficient - reduced
                # Metabolic heat generation (increases with substrate)
                metabolic_heat = 0.01 * substrate_array[i-1] / (5.0 + substrate_array[i-1])  # reduced
                dT = (K_T * heat_cool + metabolic_heat - (temp_array[i-1] - T_ambient) * heat_loss_coeff) / tau_T
                temp_array[i] = temp_array[i-1] + dT * dt
                
                # Substrate dynamics with consumption
                tau_S = 1.5   # time constant [hours] - faster response
                K_S = 0.8     # gain [g/L / control unit] - increased to match consumption
                # Monod-like consumption rate
                consumption_rate = 0.3 * substrate_array[i-1] / (2.0 + substrate_array[i-1])  # reduced
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
        
        # Get fuzzy parameters for visualization
        fuzzy_params = {
            'pH_error_range': st.session_state.get('pH_err_range', (-3.0, 3.0)),
            'pH_error_optimal_width': st.session_state.get('pH_opt_width', 0.5),
            'pH_output_range': st.session_state.get('pH_out_range', 1.0),
            'pH_control_aggressive': st.session_state.get('pH_aggr', 1.0),
            'substrate_influence_pH': st.session_state.get('sub_inf_pH', 0.5),
            'temp_error_range': st.session_state.get('temp_err_range', (-15.0, 15.0)),
            'temp_error_optimal_width': st.session_state.get('temp_opt_width', 3.0),
            'temp_output_range': st.session_state.get('temp_out_range', 1.0),
            'temp_control_aggressive': st.session_state.get('temp_aggr', 1.0),
            'substrate_influence_temp': st.session_state.get('sub_inf_temp', 0.5),
            'sub_error_range': st.session_state.get('sub_err_range', (-30.0, 30.0)),
            'sub_error_zero_width': st.session_state.get('sub_zero_width', 5.0),
            'feed_rate_max': st.session_state.get('feed_max', 1.0),
            'feed_sensitivity': st.session_state.get('feed_sens', 1.0)
        }
        
        # Display membership functions
        st.subheader("🔍 Fuzzy System Visualization")
        st.markdown("Preview of membership functions for each controller:")
        st.info("💡 Tip: Expand the 'Fuzzy Controller Tuning' section in the sidebar to customize membership functions and see the changes here.")
        
        with st.expander("pH Controller Membership Functions", expanded=True):
            fig_ph = visualize_ph_controller(fuzzy_params)
            st.pyplot(fig_ph)
        
        with st.expander("Temperature Controller Membership Functions"):
            fig_temp = visualize_temperature_controller(fuzzy_params)
            st.pyplot(fig_temp)
        
        with st.expander("Substrate Feed Controller Membership Functions"):
            fig_feed = visualize_substrate_controller(fuzzy_params)
            st.pyplot(fig_feed)


def create_ph_fuzzy_controller(params=None):
    """
    Create fuzzy logic controller for pH control based on substrate composition.
    
    Parameters:
    -----------
    params : dict, optional
        Dictionary containing fuzzy controller parameters:
        - pH_error_range: tuple (min, max) for pH error input range
        - pH_error_optimal_width: width of optimal pH zone
        - pH_output_range: maximum control action magnitude
        - pH_control_aggressive: control aggressiveness multiplier
        - substrate_influence_pH: substrate influence factor (0-1)
    
    Inputs:
        - pH_error: Difference between setpoint and measured pH
        - substrate: Current substrate concentration
    
    Output:
        - acid_base: Control action (-1 = add acid, +1 = add base)
    """
    # Default parameters
    if params is None:
        params = {}
    
    pH_min, pH_max = params.get('pH_error_range', (-3.0, 3.0))
    pH_opt_width = params.get('pH_error_optimal_width', 0.5)
    out_range = params.get('pH_output_range', 1.0)
    aggr = params.get('pH_control_aggressive', 1.0)
    sub_inf = params.get('substrate_influence_pH', 0.5)
    
    # Input variables
    pH_error = ctrl.Antecedent(np.linspace(pH_min, pH_max, 100), 'pH_error')
    substrate = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate')
    
    # Output variable
    acid_base = ctrl.Consequent(np.linspace(-out_range, out_range, 100), 'acid_base')
    
    # Calculate membership function points based on parameters
    span = pH_max - pH_min
    mid = (pH_max + pH_min) / 2
    
    # Validate and constrain optimal zone width to prevent invalid membership functions
    max_opt_width = span / 2.5  # Leave some margin for overlapping functions
    pH_opt_width = min(pH_opt_width, max_opt_width)
    
    # Membership functions for pH error (adjusted based on parameters)
    pH_error['very_low'] = fuzz.trapmf(pH_error.universe, 
                                       [pH_min, pH_min, pH_min + span*0.15, pH_min + span*0.35])
    pH_error['low'] = fuzz.trimf(pH_error.universe, 
                                 [pH_min + span*0.15, pH_min + span*0.35, mid])
    pH_error['optimal'] = fuzz.trimf(pH_error.universe, 
                                     [max(pH_min, mid - pH_opt_width), mid, min(pH_max, mid + pH_opt_width)])
    pH_error['high'] = fuzz.trimf(pH_error.universe, 
                                  [mid, pH_max - span*0.35, pH_max - span*0.15])
    pH_error['very_high'] = fuzz.trapmf(pH_error.universe, 
                                        [pH_max - span*0.35, pH_max - span*0.15, pH_max, pH_max])
    
    # Membership functions for substrate concentration
    substrate['very_low'] = fuzz.trapmf(substrate.universe, [0, 0, 5, 10])
    substrate['low'] = fuzz.trimf(substrate.universe, [5, 10, 15])
    substrate['medium'] = fuzz.trimf(substrate.universe, [10, 20, 30])
    substrate['high'] = fuzz.trimf(substrate.universe, [25, 35, 45])
    substrate['very_high'] = fuzz.trapmf(substrate.universe, [40, 45, 50, 50])
    
    # Membership functions for control action (scaled by aggressiveness)
    strong_mag = 0.7 * aggr * out_range
    mild_mag = 0.3 * aggr * out_range
    neutral_width = 0.2 * out_range
    
    acid_base['strong_acid'] = fuzz.trapmf(acid_base.universe, 
                                           [-out_range, -out_range, -strong_mag, -mild_mag])
    acid_base['mild_acid'] = fuzz.trimf(acid_base.universe, 
                                        [-mild_mag*2, -mild_mag, 0])
    acid_base['neutral'] = fuzz.trimf(acid_base.universe, 
                                      [-neutral_width, 0, neutral_width])
    acid_base['mild_base'] = fuzz.trimf(acid_base.universe, 
                                        [0, mild_mag, mild_mag*2])
    acid_base['strong_base'] = fuzz.trapmf(acid_base.universe, 
                                           [mild_mag, strong_mag, out_range, out_range])
    
    # Fuzzy rules for pH control based on pH error and substrate concentration
    # Substrate influence: higher substrate reduces control action due to buffering
    rules = []
    
    # Define rule weights based on substrate influence
    def get_output_modifier(substrate_level, error_level):
        """Modify output based on substrate influence"""
        # Mapping for reducing action strength
        action_reduction = {
            'strong_acid': 'mild_acid',
            'strong_base': 'mild_base',
            'mild_acid': 'neutral',
            'mild_base': 'neutral',
            'neutral': 'neutral'
        }
        
        if sub_inf < 0.3:  # Low influence - substrate doesn't matter much
            return error_level
        elif sub_inf < 0.7:  # Medium influence
            if substrate_level in ['high', 'very_high'] and error_level in ['strong_acid', 'strong_base']:
                return action_reduction.get(error_level, error_level)
            return error_level
        else:  # High influence - substrate significantly reduces action
            if substrate_level == 'very_high':
                return 'neutral'
            elif substrate_level == 'high' and error_level in ['strong_acid', 'strong_base']:
                return 'neutral'
            elif substrate_level == 'high' and error_level in ['mild_acid', 'mild_base']:
                return 'neutral'
            return error_level
    
    # Generate rules systematically
    error_levels = ['very_low', 'low', 'optimal', 'high', 'very_high']
    substrate_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    
    # Map pH errors to appropriate control actions
    error_to_action = {
        'very_low': 'strong_base',   # Very acidic → strong base
        'low': 'mild_base',          # Slightly acidic → mild base
        'optimal': 'neutral',        # Optimal → no action
        'high': 'mild_acid',         # Slightly basic → mild acid
        'very_high': 'strong_acid'   # Very basic → strong acid
    }
    
    for err in error_levels:
        base_action = error_to_action[err]
        for sub in substrate_levels:
            action = get_output_modifier(sub, base_action)
            rules.append(ctrl.Rule(pH_error[err] & substrate[sub], acid_base[action]))
    
    # Create control system
    pH_ctrl = ctrl.ControlSystem(rules)
    pH_simulation = ctrl.ControlSystemSimulation(pH_ctrl)
    
    return pH_simulation


def create_temperature_fuzzy_controller(params=None):
    """
    Create fuzzy logic controller for temperature control based on substrate composition.
    
    Parameters:
    -----------
    params : dict, optional
        Dictionary containing fuzzy controller parameters:
        - temp_error_range: tuple (min, max) for temperature error input range
        - temp_error_optimal_width: width of optimal temperature zone
        - temp_output_range: maximum control action magnitude
        - temp_control_aggressive: control aggressiveness multiplier
        - substrate_influence_temp: substrate influence factor (0-1)
    
    Inputs:
        - temp_error: Difference between setpoint and measured temperature
        - substrate: Current substrate concentration
    
    Output:
        - heat_cool: Control action (-1 = cooling, +1 = heating)
    """
    # Default parameters
    if params is None:
        params = {}
    
    temp_min, temp_max = params.get('temp_error_range', (-15.0, 15.0))
    temp_opt_width = params.get('temp_error_optimal_width', 3.0)
    out_range = params.get('temp_output_range', 1.0)
    aggr = params.get('temp_control_aggressive', 1.0)
    sub_inf = params.get('substrate_influence_temp', 0.5)
    
    # Input variables
    temp_error = ctrl.Antecedent(np.linspace(temp_min, temp_max, 100), 'temp_error')
    substrate = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate')
    
    # Output variable
    heat_cool = ctrl.Consequent(np.linspace(-out_range, out_range, 100), 'heat_cool')
    
    # Calculate membership function points based on parameters
    span = temp_max - temp_min
    mid = (temp_max + temp_min) / 2
    
    # Validate and constrain optimal zone width to prevent invalid membership functions
    max_temp_opt_width = span / 2.5  # Leave some margin for overlapping functions
    temp_opt_width = min(temp_opt_width, max_temp_opt_width)
    
    # Membership functions for temperature error (adjusted based on parameters)
    temp_error['very_cold'] = fuzz.trapmf(temp_error.universe, 
                                          [temp_min, temp_min, temp_min + span*0.15, temp_min + span*0.35])
    temp_error['cold'] = fuzz.trimf(temp_error.universe, 
                                    [temp_min + span*0.15, temp_min + span*0.35, mid])
    temp_error['optimal'] = fuzz.trimf(temp_error.universe, 
                                       [max(temp_min, mid - temp_opt_width), mid, min(temp_max, mid + temp_opt_width)])
    temp_error['hot'] = fuzz.trimf(temp_error.universe, 
                                   [mid, temp_max - span*0.35, temp_max - span*0.15])
    temp_error['very_hot'] = fuzz.trapmf(temp_error.universe, 
                                         [temp_max - span*0.35, temp_max - span*0.15, temp_max, temp_max])
    
    # Membership functions for substrate (affects metabolic heat generation)
    substrate['very_low'] = fuzz.trapmf(substrate.universe, [0, 0, 5, 10])
    substrate['low'] = fuzz.trimf(substrate.universe, [5, 10, 15])
    substrate['medium'] = fuzz.trimf(substrate.universe, [10, 20, 30])
    substrate['high'] = fuzz.trimf(substrate.universe, [25, 35, 45])
    substrate['very_high'] = fuzz.trapmf(substrate.universe, [40, 45, 50, 50])
    
    # Membership functions for control action (scaled by aggressiveness)
    strong_mag = 0.7 * aggr * out_range
    mild_mag = 0.3 * aggr * out_range
    neutral_width = 0.2 * out_range
    
    heat_cool['strong_cooling'] = fuzz.trapmf(heat_cool.universe, 
                                              [-out_range, -out_range, -strong_mag, -mild_mag])
    heat_cool['mild_cooling'] = fuzz.trimf(heat_cool.universe, 
                                           [-mild_mag*2, -mild_mag, 0])
    heat_cool['neutral'] = fuzz.trimf(heat_cool.universe, 
                                      [-neutral_width, 0, neutral_width])
    heat_cool['mild_heating'] = fuzz.trimf(heat_cool.universe, 
                                           [0, mild_mag, mild_mag*2])
    heat_cool['strong_heating'] = fuzz.trapmf(heat_cool.universe, 
                                              [mild_mag, strong_mag, out_range, out_range])
    
    # Fuzzy rules for temperature control based on temp error and substrate concentration
    # Substrate influence: higher substrate increases metabolic heat generation
    rules = []
    
    # Define rule weights based on substrate influence
    def get_output_modifier(substrate_level, error_level):
        """Modify output based on substrate influence (metabolic heat)"""
        if sub_inf < 0.3:  # Low influence - substrate doesn't affect much
            return error_level
        elif sub_inf < 0.7:  # Medium influence
            # High substrate means more metabolic heat, reduce heating or increase cooling
            if substrate_level in ['high', 'very_high']:
                if error_level in ['strong_heating', 'mild_heating']:
                    return 'mild_heating' if error_level == 'strong_heating' else 'neutral'
                elif error_level == 'mild_cooling':
                    return 'strong_cooling'
            return error_level
        else:  # High influence - substrate significantly affects temperature
            if substrate_level == 'very_high':
                if 'heating' in error_level:
                    return 'neutral'
                elif error_level == 'neutral':
                    return 'mild_cooling'
                else:  # cooling
                    return 'strong_cooling'
            elif substrate_level == 'high':
                if error_level == 'strong_heating':
                    return 'neutral'
                elif error_level == 'mild_heating':
                    return 'neutral'
                elif error_level == 'mild_cooling':
                    return 'strong_cooling'
            return error_level
    
    # Generate rules systematically
    error_levels = ['very_cold', 'cold', 'optimal', 'hot', 'very_hot']
    substrate_levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    
    # Map temperature errors to appropriate control actions
    error_to_action = {
        'very_cold': 'strong_heating',  # Very cold → strong heating
        'cold': 'mild_heating',         # Cold → mild heating
        'optimal': 'neutral',           # Optimal → no action
        'hot': 'mild_cooling',          # Hot → mild cooling
        'very_hot': 'strong_cooling'    # Very hot → strong cooling
    }
    
    for err in error_levels:
        base_action = error_to_action[err]
        for sub in substrate_levels:
            action = get_output_modifier(sub, base_action)
            rules.append(ctrl.Rule(temp_error[err] & substrate[sub], heat_cool[action]))
    
    # Create control system
    temp_ctrl = ctrl.ControlSystem(rules)
    temp_simulation = ctrl.ControlSystemSimulation(temp_ctrl)
    
    return temp_simulation


def create_substrate_fuzzy_controller(params=None):
    """
    Create fuzzy logic controller for substrate feeding based on substrate level and error.
    
    Parameters:
    -----------
    params : dict, optional
        Dictionary containing fuzzy controller parameters:
        - sub_error_range: tuple (min, max) for substrate error input range
        - sub_error_zero_width: width of zero error zone
        - feed_rate_max: maximum feed rate
        - feed_sensitivity: feed rate sensitivity multiplier
    
    Inputs:
        - substrate_error: Difference between target and current substrate
        - substrate_level: Current substrate concentration
    
    Output:
        - feed_rate: Substrate feed rate (0 to 1, normalized)
    """
    # Default parameters
    if params is None:
        params = {}
    
    sub_min, sub_max = params.get('sub_error_range', (-30.0, 30.0))
    zero_width = params.get('sub_error_zero_width', 5.0)
    max_feed = params.get('feed_rate_max', 1.0)
    sensitivity = params.get('feed_sensitivity', 1.0)
    
    # Input variables
    substrate_error = ctrl.Antecedent(np.linspace(sub_min, sub_max, 100), 'substrate_error')
    substrate_level = ctrl.Antecedent(np.linspace(0, 50, 100), 'substrate_level')
    
    # Output variable
    feed_rate = ctrl.Consequent(np.linspace(0, max_feed, 100), 'feed_rate')
    
    # Calculate membership function points based on parameters
    span = sub_max - sub_min
    mid = (sub_max + sub_min) / 2
    
    # Membership functions for substrate error (adjusted based on parameters)
    substrate_error['very_negative'] = fuzz.trapmf(substrate_error.universe, 
                                                    [sub_min, sub_min, sub_min + span*0.15, sub_min + span*0.35])
    substrate_error['negative'] = fuzz.trimf(substrate_error.universe, 
                                            [sub_min + span*0.15, sub_min + span*0.35, mid])
    substrate_error['zero'] = fuzz.trimf(substrate_error.universe, 
                                         [mid - zero_width, mid, mid + zero_width])
    substrate_error['positive'] = fuzz.trimf(substrate_error.universe, 
                                            [mid, sub_max - span*0.35, sub_max - span*0.15])
    substrate_error['very_positive'] = fuzz.trapmf(substrate_error.universe, 
                                                   [sub_max - span*0.35, sub_max - span*0.15, sub_max, sub_max])
    
    # Membership functions for substrate level
    substrate_level['very_low'] = fuzz.trapmf(substrate_level.universe, [0, 0, 5, 10])
    substrate_level['low'] = fuzz.trimf(substrate_level.universe, [5, 10, 20])
    substrate_level['medium'] = fuzz.trimf(substrate_level.universe, [15, 25, 35])
    substrate_level['high'] = fuzz.trimf(substrate_level.universe, [30, 40, 50])
    substrate_level['very_high'] = fuzz.trapmf(substrate_level.universe, [45, 50, 50, 50])
    
    # Membership functions for feed rate (scaled by sensitivity)
    # Sensitivity affects how quickly feed rate responds to errors
    # Ensure proper sorting to avoid invalid triangular functions when sensitivity > 1
    feed_rate['none'] = fuzz.trimf(feed_rate.universe, [0, 0, 0.05 * max_feed])
    feed_rate['very_low'] = fuzz.trimf(feed_rate.universe, 
                                       safe_trimf_points(0, 0.15 * sensitivity * max_feed, 0.3 * sensitivity * max_feed, max_feed))
    feed_rate['low'] = fuzz.trimf(feed_rate.universe, 
                                  safe_trimf_points(0.2 * sensitivity * max_feed, 0.35 * sensitivity * max_feed, 0.5 * sensitivity * max_feed, max_feed))
    feed_rate['medium'] = fuzz.trimf(feed_rate.universe, 
                                     safe_trimf_points(0.4 * sensitivity * max_feed, 0.55 * sensitivity * max_feed, 0.7 * sensitivity * max_feed, max_feed))
    feed_rate['high'] = fuzz.trimf(feed_rate.universe, 
                                   safe_trimf_points(0.6 * sensitivity * max_feed, 0.75 * sensitivity * max_feed, 0.9 * sensitivity * max_feed, max_feed))
    feed_rate['very_high'] = fuzz.trimf(feed_rate.universe, 
                                        safe_trimf_points(0.8 * sensitivity * max_feed, 0.95 * max_feed, max_feed, max_feed))
    
    # Fuzzy rules for substrate feeding
    # Positive error → need more substrate → increase feed
    # Negative error → too much substrate → decrease feed
    # Current level also affects decision (avoid overfeeding)
    rules = []
    
    # Define rule logic based on error and current level
    error_levels = ['very_negative', 'negative', 'zero', 'positive', 'very_positive']
    level_categories = ['very_low', 'low', 'medium', 'high', 'very_high']
    
    # Map substrate error to base feed rates
    error_to_base_feed = {
        'very_negative': 'none',      # Too much substrate → no feed
        'negative': 'none',           # Excess substrate → no feed
        'zero': 'very_low',           # At target → maintain low feed
        'positive': 'medium',         # Need substrate → moderate feed
        'very_positive': 'high'       # Need much more → high feed
    }
    
    # Adjust feed rate based on current substrate level
    def get_adjusted_feed(error_cat, level_cat):
        """Adjust feed rate based on error and current level"""
        base_feed = error_to_base_feed[error_cat]
        
        if base_feed == 'none':
            return 'none'
        
        # Reduce feed if substrate level is already high
        feed_levels = ['none', 'very_low', 'low', 'medium', 'high', 'very_high']
        base_idx = feed_levels.index(base_feed)
        
        # Adjust based on current level
        level_idx = level_categories.index(level_cat)
        
        if level_idx >= 4:  # very_high
            return 'none'
        elif level_idx >= 3:  # high
            adjusted_idx = max(0, base_idx - 2)
        elif level_idx >= 2:  # medium
            adjusted_idx = max(0, base_idx - 1)
        else:  # low or very_low
            # Increase feed if error is positive and level is low
            if 'positive' in error_cat:
                adjusted_idx = min(len(feed_levels) - 1, base_idx + 1)
            else:
                adjusted_idx = base_idx
        
        return feed_levels[adjusted_idx]
    
    # Generate rules systematically
    for err in error_levels:
        for level in level_categories:
            adjusted_feed = get_adjusted_feed(err, level)
            rules.append(ctrl.Rule(substrate_error[err] & substrate_level[level], feed_rate[adjusted_feed]))
    
    # Create control system
    feed_ctrl = ctrl.ControlSystem(rules)
    feed_simulation = ctrl.ControlSystemSimulation(feed_ctrl)
    
    return feed_simulation


def visualize_ph_controller(params=None):
    """Visualize pH controller membership functions."""
    # Default parameters
    if params is None:
        params = {}
    
    pH_min, pH_max = params.get('pH_error_range', (-3.0, 3.0))
    pH_opt_width = params.get('pH_error_optimal_width', 0.5)
    out_range = params.get('pH_output_range', 1.0)
    aggr = params.get('pH_control_aggressive', 1.0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Calculate membership function points
    span = pH_max - pH_min
    mid = (pH_max + pH_min) / 2
    
    # pH error
    pH_error = np.linspace(pH_min, pH_max, 100)
    axes[0].plot(pH_error, fuzz.trapmf(pH_error, [pH_min, pH_min, pH_min + span*0.15, pH_min + span*0.35]), 
                'b', linewidth=2, label='Very Low')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [pH_min + span*0.15, pH_min + span*0.35, mid]), 
                'g', linewidth=2, label='Low')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [mid - pH_opt_width, mid, mid + pH_opt_width]), 
                'orange', linewidth=2, label='Optimal')
    axes[0].plot(pH_error, fuzz.trimf(pH_error, [mid, pH_max - span*0.35, pH_max - span*0.15]), 
                'r', linewidth=2, label='High')
    axes[0].plot(pH_error, fuzz.trapmf(pH_error, [pH_max - span*0.35, pH_max - span*0.15, pH_max, pH_max]), 
                'm', linewidth=2, label='Very High')
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
    strong_mag = 0.7 * aggr * out_range
    mild_mag = 0.3 * aggr * out_range
    neutral_width = 0.2 * out_range
    
    acid_base = np.linspace(-out_range, out_range, 100)
    axes[2].plot(acid_base, fuzz.trapmf(acid_base, [-out_range, -out_range, -strong_mag, -mild_mag]), 
                'b', linewidth=2, label='Strong Acid')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [-mild_mag*2, -mild_mag, 0]), 
                'c', linewidth=2, label='Mild Acid')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [-neutral_width, 0, neutral_width]), 
                'orange', linewidth=2, label='Neutral')
    axes[2].plot(acid_base, fuzz.trimf(acid_base, [0, mild_mag, mild_mag*2]), 
                'g', linewidth=2, label='Mild Base')
    axes[2].plot(acid_base, fuzz.trapmf(acid_base, [mild_mag, strong_mag, out_range, out_range]), 
                'm', linewidth=2, label='Strong Base')
    axes[2].set_title('pH Control Action Membership Functions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Acid/Base Action')
    axes[2].set_ylabel('Membership Degree')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_temperature_controller(params=None):
    """Visualize temperature controller membership functions."""
    # Default parameters
    if params is None:
        params = {}
    
    temp_min, temp_max = params.get('temp_error_range', (-15.0, 15.0))
    temp_opt_width = params.get('temp_error_optimal_width', 3.0)
    out_range = params.get('temp_output_range', 1.0)
    aggr = params.get('temp_control_aggressive', 1.0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Calculate membership function points
    span = temp_max - temp_min
    mid = (temp_max + temp_min) / 2
    
    # Temperature error
    temp_error = np.linspace(temp_min, temp_max, 100)
    axes[0].plot(temp_error, fuzz.trapmf(temp_error, [temp_min, temp_min, temp_min + span*0.15, temp_min + span*0.35]), 
                'b', linewidth=2, label='Very Cold')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [temp_min + span*0.15, temp_min + span*0.35, mid]), 
                'c', linewidth=2, label='Cold')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [mid - temp_opt_width, mid, mid + temp_opt_width]), 
                'g', linewidth=2, label='Optimal')
    axes[0].plot(temp_error, fuzz.trimf(temp_error, [mid, temp_max - span*0.35, temp_max - span*0.15]), 
                'orange', linewidth=2, label='Hot')
    axes[0].plot(temp_error, fuzz.trapmf(temp_error, [temp_max - span*0.35, temp_max - span*0.15, temp_max, temp_max]), 
                'r', linewidth=2, label='Very Hot')
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
    strong_mag = 0.7 * aggr * out_range
    mild_mag = 0.3 * aggr * out_range
    neutral_width = 0.2 * out_range
    
    heat_cool = np.linspace(-out_range, out_range, 100)
    axes[2].plot(heat_cool, fuzz.trapmf(heat_cool, [-out_range, -out_range, -strong_mag, -mild_mag]), 
                'b', linewidth=2, label='Strong Cooling')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [-mild_mag*2, -mild_mag, 0]), 
                'c', linewidth=2, label='Mild Cooling')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [-neutral_width, 0, neutral_width]), 
                'g', linewidth=2, label='Neutral')
    axes[2].plot(heat_cool, fuzz.trimf(heat_cool, [0, mild_mag, mild_mag*2]), 
                'orange', linewidth=2, label='Mild Heating')
    axes[2].plot(heat_cool, fuzz.trapmf(heat_cool, [mild_mag, strong_mag, out_range, out_range]), 
                'r', linewidth=2, label='Strong Heating')
    axes[2].set_title('Temperature Control Action Membership Functions', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Heat/Cool Action')
    axes[2].set_ylabel('Membership Degree')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_substrate_controller(params=None):
    """Visualize substrate feed controller membership functions."""
    # Default parameters
    if params is None:
        params = {}
    
    sub_min, sub_max = params.get('sub_error_range', (-30.0, 30.0))
    zero_width = params.get('sub_error_zero_width', 5.0)
    max_feed = params.get('feed_rate_max', 1.0)
    sensitivity = params.get('feed_sensitivity', 1.0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Calculate membership function points
    span = sub_max - sub_min
    mid = (sub_max + sub_min) / 2
    
    # Substrate error
    substrate_error = np.linspace(sub_min, sub_max, 100)
    axes[0].plot(substrate_error, fuzz.trapmf(substrate_error, [sub_min, sub_min, sub_min + span*0.15, sub_min + span*0.35]), 
                'b', linewidth=2, label='Very Negative')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [sub_min + span*0.15, sub_min + span*0.35, mid]), 
                'c', linewidth=2, label='Negative')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [mid - zero_width, mid, mid + zero_width]), 
                'g', linewidth=2, label='Zero')
    axes[0].plot(substrate_error, fuzz.trimf(substrate_error, [mid, sub_max - span*0.35, sub_max - span*0.15]), 
                'orange', linewidth=2, label='Positive')
    axes[0].plot(substrate_error, fuzz.trapmf(substrate_error, [sub_max - span*0.35, sub_max - span*0.15, sub_max, sub_max]), 
                'r', linewidth=2, label='Very Positive')
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
    feed_rate = np.linspace(0, max_feed, 100)
    
    # Calculate feed rate membership function points with proper sorting to avoid invalid triangular functions
    # When sensitivity is high, ensure a <= b <= c for all trimf calls by sorting and clipping
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, [0, 0, 0.05 * max_feed]), 
                'b', linewidth=2, label='None')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, 
                                       safe_trimf_points(0, 0.15 * sensitivity * max_feed, 0.3 * sensitivity * max_feed, max_feed)), 
                'c', linewidth=2, label='Very Low')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, 
                                       safe_trimf_points(0.2 * sensitivity * max_feed, 0.35 * sensitivity * max_feed, 0.5 * sensitivity * max_feed, max_feed)), 
                'g', linewidth=2, label='Low')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, 
                                       safe_trimf_points(0.4 * sensitivity * max_feed, 0.55 * sensitivity * max_feed, 0.7 * sensitivity * max_feed, max_feed)), 
                'y', linewidth=2, label='Medium')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, 
                                       safe_trimf_points(0.6 * sensitivity * max_feed, 0.75 * sensitivity * max_feed, 0.9 * sensitivity * max_feed, max_feed)), 
                'orange', linewidth=2, label='High')
    axes[2].plot(feed_rate, fuzz.trimf(feed_rate, 
                                       safe_trimf_points(0.8 * sensitivity * max_feed, 0.95 * max_feed, max_feed, max_feed)), 
                'r', linewidth=2, label='Very High')
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
