# home_page.py
import streamlit as st

def home_page():
    st.title("Bioprocess Modeling and Control")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/Batch.png", use_container_width=True)
        st.caption("**Figure 1:** Batch Reactor")
    with col2:
        st.image("images/fed_batch.png", use_container_width=True)
        st.caption("**Figure 2:** Fed-Batch Reactor")
    with col3:
        st.image("images/continous.png", use_container_width=True)
        st.caption("**Figure 3:** Continuous Reactor")

    st.markdown("""
        Welcome to the interactive simulator for bioprocess modeling and control. 
        This tool allows you to explore different reactor operation modes, 
        microbial growth kinetics and advanced control strategies.
        
        This educational platform is based on fundamental concepts from bioprocess engineering 
        (**Bailey & Ollis, 1986; Shuler & Kargi, 2002**), classical process control 
        (**Smith & Corripio, 2005; Stephanopoulos, 1984**), and advanced control techniques 
        (**Camacho & Bordons, 2007; Rawlings et al., 2017**).
        """)

    st.markdown("---") # Visual separator

    # ========= NEW SECTION: IMPLEMENTED KINETIC MODELS =========
    st.header("🔬 Implemented kinetic models")
    st.markdown("""
        The heart of bioprocess modeling lies in mathematically describing the rates at which key biological 
        reactions occur: cell growth, substrate consumption and product formation. The kinetic models implemented 
        in this simulator are detailed below, with a particular focus on alcoholic fermentation as a case study.
        """)

    st.subheader("Specific Growth Rate ($\mu$)")
    st.markdown(r"""
        The specific growth rate (($\mu$), unit $h^{-1}$) describes how fast the biomass increases per unit of 
        existing biomass. It depends on factors such as substrate ($S$), product ($P$) and dissolved oxygen ($O_2$)
        concentration. The implemented models are:
        """)

    with st.expander("1. Simple Monod"):
        st.markdown("""The most basic model, proposed by **Monod (1949)**, assumes that growth is limited by a single substrate (S). 
        This hyperbolic relationship is analogous to Michaelis-Menten enzyme kinetics and is widely used in bioprocess engineering 
        (**Shuler & Kargi, 2002; Bailey & Ollis, 1986**).""")
        st.latex(r"""
        \mu = \mu_{max} \frac{S}{K_S + S}
        """)
        st.markdown(r"""
            * $\mu_{max}$: Maximum specific growth rate ($h^{-1}$).
            * $K_S$: Substrate affinity constant (concentration of $S$ at which $\mu = \mu_{max}/2$, units $g/L$). Also called saturation constant or half-velocity constant.
            
            **Reference:** Monod, J. (1949). "The growth of bacterial cultures." *Annual Review of Microbiology*, 3(1), 371-394.
            """)

    with st.expander("2. Sigmoidal Monod (Hill)"):
        st.markdown("""It introduces a threshold or cooperative response to the substrate, useful for modeling induction phenomena or more complex kinetics. 
        Based on the **Hill equation**, commonly used to describe cooperative binding in enzyme kinetics (**Shuler & Kargi, 2002**).""")
        st.latex(r"""
        \mu = \mu_{max} \frac{S^n}{K_S^n + S^n}
        """)
        st.markdown(r"""
            * $n$: Hill coefficient (dimensionless), controls sigmoidal shape. $n > 1$ indicates positive cooperativity, $n = 1$ reduces to simple Monod, and $n < 1$ indicates negative cooperativity.
            
            **Reference:** Hill, A. V. (1910). "The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves." *Journal of Physiology*, 40, iv-vii.
            """)

    with st.expander("3. Monod with Constraints (S, O2, P)"):
        st.markdown("""Models growth simultaneously limited by substrate and oxygen, and inhibited by product. 
        This multiplicative model combines multiple Monod terms with product inhibition, as described in **Bailey & Ollis (1986)** 
        and **Shuler & Kargi (2002)** for systems with multiple growth-limiting factors.""")
        st.latex(r"""
        \mu = \mu_{max} \left(\frac{S}{K_S + S}\right) \left(\frac{O_2}{K_O + O_2}\right) \left(\frac{K_P}{K_P + P}\right)
        """)
        st.markdown(r"""
            * $K_O$: Oxygen affinity constant ($g/L$).
            * $K_P$: Inhibition constant per product ($g/L$). *Note: This form represents simple non-competitive inhibition.*
            
            For systems with substrate inhibition at high concentrations, the **Haldane (Andrews) model** may be more appropriate:
            $\mu = \mu_{max} \dfrac{S}{K_S + S + S^2/K_I}$
            
            **References:** 
            - Haldane, J. B. S. (1930). *Enzymes*. Longmans, Green and Co.
            - Andrews, J. F. (1968). "A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates." *Biotechnology and Bioengineering*, 10(6), 707-723.
            """)

    with st.expander("4. Fermentation (Mixed Aerobic/Anaerobic)"):
        st.markdown(r"""
            It models the coexistence of metabolic pathways. The total rate is the sum of an aerobic and an anaerobic/fermentative component, modulated by $S$, $P$ and $O_2$.
            $\mu_{total} = \mu_{aerobic} + \mu_{anaerobic}$
            """)
        st.markdown("**Aerobic component:**")
        st.latex(r"""
        \mu_{aerobic} = \mu_{max, aerob} \left( \frac{S}{K_{S, aerob} + S} \right) \left( \frac{O_2}{K_{O, aerob} + O_2} \right)
        """)
        st.markdown("**Anaerobic/Fermentative Component:**")
        # We use the version implemented in the latest code
        st.latex(r"""
        \mu_{anaerobic} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( 1 - \frac{P}{K_{P, anaerob}} \right)^{n_p} \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
        """)
        # Alternative form with KP^n (less used in the final code, but common):
        # st.latex(r"""
        # \mu_{anaerobic} = \mu_{max, anaerob} \left( \frac{S}{K_{S, anaerob} + S + S^2/K_{iS, anaerob}} \right) \left( \frac{K_{P, anaerob}^{n_p}}{K_{P, anaerob}^{n_p} + P^{n_p}} \right) \left( \frac{K_{O, inhib}}{K_{O, inhib} + O_2} \right)
        # """)
        st.markdown(r"""
            **Specific Parameters:**
            * $\mu_{max, aerob}, \mu_{max, anaerob}$: Maximum growth rates ($h^{-1}$).
            * $K_{S, aerob}, K_{S, anaerob}$: Substrate affinity constants ($g/L$).
            * $K_{O, aerob}$: $O_2$ affinity constant for $\mu_{aerobic}$ ($g/L$).
            * $K_{iS, anaerob}$: Inhibition constant per substrate for $\mu_{anaerobic}$ ($g/L$).
            * $K_{P, anaerob}$: Inhibition constant per product (ethanol) for $\mu_{anaerobic}$ ($g/L$). Represents the critical concentration of $P$ that stops anaerobic growth..
            * $n_p$: Inhibition exponent per product (dimensionless).
            * $K_{O, inhib}$: Inhibition constant per $O_2$ for $\mu_{anaerobic}$ (Pasteur effect on growth, $g/L$).
            """)

    with st.expander("5. Switched Fermentation"):
        st.markdown(r"""
            Simulates a discrete metabolic change. Uses **only** $\mu_{aerobic}$ (equation seen above) during Phase 1 (aerobic) and **only** $\mu_{anaerobic}$ (equation seen above) during Phases 2 and 3 (anaerobics). It requires defining parameters for both components, but only one is active in each phase.
            """)

    st.subheader("Specific Product Formation Rate ($q_P$)")
    st.markdown(r"""
        The specific product formation rate ($q_P$, unit $g_P \cdot g_X^{-1} \cdot h^{-1}$), specifically for ethanol in this context, 
        is modeled using the **Luedeking-Piret equation** modified to include direct inhibition by oxygen. This reflects that ethanol 
        production is predominantly anaerobic (**Luedeking & Piret, 1959**).
        
        The Luedeking-Piret model distinguishes between growth-associated and non-growth-associated product formation, 
        which is critical for understanding metabolic behavior in bioprocesses (**Bailey & Ollis, 1986; Shuler & Kargi, 2002**).
        """)
    st.latex(r"""
    q_P = (\alpha \cdot \mu_{anaerob} + \beta) 
    """)
    st.markdown(r"""
        * $\mu_{anaerob}$: It is the specific growth rate calculated by the selected kinetic model. ($\mu_{anaerob}$ if Mixed/Switched).
        * $\alpha$: Growth-associated product formation coefficient ($g_P \cdot g_X^{-1}$). When $\alpha > 0$, product formation is coupled to growth.
        * $\beta$: Non-growth-associated product formation coefficient ($g_P \cdot g_X^{-1} \cdot h^{-1}$). When $\beta > 0$, product is formed even in stationary phase.
        * $K_{O,P}$: Oxygen inhibition constant on ethanol *production* ($g/L$). A low value indicates strong suppression of $P$ production by $O_2$.
        
        **Reference:** Luedeking, R., & Piret, E. L. (1959). "A kinetic study of the lactic acid fermentation. Batch process at controlled pH." 
        *Journal of Biochemical and Microbiological Technology and Engineering*, 1(4), 393-412.
        """)

    st.markdown("---") # Visual separator

    # ========= MOVED SECTION: BALANCE SHEETS =========
    st.header("📊 Theoretical Basis: Material Balances") # Adjusted title
    st.markdown("""
        Bioprocess modeling allows to describe mathematically the evolution of the variables of interest 
        (biomass concentration, substrate, product, dissolved oxygen, etc.) in a bioreactor.
        The general material balances for the three main modes of operation are presented below, assuming 
        perfect mixing. The rates $\mu$ and $q_P$ correspond to the kinetic models described above.
        
        These mass balance equations are derived from fundamental conservation principles 
        (**Bailey & Ollis, 1986; Shuler & Kargi, 2002**) and form the basis for dynamic simulation 
        and control of bioprocesses. The equations are ordinary differential equations (ODEs) solved 
        numerically using methods such as Runge-Kutta (**Press et al., 2007**), implemented in 
        this project via `scipy.integrate.solve_ivp`.
        """)

    st.subheader("🔹 Batch Mode")
    st.markdown("""
        There is no input or output of matter once the process has started ($F=0$). The volume $V$ is constant.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a_1 \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X""")
    st.markdown(r"""*Note: $OUR = (\frac{\mu}{Y_{XO}} + m_o) X$. The $k_L a$ may vary depending on the phase.*""")


    st.subheader("🔹 Fed-Batch Mode")
    st.markdown(r"""
        Feed ($F$) with concentration $S_{in}$ is added. The volume $V$ varies: $\frac{dV}{dt} = F$.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X - \frac{F}{V} \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X + \frac{F}{V} (S_{in} - S)""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X - \frac{F}{V} \cdot P""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X - \frac{F}{V} \cdot O_2""")


    st.subheader("🔹 Continuous Mode (Chemostat)")
    st.markdown(r"""
        Inflow and outflow $F$ at the same rate. Volume $V$ constant. Dilution rate $D = F/V$.
        """)
    st.latex(r"""\frac{dX}{dt} = \mu \cdot X - k_d \cdot X - D \cdot X""")
    st.latex(r"""\frac{dS}{dt} = - \frac{\mu}{Y_{XS}} \cdot X - m_s \cdot X - \frac{q_P}{Y_{PS}} \cdot X + D (S_{in} - S)""")
    st.latex(r"""\frac{dP}{dt} = q_P \cdot X - D \cdot P""")
    st.latex(r"""\frac{dO_2}{dt} = k_{L}a \cdot (C_{S}^* - O_2) - \left( \frac{\mu}{Y_{XO}} + m_o \right) \cdot X - D \cdot O_2""")

    st.markdown(r"""
        **General Parameters in Balance Sheets:**
        * $X, S, P, O_2, V, F, S_{in}$: Variables already defined.
        * $\mu, q_P$: Specific rates (defined by kinetic models).
        * $k_d$: Specific rate of cell decay ($h^{-1}$).
        * $Y_{XS}$: Biomass/substrate yield ($g_X / g_S$).
        * $Y_{PS}$: Product/substrate yield ($g_P / g_S$), used to calculate $S$ consumption for $P$.
        * $Y_{XO}$: Biomass/oxygen yield ($g_X / g_{O2}$).
        * $m_s$: Substrate-based maintenance ($g_S \cdot g_X^{-1} \cdot h^{-1}$).
        * $m_o$: Oxygen-based maintenance ($g_{O2} \cdot g_X^{-1} \cdot h^{-1}$).
        * $k_L a$: Oxygen transfer coefficient ($h^{-1}$, can be $k_{L}a_1$ or $k_{L}a_2$).
        * $C_{S}^*$: $O_2$ saturation concentration ($g/L$).
        * $D$: Dilution rate ($h^{-1}$).
        * $t$: Time ($h$).
        """)

    st.markdown("---") # Separator before advanced techniques

    # ========= MOVED SECTION: ADVANCED TECHNIQUES =========
    st.header("⚙️ Analysis and Control Techniques")
    
    st.subheader("🔹 PID Control (Regulatory Control)")
    st.markdown(r"""
        **Proportional-Integral-Derivative (PID)** controllers are the workhorse of industrial process control, 
        accounting for over 90% of control loops (**Smith & Corripio, 2005; Seborg et al., 2016**). 
        A PID controller adjusts the manipulated variable $u(t)$ based on the error $e(t) = y_{sp}(t) - y(t)$ 
        between the setpoint and the measured process variable.
        
        **PID Control Law:**
        $$u(t) = K_c \left[ e(t) + \frac{1}{\tau_I} \int_0^t e(\tau) d\tau + \tau_D \frac{de(t)}{dt} \right] + u_{bias}$$
        
        Or in velocity (incremental) form:
        $$\Delta u(t) = K_c \left[ \Delta e(t) + \frac{T_s}{\tau_I} e(t) + \frac{\tau_D}{T_s} (\Delta e(t) - \Delta e(t-T_s)) \right]$$
        
        Where:
        - $K_c$: Controller gain (proportional action strength)
        - $\tau_I$: Integral time constant (eliminates steady-state offset)
        - $\tau_D$: Derivative time constant (anticipates future error, improves response)
        - $T_s$: Sampling time
        - $u_{bias}$: Bias or nominal value of manipulated variable
        
        **Controller Tuning Methods:**
        - **Ziegler-Nichols:** Based on ultimate gain and period from closed-loop oscillation
        - **Cohen-Coon:** Uses open-loop step response characteristics (dead time, time constant)
        - **Internal Model Control (IMC):** Model-based tuning with desired closed-loop time constant
        - **Auto-tuning:** Automated identification and tuning procedures
        
        **Common PID Control Loops in Bioprocesses:**
        - **Temperature control:** Heating/cooling jacket manipulation
        - **pH control:** Acid/base addition (often split-range control)
        - **Dissolved oxygen (DO) control:** Agitation speed or air flow rate
        - **Level control:** Feed or harvest rate
        - **Pressure control:** Vent or inlet valve position
        
        **Advanced PID Strategies:**
        - **Cascade Control:** Two controllers in series (e.g., DO-agitation-motor current)
        - **Split-Range Control:** One controller driving two or more final control elements (e.g., pH with acid and base)
        - **Feedforward Control:** Anticipatory action based on measured disturbances
        - **Ratio Control:** Maintaining fixed ratio between two flows
        
        **References:**
        - Smith, C. A., & Corripio, A. B. (2005). *Principles and Practice of Automatic Process Control* (3rd ed.). John Wiley & Sons.
        - Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). *Process Dynamics and Control* (4th ed.). John Wiley & Sons.
        - Stephanopoulos, G. (1984). *Chemical Process Control: An Introduction to Theory and Practice*. Prentice Hall.
        """)
    
    st.markdown("---")
    st.subheader("🔹 Sensitivity analysis")
    st.markdown(r"""
        Evaluates how uncertainty or variations in the model parameters ($\theta$, such as $\mu_{\max}, K_S, Y_{XS}$, etc.) 
        affect the model outputs (the state variables $X, S, P, O_2$). It allows identifying the most influential parameters, 
        crucial for optimization and experimental design (**Bard, 1974; Beck & Arnold, 1977**). 
        
        A common metric is the **normalized sensitivity coefficient**:
        $$S_{ij} = \frac{\partial y_i / y_i}{\partial \theta_j / \theta_j} = \frac{\partial \ln y_i}{\partial \ln \theta_j}$$
        where $y_i$ is an output and $\theta_j$ is a parameter.
        
        Sensitivity analysis is essential for:
        - **Parameter identifiability:** Determining which parameters can be reliably estimated from data
        - **Experimental design:** Identifying optimal measurement strategies
        - **Uncertainty quantification:** Understanding how parameter uncertainty propagates to predictions
        
        **References:** 
        - Bard, Y. (1974). *Nonlinear Parameter Estimation*. Academic Press.
        - Beck, J. V., & Arnold, K. J. (1977). *Parameter Estimation in Engineering and Science*. John Wiley & Sons.
        """)
    st.markdown("---")
    st.subheader("🔹 Parameter Adjustment (Estimation)")
    st.markdown(r"""
        The process of finding the values of the model parameters ($\theta$) that best describe a set of experimental data ($y_{exp}$). 
        An objective function $J(\theta)$ that measures the discrepancy between the model predictions ($y_{model}$) and the data is minimized 
        using numerical optimization algorithms (**Bard, 1974; Nocedal & Wright, 2006**).
        
        The optimization problem is:
        """)
    st.latex(r"""
    \hat{\theta} = \arg \min_{\theta} J(\theta)
    """)
    st.markdown(r"""
        A common objective function is the **sum of weighted squared errors** (least squares):
        $$J(\theta) = \sum_{k=1}^{N} \sum_{i=1}^{M} w_{ik} (y_{i,exp}(t_k) - y_{i,model}(t_k, \theta))^2$$
        where $N$ is the number of sampling points, $M$ the number of measured variables, and $w_{ik}$ are weights.
        
        **Common optimization algorithms used:**
        - **Levenberg-Marquardt:** Efficient for nonlinear least squares (**Nocedal & Wright, 2006**)
        - **Sequential Quadratic Programming (SQP):** For constrained optimization (**Nocedal & Wright, 2006**)
        - **Trust Region Methods:** Robust convergence properties
        - **Genetic Algorithms:** Global optimization for multimodal problems
        
        **Statistical analysis** of the results includes:
        - **Coefficient of determination** ($R^2$): Goodness of fit
        - **Root Mean Square Error** (RMSE): Average prediction error
        - **Confidence intervals:** Uncertainty in parameter estimates (**Bard, 1974**)
        - **Correlation matrix:** Parameter interdependencies
        
        **References:**
        - Bard, Y. (1974). *Nonlinear Parameter Estimation*. Academic Press.
        - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
        - Beck, J. V., & Arnold, K. J. (1977). *Parameter Estimation in Engineering and Science*. John Wiley & Sons.
        """)
    st.markdown("---")
    st.subheader("🔹 Extended Kalman Filter (EKF)")
    st.markdown(r"""
        Recursive algorithm for estimating the state of **nonlinear dynamic systems** in the presence of noise. Uses a model of the system and 
        noisy measurements to obtain an optimal (in the sense of minimum variance) estimate of the state. Essential when not all state variables 
        (e.g., biomass) can be measured directly online (**Jazwinski, 1970; Simon, 2006**).
        
        The EKF extends the linear Kalman filter to nonlinear systems by **linearizing** the system dynamics around the current state estimate 
        using Jacobian matrices. It is widely used in bioprocess monitoring where direct measurement of all states is impractical or expensive 
        (**Bastin & Dochain, 1990**).
        
        **System model (discrete):**
        """)
    st.latex(r"x_{k+1} = f(x_k, u_k) + w_k \quad \text{(Process equation)}")
    st.latex(r"z_k = h(x_k) + v_k \quad \text{(Measurement equation)}")
    st.markdown(r"""
        * $x_k$: State vector at time $k$ (e.g., $[X, S, P, O_2]^T$)
        * $u_k$: Input vector (e.g., feed rate, agitation speed)
        * $z_k$: Measurement vector (e.g., temperature, pH, dissolved oxygen)
        * $w_k \sim N(0, Q)$: Process noise with covariance $Q$
        * $v_k \sim N(0, R)$: Measurement noise with covariance $R$
        * $f$: Nonlinear state transition function (bioprocess model ODEs)
        * $h$: Nonlinear measurement function
        
        **EKF Algorithm Stages:**
        
        1.  **Prediction (Time Update):** 
            - State prediction: $\hat{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k)$
            - Covariance prediction: $P_{k+1|k} = F_k P_{k|k} F_k^T + Q$
            
        2.  **Update (Measurement Update):** 
            - Kalman gain: $K_{k+1} = P_{k+1|k} H_{k+1}^T (H_{k+1} P_{k+1|k} H_{k+1}^T + R)^{-1}$
            - State update: $\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} + K_{k+1} (z_{k+1} - h(\hat{x}_{k+1|k}))$
            - Covariance update: $P_{k+1|k+1} = (I - K_{k+1} H_{k+1}) P_{k+1|k}$
            
        * $F_k = \left. \frac{\partial f}{\partial x} \right|_{\hat{x}_{k|k}}$: Jacobian of state transition function
        * $H_{k+1} = \left. \frac{\partial h}{\partial x} \right|_{\hat{x}_{k+1|k}}$: Jacobian of measurement function
        
        **Applications in bioprocesses:**
        - Real-time estimation of unmeasured biomass concentration
        - Simultaneous state and parameter estimation
        - Soft-sensor development
        - Fault detection and diagnosis
        
        **References:**
        - Jazwinski, A. H. (1970). *Stochastic Processes and Filtering Theory*. Academic Press.
        - Simon, D. (2006). *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*. John Wiley & Sons.
        - Bastin, G., & Dochain, D. (1990). *On-line Estimation and Adaptive Control of Bioreactors*. Elsevier.
        """)
    st.markdown("---")
    st.subheader("🔹 RTO Control (Real-Time Optimization)")
    st.markdown(r"""
        A strategy that optimizes an **economic objective function** by adjusting setpoints or manipulated variables, based on a (often steady-state) model 
        and measurements. It operates on a slow time scale (minutes to hours) compared to regulatory control (**Marlin, 2000; Biegler, 2010**).
        
        RTO is a **two-layer control strategy** where:
        - **Upper layer (RTO):** Solves an economic optimization problem to find optimal setpoints
        - **Lower layer (Regulatory control):** PID or MPC controllers track the RTO setpoints
        
        **Optimization Problem:**
        $$\max_{u_{opt}} \Phi(x_{ss}, u_{opt}, p)$$
        subject to:
        - $f(x_{ss}, u_{opt}, p) = 0$ (steady-state model equations)
        - $g(x_{ss}, u_{opt}, p) \le 0$ (process constraints)
        - $u_{min} \le u_{opt} \le u_{max}$ (manipulated variable bounds)
        
        Where:
        - $\Phi$: Economic objective function (e.g., maximize productivity, minimize cost)
        - $x_{ss}$: Steady-state values of process variables
        - $u_{opt}$: Optimal manipulated variables (e.g., feed rate, temperature setpoint)
        - $p$: Model parameters
        
        **Applications in bioprocesses:**
        - Maximizing product concentration in fed-batch fermentation
        - Minimizing substrate consumption while maintaining productivity
        - Optimizing substrate feed rate to avoid overflow metabolism
        
        **Implementation considerations:**
        - Model-plant mismatch requires periodic re-optimization
        - Gradient-based methods (e.g., IPOPT solver) for efficiency
        - Constraint handling crucial for safe operation
        
        **References:**
        - Marlin, T. E. (2000). *Process Control: Designing Processes and Control Systems for Dynamic Performance* (2nd ed.). McGraw-Hill.
        - Biegler, L. T. (2010). *Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes*. SIAM.
        """)
    st.markdown("---")
    st.subheader("🔹 NMPC Control (Nonlinear Model Predictive Control)")
    st.markdown(r"""
        Uses a **nonlinear dynamic model** to predict the future behavior of the process over a prediction horizon ($N_p$) and computes 
        optimal control actions over a control horizon ($N_c$) by minimizing an objective function $J$ subject to constraints 
        (**Camacho & Bordons, 2007; Rawlings et al., 2017**). 
        
        NMPC applies only the **first control action** (receding horizon principle) and repeats the optimization at each sampling time, 
        providing feedback to handle disturbances and model uncertainty.
        
        **Optimization Problem (at each time step $k$):**
        $$\min_{\Delta U_k} J = \sum_{j=1}^{N_p} ||\hat{y}_{k+j|k} - y_{sp, k+j}||^2_Q + \sum_{j=0}^{N_c-1} ||\Delta u_{k+j|k}||^2_R$$
        
        subject to:
        - **Dynamic model:** $x_{k+j+1|k} = f(x_{k+j|k}, u_{k+j|k})$ for $j = 0, ..., N_p-1$
        - **Output equation:** $\hat{y}_{k+j|k} = h(x_{k+j|k})$
        - **Input constraints:** $u_{min} \le u_{k+j|k} \le u_{max}$
        - **Input rate constraints:** $\Delta u_{min} \le \Delta u_{k+j|k} \le \Delta u_{max}$
        - **Output constraints:** $y_{min} \le \hat{y}_{k+j|k} \le y_{max}$
        
        Where:
        - $\hat{y}_{k+j|k}$: Predicted output at time $k+j$ given information at time $k$
        - $y_{sp, k+j}$: Output setpoint trajectory
        - $\Delta u_{k+j|k} = u_{k+j|k} - u_{k+j-1|k}$: Control increment
        - $\Delta U_k = [\Delta u_{k|k}, \Delta u_{k+1|k}, ..., \Delta u_{k+N_c-1|k}]^T$: Control increment vector
        - $Q$: Output error weighting matrix (emphasizes tracking)
        - $R$: Control increment weighting matrix (penalizes aggressive control action)
        - $N_p$: Prediction horizon (how far ahead to predict)
        - $N_c \le N_p$: Control horizon (degrees of freedom in optimization)
        
        **Key advantages of NMPC:**
        - Handles nonlinear process dynamics explicitly
        - Systematic constraint handling (hard and soft constraints)
        - Anticipatory control through prediction
        - Multi-variable control with trade-offs between objectives
        
        **Implementation with CasADi:**
        This project uses **CasADi** (**Andersson et al., 2019**) for efficient symbolic differentiation and **IPOPT** 
        for solving the nonlinear programming (NLP) problem at each sampling time.
        
        **Applications in bioprocesses:**
        - Fed-batch bioreactor control with substrate feeding optimization
        - Maintaining dissolved oxygen within tight bounds
        - Multi-objective control (productivity vs. product quality)
        - Handling input saturation and rate limits
        
        **Comparison with PID control:**
        - **PID:** Simple, robust, but limited to SISO and linear approximations
        - **NMPC:** Complex, requires model, but handles MIMO, nonlinearity, and constraints
        
        **References:**
        - Camacho, E. F., & Bordons, C. (2007). *Model Predictive Control* (2nd ed.). Springer-Verlag.
        - Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
        - Andersson, J. A. E., et al. (2019). "CasADi: a software framework for nonlinear optimization and optimal control." *Mathematical Programming Computation*, 11(1), 1-36.
        """)

# To be able to run this page individually if necessary
if __name__ == "__main__":
    import os
    # (Code to create dummy images without changes)
    if not os.path.exists("images"): os.makedirs("images")
    dummy_files = ["images/Batch.png", "images/fed_batch.png", "images/continous.png"]
    for f_path in dummy_files:
        if not os.path.exists(f_path):
            try:
                with open(f_path, 'w') as fp: pass
                print(f"Dummy file created: {f_path}")
            except Exception as e:
                print(f"Unable to create dummy file {f_path}: {e}")
                try:
                    from PIL import Image
                    img = Image.new('RGB', (60, 30), color = 'red'); img.save(f_path)
                    print(f"Placeholder image created: {f_path}")
                except ImportError: print("PIL not found, unable to create image.")
                except Exception as e_img: print(f"Error creating image {f_path}: {e_img}")
    home_page()