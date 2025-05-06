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
        The specific growth rate ($\mu$, unit $h^{-1}$) describes how fast the biomass increases per unit of 
        existing biomass. It depends on factors such as substrate ($S$), product ($P$) and dissolved oxygen ($O_2$)
        concentration. The implemented models are:
        """)

    with st.expander("1. Simple Monod"):
        st.markdown("The most basic model, assumes that growth is limited by a single substrate. ($S$).")
        st.latex(r"""
        \mu = \mu_{max} \frac{S}{K_S + S}
        """)
        st.markdown(r"""
            * $\mu_{max}$: Maximum specific growth rate ($h^{-1}$).
            * $K_S$: Substrate affinity constant (concentration of $S$ at which $\mu = \mu_{max}/2$, units $g/L$).
            """)

    with st.expander("2. Sigmoidal Monod (Hill)"):
        st.markdown("It introduces a threshold or cooperative response to the substrate, useful for modeling induction phenomena or more complex kinetics.")
        st.latex(r"""
        \mu = \mu_{max} \frac{S^n}{K_S^n + S^n}
        """)
        st.markdown(r"""
            * $n$: Hill coefficient (dimensionless), fits sigmoidal shape. $n > 1$ indicates cooperativity.
            """)

    with st.expander("3. Monod with Constraints (S, O2, P)"):
        st.markdown("Models growth simultaneously limited by substrate and oxygen, and inhibited by product.")
        st.latex(r"""
        \mu = \mu_{max} \left(\frac{S}{K_S + S}\right) \left(\frac{O_2}{K_O + O_2}\right) \left(\frac{K_P}{K_P + P}\right)
        """)
        st.markdown(r"""
            * $K_O$: Oxygen affinity constant ($g/L$).
            * $K_P$: Inhibition constant per product ($g/L$). *Note: This form is a simple non-competitive inhibition.*
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
        The specific product formation rate ($q_P$, unit $g_P \cdot g_X^{-1} \cdot h^{-1}$), specifically for ethanol in this context, is modeled using the Luedeking-Piret equation modified to include direct inhibition by oxygen. This reflects that ethanol production is predominantly anaerobic.
        """)
    st.latex(r"""
    q_P = (\alpha \cdot \mu + \beta) \left( \frac{K_{O,P}}{K_{O,P} + O_2} \right)
    """)
    st.markdown(r"""
        * $\mu$: It is the specific growth rate calculated by the selected kinetic model. ($\mu_{total}$ if Mixed/Switched).
        * $\alpha$: Growth-associated product formation coefficient ($g_P \cdot g_X^{-1}$).
        * $\beta$: Non-growth-associated product formation coefficient ($g_P \cdot g_X^{-1} \cdot h^{-1}$).
        * $K_{O,P}$: Oxygen inhibition constant on ethanol *production* ($g/L$). A low value indicates strong suppression of $P$ production by $O_2$.
        """)

    st.markdown("---") # Visual separator

    # ========= MOVED SECTION: BALANCE SHEETS =========
    st.header("📊 Theoretical Basis: Material Balances") # Adjusted title
    st.markdown("""
        Bioprocess modeling allows to describe mathematically the evolution of the variables of interest 
        (biomass concentration, substrate, product, dissolved oxygen, etc.) in a bioreactor.
        The general material balances for the three main modes of operation are presented below, assuming 
        a perfect mixing of the three main modes of operation. The following are the general matter balances 
        for the three main modes of operation, assuming perfect mixing. The rates $mu$ and $q_P$ correspond 
        to the kinetic models described above.
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
    st.header("⚙️ Advanced Analysis and Control Techniques")

    # (The Sensitivity, Adjustment, EKF, RTO, NMPC subsections are not modified.)
    st.subheader("🔹 Sensitivity analysis")
    st.markdown(r"""
        Evaluates how uncertainty or variations in the model parameters ($theta$, such as $$mu_{max}, K_S, Y_{XS}$, etc.) 
        affect the model outputs (the state variables $X, S, P, O_2$). It allows identifying the most influential parameters, 
        crucial for optimization and experimental design. A common metric is the normalized sensitivity coefficient:
        $S_{ij} = \frac{\partial y_i / y_i}{\partial \theta_j / \theta_j} = \frac{\partial \ln y_i}{\partial \ln \theta_j}$
        where $y_i$ is an output $\theta_j$ is a parameter.
        """)
    st.markdown("---")
    st.subheader("🔹 Parameter Adjustment (Estimation)")
    st.markdown(r"""
        The process of finding the values of the model parameters ($\theta$) that best describe a set of experimental data ($y_{exp}$). 
        An objective function $J(\theta)$ that measures the discrepancy between the model predictions ($y_{model}$) and the data is minimized.
        The optimization problem is:
        """)
    st.latex(r"""
    \hat{\theta} = \arg \min_{\theta} J(\theta)
    """)
    st.markdown(r"""
        A common objective function is the sum of weighted squared errors:
        $J(\theta) = \sum_{k=1}^{N} \sum_{i=1}^{M} w_{ik} (y_{i,exp}(t_k) - y_{i,model}(t_k, \theta))^2$
        where $N$ is the number of sampling point, $M$ the number of measured variables, and $w_{ik}$ are weights.
        Optimization algorithms (Levenberg-Marquardt, SQP, genetics, etc.) are used.
        """)
    st.markdown("---")
    st.subheader("🔹 Extended Kalman Filter (EKF)")
    st.markdown(r"""
        Recursive algorithm for estimating the state of nonlinear dynamic systems in the presence of noise. Uses a model of the system and 
        noisy measurements to obtain an optimal (in the sense of minimum variance) estimate of the state. Essential when not all state variables 
        (e.g., biomass) can be measured directly online.
        **System model (discrete):**
        """)
    st.latex(r"x_{k+1} = f(x_k, u_k) + w_k \quad \text{(Process equation)}")
    st.latex(r"z_k = h(x_k) + v_k \quad \text{(Measurement equation)}")
    st.markdown(r"""
        * $x_k, u_k, z_k$: State, input and measurement in $k$.
        * $w_k, v_k$: Process noise ($Q$) and measurement ($R$).
        * $f, h$: Nonlinear functions.
        **EKF Stages:**
        1.  **Prediction:** $\hat{x}_{k+1|k} = f(\hat{x}_{k|k}, u_k)$, $P_{k+1|k} = F_k P_{k|k} F_k^T + Q$.
        2.  **Update:** $K_{k+1} = P_{k+1|k} H_{k+1}^T (H_{k+1} P_{k+1|k} H_{k+1}^T + R)^{-1}$, $\hat{x}_{k+1|k+1} = \hat{x}_{k+1|k} + K_{k+1} (z_{k+1} - h(\hat{x}_{k+1|k}))$, $P_{k+1|k+1} = (I - K_{k+1} H_{k+1}) P_{k+1|k}$.
        * $F_k, H_{k+1}$: Jacobians of $f$ and $h$.
        """)
    st.markdown("---")
    st.subheader("🔹 RTO Control (Real-Time Optimization)")
    st.markdown(r"""
        A strategy that optimizes an economic objective function by adjusting setpoints or manipulated variables, based on a (often stationary) model 
        and measurements. It operates on a slow time scale.
        **Problem:** $\max_{u_{opt}} \Phi(x_{ss}, u_{opt}, p)$ subject to $f(x_{ss}, u_{opt}, p) = 0$, $g(x_{ss}, u_{opt}, p) \le 0$, $u_{min} \le u_{opt} \le u_{max}$.
        """)
    st.markdown("---")
    st.subheader("🔹 NMPC Control (Nonlinear Model Predictive Control)")
    st.markdown(r"""
        Use a nonlinear dynamic model to predict the future ($N_p$) and compute optimal control actions ($$Delta U$ on $N_c$) by minimizing an objective 
        function $J$ subject to constraints. Apply only the first action and repeat.
        **Problem:** $\min_{\Delta U_k} J = \sum_{j=1}^{N_p} ||\hat{y}_{k+j|k} - y_{sp, k+j}||^2_Q + \sum_{j=0}^{N_c-1} ||\Delta u_{k+j|k}||^2_R$ subject to the dynamic model and constraints in $u, \Delta u, y$.
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