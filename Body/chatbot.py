import streamlit as st
import os


def chatbot_page():
    """
    AI Assistant for bioprocess modeling and control.

    Capabilities:
    - Explain kinetic models with LaTeX equations
    - Describe numerical methods (orthogonal collocation, optimization)
    - Provide links to code and references
    - Answer questions about typical parameters
    """

    st.header("🤖 Bioprocess AI Assistant")
    st.markdown("""
    Ask me about:
    - 🔬 **Kinetic Models** (Monod, Haldane, mixed aerobic/anaerobic fermentation)
    - 🧮 **Numerical Methods** (Runge-Kutta, orthogonal collocation, optimization)
    - 📊 **Advanced Control** (NMPC, RTO, EKF)
    - 📚 **Bibliographic References**
    - 💻 **Code Implementations**
    """)

    # Configure OpenAI client
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.warning("⚠️ Configure your OpenAI API key in `.streamlit/secrets.toml`")
        st.code(
            '# .streamlit/secrets.toml\nOPENAI_API_KEY = "sk-..."',
            language="toml",
        )
        st.info(
            "Get your API key at https://platform.openai.com/api-keys and add it to "
            "`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`)."
        )
        return

    try:
        from openai import OpenAI
    except ImportError:
        st.error(
            "The `openai` package is not installed. "
            "Run `pip install openai>=1.12.0` and restart the app."
        )
        return

    client = OpenAI(api_key=api_key)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Button to clear chat history
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chatbot_response(client, st.session_state.messages)
                st.markdown(response)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})


def get_chatbot_response(client, messages):
    """
    Generates a chatbot response with enriched project context.

    Parameters
    ----------
    client : openai.OpenAI
        Configured OpenAI client.
    messages : list[dict]
        Conversation history (role / content pairs).

    Returns
    -------
    str
        Assistant reply text.
    """

    system_prompt = """
You are an expert assistant in bioprocess modeling and control, specialised in teaching
biochemical engineering.

## PROJECT CONTEXT: Biocontrol Modeling Project

### Implemented Kinetic Models

1. **Simple Monod** (Utils/kinetics.py lines 1-38)
   Equation: μ = μ_max * S / (K_S + S)
   - μ_max: Maximum specific growth rate [1/h]
   - K_S: Saturation constant [g/L]
   - Application: Single-substrate-limited growth
   - References: Monod (1949), Bailey & Ollis (1986) Ch. 7

2. **Sigmoidal Monod / Hill** (Utils/kinetics.py lines 40-81)
   Equation: μ = μ_max * S^n / (K_S^n + S^n)
   - n: Hill coefficient (cooperativity)
   - Application: Systems with enzymatic induction
   - References: Hill (1910), Shuler & Kargi (2002)

3. **Complete Model** (Utils/kinetics.py lines 83-129)
   Equation: μ = μ_max * (S/(K_S+S)) * (O2/(K_O+O2)) * (K_P/(K_P+P))
   - Dual limitation: substrate and oxygen
   - Non-competitive product inhibition
   - Application: Aerobic bioprocesses with product formation

4. **Haldane/Aiba** (Utils/kinetics.py lines 131-168)
   Equation: μ = μ_max * S / (K_S + S + S²/K_iS)
   - Substrate inhibition at high concentrations
   - Application: Phenol/ethanol degradation
   - References: Andrews (1968), Haldane (1930)

5. **Mixed Aerobic/Anaerobic Fermentation** (Utils/kinetics.py lines 170-282)
   Equation: μ_total = μ_aerobic + μ_anaerobic

   μ_aerobic = μ_max_aer * (S/(K_S_aer+S)) * (O2/(K_O_aer+O2))

   μ_anaerobic = μ_max_anaer * (S/(K_S_anaer+S+S²/K_iS_anaer)) *
                  (1-P/K_P_anaer)^n_p * (K_O_inhib/(K_O_inhib+O2))

   - Dual model for Saccharomyces cerevisiae
   - Pasteur effect: O2 inhibits fermentation
   - Ethanol inhibition: (1-P/K_P)^n_p
   - Application: Alcoholic fermentation, bioethanol production
   - References: Bailey & Ollis (1986), Luedeking & Piret (1959)

### Numerical Methods Used

#### 1. ODE Integration (scipy.integrate.solve_ivp)

**RK45 – Runge-Kutta order 4-5**
- Explicit adaptive method
- Automatic step-size control
- Used in: Body/modeling/lote.py, lote_alimentado.py, continuo.py
- Reference: Press et al. (2007) "Numerical Recipes"

**Radau (for stiff systems)**
- Implicit Runge-Kutta method
- Stable for stiff systems (fast kinetics)
- Used when RK45 diverges or is too slow
- Reference: Hairer & Wanner (1996)

#### 2. Optimization

**Levenberg-Marquardt** (scipy.optimize.least_squares)
- Non-linear parameter fitting
- Least-squares with trust-region
- Used in: Body/parameter_estimation/
- Reference: Nocedal & Wright (2006)

**SQP – Sequential Quadratic Programming** (scipy.optimize.minimize)
- Optimisation with non-linear constraints
- Local quadratic approximation of the problem
- Reference: Nocedal & Wright (2006) Ch. 18

**IPOPT – Interior Point Optimizer** (via CasADi)
- Large-scale non-linear optimisation
- Used in: NMPC, RTO (Body/control/avanzado/)
- Interior-point method with line search
- Reference: Wächter & Biegler (2006)

#### 3. Orthogonal Collocation (CasADi)

**Implementation in NMPC** (Body/control/avanzado/nmpc.py)
- Discretisation of the optimal control problem
- Radau IIA polynomials (collocation points)
- Converts continuous control problem into NLP (Non-linear Programming)
- Prediction horizon: N_p steps
- Control horizon: N_c steps
- Method: Direct transcription

Process:
1. State x(t) approximated by Lagrange polynomials
2. ODEs evaluated at Radau collocation points
3. Algebraic constraints at each interval
4. Optimisation variables: x_k, u_k at each node

Reference:
- Biegler (2010) "Nonlinear Programming"
- Andersson et al. (2019) "CasADi framework"

#### 4. Automatic Differentiation (CasADi)

**Automatic Differentiation (AD)**
- Exact computation of gradients and Jacobians
- Used in: NMPC, RTO, EKF
- Types: Forward mode and Reverse mode (backpropagation)
- Implementation: Utils/kinetics.py *_rto() functions
- Advantages: Computational efficiency, numerical accuracy

#### 5. Extended Kalman Filter (EKF)

**Local linearisation** (Body/estimation/ekf.py)
- Jacobian F_k = ∂f/∂x evaluated at each step
- Jacobian H_k = ∂h/∂x of the measurement equation
- Prediction: x̂_{k+1|k} = f(x̂_{k|k}, u_k)
- Update: x̂_{k+1|k+1} = x̂_{k+1|k} + K_{k+1}(z_{k+1} - h(x̂_{k+1|k}))
- Kalman gain: K = P H^T (H P H^T + R)^{-1}
- Reference: Simon (2006), Jazwinski (1970)

### Mass Balances (ODEs)

**Batch mode**:
- dX/dt = μX - k_d X
- dS/dt = -(μ/Y_XS + m_s)X
- dP/dt = q_P X
- dO2/dt = k_L a(C_S* - O2) - (μ/Y_XO + m_o)X

**Fed-Batch mode**:
- dX/dt = μX - k_d X - (F/V)X
- dS/dt = -(μ/Y_XS + m_s)X + (F/V)(S_in - S)
- dP/dt = q_P X - (F/V)P
- dO2/dt = k_L a(C_S* - O2) - (μ/Y_XO + m_o)X - (F/V)O2
- dV/dt = F

**Continuous mode (Chemostat)**:
- D = F/V (dilution rate)
- dX/dt = (μ - k_d - D)X
- dS/dt = D(S_in - S) - (μ/Y_XS + m_s)X
- dP/dt = q_P X - DP
- dO2/dt = k_L a(C_S* - O2) - (μ/Y_XO + m_o)X - D O2

References: Bailey & Ollis (1986) Ch. 6, Shuler & Kargi (2002) Ch. 9

### Key Bibliographic References

**Bioengineering:**
- Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals (2nd ed.). McGraw-Hill.
- Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall.
- Luedeking, R., & Piret, E. L. (1959). J. Biochem. Microbiol. Technol. Eng., 1(4), 393-412.

**Process Control:**
- Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic Process Control (3rd ed.). Wiley.
- Camacho, E. F., & Bordons, C. (2007). Model Predictive Control (2nd ed.). Springer.
- Rawlings, J. B., et al. (2017). Model Predictive Control: Theory, Computation, and Design (2nd ed.).

**Optimisation:**
- Biegler, L. T. (2010). Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM.
- Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer.
- Andersson, J. A. E., et al. (2019). Math. Program. Comput., 11(1), 1-36. [CasADi]

**Numerical Methods:**
- Press, W. H., et al. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge.
- Hairer, E., & Wanner, G. (1996). Solving Ordinary Differential Equations II: Stiff Problems. Springer.

**State Estimation:**
- Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches. Wiley.
- Bastin, G., & Dochain, D. (1990). On-line Estimation and Adaptive Control of Bioreactors. Elsevier.

## RESPONSE INSTRUCTIONS

When the user asks about:

1. **Kinetic models**:
   - Explain the equation in LaTeX format
   - Define each parameter with units
   - Mention practical applications
   - Provide a link to the code: https://github.com/CesarGarech/Biocontrol_modeling_project/blob/main/Utils/kinetics.py#L[line]
   - Cite relevant references

2. **Numerical methods**:
   - Explain the mathematical foundation
   - Describe when and why it is used
   - Mention advantages/limitations
   - Indicate where it is implemented in the code
   - Provide bibliographic references

3. **Parameters**:
   - Give typical ranges with sources
   - Explain the physical/biological meaning
   - Suggest how to adjust them (Parameter Estimation module)

4. **Implementations**:
   - Generate direct links to GitHub
   - Explain the code flow
   - Relate to theory

**RESPONSE FORMAT:**
- Use Markdown with LaTeX for equations: $\\mu_{max}$
- Include emojis for clarity: 🔬 📊 📚 💡 ⚠️
- Generate clickable links
- Be concise but complete
- Use educational but technical language

**BASE REPOSITORY:**
https://github.com/CesarGarech/Biocontrol_modeling_project
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    return response.choices[0].message.content
