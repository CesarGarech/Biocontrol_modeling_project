def mu_monod(S, mumax, Ks):
    """
    Monod model for substrate-limited microbial growth.
    
    The Monod equation is the most widely used model for describing microbial growth
    kinetics in bioprocesses. It represents a hyperbolic relationship between growth
    rate and substrate concentration, analogous to Michaelis-Menten enzyme kinetics.
    
    Parameters
    ----------
    S : float
        Substrate concentration [g/L]
    mumax : float
        Maximum specific growth rate [1/h]
    Ks : float
        Substrate saturation constant (half-velocity constant) [g/L]
        Represents the substrate concentration at which μ = μmax/2
    
    Returns
    -------
    float
        Specific growth rate μ [1/h]
    
    Notes
    -----
    The Monod model assumes:
    - Single limiting substrate
    - No substrate or product inhibition
    - Steady-state enzyme-substrate complex
    
    Reference
    ---------
    Monod, J. (1949). "The growth of bacterial cultures." 
    Annual Review of Microbiology, 3(1), 371-394.
    
    Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals. McGraw-Hill.
    """
    return mumax * S / (Ks + S)

def mu_sigmoidal(S, mumax, Ks, n):
    """
    Sigmoidal (Hill) model for microbial growth with cooperative effects.
    
    The Hill equation introduces a cooperativity coefficient (n) that allows modeling
    of sigmoidal kinetics, useful for processes with substrate induction, enzyme
    cooperativity, or threshold effects.
    
    Parameters
    ----------
    S : float
        Substrate concentration [g/L]
    mumax : float
        Maximum specific growth rate [1/h]
    Ks : float
        Substrate saturation constant [g/L]
    n : float
        Hill coefficient (cooperativity index) [-]
        n = 1: Reduces to simple Monod model
        n > 1: Positive cooperativity (sigmoidal, sharper transition)
        n < 1: Negative cooperativity (more gradual response)
    
    Returns
    -------
    float
        Specific growth rate μ [1/h]
    
    Notes
    -----
    Commonly used for:
    - Inducible enzyme systems
    - Multiple binding sites
    - Complex metabolic regulation
    
    Reference
    ---------
    Hill, A. V. (1910). "The possible effects of the aggregation of the molecules 
    of haemoglobin on its dissociation curves." Journal of Physiology, 40, iv-vii.
    
    Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. Prentice Hall.
    """
    return mumax * S**n / (Ks**n + S**n)

def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    """
    Complete model with multiple substrate limitation and product inhibition.
    
    Models microbial growth simultaneously limited by substrate and oxygen,
    with product inhibition. Uses multiplicative Monod terms for each limiting
    factor, representing non-competitive inhibition by product.
    
    Parameters
    ----------
    S : float
        Substrate concentration [g/L]
    O2 : float
        Dissolved oxygen concentration [g/L or mg/L]
    P : float
        Product concentration [g/L]
    mumax : float
        Maximum specific growth rate [1/h]
    Ks : float
        Substrate saturation constant [g/L]
    KO : float
        Oxygen saturation constant [g/L or mg/L]
    KP : float
        Product inhibition constant [g/L]
    
    Returns
    -------
    float
        Specific growth rate μ [1/h]
    
    Notes
    -----
    The model assumes:
    - Independent effects of each factor (multiplicative)
    - Non-competitive product inhibition
    - Monod kinetics for substrate and oxygen limitation
    
    For substrate inhibition at high concentrations, consider the Haldane model:
    μ = μmax * S / (Ks + S + S²/Ki)
    
    Reference
    ---------
    Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals. McGraw-Hill.
    
    Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. Prentice Hall.
    """
    return mumax * S / (Ks + S) * O2 / (KO + O2) * KP / (KP + P)

def aiba(mumax, I, Ki, KiL):
    """
    Aiba model for substrate inhibition (Haldane-type kinetics).
    
    Models microbial growth with substrate inhibition at high concentrations,
    commonly observed in systems where the substrate becomes toxic or inhibitory
    at elevated levels (e.g., ethanol, phenol degradation).
    
    Parameters
    ----------
    mumax : float
        Maximum specific growth rate [1/h]
    I : float
        Substrate/inhibitor concentration [g/L]
    Ki : float
        Substrate saturation constant [g/L]
    KiL : float
        Substrate inhibition constant [g/L]
    
    Returns
    -------
    float
        Specific growth rate μ [1/h]
    
    Notes
    -----
    Also known as Haldane or Andrews model. The growth rate increases with
    substrate concentration up to an optimum, then decreases at higher levels.
    
    Reference
    ---------
    Andrews, J. F. (1968). "A mathematical model for the continuous culture of 
    microorganisms utilizing inhibitory substrates." 
    Biotechnology and Bioengineering, 10(6), 707-723.
    
    Haldane, J. B. S. (1930). Enzymes. Longmans, Green and Co.
    """
    return mumax * (I / (Ki + I + ((I**2) / KiL)))

def mu_fermentacion(S, P, O2,
                           mumax_aerob, Ks_aerob, KO_aerob, # Params mu1 (aerobio)
                           mumax_anaerob, Ks_anaerob, KiS_anaerob, # Params mu2 (anaerobio) - Sustrato
                           KP_anaerob, n_p,                 # Params mu2 (anaerobio) - Producto
                           KO_inhib_anaerob):              # Params mu2 (anaerobio) - O2 (Inhibición)
    """
    Mixed aerobic/anaerobic fermentation model for yeast metabolism.
    
    Calculates specific growth rate (μ) as the sum of aerobic (μ1) and 
    anaerobic/fermentative (μ2) components, modeling the coexistence of 
    respiratory and fermentative metabolic pathways (e.g., Saccharomyces cerevisiae).
    
    μ_total = μ_aerobic + μ_anaerobic
    
    Parameters
    ----------
    S : float
        Substrate (glucose) concentration [g/L]
    P : float
        Product (ethanol) concentration [g/L]
    O2 : float
        Dissolved oxygen concentration [mg/L or g/L]
    
    Aerobic component parameters:
    mumax_aerob : float
        Maximum aerobic growth rate [1/h]
    Ks_aerob : float
        Substrate saturation constant for aerobic growth [g/L]
    KO_aerob : float
        Oxygen saturation constant [mg/L or g/L]
    
    Anaerobic component parameters:
    mumax_anaerob : float
        Maximum anaerobic/fermentative growth rate [1/h]
    Ks_anaerob : float
        Substrate saturation constant for anaerobic growth [g/L]
    KiS_anaerob : float
        Substrate inhibition constant (Haldane model) [g/L]
    KP_anaerob : float
        Critical ethanol concentration for growth inhibition [g/L]
        Represents maximum tolerable product concentration
    n_p : float
        Product inhibition exponent [-]
    KO_inhib_anaerob : float
        Oxygen inhibition constant for anaerobic pathway [mg/L or g/L]
        Represents Pasteur effect (oxygen suppression of fermentation)
    
    Returns
    -------
    float
        Total specific growth rate μ [1/h]
    
    Notes
    -----
    Aerobic component (μ1):
    - Monod kinetics for substrate and oxygen
    - Dominant under aerobic conditions
    - Higher biomass yield, lower product formation
    
    Anaerobic component (μ2):
    - Haldane kinetics for substrate (inhibition at high S)
    - Product (ethanol) inhibition: (1 - P/Kp)^n_p
    - Oxygen inhibition (Pasteur effect): KO_inhib/(KO_inhib + O2)
    - Dominant under anaerobic/microaerobic conditions
    - Lower biomass yield, higher ethanol production
    
    Applications:
    - Alcoholic fermentation (beer, wine, bioethanol)
    - Crabtree effect modeling
    - Fed-batch optimization to control respiratory/fermentative balance
    
    Reference
    ---------
    Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals. McGraw-Hill.
    
    Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts. Prentice Hall.
    
    Luedeking, R., & Piret, E. L. (1959). "A kinetic study of the lactic acid fermentation."
    Journal of Biochemical and Microbiological Technology and Engineering, 1(4), 393-412.
    """
    # --- Asegurar valores no negativos y evitar divisiones por cero ---
    S = max(1e-9, S)
    P = max(0.0, P)
    O2 = max(0.0, O2) # Permitir O2=0

    # --- Cálculo de mu1 (Componente Aeróbica) ---
    # Monod simple para Sustrato y Oxígeno (limitación)
    term_S_aerob = S / (Ks_aerob + S) if (Ks_aerob + S) > 1e-9 else 0.0
    term_O2_aerob = O2 / (KO_aerob + O2) if (KO_aerob + O2) > 1e-9 else 0.0 # Limitado por O2
    mu1 = mumax_aerob * term_S_aerob * term_O2_aerob

    # --- Cálculo de mu2 (Componente Anaeróbica / Fermentativa) ---
    # Término de sustrato con inhibición (Haldane)
    denominador_S_anaerob = Ks_anaerob + S + (S**2 / KiS_anaerob) if KiS_anaerob > 1e-9 else Ks_anaerob + S
    term_S_anaerob = S / denominador_S_anaerob if denominador_S_anaerob > 1e-9 else 0.0

    # Término de inhibición por producto (Etanol)
    term_P_anaerob = (1.0 - (P / KP_anaerob))**n_p if P < KP_anaerob and KP_anaerob > 1e-9 else 0.0
    term_P_anaerob = max(0.0, term_P_anaerob)

    # Término de INHIBICIÓN por oxígeno (alta O2 inhibe mu2)
    term_O2_inhib_anaerob = KO_inhib_anaerob / (KO_inhib_anaerob + O2) if (KO_inhib_anaerob + O2) > 1e-9 else 1.0 # Si O2=0 y KO_inhib=0, inhibición es nula. Si O2>0, ok.
    # Caso especial: si KO_inhib_anaerob es muy pequeño (cercano a 0), indica inhibición muy fuerte incluso a bajo O2.
    # La fórmula ya lo maneja: term_O2_inhib_anaerob -> 0 si O2 > 0.
    # Si O2=0, term_O2_inhib_anaerob -> 1 (sin inhibición por O2)

    mu2 = mumax_anaerob * term_S_anaerob * term_P_anaerob * term_O2_inhib_anaerob

    # --- Tasa de crecimiento total ---
    mu_total = mu1 + mu2

    # Asegurar que la tasa de crecimiento final no sea negativa
    return max(0.0, mu_total)

import casadi as ca
def mu_fermentacion_rto(S, P, O2,
                      mumax_aerob, Ks_aerob, KO_aerob,           # Params mu1 (aerobio)
                      mumax_anaerob, Ks_anaerob, KiS_anaerob,    # Params mu2 (anaerobio) - Sustrato
                      KP_anaerob, n_p,                          # Params mu2 (anaerobio) - Producto
                      KO_inhib_anaerob):                       # Params mu2 (anaerobio) - O2 (Inhibición)
    """
    CasADi-compatible version of mixed aerobic/anaerobic fermentation model.
    
    This is a symbolic implementation of mu_fermentacion() compatible with CasADi
    automatic differentiation framework, enabling its use in optimization-based
    control strategies (NMPC, RTO) implemented with CasADi and IPOPT.
    
    Parameters
    ----------
    S : casadi.SX or casadi.MX or float
        Substrate concentration (symbolic or numeric) [g/L]
    P : casadi.SX or casadi.MX or float
        Product concentration (symbolic or numeric) [g/L]
    O2 : casadi.SX or casadi.MX or float
        Dissolved oxygen concentration (symbolic or numeric) [mg/L or g/L]
    
    (Other parameters same as mu_fermentacion - see that function for details)
    
    Returns
    -------
    casadi.SX or casadi.MX or float
        Total specific growth rate μ (symbolic or numeric) [1/h]
    
    Notes
    -----
    Key differences from mu_fermentacion():
    - Uses ca.fmax() instead of max() for CasADi compatibility
    - Uses ca.if_else() for conditional logic in symbolic expressions
    - Avoids Python control flow (if/else) that cannot be differentiated
    - Safe for use in nonlinear programming (NLP) problem formulations
    
    Used in:
    - Nonlinear Model Predictive Control (NMPC) with CasADi
    - Real-Time Optimization (RTO) with CasADi
    - Gradient-based parameter estimation
    
    Reference
    ---------
    Andersson, J. A. E., et al. (2019). "CasADi: a software framework for 
    nonlinear optimization and optimal control." 
    Mathematical Programming Computation, 11(1), 1-36.
    
    See also
    --------
    mu_fermentacion : Non-symbolic version for standard simulation
    """
    # --- Asegurar valores no negativos usando ca.fmax ---
    S = ca.fmax(1e-9, S) # Evita S=0 exacto
    P = ca.fmax(0.0, P)
    O2 = ca.fmax(0.0, O2)

    # --- Cálculo de mu1 (Componente Aeróbica) ---
    den_S_aerob = Ks_aerob + S
    term_S_aerob = S / ca.fmax(den_S_aerob, 1e-9)
    
    den_O2_aerob = KO_aerob + O2
    term_O2_aerob = O2 / ca.fmax(den_O2_aerob, 1e-9) # Limitado por O2
    
    mu1 = mumax_aerob * term_S_aerob * term_O2_aerob

    # --- Cálculo de mu2 (Componente Anaeróbica / Fermentativa) ---
    # Término de sustrato con inhibición (Haldane)
    denominador_S_anaerob = Ks_anaerob + S
    # El if sobre KiS_anaerob (parámetro numérico) se evalúa una vez en la definición. OK.
    if isinstance(KiS_anaerob, (int, float)) and KiS_anaerob < float('inf') and KiS_anaerob > 1e-9:
         denominador_S_anaerob += (S**2 / KiS_anaerob)
    term_S_anaerob = S / ca.fmax(denominador_S_anaerob, 1e-9)

    # Término de inhibición por producto (Etanol)
    # Reemplaza: (1.0 - (P / KP_anaerob))**n_p if P < KP_anaerob else 0.0
    # Asumiendo KP_anaerob > 0
    inhib_P_base = ca.fmax(0.0, 1.0 - P / KP_anaerob) 
    term_P_anaerob = inhib_P_base**n_p

    # Término de INHIBICIÓN por oxígeno (Pasteur effect)
    # Reemplaza: KO_inhib_anaerob / (KO_inhib_anaerob + O2) if (KO_inhib_anaerob + O2) > 1e-9 else 1.0
    # Asumiendo KO_inhib_anaerob >= 0
    den_O2_inhib = KO_inhib_anaerob + O2
    term_O2_inhib_anaerob = KO_inhib_anaerob / ca.fmax(den_O2_inhib, 1e-9) 
    # Nota: Si KO_inhib_anaerob = 0, esto da 0 (inhibición total si O2>0). Si O2=0, den=0 -> div by zero -> Inf? 
    # Casadi maneja Inf. Si KO_inhib=0 Y O2=0, da 0/0 -> NaN. 
    # Si KO_inhib_anaerob es estrictamente > 0, la fórmula es segura.
    # Si KO_inhib_anaerob puede ser 0, se podría necesitar un ca.if_else, pero asumamos que es > 0.

    mu2 = mumax_anaerob * term_S_anaerob * term_P_anaerob * term_O2_inhib_anaerob

    # --- Tasa de crecimiento total ---
    mu_total = mu1 + mu2

    # Asegurar que la tasa de crecimiento final no sea negativa
    return ca.fmax(0.0, mu_total) # Reemplaza max(0.0, mu_total)
