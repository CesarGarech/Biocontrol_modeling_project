def mu_monod(S, mumax, Ks):
    return mumax * S / (Ks + S)

def mu_sigmoidal(S, mumax, Ks, n):
    return mumax * S**n / (Ks**n + S**n)

def mu_completa(S, O2, P, mumax, Ks, KO, KP):
    return mumax * S / (Ks + S) * O2 / (KO + O2) * KP / (KP + P)

def mu_fermentacion(S, P, O2,
                           mumax_aerob, Ks_aerob, KO_aerob, # Params mu1 (aerobio)
                           mumax_anaerob, Ks_anaerob, KiS_anaerob, # Params mu2 (anaerobio) - Sustrato
                           KP_anaerob, n_p,                 # Params mu2 (anaerobio) - Producto
                           KO_inhib_anaerob):              # Params mu2 (anaerobio) - O2 (Inhibición)
    """
    Calcula la tasa de crecimiento específica (mu) como la suma de un componente
    aeróbico (mu1) y uno anaeróbico/fermentativo (mu2).
    mu = mu1 + mu2

    Parámetros:
        S, P, O2: Concentraciones actuales [g/L, g/L, mg/L]
        *_aerob: Parámetros para mu1 (Monod simple S y O2)
        *_anaerob: Parámetros para mu2 (Haldane S, Inhibición P, Inhibición O2)
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
    Calcula la tasa de crecimiento específica (mu) como la suma de un componente
    aeróbico (mu1) y uno anaeróbico/fermentativo (mu2).
    **Versión compatible con CasADi.**

    Parámetros:
        S, P, O2: Variables simbólicas o numéricas [g/L, g/L, mg/L]
        ... resto de parámetros (numéricos)
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
