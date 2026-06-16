# -*- coding: utf-8 -*-
"""
Knowledge Base for the Biocontrol AI Guide (Llama / Ollama assistant)
=====================================================================

This module is the single source of truth that "grounds" the Llama model so the
chatbot can answer questions and suggest parameters for *every* screen of the
application: modeling, parameter optimization, state estimation, regulatory and
advanced control, and digital twins.

Each entry of :data:`PAGE_KNOWLEDGE` is keyed by the exact ``selected_page``
string used by ``main.py`` (including emojis where relevant) and contains:

* ``section``     - high level area the page belongs to.
* ``method``      - main method / model implemented by the page.
* ``description`` - short educational summary of the page.
* ``equations``   - list of the governing equations shown on the page.
* ``parameters``  - dict ``name -> {min, max, default, unit, description}`` with
                    the typical ranges/defaults exposed in the page UI.
* ``references``  - list of keys into ``CURATED_REFERENCES`` (in ``llm_helper``).
* ``examples``    - example questions a user could ask on this page.

The data is consumed by ``llm_helper.build_context_prompt`` (to inject the page
context into the Llama prompt) and by ``llm_ui_component`` (for page-aware
parameter suggestions, including an offline fallback when Ollama is not
running).
"""

from typing import Any, Dict, List, Optional


def _p(default: Any, unit: str, description: str,
       minimum: Any = None, maximum: Any = None) -> Dict[str, Any]:
    """Small helper to build a parameter descriptor with a stable schema."""
    return {
        "min": minimum,
        "max": maximum,
        "default": default,
        "unit": unit,
        "description": description,
    }


# ===========================================================================
# PER-PAGE KNOWLEDGE BASE
# ===========================================================================
PAGE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    # ----------------------------------------------------------------- Home
    "🏠 Home": {
        "section": "Overview",
        "method": "Educational landing page and kinetic model reference",
        "description": (
            "Introduction to the simulator. Documents the kinetic models "
            "(Monod, Sigmoidal/Hill, multi-limited Monod, mixed aerobic/"
            "anaerobic fermentation), the mass balances for batch, fed-batch "
            "and continuous reactors, the Luedeking-Piret product model and "
            "PID control basics."
        ),
        "equations": [
            "Monod: mu = mumax * S / (Ks + S)",
            "Sigmoidal (Hill): mu = mumax * S^n / (Ks^n + S^n)",
            "Multi-limited: mu = mumax * S/(Ks+S) * O2/(KO+O2) * KP/(KP+P)",
            "Luedeking-Piret: qP = alpha*mu + beta",
            "PID: u(t) = Kc[e + (1/Ti) integral(e) dt + Td de/dt] + u_bias",
        ],
        "parameters": {},
        "references": ["bioprocess", "monod", "pid_control"],
        "examples": [
            "What kinetic models are available in this app?",
            "What is the difference between batch, fed-batch and continuous reactors?",
        ],
    },

    # =================================================== MODELING ==========
    "Batch": {
        "section": "Modeling",
        "method": "Batch bioreactor simulation (fixed volume)",
        "description": (
            "Simulates a batch fermentation with selectable kinetics (simple "
            "Monod, sigmoidal, or multi-limited with O2/product effects). "
            "Solves a 4-state ODE system for biomass (X), substrate (S), "
            "product (P) and dissolved oxygen (O2)."
        ),
        "equations": [
            "dX/dt = mu*X - Kd*X",
            "dS/dt = -(mu/Yxs)*X - ms*X",
            "dP/dt = Ypx*mu*X",
            "dO2/dt = kLa*(Cs - O2) - (mu/Yxo)*X - mo*X",
            "mu = mumax * S/(Ks + S)  (Monod option)",
        ],
        "parameters": {
            "mumax": _p(0.3, "1/h", "Maximum specific growth rate", 0.1, 1.0),
            "Ks": _p(0.1, "g/L", "Monod saturation constant", 0.01, 1.0),
            "Yxs": _p(0.5, "g/g", "Biomass/substrate yield", 0.1, 1.0),
            "Ypx": _p(0.3, "g/g", "Product/biomass yield", 0.1, 1.0),
            "Yxo": _p(0.3, "g/g", "Biomass/oxygen yield", 0.1, 1.0),
            "kLa": _p(20, "1/h", "Oxygen volumetric mass-transfer coefficient", 0.1, 100),
            "Cs": _p(8, "mg/L", "Saturated dissolved-oxygen concentration", 0.1, 10),
            "ms": _p(0.005, "g/g/h", "Substrate maintenance coefficient", 0.0, 0.5),
            "Kd": _p(0.005, "1/h", "Cell death/decay constant", 0.0, 0.5),
            "mo": _p(0.05, "g/g/h", "Oxygen maintenance coefficient", 0.0, 0.5),
            "X0": _p(0.5, "g/L", "Initial biomass", 0.1, 10),
            "S0": _p(20, "g/L", "Initial substrate", 0.1, 100),
        },
        "references": ["monod", "bioprocess"],
        "examples": [
            "What does Ks mean in the Monod equation?",
            "Why does biomass stop growing when substrate is depleted in a batch?",
        ],
    },

    "Fed-Batch": {
        "section": "Modeling",
        "method": "Fed-batch bioreactor simulation (variable volume)",
        "description": (
            "Extends the batch model with a feed stream and variable volume. "
            "Supports constant, exponential, step and linear feeding strategies "
            "and tracks the dilution effect of the feed on every state."
        ),
        "equations": [
            "dX/dt = (mu - Kd)*X - (F/V)*X",
            "dS/dt = -((mu/Yxs) + ms)*X + (F/V)*(Sin - S)",
            "dP/dt = Ypx*mu*X - (F/V)*P",
            "dO2/dt = kLa*(Cs - O2) - ((mu/Yxo) + mo)*X - (F/V)*O2",
            "dV/dt = F",
        ],
        "parameters": {
            "mumax": _p(0.4, "1/h", "Maximum specific growth rate", 0.1, 1.0),
            "Ks": _p(0.5, "g/L", "Monod saturation constant", 0.01, 2.0),
            "n": _p(2, "-", "Hill exponent (sigmoidal option)", 1, 5),
            "KO": _p(0.5, "mg/L", "Oxygen saturation constant", 0.1, 5),
            "KP": _p(5, "g/L", "Product inhibition constant", 0.1, 10),
            "kLa": _p(50, "1/h", "Oxygen volumetric mass-transfer coefficient", 1, 200),
            "Cs": _p(8, "mg/L", "Saturated dissolved-oxygen concentration", 5, 15),
            "Sin": _p(150, "g/L", "Substrate concentration in the feed", 50, 300),
            "F_base": _p(0.5, "L/h", "Base feed flow rate", 0.01, 5),
            "V0": _p(3, "L", "Initial volume", 1, 100),
            "X0": _p(1, "g/L", "Initial biomass", 0.1, 50),
        },
        "references": ["monod", "bioprocess"],
        "examples": [
            "Which feeding strategy avoids substrate inhibition in fed-batch?",
            "How does the feed rate F affect the dilution of biomass?",
        ],
    },

    "Continuous": {
        "section": "Modeling",
        "method": "Continuous culture (chemostat) simulation",
        "description": (
            "Models a continuous (chemostat) reactor where the dilution rate D "
            "sets the steady state and the washout condition. Useful to explore "
            "steady-state biomass/substrate vs. dilution rate."
        ),
        "equations": [
            "dX/dt = (mu - Kd - D)*X",
            "dS/dt = -(mu/Yxs)*X - ms*X + D*(Sin - S)",
            "dP/dt = Ypx*mu*X - D*P",
            "dO2/dt = kLa*(Cs - O2) - (mu/Yxo)*X - mo*X - D*O2",
            "At steady state (no Kd): mu = D",
        ],
        "parameters": {
            "mumax": _p(0.3, "1/h", "Maximum specific growth rate", 0.1, 1.0),
            "Ks": _p(0.1, "g/L", "Monod saturation constant", 0.01, 1.0),
            "Yxs": _p(0.5, "g/g", "Biomass/substrate yield", 0.1, 1.0),
            "kLa": _p(20, "1/h", "Oxygen volumetric mass-transfer coefficient", 0.1, 100),
            "Cs": _p(8, "mg/L", "Saturated dissolved-oxygen concentration", 0.1, 10),
            "Sin": _p(50, "g/L", "Substrate concentration in the feed", 0, 100),
            "D": _p(0.01, "1/h", "Dilution rate (F/V); washout if D > mumax", 0.0, 1.0),
        },
        "references": ["monod", "bioprocess"],
        "examples": [
            "What is the washout dilution rate of a chemostat?",
            "Why does mu equal D at steady state in a chemostat?",
        ],
    },

    "Fermentation": {
        "section": "Modeling",
        "method": "Multi-phase alcoholic fermentation (fed-batch)",
        "description": (
            "Three-phase alcoholic fermentation (batch -> fed-batch -> batch "
            "depletion) with mixed aerobic/anaerobic kinetics, the Pasteur "
            "effect (O2 inhibits fermentation) and Luedeking-Piret ethanol "
            "production."
        ),
        "equations": [
            "mu_aer = mumax_aer * S/(Ksa+S) * O2/(KOa+O2)",
            "mu_an = mumax_an * S/(Ksan+S+S^2/Kian) * (1-P/KPan)^np * KOinhib/(KOinhib+O2)",
            "qP = alpha*mu_an + beta   (Luedeking-Piret)",
        ],
        "parameters": {
            "mumax_aer": _p(0.4, "1/h", "Max aerobic specific growth rate", 0.1, 1.0),
            "Ks_aer": _p(0.5, "g/L", "Aerobic substrate saturation constant", 0.01, 10),
            "KO_aer": _p(0.2, "mg/L", "Aerobic oxygen saturation constant", 0.01, 5),
            "mumax_an": _p(0.15, "1/h", "Max anaerobic specific growth rate", 0.05, 0.8),
            "Ks_an": _p(1.0, "g/L", "Anaerobic substrate saturation constant", 0.1, 20),
            "Ki_an": _p(150, "g/L", "Substrate inhibition constant (anaerobic)", 50, 500),
            "KP_an": _p(80, "g/L", "Ethanol inhibition constant", 20, 150),
            "np": _p(1.0, "-", "Ethanol inhibition exponent", 0.5, 3.0),
            "alpha": _p(4.5, "gP/gX", "Growth-associated ethanol coefficient", 0.0, 10),
            "beta": _p(0.4, "gP/gX/h", "Non-growth-associated ethanol coefficient", 0.0, 1.5),
            "Yps": _p(0.45, "g/g", "Ethanol/substrate yield (max ~0.51)", 0.1, 0.51),
            "kLa": _p(100, "1/h", "Oxygen volumetric mass-transfer coefficient", 10, 400),
        },
        "references": ["luedeking_piret", "bioprocess"],
        "examples": [
            "What is the Pasteur effect in alcoholic fermentation?",
            "What is the maximum theoretical ethanol yield Yps?",
        ],
    },

    # ============================================ SENSITIVITY ANALYSIS =====
    "📈 Sensitivity Analysis": {
        "section": "Analysis",
        "method": "Local one-at-a-time sensitivity analysis (batch model)",
        "description": (
            "Varies a single chosen parameter (mumax, Ks, Yxs or Kd) over a "
            "user range, runs several batch simulations and reports how outputs "
            "(max biomass, min substrate, max product, peak time) respond, "
            "including a variation coefficient sigma/mu."
        ),
        "equations": [
            "Sensitivity coefficient ~ (delta_output/output)/(delta_param/param)",
            "Variation coefficient = std(output)/mean(output)",
            "Batch model: dX/dt = mu*X - Kd*X with mu = mumax*S/(Ks+S)",
        ],
        "parameters": {
            "mumax": _p(0.5, "1/h", "Base maximum specific growth rate", 0.1, 2.0),
            "Ks": _p(0.2, "g/L", "Base Monod saturation constant", 0.01, 5.0),
            "Yxs": _p(0.5, "g/g", "Base biomass/substrate yield", 0.1, 1.0),
            "Kd": _p(0.01, "1/h", "Base death constant", 0.0, 0.5),
            "variation_range": _p((0, 100), "%", "Percentage variation range", -50, 200),
            "n_simulations": _p(5, "-", "Number of simulations in the sweep", 2, 50),
        },
        "references": ["parameter_estimation", "bioprocess"],
        "examples": [
            "Which kinetic parameter most affects final biomass?",
            "How do I interpret a high variation coefficient in a sensitivity sweep?",
        ],
    },

    # ============================================ PARAMETER ESTIMATION =====
    "Batch Parameter Adjustment": {
        "section": "Parameter Optimization",
        "method": "Batch kinetic parameter fitting (least squares)",
        "description": (
            "Fits batch Monod kinetic parameters to experimental data (Excel) "
            "by minimizing the RMSE between simulation and measurements using "
            "scipy.optimize (L-BFGS-B, Nelder-Mead or Differential Evolution). "
            "Reports R2, RMSE and confidence intervals."
        ),
        "equations": [
            "Objective: min sum( (y_meas - y_sim(theta))^2 )",
            "RMSE = sqrt( mean( (y_meas - y_sim)^2 ) )",
            "Model: Monod batch ODEs (X, S, P, O2)",
        ],
        "parameters": {
            "mumax": _p(0.3, "1/h", "Fitted maximum specific growth rate", 0.01, 2.0),
            "Ks": _p(0.5, "g/L", "Fitted Monod saturation constant", 0.01, 20.0),
            "Yxs": _p(0.5, "g/g", "Fitted biomass/substrate yield", 0.01, 0.8),
            "Kd": _p(0.01, "1/h", "Fitted death constant", 0.0, 1.0),
            "Ypx": _p(0.3, "g/g", "Fitted product/biomass yield", 0.0, 10.0),
            "method": _p("L-BFGS-B", "-", "Optimizer (L-BFGS-B/Nelder-Mead/DE)"),
        },
        "references": ["parameter_estimation", "monod", "bioprocess"],
        "examples": [
            "Which optimizer is most robust for kinetic fitting?",
            "How do I read the confidence interval of a fitted parameter?",
        ],
    },

    "Fed-Batch Parameter Adjustment": {
        "section": "Parameter Optimization",
        "method": "Fed-batch kinetic parameter fitting with substrate inhibition",
        "description": (
            "Fits fed-batch kinetic parameters (including a substrate-inhibition "
            "constant Ksi) to experimental data, accounting for the feeding "
            "strategy and variable volume dynamics."
        ),
        "equations": [
            "Objective: min RMSE over [mumax, Ks, Yxs, Kd, Ypx, Ksi]",
            "Inhibition (Haldane): mu = mumax*S/(Ks + S + S^2/Ksi)",
            "Fed-batch mass balances with dilution F/V",
        ],
        "parameters": {
            "mumax": _p(0.4, "1/h", "Fitted maximum specific growth rate", 0.01, 2.0),
            "Ks": _p(0.5, "g/L", "Fitted Monod saturation constant", 0.01, 20.0),
            "Yxs": _p(0.5, "g/g", "Fitted biomass/substrate yield", 0.01, 0.8),
            "Kd": _p(0.01, "1/h", "Fitted death constant", 0.0, 1.0),
            "Ypx": _p(0.3, "g/g", "Fitted product/biomass yield", 0.0, 10.0),
            "Ksi": _p(100, "g/L", "Substrate inhibition constant", 1.0, 1000.0),
        },
        "references": ["parameter_estimation", "monod", "bioprocess"],
        "examples": [
            "When should I include a substrate-inhibition term Ksi?",
            "How does the feeding strategy affect parameter identifiability?",
        ],
    },

    "Fermentation Parameter Adjustment": {
        "section": "Parameter Optimization",
        "method": "Alcoholic fermentation kinetic parameter fitting",
        "description": (
            "Fits the mixed aerobic/anaerobic fermentation model (up to 5 model "
            "choices) to experimental data using a weighted/scaled SSE, a "
            "numerical Jacobian and 95% confidence intervals."
        ),
        "equations": [
            "Objective: min weighted SSE over the selected parameters",
            "Mixed kinetics: mu = mu_aer + mu_an (Pasteur effect)",
            "qP = alpha*mu_an + beta (Luedeking-Piret)",
        ],
        "parameters": {
            "mumax_aer": _p(0.4, "1/h", "Max aerobic growth rate", 0.05, 1.0),
            "mumax_an": _p(0.15, "1/h", "Max anaerobic growth rate", 0.05, 1.0),
            "Ki_an": _p(150, "g/L", "Substrate inhibition constant", 50, 500),
            "KP_an": _p(80, "g/L", "Ethanol inhibition constant", 20, 150),
            "alpha": _p(4.5, "gP/gX", "Growth-associated ethanol coefficient", 0.0, 10),
            "beta": _p(0.4, "gP/gX/h", "Non-growth-associated ethanol coefficient", 0.0, 1.5),
        },
        "references": ["luedeking_piret", "parameter_estimation", "bioprocess"],
        "examples": [
            "How do I weight ethanol vs. biomass residuals in the fit?",
            "Which fermentation model should I choose for my data?",
        ],
    },

    # ============================================== STATE ESTIMATION =======
    "EKF": {
        "section": "State Estimation",
        "method": "Extended Kalman Filter (state and parameter estimation)",
        "description": (
            "Estimates unmeasured states (X, S, P) and kinetic parameters "
            "(mumax, Yxs) from noisy indirect measurements (dissolved oxygen, "
            "pH, temperature) using a CasADi-based discrete EKF with "
            "prediction/correction steps."
        ),
        "equations": [
            "Predict: x_k|k-1 = f(x_k-1), P = F P F' + Q",
            "Update: K = P H'(H P H' + R)^-1, x = x + K(y - h(x))",
            "Measurements: DO = ODsat - kOUR*X ; pH = pH0 - kacid*(P-Pref)",
        ],
        "parameters": {
            "X0_est": _p(0.05, "g/L", "Initial biomass estimate", 0.01, 5),
            "S0_est": _p(5, "g/L", "Initial substrate estimate", 0.1, 50),
            "mumax_est": _p(0.40, "1/h", "Initial mumax estimate", 0.1, 1),
            "Yxs_est": _p(0.50, "g/g", "Initial Yxs estimate", 0.1, 1),
            "q_X": _p(1e-5, "-", "Process noise variance for X", 1e-8, 1e-2),
            "r_DO": _p(0.05, "(mg/L)^2", "Measurement noise variance for DO", 1e-4, 1),
            "r_pH": _p(0.02, "-", "Measurement noise variance for pH", 1e-4, 1),
            "r_T": _p(0.5, "K^2", "Measurement noise variance for temperature", 1e-2, 5),
        },
        "references": ["ekf", "bioprocess"],
        "examples": [
            "How do I tune Q and R in the EKF?",
            "Why does increasing R make the EKF trust the model more?",
        ],
    },

    "ANN": {
        "section": "State Estimation",
        "method": "Artificial Neural Network soft sensor",
        "description": (
            "Two-stage soft sensor: generates a noisy training dataset from a "
            "batch simulation, then trains a feed-forward ANN to predict states "
            "(X, S, P) and parameters (mumax, Yxs) from DO/pH/T measurements."
        ),
        "equations": [
            "Architecture: 3 inputs (DO, pH, T) -> hidden layers -> 5 outputs",
            "Loss: MSE = mean( (y_true - y_pred)^2 )",
        ],
        "parameters": {
            "sim_time": _p(40, "h", "Simulation horizon for data generation", 10, 100),
            "val_size": _p(20, "%", "Validation set fraction", 5, 50),
            "hidden1": _p(64, "neurons", "Neurons in hidden layer 1", 1, 256),
            "hidden2": _p(64, "neurons", "Neurons in hidden layer 2", 1, 256),
            "activation": _p("relu", "-", "Activation function (relu/tanh/sigmoid)"),
            "optimizer": _p("adam", "-", "Optimizer (adam/rmsprop/sgd)"),
            "learning_rate": _p(1e-3, "-", "Learning rate", 1e-5, 1e-1),
            "epochs": _p(200, "-", "Training epochs", 10, 1000),
            "batch_size": _p(16, "-", "Mini-batch size (8/16/32/64)", 8, 64),
        },
        "references": ["bioprocess", "parameter_estimation"],
        "examples": [
            "How many neurons should I use for an ANN soft sensor?",
            "What learning rate avoids divergence when training the ANN?",
        ],
    },

    # ============================================ REGULATORY CONTROL =======
    "Identification (pH)": {
        "section": "Regulatory Control",
        "method": "Process identification (pH transfer function)",
        "description": (
            "Identifies a low-order transfer function (gain, time constant, "
            "dead time) for the pH loop from a step test, providing the model "
            "used to tune the pH controller."
        ),
        "equations": [
            "FOPDT: G(s) = K e^(-theta s) / (tau s + 1)",
            "Least-squares fit of K, tau, theta to step-response data",
        ],
        "parameters": {
            "K": _p(1.0, "pH/unit", "Process steady-state gain", 0.1, 10),
            "tau": _p(5.0, "min", "Process time constant", 0.1, 60),
            "theta": _p(0.5, "min", "Dead time", 0.0, 30),
        },
        "references": ["pid_control"],
        "examples": [
            "How do I obtain a FOPDT model from a step test?",
            "Why is dead time critical when tuning a pH controller?",
        ],
    },

    "Temperature": {
        "section": "Regulatory Control",
        "method": "PID temperature control",
        "description": (
            "Simulates a PID loop controlling bioreactor temperature through a "
            "jacket, with first-order process and sensor dynamics."
        ),
        "equations": [
            "PID: u = Kc[e + (1/Ti) integral(e) dt + Td de/dt]",
            "First-order process: tau dT/dt = -T + K u",
        ],
        "parameters": {
            "Kc": _p(1.0, "-", "Proportional gain", 0.1, 10.0),
            "Ti": _p(1.0, "min", "Integral (reset) time", 0.1, 10.0),
            "Td": _p(0.1, "min", "Derivative time", 0.01, 1.0),
            "setpoint": _p(35.0, "°C", "Temperature setpoint", 20, 40),
        },
        "references": ["pid_control"],
        "examples": [
            "How do I tune Kc, Ti and Td for temperature?",
            "What causes integral windup in a temperature loop?",
        ],
    },

    "pH": {
        "section": "Regulatory Control",
        "method": "Split-range PID pH control (acid/base)",
        "description": (
            "Controls pH using a split-range PID that drives an acid pump and a "
            "base pump from a single control signal."
        ),
        "equations": [
            "PID: u = Kc[e + (1/Ti) integral(e) dt + Td de/dt]",
            "Split range: u>0 -> base pump, u<0 -> acid pump",
        ],
        "parameters": {
            "Kc": _p(1.0, "-", "Proportional gain", 0.1, 10.0),
            "Ti": _p(1.0, "min", "Integral time", 0.1, 10.0),
            "Td": _p(0.1, "min", "Derivative time", 0.01, 1.0),
            "setpoint": _p(7.0, "pH", "pH setpoint", 3, 11),
        },
        "references": ["pid_control"],
        "examples": [
            "How does split-range control switch between acid and base?",
            "Why is pH control nonlinear near the setpoint?",
        ],
    },

    "Oxygen": {
        "section": "Regulatory Control",
        "method": "Dissolved oxygen PID control via agitation",
        "description": (
            "Controls dissolved oxygen by manipulating agitation speed (RPM), "
            "which changes kLa and therefore oxygen transfer."
        ),
        "equations": [
            "PID on DO error drives RPM (hence kLa)",
            "dO2/dt = kLa(RPM)*(Cs - O2) - OUR",
        ],
        "parameters": {
            "Kc": _p(1.0, "-", "Proportional gain", 0.1, 10.0),
            "Ti": _p(1.0, "min", "Integral time", 0.1, 10.0),
            "Td": _p(0.1, "min", "Derivative time", 0.01, 1.0),
            "DO_setpoint": _p(30.0, "% sat", "Dissolved oxygen setpoint", 0, 100),
        },
        "references": ["pid_control", "bioprocess"],
        "examples": [
            "How does agitation speed affect kLa and dissolved oxygen?",
            "What DO setpoint is typical for aerobic cultures?",
        ],
    },

    "Cascade-Oxygen": {
        "section": "Regulatory Control",
        "method": "Cascade control of dissolved oxygen (DO + RPM loops)",
        "description": (
            "Two-loop cascade: a master DO controller sets the RPM setpoint for "
            "a fast inner agitation-speed controller, improving disturbance "
            "rejection."
        ),
        "equations": [
            "Outer loop: PID(DO_error) -> RPM_setpoint",
            "Inner loop: PID(RPM_error) -> motor signal",
        ],
        "parameters": {
            "Kc_outer": _p(1.0, "-", "Master (DO) proportional gain", 0.1, 10.0),
            "Ti_outer": _p(2.0, "min", "Master integral time", 0.1, 20.0),
            "Kc_inner": _p(2.0, "-", "Slave (RPM) proportional gain", 0.1, 20.0),
            "Ti_inner": _p(0.5, "min", "Slave integral time", 0.05, 10.0),
            "DO_setpoint": _p(30.0, "% sat", "Dissolved oxygen setpoint", 0, 100),
        },
        "references": ["pid_control", "bioprocess"],
        "examples": [
            "Why is the inner loop tuned faster than the outer loop in cascade control?",
            "When does cascade control outperform a single DO loop?",
        ],
    },

    "On-Off Feeding": {
        "section": "Regulatory Control",
        "method": "On-off (bang-bang) substrate feeding",
        "description": (
            "Controls substrate concentration with an on-off feed valve and a "
            "dead-band, switching the feed within an allowed time window."
        ),
        "equations": [
            "If S < S_low -> feed ON; if S > S_high -> feed OFF (hysteresis)",
            "Monod growth with on-off feed term in the mass balance",
        ],
        "parameters": {
            "S_low": _p(2.0, "g/L", "Lower substrate threshold (feed on)", 0.1, 50),
            "S_high": _p(5.0, "g/L", "Upper substrate threshold (feed off)", 0.1, 50),
            "F_on": _p(0.5, "L/h", "Feed flow when on", 0.01, 5),
            "t_start": _p(2.0, "h", "Feeding window start", 0, 100),
            "t_end": _p(24.0, "h", "Feeding window end", 0, 200),
        },
        "references": ["pid_control", "bioprocess"],
        "examples": [
            "How wide should the dead-band be for on-off feeding?",
            "What is the downside of bang-bang control vs. PID?",
        ],
    },

    # ============================================== ADVANCED CONTROL =======
    "RTO": {
        "section": "Advanced Control",
        "method": "Real-Time Optimization (dynamic optimization via collocation)",
        "description": (
            "Computes the optimal feed-rate profile that maximizes final "
            "product amount (P*V) subject to constraints, using CasADi with "
            "orthogonal collocation and IPOPT."
        ),
        "equations": [
            "max integral / final P*V",
            "subject to model ODEs and bounds on S, V, F",
        ],
        "parameters": {
            "F_max": _p(1.5, "L/h", "Maximum feed flow", 0.0, 5.0),
            "V_max": _p(10.0, "L", "Maximum reactor volume", 1, 100),
            "S_max": _p(15.0, "g/L", "Maximum allowed substrate", 0, 100),
            "N_elements": _p(20, "-", "Number of collocation elements", 5, 100),
            "t_final": _p(24.0, "h", "Optimization horizon", 1, 200),
        },
        "references": ["rto", "mpc"],
        "examples": [
            "What objective does RTO maximize in a fed-batch?",
            "Why use orthogonal collocation for dynamic optimization?",
        ],
    },

    "RTO Ferm": {
        "section": "Advanced Control",
        "method": "Real-Time Optimization for alcoholic fermentation (3-phase)",
        "description": (
            "Phase-dependent dynamic optimization of the fermentation feed "
            "profile, maximizing ethanol production with slack variables for "
            "soft constraints across batch/fed-batch/batch phases."
        ),
        "equations": [
            "max final ethanol (P*V) with phase-dependent kinetics",
            "soft constraints via slack penalties",
        ],
        "parameters": {
            "F_max": _p(1.0, "L/h", "Maximum feed flow", 0.0, 5.0),
            "V_max": _p(2.0, "L", "Maximum reactor volume", 0.5, 100),
            "t_batch_end": _p(6.0, "h", "End of initial batch phase", 0, 50),
            "t_feed_end": _p(24.0, "h", "End of feeding phase", 0, 100),
            "N_elements": _p(30, "-", "Number of collocation elements", 5, 100),
        },
        "references": ["rto", "luedeking_piret"],
        "examples": [
            "How does phase switching affect the optimal feed profile?",
            "Why add slack variables to the fermentation RTO?",
        ],
    },

    "NMPC": {
        "section": "Advanced Control",
        "method": "Nonlinear Model Predictive Control (receding horizon)",
        "description": (
            "Solves, at each step, a finite-horizon nonlinear optimal control "
            "problem over the bioreactor model to compute feed (F_S) and jacket "
            "duty (Q_j), applying only the first move (receding horizon)."
        ),
        "equations": [
            "min sum Qx*(x-xsp)^2 + Wu*du^2 over horizon N",
            "subject to nonlinear ODE model and input/state constraints",
        ],
        "parameters": {
            "N": _p(10, "-", "Prediction horizon", 1, 30),
            "M": _p(4, "-", "Control horizon", 1, 20),
            "Q_X": _p(1.0, "-", "Weight on biomass tracking", 0.1, 100),
            "Q_T": _p(10.0, "-", "Weight on temperature tracking", 0.001, 100),
            "FS_max": _p(1.5, "L/h", "Maximum feed flow", 0.0, 1.5),
            "dFS_max": _p(0.1, "L/h", "Max feed rate change per step", 0.01, 1),
        },
        "references": ["mpc", "rto"],
        "examples": [
            "What is the difference between prediction and control horizon?",
            "How do the cost weights trade off tracking vs. input effort?",
        ],
    },

    "LMPC": {
        "section": "Advanced Control",
        "method": "Linear Model Predictive Control",
        "description": (
            "MPC based on a linearized 2x2 MIMO transfer-function model of the "
            "reactor, minimizing a quadratic cost subject to constraints; faster "
            "but less accurate than NMPC far from the operating point."
        ),
        "equations": [
            "Linear prediction model y = G u (step-response / state space)",
            "min ||y - ysp||_Q^2 + ||du||_R^2 subject to constraints",
        ],
        "parameters": {
            "N": _p(10, "-", "Prediction horizon", 1, 30),
            "M": _p(3, "-", "Control horizon", 1, 20),
            "Q": _p(1.0, "-", "Output tracking weight", 0.01, 100),
            "R": _p(0.1, "-", "Input move weight", 0.001, 10),
        },
        "references": ["mpc", "pid_control"],
        "examples": [
            "When is LMPC preferable to NMPC?",
            "How does the move-suppression weight R affect aggressiveness?",
        ],
    },

    "EKF-NMPC": {
        "section": "Advanced Control",
        "method": "Output-feedback NMPC with an EKF state estimator",
        "description": (
            "Closed-loop combination where an EKF reconstructs states and "
            "parameters from noisy DO/pH/T measurements and feeds them to an "
            "NMPC controller computing feed and jacket duty."
        ),
        "equations": [
            "EKF predict/update -> x_hat",
            "NMPC: min cost over horizon using x_hat as initial state",
        ],
        "parameters": {
            "N": _p(10, "-", "NMPC prediction horizon", 1, 30),
            "M": _p(4, "-", "NMPC control horizon", 1, 20),
            "q_X": _p(1e-5, "-", "EKF process noise for X", 1e-8, 1e-2),
            "r_DO": _p(0.05, "(mg/L)^2", "EKF DO measurement noise", 1e-4, 1),
            "FS_max": _p(1.5, "L/h", "Maximum feed flow", 0.0, 1.5),
        },
        "references": ["mpc", "ekf"],
        "examples": [
            "Why combine an EKF with NMPC?",
            "How does estimator noise tuning affect closed-loop NMPC?",
        ],
    },

    "Fuzzy Control": {
        "section": "Advanced Control",
        "method": "Mamdani fuzzy logic control (pH, temperature, feeding)",
        "description": (
            "Mamdani fuzzy controller that fuzzifies pH/temperature/substrate "
            "errors, applies expert IF-THEN rules and defuzzifies with the "
            "centroid method to drive acid/base, heating/cooling and feed."
        ),
        "equations": [
            "Fuzzification with triangular/trapezoidal membership functions",
            "AND=min, OR=max, implication=min, aggregation=max",
            "Centroid defuzzification: u = sum(mu*u)/sum(mu)",
        ],
        "parameters": {
            "pH_setpoint": _p(7.0, "pH", "pH setpoint", 3, 11),
            "T_setpoint": _p(35.0, "°C", "Temperature setpoint", 20, 40),
            "S_target": _p(10.0, "g/L", "Substrate target", 0, 100),
            "pH_aggressiveness": _p(1.3, "-", "pH control aggressiveness", 0.5, 2),
            "T_aggressiveness": _p(1.3, "-", "Temperature control aggressiveness", 0.5, 2),
            "feed_sensitivity": _p(1.2, "-", "Feed-rate sensitivity", 0.5, 2),
        },
        "references": ["fuzzy_control", "pid_control"],
        "examples": [
            "How do membership functions encode expert knowledge?",
            "What is centroid defuzzification?",
        ],
    },

    # ================================================== DIGITAL TWIN =======
    "🏭 Digital Twin": {
        "section": "Digital Twin",
        "method": "Digital twin for distillation columns (DWSIM + SCADA + ML)",
        "description": (
            "Integrates rigorous distillation simulation (DWSIM / FUG shortcut "
            "method), SCADA data analysis with outlier detection (IQR), moving-"
            "average filtering and WLS data reconciliation, plus machine-"
            "learning prediction of distillate ethanol composition (Neural "
            "Network, Random Forest, Decision Tree, SVR, Gradient Boosting)."
        ),
        "equations": [
            "FUG shortcut: Fenske (Nmin), Underwood (Rmin), Gilliland (N), Kirkbride (feed)",
            "WLS reconciliation: min sum((y_meas - x)^2/sigma^2) s.t. mass balance",
            "IQR outlier rule: outside [Q1-1.5 IQR, Q3+1.5 IQR]",
            "ML metrics: R2, MSE, MAE, RMSE",
        ],
        "parameters": {
            "feed_flow": _p(1000, "kmol/h", "Feed molar flow", 500, 2000),
            "feed_T": _p(60, "°C", "Feed temperature", 20, 100),
            "feed_P": _p(2.0, "bar", "Feed pressure", 1.0, 5.0),
            "ethanol_feed": _p(50, "mol%", "Ethanol mole fraction in feed", 30, 80),
            "reflux_mult": _p(1.5, "-", "Reflux ratio multiplier (R/Rmin)", 0.5, 3.0),
            "MA_window": _p(7, "-", "Moving-average window (odd)", 3, 21),
            "outlier_fraction": _p(5, "%", "Injected outlier fraction", 0, 20),
            "test_size": _p(0.2, "fraction", "ML test-set size", 0.1, 0.5),
        },
        "references": ["bioprocess", "parameter_estimation"],
        "examples": [
            "How does WLS data reconciliation enforce mass balance?",
            "Which ML model best predicts distillate ethanol composition?",
            "How does the reflux ratio multiplier affect the number of stages?",
        ],
    },
}


# ===========================================================================
# PAGE-NAME NORMALIZATION
# ===========================================================================
# Maps alternative / aggregate page names to the canonical knowledge keys.
_PAGE_ALIASES: Dict[str, str] = {
    "Home": "🏠 Home",
    "🔬 Models": "Batch",
    "Models": "Batch",
    "Sensitivity Analysis": "📈 Sensitivity Analysis",
    "🔧 Parameter Adjustment": "Batch Parameter Adjustment",
    "Parameter Adjustment": "Batch Parameter Adjustment",
    "📊 State Estimation": "EKF",
    "State Estimation": "EKF",
    "⚙️ Control": "Temperature",
    "Control": "Temperature",
    "Digital Twin": "🏭 Digital Twin",
}


def resolve_page_key(page_name: Optional[str]) -> Optional[str]:
    """Return the canonical knowledge-base key for a given page name."""
    if not page_name:
        return None
    if page_name in PAGE_KNOWLEDGE:
        return page_name
    if page_name in _PAGE_ALIASES:
        return _PAGE_ALIASES[page_name]
    # Tolerate emoji-stripped / case-insensitive matches.
    stripped = page_name.strip()
    for key in PAGE_KNOWLEDGE:
        if key.lower().lstrip("🏠📈🔬🔧📊⚙️🏭 ").strip() == stripped.lower():
            return key
    return None


def get_page_knowledge(page_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the knowledge dict for a page, or None if unknown."""
    key = resolve_page_key(page_name)
    if key is None:
        return None
    return PAGE_KNOWLEDGE[key]


def get_page_parameters(page_name: Optional[str]) -> Dict[str, Any]:
    """Return the parameter descriptors for a page (empty dict if unknown)."""
    knowledge = get_page_knowledge(page_name)
    if not knowledge:
        return {}
    return knowledge.get("parameters", {})


def format_parameter_table(parameters: Dict[str, Any]) -> str:
    """Render a Markdown table of parameter ranges for offline display."""
    if not parameters:
        return "_No parameter ranges are registered for this page._"
    lines = [
        "| Parameter | Typical | Range | Unit | Description |",
        "|-----------|---------|-------|------|-------------|",
    ]
    for name, info in parameters.items():
        default = info.get("default", "?")
        mn, mx = info.get("min"), info.get("max")
        rng = f"{mn} – {mx}" if mn is not None and mx is not None else "—"
        unit = info.get("unit", "")
        desc = info.get("description", "")
        lines.append(f"| `{name}` | {default} | {rng} | {unit} | {desc} |")
    return "\n".join(lines)
