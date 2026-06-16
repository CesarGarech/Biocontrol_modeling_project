# -*- coding: utf-8 -*-
"""
LLM Helper Module for Bioprocess Modeling Project
Provides connection to Ollama API and contextual prompt generation
"""

import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

# Page knowledge base used to ground the Llama model on every app screen.
try:
    from Utils.llm_knowledge_base import (
        get_page_knowledge,
        get_page_parameters,
        format_parameter_table,
        resolve_page_key,
    )
except Exception:  # pragma: no cover - keep helper importable in isolation
    get_page_knowledge = lambda page: None  # type: ignore
    get_page_parameters = lambda page: {}  # type: ignore
    format_parameter_table = lambda params: ""  # type: ignore
    resolve_page_key = lambda page: None  # type: ignore


# ==================== CURATED REFERENCES ====================
CURATED_REFERENCES = {
    "monod": [
        "Monod, J. (1949). 'The growth of bacterial cultures.' Annual Review of Microbiology, 3(1), 371-394.",
        "Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall."
    ],
    "luedeking_piret": [
        "Luedeking, R., & Piret, E. L. (1959). 'A kinetic study of the lactic acid fermentation. Batch process at controlled pH.' Journal of Biochemical and Microbiological Technology and Engineering, 1(4), 393-412."
    ],
    "pid_control": [
        "Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic Process Control (3rd ed.). John Wiley & Sons.",
        "Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). Process Dynamics and Control (4th ed.). John Wiley & Sons."
    ],
    "mpc": [
        "Camacho, E. F., & Bordons, C. (2007). Model Predictive Control (2nd ed.). Springer-Verlag.",
        "Rawlings, J. B., Mayne, D. Q., & Diehl, M. (2017). Model Predictive Control: Theory, Computation, and Design (2nd ed.). Nob Hill Publishing."
    ],
    "ekf": [
        "Jazwinski, A. H. (1970). Stochastic Processes and Filtering Theory. Academic Press.",
        "Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches. John Wiley & Sons."
    ],
    "rto": [
        "Biegler, L. T. (2010). Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. SIAM.",
        "Marlin, T. E. (2000). Process Control: Designing Processes and Control Systems for Dynamic Performance (2nd ed.). McGraw-Hill."
    ],
    "bioprocess": [
        "Bailey, J. E., & Ollis, D. F. (1986). Biochemical Engineering Fundamentals (2nd ed.). McGraw-Hill.",
        "Shuler, M. L., & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall."
    ],
    "parameter_estimation": [
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press."
    ],
    "fuzzy_control": [
        "Zadeh, L. A. (1965). 'Fuzzy sets.' Information and Control, 8(3), 338-353.",
        "Passino, K. M., & Yurkovich, S. (1998). Fuzzy Control. Addison Wesley Longman."
    ]
}

# ==================== OLLAMA API CONFIGURATION ====================
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Base Llama model that the assistant is built on.
BASE_MODEL = "llama3.1:8b"

# Name of the customized ("re-trained") domain model created from BASE_MODEL
# using the Modelfile in installer/ollama/biocontrol_assistant.Modelfile.
# When this model exists in Ollama it is preferred because it already embeds the
# Biocontrol system prompt and recommended generation parameters.
CUSTOM_MODEL_NAME = "biocontrol-llama"

DEFAULT_MODEL = CUSTOM_MODEL_NAME
AVAILABLE_MODELS = [
    CUSTOM_MODEL_NAME,   # Customized Biocontrol assistant (preferred)
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "mistral:7b",
    "phi3:mini"
]

# ==================== PROMPT TEMPLATES ====================
SYSTEM_PROMPT = """You are the Biocontrol AI Guide, an educational assistant embedded in an interactive
Streamlit application for modeling, simulation, analysis and control of bioprocesses
(developed at LADES - COPPE/UFRJ).

You must be able to help with EVERY screen of the application:
1. Modeling: Batch, Fed-Batch, Continuous (chemostat) and Alcoholic Fermentation reactors.
2. Sensitivity Analysis of kinetic parameters.
3. Parameter Optimization: fitting Batch, Fed-Batch and Fermentation kinetic parameters to data.
4. State Estimation: Extended Kalman Filter (EKF) and Artificial Neural Network (ANN) soft sensors.
5. Regulatory Control: pH identification, PID temperature, split-range pH, dissolved oxygen,
   cascade oxygen and on-off feeding.
6. Advanced Control: RTO, RTO for fermentation, NMPC, LMPC, EKF-NMPC and Fuzzy control.
7. Digital Twin: distillation column simulation (DWSIM/FUG), SCADA data reconciliation and
   machine-learning composition prediction.

Your role is to:
1. Explain mathematical equations and models clearly and educationally.
2. Describe simulation, estimation, optimization and control methods in accessible language.
3. Suggest reasonable parameter ranges based on the literature AND on the parameters of the
   current screen that are provided to you in the context.
4. Recommend appropriate bibliographic references.

IMPORTANT:
- Use the page context (method, equations and parameters) provided to you to give answers that
  are specific to the screen the user is currently on.
- When suggesting parameter values, prefer the typical ranges of the current screen and always
  add a disclaimer that they must be validated experimentally.
- Respond in English, clearly and concisely.
- If you do not have sufficient information, state it honestly."""

def build_context_prompt(page_name: str, user_question: str, 
                         equations: Optional[List[str]] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         method: Optional[str] = None,
                         use_knowledge_base: bool = True) -> str:
    """Build a contextual prompt based on current page and user input.

    When ``use_knowledge_base`` is True (default), the per-page knowledge base
    (method, description, equations and typical parameter ranges for the current
    screen) is automatically injected so the model can answer/suggest parameters
    specifically for the screen the user is on. Explicitly passed ``equations``,
    ``parameters`` or ``method`` always take precedence over the knowledge base.
    """
    knowledge = get_page_knowledge(page_name) if use_knowledge_base else None

    if knowledge:
        method = method or knowledge.get("method")
        if equations is None and knowledge.get("equations"):
            equations = knowledge["equations"]
        if parameters is None and knowledge.get("parameters"):
            # Flatten descriptor dicts into a readable "typical [range] unit" string.
            parameters = {}
            for name, info in knowledge["parameters"].items():
                if isinstance(info, dict):
                    default = info.get("default")
                    mn, mx = info.get("min"), info.get("max")
                    unit = info.get("unit", "")
                    desc = info.get("description", "")
                    rng = f" (range {mn}-{mx})" if mn is not None and mx is not None else ""
                    unit_str = f" {unit}" if unit and unit != "-" else ""
                    parameters[name] = f"typical {default}{rng}{unit_str} - {desc}".strip()
                else:
                    parameters[name] = info

    context_parts = [f"Context: I am on the '{page_name}' page of the bioprocess modeling app."]

    if knowledge:
        context_parts.append(f"Application area: {knowledge.get('section', 'N/A')}")
        if knowledge.get("description"):
            context_parts.append(f"Page description: {knowledge['description']}")

    if method:
        context_parts.append(f"Current Method/Model: {method}")
    
    if equations:
        context_parts.append("\nRelevant equations:")
        for eq in equations:
            context_parts.append(f"  - {eq}")
    
    if parameters:
        context_parts.append("\nParameters used on this screen (typical values and ranges):")
        for key, value in parameters.items():
            context_parts.append(f"  - {key}: {value}")
    
    context_parts.append(f"\nUser question: {user_question}")
    
    return "\n".join(context_parts)


def get_relevant_references(page_name: str, keywords: List[str]) -> List[str]:
    """Get relevant bibliographic references based on page and keywords.

    The reference keys are taken from the per-page knowledge base so that every
    screen (modeling, parameter optimization, estimation, regulatory/advanced
    control and digital twin) returns curated references. A legacy mapping is
    kept as a fallback for page names not present in the knowledge base.
    """
    references: List[str] = []

    # Preferred source: per-page knowledge base.
    knowledge = get_page_knowledge(page_name)
    ref_keys: List[str] = []
    if knowledge and knowledge.get("references"):
        ref_keys = list(knowledge["references"])
    else:
        # Legacy fallback mapping (kept for backward compatibility).
        legacy_mapping = {
            "Batch": ["monod", "bioprocess"],
            "Fed-Batch": ["monod", "bioprocess"],
            "Continuous": ["monod", "bioprocess"],
            "Fermentation": ["luedeking_piret", "bioprocess"],
            "Temperature": ["pid_control"],
            "pH": ["pid_control"],
            "Oxygen": ["pid_control"],
            "RTO": ["rto", "mpc"],
            "NMPC": ["mpc"],
            "EKF": ["ekf"],
            "ANN": ["bioprocess", "parameter_estimation"],
            "Fuzzy Control": ["fuzzy_control"],
        }
        ref_keys = legacy_mapping.get(page_name, [])

    for ref_key in ref_keys:
        if ref_key in CURATED_REFERENCES:
            references.extend(CURATED_REFERENCES[ref_key])

    if not references:
        references.extend(CURATED_REFERENCES["bioprocess"])

    seen = set()
    unique_refs = []
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)

    return unique_refs[:5]


def check_ollama_availability(base_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, str]:
    """Check if Ollama service is available and list local models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return True, f"Connected. Available models: {', '.join(models[:3])}"
        else:
            return False, f"Connection error (code {response.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Make sure it is running (ollama serve)"
    except requests.exceptions.Timeout:
        return False, "Timeout connecting to Ollama"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def model_exists(model_name: str, base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Return True if ``model_name`` is already available locally in Ollama.

    Matches both the exact tag and the bare name (e.g. ``llama3.1`` matches
    ``llama3.1:8b``) so the helper works regardless of how the user typed it.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False
        local = [m.get("name", "") for m in response.json().get("models", [])]
        if model_name in local:
            return True
        base = model_name.split(":")[0]
        return any(name.split(":")[0] == base for name in local)
    except Exception:
        return False


def _extract_error_message(response: requests.Response) -> str:
    """Best-effort extraction of Ollama's JSON ``error`` field from a response."""
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    except Exception:
        pass
    return response.text[:200] if response.text else ""


def pull_ollama_model(model_name: str, base_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, str]:
    """
    Programmatically pull (download) an Ollama model.
    Uses a long timeout as model downloads can take minutes depending on bandwidth.
    """
    # The customized assistant only exists locally; it is NOT published on the
    # Ollama registry, so attempting to ``pull`` it returns a 500 error. Guide
    # the user to build it instead of hitting the registry.
    if model_name == CUSTOM_MODEL_NAME:
        return False, (
            f"'{CUSTOM_MODEL_NAME}' is a local custom model and cannot be downloaded "
            f"from the Ollama registry. Pull the base model '{BASE_MODEL}' first, then "
            f"use '🛠️ Build Custom Assistant' (or run "
            f"`python installer/ollama/build_model.py`) to create it."
        )

    try:
        url = f"{base_url}/api/pull"
        # The newer API uses 'model'; 'name' is kept for backward compatibility.
        data = {"model": model_name, "name": model_name, "stream": False}
        # Timeout is set to 600 seconds (10 minutes) to allow large models to download
        response = requests.post(url, json=data, timeout=600)

        if response.status_code == 200:
            return True, f"✅ Model '{model_name}' successfully downloaded and ready."
        else:
            detail = _extract_error_message(response)
            detail = f" - {detail}" if detail else ""
            return False, f"Server error while pulling: {response.status_code}{detail}"

    except requests.exceptions.Timeout:
        return False, "⏳ Request timed out, but the download might still be running in the background. Check your terminal."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Verify it is running."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def query_ollama(prompt: str, model: str = DEFAULT_MODEL, 
                base_url: str = DEFAULT_OLLAMA_URL,
                temperature: float = 0.7,
                max_tokens: int = 1000) -> Tuple[bool, str]:
    """Send a query to Ollama API and get response.

    For the customized ``biocontrol-llama`` model the system prompt is already
    embedded in the Modelfile, so it is not prepended again. For any other base
    model the global ``SYSTEM_PROMPT`` is prepended to ground the answer.
    """
    try:
        url = f"{base_url}/api/generate"
        if model == CUSTOM_MODEL_NAME:
            full_prompt = prompt
        else:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        data = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get('response', 'No response')
        elif response.status_code == 404:
            # Ollama returns 404 when the requested model is not installed locally.
            if model == CUSTOM_MODEL_NAME:
                hint = (
                    f"The custom model '{model}' is not installed. Pull the base model "
                    f"'{BASE_MODEL}' and click '🛠️ Build Custom Assistant' (or run "
                    f"`python installer/ollama/build_model.py`) to create it."
                )
            else:
                hint = (
                    f"Model '{model}' is not installed. Click '⬇️ Download Model' "
                    f"or run `ollama pull {model}` first."
                )
            return False, f"Model not found (404). {hint}"
        else:
            detail = _extract_error_message(response)
            detail = f" - {detail}" if detail else ""
            return False, f"Server error: {response.status_code}{detail}"
            
    except requests.exceptions.Timeout:
        return False, "Request took too long. Try a smaller model or verify if it's downloaded."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Verify it is running."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# ==================== CUSTOM MODEL ("RE-TRAINING") ====================

# Recommended generation parameters embedded into the customized model so the
# assistant gives focused, educational answers. Shared by the Modelfile builder
# and the structured (modern) /api/create payload.
CUSTOM_MODEL_PARAMETERS: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.9,
    "num_ctx": 4096,
}


def build_modelfile(base_model: str = BASE_MODEL) -> str:
    """Build the Ollama Modelfile content used to create the customized model.

    This is the practical way to "re-train" / specialize Llama for this app
    without GPU fine-tuning: the domain ``SYSTEM`` prompt and recommended
    generation parameters are embedded into a derived Ollama model so the
    chatbot is grounded on every screen of the application.
    """
    escaped_prompt = SYSTEM_PROMPT.replace('"""', '\\"\\"\\"')
    params = "".join(
        f"PARAMETER {name} {value}\n" for name, value in CUSTOM_MODEL_PARAMETERS.items()
    )
    return (
        f"FROM {base_model}\n\n"
        f"# Recommended generation parameters for educational, focused answers\n"
        f"{params}\n"
        f'SYSTEM """\n{escaped_prompt}\n"""\n'
    )


def create_custom_model(base_url: str = DEFAULT_OLLAMA_URL,
                        base_model: str = BASE_MODEL,
                        model_name: str = CUSTOM_MODEL_NAME) -> Tuple[bool, str]:
    """Create the customized Biocontrol model in Ollama via the /api/create API.

    Requires that the base model (e.g. llama3.1:8b) is already pulled. The call
    can take a while the first time because Ollama materializes the new model.

    Newer Ollama releases (v0.5+) replaced the deprecated ``modelfile`` body
    field with a structured schema (``from``/``system``/``parameters``). Sending
    only ``modelfile`` now fails with
    ``400 - {"error":"neither 'from' or 'files' was specified"}``. This function
    therefore sends the modern schema first and transparently falls back to the
    legacy ``modelfile`` payload for older Ollama servers.
    """
    # Helpful pre-flight: if the base model is missing the create call fails.
    if not model_exists(base_model, base_url):
        return False, (
            f"Base model '{base_model}' is not installed. Pull it first with "
            f"'⬇️ Download Model' or `ollama pull {base_model}`, then build the "
            f"custom assistant again."
        )

    url = f"{base_url}/api/create"
    # Modern structured payload (Ollama v0.5+). 'model' is the new key for the
    # created model name; 'from' is the base model to derive from.
    modern_payload = {
        "model": model_name,
        "from": base_model,
        "system": SYSTEM_PROMPT,
        "parameters": CUSTOM_MODEL_PARAMETERS,
        "stream": False,
    }
    # Legacy payload for older Ollama servers that still accept 'modelfile'.
    legacy_payload = {
        "name": model_name,
        "modelfile": build_modelfile(base_model),
        "stream": False,
    }

    try:
        response = requests.post(url, json=modern_payload, timeout=600)

        # Older servers may reject the modern schema; retry with the legacy one.
        if response.status_code == 400:
            response = requests.post(url, json=legacy_payload, timeout=600)

        if response.status_code == 200:
            return True, (
                f"✅ Custom model '{model_name}' created from '{base_model}'. "
                f"Select it in the model list to use the grounded assistant."
            )
        detail = _extract_error_message(response)
        detail = f" - {detail}" if detail else ""
        return False, f"Server error while creating model: {response.status_code}{detail}"
    except requests.exceptions.Timeout:
        return False, "⏳ Model creation timed out. It may still be finishing in the background."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Verify it is running (ollama serve)."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def format_response_with_references(llm_response: str, references: List[str]) -> str:
    """Format LLM response with references and disclaimers."""
    formatted = f"{llm_response}\n\n"
    formatted += "---\n\n"
    formatted += "**📚 Relevant references:**\n\n"
    for i, ref in enumerate(references, 1):
        formatted += f"{i}. {ref}\n"
    formatted += "\n**⚠️ Note:** This information is for guidance and must be experimentally validated. "
    formatted += "Always consult specialized literature for your specific application."
    
    return formatted


def suggest_parameter_ranges(parameter_name: str, model_type: str,
                             page_name: Optional[str] = None) -> Dict[str, Any]:
    """Suggest typical parameter ranges based on literature.

    If ``page_name`` is given, the per-page knowledge base is consulted first so
    the suggestion is specific to the current screen. Otherwise (or if the
    parameter is not registered for that page) a global table is used.
    """
    # 1) Page-specific parameters take priority.
    if page_name:
        page_params = get_page_parameters(page_name)
        param_lower = parameter_name.lower()
        for key, info in page_params.items():
            if not isinstance(info, dict):
                continue
            if key.lower() in param_lower or param_lower in key.lower():
                return {
                    "min": info.get("min"),
                    "max": info.get("max"),
                    "typical": info.get("default"),
                    "unit": info.get("unit", "?"),
                    "description": info.get("description", ""),
                }

    # 2) Global fallback table.
    ranges = {
        "mumax": {"min": 0.1, "max": 1.5, "typical": 0.5, "unit": "h⁻¹", 
                  "description": "Maximum specific growth rate"},
        "Ks": {"min": 0.01, "max": 5.0, "typical": 0.5, "unit": "g/L",
               "description": "Monod saturation constant"},
        "Yxs": {"min": 0.3, "max": 0.8, "typical": 0.5, "unit": "g_X/g_S",
                "description": "Biomass/substrate yield"},
        "kd": {"min": 0.001, "max": 0.05, "typical": 0.01, "unit": "h⁻¹",
               "description": "Cell death constant"},
        "kla": {"min": 10, "max": 300, "typical": 100, "unit": "h⁻¹",
                "description": "Volumetric mass transfer coefficient for oxygen"},
        "Kc": {"min": 0.1, "max": 10.0, "typical": 1.0, "unit": "adim.",
               "description": "Proportional controller gain"},
        "Ti": {"min": 0.1, "max": 10.0, "typical": 1.0, "unit": "min",
               "description": "Integral time"},
        "Td": {"min": 0.01, "max": 1.0, "typical": 0.1, "unit": "min",
               "description": "Derivative time"}
    }
    
    param_lower = parameter_name.lower()
    for key, value in ranges.items():
        if key.lower() in param_lower or param_lower in key.lower():
            return value
    
    return {"min": None, "max": None, "typical": None, "unit": "?",
            "description": "Parameter not recognized in the database"}