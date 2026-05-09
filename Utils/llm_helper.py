# -*- coding: utf-8 -*-
"""
LLM Helper Module for Bioprocess Modeling Project
Provides connection to Ollama API and contextual prompt generation
"""

import json
import requests
from typing import Dict, List, Optional, Tuple
import streamlit as st


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
DEFAULT_MODEL = "llama3.1:8b"
AVAILABLE_MODELS = [
    "llama3.1:8b",
    "llama3.2:3b", 
    "qwen2.5:7b",
    "mistral:7b",
    "phi3:mini"
]

# ==================== PROMPT TEMPLATES ====================
SYSTEM_PROMPT = """Eres un asistente educativo especializado en ingeniería de bioprocesos, modelado matemático y control de procesos. 

Tu rol es:
1. Explicar ecuaciones y modelos matemáticos de forma clara y educativa
2. Describir métodos de simulación, estimación y control en lenguaje accesible
3. Sugerir rangos razonables de parámetros basados en literatura
4. Recomendar referencias bibliográficas apropiadas

IMPORTANTE:
- Solo usa información técnica fundamentada y referencias que te proporcionen
- Indica rangos típicos de parámetros con disclaimers sobre validación experimental
- Siempre menciona que las sugerencias son orientativas y requieren validación
- Responde en español de forma clara y concisa
- Si no tienes información suficiente, indícalo honestamente"""

def build_context_prompt(page_name: str, user_question: str, 
                         equations: Optional[List[str]] = None,
                         parameters: Optional[Dict[str, any]] = None,
                         method: Optional[str] = None) -> str:
    """
    Build a contextual prompt based on current page and user input.
    
    Args:
        page_name: Name of the current page/module
        user_question: User's question or request
        equations: List of equations shown in the current page
        parameters: Dictionary of current parameter values
        method: Current method/model being used
        
    Returns:
        Formatted prompt string for the LLM
    """
    context_parts = [f"Contexto: Estoy en la página '{page_name}' de la aplicación de modelado de bioprocesos."]
    
    if method:
        context_parts.append(f"Método/Modelo actual: {method}")
    
    if equations:
        context_parts.append("\nEcuaciones relevantes:")
        for eq in equations:
            context_parts.append(f"  - {eq}")
    
    if parameters:
        context_parts.append("\nParámetros actuales del usuario:")
        for key, value in parameters.items():
            context_parts.append(f"  - {key}: {value}")
    
    context_parts.append(f"\nPregunta del usuario: {user_question}")
    
    return "\n".join(context_parts)


def get_relevant_references(page_name: str, keywords: List[str]) -> List[str]:
    """
    Get relevant bibliographic references based on page and keywords.
    
    Args:
        page_name: Name of the current page
        keywords: List of relevant keywords
        
    Returns:
        List of reference strings
    """
    references = []
    
    # Map page names to reference categories
    page_mapping = {
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
        "Fuzzy Control": ["fuzzy_control"]
    }
    
    # Get references for the current page
    if page_name in page_mapping:
        for ref_key in page_mapping[page_name]:
            if ref_key in CURATED_REFERENCES:
                references.extend(CURATED_REFERENCES[ref_key])
    
    # Add general bioprocess references if no specific ones found
    if not references:
        references.extend(CURATED_REFERENCES["bioprocess"])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_refs = []
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    
    return unique_refs[:5]  # Return max 5 references


def check_ollama_availability(base_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, str]:
    """
    Check if Ollama service is available.
    
    Args:
        base_url: Base URL for Ollama API
        
    Returns:
        Tuple of (is_available: bool, message: str)
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            return True, f"Conectado. Modelos disponibles: {', '.join(models[:3])}"
        else:
            return False, f"Error de conexión (código {response.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "No se puede conectar con Ollama. Asegúrate de que esté ejecutándose (ollama serve)"
    except requests.exceptions.Timeout:
        return False, "Tiempo de espera agotado al conectar con Ollama"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"


def query_ollama(prompt: str, model: str = DEFAULT_MODEL, 
                base_url: str = DEFAULT_OLLAMA_URL,
                temperature: float = 0.7,
                max_tokens: int = 1000) -> Tuple[bool, str]:
    """
    Send a query to Ollama API and get response.
    
    Args:
        prompt: The prompt to send
        model: Model name to use
        base_url: Base URL for Ollama API
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        
    Returns:
        Tuple of (success: bool, response: str)
    """
    try:
        # Prepare the request
        url = f"{base_url}/api/generate"
        data = {
            "model": model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Make the request
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return True, result.get('response', 'Sin respuesta')
        else:
            return False, f"Error del servidor: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "La solicitud tardó demasiado tiempo. Intenta con un modelo más pequeño."
    except requests.exceptions.ConnectionError:
        return False, "No se puede conectar con Ollama. Verifica que esté ejecutándose."
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"


def format_response_with_references(llm_response: str, references: List[str]) -> str:
    """
    Format LLM response with references and disclaimers.
    
    Args:
        llm_response: Raw response from LLM
        references: List of reference strings
        
    Returns:
        Formatted response string
    """
    formatted = f"{llm_response}\n\n"
    formatted += "---\n\n"
    formatted += "**📚 Referencias relevantes:**\n\n"
    for i, ref in enumerate(references, 1):
        formatted += f"{i}. {ref}\n"
    formatted += "\n**⚠️ Nota:** Esta información es orientativa y debe validarse experimentalmente. "
    formatted += "Consulta siempre literatura especializada para tu aplicación específica."
    
    return formatted


def suggest_parameter_ranges(parameter_name: str, model_type: str) -> Dict[str, any]:
    """
    Suggest typical parameter ranges based on literature.
    
    Args:
        parameter_name: Name of the parameter
        model_type: Type of model (e.g., 'Monod', 'PID', etc.)
        
    Returns:
        Dictionary with suggested ranges and units
    """
    # Common bioprocess parameters
    ranges = {
        "mumax": {"min": 0.1, "max": 1.5, "typical": 0.5, "unit": "h⁻¹", 
                  "description": "Tasa específica máxima de crecimiento"},
        "Ks": {"min": 0.01, "max": 5.0, "typical": 0.5, "unit": "g/L",
               "description": "Constante de saturación de Monod"},
        "Yxs": {"min": 0.3, "max": 0.8, "typical": 0.5, "unit": "g_X/g_S",
                "description": "Rendimiento biomasa/sustrato"},
        "kd": {"min": 0.001, "max": 0.05, "typical": 0.01, "unit": "h⁻¹",
               "description": "Constante de muerte celular"},
        "kla": {"min": 10, "max": 300, "typical": 100, "unit": "h⁻¹",
                "description": "Coeficiente de transferencia de oxígeno"},
        # PID parameters
        "Kc": {"min": 0.1, "max": 10.0, "typical": 1.0, "unit": "adim.",
               "description": "Ganancia proporcional del controlador"},
        "Ti": {"min": 0.1, "max": 10.0, "typical": 1.0, "unit": "min",
               "description": "Tiempo integral"},
        "Td": {"min": 0.01, "max": 1.0, "typical": 0.1, "unit": "min",
               "description": "Tiempo derivativo"}
    }
    
    param_lower = parameter_name.lower()
    for key, value in ranges.items():
        if key.lower() in param_lower or param_lower in key.lower():
            return value
    
    return {"min": None, "max": None, "typical": None, "unit": "?",
            "description": "Parámetro no reconocido en la base de datos"}
