# -*- coding: utf-8 -*-
"""
LLM UI Component for Streamlit Sidebar
Provides the AI Guide interface
"""

import streamlit as st
from typing import Optional
from Utils.llm_helper import (
    check_ollama_availability,
    query_ollama,
    build_context_prompt,
    get_relevant_references,
    format_response_with_references,
    suggest_parameter_ranges,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL
)


def initialize_llm_session_state():
    """Initialize session state variables for LLM component."""
    if 'llm_enabled' not in st.session_state:
        st.session_state.llm_enabled = False
    if 'llm_chat_history' not in st.session_state:
        st.session_state.llm_chat_history = []
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = DEFAULT_MODEL
    if 'llm_ollama_url' not in st.session_state:
        st.session_state.llm_ollama_url = DEFAULT_OLLAMA_URL
    if 'llm_last_check' not in st.session_state:
        st.session_state.llm_last_check = None


def render_llm_sidebar(current_page: str):
    """
    Render the LLM AI Guide component in the sidebar.
    
    Args:
        current_page: Name of the current page for context
    """
    initialize_llm_session_state()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Guide (Beta)")
    
    # Enable/disable toggle
    llm_enabled = st.sidebar.checkbox(
        "Activar Asistente IA",
        value=st.session_state.llm_enabled,
        key="llm_enable_checkbox",
        help="Activa el asistente basado en Ollama para obtener ayuda contextual"
    )
    st.session_state.llm_enabled = llm_enabled
    
    if not llm_enabled:
        st.sidebar.info("📖 El asistente está desactivado. Actívalo para obtener ayuda con ecuaciones, métodos y parámetros.")
        return
    
    # Configuration expander
    with st.sidebar.expander("⚙️ Configuración", expanded=False):
        # Ollama URL
        ollama_url = st.text_input(
            "URL de Ollama",
            value=st.session_state.llm_ollama_url,
            help="URL del servidor Ollama (por defecto: http://localhost:11434)"
        )
        st.session_state.llm_ollama_url = ollama_url
        
        # Model selection
        model = st.selectbox(
            "Modelo",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.llm_model) if st.session_state.llm_model in AVAILABLE_MODELS else 0,
            help="Selecciona el modelo de Ollama a usar. Modelos más pequeños son más rápidos."
        )
        st.session_state.llm_model = model
        
        # Check connection button
        if st.button("🔍 Verificar Conexión", key="llm_check_connection"):
            with st.spinner("Verificando..."):
                is_available, message = check_ollama_availability(ollama_url)
                if is_available:
                    st.success(f"✅ {message}")
                    st.session_state.llm_last_check = "success"
                else:
                    st.error(f"❌ {message}")
                    st.session_state.llm_last_check = "failed"
        
        # Show last check status
        if st.session_state.llm_last_check == "success":
            st.caption("✅ Última verificación: OK")
        elif st.session_state.llm_last_check == "failed":
            st.caption("❌ Última verificación: Falló")
    
    # Disclaimer
    st.sidebar.info("ℹ️ **Uso educativo:** Las respuestas son orientativas y deben validarse.")
    
    # Quick actions
    st.sidebar.markdown("**Acciones rápidas:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📖 Explicar método", key="llm_explain_method"):
            question = f"Explica el método o modelo usado en la página '{current_page}' de forma educativa y clara."
            _process_question(question, current_page)
    
    with col2:
        if st.button("📊 Sugerir parámetros", key="llm_suggest_params"):
            question = f"Sugiere rangos típicos de parámetros para el modelo/método en la página '{current_page}'."
            _process_question(question, current_page)
    
    # Chat interface
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💬 Pregunta al asistente:**")
    
    user_question = st.sidebar.text_area(
        "Tu pregunta:",
        height=100,
        key="llm_user_question",
        placeholder="Ej: ¿Qué significa el parámetro Ks en la ecuación de Monod?"
    )
    
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🚀 Preguntar", key="llm_ask_button", use_container_width=True):
            if user_question.strip():
                _process_question(user_question, current_page)
            else:
                st.sidebar.warning("Por favor escribe una pregunta.")
    
    with col2:
        if st.button("🗑️ Limpiar", key="llm_clear_button", use_container_width=True):
            st.session_state.llm_chat_history = []
            st.rerun()
    
    # Show chat history
    if st.session_state.llm_chat_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📜 Historial:**")
        
        # Show only last 3 interactions
        for i, interaction in enumerate(st.session_state.llm_chat_history[-3:]):
            with st.sidebar.expander(f"Q{len(st.session_state.llm_chat_history)-3+i+1}: {interaction['question'][:50]}...", expanded=(i == len(st.session_state.llm_chat_history[-3:])-1)):
                st.markdown(f"**P:** {interaction['question']}")
                st.markdown(f"**R:** {interaction['response']}")


def _process_question(question: str, current_page: str):
    """
    Process a question and get response from LLM.
    
    Args:
        question: User's question
        current_page: Current page name for context
    """
    with st.sidebar.spinner("🤔 Pensando..."):
        # Build context
        prompt = build_context_prompt(
            page_name=current_page,
            user_question=question
        )
        
        # Get references
        references = get_relevant_references(current_page, [])
        
        # Query LLM
        success, response = query_ollama(
            prompt=prompt,
            model=st.session_state.llm_model,
            base_url=st.session_state.llm_ollama_url
        )
        
        if success:
            # Format with references
            formatted_response = format_response_with_references(response, references)
            
            # Add to history
            st.session_state.llm_chat_history.append({
                'question': question,
                'response': formatted_response,
                'page': current_page
            })
            
            st.sidebar.success("✅ Respuesta generada")
            st.rerun()
        else:
            st.sidebar.error(f"❌ Error: {response}")


def show_llm_response_in_main_area():
    """
    Optionally show the last LLM response in the main content area.
    Call this from the page if you want to show responses there instead of sidebar.
    """
    if st.session_state.get('llm_enabled', False) and st.session_state.llm_chat_history:
        last_interaction = st.session_state.llm_chat_history[-1]
        
        with st.expander("💬 Última respuesta del Asistente IA", expanded=True):
            st.markdown(f"**Pregunta:** {last_interaction['question']}")
            st.markdown("**Respuesta:**")
            st.markdown(last_interaction['response'])
