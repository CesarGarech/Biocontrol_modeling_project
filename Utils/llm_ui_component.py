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
        "Enable AI Assistant",
        value=st.session_state.llm_enabled,
        key="llm_enable_checkbox",
        help="Enable the Ollama-based assistant for contextual help"
    )
    st.session_state.llm_enabled = llm_enabled
    
    if not llm_enabled:
        st.sidebar.info("📖 The assistant is disabled. Enable it to get help with equations, methods, and parameters.")
        return
    
    # Configuration expander
    with st.sidebar.expander("⚙️ Settings", expanded=False):
        # Ollama URL
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.llm_ollama_url,
            help="Ollama server URL (default: http://localhost:11434)"
        )
        st.session_state.llm_ollama_url = ollama_url
        
        # Model selection
        model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.llm_model) if st.session_state.llm_model in AVAILABLE_MODELS else 0,
            help="Select the Ollama model to use. Smaller models are faster."
        )
        st.session_state.llm_model = model
        
        # Check connection button
        if st.button("🔍 Check Connection", key="llm_check_connection"):
            with st.spinner("Checking..."):
                is_available, message = check_ollama_availability(ollama_url)
                if is_available:
                    st.success(f"✅ {message}")
                    st.session_state.llm_last_check = "success"
                else:
                    st.error(f"❌ {message}")
                    st.session_state.llm_last_check = "failed"
        
        # Show last check status
        if st.session_state.llm_last_check == "success":
            st.caption("✅ Last check: OK")
        elif st.session_state.llm_last_check == "failed":
            st.caption("❌ Last check: Failed")
    
    # Disclaimer
    st.sidebar.info("ℹ️ **Educational Use:** Responses are for guidance and must be validated.")
    
    # Quick actions
    st.sidebar.markdown("**Quick Actions:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📖 Explain method", key="llm_explain_method"):
            question = f"Explain the method or model used in the '{current_page}' page clearly and educationally."
            _process_question(question, current_page)
    
    with col2:
        if st.button("📊 Suggest params", key="llm_suggest_params"):
            question = f"Suggest typical parameter ranges for the model/method in the '{current_page}' page."
            _process_question(question, current_page)
    
    # Chat interface
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💬 Ask the assistant:**")
    
    user_question = st.sidebar.text_area(
        "Your question:",
        height=100,
        key="llm_user_question",
        placeholder="Ex: What does the Ks parameter mean in the Monod equation?"
    )
    
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🚀 Ask", key="llm_ask_button", use_container_width=True):
            if user_question.strip():
                _process_question(user_question, current_page)
            else:
                st.sidebar.warning("Please write a question.")
    
    with col2:
        if st.button("🗑️ Clear", key="llm_clear_button", use_container_width=True):
            st.session_state.llm_chat_history = []
            st.rerun()
    
    # Show chat history
    if st.session_state.llm_chat_history:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📜 History:**")
        
        # Show only last 3 interactions
        for i, interaction in enumerate(st.session_state.llm_chat_history[-3:]):
            # Calculate question number for display
            question_number = len(st.session_state.llm_chat_history) - 3 + i + 1
            question_preview = interaction['question'][:50]
            
            with st.sidebar.expander(f"Q{question_number}: {question_preview}...", expanded=(i == len(st.session_state.llm_chat_history[-3:])-1)):
                st.markdown(f"**Q:** {interaction['question']}")
                st.markdown(f"**A:** {interaction['response']}")


def _process_question(question: str, current_page: str):
    """
    Process a question and get response from LLM.
    
    Args:
        question: User's question
        current_page: Current page name for context
    """
    with st.sidebar.spinner("🤔 Thinking..."):
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
            
            st.sidebar.success("✅ Response generated")
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
        
        with st.expander("💬 Last response from AI Assistant", expanded=True):
            st.markdown(f"**Question:** {last_interaction['question']}")
            st.markdown("**Answer:**")
            st.markdown(last_interaction['response'])