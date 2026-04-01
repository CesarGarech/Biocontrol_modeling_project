import streamlit as st



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Definir la página inicial
if "page" not in st.session_state:
    st.session_state.page = "Home"
    
    


# ----------------------
# Diseño de la pantalla
# ----------------------
if st.session_state.page == "Home":
    st.image("Images/LogoSemillero.png", width=300)
    st.markdown("<div id='tituloMenu'>🏠 HOME</div>", unsafe_allow_html=True)
    st.markdown("""<div id='textoMenu'> Welcome to the interactive simulator for bioprocess modeling and control. 
        This tool allows you to explore different reactor operation modes, 
        microbial growth kinetics and advanced control strategies.</div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.image("Images/Modelos.png", width=120)
        if st.button("Models", key="models_btn"):
            st.session_state.page = "Models"

    with col2:
        st.image("Images/analisisSensibilidad.png", width=120)
        if st.button("Sensitivity Analysis", key="sensitivity_btn"):
            st.session_state.page = "Sensitivity Analysis"
            
    with col3:
        st.image("Images/ajusteParametros.png", width=120)
        if st.button("Parameter Adjust", key="parameter_btn"):
            st.session_state.page = "Parameter Adjust"
            
    with col4:
        st.image("Images/estimacionEstados.png", width=120)
        if st.button("State Estimation", key="state_btn"):
            st.session_state.page = "State Estimation"
            
    with col5:
        st.image("Images/control.png", width=120)
        if st.button("Control", key="control_btn"):
            st.session_state.page = "Control"

    with col6:
        st.image("Images/teoria.png", width=120)
        if st.button("Theory", key="config_btn"):
            st.session_state.page = "Theory"



#Redirigiendo a las paginas seleccionadas

#Pagina modelos
elif st.session_state.page == "Models":
    st.title("Availabre Models")

    modelo = st.selectbox(
        "Choose a model:",
        ["Batch", "Fed-Batch", "Continuous", "Fermentation"],
        key="modelo_selector"
    )

    if modelo == "Batch":
        from Body.modeling import lote
        lote.lote_page()
    elif modelo == "Fed-Batch":
        from Body.modeling import lote_alimentado
        lote_alimentado.lote_alimentado_page()
    elif modelo == "Continuous":
        from Body.modeling import continuo
        continuo.continuo_page()
    elif modelo == "Fermentation":
        from Body.modeling import ferm_alcohol
        ferm_alcohol.fermentacion_alcoholica_page()
    
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"

#Pagina analisis de sensibilidad
elif st.session_state.page == "Sensitivity Analysis":
    from Body import analysis
    analysis.analysis_page()
    
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"

#Pagina ajuste de parametros
elif st.session_state.page == "Parameter Adjust":
    st.title("Available Parameter Adjustment Models")
    ajuste = st.selectbox(
            "Choose a model for parameter adjustment:",
            ["Batch Parameter Adjustment", "Fed-Batch Parameter Adjustment","Fermentation Parameter Adjustment"],
            key="ajuste_selector"
        )
    if ajuste == "Batch Parameter Adjustment":
        from Body.estimacion_parametros import ajuste_parametros_lote
        ajuste_parametros_lote.ajuste_parametros_page()
    elif ajuste == "Fed-Batch Parameter Adjustment":
        from Body.estimacion_parametros import ajuste_parametros_lote_alimentado
        ajuste_parametros_lote_alimentado.ajuste_parametros_lote_alimentado_page()
    elif ajuste == "Fermentation Parameter Adjustment":
        from Body.estimacion_parametros import ajuste_parametros_ferm
        ajuste_parametros_ferm.ajuste_parametros_ferm_page()
    
    if st.button("⬅️  Back to Home"):
        st.session_state.page = "Home"

elif st.session_state.page == "State Estimation":
    st.title("State Estimation")
    estimation = st.selectbox("Choose a model for state estimation:",
                              ["Artificial Neural Networks", "Kalman Filter"],key="estimation_selector")
    if estimation == "Artificial Neural Networks":
        from Body.estimation import ann
        ann.ann_page()
    elif estimation == "Kalman Filter":
        from Body.estimation import ekf
        ekf.ekf_page()
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"

elif st.session_state.page == "Control":
    st.title("Control Strategies")
    control = st.selectbox("Choose a control strategy:",
                           ["Advanced Control", "Regulatory Control"],key="control__strategy_selector")
    
    if control == "Advanced Control":
        control_adv=st.selectbox("Choose a model for advanced control:",
                                 ['RTO','RTO Ferm','NMPC'],key="control_adv_selector")
        if control_adv == "RTO":
            from Body.control.avanzado import rto
            rto.rto_page()
        #elif control_adv == "RTO Ferm":
            #from Body.control.avanzado import rto_ferm
            #rto_ferm.rto_ferm_page()
        elif control_adv == "NMPC":
            from Body.control.avanzado import nmpc
            nmpc.nmpc_page()
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            
    elif control =='Regulatory Control':
        control_reg=st.selectbox("Choose a model for regulatory control:",
                                  ['Identification (pH)','Temperature','pH','Oxygen','On-Off Feeding'],key="control_reg_selector")
        if control_reg == "Identification (pH)":
            from Body.control.regulatorio import reg_ident
            reg_ident.ph_identification_page()
        elif control_reg == "Temperature":
            from Body.control.regulatorio import reg_temp
            reg_temp.regulatorio_temperatura_page()
        elif control_reg == "pH":
            from Body.control.regulatorio import reg_ph
            reg_ph.regulatorio_ph_page()
        elif control_reg == "Oxygen":
            from Body.control.regulatorio import reg_oxigeno
            reg_oxigeno.regulatorio_oxigeno_page()
        elif control_reg == "On-Off Feeding":
            from Body.control.regulatorio import reg_feed_onoff
            reg_feed_onoff.regulatorio_feed_onoff_page()
    
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
    
elif st.session_state.page == "Theory":
    from theory import theory_page
    theory_page()
        
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
    