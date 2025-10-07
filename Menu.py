import streamlit as st
import base64

st.set_page_config(layout="wide")

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("style.css")

# Definir la página inicial
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Revisar query params para cambiar de página
query_page = st.query_params.get("page")
if query_page and query_page != st.session_state.page:
    st.session_state.page = query_page
    st.query_params.clear()
    st.rerun()
    
# ----------------------
# Diseño de la pantalla
# ----------------------
if st.session_state.page == "Home":
    st.markdown("<h1 id='titulo'>Innovacoding</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""<div id='textoMenu'> Lorem ipsum dolor sit amet consectetur adipisicing elit. Ipsa, sequi laborum modi impedit tempora asperiores eligendi omnis id dignissimos molestias rem, optio sed aliquam. Id recusandae autem libero soluta ipsam?" \
        "Lorem ipsum dolor sit amet consectetur adipisicing elit. Ipsa, sequi laborum modi impedit tempora asperiores eligendi omnis id dignissimos molestias rem, optio sed aliquam. Id recusandae autem libero soluta ipsam? </div>""", unsafe_allow_html=True)

    with col2:
        st.image("Images/LogoSemillero.png",width="stretch")
        st.caption("**Figure 1:** InnovaCoding Logo")
    st.markdown("<div id='tituloMenu'>🏠 HOME</div>", unsafe_allow_html=True)
    st.markdown("""<div id='textoMenu'> Welcome to the interactive simulator for bioprocess modeling and control. 
        This tool allows you to explore different reactor operation modes, 
        microbial growth kinetics and advanced control strategies.</div>""", unsafe_allow_html=True)

    # convertir imagen
    def img_to_base64(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return ""
    # tarjeta con imagen para el menu
    def image_card_button(img_path, label, target_page, key, text_color="#1e1e1e"):
        img_b64 = img_to_base64(img_path)
        
        if not img_b64:
            st.warning(f"Image not found: {img_path}")
            return

        st.markdown(
            f"""
            <a href="?page={target_page}" class="image-card {key}">
                <img src="data:image/png;base64,{img_b64}" alt="{label}">
                <div class="label">{label}</div>
            </a>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------
    # Distribución en columnas
    # ----------------------
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        image_card_button("Images/Modelos.png", "Models", "Models", key="models")

    with col2:
        image_card_button("Images/analisisSensibilidad.png", "Sensitivity Analysis", "Sensitivity Analysis", key="analysis")

    with col3:
        image_card_button("Images/ajusteParametros.png", "Parameter Adjust", "Parameter Adjust", key="adjust")

    with col4:
        image_card_button("Images/estimacionEstados.png", "State Estimation", "State Estimation", key="estimation")

    with col5:
        image_card_button("Images/control.png", "Control", "Control", key="control")

    with col6:
        image_card_button("Images/teoria.png", "Theory", "Theory", key="theory")


#Redirigiendo a las paginas seleccionadas

#Pagina modelos
elif st.session_state.page == "Models":
    from Body.modeling import lote, lote_alimentado, continuo, ferm_alcohol
    st.title("Available Models")

    modelo = st.selectbox(
        "Choose a model:",
        ["Batch", "Fed-Batch", "Continuous", "Fermentation"],
        key="modelo_selector"
    )

    if modelo == "Batch":
        lote.lote_page()
    elif modelo == "Fed-Batch":
        lote_alimentado.lote_alimentado_page()
    elif modelo == "Continuous":
        continuo.continuo_page()
    elif modelo == "Fermentation":
        ferm_alcohol.fermentacion_alcoholica_page()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

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
        from Body.estimacion_parametros import ajuste_parametros_lote_alim
        ajuste_parametros_lote_alim.ajuste_parametros_fedbatch_page()
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
    