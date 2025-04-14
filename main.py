import streamlit as st
import os

def main():
    st.set_page_config(page_title="Modelado de Bioprocesos", layout="wide")

    menu = st.sidebar.selectbox(
        "Seleccione una opción",
        [
            "Home",
            "Lote",
            "Lote Alimentado",
            "Continuo",
            "Análisis de Sensibilidad",
            "Ajuste de Parámetros",
            "Estimacion de estados",
            "Control RTO",
            "Control NMPC",
        ],
    )

    if menu == "Home":
        from Body import home
        home.home_page()  # Call the home_page function
    elif menu == "Lote":
        from Body.modeling import lote
        lote.lote_page()
    elif menu == "Lote Alimentado":
        from Body.modeling import lote_alimentado
        lote_alimentado.lote_alimentado_page()  # Create this file
    elif menu == "Continuo":
        from Body.modeling import continuo
        continuo.continuo_page()  # Create this file
    elif menu == "Análisis de Sensibilidad":
        from Body import analysis
        analysis.analysis_page() # Create this file
    elif menu == "Ajuste de Parámetros":
        from Body import ajuste_parametros
        ajuste_parametros.ajuste_parametros_page() # Create this file
    elif menu == "Estimacion de estados":
        from Body.estimation import ekf
        ekf.ekf_page() # Create this file
    elif menu == "Control RTO":
        from Body.control import rto
        rto.rto_page() # Create this file
    elif menu == "Control NMPC":
        from Body.control import nmpc
        nmpc.nmpc_page() # Create this file

if __name__ == "__main__":
    main()