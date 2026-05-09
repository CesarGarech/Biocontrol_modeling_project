"""
Digital Twin — Main Router
==========================
Entry point called from main.py. Renders a sub-menu in the sidebar and
dispatches to one of the two Digital Twin sub-pages:
  1. Simulación DWSIM  — feed conditions → design-point scaling
  2. Análisis de Datos — SCADA generation, IQR, MA filter, WLS, KPIs
"""
import streamlit as st


def digital_twin_page():
    st.sidebar.markdown("---")
    dt_option = st.sidebar.radio(
        "Digital Twin — Options:",
        ["DWSIM Simulation", "Data Analysis", "ML Prediction"],
        key="radio_dt_sub",
    )

    if dt_option == "DWSIM Simulation":
        from Body.digital_twin.simulacion_dwsim import simulacion_dwsim_page
        simulacion_dwsim_page()
    elif dt_option == "Data Analysis":
        from Body.digital_twin.analisis_datos import analisis_datos_page
        analisis_datos_page()
    else:
        from Body.digital_twin.ml_prediction import ml_prediction_page
        ml_prediction_page()
