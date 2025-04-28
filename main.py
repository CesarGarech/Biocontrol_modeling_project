import streamlit as st
import os

# Asumiendo que la carpeta con tus módulos se llama 'Body'

# 1. Define la estructura del menú jerárquico
menu_structure = {
    "🏠 Home": None,  # Sin submenú. Usar emojis puede ayudar visualmente.
    "🔬 Modelos": ["Lote", "Lote Alimentado", "Continuo", "Fermentacion"],
    "📈 Análisis de Sensibilidad": None,
    "🔧 Ajuste de Parámetros": None,
    "📊 Estimación de Estados": ["EKF"], # Puedes añadir más estimadores aquí
    # "⚙️ Control": ["RTO", "NMPC"]
    "⚙️ Control": ["RTO", "RTO Ferm", "NMPC"]
}

def main():
    st.set_page_config(page_title="Modelado de Bioprocesos", layout="wide")

    # --- Navegación en la Barra Lateral ---
    st.sidebar.title("Navegación Principal") # Título opcional para la barra lateral

    # Widget para seleccionar la categoría principal
    main_category = st.sidebar.selectbox(
        "Seleccione una sección:",
        list(menu_structure.keys()) # Obtiene las claves principales: Home, Modelos, etc.
    )

    sub_options = menu_structure[main_category]
    selected_page = main_category # Por defecto, si no hay submenú

    # Si la categoría principal tiene sub-opciones, muestra otro widget
    if sub_options:
        st.sidebar.markdown("---") # Separador visual
        # Puedes usar radio si son pocas opciones, o selectbox si son muchas
        # Aquí usamos radio como ejemplo
        sub_selection = st.sidebar.radio(
             f"Detalle - {main_category.split(' ')[1]}:", # Título dinámico, e.g., "Detalle - Modelos:"
             sub_options,
             key=f"radio_{main_category}" # Clave única para evitar problemas de estado
        )
        selected_page = sub_selection # La página final es la sub-opción seleccionada
    # --- Fin Navegación ---


    # --- Carga de la Página Seleccionada ---
    # Ahora, la lógica 'if/elif' se basa en la 'selected_page' final
    # Nota: Los nombres aquí deben coincidir exactamente con las claves (si no hay submenú)
    # o los valores de las listas (si hay submenú).

    if selected_page == "🏠 Home":
        from Body import home
        home.home_page()
    elif selected_page == "Lote":
        from Body.modeling import lote
        lote.lote_page()
    elif selected_page == "Lote Alimentado":
        from Body.modeling import lote_alimentado
        lote_alimentado.lote_alimentado_page()
    elif selected_page == "Continuo":
        from Body.modeling import continuo
        continuo.continuo_page()
    elif selected_page == "Fermentacion":
        from Body.modeling import ferm_alcohol
        ferm_alcohol.fermentacion_alcoholica_page()    
    elif selected_page == "📈 Análisis de Sensibilidad":
        from Body import analysis
        analysis.analysis_page()
    elif selected_page == "🔧 Ajuste de Parámetros":
        from Body import ajuste_parametros
        ajuste_parametros.ajuste_parametros_page()
    elif selected_page == "EKF": # Nombre de la sub-opción
        from Body.estimation import ekf
        ekf.ekf_page()
    elif selected_page == "RTO": # Nombre de la sub-opción
        from Body.control import rto
        rto.rto_page()
    elif selected_page == "RTO Ferm": # Nombre de la sub-opción
        from Body.control import rto_ferm
        rto_ferm.rto_fermentation_page()
    elif selected_page == "NMPC": # Nombre de la sub-opción
        from Body.control import nmpc
        nmpc.nmpc_page()
    # Puedes añadir un 'else' para manejar casos no esperados si lo deseas
    # else:
    #     st.error(f"Página '{selected_page}' no implementada.")

if __name__ == "__main__":
    main()