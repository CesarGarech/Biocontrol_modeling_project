import streamlit as st
import os
import sys # Necesario para añadir carpetas al path (si Utils está fuera)

# --- Añadir carpetas relevantes al path (Ajusta según tu estructura) ---
# Esto es importante si 'Utils' u otros módulos están en carpetas diferentes
# Obtener el directorio del script actual (main.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Añadir el directorio padre (si Body y Utils están al mismo nivel que main.py o en subcarpetas)
# sys.path.append(os.path.dirname(script_dir))
# Añadir carpetas específicas si es necesario
# sys.path.append(os.path.join(script_dir, 'Body'))
# sys.path.append(os.path.join(script_dir, 'Utils')) # Si Utils es una carpeta separada

# --- Asumiendo que Utils.kinetics está accesible ---
try:
    from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion
except ImportError:
    # Puedes mantener las funciones dummy o simplemente pasar
    # print("Advertencia: Módulo Utils.kinetics no encontrado.")
    pass


# 1. Define la NUEVA estructura del menú jerárquico
menu_structure = {
    "🏠 Home": None,
    "🔬 Models": ["Batch", "Fed-Batch", "Continuous", "Fermentation"],
    "📈 Sensitivity Analysis": None,
    "🔧 Parameter Adjustment": ["Batch Parameter Adjustment", "Fed-Batch Parameter Adjustment", "Fermentation Parameter Adjustment"],
    "📊 State Estimation": None,
    # --- NUEVA ESTRUCTURA PARA CONTROL ---
    "⚙️ Control": {
        # "Regulatorio": ["Temperatura", "pH", "Oxigeno"],
        "Regulatory": ["Temperature", "pH", "Oxygen", "On-Off Feeding"],
        "Advanced": ["RTO", "RTO Ferm", "NMPC"]
    }
}

def main():
    st.set_page_config(page_title="Bioprocess Modeling", layout="wide")

    # --- Navegación en la Barra Lateral (MODIFICADA) ---
    st.sidebar.title("Main Navigation")

    # Widget para seleccionar la categoría principal
    main_category = st.sidebar.selectbox(
        "Select a section:",
        list(menu_structure.keys()),
        key="main_cat_select" # Añadir key por robustez
    )

    sub_options = menu_structure[main_category]
    selected_page = main_category # Página por defecto si no hay submenú

    # --- Lógica MODIFICADA para manejar submenús simples y anidados ---
    if isinstance(sub_options, list): # Caso: Submenú de un nivel (ej. Modelos)
        st.sidebar.markdown("---")
        sub_selection = st.sidebar.radio(
             # Usar split() con cuidado, asegurarse que el emoji no interfiera
            f"Detail - {main_category.split(' ')[-1]}:", # Título dinámico
            sub_options,
            key=f"radio_sub_{main_category.replace(' ', '_')}" # Clave única
        )
        selected_page = sub_selection # La página final es la sub-opción

    elif isinstance(sub_options, dict): # Caso: Submenú de dos niveles (ej. Control)
        st.sidebar.markdown("---")
        # Primer Nivel de Submenú (Regulatorio / Avanzado)
        sub_level1_selection = st.sidebar.selectbox(
            f"Type - {main_category.split(' ')[-1]}:",
            list(sub_options.keys()), # Obtiene ["Regulatorio", "Avanzado"]
            key=f"select_sub1_{main_category.replace(' ', '_')}"
        )

        # Segundo Nivel de Submenú (Opciones dentro de Regulatorio o Avanzado)
        sub_level2_options = sub_options[sub_level1_selection]
        if sub_level2_options: # Asegurarse que hay opciones de segundo nivel
            st.sidebar.markdown("---") # Otro separador
            sub_level2_selection = st.sidebar.radio(
                f"Option - {sub_level1_selection}:",
                sub_level2_options, # Lista como ["Temperatura", "pH", ...] o ["RTO", ...]
                key=f"radio_sub2_{main_category.replace(' ', '_')}_{sub_level1_selection}"
            )
            selected_page = sub_level2_selection # La página final es la del segundo nivel

    # --- Fin Navegación ---


    # --- Carga de la Página Seleccionada (AÑADIR NUEVAS PÁGINAS DE CONTROL) ---
    # La lógica if/elif ahora busca el valor final de 'selected_page'

    st.subheader(f"Selected Page: {selected_page}") # Para depuración
    st.markdown("---")

    # --- Carga dinámica de módulos ---
    try:
        if selected_page == "🏠 Home":
            from Body import home
            home.home_page()
        elif selected_page == "Batch":
            from Body.modeling import lote
            lote.lote_page()
        elif selected_page == "Fed-Batch":
            from Body.modeling import lote_alimentado
            lote_alimentado.lote_alimentado_page()
        elif selected_page == "Continuous":
            from Body.modeling import continuo
            continuo.continuo_page()
        elif selected_page == "Fermentation":
            from Body.modeling import ferm_alcohol
            ferm_alcohol.fermentacion_alcoholica_page()
        elif selected_page == "📈 Sensitivity Analysis":
            from Body import analysis # Asumiendo que existe analysis.py
            analysis.analysis_page() # Asumiendo que tiene esta función
        elif selected_page == "Batch Parameter Adjustment":
            from Body.estimacion_parametros import ajuste_parametros_lote
            ajuste_parametros_lote.ajuste_parametros_page()
        elif selected_page == "Fed-Batch Parameter Adjustment":
            from Body.estimacion_parametros import ajuste_parametros_lote_alim
            ajuste_parametros_lote_alim.ajuste_parametros_fedbatch_page()
        elif selected_page == "Fermentation Parameter Adjustment":
            from Body.estimacion_parametros import ajuste_parametros_ferm
            ajuste_parametros_ferm.ajuste_parametros_ferm_page()
        elif selected_page == "📊 State Estimation":
            from Body.estimation import ekf # Asumiendo que existe ekf.py
            ekf.ekf_page() # Asumiendo que tiene esta función

        # --- PÁGINAS DE CONTROL REGULATORIO ---
        elif selected_page == "Temperature":
            # Asegúrate de tener este archivo y función
            from Body.control.regulatorio import reg_temp
            reg_temp.regulatorio_temperatura_page()
        elif selected_page == "pH":
            # Asegúrate de tener este archivo y función
            from Body.control.regulatorio import reg_ph
            reg_ph.regulatorio_ph_page()
        elif selected_page == "Oxygen":
             # Asegúrate de tener este archivo y función
            from Body.control.regulatorio import reg_oxigeno
            reg_oxigeno.regulatorio_oxigeno_page()
        elif selected_page == "On-Off Feeding":
             # Asegúrate de tener este archivo y función
            from Body.control.regulatorio import reg_feed_onoff
            reg_feed_onoff.regulatorio_feed_onoff_page()

        # --- PÁGINAS DE CONTROL AVANZADO (YA EXISTENTES) ---
        elif selected_page == "RTO":
            from Body.control.avanzado import rto
            rto.rto_page()
        elif selected_page == "RTO Ferm":
            from Body.control.avanzado import rto_ferm
            rto_ferm.rto_fermentation_page()
        elif selected_page == "NMPC":
            from Body.control.avanzado import nmpc
            nmpc.nmpc_page()
        else:
            # Mostrar Home o un mensaje si la página no coincide con ninguna carga
             st.warning(f"'{selected_page}' page selected, displaying Home by default or unimplemented page.")
             from Body import home
             home.home_page()

    except ModuleNotFoundError as e:
         st.error(f"Error importing the module for '{selected_page}': {e}")
         st.error(f"Check that the file '{e.name}.py' exist in the correct folder (ej. Body/control/) and has no syntax errors.")
         st.info("Displaying Home Page.")
         from Body import home
         home.home_page()
    except AttributeError as e:
         st.error(f"Error calling the page function for '{selected_page}': {e}")
         st.error(f"Make sure the imported file contains a page function with the correct name (e.g. '{selected_page.lower()}_page()').")
         st.info("Displaying Home Page.")
         from Body import home
         home.home_page()
    except Exception as e:
         st.error(f"An unexpected error occurred while loading the page '{selected_page}':")
         st.exception(e)
         st.info("Displaying Homa Page.")
         from Body import home
         home.home_page()


if __name__ == "__main__":
    main()