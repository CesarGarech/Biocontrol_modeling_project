import streamlit as st
import os
import sys

# --- Add relevant folders to path (Adjust according to your structure) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(script_dir))
# sys.path.append(os.path.join(script_dir, 'Body'))
# sys.path.append(os.path.join(script_dir, 'Utils'))

# --- Assuming that Utils.kinetics is accessible ---
try:
    from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion
except ImportError:
    pass

# 1. Define the NEW hierarchical menu structure
# --- MODIFICATION HERE ---
menu_structure = {
    "🏠 Home": None,
    "🔬 Models": ["Batch", "Fed-Batch", "Continuous", "Fermentation"],
    "📈 Sensitivity Analysis": None,
    "🔧 Parameter Adjustment": ["Batch Parameter Adjustment", "Fed-Batch Parameter Adjustment", "Fermentation Parameter Adjustment"],
    # La clave "State Estimation" ahora contiene una lista para crear un submenú
    "📊 State Estimation": ["EKF", "ANN"],
    "⚙️ Control": {
        "Regulatory": ["Identification (pH)","Temperature", "pH", "Oxygen", "Cascade-Oxygen", "On-Off Feeding"],
        "Advanced": ["RTO", "RTO Ferm", "NMPC"]
    }
}

def main():
    st.set_page_config(page_title="Bioprocess Modeling", layout="wide")

    # --- Navigation in the Sidebar (no changes needed here) ---
    st.sidebar.title("Main Navigation")

    # Widget para seleccionar la categoría principal
    main_category = st.sidebar.selectbox(
        "Select a section:",
        list(menu_structure.keys()),
        key="main_cat_select"
    )

    sub_options = menu_structure[main_category]
    selected_page = main_category

    # La lógica existente ya maneja submenús basados en listas, por lo que
    # "State Estimation" funcionará automáticamente sin cambios en esta sección.
    if isinstance(sub_options, list):
        st.sidebar.markdown("---")
        sub_selection = st.sidebar.radio(
            f"Detail - {main_category.split(' ')[-1]}:",
            sub_options,
            key=f"radio_sub_{main_category.replace(' ', '_')}"
        )
        selected_page = sub_selection

    elif isinstance(sub_options, dict):
        st.sidebar.markdown("---")
        sub_level1_selection = st.sidebar.selectbox(
            f"Type - {main_category.split(' ')[-1]}:",
            list(sub_options.keys()),
            key=f"select_sub1_{main_category.replace(' ', '_')}"
        )

        sub_level2_options = sub_options[sub_level1_selection]
        if sub_level2_options:
            st.sidebar.markdown("---")
            sub_level2_selection = st.sidebar.radio(
                f"Option - {sub_level1_selection}:",
                sub_level2_options,
                key=f"radio_sub2_{main_category.replace(' ', '_')}_{sub_level1_selection}"
            )
            selected_page = sub_level2_selection

    # --- End Navigation ---


    # --- Loading Selected Page (ADD NEW CONTROL PAGES) ---
    st.subheader(f"Selected Page: {selected_page}")
    st.markdown("---")

    # --- Dynamic module loading ---
    try:
        if selected_page == "🏠 Home":
            from Body import home
            home.home_page()
        elif selected_page == "Batch":
            from Body.modeling import batch
            batch.batch_page()
        elif selected_page == "Fed-Batch":
            from Body.modeling import fed_batch
            fed_batch.fed_batch_page()
        elif selected_page == "Continuous":
            from Body.modeling import continuo
            continuo.continuo_page()
        elif selected_page == "Fermentation":
            from Body.modeling import ferm_alcohol
            ferm_alcohol.fermentacion_alcoholica_page()
        elif selected_page == "📈 Sensitivity Analysis":
            from Body import analysis
            analysis.analysis_page()
        elif selected_page == "Batch Parameter Adjustment":
            from Body.parameter_estimation import parameter_fitting_batch
            parameter_fitting_batch.parameter_fitting_page()
        elif selected_page == "Fed-Batch Parameter Adjustment":
            from Body.parameter_estimation import parameter_fitting_fed_batch
            parameter_fitting_fed_batch.parameter_fitting_fedbatch_page()
        elif selected_page == "Fermentation Parameter Adjustment":
            from Body.parameter_estimation import parameter_fitting_fermentation
            parameter_fitting_fermentation.parameter_fitting_ferm_page()

        # --- MODIFICATION IN LOADING LOGIC ---
        # The previous condition for "State Estimation" is removed and replaced by the following two.
        
        elif selected_page == "EKF":
            # We assume the EKF page is in Body/estimation/ekf.py with an ekf_page() function
            from Body.estimation import ekf
            ekf.ekf_page()

        elif selected_page == "ANN":
            # We assume the new ANN page will be in Body/estimation/ann.py with an ann_page() function
            # This is the module we will create next.
            from Body.estimation import ann
            ann.ann_page()

        # --- REGULATORY CONTROL PAGES ---
        elif selected_page == "Identification (pH)":
            from Body.control.regulatorio import reg_ident
            reg_ident.ph_identification_page()
        elif selected_page == "Temperature":
            from Body.control.regulatorio import reg_temp
            reg_temp.regulatorio_temperatura_page()
        elif selected_page == "pH":
            from Body.control.regulatorio import reg_ph
            reg_ph.regulatorio_ph_page()
        elif selected_page == "Oxygen":
            from Body.control.regulatorio import reg_oxygen
            reg_oxygen.regulatorio_oxigeno_page()
        elif selected_page == "Cascade-Oxygen": # Exact name to use in the menu
             from Body.control.regulatorio import reg_cascade_oxigen
             reg_cascade_oxigen.regulatorio_cascade_oxigen_page()
        elif selected_page == "On-Off Feeding":
            from Body.control.regulatorio import reg_feed_onoff
            reg_feed_onoff.regulatorio_feed_onoff_page()

        # --- ADVANCED CONTROL PAGES (ALREADY EXISTING) ---
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
            st.warning(f"'{selected_page}' page selected, but no specific loader found. Displaying Home.")
            from Body import home
            home.home_page()

    except ModuleNotFoundError as e:
         st.error(f"Error importing the module for '{selected_page}': {e}")
         st.error(f"Check that the file '{e.name.replace('.', '/')}.py' exists in the correct folder (e.g., Body/estimation/) and has no syntax errors.")
         st.info("Displaying Home Page as a fallback.")
         from Body import home
         home.home_page()
    except AttributeError as e:
         st.error(f"Error calling the page function for '{selected_page}': {e}")
         st.error(f"Make sure the imported file contains a page function with the correct name (e.g., '{selected_page.lower()}_page()').")
         st.info("Displaying Home Page as a fallback.")
         from Body import home
         home.home_page()
    except Exception as e:
         st.error(f"An unexpected error occurred while loading the page '{selected_page}':")
         st.exception(e)
         st.info("Displaying Home Page as a fallback.")
         from Body import home
         home.home_page()

if __name__ == "__main__":
    main()

