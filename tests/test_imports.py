"""
Tests de importación y ejecución de todos los módulos del proyecto.

Verifica que todos los archivos Python pueden importarse sin errores
de sintaxis o de importación circular. No requiere Streamlit activo.
"""
import sys
import os
import importlib
import pytest

# Raíz del proyecto
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─── Módulos a verificar ─────────────────────────────────────────────────────

MODULES_TO_IMPORT = [
    "Utils.kinetics",
    "Utils.llm_helper",
    "Body.modeling.lote",
    "Body.modeling.lote_alimentado",
    "Body.modeling.continuo",
    "Body.modeling.ferm_alcohol",
    "Body.estimation.ekf",
    "Body.estimation.ann",
    "Body.control.regulatorio.reg_temp",
    "Body.control.regulatorio.reg_ph",
    "Body.control.regulatorio.reg_oxigeno",
    "Body.control.regulatorio.reg_cascade_oxigen",
    "Body.control.regulatorio.reg_feed_onoff",
    "Body.control.regulatorio.reg_ident",
    "Body.control.avanzado.rto",
    "Body.control.avanzado.rto_ferm",
    "Body.control.avanzado.nmpc",
    "Body.control.avanzado.lmpc",
    "Body.control.avanzado.ekf_nmpc",
    "Body.control.avanzado.fuzzy_control",
    "Body.digital_twin.gemelo_digital",
    "Body.digital_twin.ml_prediction",
    "Body.parameter_estimation.ajuste_parametros_lote",
    "Body.parameter_estimation.ajuste_parametros_lote_alim",
    "Body.parameter_estimation.ajuste_parametros_ferm",
    "Body.analysis",
    "Body.home",
]


def _mock_streamlit():
    """
    Instala un mock mínimo de streamlit para que los módulos puedan
    importarse sin levantar la aplicación Streamlit.
    """
    import types
    import unittest.mock as mock

    if "streamlit" not in sys.modules:
        st_mock = types.ModuleType("streamlit")
        # Attributes que los módulos usan a nivel de módulo
        for attr in [
            "header", "subheader", "markdown", "latex", "info", "warning",
            "error", "success", "write", "caption", "stop", "sidebar",
            "expander", "columns", "tabs", "button", "selectbox", "radio",
            "slider", "number_input", "text_input", "file_uploader", "spinner",
            "pyplot", "dataframe", "metric", "set_page_config", "exception",
            "checkbox", "multiselect",
        ]:
            setattr(st_mock, attr, mock.MagicMock())

        # st.sidebar también es un objeto con los mismos métodos
        sidebar_mock = types.SimpleNamespace()
        for attr in [
            "header", "subheader", "markdown", "latex", "slider",
            "number_input", "selectbox", "radio", "button", "expander",
            "title", "caption", "checkbox",
        ]:
            setattr(sidebar_mock, attr, mock.MagicMock())
        st_mock.sidebar = sidebar_mock

        sys.modules["streamlit"] = st_mock


@pytest.fixture(scope="module", autouse=True)
def setup_mocks():
    """Configura los mocks necesarios antes de ejecutar los tests."""
    _mock_streamlit()


@pytest.mark.parametrize("module_path", MODULES_TO_IMPORT)
def test_module_importable(module_path):
    """
    Verifica que el módulo se puede importar sin lanzar excepción.
    Los módulos de UI (streamlit) pueden fallar en tiempo de ejecución
    pero no deben tener errores de sintaxis ni imports rotos.
    """
    try:
        mod = importlib.import_module(module_path)
        assert mod is not None
    except ImportError as exc:
        # Permitir fallos por dependencias opcionales pesadas (tensorflow, skfuzzy)
        optional_heavy = ["tensorflow", "skfuzzy", "pythonnet", "clr",
                          "networkx", "System", "dwsim"]
        if any(dep in str(exc) for dep in optional_heavy):
            pytest.skip(f"Dependencia opcional no instalada: {exc}")
        else:
            pytest.fail(f"ImportError inesperado en {module_path}: {exc}")
    except SyntaxError as exc:
        pytest.fail(f"Error de sintaxis en {module_path}: {exc}")
    except Exception as exc:
        # Otros errores de ejecución (ej. configuración de página streamlit)
        # son aceptables en tiempo de importación si no son errores de código
        error_msg = str(exc)
        runtime_ok = [
            "set_page_config",
            "cannot set",
            "session_state",
            "ScriptRunContext",
        ]
        if any(msg in error_msg for msg in runtime_ok):
            pass  # Ignorar errores de contexto de Streamlit
        else:
            pytest.fail(f"Error inesperado al importar {module_path}: {type(exc).__name__}: {exc}")


# ─── Tests de existencia de funciones clave ──────────────────────────────────

class TestFunctionPresence:
    """Verifica que las funciones de página existen con los nombres correctos."""

    def test_kinetics_functions_exist(self):
        """Todas las funciones cinéticas deben estar en Utils.kinetics."""
        from Utils.kinetics import (
            mu_monod, mu_sigmoidal, mu_completa, aiba,
            mu_fermentacion, mu_fermentacion_rto,
            mu_monod_rto, mu_sigmoidal_rto, mu_completa_rto,
        )
        for fn in [mu_monod, mu_sigmoidal, mu_completa, aiba,
                   mu_fermentacion, mu_fermentacion_rto,
                   mu_monod_rto, mu_sigmoidal_rto, mu_completa_rto]:
            assert callable(fn)

    def test_kinetics_return_non_negative(self):
        """Todas las funciones cinéticas deben retornar valores ≥ 0."""
        from Utils.kinetics import mu_monod, mu_sigmoidal, mu_completa, mu_fermentacion
        assert mu_monod(0.0, 0.5, 0.1) >= 0
        assert mu_sigmoidal(0.0, 0.5, 0.1, 2) >= 0
        assert mu_completa(0.0, 0.0, 0.0, 0.5, 0.1, 0.5, 50.0) >= 0
        assert mu_fermentacion(
            0.0, 0.0, 0.0,
            0.4, 0.5, 0.2,
            0.15, 1.0, 150.0, 80.0, 1.0, 0.1
        ) >= 0
