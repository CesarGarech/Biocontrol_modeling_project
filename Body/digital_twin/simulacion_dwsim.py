"""
Digital Twin — Sub-page 1: DWSIM Interactive Simulation
========================================================
Permite al usuario especificar condiciones de la corriente de alimentación
y parámetros de la columna de destilación.  Al presionar "🚀 Run Simulation"
se intenta conectar con DWSIM vía DWSIMInterface; si DWSIM no está disponible
se usa el escalado analítico del punto de diseño como alternativa.
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Agregar carpeta Simulation al path ───────────────────────────────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── Importar interfaz DWSIM (con manejo seguro si pythonnet no está disponible)
try:
    from dwsim_interface import DWSIMInterface, DWSIMInterfaceError  # noqa: E402
    from dwsim_data_generator import validate_dwsim_installation     # noqa: E402
    _DWSIM_IMPORTS_OK = True
except Exception:
    _DWSIM_IMPORTS_OK = False
    DWSIMInterface = None          # type: ignore[assignment,misc]
    DWSIMInterfaceError = RuntimeError  # type: ignore[assignment,misc]

    def validate_dwsim_installation():  # type: ignore[misc]
        """Stub cuando los módulos DWSIM no están disponibles."""
        return False, "Módulos DWSIM no pudieron importarse."

# ── Constantes locales ────────────────────────────────────────────────────────
_TARGET_ETHANOL_BOTTOM = 0.02    # fracción molar de etanol en fondos (diseño)
_MW_ETHANOL = 46.068             # g/mol
_MW_WATER = 18.015               # g/mol

_FLOW_FEED_BASE_KGH = _cfg.FLOW_FEED_BASE
_SPLIT_TOP = _cfg.SPLIT_TOP
_SPLIT_BOTTOM = _cfg.SPLIT_BOTTOM
_Q_COND_BASE_KW = _cfg.Q_COND_BASE
_Q_REB_BASE_KW = _cfg.Q_REB_BASE
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP


def _average_mw(x_eth: float) -> float:
    """Peso molecular promedio para mezcla etanol-agua."""
    return x_eth * _MW_ETHANOL + (1.0 - x_eth) * _MW_WATER


def _run_analytic_simulation(
    F_feed_kmolh: float,
    T_feed: float,
    P_feed_bar: float,
    x_eth: float,
    reflux_ratio_col: float,
) -> dict:
    """
    Calcula resultados de la columna usando escalado analítico del punto de diseño.

    Parámetros
    ----------
    F_feed_kmolh : float
        Flujo molar de alimentación (kmol/h).
    T_feed : float
        Temperatura de alimentación (°C).
    P_feed_bar : float
        Presión de alimentación (bar).
    x_eth : float
        Fracción molar de etanol en la alimentación.
    reflux_ratio_col : float
        Razón de reflujo de la columna.

    Retorna
    -------
    dict con resultados del escalado.
    """
    mw_feed = _average_mw(x_eth)
    F_feed_kgh = F_feed_kmolh * mw_feed
    scale = F_feed_kgh / _FLOW_FEED_BASE_KGH

    F_top_kgh = F_feed_kgh * _SPLIT_TOP
    F_bot_kgh = F_feed_kgh * _SPLIT_BOTTOM
    mw_top = _average_mw(_TARGET_ETHANOL_TOP)
    mw_bot = _average_mw(_TARGET_ETHANOL_BOTTOM)
    F_top_kmolh = F_top_kgh / mw_top
    F_bot_kmolh = F_bot_kgh / mw_bot

    Q_cond_kw = _Q_COND_BASE_KW * scale
    Q_reb_kw = _Q_REB_BASE_KW * scale

    dh_vap_ethanol = 38.56
    dh_vap_water = 40.65
    dh_vap_top = (
        _TARGET_ETHANOL_TOP * dh_vap_ethanol
        + (1 - _TARGET_ETHANOL_TOP) * dh_vap_water
    )
    V_vapour = Q_cond_kw / dh_vap_top
    rr_calc = max(0.0, (V_vapour - F_top_kmolh) / F_top_kmolh) if F_top_kmolh > 0 else 0.0

    ethanol_recovery = (
        F_top_kmolh * _TARGET_ETHANOL_TOP
        / max(F_feed_kmolh * x_eth, 1e-9) * 100
    )

    T_bp = 78.4 + (P_feed_bar * 10 - 101.325) * 0.03
    cp_liquid = 2.9
    lambda_feed = dh_vap_top * mw_feed / 1000.0
    q = 1.0 + cp_liquid * max(T_bp - T_feed, 0.0) / lambda_feed

    # Estimación de etapas teóricas con razón de reflujo (Fenske simplificado)
    n_stages = max(5, int(10 * reflux_ratio_col / max(rr_calc, 0.1)))

    return {
        "source": "analytic",
        "F_feed_kmolh": F_feed_kmolh,
        "F_feed_kgh": F_feed_kgh,
        "F_top_kgh": F_top_kgh,
        "F_bot_kgh": F_bot_kgh,
        "F_top_kmolh": F_top_kmolh,
        "F_bot_kmolh": F_bot_kmolh,
        "T_feed_C": T_feed,
        "T_top_C": 78.4,
        "T_bot_C": 100.0,
        "P_feed_bar": P_feed_bar,
        "Q_cond_kw": Q_cond_kw,
        "Q_reb_kw": Q_reb_kw,
        "scale": scale,
        "reflux_ratio": rr_calc,
        "n_stages": n_stages,
        "ethanol_recovery": ethanol_recovery,
        "q": q,
        "mw_feed": mw_feed,
        "mw_top": mw_top,
        "mw_bot": mw_bot,
        "x_eth_feed": x_eth,
        "x_eth_top": _TARGET_ETHANOL_TOP,
        "x_eth_bot": _TARGET_ETHANOL_BOTTOM,
        "x_wat_feed": 1.0 - x_eth,
        "x_wat_top": 1.0 - _TARGET_ETHANOL_TOP,
        "x_wat_bot": 1.0 - _TARGET_ETHANOL_BOTTOM,
    }


def _run_dwsim_simulation(
    F_feed_kmolh: float,
    T_feed: float,
    P_feed_bar: float,
    x_eth: float,
    x_water: float,
    lk: str,
    hk: str,
    lk_bottoms: float,
    hk_distillate: float,
    reflux_ratio_col: float,
) -> dict:
    """
    Ejecuta la simulación real en DWSIM usando DWSIMInterface.

    Parámetros
    ----------
    F_feed_kmolh : float  — Flujo molar alimentación (kmol/h)
    T_feed       : float  — Temperatura alimentación (°C)
    P_feed_bar   : float  — Presión alimentación (bar)
    x_eth        : float  — Fracción molar etanol en alimentación
    x_water      : float  — Fracción molar agua en alimentación
    lk           : str    — Compuesto clave ligero
    hk           : str    — Compuesto clave pesado
    lk_bottoms   : float  — Fracción molar LK en fondos
    hk_distillate: float  — Fracción molar HK en destilado
    reflux_ratio_col : float — Razón de reflujo

    Retorna
    -------
    dict con resultados extraídos de DWSIM.

    Lanza
    -----
    DWSIMInterfaceError si la simulación falla.
    """
    _KG_S_TO_KG_H = 3600.0
    _W_TO_KW = 1e-3
    _MOL_S_TO_KMOL_H = 3.6

    with DWSIMInterface(_cfg.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(_cfg.SIMULATION_FILE)

        # Configurar corriente de alimentación
        dwsim.set_stream_conditions(
            _cfg.TAG_FEED,
            molar_flow=F_feed_kmolh,
            temperature=T_feed,
            pressure=P_feed_bar,
            composition={"Ethanol": x_eth, "Water": x_water},
        )

        # Configurar parámetros de columna
        dwsim.set_column_parameters(
            _cfg.TAG_COLUMN,
            light_key=lk,
            heavy_key=hk,
            lk_bottoms=lk_bottoms,
            hk_distillate=hk_distillate,
            reflux_ratio=reflux_ratio_col,
        )

        # Ejecutar simulación
        dwsim.run_simulation()

        # Extraer propiedades de corrientes
        feed_mflow = dwsim.get_stream_property(_cfg.TAG_FEED, "MassFlow") * _KG_S_TO_KG_H
        top_mflow  = dwsim.get_stream_property(_cfg.TAG_TOP, "MassFlow") * _KG_S_TO_KG_H
        bot_mflow  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MassFlow") * _KG_S_TO_KG_H

        feed_molflow = dwsim.get_stream_property(_cfg.TAG_FEED, "MolarFlow") * _MOL_S_TO_KMOL_H
        top_molflow  = dwsim.get_stream_property(_cfg.TAG_TOP, "MolarFlow") * _MOL_S_TO_KMOL_H
        bot_molflow  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MolarFlow") * _MOL_S_TO_KMOL_H

        feed_T = dwsim.get_stream_property(_cfg.TAG_FEED, "Temperature") - 273.15
        top_T  = dwsim.get_stream_property(_cfg.TAG_TOP, "Temperature") - 273.15
        bot_T  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "Temperature") - 273.15

        feed_P = dwsim.get_stream_property(_cfg.TAG_FEED, "Pressure") / 1e5
        top_P  = dwsim.get_stream_property(_cfg.TAG_TOP, "Pressure") / 1e5
        bot_P  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "Pressure") / 1e5

        # Composiciones
        x_eth_feed = dwsim.get_stream_property(_cfg.TAG_FEED, "MoleFraction", "Ethanol")
        x_eth_top  = dwsim.get_stream_property(_cfg.TAG_TOP, "MoleFraction", "Ethanol")
        x_eth_bot  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MoleFraction", "Ethanol")
        x_wat_feed = dwsim.get_stream_property(_cfg.TAG_FEED, "MoleFraction", "Water")
        x_wat_top  = dwsim.get_stream_property(_cfg.TAG_TOP, "MoleFraction", "Water")
        x_wat_bot  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MoleFraction", "Water")

        # Propiedades de equipo
        q_cond = abs(dwsim.get_equipment_property(_cfg.TAG_COLUMN, "DutyCondenser")) * _W_TO_KW
        q_reb  = abs(dwsim.get_equipment_property(_cfg.TAG_COLUMN, "DutyReboiler")) * _W_TO_KW
        rr     = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "RefluxRatio")
        n_stg  = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "NumberOfStages")

    mw_feed = feed_mflow / max(feed_molflow, 1e-9)
    mw_top  = top_mflow / max(top_molflow, 1e-9)
    mw_bot  = bot_mflow / max(bot_molflow, 1e-9)
    scale = feed_mflow / _FLOW_FEED_BASE_KGH

    ethanol_recovery = (
        top_molflow * x_eth_top
        / max(feed_molflow * x_eth_feed, 1e-9) * 100
    )

    return {
        "source": "dwsim",
        "F_feed_kmolh": feed_molflow,
        "F_feed_kgh": feed_mflow,
        "F_top_kgh": top_mflow,
        "F_bot_kgh": bot_mflow,
        "F_top_kmolh": top_molflow,
        "F_bot_kmolh": bot_molflow,
        "T_feed_C": feed_T,
        "T_top_C": top_T,
        "T_bot_C": bot_T,
        "P_feed_bar": feed_P,
        "Q_cond_kw": q_cond,
        "Q_reb_kw": q_reb,
        "scale": scale,
        "reflux_ratio": rr,
        "n_stages": n_stg,
        "ethanol_recovery": ethanol_recovery,
        "q": None,
        "mw_feed": mw_feed,
        "mw_top": mw_top,
        "mw_bot": mw_bot,
        "x_eth_feed": x_eth_feed,
        "x_eth_top": x_eth_top,
        "x_eth_bot": x_eth_bot,
        "x_wat_feed": x_wat_feed,
        "x_wat_top": x_wat_top,
        "x_wat_bot": x_wat_bot,
    }


def simulacion_dwsim_page():
    """Página principal del simulador DWSIM interactivo."""
    st.header("⚙️ DWSIM Interactive Simulation")
    st.markdown("""
    Configure las condiciones de **alimentación** y **columna** en el panel lateral,
    luego presione **🚀 Run Simulation**.  Si DWSIM está instalado se ejecutará la
    simulación real; de lo contrario se usará el escalado analítico del punto de diseño.
    """)

    # ── Verificar disponibilidad de DWSIM ────────────────────────────────────
    dwsim_ok, dwsim_msg = validate_dwsim_installation()
    if dwsim_ok:
        st.success(f"✅ DWSIM disponible — {dwsim_msg}")
    else:
        st.warning(
            f"⚠️ DWSIM no disponible (se usará escalado analítico como alternativa).  \n"
            f"Detalle: {dwsim_msg}"
        )

    st.markdown("---")

    # ── Referencia de diseño ─────────────────────────────────────────────────
    st.subheader("Referencia de Diseño DWSIM")
    st.caption(
        f"Flowsheet: `ethanol.dwxmz` — Columna: **{_cfg.TAG_COLUMN}** "
        f"| Feed: `{_cfg.TAG_FEED}` | Top: `{_cfg.TAG_TOP}` "
        f"| Bottom: `{_cfg.TAG_BOTTOM}` "
        f"| Condensador: `{_cfg.TAG_R_COND}` | Rehervidor: `{_cfg.TAG_Q_REB}`"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Alimentación base", f"{_FLOW_FEED_BASE_KGH:,.0f} kg/h")
        st.metric("Split destilado", f"{_SPLIT_TOP*100:.0f} %")
    with col_b:
        st.metric("Deber condensador (base)", f"{_Q_COND_BASE_KW:,.2f} kW")
        st.metric("Deber rehervidor (base)", f"{_Q_REB_BASE_KW:,.2f} kW")
    with col_c:
        st.metric("EtOH obj. destilado", f"{_TARGET_ETHANOL_TOP*100:.0f} mol%")
        st.metric("EtOH obj. fondos", f"{_TARGET_ETHANOL_BOTTOM*100:.0f} mol%")

    st.markdown("---")

    # ── Sidebar: parámetros de entrada ───────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Parámetros de Entrada")

        # ── Corriente de alimentación ─────────────────────────────────────────
        with st.expander("1. Corriente de Alimentación (Feed)", expanded=True):
            F_feed_kmolh = st.number_input(
                "Flujo molar (kmol/h)", min_value=1.0, max_value=5_000.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["molar_flow"]),
                step=5.0, key="dwsim_F",
            )
            T_feed = st.number_input(
                "Temperatura (°C)", min_value=20.0, max_value=150.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["temperature"]),
                step=1.0, key="dwsim_T",
            )
            P_feed_bar = st.number_input(
                "Presión (bar)", min_value=0.5, max_value=50.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["pressure"]),
                step=0.5, key="dwsim_P",
            )

        with st.expander("2. Composición Molar", expanded=True):
            st.markdown("Las fracciones deben sumar **1.0**")
            x_eth = st.slider(
                "Etanol (fracción molar)", 0.0, 1.0,
                float(_cfg.DEFAULT_FEED_CONDITIONS["composition"]["Ethanol"]),
                0.01, key="dwsim_xeth",
            )
            x_water = st.slider(
                "Agua (fracción molar)", 0.0, 1.0,
                round(max(0.0, 1.0 - x_eth), 2), 0.01, key="dwsim_xwat",
            )
            comp_sum = round(x_eth + x_water, 4)
            if abs(comp_sum - 1.0) > 1e-3:
                st.error(f"⚠️ Suma de composiciones = {comp_sum:.4f}  (debe ser 1.0)")
            else:
                st.success(f"Suma composiciones: {comp_sum:.4f} ✓")

        # ── Parámetros de columna ─────────────────────────────────────────────
        with st.expander("3. Parámetros de Columna", expanded=True):
            col_defaults = _cfg.DEFAULT_COLUMN_PARAMETERS
            lk = st.selectbox(
                "Light Key Compound (LK)", ["Ethanol", "Water"],
                index=0 if col_defaults["light_key"] == "Ethanol" else 1,
                key="dwsim_lk",
            )
            hk = st.selectbox(
                "Heavy Key Compound (HK)", ["Ethanol", "Water"],
                index=1 if col_defaults["heavy_key"] == "Water" else 0,
                key="dwsim_hk",
            )
            if lk == hk:
                st.error("⚠️ LK y HK deben ser compuestos distintos.")

            lk_bottoms = st.number_input(
                "LK Mole Fraction in Bottoms", min_value=0.0, max_value=1.0,
                value=float(col_defaults["lk_bottoms"]), step=0.005,
                format="%.4f", key="dwsim_lk_bot",
            )
            hk_distillate = st.number_input(
                "HK Mole Fraction in Distillate", min_value=0.0, max_value=1.0,
                value=float(col_defaults["hk_distillate"]), step=0.005,
                format="%.4f", key="dwsim_hk_dist",
            )
            reflux_ratio_col = st.number_input(
                "Reflux Ratio (L/D)", min_value=0.01, max_value=20.0,
                value=float(col_defaults["reflux_ratio"]), step=0.05,
                format="%.2f", key="dwsim_rr",
            )
            if reflux_ratio_col <= 0:
                st.error("⚠️ La razón de reflujo debe ser mayor que 0.")

    # ── Botón de ejecución ────────────────────────────────────────────────────
    st.markdown(
        "Configure los parámetros en el panel lateral y presione el botón para ejecutar la simulación."
    )

    run_clicked = st.button("🚀 Run Simulation", key="btn_dwsim_run", type="primary")

    if run_clicked:
        # ── Validaciones ──────────────────────────────────────────────────────
        errors = []
        if abs(x_eth + x_water - 1.0) > 1e-3:
            errors.append(f"La suma de composiciones es {x_eth + x_water:.4f}; debe ser 1.0 ± 0.001.")
        if lk == hk:
            errors.append("Light Key y Heavy Key deben ser compuestos distintos.")
        if reflux_ratio_col <= 0:
            errors.append("La razón de reflujo debe ser mayor que 0.")
        if not (0.0 <= lk_bottoms <= 1.0):
            errors.append("LK Mole Fraction in Bottoms debe estar entre 0 y 1.")
        if not (0.0 <= hk_distillate <= 1.0):
            errors.append("HK Mole Fraction in Distillate debe estar entre 0 y 1.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            # Limpiar resultados previos
            st.session_state.pop("sim_results", None)

            if dwsim_ok and _DWSIM_IMPORTS_OK:
                # ── Intentar simulación real con DWSIM ────────────────────────
                with st.spinner("Conectando con DWSIM y ejecutando simulación…"):
                    try:
                        results = _run_dwsim_simulation(
                            F_feed_kmolh, T_feed, P_feed_bar, x_eth, x_water,
                            lk, hk, lk_bottoms, hk_distillate, reflux_ratio_col,
                        )
                        st.success("✅ Simulación DWSIM completada exitosamente.")
                    except Exception as exc:  # DWSIMInterfaceError o cualquier otra
                        st.warning(
                            f"⚠️ DWSIM falló ({exc}). Usando escalado analítico como alternativa."
                        )
                        results = _run_analytic_simulation(
                            F_feed_kmolh, T_feed, P_feed_bar, x_eth, reflux_ratio_col,
                        )
            else:
                # ── Fallback: escalado analítico ──────────────────────────────
                with st.spinner("Calculando resultados (escalado analítico)…"):
                    results = _run_analytic_simulation(
                        F_feed_kmolh, T_feed, P_feed_bar, x_eth, reflux_ratio_col,
                    )
                st.info("ℹ️ Resultados obtenidos mediante escalado analítico del punto de diseño.")

            st.session_state["sim_results"] = results

    # ── Mostrar resultados (solo si existen en session_state) ─────────────────
    if "sim_results" not in st.session_state:
        st.info("👆 Configure los parámetros y presione **🚀 Run Simulation** para ver resultados.")
        return

    r = st.session_state["sim_results"]
    _source_label = "🔬 DWSIM (real)" if r.get("source") == "dwsim" else "📐 Escalado analítico"
    st.caption(f"Fuente de resultados: **{_source_label}**")

    st.subheader("📊 Resultados de la Simulación")

    # ── Tabla de corrientes ───────────────────────────────────────────────────
    st.markdown("#### Tabla de Corrientes")
    stream_df = pd.DataFrame({
        "Corriente": ["Feed", "Destilado (Top)", "Fondos (Bottom)"],
        "Flujo molar (kmol/h)": [
            f"{r['F_feed_kmolh']:.2f}",
            f"{r['F_top_kmolh']:.2f}",
            f"{r['F_bot_kmolh']:.2f}",
        ],
        "Flujo másico (kg/h)": [
            f"{r['F_feed_kgh']:,.1f}",
            f"{r['F_top_kgh']:,.1f}",
            f"{r['F_bot_kgh']:,.1f}",
        ],
        "Temperatura (°C)": [
            f"{r['T_feed_C']:.1f}",
            f"{r['T_top_C']:.1f}",
            f"{r['T_bot_C']:.1f}",
        ],
        "Presión (bar)": [
            f"{r['P_feed_bar']:.2f}",
            "—",
            "—",
        ],
        "x Etanol": [
            f"{r['x_eth_feed']:.4f}",
            f"{r['x_eth_top']:.4f}",
            f"{r['x_eth_bot']:.4f}",
        ],
        "x Agua": [
            f"{r['x_wat_feed']:.4f}",
            f"{r['x_wat_top']:.4f}",
            f"{r['x_wat_bot']:.4f}",
        ],
    })
    st.dataframe(stream_df, use_container_width=True)

    # ── Tabla de equipos ──────────────────────────────────────────────────────
    st.markdown("#### Tabla de Equipos")
    equip_df = pd.DataFrame({
        "Parámetro": [
            "Deber condensador (kW)",
            "Deber rehervidor (kW)",
            "Razón de reflujo (L/D)",
            "Número de etapas teóricas",
        ],
        "Valor": [
            f"{r['Q_cond_kw']:.1f}",
            f"{r['Q_reb_kw']:.1f}",
            f"{r['reflux_ratio']:.2f}",
            f"{r['n_stages']:.0f}",
        ],
    })
    st.dataframe(equip_df, use_container_width=True)

    # ── Métricas clave ────────────────────────────────────────────────────────
    st.markdown("#### Métricas Clave del Proceso")
    scale = r["scale"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condensador", f"{r['Q_cond_kw']:.1f} kW",
              delta=f"{(scale-1)*100:+.1f}% vs diseño")
    c2.metric("Rehervidor", f"{r['Q_reb_kw']:.1f} kW",
              delta=f"{(scale-1)*100:+.1f}% vs diseño")
    c3.metric("Reflujo L/D", f"{r['reflux_ratio']:.2f}")
    c4.metric("Recuperación EtOH", f"{min(r['ethanol_recovery'], 100.0):.1f} %")

    if r.get("q") is not None:
        c5, c6 = st.columns(2)
        c5.metric("Parámetro q",  f"{r['q']:.3f}",
                  help="q=1: líquido saturado; q>1: sub-enfriado; q<1: vapor parcial")
        c6.metric("PM alimentación", f"{r['mw_feed']:.3f} g/mol")

    st.markdown("---")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    st.markdown("#### Visualización de Resultados")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: flujos másicos (barras horizontales)
    labels = ["Feed", "Destilado", "Fondos"]
    values_kgh = [r["F_feed_kgh"], r["F_top_kgh"], r["F_bot_kgh"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    axes[0].barh(labels, values_kgh, color=colors)
    axes[0].set_xlabel("Flujo (kg/h)")
    axes[0].set_title("Flujos Másicos de Corrientes")
    for i, v in enumerate(values_kgh):
        axes[0].text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=8)

    # Panel 2: comparación de composiciones (antes/después)
    x_eth_vals = [r["x_eth_feed"], r["x_eth_top"], r["x_eth_bot"]]
    x_wat_vals = [r["x_wat_feed"], r["x_wat_top"], r["x_wat_bot"]]
    bar_w = 0.35
    x_pos = np.arange(3)
    axes[1].bar(x_pos - bar_w / 2, x_eth_vals, bar_w, label="Etanol", color="#55A868")
    axes[1].bar(x_pos + bar_w / 2, x_wat_vals, bar_w, label="Agua", color="#4C72B0")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Feed", "Destilado", "Fondos"])
    axes[1].set_ylabel("Fracción molar")
    axes[1].set_title("Composiciones por Corriente")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.1)

    # Panel 3: balance de energía
    duty_labels = ["Condensador", "Rehervidor"]
    duty_vals = [r["Q_cond_kw"], r["Q_reb_kw"]]
    duty_colors = ["#8172B2", "#CCB974"]
    axes[2].bar(duty_labels, duty_vals, color=duty_colors, width=0.4)
    axes[2].set_ylabel("Deber (kW)")
    axes[2].set_title("Balance de Energía")
    for i, v in enumerate(duty_vals):
        axes[2].text(i, v + max(duty_vals) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Ecuaciones de escalado ────────────────────────────────────────────────
    with st.expander("📐 Ecuaciones de Escalado Usadas (modo analítico)"):
        st.latex(r"\dot{F}_{\text{scaled}} = \dot{F}_{\text{design}} \times \alpha, "
                 r"\quad \alpha = \frac{\dot{m}_{\text{feed}}}{\dot{m}_{\text{feed,design}}}")
        st.latex(r"\dot{Q}_{\text{cond/reb,scaled}} = \dot{Q}_{\text{cond/reb,design}} \times \alpha")
        st.latex(r"\frac{L}{D} \approx \frac{\dot{V} - \dot{D}}{\dot{D}}, "
                 r"\quad \dot{V} = \frac{\dot{Q}_{\text{cond}}}{\Delta H_{\text{vap,top}}}")
        st.markdown("""
        Estas son **relaciones de escalado de primer orden** válidas cerca del punto de diseño.
        Desviaciones grandes de la composición o flujo base requieren simulación rigurosa en DWSIM.
        """)
