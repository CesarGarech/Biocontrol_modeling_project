"""
Digital Twin — Sub-page 1: DWSIM Interactive Simulation
========================================================
Lets the user specify feed-stream conditions and distillation-column
parameters.  Pressing "🚀 Run Simulation" attempts to connect to DWSIM
via DWSIMInterface; if DWSIM is unavailable, analytic design-point
scaling is used as a fallback.
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Add Simulation folder to Python path ─────────────────────────────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── Import DWSIM interface (safe import when pythonnet is unavailable) ────────
try:
    from dwsim_interface import DWSIMInterface, DWSIMInterfaceError  # noqa: E402
    from dwsim_data_generator import validate_dwsim_installation     # noqa: E402
    _DWSIM_IMPORTS_OK = True
except Exception:
    _DWSIM_IMPORTS_OK = False
    DWSIMInterface = None          # type: ignore[assignment,misc]
    DWSIMInterfaceError = RuntimeError  # type: ignore[assignment,misc]

    def validate_dwsim_installation():  # type: ignore[misc]
        """Stub when DWSIM modules are not available."""
        return False, "DWSIM modules could not be imported."

# ── Local constants ───────────────────────────────────────────────────────────
_TARGET_ETHANOL_BOTTOM = 0.02    # ethanol mole fraction in bottoms (design)
_MW_ETHANOL = 46.068             # g/mol
_MW_WATER = 18.015               # g/mol

_FLOW_FEED_BASE_KGH = _cfg.FLOW_FEED_BASE
_SPLIT_TOP = _cfg.SPLIT_TOP
_SPLIT_BOTTOM = _cfg.SPLIT_BOTTOM
_Q_COND_BASE_KW = _cfg.Q_COND_BASE
_Q_REB_BASE_KW = _cfg.Q_REB_BASE
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP


def _average_mw(x_eth: float) -> float:
    """Average molecular weight for an ethanol-water mixture."""
    return x_eth * _MW_ETHANOL + (1.0 - x_eth) * _MW_WATER


def _run_analytic_simulation(
    F_feed_kmolh: float,
    T_feed: float,
    P_feed_bar: float,
    x_eth: float,
    reflux_ratio_col: float,
) -> dict:
    """
    Compute column results using analytic design-point scaling.

    Parameters
    ----------
    F_feed_kmolh : float
        Feed molar flow rate (kmol/h).
    T_feed : float
        Feed temperature (°C).
    P_feed_bar : float
        Feed pressure (bar).
    x_eth : float
        Ethanol mole fraction in the feed.
    reflux_ratio_col : float
        Column reflux ratio.

    Returns
    -------
    dict with scaled results.
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

    # Theoretical stage estimate: simplified Fenske approximation
    # N ≈ 10 × (R / R_min); R_min ≈ rr_calc computed at the design base
    _R_MIN_FACTOR = 10.0   # Fenske proportionality factor (simplified)
    _R_MIN_FLOOR = 0.1     # floor for R_min to avoid division by zero
    _N_STAGES_MIN = 5      # minimum theoretical stages (practical lower limit)
    n_stages = max(_N_STAGES_MIN, int(_R_MIN_FACTOR * reflux_ratio_col / max(rr_calc, _R_MIN_FLOOR)))

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
) -> dict:
    """
    Run a live simulation in DWSIM using DWSIMInterface.

    Column parameters are left at their DWSIM defaults; only the feed
    stream conditions are set by the caller.

    Parameters
    ----------
    F_feed_kmolh : float  — Feed molar flow rate (kmol/h)
    T_feed       : float  — Feed temperature (°C)
    P_feed_bar   : float  — Feed pressure (bar)
    x_eth        : float  — Ethanol mole fraction in the feed
    x_water      : float  — Water mole fraction in the feed

    Returns
    -------
    dict with results extracted from DWSIM.

    Raises
    ------
    DWSIMInterfaceError if the simulation fails.
    """
    _KG_S_TO_KG_H = 3600.0
    _W_TO_KW = 1e-3
    _MOL_S_TO_KMOL_H = 3.6

    with DWSIMInterface(_cfg.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(_cfg.SIMULATION_FILE)

        # Set feed stream conditions only — column parameters use DWSIM defaults
        dwsim.set_stream_conditions(
            _cfg.TAG_FEED,
            molar_flow=F_feed_kmolh,
            temperature=T_feed,
            pressure=P_feed_bar,
            composition={"Ethanol": x_eth, "Water": x_water},
        )

        # Run simulation
        dwsim.run_simulation()

        # Extract stream properties
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

        # Compositions
        x_eth_feed = dwsim.get_stream_property(_cfg.TAG_FEED, "MoleFraction", "Ethanol")
        x_eth_top  = dwsim.get_stream_property(_cfg.TAG_TOP, "MoleFraction", "Ethanol")
        x_eth_bot  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MoleFraction", "Ethanol")
        x_wat_feed = dwsim.get_stream_property(_cfg.TAG_FEED, "MoleFraction", "Water")
        x_wat_top  = dwsim.get_stream_property(_cfg.TAG_TOP, "MoleFraction", "Water")
        x_wat_bot  = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MoleFraction", "Water")

        # Condenser / reboiler duty — read from dedicated energy streams
        # (ShortcutColumn does not expose CondenserDuty / ReboilerDuty directly)
        q_cond = abs(dwsim.get_equipment_property(_cfg.TAG_R_COND, "Duty")) * _W_TO_KW
        q_reb  = abs(dwsim.get_equipment_property(_cfg.TAG_Q_REB,  "Duty")) * _W_TO_KW

        # Reflux ratio and stage count from the column — use defaults if unavailable
        try:
            rr = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "RefluxRatio")
        except Exception:
            rr = float(_cfg.DEFAULT_COLUMN_PARAMETERS["reflux_ratio"])
        try:
            n_stg = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "NumberOfStages")
        except Exception:
            n_stg = 0.0

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
    """Main page for the interactive DWSIM simulator."""
    st.header("⚙️ DWSIM Interactive Simulation")
    st.markdown("""
    Configure **feed** conditions and **column** parameters in the sidebar,
    then press **🚀 Run Simulation**.  If DWSIM is installed the live
    simulation will run; otherwise analytic design-point scaling is used.
    """)

    # ── Check DWSIM availability ──────────────────────────────────────────────
    dwsim_ok, dwsim_msg = validate_dwsim_installation()
    if dwsim_ok:
        st.success(f"✅ DWSIM available — {dwsim_msg}")
    else:
        st.warning(
            f"⚠️ DWSIM not available (analytic scaling will be used as fallback).  \n"
            f"Detail: {dwsim_msg}"
        )

    st.markdown("---")

    # ── Design reference ──────────────────────────────────────────────────────
    st.subheader("DWSIM Design Reference")
    st.caption(
        f"Flowsheet: `ethanol.dwxmz` — Column: **{_cfg.TAG_COLUMN}** "
        f"| Feed: `{_cfg.TAG_FEED}` | Top: `{_cfg.TAG_TOP}` "
        f"| Bottom: `{_cfg.TAG_BOTTOM}` "
        f"| Condenser: `{_cfg.TAG_R_COND}` | Reboiler: `{_cfg.TAG_Q_REB}`"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Base Feed Flow", f"{_FLOW_FEED_BASE_KGH:,.0f} kg/h")
        st.metric("Distillate Split", f"{_SPLIT_TOP*100:.0f} %")
    with col_b:
        st.metric("Condenser Duty (base)", f"{_Q_COND_BASE_KW:,.2f} kW")
        st.metric("Reboiler Duty (base)", f"{_Q_REB_BASE_KW:,.2f} kW")
    with col_c:
        st.metric("EtOH target distillate", f"{_TARGET_ETHANOL_TOP*100:.0f} mol%")
        st.metric("EtOH target bottoms", f"{_TARGET_ETHANOL_BOTTOM*100:.0f} mol%")

    st.markdown("---")

    # ── Sidebar: input parameters ─────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Input Parameters")

        # ── Feed stream ───────────────────────────────────────────────────────
        with st.expander("1. Feed Stream", expanded=True):
            F_feed_kmolh = st.number_input(
                "Molar flow (kmol/h)", min_value=1.0, max_value=5_000.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["molar_flow"]),
                step=5.0, key="dwsim_F",
            )
            T_feed = st.number_input(
                "Temperature (°C)", min_value=20.0, max_value=150.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["temperature"]),
                step=1.0, key="dwsim_T",
            )
            P_feed_bar = st.number_input(
                "Pressure (bar)", min_value=0.5, max_value=50.0,
                value=float(_cfg.DEFAULT_FEED_CONDITIONS["pressure"]),
                step=0.5, key="dwsim_P",
            )

        with st.expander("2. Molar Composition", expanded=True):
            st.markdown("Fractions must sum to **1.0**")
            x_eth = st.slider(
                "Ethanol (mole fraction)", 0.0, 1.0,
                float(_cfg.DEFAULT_FEED_CONDITIONS["composition"]["Ethanol"]),
                0.01, key="dwsim_xeth",
            )
            x_water = st.slider(
                "Water (mole fraction)", 0.0, 1.0,
                round(max(0.0, 1.0 - x_eth), 2), 0.01, key="dwsim_xwat",
            )
            comp_sum = round(x_eth + x_water, 4)
            if abs(comp_sum - 1.0) > 1e-3:
                st.error(f"⚠️ Composition sum = {comp_sum:.4f}  (must be 1.0)")
            else:
                st.success(f"Composition sum: {comp_sum:.4f} ✓")

        # ── Column parameters ─────────────────────────────────────────────────
        # Column parameters are not exposed to the user; DWSIM defaults are used.

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown(
        "Configure feed conditions in the sidebar and press the button to run the simulation."
    )

    run_clicked = st.button("🚀 Run Simulation", key="btn_dwsim_run", type="primary")

    # Fixed reflux ratio from config (not exposed to the user)
    _default_rr = float(_cfg.DEFAULT_COLUMN_PARAMETERS["reflux_ratio"])

    if run_clicked:
        # ── Validation ────────────────────────────────────────────────────────
        errors = []
        if abs(x_eth + x_water - 1.0) > 1e-3:
            errors.append(f"Composition sum is {x_eth + x_water:.4f}; must be 1.0 ± 0.001.")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            # Clear any previous results
            st.session_state.pop("sim_results", None)

            if dwsim_ok and _DWSIM_IMPORTS_OK:
                # ── Attempt live DWSIM simulation ─────────────────────────────
                with st.spinner("Connecting to DWSIM and running simulation…"):
                    try:
                        results = _run_dwsim_simulation(
                            F_feed_kmolh, T_feed, P_feed_bar, x_eth, x_water,
                        )
                        st.success("✅ DWSIM simulation completed successfully.")
                    except Exception as exc:  # DWSIMInterfaceError or any other
                        st.warning(
                            f"⚠️ DWSIM failed ({exc}). Using analytic scaling as fallback."
                        )
                        results = _run_analytic_simulation(
                            F_feed_kmolh, T_feed, P_feed_bar, x_eth, _default_rr,
                        )
            else:
                # ── Fallback: analytic scaling ────────────────────────────────
                with st.spinner("Computing results (analytic scaling)…"):
                    results = _run_analytic_simulation(
                        F_feed_kmolh, T_feed, P_feed_bar, x_eth, _default_rr,
                    )
                st.info("ℹ️ Results obtained via analytic scaling of the design point.")

            st.session_state["sim_results"] = results

    # ── Display results (only if present in session_state) ────────────────────
    if "sim_results" not in st.session_state:
        st.info("👆 Configure parameters and press **🚀 Run Simulation** to view results.")
        return

    r = st.session_state["sim_results"]
    _source_label = "🔬 DWSIM (live)" if r.get("source") == "dwsim" else "📐 Analytic scaling"
    st.caption(f"Results source: **{_source_label}**")

    st.subheader("📊 Simulation Results")

    # ── Stream table ──────────────────────────────────────────────────────────
    st.markdown("#### Stream Table")
    stream_df = pd.DataFrame({
        "Stream": ["Feed", "Distillate (Top)", "Bottoms (Bottom)"],
        "Molar Flow (kmol/h)": [
            f"{r['F_feed_kmolh']:.2f}",
            f"{r['F_top_kmolh']:.2f}",
            f"{r['F_bot_kmolh']:.2f}",
        ],
        "Mass Flow (kg/h)": [
            f"{r['F_feed_kgh']:,.1f}",
            f"{r['F_top_kgh']:,.1f}",
            f"{r['F_bot_kgh']:,.1f}",
        ],
        "Temperature (°C)": [
            f"{r['T_feed_C']:.1f}",
            f"{r['T_top_C']:.1f}",
            f"{r['T_bot_C']:.1f}",
        ],
        "Pressure (bar)": [
            f"{r['P_feed_bar']:.2f}",
            "—",
            "—",
        ],
        "x Ethanol": [
            f"{r['x_eth_feed']:.4f}",
            f"{r['x_eth_top']:.4f}",
            f"{r['x_eth_bot']:.4f}",
        ],
        "x Water": [
            f"{r['x_wat_feed']:.4f}",
            f"{r['x_wat_top']:.4f}",
            f"{r['x_wat_bot']:.4f}",
        ],
    })
    st.dataframe(stream_df, use_container_width=True)

    # ── Equipment table ───────────────────────────────────────────────────────
    st.markdown("#### Equipment Table")
    equip_df = pd.DataFrame({
        "Parameter": [
            "Condenser Duty (kW)",
            "Reboiler Duty (kW)",
            "Reflux Ratio (L/D)",
            "Number of Theoretical Stages",
        ],
        "Value": [
            f"{r['Q_cond_kw']:.1f}",
            f"{r['Q_reb_kw']:.1f}",
            f"{r['reflux_ratio']:.2f}",
            f"{r['n_stages']:.0f}",
        ],
    })
    st.dataframe(equip_df, use_container_width=True)

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.markdown("#### Key Process Metrics")
    scale = r["scale"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condenser", f"{r['Q_cond_kw']:.1f} kW",
              delta=f"{(scale-1)*100:+.1f}% vs design")
    c2.metric("Reboiler", f"{r['Q_reb_kw']:.1f} kW",
              delta=f"{(scale-1)*100:+.1f}% vs design")
    c3.metric("Reflux L/D", f"{r['reflux_ratio']:.2f}")
    c4.metric("EtOH Recovery", f"{min(r['ethanol_recovery'], 100.0):.1f} %")

    if r.get("q") is not None:
        c5, c6 = st.columns(2)
        c5.metric("q-parameter", f"{r['q']:.3f}",
                  help="q=1: saturated liquid; q>1: sub-cooled; q<1: partial vapour")
        c6.metric("Feed MW", f"{r['mw_feed']:.3f} g/mol")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("#### Results Visualization")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: mass flows (horizontal bar)
    labels = ["Feed", "Distillate", "Bottoms"]
    values_kgh = [r["F_feed_kgh"], r["F_top_kgh"], r["F_bot_kgh"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    axes[0].barh(labels, values_kgh, color=colors)
    axes[0].set_xlabel("Flow (kg/h)")
    axes[0].set_title("Stream Mass Flows")
    for i, v in enumerate(values_kgh):
        axes[0].text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=8)

    # Panel 2: composition comparison
    x_eth_vals = [r["x_eth_feed"], r["x_eth_top"], r["x_eth_bot"]]
    x_wat_vals = [r["x_wat_feed"], r["x_wat_top"], r["x_wat_bot"]]
    bar_w = 0.35
    x_pos = np.arange(3)
    axes[1].bar(x_pos - bar_w / 2, x_eth_vals, bar_w, label="Ethanol", color="#55A868")
    axes[1].bar(x_pos + bar_w / 2, x_wat_vals, bar_w, label="Water", color="#4C72B0")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Feed", "Distillate", "Bottoms"])
    axes[1].set_ylabel("Mole Fraction")
    axes[1].set_title("Stream Compositions")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.1)

    # Panel 3: energy balance
    duty_labels = ["Condenser", "Reboiler"]
    duty_vals = [r["Q_cond_kw"], r["Q_reb_kw"]]
    duty_colors = ["#8172B2", "#CCB974"]
    axes[2].bar(duty_labels, duty_vals, color=duty_colors, width=0.4)
    axes[2].set_ylabel("Duty (kW)")
    axes[2].set_title("Energy Balance")
    for i, v in enumerate(duty_vals):
        axes[2].text(i, v + max(duty_vals) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Scaling equations ─────────────────────────────────────────────────────
    with st.expander("📐 Scaling Equations Used (analytic mode)"):
        st.latex(r"\dot{F}_{\text{scaled}} = \dot{F}_{\text{design}} \times \alpha, "
                 r"\quad \alpha = \frac{\dot{m}_{\text{feed}}}{\dot{m}_{\text{feed,design}}}")
        st.latex(r"\dot{Q}_{\text{cond/reb,scaled}} = \dot{Q}_{\text{cond/reb,design}} \times \alpha")
        st.latex(r"\frac{L}{D} \approx \frac{\dot{V} - \dot{D}}{\dot{D}}, "
                 r"\quad \dot{V} = \frac{\dot{Q}_{\text{cond}}}{\Delta H_{\text{vap,top}}}")
        st.markdown("""
        These are **first-order scaling relationships** valid near the design point.
        Large deviations in composition or base flow require rigorous simulation in DWSIM.
        """)
