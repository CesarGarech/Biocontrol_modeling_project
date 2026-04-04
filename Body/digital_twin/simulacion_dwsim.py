"""
Digital Twin — Sub-page 1: DWSIM Rigorous Simulation
=====================================================
Executes a rigorous steady-state DWSIM simulation of the ethanol
distillation column.  If DWSIM / pythonnet is not available the page
gracefully falls back to the original linear-scaling approximation so the
app never crashes.

Simulation mode indicator
  🟢  DWSIM Rigorous  — real DWSIM COM automation via pythonnet
  🟡  Scaling Fallback — analytic scale-up of the DWSIM design point
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Shared configuration from Simulation/config.py ──────────────────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── DWSIM interface module ───────────────────────────────────────────────────
import dwsim_interface as _dwsim  # noqa: E402

# ── Local-only constants ─────────────────────────────────────────────────────
_TARGET_ETHANOL_BOTTOM = 0.02    # mol fraction of ethanol in bottoms (design target)
_MW_ETHANOL = 46.068             # g/mol
_MW_WATER = 18.015               # g/mol

# Convenient aliases matching config naming (used by scaling fallback)
_FLOW_FEED_BASE_KGH = _cfg.FLOW_FEED_BASE
_SPLIT_TOP = _cfg.SPLIT_TOP
_SPLIT_BOTTOM = _cfg.SPLIT_BOTTOM
_Q_COND_BASE_KW = _cfg.Q_COND_BASE
_Q_REB_BASE_KW = _cfg.Q_REB_BASE
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP


def _average_mw(x_eth: float) -> float:
    """Mean molecular weight for an ethanol-water mixture."""
    return x_eth * _MW_ETHANOL + (1.0 - x_eth) * _MW_WATER


# ── DWSIM session-state helpers ──────────────────────────────────────────────

def _get_dwsim_objects():
    """Return (interf, sim) from session state, or (None, None)."""
    return (
        st.session_state.get("dwsim_interf"),
        st.session_state.get("dwsim_sim"),
    )


def _init_dwsim_session():
    """
    Initialise DWSIM interf + sim and cache in session_state.
    Returns (interf, sim, error_message).
    """
    interf, sim = _get_dwsim_objects()
    if interf is not None and sim is not None:
        return interf, sim, ""

    try:
        interf = _dwsim.init_dwsim()
        sim = _dwsim.load_simulation(interf)
        st.session_state["dwsim_interf"] = interf
        st.session_state["dwsim_sim"] = sim
        return interf, sim, ""
    except Exception as exc:
        return None, None, str(exc)


# ── Scaling fallback calculation ─────────────────────────────────────────────

def _run_scaling(F_feed_kmolh, x_eth, T_feed, P_feed):
    """Compute stream flows / duties by linear scaling from the DWSIM design point."""
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
    reflux_ratio = max(0.0, (V_vapour - F_top_kmolh) / F_top_kmolh) if F_top_kmolh > 0 else 0.0
    ethanol_recovery = (
        F_top_kmolh * _TARGET_ETHANOL_TOP
        / max(F_feed_kmolh * x_eth, 1e-9) * 100
    )
    T_bp = 78.4 + (P_feed - 101.325) * 0.03
    cp_liquid = 2.9
    lambda_feed = dh_vap_top * mw_feed / 1000.0
    q = 1.0 + cp_liquid * max(T_bp - T_feed, 0.0) / lambda_feed

    return {
        "F_feed_kmolh": F_feed_kmolh,
        "F_feed_kgh": F_feed_kgh,
        "F_top_kgh": F_top_kgh,
        "F_bot_kgh": F_bot_kgh,
        "F_top_kmolh": F_top_kmolh,
        "F_bot_kmolh": F_bot_kmolh,
        "Q_cond_kw": Q_cond_kw,
        "Q_reb_kw": Q_reb_kw,
        "scale": scale,
        "reflux_ratio": reflux_ratio,
        "ethanol_recovery": ethanol_recovery,
        "q": q,
        "mw_feed": mw_feed,
        "mw_top": mw_top,
        "mw_bot": mw_bot,
        "x_eth": x_eth,
        "x_top_eth": _TARGET_ETHANOL_TOP,
        "x_bot_eth": _TARGET_ETHANOL_BOTTOM,
        "mode": "scaling",
        "converged": True,
        "solver_status": "N/A (scaling fallback)",
    }


def _run_dwsim(interf, sim, T_feed, P_feed, F_feed_kmolh, x_eth, x_water):
    """Run the real DWSIM simulation and return a results dict."""
    mw_feed = _average_mw(x_eth)
    F_feed_kgh = F_feed_kmolh * mw_feed

    # Set feed conditions
    _dwsim.set_feed_conditions(
        sim,
        temperature_C=T_feed,
        pressure_kPa=P_feed,
        flow_kgh=F_feed_kgh,
        x_ethanol=x_eth,
        x_water=x_water,
    )

    # Solve
    converged, err_msg = _dwsim.run_simulation(
        interf, sim, timeout_seconds=getattr(_cfg, "DWSIM_TIMEOUT", 120)
    )

    # Read results
    raw = _dwsim.read_results(sim)

    feed_res = raw.get("feed", {})
    top_res = raw.get("top", {})
    bot_res = raw.get("bottom", {})
    cond_res = raw.get("r_cond", {})
    reb_res = raw.get("q_reb", {})

    F_top_kgh = top_res.get("mass_flow_kgh", 0.0)
    F_bot_kgh = bot_res.get("mass_flow_kgh", 0.0)
    F_top_kmolh = top_res.get("molar_flow_kmolh", 0.0)
    F_bot_kmolh = bot_res.get("molar_flow_kmolh", 0.0)
    Q_cond_kw = abs(cond_res.get("energy_kw", 0.0))
    Q_reb_kw = abs(reb_res.get("energy_kw", 0.0))

    top_comp = top_res.get("composition", [_TARGET_ETHANOL_TOP, 1 - _TARGET_ETHANOL_TOP])
    bot_comp = bot_res.get("composition", [_TARGET_ETHANOL_BOTTOM, 1 - _TARGET_ETHANOL_BOTTOM])
    x_top_eth = top_comp[0] if len(top_comp) > 0 else _TARGET_ETHANOL_TOP
    x_bot_eth = bot_comp[0] if len(bot_comp) > 0 else _TARGET_ETHANOL_BOTTOM

    mw_top = _average_mw(x_top_eth)
    mw_bot = _average_mw(x_bot_eth)

    scale = F_feed_kgh / _FLOW_FEED_BASE_KGH

    dh_vap_ethanol = 38.56
    dh_vap_water = 40.65
    dh_vap_top = x_top_eth * dh_vap_ethanol + (1 - x_top_eth) * dh_vap_water
    V_vapour = Q_cond_kw / dh_vap_top if dh_vap_top > 0 else 0.0
    reflux_ratio = max(0.0, (V_vapour - F_top_kmolh) / F_top_kmolh) if F_top_kmolh > 0 else 0.0
    ethanol_recovery = (
        F_top_kmolh * x_top_eth / max(F_feed_kmolh * x_eth, 1e-9) * 100
    )
    T_bp = 78.4 + (P_feed - 101.325) * 0.03
    cp_liquid = 2.9
    lambda_feed = dh_vap_top * mw_feed / 1000.0
    q = 1.0 + cp_liquid * max(T_bp - T_feed, 0.0) / lambda_feed

    return {
        "F_feed_kmolh": F_feed_kmolh,
        "F_feed_kgh": F_feed_kgh,
        "F_top_kgh": F_top_kgh,
        "F_bot_kgh": F_bot_kgh,
        "F_top_kmolh": F_top_kmolh,
        "F_bot_kmolh": F_bot_kmolh,
        "Q_cond_kw": Q_cond_kw,
        "Q_reb_kw": Q_reb_kw,
        "scale": scale,
        "reflux_ratio": reflux_ratio,
        "ethanol_recovery": ethanol_recovery,
        "q": q,
        "mw_feed": mw_feed,
        "mw_top": mw_top,
        "mw_bot": mw_bot,
        "x_eth": x_eth,
        "x_top_eth": x_top_eth,
        "x_bot_eth": x_bot_eth,
        "mode": "dwsim",
        "converged": converged,
        "solver_status": "Converged" if converged else f"Not converged: {err_msg}",
        "dwsim_raw": raw,
    }


# ── Main page ─────────────────────────────────────────────────────────────────

def simulacion_dwsim_page():
    # ── Simulation mode indicator ─────────────────────────────────────────────
    if _dwsim.DWSIM_AVAILABLE:
        st.success("🟢 **DWSIM Rigorous** — pythonnet detected; real DWSIM simulation available.")
    else:
        st.warning(
            "🟡 **Scaling Fallback** — DWSIM/pythonnet not available.  "
            "Results are analytic scale-ups of the design point.  \n"
            "To enable rigorous simulation: `pip install pythonnet` and set "
            "the `DWSIM_PATH` environment variable to your DWSIM installation."
        )

    st.header("⚙️ DWSIM Distillation Simulation")
    st.markdown("""
    This page runs a **rigorous steady-state DWSIM simulation** of the ethanol
    distillation column when DWSIM is available, or falls back to a linear
    scaling approximation of the DWSIM design point.

    The reference flowsheet (`ethanol.dwxmz`) defines the base configuration;
    you can override the feed conditions in the sidebar.
    """)

    st.markdown("---")

    # ── Column design reference ───────────────────────────────────────────────
    st.subheader("Column Design Reference (DWSIM)")
    st.caption(
        f"Flowsheet: `ethanol.dwxmz` — Column tag: **{_cfg.TAG_COLUMN}** "
        f"| Feed: `{_cfg.TAG_FEED}` | Top: `{_cfg.TAG_TOP}` "
        f"| Bottom: `{_cfg.TAG_BOTTOM}` "
        f"| Condenser: `{_cfg.TAG_R_COND}` | Reboiler: `{_cfg.TAG_Q_REB}`"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Base Feed", f"{_FLOW_FEED_BASE_KGH:,.0f} kg/h")
        st.metric("Distillate split", f"{_SPLIT_TOP * 100:.0f} %")
    with col_b:
        st.metric("Condenser duty (base)", f"{_Q_COND_BASE_KW:,.2f} kW")
        st.metric("Reboiler duty (base)", f"{_Q_REB_BASE_KW:,.2f} kW")
    with col_c:
        st.metric("Target EtOH in distillate", f"{_TARGET_ETHANOL_TOP * 100:.0f} mol%")
        st.metric("Target EtOH in bottoms", f"{_TARGET_ETHANOL_BOTTOM * 100:.0f} mol%")

    st.markdown("---")

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Feed Conditions")

        with st.expander("1. Thermal Conditions", expanded=True):
            T_feed = st.number_input(
                "Feed Temperature T (°C)", min_value=20.0, max_value=150.0,
                value=30.0, step=1.0, key="dwsim_T"
            )
            P_feed = st.number_input(
                "Feed Pressure P (kPa)", min_value=50.0, max_value=500.0,
                value=100.0, step=1.0, key="dwsim_P"
            )

        with st.expander("2. Feed Flow Rate", expanded=True):
            F_feed_kmolh = st.number_input(
                "Total Feed Flow (kmol/h)", min_value=1.0, max_value=5_000.0,
                value=100.0, step=5.0, key="dwsim_F"
            )

        with st.expander("3. Molar Composition", expanded=True):
            st.markdown("Fractions must sum to **1.0**")
            x_eth = st.slider(
                "Ethanol mole fraction", 0.0, 1.0, 0.40, 0.01, key="dwsim_xeth"
            )
            x_water = st.slider(
                "Water mole fraction", 0.0, 1.0,
                round(max(0.0, 1.0 - x_eth), 2), 0.01, key="dwsim_xwat"
            )
            x_other = round(max(0.0, 1.0 - x_eth - x_water), 4)
            st.info(f"Other (inerts) mole fraction: **{x_other:.4f}**")

            total = x_eth + x_water + x_other
            if abs(total - 1.0) > 1e-3:
                st.error(f"⚠️ Composition sums to {total:.4f} — adjust sliders so total = 1.0")

        with st.expander("4. Column Operating Pressure", expanded=False):
            P_column_kPa = st.number_input(
                "Column top pressure (kPa)", min_value=10.0, max_value=300.0,
                value=101.325, step=1.0, key="dwsim_Pcol"
            )

    # ── Run button ────────────────────────────────────────────────────────────
    st.markdown(
        "Set the feed conditions in the sidebar, then click **▶ Run Simulation** to compute results."
    )

    if st.button("▶ Run Simulation", key="btn_dwsim_run"):
        use_dwsim = _dwsim.DWSIM_AVAILABLE

        if use_dwsim:
            with st.spinner("Initialising DWSIM and solving flowsheet…"):
                interf, sim, init_err = _init_dwsim_session()
                if init_err:
                    st.error(
                        f"⚠️ DWSIM initialisation failed — falling back to scaling.\n\n{init_err}"
                    )
                    use_dwsim = False
                else:
                    try:
                        results = _run_dwsim(
                            interf, sim, T_feed, P_feed, F_feed_kmolh, x_eth, x_water
                        )
                        if not results["converged"]:
                            st.warning(
                                f"⚠️ Solver did not converge: {results['solver_status']}\n\n"
                                "Showing last-iteration values."
                            )
                    except Exception as exc:
                        st.error(
                            f"⚠️ DWSIM simulation error — falling back to scaling.\n\n{exc}"
                        )
                        use_dwsim = False

        if not use_dwsim:
            results = _run_scaling(F_feed_kmolh, x_eth, T_feed, P_feed)

        st.session_state["sim_results"] = results

    # ── Results display ───────────────────────────────────────────────────────
    if "sim_results" not in st.session_state:
        st.info("👆 Configure the feed conditions and click **▶ Run Simulation** to see results.")
        return

    r = st.session_state["sim_results"]
    is_dwsim_mode = r.get("mode") == "dwsim"

    st.subheader("📊 Simulation Results")

    # Stream table
    x_top_eth_res = r.get("x_top_eth", _TARGET_ETHANOL_TOP)
    x_bot_eth_res = r.get("x_bot_eth", _TARGET_ETHANOL_BOTTOM)
    stream_data = {
        "Stream": ["Feed", "Distillate (Top)", "Bottoms"],
        "Flow (kmol/h)": [
            f"{r['F_feed_kmolh']:.2f}",
            f"{r['F_top_kmolh']:.2f}",
            f"{r['F_bot_kmolh']:.2f}",
        ],
        "Flow (kg/h)": [
            f"{r['F_feed_kgh']:,.1f}",
            f"{r['F_top_kgh']:,.1f}",
            f"{r['F_bot_kgh']:,.1f}",
        ],
        "EtOH mol frac": [
            f"{r['x_eth']:.4f}",
            f"{x_top_eth_res:.4f}",
            f"{x_bot_eth_res:.4f}",
        ],
        "Avg MW (g/mol)": [
            f"{r['mw_feed']:.3f}",
            f"{r['mw_top']:.3f}",
            f"{r['mw_bot']:.3f}",
        ],
    }
    st.dataframe(pd.DataFrame(stream_data), use_container_width=True)

    # KPI metrics
    scale = r["scale"]
    st.markdown("#### Key Process Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condenser Duty", f"{r['Q_cond_kw']:.1f} kW",
              delta=f"{(scale - 1) * 100:+.1f}% vs design")
    c2.metric("Reboiler Duty", f"{r['Q_reb_kw']:.1f} kW",
              delta=f"{(scale - 1) * 100:+.1f}% vs design")
    c3.metric("Reflux Ratio (L/D)", f"{r['reflux_ratio']:.2f}")
    c4.metric("EtOH Recovery", f"{min(r['ethanol_recovery'], 100.0):.1f} %")

    st.caption(
        "**Delta arrows**: ↑ green = operating above design point; ↓ red = operating below design point. "
        "A higher EtOH Recovery (close to 100 %) indicates efficient separation."
    )

    st.markdown("#### Feed Thermal Condition")
    c5, c6 = st.columns(2)
    c5.metric("q-parameter", f"{r['q']:.3f}",
              help="q=1: saturated liquid; q>1: sub-cooled liquid; q<1: partial vapour")
    c6.metric("Feed MW", f"{r['mw_feed']:.3f} g/mol")

    # Stream flow bar charts
    st.markdown("#### Stream Flow Breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Feed", "Distillate", "Bottoms"]
    values_kgh = [r["F_feed_kgh"], r["F_top_kgh"], r["F_bot_kgh"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    axes[0].barh(labels, values_kgh, color=colors)
    axes[0].set_xlabel("Flow (kg/h)")
    axes[0].set_title("Mass Flow Rates")
    for i, v in enumerate(values_kgh):
        axes[0].text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=9)

    duty_labels = ["Condenser", "Reboiler"]
    duty_vals = [r["Q_cond_kw"], r["Q_reb_kw"]]
    duty_colors = ["#8172B2", "#CCB974"]
    axes[1].bar(duty_labels, duty_vals, color=duty_colors, width=0.4)
    axes[1].set_ylabel("Duty (kW)")
    axes[1].set_title("Energy Duties")
    for i, v in enumerate(duty_vals):
        axes[1].text(i, v + 5, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Details expander ──────────────────────────────────────────────────────
    if is_dwsim_mode:
        with st.expander("🔬 DWSIM Simulation Details"):
            st.markdown(f"**Flowsheet file:** `{_cfg.SIMULATION_FILE}`")
            st.markdown(f"**Solver status:** {r.get('solver_status', 'N/A')}")
            compounds = getattr(_cfg, "DWSIM_COMPOUNDS", ["Ethanol", "Water"])
            st.markdown(f"**Compounds:** {', '.join(compounds)}")
            st.markdown(f"**Timeout:** {getattr(_cfg, 'DWSIM_TIMEOUT', 120)} s")
            raw = r.get("dwsim_raw")
            if raw:
                safe_raw = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        safe_raw[k] = {
                            kk: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk, vv in v.items()
                            if kk != "composition"
                        }
                    else:
                        safe_raw[k] = v
                st.json(safe_raw)
    else:
        with st.expander("📐 Scaling Equations Used"):
            st.latex(r"\dot{F}_{\text{scaled}} = \dot{F}_{\text{design}} \times \alpha, "
                     r"\quad \alpha = \frac{\dot{m}_{\text{feed}}}{\dot{m}_{\text{feed,design}}}")
            st.latex(r"\dot{Q}_{\text{cond/reb,scaled}} = \dot{Q}_{\text{cond/reb,design}} \times \alpha")
            st.latex(r"\frac{L}{D} \approx \frac{\dot{V} - \dot{D}}{\dot{D}}, "
                     r"\quad \dot{V} = \frac{\dot{Q}_{\text{cond}}}{\Delta H_{\text{vap,top}}}")
            st.latex(r"q = 1 + \frac{c_{p,L}\,(T_{\text{bubble}} - T_{\text{feed}})}{\lambda_{\text{feed}}}")
            st.markdown("""
            These are **first-order scaling relationships** valid near the design point.
            Large departures from the base composition or flow would require a full
            rigorous simulation in DWSIM or Aspen.
            """)

