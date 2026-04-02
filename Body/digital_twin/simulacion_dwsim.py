"""
Digital Twin — Sub-page 1: DWSIM Design-Point Simulation
=========================================================
Allows the user to specify feed conditions (T, P, composition, flow rate)
and computes the expected column performance by scaling the DWSIM steady-state
design point stored in Simulation/config.py.
No DWSIM COM runtime is required; all calculations are analytic scale-ups of
the reference design.
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Import shared configuration from Simulation/config.py ───────────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── Local-only constants not defined in config.py ───────────────────────────
_TARGET_ETHANOL_BOTTOM = 0.02    # mol fraction of ethanol in bottoms (design target)
_MW_ETHANOL = 46.068             # g/mol
_MW_WATER = 18.015               # g/mol

# Convenient aliases matching config naming
_FLOW_FEED_BASE_KGH = _cfg.FLOW_FEED_BASE
_SPLIT_TOP = _cfg.SPLIT_TOP
_SPLIT_BOTTOM = _cfg.SPLIT_BOTTOM
_Q_COND_BASE_KW = _cfg.Q_COND_BASE
_Q_REB_BASE_KW = _cfg.Q_REB_BASE
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP

_FLOW_FEED_BASE_KMOLH = _FLOW_FEED_BASE_KGH / (
    _TARGET_ETHANOL_TOP * _MW_ETHANOL + (1 - _TARGET_ETHANOL_TOP) * _MW_WATER
)  # approximate kmol/h at base conditions


def _average_mw(x_eth: float) -> float:
    """Mean molecular weight for an ethanol-water mixture."""
    return x_eth * _MW_ETHANOL + (1.0 - x_eth) * _MW_WATER


def simulacion_dwsim_page():
    st.header("⚙️ DWSIM Design-Point Simulation")
    st.markdown("""
    This page scales the **DWSIM steady-state design** of a continuous ethanol
    distillation column to the feed conditions you specify.
    The reference flowsheet (`ethanol.dwxmz`) defines the base split ratios
    and energy duties; all results are linearly scaled to the new feed.
    """)

    st.markdown("---")

    # ── Transfer-function summary ─────────────────────────────────────────────
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
        st.metric("Distillate split", f"{_SPLIT_TOP*100:.0f} %")
    with col_b:
        st.metric("Condenser duty (base)", f"{_Q_COND_BASE_KW:,.2f} kW")
        st.metric("Reboiler duty (base)", f"{_Q_REB_BASE_KW:,.2f} kW")
    with col_c:
        st.metric("Target EtOH in distillate", f"{_TARGET_ETHANOL_TOP*100:.0f} mol%")
        st.metric("Target EtOH in bottoms", f"{_TARGET_ETHANOL_BOTTOM*100:.0f} mol%")

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
    st.markdown("Set the feed conditions in the sidebar, then click **▶ Run Simulation** to compute results.")
    if st.button("▶ Run Simulation", key="btn_dwsim_run"):
        # ── Calculations ──────────────────────────────────────────────────────
        mw_feed = _average_mw(x_eth)           # g/mol
        F_feed_kgh = F_feed_kmolh * mw_feed    # kg/h

        # Scale factor vs. DWSIM reference (mass-flow based)
        scale = F_feed_kgh / _FLOW_FEED_BASE_KGH

        # Stream flows
        F_top_kgh = F_feed_kgh * _SPLIT_TOP
        F_bot_kgh = F_feed_kgh * _SPLIT_BOTTOM
        mw_top = _average_mw(_TARGET_ETHANOL_TOP)
        mw_bot = _average_mw(_TARGET_ETHANOL_BOTTOM)
        F_top_kmolh = F_top_kgh / mw_top
        F_bot_kmolh = F_bot_kgh / mw_bot

        # Energy duties (linear scale)
        Q_cond_kw = _Q_COND_BASE_KW * scale
        Q_reb_kw = _Q_REB_BASE_KW * scale

        # Derived metrics
        dh_vap_ethanol = 38.56     # kJ/mol  latent heat of vaporisation at 78 °C
        dh_vap_water = 40.65       # kJ/mol
        dh_vap_top = (
            _TARGET_ETHANOL_TOP * dh_vap_ethanol
            + (1 - _TARGET_ETHANOL_TOP) * dh_vap_water
        )  # kJ/mol
        V_vapour = Q_cond_kw / dh_vap_top   # kmol/h vapour from condenser
        reflux_ratio = max(0.0, (V_vapour - F_top_kmolh) / F_top_kmolh) if F_top_kmolh > 0 else 0.0

        ethanol_recovery = (
            F_top_kmolh * _TARGET_ETHANOL_TOP
            / max(F_feed_kmolh * x_eth, 1e-9) * 100
        )

        # Thermal condition (q-parameter)
        T_bp = 78.4 + (P_feed - 101.325) * 0.03     # rough bubble-point shift
        cp_liquid = 2.9     # kJ/(kg·K)  approximate
        lambda_feed = dh_vap_top * mw_feed / 1000.0  # kJ/kg
        q = 1.0 + cp_liquid * max(T_bp - T_feed, 0.0) / lambda_feed

        # Store all computed results in session state
        st.session_state["sim_results"] = {
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
        }

    # ── Results Display (only from stored state) ──────────────────────────────
    if "sim_results" not in st.session_state:
        st.info("👆 Configure the feed conditions and click **▶ Run Simulation** to see results.")
        return

    r = st.session_state["sim_results"]
    F_feed_kmolh = r["F_feed_kmolh"]
    F_feed_kgh = r["F_feed_kgh"]
    F_top_kgh = r["F_top_kgh"]
    F_bot_kgh = r["F_bot_kgh"]
    F_top_kmolh = r["F_top_kmolh"]
    F_bot_kmolh = r["F_bot_kmolh"]
    Q_cond_kw = r["Q_cond_kw"]
    Q_reb_kw = r["Q_reb_kw"]
    scale = r["scale"]
    reflux_ratio = r["reflux_ratio"]
    ethanol_recovery = r["ethanol_recovery"]
    q = r["q"]
    mw_feed = r["mw_feed"]
    mw_top = r["mw_top"]
    mw_bot = r["mw_bot"]
    x_eth_res = r["x_eth"]

    st.subheader("📊 Scaled Simulation Results")

    # Stream table
    stream_data = {
        "Stream": ["Feed", "Distillate (Top)", "Bottoms"],
        "Flow (kmol/h)": [
            f"{F_feed_kmolh:.2f}",
            f"{F_top_kmolh:.2f}",
            f"{F_bot_kmolh:.2f}",
        ],
        "Flow (kg/h)": [
            f"{F_feed_kgh:,.1f}",
            f"{F_top_kgh:,.1f}",
            f"{F_bot_kgh:,.1f}",
        ],
        "EtOH mol frac": [
            f"{x_eth_res:.4f}",
            f"{_TARGET_ETHANOL_TOP:.4f}",
            f"{_TARGET_ETHANOL_BOTTOM:.4f}",
        ],
        "Avg MW (g/mol)": [
            f"{mw_feed:.3f}",
            f"{mw_top:.3f}",
            f"{mw_bot:.3f}",
        ],
    }
    st.dataframe(pd.DataFrame(stream_data), use_container_width=True)

    # KPI metrics
    st.markdown("#### Key Process Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Condenser Duty", f"{Q_cond_kw:.1f} kW", delta=f"{(scale-1)*100:+.1f}% vs design")
    c2.metric("Reboiler Duty", f"{Q_reb_kw:.1f} kW", delta=f"{(scale-1)*100:+.1f}% vs design")
    c3.metric("Reflux Ratio (L/D)", f"{reflux_ratio:.2f}")
    c4.metric("EtOH Recovery", f"{min(ethanol_recovery, 100.0):.1f} %")

    st.caption(
        "**Delta arrows**: ↑ green = operating above design point; ↓ red = operating below design point. "
        "A higher EtOH Recovery (close to 100 %) indicates efficient separation."
    )

    st.markdown("#### Feed Thermal Condition")
    c5, c6 = st.columns(2)
    c5.metric("q-parameter", f"{q:.3f}",
              help="q=1: saturated liquid; q>1: sub-cooled liquid; q<1: partial vapour")
    c6.metric("Feed MW", f"{mw_feed:.3f} g/mol")

    # ── Visual: Stream Sankey-style bar chart ─────────────────────────────────
    st.markdown("#### Stream Flow Breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: mass flows
    labels = ["Feed", "Distillate", "Bottoms"]
    values_kgh = [F_feed_kgh, F_top_kgh, F_bot_kgh]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    axes[0].barh(labels, values_kgh, color=colors)
    axes[0].set_xlabel("Flow (kg/h)")
    axes[0].set_title("Mass Flow Rates")
    for i, v in enumerate(values_kgh):
        axes[0].text(v * 1.01, i, f"{v:,.0f}", va="center", fontsize=9)

    # Right: energy duties
    duty_labels = ["Condenser", "Reboiler"]
    duty_vals = [Q_cond_kw, Q_reb_kw]
    duty_colors = ["#8172B2", "#CCB974"]
    axes[1].bar(duty_labels, duty_vals, color=duty_colors, width=0.4)
    axes[1].set_ylabel("Duty (kW)")
    axes[1].set_title("Energy Duties")
    for i, v in enumerate(duty_vals):
        axes[1].text(i, v + 5, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Theory / equations ────────────────────────────────────────────────────
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
