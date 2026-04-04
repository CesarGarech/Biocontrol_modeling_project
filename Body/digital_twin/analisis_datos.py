"""
Digital Twin — Sub-page 2: SCADA Data Generation & Analysis
============================================================
Three-step interactive pipeline:
  Step 1 — Generate raw noisy SCADA data and visualise it.
  Step 2 — Inject outliers (manual time indices or random fraction) and
            display IQR-detected anomalies overlaid on the raw signal.
  Step 3 — Run the full analysis: MA filter → WLS reconciliation → KPIs.
All design-point constants are sourced from Simulation/config.py.
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

# ── Import shared configuration from Simulation/config.py ───────────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402
import dwsim_interface as _dwsim  # noqa: E402  (import-safe; check _dwsim.DWSIM_AVAILABLE)

_EPSILON = 1e-9   # numerical guard against division by zero

# ── Design-point aliases (sourced from config) ───────────────────────────────
_FLOW_FEED_BASE = _cfg.FLOW_FEED_BASE
_SPLIT_TOP = _cfg.SPLIT_TOP
_SPLIT_BOTTOM = _cfg.SPLIT_BOTTOM
_Q_COND_BASE = _cfg.Q_COND_BASE
_Q_REB_BASE = _cfg.Q_REB_BASE
_DESIGN_ENERGY_RATIO = _Q_COND_BASE / _Q_REB_BASE
_MAX_MASS_ERROR = _cfg.MAX_MASS_BALANCE_ERROR

# Traffic-light thresholds (% of time steps that must pass the KPI target)
_TL_GREEN = 80.0   # ≥ 80 %  → green
_TL_YELLOW = 60.0  # 60–80 % → yellow  (< 60 % → red)


# ── Helper: generate raw noisy data (no outliers) ────────────────────────────

def _generate_raw_data(
    n_points: int,
    sigma_feed: float,
    sigma_top: float,
    sigma_bottom: float,
    sigma_q_cond: float,
    sigma_q_reb: float,
    seed: int,
) -> pd.DataFrame:
    """Generate synthetic SCADA readings with Gaussian noise only."""
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range(start="2026-01-01 08:00", periods=n_points, freq="h")
    trend = np.sin(np.linspace(0, np.pi, n_points)) * 500.0
    feed_real = _FLOW_FEED_BASE + trend

    df = pd.DataFrame({"Timestamp": timestamp})
    df["F_feed_raw"] = rng.normal(feed_real, sigma_feed)
    df["F_top_raw"] = rng.normal(feed_real * _SPLIT_TOP, sigma_top)
    df["F_bottom_raw"] = rng.normal(feed_real * _SPLIT_BOTTOM, sigma_bottom)
    df["Q_cond_raw"] = rng.normal(_Q_COND_BASE, sigma_q_cond, n_points)
    df["Q_reb_raw"] = rng.normal(_Q_REB_BASE, sigma_q_reb, n_points)
    return df


# ── Helper: inject outliers at given indices ─────────────────────────────────

def _inject_outliers(df: pd.DataFrame, outlier_indices: list[int], seed: int) -> pd.DataFrame:
    """Return a copy of df with spike/drop anomalies at the given row indices."""
    rng = np.random.default_rng(seed + 99)
    raw_cols = ["F_feed_raw", "F_top_raw", "F_bottom_raw", "Q_cond_raw", "Q_reb_raw"]
    df_out = df.copy()
    for idx in outlier_indices:
        if 0 <= idx < len(df_out):
            col = rng.choice(raw_cols)
            df_out.loc[idx, col] *= float(rng.choice([1.8, 0.3]))
    return df_out


# ── Helper: IQR outlier detection + MA filter ────────────────────────────────

def _clean_and_filter(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Flag outliers (IQR), interpolate, then apply centred moving-average."""
    df_c = df.copy()
    raw_cols = ["F_feed_raw", "F_top_raw", "F_bottom_raw", "Q_cond_raw", "Q_reb_raw"]
    for col in raw_cols:
        q1 = df_c[col].quantile(0.25)
        q3 = df_c[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df_c[col] < lo) | (df_c[col] > hi)
        df_c[f"{col}_outlier"] = mask
        df_c.loc[mask, col] = np.nan
        df_c[col] = df_c[col].interpolate(method="linear")
        filt_col = col.replace("_raw", "_filtered")
        df_c[filt_col] = df_c[col].rolling(window=window, center=True).mean()
    return df_c.dropna().reset_index(drop=True)


# ── Helper: WLS reconciliation ────────────────────────────────────────────────

def _reconcile(
    df: pd.DataFrame,
    sigma_feed: float,
    sigma_top: float,
    sigma_bottom: float,
) -> pd.DataFrame:
    """Row-wise SLSQP reconciliation enforcing F_feed = F_top + F_bottom.

    Only adds NEW columns; never duplicates columns already present in df.
    """
    sigma_mass = np.array([sigma_feed, sigma_top, sigma_bottom])
    results = []

    for _, row in df.iterrows():
        y = np.array([
            row["F_feed_filtered"],
            row["F_top_filtered"],
            row["F_bottom_filtered"],
        ])

        def obj(x, _y=y, _s=sigma_mass):
            return np.sum(((_y - x) / _s) ** 2)

        def con(x):
            return x[0] - x[1] - x[2]

        sol = minimize(
            obj, x0=y.copy(),
            constraints={"type": "eq", "fun": con},
            method="SLSQP",
        )
        F_f, F_t, F_b = sol.x

        err_before = abs(y[0] - y[1] - y[2]) / (y[0] + _EPSILON) * 100.0
        actual_split = F_t / (F_f + _EPSILON)
        kpi_sep = 100.0 - abs(actual_split - _SPLIT_TOP) / _SPLIT_TOP * 100.0

        q_cond = row["Q_cond_filtered"]
        q_reb = row["Q_reb_filtered"]
        actual_ratio = q_cond / (q_reb + _EPSILON)
        kpi_energy = (
            100.0
            - abs(actual_ratio - _DESIGN_ENERGY_RATIO) / _DESIGN_ENERGY_RATIO * 100.0
        )

        results.append({
            "Timestamp": row["Timestamp"],
            "F_feed_rec": F_f,
            "F_top_rec": F_t,
            "F_bottom_rec": F_b,
            # NOTE: Q_cond_filtered / Q_reb_filtered are already in df — omitted
            "Error_Mass_Before_%": err_before,
            "KPI_Separation_%": kpi_sep,
            "KPI_Energy_%": kpi_energy,
        })

    df_rec = pd.DataFrame(results).drop("Timestamp", axis=1)
    return pd.concat([df.reset_index(drop=True), df_rec], axis=1)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_raw_signals(df: pd.DataFrame, title_suffix: str = "") -> plt.Figure:
    """3×2 subplot of all five raw signals (style matches dashboard.plot_outliers)."""
    fig, axs = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"SCADA Raw Signals — Ethanol Distillation Column{title_suffix}",
        fontsize=13, fontweight="bold",
    )

    mass_cfg = [
        ("F_feed_raw", "Feed Flow", "gray"),
        ("F_top_raw", "Top (Distillate) Flow", "tab:blue"),
        ("F_bottom_raw", "Bottom Flow", "tab:orange"),
    ]
    energy_cfg = [
        ("Q_cond_raw", "Condenser Duty (Q_cond)", "tab:cyan"),
        ("Q_reb_raw", "Reboiler Duty (Q_reb)", "tab:red"),
    ]

    for i, (col, title, color) in enumerate(mass_cfg):
        ax = axs[i, 0]
        ax.plot(df["Timestamp"], df[col], color=color, alpha=0.7, label="Raw Signal")
        # Mark outliers if IQR flags exist
        out_col = f"{col}_outlier"
        if out_col in df.columns:
            outliers = df[df[out_col]]
            ax.scatter(
                outliers["Timestamp"], outliers[col],
                color="red", s=50, zorder=5, label="Outlier (IQR)",
            )
        ax.set_title(title)
        ax.set_ylabel("Flow (kg/h)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    for i, (col, title, color) in enumerate(energy_cfg):
        ax = axs[i, 1]
        ax.plot(df["Timestamp"], df[col], color=color, alpha=0.7, label="Raw Signal")
        out_col = f"{col}_outlier"
        if out_col in df.columns:
            outliers = df[df[out_col]]
            ax.scatter(
                outliers["Timestamp"], outliers[col],
                color="red", s=50, zorder=5, label="Outlier (IQR)",
            )
        ax.set_title(title)
        ax.set_ylabel("Duty (kW)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    axs[2, 1].axis("off")
    axs[2, 0].set_xlabel("Time")
    axs[1, 1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def _plot_dashboard(df: pd.DataFrame, threshold_mass: float) -> plt.Figure:
    """4-panel reconciliation + KPI dashboard (style matches dashboard.plot_dashboard)."""
    fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"Digital Twin Results — Ethanol Column {_cfg.TAG_COLUMN}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — Feed signal treatment
    axs[0].plot(df["Timestamp"], df["F_feed_raw"], color="lightgray", label="Raw")
    axs[0].plot(df["Timestamp"], df["F_feed_filtered"],
                color="blue", alpha=0.6, label="Filtered (MA)")
    axs[0].plot(df["Timestamp"], df["F_feed_rec"],
                color="red", ls="--", label="Reconciled (WLS)")
    axs[0].set_title("Signal Treatment: Feed Flow ($F_{feed}$)")
    axs[0].set_ylabel("Flow (kg/h)")
    axs[0].legend(fontsize=9)
    axs[0].grid(True)

    # Panel 2 — Mass balance closure error
    axs[1].plot(df["Timestamp"], df["Error_Mass_Before_%"],
                color="orange", label="Error before reconciliation")
    axs[1].axhline(y=0, color="green", ls="--", label="Reconciled balance (0 %)")
    axs[1].axhline(y=threshold_mass, color="red", ls=":",
                   label=f"Tolerance ({threshold_mass} %)")
    axs[1].set_title("Mass Conservation — Balance Closure")
    axs[1].set_ylabel("Closure Error (%)")
    axs[1].legend(fontsize=9)
    axs[1].grid(True)

    # Panel 3 — Separation adherence KPI
    axs[2].plot(df["Timestamp"], df["KPI_Separation_%"],
                color="purple", label="Separation Adherence")
    axs[2].axhline(y=100, color="gray", ls=":")
    axs[2].set_title(f"KPI: Adherence to Target Distillate Split ({_cfg.SPLIT_TOP*100:.0f} %)")
    axs[2].set_ylabel("Adherence (%)")
    axs[2].legend(fontsize=9)
    axs[2].grid(True)

    # Panel 4 — Energy efficiency KPI
    axs[3].plot(df["Timestamp"], df["KPI_Energy_%"],
                color="teal", label="Energy Efficiency")
    axs[3].axhline(y=100, color="gray", ls=":")
    axs[3].set_title("KPI: Energy Ratio Adherence (Q_cond / Q_reb)")
    axs[3].set_ylabel("Efficiency (%)")
    axs[3].set_xlabel("Time")
    axs[3].legend(fontsize=9)
    axs[3].grid(True)

    fig.tight_layout()
    return fig


# ── Streamlit page ────────────────────────────────────────────────────────────

def analisis_datos_page():
    st.header("📡 SCADA Data Generation & Analysis")
    st.markdown("""
    Follow the **three steps** below.  
    Each step builds on the previous one — changes to sidebar parameters
    only take effect when you click the corresponding action button.
    """)
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Analysis Parameters")

        with st.expander("1. Data Generation", expanded=True):
            n_points = st.slider("Number of SCADA points", 50, 300, _cfg.N_POINTS, 10, key="da_n")
            seed = int(st.number_input("Random seed", 0, 9999, _cfg.SEED, key="da_seed"))

        with st.expander("2. Sensor Noise (σ)", expanded=True):
            sigma_feed = st.number_input("σ Feed (kg/h)", 10.0, 1000.0, float(_cfg.SIGMA_FEED), 10.0, key="da_sf")
            sigma_top = st.number_input("σ Distillate (kg/h)", 10.0, 500.0, float(_cfg.SIGMA_TOP), 10.0, key="da_st")
            sigma_bot = st.number_input("σ Bottoms (kg/h)", 10.0, 500.0, float(_cfg.SIGMA_BOTTOM), 10.0, key="da_sb")
            sigma_qc = st.number_input("σ Q_cond (kW)", 1.0, 200.0, float(_cfg.SIGMA_Q_COND), 5.0, key="da_sqc")
            sigma_qr = st.number_input("σ Q_reb (kW)", 1.0, 200.0, float(_cfg.SIGMA_Q_REB), 5.0, key="da_sqr")

        with st.expander("3. Outlier Injection", expanded=True):
            use_random = st.checkbox("Use random outliers", value=False, key="da_random_out")
            if use_random:
                outlier_frac = st.slider(
                    "Outlier fraction", 0.0, 0.20, 0.05, 0.01, key="da_outfrac"
                )
                outlier_indices_str = None
            else:
                st.markdown("Enter time-step indices (0-based, comma-separated):")
                outlier_indices_str = st.text_input(
                    "Outlier indices", value="12, 22, 30, 48, 55, 67, 75, 85, 92",
                    key="da_out_idx"
                )
                outlier_frac = None

        with st.expander("4. Filter & Reconciliation", expanded=True):
            window = st.slider(
                "Moving-average window W", 3, 21, _cfg.WINDOW_SIZE, 2, key="da_win",
                help="Centred MA; even values rounded up to next odd"
            )
            if window % 2 == 0:
                window += 1

        with st.expander("5. KPI Targets", expanded=True):
            target_sep = st.slider("Target Separation Adherence (%)", 70.0, 99.0, 90.0, 1.0, key="da_tsep")
            target_energy = st.slider("Target Energy Efficiency (%)", 70.0, 99.0, 85.0, 1.0, key="da_teng")
            threshold_mass = st.slider(
                "Max Mass Balance Error (%)", 0.5, 10.0,
                float(_cfg.MAX_MASS_BALANCE_ERROR), 0.5, key="da_tmass"
            )

        with st.expander("6. Batch Simulation Feed", expanded=False):
            st.markdown("Feed conditions used for the point-by-point simulation.")
            batch_T_feed = st.number_input(
                "Feed Temperature (°C)", 20.0, 150.0, 30.0, 1.0, key="da_bT"
            )
            batch_P_feed = st.number_input(
                "Feed Pressure (kPa)", 50.0, 500.0, 100.0, 1.0, key="da_bP"
            )
            batch_x_eth = st.slider(
                "Ethanol mole fraction", 0.0, 1.0,
                0.40,
                0.01, key="da_bxeth"
            )
            batch_x_water = round(max(0.0, 1.0 - batch_x_eth), 4)

    # ── Step 1: Generate raw data ──────────────────────────────────────────────
    st.subheader("🔵 Step 1 — Generate Raw SCADA Data")
    if st.button("▶ Generate Raw Data", key="btn_gen"):
        df_raw = _generate_raw_data(
            n_points, sigma_feed, sigma_top, sigma_bot,
            sigma_qc, sigma_qr, seed,
        )
        st.session_state["da_df_raw"] = df_raw
        # Clear downstream results when raw data is regenerated
        st.session_state.pop("da_df_with_outliers", None)
        st.session_state.pop("da_df_final", None)

    if "da_df_raw" in st.session_state:
        df_raw = st.session_state["da_df_raw"]
        st.success(f"Raw data generated — {len(df_raw)} time steps.")
        fig1 = _plot_raw_signals(df_raw, " — Gaussian Noise Only")
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("👆 Click **Generate Raw Data** to start.")

    st.markdown("---")

    # ── Step 2: Inject outliers & visualise ───────────────────────────────────
    st.subheader("🟡 Step 2 — Inject Outliers & IQR Detection")

    if "da_df_raw" not in st.session_state:
        st.info("Complete Step 1 first.")
    else:
        if st.button("▶ Inject Outliers", key="btn_outlier"):
            df_raw = st.session_state["da_df_raw"]

            # Resolve outlier indices
            if use_random:
                rng_tmp = np.random.default_rng(seed + 1)
                n_out = max(1, int(n_points * outlier_frac))
                indices = list(rng_tmp.choice(n_points, size=n_out, replace=False))
            else:
                try:
                    indices = [int(x.strip()) for x in outlier_indices_str.split(",") if x.strip()]
                except ValueError:
                    st.error("Invalid index list — use comma-separated integers like: 10, 25, 40")
                    indices = []

            df_with_out = _inject_outliers(df_raw, indices, seed)

            # Run IQR detection on the data with injected outliers (for display)
            df_iqr = _add_iqr_flags(df_with_out)
            st.session_state["da_df_with_outliers"] = df_with_out
            st.session_state["da_df_iqr"] = df_iqr
            st.session_state["da_outlier_indices"] = indices
            st.session_state.pop("da_df_final", None)

        if "da_df_iqr" in st.session_state:
            df_iqr = st.session_state["da_df_iqr"]
            indices = st.session_state.get("da_outlier_indices", [])
            st.success(
                f"Outliers injected at indices: {indices}  |  "
                f"IQR detected: {sum(df_iqr[[c for c in df_iqr.columns if c.endswith('_outlier')]].any(axis=1))} affected rows"
            )
            fig2 = _plot_raw_signals(df_iqr, " — IQR Anomaly Detection")
            st.pyplot(fig2)
            plt.close(fig2)

            # Outlier count table
            with st.expander("📌 Outlier Count per Sensor"):
                import re as _re
                out_cols = [c for c in df_iqr.columns if c.endswith("_outlier")]
                summary = {
                    _re.sub(r"_raw_outlier$|_outlier$", "", c): int(df_iqr[c].sum())
                    for c in out_cols
                }
                st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Outliers (IQR)"]))
        else:
            st.info("👆 Click **Inject Outliers** to proceed.")

    st.markdown("---")

    # ── Step 3: Full analysis pipeline ────────────────────────────────────────
    st.subheader("🟢 Step 3 — MA Filter → WLS Reconciliation → KPIs")

    if "da_df_with_outliers" not in st.session_state:
        st.info("Complete Steps 1 & 2 first.")
    else:
        if st.button("▶ Run Analysis", key="btn_analyze"):
            with st.spinner("Running MA filter and WLS reconciliation…"):
                df_filt = _clean_and_filter(st.session_state["da_df_with_outliers"], window)
                df_final = _reconcile(df_filt, sigma_feed, sigma_top, sigma_bot)
            st.session_state["da_df_final"] = df_final
            st.session_state["da_df_filt"] = df_filt
            # Snapshot the thresholds used for this run so display does not
            # change when the user adjusts sidebar sliders without re-running.
            st.session_state["da_thresholds"] = {
                "target_sep": target_sep,
                "target_energy": target_energy,
                "threshold_mass": threshold_mass,
            }

        if "da_df_final" in st.session_state:
            df_final = st.session_state["da_df_final"]
            thr = st.session_state["da_thresholds"]
            target_sep_disp = thr["target_sep"]
            target_energy_disp = thr["target_energy"]
            threshold_mass_disp = thr["threshold_mass"]

            st.success(f"Analysis complete — {len(df_final)} usable time steps after filtering.")

            # KPI metric cards
            avg_mass_err = df_final["Error_Mass_Before_%"].mean()
            avg_sep = df_final["KPI_Separation_%"].mean()
            avg_energy = df_final["KPI_Energy_%"].mean()
            pct_sep_ok = (df_final["KPI_Separation_%"] >= target_sep_disp).mean() * 100
            pct_energy_ok = (df_final["KPI_Energy_%"] >= target_energy_disp).mean() * 100
            pct_mass_ok = (df_final["Error_Mass_Before_%"] <= threshold_mass_disp).mean() * 100

            def _traffic_light(val: float) -> str:
                return "🟢" if val >= _TL_GREEN else ("🟡" if val >= _TL_YELLOW else "🔴")

            c1, c2, c3 = st.columns(3)
            c1.metric(
                f"{_traffic_light(pct_sep_ok)} Separation Adherence",
                f"{avg_sep:.1f} %",
                delta=f"{avg_sep - target_sep_disp:+.1f}% vs target",
                help=f"{pct_sep_ok:.0f}% of steps ≥ {target_sep_disp}%",
            )
            c2.metric(
                f"{_traffic_light(pct_energy_ok)} Energy Efficiency",
                f"{avg_energy:.1f} %",
                delta=f"{avg_energy - target_energy_disp:+.1f}% vs target",
                help=f"{pct_energy_ok:.0f}% of steps ≥ {target_energy_disp}%",
            )
            c3.metric(
                f"{_traffic_light(pct_mass_ok)} Mass Balance Error",
                f"{avg_mass_err:.2f} %",
                delta=f"{avg_mass_err - threshold_mass_disp:+.2f}% vs threshold",
                delta_color="inverse",
                help=f"{pct_mass_ok:.0f}% of steps within ±{threshold_mass_disp}%",
            )

            # ── KPI colour legend ─────────────────────────────────────────────
            with st.expander("ℹ️ How to read these KPIs"):
                st.markdown(f"""
**Status icon** (header of each card):
- 🟢 **Green** — KPI meets the target in **≥ {_TL_GREEN:.0f} %** of time steps. The process is operating well.
- 🟡 **Yellow** — KPI meets the target in **{_TL_YELLOW:.0f}–{_TL_GREEN:.0f} %** of time steps. Monitor closely.
- 🔴 **Red** — KPI meets the target in **< {_TL_YELLOW:.0f} %** of time steps. Attention required.

**Delta arrow** (small number below the main value):
- ↑ **Green delta** — the average KPI is *above* the target you set → favourable.
- ↓ **Red delta** — the average KPI is *below* the target you set → needs improvement.
- For **Mass Balance Error** the arrow is inverted: ↑ red means the error exceeds the threshold.

**Individual KPI meanings**:
- *Separation Adherence* — how closely the actual distillate split matches the design target ({_cfg.SPLIT_TOP*100:.0f} % of feed). 100 % = perfect adherence.
- *Energy Efficiency* — how close the Q_cond / Q_reb ratio is to the design ratio. 100 % = operating at design conditions.
- *Mass Balance Error* — percentage imbalance (F_feed − F_top − F_bottom) before reconciliation. Lower is better; target ≤ {threshold_mass_disp} %.

> Results reflect the **last run**. Changing sidebar parameters requires clicking **▶ Run Analysis** again to update the display.
""")

            st.markdown("---")

            # 4-panel dashboard
            fig3 = _plot_dashboard(df_final, threshold_mass_disp)
            st.pyplot(fig3)
            plt.close(fig3)

            # Data table
            with st.expander("📋 View Reconciled Data Table"):
                display_cols = [
                    "Timestamp", "F_feed_rec", "F_top_rec", "F_bottom_rec",
                    "Q_cond_filtered", "Q_reb_filtered",
                    "Error_Mass_Before_%", "KPI_Separation_%", "KPI_Energy_%",
                ]
                display_df = df_final[display_cols].copy()
                st.dataframe(
                    display_df.style.format({
                        "F_feed_rec": "{:.1f}",
                        "F_top_rec": "{:.1f}",
                        "F_bottom_rec": "{:.1f}",
                        "Q_cond_filtered": "{:.2f}",
                        "Q_reb_filtered": "{:.2f}",
                        "Error_Mass_Before_%": "{:.3f}",
                        "KPI_Separation_%": "{:.2f}",
                        "KPI_Energy_%": "{:.2f}",
                    }),
                    use_container_width=True,
                )
                csv = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV", data=csv,
                    file_name="digital_twin_results.csv", mime="text/csv",
                )

            # ── Step 4: Batch Simulation ──────────────────────────────────────
            st.markdown("---")
            st.subheader("🔵 Step 4 — Point-by-Point Simulation (DWSIM / Scaling)")

            if _dwsim.DWSIM_AVAILABLE:
                st.success(
                    "🟢 **DWSIM available** — batch simulation will use DWSIM for each point "
                    "(falls back to scaling on per-row errors)."
                )
            else:
                st.warning(
                    "🟡 **DWSIM not available** — batch simulation will use the linear-scaling "
                    "fallback for each data point."
                )

            st.markdown(
                "Runs a simulation for **every filtered/reconciled data point**, using the "
                "reconciled feed flow (`F_feed_rec`) and the feed conditions from the sidebar "
                "*(Section 6)*. The predicted stream flows and duties are then compared with the "
                "SCADA measurements."
            )

            if st.button("▶ Run Batch Simulation", key="btn_batch_sim"):
                interf = st.session_state.get("dwsim_interf")
                sim_obj = st.session_state.get("dwsim_sim")

                # If DWSIM objects are not already in session state, try to initialise
                if _dwsim.DWSIM_AVAILABLE and (interf is None or sim_obj is None):
                    with st.spinner("Initialising DWSIM…"):
                        try:
                            interf = _dwsim.init_dwsim()
                            sim_obj = _dwsim.load_simulation(interf)
                            st.session_state["dwsim_interf"] = interf
                            st.session_state["dwsim_sim"] = sim_obj
                        except Exception as exc:
                            st.warning(f"DWSIM init failed ({exc}); using scaling fallback.")
                            interf = None
                            sim_obj = None

                progress = st.progress(0)
                with st.spinner(f"Simulating {len(df_final)} data points…"):
                    df_batch = _run_batch_simulation(
                        df_final,
                        T_feed=batch_T_feed,
                        P_feed=batch_P_feed,
                        x_eth=batch_x_eth,
                        x_water=batch_x_water,
                        interf=interf,
                        sim=sim_obj,
                        progress_bar=progress,
                    )
                progress.empty()
                st.session_state["da_df_batch"] = df_batch

            if "da_df_batch" in st.session_state:
                df_batch = st.session_state["da_df_batch"]
                modes = df_batch["sim_mode"].value_counts().to_dict()
                mode_str = ", ".join(f"{v} {k}" for k, v in modes.items())
                st.success(f"Batch simulation complete — {len(df_batch)} points ({mode_str}).")

                fig4 = _plot_comparison(df_batch)
                st.pyplot(fig4)
                plt.close(fig4)

                with st.expander("📋 View Batch Simulation Data"):
                    batch_cols = [
                        "Timestamp", "F_feed_rec",
                        "F_top_rec", "sim_F_top_kgh",
                        "F_bottom_rec", "sim_F_bot_kgh",
                        "Q_cond_filtered", "sim_Q_cond_kw",
                        "Q_reb_filtered",  "sim_Q_reb_kw",
                        "sim_mode",
                    ]
                    avail = [c for c in batch_cols if c in df_batch.columns]
                    st.dataframe(
                        df_batch[avail].style.format(
                            {c: "{:.1f}" for c in avail
                             if c not in ("Timestamp", "sim_mode")}
                        ),
                        use_container_width=True,
                    )
                    csv_b = df_batch[avail].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Batch Results CSV", data=csv_b,
                        file_name="batch_simulation_results.csv", mime="text/csv",
                    )
            else:
                st.info("👆 Click **▶ Run Batch Simulation** to compare SCADA data with simulation predictions.")
        else:
            st.info("👆 Click **Run Analysis** to see reconciliation results and KPIs.")


# ── IQR flag helper (no in-place modification) ───────────────────────────────

def _add_iqr_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with boolean _outlier columns added (no replacement)."""
    df_c = df.copy()
    raw_cols = ["F_feed_raw", "F_top_raw", "F_bottom_raw", "Q_cond_raw", "Q_reb_raw"]
    for col in raw_cols:
        q1 = df_c[col].quantile(0.25)
        q3 = df_c[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_c[f"{col}_outlier"] = (df_c[col] < lo) | (df_c[col] > hi)
    return df_c


# ── Batch simulation helpers ──────────────────────────────────────────────────

_MW_ETHANOL = 46.068
_MW_WATER = 18.015


def _scaling_row(F_feed_kgh: float, x_eth: float) -> dict:
    """Compute predicted top/bottom flows and duties via linear scaling."""
    scale = F_feed_kgh / (_FLOW_FEED_BASE + _EPSILON)
    return {
        "sim_F_top_kgh":   F_feed_kgh * _SPLIT_TOP,
        "sim_F_bot_kgh":   F_feed_kgh * _SPLIT_BOTTOM,
        "sim_Q_cond_kw":   _Q_COND_BASE * scale,
        "sim_Q_reb_kw":    _Q_REB_BASE * scale,
        "sim_mode":        "scaling",
    }


def _run_batch_simulation(
    df_rec: pd.DataFrame,
    T_feed: float,
    P_feed: float,
    x_eth: float,
    x_water: float,
    interf=None,
    sim=None,
    progress_bar=None,
) -> pd.DataFrame:
    """
    Run a simulation (DWSIM or scaling fallback) for every row in *df_rec*.

    Parameters
    ----------
    df_rec : pd.DataFrame
        Reconciled SCADA data; must contain ``F_feed_rec`` column (kg/h).
    T_feed, P_feed, x_eth, x_water : float
        Feed temperature (°C), pressure (kPa), ethanol and water mole fractions.
    interf, sim : optional
        DWSIM automation objects.  When both are supplied DWSIM is used row-by-row,
        with scaling as fall-back on any per-row exception.
    progress_bar : st.progress or None
        Streamlit progress bar updated after each row.

    Returns
    -------
    pd.DataFrame
        Same length as *df_rec* with additional columns:
        ``sim_F_top_kgh``, ``sim_F_bot_kgh``, ``sim_Q_cond_kw``,
        ``sim_Q_reb_kw``, ``sim_mode``.
    """
    use_dwsim = (interf is not None and sim is not None
                 and _dwsim.DWSIM_AVAILABLE)
    rows_out = []
    n = len(df_rec)

    for i, (_, row) in enumerate(df_rec.iterrows()):
        F_feed = float(row["F_feed_rec"])

        if use_dwsim:
            try:
                _dwsim.set_feed_conditions(
                    sim,
                    temperature_C=T_feed,
                    pressure_kPa=P_feed,
                    flow_kgh=F_feed,
                    x_ethanol=x_eth,
                    x_water=x_water,
                )
                _dwsim.run_simulation(interf, sim)
                raw = _dwsim.read_results(sim)
                top = raw.get("top", {})
                bot = raw.get("bottom", {})
                cond = raw.get("r_cond", {})
                reb = raw.get("q_reb", {})
                rows_out.append({
                    "sim_F_top_kgh":  top.get("mass_flow_kgh", 0.0),
                    "sim_F_bot_kgh":  bot.get("mass_flow_kgh", 0.0),
                    "sim_Q_cond_kw":  abs(cond.get("energy_kw", 0.0)),
                    "sim_Q_reb_kw":   abs(reb.get("energy_kw", 0.0)),
                    "sim_mode":       "dwsim",
                })
            except Exception:
                rows_out.append(_scaling_row(F_feed, x_eth))
        else:
            rows_out.append(_scaling_row(F_feed, x_eth))

        if progress_bar is not None:
            progress_bar.progress((i + 1) / n)

    df_sim = pd.DataFrame(rows_out)
    return pd.concat(
        [df_rec.reset_index(drop=True), df_sim.reset_index(drop=True)], axis=1
    )


def _plot_comparison(df: pd.DataFrame) -> plt.Figure:
    """2×2 chart comparing SCADA measurements vs simulation predictions."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Digital Twin — SCADA Measurements vs Simulation Predictions",
        fontsize=13, fontweight="bold",
    )

    pairs = [
        (axs[0, 0], "F_top_rec",      "sim_F_top_kgh",  "Distillate Flow",    "Flow (kg/h)"),
        (axs[0, 1], "F_bottom_rec",    "sim_F_bot_kgh",  "Bottoms Flow",       "Flow (kg/h)"),
        (axs[1, 0], "Q_cond_filtered", "sim_Q_cond_kw",  "Condenser Duty",     "Duty (kW)"),
        (axs[1, 1], "Q_reb_filtered",  "sim_Q_reb_kw",   "Reboiler Duty",      "Duty (kW)"),
    ]
    ts = df["Timestamp"] if "Timestamp" in df.columns else df.index

    for ax, meas_col, sim_col, title, ylabel in pairs:
        ax.plot(ts, df[meas_col], color="steelblue", alpha=0.8, label="SCADA (reconciled/filtered)")
        if sim_col in df.columns:
            ax.plot(ts, df[sim_col], color="tomato", ls="--", label="Simulation")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.5)

    for ax in axs[1]:
        ax.set_xlabel("Time")
    fig.tight_layout()
    return fig
