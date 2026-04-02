"""
Digital Twin — Sub-page 2: SCADA Data Generation & Analysis
============================================================
Lets the user configure noise levels, outlier injection, moving-average
window, and KPI targets; then runs the full data-quality pipeline:
  1. Generate synthetic SCADA data (Gaussian noise + injected outliers)
  2. Detect outliers with the IQR method
  3. Apply centred moving-average filter
  4. Reconcile mass balance with Weighted Least Squares (SLSQP)
  5. Calculate and visualise KPIs
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

_EPSILON = 1e-9   # numerical guard against division by zero

# ── Design-point constants (from Simulation/config.py / ethanol.dwxmz) ──────
_FLOW_FEED_BASE = 10_000.0   # kg/h
_SPLIT_TOP = 0.35
_SPLIT_BOTTOM = 0.65
_Q_COND_BASE = 1_207.87      # kW
_Q_REB_BASE = 1_524.29       # kW
_DESIGN_ENERGY_RATIO = _Q_COND_BASE / _Q_REB_BASE


# ── 1. Data generation ────────────────────────────────────────────────────────

def _generate_raw_data(
    n_points: int,
    sigma_feed: float,
    sigma_top: float,
    sigma_bottom: float,
    sigma_q_cond: float,
    sigma_q_reb: float,
    outlier_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Generate synthetic SCADA data with Gaussian noise and random outliers."""
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

    # Inject outliers at random indices
    n_outliers = max(1, int(n_points * outlier_fraction))
    outlier_indices = rng.choice(n_points, size=n_outliers, replace=False)
    for idx in outlier_indices:
        col = rng.choice(["F_feed_raw", "F_top_raw", "F_bottom_raw",
                           "Q_cond_raw", "Q_reb_raw"])
        df.loc[idx, col] *= rng.choice([1.8, 0.3])   # spike or drop

    return df


# ── 2. IQR outlier detection + MA filter ─────────────────────────────────────

def _clean_and_filter(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Detect outliers (IQR), interpolate, then apply moving-average."""
    df_c = df.copy()
    raw_cols = ["F_feed_raw", "F_top_raw", "F_bottom_raw", "Q_cond_raw", "Q_reb_raw"]

    for col in raw_cols:
        q1 = df_c[col].quantile(0.25)
        q3 = df_c[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (df_c[col] < lo) | (df_c[col] > hi)
        df_c[f"{col}_outlier"] = outlier_mask
        df_c.loc[outlier_mask, col] = np.nan
        df_c[col] = df_c[col].interpolate(method="linear")
        filtered_name = col.replace("_raw", "_filtered")
        df_c[filtered_name] = df_c[col].rolling(window=window, center=True).mean()

    return df_c.dropna().reset_index(drop=True)


# ── 3. WLS data reconciliation ────────────────────────────────────────────────

def _reconcile(df: pd.DataFrame,
               sigma_feed: float, sigma_top: float, sigma_bottom: float
               ) -> pd.DataFrame:
    """Row-wise SLSQP reconciliation enforcing F_feed = F_top + F_bottom."""
    sigma_mass = np.array([sigma_feed, sigma_top, sigma_bottom])
    results = []

    for _, row in df.iterrows():
        y = np.array([row["F_feed_filtered"], row["F_top_filtered"], row["F_bottom_filtered"]])

        def obj(x):
            return np.sum(((y - x) / sigma_mass) ** 2)

        def con(x):
            return x[0] - x[1] - x[2]

        sol = minimize(obj, x0=y.copy(),
                       constraints={"type": "eq", "fun": con},
                       method="SLSQP")
        F_f, F_t, F_b = sol.x

        # KPI 1 — mass balance closure error BEFORE reconciliation
        err_before = abs(y[0] - y[1] - y[2]) / (y[0] + _EPSILON) * 100.0

        # KPI 2 — separation adherence
        actual_split = F_t / (F_f + _EPSILON)
        kpi_sep = 100.0 - abs(actual_split - _SPLIT_TOP) / _SPLIT_TOP * 100.0

        # KPI 3 — energy efficiency
        q_cond = row["Q_cond_filtered"]
        q_reb = row["Q_reb_filtered"]
        actual_ratio = q_cond / (q_reb + _EPSILON)
        kpi_energy = 100.0 - abs(actual_ratio - _DESIGN_ENERGY_RATIO) / _DESIGN_ENERGY_RATIO * 100.0

        results.append({
            "Timestamp": row["Timestamp"],
            "F_feed_rec": F_f,
            "F_top_rec": F_t,
            "F_bottom_rec": F_b,
            "Q_cond_filtered": q_cond,
            "Q_reb_filtered": q_reb,
            "Error_Mass_Before_%": err_before,
            "KPI_Separation_%": kpi_sep,
            "KPI_Energy_%": kpi_energy,
        })

    df_rec = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True),
                      df_rec.drop("Timestamp", axis=1)], axis=1)


# ── 4. Visualisation helpers ──────────────────────────────────────────────────

def _plot_signal_treatment(df: pd.DataFrame, col_base: str, label: str,
                            color_raw: str = "#4C72B0",
                            color_filt: str = "#55A868",
                            color_rec: str = "#C44E52") -> plt.Figure:
    """Plot raw / filtered / reconciled for one signal with outliers marked."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    t = np.arange(len(df))

    ax.plot(t, df[f"{col_base}_raw"], color=color_raw, lw=1, alpha=0.6, label="Raw")
    # Outlier markers
    mask = df.get(f"{col_base}_raw_outlier", pd.Series(False, index=df.index))
    ax.scatter(t[mask.values], df.loc[mask, f"{col_base}_raw"],
               color="red", zorder=5, s=40, label="Outlier")
    ax.plot(t, df[f"{col_base}_filtered"], color=color_filt, lw=1.5, label="MA-filtered")
    rec_col = f"{col_base}_rec"
    if rec_col in df.columns:
        ax.plot(t, df[rec_col], color=color_rec, lw=1.5, ls="--", label="Reconciled")
    ax.set_title(f"Signal Treatment — {label}")
    ax.set_xlabel("Time step")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_kpi_dashboard(df: pd.DataFrame,
                        target_sep: float,
                        target_energy: float,
                        threshold_mass: float) -> plt.Figure:
    """Four-panel KPI dashboard."""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    t = np.arange(len(df))

    # Panel 1 — mass balance error
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(t, 0, df["Error_Mass_Before_%"], alpha=0.4, color="#C44E52")
    ax1.axhline(threshold_mass, color="red", ls="--", lw=1.2, label=f"Threshold {threshold_mass}%")
    ax1.set_title("Mass Balance Closure Error (%)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Error (%)")
    ax1.legend(fontsize=8)

    # Panel 2 — separation KPI
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["KPI_Separation_%"], color="#55A868", lw=1.5)
    ax2.axhline(target_sep, color="green", ls="--", lw=1.2, label=f"Target {target_sep}%")
    ax2.fill_between(t, target_sep, df["KPI_Separation_%"],
                     where=df["KPI_Separation_%"] < target_sep,
                     alpha=0.3, color="orange", label="Below target")
    ax2.set_title("KPI — Separation Adherence (%)")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("KPI (%)")
    ax2.set_ylim(max(0, df["KPI_Separation_%"].min() - 5), 105)
    ax2.legend(fontsize=8)

    # Panel 3 — energy KPI
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, df["KPI_Energy_%"], color="#8172B2", lw=1.5)
    ax3.axhline(target_energy, color="purple", ls="--", lw=1.2, label=f"Target {target_energy}%")
    ax3.fill_between(t, target_energy, df["KPI_Energy_%"],
                     where=df["KPI_Energy_%"] < target_energy,
                     alpha=0.3, color="pink", label="Below target")
    ax3.set_title("KPI — Energy Efficiency (%)")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("KPI (%)")
    ax3.set_ylim(max(0, df["KPI_Energy_%"].min() - 5), 105)
    ax3.legend(fontsize=8)

    # Panel 4 — reconciled streams
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, df["F_feed_rec"], label="Feed (rec.)", lw=1.5)
    ax4.plot(t, df["F_top_rec"], label="Distillate (rec.)", lw=1.5, ls="--")
    ax4.plot(t, df["F_bottom_rec"], label="Bottoms (rec.)", lw=1.5, ls=":")
    ax4.set_title("Reconciled Mass Flows (kg/h)")
    ax4.set_xlabel("Time step")
    ax4.set_ylabel("kg/h")
    ax4.legend(fontsize=8)

    fig.suptitle("Digital Twin — KPI Dashboard", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── Streamlit page ────────────────────────────────────────────────────────────

def analisis_datos_page():
    st.header("📡 SCADA Data Generation & Analysis")
    st.markdown("""
    Configure the synthetic data generator, noise levels, and quality targets below.
    The pipeline runs automatically: raw data → outlier detection (IQR) →
    moving-average filter → WLS reconciliation → KPI calculation.
    """)
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Analysis Parameters")

        with st.expander("1. Data Generation", expanded=True):
            n_points = st.slider("Number of SCADA points", 50, 300, 100, 10, key="da_n")
            seed = st.number_input("Random seed", 0, 9999, 42, key="da_seed")
            outlier_frac = st.slider("Outlier fraction", 0.0, 0.15, 0.05, 0.01,
                                     key="da_outfrac",
                                     help="Fraction of readings replaced with anomalous spikes/drops")

        with st.expander("2. Sensor Noise (σ)", expanded=True):
            sigma_feed = st.number_input("σ Feed (kg/h)", 10.0, 1000.0, 350.0, 10.0, key="da_sf")
            sigma_top = st.number_input("σ Distillate (kg/h)", 10.0, 500.0, 150.0, 10.0, key="da_st")
            sigma_bot = st.number_input("σ Bottoms (kg/h)", 10.0, 500.0, 200.0, 10.0, key="da_sb")
            sigma_qc = st.number_input("σ Q_cond (kW)", 1.0, 200.0, 40.0, 5.0, key="da_sqc")
            sigma_qr = st.number_input("σ Q_reb (kW)", 1.0, 200.0, 50.0, 5.0, key="da_sqr")

        with st.expander("3. Filter Settings", expanded=True):
            window = st.slider("Moving-average window W", 3, 21, 5, 2, key="da_win",
                               help="Must be odd for centred MA; even values are rounded up internally")
            if window % 2 == 0:
                window += 1

        with st.expander("4. KPI Targets", expanded=True):
            target_sep = st.slider("Target Separation Adherence (%)", 70.0, 99.0, 90.0, 1.0, key="da_tsep")
            target_energy = st.slider("Target Energy Efficiency (%)", 70.0, 99.0, 85.0, 1.0, key="da_teng")
            threshold_mass = st.slider("Max Mass Balance Error (%)", 0.5, 10.0, 2.0, 0.5, key="da_tmass")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Running data pipeline…"):
        df_raw = _generate_raw_data(
            n_points, sigma_feed, sigma_top, sigma_bot,
            sigma_qc, sigma_qr, outlier_frac, int(seed)
        )
        df_filt = _clean_and_filter(df_raw, window)
        df_final = _reconcile(df_filt, sigma_feed, sigma_top, sigma_bot)

    st.success(f"Pipeline complete — {len(df_final)} usable time steps after filtering.")

    # ── KPI summary cards ─────────────────────────────────────────────────────
    st.subheader("📊 KPI Summary")
    avg_mass_err = df_final["Error_Mass_Before_%"].mean()
    avg_sep = df_final["KPI_Separation_%"].mean()
    avg_energy = df_final["KPI_Energy_%"].mean()

    pct_sep_ok = (df_final["KPI_Separation_%"] >= target_sep).mean() * 100
    pct_energy_ok = (df_final["KPI_Energy_%"] >= target_energy).mean() * 100
    pct_mass_ok = (df_final["Error_Mass_Before_%"] <= threshold_mass).mean() * 100

    def _traffic_light(value: float, green_thresh: float, yellow_thresh: float) -> str:
        if value >= green_thresh:
            return "🟢"
        if value >= yellow_thresh:
            return "🟡"
        return "🔴"

    c1, c2, c3 = st.columns(3)
    sep_light = _traffic_light(pct_sep_ok, 80, 60)
    energy_light = _traffic_light(pct_energy_ok, 80, 60)
    mass_light = _traffic_light(pct_mass_ok, 80, 60)

    c1.metric(
        f"{sep_light} Separation Adherence",
        f"{avg_sep:.1f} %",
        delta=f"{avg_sep - target_sep:+.1f}% vs target",
        help=f"{pct_sep_ok:.0f}% of steps meet the ≥{target_sep}% target",
    )
    c2.metric(
        f"{energy_light} Energy Efficiency",
        f"{avg_energy:.1f} %",
        delta=f"{avg_energy - target_energy:+.1f}% vs target",
        help=f"{pct_energy_ok:.0f}% of steps meet the ≥{target_energy}% target",
    )
    c3.metric(
        f"{mass_light} Mass Balance Error",
        f"{avg_mass_err:.2f} %",
        delta=f"{avg_mass_err - threshold_mass:+.2f}% vs threshold",
        delta_color="inverse",
        help=f"{pct_mass_ok:.0f}% of steps are within ±{threshold_mass}%",
    )

    st.markdown("---")

    # ── Signal treatment plots ─────────────────────────────────────────────────
    st.subheader("🔍 Signal Treatment")
    tab1, tab2, tab3 = st.tabs(["Feed Flow", "Distillate Flow", "Bottoms Flow"])

    with tab1:
        fig_f = _plot_signal_treatment(df_final, "F_feed", "Feed Flow (kg/h)")
        st.pyplot(fig_f)
        plt.close(fig_f)

    with tab2:
        fig_t = _plot_signal_treatment(df_final, "F_top", "Distillate Flow (kg/h)")
        st.pyplot(fig_t)
        plt.close(fig_t)

    with tab3:
        fig_b = _plot_signal_treatment(df_final, "F_bottom", "Bottoms Flow (kg/h)")
        st.pyplot(fig_b)
        plt.close(fig_b)

    st.markdown("---")

    # ── KPI dashboard ─────────────────────────────────────────────────────────
    st.subheader("📈 KPI Dashboard")
    fig_kpi = _plot_kpi_dashboard(df_final, target_sep, target_energy, threshold_mass)
    st.pyplot(fig_kpi)
    plt.close(fig_kpi)

    st.markdown("---")

    # ── Data table (expandable) ────────────────────────────────────────────────
    with st.expander("📋 View Reconciled Data Table"):
        display_cols = [
            "Timestamp", "F_feed_rec", "F_top_rec", "F_bottom_rec",
            "Q_cond_filtered", "Q_reb_filtered",
            "Error_Mass_Before_%", "KPI_Separation_%", "KPI_Energy_%",
        ]
        st.dataframe(
            df_final[display_cols].style.format({
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
        csv = df_final[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="digital_twin_results.csv", mime="text/csv")

    # ── Outlier statistics ─────────────────────────────────────────────────────
    with st.expander("📌 Outlier Detection Summary"):
        outlier_cols = [c for c in df_filt.columns if c.endswith("_outlier")]
        if outlier_cols:
            summary = {c.replace("_raw_outlier", "").replace("_outlier", ""):
                       int(df_filt[c].sum()) for c in outlier_cols}
            st.table(pd.DataFrame.from_dict(
                summary, orient="index", columns=["Outliers detected"]
            ))
        else:
            st.info("No outlier columns found.")
