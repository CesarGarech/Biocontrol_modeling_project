"""
Digital Twin — Sub-page 2: SCADA Data Generation & Analysis
============================================================
Pipeline interactivo de tres pasos:
  Paso 1 — Generar datos SCADA con ruido (desde DWSIM o sintéticos) y visualizarlos.
  Paso 2 — Inyectar outliers (índices manuales o fracción aleatoria) y mostrar
            anomalías detectadas por IQR superpuestas a la señal raw.
  Paso 3 — Análisis completo: filtro MA → reconciliación WLS → KPIs.
Las constantes del punto de diseño se importan desde Simulation/config.py.
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Importar configuración compartida desde Simulation/config.py ─────────────
_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Simulation"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── Importar generador de datos DWSIM (con fallback seguro) ──────────────────
try:
    from dwsim_data_generator import (  # noqa: E402
        generate_dwsim_data,
        validate_dwsim_installation,
    )
    _DWSIM_GENERATOR_OK = True
except Exception:
    _DWSIM_GENERATOR_OK = False

    def validate_dwsim_installation():  # type: ignore[misc]
        """Stub cuando los módulos DWSIM no están disponibles."""
        return False, "Módulos DWSIM no pudieron importarse."

    def generate_dwsim_data(n_points, perturbations=None):  # type: ignore[misc]
        """Stub — nunca se llama en producción cuando _DWSIM_GENERATOR_OK es False."""
        raise RuntimeError("dwsim_data_generator no disponible.")

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
    """Página del pipeline de análisis de datos SCADA de la columna de destilación."""
    st.header("📡 SCADA Data Generation & Analysis")
    st.markdown("""
    Siga los **tres pasos** a continuación.  
    Cada paso usa los resultados del anterior — los cambios en parámetros del
    panel lateral solo surten efecto al presionar el botón correspondiente.
    """)
    st.markdown("---")

    # ── Verificar disponibilidad de DWSIM ────────────────────────────────────
    dwsim_ok, dwsim_msg = validate_dwsim_installation()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Parámetros de Análisis")

        with st.expander("1. Generación de Datos", expanded=True):
            use_dwsim_gen = st.checkbox(
                "Usar DWSIM para generación de datos",
                value=False,
                key="da_use_dwsim",
                help="Requiere DWSIM instalado. Si no está disponible se usará generación sintética.",
                disabled=not (dwsim_ok and _DWSIM_GENERATOR_OK),
            )
            n_points = st.slider("Número de puntos SCADA", 50, 300, _cfg.N_POINTS, 10, key="da_n")
            seed = int(st.number_input("Semilla aleatoria", 0, 9999, _cfg.SEED, key="da_seed"))

        with st.expander("2. Ruido de Sensores (σ)", expanded=True):
            sigma_feed = st.number_input("σ Feed (kg/h)", 10.0, 1000.0, float(_cfg.SIGMA_FEED), 10.0, key="da_sf")
            sigma_top = st.number_input("σ Destilado (kg/h)", 10.0, 500.0, float(_cfg.SIGMA_TOP), 10.0, key="da_st")
            sigma_bot = st.number_input("σ Fondos (kg/h)", 10.0, 500.0, float(_cfg.SIGMA_BOTTOM), 10.0, key="da_sb")
            sigma_qc = st.number_input("σ Q_cond (kW)", 1.0, 200.0, float(_cfg.SIGMA_Q_COND), 5.0, key="da_sqc")
            sigma_qr = st.number_input("σ Q_reb (kW)", 1.0, 200.0, float(_cfg.SIGMA_Q_REB), 5.0, key="da_sqr")

        with st.expander("3. Inyección de Outliers", expanded=True):
            use_random = st.checkbox("Usar outliers aleatorios", value=False, key="da_random_out")
            if use_random:
                outlier_frac = st.slider(
                    "Fracción de outliers", 0.0, 0.20, 0.05, 0.01, key="da_outfrac"
                )
                outlier_indices_str = None
            else:
                st.markdown("Índices de tiempo (base 0, separados por coma):")
                outlier_indices_str = st.text_input(
                    "Índices de outliers", value="12, 22, 30, 48, 55, 67, 75, 85, 92",
                    key="da_out_idx"
                )
                outlier_frac = None

        with st.expander("4. Filtrado y Reconciliación", expanded=True):
            window = st.slider(
                "Ventana de media móvil W", 3, 21, _cfg.WINDOW_SIZE, 2, key="da_win",
                help="MA centrada; valores pares se redondean al siguiente impar"
            )
            if window % 2 == 0:
                window += 1

        with st.expander("5. Objetivos de KPI", expanded=True):
            target_sep = st.slider("Target Adherencia Separación (%)", 70.0, 99.0, 90.0, 1.0, key="da_tsep")
            target_energy = st.slider("Target Eficiencia Energética (%)", 70.0, 99.0, 85.0, 1.0, key="da_teng")
            threshold_mass = st.slider(
                "Error Máx. Balance de Masa (%)", 0.5, 10.0,
                float(_cfg.MAX_MASS_BALANCE_ERROR), 0.5, key="da_tmass"
            )

    # ── Paso 1: Generar datos raw ─────────────────────────────────────────────
    st.subheader("🔵 Paso 1 — Generar Datos SCADA")

    # Mostrar estado de DWSIM
    if use_dwsim_gen and dwsim_ok and _DWSIM_GENERATOR_OK:
        st.info(f"🔬 Modo DWSIM activo. {dwsim_msg}")
        btn_label = "📊 Generate DWSIM Data"
    elif use_dwsim_gen and not (dwsim_ok and _DWSIM_GENERATOR_OK):
        st.warning(
            f"⚠️ DWSIM no disponible. Se usará generación sintética.  \nDetalle: {dwsim_msg}"
        )
        btn_label = "▶ Generar Datos Sintéticos"
    else:
        btn_label = "▶ Generate Raw Data"

    if st.button(btn_label, key="btn_gen"):
        # Validar que ventana sea menor que n_points
        if n_points < window:
            st.error(
                f"❌ El número de puntos ({n_points}) debe ser mayor que "
                f"la ventana de filtro ({window})."
            )
        else:
            use_live_dwsim = use_dwsim_gen and dwsim_ok and _DWSIM_GENERATOR_OK
            if use_live_dwsim:
                with st.spinner("Ejecutando simulaciones DWSIM para generar datos…"):
                    try:
                        # Perturbaciones sinusoidales en el flujo másico de alimentación
                        perturbations = (
                            np.sin(np.linspace(0, np.pi, n_points))
                            * _cfg.PERTURBATION_AMPLITUDE
                        )
                        df_raw = generate_dwsim_data(n_points, perturbations=perturbations)
                        st.success("✅ Datos generados desde DWSIM exitosamente.")
                    except Exception as exc:
                        st.warning(
                            f"⚠️ DWSIM falló ({exc}). Usando generación sintética."
                        )
                        df_raw = _generate_raw_data(
                            n_points, sigma_feed, sigma_top, sigma_bot, sigma_qc, sigma_qr, seed,
                        )
            else:
                df_raw = _generate_raw_data(
                    n_points, sigma_feed, sigma_top, sigma_bot,
                    sigma_qc, sigma_qr, seed,
                )
            st.session_state["da_df_raw"] = df_raw
            # Limpiar resultados de pasos siguientes al regenerar datos
            st.session_state.pop("da_df_with_outliers", None)
            st.session_state.pop("da_df_final", None)

    if "da_df_raw" in st.session_state:
        df_raw = st.session_state["da_df_raw"]
        st.success(f"Datos generados — {len(df_raw)} puntos temporales.")
        fig1 = _plot_raw_signals(df_raw, " — Señales con Ruido Gaussiano")
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("👆 Presione el botón para generar datos.")

    st.markdown("---")

    # ── Paso 2: Inyectar outliers & visualizar ────────────────────────────────
    st.subheader("🟡 Paso 2 — Inyección de Outliers y Detección IQR")

    if "da_df_raw" not in st.session_state:
        st.info("Complete el Paso 1 primero.")
    else:
        if st.button("⚠️ Inject Outliers", key="btn_outlier"):
            df_raw = st.session_state["da_df_raw"]
            n_pts_actual = len(df_raw)

            # Resolver índices de outliers
            if use_random:
                rng_tmp = np.random.default_rng(seed + 1)
                n_out = max(1, int(n_pts_actual * outlier_frac))
                indices = list(rng_tmp.choice(n_pts_actual, size=n_out, replace=False))
            else:
                try:
                    indices = [int(x.strip()) for x in outlier_indices_str.split(",") if x.strip()]
                except ValueError:
                    st.error("Lista de índices inválida — use enteros separados por coma: 10, 25, 40")
                    indices = []

            df_with_out = _inject_outliers(df_raw, indices, seed)

            # Detección IQR sobre los datos con outliers inyectados
            df_iqr = _add_iqr_flags(df_with_out)
            st.session_state["da_df_with_outliers"] = df_with_out
            st.session_state["da_df_iqr"] = df_iqr
            st.session_state["da_outlier_indices"] = indices
            st.session_state.pop("da_df_final", None)

        if "da_df_iqr" in st.session_state:
            df_iqr = st.session_state["da_df_iqr"]
            indices = st.session_state.get("da_outlier_indices", [])
            st.success(
                f"Outliers inyectados en índices: {indices}  |  "
                f"IQR detectó: {sum(df_iqr[[c for c in df_iqr.columns if c.endswith('_outlier')]].any(axis=1))} filas afectadas"
            )
            fig2 = _plot_raw_signals(df_iqr, " — Detección de Anomalías IQR")
            st.pyplot(fig2)
            plt.close(fig2)

            # Tabla de conteo de outliers
            with st.expander("📌 Conteo de Outliers por Sensor"):
                import re as _re
                out_cols = [c for c in df_iqr.columns if c.endswith("_outlier")]
                summary = {
                    _re.sub(r"_raw_outlier$|_outlier$", "", c): int(df_iqr[c].sum())
                    for c in out_cols
                }
                st.table(pd.DataFrame.from_dict(summary, orient="index", columns=["Outliers (IQR)"]))
        else:
            st.info("👆 Presione **⚠️ Inject Outliers** para continuar.")

    st.markdown("---")

    # ── Paso 3: Pipeline de análisis completo ─────────────────────────────────
    st.subheader("🟢 Paso 3 — Filtro MA → Reconciliación WLS → KPIs")

    if "da_df_with_outliers" not in st.session_state:
        st.info("Complete los Pasos 1 y 2 primero.")
    else:
        if st.button("⚖️ Run Analysis", key="btn_analyze"):
            with st.spinner("Aplicando filtro MA y reconciliación WLS…"):
                df_filt = _clean_and_filter(st.session_state["da_df_with_outliers"], window)
                df_final = _reconcile(df_filt, sigma_feed, sigma_top, sigma_bot)
            st.session_state["da_df_final"] = df_final
            st.session_state["da_df_filt"] = df_filt
            # Guardar umbrales usados para que el display no cambie al mover sliders
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

            st.success(f"Análisis completo — {len(df_final)} pasos temporales utilizables tras el filtrado.")

            # Tarjetas de métricas KPI
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
                f"{_traffic_light(pct_sep_ok)} Adherencia Separación",
                f"{avg_sep:.1f} %",
                delta=f"{avg_sep - target_sep_disp:+.1f}% vs objetivo",
                help=f"{pct_sep_ok:.0f}% de pasos ≥ {target_sep_disp}%",
            )
            c2.metric(
                f"{_traffic_light(pct_energy_ok)} Eficiencia Energética",
                f"{avg_energy:.1f} %",
                delta=f"{avg_energy - target_energy_disp:+.1f}% vs objetivo",
                help=f"{pct_energy_ok:.0f}% de pasos ≥ {target_energy_disp}%",
            )
            c3.metric(
                f"{_traffic_light(pct_mass_ok)} Error Balance de Masa",
                f"{avg_mass_err:.2f} %",
                delta=f"{avg_mass_err - threshold_mass_disp:+.2f}% vs umbral",
                delta_color="inverse",
                help=f"{pct_mass_ok:.0f}% de pasos dentro de ±{threshold_mass_disp}%",
            )

            # ── Leyenda de KPIs ───────────────────────────────────────────────
            with st.expander("ℹ️ Cómo interpretar estos KPIs"):
                st.markdown(f"""
**Indicador de semáforo** (encabezado de cada tarjeta):
- 🟢 **Verde** — KPI cumple el objetivo en **≥ {_TL_GREEN:.0f} %** de los pasos. El proceso opera bien.
- 🟡 **Amarillo** — KPI cumple el objetivo en **{_TL_YELLOW:.0f}–{_TL_GREEN:.0f} %** de los pasos. Monitoreo cercano recomendado.
- 🔴 **Rojo** — KPI cumple el objetivo en **< {_TL_YELLOW:.0f} %** de los pasos. Se requiere atención.

**Flecha delta** (número pequeño debajo del valor principal):
- ↑ **Delta verde** — el KPI promedio es *superior* al objetivo fijado → favorable.
- ↓ **Delta rojo** — el KPI promedio es *inferior* al objetivo fijado → requiere mejora.
- Para **Error de Balance de Masa** la flecha está invertida: ↑ rojo significa que el error supera el umbral.

**Significado de cada KPI**:
- *Adherencia Separación* — qué tan cerca está el split real del destilado al objetivo de diseño ({_cfg.SPLIT_TOP*100:.0f} % del feed). 100 % = adherencia perfecta.
- *Eficiencia Energética* — qué tan cercana está la razón Q_cond / Q_reb al valor de diseño. 100 % = operación en condiciones de diseño.
- *Error Balance de Masa* — desbalance porcentual (F_feed − F_top − F_bottom) antes de la reconciliación. Menor es mejor; objetivo ≤ {threshold_mass_disp} %.

> Los resultados reflejan la **última ejecución**. Cambiar parámetros del panel lateral requiere presionar **⚖️ Run Analysis** nuevamente.
""")

            st.markdown("---")

            # Dashboard de 4 paneles
            fig3 = _plot_dashboard(df_final, threshold_mass_disp)
            st.pyplot(fig3)
            plt.close(fig3)

            # Tabla de datos reconciliados
            with st.expander("📋 Ver Tabla de Datos Reconciliados"):
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
        else:
            st.info("👆 Presione **⚖️ Run Analysis** para ver resultados de reconciliación y KPIs.")


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
