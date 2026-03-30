"""
Digital Twin - Distillation Column

This module implements a complete digital twin pipeline for a distillation column:

1. DWSIM integration (mock mode) - generates synthetic distillation time-series data
2. Excel sensor data ingestion with schema validation
3. Data treatment: outlier detection/removal by IQR (with before/after plots)
4. Signal filtering: moving average or low-pass Butterworth filter (user selectable)
5. Data reconciliation: Weighted Least Squares (WLS) with balance constraints
6. KPIs and adherence indicators

References:
-----------
- Narasimhan, S., & Jordache, C. (1999). Data Reconciliation & Gross Error Detection.
  Gulf Professional Publishing.
- Mah, R. S. H. (1990). Chemical Process Structures and Information Flows.
  Butterworth-Heinemann.
- Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle, F. J. (2016).
  Process Dynamics and Control (4th ed.). Wiley.
- Smith, C. A., & Corripio, A. B. (2005). Principles and Practice of Automatic
  Process Control. Wiley.
"""

import io
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "..", "..", "Output", "digital_twin")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column names expected in the sensor dataset
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Time", "F", "D", "B", "zF", "xD", "xB", "R", "QR", "QC", "T_top", "T_bot"]

COLUMN_UNITS = {
    "Time": "min",
    "F": "mol/min",
    "D": "mol/min",
    "B": "mol/min",
    "zF": "mol frac",
    "xD": "mol frac",
    "xB": "mol frac",
    "R": "-",
    "QR": "kW",
    "QC": "kW",
    "T_top": "°C",
    "T_bot": "°C",
}

# Variables used in mass/component balance reconciliation
BALANCE_VARS = ["F", "D", "B", "zF", "xD", "xB"]

# ===========================================================================
# 1. MOCK DWSIM SIMULATION
# ===========================================================================


def simulate_distillation_column(n_points: int = 120, dt: float = 1.0,
                                  noise_level: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """
    Mock simulation of a binary distillation column (ethanol-water).

    Generates a realistic time-series representing the "true" plant state plus
    sensor noise, mimicking what DWSIM would produce via its automation API.

    Column operating conditions (nominal):
    - Feed: F = 100 mol/min, z_F = 0.40 (mol frac ethanol)
    - Distillate: D = 38 mol/min, x_D = 0.92 (mol frac ethanol)
    - Bottoms: B = 62 mol/min, x_B = 0.05 (mol frac ethanol)
    - Reflux ratio: R = 2.5
    - Reboiler duty: Q_R = 450 kW
    - Condenser duty: Q_C = 420 kW
    - Top temperature: T_top = 78 °C
    - Bottom temperature: T_bot = 100 °C

    Parameters
    ----------
    n_points : int
        Number of time points.
    dt : float
        Sampling interval in minutes.
    noise_level : float
        Relative noise amplitude (fraction of nominal value).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: Time, F, D, B, zF, xD, xB, R, QR, QC, T_top, T_bot
        Also returns twin (noise-free) values with prefix 'twin_'.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) * dt

    # --- Nominal (steady-state) values ---
    F0 = 100.0
    D0 = 38.0
    B0 = 62.0
    zF0 = 0.40
    xD0 = 0.92
    xB0 = 0.05
    R0 = 2.5
    QR0 = 450.0
    QC0 = 420.0
    T_top0 = 78.0
    T_bot0 = 100.0

    # --- Slow drift + sinusoidal disturbance (true plant) ---
    drift = 0.005 * np.sin(2 * np.pi * t / (n_points * dt * 0.5))
    step = np.where(t >= n_points * dt * 0.6, 0.03, 0.0)  # feed step at 60% time

    F_true = F0 * (1 + drift + step)
    D_true = D0 * (1 + drift * 0.8 + step * 0.6)
    B_true = F_true - D_true  # mass balance satisfied
    zF_true = np.clip(zF0 + 0.02 * drift, 0.01, 0.99)
    xD_true = np.clip(xD0 - 0.01 * drift - step * 0.05, 0.01, 0.99)
    xB_true = np.clip(xB0 + 0.005 * drift + step * 0.01, 0.001, 0.5)
    R_true = R0 * (1 + 0.5 * drift)
    QR_true = QR0 * (1 + drift + step * 0.3)
    QC_true = QC0 * (1 + drift + step * 0.25)
    T_top_true = T_top0 + 2 * drift * 10
    T_bot_true = T_bot0 + 3 * drift * 10

    def add_noise(arr, ref, level, rng, outlier_frac=0.04):
        """
        Add Gaussian noise and inject random spike outliers.

        Parameters
        ----------
        arr : np.ndarray
            True (noise-free) signal array.
        ref : float
            Nominal reference value used to scale noise and outlier amplitudes.
        level : float
            Relative noise standard deviation (fraction of ref).
        rng : np.random.Generator
            NumPy random generator instance.
        outlier_frac : float
            Fraction of points to replace with spike outliers (default 0.04 = 4%).

        Returns
        -------
        np.ndarray
            Noisy signal with injected outliers.
        """
        noise = rng.normal(0, level * ref, size=arr.shape)
        noisy = arr + noise
        # Inject outliers
        n_outliers = max(1, int(outlier_frac * len(arr)))
        idx = rng.choice(len(arr), n_outliers, replace=False)
        noisy[idx] += rng.choice([-1, 1], n_outliers) * ref * rng.uniform(0.15, 0.40, n_outliers)
        return noisy

    # --- Sensor (noisy) measurements ---
    F_meas = add_noise(F_true, F0, noise_level, rng)
    D_meas = add_noise(D_true, D0, noise_level, rng)
    B_meas = add_noise(B_true, B0, noise_level, rng)
    zF_meas = np.clip(add_noise(zF_true, zF0, noise_level, rng), 0.01, 0.99)
    xD_meas = np.clip(add_noise(xD_true, xD0, noise_level, rng), 0.01, 0.99)
    xB_meas = np.clip(add_noise(xB_true, xB0, noise_level * 1.5, rng), 0.001, 0.5)
    R_meas = add_noise(R_true, R0, noise_level, rng)
    QR_meas = add_noise(QR_true, QR0, noise_level, rng)
    QC_meas = add_noise(QC_true, QC0, noise_level, rng)
    T_top_meas = add_noise(T_top_true, T_top0, noise_level, rng)
    T_bot_meas = add_noise(T_bot_true, T_bot0, noise_level, rng)

    df = pd.DataFrame({
        "Time": t,
        # Sensor measurements
        "F": F_meas, "D": D_meas, "B": B_meas,
        "zF": zF_meas, "xD": xD_meas, "xB": xB_meas,
        "R": R_meas, "QR": QR_meas, "QC": QC_meas,
        "T_top": T_top_meas, "T_bot": T_bot_meas,
        # True (twin) values for KPI computation
        "twin_F": F_true, "twin_D": D_true, "twin_B": B_true,
        "twin_zF": zF_true, "twin_xD": xD_true, "twin_xB": xB_true,
        "twin_R": R_true, "twin_QR": QR_true, "twin_QC": QC_true,
        "twin_T_top": T_top_true, "twin_T_bot": T_bot_true,
    })
    return df


# ===========================================================================
# 2. EXCEL DATA INGESTION
# ===========================================================================


def load_excel_sensor_data(file_obj) -> tuple[pd.DataFrame, list[str]]:
    """
    Load sensor data from an Excel file uploaded by the user.

    Expected schema: columns listed in REQUIRED_COLUMNS.
    Twin columns (twin_*) are optional; if absent, only sensor-vs-sensor KPIs
    will be available.

    Parameters
    ----------
    file_obj : file-like
        Excel file object from st.file_uploader.

    Returns
    -------
    df : pd.DataFrame
        Validated sensor data.
    warnings_list : list[str]
        Non-fatal validation messages.
    """
    warnings_list = []
    df = pd.read_excel(file_obj, engine="openpyxl")

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}. Got: {list(df.columns)}"
        )

    # Type coercion
    for col in REQUIRED_COLUMNS:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            warnings_list.append(f"Column '{col}' could not be converted to numeric.")

    # Check NaN percentage
    for col in REQUIRED_COLUMNS[1:]:  # skip Time
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 20:
            warnings_list.append(f"Column '{col}' has {nan_pct:.1f}% NaN values.")

    return df, warnings_list


def create_example_excel() -> bytes:
    """Generate an example Excel file with synthetic sensor data."""
    df = simulate_distillation_column(n_points=100, noise_level=0.04, seed=0)
    # Keep only sensor columns + twin columns
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sensor_Data")
        # Info sheet
        info = pd.DataFrame({
            "Column": REQUIRED_COLUMNS,
            "Units": [COLUMN_UNITS[c] for c in REQUIRED_COLUMNS],
            "Description": [
                "Time index", "Feed flow rate", "Distillate flow rate",
                "Bottoms flow rate", "Feed composition (mol frac)",
                "Distillate composition (mol frac)", "Bottoms composition (mol frac)",
                "Reflux ratio", "Reboiler duty", "Condenser duty",
                "Top stage temperature", "Bottom stage temperature",
            ],
        })
        info.to_excel(writer, index=False, sheet_name="Column_Info")
    buf.seek(0)
    return buf.read()


# ===========================================================================
# 3. OUTLIER DETECTION / REMOVAL (IQR)
# ===========================================================================


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str],
                         k: float = 1.5) -> dict:
    """
    Detect outliers using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to analyse.
    k : float
        IQR multiplier (default 1.5; use 3.0 for extreme outliers only).

    Returns
    -------
    dict with keys:
        'mask'   : DataFrame[bool], True where a value is an outlier.
        'bounds' : dict col -> (lower, upper).
        'counts' : dict col -> int (number of outliers).
    """
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    bounds = {}
    counts = {}
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        outlier_mask = (df[col] < lo) | (df[col] > hi)
        mask[col] = outlier_mask
        bounds[col] = (lo, hi)
        counts[col] = int(outlier_mask.sum())
    return {"mask": mask, "bounds": bounds, "counts": counts}


def remove_outliers_iqr(df: pd.DataFrame, outlier_result: dict,
                         columns: list[str]) -> pd.DataFrame:
    """
    Replace outlier values with NaN and forward/backward fill.

    Parameters
    ----------
    df : pd.DataFrame
        Original data.
    outlier_result : dict
        Output of detect_outliers_iqr.
    columns : list[str]
        Columns to clean.

    Returns
    -------
    pd.DataFrame
        Data with outliers replaced and filled.
    """
    df_clean = df.copy()
    mask = outlier_result["mask"]
    for col in columns:
        df_clean.loc[mask[col], col] = np.nan
    # Fill NaN: forward then backward
    df_clean[columns] = df_clean[columns].ffill().bfill()
    return df_clean


def plot_outliers(df: pd.DataFrame, outlier_result: dict, columns: list[str],
                  time_col: str = "Time") -> dict:
    """
    Generate before/after outlier plots for each variable.

    Returns
    -------
    dict col -> (fig_before, fig_after)
    """
    figures = {}
    t = df[time_col].values
    mask = outlier_result["mask"]
    bounds = outlier_result["bounds"]

    for col in columns:
        lo, hi = bounds[col]
        outlier_idx = mask[col]
        clean_idx = ~outlier_idx

        # --- Figure: before (with outliers marked) ---
        fig_before, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, df[col], color="steelblue", linewidth=1, label="Data")
        ax.scatter(t[outlier_idx], df[col][outlier_idx],
                   color="red", s=40, zorder=5, label=f"Outliers (n={outlier_idx.sum()})")
        ax.axhline(lo, color="orange", linestyle="--", linewidth=0.9, label=f"Lower fence ({lo:.3g})")
        ax.axhline(hi, color="orange", linestyle="--", linewidth=0.9, label=f"Upper fence ({hi:.3g})")
        ax.set_xlabel(f"Time [{COLUMN_UNITS.get(time_col, '')}]")
        ax.set_ylabel(f"{col} [{COLUMN_UNITS.get(col, '')}]")
        ax.set_title(f"{col} — IQR Outlier Detection (k={outlier_result.get('k', 1.5):.1f})")
        ax.legend(fontsize=7)
        fig_before.tight_layout()

        # --- Figure: after (outliers removed) ---
        fig_after, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(t[clean_idx], df[col][clean_idx],
                 color="green", linewidth=1, label="Clean data")
        ax2.set_xlabel(f"Time [{COLUMN_UNITS.get(time_col, '')}]")
        ax2.set_ylabel(f"{col} [{COLUMN_UNITS.get(col, '')}]")
        ax2.set_title(f"{col} — After Outlier Removal")
        ax2.legend(fontsize=7)
        fig_after.tight_layout()

        figures[col] = (fig_before, fig_after)

    return figures


# ===========================================================================
# 4. SIGNAL FILTERING
# ===========================================================================


def apply_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Apply a centred moving-average filter.

    Parameters
    ----------
    series : pd.Series
    window : int
        Number of samples in the window (odd recommended).

    Returns
    -------
    pd.Series
        Filtered signal (NaNs at edges filled with edge values).
    """
    filtered = series.rolling(window=window, center=True, min_periods=1).mean()
    return filtered


def apply_lowpass_filter(series: pd.Series, cutoff_freq: float,
                          fs: float, order: int = 4) -> pd.Series:
    """
    Apply a Butterworth low-pass filter.

    Parameters
    ----------
    series : pd.Series
    cutoff_freq : float
        Cutoff frequency in Hz (must be < fs/2).
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (default 4).

    Returns
    -------
    pd.Series
        Filtered signal.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    norm_cutoff = np.clip(norm_cutoff, 1e-4, 0.9999)
    b, a = signal.butter(order, norm_cutoff, btype="low", analog=False)
    # Use filtfilt for zero-phase filtering
    arr = series.fillna(series.median()).values
    filtered = signal.filtfilt(b, a, arr)
    return pd.Series(filtered, index=series.index, name=series.name)


def filter_dataframe(df: pd.DataFrame, columns: list[str],
                     method: str, **kwargs) -> pd.DataFrame:
    """
    Apply the selected filter to all specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    method : str
        "moving_average" or "lowpass".
    **kwargs
        window (int) for moving_average; cutoff_freq (float), fs (float) for lowpass.

    Returns
    -------
    pd.DataFrame with filtered columns appended as 'col_filtered'.
    """
    df_out = df.copy()
    for col in columns:
        if method == "moving_average":
            df_out[col + "_filtered"] = apply_moving_average(df[col], kwargs["window"])
        elif method == "lowpass":
            df_out[col + "_filtered"] = apply_lowpass_filter(
                df[col], kwargs["cutoff_freq"], kwargs["fs"], kwargs.get("order", 4)
            )
    return df_out


def plot_filter_comparison(df: pd.DataFrame, columns: list[str],
                            time_col: str = "Time") -> dict:
    """
    Generate comparison plots (raw vs filtered) for each variable.

    Returns
    -------
    dict col -> fig
    """
    figures = {}
    t = df[time_col].values
    for col in columns:
        fcol = col + "_filtered"
        if fcol not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, df[col], color="lightblue", linewidth=0.8, alpha=0.9, label="Raw")
        ax.plot(t, df[fcol], color="navy", linewidth=1.5, label="Filtered")
        ax.set_xlabel(f"Time [{COLUMN_UNITS.get(time_col, '')}]")
        ax.set_ylabel(f"{col} [{COLUMN_UNITS.get(col, '')}]")
        ax.set_title(f"{col} — Raw vs Filtered Signal")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures[col] = fig
    return figures


# ===========================================================================
# 5. DATA RECONCILIATION (WLS)
# ===========================================================================


def reconcile_data_wls(df: pd.DataFrame,
                        meas_cols: list[str],
                        A: np.ndarray,
                        b_vec: np.ndarray,
                        sigma: np.ndarray | None = None) -> pd.DataFrame:
    """
    Weighted Least Squares (WLS) data reconciliation.

    Minimises:  (x - x̂)ᵀ Σ⁻¹ (x - x̂)
    subject to: A x̂ = b   (linear balance constraints)

    Analytical solution:
        x̂ = x - Σ Aᵀ (A Σ Aᵀ)⁻¹ (A x - b)

    Parameters
    ----------
    df : pd.DataFrame
        Contains columns meas_cols (sensor measurements, already filtered/cleaned).
    meas_cols : list[str]
        Variables to reconcile (must match columns of A).
    A : np.ndarray, shape (n_constraints, n_vars)
        Constraint matrix.
    b_vec : np.ndarray, shape (n_constraints,)
        Right-hand side of constraints.
    sigma : np.ndarray, shape (n_vars, n_vars) or None
        Measurement covariance matrix. If None, uses identity (equally weighted).

    Returns
    -------
    pd.DataFrame
        Original columns plus reconciled columns (suffix '_rec') and
        adjustment columns (suffix '_adj').
    """
    df_out = df.copy()
    n_vars = len(meas_cols)

    if sigma is None:
        sigma = np.eye(n_vars)

    # Row-by-row reconciliation
    rec_data = {col: [] for col in meas_cols}
    adj_data = {col: [] for col in meas_cols}

    for _, row in df[meas_cols].iterrows():
        x_meas = row.values.astype(float)
        residual = A @ x_meas - b_vec  # constraint residual before reconciliation
        try:
            # WLS analytical solution
            ASAT = A @ sigma @ A.T
            adjustment = sigma @ A.T @ np.linalg.solve(ASAT, residual)
            x_rec = x_meas - adjustment
        except np.linalg.LinAlgError:
            x_rec = x_meas.copy()
            adjustment = np.zeros(n_vars)

        for i, col in enumerate(meas_cols):
            rec_data[col].append(x_rec[i])
            adj_data[col].append(x_meas[i] - x_rec[i])

    for col in meas_cols:
        df_out[col + "_rec"] = rec_data[col]
        df_out[col + "_adj"] = adj_data[col]

    return df_out


def build_distillation_constraints() -> tuple[np.ndarray, np.ndarray]:
    """
    Build the balance constraint matrix for the distillation column.

    Variables order: [F, D, B, zF, xD, xB]

    Constraints:
      1. Overall mass balance:    F - D - B = 0
      2. Component balance:       F*zF - D*xD - B*xB  ≈ 0
         (linearised around nominal: F0=100, D0=38, B0=62, zF0=0.4, xD0=0.92, xB0=0.05)

    Returns
    -------
    A : np.ndarray, shape (2, 6)
    b : np.ndarray, shape (2,)
    """
    # Nominal values for linearisation
    F0, D0, B0 = 100.0, 38.0, 62.0
    zF0, xD0, xB0 = 0.40, 0.92, 0.05

    # Constraint 1: F - D - B = 0
    # [F, D, B, zF, xD, xB]
    A1 = np.array([1.0, -1.0, -1.0, 0.0, 0.0, 0.0])

    # Constraint 2 (linearised component balance):
    # F*zF - D*xD - B*xB ≈ (zF0)*dF + F0*dzF - (xD0)*dD - D0*dxD - (xB0)*dB - B0*dxB = 0
    # Around nominal: F0*zF0 - D0*xD0 - B0*xB0 = 100*0.4 - 38*0.92 - 62*0.05 = 40 - 34.96 - 3.1 = 1.94 ≈ 0
    # Linearised Jacobian row:
    A2 = np.array([zF0, -xD0, -xB0, F0, -D0, -B0])
    b2 = F0 * zF0 - D0 * xD0 - B0 * xB0  # nominal RHS

    A = np.vstack([A1, A2])
    b = np.array([0.0, b2])
    return A, b


def plot_reconciliation(df: pd.DataFrame, columns: list[str],
                         time_col: str = "Time") -> dict:
    """
    Plot measured vs reconciled values for each variable.

    Returns
    -------
    dict col -> fig
    """
    figures = {}
    t = df[time_col].values
    for col in columns:
        rcol = col + "_rec"
        acol = col + "_adj"
        if rcol not in df.columns:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(t, df[col], color="steelblue", linewidth=0.8, alpha=0.8, label="Measured")
        ax1.plot(t, df[rcol], color="crimson", linewidth=1.3, linestyle="--", label="Reconciled")
        ax1.set_ylabel(f"{col} [{COLUMN_UNITS.get(col, '')}]")
        ax1.set_title(f"{col} — Measured vs Reconciled")
        ax1.legend(fontsize=8)

        ax2.bar(t, df[acol], color="orange", alpha=0.7, width=0.7 * (t[1] - t[0]))
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_xlabel(f"Time [{COLUMN_UNITS.get(time_col, '')}]")
        ax2.set_ylabel("Adjustment")
        ax2.set_title(f"{col} — Reconciliation Adjustments")

        fig.tight_layout()
        figures[col] = fig
    return figures


# ===========================================================================
# 6. KPIs AND ADHERENCE INDICATORS
# ===========================================================================


def compute_kpis(df: pd.DataFrame,
                 sensor_cols: list[str],
                 twin_prefix: str = "twin_",
                 band_pct: float = 5.0) -> pd.DataFrame:
    """
    Compute KPIs between sensor (cleaned/filtered/reconciled) and twin values.

    KPIs computed per variable:
    - ME   : Mean Error
    - MAE  : Mean Absolute Error
    - RMSE : Root Mean Squared Error
    - Adherence (%) : % of time points within ±band% of the twin value
    - Data availability (%) : % of non-NaN points

    Parameters
    ----------
    df : pd.DataFrame
    sensor_cols : list[str]
        Sensor column names. Twin columns are inferred as 'twin_<col>'.
    twin_prefix : str
        Prefix used for twin (simulated) column names.
    band_pct : float
        Adherence band as a percentage of the twin value (e.g., 5.0 = ±5%).

    Returns
    -------
    pd.DataFrame
        KPI table with one row per variable.
    """
    rows = []
    for col in sensor_cols:
        tcol = twin_prefix + col
        if tcol not in df.columns:
            rows.append({
                "Variable": col,
                "Units": COLUMN_UNITS.get(col, "-"),
                "ME": np.nan, "MAE": np.nan, "RMSE": np.nan,
                "Adherence (%)": np.nan,
                "Data Availability (%)": np.nan,
                "N_valid": np.nan,
            })
            continue

        sensor = df[col].dropna()
        twin = df.loc[sensor.index, tcol]

        diff = sensor - twin
        me = diff.mean()
        mae = diff.abs().mean()
        rmse = np.sqrt((diff ** 2).mean())

        # Adherence: fraction within ±band_pct% of twin
        band = (band_pct / 100.0) * twin.abs()
        adherent = (diff.abs() <= band).sum() / len(diff) * 100

        avail = df[col].notna().mean() * 100

        rows.append({
            "Variable": col,
            "Units": COLUMN_UNITS.get(col, "-"),
            "ME": round(me, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Adherence (%)": round(adherent, 2),
            "Data Availability (%)": round(avail, 2),
            "N_valid": int(df[col].notna().sum()),
        })

    return pd.DataFrame(rows)


def compute_outlier_summary(df_original: pd.DataFrame,
                             outlier_result: dict,
                             columns: list[str]) -> pd.DataFrame:
    """
    Compute outlier summary statistics.

    Returns
    -------
    pd.DataFrame with columns: Variable, Total_Points, Outliers, Outlier_Pct
    """
    rows = []
    mask = outlier_result["mask"]
    for col in columns:
        n_total = df_original[col].notna().sum()
        n_out = mask[col].sum()
        rows.append({
            "Variable": col,
            "Total Points": int(n_total),
            "Outliers": int(n_out),
            "Outlier %": round(n_out / max(n_total, 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def plot_kpi_summary(kpi_df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart summarising KPIs across variables.

    Returns
    -------
    matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["MAE", "RMSE", "Adherence (%)"]
    colors = ["steelblue", "tomato", "seagreen"]

    for ax, metric, color in zip(axes, metrics, colors):
        valid = kpi_df.dropna(subset=[metric])
        if valid.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.barh(valid["Variable"], valid[metric], color=color, alpha=0.8)
            ax.set_xlabel(metric)
            ax.set_title(metric)
            ax.invert_yaxis()

    fig.suptitle("KPI Summary — Digital Twin vs Sensor Data", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_adherence_timeline(df: pd.DataFrame, sensor_cols: list[str],
                             twin_prefix: str = "twin_",
                             band_pct: float = 5.0,
                             time_col: str = "Time") -> dict:
    """
    Plot adherence timeline (sensor vs twin with adherence band) per variable.

    Returns
    -------
    dict col -> fig
    """
    figures = {}
    t = df[time_col].values
    for col in sensor_cols:
        tcol = twin_prefix + col
        if tcol not in df.columns:
            continue
        sensor = df[col].values
        twin = df[tcol].values
        band = np.abs(twin) * (band_pct / 100.0)
        in_band = np.abs(sensor - twin) <= band

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(t, twin, color="blue", linewidth=1.5, label="Twin (simulated)")
        ax.fill_between(t, twin - band, twin + band,
                        color="blue", alpha=0.15, label=f"±{band_pct}% band")
        ax.scatter(t[in_band], sensor[in_band], color="green", s=15, alpha=0.7, label="In band")
        ax.scatter(t[~in_band], sensor[~in_band], color="red", s=20, alpha=0.8, label="Out of band")
        adherence = in_band.mean() * 100
        ax.set_xlabel(f"Time [{COLUMN_UNITS.get(time_col, '')}]")
        ax.set_ylabel(f"{col} [{COLUMN_UNITS.get(col, '')}]")
        ax.set_title(f"{col} — Adherence: {adherence:.1f}% within ±{band_pct}% of twin")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures[col] = fig
    return figures


# ===========================================================================
# 7. REPORT EXPORT
# ===========================================================================


def export_results_excel(df_sensor: pd.DataFrame,
                          df_clean: pd.DataFrame,
                          df_filtered: pd.DataFrame,
                          df_rec: pd.DataFrame,
                          kpi_df: pd.DataFrame,
                          outlier_summary: pd.DataFrame) -> bytes:
    """
    Export all pipeline results to a single Excel workbook.

    Returns
    -------
    bytes
        Excel file content.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_sensor.to_excel(writer, index=False, sheet_name="Raw_Sensor_Data")
        df_clean.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        df_filtered.to_excel(writer, index=False, sheet_name="Filtered_Data")
        df_rec.to_excel(writer, index=False, sheet_name="Reconciled_Data")
        kpi_df.to_excel(writer, index=False, sheet_name="KPIs")
        outlier_summary.to_excel(writer, index=False, sheet_name="Outlier_Summary")
    buf.seek(0)
    return buf.read()


# ===========================================================================
# 8. STREAMLIT PAGE
# ===========================================================================

_SENSOR_COLS = [c for c in REQUIRED_COLUMNS if c != "Time"]


def _init_session_state():
    """Initialise session state keys for the Digital Twin pipeline."""
    defaults = {
        "dt_raw_df": None,
        "dt_clean_df": None,
        "dt_filtered_df": None,
        "dt_rec_df": None,
        "dt_kpi_df": None,
        "dt_outlier_result": None,
        "dt_outlier_summary": None,
        "dt_filter_method": "moving_average",
        "dt_filter_params": {},
        "dt_iqr_k": 1.5,
        "dt_band_pct": 5.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def gemelo_digital_page():
    """
    Main Streamlit page for the Digital Twin — Distillation Column module.
    """
    _init_session_state()

    st.header("🏭 Digital Twin — Distillation Column")
    st.markdown(
        """
        This module implements a **complete digital twin pipeline** for a binary
        distillation column (ethanol-water). The workflow covers:

        | Step | Description |
        |------|-------------|
        | **1. Data Ingestion** | Load DWSIM mock data or upload an Excel sensor file |
        | **2. Outlier Treatment** | IQR-based detection and removal (with charts) |
        | **3. Signal Filtering** | Moving average or low-pass Butterworth filter |
        | **4. Data Reconciliation** | WLS reconciliation with mass/component balance |
        | **5. KPIs & Adherence** | ME, MAE, RMSE, adherence %, availability |
        """
    )

    with st.expander("📚 Theoretical Background", expanded=False):
        st.markdown(r"""
        #### Digital Twin Concept
        A **digital twin** is a virtual replica of a physical process that runs in
        parallel, using real (sensor) data and a process model to monitor performance,
        detect anomalies, and support decision-making.

        #### Mass Balance (Distillation Column)
        $$F = D + B$$
        $$F \cdot z_F = D \cdot x_D + B \cdot x_B$$

        where $F$, $D$, $B$ are feed, distillate, and bottoms molar flow rates, and
        $z_F$, $x_D$, $x_B$ are the corresponding ethanol mole fractions.

        #### IQR Outlier Detection
        For a variable $x$:
        $$\text{IQR} = Q_3 - Q_1, \quad
          x_{\text{low}} = Q_1 - k \cdot \text{IQR}, \quad
          x_{\text{high}} = Q_3 + k \cdot \text{IQR}$$
        Values outside $[x_{\text{low}}, x_{\text{high}}]$ are flagged as outliers.

        #### WLS Data Reconciliation
        Minimise the weighted sum of squared adjustments subject to linear balance constraints:
        $$\min_{\hat{x}} (\hat{x} - x)^\top \Sigma^{-1} (\hat{x} - x) \quad
          \text{s.t.} \quad A\hat{x} = b$$
        Analytical solution:
        $$\hat{x} = x - \Sigma A^\top (A \Sigma A^\top)^{-1} (Ax - b)$$

        #### KPI Definitions
        | KPI | Formula |
        |-----|---------|
        | ME | $\frac{1}{N}\sum(x_{\text{sensor}} - x_{\text{twin}})$ |
        | MAE | $\frac{1}{N}\sum|x_{\text{sensor}} - x_{\text{twin}}|$ |
        | RMSE | $\sqrt{\frac{1}{N}\sum(x_{\text{sensor}} - x_{\text{twin}})^2}$ |
        | Adherence | $\frac{\text{# points within band}}{N} \times 100\%$ |
        """)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 1. Data Ingestion",
        "🔍 2. Outlier Treatment",
        "📊 3. Signal Filtering",
        "⚖️ 4. Data Reconciliation",
        "📈 5. KPIs & Adherence",
    ])

    # ===================================================================
    # TAB 1 — DATA INGESTION
    # ===================================================================
    with tab1:
        st.subheader("Step 1: Data Ingestion")

        col_src, col_cfg = st.columns([1, 1])

        with col_src:
            st.markdown("#### Data Source")
            data_source = st.radio(
                "Choose data source:",
                ["DWSIM Mock Simulation", "Upload Excel File"],
                key="dt_data_source",
                help="Use mock data for demo, or upload your own sensor Excel file.",
            )

        with col_cfg:
            st.markdown("#### Configuration")
            if data_source == "DWSIM Mock Simulation":
                n_pts = st.number_input("Number of time points", 30, 500, 120, 10, key="dt_n_pts")
                dt_val = st.number_input("Sampling interval [min]", 0.5, 60.0, 1.0, 0.5, key="dt_dt")
                noise = st.slider("Noise level (fraction of nominal)", 0.01, 0.20, 0.04, 0.01,
                                  key="dt_noise")
                seed = st.number_input("Random seed", 0, 9999, 42, key="dt_seed")
            else:
                uploaded = st.file_uploader(
                    "Upload Excel sensor file (.xlsx)",
                    type=["xlsx"],
                    key="dt_excel_upload",
                )
                st.markdown("**Expected columns:**")
                cols_info = pd.DataFrame(
                    {"Column": REQUIRED_COLUMNS,
                     "Units": [COLUMN_UNITS[c] for c in REQUIRED_COLUMNS]}
                )
                st.dataframe(cols_info, use_container_width=True, hide_index=True)

                # Download example template
                example_bytes = create_example_excel()
                st.download_button(
                    "⬇️ Download Example Excel Template",
                    data=example_bytes,
                    file_name="distillation_sensor_example.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dt_dl_example",
                )

        st.markdown("---")

        if st.button("▶ Load / Generate Data", key="dt_load_btn", type="primary"):
            with st.spinner("Loading data…"):
                try:
                    if data_source == "DWSIM Mock Simulation":
                        df = simulate_distillation_column(
                            n_points=int(n_pts), dt=float(dt_val),
                            noise_level=float(noise), seed=int(seed)
                        )
                        st.session_state["dt_raw_df"] = df
                        st.success(f"✅ Mock simulation generated: {len(df)} points, "
                                   f"Δt = {dt_val} min, noise = {noise:.0%}")
                    else:
                        if uploaded is None:
                            st.error("Please upload an Excel file first.")
                        else:
                            df, warns = load_excel_sensor_data(uploaded)
                            st.session_state["dt_raw_df"] = df
                            if warns:
                                for w in warns:
                                    st.warning(w)
                            st.success(f"✅ Excel file loaded: {len(df)} rows, "
                                       f"{len(df.columns)} columns")
                    # Reset downstream results when new data is loaded
                    for k in ["dt_clean_df", "dt_filtered_df", "dt_rec_df",
                               "dt_kpi_df", "dt_outlier_result", "dt_outlier_summary"]:
                        st.session_state[k] = None
                except Exception as exc:
                    st.error(f"Error loading data: {exc}")

        if st.session_state["dt_raw_df"] is not None:
            df_raw = st.session_state["dt_raw_df"]
            st.markdown("#### Preview (sensor columns)")
            st.dataframe(df_raw[REQUIRED_COLUMNS].head(10), use_container_width=True)
            st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

            # Quick stats
            with st.expander("📊 Descriptive Statistics"):
                st.dataframe(df_raw[_SENSOR_COLS].describe().round(4), use_container_width=True)

    # ===================================================================
    # TAB 2 — OUTLIER TREATMENT
    # ===================================================================
    with tab2:
        st.subheader("Step 2: Outlier Detection & Removal (IQR)")

        if st.session_state["dt_raw_df"] is None:
            st.info("⚠️ Load data first (Step 1).")
        else:
            df_raw = st.session_state["dt_raw_df"]

            col_cfg2, col_info2 = st.columns([1, 2])
            with col_cfg2:
                st.markdown("#### IQR Parameters")
                k_val = st.number_input(
                    "IQR multiplier k", 0.5, 5.0,
                    float(st.session_state["dt_iqr_k"]), 0.1,
                    key="dt_k_input",
                    help="k=1.5 (standard), k=3.0 (extreme outliers only)"
                )
                selected_cols = st.multiselect(
                    "Variables to analyse",
                    _SENSOR_COLS,
                    default=_SENSOR_COLS,
                    key="dt_outlier_cols",
                )

            with col_info2:
                st.markdown("#### Method")
                st.markdown(r"""
                Fences:  $Q_1 - k \cdot \text{IQR}$ and $Q_3 + k \cdot \text{IQR}$

                Values outside these fences are flagged as outliers. After removal,
                NaNs are filled with forward/backward fill.
                """)

            if st.button("▶ Detect & Remove Outliers", key="dt_iqr_btn", type="primary"):
                if not selected_cols:
                    st.error("Select at least one variable.")
                else:
                    with st.spinner("Running IQR analysis…"):
                        result = detect_outliers_iqr(df_raw, selected_cols, k=k_val)
                        result["k"] = k_val  # store k for plot labels
                        df_clean = remove_outliers_iqr(df_raw, result, selected_cols)
                        st.session_state.update({
                            "dt_outlier_result": result,
                            "dt_clean_df": df_clean,
                            "dt_iqr_k": k_val,
                            "dt_outlier_summary": compute_outlier_summary(df_raw, result, selected_cols),
                            # Reset downstream
                            "dt_filtered_df": None, "dt_rec_df": None, "dt_kpi_df": None,
                        })
                    st.success("✅ Outlier analysis complete.")

            if st.session_state["dt_outlier_result"] is not None:
                result = st.session_state["dt_outlier_result"]
                summary = st.session_state["dt_outlier_summary"]

                st.markdown("#### Outlier Summary")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("#### Plots (Before / After)")
                figs = plot_outliers(df_raw, result,
                                     [c for c in selected_cols if c in result["mask"].columns])

                for col_name, (fig_b, fig_a) in figs.items():
                    with st.expander(f"🔍 {col_name}", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Before removal (outliers marked)**")
                            st.pyplot(fig_b)
                        with c2:
                            st.markdown("**After removal**")
                            st.pyplot(fig_a)
                    plt.close("all")

    # ===================================================================
    # TAB 3 — SIGNAL FILTERING
    # ===================================================================
    with tab3:
        st.subheader("Step 3: Signal Filtering")

        src_df = st.session_state["dt_clean_df"] or st.session_state["dt_raw_df"]
        if src_df is None:
            st.info("⚠️ Load data first (Step 1).")
        else:
            col_cfg3, col_info3 = st.columns([1, 2])

            with col_cfg3:
                st.markdown("#### Filter Configuration")
                filter_method = st.selectbox(
                    "Filter type",
                    ["Moving Average", "Low-Pass Butterworth"],
                    key="dt_filter_type",
                )
                filter_cols = st.multiselect(
                    "Variables to filter",
                    _SENSOR_COLS,
                    default=_SENSOR_COLS,
                    key="dt_filter_cols",
                )

                if filter_method == "Moving Average":
                    win = st.number_input(
                        "Window size (samples)", 2, 50, 7, 1,
                        key="dt_ma_window",
                        help="Number of samples in the centred moving average window.",
                    )
                    filter_params = {"method": "moving_average", "window": int(win)}
                else:  # Low-pass
                    dt_step = float(src_df["Time"].diff().median()) if len(src_df) > 1 else 1.0
                    fs_default = round(1.0 / max(dt_step, 1e-6) / 60, 4)  # Hz (time in min → /60)
                    fs_val = st.number_input(
                        "Sampling frequency fs [Hz]", 1e-6, 10.0,
                        max(fs_default, 1e-4), format="%.5f",
                        key="dt_fs",
                        help="fs = 1/Δt. If time is in minutes, Δt_min → Δt_s = Δt_min×60.",
                    )
                    fc_val = st.number_input(
                        "Cutoff frequency fc [Hz]", 1e-6, float(fs_val / 2) * 0.99,
                        max(float(fs_val) * 0.1, 1e-6), format="%.5f",
                        key="dt_fc",
                        help="Cutoff frequency (must be < fs/2).",
                    )
                    order_val = st.selectbox("Filter order", [2, 4, 6, 8], index=1, key="dt_forder")
                    filter_params = {
                        "method": "lowpass",
                        "cutoff_freq": float(fc_val),
                        "fs": float(fs_val),
                        "order": int(order_val),
                    }

            with col_info3:
                st.markdown("#### Method Details")
                if filter_method == "Moving Average":
                    st.markdown(r"""
                    **Centred Moving Average:**
                    $$\hat{x}[n] = \frac{1}{W}\sum_{k=-W/2}^{W/2} x[n+k]$$

                    Simple, non-parametric, good for removing high-frequency noise.
                    Edge effects are minimised using `min_periods=1`.
                    """)
                else:
                    st.markdown(r"""
                    **Butterworth Low-Pass Filter (zero-phase, `filtfilt`):**

                    Maximally flat magnitude response in the passband.
                    The transfer function in the $s$-domain is:
                    $$|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}$$

                    Zero-phase implementation (`scipy.signal.filtfilt`) prevents
                    phase distortion by applying the filter forward and backward.
                    """)

            if st.button("▶ Apply Filter", key="dt_filter_btn", type="primary"):
                if not filter_cols:
                    st.error("Select at least one variable.")
                else:
                    with st.spinner("Applying filter…"):
                        try:
                            method_key = filter_params.pop("method")
                            df_filt = filter_dataframe(src_df, filter_cols,
                                                       method_key, **filter_params)
                            filter_params["method"] = method_key  # restore
                            st.session_state.update({
                                "dt_filtered_df": df_filt,
                                "dt_filter_method": method_key,
                                "dt_filter_params": filter_params,
                                "dt_rec_df": None, "dt_kpi_df": None,
                            })
                            st.success("✅ Filter applied.")
                        except Exception as exc:
                            st.error(f"Filtering error: {exc}")

            if st.session_state["dt_filtered_df"] is not None:
                df_filt = st.session_state["dt_filtered_df"]
                filt_cols_present = [c for c in filter_cols
                                     if c + "_filtered" in df_filt.columns]
                st.markdown("#### Comparison Plots (Raw vs Filtered)")
                figs_f = plot_filter_comparison(df_filt, filt_cols_present)
                for col_name, fig_f in figs_f.items():
                    with st.expander(f"📊 {col_name}", expanded=False):
                        st.pyplot(fig_f)
                    plt.close(fig_f)

    # ===================================================================
    # TAB 4 — DATA RECONCILIATION
    # ===================================================================
    with tab4:
        st.subheader("Step 4: Data Reconciliation (WLS)")

        # Determine the best available input dataframe
        if st.session_state["dt_filtered_df"] is not None:
            src_rec = st.session_state["dt_filtered_df"].copy()
            # Use filtered columns if available, else raw
            for c in BALANCE_VARS:
                fc = c + "_filtered"
                if fc in src_rec.columns:
                    src_rec[c] = src_rec[fc]
            st.info("ℹ️ Using filtered data as input for reconciliation.")
        elif st.session_state["dt_clean_df"] is not None:
            src_rec = st.session_state["dt_clean_df"]
            st.info("ℹ️ Using cleaned (outlier-free) data as input for reconciliation.")
        elif st.session_state["dt_raw_df"] is not None:
            src_rec = st.session_state["dt_raw_df"]
            st.warning("⚠️ Using raw data (no filtering applied). Run Steps 2–3 for better results.")
        else:
            st.info("⚠️ Load data first (Step 1).")
            src_rec = None

        if src_rec is not None:
            col_cfg4, col_info4 = st.columns([1, 2])

            with col_cfg4:
                st.markdown("#### Configuration")
                rec_cols = st.multiselect(
                    "Variables to reconcile",
                    BALANCE_VARS,
                    default=BALANCE_VARS,
                    key="dt_rec_cols",
                    help="Must include F, D, B for mass balance; add zF, xD, xB for component balance.",
                )
                st.markdown("**Measurement uncertainties (σ):**")
                sigma_diag = []
                sigma_defaults = {"F": 2.0, "D": 1.5, "B": 1.5,
                                  "zF": 0.01, "xD": 0.01, "xB": 0.005}
                for c in rec_cols:
                    sv = st.number_input(
                        f"σ({c}) [{COLUMN_UNITS.get(c, '')}]",
                        0.001, 100.0, sigma_defaults.get(c, 1.0), format="%.4f",
                        key=f"dt_sigma_{c}"
                    )
                    sigma_diag.append(sv ** 2)  # variance

            with col_info4:
                st.markdown("#### Constraints")
                st.markdown("""
                The reconciliation enforces:

                1. **Overall mass balance:** F = D + B  
                2. **Component balance:** F·z_F ≈ D·x_D + B·x_B  
                   (linearised around nominal operating point)

                If only flow variables (F, D, B) are selected, only constraint 1 is applied.
                """)
                A_full, b_full = build_distillation_constraints()
                # Display constraint matrix
                st.markdown("Constraint matrix A (full, 6 variables):")
                A_df = pd.DataFrame(A_full, columns=BALANCE_VARS,
                                    index=["Mass balance", "Component balance"])
                st.dataframe(A_df.round(4), use_container_width=True)

            if st.button("▶ Run Reconciliation", key="dt_rec_btn", type="primary"):
                if len(rec_cols) < 2:
                    st.error("Select at least 2 variables.")
                else:
                    with st.spinner("Running WLS reconciliation…"):
                        try:
                            # Build sub-matrix for selected variables
                            col_idx = [BALANCE_VARS.index(c) for c in rec_cols]
                            A_sub = A_full[:, col_idx]
                            # Drop constraints that are all-zero after subsetting
                            row_mask = np.any(A_sub != 0, axis=1)
                            A_sub = A_sub[row_mask]
                            b_sub = b_full[row_mask]

                            sigma_mat = np.diag(sigma_diag)

                            df_rec = reconcile_data_wls(
                                src_rec, rec_cols, A_sub, b_sub, sigma=sigma_mat
                            )
                            st.session_state.update({
                                "dt_rec_df": df_rec,
                                "dt_kpi_df": None,
                            })
                            st.success("✅ Reconciliation complete.")
                        except Exception as exc:
                            st.error(f"Reconciliation error: {exc}")

            if st.session_state["dt_rec_df"] is not None:
                df_rec = st.session_state["dt_rec_df"]
                rec_cols_stored = [c for c in BALANCE_VARS if c + "_rec" in df_rec.columns]

                st.markdown("#### Reconciliation Plots")
                figs_r = plot_reconciliation(df_rec, rec_cols_stored)
                for col_name, fig_r in figs_r.items():
                    with st.expander(f"⚖️ {col_name}", expanded=False):
                        st.pyplot(fig_r)
                    plt.close(fig_r)

                # Show adjustments table
                adj_cols = [c + "_adj" for c in rec_cols_stored if c + "_adj" in df_rec.columns]
                if adj_cols:
                    with st.expander("📋 Adjustment Statistics"):
                        adj_stats = df_rec[adj_cols].describe().round(5)
                        adj_stats.columns = [c.replace("_adj", "") for c in adj_stats.columns]
                        st.dataframe(adj_stats, use_container_width=True)

    # ===================================================================
    # TAB 5 — KPIs & ADHERENCE
    # ===================================================================
    with tab5:
        st.subheader("Step 5: KPIs & Adherence Indicators")

        # Build the best available sensor dataframe
        if st.session_state["dt_rec_df"] is not None:
            kpi_src = st.session_state["dt_rec_df"].copy()
            # Use reconciled values for sensor columns in KPIs
            for c in _SENSOR_COLS:
                rc = c + "_rec"
                if rc in kpi_src.columns:
                    kpi_src[c] = kpi_src[rc]
            st.info("ℹ️ KPIs computed on reconciled sensor values.")
        elif st.session_state["dt_filtered_df"] is not None:
            kpi_src = st.session_state["dt_filtered_df"].copy()
            for c in _SENSOR_COLS:
                fc = c + "_filtered"
                if fc in kpi_src.columns:
                    kpi_src[c] = kpi_src[fc]
            st.info("ℹ️ KPIs computed on filtered sensor values.")
        elif st.session_state["dt_clean_df"] is not None:
            kpi_src = st.session_state["dt_clean_df"]
            st.info("ℹ️ KPIs computed on cleaned sensor values.")
        elif st.session_state["dt_raw_df"] is not None:
            kpi_src = st.session_state["dt_raw_df"]
            st.warning("⚠️ KPIs computed on raw sensor values.")
        else:
            kpi_src = None
            st.info("⚠️ Load data first (Step 1).")

        if kpi_src is not None:
            col_cfg5, col_info5 = st.columns([1, 2])
            with col_cfg5:
                st.markdown("#### KPI Configuration")
                band_pct = st.number_input(
                    "Adherence band (% of twin value)", 0.5, 50.0,
                    float(st.session_state["dt_band_pct"]), 0.5,
                    key="dt_band_input",
                    help="A sensor value is 'adherent' if it lies within ±band% of the twin value."
                )
                kpi_cols = st.multiselect(
                    "Variables for KPI",
                    _SENSOR_COLS,
                    default=_SENSOR_COLS,
                    key="dt_kpi_cols",
                )

            twin_available = any("twin_" + c in kpi_src.columns for c in _SENSOR_COLS)
            with col_info5:
                if not twin_available:
                    st.warning(
                        "⚠️ Twin (simulated) columns not found. "
                        "KPIs requiring sensor-vs-twin comparison will show NaN. "
                        "Use the DWSIM Mock Simulation to get twin values, "
                        "or add 'twin_*' columns to your Excel file."
                    )
                else:
                    st.success("✅ Twin columns found — full KPI comparison available.")

            if st.button("▶ Compute KPIs", key="dt_kpi_btn", type="primary"):
                with st.spinner("Computing KPIs…"):
                    try:
                        kpi_df = compute_kpis(kpi_src, kpi_cols, band_pct=band_pct)
                        st.session_state.update({
                            "dt_kpi_df": kpi_df,
                            "dt_band_pct": band_pct,
                        })
                        st.success("✅ KPIs computed.")
                    except Exception as exc:
                        st.error(f"KPI computation error: {exc}")

            if st.session_state["dt_kpi_df"] is not None:
                kpi_df = st.session_state["dt_kpi_df"]

                st.markdown("#### KPI Table")
                st.dataframe(
                    kpi_df.style.format({
                        "ME": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}",
                        "Adherence (%)": "{:.2f}", "Data Availability (%)": "{:.2f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                # Summary bar chart
                st.markdown("#### KPI Summary Chart")
                fig_kpi = plot_kpi_summary(kpi_df)
                st.pyplot(fig_kpi)
                plt.close(fig_kpi)

                # Adherence timeline plots
                if twin_available:
                    st.markdown("#### Adherence Timeline Plots")
                    adh_cols = [c for c in kpi_cols if "twin_" + c in kpi_src.columns]
                    figs_adh = plot_adherence_timeline(
                        kpi_src, adh_cols, band_pct=band_pct
                    )
                    for col_name, fig_adh in figs_adh.items():
                        with st.expander(f"📈 {col_name}", expanded=False):
                            st.pyplot(fig_adh)
                        plt.close(fig_adh)

                # Outlier summary (if available)
                if st.session_state["dt_outlier_summary"] is not None:
                    st.markdown("#### Outlier Summary")
                    st.dataframe(
                        st.session_state["dt_outlier_summary"],
                        use_container_width=True, hide_index=True,
                    )

                # Export
                st.markdown("---")
                st.markdown("#### Export Results")
                raw = st.session_state["dt_raw_df"]
                clean = st.session_state["dt_clean_df"] or raw
                filt = st.session_state["dt_filtered_df"] or clean
                rec = st.session_state["dt_rec_df"] or filt
                out_summary = (st.session_state["dt_outlier_summary"]
                               if st.session_state["dt_outlier_summary"] is not None
                               else pd.DataFrame())

                excel_bytes = export_results_excel(raw, clean, filt, rec, kpi_df, out_summary)
                st.download_button(
                    "⬇️ Download Full Report (Excel)",
                    data=excel_bytes,
                    file_name="digital_twin_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dt_dl_report",
                )
