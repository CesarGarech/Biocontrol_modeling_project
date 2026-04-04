"""
SIGNAL PROCESSING
Generates synthetic SCADA data with noise and outliers,
then detects outliers (IQR method) and applies moving-average smoothing.
"""
import numpy as np
import pandas as pd
import config


def generate_raw_data():
    """
    Step 1: Generate synthetic SCADA readings for the distillation column
    and inject anomalies to simulate real instrument failures.
    """
    print("1. Generating synthetic SCADA data and injecting anomalies...")
    np.random.seed(config.SEED)

    timestamp = pd.date_range(
        start="2026-03-16 08:00", periods=config.N_POINTS, freq="h"
    )

    # Sinusoidal operational trend on the feed (± 500 kg/h)
    trend = np.sin(np.linspace(0, np.pi, config.N_POINTS)) * 500
    feed_real = config.FLOW_FEED_BASE + trend

    df = pd.DataFrame({"Timestamp": timestamp})

    # --- Mass flow readings with Gaussian noise ---
    df["F_feed_raw"] = np.random.normal(feed_real, config.SIGMA_FEED)
    df["F_top_raw"] = np.random.normal(
        feed_real * config.SPLIT_TOP, config.SIGMA_TOP
    )
    df["F_bottom_raw"] = np.random.normal(
        feed_real * config.SPLIT_BOTTOM, config.SIGMA_BOTTOM
    )

    # --- Energy duty readings with Gaussian noise ---
    df["Q_cond_raw"] = np.random.normal(config.Q_COND_BASE, config.SIGMA_Q_COND,
                                         config.N_POINTS)
    df["Q_reb_raw"] = np.random.normal(config.Q_REB_BASE, config.SIGMA_Q_REB,
                                        config.N_POINTS)

    # --- Inject outliers (instrument spikes / drops) ---
    # Feed spikes
    df.loc[[12, 48, 85], "F_feed_raw"] *= 1.8
    # Distillate drops (sensor failure)
    df.loc[[22, 67], "F_top_raw"] *= 0.3
    # Bottoms spikes
    df.loc[[55, 92], "F_bottom_raw"] *= 1.9
    # Energy spikes
    df.loc[[30], "Q_cond_raw"] *= 1.6
    df.loc[[75], "Q_reb_raw"] *= 0.4

    return df


def clean_and_filter(df):
    """
    Step 2: Detect outliers using the IQR method, replace with interpolation,
    then smooth using a centered moving average.
    """
    print("2. Detecting outliers (IQR method) and applying moving-average filter...")
    df_clean = df.copy()

    raw_columns = [
        "F_feed_raw", "F_top_raw", "F_bottom_raw",
        "Q_cond_raw", "Q_reb_raw",
    ]

    for col in raw_columns:
        # --- A. IQR-based outlier detection ---
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)

        # Save flag for plotting
        df_clean[f"{col}_outlier"] = outliers

        # Replace outliers with NaN and interpolate
        df_clean.loc[outliers, col] = np.nan
        df_clean[col] = df_clean[col].interpolate(method="linear")

        # --- B. Moving-average smoothing ---
        filtered_col = col.replace("_raw", "_filtered")
        df_clean[filtered_col] = (
            df_clean[col]
            .rolling(window=config.WINDOW_SIZE, center=True)
            .mean()
        )

    return df_clean.dropna().reset_index(drop=True)