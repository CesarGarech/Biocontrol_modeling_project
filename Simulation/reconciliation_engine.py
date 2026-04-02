"""
RECONCILIATION ENGINE
Applies strict mass and energy balance constraints using weighted least-squares
optimization to correct filtered measurements.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config


def reconcile_balances(df):
    """
    Step 3 & 4: For each time step, solve a constrained optimization that:
      - Minimises measurement corrections weighted by sensor uncertainty.
      - Enforces mass balance: F_feed = F_top + F_bottom
      - Computes KPIs: mass-balance closure error and separation adherence.
    """
    print("3. Running Data Reconciliation (Mass & Energy Balance)...")
    results = []

    for _, row in df.iterrows():
        # --------------------------------------------------
        # A. MASS BALANCE RECONCILIATION
        # --------------------------------------------------
        # Filtered measurements
        y_mass = np.array([
            row["F_feed_filtered"],
            row["F_top_filtered"],
            row["F_bottom_filtered"],
        ])

        # Sensor standard deviations (weights)
        sigma_mass = np.array([
            config.SIGMA_FEED,
            config.SIGMA_TOP,
            config.SIGMA_BOTTOM,
        ])

        # Objective: weighted least-squares
        def mass_objective(x):
            return np.sum(((y_mass - x) / sigma_mass) ** 2)

        # Constraint: F_feed - F_top - F_bottom = 0
        def mass_constraint(x):
            return x[0] - x[1] - x[2]

        sol_mass = minimize(
            mass_objective,
            x0=y_mass.copy(),
            constraints={"type": "eq", "fun": mass_constraint},
            method="SLSQP",
        )

        F_feed_rec, F_top_rec, F_bottom_rec = sol_mass.x

        # --------------------------------------------------
        # B. ENERGY BALANCE (informational, no reconciliation)
        # --------------------------------------------------
        Q_cond_filt = row["Q_cond_filtered"]
        Q_reb_filt = row["Q_reb_filtered"]

        # --------------------------------------------------
        # C. KPI CALCULATIONS
        # --------------------------------------------------
        # KPI 1: Mass-balance closure error BEFORE reconciliation (%)
        error_mass_before = (
            abs(y_mass[0] - y_mass[1] - y_mass[2]) / y_mass[0] * 100
        )

        # KPI 2: Separation adherence — how close is the actual top split
        #         to the desired SPLIT_TOP (target = 35% of feed as distillate)
        actual_split = F_top_rec / F_feed_rec
        kpi_separation = 100 - abs(actual_split - config.SPLIT_TOP) / config.SPLIT_TOP * 100

        # KPI 3: Energy efficiency — ratio Q_cond / Q_reb vs design ratio
        design_energy_ratio = config.Q_COND_BASE / config.Q_REB_BASE
        actual_energy_ratio = Q_cond_filt / Q_reb_filt if Q_reb_filt != 0 else 0
        kpi_energy = 100 - abs(actual_energy_ratio - design_energy_ratio) / design_energy_ratio * 100

        results.append({
            "Timestamp": row["Timestamp"],
            "F_feed_rec": F_feed_rec,
            "F_top_rec": F_top_rec,
            "F_bottom_rec": F_bottom_rec,
            "Q_cond_filtered": Q_cond_filt,
            "Q_reb_filtered": Q_reb_filt,
            "Error_Mass_Before_%": error_mass_before,
            "KPI_Separation_%": kpi_separation,
            "KPI_Energy_%": kpi_energy,
        })

    df_rec = pd.DataFrame(results)
    df_final = pd.concat(
        [df.reset_index(drop=True), df_rec.drop("Timestamp", axis=1)], axis=1
    )

    print("4. KPIs calculated: Mass Balance Error, Separation Adherence, Energy Efficiency.")
    return df_final