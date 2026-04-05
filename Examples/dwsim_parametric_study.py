"""
DWSIM Parametric Study — Ethanol Distillation Column
=====================================================
Standalone example that demonstrates how to use DWSIMInterface to perform
parametric studies on an ethanol–water distillation column.

Covered scenarios
-----------------
1. Effect of reflux ratio on separation performance.
2. Effect of feed temperature and pressure on distillate purity.
3. Effect of feed composition on product yields.

All results are saved to CSV files and plotted when matplotlib is available.

Requirements
------------
* DWSIM 8+ installed (Windows).
* pythonnet ≥ 3.0.0  (``pip install pythonnet``).
* numpy, pandas, matplotlib  (``pip install numpy pandas matplotlib``).
* The simulation file ``Simulation/ethanol.dwxmz`` must exist.

Usage
-----
::

    cd <project_root>
    python Examples/dwsim_parametric_study.py

The script reads ``DWSIM_INSTALL_PATH`` and ``SIMULATION_FILE`` from
``Simulation/config.py``.  Adjust those values before running.
"""
import os
import sys

# ---------------------------------------------------------------------------
# Resolve paths so the script can be run from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")

if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import numpy as np
import pandas as pd

import config
from dwsim_interface import DWSIMInterface, DWSIMInterfaceError

# ---------------------------------------------------------------------------
# Unit-conversion constants
# ---------------------------------------------------------------------------
_MOLS_TO_KMOLH = 3.6    # mol/s → kmol/h

# ---------------------------------------------------------------------------
# Optional matplotlib import
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
    _HAS_MPLOT = True
except ImportError:
    _HAS_MPLOT = False
    print("[INFO] matplotlib not found — plots will be skipped.")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "Output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. Reflux-ratio study
# ===========================================================================

def study_reflux_ratio(
    reflux_ratios=None,
    output_csv=None,
):
    """
    Vary the reflux ratio of the shortcut column and record distillate purity
    and reboiler duty for each value.

    Parameters
    ----------
    reflux_ratios : list of float, optional
        Reflux-ratio values to test.  Defaults to the range defined by
        ``config.REFLUX_RATIO_RANGE`` with 6 evenly-spaced points.
    output_csv : str, optional
        Path for the output CSV file.  Defaults to
        ``Output/study_reflux_ratio.csv``.

    Returns
    -------
    pandas.DataFrame
    """
    if reflux_ratios is None:
        lo, hi = config.REFLUX_RATIO_RANGE
        reflux_ratios = np.linspace(lo, hi, 6).tolist()

    if output_csv is None:
        output_csv = os.path.join(_OUTPUT_DIR, "study_reflux_ratio.csv")

    results = []

    print("\n=== Study 1: Effect of Reflux Ratio ===")
    print(f"  Reflux ratios: {[round(r, 2) for r in reflux_ratios]}")

    with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(config.SIMULATION_FILE)

        # Set default feed and column parameters first
        dwsim.set_stream_conditions(
            config.TAG_FEED,
            **config.DEFAULT_FEED_CONDITIONS,
        )
        dwsim.set_column_parameters(
            config.TAG_COLUMN,
            light_key=config.DEFAULT_COLUMN_PARAMETERS["light_key"],
            heavy_key=config.DEFAULT_COLUMN_PARAMETERS["heavy_key"],
            lk_bottoms=config.DEFAULT_COLUMN_PARAMETERS["lk_bottoms"],
            hk_distillate=config.DEFAULT_COLUMN_PARAMETERS["hk_distillate"],
        )

        for rr in reflux_ratios:
            try:
                dwsim.set_column_reflux_ratio(config.TAG_COLUMN, rr)
                dwsim.run_simulation()

                top_ethanol = dwsim.get_stream_property(
                    config.TAG_TOP, "MoleFraction", component="Ethanol"
                )
                top_molar_flow = (
                    dwsim.get_stream_property(config.TAG_TOP, "MolarFlow")
                    * _MOLS_TO_KMOLH  # mol/s → kmol/h
                )
                q_reb = (
                    abs(dwsim.get_equipment_property(
                        config.TAG_COLUMN, "DutyReboiler"
                    )) / 1000  # W → kW
                )

                results.append({
                    "Reflux_Ratio": rr,
                    "Top_Ethanol_MoleFrac": top_ethanol,
                    "Top_MolarFlow_kmolh": top_molar_flow,
                    "Reboiler_Duty_kW": q_reb,
                })
                print(
                    f"  RR={rr:.2f}  →  purity={top_ethanol*100:.2f}%  "
                    f"duty={q_reb:.1f} kW"
                )

            except DWSIMInterfaceError as exc:
                print(f"  [WARN] RR={rr:.2f} failed: {exc}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Results saved to: {output_csv}")

    if _HAS_MPLOT and not df.empty:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(df["Reflux_Ratio"], df["Top_Ethanol_MoleFrac"] * 100,
                 "b-o", label="Ethanol Purity")
        ax1.set_xlabel("Reflux Ratio")
        ax1.set_ylabel("Ethanol Purity (mol%)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot(df["Reflux_Ratio"], df["Reboiler_Duty_kW"],
                 "r-s", label="Reboiler Duty")
        ax2.set_ylabel("Reboiler Duty (kW)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        plt.title("Effect of Reflux Ratio on Separation Performance")
        fig.tight_layout()
        plot_path = output_csv.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Plot saved to: {plot_path}")

    return df


# ===========================================================================
# 2. Feed-temperature and feed-pressure study
# ===========================================================================

def study_feed_conditions(
    temperatures=None,
    pressures=None,
    output_csv=None,
):
    """
    Vary feed temperature and pressure and record distillate ethanol purity.

    Parameters
    ----------
    temperatures : list of float, optional
        Feed temperatures in °C.  Defaults to 4 points within
        ``config.FEED_TEMP_RANGE``.
    pressures : list of float, optional
        Feed pressures in bar.  Defaults to 3 points within
        ``config.FEED_PRESSURE_RANGE``.
    output_csv : str, optional
        Path for the output CSV.  Defaults to
        ``Output/study_feed_conditions.csv``.

    Returns
    -------
    pandas.DataFrame
    """
    if temperatures is None:
        lo, hi = config.FEED_TEMP_RANGE
        temperatures = np.linspace(lo, hi, 4).tolist()
    if pressures is None:
        lo, hi = config.FEED_PRESSURE_RANGE
        pressures = np.linspace(lo, hi, 3).tolist()
    if output_csv is None:
        output_csv = os.path.join(_OUTPUT_DIR, "study_feed_conditions.csv")

    results = []

    print("\n=== Study 2: Effect of Feed Temperature and Pressure ===")
    print(f"  Temperatures (°C): {[round(t, 1) for t in temperatures]}")
    print(f"  Pressures (bar):   {[round(p, 1) for p in pressures]}")

    with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(config.SIMULATION_FILE)

        # Apply default column parameters once
        dwsim.set_column_parameters(config.TAG_COLUMN,
                                    **config.DEFAULT_COLUMN_PARAMETERS)

        for temp in temperatures:
            for pres in pressures:
                try:
                    dwsim.set_stream_conditions(
                        config.TAG_FEED,
                        molar_flow=config.DEFAULT_FEED_CONDITIONS["molar_flow"],
                        temperature=temp,
                        pressure=pres,
                        composition=config.DEFAULT_FEED_CONDITIONS["composition"],
                    )
                    dwsim.run_simulation()

                    top_ethanol = dwsim.get_stream_property(
                        config.TAG_TOP, "MoleFraction", component="Ethanol"
                    )
                    results.append({
                        "Feed_Temp_C": temp,
                        "Feed_Pressure_bar": pres,
                        "Top_Ethanol_MoleFrac": top_ethanol,
                    })
                    print(
                        f"  T={temp:.1f}°C  P={pres:.1f} bar  "
                        f"→  purity={top_ethanol*100:.2f}%"
                    )

                except DWSIMInterfaceError as exc:
                    print(f"  [WARN] T={temp:.1f}, P={pres:.1f} failed: {exc}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Results saved to: {output_csv}")

    if _HAS_MPLOT and not df.empty and len(pressures) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        for pres in pressures:
            subset = df[df["Feed_Pressure_bar"].round(4) == round(pres, 4)]
            ax.plot(
                subset["Feed_Temp_C"],
                subset["Top_Ethanol_MoleFrac"] * 100,
                marker="o",
                label=f"P={pres:.1f} bar",
            )
        ax.set_xlabel("Feed Temperature (°C)")
        ax.set_ylabel("Ethanol Purity (mol%)")
        ax.set_title("Effect of Feed Conditions on Distillate Purity")
        ax.legend()
        fig.tight_layout()
        plot_path = output_csv.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Plot saved to: {plot_path}")

    return df


# ===========================================================================
# 3. Feed-composition study
# ===========================================================================

def study_feed_composition(
    ethanol_fractions=None,
    output_csv=None,
):
    """
    Vary the ethanol mole fraction in the feed and record separation results.

    Parameters
    ----------
    ethanol_fractions : list of float, optional
        Ethanol mole fractions to test.  Defaults to 5 points within
        ``config.FEED_ETHANOL_RANGE``.
    output_csv : str, optional
        Path for the output CSV.  Defaults to
        ``Output/study_feed_composition.csv``.

    Returns
    -------
    pandas.DataFrame
    """
    if ethanol_fractions is None:
        lo, hi = config.FEED_ETHANOL_RANGE
        ethanol_fractions = np.linspace(lo, hi, 5).tolist()
    if output_csv is None:
        output_csv = os.path.join(_OUTPUT_DIR, "study_feed_composition.csv")

    results = []

    print("\n=== Study 3: Effect of Feed Composition ===")
    print(f"  Ethanol fractions: {[round(x, 3) for x in ethanol_fractions]}")

    with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(config.SIMULATION_FILE)
        dwsim.set_column_parameters(config.TAG_COLUMN,
                                    **config.DEFAULT_COLUMN_PARAMETERS)

        for x_eth in ethanol_fractions:
            x_water = 1.0 - x_eth
            composition = {"Ethanol": x_eth, "Water": x_water}
            try:
                dwsim.set_stream_conditions(
                    config.TAG_FEED,
                    molar_flow=config.DEFAULT_FEED_CONDITIONS["molar_flow"],
                    temperature=config.DEFAULT_FEED_CONDITIONS["temperature"],
                    pressure=config.DEFAULT_FEED_CONDITIONS["pressure"],
                    composition=composition,
                )
                dwsim.run_simulation()

                top_ethanol = dwsim.get_stream_property(
                    config.TAG_TOP, "MoleFraction", component="Ethanol"
                )
                bottom_ethanol = dwsim.get_stream_property(
                    config.TAG_BOTTOM, "MoleFraction", component="Ethanol"
                )
                top_flow = (
                    dwsim.get_stream_property(config.TAG_TOP, "MolarFlow")
                    * _MOLS_TO_KMOLH  # mol/s → kmol/h
                )

                results.append({
                    "Feed_Ethanol_MoleFrac": x_eth,
                    "Top_Ethanol_MoleFrac": top_ethanol,
                    "Bottom_Ethanol_MoleFrac": bottom_ethanol,
                    "Top_MolarFlow_kmolh": top_flow,
                })
                print(
                    f"  Feed EtOH={x_eth:.3f}  →  top={top_ethanol*100:.2f}%  "
                    f"bottom={bottom_ethanol*100:.3f}%"
                )

            except DWSIMInterfaceError as exc:
                print(f"  [WARN] x_eth={x_eth:.3f} failed: {exc}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"  Results saved to: {output_csv}")

    if _HAS_MPLOT and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["Feed_Ethanol_MoleFrac"] * 100,
                df["Top_Ethanol_MoleFrac"] * 100, "b-o", label="Distillate")
        ax.plot(df["Feed_Ethanol_MoleFrac"] * 100,
                df["Bottom_Ethanol_MoleFrac"] * 100, "r-s", label="Bottoms")
        ax.set_xlabel("Feed Ethanol (mol%)")
        ax.set_ylabel("Ethanol in Product (mol%)")
        ax.set_title("Effect of Feed Composition on Separation")
        ax.legend()
        fig.tight_layout()
        plot_path = output_csv.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Plot saved to: {plot_path}")

    return df


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    print("=" * 60)
    print("  DWSIM Parametric Study — Ethanol Distillation")
    print("=" * 60)
    print(f"  DWSIM install path : {config.DWSIM_INSTALL_PATH}")
    print(f"  Simulation file    : {config.SIMULATION_FILE}")
    print(f"  Output directory   : {_OUTPUT_DIR}")

    df_rr = study_reflux_ratio()
    df_fc = study_feed_conditions()
    df_comp = study_feed_composition()

    print("\n" + "=" * 60)
    print("  PARAMETRIC STUDY COMPLETE")
    print("=" * 60)

    if not df_rr.empty:
        best_rr = df_rr.loc[
            df_rr["Top_Ethanol_MoleFrac"].idxmax(), "Reflux_Ratio"
        ]
        best_purity = df_rr["Top_Ethanol_MoleFrac"].max() * 100
        print(f"  Best reflux ratio: {best_rr:.2f}  "
              f"(purity {best_purity:.2f} mol%)")

    return df_rr, df_fc, df_comp


if __name__ == "__main__":
    main()
