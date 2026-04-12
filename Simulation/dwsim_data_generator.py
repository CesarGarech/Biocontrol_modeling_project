"""
DWSIM DATA GENERATOR
Uses DWSIMInterface to run live DWSIM simulations and produce DataFrames
compatible with the existing Digital Twin pipeline.
"""
import logging
import os
import platform
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Import DWSIMInterface at module level so it can be patched in tests.
# On hosts without pythonnet the import will succeed because dwsim_interface
# itself only imports clr inside __init__; DWSIMInterfaceError is always
# available.
try:
    from dwsim_interface import DWSIMInterface, DWSIMInterfaceError  # noqa: F401
except Exception:  # pragma: no cover — only happens in unusual environments
    DWSIMInterface = None  # type: ignore[assignment,misc]
    DWSIMInterfaceError = RuntimeError  # type: ignore[assignment,misc]

# Conversion constants
KG_S_TO_KG_H = 3600.0          # kg/s  → kg/h
W_TO_KW = 1e-3                  # W     → kW
MOL_S_TO_KMOL_H = 3.6          # mol/s → kmol/h


def validate_dwsim_installation() -> Tuple[bool, str]:
    """
    Check that DWSIM is installed and the simulation file exists.

    Returns
    -------
    (is_valid, message) : (bool, str)
        ``is_valid`` is ``True`` when all checks pass.
        ``message`` provides a human-readable status summary.
    """
    issues = []

    # 1. Platform check
    if platform.system() != "Windows":
        issues.append(
            f"DWSIM Automation is primarily supported on Windows "
            f"(current platform: {platform.system()}). "
            "Mono-based operation may be possible but is untested."
        )

    # 2. Installation directory
    if not os.path.isdir(config.DWSIM_INSTALL_PATH):
        issues.append(
            f"DWSIM installation directory not found: "
            f"{config.DWSIM_INSTALL_PATH!r}. "
            "Update DWSIM_INSTALL_PATH in Simulation/config.py."
        )

    # 3. Automation DLL
    dll_path = config.DWSIM_DLL_PATH
    if os.path.isdir(config.DWSIM_INSTALL_PATH) and not os.path.isfile(dll_path):
        issues.append(
            f"DWSIM.Automation.dll not found at {dll_path!r}. "
            "Ensure DWSIM 8+ is installed."
        )

    # 4. Simulation file
    if not os.path.isfile(config.SIMULATION_FILE):
        issues.append(
            f"Simulation file not found: {config.SIMULATION_FILE!r}."
        )

    # 5. pythonnet
    try:
        import clr  # noqa: F401
    except ImportError:
        issues.append(
            "pythonnet is not installed. "
            "Install it with: pip install pythonnet>=3.0.0"
        )

    if issues:
        return False, " | ".join(issues)

    return True, (
        f"DWSIM installation OK at {config.DWSIM_INSTALL_PATH!r}, "
        f"simulation file found: {config.SIMULATION_FILE!r}."
    )


def generate_dwsim_data(
    n_points: int,
    perturbations: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run *n_points* DWSIM simulations with varying feed conditions and return
    a DataFrame with the same columns as
    :func:`data_processor.generate_raw_data`.

    Each simulation point:

    1. Sets the feed mass-flow to *base + perturbation[i]*.
    2. Calls ``CalculateFlowsheet``.
    3. Extracts stream and equipment properties.
    4. Adds Gaussian sensor noise identical to the synthetic pipeline.

    Parameters
    ----------
    n_points : int
        Number of simulation points (time steps) to generate.
    perturbations : array-like of float, optional
        Feed-flow perturbations in kg/h for each point.  Length must equal
        *n_points*.  If ``None`` a sinusoidal pattern of
        ±:data:`config.PERTURBATION_AMPLITUDE` kg/h is used.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Timestamp``, ``F_feed_raw``, ``F_top_raw``,
        ``F_bottom_raw``, ``Q_cond_raw``, ``Q_reb_raw``
        (identical to ``data_processor.generate_raw_data()`` output).

    Raises
    ------
    RuntimeError
        If DWSIM validation fails and no fallback data can be produced.
    """
    np.random.seed(config.SEED)

    if perturbations is None:
        perturbations = (
            np.sin(np.linspace(0, np.pi, n_points))
            * config.PERTURBATION_AMPLITUDE
        )
    perturbations = np.asarray(perturbations, dtype=float)
    if len(perturbations) != n_points:
        raise ValueError(
            f"perturbations length ({len(perturbations)}) must equal "
            f"n_points ({n_points})."
        )

    timestamp = pd.date_range(
        start="2026-03-16 08:00", periods=n_points, freq="h"
    )

    records = []
    logger.info(
        "Starting DWSIM data generation: %d simulation points.", n_points
    )

    with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(config.SIMULATION_FILE)

        for i, delta in enumerate(perturbations):
            target_flow_kg_h = config.FLOW_FEED_BASE + delta
            # DWSIM uses SI units (kg/s) internally
            target_flow_kg_s = target_flow_kg_h / KG_S_TO_KG_H
            try:
                dwsim.set_stream_property(
                    config.TAG_FEED, "MassFlow", target_flow_kg_s
                )
                dwsim.run_simulation()

                # ---- Extract stream properties --------------------------------
                feed_flow = (
                    dwsim.get_stream_property(config.TAG_FEED, "MassFlow")
                    * KG_S_TO_KG_H
                )
                top_flow = (
                    dwsim.get_stream_property(config.TAG_TOP, "MassFlow")
                    * KG_S_TO_KG_H
                )
                bottom_flow = (
                    dwsim.get_stream_property(config.TAG_BOTTOM, "MassFlow")
                    * KG_S_TO_KG_H
                )

                # ---- Extract equipment duties --------------------------------
                q_cond = abs(
                    dwsim.get_equipment_property(
                        config.TAG_R_COND, "DutyCondenser"
                    )
                ) * W_TO_KW
                q_reb = abs(
                    dwsim.get_equipment_property(
                        config.TAG_Q_REB, "DutyReboiler"
                    )
                ) * W_TO_KW

            except DWSIMInterfaceError as exc:
                logger.warning(
                    "Point %d/%d failed (%s). Using base values.", i + 1, n_points, exc
                )
                feed_flow = config.FLOW_FEED_BASE + delta
                top_flow = feed_flow * config.SPLIT_TOP
                bottom_flow = feed_flow * config.SPLIT_BOTTOM
                q_cond = config.Q_COND_BASE
                q_reb = config.Q_REB_BASE

            # ---- Add Gaussian sensor noise ----------------------------------
            records.append(
                {
                    "Timestamp": timestamp[i],
                    "F_feed_raw": float(
                        np.random.normal(feed_flow, config.SIGMA_FEED)
                    ),
                    "F_top_raw": float(
                        np.random.normal(top_flow, config.SIGMA_TOP)
                    ),
                    "F_bottom_raw": float(
                        np.random.normal(bottom_flow, config.SIGMA_BOTTOM)
                    ),
                    "Q_cond_raw": float(
                        np.random.normal(q_cond, config.SIGMA_Q_COND)
                    ),
                    "Q_reb_raw": float(
                        np.random.normal(q_reb, config.SIGMA_Q_REB)
                    ),
                }
            )

            if (i + 1) % 10 == 0 or (i + 1) == n_points:
                logger.info("  Progress: %d / %d points", i + 1, n_points)

    df = pd.DataFrame(records)
    logger.info("DWSIM data generation complete (%d rows).", len(df))
    return df
