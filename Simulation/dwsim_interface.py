"""
DWSIM Automation Interface
===========================
Reusable wrapper module for the DWSIM Python/Automation interface.
All DWSIM COM/Automation logic is encapsulated here so that other
modules (Streamlit pages, standalone scripts) only call high-level
functions.

Import-safe: importing this module will NOT raise an exception even when
``pythonnet`` is not installed or DWSIM DLLs are not present on the system.
Check ``DWSIM_AVAILABLE`` before calling any function.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Availability flag — set to True only if pythonnet can be imported
# ---------------------------------------------------------------------------
DWSIM_AVAILABLE: bool = False
_DWSIM_INIT_ERROR: str = ""

try:
    import clr  # noqa: F401  (pythonnet)
    DWSIM_AVAILABLE = True
except ImportError:
    _DWSIM_INIT_ERROR = (
        "pythonnet is not installed. "
        "Install it with:  pip install pythonnet\n"
        "On Linux/Docker you also need the Mono runtime: "
        "apt-get install -y mono-complete"
    )

# ---------------------------------------------------------------------------
# Lazy configuration import (avoids circular dependency at import time)
# ---------------------------------------------------------------------------
def _get_config():
    """Return the config module from Simulation/, adding it to sys.path if needed."""
    _sim_dir = os.path.dirname(os.path.abspath(__file__))
    if _sim_dir not in sys.path:
        sys.path.insert(0, _sim_dir)
    import config as _cfg  # noqa: PLC0415
    return _cfg


# ---------------------------------------------------------------------------
# 1. init_dwsim
# ---------------------------------------------------------------------------
def init_dwsim(dwsim_path: str | None = None):
    """
    Initialise the DWSIM Automation interface.

    Resolution order for the DWSIM installation directory:
      1. ``dwsim_path`` argument
      2. ``DWSIM_PATH`` environment variable
      3. ``config.DWSIM_INSTALL_PATH``

    Parameters
    ----------
    dwsim_path : str, optional
        Absolute path to the DWSIM installation directory (the folder
        containing ``DWSIM.Automation.dll`` etc.).

    Returns
    -------
    interf : Automation3
        The DWSIM ``Automation3`` COM interop object.

    Raises
    ------
    RuntimeError
        If ``pythonnet`` is not installed.
    FileNotFoundError
        If no DWSIM DLLs are found at the resolved path.
    """
    if not DWSIM_AVAILABLE:
        raise RuntimeError(
            "DWSIM Automation is not available.\n" + _DWSIM_INIT_ERROR
        )

    cfg = _get_config()

    # Resolve installation path
    path = (
        dwsim_path
        or os.environ.get("DWSIM_PATH", "").strip()
        or cfg.DWSIM_INSTALL_PATH
    )
    path = path.rstrip("/\\")

    if not path:
        raise FileNotFoundError(
            "DWSIM installation path is not configured. "
            "Set the DWSIM_PATH environment variable or update "
            "DWSIM_INSTALL_PATH in Simulation/config.py."
        )

    # Verify that the directory contains DWSIM DLLs
    automation_dll = os.path.join(path, "DWSIM.Automation.dll")
    if not os.path.isfile(automation_dll):
        raise FileNotFoundError(
            f"DWSIM.Automation.dll not found in: {path}\n"
            "Check that DWSIM is installed and DWSIM_PATH points to the "
            "correct directory."
        )

    # Add the installation path to the .NET assembly search path
    import clr
    sys.path.insert(0, path)
    clr.AddReference(automation_dll)

    # Load additional assemblies (non-fatal if optional DLLs are absent)
    for dll_name in [
        "DWSIM.Interfaces",
        "DWSIM.Thermodynamics",
        "DWSIM.SharedClasses",
    ]:
        dll_path = os.path.join(path, dll_name + ".dll")
        if os.path.isfile(dll_path):
            try:
                clr.AddReference(dll_path)
            except Exception:
                pass

    # CapeOpen is optional
    capeopen_dll = os.path.join(path, "CapeOpen.dll")
    if os.path.isfile(capeopen_dll):
        try:
            clr.AddReference(capeopen_dll)
        except Exception:
            pass

    # Import and instantiate the Automation3 object
    from DWSIM.Automation import Automation3  # noqa: PLC0415
    interf = Automation3()
    return interf


# ---------------------------------------------------------------------------
# 2. load_simulation
# ---------------------------------------------------------------------------
def load_simulation(interf, sim_file: str | None = None):
    """
    Load a DWSIM flowsheet file.

    Parameters
    ----------
    interf : Automation3
        DWSIM automation object returned by ``init_dwsim()``.
    sim_file : str, optional
        Absolute path to the ``.dwxmz`` simulation file.
        Defaults to ``config.SIMULATION_FILE``.

    Returns
    -------
    sim : object
        Loaded DWSIM flowsheet (simulation object).

    Raises
    ------
    FileNotFoundError
        If the simulation file does not exist.
    """
    cfg = _get_config()
    path = sim_file or cfg.SIMULATION_FILE

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DWSIM simulation file not found: {path}\n"
            "Set the SIMULATION_FILE in Simulation/config.py or provide a path."
        )

    sim = interf.LoadFlowsheet(path)
    return sim


# ---------------------------------------------------------------------------
# 3. set_feed_conditions
# ---------------------------------------------------------------------------
def set_feed_conditions(
    sim,
    temperature_C: float,
    pressure_kPa: float,
    flow_kgh: float,
    x_ethanol: float,
    x_water: float,
    feed_tag: str | None = None,
) -> None:
    """
    Write feed conditions to the DWSIM feed stream.

    Parameters
    ----------
    sim : object
        Loaded DWSIM flowsheet.
    temperature_C : float
        Feed temperature in **°C** (converted to K internally).
    pressure_kPa : float
        Feed pressure in **kPa** (converted to Pa internally).
    flow_kgh : float
        Mass flow rate in **kg/h** (converted to kg/s internally).
    x_ethanol : float
        Ethanol mole fraction (0–1).
    x_water : float
        Water mole fraction (0–1).  ``x_ethanol + x_water`` should equal 1.
    feed_tag : str, optional
        DWSIM tag of the feed stream.  Defaults to ``config.TAG_FEED``.
    """
    cfg = _get_config()
    tag = feed_tag or cfg.TAG_FEED
    stream = sim.GetFlowsheetSimulationObject(tag)

    # Unit conversions
    T_K = temperature_C + 273.15
    P_Pa = pressure_kPa * 1_000.0
    F_kgs = flow_kgh / 3_600.0

    # DWSIM ISimulationObject property-bag API (works without concrete-type cast)
    stream.SetPropertyValue("temperature", T_K)
    stream.SetPropertyValue("pressure", P_Pa)
    stream.SetPropertyValue("massflow", F_kgs)

    # Molar composition — SetOverallComposition requires a .NET Double[].
    # Explicitly marshal via System.Array to avoid pythonnet marshalling issues.
    compounds = getattr(_get_config(), "DWSIM_COMPOUNDS", ["Ethanol", "Water"])
    fractions = [x_ethanol, x_water]
    _comp_set = False
    try:
        import System  # noqa: PLC0415
        fracs_net = System.Array[System.Double](fractions)
        stream.SetOverallComposition(fracs_net)
        _comp_set = True
    except Exception:
        pass
    if not _comp_set:
        # Fallback: plain Python list (works in some pythonnet versions)
        try:
            stream.SetOverallComposition(fractions)
            _comp_set = True
        except Exception:
            pass
    if not _comp_set:
        # Last-resort: per-compound property-bag entry
        for name, frac in zip(compounds, fractions):
            try:
                stream.SetPropertyValue(f"fraction[{name}]", frac)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 4. run_simulation
# ---------------------------------------------------------------------------
def run_simulation(interf, sim, timeout_seconds: int = 120):
    """
    Solve the DWSIM flowsheet.

    Parameters
    ----------
    interf : Automation3
        DWSIM automation object.
    sim : object
        Loaded DWSIM flowsheet.
    timeout_seconds : int, optional
        Solver timeout in seconds.  Defaults to ``config.DWSIM_TIMEOUT``
        if available, otherwise 120.

    Returns
    -------
    converged : bool
        ``True`` if the solver converged successfully.
    error_message : str
        Non-empty string describing any convergence issue; empty on success.
    """
    cfg = _get_config()
    timeout = timeout_seconds or getattr(cfg, "DWSIM_TIMEOUT", 120)

    interf.CalculateFlowsheet3(sim, timeout)

    # Inspect solver messages / errors
    errors = []
    try:
        for obj_name in sim.GetFlowsheetSimulationObjectsNames():
            obj = sim.GetFlowsheetSimulationObject(obj_name)
            try:
                errs = obj.GraphicObject.ErrorInfoList
                if errs and len(errs) > 0:
                    for err in errs:
                        errors.append(str(err))
            except Exception:
                pass
    except Exception:
        pass

    converged = len(errors) == 0
    error_message = "; ".join(errors) if errors else ""
    return converged, error_message


# ---------------------------------------------------------------------------
# 5. read_results
# ---------------------------------------------------------------------------
def read_results(sim, tags: dict | None = None) -> dict:
    """
    Read stream/equipment results from the solved flowsheet.

    Parameters
    ----------
    sim : object
        Solved DWSIM flowsheet.
    tags : dict, optional
        Mapping of role → DWSIM tag string.  Supported keys:
        ``"feed"``, ``"top"``, ``"bottom"``, ``"r_cond"``, ``"q_reb"``.
        Defaults to the tags defined in ``config.py``.

    Returns
    -------
    results : dict
        Dictionary with the following structure::

            {
                "feed": {
                    "mass_flow_kgh": float,   # kg/h
                    "molar_flow_kmolh": float, # kmol/h
                    "temperature_C": float,    # °C
                    "pressure_kPa": float,     # kPa
                    "composition": list,       # mole fractions
                },
                "top": { ... },
                "bottom": { ... },
                "r_cond": {"energy_kw": float},  # kW
                "q_reb":  {"energy_kw": float},  # kW
            }
    """
    cfg = _get_config()

    # Resolve tags
    resolved = {
        "feed":    (tags or {}).get("feed",    cfg.TAG_FEED),
        "top":     (tags or {}).get("top",     cfg.TAG_TOP),
        "bottom":  (tags or {}).get("bottom",  cfg.TAG_BOTTOM),
        "r_cond":  (tags or {}).get("r_cond",  cfg.TAG_R_COND),
        "q_reb":   (tags or {}).get("q_reb",   cfg.TAG_Q_REB),
    }

    results: dict = {}

    def _pv(obj, prop):
        """Safely read a .NET property value as float; returns None when unavailable."""
        try:
            v = obj.GetPropertyValue(prop)
            return float(v) if v is not None else None
        except Exception:
            return None

    # Material streams
    for role in ("feed", "top", "bottom"):
        tag = resolved[role]
        obj = sim.GetFlowsheetSimulationObject(tag)

        # DWSIM ISimulationObject property-bag API (no concrete-type cast needed).
        # GetPropertyValue returns SI-unit values; None if the stream is unsolved.
        mass_flow_kgs   = _pv(obj, "massflow")    or 0.0   # kg/s
        molar_flow_mols = _pv(obj, "molarflow")   or 0.0   # mol/s
        temperature_K   = _pv(obj, "temperature")          # K  — keep None to detect unsolved
        pressure_Pa     = _pv(obj, "pressure")    or 0.0   # Pa

        # Molar composition — GetOverallComposition returns a float array
        composition = []
        try:
            composition = [float(v) for v in obj.GetOverallComposition()]
        except Exception:
            # Fallback: read per-compound fractions via property bag
            compounds = getattr(_get_config(), "DWSIM_COMPOUNDS", ["Ethanol", "Water"])
            for name in compounds:
                fv = _pv(obj, f"fraction[{name}]")
                composition.append(fv if fv is not None else 0.0)

        # temperature_K=None means the stream is unsolved; surface -273.15°C so it is
        # immediately obvious rather than the misleading 0°C produced by a 273.15 default.
        t_K = temperature_K if temperature_K is not None else 0.0

        results[role] = {
            "mass_flow_kgh":    mass_flow_kgs * 3_600.0,
            "molar_flow_kmolh": molar_flow_mols * 3_600.0 / 1_000.0,
            "temperature_C":    t_K - 273.15,
            "pressure_kPa":     pressure_Pa / 1_000.0,
            "composition":      composition,
        }

    # Energy streams — heatflow property (W) via the ISimulationObject property bag
    for role in ("r_cond", "q_reb"):
        tag = resolved[role]
        obj = sim.GetFlowsheetSimulationObject(tag)
        energy_W = _pv(obj, "heatflow") or 0.0
        results[role] = {
            "energy_kw": energy_W / 1_000.0,
        }

    return results


# ---------------------------------------------------------------------------
# 6. close_simulation
# ---------------------------------------------------------------------------
def close_simulation(interf, sim) -> None:
    """
    Release DWSIM resources.

    Parameters
    ----------
    interf : Automation3
        DWSIM automation object.
    sim : object
        DWSIM flowsheet to release (may be None).
    """
    try:
        if sim is not None:
            interf.ReleaseResources()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# __main__ — standalone demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("DWSIM Interface — standalone demo")
    print(f"  DWSIM_AVAILABLE = {DWSIM_AVAILABLE}")

    if not DWSIM_AVAILABLE:
        print(f"  Reason: {_DWSIM_INIT_ERROR}")
        sys.exit(1)

    cfg = _get_config()
    print(f"  DWSIM path   : {cfg.DWSIM_INSTALL_PATH}")
    print(f"  Flowsheet    : {cfg.SIMULATION_FILE}")

    # Initialise
    print("\n[1] Initialising DWSIM automation...")
    interf = init_dwsim()

    # Load flowsheet
    print("[2] Loading flowsheet...")
    sim = load_simulation(interf)

    # Set demo feed conditions
    print("[3] Setting feed conditions...")
    set_feed_conditions(
        sim,
        temperature_C=30.0,
        pressure_kPa=100.0,
        flow_kgh=10_000.0,
        x_ethanol=0.40,
        x_water=0.60,
    )

    # Run simulation
    print("[4] Running simulation...")
    converged, err = run_simulation(interf, sim)
    print(f"    Converged: {converged}")
    if err:
        print(f"    Errors   : {err}")

    # Read results
    print("[5] Reading results...")
    res = read_results(sim)
    for stream, data in res.items():
        print(f"  {stream}: {data}")

    # Clean up
    close_simulation(interf, sim)
    print("\nDone.")
