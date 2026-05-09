"""
DWSIM INTERFACE
Provides bidirectional communication between Python and DWSIM via pythonnet.
Wraps DWSIM.Automation to load, run and query flowsheet objects.

Changes vs previous version
-----------------------------
* get_equipment_property: added "CondenserPressure", "ReboilerPressure",
  "MinimumRefluxRatio", "LKMoleFractionInBottoms", "HKMoleFractionInDistillate".
* set_equipment_property: removed (was using arbitrary string names not present
  in the DWSIM API); all column writes now go through set_column_parameters().
* _get_shortcut_column: new helper that casts the flowsheet object to the
  concrete ShortcutColumn type before reading/writing properties.
"""
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Unit-conversion constants
_KMOLH_TO_MOLS = 1000.0 / 3600.0   # kmol/h → mol/s
_CELSIUS_TO_K  = 273.15             # °C offset → K
_BAR_TO_PA     = 1e5                # bar → Pa
_PA_TO_BAR     = 1e-5               # Pa → bar


class DWSIMInterfaceError(Exception):
    """Raised when the DWSIM interface encounters a fatal error."""


class DWSIMInterface:
    """
    Context-manager–friendly wrapper around the DWSIM Automation API.

    Usage::

        with DWSIMInterface(dwsim_install_path) as dwsim:
            dwsim.load_simulation("ethanol.dwxmz")
            dwsim.set_stream_conditions("Feed", molar_flow=100, ...)
            dwsim.set_column_parameters("SCOL-1", light_key="Ethanol", ...)
            dwsim.run_simulation()
            feed_flow = dwsim.get_stream_property("Feed", "MassFlow")
    """

    def __init__(self, install_path: str) -> None:
        self.install_path   = install_path
        self._automation    = None
        self._flowsheet     = None
        self._loaded_file: Optional[str] = None
        self._initialised   = False
        self._compounds_cache: Optional[List[str]] = None
        self._init_dotnet()

    # ──────────────────────────────────────────────────────────────────────────
    # .NET / DWSIM initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_dotnet(self) -> None:
        """Load the .NET runtime and all required DWSIM assemblies via pythonnet."""
        if platform.system() != "Windows":
            logger.warning(
                "DWSIM Automation is primarily supported on Windows. "
                "Attempting on %s — this may not work.",
                platform.system(),
            )

        try:
            import clr  # noqa: F401
        except ImportError as exc:
            raise DWSIMInterfaceError(
                "pythonnet is not installed.  "
                "Install it with: pip install pythonnet>=3.0.0"
            ) from exc

        if not os.path.isdir(self.install_path):
            raise DWSIMInterfaceError(
                f"DWSIM installation directory not found: {self.install_path!r}. "
                "Update DWSIM_INSTALL_PATH in Simulation/config.py."
            )

        if platform.system() == "Windows":
            try:
                import pythoncom  # type: ignore[import]
                pythoncom.CoInitialize()
                logger.debug("COM initialised.")
            except ImportError:
                logger.debug("pythoncom not available — skipping CoInitialize.")
            except Exception as exc:
                logger.debug("CoInitialize() failed (non-fatal): %s", exc)

        try:
            import clr as _clr

            _clr.AddReference("System.Runtime")
            _clr.AddReference("System.IO")

            if self.install_path not in sys.path:
                sys.path.insert(0, self.install_path)

            _DWSIM_DLLS = [
                "CapeOpen.dll",
                "DWSIM.Automation.dll",
                "DWSIM.Interfaces.dll",
                "DWSIM.GlobalSettings.dll",
                "DWSIM.SharedClasses.dll",
                "DWSIM.Thermodynamics.dll",
                "DWSIM.UnitOperations.dll",
                "DWSIM.Inspector.dll",
                "System.Buffers.dll",
                "DWSIM.Thermodynamics.ThermoC.dll",
            ]
            for _dll in _DWSIM_DLLS:
                _dll_path = os.path.join(self.install_path, _dll)
                try:
                    if os.path.isfile(_dll_path):
                        _clr.AddReference(_dll_path)
                        logger.debug("Loaded %s", _dll)
                    else:
                        logger.debug("DLL not found, skipping: %s", _dll_path)
                except Exception as exc:
                    logger.debug("AddReference(%s) skipped: %s", _dll, exc)

            try:
                from System.IO import Directory  # type: ignore[import]
                Directory.SetCurrentDirectory(self.install_path)
                logger.debug("Set .NET CWD to: %s", self.install_path)
            except Exception as exc:
                logger.debug("SetCurrentDirectory failed (non-fatal): %s", exc)

        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Failed to load DWSIM assemblies from {self.install_path!r}: {exc}"
            ) from exc

        try:
            from DWSIM.Automation import Automation3  # type: ignore[import]
            self._automation  = Automation3()
            self._initialised = True
            logger.info("DWSIM Automation3 initialised.")
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not create DWSIM Automation3 object: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def load_simulation(self, file_path: str) -> Any:
        """Load a ``.dwxmz`` file and return the flowsheet object."""
        if not self._initialised:
            raise DWSIMInterfaceError("DWSIM interface is not initialised.")
        if not os.path.isfile(file_path):
            raise DWSIMInterfaceError(f"Simulation file not found: {file_path!r}")
        try:
            self._flowsheet       = self._automation.LoadFlowsheet(file_path)
            self._loaded_file     = file_path
            self._compounds_cache = None
            logger.info("Loaded: %s", file_path)
            return self._flowsheet
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Failed to load {file_path!r}: {exc}"
            ) from exc

    def run_simulation(self) -> None:
        """Execute the DWSIM calculation for the loaded flowsheet."""
        self._require_loaded()
        try:
            error = self._automation.CalculateFlowsheet2(self._flowsheet)
            if error is not None:
                raise DWSIMInterfaceError(
                    f"Flowsheet calculation returned an error: {error}"
                )
            logger.info("Simulation completed.")
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(f"Simulation run failed: {exc}") from exc

    def close_simulation(self) -> None:
        """Release DWSIM resources for the loaded simulation."""
        if self._flowsheet is not None:
            try:
                self._automation.CloseFlowsheet(self._flowsheet)
                logger.info("Closed: %s", self._loaded_file)
            except Exception as exc:
                logger.warning("Error closing simulation: %s", exc)
            finally:
                self._flowsheet   = None
                self._loaded_file = None

    # ──────────────────────────────────────────────────────────────────────────
    # Stream property readers
    # ──────────────────────────────────────────────────────────────────────────

    def get_stream_property(
        self,
        stream_name: str,
        property_name: str,
        component: Optional[str] = None,
    ) -> float:
        """
        Extract a scalar property from a material stream.

        Parameters
        ----------
        stream_name   : stream tag in the DWSIM flowsheet.
        property_name : one of
            ``"MassFlow"``      → kg/s (SI)
            ``"MolarFlow"``     → mol/s (SI)
            ``"Temperature"``   → K (SI)
            ``"Pressure"``      → Pa (SI)
            ``"MassFraction"``  → requires *component*
            ``"MoleFraction"``  → requires *component*
        component : compound name for fraction properties.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        try:
            if property_name == "MassFlow":
                return float(stream.Phases[0].Properties.massflow)
            if property_name == "MolarFlow":
                return float(stream.Phases[0].Properties.molarflow)
            if property_name == "Temperature":
                return float(stream.Phases[0].Properties.temperature)
            if property_name == "Pressure":
                return float(stream.Phases[0].Properties.pressure)
            if property_name == "MassFraction":
                if component is None:
                    raise DWSIMInterfaceError(
                        "'component' required for MassFraction."
                    )
                return float(
                    stream.Phases[0].Compounds[component].MassFraction
                )
            if property_name == "MoleFraction":
                if component is None:
                    raise DWSIMInterfaceError(
                        "'component' required for MoleFraction."
                    )
                return float(
                    stream.Phases[0].Compounds[component].MoleFraction
                )
            raise DWSIMInterfaceError(
                f"Unknown stream property: {property_name!r}. "
                "Supported: MassFlow, MolarFlow, Temperature, Pressure, "
                "MassFraction, MoleFraction."
            )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not read {property_name!r} from stream "
                f"{stream_name!r}: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Equipment property reader
    # ──────────────────────────────────────────────────────────────────────────

    def get_equipment_property(
        self, equipment_name: str, property_name: str
    ) -> float:
        """
        Extract a scalar property from a unit operation.

        Supported property names
        -------------------------
        Generic energy streams
            ``"Duty"``                   → W (sign preserved)

        ShortcutColumn (distillation)
            ``"DutyCondenser"``          → W
            ``"DutyReboiler"``           → W
            ``"RefluxRatio"``            → dimensionless
            ``"MinimumRefluxRatio"``     → dimensionless
            ``"NumberOfStages"``         → dimensionless
            ``"LKMoleFractionInBottoms"``    → mol/mol
            ``"HKMoleFractionInDistillate"`` → mol/mol
            ``"CondenserPressure"``      → Pa (returned as Pa)
            ``"ReboilerPressure"``       → Pa (returned as Pa)
        """
        self._require_loaded()
        obj = self._get_object(equipment_name, "equipment")

        try:
            # ── Generic energy stream ─────────────────────────────────────────
            if property_name == "Duty":
                for attr in ("EnergyFlow", "get_EnergyFlow"):
                    try:
                        val = getattr(obj, attr)
                        return float(val() if callable(val) else val)
                    except Exception:
                        pass
                raise DWSIMInterfaceError(
                    f"Could not read EnergyFlow from {equipment_name!r}."
                )

            # ── ShortcutColumn properties ─────────────────────────────────────
            if property_name == "DutyCondenser":
                return float(obj.CondenserDuty)
            if property_name == "DutyReboiler":
                return float(obj.ReboilerDuty)
            if property_name == "RefluxRatio":
                return float(obj.m_refluxratio)
            if property_name == "MinimumRefluxRatio":
                return float(obj.m_Rmin)
            if property_name == "NumberOfStages":
                return float(obj.m_N)
            if property_name == "MinimumNumberOfStages":
                return float(obj.m_Nmin)
            if property_name == "FeedStage":
                return float(obj.ofs)
            if property_name == "EstimatedHeight":
                return float(obj.EstimatedHeight)
            if property_name == "LKMoleFractionInBottoms":
                return float(obj.LKMoleFractionInBottoms)
            if property_name == "HKMoleFractionInDistillate":
                return float(obj.HKMoleFractionInDistillate)
            if property_name == "CondenserPressure":
                # DWSIM stores as Pa; caller decides on unit conversion
                return float(obj.CondenserPressure)
            if property_name == "ReboilerPressure":
                return float(obj.ReboilerPressure)

            raise DWSIMInterfaceError(
                f"Unknown equipment property: {property_name!r}. "
                "Supported: Duty, DutyCondenser, DutyReboiler, RefluxRatio, "
                "MinimumRefluxRatio, NumberOfStages, MinimumNumberOfStages, FeedStage, "
                "LKMoleFractionInBottoms, HKMoleFractionInDistillate, "
                "CondenserPressure, ReboilerPressure, EstimatedHeight."
            )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not read {property_name!r} from "
                f"{equipment_name!r}: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Stream setters (user-friendly units)
    # ──────────────────────────────────────────────────────────────────────────

    def set_stream_molar_flow(
        self, stream_name: str, molar_flow_kmolh: float
    ) -> None:
        """Set molar flow in **kmol/h** (converted to mol/s internally)."""
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        mols = molar_flow_kmolh * _KMOLH_TO_MOLS
        try:
            stream.Phases[0].Properties.molarflow = mols
            logger.debug("Set %s.molarflow = %.6g mol/s", stream_name, mols)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set molar flow on {stream_name!r}: {exc}"
            ) from exc

    def set_stream_temperature(
        self, stream_name: str, temperature_celsius: float
    ) -> None:
        """Set temperature in **°C** (converted to K internally)."""
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        T_k = temperature_celsius + _CELSIUS_TO_K
        try:
            stream.Phases[0].Properties.temperature = T_k
            logger.debug("Set %s.temperature = %.2f K", stream_name, T_k)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set temperature on {stream_name!r}: {exc}"
            ) from exc

    def set_stream_pressure(
        self, stream_name: str, pressure_bar: float
    ) -> None:
        """Set pressure in **bar** (converted to Pa internally)."""
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        P_pa = pressure_bar * _BAR_TO_PA
        try:
            stream.Phases[0].Properties.pressure = P_pa
            logger.debug("Set %s.pressure = %.4g Pa", stream_name, P_pa)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set pressure on {stream_name!r}: {exc}"
            ) from exc

    def set_stream_composition(
        self, stream_name: str, composition_dict: Dict[str, float]
    ) -> None:
        """
        Set mole-fraction composition of a stream.

        Fractions are auto-normalised.  Only listed compounds are updated.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        normalised = self.validate_composition(composition_dict)
        try:
            for compound, fraction in normalised.items():
                stream.Phases[0].Compounds[compound].MoleFraction = fraction
                logger.debug(
                    "Set %s[%s].MoleFraction = %.6g", stream_name, compound, fraction
                )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set composition on {stream_name!r}: {exc}"
            ) from exc

    def set_stream_conditions(
        self,
        stream_name: str,
        molar_flow: Optional[float]          = None,
        temperature: Optional[float]         = None,
        pressure: Optional[float]            = None,
        composition: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Convenience: set any combination of feed stream conditions.

        Parameters
        ----------
        molar_flow  : kmol/h
        temperature : °C
        pressure    : bar
        composition : {compound: mole_fraction}
        """
        if molar_flow  is not None: self.set_stream_molar_flow(stream_name, molar_flow)
        if temperature is not None: self.set_stream_temperature(stream_name, temperature)
        if pressure    is not None: self.set_stream_pressure(stream_name, pressure)
        if composition is not None: self.set_stream_composition(stream_name, composition)

    # ──────────────────────────────────────────────────────────────────────────
    # Shortcut-column setters (individual)
    # ──────────────────────────────────────────────────────────────────────────

    def set_column_light_key(self, column_name: str, compound_name: str) -> None:
        """Set the LightKeyCompound of a shortcut column."""
        self._require_loaded()
        self._validate_compound_name(compound_name)
        col = self._get_shortcut_column(column_name)
        try:
            col.LightKeyCompound = compound_name
            logger.debug("Set %s.LightKeyCompound = %r", column_name, compound_name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set LightKeyCompound on {column_name!r}: {exc}"
            ) from exc

    def set_column_heavy_key(self, column_name: str, compound_name: str) -> None:
        """Set the HeavyKeyCompound of a shortcut column."""
        self._require_loaded()
        self._validate_compound_name(compound_name)
        col = self._get_shortcut_column(column_name)
        try:
            col.HeavyKeyCompound = compound_name
            logger.debug("Set %s.HeavyKeyCompound = %r", column_name, compound_name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set HeavyKeyCompound on {column_name!r}: {exc}"
            ) from exc

    def set_column_lk_fraction_bottoms(
        self, column_name: str, mole_fraction: float
    ) -> None:
        """
        Set the LK mole fraction in the bottoms stream.

        This is the **DWSIM ShortcutColumn separation spec** — it corresponds
        to the ``LKMoleFractionInBottoms`` property on the column object.
        """
        self._require_loaded()
        self._validate_mole_fraction(mole_fraction, "lk_bottoms")
        col = self._get_shortcut_column(column_name)
        try:
            col.LKMoleFractionInBottoms = mole_fraction
            logger.debug(
                "Set %s.LKMoleFractionInBottoms = %.4g", column_name, mole_fraction
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set LKMoleFractionInBottoms on {column_name!r}: {exc}"
            ) from exc

    def set_column_hk_fraction_distillate(
        self, column_name: str, mole_fraction: float
    ) -> None:
        """
        Set the HK mole fraction allowed in the distillate.

        Corresponds to ``HKMoleFractionInDistillate`` on the DWSIM column object.
        """
        self._require_loaded()
        self._validate_mole_fraction(mole_fraction, "hk_distillate")
        col = self._get_shortcut_column(column_name)
        try:
            col.HKMoleFractionInDistillate = mole_fraction
            logger.debug(
                "Set %s.HKMoleFractionInDistillate = %.4g",
                column_name, mole_fraction,
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set HKMoleFractionInDistillate on {column_name!r}: {exc}"
            ) from exc

    def set_column_reflux_ratio(
        self, column_name: str, reflux_ratio: float
    ) -> None:
        """
        Set the **actual** reflux ratio (L/D) on the shortcut column.

        Note: DWSIM accepts the actual R, not a R/R_min multiplier.
        Compute R = β × R_min before calling this method.
        """
        self._require_loaded()
        if reflux_ratio <= 0:
            raise DWSIMInterfaceError(
                f"reflux_ratio must be positive (got {reflux_ratio})."
            )
        col = self._get_shortcut_column(column_name)
        try:
            col.RefluxRatio = reflux_ratio
            logger.debug("Set %s.RefluxRatio = %.4g", column_name, reflux_ratio)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set RefluxRatio on {column_name!r}: {exc}"
            ) from exc

    def set_column_condenser_pressure(
        self, column_name: str, pressure_bar: float
    ) -> None:
        """Set the condenser pressure in **bar** (converted to Pa)."""
        self._require_loaded()
        col = self._get_shortcut_column(column_name)
        P_pa = pressure_bar * _BAR_TO_PA
        try:
            col.CondenserPressure = P_pa
            logger.debug(
                "Set %s.CondenserPressure = %.4g Pa", column_name, P_pa
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set CondenserPressure on {column_name!r}: {exc}"
            ) from exc

    def set_column_reboiler_pressure(
        self, column_name: str, pressure_bar: float
    ) -> None:
        """Set the reboiler pressure in **bar** (converted to Pa)."""
        self._require_loaded()
        col = self._get_shortcut_column(column_name)
        P_pa = pressure_bar * _BAR_TO_PA
        try:
            col.ReboilerPressure = P_pa
            logger.debug(
                "Set %s.ReboilerPressure = %.4g Pa", column_name, P_pa
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set ReboilerPressure on {column_name!r}: {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Shortcut-column convenience setter
    # ──────────────────────────────────────────────────────────────────────────

    def set_column_parameters(
        self,
        column_name: str,
        light_key: Optional[str]         = None,
        heavy_key: Optional[str]         = None,
        lk_bottoms: Optional[float]      = None,
        hk_distillate: Optional[float]   = None,
        reflux_ratio: Optional[float]    = None,
        condenser_pressure_bar: Optional[float] = None,
        reboiler_pressure_bar: Optional[float]  = None,
    ) -> None:
        """
        Set any combination of shortcut-column parameters in a single call.

        Parameters
        ----------
        column_name            : tag in the DWSIM flowsheet (e.g. ``"SCOL-1"``).
        light_key              : LK compound name (e.g. ``"Ethanol"``).
        heavy_key              : HK compound name (e.g. ``"Water"``).
        lk_bottoms             : LK mole fraction in bottoms (0 – 1).
        hk_distillate          : HK mole fraction in distillate (0 – 1).
        reflux_ratio           : Actual reflux ratio L/D (> 0).
        condenser_pressure_bar : Condenser operating pressure (bar).
        reboiler_pressure_bar  : Reboiler operating pressure (bar).

        Example
        -------
        ::

            dwsim.set_column_parameters(
                "SCOL-1",
                light_key="Ethanol",
                heavy_key="Water",
                lk_bottoms=0.02,
                hk_distillate=0.02,
                reflux_ratio=1.5,
                condenser_pressure_bar=1.013,
                reboiler_pressure_bar=1.10,
            )
        """
        if light_key               is not None:
            self.set_column_light_key(column_name, light_key)
        if heavy_key               is not None:
            self.set_column_heavy_key(column_name, heavy_key)
        if lk_bottoms              is not None:
            self.set_column_lk_fraction_bottoms(column_name, lk_bottoms)
        if hk_distillate           is not None:
            self.set_column_hk_fraction_distillate(column_name, hk_distillate)
        if reflux_ratio            is not None:
            self.set_column_reflux_ratio(column_name, reflux_ratio)
        if condenser_pressure_bar  is not None:
            self.set_column_condenser_pressure(column_name, condenser_pressure_bar)
        if reboiler_pressure_bar   is not None:
            self.set_column_reboiler_pressure(column_name, reboiler_pressure_bar)

    # ──────────────────────────────────────────────────────────────────────────
    # Compound / composition helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_available_compounds(self) -> List[str]:
        """Return compound names available in the loaded simulation (cached)."""
        self._require_loaded()
        if self._compounds_cache is not None:
            return self._compounds_cache
        try:
            self._compounds_cache = list(self._flowsheet.SelectedCompounds.Keys)
            return self._compounds_cache
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not retrieve compound list: {exc}"
            ) from exc

    def validate_composition(
        self, composition_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Validate and normalise a composition dict.

        * All compounds must exist in the simulation.
        * All fractions must be ≥ 0.
        * Result is normalised to sum = 1.0.
        """
        if not composition_dict:
            raise DWSIMInterfaceError("composition_dict must not be empty.")
        available = self.get_available_compounds()
        unknown = [c for c in composition_dict if c not in available]
        if unknown:
            raise DWSIMInterfaceError(
                f"Unknown compound(s) {unknown!r}.  "
                f"Available: {available!r}."
            )
        if any(v < 0 for v in composition_dict.values()):
            raise DWSIMInterfaceError("Mole fractions must be non-negative.")
        total = sum(composition_dict.values())
        if total == 0:
            raise DWSIMInterfaceError("Sum of mole fractions is zero.")
        if abs(total - 1.0) > 1e-9:
            logger.debug("Normalising composition (sum = %g).", total)
            return {c: v / total for c, v in composition_dict.items()}
        return dict(composition_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # Context-manager support
    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self) -> "DWSIMInterface":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_simulation()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self._flowsheet is None:
            raise DWSIMInterfaceError(
                "No simulation loaded.  Call load_simulation() first."
            )

    def _validate_mole_fraction(self, value: float, param_name: str) -> None:
        if not (0.0 <= value <= 1.0):
            raise DWSIMInterfaceError(
                f"{param_name} must be in [0, 1] (got {value})."
            )

    def _validate_compound_name(self, compound_name: str) -> None:
        available = self.get_available_compounds()
        if compound_name not in available:
            raise DWSIMInterfaceError(
                f"Unknown compound {compound_name!r}.  "
                f"Available: {available!r}."
            )

    def _get_material_stream(self, name: str) -> Any:
        """Return a material-stream object exposing the ``Phases`` attribute."""
        obj = self._get_object(name, "stream")
        try:
            typed = obj.GetAsObject()
            if typed is not None:
                return typed
        except Exception:
            pass
        return obj

    def _get_shortcut_column(self, name: str) -> Any:
        """
        Return a typed ShortcutColumn object.

        Uses the same GetAsObject() cast as _get_material_stream so that
        the concrete .NET type is returned and its properties (LightKeyCompound,
        RefluxRatio, CondenserPressure, etc.) are directly accessible.
        """
        obj = self._get_object(name, "column")
        try:
            typed = obj.GetAsObject()
            if typed is not None:
                return typed
        except Exception:
            pass
        return obj

    def _get_object(self, name: str, kind: str) -> Any:
        """Return a typed flowsheet object by name."""
        try:
            obj = self._flowsheet.GetFlowsheetSimulationObject(name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Cannot find {kind} {name!r} in the flowsheet.  "
                "Check that the name matches the DWSIM diagram exactly."
            ) from exc
        if obj is None:
            raise DWSIMInterfaceError(
                f"{kind.capitalize()} {name!r} was not found in the flowsheet."
            )
        try:
            typed = obj.GetAsObject()
            if typed is not None:
                return typed
        except Exception:
            pass
        return obj