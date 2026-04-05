"""
DWSIM INTERFACE
Provides bidirectional communication between Python and DWSIM via pythonnet.
Wraps DWSIM.Automation to load, run and query flowsheet objects.
"""
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Unit-conversion constants
_KMOLH_TO_MOLS = 1000.0 / 3600.0   # kmol/h → mol/s
_CELSIUS_TO_K = 273.15              # °C offset → K
_BAR_TO_PA = 1e5                    # bar → Pa


class DWSIMInterfaceError(Exception):
    """Raised when the DWSIM interface encounters a fatal error."""


class DWSIMInterface:
    """
    Context-manager–friendly wrapper around the DWSIM Automation API.

    Usage::

        with DWSIMInterface(dwsim_install_path) as dwsim:
            dwsim.load_simulation("ethanol.dwxmz")
            dwsim.run_simulation()
            feed_flow = dwsim.get_stream_property("Feed", "MassFlow")

    Parameters
    ----------
    install_path : str
        Absolute path to the DWSIM installation directory that contains
        ``DWSIM.Automation.dll`` and the other required assemblies.
    """

    def __init__(self, install_path: str) -> None:
        self.install_path = install_path
        self._automation = None   # DWSIM.Automation.Automation instance
        self._flowsheet = None    # IFlowsheet / loaded simulation object
        self._loaded_file: Optional[str] = None
        self._initialised = False
        self._compounds_cache: Optional[List[str]] = None
        self._material_stream_class = None  # cached MaterialStream type for casting
        self._init_dotnet()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_dotnet(self) -> None:
        """Load the .NET runtime and import DWSIM assemblies via pythonnet."""
        if platform.system() != "Windows":
            logger.warning(
                "DWSIM Automation is primarily supported on Windows. "
                "Attempting to load via Mono on %s — this may not work.",
                platform.system(),
            )

        try:
            import clr  # noqa: F401 — pythonnet
        except ImportError as exc:
            raise DWSIMInterfaceError(
                "pythonnet is not installed. "
                "Install it with: pip install pythonnet>=3.0.0"
            ) from exc

        if not os.path.isdir(self.install_path):
            raise DWSIMInterfaceError(
                f"DWSIM installation directory not found: {self.install_path!r}. "
                "Update DWSIM_INSTALL_PATH in Simulation/config.py."
            )

        # Add the DWSIM directory to the .NET assembly search path
        try:
            import clr as _clr

            _clr.AddReference  # check attribute exists (pythonnet ≥ 2)
            if self.install_path not in sys.path:
                sys.path.insert(0, self.install_path)
            _clr.AddReference("DWSIM.Automation")
            # Also load DWSIM.Objects so MaterialStream is available for casting.
            # This assembly is required when GetFlowsheetSimulationObject() returns
            # ISimulationObject and a cast to MaterialStream is needed to access Phases.
            try:
                _clr.AddReference("DWSIM.Objects")
                logger.debug("DWSIM.Objects assembly loaded.")
            except Exception:
                logger.debug(
                    "DWSIM.Objects assembly not loaded — stream casting will be "
                    "attempted lazily on first stream access."
                )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Failed to load DWSIM.Automation.dll from {self.install_path!r}. "
                f"Ensure DWSIM is installed and the path is correct. Detail: {exc}"
            ) from exc

        try:
            from DWSIM.Automation import Automation  # type: ignore[import]

            self._automation = Automation()
            self._initialised = True
            logger.info("DWSIM Automation initialised successfully.")
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not create DWSIM Automation object: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_simulation(self, file_path: str) -> Any:
        """
        Load a ``.dwxmz`` simulation file and return the flowsheet object.

        Parameters
        ----------
        file_path : str
            Path to the ``*.dwxmz`` file.

        Returns
        -------
        flowsheet
            The DWSIM ``IFlowsheet`` (or equivalent) object for the loaded
            simulation.

        Raises
        ------
        DWSIMInterfaceError
            If the file does not exist or DWSIM fails to open it.
        """
        if not self._initialised:
            raise DWSIMInterfaceError("DWSIM interface is not initialised.")

        if not os.path.isfile(file_path):
            raise DWSIMInterfaceError(
                f"Simulation file not found: {file_path!r}"
            )

        try:
            self._flowsheet = self._automation.LoadFlowsheet(file_path)
            self._loaded_file = file_path
            self._compounds_cache = None  # invalidate cache on new load
            logger.info("Loaded simulation: %s", file_path)
            return self._flowsheet
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Failed to load simulation {file_path!r}: {exc}"
            ) from exc

    def run_simulation(self) -> None:
        """
        Execute the DWSIM calculation for the currently loaded flowsheet.

        Raises
        ------
        DWSIMInterfaceError
            If no simulation is loaded or the calculation fails.
        """
        self._require_loaded()
        try:
            self._automation.CalculateFlowsheet(self._flowsheet)
            logger.info("Simulation run completed.")
        except Exception as exc:
            raise DWSIMInterfaceError(f"Simulation run failed: {exc}") from exc

    def get_stream_property(
        self,
        stream_name: str,
        property_name: str,
        component: Optional[str] = None,
    ) -> float:
        """
        Extract a scalar property from a material or energy stream.

        Parameters
        ----------
        stream_name : str
            Name of the stream as defined in the DWSIM flowsheet
            (e.g. ``"Feed"``, ``"Top"``, ``"Bottom"``).
        property_name : str
            Property to retrieve.  Supported values:

            * ``"MassFlow"``       — total mass flow rate (kg/h)
            * ``"Temperature"``   — stream temperature (K)
            * ``"Pressure"``      — stream pressure (Pa)
            * ``"MolarFlow"``     — total molar flow rate (mol/s)
            * ``"MassFraction"``  — requires ``component`` keyword
            * ``"MoleFraction"``  — requires ``component`` keyword

        component : str, optional
            Component name (e.g. ``"Ethanol"``, ``"Water"``) required when
            *property_name* is ``"MassFraction"`` or ``"MoleFraction"``.

        Returns
        -------
        float
            The requested property value in SI units as returned by DWSIM.

        Raises
        ------
        DWSIMInterfaceError
            If the stream or property does not exist.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)

        try:
            if property_name == "MassFlow":
                return float(stream.Phases[0].Properties.massflow)
            if property_name == "Temperature":
                return float(stream.Phases[0].Properties.temperature)
            if property_name == "Pressure":
                return float(stream.Phases[0].Properties.pressure)
            if property_name == "MolarFlow":
                return float(stream.Phases[0].Properties.molarflow)
            if property_name == "MassFraction":
                if component is None:
                    raise DWSIMInterfaceError(
                        "'component' must be provided for MassFraction."
                    )
                return float(
                    stream.Phases[0].Compounds[component].MassFraction
                )
            if property_name == "MoleFraction":
                if component is None:
                    raise DWSIMInterfaceError(
                        "'component' must be provided for MoleFraction."
                    )
                return float(
                    stream.Phases[0].Compounds[component].MoleFraction
                )
            raise DWSIMInterfaceError(
                f"Unknown stream property: {property_name!r}. "
                "Supported: MassFlow, Temperature, Pressure, MolarFlow, "
                "MassFraction, MoleFraction."
            )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not read {property_name!r} from stream {stream_name!r}: {exc}"
            ) from exc

    def get_equipment_property(
        self, equipment_name: str, property_name: str
    ) -> float:
        """
        Extract a scalar property from a unit operation (equipment).

        Parameters
        ----------
        equipment_name : str
            Name of the equipment as defined in the DWSIM flowsheet
            (e.g. ``"SCOL-1"``, ``"R_cond"``, ``"Q_reb"``).
        property_name : str
            Property to retrieve.  Supported values:

            * ``"DutyCondenser"``   — condenser duty (W)
            * ``"DutyReboiler"``    — reboiler duty (W)
            * ``"RefluxRatio"``     — column reflux ratio (–)
            * ``"NumberOfStages"``  — number of theoretical stages (–)
            * ``"Duty"``            — generic energy duty for energy streams (W)

        Returns
        -------
        float

        Raises
        ------
        DWSIMInterfaceError
            If the equipment or property does not exist.
        """
        self._require_loaded()
        obj = self._get_object(equipment_name, "equipment")

        try:
            if property_name == "DutyCondenser":
                return float(obj.CondenserDuty)
            if property_name == "DutyReboiler":
                return float(obj.ReboilerDuty)
            if property_name == "RefluxRatio":
                return float(obj.RefluxRatio)
            if property_name == "NumberOfStages":
                return float(obj.NumberOfStages)
            if property_name == "Duty":
                # Generic energy-stream duty
                return float(obj.EnergyFlow)
            raise DWSIMInterfaceError(
                f"Unknown equipment property: {property_name!r}. "
                "Supported: DutyCondenser, DutyReboiler, RefluxRatio, "
                "NumberOfStages, Duty."
            )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not read {property_name!r} from equipment "
                f"{equipment_name!r}: {exc}"
            ) from exc

    def set_stream_property(
        self,
        stream_name: str,
        property_name: str,
        value: float,
    ) -> None:
        """
        Set a scalar property on a material stream (input specification).

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet.
        property_name : str
            Property to set.  Supported: ``"MassFlow"``, ``"Temperature"``,
            ``"Pressure"``.
        value : float
            New value in SI units (kg/s, K, Pa respectively).

        Raises
        ------
        DWSIMInterfaceError
            If the stream or property is not writable.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)

        try:
            if property_name == "MassFlow":
                stream.Phases[0].Properties.massflow = value
            elif property_name == "Temperature":
                stream.Phases[0].Properties.temperature = value
            elif property_name == "Pressure":
                stream.Phases[0].Properties.pressure = value
            else:
                raise DWSIMInterfaceError(
                    f"Property {property_name!r} is not settable via this "
                    "interface. Settable properties: MassFlow, Temperature, Pressure."
                )
            logger.debug(
                "Set %s.%s = %s", stream_name, property_name, value
            )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set {property_name!r} on stream {stream_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # High-level stream setters (user-friendly units)
    # ------------------------------------------------------------------

    def set_stream_molar_flow(
        self, stream_name: str, molar_flow_kmolh: float
    ) -> None:
        """
        Set the molar flow rate of a material stream.

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet (e.g. ``"Feed"``).
        molar_flow_kmolh : float
            Molar flow rate in **kmol/h**.  Converted to mol/s internally.

        Raises
        ------
        DWSIMInterfaceError
            If the stream does not exist or the property cannot be set.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        molar_flow_mols = molar_flow_kmolh * _KMOLH_TO_MOLS
        try:
            stream.Phases[0].Properties.molarflow = molar_flow_mols
            logger.debug(
                "Set %s.molarflow = %.6g mol/s (%.4g kmol/h)",
                stream_name, molar_flow_mols, molar_flow_kmolh,
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set molar flow on stream {stream_name!r}: {exc}"
            ) from exc

    def set_stream_temperature(
        self, stream_name: str, temperature_celsius: float
    ) -> None:
        """
        Set the temperature of a material stream.

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet.
        temperature_celsius : float
            Temperature in **°C**.  Converted to K internally (+273.15).

        Raises
        ------
        DWSIMInterfaceError
            If the stream does not exist or the property cannot be set.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        temperature_k = temperature_celsius + _CELSIUS_TO_K
        try:
            stream.Phases[0].Properties.temperature = temperature_k
            logger.debug(
                "Set %s.temperature = %.2f K (%.2f °C)",
                stream_name, temperature_k, temperature_celsius,
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set temperature on stream {stream_name!r}: {exc}"
            ) from exc

    def set_stream_pressure(
        self, stream_name: str, pressure_bar: float
    ) -> None:
        """
        Set the pressure of a material stream.

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet.
        pressure_bar : float
            Pressure in **bar**.  Converted to Pa internally (×100 000).

        Raises
        ------
        DWSIMInterfaceError
            If the stream does not exist or the property cannot be set.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        pressure_pa = pressure_bar * _BAR_TO_PA
        try:
            stream.Phases[0].Properties.pressure = pressure_pa
            logger.debug(
                "Set %s.pressure = %.4g Pa (%.4g bar)",
                stream_name, pressure_pa, pressure_bar,
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set pressure on stream {stream_name!r}: {exc}"
            ) from exc

    def set_stream_composition(
        self, stream_name: str, composition_dict: Dict[str, float]
    ) -> None:
        """
        Set the mole-fraction composition of a material stream.

        The provided fractions are automatically normalised to sum to 1.0.
        Only the compounds listed in *composition_dict* are updated; other
        compounds already present in the stream are **not** modified.

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet (e.g. ``"Feed"``).
        composition_dict : dict
            Mapping of compound name → mole fraction.
            Example: ``{"Ethanol": 0.1, "Water": 0.9}``

        Raises
        ------
        DWSIMInterfaceError
            If the stream does not exist, a compound name is unknown, or
            fractions are invalid.
        """
        self._require_loaded()
        stream = self._get_material_stream(stream_name)
        normalised = self.validate_composition(composition_dict)
        try:
            for compound, fraction in normalised.items():
                stream.Phases[0].Compounds[compound].MoleFraction = fraction
                logger.debug(
                    "Set %s compound %s MoleFraction = %.6g",
                    stream_name, compound, fraction,
                )
        except DWSIMInterfaceError:
            raise
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set composition on stream {stream_name!r}: {exc}"
            ) from exc

    def set_stream_conditions(
        self,
        stream_name: str,
        molar_flow: Optional[float] = None,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        composition: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Convenience method to set multiple stream conditions in a single call.

        All parameters are optional; only the provided ones are applied.

        Parameters
        ----------
        stream_name : str
            Name of the stream in the DWSIM flowsheet (e.g. ``"Feed"``).
        molar_flow : float, optional
            Molar flow rate in **kmol/h**.
        temperature : float, optional
            Temperature in **°C**.
        pressure : float, optional
            Pressure in **bar**.
        composition : dict, optional
            Mole-fraction dictionary, e.g. ``{"Ethanol": 0.1, "Water": 0.9}``.

        Example
        -------
        ::

            dwsim.set_stream_conditions(
                "Feed",
                molar_flow=100,                          # kmol/h
                temperature=30,                           # °C
                pressure=10,                              # bar
                composition={"Ethanol": 0.1, "Water": 0.9},
            )
        """
        if molar_flow is not None:
            self.set_stream_molar_flow(stream_name, molar_flow)
        if temperature is not None:
            self.set_stream_temperature(stream_name, temperature)
        if pressure is not None:
            self.set_stream_pressure(stream_name, pressure)
        if composition is not None:
            self.set_stream_composition(stream_name, composition)

    # ------------------------------------------------------------------
    # Shortcut-column setters
    # ------------------------------------------------------------------

    def set_column_light_key(
        self, column_name: str, compound_name: str
    ) -> None:
        """
        Set the light-key compound for a shortcut distillation column.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet (e.g. ``"SCOL-1"``).
        compound_name : str
            Exact compound name as known to DWSIM (e.g. ``"Ethanol"``).

        Raises
        ------
        DWSIMInterfaceError
            If the column does not exist or the compound is not in the
            simulation.
        """
        self._require_loaded()
        self._validate_compound_name(compound_name)
        col = self._get_object(column_name, "column")
        try:
            col.LightKeyCompound = compound_name
            logger.debug("Set %s.LightKeyCompound = %r", column_name, compound_name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set LightKeyCompound on column {column_name!r}: {exc}"
            ) from exc

    def set_column_heavy_key(
        self, column_name: str, compound_name: str
    ) -> None:
        """
        Set the heavy-key compound for a shortcut distillation column.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet (e.g. ``"SCOL-1"``).
        compound_name : str
            Exact compound name as known to DWSIM (e.g. ``"Water"``).

        Raises
        ------
        DWSIMInterfaceError
            If the column does not exist or the compound is not in the
            simulation.
        """
        self._require_loaded()
        self._validate_compound_name(compound_name)
        col = self._get_object(column_name, "column")
        try:
            col.HeavyKeyCompound = compound_name
            logger.debug("Set %s.HeavyKeyCompound = %r", column_name, compound_name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set HeavyKeyCompound on column {column_name!r}: {exc}"
            ) from exc

    def set_column_lk_fraction_bottoms(
        self, column_name: str, mole_fraction: float
    ) -> None:
        """
        Set the light-key mole fraction in the bottoms stream.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet.
        mole_fraction : float
            Target mole fraction of the light key in the bottoms (0 – 1).

        Raises
        ------
        DWSIMInterfaceError
            If the value is outside [0, 1] or the property cannot be set.
        """
        self._require_loaded()
        self._validate_mole_fraction(mole_fraction, "lk_bottoms")
        col = self._get_object(column_name, "column")
        try:
            col.LKMoleFractionInBottoms = mole_fraction
            logger.debug(
                "Set %s.LKMoleFractionInBottoms = %.4g", column_name, mole_fraction
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set LKMoleFractionInBottoms on column "
                f"{column_name!r}: {exc}"
            ) from exc

    def set_column_hk_fraction_distillate(
        self, column_name: str, mole_fraction: float
    ) -> None:
        """
        Set the heavy-key mole fraction in the distillate stream.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet.
        mole_fraction : float
            Target mole fraction of the heavy key in the distillate (0 – 1).

        Raises
        ------
        DWSIMInterfaceError
            If the value is outside [0, 1] or the property cannot be set.
        """
        self._require_loaded()
        self._validate_mole_fraction(mole_fraction, "hk_distillate")
        col = self._get_object(column_name, "column")
        try:
            col.HKMoleFractionInDistillate = mole_fraction
            logger.debug(
                "Set %s.HKMoleFractionInDistillate = %.4g",
                column_name, mole_fraction,
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set HKMoleFractionInDistillate on column "
                f"{column_name!r}: {exc}"
            ) from exc

    def set_column_reflux_ratio(
        self, column_name: str, reflux_ratio: float
    ) -> None:
        """
        Set the reflux ratio of a shortcut distillation column.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet (e.g. ``"SCOL-1"``).
        reflux_ratio : float
            Reflux ratio (dimensionless, must be > 0).

        Raises
        ------
        DWSIMInterfaceError
            If the value is not positive or the property cannot be set.
        """
        self._require_loaded()
        if reflux_ratio <= 0:
            raise DWSIMInterfaceError(
                f"reflux_ratio must be positive (got {reflux_ratio})."
            )
        col = self._get_object(column_name, "column")
        try:
            col.RefluxRatio = reflux_ratio
            logger.debug(
                "Set %s.RefluxRatio = %.4g", column_name, reflux_ratio
            )
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not set RefluxRatio on column {column_name!r}: {exc}"
            ) from exc

    def set_column_parameters(
        self,
        column_name: str,
        light_key: Optional[str] = None,
        heavy_key: Optional[str] = None,
        lk_bottoms: Optional[float] = None,
        hk_distillate: Optional[float] = None,
        reflux_ratio: Optional[float] = None,
    ) -> None:
        """
        Convenience method to set multiple column parameters in a single call.

        All parameters are optional; only the provided ones are applied.

        Parameters
        ----------
        column_name : str
            Name of the column in the DWSIM flowsheet (e.g. ``"SCOL-1"``).
        light_key : str, optional
            Light-key compound name (e.g. ``"Ethanol"``).
        heavy_key : str, optional
            Heavy-key compound name (e.g. ``"Water"``).
        lk_bottoms : float, optional
            Light-key mole fraction in bottoms (0 – 1).
        hk_distillate : float, optional
            Heavy-key mole fraction in distillate (0 – 1).
        reflux_ratio : float, optional
            Reflux ratio (> 0).

        Example
        -------
        ::

            dwsim.set_column_parameters(
                "SCOL-1",
                light_key="Ethanol",
                heavy_key="Water",
                lk_bottoms=0.05,
                hk_distillate=0.1,
                reflux_ratio=1.1,
            )
        """
        if light_key is not None:
            self.set_column_light_key(column_name, light_key)
        if heavy_key is not None:
            self.set_column_heavy_key(column_name, heavy_key)
        if lk_bottoms is not None:
            self.set_column_lk_fraction_bottoms(column_name, lk_bottoms)
        if hk_distillate is not None:
            self.set_column_hk_fraction_distillate(column_name, hk_distillate)
        if reflux_ratio is not None:
            self.set_column_reflux_ratio(column_name, reflux_ratio)

    # ------------------------------------------------------------------
    # Compound / composition helpers
    # ------------------------------------------------------------------

    def get_available_compounds(self) -> List[str]:
        """
        Return the list of compound names available in the loaded simulation.

        The result is cached after the first call; the cache is cleared
        whenever :meth:`load_simulation` is called.

        Returns
        -------
        list of str
            Compound names exactly as defined in the DWSIM flowsheet.

        Raises
        ------
        DWSIMInterfaceError
            If no simulation is loaded or the compound list cannot be read.
        """
        self._require_loaded()
        if self._compounds_cache is not None:
            return self._compounds_cache
        try:
            self._compounds_cache = list(self._flowsheet.SelectedCompounds.Keys)
            return self._compounds_cache
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Could not retrieve compound list from flowsheet: {exc}"
            ) from exc

    def validate_composition(
        self, composition_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Validate and normalise a composition dictionary.

        * All compound names must exist in the loaded simulation.
        * All mole fractions must be non-negative.
        * The returned dictionary is normalised so that fractions sum to 1.0.

        Parameters
        ----------
        composition_dict : dict
            Mapping of compound name → mole fraction.

        Returns
        -------
        dict
            Normalised composition dictionary.

        Raises
        ------
        DWSIMInterfaceError
            If the dict is empty, contains unknown compounds, negative
            fractions, or if all fractions are zero.
        """
        if not composition_dict:
            raise DWSIMInterfaceError("composition_dict must not be empty.")

        available = self.get_available_compounds()
        unknown = [c for c in composition_dict if c not in available]
        if unknown:
            raise DWSIMInterfaceError(
                f"Unknown compound(s) {unknown!r}. "
                f"Available compounds: {available!r}."
            )

        if any(v < 0 for v in composition_dict.values()):
            raise DWSIMInterfaceError(
                "Mole fractions must be non-negative."
            )

        total = sum(composition_dict.values())
        if total == 0:
            raise DWSIMInterfaceError(
                "Sum of mole fractions is zero; cannot normalise."
            )

        if abs(total - 1.0) > 1e-9:
            logger.debug(
                "Normalising composition: fractions summed to %g (expected 1.0).",
                total,
            )
            return {c: v / total for c, v in composition_dict.items()}

        return dict(composition_dict)

    def close_simulation(self) -> None:
        """Release DWSIM resources associated with the loaded simulation."""
        if self._flowsheet is not None:
            try:
                self._automation.CloseFlowsheet(self._flowsheet)
                logger.info("Simulation closed: %s", self._loaded_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing simulation: %s", exc)
            finally:
                self._flowsheet = None
                self._loaded_file = None

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DWSIMInterface":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_simulation()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._flowsheet is None:
            raise DWSIMInterfaceError(
                "No simulation is loaded. Call load_simulation() first."
            )

    def _validate_mole_fraction(self, value: float, param_name: str) -> None:
        """Raise DWSIMInterfaceError if *value* is outside [0, 1]."""
        if not (0.0 <= value <= 1.0):
            raise DWSIMInterfaceError(
                f"{param_name} must be in [0, 1] (got {value})."
            )

    def _validate_compound_name(self, compound_name: str) -> None:
        """Raise DWSIMInterfaceError if *compound_name* is not in the simulation."""
        available = self.get_available_compounds()
        if compound_name not in available:
            raise DWSIMInterfaceError(
                f"Unknown compound {compound_name!r}. "
                f"Available compounds: {available!r}."
            )

    def _get_material_stream(self, name: str) -> Any:
        """
        Return a material-stream object that exposes the ``Phases`` attribute.

        In DWSIM's external Automation context,
        ``GetFlowsheetSimulationObject()`` returns an ``ISimulationObject``
        interface which does not expose ``Phases``.  This helper casts the
        object to the concrete ``MaterialStream`` class so that stream-specific
        properties (temperature, pressure, molar flow, composition …) are
        accessible.

        If the object already has a ``Phases`` attribute (e.g. when running
        inside DWSIM's own scripting console) it is returned unchanged.
        """
        obj = self._get_object(name, "stream")

        # Fast path: already a properly typed object (in-process scripting,
        # or a future DWSIM automation version that returns typed objects).
        if hasattr(obj, "Phases"):
            return obj

        # Slow path: ISimulationObject returned by external Automation DLL.
        # Load MaterialStream type once and cache it.
        if self._material_stream_class is None:
            try:
                import clr as _clr  # noqa: PLC0415

                try:
                    _clr.AddReference("DWSIM.Objects")
                except Exception:
                    pass  # may already be loaded

                from DWSIM.Objects.Streams import MaterialStream  # type: ignore[import]

                self._material_stream_class = MaterialStream
                logger.debug("Cached MaterialStream class for stream casting.")
            except Exception as exc:
                raise DWSIMInterfaceError(
                    "Cannot load 'MaterialStream' from DWSIM.Objects.dll. "
                    f"Ensure DWSIM.Objects.dll is present in {self.install_path!r}. "
                    f"Detail: {exc}"
                ) from exc

        try:
            return self._material_stream_class(obj)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Cannot cast stream {name!r} from ISimulationObject to "
                f"MaterialStream: {exc}"
            ) from exc

    def _get_object(self, name: str, kind: str) -> Any:
        """Return a flowsheet object by name, raising a clear error if missing."""
        try:
            obj = self._flowsheet.GetFlowsheetSimulationObject(name)
        except Exception as exc:
            raise DWSIMInterfaceError(
                f"Cannot find {kind} {name!r} in the loaded flowsheet. "
                "Check that the name matches the DWSIM diagram exactly."
            ) from exc
        if obj is None:
            raise DWSIMInterfaceError(
                f"{kind.capitalize()} {name!r} was not found in the flowsheet."
            )
        return obj
