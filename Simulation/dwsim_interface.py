"""
DWSIM INTERFACE
Provides bidirectional communication between Python and DWSIM via pythonnet.
Wraps DWSIM.Automation to load, run and query flowsheet objects.
"""
import logging
import os
import platform
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        stream = self._get_object(stream_name, "stream")

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
        stream = self._get_object(stream_name, "stream")

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
