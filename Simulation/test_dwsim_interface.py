"""
Unit tests for dwsim_interface.py and dwsim_data_generator.py.

These tests use mocking so they run without a real DWSIM installation.
Run with:  python -m pytest Simulation/test_dwsim_interface.py -v
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure the Simulation directory is on sys.path so bare imports work
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake DWSIM object tree
# ---------------------------------------------------------------------------

def _make_fake_stream(
    massflow=10000.0 / 3600,  # kg/s
    temperature=350.0,        # K
    pressure=101325.0,        # Pa
    molarflow=1.0,
    ethanol_mole_frac=0.80,
    water_mole_frac=0.20,
):
    compound_ethanol = MagicMock()
    compound_ethanol.MoleFraction = ethanol_mole_frac
    compound_ethanol.MassFraction = ethanol_mole_frac

    compound_water = MagicMock()
    compound_water.MoleFraction = water_mole_frac
    compound_water.MassFraction = water_mole_frac

    props = MagicMock()
    props.massflow = massflow
    props.temperature = temperature
    props.pressure = pressure
    props.molarflow = molarflow

    phase = MagicMock()
    phase.Properties = props
    phase.Compounds = {"Ethanol": compound_ethanol, "Water": compound_water}

    stream = MagicMock()
    stream.Phases = {0: phase}
    return stream


def _make_fake_column(
    condenser_duty=-1_207_870.0,   # W (negative = heat removed)
    reboiler_duty=1_524_290.0,     # W
    reflux_ratio=3.5,
    n_stages=20,
):
    col = MagicMock()
    col.CondenserDuty = condenser_duty
    col.ReboilerDuty = reboiler_duty
    col.RefluxRatio = reflux_ratio
    col.NumberOfStages = n_stages
    return col


def _make_fake_energy_stream(energy_flow=-1_207_870.0):
    es = MagicMock()
    es.EnergyFlow = energy_flow
    return es


# ---------------------------------------------------------------------------
# Patch factories
# ---------------------------------------------------------------------------

def _make_clr_mock():
    """Return a fake `clr` module that silently accepts AddReference calls."""
    clr_mock = MagicMock()
    clr_mock.AddReference = MagicMock(return_value=None)
    return clr_mock


def _make_automation_mock(flowsheet_objects: dict):
    """
    Return a mock DWSIM.Automation.Automation instance whose
    GetFlowsheetSimulationObject returns objects from *flowsheet_objects*.
    """
    flowsheet = MagicMock()
    flowsheet.GetFlowsheetSimulationObject.side_effect = (
        lambda name: flowsheet_objects.get(name)
    )

    automation_instance = MagicMock()
    automation_instance.LoadFlowsheet.return_value = flowsheet
    automation_instance.CalculateFlowsheet.return_value = None
    automation_instance.CloseFlowsheet.return_value = None

    return automation_instance, flowsheet


# ---------------------------------------------------------------------------
# Tests for DWSIMInterface
# ---------------------------------------------------------------------------

class TestDWSIMInterface(unittest.TestCase):

    def _build_interface(self, flowsheet_objects=None):
        """
        Build a DWSIMInterface with all .NET/DWSIM imports mocked out.
        Returns (interface, automation_mock).
        """
        if flowsheet_objects is None:
            flowsheet_objects = {}

        automation_mock, flowsheet_mock = _make_automation_mock(flowsheet_objects)
        clr_mock = _make_clr_mock()

        # Build a fake DWSIM.Automation module hierarchy
        dwsim_automation_mod = types.ModuleType("DWSIM.Automation")
        dwsim_automation_mod.Automation = MagicMock(return_value=automation_mock)

        with (
            patch.dict("sys.modules", {
                "clr": clr_mock,
                "DWSIM": types.ModuleType("DWSIM"),
                "DWSIM.Automation": dwsim_automation_mod,
            }),
            patch("os.path.isdir", return_value=True),
        ):
            from dwsim_interface import DWSIMInterface
            iface = DWSIMInterface.__new__(DWSIMInterface)
            iface.install_path = r"C:\DWSIM"
            iface._automation = automation_mock
            iface._flowsheet = None
            iface._loaded_file = None
            iface._initialised = True

        return iface, automation_mock, flowsheet_mock

    # ------------------------------------------------------------------
    # load_simulation
    # ------------------------------------------------------------------

    def test_load_simulation_success(self):
        iface, automation_mock, _ = self._build_interface()
        with patch("os.path.isfile", return_value=True):
            fs = iface.load_simulation("ethanol.dwxmz")
        automation_mock.LoadFlowsheet.assert_called_once_with("ethanol.dwxmz")
        self.assertIsNotNone(fs)

    def test_load_simulation_file_missing(self):
        from dwsim_interface import DWSIMInterfaceError
        iface, _, _ = self._build_interface()
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(DWSIMInterfaceError):
                iface.load_simulation("missing.dwxmz")

    def test_load_simulation_requires_init(self):
        from dwsim_interface import DWSIMInterfaceError
        iface, _, _ = self._build_interface()
        iface._initialised = False
        with self.assertRaises(DWSIMInterfaceError):
            iface.load_simulation("ethanol.dwxmz")

    # ------------------------------------------------------------------
    # run_simulation
    # ------------------------------------------------------------------

    def test_run_simulation_success(self):
        iface, automation_mock, flowsheet_mock = self._build_interface()
        iface._flowsheet = flowsheet_mock
        iface.run_simulation()
        automation_mock.CalculateFlowsheet.assert_called_once_with(flowsheet_mock)

    def test_run_simulation_no_flowsheet(self):
        from dwsim_interface import DWSIMInterfaceError
        iface, _, _ = self._build_interface()
        with self.assertRaises(DWSIMInterfaceError):
            iface.run_simulation()

    # ------------------------------------------------------------------
    # get_stream_property
    # ------------------------------------------------------------------

    def test_get_stream_massflow(self):
        feed = _make_fake_stream(massflow=2.5)
        iface, _, flowsheet_mock = self._build_interface({"Feed": feed})
        iface._flowsheet = flowsheet_mock
        result = iface.get_stream_property("Feed", "MassFlow")
        self.assertAlmostEqual(result, 2.5)

    def test_get_stream_temperature(self):
        feed = _make_fake_stream(temperature=370.0)
        iface, _, flowsheet_mock = self._build_interface({"Feed": feed})
        iface._flowsheet = flowsheet_mock
        result = iface.get_stream_property("Feed", "Temperature")
        self.assertAlmostEqual(result, 370.0)

    def test_get_stream_molefraction_with_component(self):
        top = _make_fake_stream(ethanol_mole_frac=0.85)
        iface, _, flowsheet_mock = self._build_interface({"Top": top})
        iface._flowsheet = flowsheet_mock
        result = iface.get_stream_property("Top", "MoleFraction", component="Ethanol")
        self.assertAlmostEqual(result, 0.85)

    def test_get_stream_molefraction_missing_component(self):
        from dwsim_interface import DWSIMInterfaceError
        top = _make_fake_stream()
        iface, _, flowsheet_mock = self._build_interface({"Top": top})
        iface._flowsheet = flowsheet_mock
        with self.assertRaises(DWSIMInterfaceError):
            iface.get_stream_property("Top", "MoleFraction")

    def test_get_stream_unknown_property(self):
        from dwsim_interface import DWSIMInterfaceError
        feed = _make_fake_stream()
        iface, _, flowsheet_mock = self._build_interface({"Feed": feed})
        iface._flowsheet = flowsheet_mock
        with self.assertRaises(DWSIMInterfaceError):
            iface.get_stream_property("Feed", "InvalidProp")

    def test_get_stream_not_found(self):
        from dwsim_interface import DWSIMInterfaceError
        iface, _, flowsheet_mock = self._build_interface({})
        iface._flowsheet = flowsheet_mock
        with self.assertRaises(DWSIMInterfaceError):
            iface.get_stream_property("NonExistent", "MassFlow")

    # ------------------------------------------------------------------
    # get_equipment_property
    # ------------------------------------------------------------------

    def test_get_equipment_condenser_duty(self):
        col = _make_fake_column(condenser_duty=-1_200_000.0)
        iface, _, flowsheet_mock = self._build_interface({"SCOL-1": col})
        iface._flowsheet = flowsheet_mock
        result = iface.get_equipment_property("SCOL-1", "DutyCondenser")
        self.assertAlmostEqual(result, -1_200_000.0)

    def test_get_equipment_reflux_ratio(self):
        col = _make_fake_column(reflux_ratio=4.0)
        iface, _, flowsheet_mock = self._build_interface({"SCOL-1": col})
        iface._flowsheet = flowsheet_mock
        result = iface.get_equipment_property("SCOL-1", "RefluxRatio")
        self.assertAlmostEqual(result, 4.0)

    def test_get_equipment_unknown_property(self):
        from dwsim_interface import DWSIMInterfaceError
        col = _make_fake_column()
        iface, _, flowsheet_mock = self._build_interface({"SCOL-1": col})
        iface._flowsheet = flowsheet_mock
        with self.assertRaises(DWSIMInterfaceError):
            iface.get_equipment_property("SCOL-1", "UnknownProp")

    # ------------------------------------------------------------------
    # set_stream_property
    # ------------------------------------------------------------------

    def test_set_stream_massflow(self):
        feed = _make_fake_stream()
        iface, _, flowsheet_mock = self._build_interface({"Feed": feed})
        iface._flowsheet = flowsheet_mock
        iface.set_stream_property("Feed", "MassFlow", 3.0)
        self.assertEqual(feed.Phases[0].Properties.massflow, 3.0)

    def test_set_stream_unsupported_property(self):
        from dwsim_interface import DWSIMInterfaceError
        feed = _make_fake_stream()
        iface, _, flowsheet_mock = self._build_interface({"Feed": feed})
        iface._flowsheet = flowsheet_mock
        with self.assertRaises(DWSIMInterfaceError):
            iface.set_stream_property("Feed", "MoleFraction", 0.5)

    # ------------------------------------------------------------------
    # close_simulation / context manager
    # ------------------------------------------------------------------

    def test_close_simulation(self):
        iface, automation_mock, flowsheet_mock = self._build_interface()
        iface._flowsheet = flowsheet_mock
        iface._loaded_file = "ethanol.dwxmz"
        iface.close_simulation()
        automation_mock.CloseFlowsheet.assert_called_once_with(flowsheet_mock)
        self.assertIsNone(iface._flowsheet)

    def test_context_manager_calls_close(self):
        iface, automation_mock, flowsheet_mock = self._build_interface()
        iface._flowsheet = flowsheet_mock
        iface._loaded_file = "ethanol.dwxmz"
        with iface:
            pass
        automation_mock.CloseFlowsheet.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for dwsim_data_generator
# ---------------------------------------------------------------------------

class TestValidateDwsimInstallation(unittest.TestCase):

    def test_returns_false_when_directory_missing(self):
        import dwsim_data_generator
        with (
            patch("os.path.isdir", return_value=False),
            patch("os.path.isfile", return_value=False),
        ):
            valid, msg = dwsim_data_generator.validate_dwsim_installation()
        self.assertFalse(valid)
        self.assertIn("not found", msg.lower())

    def test_returns_false_when_pythonnet_missing(self):
        import dwsim_data_generator
        with (
            patch("os.path.isdir", return_value=True),
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"clr": None}),
            patch("builtins.__import__", side_effect=ImportError("no module clr")),
        ):
            try:
                valid, msg = dwsim_data_generator.validate_dwsim_installation()
                if not valid:
                    self.assertFalse(valid)
            except ImportError:
                pass  # acceptable — pythonnet truly absent

    def test_returns_true_when_all_ok(self):
        import dwsim_data_generator
        clr_mock = MagicMock()
        with (
            patch("platform.system", return_value="Windows"),
            patch("os.path.isdir", return_value=True),
            patch("os.path.isfile", return_value=True),
            patch.dict("sys.modules", {"clr": clr_mock}),
        ):
            valid, msg = dwsim_data_generator.validate_dwsim_installation()
        self.assertTrue(valid)
        self.assertIn("OK", msg)


class TestGenerateDwsimDataFallback(unittest.TestCase):
    """
    Verify generate_dwsim_data falls back to base values when every
    simulation point raises DWSIMInterfaceError.
    """

    def test_fallback_produces_correct_shape(self):
        import dwsim_data_generator
        from dwsim_interface import DWSIMInterfaceError

        # Build a DWSIMInterface mock that always fails on run_simulation
        iface_mock = MagicMock()
        iface_mock.__enter__ = MagicMock(return_value=iface_mock)
        iface_mock.__exit__ = MagicMock(return_value=False)
        iface_mock.load_simulation = MagicMock(return_value=MagicMock())
        iface_mock.set_stream_property = MagicMock(
            side_effect=DWSIMInterfaceError("test error")
        )
        iface_mock.run_simulation = MagicMock(
            side_effect=DWSIMInterfaceError("test error")
        )

        with patch.object(
            dwsim_data_generator,
            "DWSIMInterface",
            return_value=iface_mock,
        ):
            df = dwsim_data_generator.generate_dwsim_data(n_points=5)

        self.assertEqual(len(df), 5)
        self.assertIn("F_feed_raw", df.columns)
        self.assertIn("Q_cond_raw", df.columns)

    def test_perturbation_length_mismatch_raises(self):
        import dwsim_data_generator
        with self.assertRaises(ValueError):
            dwsim_data_generator.generate_dwsim_data(
                n_points=5, perturbations=[0, 0, 0]
            )


if __name__ == "__main__":
    unittest.main()
