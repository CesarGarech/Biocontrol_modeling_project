# DWSIM Integration Guide

This document explains how to configure and use the live DWSIM integration for
the Ethanol Distillation Digital Twin.

---

## Prerequisites

| Requirement | Version / Notes |
|-------------|-----------------|
| DWSIM | 8.x (Windows installer from https://dwsim.org) |
| Python | 3.9 – 3.12 |
| pythonnet | ≥ 3.0.0  (`pip install pythonnet>=3.0.0`) |
| .NET Runtime | 6.0 or 8.0 LTS (required by pythonnet 3) |
| OS | Windows 10/11 (primary). Linux/macOS via Mono is experimental. |

---

## Installation Steps

### 1. Install DWSIM

Download and run the DWSIM installer from the official website:
https://dwsim.org/index.php/download/

The default installation path is:
```
C:\Users\<username>\AppData\Local\DWSIM\
```

### 2. Install .NET Runtime

Download .NET 6 or 8 LTS from:
https://dotnet.microsoft.com/download

Verify with:
```bash
dotnet --version
```

### 3. Install Python dependencies

```bash
pip install pythonnet>=3.0.0
```

---

## Configuration

Open `Simulation/config.py` and adjust the following variables:

```python
# Path to DWSIM installation directory
DWSIM_INSTALL_PATH = r"C:\Users\<your_username>\AppData\Local\DWSIM\"

# Set to True to use live DWSIM simulations instead of synthetic data
USE_DWSIM_LIVE = True
```

All other DWSIM settings (`STREAM_PROPERTIES`, `EQUIPMENT_PROPERTIES`,
`PERTURBATION_AMPLITUDE`, etc.) are defined in the same file and can be
adjusted as needed.

---

## Communication Flow

```
Python (main2.py)
    │
    ▼
dwsim_data_generator.generate_dwsim_data()
    │
    ▼
DWSIMInterface (dwsim_interface.py)
    │  pythonnet (.NET bridge)
    ▼
DWSIM.Automation.dll
    │
    ▼
ethanol.dwxmz  (DWSIM flowsheet)
    │
    ▼
Calculated stream & equipment properties
    │
    ▼
DataFrame → data_processor → reconciliation_engine → dashboard
```

---

## Usage Examples

### Basic usage

```python
import sys
sys.path.insert(0, "Simulation/")

from dwsim_interface import DWSIMInterface
import config

with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
    dwsim.load_simulation(config.SIMULATION_FILE)
    dwsim.run_simulation()

    feed_flow = dwsim.get_stream_property("Feed", "MassFlow")   # kg/s
    top_ethanol = dwsim.get_stream_property(
        "Top", "MoleFraction", component="Ethanol"
    )
    condenser_duty = dwsim.get_equipment_property("R_cond", "DutyCondenser")  # W

    print(f"Feed flow:        {feed_flow * 3600:.1f} kg/h")
    print(f"Top ethanol:      {top_ethanol * 100:.2f} mol%")
    print(f"Condenser duty:  {abs(condenser_duty) / 1000:.2f} kW")
```

### Modifying feed conditions and re-running

```python
with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
    dwsim.load_simulation(config.SIMULATION_FILE)

    # Increase feed by 500 kg/h
    new_flow = (config.FLOW_FEED_BASE + 500) / 3600  # convert to kg/s
    dwsim.set_stream_property("Feed", "MassFlow", new_flow)
    dwsim.run_simulation()

    feed = dwsim.get_stream_property("Feed", "MassFlow") * 3600
    print(f"New feed flow: {feed:.1f} kg/h")
```

### Setting feed conditions with user-friendly units

```python
with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
    dwsim.load_simulation(config.SIMULATION_FILE)

    # Set all feed conditions in one call (user-friendly units)
    dwsim.set_stream_conditions(
        "Feed",
        molar_flow=100,                          # kmol/h
        temperature=30,                           # °C
        pressure=10,                              # bar
        composition={"Ethanol": 0.1, "Water": 0.9},  # mole fractions
    )

    dwsim.run_simulation()

    top_flow = dwsim.get_stream_property("Top", "MolarFlow") * 3.6  # kmol/h
    bottom_flow = dwsim.get_stream_property("Bottom", "MolarFlow") * 3.6
    print(f"Distillate: {top_flow:.2f} kmol/h")
    print(f"Bottoms:    {bottom_flow:.2f} kmol/h")
```

### Configuring the distillation column

```python
with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
    dwsim.load_simulation(config.SIMULATION_FILE)

    # Configure all column parameters in one call
    dwsim.set_column_parameters(
        "SCOL-1",
        light_key="Ethanol",
        heavy_key="Water",
        lk_bottoms=0.05,      # 5% ethanol allowed in bottoms
        hk_distillate=0.1,    # 10% water allowed in distillate
        reflux_ratio=1.1,
    )

    dwsim.run_simulation()

    top_ethanol = dwsim.get_stream_property("Top", "MoleFraction", component="Ethanol")
    bottom_ethanol = dwsim.get_stream_property("Bottom", "MoleFraction", component="Ethanol")
    print(f"Distillate purity: {top_ethanol*100:.2f}% ethanol")
    print(f"Bottoms ethanol:   {bottom_ethanol*100:.3f}%")
```

### Parametric study — effect of reflux ratio

```python
import pandas as pd

reflux_ratios = [0.5, 1.0, 1.5, 2.0, 2.5]
results = []

with DWSIMInterface(config.DWSIM_INSTALL_PATH) as dwsim:
    dwsim.load_simulation(config.SIMULATION_FILE)
    dwsim.set_stream_conditions("Feed", **config.DEFAULT_FEED_CONDITIONS)

    for rr in reflux_ratios:
        dwsim.set_column_reflux_ratio("SCOL-1", rr)
        dwsim.run_simulation()

        purity = dwsim.get_stream_property("Top", "MoleFraction", component="Ethanol")
        q_reb = abs(dwsim.get_equipment_property("SCOL-1", "DutyReboiler")) / 1000

        results.append({"Reflux_Ratio": rr, "Purity_%": purity * 100, "Reboiler_kW": q_reb})

df = pd.DataFrame(results)
print(df)
```

### Running the pipeline with live DWSIM

```python
# In Simulation/config.py, set:
USE_DWSIM_LIVE = True

# Then run:
python Simulation/main2.py
```

---

## Supported Properties

### Stream properties (`get_stream_property`)

| Property | Description | SI Unit |
|----------|-------------|---------|
| `MassFlow` | Total mass flow rate | kg/s |
| `Temperature` | Stream temperature | K |
| `Pressure` | Stream pressure | Pa |
| `MolarFlow` | Total molar flow rate | mol/s |
| `MassFraction` | Component mass fraction (needs `component=`) | – |
| `MoleFraction` | Component mole fraction (needs `component=`) | – |

### Equipment properties (`get_equipment_property`)

| Property | Description | SI Unit |
|----------|-------------|---------|
| `DutyCondenser` | Condenser heat duty | W |
| `DutyReboiler` | Reboiler heat duty | W |
| `RefluxRatio` | Column reflux ratio | – |
| `NumberOfStages` | Number of theoretical stages | – |
| `Duty` | Generic energy-stream duty | W |

### Settable stream properties (`set_stream_property`)

| Property | SI Unit |
|----------|---------|
| `MassFlow` | kg/s |
| `Temperature` | K |
| `Pressure` | Pa |

---

## User-Friendly Stream Setters

These methods accept engineering units and convert them internally.

| Python method | Input unit | DWSIM internal unit | Conversion |
|---------------|-----------|---------------------|------------|
| `set_stream_molar_flow` | kmol/h | mol/s | ÷ 3.6 |
| `set_stream_temperature` | °C | K | + 273.15 |
| `set_stream_pressure` | bar | Pa | × 100 000 |
| `set_stream_composition` | mole fractions | mole fractions | normalised to 1.0 |

### `set_stream_conditions` — set multiple properties at once

```python
dwsim.set_stream_conditions(
    stream_name,
    molar_flow=None,    # kmol/h
    temperature=None,   # °C
    pressure=None,      # bar
    composition=None,   # {"Ethanol": 0.1, "Water": 0.9}
)
```

All parameters are optional; only the provided ones are applied.

---

## Shortcut Column Setters

| Python method | Parameter | Type | Validation |
|---------------|-----------|------|------------|
| `set_column_light_key` | compound name | str | must exist in simulation |
| `set_column_heavy_key` | compound name | str | must exist in simulation |
| `set_column_lk_fraction_bottoms` | mole fraction | float | must be in [0, 1] |
| `set_column_hk_fraction_distillate` | mole fraction | float | must be in [0, 1] |
| `set_column_reflux_ratio` | reflux ratio | float | must be > 0 |

### `set_column_parameters` — set multiple column parameters at once

```python
dwsim.set_column_parameters(
    column_name,
    light_key=None,      # e.g. "Ethanol"
    heavy_key=None,      # e.g. "Water"
    lk_bottoms=None,     # mole fraction [0, 1]
    hk_distillate=None,  # mole fraction [0, 1]
    reflux_ratio=None,   # > 0
)
```

---

## Compound / Composition Helpers

### `get_available_compounds()`

Returns the list of compound names available in the loaded simulation.
Result is cached after the first call.

```python
compounds = dwsim.get_available_compounds()
# e.g. ["Ethanol", "Water"]
```

### `validate_composition(composition_dict)`

Validates and normalises a composition dictionary.

```python
normalised = dwsim.validate_composition({"Ethanol": 1.0, "Water": 3.0})
# returns {"Ethanol": 0.25, "Water": 0.75}
```

Raises `DWSIMInterfaceError` if:
- The dictionary is empty.
- A compound name is not in the simulation.
- Any mole fraction is negative.
- All mole fractions are zero.

---

## Parametric Study Script

A ready-to-run parametric study is provided in
`Examples/dwsim_parametric_study.py`.  It performs three studies:

1. **Reflux-ratio sweep** — records distillate purity and reboiler duty.
2. **Feed-condition grid** — varies temperature and pressure.
3. **Feed-composition sweep** — varies ethanol mole fraction in the feed.

Run it with:

```bash
python Examples/dwsim_parametric_study.py
```

CSV results and PNG plots are saved to the `Output/` directory.

---

## Troubleshooting

### `pythonnet is not installed`
Install it: `pip install pythonnet>=3.0.0`

### `DWSIM installation directory not found`
Check `DWSIM_INSTALL_PATH` in `config.py`.  The path must contain
`DWSIM.Automation.dll`.

### `Failed to load DWSIM.Automation.dll`
- Verify DWSIM 8+ is installed.
- Verify .NET 6/8 runtime is installed.
- On 64-bit Windows, ensure you use the 64-bit Python interpreter.

### `Cannot find stream 'Feed' in the loaded flowsheet`
The stream name in the flowsheet must match the tag in `config.py` exactly
(case-sensitive).  Open the `.dwxmz` file in DWSIM and verify the names.

### `Unknown compound 'Ethanol'`
Call `dwsim.get_available_compounds()` to see the exact compound names used
in the loaded simulation and update your code accordingly.

### Simulation converges to wrong values
- Check that the feed composition and thermodynamic model are correctly
  specified in the `.dwxmz` file.
- Increase the number of DWSIM solver iterations in the flowsheet settings.

### Linux / macOS
DWSIM Automation relies on .NET/COM interop that is not fully supported
outside Windows.  A warning is printed at startup.  The code will still
attempt to proceed using Mono; results may vary.

---

## Running the Tests

```bash
cd Simulation/
python -m pytest test_dwsim_interface.py -v
```

All tests use mocks and do **not** require a real DWSIM installation.

