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
