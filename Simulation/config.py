"""
CONFIGURATION FILE
Central control panel for the Ethanol Distillation Digital Twin.
Defines paths, base operating conditions, sensor uncertainties, and KPI targets.
"""
import os

# ==========================================
# 1. SYSTEM PATHS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# DWSIM installation path — checks DWSIM_PATH env var first (Docker-friendly).
# Leave empty to force explicit configuration via the DWSIM_PATH environment variable.
DWSIM_INSTALL_PATH = os.environ.get(
    "DWSIM_PATH",
    r"C:\Users\cesar\AppData\Local\DWSIM"
)

# DWSIM simulation file
SIMULATION_FILE = os.path.join(CURRENT_DIR, "ethanol.dwxmz")

# Output files
RAW_DATA_FILE = os.path.join(CURRENT_DIR, "scada_data_raw.csv")
CLEAN_DATA_FILE = os.path.join(CURRENT_DIR, "scada_data_clean.csv")
RESULTS_FILE = os.path.join(CURRENT_DIR, "digital_twin_results.csv")

# ==========================================
# 2. DWSIM STREAM / EQUIPMENT TAGS
# ==========================================
# These names MUST match the DWSIM flowsheet exactly
TAG_FEED = "Feed"
TAG_TOP = "Top"
TAG_BOTTOM = "Bottom"
TAG_COLUMN = "SCOL-1"
TAG_R_COND = "Q_cond"     # Condenser duty (energy stream)
TAG_Q_REB = "Q_reb"       # Reboiler duty (energy stream)

# ==========================================
# 3. BASE OPERATING CONDITIONS (from DWSIM steady-state)
# ==========================================
# Mass flow rates (kg/h)
FLOW_FEED_BASE = 10000.0   # Feed to column
SPLIT_TOP = 0.35           # Fraction of feed that exits as distillate (Top)
SPLIT_BOTTOM = 0.65        # Fraction of feed that exits as bottoms (Bottom)

# Energy duties (kW) — from DWSIM diagram
Q_COND_BASE = 1207.87      # Condenser duty
Q_REB_BASE = 1524.29       # Reboiler duty

# Desired ethanol mole fraction in distillate (separation target)
TARGET_ETHANOL_TOP = 0.80   # 80 mol% ethanol in Top stream

# ==========================================
# 4. SENSOR UNCERTAINTIES (Standard Deviations)
# ==========================================
# Mass flow sensors (kg/h)
SIGMA_FEED = 350.0
SIGMA_TOP = 150.0
SIGMA_BOTTOM = 200.0

# Energy sensors (kW)
SIGMA_Q_COND = 40.0
SIGMA_Q_REB = 50.0

# ==========================================
# 5. DATA GENERATION SETTINGS
# ==========================================
N_POINTS = 100          # Number of hourly data points
WINDOW_SIZE = 5         # Moving average filter window
SEED = 42               # Random seed for reproducibility

# ==========================================
# 6. KPI THRESHOLDS
# ==========================================
MAX_MASS_BALANCE_ERROR = 1.0    # Max tolerable mass balance error (%)
MAX_ENERGY_BALANCE_ERROR = 2.0  # Max tolerable energy balance error (%)

# ==========================================
# 7. DWSIM AUTOMATION SETTINGS
# ==========================================
DWSIM_TIMEOUT = 120                          # Solver timeout in seconds
DWSIM_COMPOUNDS = ["Ethanol", "Water"]       # Compound order in the flowsheet