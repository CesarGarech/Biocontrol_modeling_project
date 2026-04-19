"""
Digital Twin — Sub-page 1: DWSIM Interactive Simulation
========================================================
Lets the user specify feed-stream conditions AND shortcut distillation-column
parameters (LK, HK, key compositions, reflux ratio, condenser/reboiler
pressures, tray height).

Pressing "🚀 Run Simulation" attempts to connect to DWSIM via DWSIMInterface;
if DWSIM is unavailable, the Fenske-Underwood-Gilliland analytic model is used.

Key bug fixes vs previous version
-----------------------------------
* Underwood θ now solved with Brent's method (scipy.optimize.brentq) instead
  of Newton-Raphson.  Newton collapsed to the boundary θ≈1 every time.
* The Underwood pinch-equation denominator (1−θ) is kept signed (negative
  when θ > 1).  The old code applied max(…, 1e-12) which replaced the correct
  negative value with ≈0, causing R_min → 2×10^10.
* ΔHvap and cp are in consistent units throughout (kJ/kmol).
  Q_cond = V [kmol/h] × ΔHvap [kJ/kmol] / 3600 [s/h] = kW  ✓
* DWSIM live path uses set_column_parameters() (correct API) instead of the
  removed set_equipment_property() generic method.
* N_stages and feed_tray are calculated (Fenske-Gilliland + Kirkbride), not
  user inputs — DWSIM ShortcutColumn does not accept them as specs.
* Added: condenser pressure, reboiler pressure, tray height (sidebar).

Unit conventions
-----------------
  Flows     kmol/h  (molar),  kg/h (mass)
  Energy    kW
  Enthalpy  kJ/kmol
  Pressure  bar (user), Pa (DWSIM API)
  Temp      °C (user), K (DWSIM API)
  cp        kJ/(kmol·°C)   [water ≈ 75.3, ethanol ≈ 112.4]
  ΔHvap     kJ/kmol        [water ≈ 40 650, ethanol ≈ 38 560]
"""
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq as _brentq

# ── Add Simulation folder to Python path ─────────────────────────────────────
_SIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Simulation")
)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
import config as _cfg  # noqa: E402

# ── Import DWSIM interface (safe import when pythonnet is unavailable) ────────
try:
    from dwsim_interface import DWSIMInterface, DWSIMInterfaceError  # noqa: E402
    from dwsim_data_generator import validate_dwsim_installation     # noqa: E402
    _DWSIM_IMPORTS_OK = True
except Exception:
    _DWSIM_IMPORTS_OK = False
    DWSIMInterface     = None           # type: ignore[assignment,misc]
    DWSIMInterfaceError = RuntimeError  # type: ignore[assignment,misc]

    def validate_dwsim_installation():  # type: ignore[misc]
        return False, "DWSIM modules could not be imported."

# ── Physical constants ────────────────────────────────────────────────────────
_MW_ETHANOL    = 46.068    # g/mol
_MW_WATER      = 18.015    # g/mol
_DHV_ETHANOL   = 38_560.0  # kJ/kmol
_DHV_WATER     = 40_650.0  # kJ/kmol
_CP_ETH_L      = 112.4     # kJ/(kmol·°C)  liquid
_CP_WAT_L      =  75.3     # kJ/(kmol·°C)  liquid
_ALPHA_ETH_H2O =   2.2     # relative volatility ethanol/water at ~1 atm

# ── Design-point constants from config ───────────────────────────────────────
_FLOW_FEED_BASE_KGH = _cfg.FLOW_FEED_BASE
_Q_COND_BASE_KW     = _cfg.Q_COND_BASE
_Q_REB_BASE_KW      = _cfg.Q_REB_BASE
_TARGET_ETHANOL_TOP = _cfg.TARGET_ETHANOL_TOP

# ── Sidebar defaults ──────────────────────────────────────────────────────────
_DEFAULT_LK              = "Ethanol"
_DEFAULT_HK              = "Water"
_DEFAULT_LK_BOT          = 0.02    # LK mole fraction in bottoms
_DEFAULT_HK_DIST         = 0.02    # HK mole fraction in distillate
_DEFAULT_RR_MULT         = 1.1     # R
_DEFAULT_P_COND_BAR      = 1.013   # bar  (atmospheric)
_DEFAULT_P_REB_BAR       = 1.10    # bar
_DEFAULT_TRAY_HEIGHT_M   = 0.60    # m
_CONDENSER_OPTIONS       = ["Total", "Partial"]


# ─────────────────────────────────────────────────────────────────────────────
# Mixture property helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mw(x: float) -> float:
    """Average MW [g/mol] for ethanol(x) – water(1-x) mixture."""
    return x * _MW_ETHANOL + (1.0 - x) * _MW_WATER

def _dhv(x: float) -> float:
    """Average ΔHvap [kJ/kmol] for ethanol(x) – water mixture."""
    return x * _DHV_ETHANOL + (1.0 - x) * _DHV_WATER

def _cp_l(x: float) -> float:
    """Average liquid cp [kJ/(kmol·°C)] for ethanol(x) – water mixture."""
    return x * _CP_ETH_L + (1.0 - x) * _CP_WAT_L


# ─────────────────────────────────────────────────────────────────────────────
# Fenske-Underwood-Gilliland + Kirkbride shortcut method
# ─────────────────────────────────────────────────────────────────────────────

def _underwood_feed_eq(theta: float, alpha: float, z_lk: float, q: float) -> float:
    """
    Underwood feed equation (binary):
        f(θ) = α·z_LK/(α−θ) + z_HK/(1−θ) − (1−q)

    Root lies strictly in (1, α).
    IMPORTANT: denominator (1−θ) is kept signed — it is NEGATIVE when θ > 1,
    which is the correct physical value.  Never apply max(…, ε) here.
    """
    return (
        alpha * z_lk / (alpha - theta)
        + (1.0 - z_lk) / (1.0 - theta)
        - (1.0 - q)
    )


def _underwood_theta(alpha: float, z_lk: float, q: float) -> float:
    """
    Solve the Underwood feed equation for θ ∈ (1, α) using Brent's method.

    Brent is used instead of Newton-Raphson because the function has poles
    at both boundaries (θ = 1 and θ = α), making Newton diverge to the
    boundary on nearly every starting guess.

    For q ≈ 1 (saturated-liquid feed) a tiny perturbation is applied to
    avoid the removable singularity.
    """
    if abs(q - 1.0) < 1e-6:
        q = 1.0 + 1e-6
    lo, hi = 1.0 + 1e-9, alpha - 1e-9
    try:
        return _brentq(
            _underwood_feed_eq, lo, hi,
            args=(alpha, z_lk, q),
            xtol=1e-12, rtol=1e-12, maxiter=500,
        )
    except ValueError:
        return (lo + hi) / 2.0


def _r_min_underwood(alpha: float, z_lk: float, x_d_lk: float, q: float) -> float:
    """
    Minimum reflux ratio (Underwood pinch, binary):
        R_min + 1 = α·x_D,LK/(α−θ) + x_D,HK/(1−θ)

    The denominator (1−θ) is NEGATIVE when θ > 1 — this is correct and must
    NOT be replaced by a positive ε (that causes R_min → ∞).
    """
    theta   = _underwood_theta(alpha, z_lk, q)
    x_d_hk  = 1.0 - x_d_lk
    rmin_p1 = (
        alpha * x_d_lk / (alpha - theta)
        + x_d_hk / (1.0 - theta)          # signed denominator — correct
    )
    return max(rmin_p1 - 1.0, 0.01)


def _fenske_nmin(alpha: float, x_d_lk: float, x_b_lk: float) -> float:
    """Fenske minimum stages at total reflux (binary)."""
    x_d_hk = max(1.0 - x_d_lk, 1e-9)
    x_b_hk = max(1.0 - x_b_lk, 1e-9)
    return np.log(
        (x_d_lk / x_d_hk) * (x_b_hk / max(x_b_lk, 1e-9))
    ) / np.log(alpha)


def _gilliland_n(n_min: float, r_min: float, r: float) -> float:
    """
    Gilliland correlation — Molokanov (1972) approximation.
        X = (R − R_min)/(R + 1)
        Y = 1 − exp[(1 + 54.4X)/(11 + 117.2X) · (X−1)/√X]
        N = (N_min + Y)/(1 − Y)
    """
    x = float(np.clip((r - r_min) / (r + 1.0), 1e-6, 1.0 - 1e-6))
    y = float(np.clip(
        1.0 - np.exp(((1.0 + 54.4 * x) / (11.0 + 117.2 * x)) * (x - 1.0) / x ** 0.5),
        1e-6, 1.0 - 1e-6,
    ))
    return (n_min + y) / (1.0 - y)


def _kirkbride_feed_tray(n_total: float,
                         z_hk: float, z_lk: float,
                         x_b_lk: float, x_d_hk: float,
                         B: float, D: float) -> int:
    """
    Kirkbride (1944) equation for optimal feed tray location.
        (N_rect / N_strip)² = (z_HK/z_LK) · (x_B,LK / x_D,HK)² · (B/D)
    Returns tray number counted from the top (1-based).
    """
    ratio_sq = (
        (z_hk / max(z_lk, 1e-9))
        * (x_b_lk / max(x_d_hk, 1e-9)) ** 2
        * (B / max(D, 1e-9))
    )
    ratio   = ratio_sq ** 0.5           # N_rect / N_strip
    n_rect  = ratio / (1.0 + ratio) * n_total
    return max(1, min(int(round(n_rect)), int(n_total) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Analytic simulation (FUG)
# ─────────────────────────────────────────────────────────────────────────────

def _run_analytic_simulation(
    F_feed_kmolh: float,
    T_feed_C:     float,
    P_feed_bar:   float,
    x_eth:        float,
    lk_bot:       float,
    hk_dist:      float,
    rr_multiplier: float,
    P_cond_bar:   float,
    P_reb_bar:    float,
    tray_height_m: float,
    condenser_type: str,
) -> dict:
    """
    Fenske-Underwood-Gilliland shortcut distillation model.

    Separation specs
    ----------------
    lk_bot  : mole fraction of LK (Ethanol) allowed in the bottoms
    hk_dist : mole fraction of HK (Water)   allowed in the distillate
    These are the same specs accepted by the DWSIM ShortcutColumn.
    """
    alpha = _ALPHA_ETH_H2O
    z_lk  = x_eth
    z_hk  = 1.0 - x_eth
    F     = F_feed_kmolh

    # ── Derive distillate/bottoms compositions from specs ─────────────────────
    x_d_lk = 1.0 - hk_dist   # = x_D,Ethanol
    x_d_hk = hk_dist
    x_b_lk = lk_bot
    x_b_hk = 1.0 - lk_bot

    # ── Overall material balance (two-component) ──────────────────────────────
    # [x_d_lk  x_b_lk] [D]   [F·z_lk]
    # [x_d_hk  x_b_hk] [B] = [F·z_hk]
    A_mat = np.array([[x_d_lk, x_b_lk], [x_d_hk, x_b_hk]])
    b_vec = np.array([F * z_lk, F * z_hk])
    try:
        D, B = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        D = F * z_lk / max(x_d_lk, 1e-9)
        B = F - D
    D = max(float(D), 1e-6)
    B = max(float(B), 1e-6)

    # ── Feed condition q ──────────────────────────────────────────────────────
    # Approximate bubble-point temperature at feed pressure (Clausius-Clapeyron)
    T_bp  = 78.4 + (P_feed_bar * 100.0 - 101.325) * 0.04   # °C
    lam_f = _dhv(z_lk)
    cp_f  = _cp_l(z_lk)
    q = 1.0 + cp_f * max(T_bp - T_feed_C, 0.0) / lam_f

    # ── Fenske N_min ──────────────────────────────────────────────────────────
    n_min = max(_fenske_nmin(alpha, x_d_lk, x_b_lk), 1.0)

    # ── Underwood R_min ───────────────────────────────────────────────────────
    R_min = _r_min_underwood(alpha, z_lk, x_d_lk, q)

    # ── Actual reflux → Gilliland N ───────────────────────────────────────────
    R = rr_multiplier
    N = _gilliland_n(n_min, R_min, R)
    if condenser_type == "Partial":
        N = max(N - 1.0, 1.0)
    N_int = max(int(np.ceil(N)), 2)

    # ── Kirkbride feed tray ───────────────────────────────────────────────────
    feed_tray = _kirkbride_feed_tray(N_int, z_hk, z_lk, x_b_lk, x_d_hk, B, D)

    # ── Column height ─────────────────────────────────────────────────────────
    col_height = N_int * tray_height_m

    # ── Stream mass flows ─────────────────────────────────────────────────────
    F_kgh = F * _mw(z_lk)
    D_kgh = D * _mw(x_d_lk)
    B_kgh = B * _mw(x_b_lk)

    # ── Internal vapour/liquid rates ──────────────────────────────────────────
    V_rect  = D * (R + 1.0)          # kmol/h  rectifying vapour
    L_rect  = R * D                  # kmol/h  rectifying liquid
    V_strip = V_rect - F * (1.0 - q) # kmol/h  stripping vapour

    # ── Energy balance ────────────────────────────────────────────────────────
    # Q_cond = V_rect [kmol/h] × ΔHvap_dist [kJ/kmol] / 3600 [s/h]  → kW
    Q_cond = V_rect  * _dhv(x_d_lk) / 3600.0
    Q_reb  = max(V_strip * _dhv(x_b_lk) / 3600.0, Q_cond * 0.90)

    # ── Approximate stream temperatures ──────────────────────────────────────
    T_top = 78.4 + x_d_hk * 21.6 + (P_cond_bar * 100.0 - 101.325) * 0.04
    T_bot = 78.4 + x_b_hk * 21.6 + (P_reb_bar  * 100.0 - 101.325) * 0.04

    scale = F_kgh / max(_FLOW_FEED_BASE_KGH, 1.0)
    eth_recovery = D * x_d_lk / max(F * z_lk, 1e-9) * 100.0

    return {
        "source": "analytic",
        "F_feed_kmolh": F,     "F_feed_kgh": F_kgh,
        "F_top_kmolh":  D,     "F_top_kgh":  D_kgh,
        "F_bot_kmolh":  B,     "F_bot_kgh":  B_kgh,
        "T_feed_C": T_feed_C,  "T_top_C": T_top, "T_bot_C": T_bot,
        "P_feed_bar": P_feed_bar, "P_cond_bar": P_cond_bar, "P_reb_bar": P_reb_bar,
        "Q_cond_kw": Q_cond,   "Q_reb_kw": Q_reb,
        "reflux_ratio":     R,
        "reflux_ratio_min": R_min,
        "rr_multiplier":    rr_multiplier,
        "n_min":            n_min,
        "n_stages":         float(N_int),
        "feed_tray":        feed_tray,
        "tray_height_m":    tray_height_m,
        "column_height_m":  col_height,
        "condenser_type":   condenser_type,
        "x_eth_feed": z_lk,   "x_eth_top": x_d_lk, "x_eth_bot": x_b_lk,
        "x_wat_feed": z_hk,   "x_wat_top": x_d_hk, "x_wat_bot": x_b_hk,
        "q":              q,
        "V_kmolh":        V_rect,
        "L_kmolh":        L_rect,
        "mw_feed":        _mw(z_lk),
        "mw_top":         _mw(x_d_lk),
        "mw_bot":         _mw(x_b_lk),
        "scale":          scale,
        "ethanol_recovery": eth_recovery,
        "lk_bot":  lk_bot,
        "hk_dist": hk_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DWSIM live simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_dwsim_simulation(
    F_feed_kmolh: float, T_feed_C: float, P_feed_bar: float,
    x_eth: float, x_water: float,
    lk: str, hk: str,
    lk_bot: float, hk_dist: float,
    rr_multiplier: float,
    P_cond_bar: float, P_reb_bar: float,
    tray_height_m: float, condenser_type: str,
) -> dict:
    """
    Run a live DWSIM shortcut-column simulation.

    N_stages and feed_tray are calculated analytically (FUG + Kirkbride) and
    reported as output — the DWSIM ShortcutColumn does not accept them as
    input specifications.

    DWSIM column specs used:
      • LightKeyCompound  / HeavyKeyCompound
      • LKMoleFractionInBottoms  / HKMoleFractionInDistillate
      • RefluxRatio  (actual R = β × R_min, computed from FUG before call)
      • CondenserPressure / ReboilerPressure  (Pa)
    """
    _KGS_TO_KGH   = 3600.0
    _MOLS_TO_KMOLH = 3.6

    # Pre-compute actual R from FUG so DWSIM receives R, not the multiplier β
    _a = _run_analytic_simulation(
        F_feed_kmolh, T_feed_C, P_feed_bar, x_eth,
        lk_bot=lk_bot, hk_dist=hk_dist, rr_multiplier=rr_multiplier,
        P_cond_bar=P_cond_bar, P_reb_bar=P_reb_bar,
        tray_height_m=tray_height_m, condenser_type=condenser_type,
    )
    R_actual = _a["reflux_ratio"]

    with DWSIMInterface(_cfg.DWSIM_INSTALL_PATH) as dwsim:
        dwsim.load_simulation(_cfg.SIMULATION_FILE)

        # Feed stream
        dwsim.set_stream_conditions(
            _cfg.TAG_FEED,
            molar_flow=F_feed_kmolh, temperature=T_feed_C, pressure=P_feed_bar,
            composition={"Ethanol": x_eth, "Water": x_water},
        )

        # Shortcut column — using the dedicated API (correct property names)
        dwsim.set_column_parameters(
            _cfg.TAG_COLUMN,
            light_key=lk, heavy_key=hk,
            lk_bottoms=lk_bot, hk_distillate=hk_dist,
            reflux_ratio=R_actual,
            condenser_pressure_bar=P_cond_bar,
            reboiler_pressure_bar=P_reb_bar,
        )

        dwsim.run_simulation()

        # Read stream results
        def _mfl(tag): return dwsim.get_stream_property(tag, "MassFlow")  * _KGS_TO_KGH
        def _nfl(tag): return dwsim.get_stream_property(tag, "MolarFlow") * _MOLS_TO_KMOLH
        def _T(tag):   return dwsim.get_stream_property(tag, "Temperature") - 273.15

        F_kgh  = _mfl(_cfg.TAG_FEED);  D_kgh  = _mfl(_cfg.TAG_TOP);  B_kgh  = _mfl(_cfg.TAG_BOTTOM)
        F_kmolh = _nfl(_cfg.TAG_FEED); D_kmolh = _nfl(_cfg.TAG_TOP); B_kmolh = _nfl(_cfg.TAG_BOTTOM)
        T_f = _T(_cfg.TAG_FEED); T_t = _T(_cfg.TAG_TOP); T_b = _T(_cfg.TAG_BOTTOM)
        P_f = dwsim.get_stream_property(_cfg.TAG_FEED, "Pressure") * 1e-5

        x_eth_f = dwsim.get_stream_property(_cfg.TAG_FEED,   "MoleFraction", "Ethanol")
        x_eth_t = dwsim.get_stream_property(_cfg.TAG_TOP,    "MoleFraction", "Ethanol")
        x_eth_b = dwsim.get_stream_property(_cfg.TAG_BOTTOM, "MoleFraction", "Ethanol")

        q_cond = abs(dwsim.get_equipment_property(_cfg.TAG_R_COND, "Duty"))
        q_reb  = abs(dwsim.get_equipment_property(_cfg.TAG_Q_REB,  "Duty")) 

        # Column properties
        try:
            N      = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "NumberOfStages")
        except:
            N = None

        try:
            R_min  = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "MinimumRefluxRatio")
        except:
            R_min = None
        # Opcionales (pueden fallar según versión)
        try:
            N_min = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "MinimumNumberOfStages")
        except:
            N_min = None

        try:
            feed_stage = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "FeedStage")
        except:
            feed_stage = None

        try:
            estimated_height = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "EstimatedHeight")
        except:
            estimated_height = None

        try:
            rr_out = dwsim.get_equipment_property(_cfg.TAG_COLUMN, "RefluxRatio")
        except Exception:
            rr_out = R_actual

    scale = F_kgh / max(_FLOW_FEED_BASE_KGH, 1.0)
    eth_rec = D_kmolh * x_eth_t / max(F_kmolh * x_eth_f, 1e-9) * 100.0

    return {
        "source": "dwsim",
        "F_feed_kmolh": F_kmolh, "F_feed_kgh": F_kgh,
        "F_top_kmolh":  D_kmolh, "F_top_kgh":  D_kgh,
        "F_bot_kmolh":  B_kmolh, "F_bot_kgh":  B_kgh,
        "T_feed_C": T_f, "T_top_C": T_t, "T_bot_C": T_b,
        "P_feed_bar": P_f, "P_cond_bar": P_cond_bar, "P_reb_bar": P_reb_bar,
        "Q_cond_kw": q_cond, "Q_reb_kw": q_reb,
        "reflux_ratio":     R_actual,
        "reflux_ratio_min": R_min if R_min else _a["reflux_ratio_min"],
        "rr_multiplier":    rr_multiplier,
        "n_min":            N_min if N_min else _a["n_min"],
        "n_stages":         N if N else _a["n_stages"],
        "feed_tray":        feed_stage if feed_stage else _a["feed_tray"],
        "tray_height_m":    tray_height_m,
        "column_height_m":  estimated_height if estimated_height else _a["column_height_m"],
        "condenser_type":   condenser_type,
        "x_eth_feed": x_eth_f, "x_eth_top": x_eth_t, "x_eth_bot": x_eth_b,
        "x_wat_feed": 1-x_eth_f, "x_wat_top": 1-x_eth_t, "x_wat_bot": 1-x_eth_b,
        "q":    _a["q"],
        "V_kmolh": _a["V_kmolh"], "L_kmolh": _a["L_kmolh"],
        "mw_feed": F_kgh/max(F_kmolh,1e-9),
        "mw_top":  D_kgh/max(D_kmolh,1e-9),
        "mw_bot":  B_kgh/max(B_kmolh,1e-9),
        "scale":   scale,
        "ethanol_recovery": eth_rec,
        "lk_bot":  lk_bot, "hk_dist": hk_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page
# ─────────────────────────────────────────────────────────────────────────────

def simulacion_dwsim_page():
    """Main page for the interactive DWSIM simulator."""
    st.header("⚙️ DWSIM Interactive Simulation")
    st.markdown(
        "Configure **feed** conditions and **shortcut column** parameters in the "
        "sidebar, then press **🚀 Run Simulation**.  "
        "DWSIM is used when available; otherwise the "
        "**Fenske-Underwood-Gilliland** analytic model runs as fallback."
    )

    dwsim_ok, dwsim_msg = validate_dwsim_installation()
    if dwsim_ok:
        st.success(f"✅ DWSIM available — {dwsim_msg}")
    else:
        st.warning(f"⚠️ DWSIM not available (F-U-G analytic fallback).  Detail: {dwsim_msg}")

    st.markdown("---")

    # Design reference banner
    st.subheader("Design Reference")
    st.caption(
        f"Flowsheet: `ethanol.dwxmz` | Column: **{_cfg.TAG_COLUMN}** "
        f"| Feed: `{_cfg.TAG_FEED}` | Top: `{_cfg.TAG_TOP}` | Bottom: `{_cfg.TAG_BOTTOM}` "
        f"| Cond energy: `{_cfg.TAG_R_COND}` | Reb energy: `{_cfg.TAG_Q_REB}`"
    )
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Base feed flow",   f"{_FLOW_FEED_BASE_KGH:,.0f} kg/h")
    dc2.metric("Condenser (base)", f"{_Q_COND_BASE_KW:,.1f} kW")
    dc3.metric("EtOH target top",  f"{_TARGET_ETHANOL_TOP*100:.0f} mol%")
    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Input Parameters")

        # 1 — Feed stream
        with st.expander("1. Feed Stream", expanded=True):
            F_feed_kmolh = st.number_input(
                "Molar flow (kmol/h)", 1.0, 5_000.0,
                float(_cfg.DEFAULT_FEED_CONDITIONS["molar_flow"]), 5.0, key="F")
            T_feed_C = st.number_input(
                "Temperature (°C)", 20.0, 150.0,
                float(_cfg.DEFAULT_FEED_CONDITIONS["temperature"]), 1.0, key="T")
            P_feed_bar = st.number_input(
                "Pressure (bar)", 0.5, 50.0,
                float(_cfg.DEFAULT_FEED_CONDITIONS["pressure"]), 0.5, key="P")

        # 2 — Composition
        with st.expander("2. Molar Composition", expanded=True):
            x_eth = st.slider(
                "Ethanol (mole fraction)", 0.01, 0.99,
                float(_cfg.DEFAULT_FEED_CONDITIONS["composition"]["Ethanol"]),
                0.01, key="xeth")
            x_water = round(1.0 - x_eth, 4)
            st.markdown(f"**Water: `{x_water:.4f}`** *(auto)*")

        # 3 — Key components & purity specs
        with st.expander("3. Key Components & Purity Specs", expanded=True):
            st.markdown(
                "DWSIM ShortcutColumn separation specs:  \n"
                "• **x_B,LK** — LK mole fraction in bottoms  \n"
                "• **x_D,HK** — HK mole fraction in distillate"
            )
            _COMPS = ["Ethanol", "Water"]
            lk = st.selectbox("Light Key (LK)", _COMPS,
                                _COMPS.index(_DEFAULT_LK), key="lk")
            hk = st.selectbox("Heavy Key (HK)", [c for c in _COMPS if c != lk],
                                0, key="hk")
            lk_bot = st.number_input(
                f"x_B,LK  ({lk} in bottoms)",
                0.0001, 0.20, _DEFAULT_LK_BOT, 0.005, format="%.4f",
                key="lk_bot",
                help="Lower → purer bottoms; more stages / energy required.")
            hk_dist = st.number_input(
                f"x_D,HK  ({hk} in distillate)",
                0.0001, 0.20, _DEFAULT_HK_DIST, 0.005, format="%.4f",
                key="hk_dist",
                help="Lower → purer distillate; more stages / energy required.")

        # 4 — Reflux ratio
        with st.expander("4. Reflux Ratio", expanded=True):
            rr_multiplier = st.number_input(
                "R ", 1.01, 10.0,
                _DEFAULT_RR_MULT, 0.05, key="rr_mult")

        # 5 — Column pressures
        with st.expander("5. Column Pressures", expanded=True):
            P_cond_bar = st.number_input(
                "Condenser pressure (bar)", 0.1, 20.0,
                _DEFAULT_P_COND_BAR, 0.05, format="%.3f", key="Pc",
                help="Atmospheric ≈ 1.013 bar.  Vacuum operation < 1 bar.")
            P_reb_bar = st.number_input(
                "Reboiler pressure (bar)", 0.1, 20.0,
                _DEFAULT_P_REB_BAR, 0.05, format="%.3f", key="Pr",
                help="Should be ≥ condenser pressure (column ΔP).")
            if P_reb_bar < P_cond_bar - 1e-3:
                st.warning("⚠️ Reboiler P should be ≥ condenser P.")

        # 6 — Tray geometry
        with st.expander("6. Tray / Stage Height", expanded=False):
            tray_height_m = st.number_input(
                "Stage height (m)", 0.20, 1.50,
                _DEFAULT_TRAY_HEIGHT_M, 0.05, key="tray_h",
                help="Sieve tray: 0.45–0.75 m.  Structured packing HETP: 0.3–0.6 m.")

        # 7 — Condenser type
        with st.expander("7. Condenser Type", expanded=False):
            condenser_type = st.radio(
                "Condenser type", _CONDENSER_OPTIONS,
                _CONDENSER_OPTIONS.index("Total"),
                horizontal=True, key="cond_type",
                help="Total: liquid distillate.  Partial: vapour distillate (−1 stage).")

    # ─────────────────────────────────────────────────────────────────────────
    # Run button
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("Configure parameters in the sidebar then run:")
    run_clicked = st.button("🚀 Run Simulation", key="btn_run", type="primary")

    if run_clicked:
        errors = []
        if P_reb_bar < P_cond_bar - 1e-3:
            errors.append("Reboiler pressure must be ≥ condenser pressure.")
        if lk_bot + (1.0 - hk_dist) < 0.01:
            errors.append("Separation specs are infeasible (compositions don't leave room for separation).")
        if errors:
            for e in errors: st.error(f"❌ {e}")
        else:
            st.session_state.pop("sim_results", None)
            kw = dict(
                F_feed_kmolh=F_feed_kmolh, T_feed_C=T_feed_C, P_feed_bar=P_feed_bar,
                x_eth=x_eth,
                lk_bot=lk_bot, hk_dist=hk_dist, rr_multiplier=rr_multiplier,
                P_cond_bar=P_cond_bar, P_reb_bar=P_reb_bar,
                tray_height_m=tray_height_m, condenser_type=condenser_type,
            )
            if dwsim_ok and _DWSIM_IMPORTS_OK:
                with st.spinner("Connecting to DWSIM…"):
                    try:
                        results = _run_dwsim_simulation(
                            **kw, x_water=x_water, lk=lk, hk=hk)
                        st.success("✅ DWSIM completed.")
                    except Exception as exc:
                        st.warning(f"⚠️ DWSIM failed ({exc}).  Using F-U-G fallback.")
                        results = _run_analytic_simulation(**kw)
            else:
                with st.spinner("Running Fenske-Underwood-Gilliland model…"):
                    results = _run_analytic_simulation(**kw)
                st.info("ℹ️ Analytic F-U-G results (DWSIM not available).")

            st.session_state["sim_results"] = results

    # ─────────────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────────────
    if "sim_results" not in st.session_state:
        st.info("👆 Set parameters in the sidebar and press **🚀 Run Simulation**.")
        return

    r = st.session_state["sim_results"]
    src = ("🔬 DWSIM (live)" if r.get("source") == "dwsim"
           else "📐 Fenske-Underwood-Gilliland (analytic)")
    st.caption(f"Source: **{src}**")
    st.subheader("📊 Simulation Results")

    # Column parameters summary
    st.markdown("#### Column Parameters")
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    p1.metric("Condenser type",       r.get("condenser_type","—"))
    p2.metric("Theoretical stages",   f"{r['n_stages']:.0f}")
    p3.metric("N_min (Fenske)",        f"{r.get('n_min',0):.1f}")
    p4.metric("Feed tray",  f"{r['feed_tray']:.2f}")
    p5.metric("Column height",         f"{r.get('column_height_m',0):.1f} m")
    p6.metric("Stage/tray height",     f"{r.get('tray_height_m',0):.2f} m")

    # Stream table
    st.markdown("#### Stream Table")
    st.dataframe(pd.DataFrame({
        "Stream":              ["Feed", "Distillate", "Bottoms"],
        "Molar Flow (kmol/h)": [f"{r['F_feed_kmolh']:.2f}", f"{r['F_top_kmolh']:.2f}", f"{r['F_bot_kmolh']:.2f}"],
        "Mass Flow (kg/h)":    [f"{r['F_feed_kgh']:,.1f}",  f"{r['F_top_kgh']:,.1f}",  f"{r['F_bot_kgh']:,.1f}"],
        "Temp (°C)":           [f"{r['T_feed_C']:.1f}",     f"{r['T_top_C']:.1f}",     f"{r['T_bot_C']:.1f}"],
        "Pressure (bar)":      [f"{r['P_feed_bar']:.3f}",   f"{r['P_cond_bar']:.3f}",  f"{r['P_reb_bar']:.3f}"],
        "x Ethanol":           [f"{r['x_eth_feed']:.4f}",   f"{r['x_eth_top']:.4f}",   f"{r['x_eth_bot']:.4f}"],
        "x Water":             [f"{r['x_wat_feed']:.4f}",   f"{r['x_wat_top']:.4f}",   f"{r['x_wat_bot']:.4f}"],
    }), use_container_width=True)

    # Equipment table
    st.markdown("#### Equipment Summary")
    eq = {
        "Condenser duty (kW)":          f"{r['Q_cond_kw']:.2f}",
        "Reboiler duty (kW)":           f"{r['Q_reb_kw']:.2f}",
        "Condenser pressure (bar)":     f"{r['P_cond_bar']:.3f}",
        "Reboiler pressure (bar)":      f"{r['P_reb_bar']:.3f}",
        "Actual reflux ratio R (L/D)":  f"{r['reflux_ratio']:.4f}",
        "Minimum reflux ratio R_min":   f"{r['reflux_ratio_min']:.4f}",
        "R / R_min (β)":                f"{r['rr_multiplier']:.3f}×",
        "N_min — Fenske":               f"{r.get('n_min',0):.1f}",
        "Theoretical stages N":         f"{r['n_stages']:.0f}",
        "Feed tray":         f"{r['feed_tray']:.2f}",
        "Stage height (m)":             f"{r.get('tray_height_m',0):.2f}",
        "Column height (m)":            f"{r.get('column_height_m',0):.1f}",
        "Condenser type":               r.get("condenser_type","—"),
        "q-parameter":                  f"{r.get('q',0):.3f}",
        "Ethanol recovery (%)":         f"{min(r['ethanol_recovery'],100.0):.2f}",
    }
    st.dataframe(pd.DataFrame({"Parameter": list(eq.keys()), "Value": list(eq.values())}),
                 use_container_width=True)

    # Key metrics
    st.markdown("#### Key Metrics")
    sc = r["scale"]
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Condenser",   f"{r['Q_cond_kw']:.1f} kW", delta=f"{(sc-1)*100:+.1f}% vs design")
    m2.metric("Reboiler",    f"{r['Q_reb_kw']:.1f} kW",  delta=f"{(sc-1)*100:+.1f}% vs design")
    m3.metric("Reflux R",    f"{r['reflux_ratio']:.3f}")
    m4.metric("EtOH recovery", f"{min(r['ethanol_recovery'],100.0):.1f}%")
    m5,m6,m7,m8 = st.columns(4)
    m5.metric("Stages N",        f"{r['n_stages']:.0f}")
    m6.metric("Feed tray",       f"{r['feed_tray']:.2f}")
    m7.metric("Column height",   f"{r.get('column_height_m',0):.1f} m")
    m8.metric("q-parameter",     f"{r.get('q',0):.3f}",
              help="1=sat.liq | >1=subcooled | <1=part.vapour")

    st.markdown("---")

    # Charts
    st.markdown("#### Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    # 1 — Mass flows
    axes[0].barh(["Feed","Distillate","Bottoms"],
                 [r["F_feed_kgh"],r["F_top_kgh"],r["F_bot_kgh"]],
                 color=["#4C72B0","#55A868","#C44E52"])
    axes[0].set_xlabel("Flow (kg/h)"); axes[0].set_title("Stream Mass Flows")
    for i,v in enumerate([r["F_feed_kgh"],r["F_top_kgh"],r["F_bot_kgh"]]):
        axes[0].text(v*1.01, i, f"{v:,.0f}", va="center", fontsize=8)

    # 2 — Compositions
    xp = np.arange(3); bw = 0.35
    axes[1].bar(xp-bw/2,[r["x_eth_feed"],r["x_eth_top"],r["x_eth_bot"]],bw,label="Ethanol",color="#55A868")
    axes[1].bar(xp+bw/2,[r["x_wat_feed"],r["x_wat_top"],r["x_wat_bot"]],bw,label="Water",  color="#4C72B0")
    axes[1].set_xticks(xp); axes[1].set_xticklabels(["Feed","Distillate","Bottoms"])
    axes[1].set_ylabel("Mole Fraction"); axes[1].set_title("Stream Compositions")
    axes[1].legend(fontsize=8); axes[1].set_ylim(0,1.1)

    # 3 — Energy
    axes[2].bar(["Condenser","Reboiler"],[r["Q_cond_kw"],r["Q_reb_kw"]],
                color=["#8172B2","#CCB974"],width=0.4)
    axes[2].set_ylabel("Duty (kW)"); axes[2].set_title("Energy Balance")
    for i,v in enumerate([r["Q_cond_kw"],r["Q_reb_kw"]]):
        axes[2].text(i, v*1.01, f"{v:.1f}", ha="center", fontsize=9)

    # Equations reference
    with st.expander("📐 Analytic Method — Fenske-Underwood-Gilliland-Kirkbride"):
        st.markdown("**Material balance from purity specs:**")
        st.latex(r"\begin{bmatrix}x_{D,LK}&x_{B,LK}\\x_{D,HK}&x_{B,HK}\end{bmatrix}"
                 r"\begin{bmatrix}D\\B\end{bmatrix}="
                 r"\begin{bmatrix}Fz_{LK}\\Fz_{HK}\end{bmatrix}")
        st.markdown("**Underwood θ** (Brent root in (1, α)):")
        st.latex(r"\frac{\alpha z_{LK}}{\alpha-\theta}+\frac{z_{HK}}{1-\theta}=1-q"
                 r"\quad \theta\in(1,\alpha)")
        st.markdown("**Minimum reflux:**")
        st.latex(r"R_{min}+1=\frac{\alpha x_{D,LK}}{\alpha-\theta}+\frac{x_{D,HK}}{1-\theta}")
        st.markdown("**Fenske N_min:**")
        st.latex(r"N_{min}=\frac{\ln\!\left[\dfrac{x_{D,LK}}{x_{D,HK}}\cdot\dfrac{x_{B,HK}}{x_{B,LK}}\right]}{\ln\alpha}")
        st.markdown("**Gilliland N** (Molokanov):")
        st.latex(r"X=\frac{R-R_{min}}{R+1},\;"
                 r"Y=1-\exp\!\left[\frac{1+54.4X}{11+117.2X}\cdot\frac{X-1}{\sqrt{X}}\right],"
                 r"\;N=\frac{N_{min}+Y}{1-Y}")
        st.markdown("**Condenser duty:**")
        st.latex(r"Q_C=\frac{V\cdot\Delta H_{vap,D}}{3600}\;[\text{kW}]"
                 r"\quad V=D(R+1)\;[\text{kmol/h}],\;\Delta H_{vap}\;[\text{kJ/kmol}]")
        st.markdown("**Kirkbride feed tray:**")
        st.latex(r"\left(\frac{N_{rect}}{N_{strip}}\right)^2"
                 r"=\frac{z_{HK}}{z_{LK}}\left(\frac{x_{B,LK}}{x_{D,HK}}\right)^2\frac{B}{D}")