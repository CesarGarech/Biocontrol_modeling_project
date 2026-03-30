"""
Unit tests for the Digital Twin — Distillation Column module.

Tests cover:
- IQR outlier detection and removal
- Moving average and low-pass Butterworth filters
- Excel parsing and schema validation
- WLS data reconciliation
"""

import io
import sys
import os

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Body.digital_twin.gemelo_digital import (
    BALANCE_VARS,
    REQUIRED_COLUMNS,
    apply_lowpass_filter,
    apply_moving_average,
    build_distillation_constraints,
    compute_kpis,
    compute_outlier_summary,
    create_example_excel,
    detect_outliers_iqr,
    filter_dataframe,
    load_excel_sensor_data,
    reconcile_data_wls,
    remove_outliers_iqr,
    simulate_distillation_column,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_df():
    """Small synthetic dataframe for testing (no noise, deterministic)."""
    return simulate_distillation_column(n_points=80, noise_level=0.04, seed=7)


@pytest.fixture
def clean_series():
    """Clean sinusoidal signal with a few injected outliers."""
    t = np.linspace(0, 10 * np.pi, 200)
    s = np.sin(t)
    s_dirty = s.copy()
    s_dirty[[10, 50, 100, 150]] = [5.0, -6.0, 4.5, -5.5]
    return pd.Series(s_dirty)


# ===========================================================================
# 1. Mock Simulation
# ===========================================================================


class TestSimulateDistillationColumn:
    def test_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)

    def test_required_columns_present(self, sample_df):
        for col in REQUIRED_COLUMNS:
            assert col in sample_df.columns, f"Column '{col}' missing"

    def test_twin_columns_present(self, sample_df):
        for col in REQUIRED_COLUMNS[1:]:  # skip Time
            assert "twin_" + col in sample_df.columns

    def test_n_points(self):
        df = simulate_distillation_column(n_points=50)
        assert len(df) == 50

    def test_no_nans_in_required(self, sample_df):
        for col in REQUIRED_COLUMNS:
            assert sample_df[col].isna().sum() == 0, f"NaN found in '{col}'"

    def test_mass_balance_twin(self, sample_df):
        """Twin values should satisfy F = D + B within numerical tolerance."""
        residual = sample_df["twin_F"] - sample_df["twin_D"] - sample_df["twin_B"]
        assert (residual.abs() < 1e-8).all(), "Twin mass balance violated"

    def test_compositions_in_range(self, sample_df):
        for col in ["zF", "xD", "xB"]:
            assert sample_df[col].between(0, 1).all(), f"Composition '{col}' out of [0,1]"


# ===========================================================================
# 2. IQR Outlier Detection
# ===========================================================================


class TestOutlierDetectionIQR:
    def test_detects_injected_outliers(self):
        """Known outliers should be flagged."""
        n = 100
        rng = np.random.default_rng(0)
        data = rng.normal(50, 2, n)
        # Inject 3 clear outliers
        data[[10, 40, 80]] = [200.0, -100.0, 300.0]
        df = pd.DataFrame({"x": data})
        result = detect_outliers_iqr(df, ["x"], k=1.5)
        assert result["counts"]["x"] >= 3

    def test_no_false_positives_clean_data(self):
        """Nearly no outliers in perfectly normal data with large k."""
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, 200)
        df = pd.DataFrame({"x": data})
        result = detect_outliers_iqr(df, ["x"], k=3.0)
        # Very few (or zero) outliers expected
        assert result["counts"]["x"] <= 5

    def test_bounds_structure(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float)})
        result = detect_outliers_iqr(df, ["x"])
        assert "x" in result["bounds"]
        lo, hi = result["bounds"]["x"]
        assert lo < hi

    def test_mask_shape(self, sample_df):
        result = detect_outliers_iqr(sample_df, ["F", "D"], k=1.5)
        assert result["mask"].shape == (len(sample_df), 2)

    def test_remove_replaces_outliers_with_valid(self, sample_df):
        result = detect_outliers_iqr(sample_df, ["F"], k=1.5)
        df_clean = remove_outliers_iqr(sample_df, result, ["F"])
        assert df_clean["F"].isna().sum() == 0, "NaN values not filled after removal"
        lo, hi = result["bounds"]["F"]
        # After removal, values should generally be within a wider range
        assert df_clean["F"].between(lo * 0.5, hi * 2).all()

    def test_k_parameter_effect(self, sample_df):
        """Higher k → fewer outliers detected."""
        res15 = detect_outliers_iqr(sample_df, ["F"], k=1.5)
        res30 = detect_outliers_iqr(sample_df, ["F"], k=3.0)
        assert res15["counts"]["F"] >= res30["counts"]["F"]


# ===========================================================================
# 3. Signal Filtering
# ===========================================================================


class TestMovingAverage:
    def test_output_length(self, clean_series):
        out = apply_moving_average(clean_series, window=5)
        assert len(out) == len(clean_series)

    def test_reduces_outlier_magnitude(self, clean_series):
        """Moving average should damp injected outliers."""
        filtered = apply_moving_average(clean_series, window=9)
        assert filtered.abs().max() < clean_series.abs().max()

    def test_no_nans_after_filtering(self, clean_series):
        out = apply_moving_average(clean_series, window=7)
        assert out.isna().sum() == 0

    def test_constant_series_unchanged(self):
        s = pd.Series(np.ones(50) * 5.0)
        out = apply_moving_average(s, window=5)
        np.testing.assert_allclose(out.values, 5.0, atol=1e-10)

    def test_filter_dataframe_adds_filtered_cols(self, sample_df):
        df_out = filter_dataframe(sample_df, ["F", "D"], "moving_average", window=5)
        assert "F_filtered" in df_out.columns
        assert "D_filtered" in df_out.columns
        assert len(df_out) == len(sample_df)


class TestLowPassFilter:
    def test_output_length(self, clean_series):
        out = apply_lowpass_filter(clean_series, cutoff_freq=0.05, fs=1.0)
        assert len(out) == len(clean_series)

    def test_attenuates_high_frequency_noise(self):
        """Low-pass should attenuate a high-frequency component."""
        t = np.linspace(0, 1, 500)
        fs = 500.0
        low = np.sin(2 * np.pi * 5 * t)   # 5 Hz — below cutoff
        high = 0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz — above cutoff
        noisy = pd.Series(low + high)
        filtered = apply_lowpass_filter(noisy, cutoff_freq=20.0, fs=fs)
        # Residual high-freq should be attenuated
        residual = filtered.values - low
        assert residual.std() < 0.5 * high.std() + 0.02

    def test_no_nans(self, clean_series):
        out = apply_lowpass_filter(clean_series, cutoff_freq=0.1, fs=1.0)
        assert out.isna().sum() == 0

    def test_filter_dataframe_lowpass(self, sample_df):
        df_out = filter_dataframe(
            sample_df, ["F"], "lowpass",
            cutoff_freq=0.01, fs=1.0 / 60, order=4  # 1-min sampling → fs=1/60 Hz
        )
        assert "F_filtered" in df_out.columns


# ===========================================================================
# 4. Excel Parsing
# ===========================================================================


class TestExcelParsing:
    def test_load_example_excel(self):
        """The bundled example Excel should load without errors."""
        excel_bytes = create_example_excel()
        buf = io.BytesIO(excel_bytes)
        df, warns = load_excel_sensor_data(buf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        df_bad = pd.DataFrame({"Time": [1, 2, 3], "F": [100, 101, 102]})
        buf = io.BytesIO()
        df_bad.to_excel(buf, index=False, engine="xlsxwriter")
        buf.seek(0)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_excel_sensor_data(buf)

    def test_numeric_coercion(self):
        """String values should be coerced to NaN, not crash."""
        df_ok = simulate_distillation_column(n_points=20, seed=0)
        df_ok_cols = df_ok[REQUIRED_COLUMNS].copy()
        # Inject a string in F column (store as object dtype to avoid FutureWarning)
        df_ok_cols["F"] = df_ok_cols["F"].astype(object)
        df_ok_cols.loc[5, "F"] = "bad_value"
        buf = io.BytesIO()
        df_ok_cols.to_excel(buf, index=False, engine="xlsxwriter")
        buf.seek(0)
        loaded, warns = load_excel_sensor_data(buf)
        assert isinstance(loaded, pd.DataFrame)


# ===========================================================================
# 5. Data Reconciliation (WLS)
# ===========================================================================


class TestDataReconciliation:
    def test_mass_balance_satisfied_after_reconciliation(self):
        """After WLS reconciliation, F - D - B should be close to zero."""
        df = simulate_distillation_column(n_points=50, noise_level=0.05, seed=3)
        A, b = build_distillation_constraints()
        # Use only mass balance (row 0) and only F, D, B (cols 0-2)
        A_sub = A[0:1, 0:3]
        b_sub = b[0:1]
        sigma = np.diag([4.0, 2.25, 2.25])  # σ² = σ²_F, σ²_D, σ²_B
        df_rec = reconcile_data_wls(df, ["F", "D", "B"], A_sub, b_sub, sigma)

        residual = df_rec["F_rec"] - df_rec["D_rec"] - df_rec["B_rec"]
        assert (residual.abs() < 1e-6).all(), "Mass balance not satisfied after reconciliation"

    def test_reconciliation_reduces_balance_error(self):
        """Reconciliation should reduce the mean absolute balance error."""
        df = simulate_distillation_column(n_points=50, noise_level=0.08, seed=4)
        A, b = build_distillation_constraints()
        A_sub = A[0:1, 0:3]
        b_sub = b[0:1]
        df_rec = reconcile_data_wls(df, ["F", "D", "B"], A_sub, b_sub)

        err_before = (df["F"] - df["D"] - df["B"]).abs().mean()
        err_after = (df_rec["F_rec"] - df_rec["D_rec"] - df_rec["B_rec"]).abs().mean()
        assert err_after < err_before + 1e-6  # after ≤ before

    def test_output_columns_created(self, sample_df):
        A, b = build_distillation_constraints()
        A_sub = A[0:1, 0:3]
        b_sub = b[0:1]
        df_rec = reconcile_data_wls(sample_df, ["F", "D", "B"], A_sub, b_sub)
        for col in ["F_rec", "D_rec", "B_rec", "F_adj", "D_adj", "B_adj"]:
            assert col in df_rec.columns

    def test_build_constraints_shape(self):
        A, b = build_distillation_constraints()
        assert A.shape == (2, 6)
        assert b.shape == (2,)


# ===========================================================================
# 6. KPI Computation
# ===========================================================================


class TestKPIComputation:
    def test_kpi_dataframe_structure(self, sample_df):
        kpi = compute_kpis(sample_df, ["F", "D"], band_pct=5.0)
        assert "Variable" in kpi.columns
        assert "MAE" in kpi.columns
        assert "RMSE" in kpi.columns
        assert "Adherence (%)" in kpi.columns
        assert len(kpi) == 2

    def test_kpi_values_are_finite(self, sample_df):
        kpi = compute_kpis(sample_df, ["F", "T_top"], band_pct=5.0)
        for col in ["ME", "MAE", "RMSE"]:
            assert kpi[col].dropna().apply(np.isfinite).all()

    def test_adherence_zero_for_large_errors(self):
        """If sensor is always far from twin, adherence should be 0%."""
        df = pd.DataFrame({
            "Time": [0, 1, 2],
            "F": [100.0, 100.0, 100.0],
            "twin_F": [200.0, 200.0, 200.0],
        })
        kpi = compute_kpis(df, ["F"], band_pct=5.0)
        assert kpi.loc[kpi["Variable"] == "F", "Adherence (%)"].values[0] == 0.0

    def test_adherence_100_for_perfect_match(self):
        """If sensor equals twin, adherence should be 100%."""
        vals = [100.0, 101.0, 99.5]
        df = pd.DataFrame({"Time": [0, 1, 2], "F": vals, "twin_F": vals})
        kpi = compute_kpis(df, ["F"], band_pct=5.0)
        assert kpi.loc[kpi["Variable"] == "F", "Adherence (%)"].values[0] == 100.0

    def test_outlier_summary_structure(self, sample_df):
        result = detect_outliers_iqr(sample_df, ["F", "D"], k=1.5)
        summary = compute_outlier_summary(sample_df, result, ["F", "D"])
        assert list(summary.columns) == ["Variable", "Total Points", "Outliers", "Outlier %"]
        assert len(summary) == 2
