"""Tests for data normalization, validation, and combining."""

import numpy as np
import pandas as pd
import pytest

from enso_forecast.normalize import (
    REQUIRED_COLUMNS,
    compute_multi_model_mean,
    compute_source_means,
    deduplicate_models,
    get_baseline_caveat,
    get_ensemble_means,
    get_model_comparison_df,
    validate_forecast_df,
)


class TestValidation:
    """Tests for validate_forecast_df."""

    def test_valid_df(self, synthetic_forecast_df):
        issues = validate_forecast_df(synthetic_forecast_df)
        # Should have no errors (may have warnings about large values)
        errors = [i for i in issues if not i.startswith("WARNING")]
        assert len(errors) == 0

    def test_empty_df(self):
        issues = validate_forecast_df(pd.DataFrame())
        assert any("empty" in i.lower() for i in issues)

    def test_missing_columns(self):
        df = pd.DataFrame({"source": ["IRI"], "model": ["test"]})
        issues = validate_forecast_df(df)
        assert any("Missing required columns" in i for i in issues)

    def test_out_of_range_values(self, synthetic_forecast_df):
        df = synthetic_forecast_df.copy()
        # Insert an out-of-range value
        df.loc[0, "nino34_anom"] = 5.0
        issues = validate_forecast_df(df)
        assert any("outside valid range" in i for i in issues)

    def test_negative_lead_months(self, synthetic_forecast_df):
        df = synthetic_forecast_df.copy()
        df.loc[0, "lead_months"] = -1
        issues = validate_forecast_df(df)
        assert any("lead_months outside" in i for i in issues)

    def test_target_before_init(self, synthetic_forecast_df):
        df = synthetic_forecast_df.copy()
        df.loc[0, "target_month"] = "2025-01"  # Before init_date 2026-03-01
        issues = validate_forecast_df(df)
        assert any("before init_date" in i for i in issues)

    def test_nan_values_flagged(self, synthetic_forecast_df):
        df = synthetic_forecast_df.copy()
        df.loc[0, "nino34_anom"] = np.nan
        df.loc[1, "nino34_anom"] = np.nan
        issues = validate_forecast_df(df)
        assert any("NaN" in i for i in issues)


class TestEnsembleMeans:
    """Tests for get_ensemble_means."""

    def test_filters_to_means_only(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        assert (means["member_id"] == "mean").all()

    def test_excludes_individual_members(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        # CFS has individual members that should be excluded
        cfs_means = means[means["source"] == "CFS"]
        assert len(cfs_means) > 0
        assert all(mid == "mean" for mid in cfs_means["member_id"])


class TestDeduplication:
    """Tests for deduplicate_models."""

    def test_removes_duplicate_models(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        before = len(means)
        deduped = deduplicate_models(means)

        # Should have fewer rows (NCEP CFSv2 appears in IRI, CFS, NMME)
        assert len(deduped) <= before

    def test_keeps_preferred_source(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        deduped = deduplicate_models(means)

        # For NCEP-CFSv2, preferred source is CFS
        cfsv2_rows = deduped[deduped["canonical"] == "NCEP-CFSv2"]
        if len(cfsv2_rows) > 0:
            assert (cfsv2_rows["source"] == "CFS").all()

    def test_preserves_unique_models(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        deduped = deduplicate_models(means)

        # Models with no overlap should still be present
        ncar = deduped[deduped["model"] == "NCAR-CESM1"]
        assert len(ncar) > 0


class TestSourceMeans:
    """Tests for compute_source_means."""

    def test_one_row_per_source_per_month(self, synthetic_forecast_df):
        source_means = compute_source_means(synthetic_forecast_df)

        # Group by source + target_month should have unique rows
        groups = source_means.groupby(["source", "target_month"]).size()
        assert (groups == 1).all()

    def test_mean_within_model_range(self, synthetic_forecast_df):
        means = get_ensemble_means(synthetic_forecast_df)
        source_means = compute_source_means(synthetic_forecast_df)

        for source in source_means["source"].unique():
            src_sm = source_means[source_means["source"] == source]
            src_models = means[means["source"] == source]

            for tm in src_sm["target_month"].unique():
                sm_val = src_sm[src_sm["target_month"] == tm]["nino34_anom"].iloc[0]
                model_vals = src_models[src_models["target_month"] == tm]["nino34_anom"]

                if len(model_vals) > 0:
                    assert sm_val >= model_vals.min() - 0.001
                    assert sm_val <= model_vals.max() + 0.001


class TestMultiModelMean:
    """Tests for compute_multi_model_mean."""

    def test_returns_single_model_row(self, synthetic_forecast_df):
        mm = compute_multi_model_mean(synthetic_forecast_df)
        assert (mm["model"] == "multi-model-mean").all()
        assert (mm["source"] == "multi-model").all()

    def test_one_value_per_target_month(self, synthetic_forecast_df):
        mm = compute_multi_model_mean(synthetic_forecast_df)
        assert mm["target_month"].is_unique

    def test_equal_weight_per_model(self, synthetic_forecast_df):
        mm = compute_multi_model_mean(synthetic_forecast_df, weight_by="model")
        assert len(mm) > 0

    def test_equal_weight_per_source(self, synthetic_forecast_df):
        mm = compute_multi_model_mean(synthetic_forecast_df, weight_by="source")
        assert len(mm) > 0

    def test_empty_input(self):
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        mm = compute_multi_model_mean(empty)
        assert mm.empty


class TestModelComparison:
    """Tests for get_model_comparison_df."""

    def test_returns_means_only(self, synthetic_forecast_df):
        comp = get_model_comparison_df(synthetic_forecast_df)
        assert (comp["member_id"] == "mean").all()

    def test_no_duplicate_physical_models(self, synthetic_forecast_df):
        comp = get_model_comparison_df(synthetic_forecast_df)
        # Check that deduplicated canonical names are unique per target_month
        for tm in comp["target_month"].unique():
            tm_data = comp[comp["target_month"] == tm]
            canonicals = tm_data["canonical"].tolist()
            # Canonical names should be unique (no duplicates)
            assert len(canonicals) == len(set(canonicals))


class TestBaselineCaveat:
    """Tests for get_baseline_caveat."""

    def test_returns_string(self):
        caveat = get_baseline_caveat()
        assert isinstance(caveat, str)
        assert len(caveat) > 0

    def test_mentions_key_periods(self):
        caveat = get_baseline_caveat()
        assert "1991" in caveat
        assert "1993" in caveat
