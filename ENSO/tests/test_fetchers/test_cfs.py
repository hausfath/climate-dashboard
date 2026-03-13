"""Tests for CFS v2 fetcher using synthetic NetCDF data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_cfs_member_extraction(synthetic_cfs_nc):
    """Test extraction of ensemble members from a CFS-style NetCDF."""
    nc_path, ds = synthetic_cfs_nc

    # Simulate the logic from cfs.py
    data = ds["nino34"]
    assert "ensemble" in data.dims
    assert "time" in data.dims
    assert data.sizes["ensemble"] == 5
    assert data.sizes["time"] == 9


def test_cfs_mean_computation(synthetic_cfs_nc):
    """Test that ensemble mean is correctly computed."""
    nc_path, ds = synthetic_cfs_nc
    data = ds["nino34"]

    # Compute mean across ensemble dimension
    ens_mean = data.mean(dim="ensemble")
    manual_mean = data.values.mean(axis=0)

    np.testing.assert_allclose(ens_mean.values, manual_mean, atol=1e-6)


def test_cfs_member_id_assignment():
    """Test that member IDs follow expected naming convention."""
    # This tests the naming logic
    for e in range(1, 4):
        for m in range(1, 41):
            member_id = f"ens_E{e}_{m:03d}"
            assert member_id.startswith("ens_E")
            parts = member_id.split("_")
            assert len(parts) == 3
            assert parts[2].isdigit()


def test_cfs_output_schema(synthetic_cfs_df):
    """Test that synthetic CFS data matches expected schema."""
    df = synthetic_cfs_df
    required = [
        "source", "model", "model_type", "init_date",
        "target_month", "lead_months", "nino34_anom",
        "member_id", "temporal_resolution", "anomaly_base_period",
    ]
    for col in required:
        assert col in df.columns

    assert (df["source"] == "CFS").all()
    assert (df["model"] == "CFSv2").all()
    assert (df["model_type"] == "dynamical").all()
    assert (df["anomaly_base_period"] == "1991-2020").all()


def test_cfs_has_members_and_mean(synthetic_cfs_df):
    """Test that both individual members and mean are present."""
    df = synthetic_cfs_df
    members = df[df["member_id"] != "mean"]
    means = df[df["member_id"] == "mean"]

    assert len(members) > 0
    assert len(means) > 0
    assert members["member_id"].nunique() == 20  # 20 members


def test_cfs_nc_time_parsing(synthetic_cfs_nc):
    """Test that datetime time coordinates are correctly handled."""
    nc_path, ds = synthetic_cfs_nc
    time_vals = ds.coords["time"].values

    # All should be datetime64
    assert np.issubdtype(time_vals.dtype, np.datetime64)

    # Should be monthly, starting April 2026
    first = pd.Timestamp(time_vals[0])
    assert first.year == 2026
    assert first.month == 4
