"""Tests for C3S/Copernicus fetcher."""

import numpy as np
import pytest
import xarray as xr

from enso_forecast.config import (
    C3S_ANOMALY_BASE_PERIOD,
    C3S_DATASET,
    C3S_MODELS,
    C3S_VARIABLE,
)


def test_c3s_model_config():
    """Test that C3S model configuration is complete."""
    assert len(C3S_MODELS) == 9

    for name, info in C3S_MODELS.items():
        assert "system" in info
        assert "originating_centre" in info
        assert info["system"].isdigit() or info["system"].isalnum()


def test_c3s_request_parameters():
    """Test that CDS API request parameters are correctly structured."""
    year, month = 2026, 3

    for model_name, model_info in list(C3S_MODELS.items())[:1]:
        request = {
            "originating_centre": model_info["originating_centre"],
            "system": model_info["system"],
            "variable": C3S_VARIABLE,
            "product_type": "ensemble_mean",
            "year": str(year),
            "month": f"{month:02d}",
            "leadtime_month": [str(i) for i in range(1, 7)],
            "area": [5, -170, -5, -120],
            "data_format": "netcdf",
        }

        assert request["variable"] == "sea_surface_temperature_anomaly"
        assert request["product_type"] == "ensemble_mean"
        assert len(request["leadtime_month"]) == 6
        assert request["area"] == [5, -170, -5, -120]
        assert request["year"] == "2026"
        assert request["month"] == "03"


def test_c3s_area_mean_computation(tmp_path):
    """Test area mean computation from C3S-like gridded data."""
    # Create C3S-style NetCDF (already subsetted to Nino3.4 region)
    lat = np.arange(-5, 6, 1.0)
    lon = np.arange(-170, -119, 1.0)
    lead_time = np.arange(1, 7)
    np.random.seed(88)

    data = np.random.uniform(-1.0, 1.0, (len(lead_time), len(lat), len(lon)))
    ds = xr.Dataset(
        {"sst_anomaly": (["forecastMonth", "latitude", "longitude"], data)},
        coords={
            "forecastMonth": lead_time,
            "latitude": lat,
            "longitude": lon,
        },
    )

    # Compute area mean
    weights = np.cos(np.deg2rad(ds.coords["latitude"]))
    weighted = ds["sst_anomaly"].weighted(weights)
    area_mean = weighted.mean(dim=["latitude", "longitude"])

    assert area_mean.sizes["forecastMonth"] == 6
    assert all(abs(v) < 2.0 for v in area_mean.values)


def test_c3s_anomaly_base_period():
    """Test that C3S uses the correct anomaly base period."""
    assert C3S_ANOMALY_BASE_PERIOD == "1993-2016"


def test_c3s_no_api_key_raises():
    """Test that missing API key raises an informative error."""
    import enso_forecast.config as config
    original = config.CDS_API_KEY
    try:
        config.CDS_API_KEY = ""
        from enso_forecast.fetchers.c3s import fetch_c3s
        with pytest.raises(ValueError, match="CDS API key"):
            fetch_c3s()
    finally:
        config.CDS_API_KEY = original
