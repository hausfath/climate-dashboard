"""Tests for NMME fetcher using synthetic gridded NetCDF data."""

import numpy as np
import pytest
import xarray as xr

from enso_forecast.config import NINO34_LAT_BOUNDS, NINO34_LON_BOUNDS_360


def test_nino34_region_extraction(synthetic_gridded_nc):
    """Test that Nino3.4 region is correctly extracted from gridded data."""
    nc_path, ds = synthetic_gridded_nc

    lat_min, lat_max = NINO34_LAT_BOUNDS  # -5, 5
    lon_min, lon_max = NINO34_LON_BOUNDS_360  # 190, 240

    region = ds["tmpsfc"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Should select lats from -4 to 4 (stepping by 2: -4, -2, 0, 2, 4)
    assert region.sizes["lat"] == 5
    # Lons from 190 to 240 (stepping by 2)
    assert region.sizes["lon"] == 26


def test_cosine_weighted_mean(synthetic_gridded_nc):
    """Test cosine-latitude-weighted spatial mean computation."""
    nc_path, ds = synthetic_gridded_nc

    lat_min, lat_max = NINO34_LAT_BOUNDS
    lon_min, lon_max = NINO34_LON_BOUNDS_360

    region = ds["tmpsfc"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Cosine weighting
    weights = np.cos(np.deg2rad(region.coords["lat"]))
    weighted = region.weighted(weights)
    area_mean = weighted.mean(dim=["lat", "lon"])

    # Should have 9 time steps
    assert area_mean.sizes["time"] == 9

    # Unweighted mean for comparison
    simple_mean = region.mean(dim=["lat", "lon"])

    # At equatorial latitudes (-5 to 5), cosine weighting effect is tiny
    # but the values should differ slightly
    assert not np.allclose(area_mean.values, simple_mean.values)
    # But they should be very close
    np.testing.assert_allclose(area_mean.values, simple_mean.values, atol=0.01)


def test_longitude_convention_360(synthetic_gridded_nc):
    """Test handling of 0-360 longitude convention."""
    nc_path, ds = synthetic_gridded_nc
    lon = ds.coords["lon"]

    # Our fixture uses 180-250, which is 0-360 convention
    assert float(lon.max()) > 180
    assert float(lon.min()) >= 0


def test_longitude_convention_180(tmp_path):
    """Test handling of -180/180 longitude convention."""
    lat = np.arange(-10, 11, 2.0)
    lon = np.arange(-180, -109, 2.0)  # Western hemisphere
    time_vals = np.array(["2026-04-01", "2026-05-01"], dtype="datetime64")
    np.random.seed(42)

    data = np.random.uniform(-1.0, 1.0, (2, len(lat), len(lon)))
    ds = xr.Dataset(
        {"tmpsfc": (["time", "lat", "lon"], data)},
        coords={"time": time_vals, "lat": lat, "lon": lon},
    )

    from enso_forecast.config import NINO34_LON_BOUNDS_180
    lon_min, lon_max = NINO34_LON_BOUNDS_180  # -170, -120

    region = ds["tmpsfc"].sel(
        lat=slice(-5, 5),
        lon=slice(lon_min, lon_max),
    )

    # Should capture lons from -170 to -120 that fall in our range
    # Our grid goes from -180 to -110, so -170 to -120 should have some data
    assert region.sizes["lon"] > 0


def test_nmme_output_values_in_range(synthetic_gridded_nc):
    """Test that computed Nino3.4 values are within expected range."""
    nc_path, ds = synthetic_gridded_nc

    region = ds["tmpsfc"].sel(
        lat=slice(*NINO34_LAT_BOUNDS),
        lon=slice(*NINO34_LON_BOUNDS_360),
    )

    weights = np.cos(np.deg2rad(region.coords["lat"]))
    area_mean = region.weighted(weights).mean(dim=["lat", "lon"])

    # All values should be within -4 to +4 (our synthetic data is -1.5 to 1.5)
    assert float(area_mean.min()) >= -4.0
    assert float(area_mean.max()) <= 4.0
