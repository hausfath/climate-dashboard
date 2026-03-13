"""Shared test fixtures: synthetic data generators for all test modules."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def synthetic_forecast_df():
    """Generate a synthetic multi-source forecast DataFrame."""
    records = []
    np.random.seed(42)

    # IRI models (seasonal)
    for model in ["ECMWF-IRI", "NCEP CFSv2", "UKMO", "CSU CLIPR"]:
        model_type = "statistical" if model == "CSU CLIPR" else "dynamical"
        for lead in range(1, 7):
            target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
            records.append({
                "source": "IRI",
                "model": model,
                "model_type": model_type,
                "init_date": "2026-03-01",
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(np.random.uniform(-1.5, 1.5), 2),
                "member_id": "mean",
                "temporal_resolution": "seasonal_3mo",
                "anomaly_base_period": "varies",
            })

    # CFS members
    for member in range(1, 6):
        for lead in range(1, 10):
            target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
            records.append({
                "source": "CFS",
                "model": "CFSv2",
                "model_type": "dynamical",
                "init_date": "2026-03-01",
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(np.random.uniform(-2.0, 2.0), 2),
                "member_id": f"ens_E1_{member:03d}",
                "temporal_resolution": "monthly",
                "anomaly_base_period": "1991-2020",
            })

    # CFS mean
    for lead in range(1, 10):
        target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
        records.append({
            "source": "CFS",
            "model": "CFSv2",
            "model_type": "dynamical",
            "init_date": "2026-03-01",
            "target_month": target_dt.strftime("%Y-%m"),
            "lead_months": lead,
            "nino34_anom": round(np.random.uniform(-1.0, 1.0), 2),
            "member_id": "mean",
            "temporal_resolution": "monthly",
            "anomaly_base_period": "1991-2020",
        })

    # NMME models
    for model in ["NCEP-CFSv2", "ECCC-CanESM5", "NCAR-CESM1"]:
        for lead in range(1, 10):
            target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
            records.append({
                "source": "NMME",
                "model": model,
                "model_type": "dynamical",
                "init_date": "2026-03-01",
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(np.random.uniform(-1.5, 1.5), 2),
                "member_id": "mean",
                "temporal_resolution": "monthly",
                "anomaly_base_period": "model-specific",
            })

    # C3S models
    for model in ["ECMWF", "UKMO", "Meteo-France"]:
        for lead in range(1, 7):
            target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
            records.append({
                "source": "C3S",
                "model": model,
                "model_type": "dynamical",
                "init_date": "2026-03-01",
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(np.random.uniform(-1.5, 1.5), 2),
                "member_id": "mean",
                "temporal_resolution": "monthly",
                "anomaly_base_period": "1993-2016",
            })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_observed_df():
    """Generate synthetic observed monthly Nino3.4 data."""
    dates = pd.date_range("2024-01-01", "2026-03-01", freq="MS")
    np.random.seed(123)
    return pd.DataFrame({
        "year": dates.year,
        "month": dates.month,
        "nino34_anom": np.round(np.random.uniform(-1.5, 1.5, len(dates)), 2),
        "date": dates,
    })


@pytest.fixture
def synthetic_cfs_df():
    """Generate synthetic CFS data with individual ensemble members."""
    records = []
    np.random.seed(99)
    for member in range(1, 21):
        for lead in range(1, 10):
            target_dt = pd.Timestamp("2026-03-01") + pd.DateOffset(months=lead)
            records.append({
                "source": "CFS",
                "model": "CFSv2",
                "model_type": "dynamical",
                "init_date": "2026-03-01",
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(np.random.uniform(-2.5, 2.5), 2),
                "member_id": f"ens_E1_{member:03d}",
                "temporal_resolution": "monthly",
                "anomaly_base_period": "1991-2020",
            })

    # Add mean
    df = pd.DataFrame(records)
    mean_df = df.groupby("target_month", as_index=False).agg(
        nino34_anom=("nino34_anom", "mean"),
        lead_months=("lead_months", "first"),
    )
    mean_df["source"] = "CFS"
    mean_df["model"] = "CFSv2"
    mean_df["model_type"] = "dynamical"
    mean_df["init_date"] = "2026-03-01"
    mean_df["member_id"] = "mean"
    mean_df["temporal_resolution"] = "monthly"
    mean_df["anomaly_base_period"] = "1991-2020"

    return pd.concat([df, mean_df], ignore_index=True)


@pytest.fixture
def synthetic_gridded_nc(tmp_path):
    """Create a small synthetic gridded NetCDF file (like NMME)."""
    lat = np.arange(-10, 11, 2.0)
    lon = np.arange(180, 251, 2.0)
    time = pd.date_range("2026-04-01", periods=9, freq="MS")
    np.random.seed(77)

    data = np.random.uniform(-1.5, 1.5, (len(time), len(lat), len(lon)))

    ds = xr.Dataset(
        {"tmpsfc": (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    nc_path = tmp_path / "test_nmme.nc"
    ds.to_netcdf(nc_path)
    return nc_path, ds


@pytest.fixture
def synthetic_cfs_nc(tmp_path):
    """Create a small synthetic CFS-style NetCDF with ensemble members."""
    time = pd.date_range("2026-04-01", periods=9, freq="MS")
    members = np.arange(5)
    np.random.seed(55)

    data = np.random.uniform(-2.0, 2.0, (len(members), len(time)))

    ds = xr.Dataset(
        {"nino34": (["ensemble", "time"], data)},
        coords={"time": time, "ensemble": members},
    )

    nc_path = tmp_path / "test_cfs.nc"
    ds.to_netcdf(nc_path)
    return nc_path, ds
