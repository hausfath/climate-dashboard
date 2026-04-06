"""Fetch C3S/Copernicus seasonal SST anomaly forecasts via CDS API."""

import logging
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from enso_forecast.config import (
    C3S_ANOMALY_BASE_PERIOD,
    C3S_DATASET,
    C3S_MODELS,
    C3S_VARIABLE,
    CDS_API_KEY,
    CDS_API_URL,
    FORECASTS_DIR,
    NINO34_LAT_BOUNDS,
    NINO34_LON_BOUNDS_180,
    NINO34_LON_BOUNDS_360,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


def _compute_nino34_area_mean(ds: xr.Dataset) -> xr.DataArray:
    """Compute cosine-latitude-weighted Nino3.4 area mean from gridded C3S data.

    Preserves all non-spatial dimensions (number/member, forecastMonth, etc.).
    Squeezes only singleton time dimensions like forecast_reference_time and
    indexing_time.
    """
    # Find the SST anomaly variable
    var_name = None
    for v in ds.data_vars:
        if "sst" in v.lower() or "anom" in v.lower() or "temperature" in v.lower():
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

    # Identify lat/lon dims
    lat_name = lon_name = None
    for dim in data.dims:
        dl = dim.lower()
        if "lat" in dl:
            lat_name = dim
        elif "lon" in dl:
            lon_name = dim

    if lat_name is None or lon_name is None:
        # Fallback: average over everything that isn't a known non-spatial dim
        keep = {"number", "forecastmonth", "forecastMonth", "leadtime_month"}
        spatial = [d for d in data.dims if d.lower() not in {k.lower() for k in keep}]
        if spatial:
            return data.mean(dim=spatial)
        return data

    lat = ds.coords[lat_name]

    # Squeeze singleton time dims (forecast_reference_time, indexing_time)
    # but preserve number (ensemble member) and forecastMonth
    squeeze_dims = []
    for dim in data.dims:
        if dim in (lat_name, lon_name):
            continue
        if data.sizes[dim] == 1 and dim.lower() not in ("number", "forecastmonth"):
            squeeze_dims.append(dim)
    if squeeze_dims:
        data = data.squeeze(dim=squeeze_dims, drop=True)

    # Cosine weighting and area mean
    weights = np.cos(np.deg2rad(lat))
    weighted = data.weighted(weights)
    return weighted.mean(dim=[lat_name, lon_name])


def _extract_records(
    nino34: xr.DataArray,
    model_name: str,
    init_date: str,
) -> list[dict]:
    """Extract forecast records from a Nino3.4 DataArray.

    Handles both ensemble-mean (no 'number' dim) and member-level data.
    """
    records = []

    # Identify forecastMonth dim
    fm_dim = None
    for dim in nino34.dims:
        if "forecast" in dim.lower() or "lead" in dim.lower():
            fm_dim = dim
            break
    if fm_dim is None and len(nino34.dims) == 1:
        fm_dim = nino34.dims[0]

    # Identify member dim
    member_dim = None
    for dim in nino34.dims:
        if "number" in dim.lower() or "member" in dim.lower():
            member_dim = dim
            break

    init_dt = datetime.strptime(init_date, "%Y-%m-%d")

    if member_dim is not None:
        n_members = nino34.sizes[member_dim]
        member_coords = nino34.coords[member_dim].values
    else:
        n_members = 0
        member_coords = []

    def _add_records(data_1d, member_id: str):
        """Add records from a 1D (forecastMonth) array.

        C3S forecastMonth convention: forecastMonth=1 is the init month itself
        (e.g., init March 2026, forecastMonth=1 = March 2026).
        So lead_months = forecastMonth - 1, and
        target_month = init_date + (forecastMonth - 1) months.
        """
        if fm_dim is None:
            val = float(data_1d.values)
            if not np.isnan(val):
                # Single value — assume it's forecastMonth=1 (the init month)
                records.append({
                    "source": "C3S",
                    "model": model_name,
                    "model_type": "dynamical",
                    "init_date": init_date,
                    "target_month": pd.Timestamp(init_date).strftime("%Y-%m"),
                    "lead_months": 0,
                    "nino34_anom": round(float(val), 4),
                    "member_id": member_id,
                    "temporal_resolution": "monthly",
                    "anomaly_base_period": C3S_ANOMALY_BASE_PERIOD,
                })
            return

        fm_coords = nino34.coords.get(fm_dim)
        for t_idx in range(data_1d.sizes[fm_dim]):
            val = float(data_1d.isel({fm_dim: t_idx}).values)
            if np.isnan(val):
                continue

            # forecastMonth is 1-indexed: 1 = init month, 2 = init+1, etc.
            if fm_coords is not None:
                forecast_month_num = int(fm_coords.values[t_idx])
            else:
                forecast_month_num = t_idx + 1

            lead = forecast_month_num - 1  # 0-based lead from init
            target_dt = init_dt + pd.DateOffset(months=lead)
            target_month = target_dt.strftime("%Y-%m")
            records.append({
                "source": "C3S",
                "model": model_name,
                "model_type": "dynamical",
                "init_date": init_date,
                "target_month": target_month,
                "lead_months": lead,
                "nino34_anom": round(float(val), 4),
                "member_id": member_id,
                "temporal_resolution": "monthly",
                "anomaly_base_period": C3S_ANOMALY_BASE_PERIOD,
            })

    if member_dim is not None:
        # Individual members
        for m_idx in range(n_members):
            m_num = int(member_coords[m_idx])
            member_id = f"ens_{m_num:03d}"
            member_data = nino34.isel({member_dim: m_idx})
            _add_records(member_data, member_id)

        # Also compute and store ensemble mean
        ens_mean = nino34.mean(dim=member_dim)
        _add_records(ens_mean, "mean")
    else:
        # Already ensemble mean
        _add_records(nino34, "mean")

    return records


def _fetch_c3s_month(
    year: int,
    month: int,
    include_members: bool,
    only_models: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Fetch C3S data for a specific init month.

    Args:
        only_models: If provided, only fetch these model names.

    Returns:
        (records, succeeded_models) — the fetched records and the set of
        model names that returned data.
    """
    import cdsapi

    init_date = f"{year}-{month:02d}-01"
    c3s_raw = RAW_DIR / "c3s" / f"{year}{month:02d}"
    c3s_raw.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)
    product_type = "monthly_mean" if include_members else "ensemble_mean"
    all_records = []
    succeeded_models: set[str] = set()

    for model_name, model_info in C3S_MODELS.items():
        if only_models is not None and model_name not in only_models:
            continue
        logger.info("Fetching C3S %s (system %s, %s)...",
                     model_name, model_info["system"], product_type)

        suffix = "members" if include_members else "ensmean"
        nc_path = c3s_raw / f"{model_info['originating_centre']}_{year}{month:02d}_{suffix}.nc"

        try:
            client.retrieve(
                C3S_DATASET,
                {
                    "originating_centre": model_info["originating_centre"],
                    "system": model_info["system"],
                    "variable": C3S_VARIABLE,
                    "product_type": product_type,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "leadtime_month": [str(i) for i in range(1, model_info.get("max_lead_months", 6) + 1)],
                    "area": [5, -170, -5, -120],  # N, W, S, E
                    "data_format": "netcdf",
                },
                str(nc_path),
            )
        except Exception as e:
            logger.warning("Failed to fetch C3S %s: %s", model_name, e)
            continue

        try:
            ds = xr.open_dataset(nc_path)
            nino34 = _compute_nino34_area_mean(ds)
            logger.info("C3S %s: Nino3.4 dims=%s, shape=%s",
                        model_name, nino34.dims, nino34.shape)

            model_records = _extract_records(nino34, model_name, init_date)
            all_records.extend(model_records)
            succeeded_models.add(model_name)

            n_members = 0
            for dim in nino34.dims:
                if "number" in dim.lower():
                    n_members = nino34.sizes[dim]
            n_leads = max(1, nino34.sizes.get("forecastMonth", 1))

            if n_members > 0:
                logger.info("C3S %s: %d members x %d lead times + mean",
                            model_name, n_members, n_leads)
            else:
                logger.info("C3S %s: %d lead times (ensemble mean)",
                            model_name, n_leads)

            ds.close()

        except Exception as e:
            logger.warning("Error processing C3S %s: %s", model_name, e)
            continue

    return all_records, succeeded_models


def fetch_c3s(
    year: int | None = None,
    month: int | None = None,
    include_members: bool = True,
) -> pd.DataFrame:
    """Fetch C3S seasonal forecasts for all configured models.

    Tries the current month first for each model; for any models whose
    data is not yet available, falls back to the previous month.

    Returns DataFrame in standard forecast schema.
    """
    if not CDS_API_KEY:
        raise ValueError(
            "CDS API key not configured. Set CDS_API_KEY environment variable "
            "or edit enso_forecast/config.py. Get a key at "
            "https://cds.climate.copernicus.eu/"
        )

    try:
        import cdsapi
    except ImportError:
        raise ImportError("cdsapi package required. Install with: pip install cdsapi")

    if year is None:
        year = date.today().year
    if month is None:
        month = date.today().month

    # Try current month for all models
    all_records, succeeded = _fetch_c3s_month(year, month, include_members)

    # Fall back to previous month for any models that didn't return data
    missing = set(C3S_MODELS.keys()) - succeeded
    if missing:
        prev = pd.Timestamp(f"{year}-{month:02d}-01") - pd.DateOffset(months=1)
        logger.info(
            "C3S: %d models missing for %d-%02d (%s), falling back to %d-%02d",
            len(missing), year, month, ", ".join(sorted(missing)),
            prev.year, prev.month,
        )
        prev_records, prev_succeeded = _fetch_c3s_month(
            prev.year, prev.month, include_members, only_models=missing,
        )
        all_records.extend(prev_records)
        succeeded |= prev_succeeded

        still_missing = missing - prev_succeeded
        if still_missing:
            logger.warning("C3S: no data for models: %s", ", ".join(sorted(still_missing)))

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        n_models = df["model"].nunique()
        n_members = df[df["member_id"] != "mean"]["member_id"].nunique()
        logger.info("C3S: %d records from %d models (%d individual members)",
                     len(df), n_models, n_members)
    else:
        logger.info("C3S: 0 records")
    return df


def save_c3s(force: bool = False) -> pd.DataFrame:
    """Fetch and save C3S data."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "C3S"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("C3S data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_c3s(include_members=True)
    if df.empty:
        logger.warning("C3S fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved C3S data to %s", out_path)
    return df
