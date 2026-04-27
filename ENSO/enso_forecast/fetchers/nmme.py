"""Fetch NMME Nino3.4 forecasts from NOAA CPC (ensemble means and individual members)."""

import logging
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from enso_forecast.config import (
    FORECASTS_DIR,
    MAX_RETRIES,
    NINO34_LAT_BOUNDS,
    NINO34_LON_BOUNDS_360,
    RAW_DIR,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)

logger = logging.getLogger(__name__)

# Base URL for NMME data (ensemble means and individual members)
NMME_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/"

# Model names as they appear in filenames
NMME_FILE_MODELS = [
    "CFSv2",
    "CanESM5",
    "GEM5.2_NEMO",
    "NCAR_CESM1",
    "NCAR_CCSM4",
    "NASA_GEOS5v2",
]

NMME_MODEL_DISPLAY = {
    "CFSv2": "NCEP-CFSv2",
    "CanESM5": "ECCC-CanESM5",
    "GEM5.2_NEMO": "ECCC-GEM5.2-NEMO",
    "NCAR_CESM1": "NCAR-CESM1",
    "NCAR_CCSM4": "NCAR-CCSM4",
    "NASA_GEOS5v2": "NASA-GEOS-S2S-2",
    "NMME": "NMME",
}


def _find_latest_nmme_date() -> str:
    """Find the most recent NMME date from the ENSMEAN directory listing."""
    ensmean_url = NMME_BASE_URL + "ENSMEAN/"
    logger.info("Checking NMME directory listing at %s", ensmean_url)
    resp = requests.get(ensmean_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    pattern = re.compile(r"(\d{6})0800")
    dates = sorted(set(pattern.findall(resp.text)))
    if not dates:
        raise ValueError("No NMME date directories found")

    latest = dates[-1]
    logger.info("Found %d NMME dates, latest: %s", len(dates), latest)
    return latest


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with retry logic. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, headers=REQUEST_HEADERS, timeout=300, stream=True,
            )
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Download failed (attempt %d): %s", attempt + 1, e)
                time.sleep(RETRY_DELAY)
            else:
                logger.warning("Download failed after %d attempts: %s", MAX_RETRIES, e)
                return False
    return False


def _compute_area_mean(
    ds: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Cosine-latitude-weighted area mean over a lat/lon box.

    Preserves non-spatial dimensions (ensmem, target, etc.). When ``lon_bounds``
    is None, averages over all longitudes (used for the tropical 20S-20N band).
    """
    var_name = None
    for v in ds.data_vars:
        if v.lower() in ("fcst", "anom"):
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

    lat_name = lon_name = None
    for dim in data.dims:
        dl = dim.lower()
        if "lat" in dl:
            lat_name = dim
        elif "lon" in dl:
            lon_name = dim
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon dims from {data.dims}")

    lat = ds.coords[lat_name]
    lat_min, lat_max = lat_bounds

    squeeze_dims = [
        d for d in data.dims
        if d not in (lat_name, lon_name) and data.sizes[d] == 1
        and d.lower() not in ("ensmem", "target")
    ]
    if squeeze_dims:
        data = data.squeeze(dim=squeeze_dims, drop=True)

    lat_slice = slice(lat_max, lat_min) if float(lat[0]) > float(lat[-1]) else slice(lat_min, lat_max)
    sel = {lat_name: lat_slice}
    if lon_bounds is not None:
        lon_min, lon_max = lon_bounds
        sel[lon_name] = slice(lon_min, lon_max)
    region = data.sel(sel)

    if region.sizes[lat_name] == 0 or region.sizes[lon_name] == 0:
        raise ValueError(f"Empty area selection for lat={lat_bounds} lon={lon_bounds}")

    weights = np.cos(np.deg2rad(region.coords[lat_name]))
    return region.weighted(weights).mean(dim=[lat_name, lon_name])


def _compute_nino34_mean(ds: xr.Dataset) -> xr.DataArray:
    """Cosine-latitude-weighted Niño 3.4 area mean from gridded NMME data."""
    return _compute_area_mean(ds, NINO34_LAT_BOUNDS, NINO34_LON_BOUNDS_360)


# Tropical band used for rONI denominator (L'Heureux et al. 2024)
TROPICAL_LAT_BOUNDS = (-20.0, 20.0)


def _compute_tropical_mean(ds: xr.Dataset) -> xr.DataArray:
    """Cosine-latitude-weighted 20S-20N tropical-mean SSTA (all longitudes)."""
    return _compute_area_mean(ds, TROPICAL_LAT_BOUNDS, lon_bounds=None)


def _decode_target_times(ds: xr.Dataset, time_dim: str) -> list[pd.Timestamp]:
    """Decode NMME 'months since 1960' target times."""
    target_vals = ds.coords[time_dim].values
    ref_date = pd.Timestamp("1960-01-01")
    return [ref_date + pd.DateOffset(months=int(m)) for m in target_vals]


def _process_member_file(
    nc_path: Path, file_model: str, display_name: str, init_date: str,
) -> list[dict]:
    """Process an NMME file with individual ensemble members."""
    ds = xr.open_dataset(nc_path, decode_times=False)
    nino34 = _compute_nino34_mean(ds)
    try:
        tropical = _compute_tropical_mean(ds)
    except Exception as e:
        logger.warning("NMME %s: tropical-mean failed (%s); rONI will be NaN", display_name, e)
        tropical = None

    # Find dims
    ens_dim = target_dim = None
    for dim in nino34.dims:
        if "ens" in dim.lower():
            ens_dim = dim
        elif "target" in dim.lower() or "time" in dim.lower():
            target_dim = dim
    if target_dim is None and len(nino34.dims) == 1:
        target_dim = nino34.dims[0]

    target_dates = _decode_target_times(ds, target_dim) if target_dim else []
    init_dt = datetime.strptime(init_date, "%Y-%m-%d")

    def _trop_val(member_idx, t_idx):
        if tropical is None:
            return float("nan")
        if ens_dim and ens_dim in tropical.dims and member_idx is not None:
            return float(tropical.isel({ens_dim: member_idx, target_dim: t_idx}).values)
        return float(tropical.isel({target_dim: t_idx}).values)

    records = []

    def _make_record(member_id, target_dt, n34_val, trop_val):
        target_month = target_dt.strftime("%Y-%m")
        lead = (target_dt.year - init_dt.year) * 12 + (target_dt.month - init_dt.month)
        roni = n34_val - trop_val if not np.isnan(trop_val) else float("nan")
        return {
            "source": "NMME",
            "model": display_name,
            "model_type": "dynamical",
            "init_date": init_date,
            "target_month": target_month,
            "lead_months": max(lead, 0),
            "nino34_anom": round(n34_val, 4),
            "tropical_mean_anom": round(trop_val, 4) if not np.isnan(trop_val) else float("nan"),
            "roni_anom": round(roni, 4) if not np.isnan(roni) else float("nan"),
            "member_id": member_id,
            "temporal_resolution": "monthly",
            "anomaly_base_period": "model-specific",
        }

    if ens_dim:
        n_members = nino34.sizes[ens_dim]
        ens_coords = ds.coords[ens_dim].values

        for m_idx in range(n_members):
            member_id = f"ens_{int(ens_coords[m_idx]):03d}"
            member_n34 = nino34.isel({ens_dim: m_idx})

            for t_idx, target_dt in enumerate(target_dates):
                val = float(member_n34.isel({target_dim: t_idx}).values)
                if np.isnan(val):
                    continue
                trop = _trop_val(m_idx, t_idx)
                records.append(_make_record(member_id, target_dt, val, trop))

        # Ensemble mean
        ens_mean_n34 = nino34.mean(dim=ens_dim)
        ens_mean_trop = tropical.mean(dim=ens_dim) if tropical is not None and ens_dim in tropical.dims else tropical
        for t_idx, target_dt in enumerate(target_dates):
            val = float(ens_mean_n34.isel({target_dim: t_idx}).values)
            if np.isnan(val):
                continue
            if ens_mean_trop is None:
                trop = float("nan")
            else:
                trop = float(ens_mean_trop.isel({target_dim: t_idx}).values)
            records.append(_make_record("mean", target_dt, val, trop))

        logger.info("NMME %s: %d members x %d lead times", display_name, n_members, len(target_dates))
    else:
        for t_idx, target_dt in enumerate(target_dates):
            val = float(nino34.isel({target_dim: t_idx}).values)
            if np.isnan(val):
                continue
            trop = _trop_val(None, t_idx)
            records.append(_make_record("mean", target_dt, val, trop))
        logger.info("NMME %s: %d lead times (ensemble mean only)", display_name, len(target_dates))

    ds.close()
    return records


def fetch_nmme(
    yyyymm: str | None = None,
    include_members: bool = True,
) -> pd.DataFrame:
    """Download and process NMME data for a given month.

    Args:
        yyyymm: Target month as YYYYMM string. If None, uses latest available.
        include_members: If True, download individual ensemble member files
            (~75MB each). If False, only download ENSMEAN files (~2MB each).
    """
    if yyyymm is None:
        yyyymm = _find_latest_nmme_date()

    nmme_raw = RAW_DIR / "nmme" / yyyymm
    nmme_raw.mkdir(parents=True, exist_ok=True)
    init_date = f"{yyyymm[:4]}-{yyyymm[4:6]}-01"

    all_records = []

    for file_model in NMME_FILE_MODELS:
        display_name = NMME_MODEL_DISPLAY.get(file_model, file_model)

        if include_members:
            # Individual member files: {MODEL}/{YYYYMM}0800/{MODEL}.tmpsfc.{YYYYMM}.anom.nc
            filename = f"{file_model}.tmpsfc.{yyyymm}.anom.nc"
            file_url = f"{NMME_BASE_URL}{file_model}/{yyyymm}0800/{filename}"
        else:
            # ENSMEAN files: ENSMEAN/{YYYYMM}0800/{MODEL}.tmpsfc.{YYYYMM}.ENSMEAN.anom.nc
            filename = f"{file_model}.tmpsfc.{yyyymm}.ENSMEAN.anom.nc"
            file_url = f"{NMME_BASE_URL}ENSMEAN/{yyyymm}0800/{filename}"

        local_path = nmme_raw / filename
        logger.info("Fetching NMME %s from %s", display_name, file_url)

        if not _download_file(file_url, local_path):
            logger.warning("Skipping NMME %s: download failed", display_name)
            continue

        try:
            records = _process_member_file(local_path, file_model, display_name, init_date)
            all_records.extend(records)
        except Exception as e:
            logger.warning("Error processing NMME %s: %s", display_name, e)
            continue

    # Also fetch NMME overall mean from ENSMEAN directory
    nmme_mean_filename = f"NMME.tmpsfc.{yyyymm}.ENSMEAN.anom.nc"
    nmme_mean_url = f"{NMME_BASE_URL}ENSMEAN/{yyyymm}0800/{nmme_mean_filename}"
    nmme_mean_path = nmme_raw / nmme_mean_filename
    logger.info("Fetching NMME overall mean")

    if _download_file(nmme_mean_url, nmme_mean_path):
        try:
            records = _process_member_file(
                nmme_mean_path, "NMME", "NMME", init_date,
            )
            all_records.extend(records)
        except Exception as e:
            logger.warning("Error processing NMME overall mean: %s", e)

    df = pd.DataFrame(all_records)

    # Filter out trailing zero-fill values from NMME overall mean
    if len(df) > 0:
        nmme_overall = df[df["model"] == "NMME"]
        if len(nmme_overall) > 0:
            last_valid_idx = nmme_overall[nmme_overall["nino34_anom"] != 0.0].index
            if len(last_valid_idx) > 0:
                trailing_zeros = nmme_overall[
                    (nmme_overall["nino34_anom"] == 0.0)
                    & (nmme_overall.index > last_valid_idx.max())
                ]
                if len(trailing_zeros) > 0:
                    df = df.drop(trailing_zeros.index).reset_index(drop=True)
                    logger.info("Dropped %d trailing zero-fill values from NMME mean", len(trailing_zeros))

    if len(df) > 0:
        n_models = df["model"].nunique()
        n_members = df[df["member_id"] != "mean"]["member_id"].nunique()
        logger.info("NMME: %d records from %d models (%d individual members)",
                     len(df), n_models, n_members)
    else:
        logger.info("NMME: 0 records")

    return df


def save_nmme(force: bool = False) -> pd.DataFrame:
    """Fetch and save NMME data."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "NMME"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("NMME data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_nmme(include_members=True)
    if df.empty:
        logger.warning("NMME fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved NMME data to %s", out_path)
    return df
