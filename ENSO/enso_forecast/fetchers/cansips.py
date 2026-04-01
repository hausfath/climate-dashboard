"""Fetch CanSIPS Nino3.4 ensemble forecasts from MSC Datamart (GRIB2 + CSV)."""

import calendar
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from enso_forecast.config import (
    CANSIPS_CSV_BASE,
    CANSIPS_GRIB_BASE,
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

# Month abbreviations used in the CanSIPS CSV (note: French "MAI" for May)
_CSV_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAI", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Member ranges per sub-model in the 1-degree GRIB2 files
# Verified by comparing spread patterns with NMME's separate model files:
# GEM-NEMO is the warmer model (matching NMME GEM5.2-NEMO) = members 21-40
# CanESM5 is the cooler model (matching NMME CanESM5) = members 1-20
_CANESM5_MEMBERS = range(1, 21)     # perturbation numbers 1-20
_GEM_NEMO_MEMBERS = range(21, 41)   # perturbation numbers 21-40


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with retry logic. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, stream=True,
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


def _get_init_date(year: int, month: int) -> date:
    """Return the last day of the given month (CanSIPS init convention)."""
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


def _fetch_ensemble_mean_csv(init_dt: date) -> dict[int, float]:
    """Download the CanSIPS CSV and parse the Nino3.4 ensemble mean.

    Returns dict mapping lead_month index (0-11) to Nino3.4 anomaly.
    """
    # Forecast period: month after init through 12 months out
    first_month = init_dt.month % 12 + 1
    first_year = init_dt.year + (1 if init_dt.month == 12 else 0)
    last_dt = pd.Timestamp(f"{first_year}-{first_month:02d}-01") + pd.DateOffset(months=11)

    filename = (
        f"{init_dt.strftime('%Y%m%d')}00_indices_month_"
        f"{first_year}{first_month:02d}_{last_dt.year}{last_dt.month:02d}.csv"
    )
    url = f"{CANSIPS_CSV_BASE}/{filename}"
    logger.info("Fetching CanSIPS CSV from %s", url)

    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # Parse CSV
    from io import StringIO
    csv_df = pd.read_csv(StringIO(resp.text))

    # Find Nino3.4 row (case-insensitive)
    nino_row = csv_df[csv_df["INDEX"].str.strip().str.lower() == "nino3.4"]
    if nino_row.empty:
        raise ValueError("Nino3.4 row not found in CanSIPS CSV")

    row = nino_row.iloc[0]
    anomalies = {}
    col_names = [c for c in csv_df.columns if c != "INDEX"]

    for i, col in enumerate(col_names):
        val = row[col]
        try:
            anomalies[i] = float(val)
        except (ValueError, TypeError):
            logger.warning("Could not parse CSV value for %s: %s", col, val)

    logger.info("CanSIPS CSV: %d lead months, values %.2f to %.2f",
                len(anomalies),
                min(anomalies.values()) if anomalies else 0,
                max(anomalies.values()) if anomalies else 0)
    return anomalies


def _compute_nino34_per_member(grib_path: Path) -> dict[int, float]:
    """Compute cosine-lat-weighted Nino3.4 area mean SST per ensemble member.

    Returns dict mapping perturbation number to Nino3.4 SST (Kelvin).
    """
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    # Find SST variable
    var_name = None
    for v in ds.data_vars:
        vl = v.lower()
        if "sst" in vl or "tmp" in vl or "water" in vl or "t" == vl:
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

    # Identify lat/lon dims
    lat_name = lon_name = member_dim = None
    for dim in data.dims:
        dl = dim.lower()
        if "lat" in dl:
            lat_name = dim
        elif "lon" in dl:
            lon_name = dim
        elif "number" in dl or "pert" in dl or "member" in dl:
            member_dim = dim

    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot identify lat/lon dims from {data.dims}")

    # Squeeze singleton dims (preserve member dim)
    squeeze_dims = [
        d for d in data.dims
        if d not in (lat_name, lon_name, member_dim) and data.sizes[d] == 1
    ]
    if squeeze_dims:
        data = data.squeeze(dim=squeeze_dims, drop=True)

    # Select Nino3.4 region
    lat = ds.coords[lat_name]
    lon_min, lon_max = NINO34_LON_BOUNDS_360
    lat_min, lat_max = NINO34_LAT_BOUNDS

    if float(lat[0]) > float(lat[-1]):
        region = data.sel(
            {lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)}
        )
    else:
        region = data.sel(
            {lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)}
        )

    if region.sizes[lat_name] == 0 or region.sizes[lon_name] == 0:
        raise ValueError("Empty Nino3.4 selection")

    # Cosine-latitude weighted area mean
    weights = np.cos(np.deg2rad(region.coords[lat_name]))
    area_mean = region.weighted(weights).mean(dim=[lat_name, lon_name])

    # Extract per-member values
    results = {}
    if member_dim is not None:
        member_coords = ds.coords[member_dim].values
        for m_idx in range(area_mean.sizes[member_dim]):
            member_num = int(member_coords[m_idx])
            val = float(area_mean.isel({member_dim: m_idx}).values)
            if not np.isnan(val):
                results[member_num] = val
    else:
        results[0] = float(area_mean.values)

    ds.close()
    return results


def fetch_cansips(
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Download and process CanSIPS ensemble forecasts.

    Downloads GRIB2 files for individual ensemble members and the CSV
    for ensemble mean anomalies. Converts raw SST to anomalies using:
        anomaly_i = (SST_i - ensemble_mean_SST) + csv_mean_anomaly

    Members 1-20 → CanSIPS-GEM-NEMO, members 21-40 → CanSIPS-CanESM5.

    Returns DataFrame in standard forecast schema with both sub-models.
    """
    if year is None:
        year = date.today().year
    if month is None:
        month = date.today().month

    # CanSIPS is initialized at end-of-month, so data for the current month
    # may not be available yet (especially early in the month). Try the
    # current month first, then fall back to the previous month.
    init_dt = _get_init_date(year, month)
    csv_anomalies = None
    try:
        csv_anomalies = _fetch_ensemble_mean_csv(init_dt)
    except Exception as e:
        logger.warning("CanSIPS CSV not available for %s: %s. Trying previous month.", init_dt, e)
        prev = pd.Timestamp(f"{year}-{month:02d}-01") - pd.DateOffset(months=1)
        year, month = prev.year, prev.month
        init_dt = _get_init_date(year, month)
        try:
            csv_anomalies = _fetch_ensemble_mean_csv(init_dt)
        except Exception as e2:
            logger.error("Failed to fetch CanSIPS CSV for %s: %s", init_dt, e2)
            return pd.DataFrame()

    init_date_str = init_dt.isoformat()
    yyyymm = f"{year}{month:02d}"

    raw_dir = RAW_DIR / "cansips" / yyyymm
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Download and process GRIB2 files for each lead month
    all_records = []

    for lead_idx in range(12):
        filename = f"{yyyymm}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{lead_idx:02d}M.grib2"
        url = f"{CANSIPS_GRIB_BASE}/{year}/{month:02d}/{filename}"
        local_path = raw_dir / filename

        logger.info("Fetching CanSIPS lead %d from %s", lead_idx, url)
        if not _download_file(url, local_path):
            logger.warning("Skipping CanSIPS lead %d: download failed", lead_idx)
            continue

        try:
            member_ssts = _compute_nino34_per_member(local_path)
        except Exception as e:
            logger.warning("Error processing CanSIPS lead %d: %s", lead_idx, e)
            continue

        if not member_ssts or lead_idx not in csv_anomalies:
            logger.warning("Skipping lead %d: no data", lead_idx)
            continue

        # Compute ensemble mean SST across all members
        all_sst_vals = list(member_ssts.values())
        ensemble_mean_sst = np.mean(all_sst_vals)
        csv_mean_anom = csv_anomalies[lead_idx]

        # Target month for this lead
        target_dt = pd.Timestamp(f"{year}-{month:02d}-01") + pd.DateOffset(months=lead_idx + 1)
        target_month = target_dt.strftime("%Y-%m")

        # Convert each member SST to anomaly and assign sub-model
        for member_num, sst_val in member_ssts.items():
            anomaly = (sst_val - ensemble_mean_sst) + csv_mean_anom

            if member_num in _GEM_NEMO_MEMBERS:
                model_name = "CanSIPS-GEM-NEMO"
            elif member_num in _CANESM5_MEMBERS:
                model_name = "CanSIPS-CanESM5"
            else:
                model_name = "CanSIPS"

            all_records.append({
                "source": "CanSIPS",
                "model": model_name,
                "model_type": "dynamical",
                "init_date": init_date_str,
                "target_month": target_month,
                "lead_months": lead_idx + 1,
                "nino34_anom": round(anomaly, 4),
                "member_id": f"ens_{member_num:03d}",
                "temporal_resolution": "monthly",
                "anomaly_base_period": "1991-2020",
            })

    df = pd.DataFrame(all_records)

    if df.empty:
        logger.warning("No CanSIPS data extracted")
        return df

    # Compute per-sub-model ensemble means
    member_df = df[df["member_id"] != "mean"].copy()
    mean_rows = []
    for model_name in ["CanSIPS-GEM-NEMO", "CanSIPS-CanESM5"]:
        model_members = member_df[member_df["model"] == model_name]
        if model_members.empty:
            continue

        model_mean = (
            model_members.groupby("target_month", as_index=False)
            .agg({"nino34_anom": "mean", "lead_months": "first"})
        )
        model_mean["source"] = "CanSIPS"
        model_mean["model"] = model_name
        model_mean["model_type"] = "dynamical"
        model_mean["init_date"] = init_date_str
        model_mean["member_id"] = "mean"
        model_mean["temporal_resolution"] = "monthly"
        model_mean["anomaly_base_period"] = "1991-2020"
        mean_rows.append(model_mean)

    if mean_rows:
        df = pd.concat([df] + mean_rows, ignore_index=True)

    n_gem = df[(df["model"] == "CanSIPS-GEM-NEMO") & (df["member_id"] != "mean")]["member_id"].nunique()
    n_esm = df[(df["model"] == "CanSIPS-CanESM5") & (df["member_id"] != "mean")]["member_id"].nunique()
    logger.info(
        "CanSIPS: %d total records, GEM-NEMO=%d members, CanESM5=%d members, "
        "%d target months, init=%s",
        len(df), n_gem, n_esm, df["target_month"].nunique(), init_date_str,
    )
    return df


def save_cansips(force: bool = False) -> pd.DataFrame:
    """Fetch and save CanSIPS ensemble data."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "CanSIPS"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("CanSIPS data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_cansips()
    if df.empty:
        logger.warning("CanSIPS fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved CanSIPS data to %s", out_path)
    return df
