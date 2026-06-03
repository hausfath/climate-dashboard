"""Fetch CFS v2 Nino3.4 ensemble plume data from NOAA CPC."""

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from enso_forecast.config import (
    CFS_E3_URL,
    CFS_URLS,
    FORECASTS_DIR,
    MAX_RETRIES,
    RAW_DIR,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    RNINO34_E3_URL,
    RNINO34_URLS,
)

logger = logging.getLogger(__name__)


def _download_netcdf(url: str, dest: Path) -> Path:
    """Download a NetCDF file with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info("Downloading %s (attempt %d/%d)", url, attempt + 1, MAX_RETRIES)
            resp = requests.get(
                url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, stream=True
            )
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Saved to %s", dest)
            return dest
        except (requests.RequestException, IOError) as e:
            logger.warning("Download failed: %s", e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
    return dest  # unreachable but satisfies type checker


def _parse_initial_time_attr(ds: xr.Dataset) -> str | None:
    """Parse the earliest member init date from the `initial_time` attr.

    CFSv2 NetCDFs from CPC carry this on the `anom` variable, formatted as
    "YYYYMMDD - YYYYMMDD" (earliest – latest run). The first date is the
    earliest of the rolling member init timestamps in the file.
    """
    for source in (ds["anom"].attrs if "anom" in ds.data_vars else {}, ds.attrs):
        raw = source.get("initial_time")
        if not raw:
            continue
        first_token = str(raw).split("-")[0].strip()
        try:
            return pd.to_datetime(first_token, format="%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _parse_initial_time_window(ds: xr.Dataset) -> tuple[str, str] | None:
    """Return (earliest_init, latest_init) as YYYY-MM-DD from the
    ``initial_time`` attribute "YYYYMMDD - YYYYMMDD", or None if absent.
    """
    for source in (ds["anom"].attrs if "anom" in ds.data_vars else {}, ds.attrs):
        raw = source.get("initial_time")
        if not raw:
            continue
        parts = [p.strip() for p in str(raw).split("-")]
        if len(parts) < 2:
            continue
        try:
            earliest = pd.to_datetime(parts[0], format="%Y%m%d")
            latest = pd.to_datetime(parts[1], format="%Y%m%d")
            return earliest.strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _detect_init_date(ds: xr.Dataset) -> str:
    """Detect the effective initialization date from CFS data.

    Preferred: the earliest member init timestamp from the file's
    ``initial_time`` attribute (e.g. "20260423 - 20260502" → "2026-04-23"),
    which reflects the oldest of the rolling 40 4×daily runs in the E3 file.

    Fallback: ensemble-spread detection. The CFS NetCDF includes historical
    verification months (members identical, std=0) followed by forecast
    months (members diverge). Init is then the month before the first month
    with ensemble spread — month-resolution only.
    """
    earliest = _parse_initial_time_attr(ds)
    if earliest is not None:
        return earliest

    data = ds["anom"].squeeze(drop=True)

    # Find ensemble and time dims
    ens_dim = time_dim = None
    for dim in data.dims:
        if "ens" in dim.lower():
            ens_dim = dim
        elif "time" in dim.lower():
            time_dim = dim
        elif dim in ds.coords and np.issubdtype(ds.coords[dim].dtype, np.datetime64):
            time_dim = dim

    if ens_dim is None or time_dim is None:
        # Fallback: use the history attribute creation date
        history = ds.attrs.get("history", "")
        if "created:" in history:
            try:
                created_str = history.split("created:")[1].strip().split()[0]
                return pd.Timestamp(created_str).replace(day=1).strftime("%Y-%m-%d")
            except Exception:
                pass
        return date.today().replace(day=1).isoformat()

    time_values = ds.coords[time_dim].values
    stds = data.std(dim=ens_dim)

    for i, t in enumerate(time_values):
        s = float(stds.isel({time_dim: i}))
        if s > 0.001:
            # First month with spread — init is the month before
            forecast_start = pd.Timestamp(t)
            init = forecast_start - pd.DateOffset(months=1)
            return init.replace(day=1).strftime("%Y-%m-%d")

    # No spread found — use last verification month
    last_t = pd.Timestamp(time_values[-1])
    return last_t.replace(day=1).strftime("%Y-%m-%d")


def _process_cfs_file(
    nc_path: Path, ensemble_label: str
) -> tuple[list[dict], str]:
    """Process a single CFS NetCDF file.

    Returns (records, init_date).
    """
    ds = xr.open_dataset(nc_path)
    init_date = _detect_init_date(ds)
    # CPC's CFSv2 NetCDF tags the 40-member rolling window with a date
    # range on the ``initial_time`` attribute. The notes attribute clarifies
    # member ordering: "ENS=1 to 40 forecast members from the latest to
    # earliest initial times; ENS=41 is observation". We use that ordering
    # to assign a per-member init_date evenly spaced across the window
    # rather than labelling all 40 members with the same earliest date.
    init_window = _parse_initial_time_window(ds)
    logger.info("CFS %s: detected init_date = %s, window = %s",
                ensemble_label, init_date, init_window)

    # Find variable
    var_name = None
    for v in ds.data_vars:
        if "anom" in v.lower() or "nino" in v.lower() or "sst" in v.lower():
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name].squeeze(drop=True)

    # Identify dims
    ens_dim = time_dim = None
    for dim in data.dims:
        if dim in ds.coords and np.issubdtype(ds.coords[dim].dtype, np.datetime64):
            time_dim = dim
        elif "ens" in dim.lower() or "member" in dim.lower():
            ens_dim = dim
        elif "time" in dim.lower():
            time_dim = dim

    if time_dim is None or ens_dim is None:
        # Fallback for non-standard dims
        for dim in data.dims:
            if dim == time_dim or dim == ens_dim:
                continue
            if data.sizes[dim] > 20:
                ens_dim = dim
            else:
                time_dim = dim

    time_values = ds.coords[time_dim].values if time_dim else []

    records = []

    if ens_dim and time_dim:
        n_members = data.sizes[ens_dim]

        # Build per-member init dates. Members 1..40 are rolling forecast
        # runs from latest → earliest across the ``initial_time`` window;
        # member 41 is the observation row (kept tagged with the latest
        # init for compatibility with downstream filters).
        def _member_init_date(m_idx: int) -> str:
            if init_window is None or m_idx >= 40:
                return init_date
            earliest = pd.Timestamp(init_window[0])
            latest = pd.Timestamp(init_window[1])
            span_days = (latest - earliest).days
            if span_days <= 0:
                return latest.strftime("%Y-%m-%d")
            # ENS=1 (m_idx=0) → latest, ENS=40 (m_idx=39) → earliest.
            offset_days = round(span_days * m_idx / 39)
            return (latest - pd.Timedelta(days=offset_days)).strftime("%Y-%m-%d")

        for m_idx in range(n_members):
            member_id = f"ens_{ensemble_label}_{m_idx + 1:03d}"
            member_data = data.isel({ens_dim: m_idx})
            member_init = _member_init_date(m_idx)

            for t_idx in range(len(time_values)):
                val = float(member_data.isel({time_dim: t_idx}).values)
                if np.isnan(val):
                    continue

                t_val = time_values[t_idx]
                target_dt = pd.Timestamp(t_val) if np.issubdtype(type(t_val), np.datetime64) else None
                if target_dt is None:
                    continue
                target_month = target_dt.strftime("%Y-%m")

                init_dt = pd.Timestamp(member_init)
                lead = (target_dt.year - init_dt.year) * 12 + (target_dt.month - init_dt.month)

                records.append({
                    "source": "CFS",
                    "model": "CFSv2",
                    "model_type": "dynamical",
                    "init_date": member_init,
                    "target_month": target_month,
                    "nino34_anom": val,
                    "member_id": member_id,
                    "lead_months": max(lead, 0),
                    "temporal_resolution": "monthly",
                    "anomaly_base_period": "1991-2020",
                })

    ds.close()
    return records, init_date


def fetch_cfs(e3_only: bool = True) -> pd.DataFrame:
    """Download and parse CFS v2 Nino3.4 ensemble data.

    Args:
        e3_only: If True (default), only download the latest 10-day ensemble (E3).
            If False, download all three ensemble groups (E1, E2, E3).

    Returns DataFrame with individual ensemble members and ensemble mean.
    """
    cfs_raw = RAW_DIR / "cfs"
    cfs_raw.mkdir(parents=True, exist_ok=True)

    all_records = []
    init_date = None

    if e3_only:
        urls = [("E3", CFS_E3_URL)]
        rurls = [("E3", RNINO34_E3_URL)]
    else:
        urls = [(f"E{i}", url) for i, url in enumerate(CFS_URLS, start=1)]
        rurls = [(f"E{i}", url) for i, url in enumerate(RNINO34_URLS, start=1)]

    for ensemble_label, url in urls:
        nc_path = cfs_raw / f"nino34Mon_{ensemble_label}.nc"
        _download_netcdf(url, nc_path)

        records, file_init = _process_cfs_file(nc_path, ensemble_label)
        all_records.extend(records)

        if init_date is None:
            init_date = file_init

    df = pd.DataFrame(all_records)

    if len(df) == 0:
        logger.warning("No CFS data extracted")
        return df

    # Fetch published rONI (relative Niño 3.4) and merge by member/target_month.
    # Same NetCDF schema as nino34Mon — the `anom` variable holds Niño 3.4 SSTA
    # minus tropical-mean SSTA per ensemble member.
    roni_records = []
    for ensemble_label, rurl in rurls:
        try:
            rnc_path = cfs_raw / f"rnino34Mon_{ensemble_label}.nc"
            _download_netcdf(rurl, rnc_path)
            recs, _ = _process_cfs_file(rnc_path, ensemble_label)
            roni_records.extend(recs)
        except Exception as e:
            logger.warning("Failed to fetch CFSv2 rONI for %s: %s", ensemble_label, e)

    if roni_records:
        rdf = pd.DataFrame(roni_records)[["member_id", "target_month", "nino34_anom"]]
        rdf = rdf.rename(columns={"nino34_anom": "roni_anom"})
        df = df.merge(rdf, on=["member_id", "target_month"], how="left")
        df["tropical_mean_anom"] = df["nino34_anom"] - df["roni_anom"]
    else:
        df["roni_anom"] = np.nan
        df["tropical_mean_anom"] = np.nan

    # Compute ensemble mean per target_month
    member_df = df[df["member_id"] != "mean"].copy()
    if len(member_df) > 0:
        agg_spec = {"nino34_anom": "mean", "lead_months": "first"}
        if "roni_anom" in member_df.columns:
            agg_spec["roni_anom"] = "mean"
            agg_spec["tropical_mean_anom"] = "mean"
        mean_df = (
            member_df.groupby("target_month", as_index=False)
            .agg(agg_spec)
        )
        mean_df["source"] = "CFS"
        mean_df["model"] = "CFSv2"
        mean_df["model_type"] = "dynamical"
        mean_df["init_date"] = init_date
        mean_df["member_id"] = "mean"
        mean_df["temporal_resolution"] = "monthly"
        mean_df["anomaly_base_period"] = "1991-2020"
        df = pd.concat([df, mean_df], ignore_index=True)

    n_members = df[df["member_id"] != "mean"]["member_id"].nunique()
    logger.info(
        "CFS: %d total records, %d unique members, %d target months, init=%s",
        len(df), n_members, df["target_month"].nunique(), init_date,
    )
    return df


def save_cfs(force: bool = False) -> pd.DataFrame:
    """Fetch and save CFS v2 ensemble data."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "CFS"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("CFS data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_cfs(e3_only=True)
    if df.empty:
        logger.warning("CFS fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved CFS data to %s", out_path)
    return df
