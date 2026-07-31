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
    OBSERVED_DIR,
    RAW_DIR,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)

# Tropical band for rONI denominator (L'Heureux et al. 2024)
TROPICAL_LAT_BOUNDS = (-20.0, 20.0)

# Observed ENSO -> tropical-mean teleconnection, used to evolve the
# ensemble-mean tropical anomaly across forecast leads (see
# _evolved_tropical_anomalies). Fit on OISSTv2.1 daily-box monthly means
# 1982-2026, lowess-detrended: tropics lag Nino 3.4 by 2 months with
# slope 0.192 C/C (r = 0.84). Caveat: linear extrapolation beyond the
# fitted Nino 3.4 range (max ~2.6 C observed) — the 2026 forecast event
# exceeds it, and dynamical models (NMME/C3S) suggest mild saturation.
TROPICS_ENSO_SLOPE = 0.192
TROPICS_ENSO_LAG = 2

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


def _compute_area_means_per_member(
    grib_path: Path,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute per-member cosine-lat-weighted area-mean SST (Kelvin) for
    Niño 3.4 and the 20S-20N tropical band.

    Returns (nino34_ssts, tropical_ssts) — each a dict {member_num: SST}.
    """
    ds = xr.open_dataset(grib_path, engine="cfgrib")

    var_name = None
    for v in ds.data_vars:
        vl = v.lower()
        if "sst" in vl or "tmp" in vl or "water" in vl or "t" == vl:
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

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

    squeeze_dims = [
        d for d in data.dims
        if d not in (lat_name, lon_name, member_dim) and data.sizes[d] == 1
    ]
    if squeeze_dims:
        data = data.squeeze(dim=squeeze_dims, drop=True)

    lat = ds.coords[lat_name]
    descending = float(lat[0]) > float(lat[-1])

    def _area_mean(lat_bounds, lon_bounds):
        lat_min, lat_max = lat_bounds
        lat_slice = slice(lat_max, lat_min) if descending else slice(lat_min, lat_max)
        sel = {lat_name: lat_slice}
        if lon_bounds is not None:
            sel[lon_name] = slice(lon_bounds[0], lon_bounds[1])
        region = data.sel(sel)
        if region.sizes[lat_name] == 0 or region.sizes[lon_name] == 0:
            raise ValueError(f"Empty selection lat={lat_bounds} lon={lon_bounds}")
        w = np.cos(np.deg2rad(region.coords[lat_name]))
        return region.weighted(w).mean(dim=[lat_name, lon_name])

    nino34_mean = _area_mean(NINO34_LAT_BOUNDS, NINO34_LON_BOUNDS_360)
    tropical_mean = _area_mean(TROPICAL_LAT_BOUNDS, None)

    def _to_dict(mean_arr):
        out = {}
        if member_dim is not None:
            mc = ds.coords[member_dim].values
            for m_idx in range(mean_arr.sizes[member_dim]):
                v = float(mean_arr.isel({member_dim: m_idx}).values)
                if not np.isnan(v):
                    out[int(mc[m_idx])] = v
        else:
            out[0] = float(mean_arr.values)
        return out

    n34_d, trop_d = _to_dict(nino34_mean), _to_dict(tropical_mean)
    ds.close()
    return n34_d, trop_d


def _compute_nino34_per_member(grib_path: Path) -> dict[int, float]:
    """Backwards-compatible wrapper returning only the Niño 3.4 dict."""
    n34, _ = _compute_area_means_per_member(grib_path)
    return n34


def _load_obs_nino34_monthly() -> pd.Series:
    """Observed monthly Niño 3.4 anomalies indexed by month start."""
    path = OBSERVED_DIR / "nino34_monthly.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    obs = pd.read_csv(path)
    obs["date"] = pd.to_datetime(obs["date"])
    return obs.set_index("date")["nino34_anom"].dropna().sort_index()


def _observed_tropical_anchor(init_dt: date) -> tuple[float, pd.Timestamp | None]:
    """(tropical-mean SSTA, anchor month) in the BARE convention: obs
    Niño 3.4 anomaly minus the de-scaled monthly relative index, both from
    the latest calendar month available in both observed files.

    The de-scaling matters: NOAA's published rNINO3.4 (rnino_monthly.csv)
    carries the L'Heureux et al. (2024) variance-restoration factor, while
    the member rONI computed here is the bare difference that gets scaled
    downstream in normalize.apply_roni_scaling. Subtracting *scaled* RONI
    from Niño 3.4 — or pairing different months during a fast-evolving
    event — inflated this baseline by up to ~0.4 °C (bug fixed 2026-07-31;
    verified against tropical-box anomalies from the OISST daily pipeline).
    """
    from enso_forecast.normalize import RONI_SCALING_MONTHLY

    nino_path = OBSERVED_DIR / "nino34_monthly.csv"
    rnino_path = OBSERVED_DIR / "rnino_monthly.csv"
    if not (nino_path.exists() and rnino_path.exists()):
        return 0.0, None
    obs = pd.read_csv(nino_path)
    rnino = pd.read_csv(rnino_path)
    obs["date"] = pd.to_datetime(obs["date"])
    rnino["date"] = pd.to_datetime(rnino["date"])
    cutoff = pd.Timestamp(init_dt)

    merged = obs.merge(rnino[["date", "rnino34"]], on="date", how="inner")
    merged = merged[merged["date"] <= cutoff].dropna(
        subset=["nino34_anom", "rnino34"]).sort_values("date")
    if merged.empty:
        return 0.0, None
    last = merged.iloc[-1]
    a = RONI_SCALING_MONTHLY[int(last["date"].month)]
    return (float(last["nino34_anom"]) - float(last["rnino34"]) / a,
            pd.Timestamp(last["date"]))


def _evolved_tropical_anomalies(
    init_dt: date,
    csv_anomalies: dict[int, float],
    first_month: pd.Timestamp,
) -> dict[int, float]:
    """Per-lead ensemble-mean tropical anomaly (BARE convention).

    Starts from the observed anchor (_observed_tropical_anchor) and evolves
    it along the observed ENSO->tropics teleconnection:

        trop(T) = trop(anchor) + SLOPE * (n34(T - LAG) - n34(anchor - LAG))

    where the Niño 3.4 trajectory splices observed monthly values with the
    run's own CSV ensemble means (gaps, e.g. the init month, interpolated
    linearly). A constant baseline was used before 2026-07-31; during the
    2026 event that pinned the tropics ~0.3-0.6 °C below what NMME and C3S
    members forecast for the same months and forced the two sub-models'
    tropical anomalies to be mirror images around it (sending CanESM5's
    tropics negative in mid-2027). The evolved form reproduces the observed
    July-2026 tropical anomaly (+0.79 predicted vs +0.78 OISST daily).
    """
    anchor_trop, anchor_month = _observed_tropical_anchor(init_dt)
    if anchor_month is None:
        return {lead: anchor_trop for lead in csv_anomalies}

    obs = _load_obs_nino34_monthly()
    fc = pd.Series({first_month + pd.DateOffset(months=lead): val
                    for lead, val in csv_anomalies.items()})
    n34 = pd.concat([obs[~obs.index.isin(fc.index)], fc]).sort_index()
    n34 = n34.resample("MS").mean().interpolate(limit_area="inside")

    def n34_at(month: pd.Timestamp) -> float:
        if month in n34.index:
            return float(n34.loc[month])
        prior = n34.loc[:month]
        return float(prior.iloc[-1]) if not prior.empty else 0.0

    ref = n34_at(anchor_month - pd.DateOffset(months=TROPICS_ENSO_LAG))
    out = {}
    for lead in csv_anomalies:
        lag_month = (first_month + pd.DateOffset(months=lead)
                     - pd.DateOffset(months=TROPICS_ENSO_LAG))
        out[lead] = anchor_trop + TROPICS_ENSO_SLOPE * (n34_at(lag_month) - ref)
    return out


def fetch_cansips(
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Download and process CanSIPS ensemble forecasts.

    Downloads GRIB2 files for individual ensemble members and the CSV
    for ensemble mean anomalies. Converts raw SST to anomalies using:
        anomaly_i = (SST_i - ensemble_mean_SST) + csv_mean_anomaly

    Members 1-20 → CanSIPS-CanESM5, members 21-40 → CanSIPS-GEM-NEMO.

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

    # Datamart labels GRIB files by the run's official init month, which is
    # the month AFTER the CSV's init stamp: the run posted 2026-06-30 is CSV
    # 2026063000 but GRIB 202607_* (reference time 2026-07-01 00Z), and P00M
    # is valid for the label month itself. Using the CSV month here would
    # silently pair the new CSV with the previous month's run.
    grib_dt = pd.Timestamp(f"{year}-{month:02d}-01") + pd.DateOffset(months=1)
    grib_yyyymm = f"{grib_dt.year}{grib_dt.month:02d}"

    raw_dir = RAW_DIR / "cansips" / grib_yyyymm
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble-mean tropical anomaly per lead: observed anchor evolved along
    # the ENSO->tropics teleconnection (grib_dt is the first forecast month,
    # matching csv_anomalies lead 0).
    trop_by_lead = _evolved_tropical_anomalies(init_dt, csv_anomalies, grib_dt)
    logger.info(
        "CanSIPS: tropical-mean anomaly anchor %+.3f °C, evolved %+.3f..%+.3f "
        "across leads",
        _observed_tropical_anchor(init_dt)[0],
        min(trop_by_lead.values()), max(trop_by_lead.values()),
    )

    all_records = []

    for lead_idx in range(12):
        filename = f"{grib_yyyymm}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{lead_idx:02d}M.grib2"
        url = f"{CANSIPS_GRIB_BASE}/{grib_dt.year}/{grib_dt.month:02d}/{filename}"
        local_path = raw_dir / filename

        logger.info("Fetching CanSIPS lead %d from %s", lead_idx, url)
        if not _download_file(url, local_path):
            logger.warning("Skipping CanSIPS lead %d: download failed", lead_idx)
            continue

        try:
            member_n34_ssts, member_trop_ssts = _compute_area_means_per_member(local_path)
        except Exception as e:
            logger.warning("Error processing CanSIPS lead %d: %s", lead_idx, e)
            continue

        if not member_n34_ssts or lead_idx not in csv_anomalies:
            logger.warning("Skipping lead %d: no data", lead_idx)
            continue

        ensemble_mean_n34_sst = float(np.mean(list(member_n34_ssts.values())))
        ensemble_mean_trop_sst = (
            float(np.mean(list(member_trop_ssts.values()))) if member_trop_ssts else None
        )
        csv_mean_anom = csv_anomalies[lead_idx]

        # P{lead}M is valid for grib label month + lead, matching CSV lead order
        target_dt = grib_dt + pd.DateOffset(months=lead_idx)
        target_month = target_dt.strftime("%Y-%m")

        for member_num, sst_val in member_n34_ssts.items():
            anomaly = (sst_val - ensemble_mean_n34_sst) + csv_mean_anom

            if member_trop_ssts and ensemble_mean_trop_sst is not None and member_num in member_trop_ssts:
                trop_anom = (
                    (member_trop_ssts[member_num] - ensemble_mean_trop_sst)
                    + trop_by_lead[lead_idx]
                )
                roni = anomaly - trop_anom
            else:
                trop_anom = float("nan")
                roni = float("nan")

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
                "tropical_mean_anom": round(trop_anom, 4) if not np.isnan(trop_anom) else float("nan"),
                "roni_anom": round(roni, 4) if not np.isnan(roni) else float("nan"),
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

        agg_spec = {"nino34_anom": "mean", "lead_months": "first"}
        if "tropical_mean_anom" in model_members.columns:
            agg_spec["tropical_mean_anom"] = "mean"
            agg_spec["roni_anom"] = "mean"
        model_mean = (
            model_members.groupby("target_month", as_index=False)
            .agg(agg_spec)
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
