"""Fetch ECMWF EC46 (extended-range, 46-day) global-mean 2m temperature
forecasts for the dashboard's daily anomaly plots.

Source: Open-Meteo seasonal-forecast API with `models=ecmwf_ec46`.
  - 6-hourly 2m temperature + 50 perturbed members per gridpoint.
  - 46-day forecast horizon, weekly init cycle (Mon/Thu typical).

The dashboard needs only daily global-mean GSAT, not gridded fields. We
sample on a coarse global grid, area-weight by cos(lat), and reduce to
percentile summaries (5/25/50/75/95 + mean) per day. The output CSV is
~5 KB so it commits cleanly with the other dashboard data.

Output: ``data/era5_forecast_ec46_<init-date>.csv`` with columns
  date, t2m_mean, t2m_p5, t2m_p25, t2m_p50, t2m_p75, t2m_p95,
  init_date, n_members.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DASHBOARD_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = DASHBOARD_ROOT / "data"
API = "https://seasonal-api.open-meteo.com/v1/seasonal"
N_MEMBERS = 50
FORECAST_DAYS = 46
MIN_TIMESTEPS_PER_DAY = 4  # require all 4 6-hourly steps; protects
                            # against the diurnal-coverage artifact when
                            # Open-Meteo's last day has only the 00:00 UTC
                            # timestep populated.
LAT_STEP_DEFAULT = 10.0
LON_STEP_DEFAULT = 15.0
REQUEST_TIMEOUT = 30
RETRY_MAX = 2
SLEEP_BETWEEN = 0.05


def _grid(lat_step: float, lon_step: float) -> list[tuple[float, float]]:
    """Cell-centred global grid covering the full sphere."""
    lats = np.arange(-90 + lat_step / 2, 90, lat_step)
    lons = np.arange(-180 + lon_step / 2, 180, lon_step)
    return [(float(la), float(lo)) for la in lats for lo in lons]


def _fetch_point(lat: float, lon: float) -> dict | None:
    params = (
        f"latitude={lat:.3f}&longitude={lon:.3f}"
        f"&models=ecmwf_ec46"
        f"&hourly=temperature_2m"
        f"&forecast_days={FORECAST_DAYS}"
        f"&ensemble_members={N_MEMBERS}"
    )
    url = f"{API}?{params}"
    last_err = None
    for attempt in range(RETRY_MAX + 1):
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, json.JSONDecodeError,
                TimeoutError) as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    logger.warning("EC46 fetch (%.1f, %.1f) failed after %d retries: %s",
                   lat, lon, RETRY_MAX + 1, last_err)
    return None


def _point_daily_per_member(payload: dict) -> pd.DataFrame:
    """Aggregate 6-hourly t2m → daily mean per member (long DataFrame).
    Days with fewer than MIN_TIMESTEPS_PER_DAY valid timesteps are
    dropped — this handles the trailing-day diurnal-coverage artifact."""
    h = payload["hourly"]
    t = pd.to_datetime(h["time"])
    member_keys = sorted(k for k in h if k.startswith("temperature_2m_member"))
    rows = []
    for mk in member_keys:
        m_id = mk.replace("temperature_2m_", "")
        ser = pd.Series(h[mk], index=t).dropna()
        if ser.empty:
            continue
        grouped = ser.groupby(ser.index.normalize())
        counts = grouped.size()
        daily = grouped.mean()
        keep = counts[counts >= MIN_TIMESTEPS_PER_DAY].index
        daily = daily.loc[keep]
        for d, v in daily.items():
            rows.append({"date": d, "member": m_id, "t2m_C": float(v)})
    return pd.DataFrame(rows)


def _global_mean(per_pt: pd.DataFrame) -> pd.DataFrame:
    """Reduce (date, member, lat, lon, t2m_C) → (date, member, t2m_C),
    cos(lat) area-weighting."""
    per_pt = per_pt.copy()
    per_pt["w"] = np.cos(np.deg2rad(per_pt["lat"]))

    def _wmean(g):
        return float(np.average(g["t2m_C"], weights=g["w"]))

    return per_pt.groupby(["date", "member"]).apply(
        lambda g: pd.Series({"t2m_C": _wmean(g)}),
        include_groups=False,
    ).reset_index()


def _percentile_summary(member_long: pd.DataFrame, init_date: pd.Timestamp,
                        n_members: int) -> pd.DataFrame:
    g = member_long.groupby("date")["t2m_C"]
    out = pd.DataFrame({
        "t2m_mean": g.mean(),
        "t2m_p5": g.quantile(0.05),
        "t2m_p25": g.quantile(0.25),
        "t2m_p50": g.quantile(0.50),
        "t2m_p75": g.quantile(0.75),
        "t2m_p95": g.quantile(0.95),
    }).reset_index()
    out["init_date"] = init_date.strftime("%Y-%m-%d")
    out["n_members"] = n_members
    return out


def fetch_ec46(lat_step: float = LAT_STEP_DEFAULT,
               lon_step: float = LON_STEP_DEFAULT) -> pd.DataFrame:
    """Pull the latest EC46 init from Open-Meteo, reduce to per-day
    percentile summary. Returns the summary DataFrame; empty if fetch
    fails badly."""
    grid_pts = _grid(lat_step, lon_step)
    logger.info("EC46: fetching %d gridpoints (%.1f°×%.1f°) from Open-Meteo",
                len(grid_pts), lat_step, lon_step)

    parts: list[pd.DataFrame] = []
    failed = 0
    t0 = time.time()
    for i, (lat, lon) in enumerate(grid_pts):
        payload = _fetch_point(lat, lon)
        if payload is None:
            failed += 1
            continue
        df = _point_daily_per_member(payload)
        if df.empty:
            failed += 1
            continue
        df["lat"] = lat
        df["lon"] = lon
        parts.append(df)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  EC46 [%d/%d] ok (%.0fs, %d fails)",
                        i + 1, len(grid_pts), elapsed, failed)
        time.sleep(SLEEP_BETWEEN)

    elapsed = time.time() - t0
    logger.info("EC46: fetch done — %d/%d ok (%.0fs)",
                len(grid_pts) - failed, len(grid_pts), elapsed)

    if not parts:
        logger.error("EC46: all gridpoints failed")
        return pd.DataFrame()

    per_pt = pd.concat(parts, ignore_index=True)
    member_long = _global_mean(per_pt)

    n_members = member_long["member"].nunique()
    init_date = member_long["date"].min()
    summary = _percentile_summary(member_long, init_date, n_members)

    logger.info("EC46: %d forecast days × %d members; init %s",
                len(summary), n_members, init_date.strftime("%Y-%m-%d"))
    return summary


def save_ec46(force: bool = False) -> pd.DataFrame:
    """Fetch latest EC46 forecast and save to
    ``data/era5_forecast_ec46_<init-date>.csv``.

    Skips the fetch if today's local date hasn't crossed the init date of
    the most recent on-disk file (cheap idempotency for daily cron). On
    fetch failure, the existing file is preserved (no regression).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DATA_DIR.glob("era5_forecast_ec46_*.csv"))
    today = pd.Timestamp(date.today())

    if not force and existing:
        latest = existing[-1]
        try:
            init_str = latest.stem.split("era5_forecast_ec46_")[-1]
            latest_init = pd.Timestamp(init_str)
        except Exception:
            latest_init = None
        if latest_init is not None and (today - latest_init).days < 3:
            # EC46 publishes Mon/Thu; if our latest init is within 3 days
            # we're already current.
            logger.info("EC46: %s is fresh (%s), skipping fetch",
                        latest.name, latest_init.strftime("%Y-%m-%d"))
            return pd.read_csv(latest)

    df = fetch_ec46()
    if df.empty:
        if existing:
            logger.warning("EC46: fetch failed, keeping previous file %s",
                           existing[-1].name)
            return pd.read_csv(existing[-1])
        return df

    init_date = df["init_date"].iloc[0]
    out_path = DATA_DIR / f"era5_forecast_ec46_{init_date}.csv"
    df.to_csv(out_path, index=False)
    logger.info("EC46: saved %s", out_path)

    # Cleanup older files — keep only the most recent.
    for old in existing:
        if old != out_path:
            try:
                old.unlink()
                logger.info("EC46: removed older %s", old.name)
            except OSError:
                pass

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    save_ec46(force=True)
