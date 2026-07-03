"""Fetch SINTEX-F ENSO forecasts from JAMSTEC's APL VirtualEarth service.

Two endpoints:

* **Indices CSV** — monthly Niño 3.4 SSTA per ensemble member:
    https://www.jamstec.go.jp/virtualearth/data/SINTEX/SINTEX_Nino34.csv
  Columns: time, Obs, Mean, <system means>, then 24 individual members
  (N1K1 … N3K2hiV: initialization × physics variants of SINTEX-F2).
  Observed months carry the same value in every member column (the
  "bridge"); forecast months diverge.

* **Gridded seasonal SSTA JSON** — per member, global 1.125° grid:
    https://www.jamstec.go.jp/virtualearth/data/SINTEX/SINTEX_sst_{member}_all_{period}.json
  Periods (e.g. JJA2026, SON2026, DJF2027) come from SINTEX_catalog.json.
  Used to compute the 20S–20N cos-weighted tropical-mean SSTA per member,
  which is assigned to each month of the season → RONI = Niño3.4 − tropical.
  Months beyond the gridded seasons get NaN RONI (ONI still available).

Anomaly baseline is 1983–2015 (per JAMSTEC's data page); the shift to
1991–2020 happens downstream in ``normalize.adjust_sintexf_baseline`` so raw
files preserve the source convention, mirroring how C3S is handled.

Update cadence: monthly; the new run (initialized early in month M, observed
bridge through M−1) has been observed to appear within the first days of M.
"""

import io
import logging
import re
from datetime import date

import numpy as np
import pandas as pd
import requests

from enso_forecast.config import FORECASTS_DIR, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# JAMSTEC 302-redirects non-browser user agents to an error page, so this
# fetcher needs a browser-like UA (config's tool UA breaks it).
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; climate-dashboard)"}

BASE = "https://www.jamstec.go.jp/virtualearth/data/SINTEX"
INDICES_URL = f"{BASE}/SINTEX_Nino34.csv"
CATALOG_URL = f"{BASE}/SINTEX_catalog.json"
GRID_URL = f"{BASE}/SINTEX_sst_{{member}}_all_{{period}}.json"

# Season label (3-letter month initials) → center calendar month
SEASON_CENTER = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def _member_columns(df: pd.DataFrame) -> list[str]:
    """Individual ensemble members: N{i}K{j} plus hi/V variants."""
    return [c for c in df.columns if re.fullmatch(r"N\d+K\d+(hi)?(V)?", c)]


def _season_months(period: str) -> list[str]:
    """'DJF2027' → ['2026-12', '2027-01', '2027-02'] (year = center month's)."""
    season, year = period[:3], int(period[3:])
    center = SEASON_CENTER[season]
    months = []
    for off in (-1, 0, 1):
        m = center + off
        y = year
        if m < 1:
            m, y = m + 12, y - 1
        elif m > 12:
            m, y = m - 12, y + 1
        months.append(f"{y}-{m:02d}")
    return months


def _tropical_mean_from_grid(payload) -> float:
    """20S–20N cos-weighted mean SSTA from a VirtualEarth grid JSON."""
    item = payload[0] if isinstance(payload, list) else payload
    h = item["header"]
    vals = np.array(item["data"], dtype=float)
    grid = vals.reshape(h["ny"], h["nx"])
    lats = np.linspace(h["la1"], h["la2"], h["ny"])
    band = (lats >= -20) & (lats <= 20)
    sub = grid[band, :]
    w = np.cos(np.deg2rad(lats[band]))[:, None] * np.ones(h["nx"])
    ok = ~np.isnan(sub)
    if ok.sum() == 0:
        return float("nan")
    return float(np.nansum(sub * w) / np.sum(np.where(ok, w, 0.0)))


def _fetch_tropical_means(members: list[str]) -> dict[tuple[str, str], float]:
    """{(member, 'YYYY-MM'): tropical_mean} for every catalog period."""
    try:
        cat = requests.get(CATALOG_URL, headers=REQUEST_HEADERS,
                           timeout=REQUEST_TIMEOUT).json()
        periods = list(cat.get("period", {}).keys())
    except Exception as e:
        logger.warning("SINTEX-F: catalog fetch failed (%s); RONI will be NaN", e)
        return {}

    out: dict[tuple[str, str], float] = {}
    for period in periods:
        months = _season_months(period)
        for member in members:
            url = GRID_URL.format(member=member.lower(), period=period)
            try:
                resp = requests.get(url, headers=REQUEST_HEADERS,
                                    timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                trop = _tropical_mean_from_grid(resp.json())
            except Exception as e:
                logger.warning("SINTEX-F: grid fetch failed for %s %s (%s)",
                               member, period, e)
                continue
            for ym in months:
                out[(member, ym)] = trop
        logger.info("SINTEX-F: tropical means for %s (%d members)",
                    period, len(members))
    return out


def fetch_sintexf() -> pd.DataFrame:
    """Fetch the current SINTEX-F run as a standard forecast DataFrame."""
    resp = requests.get(INDICES_URL, headers=REQUEST_HEADERS,
                        timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    raw["time"] = pd.to_datetime(raw["time"])

    members = _member_columns(raw)
    if not members:
        logger.warning("SINTEX-F: no member columns found")
        return pd.DataFrame()

    # Forecast months: member values present and diverging from each other
    # (observed bridge months have identical values across all members).
    mem_vals = raw[members]
    has_data = mem_vals.notna().any(axis=1)
    diverged = mem_vals.std(axis=1) > 1e-6
    fc = raw[has_data & diverged].copy()
    if fc.empty:
        logger.warning("SINTEX-F: no forecast months found")
        return pd.DataFrame()

    # Init month: the month before the first forecast month (CFS convention).
    first_fc = fc["time"].min()
    init = (first_fc - pd.DateOffset(months=1)).replace(day=1)
    init_date = init.strftime("%Y-%m-%d")

    trop = _fetch_tropical_means(members)

    rows = []
    for _, r in fc.iterrows():
        tm = r["time"].strftime("%Y-%m")
        lead = ((r["time"].year - init.year) * 12
                + (r["time"].month - init.month))
        for member in members:
            val = r[member]
            if pd.isna(val):
                continue
            t = trop.get((member, tm), float("nan"))
            rows.append({
                "source": "SINTEX-F",
                "model": "SINTEX-F",
                "model_type": "dynamical",
                "init_date": init_date,
                "target_month": tm,
                "lead_months": lead,
                "nino34_anom": round(float(val), 4),
                "tropical_mean_anom": round(t, 4) if not np.isnan(t) else np.nan,
                "roni_anom": (round(float(val) - t, 4)
                              if not np.isnan(t) else np.nan),
                "member_id": member,
                "temporal_resolution": "monthly",
                "anomaly_base_period": "1983-2015",
            })
    df = pd.DataFrame(rows)

    # Ensemble-mean rows (member_id="mean"): downstream plume code draws the
    # per-model mean line and counts models-per-month from these.
    mean_rows = []
    for tm, grp in df.groupby("target_month"):
        mean_rows.append({
            "source": "SINTEX-F",
            "model": "SINTEX-F",
            "model_type": "dynamical",
            "init_date": init_date,
            "target_month": tm,
            "lead_months": int(grp["lead_months"].iloc[0]),
            "nino34_anom": round(float(grp["nino34_anom"].mean()), 4),
            "tropical_mean_anom": (round(float(grp["tropical_mean_anom"].mean()), 4)
                                   if grp["tropical_mean_anom"].notna().any() else np.nan),
            "roni_anom": (round(float(grp["roni_anom"].mean()), 4)
                          if grp["roni_anom"].notna().any() else np.nan),
            "member_id": "mean",
            "temporal_resolution": "monthly",
            "anomaly_base_period": "1983-2015",
        })
    df = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)

    logger.info("SINTEX-F: %d records, init %s, %d members, targets %s → %s",
                len(df), init_date, len(members),
                df["target_month"].min(), df["target_month"].max())
    return df


def save_sintexf(force: bool = False) -> pd.DataFrame:
    """Fetch and save SINTEX-F data (standard fetcher interface)."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "SINTEX-F"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("SINTEX-F data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_sintexf()
    if df.empty:
        logger.warning("SINTEX-F fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved SINTEX-F data to %s", out_path)
    return df
