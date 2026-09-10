"""Monthly ENSO coupling observations: SOI, 850 hPa trade winds, warm water volume.

Feeds the ENSO tab's "Is it coupled?" section and the Atmosphere KPI card.
All three series are MONTHLY products published a few weeks after month end,
so they are deliberately labelled as such on the dashboard (unlike the daily
OISST Niño 3.4 index they sit beside).

Sources (NOAA CPC fixed-width index files; NOAA/PMEL WWV):
  - SOI (standardized Tahiti − Darwin SLP), 1951–:  cpc.ncep.noaa.gov/data/indices/soi
  - 850 hPa zonal wind anomaly indices, 1979–:       .../wpac850, cpac850, epac850
      West 135°E–180°, Central 175°W–140°W, East 135°W–120°W; 5°N–5°S;
      CDAS/NCEP–NCAR reanalysis, 1981–2010 base. The index is the EASTERLY
      trade speed, so a positive anomaly = stronger trades, negative =
      weakened trades (El Niño-like). Verified on the data: 1997/2015/2026
      strongly negative, 2010/2020 positive, ORIGINAL block ≈ +5 to +11 m/s.
  - Warm Water Volume anomaly (T > 20 °C, 5°N–5°S, 120°E–80°W), 1980–:
      pmel.noaa.gov/tao/wwv/data/wwv.dat  (derived from ocean analyses, in m³)

Derived files (small, committed by the daily cron):
  data/enso_obs/soi_trade_winds.csv   date, soi, wind_wpac, wind_cpac, wind_epac
  data/enso_obs/wwv.csv               date, wwv_anom_1e14_m3

Parsing follows the fixed-width approach from the proposal in PR #2 (Al
Khourdajie): CPC's current-year rows carry -999.9 sentinels that collide with
neighbouring values under whitespace splitting ("-6.1-999.9").
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OBS_DIR = DATA_DIR / "enso_obs"
SOI_WINDS_FILE = OBS_DIR / "soi_trade_winds.csv"
WWV_FILE = OBS_DIR / "wwv.csv"

CPC_SOURCES = {
    # column: (url, marker that must appear in the payload, block, |max| sanity)
    "soi": ("https://www.cpc.ncep.noaa.gov/data/indices/soi",
            "(STAND TAHITI - STAND DARWIN)", "STANDARDIZED DATA", 6.0),
    "wind_wpac": ("https://www.cpc.ncep.noaa.gov/data/indices/wpac850",
                  "(135E-180W)", "ANOMALY", 15.0),
    "wind_cpac": ("https://www.cpc.ncep.noaa.gov/data/indices/cpac850",
                  "(175W-140W)", "ANOMALY", 15.0),
    "wind_epac": ("https://www.cpc.ncep.noaa.gov/data/indices/epac850",
                  "(135W-120W)", "ANOMALY", 15.0),
}
WWV_URL = "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat"
WWV_MARKER = "Warm Water Volume"
SENTINEL = -999.0
UA = {"User-Agent": "climate-dashboard (dashboard.theclimatebrink.com)"}
REFRESH_DAYS = 3   # monthly products: re-pull every few days is plenty

WIND_LABELS = {
    "wind_wpac": "West Pacific (135°E–180°)",
    "wind_cpac": "Central Pacific (175°W–140°W)",
    "wind_epac": "East Pacific (135°W–120°W)",
}


# ---------------------------------------------------------------------------
# CPC fixed-width index files
# ---------------------------------------------------------------------------
def parse_cpc_blocks(text: str, name: str) -> dict[str, pd.DataFrame]:
    """Split a CPC index file into wide (year × month) blocks keyed by the
    block subtitle ('ANOMALY', 'STANDARDIZED DATA', 'ORIGINAL DATA').

    Layout per block: title line, subtitle line, blank, 'YEAR JAN … DEC'
    header, then fixed-width rows (YEAR in cols 0–3, twelve 6-char fields).
    """
    blocks: dict[str, pd.DataFrame] = {}
    subtitle: str | None = None
    rows: list[tuple[int, list[float]]] = []

    def flush() -> None:
        nonlocal rows
        if subtitle is None or not rows:
            rows = []
            return
        years = [y for y, _ in rows]
        if years != sorted(set(years)):
            raise ValueError(f"{name}: block {subtitle!r} years not increasing")
        blocks[subtitle] = pd.DataFrame([v for _, v in rows], index=years,
                                        columns=range(1, 13))
        rows = []

    pending_title = False
    for raw in text.split("\n"):
        line = raw.rstrip("\r\n")
        stripped = " ".join(line.split())
        if not stripped:
            continue
        if stripped.startswith("YEAR "):
            pending_title = False
            continue
        if len(line) >= 10 and line[:4].isdigit():
            year = int(line[:4])
            if not 1900 <= year <= 2100:
                raise ValueError(f"{name}: implausible year {year}")
            padded = line.ljust(4 + 6 * 12)
            vals = []
            for i in range(12):
                field = padded[4 + 6 * i: 10 + 6 * i].strip()
                if not field:
                    raise ValueError(f"{name}: empty field {i + 1}, year {year}")
                v = float(field)
                vals.append(np.nan if v <= SENTINEL else v)
            rows.append((year, vals))
            continue
        # A text line: the first of a pair is the block title, the second the
        # subtitle that names the block.
        if pending_title:
            flush()
            subtitle = stripped
            pending_title = False
        else:
            pending_title = True
    flush()
    return blocks


def block_to_series(block: pd.DataFrame, name: str, max_abs: float) -> pd.Series:
    """Wide block → monthly Series (first-of-month index). Sentinel NaNs may
    only form a contiguous tail (the not-yet-published months)."""
    long = block.stack(future_stack=True)
    idx = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in long.index])
    s = pd.Series(long.values, index=idx, name=name).sort_index()
    valid = s.dropna()
    if valid.empty:
        raise ValueError(f"{name}: no valid values")
    s = s.loc[:valid.index[-1]]
    if s.isna().any():
        gaps = s[s.isna()].index
        raise ValueError(f"{name}: interior gaps at "
                         f"{[str(d.date()) for d in gaps[:5]]}")
    if float(s.abs().max()) > max_abs:
        raise ValueError(f"{name}: |value| {s.abs().max():.1f} exceeds {max_abs}")
    return s


def fetch_cpc_series(name: str) -> pd.Series:
    url, marker, block_name, max_abs = CPC_SOURCES[name]
    resp = requests.get(url, headers=UA, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if marker not in text[:400]:
        raise ValueError(f"{name}: expected marker {marker!r} missing — "
                         f"layout changed? ({url})")
    blocks = parse_cpc_blocks(text, name)
    if block_name not in blocks:
        raise ValueError(f"{name}: block {block_name!r} not found; have "
                         f"{list(blocks)}")
    return block_to_series(blocks[block_name], name, max_abs)


# ---------------------------------------------------------------------------
# PMEL warm water volume
# ---------------------------------------------------------------------------
def parse_wwv(text: str) -> pd.Series:
    lines = text.split("\n")
    if WWV_MARKER not in lines[0]:
        raise ValueError("wwv.dat: header marker missing — layout changed?")
    try:
        head = next(i for i, l in enumerate(lines)
                    if l.split()[:3] == ["date", "Volume", "Anomaly"])
    except StopIteration:
        raise ValueError("wwv.dat: 'date Volume Anomaly' header not found")
    dates, anoms = [], []
    for ln in lines[head + 1:]:
        parts = ln.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"wwv.dat: expected 3 columns, got {ln!r}")
        ym, vol_s, anom_s = parts
        vol, anom = float(vol_s), float(anom_s)
        if not 1.5e15 < vol < 4.0e15:
            raise ValueError(f"wwv.dat: volume {vol:.3e} outside physical range at {ym}")
        if abs(anom) > 1.5e15:
            raise ValueError(f"wwv.dat: anomaly {anom:.3e} implausible at {ym}")
        dates.append(pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"))
        anoms.append(anom / 1e14)
    s = pd.Series(anoms, index=pd.DatetimeIndex(dates), name="wwv_anom_1e14_m3").sort_index()
    if s.index.has_duplicates:
        raise ValueError("wwv.dat: duplicate months")
    return s


def fetch_wwv() -> pd.Series:
    resp = requests.get(WWV_URL, headers=UA, timeout=60)
    resp.raise_for_status()
    if len(resp.text) < 5000:
        raise ValueError(f"wwv.dat: implausibly short payload ({len(resp.text)} bytes)")
    return parse_wwv(resp.text)


# ---------------------------------------------------------------------------
# Update / load
# ---------------------------------------------------------------------------
def _fresh(path: Path) -> bool:
    """True when the file exists, was written within REFRESH_DAYS, and already
    carries last month's value (so a late-posting month still triggers a pull)."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age > timedelta(days=REFRESH_DAYS):
        return False
    try:
        last = pd.to_datetime(pd.read_csv(path)["date"]).max()
    except Exception:
        return False
    prev_month = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
    return last >= pd.Timestamp(prev_month)


def update_enso_observations(force: bool = False) -> None:
    """Refresh both derived CSVs (each independently; a failure in one source
    leaves the other's file intact)."""
    OBS_DIR.mkdir(parents=True, exist_ok=True)

    if force or not _fresh(SOI_WINDS_FILE):
        try:
            series = [fetch_cpc_series(n) for n in CPC_SOURCES]
            df = pd.concat(series, axis=1).sort_index()
            df.index.name = "date"
            df.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")) \
              .to_csv(SOI_WINDS_FILE, index=False, float_format="%.2f")
            logger.info("SOI/trade winds updated through %s",
                        df.dropna(how="all").index.max().strftime("%b %Y"))
        except Exception as e:
            logger.error("SOI/trade-wind update failed: %s", e)
    else:
        logger.info("SOI/trade winds fresh; skipping fetch")

    if force or not _fresh(WWV_FILE):
        try:
            s = fetch_wwv()
            out = s.reset_index().rename(columns={"index": "date"})
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
            out.to_csv(WWV_FILE, index=False, float_format="%.3f")
            logger.info("WWV updated through %s", s.index.max().strftime("%b %Y"))
        except Exception as e:
            logger.error("WWV update failed: %s", e)
    else:
        logger.info("WWV fresh; skipping fetch")


def load_enso_observations() -> dict:
    """{'soi_winds': DataFrame(date index), 'wwv': Series} — empty objects
    when files are absent so callers can degrade gracefully."""
    out = {"soi_winds": pd.DataFrame(), "wwv": pd.Series(dtype=float)}
    if SOI_WINDS_FILE.exists():
        df = pd.read_csv(SOI_WINDS_FILE, parse_dates=["date"]).set_index("date")
        out["soi_winds"] = df
    if WWV_FILE.exists():
        w = pd.read_csv(WWV_FILE, parse_dates=["date"]).set_index("date")
        out["wwv"] = w["wwv_anom_1e14_m3"]
    return out


def latest_coupling_status(obs: dict | None = None) -> dict | None:
    """Latest common month of SOI + central-Pacific trades for the KPI card.
    Returns None if nothing is loaded."""
    obs = obs or load_enso_observations()
    df = obs["soi_winds"]
    if df.empty or "soi" not in df or "wind_cpac" not in df:
        return None
    both = df[["soi", "wind_cpac"]].dropna()
    if both.empty:
        return None
    last = both.index.max()
    row = both.loc[last]
    soi, cpac = float(row["soi"]), float(row["wind_cpac"])
    # Descriptive reading, not a classification: both signs pointing the same
    # way is what "coupled" means for the Bjerknes feedback.
    if soi <= -0.5 and cpac < 0:
        reading = "Atmosphere coupled to El Niño"
    elif soi >= 0.5 and cpac > 0:
        reading = "Atmosphere coupled to La Niña"
    else:
        reading = "Weak or mixed atmospheric response"
    wwv = obs["wwv"]
    wwv_last = (float(wwv.iloc[-1]), wwv.index[-1]) if not wwv.empty else None
    return {"month": last, "month_label": last.strftime("%b %Y"), "soi": soi,
            "wind_cpac": cpac, "reading": reading, "wwv": wwv_last}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    update_enso_observations(force="--force" in sys.argv)
    st = latest_coupling_status()
    print(st)
