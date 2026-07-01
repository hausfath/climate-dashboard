"""Daily Niño 3.4 and RONI from OISSTv2.1.

Fetches area subsets of NOAA OISST v2.1 daily SST from the CoastWatch ERDDAP
(final aggregate for history, NRT aggregate for the most recent ~2 weeks),
computes cos-weighted box means for the Niño 3.4 region and the 20S–20N
tropical belt, and converts them to anomalies against a 1991–2020 day-of-year
climatology built from the SAME spatially-strided sampling — so any
subsampling bias cancels in the anomaly.

Daily RONI follows L'Heureux et al. (2024): the Niño 3.4 anomaly minus the
tropical-mean anomaly, scaled by the monthly variance-restoration factor.

Outputs (committed to git by the daily cron):
  - data/nino34_daily_climatology.csv  (one-time; rebuild with --clim)
  - data/nino34_daily.csv              (refreshed daily)
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / 'data'
CLIM_FILE = DATA_DIR / 'nino34_daily_climatology.csv'
DAILY_FILE = DATA_DIR / 'nino34_daily.csv'

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
DS_FINAL = "ncdcOisst21Agg_LonPM180"      # lags ~2 weeks
DS_NRT = "ncdcOisst21NrtAgg_LonPM180"     # near-real-time

# Box definitions: (lat0, lat1, lat_stride, lon0, lon1, lon_stride).
# Grid cells sit at *.875/*.625/… (0.25° centered); strides subsample to
# ~1° (Niño 3.4) and ~2°×4° (tropics) — identical for clim and daily.
NINO34_BOX = (-4.875, 4.875, 4, -169.875, -120.125, 4)
TROPICS_BOX = (-19.875, 19.875, 8, -179.875, 179.875, 16)

CLIM_START, CLIM_END = 1991, 2020

# L'Heureux et al. (2024) monthly variance-restoration factors, ERSSTv5
# 1950–2020 (see ENSO/enso_forecast/normalize.py and
# temp_files/compute_roni_scaling.py for the derivation).
RONI_SCALING_MONTHLY = {
    1: 1.2371, 2: 1.2773, 3: 1.3364, 4: 1.3894,
    5: 1.3084, 6: 1.2479, 7: 1.1654, 8: 1.1563,
    9: 1.1887, 10: 1.2186, 11: 1.2168, 12: 1.2389,
}


def _griddap_url(dataset: str, start: str, end: str, box: tuple,
                 time_stride: int = 1) -> str:
    lat0, lat1, lat_s, lon0, lon1, lon_s = box
    return (
        f"{ERDDAP}/{dataset}.nc?sst"
        f"%5B({start}T12:00:00Z):{time_stride}:({end}T12:00:00Z)%5D"
        f"%5B(0.0):1:(0.0)%5D"
        f"%5B({lat0}):{lat_s}:({lat1})%5D"
        f"%5B({lon0}):{lon_s}:({lon1})%5D"
    )


def _fetch_box_means(dataset: str, start: str, end: str, box: tuple,
                     time_stride: int = 1) -> pd.Series:
    """Fetch a box subset and return the cos-weighted daily spatial mean."""
    url = _griddap_url(dataset, start, end, box, time_stride)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with xr.open_dataset(io.BytesIO(resp.content)) as ds:
        sst = ds['sst'].squeeze('zlev', drop=True)
        w = np.cos(np.deg2rad(sst.latitude))
        mean = sst.weighted(w).mean(dim=('latitude', 'longitude'))
        s = mean.to_pandas()
    s.index = pd.to_datetime(s.index).normalize()
    return s.dropna()


def _dataset_end(dataset: str) -> date:
    url = f"{ERDDAP}/{dataset}.json?time%5Blast%5D"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.Timestamp(resp.json()['table']['rows'][0][0]).date()


def _doy_key(idx: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year key that maps Feb 29 onto Feb 28 so leap years align."""
    doy = idx.dayofyear.values.astype(int)
    leap = idx.is_leap_year
    after_feb28 = doy > 59
    doy = doy - (leap & after_feb28).astype(int)
    return doy  # 1..365


def build_climatology(force: bool = False) -> pd.DataFrame:
    """One-time 1991–2020 day-of-year climatology for both boxes (time
    stride 2 keeps the download modest; a 15-day circular smooth follows)."""
    if CLIM_FILE.exists() and not force:
        return pd.read_csv(CLIM_FILE)

    frames = {}
    for name, box in [('nino34', NINO34_BOX), ('tropics', TROPICS_BOX)]:
        chunks = []
        for y0 in range(CLIM_START, CLIM_END + 1, 5):
            y1 = min(y0 + 4, CLIM_END)
            logger.info(f"Climatology fetch {name} {y0}–{y1}...")
            chunks.append(_fetch_box_means(
                DS_FINAL, f"{y0}-01-01", f"{y1}-12-31", box, time_stride=2))
        frames[name] = pd.concat(chunks)

    rows = []
    for name, s in frames.items():
        doy = _doy_key(s.index)
        by_doy = pd.Series(s.values, index=doy).groupby(level=0).mean()
        # circular 15-day smooth
        ext = pd.concat([by_doy.iloc[-15:], by_doy, by_doy.iloc[:15]])
        smooth = ext.rolling(15, center=True, min_periods=1).mean()
        smooth = smooth.iloc[15:-15]
        rows.append(smooth.rename(f'{name}_clim'))
    clim = pd.concat(rows, axis=1)
    clim.index.name = 'doy'
    clim = clim.reset_index()

    DATA_DIR.mkdir(exist_ok=True)
    clim.to_csv(CLIM_FILE, index=False)
    logger.info(f"Wrote {CLIM_FILE} ({len(clim)} rows)")
    return clim


def update_nino34_daily(force: bool = False) -> pd.DataFrame:
    """Refresh data/nino34_daily.csv from Jan 1 of last year to present."""
    if not CLIM_FILE.exists():
        raise FileNotFoundError(
            f"{CLIM_FILE} missing — run build_climatology() once first")
    clim = pd.read_csv(CLIM_FILE).set_index('doy')

    today = date.today()
    start = date(today.year - 1, 1, 1)

    existing = None
    if DAILY_FILE.exists() and not force:
        existing = pd.read_csv(DAILY_FILE, parse_dates=['date'])
        if not existing.empty:
            # refetch a small overlap so NRT days get replaced by final data
            start = max(start,
                        (existing['date'].max() - pd.Timedelta(days=21)).date())

    final_end = _dataset_end(DS_FINAL)
    nrt_end = _dataset_end(DS_NRT)

    series = {}
    for name, box in [('nino34', NINO34_BOX), ('tropics', TROPICS_BOX)]:
        parts = []
        if start <= final_end:
            parts.append(_fetch_box_means(
                DS_FINAL, str(start), str(final_end), box))
        nrt_start = max(start, final_end + timedelta(days=1))
        if nrt_start <= nrt_end:
            parts.append(_fetch_box_means(
                DS_NRT, str(nrt_start), str(nrt_end), box))
        series[name] = pd.concat(parts).sort_index()
    df = pd.DataFrame({'nino34': series['nino34'],
                       'tropics': series['tropics']}).dropna()
    df.index.name = 'date'
    df = df.reset_index()

    doy = _doy_key(pd.DatetimeIndex(df['date']))
    df['nino34_anom'] = df['nino34'].values - clim.loc[doy, 'nino34_clim'].values
    trop_anom = df['tropics'].values - clim.loc[doy, 'tropics_clim'].values
    months = pd.DatetimeIndex(df['date']).month
    scale = np.array([RONI_SCALING_MONTHLY[m] for m in months])
    df['roni_anom'] = (df['nino34_anom'].values - trop_anom) * scale
    df = df[['date', 'nino34', 'nino34_anom', 'roni_anom']].round(4)

    if existing is not None and not existing.empty:
        keep = existing[existing['date'] < pd.Timestamp(start)]
        df = pd.concat([keep, df], ignore_index=True)
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date')

    df.to_csv(DAILY_FILE, index=False)
    logger.info(f"Wrote {DAILY_FILE}: {len(df)} rows through "
                f"{df['date'].max().date()}")
    return df


def load_daily_status() -> dict | None:
    """Latest daily reading for UI cards: values plus their date."""
    try:
        df = pd.read_csv(DAILY_FILE, parse_dates=['date'])
        if df.empty:
            return None
        last = df.iloc[-1]
        return {
            'date': last['date'].strftime('%b %-d'),
            'nino34_anom': float(last['nino34_anom']),
            'roni_anom': float(last['roni_anom']),
        }
    except Exception:
        return None


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    if '--clim' in sys.argv:
        build_climatology(force='--force' in sys.argv)
    update_nino34_daily(force='--force' in sys.argv)
    print(load_daily_status())
