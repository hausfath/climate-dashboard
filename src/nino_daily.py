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
import tempfile
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
HISTORY_FILE = DATA_DIR / 'nino34_daily_history.csv'

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
DS_FINAL = "ncdcOisst21Agg_LonPM180"      # lags ~2 weeks
DS_NRT = "ncdcOisst21NrtAgg_LonPM180"     # near-real-time

# NCEI hosts the per-day source files the ERDDAP aggregates; used as a
# fallback since 2026-07-14, when CoastWatch started 403ing GitHub runners.
NCEI_BASE = ("https://www.ncei.noaa.gov/data/"
             "sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr")

# CoastWatch blocks anonymous default UAs; identify ourselves politely.
HEADERS = {"User-Agent": ("climate-dashboard/1.0 "
                          "(github.com/hausfath/climate-dashboard; "
                          "hausfath@gmail.com)")}

# Box definitions: (lat0, lat1, lat_stride, lon0, lon1, lon_stride).
# Grid cells sit at *.875/*.625/… (0.25° centered); strides subsample to
# ~1° (Niño boxes) and ~2°×4° (tropics) — identical for clim and daily.
NINO34_BOX = (-4.875, 4.875, 4, -169.875, -120.125, 4)
TROPICS_BOX = (-19.875, 19.875, 8, -179.875, 179.875, 16)
NINO12_BOX = (-9.875, -0.125, 4, -89.875, -80.125, 4)
NINO3_BOX = (-4.875, 4.875, 4, -149.875, -90.125, 4)
# Niño 4 (160°E–150°W) crosses the dateline: two PM180 sub-boxes for the
# ERDDAP path, combined with longitude-count weights (20:30 grid points).
NINO4_BOX_E = (-4.875, 4.875, 4, 160.125, 179.875, 4)
NINO4_BOX_W = (-4.875, 4.875, 4, -179.875, -150.125, 4)

# All history columns beyond nino34/tropics (backfilled 2026-08 from PSL
# NCSS on the identical strided grid; see temp_files/fetch_nino_regions_ncss.py)
EXTRA_REGIONS = ('nino12', 'nino3', 'nino4')

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
    resp = requests.get(url, headers=HEADERS, timeout=300)
    resp.raise_for_status()
    with xr.open_dataset(io.BytesIO(resp.content)) as ds:
        sst = ds['sst'].squeeze('zlev', drop=True)
        w = np.cos(np.deg2rad(sst.latitude))
        mean = sst.weighted(w).mean(dim=('latitude', 'longitude'))
        s = mean.to_pandas()
    s.index = pd.to_datetime(s.index).normalize()
    return s.dropna()


def _box_coords(box: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Exact strided grid centers ERDDAP would return for a box, with
    longitudes mapped to NCEI's 0–360 convention."""
    lat0, lat1, lat_s, lon0, lon1, lon_s = box
    lats = np.arange(lat0, lat1 + 1e-6, 0.25 * lat_s)
    lons = np.arange(lon0, lon1 + 1e-6, 0.25 * lon_s) % 360
    return lats, lons


def _all_region_coords() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{column: (lats, lons in 0-360)} for every history column. Niño 4 is
    contiguous in 0-360 (160.125..209.875), so its two PM180 sub-boxes just
    concatenate."""
    coords = {name: _box_coords(box) for name, box in
              [('nino34', NINO34_BOX), ('tropics', TROPICS_BOX),
               ('nino12', NINO12_BOX), ('nino3', NINO3_BOX)]}
    lats_e, lons_e = _box_coords(NINO4_BOX_E)
    _, lons_w = _box_coords(NINO4_BOX_W)
    coords['nino4'] = (lats_e, np.concatenate([lons_e, lons_w]))
    return coords


def _fetch_region_erddap(dataset: str, start: str, end: str,
                         name: str) -> pd.Series:
    """One region's box-mean series from ERDDAP (two-part fetch for Niño 4)."""
    boxes = {'nino34': NINO34_BOX, 'tropics': TROPICS_BOX,
             'nino12': NINO12_BOX, 'nino3': NINO3_BOX}
    if name == 'nino4':
        e = _fetch_box_means(dataset, start, end, NINO4_BOX_E)
        w = _fetch_box_means(dataset, start, end, NINO4_BOX_W)
        return (e * 20 + w * 30) / 50   # longitude-count weights
    return _fetch_box_means(dataset, start, end, boxes[name])


def _fetch_daily_means_ncei(start: date, end: date) -> pd.DataFrame:
    """ERDDAP-free fallback: NCEI's per-day OISST v2.1 files (the same data
    the ERDDAP aggregates), sampled on the identical strided grid so
    anomalies stay consistent with the committed climatology. Returns a
    date-indexed frame with a column per region box plus 'tropics'."""
    boxes = _all_region_coords()
    rows = {}
    misses = 0
    d = start
    while d <= end and misses < 3:
        content = None
        for suffix in ('', '_preliminary'):
            url = (f"{NCEI_BASE}/{d:%Y%m}/"
                   f"oisst-avhrr-v02r01.{d:%Y%m%d}{suffix}.nc")
            resp = requests.get(url, headers=HEADERS, timeout=120)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            content = resp.content
            break
        if content is None:
            misses += 1
            d += timedelta(days=1)
            continue
        misses = 0
        # NetCDF4/HDF5 can't be opened from memory without h5netcdf;
        # round-trip through a temp file for the netCDF4 engine.
        with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
            tmp.write(content)
            tmp.flush()
            with xr.open_dataset(tmp.name, engine='netcdf4') as ds:
                sst = ds['sst'].squeeze(('time', 'zlev'), drop=True)
                row = {}
                for name, (lats, lons) in boxes.items():
                    sub = sst.sel(lat=lats, lon=lons)
                    w = np.cos(np.deg2rad(sub.lat))
                    row[name] = float(
                        sub.weighted(w).mean(dim=('lat', 'lon')))
                rows[pd.Timestamp(d)] = row
        d += timedelta(days=1)
    if not rows:
        raise RuntimeError(f"NCEI fallback: no OISST files found "
                           f"for {start}..{end}")
    df = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    logger.info(f"NCEI fallback: {len(df)} days through {df.index.max().date()}")
    return df


def _dataset_end(dataset: str) -> date:
    url = f"{ERDDAP}/{dataset}.json?time%5Blast%5D"
    resp = requests.get(url, headers=HEADERS, timeout=60)
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

    try:
        final_end = _dataset_end(DS_FINAL)
        nrt_end = _dataset_end(DS_NRT)

        series = {}
        for name in ('nino34', 'tropics') + EXTRA_REGIONS:
            parts = []
            if start <= final_end:
                parts.append(_fetch_region_erddap(
                    DS_FINAL, str(start), str(final_end), name))
            nrt_start = max(start, final_end + timedelta(days=1))
            if nrt_start <= nrt_end:
                parts.append(_fetch_region_erddap(
                    DS_NRT, str(nrt_start), str(nrt_end), name))
            series[name] = pd.concat(parts).sort_index()
        df = pd.DataFrame(series).dropna(subset=['nino34', 'tropics'])
    except Exception as e:
        logger.warning(f"ERDDAP fetch failed ({e}); "
                       f"falling back to NCEI daily files")
        df = _fetch_daily_means_ncei(start, today).dropna(
            subset=['nino34', 'tropics'])
    df.index.name = 'date'
    # float32 from the NetCDF would defeat .round(4) in the CSV output
    df = df.astype('float64').reset_index()

    doy = _doy_key(pd.DatetimeIndex(df['date']))
    df['nino34_anom'] = df['nino34'].values - clim.loc[doy, 'nino34_clim'].values
    trop_anom = df['tropics'].values - clim.loc[doy, 'tropics_clim'].values
    months = pd.DatetimeIndex(df['date']).month
    scale = np.array([RONI_SCALING_MONTHLY[m] for m in months])
    df['roni_anom'] = (df['nino34_anom'].values - trop_anom) * scale
    hist_cols = ['date', 'nino34', 'tropics'] + [
        r for r in EXTRA_REGIONS if r in df.columns]
    _merge_into_history(df[hist_cols])
    df = df[['date', 'nino34', 'nino34_anom', 'roni_anom']].round(4)

    if existing is not None and not existing.empty:
        keep = existing[existing['date'] < pd.Timestamp(start)]
        df = pd.concat([keep, df], ignore_index=True)
    df = df.drop_duplicates(subset='date', keep='last').sort_values('date')

    df.to_csv(DAILY_FILE, index=False)
    logger.info(f"Wrote {DAILY_FILE}: {len(df)} rows through "
                f"{df['date'].max().date()}")
    return df


def _merge_into_history(new_rows: pd.DataFrame) -> None:
    """Fold freshly fetched absolute box means into the full-record history
    file (date, nino34, tropics) that feeds the daily year-lines figure.
    New rows win on overlap, so NRT days get replaced once final data land.
    No-op if the history file hasn't been built yet (one-time local build;
    see temp_files/build_tropics_history.py)."""
    if not HISTORY_FILE.exists():
        logger.warning("%s missing — skipping history top-up", HISTORY_FILE)
        return
    hist = pd.read_csv(HISTORY_FILE, parse_dates=['date'])
    merged = (pd.concat([hist, new_rows.round(4)], ignore_index=True)
              .drop_duplicates(subset='date', keep='last')
              .sort_values('date'))
    merged.to_csv(HISTORY_FILE, index=False)
    logger.info("Topped up %s: %d rows through %s", HISTORY_FILE.name,
                len(merged), merged['date'].max().date())


def load_daily_status() -> dict | None:
    """Latest daily reading for UI cards: values, date, and 30-day means."""
    try:
        df = pd.read_csv(DAILY_FILE, parse_dates=['date'])
        if df.empty:
            return None
        last = df.iloc[-1]
        win = df[df['date'] > last['date'] - pd.Timedelta(days=30)]
        return {
            'date': last['date'].strftime('%b %-d'),
            'nino34_anom': float(last['nino34_anom']),
            'roni_anom': float(last['roni_anom']),
            'nino34_30d': float(win['nino34_anom'].mean()),
            'roni_30d': float(win['roni_anom'].mean()),
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
