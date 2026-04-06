"""
Models vs. Observations module.

Loads CMIP3/5/6 model ensembles and observational temperature records,
computes statistics, and creates Plotly visualizations for the dashboard.
"""

import re
import ssl
import logging
from pathlib import Path
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / 'data' / 'cmip'
CMIP_FILES = {
    'cmip3': DATA_DIR / 'cmip3_global.txt',
    'cmip5': DATA_DIR / 'cmip5_global.txt',
    'cmip6': DATA_DIR / 'cmip6_global.txt',
}
OBS_CACHE = DATA_DIR / 'combined_obs_1981_2010.csv'

# ── Observational data source URLs ────────────────────────────────────────────
GISTEMP_URL = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt'
def _noaa_url():
    return f'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/tavg/1/0/1850-{datetime.today().year}/data.csv'
HADCRUT5_URL = 'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.1.0.0/analysis/diagnostics/HadCRUT.5.1.0.0.analysis.summary_series.global.monthly.csv'
BERKELEY_URL = 'https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global_TAVG_monthly.txt'

# ── Rebaseline offsets: shift from 1981-2010 baseline to 1850-1900 ────────────
# Computed from existing combined_obs_1981_2010.csv (mean of 1850-1900 period)
PREINDUSTRIAL_OFFSETS = {
    'hadcrut5': -0.7021,
    'gistemp':  -0.6398,
    'noaa':     -0.5991,
    'berkeley': -0.6903,
    'copernicus': -0.6611,  # Average of HadCRUT5/Berkeley/NOAA (records extending to 1850)
}

# ── Theme colors for this tab ─────────────────────────────────────────────────
OBS_COLORS = {
    'light': {
        'hadcrut5':   '#d62728',   # red
        'gistemp':    '#ff7f0e',   # orange
        'noaa':       '#2ca02c',   # green
        'berkeley':   '#9467bd',   # purple
        'copernicus': '#8c564b',   # brown
        'era5_daily': '#1f77b4',   # blue (daily ERA5 extension)
    },
    'dark': {
        'hadcrut5':   '#ff6b6b',   # coral
        'gistemp':    '#ff9f43',   # amber
        'noaa':       '#2ed573',   # green
        'berkeley':   '#a29bfe',   # lavender
        'copernicus': '#fd79a8',   # pink
        'era5_daily': '#74b9ff',   # sky blue
    }
}

MODEL_COLORS = {
    'light': {
        'mean': '#1f77b4',
        'band': 'rgba(31, 119, 180, 0.15)',
        'band_outline': 'rgba(31, 119, 180, 0.0)',
    },
    'dark': {
        'mean': '#4ecdc4',
        'band': 'rgba(78, 205, 196, 0.15)',
        'band_outline': 'rgba(78, 205, 196, 0.0)',
    }
}

HIST_COLORS = {
    'light': 'rgba(214, 39, 40, 0.65)',
    'dark':  'rgba(255, 107, 107, 0.65)',
}


# ── CMIP data loading ─────────────────────────────────────────────────────────

def _decimal_year_to_date(val: str) -> pd.Timestamp:
    """Convert decimal-year string (e.g. '1850.0417') to pd.Timestamp."""
    year = int(val.split('.')[0])
    frac = float('0.' + val.split('.')[1])
    month = min(12, max(1, round(frac * 12) + 1))
    return pd.Timestamp(year, month, 1)


def load_cmip_ensemble(gen: str) -> pd.DataFrame:
    """
    Parse a CMIP text file into a wide DataFrame.

    Returns DataFrame with columns ['date', 'ens_0', 'ens_1', ...].
    The anomalies are in their original baseline (approximately 1961-1990 for
    CMIP; the exact baseline varies by model but is consistent within a file).
    """
    path = CMIP_FILES.get(gen)
    if path is None or not path.exists():
        logger.warning(f"CMIP file not found: {path}")
        return pd.DataFrame()

    members = []
    current = []
    with open(path, 'r') as fh:
        for line in fh:
            if re.match(r"# ensemble member", line):
                if current:
                    members.append(current)
                    current = []
            elif re.match(r"^\s*\d{4}\.\d+\s+-?\d+\.\d+", line):
                parts = line.strip().split()
                current.append(parts)
    if current:
        members.append(current)

    if not members:
        return pd.DataFrame()

    dfs = []
    for i, member in enumerate(members):
        dates = [_decimal_year_to_date(row[0]) for row in member]
        vals  = [float(row[1]) for row in member]
        dfs.append(pd.DataFrame({'date': dates, f'ens_{i}': vals}))

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on='date', how='outer')

    df = df.sort_values('date').reset_index(drop=True)
    logger.info(f"Loaded {gen}: {len(df)} months, {len(members)} members")
    return df


def compute_ensemble_stats(df: pd.DataFrame,
                            baseline_start: int = 1850,
                            baseline_end: int = 1900) -> pd.DataFrame:
    """
    Rebaseline ensemble to 1850-1900 mean and compute mean/p05/p95.

    Returns DataFrame with columns: date, mean, p05, p95.
    """
    if df.empty:
        return pd.DataFrame(columns=['date', 'mean', 'p05', 'p95'])

    ens_cols = [c for c in df.columns if c.startswith('ens_')]
    data = df[ens_cols].copy()

    # Rebaseline each member to 1850-1900
    mask = (df['date'].dt.year >= baseline_start) & (df['date'].dt.year <= baseline_end)
    for col in ens_cols:
        base_mean = data.loc[mask, col].mean()
        if not np.isnan(base_mean):
            data[col] = data[col] - base_mean

    result = pd.DataFrame({'date': df['date']})
    result['mean'] = data.mean(axis=1)
    result['p05']  = data.quantile(0.05, axis=1)
    result['p95']  = data.quantile(0.95, axis=1)
    return result


# ── Observational data ────────────────────────────────────────────────────────

def _import_gistemp() -> pd.DataFrame:
    """Import NASA GISTEMP v4."""
    df = pd.read_csv(GISTEMP_URL, sep=r'\s+', skiprows=7)
    df = df[~df['Year'].isin(['Year'])]
    long = pd.melt(df, id_vars=['Year'], var_name='Month', value_name='Anomaly')
    long = long[long['Month'].str.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$')]
    long['Anomaly'] = pd.to_numeric(long['Anomaly'], errors='coerce') / 100.0
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    long['month'] = long['Month'].map(month_map)
    long['year'] = pd.to_numeric(long['Year'], errors='coerce')
    long = long.dropna(subset=['year','month','Anomaly']).sort_values(['year','month'])
    return long[['year','month']].assign(gistemp=long['Anomaly'].values).reset_index(drop=True)


def _import_noaa() -> pd.DataFrame:
    """Import NOAA GlobalTemp."""
    df = pd.read_csv(_noaa_url(), skiprows=5, names=['date', 'noaa'])
    df['year']  = df['date'].astype(str).str[:4].astype(int)
    df['month'] = df['date'].astype(str).str[4:6].astype(int)
    df['noaa']  = pd.to_numeric(df['noaa'], errors='coerce')
    return df[['year','month','noaa']].dropna().reset_index(drop=True)


def _import_hadcrut5() -> pd.DataFrame:
    """Import HadCRUT5."""
    df = pd.read_csv(HADCRUT5_URL)
    df = df.rename(columns={'Time': 'date', 'Anomaly (deg C)': 'hadcrut5'})
    df['date']  = pd.to_datetime(df['date'], format='%Y-%m')
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df[['year','month','hadcrut5']].reset_index(drop=True)


def _import_berkeley() -> pd.DataFrame:
    """Import Berkeley Earth."""
    df = pd.read_csv(BERKELEY_URL, comment='%', sep=r'\s+', header=None,
                     usecols=[0,1,2], names=['year','month','berkeley'])
    df = df[pd.to_numeric(df['year'], errors='coerce').notna()]
    df['year']    = df['year'].astype(int)
    df['month']   = df['month'].astype(int)
    df['berkeley'] = pd.to_numeric(df['berkeley'], errors='coerce')
    return df[['year','month','berkeley']].dropna().reset_index(drop=True)


def _import_copernicus() -> pd.DataFrame:
    """Derive monthly Copernicus/ERA5 anomalies from the local ERA5 daily CSV.

    The daily file (1991-2020 baseline, back to 1940) is already fetched for the
    Global Temperature tab, so we aggregate to monthly means instead of scraping
    the Copernicus website.
    """
    era5_csv = DATA_DIR.parent / 'era5_daily_series_2t_global.csv'
    try:
        df = pd.read_csv(era5_csv, comment='#')
        df['date'] = pd.to_datetime(df['date'])
        df['year']  = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['ano_91-20'] = pd.to_numeric(df['ano_91-20'], errors='coerce')
        monthly = df.groupby(['year', 'month'])['ano_91-20'].mean().reset_index()
        monthly.rename(columns={'ano_91-20': 'copernicus'}, inplace=True)
        return monthly[['year', 'month', 'copernicus']].dropna().reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Copernicus/ERA5 daily import failed: {e}")
        return pd.DataFrame(columns=['year', 'month', 'copernicus'])


def _rebaseline(df: pd.DataFrame, col: str, start: int = 1981, end: int = 2010) -> pd.DataFrame:
    """Subtract mean anomaly over [start, end] from column col."""
    mask = df['year'].between(start, end)
    mean = df.loc[mask, col].mean()
    df = df.copy()
    df[col] = df[col] - mean
    return df


def fetch_obs_data(cache_path: Path = OBS_CACHE, force: bool = False) -> pd.DataFrame:
    """
    Fetch all 5 observational datasets, rebaseline to 1981-2010, merge, cache.
    Falls back to cached file if any fetch fails.
    """
    cache_path = Path(cache_path)

    # Check if cache is fresh (updated this month)
    if not force and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        today = datetime.today()
        if mtime.year == today.year and mtime.month == today.month:
            logger.info("Obs cache is current, skipping fetch")
            return load_obs_data(cache_path)

    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        logger.info("Fetching observational temperature datasets...")
        dfs = [
            _rebaseline(_import_hadcrut5(), 'hadcrut5'),
            _rebaseline(_import_gistemp(),  'gistemp'),
            _rebaseline(_import_noaa(),     'noaa'),
            _rebaseline(_import_berkeley(), 'berkeley'),
            _rebaseline(_import_copernicus(), 'copernicus'),
        ]
        merged = reduce(
            lambda l, r: pd.merge(l, r, on=['year','month'], how='outer'), dfs
        ).sort_values(['year','month']).reset_index(drop=True)
        merged = merged.round(4)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(cache_path, index=False)
        logger.info(f"Obs data saved to {cache_path}: {len(merged)} rows")
        return merged
    except Exception as e:
        logger.error(f"Failed to fetch obs data: {e}. Using cache.")
        if cache_path.exists():
            return load_obs_data(cache_path)
        raise


def load_obs_data(csv_path: Path = OBS_CACHE) -> pd.DataFrame:
    """
    Load cached observational CSV and shift baseline from 1981-2010 to 1850-1900.

    Returns DataFrame with columns: date, hadcrut5, gistemp, noaa, berkeley, copernicus
    All values are anomalies relative to the 1850-1900 preindustrial mean.
    """
    df = pd.read_csv(csv_path)
    # Drop any unnamed index column
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df['year']  = df['year'].astype(int)
    df['month'] = df['month'].astype(int)

    # Shift each series from 1981-2010 baseline to 1850-1900 baseline
    for col, offset in PREINDUSTRIAL_OFFSETS.items():
        if col in df.columns:
            df[col] = df[col] - offset  # offset is already negative, so subtract adds ~0.7

    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))
    return df


def bridge_with_daily_era5(obs_df: pd.DataFrame,
                            era5_daily: pd.DataFrame) -> pd.Series:
    """
    Aggregate daily ERA5 to monthly means and return as a Series indexed by date,
    using the same preindustrial baseline as obs_df.

    era5_daily must have columns: date (datetime), anomaly (already preindustrial-referenced).
    """
    if era5_daily is None or era5_daily.empty:
        return pd.Series(dtype=float)

    df = era5_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    monthly = df.groupby(['year','month'])['anomaly'].mean().reset_index()
    monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

    # Only return months not already in the copernicus column of obs_df
    last_obs = obs_df['date'].max() if 'date' in obs_df.columns else pd.Timestamp('1900-01-01')
    return monthly[monthly['date'] > last_obs].set_index('date')['anomaly']


# ── Trend calculations ─────────────────────────────────────────────────────────

def _trend_cdecade(x_dates: pd.Series, y_vals: pd.Series) -> float:
    """Compute linear trend in °C/decade via linregress."""
    x = (x_dates - x_dates.iloc[0]).dt.days.values.astype(float)
    slope, *_ = linregress(x, y_vals.values)
    return slope * 365.25 * 10   # °C/decade


def calculate_member_trends(cmip_df: pd.DataFrame,
                             start: str, end: str) -> np.ndarray:
    """Return array of per-member warming trends (°C/decade)."""
    if cmip_df.empty:
        return np.array([])
    mask = (cmip_df['date'] >= start) & (cmip_df['date'] <= end)
    sub  = cmip_df[mask].dropna(subset=['date'])
    ens_cols = [c for c in sub.columns if c.startswith('ens_')]

    # Rebaseline to 1850-1900 before trend
    stats = compute_ensemble_stats(cmip_df)
    stats_sub = stats[mask]

    trends = []
    for col in ens_cols:
        s = sub[col].dropna()
        if len(s) < 12:
            continue
        trends.append(_trend_cdecade(sub.loc[s.index, 'date'], s))
    return np.array(trends)


def calculate_obs_trends(obs_df: pd.DataFrame,
                          start: str, end: str) -> dict:
    """Compute trend (°C/decade) for each observational dataset."""
    obs_cols = ['hadcrut5', 'gistemp', 'noaa', 'berkeley', 'copernicus']
    mask = (obs_df['date'] >= start) & (obs_df['date'] <= end)
    sub  = obs_df[mask]
    result = {}
    for col in obs_cols:
        if col not in sub.columns:
            continue
        s = sub[col].dropna()
        if len(s) < 24:
            result[col] = None
            continue
        result[col] = _trend_cdecade(sub.loc[s.index, 'date'], s)
    return result


def compute_variable_start_trends(cmip_df: pd.DataFrame,
                                   obs_df: pd.DataFrame,
                                   end_date: str = None,
                                   min_years: int = 15) -> dict:
    """
    For every start year from 1970 to (end_year - min_years), compute:
    - model mean, p05, p95 trend
    - obs trends for each dataset

    Returns dict with keys:
      start_years, model_mean, model_p05, model_p95, obs_{name}
    """
    if end_date is None:
        end_date = obs_df['date'].max().strftime('%Y-%m-%d')

    end_dt = pd.Timestamp(end_date)
    max_start_year = end_dt.year - min_years

    start_years = list(range(1970, max_start_year + 1))
    obs_cols = ['hadcrut5', 'gistemp', 'noaa', 'berkeley', 'copernicus']

    model_means, model_p05s, model_p95s = [], [], []
    obs_trends_by_col = {c: [] for c in obs_cols}

    ens_cols = [c for c in cmip_df.columns if c.startswith('ens_')]
    # Rebaseline cmip data once
    stats = compute_ensemble_stats(cmip_df)
    # Merge stats back onto cmip_df for individual member trends
    cmip_rebased = cmip_df.copy()
    for c in ens_cols:
        mask_bl = (cmip_df['date'].dt.year >= 1850) & (cmip_df['date'].dt.year <= 1900)
        base_mean = cmip_df.loc[mask_bl, c].mean()
        if not np.isnan(base_mean):
            cmip_rebased[c] = cmip_df[c] - base_mean

    for sy in start_years:
        start = f'{sy}-01-01'
        mask_m = (cmip_rebased['date'] >= start) & (cmip_rebased['date'] <= end_date)
        sub_m  = cmip_rebased[mask_m]

        member_trends = []
        for col in ens_cols:
            s = sub_m[col].dropna()
            if len(s) < 12:
                continue
            member_trends.append(_trend_cdecade(sub_m.loc[s.index, 'date'], s))

        if member_trends:
            model_means.append(np.mean(member_trends))
            model_p05s.append(np.percentile(member_trends, 5))
            model_p95s.append(np.percentile(member_trends, 95))
        else:
            model_means.append(np.nan)
            model_p05s.append(np.nan)
            model_p95s.append(np.nan)

        mask_o = (obs_df['date'] >= start) & (obs_df['date'] <= end_date)
        sub_o  = obs_df[mask_o]
        for col in obs_cols:
            if col not in sub_o.columns:
                obs_trends_by_col[col].append(np.nan)
                continue
            s = sub_o[col].dropna()
            if len(s) < 24:
                obs_trends_by_col[col].append(np.nan)
            else:
                obs_trends_by_col[col].append(
                    _trend_cdecade(sub_o.loc[s.index, 'date'], s)
                )

    result = {
        'start_years': start_years,
        'model_mean':  model_means,
        'model_p05':   model_p05s,
        'model_p95':   model_p95s,
    }
    for col in obs_cols:
        result[f'obs_{col}'] = obs_trends_by_col[col]
    return result


# ── Statistics card data ──────────────────────────────────────────────────────

def _trend_card(cmip_df, obs_df, start_year, last_date):
    """Compute obs mean trend, model mean trend, and model 5-95 range for a period."""
    start = f'{start_year}-01-01'
    end = last_date.strftime('%Y-%m-%d')
    obs_trends = calculate_obs_trends(obs_df, start, end)
    obs_vals = [v for v in obs_trends.values() if v is not None]
    obs_mean = np.mean(obs_vals) if obs_vals else None
    model_trends = calculate_member_trends(cmip_df, start, end)
    model_mean = np.mean(model_trends) if len(model_trends) > 0 else None
    model_p05 = np.percentile(model_trends, 5) if len(model_trends) > 0 else None
    model_p95 = np.percentile(model_trends, 95) if len(model_trends) > 0 else None
    return obs_mean, model_mean, model_p05, model_p95


def compute_model_obs_cards(cmip_df: pd.DataFrame, obs_df: pd.DataFrame,
                            cmip_label: str = 'CMIP6') -> dict:
    """Compute values for the 4 statistics cards.

    Cards:
      1. Model vs Observed warming since preindustrial (last 12 months)
      2. 1970-Present trend
      3. Past 25 years trend (dynamic start year)
      4. Past 15 years trend (dynamic start year)
    """
    stats = compute_ensemble_stats(cmip_df)
    current_year = datetime.today().year

    # Card 1: Current warming level via LOWESS smooth (20-year bandwidth)
    last_date = obs_df['date'].max()
    obs_cols = ['hadcrut5', 'gistemp', 'noaa', 'berkeley', 'copernicus']
    obs_mean_ts = obs_df[obs_cols].mean(axis=1)
    obs_time = obs_df['date'].dt.year + (obs_df['date'].dt.month - 0.5) / 12
    valid = obs_mean_ts.notna()
    t_valid = obs_time[valid].values
    y_valid = obs_mean_ts[valid].values
    span_years = t_valid[-1] - t_valid[0]
    frac = min(20.0 / span_years, 1.0) if span_years > 0 else 1.0
    smoothed = lowess(y_valid, t_valid, frac=frac, return_sorted=True)
    obs_recent_mean = smoothed[-1, 1]  # last smoothed value = current warming level

    # Model LOWESS — truncate to obs time range so endpoint matches present
    stats_trunc = stats[stats['date'] <= last_date].copy()
    model_time = stats_trunc['date'].dt.year + (stats_trunc['date'].dt.month - 0.5) / 12
    m_span = model_time.values[-1] - model_time.values[0] if len(model_time) > 1 else 1
    m_frac = min(20.0 / m_span, 1.0)
    model_smooth = lowess(stats_trunc['mean'].values, model_time.values, frac=m_frac, return_sorted=True)
    model_recent_mean = model_smooth[-1, 1]
    if 'p05' in stats_trunc.columns:
        p05_smooth = lowess(stats_trunc['p05'].values, model_time.values, frac=m_frac, return_sorted=True)
        model_p05_recent = p05_smooth[-1, 1]
    else:
        model_p05_recent = None
    if 'p95' in stats_trunc.columns:
        p95_smooth = lowess(stats_trunc['p95'].values, model_time.values, frac=m_frac, return_sorted=True)
        model_p95_recent = p95_smooth[-1, 1]
    else:
        model_p95_recent = None

    # Card 2: 1970-Present trend
    obs_1970, model_1970, m05_1970, m95_1970 = _trend_card(cmip_df, obs_df, 1970, last_date)

    # Card 3: Past 25 years trend
    start_25 = current_year - 25
    obs_25, model_25, m05_25, m95_25 = _trend_card(cmip_df, obs_df, start_25, last_date)

    # Card 4: Past 15 years trend
    start_15 = current_year - 15
    obs_15, model_15, m05_15, m95_15 = _trend_card(cmip_df, obs_df, start_15, last_date)

    def _fmt(v, suffix=''):
        return f'{v:.2f}{suffix}' if v is not None else 'N/A'

    def _fmt_range(lo, hi, suffix=''):
        if lo is not None and hi is not None:
            return f'{lo:.2f}–{hi:.2f}{suffix}'
        return 'N/A'

    return {
        'cmip_label': cmip_label,
        # Card 1
        'obs_warming': _fmt(obs_recent_mean, '°C'),
        'model_warming': _fmt(model_recent_mean, '°C'),
        'model_warming_range': _fmt_range(model_p05_recent, model_p95_recent, '°C'),
        # Card 2
        'obs_trend_1970': _fmt(obs_1970, '°C/dec'),
        'model_trend_1970': _fmt(model_1970, '°C/dec'),
        'model_range_1970': _fmt_range(m05_1970, m95_1970, '°C/dec'),
        # Card 3
        'start_25': start_25,
        'obs_trend_25': _fmt(obs_25, '°C/dec'),
        'model_trend_25': _fmt(model_25, '°C/dec'),
        'model_range_25': _fmt_range(m05_25, m95_25, '°C/dec'),
        # Card 4
        'start_15': start_15,
        'obs_trend_15': _fmt(obs_15, '°C/dec'),
        'model_trend_15': _fmt(model_15, '°C/dec'),
        'model_range_15': _fmt_range(m05_15, m95_15, '°C/dec'),
    }


def compute_running_scorecard(cmip_stats: pd.DataFrame,
                               obs_df: pd.DataFrame,
                               end_year: int) -> dict:
    """
    Compute running scorecard metrics through end_year:
    - pct_in_range: % of obs months inside model 5th-95th percentile
    - trend_diff: obs mean trend minus model mean trend (°C/decade)
    """
    obs_cols = ['hadcrut5','gistemp','noaa','berkeley','copernicus']
    mask_s = cmip_stats['date'].dt.year <= end_year
    mask_o = (obs_df['date'].dt.year >= 1970) & (obs_df['date'].dt.year <= end_year)

    s_sub = cmip_stats[mask_s]
    o_sub = obs_df[mask_o]

    merged = o_sub.merge(s_sub[['date','p05','p95']], on='date', how='inner')
    if merged.empty:
        return {'pct_in_range': 'N/A', 'trend_diff': 'N/A'}

    obs_mean = merged[obs_cols].mean(axis=1)
    in_range = ((obs_mean >= merged['p05']) & (obs_mean <= merged['p95'])).sum()
    total    = obs_mean.notna().sum()
    pct = int(round(in_range / total * 100)) if total > 0 else 0

    # Trend difference
    s_end = f'{end_year}-12-31'
    obs_t = calculate_obs_trends(o_sub, '1970-01-01', s_end)
    obs_t_vals = [v for v in obs_t.values() if v is not None]
    obs_trend_mean = np.mean(obs_t_vals) if obs_t_vals else None

    s_trend_sub = s_sub[s_sub['date'] >= '1970-01-01']
    if len(s_trend_sub) >= 12:
        model_trend = _trend_cdecade(s_trend_sub['date'], s_trend_sub['mean'])
    else:
        model_trend = None

    if obs_trend_mean is not None and model_trend is not None:
        diff = obs_trend_mean - model_trend
        trend_diff_str = f'{diff:+.2f}°C/dec'
    else:
        trend_diff_str = 'N/A'

    return {
        'pct_in_range': f'{pct}%',
        'trend_diff':   trend_diff_str,
    }


# ── Plot functions ─────────────────────────────────────────────────────────────

def _base_layout(theme: dict, title: str, height: int = 500) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=theme['text_color'])),
        height=height,
        template=theme['template'],
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color']),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    font=dict(size=11, color=theme['text_color'])),
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(gridcolor=theme['grid_color'], showgrid=True),
        yaxis=dict(gridcolor=theme['grid_color'], showgrid=True),
        hovermode='x unified',
    )


def create_models_vs_obs_timeseries(
    cmip_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    rolling: bool = False,
    dark_mode: bool = False,
    gen_label: str = 'CMIP6',
    baseline: str = '1850-1900',
) -> go.Figure:
    """
    Visualization 1: Model envelope vs. all observational datasets, 1900-2040.

    baseline: one of '1850-1900', '1951-1980', '1961-1990', '1971-2000', '1981-2010'.
    """
    from src.dashboard import get_theme
    theme = get_theme(dark_mode)
    mc    = MODEL_COLORS['dark' if dark_mode else 'light']
    oc    = OBS_COLORS['dark' if dark_mode else 'light']

    bl_start, bl_end = (int(x) for x in baseline.split('-'))

    stats = compute_ensemble_stats(cmip_df, baseline_start=bl_start, baseline_end=bl_end)

    # Filter to 1900-2040
    stats = stats[(stats['date'].dt.year >= 1900) & (stats['date'].dt.year <= 2040)]

    if rolling:
        stats = stats.set_index('date').rolling(12, min_periods=6).mean().reset_index()

    # Rebaseline obs from 1850-1900 to the selected baseline
    obs_cols_ordered = ['hadcrut5','gistemp','noaa','berkeley','copernicus']
    obs_to_plot = obs_df.copy()
    if baseline != '1850-1900':
        bl_mask = (obs_to_plot['date'].dt.year >= bl_start) & (obs_to_plot['date'].dt.year <= bl_end)
        for c in obs_cols_ordered:
            if c in obs_to_plot.columns:
                bl_mean = obs_to_plot.loc[bl_mask, c].mean()
                if not np.isnan(bl_mean):
                    obs_to_plot[c] = obs_to_plot[c] - bl_mean

    fig = go.Figure()

    # Model 5th-95th band
    fig.add_trace(go.Scatter(
        x=stats['date'], y=stats['p05'],
        mode='lines', line=dict(width=0), showlegend=False,
        hoverinfo='skip', name='_p05',
    ))
    fig.add_trace(go.Scatter(
        x=stats['date'], y=stats['p95'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor=mc['band'],
        name=f'{gen_label} 5th–95th %ile',
        hovertemplate='%{y:.2f}°C<extra>Model range</extra>',
    ))

    # Model ensemble mean
    fig.add_trace(go.Scatter(
        x=stats['date'], y=stats['mean'],
        mode='lines', line=dict(color=mc['mean'], width=2),
        name=f'{gen_label} mean',
        hovertemplate='%{y:.2f}°C<extra>' + gen_label + ' mean</extra>',
    ))

    # Observational datasets
    obs_labels = {
        'hadcrut5':   'HadCRUT5',
        'gistemp':    'GISTEMP v4',
        'noaa':       'NOAA GlobalTemp',
        'berkeley':   'Berkeley Earth',
        'copernicus': 'Copernicus/ERA5',
    }
    if rolling:
        for c in obs_cols_ordered:
            if c in obs_to_plot.columns:
                obs_to_plot[c] = obs_to_plot[c].rolling(12, min_periods=6).mean()

    for col in obs_cols_ordered:
        if col not in obs_to_plot.columns:
            continue
        sub = obs_to_plot[obs_to_plot['date'].dt.year >= 1900]
        s = sub[['date', col]].dropna()
        fig.add_trace(go.Scatter(
            x=s['date'], y=s[col],
            mode='lines', line=dict(color=oc[col], width=1.5),
            name=obs_labels[col],
            hovertemplate='%{y:.2f}°C<extra>' + obs_labels[col] + '</extra>',
        ))

    # 1.5°C reference line (only meaningful for 1850-1900 baseline)
    if baseline == '1850-1900':
        fig.add_hline(y=1.5, line_dash='dash',
                      line_color=theme.get('threshold_color', 'orange'),
                      annotation_text='1.5°C', annotation_position='top left',
                      annotation_font_color=theme['text_color'])

    bl_label = f'{bl_start}\u2013{bl_end}'
    layout = _base_layout(theme, f'Climate Models vs. Observations ({gen_label})', height=520)
    layout.update(
        yaxis_title=f'Temperature anomaly (\u00b0C, rel. {bl_label})',
        xaxis=dict(
            gridcolor=theme['grid_color'], showgrid=True,
            rangeslider=dict(visible=True, thickness=0.05),
            range=['1900-01-01', '2040-12-31'],
        ),
    )
    fig.update_layout(layout)
    return fig


def create_trend_explorer(
    cmip_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    dark_mode: bool = False,
    gen_label: str = 'CMIP6',
) -> go.Figure:
    """
    Visualization 2: Warming trends by start year.
    X = start year (1970-2010), Y = trend to present (°C/decade).
    """
    from src.dashboard import get_theme
    theme = get_theme(dark_mode)
    mc    = MODEL_COLORS['dark' if dark_mode else 'light']
    oc    = OBS_COLORS['dark' if dark_mode else 'light']

    data = compute_variable_start_trends(cmip_df, obs_df)
    syears = data['start_years']

    fig = go.Figure()

    # Model band
    fig.add_trace(go.Scatter(
        x=syears, y=data['model_p05'],
        mode='lines', line=dict(width=0), showlegend=False,
        hoverinfo='skip', name='_p05',
    ))
    fig.add_trace(go.Scatter(
        x=syears, y=data['model_p95'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor=mc['band'],
        name=f'{gen_label} 5th–95th %ile',
        hovertemplate='%{y:.3f}°C/dec<extra>Model range</extra>',
    ))

    # Model mean
    fig.add_trace(go.Scatter(
        x=syears, y=data['model_mean'],
        mode='lines', line=dict(color=mc['mean'], width=2.5),
        name=f'{gen_label} mean',
        hovertemplate='%{y:.3f}°C/dec<extra>' + gen_label + ' mean</extra>',
    ))

    # Observational trends
    obs_labels = {
        'hadcrut5':   'HadCRUT5',
        'gistemp':    'GISTEMP v4',
        'noaa':       'NOAA GlobalTemp',
        'berkeley':   'Berkeley Earth',
        'copernicus': 'Copernicus/ERA5',
    }
    for col, label in obs_labels.items():
        key = f'obs_{col}'
        if key not in data:
            continue
        vals = data[key]
        fig.add_trace(go.Scatter(
            x=syears, y=vals,
            mode='lines', line=dict(color=oc[col], width=1.5, dash='dash'),
            name=label,
            hovertemplate='%{y:.3f}°C/dec<extra>' + label + '</extra>',
        ))

    layout = _base_layout(theme, 'Warming Trend by Start Year (to present)', height=500)
    layout.update(
        xaxis_title='Start Year of Trend',
        yaxis_title='Warming Trend (°C/decade)',
        xaxis=dict(gridcolor=theme['grid_color'], showgrid=True,
                   tickmode='linear', dtick=5),
    )
    fig.update_layout(layout)
    return fig


def create_trend_histogram_grid(
    cmip_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    dark_mode: bool = False,
    gen_label: str = 'CMIP6',
) -> go.Figure:
    """
    Visualization 3: 1×3 grid of trend histograms.
    Columns = three time periods for the selected CMIP generation.
    """
    from src.dashboard import get_theme
    theme = get_theme(dark_mode)
    oc    = OBS_COLORS['dark' if dark_mode else 'light']
    hist_color = HIST_COLORS['dark' if dark_mode else 'light']

    last_obs = obs_df['date'].max().strftime('%Y-%m-%d')
    periods = [
        ('1970-01-01', last_obs, '1970–present'),
        ('2001-01-01', last_obs, '2001–present'),
        ('2011-01-01', last_obs, '2011–present'),
    ]
    obs_labels = {
        'hadcrut5':   'HadCRUT5',
        'gistemp':    'GISTEMP',
        'noaa':       'NOAA',
        'berkeley':   'Berkeley',
        'copernicus': 'Copernicus',
    }

    subplot_titles = [p[2] for p in periods]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    for col_i, (start, end, period_label) in enumerate(periods):
        c = col_i + 1
        obs_tr = calculate_obs_trends(obs_df, start, end)
        member_trends = calculate_member_trends(cmip_df, start, end)

        if len(member_trends) > 0:
            fig.add_trace(go.Histogram(
                x=member_trends, nbinsx=15,
                marker_color=hist_color,
                marker_line_color=theme['text_color'],
                marker_line_width=0.5,
                showlegend=False,
                hovertemplate='Trend: %{x:.3f}°C/dec<br>Count: %{y}<extra></extra>',
            ), row=1, col=c)

        # Observed trend lines
        for obs_col, label in obs_labels.items():
            val = obs_tr.get(obs_col)
            if val is None:
                continue
            fig.add_vline(
                x=val, row=1, col=c,
                line_dash='dash', line_color=oc[obs_col], line_width=2,
            )

    fig.update_layout(
        height=380,
        template=theme['template'],
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'], size=11),
        showlegend=False,
        margin=dict(l=50, r=30, t=60, b=60),
        title=dict(
            text=f'{gen_label} Warming Trends vs. Observations',
            font=dict(size=14, color=theme['text_color']),
        ),
    )
    fig.update_annotations(font_size=12)
    for c in range(1, 4):
        fig.update_xaxes(
            title_text='°C/decade',
            gridcolor=theme['grid_color'],
            row=1, col=c,
        )
        fig.update_yaxes(
            gridcolor=theme['grid_color'],
            row=1, col=c,
        )
    return fig


def create_animated_scorecard(
    cmip_stats: pd.DataFrame,
    obs_df: pd.DataFrame,
    end_year: int = None,
    dark_mode: bool = False,
) -> go.Figure:
    """
    Visualization 4: Animated replay — obs line draws into model envelope.
    This returns the figure for the given end_year (used by the slider callback).
    """
    from src.dashboard import get_theme
    theme = get_theme(dark_mode)
    mc    = MODEL_COLORS['dark' if dark_mode else 'light']
    oc    = OBS_COLORS['dark' if dark_mode else 'light']

    if end_year is None:
        end_year = obs_df['date'].dt.year.max()

    # Full model stats (background)
    full_stats = cmip_stats[cmip_stats['date'].dt.year >= 1970]
    # Visible obs data (truncated to end_year)
    obs_cols = ['hadcrut5','gistemp','noaa','berkeley','copernicus']
    obs_visible = obs_df[
        (obs_df['date'].dt.year >= 1970) & (obs_df['date'].dt.year <= end_year)
    ].copy()
    obs_mean = obs_visible[obs_cols].mean(axis=1)
    obs_visible = obs_visible.assign(obs_mean=obs_mean)

    fig = go.Figure()

    # Static model band (always shown)
    fig.add_trace(go.Scatter(
        x=full_stats['date'], y=full_stats['p05'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=full_stats['date'], y=full_stats['p95'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor=mc['band'],
        name='CMIP6 5th–95th %ile',
        hovertemplate='%{y:.2f}°C<extra>Model range</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=full_stats['date'], y=full_stats['mean'],
        mode='lines', line=dict(color=mc['mean'], width=1.5, dash='dot'),
        name='CMIP6 mean', opacity=0.5,
        hovertemplate='%{y:.2f}°C<extra>Model mean</extra>',
    ))

    # Observed mean line (truncated)
    visible = obs_visible.dropna(subset=['obs_mean'])
    fig.add_trace(go.Scatter(
        x=visible['date'], y=visible['obs_mean'],
        mode='lines', line=dict(color=oc['berkeley'], width=2.5),
        name='Obs. mean (5 datasets)',
        hovertemplate='%{y:.2f}°C<extra>Obs. mean</extra>',
    ))

    # 1.5°C reference
    fig.add_hline(y=1.5, line_dash='dash',
                  line_color=theme.get('threshold_color', 'orange'),
                  annotation_text='1.5°C', annotation_position='top left',
                  annotation_font_color=theme['text_color'])

    # Key event markers (add_vline with date strings + annotations separately to avoid plotly bug)
    key_events = {
        1991: ('Pinatubo', -0.1),
        1998: ('El Niño', 0.8),
        2016: ('El Niño', 1.1),
        2023: ('Record warmth', 1.4),
    }
    for yr, (label, y_pos) in key_events.items():
        if yr <= end_year:
            fig.add_vline(x=f'{yr}-06-01', line_dash='dot',
                          line_color=theme['grid_color'], line_width=1)
            fig.add_annotation(
                x=f'{yr}-06-01', y=1.0, yref='paper',
                text=label, showarrow=False, xanchor='left',
                font=dict(size=9, color=theme['text_color']),
                bgcolor='rgba(0,0,0,0)',
            )

    layout = _base_layout(theme, f'Observations vs. Model Envelope (through {end_year})', height=480)
    layout.update(
        yaxis_title='Temperature anomaly (°C, rel. 1850–1900)',
        xaxis=dict(
            gridcolor=theme['grid_color'], showgrid=True,
            range=['1970-01-01', f'{max(end_year + 1, 2027)}-01-01'],
        ),
    )
    fig.update_layout(layout)
    return fig


# ── Static image generation ───────────────────────────────────────────────────

def generate_models_static_images(
    obs_df: pd.DataFrame,
    cmip3: pd.DataFrame,
    cmip5: pd.DataFrame,
    cmip6: pd.DataFrame,
    assets_dir: Path,
) -> None:
    """Pre-render all Models vs. Obs static images for dark and light modes."""
    import plotly.io as pio

    for dark_mode in [True, False]:
        mode = 'dark' if dark_mode else 'light'
        logger.info(f"Generating models static images ({mode})...")

        try:
            fig1 = create_models_vs_obs_timeseries(cmip6, obs_df, rolling=False, dark_mode=dark_mode)
            pio.write_image(fig1, assets_dir / f'models_timeseries_{mode}.png',
                            width=1200, height=520, scale=1.5)
        except Exception as e:
            logger.error(f"Failed to generate models_timeseries_{mode}: {e}")

        try:
            fig2 = create_trend_explorer(cmip6, obs_df, dark_mode=dark_mode)
            pio.write_image(fig2, assets_dir / f'models_trend_explorer_{mode}.png',
                            width=1200, height=500, scale=1.5)
        except Exception as e:
            logger.error(f"Failed to generate models_trend_explorer_{mode}: {e}")

        try:
            fig3 = create_trend_histogram_grid(cmip6, obs_df, dark_mode=dark_mode, gen_label='CMIP6')
            pio.write_image(fig3, assets_dir / f'models_histograms_{mode}.png',
                            width=1200, height=380, scale=1.5)
        except Exception as e:
            logger.error(f"Failed to generate models_histograms_{mode}: {e}")

    logger.info("Models static images generation complete")
