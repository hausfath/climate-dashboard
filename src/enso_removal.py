"""ENSO removal for the global temperature series.

Fits monthly preindustrial-adjusted ERA5 anomalies (1950-present) against a
quadratic time trend plus ONI lagged by three months, then returns the
de-meaned ENSO component per day so it can be subtracted from the daily
series without shifting the long-run mean or trend.

Method notes (see temp_files/figure_candidates/BRAINSTORM.md review):
- The lag is fixed at 3 months rather than searched: lags 1-5 are
  statistically indistinguishable in this data and ~3 months is the
  published-literature value (Foster & Rahmstorf 2011).
- ONI enters linearly only; a quadratic ONI term is not significant.
- Volcanic forcing is deliberately not modeled; the residual dips in
  1964/1983/1992 are volcanic cooling, and captions should say so.
"""
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

ONI_FILE = Path(DATA_DIR) / 'enso_combined.csv'
LAG_MONTHS = 3
FIT_START_YEAR = 1950


@lru_cache(maxsize=1)
def _fit_enso_coefficient(anom_key: tuple) -> tuple:
    """OLS of monthly anomaly on quadratic trend + lagged ONI.

    anom_key is a hashable tuple of (year, month, anomaly) rows so the fit
    is cached across callbacks for the lifetime of the loaded data.
    Returns (oni_coef, monthly DataFrame with the lagged ONI attached).
    """
    mo = pd.DataFrame(list(anom_key), columns=['year', 'month', 'anom'])

    oni = pd.read_csv(ONI_FILE)
    oni = oni[~oni['is_forecast']][['year', 'month', 'oni']]
    oni['oni_lag'] = oni['oni'].shift(LAG_MONTHS)
    mo = mo.merge(oni[['year', 'month', 'oni_lag']], on=['year', 'month'],
                  how='left')

    fit = mo[(mo['year'] >= FIT_START_YEAR)].dropna(subset=['anom', 'oni_lag'])
    t = fit['year'] + (fit['month'] - 0.5) / 12
    t = t - t.mean()
    X = np.column_stack([np.ones(len(fit)), t, t ** 2, fit['oni_lag']])
    beta, *_ = np.linalg.lstsq(X, fit['anom'].values, rcond=None)
    oni_coef = float(beta[3])
    logger.info("ENSO removal fit: %.3f °C per unit ONI (lag %d months, "
                "%d monthly obs)", oni_coef, LAG_MONTHS, len(fit))
    return oni_coef, mo


def enso_component_by_month(df_adj: pd.DataFrame) -> tuple:
    """De-meaned monthly ENSO component for a preindustrial-adjusted daily df.

    Returns (component keyed by (year, month), oni_coef). Days whose month
    has no lagged ONI (pre-1870 tail, if any) get component 0.
    """
    d = df_adj[['date', 'anomaly']].copy()
    d['year'] = d['date'].dt.year
    d['month'] = d['date'].dt.month
    mo = (d.groupby(['year', 'month'], as_index=False)['anomaly'].mean()
          .rename(columns={'anomaly': 'anom'}))
    key = tuple(mo.itertuples(index=False, name=None))
    oni_coef, merged = _fit_enso_coefficient(key)

    comp = oni_coef * merged['oni_lag']
    comp = comp - comp[merged['year'] >= FIT_START_YEAR].mean()
    merged['component'] = comp.fillna(0.0)
    lookup = merged.set_index(['year', 'month'])['component']
    return lookup, oni_coef


def remove_enso(df_adj: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the daily preindustrial-adjusted df with the ENSO
    component subtracted from ``anomaly``."""
    lookup, _ = enso_component_by_month(df_adj)
    out = df_adj.copy()
    idx = pd.MultiIndex.from_arrays([out['date'].dt.year, out['date'].dt.month])
    out['anomaly'] = out['anomaly'].values - lookup.reindex(idx).fillna(0.0).values
    return out
