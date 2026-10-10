#!/usr/bin/env python3.13
"""Forecast windows for the model checks: each region is tested over its full observed window, truncated to the
months every forecast system covers.

Rule (27 Sep 2026, Zeke):
  model window = the region's observed window (hit_rates season code, e.g. "SOND", "DJFMA", "MAM1")
                 ∩ the months covered by all 13 systems (NMME 6 + C3S 7). C3S runs 6 months from the start month,
                 so a September start ends in February and an October start in March.
  if that covers less than half of the observed window: ∩ the NMME horizon (9 months), NMME models only (6).
  if that still covers less than half: no forecast ("beyond range"). The half rule keeps a season that is mostly
  over (or mostly beyond the horizon) from being judged on one or two months.
The start month and both horizons are read from the raw files (raw/nmme_YYYYMM, the C3S forecastMonth axis), so a
new initialization only needs the raw folders swapped.

Monthly anomalies:  derived/monthly_anoms.nc   prate (model, lead, lat, lon), mm/day, lead 0 = start month.
                    Built from the same readers as process.py (seasonal_anoms.nc = 3-month means of these).
Window normals:     window_obs(months) -> (clim, detrended SD) from GPCP monthly 1991–2020, as obs_ref.nc does for
                    3-month seasons. Windows are labelled by the year of their LAST month, which is how
                    process.season_series labels DJF, so 3-month windows reproduce obs_ref.nc exactly.
"""
from pathlib import Path
import re

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "derived"
MON = "JFMAMJJASOND"
MNAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_src = (HERE / "process.py").read_text()
P = {"__file__": str(HERE / "process.py")}
exec(_src[:_src.index("# ---------------- observations")], P)   # readers, grids, model lists (no side effects)
NMME, C3S = P["NMME"], P["C3S_IND"]

from initcfg import INIT_YEAR, INIT_MONTH, NMME_LEADS, C3S_LEADS   # noqa: E402  (ELNINO_INIT or newest raw folder)
INIT = INIT_YEAR * 12 + INIT_MONTH - 1                           # absolute month index
ALL_END, NMME_END = INIT + C3S_LEADS - 1, INIT + NMME_LEADS - 1


def obs_window(code, onset_year=INIT_YEAR):
    """hit_rates season code -> absolute month indices. 'SOND' = Sep–Dec of the onset year, 'DJF' = Dec–Feb,
    'MAM1' = Mar–May of the following year, 'ON0' = Oct–Nov of the onset year."""
    d = code[-1] if code[-1].isdigit() else None
    s = code.rstrip("0123456789")
    if d == "1":
        i = MON.index(s); return [(onset_year + 1) * 12 + i + k for k in range(len(s))]
    i = (MON + MON).index(s)
    return [onset_year * 12 + i + k for k in range(len(s))]


MIN_COVER = 0.5   # a window is only checked if the forecast covers at least half of its months


def model_window(code):
    """-> (absolute months, ensemble label 'all'|'nmme'|None). A window that is mostly before the start month
    (e.g. the 2026 Sahel monsoon) or mostly beyond the horizon gets no model check."""
    w = obs_window(code)
    a = [m for m in w if INIT <= m <= ALL_END]
    if len(a) >= MIN_COVER * len(w):
        return a, "all"
    n = [m for m in w if INIT <= m <= NMME_END]
    if len(n) >= MIN_COVER * len(w):
        return n, "nmme"
    return [], None


def label(months):
    if not months:
        return ""
    a, b = months[0], months[-1]
    ya, yb = a // 12, b // 12
    yr = f"{ya}" if ya == yb else f"{ya}–{str(yb)[2:]}"
    return f"{MNAME[a % 12]}–{MNAME[b % 12]} {yr}" if a != b else f"{MNAME[a % 12]} {ya}"


def ensemble(kind):
    return NMME + C3S if kind == "all" else NMME if kind == "nmme" else []


# ---------------------------------------------------------------- model fields
def monthly_anoms():
    f = OUT / "monthly_anoms.nc"
    tag = f"{INIT_YEAR}{INIT_MONTH:02d}"
    if f.exists():
        ds = xr.open_dataset(f)
        if ds.attrs.get("init") == tag and "c3s_leads" in ds.attrs:
            return ds.prate
        ds.close()                                  # a new initialization: rebuild from the raw files
    L = max(NMME_LEADS, C3S_LEADS)
    arr = np.full((len(NMME) + len(C3S), L, P["LAT"].size, P["LON"].size), np.nan, np.float32)
    for i, m in enumerate(NMME):
        arr[i, :NMME_LEADS] = P["nmme_monthly"](m, "prate")
    for j, c in enumerate(C3S):
        da = P["c3s_monthly"](c, "prate")
        for lead in range(C3S_LEADS):
            arr[len(NMME) + j, lead] = da.sel(forecastMonth=lead + 1).values     # forecastMonth 1 = start month
    ds = xr.Dataset({"prate": (("model", "lead", "lat", "lon"), arr)},
                    coords={"model": NMME + C3S, "lead": np.arange(L), "lat": P["LAT"], "lon": P["LON"]},
                    attrs={"init": tag, "nmme_leads": NMME_LEADS, "c3s_leads": C3S_LEADS, "note": f"{MNAME[INIT_MONTH - 1]} {INIT_YEAR} start; lead 0 = start month; mm/day anomaly vs each system's hindcast climatology"})
    ds.to_netcdf(f, encoding={"prate": {"zlib": True, "complevel": 4}})
    return ds.prate


def window_mean(prate, model, months):
    return prate.sel(model=model, lead=[m - INIT for m in months]).mean("lead").values


# ---------------------------------------------------------------- observed normals for a window
_gp = None
_cache = {}


def window_obs(months):
    """(clim, detrended SD) of the window mean, GPCP 1991–2020, on the model grid."""
    global _gp
    key = tuple(m % 12 for m in months) + (months[-1] // 12 - months[0] // 12,)
    if key in _cache:
        return _cache[key]
    if _gp is None:
        _gp = xr.open_dataset(RAW / "obs" / "gpcp_precip.mon.mean.nc", drop_variables=["time_bnds"]).precip
    t = _gp.time.dt
    idx = t.year.values * 12 + t.month.values - 1
    span = months[-1] - months[0]
    offs = [m - months[-1] for m in months]                 # relative to the last month
    last_moy = months[-1] % 12
    ser = {}
    for y in range(int(t.year.values.min()), int(t.year.values.max()) + 1):
        want = [y * 12 + last_moy + o for o in offs]
        pos = [np.where(idx == w)[0] for w in want]
        if all(len(p) for p in pos):
            ser[y] = _gp.isel(time=[p[0] for p in pos]).mean("time")
    s = xr.concat(list(ser.values()), dim="syear").assign_coords(syear=list(ser.keys()))
    clim = P["wrap_interp"](s.sel(syear=slice(1991, 2020)).mean("syear"), "lat", "lon").values
    sub = s.sel(syear=slice(1991, 2020))
    fit = xr.polyval(sub.syear, sub.polyfit("syear", 1).polyfit_coefficients)
    sd = P["wrap_interp"]((sub - fit).std("syear"), "lat", "lon").values
    _cache[key] = (clim, sd)
    return clim, sd


# ---------------------------------------------------------------- typical neutral year (hit_rates/analyze.py method)
# Neutral years: onset years 1951–2023 with |Nov–Jan ONI| < 0.5 (CPC ONI, same file as analyze.py).
# Typical neutral year for onset year Y: Theil-Sen trend fitted to the neutral years, evaluated at Y, plus the median
# neutral-year residual. For this event Y = INIT_YEAR, so the trend is extrapolated to 2026.
def _neutral_years():
    import pandas as pd
    t = pd.read_csv(HERE.parent / "oni_cpc_2026-09-15.txt", sep=r"\s+")
    t = t[t.SEAS == "NDJ"]
    oni = dict(zip(t.YR.values, t.ANOM.values))
    return [y for y in range(1951, 2024) if abs(oni.get(y, np.nan)) < 0.5]


NEUTRAL = _neutral_years()


def theil_typical(years, Y, target):
    """years (n,), Y (n, ...) -> typical neutral value at `target` (and the Theil-Sen slope). Same estimator as
    scipy.stats.theilslopes (median pairwise slope; intercept = median(y) - slope * median(x))."""
    x = np.asarray(years, float)
    i, j = np.triu_indices(len(x), 1)
    dx = (x[j] - x[i]).reshape((-1,) + (1,) * (Y.ndim - 1))
    sl = np.median((Y[j] - Y[i]) / dx, axis=0)
    ic = np.median(Y, axis=0) - sl * np.median(x)
    res = Y - (ic + sl * x.reshape((-1,) + (1,) * (Y.ndim - 1)))
    return ic + sl * target + np.median(res, axis=0), sl


def window_typical(months):
    """Typical neutral-year mean rainfall (mm/day) over the window for this event, per cell, from GPCP monthly,
    computed on the native 2.5° grid and interpolated like obs_ref.nc. Returns (field, n neutral years used)."""
    global _gp
    key = ("typ",) + tuple(m - INIT for m in months)
    if key in _cache:
        return _cache[key]
    if _gp is None:
        _gp = xr.open_dataset(RAW / "obs" / "gpcp_precip.mon.mean.nc", drop_variables=["time_bnds"]).precip
    t = _gp.time.dt
    pos = {int(a) * 12 + int(b) - 1: k for k, (a, b) in enumerate(zip(t.year.values, t.month.values))}
    yrs, fields = [], []
    for y in NEUTRAL:
        want = [m + (y - INIT_YEAR) * 12 for m in months]
        if all(w in pos for w in want):
            yrs.append(y)
            fields.append(_gp.isel(time=[pos[w] for w in want]).mean("time").values)
    Y = np.array(fields)
    typ, _ = theil_typical(yrs, Y, INIT_YEAR)
    da = xr.DataArray(typ, coords={"lat": _gp.lat, "lon": _gp.lon}, dims=("lat", "lon"))
    f = P["wrap_interp"](da, "lat", "lon").values
    clim, _ = window_obs(months)
    # where the extrapolated trend drives the typical year to (nearly) zero the ratio is meaningless: mask like deserts
    f = np.where((f > 0.1) & (f > 0.2 * clim), f, np.nan)
    out = (f, len(yrs))
    _cache[key] = out
    return out


if __name__ == "__main__":
    print(f"start {MNAME[INIT_MONTH - 1]} {INIT_YEAR}; all 13 systems to {label([ALL_END])}; NMME to {label([NMME_END])}")
    for c in ["SOND", "DJFMA", "ONDJFMAM", "SONDJF", "NDJFM", "ON0", "MAM1", "JJA1", "SONDJFMAM", "OND", "DJF"]:
        w, k = model_window(c)
        print(f"  {c:10s} -> {label(w) or 'beyond range':18s} {k}")
    monthly_anoms()
    print("monthly_anoms.nc ok")
