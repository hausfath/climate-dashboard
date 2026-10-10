#!/usr/bin/env python3.13
"""Seasonal multi-model patterns for one NMME + C3S initialization (ELNINO_INIT=YYYYMM, else the newest raw/nmme_* folder).

Outputs (derived/, small; regenerate figures without re-downloading):
  seasonal_anoms.nc   per-model seasonal-mean anomalies: prate (mm/day), t2m raw (°C),
                      t2m_adj (°C, trend-adjusted), on the NMME 1° grid
  obs_ref.nc          GPCP precip climatology/SD, ERA5 T2m trend/SD per season
  metrics.nc          multi-model standardized mean, sign agreement, robust masks,
                      NMME calibrated tercile probabilities
  validation.txt      reproduction / consistency checks
See METHODS.md for choices and caveats.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import shapely
import cartopy.io.shapereader as shpreader
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).parent
RAW, OUT = ROOT / "raw", ROOT / "derived"
OUT.mkdir(exist_ok=True)

LAT = np.arange(90, -91, -1.0)
LON = np.arange(0, 360, 1.0)

from initcfg import (INIT_LABEL, INIT_YM, NMME_DIR, C3S_DIR, PROB_FILE, SEASONS, SEASON_DEF, PROB_LEAD,
                     ALL_MODEL_SEASONS, ens_for)   # noqa: E402  seasons follow from the init month
OBS_MONTHS = {s: m for s, m in SEASON_DEF.items()}

from initcfg import NMME, C3S_IND, C3S_DUP   # noqa: E402

# Hindcast-climatology midpoints (decimal year). NMME periods read from the
# long_name of CPC's /NMME/clim files, and verified: realtime anom == fcst - clim
# to <1e-3 K for GEM5.2_NEMO and CFSv2. CanESM5 / NCAR_CESM1 have no clim file on
# CPC -> baseline unknown -> excluded from temperature agreement.
BASE_MID = {"CFSv2": 2006.0, "NASA_GEOS5v2": 2006.0, "NCAR_CCSM4": 2006.0,  # 1992-2019
            "GEM5.2_NEMO": 1996.5,                                         # 1982-2010
            "CanESM5": None, "NCAR_CESM1": None}
BASE_MID.update({c: 2005.0 for c in C3S_IND + C3S_DUP})                    # 1993-2016


def wrap_interp(da, lat_name, lon_name):
    """Bilinear interp onto the 1° grid, periodic in longitude."""
    da = da.sortby(lat_name)
    lon = da[lon_name].values
    da = xr.concat([da.isel({lon_name: slice(-2, None)}).assign_coords({lon_name: lon[-2:] - 360}),
                    da,
                    da.isel({lon_name: slice(0, 2)}).assign_coords({lon_name: lon[:2] + 360})], lon_name)
    out = da.interp({lat_name: LAT, lon_name: LON})
    out = out.rename({lat_name: "lat", lon_name: "lon"})
    # rows poleward of the source grid: copy the nearest valid latitude row
    ax = out.get_axis_num("lat")
    v = np.moveaxis(out.values, ax, 0).copy()
    ok = np.where(np.isfinite(v).reshape(v.shape[0], -1).any(1))[0]
    v[: ok[0]] = v[ok[0]]
    v[ok[-1] + 1:] = v[ok[-1]]
    out.values = np.moveaxis(v, 0, ax)
    return out


# ---------------- model data ----------------
def nmme_monthly(model, var):
    ds = xr.open_dataset(NMME_DIR / f"{model}.{var}.{INIT_YM}.ENSMEAN.anom.nc", decode_times=False)
    x = ds.fcst.values  # (9, lat, lon)
    if var == "prate":
        x = x * 86400.0  # mm/s -> mm/day
    return x


def c3s_monthly(centre, var):
    short = {"prate": "tprate", "tmp2m": "t2m"}[var]
    ds = xr.open_dataset(C3S_DIR / f"{centre}_{short}.nc")
    da = ds[list(ds.data_vars)[0]]
    if "forecast_reference_time" in da.dims:  # absent for lagged-ensemble systems (BOM, JMA, NCEP)
        da = da.isel(forecast_reference_time=0)
    if var == "prate":
        da = da * 1000.0 * 86400.0  # m/s -> mm/day
    da = wrap_interp(da, "latitude", "longitude")
    return da.transpose("forecastMonth", "lat", "lon")


def seasonal(var):
    """dict season -> dict model -> 2-D array."""
    out = {s: {} for s in SEASONS}
    for m in NMME:
        x = nmme_monthly(m, var)
        for s, (ni, _, _) in SEASONS.items():
            out[s][m] = x[ni].mean(0)
    for c in C3S_IND + C3S_DUP:
        da = c3s_monthly(c, var)
        for s, (_, ci, _) in SEASONS.items():
            if ci is not None:
                out[s][c] = da.sel(forecastMonth=ci).mean("forecastMonth").values
    return out


# ---------------- observations ----------------
def season_series(da, months, time_name):
    """Seasonal means by 'season year' (DJF labelled by Jan-Feb year)."""
    t = da[time_name].dt
    yr = t.year.values.copy()
    if 12 in months and 1 in months:
        yr = np.where(t.month.values == 12, yr + 1, yr)
    sel = np.isin(t.month.values, months)
    d = da.isel({time_name: sel}).assign_coords(syear=(time_name, yr[sel]))
    g = d.groupby("syear")
    n = g.count(time_name)
    return g.mean(time_name).where(n == len(months))


def detrended_sd(s, y0, y1):
    s = s.sel(syear=slice(y0, y1))
    p = s.polyfit("syear", 1)
    fit = xr.polyval(s.syear, p.polyfit_coefficients)
    return (s - fit).std("syear")


gp = xr.open_dataset(RAW / "obs" / "gpcp_precip.mon.mean.nc", drop_variables=["time_bnds"]).precip
e5 = xr.open_dataset(RAW / "era5" / "era5_t2m_monthly_1979_2025_1deg.nc").t2m - 273.15
if "expver" in e5.coords:
    e5 = e5.drop_vars("expver")

obs = {}
for s, months in OBS_MONTHS.items():
    ps = season_series(gp, months, "time")
    obs[f"pr_clim_{s}"] = wrap_interp(ps.sel(syear=slice(1991, 2020)).mean("syear"), "lat", "lon")
    obs[f"pr_sd_{s}"] = wrap_interp(detrended_sd(ps, 1991, 2020), "lat", "lon")
    ts = season_series(e5, months, "valid_time").rename({"latitude": "lat", "longitude": "lon"})
    ts = ts.sel(syear=slice(1979, 2025))
    tr = ts.polyfit("syear", 1).polyfit_coefficients.sel(degree=1)
    tr91 = ts.sel(syear=slice(1991, 2025)).polyfit("syear", 1).polyfit_coefficients.sel(degree=1)
    sm = lambda a: gaussian_filter(a.values, sigma=1.5, mode=["nearest", "wrap"])
    obs[f"t_trend_{s}"] = (("lat", "lon"), sm(tr))          # °C / yr, 1979-2025
    obs[f"t_trend91_{s}"] = (("lat", "lon"), sm(tr91))      # sensitivity: 1991-2025
    obs[f"t_sd_{s}"] = detrended_sd(ts, 1991, 2020)
obs = xr.Dataset({k: (v if isinstance(v, tuple) else (("lat", "lon"), np.asarray(v))) for k, v in obs.items()},
                 coords={"lat": LAT, "lon": LON})

# land mask (Natural Earth 110m)
land_geom = shapely.union_all(list(shpreader.Reader(shpreader.natural_earth("110m", "physical", "land")).geometries()))
LON2, LAT2 = np.meshgrid(np.where(LON > 180, LON - 360, LON), LAT)
land = shapely.contains_xy(land_geom, LON2, LAT2)
obs["land"] = (("lat", "lon"), land)
obs.to_netcdf(OUT / "obs_ref.nc", encoding={v: {"zlib": True, "complevel": 4} for v in obs.data_vars})

# ---------------- seasonal anomalies ----------------
pr = seasonal("prate")
t2 = seasonal("tmp2m")
coslat = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
band = (np.abs(LAT2) <= 60) & land


def trend_adjust(s, model, x, trend_key="t_trend"):
    mid = BASE_MID[model]
    if mid is None:
        return np.full_like(x, np.nan)
    return x - obs[f"{trend_key}_{s}"].values * (SEASONS[s][2] - mid)


models_all = NMME + C3S_IND + C3S_DUP
arr = {k: np.full((len(models_all), len(SEASONS), LAT.size, LON.size), np.nan, np.float32)
       for k in ["prate", "t2m", "t2m_adj", "t2m_adj91", "t2m_lmr"]}
for j, s in enumerate(SEASONS):
    for i, m in enumerate(models_all):
        if m not in pr[s]:
            continue
        arr["prate"][i, j] = pr[s][m]
        arr["t2m"][i, j] = t2[s][m]
        arr["t2m_adj"][i, j] = trend_adjust(s, m, t2[s][m])
        arr["t2m_adj91"][i, j] = trend_adjust(s, m, t2[s][m], "t_trend91")
        lm = np.nansum(t2[s][m] * coslat * band) / np.sum(coslat * band)
        arr["t2m_lmr"][i, j] = t2[s][m] - lm  # sensitivity: land-mean removed
anoms = xr.Dataset({k: (("model", "season", "lat", "lon"), v) for k, v in arr.items()},
                   coords={"model": models_all, "season": list(SEASONS), "lat": LAT, "lon": LON})
anoms.attrs["note"] = (f"{INIT_LABEL} init. prate mm/day, t2m °C, vs each system's hindcast climatology. "
                       "t2m_adj: minus ERA5 1979-2025 trend x (target - baseline midpoint). "
                       "t2m_lmr: minus model's own 60S-60N land-mean anomaly.")
anoms.to_netcdf(OUT / "seasonal_anoms.nc", encoding={v: {"zlib": True, "complevel": 4} for v in anoms.data_vars})

# ---------------- metrics ----------------
AGREE, ZMIN = 0.8, 0.5
met = {}
for s in SEASONS:
    ens = [m for m in ens_for(s) if m in pr[s]]
    sub = anoms.sel(season=s, model=ens)
    for var, key, sdk in [("pr", "prate", "pr_sd"), ("t", "t2m_adj", "t_sd"),
                          ("t91", "t2m_adj91", "t_sd"), ("tlmr", "t2m_lmr", "t_sd")]:
        x = sub[key]
        valid = x.notnull().mean(("lat", "lon")) > 0.99  # tolerate isolated missing cells
        x = x.sel(model=valid)
        sd = obs[f"{sdk}_{s}"].clip(min=0.05 if var == "pr" else 0.1)
        z = (x / sd)
        mmm = z.mean("model")
        agree = (np.sign(x) == np.sign(mmm)).where(x.notnull()).mean("model")
        n = int(x.sizes["model"])
        need = AGREE if n >= 7 else (n - 1) / n  # 6-model seasons: >=5/6
        met[f"{var}_z_{s}"] = mmm
        met[f"{var}_agree_{s}"] = agree
        met[f"{var}_robust_{s}"] = ((agree >= need - 1e-9) & (np.abs(mmm) >= ZMIN)).astype("i1") * np.sign(mmm)
        met[f"{var}_n_{s}"] = xr.DataArray(n)
metrics = xr.Dataset({k: v.drop_vars([c for c in ("season", "model") if c in v.coords]) for k, v in met.items()})

# NMME calibrated (skill-adjusted) tercile probabilities; target 800 = SON (checked below)
prob = xr.open_dataset(PROB_FILE, decode_times=False)
_t0 = float(prob.target.values[0])          # first code = the season starting at lead 0 (check [3] below verifies)
tgt = {s: _t0 + L for s, L in PROB_LEAD.items() if len(SEASON_DEF[s]) == 3 and (_t0 + L) in prob.target.values}
for s, t in tgt.items():
    # stored as fractions despite units="percent" attribute (checked: above+normal+below = 1.0)
    pa = prob.prob_above.sel(target=t).values * 100
    pb = prob.prob_below.sel(target=t).values * 100
    metrics[f"pr_pabove_{s}"] = (("lat", "lon"), pa)
    metrics[f"pr_pbelow_{s}"] = (("lat", "lon"), pb)
metrics = metrics.assign_coords(lat=LAT, lon=LON)
metrics.attrs.update(agree_threshold=AGREE, z_threshold=ZMIN,
                     note="z = anomaly / obs detrended 1991-2020 interannual SD (GPCP precip, ERA5 T2m). "
                          f"Ensemble = 6 NMME + 7 C3S independent centres ({', '.join(ALL_MODEL_SEASONS)}); NMME only ({', '.join(s for s in SEASONS if s not in ALL_MODEL_SEASONS)}).")
metrics.to_netcdf(OUT / "metrics.nc", encoding={v: {"zlib": True, "complevel": 4} for v in metrics.data_vars if metrics[v].ndim})

# ---------------- validation ----------------
def pcorr(a, b, mask=None):
    w = coslat.copy()
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= mask
    a, b, w = a[ok], b[ok], w[ok]
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))


lines = []
for var in ["prate", "tmp2m"]:
    mine = np.mean([nmme_monthly(m, var) for m in NMME], axis=0)
    cpc = xr.open_dataset(NMME_DIR / f"NMME.{var}.{INIT_YM}.ENSMEAN.anom.nc", decode_times=False).fcst.values
    cpc = cpc[:9]
    if var == "prate":
        cpc = cpc * 86400
    r = [pcorr(mine[i], cpc[i]) for i in range(9)]
    d = np.nanmax(np.abs(mine - cpc))
    lines.append(f"[1] NMME {var}: equal-weight mean of 6 model ENSMEANs vs CPC NMME.ENSMEAN file: "
                 f"pattern r min {min(r):.4f} over 9 leads; max |diff| {d:.3g}")
for s in [ALL_MODEL_SEASONS[0], "DJF"]:
    for dup, ref in [("ncep", ["CFSv2"]), ("eccc", ["CanESM5", "GEM5.2_NEMO"])]:
        a = pr[s][dup]; b = np.mean([pr[s][m] for m in ref], axis=0)
        ta = t2[s][dup]; tb = np.mean([t2[s][m] for m in ref], axis=0)
        lines.append(f"[2] {s} C3S {dup} vs NMME {'+'.join(ref)}: precip r {pcorr(a, b):.3f}, "
                     f"T2m r {pcorr(ta, tb):.3f}")
# tercile-target convention: correlate (P_above - P_below) with NMME ENSMEAN seasonal mean for each target
cpcp = xr.open_dataset(NMME_DIR / f"NMME.prate.{INIT_YM}.ENSMEAN.anom.nc", decode_times=False).fcst.values[:9]
for t in prob.target.values[:3]:
    pd_ = (prob.prob_above.sel(target=t) - prob.prob_below.sel(target=t)).values
    rs = {f"months {k}-{k+2}": round(pcorr(pd_, cpcp[k:k + 3].mean(0)), 3) for k in range(0, 4)}
    lines.append(f"[3] NMME prob target {t:.0f}: r(Pabove-Pbelow, ENSMEAN mean of lead months) {rs}")
# baseline / trend-adjustment size
for s in [ALL_MODEL_SEASONS[0], "DJF"]:
    for m in ["CFSv2", "GEM5.2_NEMO", "ecmwf"]:
        off = obs[f"t_trend_{s}"].values * (SEASONS[s][2] - BASE_MID[m])
        lines.append(f"[4] {s} {m}: trend offset removed, 60S-60N land mean {np.nansum(off*coslat*band)/np.sum(coslat*band):.2f} °C")
    raw_frac = float(((anoms.sel(season=s, model=NMME + C3S_IND).t2m > 0).mean("model") >= 0.8).where(band).mean())
    adj = metrics[f"t_agree_{s}"].where(band)
    lines.append(f"[5] {s}: fraction of 60S-60N land with >=80% models warm, raw {raw_frac:.2f}; "
                 f"trend-adjusted robust warm {float((metrics[f't_robust_{s}'] > 0).where(band).mean()):.2f}, "
                 f"robust cold {float((metrics[f't_robust_{s}'] < 0).where(band).mean()):.2f}")
(OUT / "validation.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
