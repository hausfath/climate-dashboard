#!/usr/bin/env python3.13
"""Region-level tests of the literature-assessed map regions (and candidates) against
the Sep 2026 multi-model forecasts. Pre-registered: regions, expected signs and seasons
are fixed here from the map / dossier / reviewer list BEFORE looking at region results.

Output: derived/region_table.csv, derived/region_table.md
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import shapely
from shapely.geometry import Polygon, box
from scipy.stats import binom

ROOT = Path(__file__).parent
a = xr.open_dataset(ROOT / "derived" / "seasonal_anoms.nc")
m = xr.open_dataset(ROOT / "derived" / "metrics.nc")
o = xr.open_dataset(ROOT / "derived" / "obs_ref.nc")
LAT, LON = a.lat.values, a.lon.values
import sys as _sys; _sys.path.insert(0, str(ROOT)); from initcfg import NMME, SEASONS as _SEASONS   # noqa: E402
NMME = list(NMME)
C3S_IND = ["ecmwf", "ukmo", "meteo_france", "dwd", "cmcc", "jma", "bom"]

# current map polygons (same buffering as the drawn map)
LITP = json.loads((ROOT / "lit_polygons_v4.json").read_text())  # literature polygons as of map v4
BUF = {"peru": 0.8, "camerica": 1.0, "safrica": 1.4}
POLY = {k: Polygon(v["pts"]).buffer(BUF.get(k, 2.0), join_style=1) for k, v in LITP.items()}


def b(w, s, e, n):  # lon/lat box, lon in -180..180 or 0..360
    return box(w, s, e, n)


CAND = {
    "socal": b(-121, 32.5, -114, 35.5), "norcal": b(-124.5, 38, -120, 42),
    "nmexico_sw": b(-115, 25, -100, 33), "nsam": b(-77, 0, -55, 12),
    "philippines": b(117, 5, 127, 19), "mainland_sea": b(97, 10, 110, 22),
    "schina": b(105, 22, 122, 30), "hawaii": b(-161, 18, -154, 23),
    "antilles": b(-85, 18, -72, 24), "drycorridor": b(-92, 10, -83, 16),
    "alaska_wcan": b(-170, 55, -120, 70), "se_us": b(-95, 28, -78, 36),
}
ALLPOLY = {**POLY, **CAND}
OCEAN_OK = {"maritime", "wpacific", "cpacific", "hawaii", "philippines", "peru"}

# (region, variable, season, expected sign, source of expectation)
TESTS = [
    ("maritime", "pr", "SON", -1, "map"), ("maritime", "pr", "DJF", -1, "map"),
    ("safrica", "pr", "DJF", -1, "map"), ("safrica", "pr", "JFM", -1, "map"),
    ("amazon", "pr", "SON", -1, "map"), ("amazon", "pr", "DJF", -1, "map"), ("amazon", "pr", "MAM", -1, "map"),
    ("nebrazil", "pr", "MAM", -1, "map"),
    ("camerica", "pr", "DJF", -1, "map"),
    ("wpacific", "pr", "SON", -1, "map"), ("wpacific", "pr", "DJF", -1, "map"),
    ("australia", "pr", "SON", -1, "map"), ("australia", "pr", "OND", -1, "map"), ("australia", "pr", "DJF", -1, "map"),
    ("horn", "pr", "OND", +1, "map"), ("horn", "pr", "DJF", +1, "map"),
    ("peru", "pr", "DJF", +1, "map"), ("peru", "pr", "MAM", +1, "map"),
    ("sesa", "pr", "SON", +1, "map"), ("sesa", "pr", "DJF", +1, "map"),
    ("gulf", "pr", "DJF", +1, "map"), ("gulf", "pr", "JFM", +1, "map"),
    ("sindia", "pr", "OND", +1, "map"),
    ("pnw", "pr", "DJF", -1, "map"), ("pnw", "pr", "JFM", -1, "map"),
    ("ohio", "pr", "DJF", -1, "map"), ("ohio", "pr", "JFM", -1, "map"),
    ("cpacific", "pr", "SON", +1, "map"), ("cpacific", "pr", "DJF", +1, "map"),
    ("pnw", "t", "DJF", +1, "map"), ("ohio", "t", "DJF", +1, "map"),
    ("socal", "pr", "DJF", +1, "candidate"), ("socal", "pr", "JFM", +1, "candidate"),
    ("norcal", "pr", "DJF", +1, "candidate"), ("norcal", "pr", "JFM", +1, "candidate"),
    ("nmexico_sw", "pr", "DJF", +1, "candidate"), ("nmexico_sw", "pr", "JFM", +1, "candidate"),
    ("nsam", "pr", "SON", -1, "candidate"), ("nsam", "pr", "DJF", -1, "candidate"),
    ("philippines", "pr", "DJF", -1, "candidate"), ("philippines", "pr", "MAM", -1, "candidate"),
    ("mainland_sea", "pr", "MAM", -1, "candidate"),
    ("schina", "pr", "DJF", +1, "candidate"), ("schina", "pr", "MAM", +1, "candidate"),
    ("hawaii", "pr", "DJF", -1, "candidate"), ("hawaii", "pr", "JFM", -1, "candidate"),
    ("antilles", "pr", "DJF", +1, "reviewer"), ("drycorridor", "pr", "DJF", -1, "candidate"),
    ("alaska_wcan", "t", "DJF", +1, "candidate"), ("se_us", "t", "DJF", -1, "candidate"),
]

TESTS = [t for t in TESTS if t[2] in _SEASONS]   # seasons that begin before the start month have no forecast
LON2, LAT2 = np.meshgrid(np.where(LON > 180, LON - 360, LON), LAT)
W = np.cos(np.deg2rad(LAT2))
land = o.land.values.astype(bool)


def mask_for(name):
    g = ALLPOLY[name]
    # polygons may use 0..360 (cpacific) or -180..180
    msk = shapely.contains_xy(g, LON2, LAT2) | shapely.contains_xy(g, np.where(LON2 < 0, LON2 + 360, LON2), LAT2)
    if name not in OCEAN_OK:
        msk &= land
    return msk


def rmean(field, msk):
    ok = msk & np.isfinite(field)
    return float(np.sum(field[ok] * W[ok]) / np.sum(W[ok])) if ok.any() else np.nan


rows = []
for reg, var, s, exp, srcx in TESTS:
    msk = mask_for(reg)
    ens = [x for x in NMME + C3S_IND if not np.all(np.isnan(a.prate.sel(model=x, season=s).values))]
    if var == "pr":
        sd = o[f"pr_sd_{s}"].clip(min=0.05).values
        per = [rmean(a.prate.sel(model=x, season=s).values / sd, msk) for x in ens]
        clim = rmean(o[f"pr_clim_{s}"].values, msk)
        pexp = rmean(m[f"pr_pabove_{s}"].values if exp > 0 else m[f"pr_pbelow_{s}"].values, msk)
        rob = m[f"pr_robust_{s}"].values
    else:
        sd = o[f"t_sd_{s}"].clip(min=0.1).values
        per = [rmean(a.t2m_adj.sel(model=x, season=s).values / sd, msk) for x in ens]
        per = [p for p in per if np.isfinite(p)]
        per_lmr = [rmean(a.t2m_lmr.sel(model=x, season=s).values / sd, msk) for x in ens]
        clim, pexp = np.nan, np.nan
        rob = np.where((m[f"t_robust_{s}"] * exp > 0) & (m[f"t91_robust_{s}"] * exp > 0) &
                       (m[f"tlmr_robust_{s}"] * exp > 0), exp, 0)
    per = np.array(per)
    n = len(per)
    k = int(np.sum(np.sign(per) == exp))
    rob_frac = float(np.sum((rob * exp > 0) & msk) / max(msk.sum(), 1))
    rows.append(dict(region=reg, var=var, season=s, expected="wet" if (var == "pr" and exp > 0) else
                     "dry" if var == "pr" else ("warm" if exp > 0 else "cool"), source=srcx,
                     cells=int(msk.sum()), clim_mm_day=round(clim, 2) if np.isfinite(clim) else None,
                     mmm_z=round(float(per.mean()), 2), agree=f"{k}/{n}",
                     p_sign=binom.sf(k - 1, n, 0.5),
                     nmme_p_expected_tercile=round(pexp, 0) if np.isfinite(pexp) else None,
                     robust_frac=round(rob_frac, 2),
                     lmr_z=round(float(np.mean(per_lmr)), 2) if var == "t" else None))

df = pd.DataFrame(rows)
# Benjamini-Hochberg across all tests (q = 0.10); models are not independent -> p optimistic
p = df.p_sign.values
order = np.argsort(p)
bh = np.empty_like(p)
ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
bh[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
df["q_BH"] = bh.round(4)
df["p_sign"] = df.p_sign.round(4)


def verdict(r):
    n = int(r.agree.split("/")[1]); k = int(r.agree.split("/")[0])
    if r.clim_mm_day is not None and r.clim_mm_day < 0.5:
        return "dry season (uninformative)"
    if r.q_BH <= 0.10 and abs(r.mmm_z) >= 0.5 and r.robust_frac >= 0.25:
        return "SUPPORTED (robust)"
    if r.q_BH <= 0.10 and abs(r.mmm_z) >= 0.25:
        return "supported (moderate)"
    if k <= n / 2:
        return "CONTRADICTED / opposite sign" if (np.sign(r.mmm_z) != (1 if r.expected in ("wet", "warm") else -1)
                                                  and abs(r.mmm_z) >= 0.25) else "no signal"
    return "weak / inconclusive"


df["verdict"] = df.apply(verdict, axis=1)
df.to_csv(ROOT / "derived" / "region_table.csv", index=False)
(ROOT / "derived" / "region_table.md").write_text("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n")
pd.set_option("display.width", 250, "display.max_rows", 200)
print(df.drop(columns=["source", "cells"]).to_string(index=False))
