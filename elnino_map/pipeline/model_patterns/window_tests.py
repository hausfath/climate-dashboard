#!/usr/bin/env python3.13
"""Full-window model checks for every region in the blog's hit grid (windows.py has the rule).

Two tests, each with its original method; only the time window changes:
  "box"  the pre-registered literature region / candidate box (regions.py ALLPOLY): count of models whose regional
         mean standardized anomaly (anomaly / GPCP detrended SD, per cell) has the expected sign. This is what the
         hit grid and the blog text use where such a test exists.
  "poly" the final map polygon (hit_rates/extract.py SHAPES): count of models whose regional-mean anomaly has the
         expected sign (model_counts.py method). Used for rows marked † or "own" in the hit grid, and for the
         dashboard tiles (per-model % of normal).
Step 1 reproduces model_counts.csv and region_table.csv exactly with each region's old 3-month season. Step 2 runs
the full windows. Output: derived/window_tests.csv (the old tables are left untouched).
usage: python3.13 model_patterns/window_tests.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "hit_rates"))
import windows as WN                                   # noqa: E402
from initcfg import INIT_LABEL, SEASONS as _SEASONS, SEASON_DEF, season_abs_months, ens_for   # noqa: E402
from extract import SHAPES, OCEAN_OK as OCEAN_POLY     # noqa: E402

_src = (HERE / "regions.py").read_text()
R = {"__file__": str(HERE / "regions.py")}
exec(_src[:_src.index("rows = []")], R)               # ALLPOLY, mask_for, rmean, a (seasonal), o (obs_ref)
o, a_seas = R["o"], R["a"]
land = o.land.values.astype(bool)
LON2, LAT2 = np.meshgrid(o.lon.values, o.lat.values)
W = np.cos(np.deg2rad(LAT2))
PR = WN.monthly_anoms()
def season_abs(s):
    return season_abs_months(s)


def poly_mask(key):
    g = SHAPES[key]
    msk = shapely.contains_xy(g, LON2, LAT2)
    if key not in OCEAN_POLY:
        msk &= land
    if msk.sum() == 0:
        msk = shapely.contains_xy(g.buffer(0.75), LON2, LAT2) & (land | (key in OCEAN_POLY))
    return msk


def rmean(f, msk):
    ok = msk & np.isfinite(f)
    return float(np.sum(f[ok] * W[ok]) / np.sum(W[ok])) if ok.any() else np.nan


def test(msk, months, kind, sign, method, want_pct=True):
    ens = WN.ensemble(kind)
    clim, sd = WN.window_obs(months)
    sdc = np.clip(sd, 0.05, None)
    per, pct = [], []
    for m in ens:
        f = WN.window_mean(PR, m, months)
        if np.all(np.isnan(f)):
            continue
        v = rmean(f / sdc, msk) if method == "z" else rmean(f, msk)
        per.append(v)
        pct.append(100 * rmean(f, msk) / rmean(clim, msk))
    per = np.array(per)
    return {"n": len(per), "agree": int(np.sum(np.sign(per) == sign)), "mmm_pct": round(float(np.mean(pct)), 1),
            "members": [{"model": m, "pct": round(p, 1)} for m, p in zip([x for x in ens], pct)], "cells": int(msk.sum())}


def test_typ(msk, months, kind, sign, method):
    """As test(), but against a typical neutral year (GPCP, windows.window_typical) instead of the model's own
    climatology: absolute forecast = GPCP 1991–2020 window normal + model anomaly; compared with the typical level."""
    ens = WN.ensemble(kind)
    clim, sd = WN.window_obs(months)
    typ, _ = WN.window_typical(months)
    sdc = np.clip(sd, 0.05, None)
    m_ok = msk & np.isfinite(typ)
    per, pct = [], []
    for m in ens:
        f = WN.window_mean(PR, m, months)
        if np.all(np.isnan(f)):
            continue
        absf = clim + f
        per.append(rmean((absf - typ) / sdc, m_ok) if method == "z" else rmean(absf, m_ok) - rmean(typ, m_ok))
        pct.append(100 * rmean(absf, m_ok) / rmean(typ, m_ok) - 100)
    per = np.array(per)
    return {"n": len(per), "agree": int(np.sum(np.sign(per) == sign)), "mmm_pct": round(float(np.mean(pct)), 1),
            "members": [{"model": m, "pct": round(p, 1)} for m, p in zip(ens, pct)],
            "typ_pct_of_normal": round(100 * rmean(typ, m_ok) / rmean(clim, m_ok), 1)}


# ---------------------------------------------------------------- step 1: reproduce the old tables
# Consistency check: with each region's 3-month season, the window machinery must reproduce model_counts.csv and
# region_table.csv exactly. Those tables are regenerated per initialization (model_counts.py, regions.py); the check
# only runs when they carry the same init as the derived fields, otherwise it is reported as skipped.
mc = pd.read_csv(ROOT / "hit_rates" / "derived" / "model_counts.csv")
rt = pd.read_csv(HERE / "derived" / "region_table.csv")
mcsrc = (ROOT / "hit_rates" / "model_counts.py").read_text()
REGMC = eval(mcsrc[mcsrc.index("REG = {") + 6:mcsrc.index("}\nrows = []") + 1])
tables_init = mc["init"].iloc[0] if "init" in mc.columns else "Sep 2026"
if tables_init != INIT_LABEL:
    print(f"step 1 skipped: reference tables are from {tables_init}, derived fields from {INIT_LABEL} "
          "(run hit_rates/model_counts.py and regions.py for this init first)")
else:
    bad = []
    for r, (sign, s, var) in REGMC.items():
        if var != "pr":
            continue
        s = s if s in _SEASONS else mc[mc.region == r].season.iloc[0]
        kind = "all" if len(ens_for(s)) > 6 else "nmme"
        t = test(poly_mask(r), season_abs(s), kind, sign, "raw", False)
        ref = mc[mc.region == r].iloc[0]
        if (t["agree"], t["n"], t["cells"]) != (ref.agree, ref.n, ref.ncells):
            bad.append(("poly", r, s, (t["agree"], t["n"], t["cells"]), (ref.agree, ref.n, ref.ncells)))
    nrt = 0
    for reg, var, s, sign, _src_ in R["TESTS"]:
        if var != "pr":
            continue
        kind = "all" if len(ens_for(s)) > 6 else "nmme"
        months = season_abs(s)
        clim, sd = WN.window_obs(months)
        assert np.allclose(clim, o[f"pr_clim_{s}"].values, equal_nan=True, atol=1e-4), f"clim {s}"
        assert np.allclose(sd, o[f"pr_sd_{s}"].values, equal_nan=True, atol=1e-4), f"sd {s}"
        t = test(R["mask_for"](reg), months, kind, sign, "z", False)
        ref = rt[(rt.region == reg) & (rt["var"] == "pr") & (rt.season == s)].iloc[0]
        nrt += 1
        if (f"{t['agree']}/{t['n']}", t["cells"]) != (ref.agree, ref.cells):
            bad.append(("box", reg, s, (t["agree"], t["n"], t["cells"]), (ref.agree, ref.cells)))
    print(f"step 1: reproduced {sum(v[2] == 'pr' for v in REGMC.values())} polygon counts (model_counts.csv) and {nrt} box tests "
          f"(region_table.csv, incl. clim + SD fields to 1e-4): {'ALL MATCH' if not bad else bad}")
    assert not bad

# ---------------------------------------------------------------- step 2: full windows
G = {"__file__": str(ROOT / "figures" / "make_hit_grid.py")}
gsrc = (ROOT / "figures" / "make_hit_grid.py").read_text()
exec(gsrc[:gsrc.index("main = sm[sm.main]")], G)
hs = pd.read_csv(ROOT / "hit_rates" / "derived" / "hit_summary.csv")
hs = hs[hs.main].set_index("region")
rows = []
for g, key, lab, exp, spec in G["ROWS"]:
    code = hs.loc[key].season
    sign = int(hs.loc[key].sign)
    temp = key in G["TEMP"]
    old, _ = G["model_txt"](key, spec)
    months, kind = WN.model_window(code)
    row = {"group": g, "key": key, "label": lab, "obs_window": code, "old_models": old, "window": WN.label(months),
           "ensemble": kind or "", "test": "", "agree": None, "n": None, "alt_agree": None, "mmm_pct": None, "poly_agree": None,
           "poly_n": None, "poly_mmm_pct": None, "members": "", "box_members": "", "box_cells": None,
           "typ_agree": None, "typ_alt_agree": None, "typ_mmm_pct": None, "typ_box_members": "", "typ_poly_agree": None,
           "typ_poly_mmm_pct": None, "typ_members": "", "typ_level": None}
    if temp:
        row.update(test="unchanged (temperature)", window="", ensemble="")
        rows.append(row); continue
    if kind is None:
        row["test"] = "beyond range"; rows.append(row); continue
    tt = None
    if isinstance(spec, tuple):
        bx = R["mask_for"](spec[0]); tt = test_typ(bx, months, kind, sign, "z"); tta = test_typ(bx, months, kind, sign, "raw")
        row.update(typ_agree=tt["agree"], typ_alt_agree=tta["agree"], typ_mmm_pct=tta["mmm_pct"], typ_box_members=json.dumps(tta["members"]))
    elif key in SHAPES:
        tt = test_typ(poly_mask(key), months, kind, sign, "raw")
        row.update(typ_agree=tt["agree"], typ_mmm_pct=tt["mmm_pct"])
    if key in SHAPES:
        tp = test_typ(poly_mask(key), months, kind, sign, "raw")
        row.update(typ_poly_agree=tp["agree"], typ_poly_mmm_pct=tp["mmm_pct"], typ_members=json.dumps(tp["members"]),
                   typ_level=tp["typ_pct_of_normal"])
    if isinstance(spec, tuple):                       # pre-registered literature region / candidate box
        box = R["mask_for"](spec[0])
        t = test(box, months, kind, sign, "z"); alt = test(box, months, kind, sign, "raw")
        row.update(test=f"box:{spec[0]}", agree=t["agree"], n=t["n"], alt_agree=alt["agree"], mmm_pct=alt["mmm_pct"],
                   box_members=json.dumps(alt["members"]), box_cells=alt["cells"])
    elif key in SHAPES:
        pm = poly_mask(key)
        t = test(pm, months, kind, sign, "raw"); alt = test(pm, months, kind, sign, "z")
        row.update(test="poly" + (" †" if spec == "dagger" else ""), agree=t["agree"], n=t["n"], alt_agree=alt["agree"], mmm_pct=t["mmm_pct"])
    if key in SHAPES:                                  # the dashboard tile values: final polygon, raw
        p = test(poly_mask(key), months, kind, sign, "raw")
        row.update(poly_agree=p["agree"], poly_n=p["n"], poly_mmm_pct=p["mmm_pct"], members=json.dumps(p["members"]))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(HERE / "derived" / "window_tests.csv", index=False)
pd.set_option("display.width", 250)
print(df[["key", "window", "ensemble", "test", "n", "agree", "typ_agree", "mmm_pct", "typ_mmm_pct", "poly_agree", "typ_poly_agree", "poly_mmm_pct", "typ_poly_mmm_pct", "typ_level"]].to_string(index=False))
