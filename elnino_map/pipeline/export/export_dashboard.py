#!/usr/bin/env python3.13
"""Export the data behind the Climate Dashboard's interactive El Niño impacts tab.

Reads what tools/export_data.py already produced for the video (data/*.json), plus the per-model seasonal
forecast fields, and writes a small static bundle into the dashboard repo:

  <dashboard>/elnino_map/data/regions.json   26 map regions + extras: polygons, observed record, model values
  <dashboard>/elnino_map/data/geo.json       land / borders / states (same as the video)
  <dashboard>/elnino_map/data/meta.json      forecast init, model list per season, headline stats
  <dashboard>/elnino_map/data/lit.json       per-region literature text (data/region_lit.json)
  <dashboard>/elnino_map/data/grid_<S>.bin   Int16 LE [n_models + 1, 181, 360]: each model's precipitation anomaly as
                                             % of normal (GPCP 1991–2020 climatology, as in the video), then the
                                             multi-model mean. lat 90..-90, lon -180..179, 1°. -32768 = no value
                                             (ocean-free desert cells with < 0.5 mm/day normal, same mask as the
                                             video's field). Whole percent, no clipping (dry-zone cells reach several thousand %;
                                             an earlier Int8 version clipped at +126%).
  <dashboard>/elnino_map/img/sst_DJF.png     NMME Dec–Feb SST anomaly (downsampled copy of the video's layer)

The map field, the stippling and the click-anywhere model readout are all computed in the browser from grid_*.bin,
so the picture and the numbers cannot drift apart. Region model counts are asserted against regions.json.

usage: python3.13 tools/export_dashboard.py [--out "/path/to/Climate Dashboard/elnino_map"]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

VIDEO = Path(__file__).resolve().parent        # export/ (the trimmed export_data.py writes data/ and img/ here)
ROOT = VIDEO.parent                            # pipeline root
sys.path.insert(0, str(ROOT / "model_patterns"))
from initcfg import INIT_LABEL, SEASONS as CFG_SEASONS, MAP_SEASONS, NMME as CFG_NMME, season_abs_months, season_label, ens_for  # noqa: E402
def _default_out():
    return ROOT.parent                         # elnino_map/ (the pipeline lives inside the map bundle)


OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else _default_out()
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)
(OUT / "img").mkdir(exist_ok=True)

NMME = ["CFSv2", "CanESM5", "GEM5.2_NEMO", "NASA_GEOS5v2", "NCAR_CCSM4", "NCAR_CESM1"]
C3S = ["ecmwf", "ukmo", "meteo_france", "dwd", "cmcc", "jma", "bom"]
LABEL = {"CFSv2": "NOAA CFSv2", "CanESM5": "Canada CanESM5", "GEM5.2_NEMO": "Canada GEM5.2", "NASA_GEOS5v2": "NASA GEOS",
         "NCAR_CCSM4": "NCAR CCSM4", "NCAR_CESM1": "NCAR CESM1", "ecmwf": "ECMWF", "ukmo": "UK Met Office",
         "meteo_france": "Météo-France", "dwd": "DWD Germany", "cmcc": "CMCC Italy", "jma": "JMA Japan", "bom": "BoM Australia"}
SEASONS = {s: season_label(s) for s in MAP_SEASONS}        # 3-month seasons the forecast covers, in order
ens = ens_for

a = xr.open_dataset(ROOT / "model_patterns" / "derived" / "seasonal_anoms.nc")
o = xr.open_dataset(ROOT / "model_patterns" / "derived" / "obs_ref.nc")
assert np.allclose(a.lat.values, np.arange(90, -91, -1)) and np.allclose(a.lon.values, np.arange(360))
land = o.land.values.astype(bool)
init = a.attrs["note"].split(" init")[0]            # "Oct 2026"
assert len(init.split()) == 2 and init == INIT_LABEL, (init, INIT_LABEL)

regions = json.loads((VIDEO / "data" / "regions.json").read_text())
stats = json.loads((VIDEO / "data" / "stats.json").read_text())

seasons = {}
# BASELINE=typical (env): express everything relative to a typical neutral year instead of the 1991–2020 normal.
# Map field: absolute forecast (GPCP 1991–2020 normal + model anomaly) as % of the GPCP typical neutral year for this
# event (model_patterns/windows.window_typical). Region model checks: window_tests.csv typ_* columns (same GPCP basis).
import os
BASELINE = os.environ.get("BASELINE", "normal")      # map shading, point readout, model tiles
assert BASELINE in ("normal", "typical"), BASELINE
# Past-event bars: % of a typical neutral year (the same reference as the x/8 checks, GPCC 1951–2023), with the
# 1991–2020 average marked. Decided 27 Sep 2026 after previewing BASELINE=typical: rebasing the map/models on a GPCP
# typical neutral year mostly shows the baseline (61% of the map shaded >=10% with a zero forecast), is noisy
# (13 GPCP neutral years), and GPCC/GPCP typical levels differ by >=10 points in a third of regions.
BARS = os.environ.get("BARS", "typical")
assert BARS in ("normal", "typical"), BARS
sys.path.insert(0, str(ROOT / "model_patterns"))
import windows as WN   # noqa: E402
SEAS_ABS = {s: [m - WN.INIT_YEAR * 12 for m in season_abs_months(s)] for s in MAP_SEASONS}   # month index from Jan of the init year
for s, label in SEASONS.items():
    ms = ens(s)
    arr = np.array([a.prate.sel(model=m, season=s).values for m in ms])      # mm/day anomaly
    cl = o[f"pr_clim_{s}"].values
    bad = ((cl < 0.5) & land) | ~(cl > 0)
    if BASELINE == "typical":
        typ, _ = WN.window_typical([WN.INIT_YEAR * 12 + k for k in SEAS_ABS[s]])
        bad |= ~np.isfinite(typ)
        pct = 100 * (cl + arr) / typ - 100
        mm = 100 * (cl + arr.mean(0)) / typ - 100
    else:
        pct = 100 * arr / np.where(cl > 0, cl, np.nan)
        mm = 100 * arr.mean(0) / np.where(cl > 0, cl, np.nan)                # same field as the video (pr_<S>.png)
    stack = np.concatenate([pct, mm[None]], 0)
    stack = np.where(bad[None] | ~np.isfinite(stack), np.nan, stack)
    stack = np.roll(stack, 180, axis=-1)                                     # lon 0..359 -> -180..179
    q = np.where(np.isfinite(stack), np.clip(np.round(stack), -100, 32000), -32768).astype("<i2")
    assert np.nanmax(np.where(np.isfinite(stack), stack, np.nan)) < 32000, "value beyond Int16 range"
    q.tofile(DATA / f"grid_{s}.bin")
    seasons[s] = {"label": label, "n": len(ms), "models": [{"id": m, "label": LABEL[m]} for m in ms],
                  "win": [SEAS_ABS[s][0], len(SEAS_ABS[s])], "sub": label.split(" ", 1)[1] if "–" in label.split(" ")[0] else label,
                  "source": "NMME + Copernicus C3S" if len(ms) > len(CFG_NMME) else f"NMME (C3S runs end in {WN.label([WN.ALL_END]).split(' ')[0]})"}
    print(f"grid_{s}.bin  {q.shape}  {q.nbytes / 1e3:.0f} kB")

# Model checks over each region's full window (model_patterns/windows.py + window_tests.py, 27 Sep 2026): the
# observed window cut at the last month all 13 systems cover (Feb for a Sep start, Mar for Oct), or the 6 NMME
# models for windows beyond it. Counts follow the blog's hit grid. Where the grid uses a test region fixed before
# the map was drawn and its count differs from the drawn polygon's, the dashboard shows that test region instead
# (southern China: 12/13 on its box vs 11/13 on the polygon), so the tab never disagrees with the post.
import pandas as pd
WT = pd.read_csv(ROOT / "model_patterns" / "derived" / "window_tests.csv").set_index("key")
BOXNOTE = {"schina": "Models tested on a box fixed before the map was drawn (105–122°E, 22–30°N)."}
changed = []
for key, R in {**regions["regions"], **regions["extras"]}.items():
    if key not in WT.index or not R.get("models"):
        continue
    r = WT.loc[key]
    if r.test == "unchanged (temperature)":
        continue
    old = R["models"]
    if r.test == "beyond range":
        R["models"] = None; changed.append(f"{key}: beyond range"); continue
    months, kind = WN.model_window(r.obs_window)
    T = BASELINE == "typical"
    c_agree, c_alt, c_poly = ("typ_agree", "typ_alt_agree", "typ_poly_agree") if T else ("agree", "alt_agree", "poly_agree")
    use_box = str(r.test).startswith("box:") and not pd.isna(r[c_poly]) and int(r[c_agree]) != int(r[c_poly])
    mem = json.loads((r.typ_box_members if use_box else r.typ_members) if T else (r.box_members if use_box else r.members))
    sign = old["sign"]
    for x in mem:
        x["label"] = LABEL[x["model"]]; x["agree"] = bool(np.sign(x["pct"]) == sign)
    agree = sum(x["agree"] for x in mem)
    if use_box:
        if agree != int(r[c_agree]):
            print(f"  NOTE {key}: headline (standardized) count {int(r[c_agree])} vs tiles (raw) {agree}; showing the tiles' count")
        if key not in BOXNOTE:
            print(f"  NOTE {key}: box and polygon counts differ ({int(r[c_agree])} vs {int(r[c_poly])}); showing the polygon")
            use_box = False; mem = json.loads(r.typ_members if T else r.members)
            for x in mem:
                x["label"] = LABEL[x["model"]]; x["agree"] = bool(np.sign(x["pct"]) == sign)
            agree = sum(x["agree"] for x in mem)
    else:
        assert agree == int(r[c_poly]), (key, agree, r[c_poly])
    seas = next((sk for sk, sm in SEAS_ABS.items() if [m - WN.INIT_YEAR * 12 for m in months] == sm), None)
    R["models"] = {"season": seas, "window": WN.label(months), "ens": kind, "sign": sign, "n": len(mem), "agree": agree,
                   "members": mem, "mmm_pct": round(float(np.mean([x["pct"] for x in mem])), 1),
                   **({"note": BOXNOTE[key]} if use_box else {})}
    if (old["agree"], old["n"], old["season"]) != (agree, len(mem), seas) or abs(old["mmm_pct"] - R["models"]["mmm_pct"]) >= 1:
        changed.append(f"{key}: {old['season']} {old['agree']}/{old['n']} {old['mmm_pct']:+.0f}% -> {WN.label(months)} {agree}/{len(mem)} {R['models']['mmm_pct']:+.0f}%")
print("model checks now over full windows; changed vs the 3-month version:\n  " + "\n  ".join(changed))

# regions: keep the published map's tier colours
for k, R in {**regions["regions"], **regions["extras"]}.items():
    R["title"] = R["title"].replace("\n", " ")
    if "detail" in R:
        R["detail"] = R["detail"].replace("\n", " · ")
    M = R.get("models")
    if M:
        assert M["agree"] == sum(m["agree"] for m in M["members"]) and M["n"] == len(M["members"]), k
json.dump(regions, open(DATA / "regions.json", "w"), separators=(",", ":"), ensure_ascii=False)
shutil.copy(VIDEO / "data" / "geo.json", DATA / "geo.json")

lit = VIDEO / "data" / "region_lit.json"
if lit.exists():
    L = json.loads(lit.read_text())
    missing = [k for k in regions["regions"] if k not in L]
    assert not missing, f"region_lit.json lacks {missing}"
    shutil.copy(lit, DATA / "lit.json")
else:
    print("WARNING: data/region_lit.json not found; lit.json not written")

meta = {
    "init": init,
    "seasons": seasons,
    "strong_events": sorted(int(y) for y in stats["oni_ndj_strong"]),
    "oni_ndj_strong": stats["oni_ndj_strong"],
    "by_event": stats["by_event"],
    "aggregate": {"hits": stats["aggregate_hits"], "n": stats["aggregate_n"]},
    "lanina_same": stats["lanina_same"],
    "moderate_hits": stats["moderate_hits"],
    "p2027_warmest": stats["p2027_warmest"],
    "n_map_regions": stats["n_map_regions"],
    "obs_source": "GPCC Full Data Monthly v2025 (1° land), GPCP / station composites for islands",
    "model_horizon": {"all": WN.label([WN.ALL_END]), "nmme": WN.label([WN.NMME_END])},
    "n_all": max(v["n"] for v in seasons.values()), "n_nmme": len(CFG_NMME),
    "baseline": BASELINE,
    "bars": BARS,
}
json.dump(meta, open(DATA / "meta.json", "w"), indent=1, ensure_ascii=False)

im = Image.open(VIDEO / "img" / "sst_DJF.png")
im.resize((1440, 720), Image.LANCZOS).save(OUT / "img" / "sst_DJF.png", optimize=True)
print(f"wrote {OUT}  (init {init}; {len(regions['regions'])} regions, {len(regions['extras'])} extras)")
