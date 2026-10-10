#!/usr/bin/env python3.13
"""El Niño 2026-27 expected regional impacts map.

Fill colour = literature confidence tier (research/impacts-literature.md plus the
research/lit_*.md searches). Do not edit tiers here without updating the dossier.
Shapes marked "model" come from model_patterns/derived/model_polygons.json: where
>=80% of 13 dynamical seasonal forecast models (NMME + C3S, Sep 2026 initialization)
agree on the literature-expected sign with |z| >= 0.5 (see model_patterns/METHODS.md).
Hand-drawn shapes remain only where models are out of range (Yangtze, N/Central India:
summer 2027) or cannot resolve the region (Peru coast; redrawn to the coastal plain in v5.1).
Palette validated for CVD separation 2026-08-13 (see figures/NOTES.md).

v5 (23 Sep 2026): model-derived shapes; region edits approved by the author
(model_patterns/REGION_REVIEW.md): Ohio Valley removed; S India hatched (forecasts
disagree); Caribbean split; S California/US Southwest/N Mexico added; N South America,
Philippines, S China, Hawaii added; season windows revised.
v5.1 (23 Sep 2026): Pacific NW -> model-derived interior BC/N Rockies/inland NW (dry, low
snowpack; research/lit_canada_nw.md); Central Canada warm added (trimmed at ~79.5°W per
literature); E Australia -> SE Australia (Sep–Nov); S India/Sri Lanka -> Sri Lanka; Peru
strip tightened to the coastal plain. No 'disagree' regions remain (hatch code retained).
v5.2 (23 Sep 2026): S California/US Southwest/N Mexico medium -> medium-high, ⚠ kept.
v5.3 (23 Sep 2026): Peru/Ecuador coast medium ⚠ -> high, no flag, Dec–May -> Dec–Apr.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, shape

HERE = Path(__file__).resolve().parent
MODEL = {k: shape(v) for k, v in json.loads(
    (HERE.parent / "model_patterns" / "derived" / "model_polygons.json").read_text()).items()}

# confidence -> fill (drought ramp / flood ramp), dark = high
DRY = {"high": "#8C510A", "medhigh": "#C4791C", "medium": "#F4C154"}
WET = {"high": "#0A5599", "medhigh": "#3F8FC5", "medium": "#7FC4EA"}
WARM = {"medium": "#D6604D"}  # CVD-checked vs palette: worst ΔE 18.7 (deutan vs dry-high)
EDGE = {"dry": "#5C3406", "wet": "#063763", "warm": "#7F2A1F"}

LAT_S, LAT_N = -47, 62

# 2027 record odds (kept outside the REGIONS ... FOOTER span, which other scripts exec) from the operational GMST forecast (separate workflow: 6-dataset blend, 14-model
# ENSO plume, Sep 2026 run); P(2027 ranks warmest) on the ONI and RONI conventions, shown as a range
import pandas as _pd
_lp = HERE.parent / "local_paths.json"   # local-only (git-ignored): where the forecast run lives on this machine
_fcf = (Path(json.loads(_lp.read_text())["forecast_dir"]) if _lp.exists() else HERE / "_none") / "rank_probabilities_wide.csv"
if _fcf.exists():
    _rk = _pd.read_csv(_fcf).set_index(["approach", "year"])
    _p27 = sorted(round(100 * float(_rk.loc[(a_, 2027), "P(rank=1)"])) for a_ in ["ONI", "RONI"])
else:   # public repo: the extract written by dashboard/tools/export_data.py
    _p27 = sorted(round(100 * v) for v in json.loads((HERE.parent / "data" / "forecast_2027_extract.json").read_text())["p2027_rank1"].values())
assert min(_p27) >= 90, "\"almost certain\" needs P >= 90%; update the wording"
_p27s = f"{_p27[0]}%" if _p27[0] == _p27[1] else f"{_p27[0]}–{_p27[1]}%"

# (key, kind, conf, flag, shape, label lon/lat, title, detail, leader target or None, align)
# shape: "model" (key in MODEL) or list of hand-drawn (lon, lat) points
# flag: None | "busted" (dashed outline: underperformed in a recent strong event; ⚠ glyph dropped in v6.3)
#       | "disagree" (hatched: this year's forecasts disagree with the literature)
REGIONS = [
    ("maritime", "dry", "high", None, "model",
     (108, -25), "Indonesia &\nMaritime Continent", "drought + fire · Sep–Dec", None, "c"),
    ("philippines", "dry", "medhigh", None, "model",
     (139, 16), "Philippines", "drought · Dec–Apr", None, "l"),
    ("safrica", "dry", "high", None, "model",
     (48, -33), "Southern Africa", "drought · Dec–Feb\nmaize / food risk", None, "l"),
    ("amazon", "dry", "high", None, "model",
     (-92, -5), "Amazon", "drought + fire\nnow–May", None, "r"),
    ("nsam", "dry", "medhigh", None, "model",
     (-46, 15), "Northern South America", "drought · Dec–Feb", None, "l"),
    ("nebrazil", "dry", "medhigh", "busted", "model",  # v6.2: flag added; Mar-May 2024 near normal (100%)
     (-27, -10), "NE Brazil", "drought\nMar–May 2027", None, "l"),
    ("drycorridor", "dry", "medhigh", "busted", "model",  # v6.2: flag added; Dec-Feb 2015-16 at the neutral median
     (-100, 13), "Central America &\nS Caribbean", "drought · Dec–Feb", None, "r"),
    ("wpacific", "dry", "medhigh", None, "model",
     (178, -32), "W Pacific islands", "drought · sea-level fall", None, "c"),
    ("hawaii", "dry", "medhigh", None, "model",
     (-160, 28), "Hawaii", "drought · Nov–Mar", None, "c"),
    ("seaustralia", "dry", "medium", None, "model",  # v6.2: flag removed; Sep-Nov verified dry in 2015 and 2023
     (134, -42), "SE Australia", "drought, fire risk · Sep–Nov", None, "r"),
    ("inlandnw", "dry", "medium", None, "model",
     (-163, 54), "Interior BC, N Rockies\n& inland Northwest", "dry, low snowpack · Dec–Feb", (-117, 51), "c"),
    ("ccanada_warm", "warm", "medium", None, "model",
     (-72, 57), "Central Canada", "warm winter · Dec–Feb", None, "l"),
    # v6 (24 Sep 2026): N & Central India removed -- decay-year monsoon dry in only 1 of 8 strong events
    # (GPCC v2025; hit_rates/METHODS.md). Southern Mekong added at medium (7/8 dry, Mar-May yr+1).
    ("mekong", "dry", "medium", None,
     [(102.6, 13.6), (104.2, 15.6), (106.6, 15.9), (108.3, 14.4), (109.0, 11.6), (107.0, 10.3),
      (105.1, 8.7), (104.4, 10.4), (102.9, 11.4)],
     (79, 19), "Southern Mekong", "drought · Mar–May 2027", None, "c"),
    ("horn", "wet", "medium", None, "model",
     (31, 29), "Horn of Africa", "floods, heavy short rains · Oct–Dec\n(size set by Indian Ocean Dipole)", None, "c"),
    ("peru", "wet", "high", None,  # v5.3: Niño 1+2 +4.6 °C mid-Sep, NMME Jan–Mar +1.9..+5.1 (NOTES)
     [(-80.3, 1.2), (-79.4, -1.0), (-79.3, -3.5), (-79.4, -6.0), (-78.2, -9.0), (-76.7, -12.0),
      (-77.3, -12.6), (-79.3, -9.2), (-81.1, -6.2), (-81.2, -4.2), (-80.8, -2.2), (-80.9, 0.2)],
     (-88.5, -16.5), "Peru / Ecuador coast", "flooding · Dec–Apr", (-79, -6), "r"),
    # v6.4 (24 Sep 2026, suggestion from a Chilean colleague; research/lit_chile_altiplano.md):
    # Altiplano: model-robust across the plateau (DJF 13/13, 93% of cells robust); outline hand-traced along the
    # plateau and trimmed south of Lake Titicaca, where the literature finds the signal weak or reversed.
    ("altiplano", "dry", "medium", None,  # obs 5/8 on this outline (6/8 box), lit medium -> medium
     [(-70.6, -16.0), (-68.6, -16.0), (-67.6, -17.2), (-66.7, -18.6), (-66.3, -20.2), (-66.7, -21.9),
      (-67.9, -22.4), (-68.9, -21.6), (-69.4, -19.6), (-70.1, -17.9), (-70.9, -16.7)],
     (-92, -27), "Altiplano", "drought · Dec–Mar", None, "r"),
    # Central Chile 35-38S: models OND 13/13, 100% of cells robust; outline = Chile between coast and Andes crest.
    ("cchile", "wet", "medium", None,
     [(-72.2, -35.0), (-70.4, -35.0), (-70.8, -36.5), (-71.1, -38.0), (-73.5, -38.0), (-73.6, -37.2), (-72.8, -36.0)],
     (-92, -38.5), "Central Chile", "wet · Oct–Nov 2026", None, "r"),
    ("sesa", "wet", "high", None, "model",
     (-36, -41), "SE South America", "heavy rain, floods\nSep–Feb", None, "l"),
    ("gulf", "wet", "high", None, "model",
     (-66, 39), "US Gulf Coast & Southeast", "wet, stormy · Dec–Feb", None, "l"),
    ("antilles", "wet", "medhigh", None, "model",
     (-64, 26), "Cuba & Bahamas", "wet · Dec–Feb", None, "l"),
    ("swus", "wet", "medhigh", "busted", "model",  # upgraded 23 Sep 2026: 6/8 strong winters wet (NOTES)
     (-132, 41), "S California, US Southwest\n& N Mexico", "wet · Dec–Mar", None, "r"),
    # v6.1 (24 Sep 2026): N California added at medium. Shape = the pre-registered model-test box
    # (124.5-120W, 38-42N; 13/13 models wet, robust fraction 1.0) clipped to California. Obs 6/8 (GPCC).
    ("norcal", "wet", "medium", None,
     [(-120.0, 38.0), (-123.0, 38.0), (-122.9, 38.2), (-123.7, 38.9), (-123.8, 39.8), (-124.4, 40.4),
      (-124.1, 41.4), (-124.2, 42.0), (-120.0, 42.0)],
     (-96, 41.9), "N California", "wet · Dec–Mar", None, "c"),
    ("srilanka", "wet", "medium", None, "model",
     (71, -9), "Sri Lanka", "heavy NE monsoon · Oct–Dec", None, "c"),
    ("schina", "wet", "medhigh", None, "model",
     (140, 34), "Southern China", "wet · Dec–May", None, "l"),
    ("yangtze", "wet", "medium", "busted",  # v6: med-high -> medium (obs 4/8 Jun-Aug, 6/8 Jun-Jul)
     [(104, 28), (112, 30), (120, 32), (122, 30), (118, 27), (108, 26)],
     (127, 47), "Yangtze basin", "flood risk\nJun–Jul 2027", (114, 29), "c"),
    ("cpacific", "wet", "medhigh", None, "model",
     (205, -16), "Central Pacific islands", "wet, inundation · Sep–May", None, "c"),
]

FOOTER = (f"GLOBAL:  2027 almost certain to be the warmest year on record ({_p27s})  ·  CO₂ growth well above trend  ·  "
          "widespread coral bleaching risk  ·  correlated multi-breadbasket crop risk")

INK, SUB = "#1A1A1A", "#4A4A4A"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["hatch.linewidth"] = 0.8

fig = plt.figure(figsize=(15, 7.7), facecolor="white")
proj = ccrs.Robinson(central_longitude=180)
pc = ccrs.PlateCarree()
ax = fig.add_axes([0.01, 0.17, 0.98, 0.67], projection=proj)
ax.set_global()
x0, x1 = ax.get_xlim()
ax.set_ylim(proj.transform_point(180, LAT_S, pc)[1], proj.transform_point(180, LAT_N, pc)[1])
ax.set_xlim(x0, x1)
ax.add_feature(cfeature.LAND, facecolor="#E9E7E1", edgecolor="none")
ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#B7B3AA")
ax.spines["geo"].set_visible(False)


def xy(lon, lat):
    return proj.transform_point(lon, lat, pc)


for key, kind, conf, flag, shp, lxy, title, detail, leader, align in REGIONS:
    if shp == "model":
        poly = MODEL[key]
    else:
        buf = {"peru": 0.8, "mekong": 0.8, "norcal": 0.3, "altiplano": 0.3, "cchile": 0.2}.get(key, 2.0)
        poly = Polygon(shp).buffer(buf, join_style=1).simplify(0.25)
    if leader is None:
        rp = poly.representative_point()
        leader = (rp.x, rp.y)
    fill = {"dry": DRY, "wet": WET, "warm": WARM}[kind][conf]
    ax.add_geometries([poly], crs=pc, facecolor=fill, alpha=0.88,
                      edgecolor=EDGE[kind], linewidth=1.4 if flag == "busted" else 0.7,
                      linestyle=(0, (4, 2)) if flag == "busted" else "solid", zorder=3)
    if flag == "disagree":
        ax.add_geometries([poly], crs=pc, facecolor="none", edgecolor="#333333",
                          hatch="////", linewidth=0, zorder=3.5)
    ax.plot(*xy(*leader), marker="o", ms=3.2, color="#333333", mec="white", mew=0.6, zorder=6)
    tx_align = {"l": "left", "r": "right", "c": "center"}[align]
    box = VPacker(children=[
        TextArea(title, textprops=dict(fontsize=10, fontweight="bold", color=INK, ha=tx_align)),
        TextArea(detail, textprops=dict(fontsize=9, color=SUB, linespacing=1.2, ha=tx_align)),
    ], align={"l": "left", "r": "right", "c": "center"}[align], pad=0, sep=1.5)
    ab = AnnotationBbox(
        box, xy(*leader), xybox=xy(*lxy), xycoords="data", boxcoords="data",
        box_alignment={"l": (0, 0.5), "r": (1, 0.5), "c": (0.5, 0.5)}[align],
        pad=0.4, frameon=True, zorder=7,
        bboxprops=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D0CCC4",
                       linewidth=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#6E6E6E", linewidth=0.8, shrinkA=0, shrinkB=2.5))
    ax.add_artist(ab)

fig.text(0.03, 0.935, "El Niño 2026–27: where the biggest impacts are expected",
         fontsize=21, fontweight="bold", color=INK)
fig.text(0.03, 0.912, "Shading = scientific confidence from the teleconnection literature and how often each signal appeared in the 8 strong El Niños since 1957.\n"
         "Shapes show where ≥80% of 13 seasonal forecast models agree (NMME + C3S, Sep 2026 start); regions beyond forecast range are indicative.",
         fontsize=11.5, color="#555555", linespacing=1.35, va="top")
fig.text(0.985, 0.95, "Confidence assessment: Z. Hausfather · basemap: Natural Earth",
         fontsize=8, color="#999999", ha="right")

# legend: entirely below the map axes (map bottom = 0.17)
lx, ly, sw, sh, gx = 0.035, 0.081, 0.018, 0.025, 0.062
cx0 = lx + 0.10
for j, lab in enumerate(["High", "Med-high", "Medium"]):
    fig.text(cx0 + j * gx, ly + 2.05 * sh, lab, fontsize=9, color="#555555", ha="center")
fig.text(cx0 + 1 * gx, ly + 2.85 * sh, "Confidence", fontsize=9, fontweight="bold",
         color="#555555", ha="center")
for i, (ramp, rowlab) in enumerate([(DRY, "Drought / dry"), (WET, "Flood / wet")]):
    y = ly + (1 - i) * sh
    fig.text(lx, y + 0.009, rowlab, fontsize=9.5, color="#222222", ha="left", va="center")
    for j, conf in enumerate(["high", "medhigh", "medium"]):
        fig.patches.append(mpatches.FancyBboxPatch(
            (cx0 + j * gx - sw / 2, y), sw, sh * 0.72,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor=ramp[conf], edgecolor="none", transform=fig.transFigure))
fx = cx0 + 2.75 * gx
fig.patches.append(mpatches.Rectangle(
    (fx, ly + 1.15 * sh), sw * 1.2, sh * 0.7, facecolor="#F4C154", edgecolor=EDGE["dry"],
    linewidth=1.3, linestyle=(0, (3, 1.6)), transform=fig.transFigure))
fig.text(fx + sw * 1.2 + 0.008, ly + 1.5 * sh,
         "dashed outline = signal underperformed in a recent strong event",
         fontsize=9, color="#555555", va="center")
fig.patches.append(mpatches.FancyBboxPatch(
    (fx, ly + 0.05 * sh), sw * 1.2, sh * 0.7, boxstyle="round,pad=0.001,rounding_size=0.004",
    facecolor=WARM["medium"], edgecolor="none", transform=fig.transFigure))
fig.text(fx + sw * 1.2 + 0.008, ly + 0.4 * sh,
         "warm winter (temperature signal, medium confidence)",
         fontsize=9, color="#555555", va="center")
fig.text(fx + 0.33, ly + 0.95 * sh, "All impacts are shifts in probability,\nnot certainties",
         fontsize=9, color="#555555", va="center", linespacing=1.4)

fig.text(0.5, 0.018, FOOTER, fontsize=10.2, color="#333333", ha="center",
         bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F2EC", edgecolor="#DDD9D0"))

out = HERE / "elnino_2026_impacts_map"
fig.savefig(str(out) + ".png", dpi=200)
fig.savefig(str(out) + ".pdf")
print("written", str(out) + ".png")
