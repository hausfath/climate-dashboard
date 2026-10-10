#!/usr/bin/env python3.13
"""Hit-rate grid: how each impacts-map region behaved in the 8 strong El Niños since 1957.

Cell colour = the region's labelled season relative to ENSO-neutral years (detrended percentile,
hit_rates/METHODS.md). Right-hand columns: count of events in the expected direction, and Sep-2026
model agreement (pre-reshape tests from model_patterns/derived/region_table.csv where they exist;
† = shape drawn from this year's models, so the count is high by construction).
All numbers are read from hit_rates/derived/*.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
HR = HERE.parent / "hit_rates" / "derived"
ev = pd.read_csv(HR / "hit_events.csv")
sm = pd.read_csv(HR / "hit_summary.csv")
mc = pd.read_csv(HR / "model_counts.csv").set_index("region")
rt = pd.read_csv(HERE.parent / "model_patterns" / "derived" / "region_table.csv")

STRONG = [1957, 1965, 1972, 1982, 1997, 2009, 2015, 2023]

# (group, region key, label, expectation text, pre-reshape region_table key/season or None)
ROWS = [
    ("Asia & the Pacific", "maritime", "Indonesia & Maritime Continent", "dry · Sep–Dec", ("maritime", "SON")),
    ("Asia & the Pacific", "philippines", "Philippines", "dry · Dec–Apr", ("philippines", "DJF")),
    ("Asia & the Pacific", "mekong", "Southern Mekong", "dry · Mar–May", ("mainland_sea", "MAM")),
    ("Asia & the Pacific", "schina", "Southern China", "wet · Dec–May", ("schina", "DJF")),
    ("Asia & the Pacific", "yangtze", "Yangtze basin", "wet · Jun–Aug", None),
    ("Asia & the Pacific", "srilanka", "Sri Lanka", "wet · Oct–Dec", "dagger"),
    ("Asia & the Pacific", "seaustralia", "SE Australia", "dry · Sep–Nov", "dagger"),
    ("Asia & the Pacific", "wpacific", "W Pacific islands", "dry · Sep–Feb", ("wpacific", "SON")),
    ("Asia & the Pacific", "cpacific", "Central Pacific islands", "wet · Sep–May", ("cpacific", "SON")),
    ("Africa", "safrica", "Southern Africa", "dry · Dec–Feb", ("safrica", "DJF")),
    ("Africa", "horn", "Horn of Africa", "wet · Oct–Dec", ("horn", "OND")),
    ("South America", "amazon", "Amazon", "dry · Oct–May", ("amazon", "DJF")),
    ("South America", "nsam", "Northern South America", "dry · Dec–Feb", ("nsam", "DJF")),
    ("South America", "nebrazil", "NE Brazil", "dry · Mar–May", ("nebrazil", "MAM")),
    ("South America", "peru", "Peru / Ecuador coast ‡", "wet · Dec–Apr", ("peru", "DJF")),
    ("South America", "sesa", "SE South America", "wet · Sep–Feb", ("sesa", "DJF")),
    ("South America", "altiplano", "Altiplano", "dry · Dec–Mar", "own"),
    ("South America", "cchile", "Central Chile (35–38°S)", "wet · Oct–Nov", "own"),
    ("North & Central America", "drycorridor", "Central America & S Caribbean", "dry · Dec–Feb", ("drycorridor", "DJF")),
    ("North & Central America", "antilles", "Cuba & Bahamas", "wet · Dec–Feb", ("antilles", "DJF")),
    ("North & Central America", "gulf", "US Gulf Coast & Southeast", "wet · Dec–Feb", ("gulf", "DJF")),
    ("North & Central America", "swus", "S California, SW US & N Mexico", "wet · Dec–Mar", ("socal", "DJF")),
    ("North & Central America", "norcal", "Northern California", "wet · Dec–Mar", ("norcal", "DJF")),
    ("North & Central America", "hawaii", "Hawaii", "dry · Nov–Mar", ("hawaii", "DJF")),
    ("North & Central America", "inlandnw", "Interior BC & inland Northwest", "dry · Dec–Feb", "dagger"),
    ("North & Central America", "ccanada_warm", "Central Canada", "warm · Dec–Feb", "dagger"),
    ("Not on the map", "nindia", "N & Central India (2027 monsoon)", "dry · Jun–Sep", None),
    ("Not on the map", "c_sahel", "Sahel (2026 monsoon)", "dry · Jul–Sep", None),
    ("Not on the map", "lit_sindia", "Southern India", "wet · Oct–Dec", "own"),
    ("Not on the map", "cl_south_n", "South-central Chile (38–42°S)", "dry · Jan–Mar", "own"),
    ("Not on the map", "c_uk_ceurope", "UK & central Europe", "wet · Oct–Dec", "own"),
    ("Not on the map", "c_scandinavia", "Scandinavia", "cold · Jan–Feb", "own"),
    ("Not on the map", "lit_ohio", "Ohio Valley", "dry · Dec–Feb", "own"),
    ("Not on the map", "c_se_alaska", "SE Alaska & N BC coast", "warm · Dec–Feb", "own"),
]
TEMP = {"ccanada_warm", "c_scandinavia", "c_se_alaska"}
PR_COL = ["#8C510A", "#D9A441", "#E6E3DD", "#5FA8DA", "#0A5599"]   # drier ... wetter
T_COL = ["#9E4AA0", "#D5A6D6", "#E6E3DD", "#EA8466", "#B2182B"]    # colder (violet) ... warmer; blue is reserved for wet
INK, SUB, MUTED = "#1A1A1A", "#4A4A4A", "#8A8A8A"


def bin5(p):
    return 0 if p < 0.15 else 1 if p < 0.40 else 2 if p <= 0.60 else 3 if p <= 0.85 else 4


WT = pd.read_csv(HERE.parent / "model_patterns" / "derived" / "window_tests.csv").set_index("key")
import sys as _sys; _sys.path.insert(0, str(HERE.parent / "model_patterns")); import windows as _WN   # noqa: E402
HORIZON = _WN.label([_WN.ALL_END])


def model_txt(key, spec):
    """Model agreement over the region's full window up to the last month all 13 systems cover (Feb for a Sep start;
    6 NMME models for windows beyond it), from model_patterns/window_tests.py. Temperature rows keep their season."""
    if key in WT.index and WT.loc[key].test not in ("unchanged (temperature)",):
        r = WT.loc[key]
        if r.test == "beyond range":
            return "beyond range", False
        return f"{int(r.agree)} of {int(r.n)}" + (" †" if spec == "dagger" else ""), spec == "dagger"
    if spec is None:
        return "beyond range", False
    if spec in ("dagger", "own"):
        r = mc.loc[key]
        return f"{int(r.agree)} of {int(r.n)}" + (" †" if spec == "dagger" else ""), spec == "dagger"
    reg, s = spec
    r = rt[(rt.region == reg) & (rt.season == s) & (rt["var"] == "pr")].iloc[0]
    a, n = r.agree.split("/")
    return f"{a} of {n}", False


main = sm[sm.main].set_index("region")
plt.rcParams["font.family"] = "DejaVu Sans"
nrow = len(ROWS)
groups = list(dict.fromkeys(g for g, *_ in ROWS))
H = 0.38 * nrow + 0.67 * len(groups) + 4.4
fig = plt.figure(figsize=(12.5, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12.5)
ax.set_ylim(H, 0)
ax.axis("off")

x_lab, x_exp, x0, cw, x_cnt, x_mod = 0.35, 3.55, 5.05, 0.62, 10.35, 11.55
y = 1.95
ax.text(0.35, 0.55, "How often have El Niño's impacts actually shown up?", fontsize=19, fontweight="bold",
        color=INK, va="center")
ax.text(0.35, 1.02, "Each cell is one of the 8 strong El Niños since 1957 (Nov–Jan Oceanic Niño Index ≥ 1.5 °C), for the season "
        "on the impacts map,\ncompared with ENSO-neutral years after removing long-term trends. GPCC, Berkeley Earth "
        "and station data.", fontsize=10.5, color=SUB, va="center", linespacing=1.4)

# header
for j, yr in enumerate(STRONG):
    ax.text(x0 + (j + 0.5) * cw, y, f"{yr}–\n{str(yr + 1)[2:]}", ha="center", va="bottom", fontsize=9, color=SUB,
            linespacing=1.1)
ax.text(x_cnt, y, "Matched\nexpectation", ha="center", va="bottom", fontsize=9, color=SUB, fontweight="bold",
        linespacing=1.1)
ax.text(x_mod, y, "Models agree\n(Sep 2026)", ha="center", va="bottom", fontsize=9, color=SUB, fontweight="bold",
        linespacing=1.1)
ax.text(x_exp, y, "Expected", ha="left", va="bottom", fontsize=9, color=SUB, fontweight="bold")
y += 0.15

rh = 0.34
prev = None
for g, key, lab, exp, spec in ROWS:
    if g != prev:
        y += 0.25
        ax.text(x_lab, y + 0.2, g.upper(), fontsize=9.5, fontweight="bold", color=MUTED, va="center")
        if g == "Not on the map":
            ax.plot([x_lab, 12.2], [y - 0.02, y - 0.02], color="#CFCBC3", lw=0.8)
        y += 0.42
        prev = g
    r = main.loc[key]
    e = ev[(ev.region == key) & (ev.source == r.source) & (ev.year0.isin(STRONG))].set_index("year0")
    cols = T_COL if key in TEMP else PR_COL
    ax.text(x_lab, y + rh / 2, lab, fontsize=10, color=INK, va="center")
    ax.text(x_exp, y + rh / 2, exp, fontsize=9, color=SUB, va="center")
    for j, yr in enumerate(STRONG):
        p = e.pct.get(yr, np.nan)
        fc = "white" if not np.isfinite(p) else cols[bin5(p)]
        ax.add_patch(Rectangle((x0 + j * cw + 0.03, y + 0.03), cw - 0.06, rh - 0.06, facecolor=fc,
                               edgecolor="#D9D5CD" if not np.isfinite(p) else "none", lw=0.6))
        if not np.isfinite(p):
            ax.text(x0 + (j + 0.5) * cw, y + rh / 2, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
        elif e.hit.get(yr) is True or e.hit.get(yr) == True:
            b = bin5(p)
            ax.text(x0 + (j + 0.5) * cw, y + rh / 2 + 0.01, "✓", ha="center", va="center", fontsize=9,
                    color="white" if b in (0, 4) else INK)
    ax.text(x_cnt, y + rh / 2, f"{int(r.hits)} of {int(r.n)}", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    mt, _ = model_txt(key, spec)
    ax.text(x_mod, y + rh / 2, mt, ha="center", va="center", fontsize=9.5,
            color=MUTED if mt == "beyond range" else SUB)
    y += rh + 0.04

# legend
y += 0.35
lx = x_lab
ax.text(lx, y, "Rain & snow:", fontsize=9, color=SUB, va="center")
labs = ["much drier", "drier", "near typical", "wetter", "much wetter"]
for k, (c, t) in enumerate(zip(PR_COL, labs)):
    xx = lx + 1.2 + k * 1.28
    ax.add_patch(Rectangle((xx, y - 0.12), 0.32, 0.24, facecolor=c))
    ax.text(xx + 0.4, y, t, fontsize=8.5, color=SUB, va="center")
y += 0.36
ax.text(lx, y, "Temperature:", fontsize=9, color=SUB, va="center")  # 3 rows: Central Canada, Scandinavia, SE Alaska
for k, (c, t) in enumerate(zip(T_COL, ["much colder", "colder", "near typical", "warmer", "much warmer"])):
    xx = lx + 1.2 + k * 1.28
    ax.add_patch(Rectangle((xx, y - 0.12), 0.32, 0.24, facecolor=c))
    ax.text(xx + 0.4, y, t, fontsize=8.5, color=SUB, va="center")
y += 0.45
ax.text(lx, y, "✓ = on the expected side of the neutral-year median (counted as a match). Much drier/wetter = beyond the "
        "15th/85th percentile of neutral years;\nnear typical = 40th–60th. Central Pacific islands use satellite-era data only "
        "(5 events). ‡ The 1° grid cannot resolve Peru's coastal plain; see text.\n"
        "Model counts are for each original literature region or a box fixed in advance, over the region's full window up to " + HORIZON + " "
        "(6 NMME models for windows after that); † = shape drawn where this year's models agree, so the count is high by construction.", fontsize=8, color=MUTED, va="top", linespacing=1.45)
ax.text(12.2, y + 1.15, "Z. Hausfather · GPCC v2025, Berkeley Earth, GHCN-D, GPCP, NMME, C3S", fontsize=7.5,
        color="#9A9A9A", ha="right", va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(HERE / f"elnino_2026_hit_grid.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight", pad_inches=0.25)
print("written")
