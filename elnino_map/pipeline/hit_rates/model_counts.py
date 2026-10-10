#!/usr/bin/env python3.13
"""Model agreement (one initialization) for each hit-rate region, on the same polygons (map v6) and the model season
closest to the region's label window. Method as model_patterns/regions.py: area-weighted regional-mean
anomaly per model (land only except OCEAN_OK), count of models with the expected sign.
13 models (NMME 6 + C3S 7) for SON/OND/DJF/JF; 6 NMME for JFM/MAM. Temperature: t2m_adj (ERA5-trend
adjusted, main treatment); CanESM5 and CESM1 excluded (undocumented baselines).
Output: derived/model_counts.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, shapely
from extract import SHAPES, OCEAN_OK

ROOT = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(ROOT / "model_patterns"))
from initcfg import INIT_LABEL, SEASONS, ens_for, season_or_fallback   # noqa: E402
a = xr.open_dataset(ROOT / "model_patterns" / "derived" / "seasonal_anoms.nc")
o = xr.open_dataset(ROOT / "model_patterns" / "derived" / "obs_ref.nc")
NMME = ["CFSv2", "CanESM5", "GEM5.2_NEMO", "NASA_GEOS5v2", "NCAR_CCSM4", "NCAR_CESM1"]
C3S = ["ecmwf", "ukmo", "meteo_france", "dwd", "cmcc", "jma", "bom"]
LON2, LAT2 = np.meshgrid(a.lon.values, a.lat.values)
W = np.cos(np.deg2rad(LAT2))
land = o.land.values.astype(bool)

# region: (sign, model season, var)
REG = {
    "maritime": (-1, "SON", "pr"), "philippines": (-1, "DJF", "pr"), "safrica": (-1, "DJF", "pr"),
    "amazon": (-1, "DJF", "pr"), "nsam": (-1, "DJF", "pr"), "nebrazil": (-1, "MAM", "pr"),
    "drycorridor": (-1, "DJF", "pr"), "wpacific": (-1, "SON", "pr"), "cpacific": (1, "SON", "pr"),
    "hawaii": (-1, "DJF", "pr"), "seaustralia": (-1, "SON", "pr"), "inlandnw": (-1, "DJF", "pr"),
    "ccanada_warm": (1, "DJF", "t"), "horn": (1, "OND", "pr"), "peru": (1, "DJF", "pr"),
    "sesa": (1, "DJF", "pr"), "gulf": (1, "DJF", "pr"), "antilles": (1, "DJF", "pr"),
    "swus": (1, "DJF", "pr"), "srilanka": (1, "OND", "pr"), "schina": (1, "DJF", "pr"),
    "mekong": (-1, "MAM", "pr"), "norcal": (1, "DJF", "pr"), "altiplano": (-1, "DJF", "pr"), "cchile": (1, "OND", "pr"), "cl_south_n": (-1, "JFM", "pr"),
    "c_uk_ceurope": (1, "OND", "pr"), "c_scandinavia": (-1, "JF", "t"), "lit_sindia": (1, "OND", "pr"),
    "c_norcal": (1, "DJF", "pr"), "lit_ohio": (-1, "DJF", "pr"), "c_se_alaska": (1, "DJF", "t"),
}
rows = []
for r, (sign, s, var) in REG.items():
    g = SHAPES[r]
    msk = shapely.contains_xy(g, LON2, LAT2)
    if r not in OCEAN_OK:
        msk &= land
    if msk.sum() == 0:  # tiny polygons: nearest cells of the dilated shape
        msk = shapely.contains_xy(g.buffer(0.75), LON2, LAT2) & (land | (r in OCEAN_OK))
    fld = a.prate if var == "pr" else a.t2m_adj
    s = season_or_fallback(s)          # e.g. SON regions use OND for an October start
    ens = ens_for(s)
    if var == "t":
        ens = [m for m in ens if m not in ("CanESM5", "NCAR_CESM1")]
    vals = []
    for m in ens:
        f = fld.sel(model=m, season=s).values
        ok = msk & np.isfinite(f)
        if ok.any():
            vals.append(np.sum(f[ok] * W[ok]) / np.sum(W[ok]))
    vals = np.array(vals)
    rows.append(dict(region=r, season=s, agree=int((np.sign(vals) == sign).sum()), n=len(vals),
                     ncells=int(msk.sum())))
df = pd.DataFrame(rows); df["init"] = INIT_LABEL
df.to_csv(Path(__file__).parent / "derived" / "model_counts.csv", index=False)
print(df.to_string(index=False))
