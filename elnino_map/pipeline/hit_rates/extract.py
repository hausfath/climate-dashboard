#!/usr/bin/env python3.13
"""Regional-mean monthly series for the El Niño 2026-27 impacts-map regions.

Inputs (hit_rates/raw/): NOAA PREC/L 1° (precip, mm/day), GHCN-CAMS 0.5° (2 m air temp, K),
plus GPCP v2.3 2.5° from model_patterns/raw/obs/.
Region shapes: the v5.3 map polygons (model_polygons.json + hand-drawn shapes in
figures/make_impacts_map.py), the frozen pre-model literature polygons (lit_polygons_v4.json,
prefixed "lit_"), and a set of fixed "not on the map" contrast boxes.

Weights = cos(lat) × fraction of each grid cell inside the polygon (5×5 sub-sampling), so small
regions (Sri Lanka, SE Australia) still get a weighted mean on coarse grids. Land-only regions use
PREC/L's own land coverage (NaN over ocean); for GPCP a land mask from the 1° obs_ref file is applied
except for the ocean/island regions in OCEAN_OK.

Output: derived/regional_monthly.csv (long format: dataset, region, time, value)
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import shapely
from shapely.geometry import Polygon, box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = HERE / "derived"
OUT.mkdir(exist_ok=True)

# ---- region shapes -------------------------------------------------------------
src = (ROOT / "figures" / "make_impacts_map.py").read_text()
ns = {"HERE": ROOT / "figures"}   # the borrowed block reads the 2027 odds relative to the figures folder
exec(src[src.index("REGIONS = ["):src.index("FOOTER =")], ns)
MODEL = {k: shapely.geometry.shape(v) for k, v in
         json.loads((ROOT / "model_patterns" / "derived" / "model_polygons.json").read_text()).items()}


def to360(g):
    return shapely.affinity.translate(g, 360) if g.bounds[0] < 0 else g


SHAPES = {}
for key, kind, conf, flag, shp, *_ in ns["REGIONS"]:
    if shp == "model":
        SHAPES[key] = MODEL[key]
    else:  # hand-drawn, buffered exactly as on the map
        SHAPES[key] = to360(Polygon(shp).buffer({"peru": 0.8, "mekong": 0.8, "norcal": 0.3, "altiplano": 0.3, "cchile": 0.2}.get(key, 2.0), join_style=1).simplify(0.25))

LITP = json.loads((ROOT / "model_patterns" / "lit_polygons_v4.json").read_text())
LBUF = {"peru": 0.8, "camerica": 1.0, "safrica": 1.4}   # same as model_patterns/regions.py
for k, v in LITP.items():
    SHAPES["lit_" + k] = to360(Polygon(v["pts"]).buffer(LBUF.get(k, 2.0), join_style=1))

# contrast regions, fixed before any hit-rate results were computed
CONTRAST = {
    "c_uk_ceurope": box(-10.5, 45, 20, 59),     # UK/Ireland + central Europe (europe.py boxes merged)
    "c_scandinavia": box(5, 55, 30, 70),
    "c_sahel": box(-18, 10, 35, 18),
    "c_mekong_south": box(102, 9, 109, 16),     # Cambodia, S Laos, S Vietnam (Räsänen et al. 2016; Nguyen et al. 2023)
    "c_norcal": box(-124.5, 38, -120, 42),       # as model_patterns/regions.py
    "c_se_alaska": box(-140, 54, -128, 60),      # Gulf of Alaska coast / Panhandle / N BC coast
}
for k, g in CONTRAST.items():
    SHAPES[k] = to360(g) if g.bounds[0] < 0 else g
# lit_ohio and lit_sindia (in LITP) serve as the Ohio Valley and S India contrasts

# simple pre-specified lat-lon boxes around the labels of model-born regions (selection-effect check;
# chosen from the map labels, not from model or observed fields)
BOXES = {
    "box_antilles": box(-85, 18, -72, 24),        # as model_patterns/regions.py candidate
    "box_inlandnw": box(-125, 45, -110, 56),
    "box_srilanka": box(79.5, 5.8, 82, 10),
    "box_seaustralia": box(140.5, -43.7, 150, -36),
    "box_ccanada": box(-102, 49, -80, 60),
    "box_hawaii": box(-161, 18, -154, 23),        # as regions.py candidate
}
for k, g in BOXES.items():
    SHAPES[k] = to360(g) if g.bounds[0] < 0 else g


def _drycorridor_wet():
    """Dry Corridor polygon restricted to 1° cells whose DJF 1991-2020 PREC/L mean is >= 1 mm/day."""
    p = xr.open_dataset(RAW / "precl_1x1.nc", drop_variables=["time_bnds"]).precip
    p = p.sel(time=slice("1991-01", "2020-12"))
    djf = p.where(p.time.dt.month.isin([12, 1, 2])).mean("time")
    cells = []
    g = SHAPES["drycorridor"]
    for la in djf.lat.values:
        for lo in djf.lon.values:
            c = box(lo - 0.5, la - 0.5, lo + 0.5, la + 0.5)
            if c.intersects(g) and float(djf.sel(lat=la, lon=lo)) >= 1.0:
                cells.append(c.intersection(g))
    return shapely.union_all(cells)


_frozen = OUT / "drycorridor_wet.json"
if (RAW / "precl_1x1.nc").exists():
    SHAPES["drycorridor_wet"] = _drycorridor_wet()
else:   # the PREC/L file is not shipped with the pipeline: use the shape frozen from it
    SHAPES["drycorridor_wet"] = shapely.geometry.shape(json.loads(_frozen.read_text())["geometry"])

# Chile proposals from a Chilean colleague (24 Sep 2026); boxes fixed from their sketch before any results
CHILE = {
    "cl_altiplano": box(-71, -22, -66, -14),
    "cl_atacama": box(-71.5, -30, -69, -18),
    "cl_central": box(-73.8, -40, -70, -30),
    "cl_south": box(-76, -52, -71.5, -40),
    # narrowed per the literature check (research/lit_chile_altiplano.md), fixed before results
    "cl_central_n": box(-73.8, -38, -70, -35),
    "cl_south_n": box(-74.5, -42, -71, -38),
}
for k, g in CHILE.items():
    SHAPES[k] = to360(g)

# validation shapes: US state outlines (Natural Earth 50m) for comparison with NCEI statewide series
import cartopy.io.shapereader as _shp
for _r in _shp.Reader(_shp.natural_earth(resolution="50m", category="cultural",
                                         name="admin_1_states_provinces")).records():
    if _r.attributes.get("adm0_a3") == "USA" and _r.attributes.get("postal") in ("CA", "AZ", "NM"):
        SHAPES["val_" + _r.attributes["postal"].lower()] = to360(_r.geometry)

OCEAN_OK = {"maritime", "wpacific", "cpacific", "hawaii", "philippines", "peru", "srilanka", "antilles",
            "lit_maritime", "lit_wpacific", "lit_cpacific"}


def frac_weights(g, lat, lon, n=5):
    """Fraction of each (lat, lon) cell inside polygon g (lon 0..360), times cos(lat)."""
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    xmin, ymin, xmax, ymax = g.bounds
    w = np.zeros((lat.size, lon.size))
    jj = np.where((lat >= ymin - dlat) & (lat <= ymax + dlat))[0]
    ii = np.where((lon >= xmin - dlon) & (lon <= xmax + dlon))[0]
    off = (np.arange(n) + 0.5) / n - 0.5
    for j in jj:
        for i in ii:
            X, Y = np.meshgrid(lon[i] + off * dlon, lat[j] + off * dlat)
            w[j, i] = shapely.contains_xy(g, X, Y).mean()
    return w * np.cos(np.deg2rad(lat))[:, None]


def regional_series(da, name, land=None):
    lat, lon = da.lat.values, da.lon.values
    vals = da.values  # (time, lat, lon)
    out = {}
    for reg, g in SHAPES.items():
        w = frac_weights(g, lat, lon)
        if land is not None and reg not in OCEAN_OK:
            w = w * land
        if w.sum() == 0:
            continue
        ok = np.isfinite(vals)
        num = np.nansum(vals * w, axis=(1, 2))
        den = np.sum(ok * w, axis=(1, 2))
        s = np.where(den > 0.5 * den.max(), num / np.where(den > 0, den, 1), np.nan)  # >=50% of the land weight ever observed
        out[reg] = s
        out[reg + "__ncells"] = np.full(s.shape, (w > 0).sum())
    df = pd.DataFrame(out, index=pd.DatetimeIndex(da.time.values))
    return df


if __name__ == "__main__":
    frames = []
    p = xr.open_dataset(RAW / "precl_1x1.nc", drop_variables=["time_bnds"]).precip.sortby("lat")
    df = regional_series(p, "precl")
    frames.append(df.assign(dataset="precl"))

    gp = xr.open_dataset(ROOT / "model_patterns" / "raw" / "obs" / "gpcp_precip.mon.mean.nc", drop_variables=["time_bnds"]).precip.sortby("lat")
    o = xr.open_dataset(ROOT / "model_patterns" / "derived" / "obs_ref.nc")
    land1 = o.land.astype(float)
    land1 = land1.assign_coords(lon=np.mod(land1.lon, 360)).sortby("lon")
    land_g = land1.interp(lat=gp.lat, lon=gp.lon, method="nearest").fillna(0).values
    frames.append(regional_series(gp, "gpcp", land=land_g).assign(dataset="gpcp"))

    tpath = RAW / "ghcncams_air.nc"
    if tpath.exists():
        t = xr.open_dataset(tpath, drop_variables=["time_bnds"]).air.sortby("lat") - 273.15
        frames.append(regional_series(t, "ghcncams").assign(dataset="ghcncams"))

    long = []
    for f in frames:
        d = f.drop(columns=[c for c in f.columns if c.endswith("__ncells")] + ["dataset"])
        nc = {c.replace("__ncells", ""): int(f[c].iloc[0]) for c in f.columns if c.endswith("__ncells")}
        m = d.reset_index(names="time").melt(id_vars="time", var_name="region", value_name="value")
        m["dataset"] = f["dataset"].iloc[0]
        m["ncells"] = m.region.map(nc)
        long.append(m)
    pd.concat(long).to_csv(OUT / "regional_monthly.csv", index=False)
    print(pd.concat(long).groupby(["dataset"]).region.nunique())
