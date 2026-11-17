#!/usr/bin/env python3.13
"""Export the El Niño Impacts map layers in standard GIS formats (GeoJSON + GeoTIFF).

Reads only the published dashboard bundle (../data/regions.json, meta.json, grid_<season>.bin); nothing is
re-derived. Polygons and confidence tiers are the author's literature assessment (frozen inputs); the rainfall
grids are the multi-model mean forecast for the initialization named in meta.json.

Outputs (this folder):
  impacts_regions.geojson        26 map regions (RFC 7946; dateline-crossing rings split into MultiPolygons)
  impacts_regions_extra.geojson  5 "also tracked" regions shown dashed/neutral in the layers panel, not on the map
  impacts_regions.csv            flat attribute table of the 26 regions
  rainfall_<S>_model_mean.tif    multi-model mean precipitation, % departure from the GPCP 1991–2020 normal
  rainfall_<S>_agreement.tif     number of models whose sign matches the multi-model mean (0..n)
  rainfall_<S>_stipple.tif       1 where the map draws an agreement dot (>=80% agree and |mean| >= 10%)
  rainfall_<S>_models.tif        one band per model (names in band descriptions), same units as model_mean
  layers_metadata.json           init, seasons, model lists, baselines, licences
"""
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping, box
from shapely.ops import unary_union
from shapely.affinity import translate

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ND = -32768
NLAT, NLON = 181, 360

meta = json.loads((DATA / "meta.json").read_text())
regions = json.loads((DATA / "regions.json").read_text())
CONF = {"high": "high", "medhigh": "medium-high", "medium": "medium"}
KIND = {"dry": "drier", "wet": "wetter", "warm": "warmer"}

# ---------------------------------------------------------------- polygons
def split_dateline(rings):
    """Rings use continuous longitudes that may run past ±180. Return an RFC 7946 geometry inside [-180, 180]."""
    poly = Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0])
    if not poly.is_valid:
        poly = poly.buffer(0)
    xs = [x for x, _ in poly.exterior.coords]
    if min(xs) >= -180 and max(xs) <= 180:
        return poly
    parts = []
    for shift in (-360, 0, 360):                       # bring every copy that overlaps the world window inside it
        p = translate(poly, xoff=shift)
        clip = p.intersection(box(-180, -90, 180, 90))
        if not clip.is_empty:
            parts.append(clip)
    geom = unary_union(parts)
    return geom if isinstance(geom, (Polygon, MultiPolygon)) else MultiPolygon([g for g in geom.geoms if isinstance(g, Polygon)])


def props(key, R, on_map):
    H, M = R.get("hits") or {}, R.get("models") or {}
    p = {
        "key": key, "title": R["title"], "detail": R.get("detail"), "on_map": on_map,
        "signal": KIND.get(R.get("kind"), R.get("kind")),
        "confidence": CONF.get(R.get("conf"), R.get("conf")),
        "underperformed_recent_event": bool(R.get("dashed")) if on_map else None,
        "fill_hex": R.get("fill") if on_map else "#8A93A6",   # extras are drawn neutral grey, dashed (app.js)
        "observed_source": H.get("source"), "observed_window": H.get("season"),
        "observed_hits": H.get("hits"), "observed_n": H.get("n"),
        "observed_reference": "strong El Niños since 1957 vs a typical neutral year" if H else None,
        "observed_units": ("°C anomaly" if H.get("source") == "berkeley" else "% of 1991–2020 normal") if H else None,
        "observed_median": H.get("median_pct_normal"),       # median of the strong events in observed_units
        "lanina_same_direction": H.get("lanina_same_dir"),
        "forecast_init": meta["init"],
        "forecast_window": M.get("window"), "forecast_ensemble": M.get("ens"),
        "models_agree": M.get("agree"), "models_n": M.get("n"),
        "model_mean_pct_of_normal": (round(100 + M["mmm_pct"], 1) if M.get("mmm_pct") is not None else None),   # the card's "Model average"
    }
    return p

def write_geojson(path, items, on_map):
    feats = []
    for key, R in items.items():
        g = split_dateline(R["rings"])
        feats.append({"type": "Feature", "id": key, "properties": props(key, R, on_map), "geometry": mapping(g)})
    fc = {"type": "FeatureCollection", "name": path.stem,
          "crs_note": "WGS84 lon/lat (RFC 7946); dateline-crossing regions are split into MultiPolygons",
          "features": feats}
    path.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")))
    return feats

feats = write_geojson(HERE / "impacts_regions.geojson", regions["regions"], True)
write_geojson(HERE / "impacts_regions_extra.geojson", regions["extras"], False)
assert len(feats) == meta["n_map_regions"] == 26, len(feats)
for f in feats:                                       # every vertex inside the world window
    for ring in (r for part in (f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]) for r in part):
        assert all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in ring), f["id"]

cols = list(feats[0]["properties"].keys())
with open(HERE / "impacts_regions.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for f in feats: w.writerow(f["properties"])

# ---------------------------------------------------------------- rasters
# cell centres at integer lat/lon (90..-90, -180..179), so the pixel grid origin is (-180.5, 90.5)
TRANSFORM = from_origin(-180.5, 90.5, 1.0, 1.0)
PROFILE = dict(driver="GTiff", height=NLAT, width=NLON, crs="EPSG:4326", transform=TRANSFORM, compress="deflate")

def write_tif(path, bands, dtype, nodata, descriptions=None, tags=None):
    bands = np.asarray(bands)
    if bands.ndim == 2: bands = bands[None]
    with rasterio.open(path, "w", count=bands.shape[0], dtype=dtype, nodata=nodata, **PROFILE) as dst:
        for k in range(bands.shape[0]):
            dst.write(bands[k].astype(dtype), k + 1)
            if descriptions: dst.set_band_description(k + 1, descriptions[k])
        dst.update_tags(**(tags or {}))
    with rasterio.open(path) as src:                   # round trip
        back = src.read()
        assert back.shape == bands.shape and np.array_equal(back, bands.astype(dtype)), path.name
        assert abs(src.bounds.left + 180.5) < 1e-9 and abs(src.bounds.top - 90.5) < 1e-9

season_meta = {}
for s, S in meta["seasons"].items():
    n = S["n"]
    g = np.fromfile(DATA / f"grid_{s}.bin", dtype="<i2").reshape(n + 1, NLAT, NLON)
    models, mm = g[:n], g[n]
    valid_m = mm != ND
    # engine.js setGrid: count models whose sign matches the mean's sign (ND members never count)
    sgn = np.sign(mm)
    agree = ((models != ND) & (np.sign(models) == sgn[None])).sum(0).astype(np.int16)
    agree = np.where(valid_m, agree, ND).astype(np.int16)
    stipple = (valid_m & (np.abs(mm) >= 10) & (agree.astype(float) / n >= 0.8 - 1e-9)).astype(np.uint8)
    tags = {"init": meta["init"], "season": S["label"], "units": "percent departure from GPCP 1991-2020 normal (0 = normal)",
            "models": ", ".join(m["label"] for m in S["models"]), "source": S["source"],
            "attribution": "Generated using Copernicus Climate Change Service information 2026; NOAA NMME; analysis Zeke Hausfather, The Climate Brink"}
    write_tif(HERE / f"rainfall_{s}_model_mean.tif", mm, "int16", ND, ["multi-model mean, % departure from normal"], tags)
    write_tif(HERE / f"rainfall_{s}_agreement.tif", agree, "int16", ND, [f"models agreeing with mean sign (of {n})"], {**tags, "units": f"count of {n} models"})
    write_tif(HERE / f"rainfall_{s}_stipple.tif", stipple, "uint8", None, ["1 = agreement dot on the map"], {**tags, "units": ">=80% of models agree and |mean| >= 10%"})
    write_tif(HERE / f"rainfall_{s}_models.tif", models, "int16", ND, [m["label"] for m in S["models"]], tags)
    season_meta[s] = {"label": S["label"], "window_months": S["win"], "n_models": n, "models": [m["label"] for m in S["models"]],
                      "source": S["source"], "stipple_cells": int(stipple.sum()), "valid_cells": int(valid_m.sum())}
    print(f"{s}: {n} models, {int(valid_m.sum())} valid cells, {int(stipple.sum())} stippled, mean range {mm[valid_m].min()}..{mm[valid_m].max()}%")

# ---------------------------------------------------------------- SST layer (page toggle "Ocean temperature forecast, Dec–Feb")
# Reproduces export/export_data.py: mean of the NMME models sharing the 1992–2019 baseline (CFSv2, NASA GEOS, NCAR CCSM4,
# whichever posted for this init), Dec–Feb, ocean only.
# Shipped untapered and unsmoothed (the page tapers it to the tropics and blurs it for display).
import sys
sys.path.insert(0, str(HERE.parent / "pipeline" / "model_patterns"))
sst_meta = None
try:
    import xarray as xr
    from initcfg import SST_DIR, SEASONS as CFG_SEASONS, INIT_LABEL, NMME
    assert INIT_LABEL == meta["init"], (INIT_LABEL, meta["init"])
    o = xr.open_dataset(HERE.parent / "pipeline" / "model_patterns" / "derived" / "obs_ref.nc")
    land = o.land.values.astype(bool)
    SST_MODELS = [m for m in ["CFSv2", "NASA_GEOS5v2", "NCAR_CCSM4"] if m in NMME]
    fields = []
    for m in SST_MODELS:
        d = xr.open_dataset(SST_DIR / f"{m}.tmpsfc.anom.nc", decode_times=False)
        assert np.allclose(d.lat.values, np.arange(-90, 91)) or np.allclose(d.lat.values, np.arange(90, -91, -1)), d.lat.values[:3]
        f = d.fcst.sel(target=[float(d.target.values[0]) + k for k in CFG_SEASONS["DJF"][0]]).mean("target")   # target = months since Jan 1960; offsets count from the init month
        f = f.sortby("lat", ascending=False).values                       # 90..-90 like the grids
        assert f.shape == (NLAT, NLON) and np.allclose(d.lon.values, np.arange(360))
        fields.append(f)
    sst = np.where(land, np.nan, np.mean(fields, 0)).astype("float32")
    lat2 = np.arange(90, -91, -1)[:, None] * np.ones((1, NLON)); lon2 = np.ones((NLAT, 1)) * np.arange(360)[None]
    nino34 = float(np.nanmean(sst[(lat2 >= -5) & (lat2 <= 5) & (lon2 >= 190) & (lon2 <= 240)]))
    stats = json.loads((HERE.parent / "pipeline" / "export" / "data" / "stats.json").read_text())
    assert abs(nino34 - stats["sst_nino34_djf_3model"]) < 0.006, (nino34, stats["sst_nino34_djf_3model"])
    sst = np.roll(sst, 180, axis=-1)
    write_tif(HERE / "sst_DJF_nmme.tif", np.where(np.isfinite(sst), sst, -9999.0), "float32", -9999.0,
              [f"NMME {len(SST_MODELS)}-model mean SST anomaly, Dec 2026–Feb 2027, °C vs 1992–2019"],
              {"init": meta["init"], "models": ", ".join(SST_MODELS), "units": "degC anomaly vs each model's 1992-2019 hindcast climatology",
               "source": "NOAA NMME (CPC realtime_anom ENSMEAN tmpsfc)", "note": "ocean cells only; the page shows this field tapered to 30S-30N and lightly smoothed"})
    sst_meta = {"models": SST_MODELS, "window": "Dec 2026–Feb 2027", "baseline": "1992–2019", "nino34_box_mean_degC": round(nino34, 2)}
    print(f"sst_DJF_nmme.tif ({len(SST_MODELS)} models): Niño 3.4 box mean {nino34:.2f} °C (stats.json {stats['sst_nino34_djf_3model']})")
except (ImportError, FileNotFoundError, OSError) as e:
    print("SST layer skipped (raw NMME SST files not available):", e)

(HERE / "layers_metadata.json").write_text(json.dumps({
    "sst_layer": sst_meta,
    "map": "El Niño 2026–27 impacts map, Climate Dashboard (dashboard.theclimatebrink.com, El Niño Impacts tab)",
    "forecast_init": meta["init"],
    "seasons": season_meta,
    "raster_grid": "1° × 1°, 181 rows (90N..90S) × 360 cols (180W..179E), cell centres on integer degrees, EPSG:4326",
    "raster_baseline": "forecast precipitation as % departure from the GPCP v2.3 1991–2020 seasonal normal; cells with normal < 0.5 mm/day are masked (nodata -32768)",
    "polygon_source": "author's assessment of the teleconnection literature and the observed record of 8 strong El Niños (1957, 1965, 1972, 1982, 1997, 2009, 2015, 2023); polygons and tiers do not change with the forecasts",
    "observed_record": meta["obs_source"],
    "model_horizon": meta["model_horizon"],
    "attribution": {"forecasts": "NOAA NMME (public); Copernicus C3S: 'Generated using Copernicus Climate Change Service information 2026'",
                    "observations": "GPCC Full Data Monthly v2025 (DWD); GPCP v2.3 (NOAA PSL); GHCN-Daily (NOAA); Berkeley Earth (central Canada temperature)",
                    "basemap": "Natural Earth (public domain)",
                    "analysis": "Zeke Hausfather, The Climate Brink"},
}, ensure_ascii=False, indent=1))
print("wrote", len(list(HERE.glob("*.tif"))), "tifs,", len(feats), "regions")
