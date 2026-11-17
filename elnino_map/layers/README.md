# El Niño Impacts map: GIS layers

The layers behind the **El Niño Impacts** tab of the [Climate Dashboard](https://dashboard.theclimatebrink.com)
(analysis: Zeke Hausfather, [The Climate Brink](https://www.theclimatebrink.com)), exported in standard formats.
Everything here is written by `build_layers.py` from the published bundle in `../data/`; nothing is re-derived.

Forecast initialization: **November 2026** (see `layers_metadata.json`). The rainfall grids are refreshed every
month when new seasonal-forecast runs post; the region polygons and confidence tiers are the author's assessment
and do not change with the forecasts.

## Files

| File | What | Format |
|---|---|---|
| `impacts_regions.geojson` | The 26 regions drawn on the map | GeoJSON (WGS84, RFC 7946; the W Pacific and Central Pacific regions are split at the dateline into MultiPolygons) |
| `impacts_regions_extra.geojson` | 5 "also tracked" regions (Indian monsoon, Ohio Valley, UK/central Europe, Scandinavia) shown neutral/dashed in the layers panel, not on the map | GeoJSON |
| `impacts_regions.csv` | Attribute table of the 26 regions | CSV |
| `rainfall_<S>_model_mean.tif` | Multi-model mean forecast precipitation, % departure from normal (0 = normal, −30 = 70% of normal) | GeoTIFF, Int16, nodata −32768 |
| `rainfall_<S>_agreement.tif` | Number of models whose sign matches the multi-model mean | GeoTIFF, Int16 |
| `rainfall_<S>_stipple.tif` | 1 where the map draws an agreement dot: ≥80% of models agree on the direction and the mean departs by ≥10% | GeoTIFF, UInt8 |
| `rainfall_<S>_models.tif` | One band per model (band descriptions carry the model names), same units as the mean | GeoTIFF, Int16 |
| `sst_DJF_nmme.tif` | The page's "Ocean temperature forecast, Dec–Feb" layer: mean Dec 2026–Feb 2027 SST anomaly of the NMME models that share a 1992–2019 baseline (CFSv2, NASA GEOS, NCAR CCSM4; NASA GEOS did not post for November, so this run uses CFSv2 and CCSM4), °C, ocean only. The page tapers it to 30°S–30°N and smooths it for display; the file is the full, unsmoothed field | GeoTIFF, Float32, nodata −9999 |
| `layers_metadata.json` | Init date, model list per season, baselines, attribution | JSON |

Seasons `<S>`: `DJF` = Dec 2026–Feb 2027 and `JFM` = Jan–Mar 2027 (12 models each: 5 NMME + 7 Copernicus C3S)
and `MAM` = Mar–May 2027 (5 NMME models; the C3S runs end in April). NASA GEOS did not post its November run and is left out.

## Region attributes

- `signal`: drier / wetter / warmer. `confidence`: high / medium-high / medium. These are the map's fill colours
  (`fill_hex`) and come from the teleconnection literature and the observed record, not from the forecast models.
- `underperformed_recent_event`: dashed outline on the map; the signal under-delivered in a recent strong El Niño.
- `observed_median` is in `observed_units` (% of the 1991–2020 normal, or °C for the central Canada temperature region).
- `observed_hits` / `observed_n`: how many of the strong El Niños since 1957 (1957, 1965, 1972, 1982, 1997, 2009,
  2015, 2023) went the expected way over the region's window, measured against a typical neutral year
  (GPCC v2025 land data; GPCP and station composites for islands; Berkeley Earth for the central Canada temperature signal).
- `models_agree` / `models_n`, `model_mean_pct_of_normal`, `forecast_window`: the November 2026 forecast check over the
  region's window, relative to the 1991–2020 normal.

## Raster grid

1° × 1°, 181 rows (90°N to 90°S) × 360 columns (180°W to 179°E), cell centres on integer degrees, EPSG:4326.
Normal = the GPCP v2.3 1991–2020 seasonal climatology; cells with a normal below 0.5 mm/day (deserts) are masked, and
very dry cells can show departures of several hundred percent. Values are floored at −100 (no rain at all; a few hundred
mean cells per season sit at that floor, mostly in deserts). The map shading saturates at ±100%.

## Attribution and licences

- Forecasts: NOAA NMME (public domain) and Copernicus C3S seasonal forecasts. The C3S licence requires this
  attribution on any derived product: **"Generated using Copernicus Climate Change Service information 2026."**
- Observations: GPCC Full Data Monthly v2025 (DWD), GPCP v2.3 (NOAA PSL), GHCN-Daily (NOAA), Berkeley Earth.
  Only regional aggregates of these are included here; each provider's own terms apply to the underlying data.
- Basemap in the dashboard: Natural Earth (public domain). Not included here.
- Region polygons, confidence tiers and the observed-record aggregates: Zeke Hausfather / The Climate Brink,
  [LICENCE TO BE SET BY AUTHOR]. Suggested credit: "Analysis: Zeke Hausfather, The Climate Brink
  (dashboard.theclimatebrink.com); forecasts: NOAA NMME and Copernicus C3S."
- Code (`build_layers.py`): MIT.

## Regenerate

```sh
cd elnino_map/layers && python3.13 build_layers.py   # needs rasterio, shapely, numpy (+ xarray and the raw NMME SST files for the SST layer)
```
