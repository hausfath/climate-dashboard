# El Niño Impacts pipeline

Everything needed to regenerate the data behind the dashboard's **El Niño Impacts** tab (`elnino_map/data/`,
`elnino_map/img/`) from a new seasonal-forecast initialization. The map's region polygons, confidence tiers and
literature text are frozen inputs (they were set by the author for the Climate Brink impacts post); the forecast
fields, the per-model tiles, the agreement counts and the season timeline are what a refresh changes.

Ported from the private El Niño Impacts project on 2026-10-10 and validated by reproducing the September 2026
bundle byte-for-byte (grid files, regions.json) before the first October run.

## Layout

| Path | What |
|---|---|
| `model_patterns/initcfg.py` | Which initialization to run (`ELNINO_INIT=YYYYMM`, else the newest `raw/nmme_*`), seasons, lead indices |
| `model_patterns/fetch_nmme.sh` | NMME per-model ensemble-mean anomalies, SST anomalies and CPC tercile probabilities (CPC FTP) |
| `model_patterns/fetch_cds.py` | C3S ensemble-mean anomalies (9 systems, 6 leads) and ERA5 monthly 2 m T 1979–2025 (CDS API) |
| `model_patterns/process.py` | Seasonal means, standardized anomalies, agreement metrics → `derived/seasonal_anoms.nc`, `metrics.nc`, `obs_ref.nc`, `validation.txt` |
| `model_patterns/regions.py` | Pre-registered region/box tests (sign tests with FDR) → `derived/region_table.csv` |
| `model_patterns/windows.py`, `window_tests.py` | Full-window model checks per region (the numbers on the cards) → `derived/window_tests.csv`, `monthly_anoms.nc` |
| `hit_rates/extract.py`, `model_counts.py` | Final map polygons (`SHAPES`) and the 3-month-season model counts → `hit_rates/derived/model_counts.csv` |
| `hit_rates/derived/hit_events.csv`, `hit_summary.csv` | **Observed record** of the 8 strong El Niños since 1957 (GPCC v2025, GPCP, station composites, Berkeley Earth). Shipped as derived tables; the gauge extraction that produced them is not part of this repo |
| `figures/make_impacts_map.py`, `make_hit_grid.py` | Region definitions (polygons, tiers, labels) and the hit-grid rows, read by the exports |
| `export/export_data.py` | Region records: polygons + observed record + model values → `export/data/regions.json`, `stats.json`, `export/img/sst_DJF.png` |
| `export/export_dashboard.py` | Writes the dashboard bundle: `../data/{regions,meta,lit,geo}.json`, `../data/grid_<season>.bin`, `../img/sst_DJF.png` |
| `data/forecast_2027_extract.json` | P(2027 warmest year) from the global temperature forecast workflow (refreshed by hand) |
| `oni_cpc_2026-09-15.txt` | CPC ONI table (strong-event list, neutral years for the "typical year" reference) |

`raw/` and the large `derived/*.nc` are git-ignored and regenerated; the small CSVs are committed.

## Refresh for a new initialization

Needs `python3.13` with the packages in `requirements.txt` (cartopy and shapely are not in the web app's
requirements), a CDS API key in `~/.cdsapirc`, and ~250 MB of raw downloads.

```sh
cd elnino_map/pipeline
export ELNINO_INIT=202610                      # the initialization to run
model_patterns/fetch_nmme.sh 202610            # CPC posts the month's folder around the 8th
python3.13 model_patterns/fetch_cds.py 202610  # C3S systems post between the ~5th (ECMWF) and ~13th; ERA5 once
python3.13 model_patterns/process.py           # seasonal fields + metrics + validation.txt
python3.13 model_patterns/regions.py           # region_table.csv
(cd hit_rates && python3.13 model_counts.py)   # model_counts.csv (carries the init)
python3.13 model_patterns/window_tests.py      # window_tests.csv (step 1 reproduces the two tables above)
(cd export && python3.13 export_data.py && python3.13 export_dashboard.py)
```

`./run_refresh.sh 202610` runs the processing and export steps in that order once the raw files are fetched.

`export_dashboard.py` prints every region whose model count or average changed, asserts the counts against the
member lists, and writes `meta.json` with the season list the map builds its timeline from. After a refresh,
delete grid files for seasons the new start no longer covers (an October start has no Sep–Nov), reread the dated
sentences in `export/data/region_lit.json`, and check `model_patterns/derived/validation.txt`: check [1] must
reproduce CPC's NMME mean (r = 1.0000) and check [3] fixes the tercile-target convention.

## Methods

`model_patterns/METHODS.md` (the September 2026 analysis, with the full-window rule of 27 Sep 2026 and the
October 2026 refresh note at the end). Data licences: NOAA NMME (public); Copernicus C3S ("Generated using
Copernicus Climate Change Service information 2026"); GPCC/GPCP/Berkeley Earth under their own terms;
Natural Earth (public domain); code MIT.
