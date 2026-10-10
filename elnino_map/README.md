# El Niño Impacts map

The interactive map behind the dashboard's **El Niño Impacts** tab. It is a static bundle (HTML, CSS and plain JavaScript, no build step). A small Flask route in `src/dashboard.py` serves it at `/elnino-map/`, and `assets/impacts_embed.js` embeds it in an iframe that loads on the first visit to the tab and follows the dashboard's dark/light toggle. It lives outside `assets/` because Dash injects every `.js`/`.css` file under `assets/` into the main page.

Standalone: `http://<host>/elnino-map/` (add `?theme=light|dark`, or `?region=<key>` to open a region card on load).

## What it shows

- **Impact regions.** The 26 polygons from the Climate Brink impacts map (Sep 2026). The shade is the assessed confidence. A dashed outline means the signal underperformed in 2015–16 or 2023–24.
- **Region cards** (click a region, or zoom in and one opens by itself):
  - the observed record in the 8 strong El Niños since 1957 (bars = % of a typical ENSO-neutral year, the same reference as the checks; line = the 1991–2020 average);
  - how often La Niña years went the same way;
  - one tile per model from this year's forecasts, averaged over the region's whole season up to the last month all 13 systems cover (Mar 2027 for the Oct start), or the 6 NMME models for seasons mostly after that (`pipeline/model_patterns/windows.py`);
  - literature text with links.
- **Point readout.** Click anywhere outside a region for the raw model values in that 1° grid cell. The card says plainly that no observed record sits behind it.
- **Season timeline.** Switch or play the 3-month seasons the current initialization covers (for the October 2026 start: Oct–Dec, Dec–Feb, Jan–Mar and Mar–May; the list comes from `meta.json`). The background shows the multi-model mean rainfall as % of normal (100% = normal), dots mark ≥80% model agreement with a departure of ≥ 10%, and regions outside the season fade. (No stop for Jun–Aug 2027 yet: the forecasts end in spring. The Yangtze region, a Jun–Jul signal, stays faded until then.)
- **Replay a past El Niño.** Each region is coloured by whether it went the expected way in that event. The tally is checked in the browser against `meta.json` `by_event`.
- **Guided tour.** The stops follow the blog post's regional order.
- **Zoom chips.** Past a zoom threshold, each visible region shows a small box with its observed and model counts. Zooming in (scroll or pinch) with the pointer over a region until it fills part of the view opens its card. This never happens after a pan, and it never moves the camera.
- **Tooltips.** Hover a model tile for the model's name and its rainfall as % of normal (the regional mean, or the grid cell in the point readout), the same level convention as the past-event bars. Hover a past-event bar for its value against the 1991–2020 average and a typical neutral year.

## Files

| Path | What |
|---|---|
| `index.html`, `app.css` | Layout and theme tokens (mirrors `assets/theme.css`) |
| `js/engine.js` | Canvas map engine, ported from the explainer video: camera, dateline wrap, regions, rainfall field and dots built from the grids |
| `js/cards.js` | Region, point, global and how-to cards. Every number is read from `data/` |
| `js/app.js` | State, pan/zoom/pinch, hover, click, tour, replay, season player, theme sync |
| `data/regions.json` | Region polygons, observed record per event (`pct_normal`, `pct_typ`, `typ_level`, `hit`), per-model values |
| `data/grid_<S>.bin` | Int16 little-endian `[n_models + 1, 181, 360]`: each model's precipitation anomaly as whole % of the GPCP 1991–2020 normal, then the multi-model mean. lat 90→−90, lon −180→179, 1°. −32768 = no value (land cells with < 0.5 mm/day normal). Not clipped: normally dry ocean cells reach several thousand % |
| `data/meta.json` | Forecast start month, seasons (label, models, month window) that the timeline is built from, model horizons, strong-event list and per-event tallies, 2027 record odds |
| `data/lit.json` | Literature text per region, the global card and the tour |
| `data/geo.json` | Natural Earth land, borders |
| `img/sst_DJF.png` | NMME Dec–Feb SST anomaly layer |

## Regenerating (after each monthly forecast refresh)

The pipeline lives in [`pipeline/`](pipeline/README.md) (ported from the Climate Brink El Niño impacts project on
2026-10-10 and validated by reproducing the September 2026 bundle byte-for-byte). For a new initialization:

```sh
cd elnino_map/pipeline
export ELNINO_INIT=202610
model_patterns/fetch_nmme.sh 202610 && python3.13 model_patterns/fetch_cds.py 202610
python3.13 model_patterns/process.py && python3.13 model_patterns/regions.py
(cd hit_rates && python3.13 model_counts.py) && python3.13 model_patterns/window_tests.py
(cd export && python3.13 export_data.py && python3.13 export_dashboard.py)
```

`export_dashboard.py` writes this folder's `data/` and `img/`, prints every region whose model count or average
changed, and asserts each region's agreement count against its member list. The season timeline, model counts and
horizons in the page are read from `meta.json`, so no JavaScript changes are needed for a new start month; delete
`grid_<season>.bin` files for seasons the new start no longer covers. After a refresh, reread the dated sentences
in `pipeline/export/data/region_lit.json` (the IMD monsoon figure, the Niño 1+2 comparison and "east-leaning so
far"). The observed record (`pipeline/hit_rates/derived/`) is a frozen input; the region polygons and confidence
tiers are the author's and do not change with the forecasts.

## Data sources and licences

- Observed precipitation: GPCC Full Data Monthly v2025 (DWD); GPCP v2.3 monthly (NOAA PSL) and GHCN-Daily station composites for island regions (NOAA). Temperature (central Canada): Berkeley Earth. Each provider's own terms apply; check them before republishing the data files.
- Seasonal forecasts: NOAA NMME (public) and Copernicus C3S multi-system seasonal forecasts. The C3S data carry the Copernicus licence, which requires attribution: "Generated using Copernicus Climate Change Service information 2026."
- Map geometry: Natural Earth (public domain).
- Code: MIT, like the rest of the dashboard.

## Two reference points

The past-event bars are measured against a typical neutral year (`meta.bars = "typical"`), so each bar lines up with its check mark. The forecast shading and model tiles stay relative to the 1991–2020 average (`meta.baseline = "normal"`). `El Nino Impacts/video/tools/export_dashboard.py` can also rebase the forecasts (`BASELINE=typical`). That option was previewed on 27 Sep 2026 and rejected; the reasons are in `hit_rates/METHODS.md` and in the tab's "How to read this" card.
