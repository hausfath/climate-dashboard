# Proposal 03 — Equatorial Pacific warm water volume vs Niño 3.4

## What the panel shows

Monthly **Warm Water Volume anomaly** (WWV: volume of water warmer than
20 °C across 5°N–5°S, 120°E–80°W — NOAA/PMEL) plotted against the observed
**Niño 3.4 SST anomaly** from the repo's committed series, 1990 → present,
with zero lines aligned across the two axes. An annotation reports the lag
correlation computed from the plotted data (WWV leading), so the precursor
relationship is shown, not asserted.

> **Proposed panel caption:** The Warm Water Volume index — the volume of
> equatorial-Pacific water warmer than 20 °C (5°N–5°S, 120°E–80°W), derived
> from PMEL's ocean analyses. Subsurface heat leads the surface by two to
> three seasons, making this the classic indicator of where ENSO goes next.

## Why it adds value

Upper-ocean heat content is the canonical ENSO *precursor*: in the
recharge-oscillator picture the equatorial band charges with warm water
before an El Niño discharges it poleward, and WWV leads eastern-Pacific SST
by roughly two to three seasons (Meinen & McPhaden 2000). The dashboard
currently shows ENSO's present (daily indices) and its forecast (model
plumes) but nothing about the *build-up* that constrains what next year can
do. This panel is the leading indicator that foreshadows the plume — e.g.
post-peak discharge visibly caps how long an event can persist.

## Where it sits

ENSO Forecast tab, section 04 ("In context"), next to the historical record —
or immediately after the plume in section 02 as forecast context.

## Data provenance

Exact retrieval timestamp and sha256 for the cached download are in
`data/MANIFEST.json` (retrieved 2026-09-07 UTC, at repo commit `dc1fb33`).

| Product | Publisher | Exact URL | Series | Cadence | Licence/terms | Type |
|---|---|---|---|---|---|---|
| Warm Water Volume index (T > 20 °C volume, 5°N–5°S, 120°E–80°W) | NOAA/PMEL, GTMBA Project Office | `https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat` (documentation: `https://www.pmel.noaa.gov/tao/wwv/`) | 1980-01 → 2026-07 at retrieval | Monthly (published in arrears) | US-Gov work; no explicit licence stated (public domain) | **Derived from ocean analyses** (gridded subsurface temperature analyses; not a direct observation) |
| Niño 3.4 anomaly | NOAA CPC via this repo | committed `ENSO/data/observed/nino34_monthly.csv` (upstream `sstoi.indices`) | 1982-01 → present | Daily cron | as above | Observation-based SST analysis |
| **Verified alternative (not plotted)**: equatorial upper-300 m temperature anomaly, 3 longitude bands incl. 130°E–80°W | NOAA CPC | `https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ocean/index/heat_content_index.txt` | 1979-01 → present | Monthly | as above | **Ocean-reanalysis-derived** (GODAS), °C anomaly vs 1981–2010 |

Reference: Meinen, C. S., and M. J. McPhaden (2000): Observations of Warm
Water Volume Changes in the Equatorial Pacific and Their Relationship to
El Niño and La Niña. *J. Climate*, **13**, 3551–3559.

## Implementation sketch

Interactive-first, exactly as the peak-intensity lollipop shipped (commit
`38719b3`; static PNGs followed later in `ad0d69d`):

1. Tiny fetcher for `wwv.dat` (~40 KB text) alongside the observed fetchers
   (`ENSO/enso_forecast/fetchers/` or `src/`), monthly-gated like the other
   monthly sources in `run.py`, cached under `ENSO/data/observed/`.
2. `create_wwv_panel()` in `src/enso_plots.py` (this script's
   `build_figure` is the prototype; templates via `src.theme`).
3. `L.panel(...)` in the ENSO tab layout, own chain-head callback; one
   `graph(True, ...)` row in `toggle_interactive_mode`; no image outputs
   until PNGs are promoted.
4. CSV export in `src/export_data.py` (→ `assets/data/wwv_nino34.csv`).
5. Units note: only the Niño 3.4 axis is °C; the WWV axis is 10¹⁴ m³. The
   dashboard's °F toggle is safe here — `src/units.py` converts only axes
   whose titles mention °C.

## Open questions for the maintainer

1. **Which product**: PMEL WWV (volume; the index the literature names) or
   CPC's upper-300 m temperature anomaly (verified alternative above; keeps
   the whole panel in °C and adds west/east bands)? Or both as a toggle?
2. Window: 1990+ (as mocked, matching the historical-context panel) or the
   full 1980+ record?
3. Would a lag-shifted WWV trace (shifted forward by the annotated lead) be
   clearer than the correlation annotation?
4. PMEL also publishes sub-basin variants (western/eastern equatorial
   Pacific) on the same data page — worth a west/east split view?
5. Update cadence: WWV posts in arrears (Jul 2026 was latest at retrieval) —
   acceptable lag for an "In context" panel?
