# Proposal 02 — SOI and equatorial 850 hPa trade-wind anomalies

## What the panel shows

A two-row monthly figure adding the **atmospheric half of the coupled ENSO
system** to a tab that is currently all ocean:

- **Row 1 — Southern Oscillation Index** (standardized Tahiti − Darwin
  sea-level pressure) as sign-colored bars. *Station-based observation.*

  > **Proposed panel caption:** The Southern Oscillation Index is the
  > standardised sea-level-pressure difference between Tahiti and Darwin — a
  > station-based observation, plotted here from the NOAA CPC series that
  > begins in 1951. Strongly negative values mean the pressure see-saw has
  > tipped and weakened trade winds are reinforcing El Niño.
- **Row 2 — Equatorial Pacific 850 hPa trade-wind anomalies** for the three
  longitude zones CPC publishes — West (135°E–180°), Central (175°W–140°W),
  East (135°W–120°W), all 5°N–5°S — with the observed Niño 3.4 anomaly (°C)
  overlaid dashed on a secondary axis. *Reanalysis-derived indices.*

  > **Proposed panel caption:** Trade-wind anomalies — the zonal wind at
  > 850 hPa over the equatorial Pacific from the NCEP/NCAR (CDAS) atmospheric
  > reanalysis, with positive values meaning stronger easterlies. Sustained
  > westerly anomalies (negative here) push warm water eastward and show the
  > atmosphere coupling to a developing event.

Mock-up window: 2015 → present (both series extend further back; see open
questions).

## Why it adds value

ENSO is a coupled phenomenon; the dashboard currently shows only the SST
side. The pressure seesaw and the trades confirm — or contradict — what SST
implies: a warm Niño 3.4 with trades intact reads very differently from one
with collapsed trades and a strongly negative SOI. Wind anomalies also lead
SST during event onset (westerly bursts), giving the plume panels useful
context. This is the standard monitoring pairing in CPC's and BoM's own ENSO
briefings, absent from the dashboard.

## Where it sits

ENSO Forecast tab, section 04 ("In context"), directly under the historical
record panel — observed atmosphere next to observed ocean. Alternative: a new
"Coupled-system check" panel in section 01.

## Data provenance

Exact retrieval timestamps and sha256 checksums for every cached file are in
`data/MANIFEST.json` (retrieved 2026-09-07 UTC, at repo commit `dc1fb33`).

| Product | Publisher | Exact URL | Series | Cadence | Licence/terms | Type |
|---|---|---|---|---|---|---|
| SOI, standardized (Tahiti − Darwin SLP) | NOAA CPC | `https://www.cpc.ncep.noaa.gov/data/indices/soi` (STANDARDIZED DATA block) | 1951-01 → present | Monthly | US-Gov work; no explicit licence stated (public domain) | Station-based **observation** |
| 850 hPa trade-wind index, West Pacific (135°E–180°, 5°N–5°S) | NOAA CPC | `https://www.cpc.ncep.noaa.gov/data/indices/wpac850` (ANOMALY block) | 1979-01 → present | Monthly | as above | **Reanalysis-derived** (CDAS/NCEP–NCAR; 1981–2010 base per CPC `Readme.index.shtml`) |
| 850 hPa trade-wind index, Central Pacific (175°W–140°W) | NOAA CPC | `https://www.cpc.ncep.noaa.gov/data/indices/cpac850` (ANOMALY block) | 1979-01 → present | Monthly | as above | as above |
| 850 hPa trade-wind index, East Pacific (135°W–120°W) | NOAA CPC | `https://www.cpc.ncep.noaa.gov/data/indices/epac850` (ANOMALY block) | 1979-01 → present | Monthly | as above | as above |
| Niño 3.4 anomaly overlay | NOAA CPC via this repo | committed `ENSO/data/observed/nino34_monthly.csv` (upstream `sstoi.indices`) | 1982-01 → present | Daily cron | as above | Observation-based SST analysis |
| **BoM SOI — NOT VERIFIED** | Bureau of Meteorology | `bom.gov.au/climate/enso/soi_monthly.txt` returned **404** from this environment (2026-08-13); not used, not approximated | — | — | BoM content is Crown copyright (typically CC BY where stated) | station-based observation |

Format note: CPC index files are fixed-width (YEAR + twelve 6-character
fields) and are parsed as such — the current year's `-999.9` sentinels
collide with neighbouring values under whitespace splitting (e.g.
`-6.1-999.9`). Sign convention verified against the data: 1997/2015/2026
El Niño years strongly negative, 2010/2020 La Niña years positive, ORIGINAL
blocks positive ≈5–11 m s⁻¹ — i.e. the index reports easterly trade speed,
positive anomaly = stronger trades.

## Implementation sketch

Interactive-first, exactly as the peak-intensity lollipop shipped (commit
`38719b3`; static PNGs followed later in `ad0d69d`):

1. Small fetcher (tiny text files, ~6 KB each) alongside the existing
   observed fetchers — e.g. `ENSO/enso_forecast/fetchers/atmos_indices.py`
   with `save_atmos_indices()`, registered in `update_enso_forecasts` in
   `run.py`; the files update monthly at CPC so the daily fetch is cheap and
   idempotent, cached under `ENSO/data/observed/`.
2. `create_soi_winds()` in `src/enso_plots.py` (this script's
   `build_figure` is the prototype; templates via `src.theme`).
3. `L.panel(...)` in the ENSO tab layout in `src/dashboard.py`, own
   chain-head callback (no PNG initially → no `update_image_sources`
   entries; one `graph(True, ...)` row in `toggle_interactive_mode`).
4. CSV export `export_soi_winds()` in `src/export_data.py`, registered in
   `generate_all_csv_exports` (→ `assets/data/soi_trade_winds.csv`).

## Open questions for the maintainer

1. **SOI source**: CPC standardized SOI (used here, 1951+) or BoM's series
   (1876+, different standardization)? The BoM `soi_monthly.txt` endpoint
   404s from this environment — if BoM is preferred, a working endpoint and
   its licence terms need confirming.
2. Zones: all three wind indices, or only the central Pacific (the classic
   westerly-burst zone) to reduce clutter?
3. Window: 2015+ (as mocked) or the full 1979+ record with a range slider?
4. The easterly-positive sign reading is verified empirically above — worth
   confirming against the CPC Climate Diagnostics Bulletin text before
   captions go live.
5. SOI block: standardized (used) vs raw anomaly; and should bars use the
   3-month running SOI (`soi.3m.txt`) instead?
6. Keep the Niño 3.4 overlay on by default, or make it a toggle like the
   dashboard's other optional layers?
