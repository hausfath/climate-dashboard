# Proposal 01 — Tropical-mean daily SST year-lines

## What the panel shows

Daily mean sea-surface temperature of the full tropical belt (20°S–20°N, all
longitudes) as year-lines by day-of-year — every year since 1982 in gray,
1997 and 2015 highlighted, the current year bold — in exactly the style of
the existing daily Niño year-lines figure
(`src/enso_plots.py::create_nino34_daily_years`). Anomalies use the same
era-relative convention as that figure: each year measured against a centered
30-year day-of-year climatology (clamped at the record edges, current year
excluded, 15-day circular smooth), so the long-term warming trend is removed
and years compare as *states* across the record.

## Why it adds value

The tropical-belt mean is literally the RONI denominator: RONI = Niño 3.4
anomaly − tropical-mean anomaly, variance-scaled. The dashboard shows the
result of that subtraction but never the subtrahend. This panel makes the
background tropical ocean state visible — how warm the belt is *for its era*,
how ENSO's imprint spreads into it (the repo's own CanSIPS teleconnection fit:
the belt lags Niño 3.4 by ~2 months at ~0.19 °C/°C — `TROPICS_ENSO_SLOPE` in
`ENSO/enso_forecast/fetchers/cansips.py`), and whether the current year is
riding unusually high the way 1997/2015 did.

## Where it sits

ENSO Forecast tab, section 01 ("Right now"), alongside the existing daily
year-lines panel. Two placements are viable — see open questions:
(a) a fifth option in the existing region toggle (1+2 / 3 / 3.4 / 4 /
**Tropics**), or (b) a standalone companion panel.

## The existing series — definition and reuse decision

**Reused: yes.** The repo already maintains exactly this series; no new fetch
is needed, and none was performed.

| Aspect | Definition (from `src/nino_daily.py`) |
|---|---|
| Product | NOAA OISST v2.1 daily SST (0.25° analysis) |
| Spatial box | `TROPICS_BOX = (-19.875, 19.875, 8, -179.875, 179.875, 16)`: 20°S–20°N, all longitudes, sampled on grid-cell centers with lat stride 8 (2°) and lon stride 16 (4°) — cos(lat)-weighted mean; identical strided sampling for climatology and daily values so subsampling bias cancels |
| Committed series | `data/nino34_daily_history.csv`, `tropics` column — absolute box-mean SST, 1981-09-01 → present, topped up daily by the cron (`_merge_into_history`) |
| Fixed climatology | 1991–2020 day-of-year, 15-day circular smooth (`data/nino34_daily_climatology.csv`, `tropics_clim`) — used for the dashboard's published daily RONI |
| This figure's anomaly | `era_relative_anomalies(..., region='tropics')` — centered 30-yr day-of-year climatology per year, windows clamped, current year excluded, 15-day circular smooth (the ONI convention; already supports `region='tropics'` unchanged) |
| Leap days | Feb 29 mapped onto Feb 28's day-of-year (`_doy_key`) |

Since this proposal was drafted, upstream commit `cab51d4` has adopted this same era-relative convention for the dashboard's daily Niño/RONI cards — the anomaly definition used here now matches the live site's.

## Data provenance

| Field | Value |
|---|---|
| Product | NOAA OISST v2.1 daily SST — tropical-belt (20°S–20°N) cos-weighted box mean |
| Publisher | NOAA NCEI (product); NOAA CoastWatch ERDDAP (service); this repo's daily cron (derived series) |
| Exact URL | Upstream: `https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180` (final) + `.../ncdcOisst21NrtAgg_LonPM180` (near-real-time); fallback `https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/`. Used here: committed `data/nino34_daily_history.csv` |
| Retrieved | Cache extracted from the committed CSV at repo commit `dc1fb33`; extraction timestamp in `data/MANIFEST.json` (2026-09-07 UTC). No network fetch performed |
| Series span | 1981-09-01 → 2026-09-05 (figure plots complete years 1982+, per `FIRST_NINO_YEAR`) |
| Update cadence | Daily — 06:00 UTC workflow + 15:00 UTC afternoon top-up |
| Licence / terms | NOAA observational data (U.S. Government work, public domain); no explicit licence text on the endpoints |
| Type | Observation-based gridded analysis (AVHRR satellite + in-situ optimum interpolation); anomaly convention derived in-repo |

## Implementation sketch

Lowest-friction path — the data column and the anomaly function already work:

1. Add a `Tropics` option to `nino-region-toggle` in `src/dashboard.py` and a
   `'tropics': 'Tropics 20°S–20°N'` entry to `NINO_REGION_LABELS` in
   `src/enso_plots.py`. `create_nino34_daily_years(region='tropics')` then
   works as-is: the history CSV already carries the column, and the figure
   already omits ONI category bands for non-3.4 regions.
2. Caption branch in `update_nino34_daily_caption` (bands omitted; note the
   RONI-denominator relationship).
3. CSV download: add `tropics` to `export_nino_daily_years` in
   `src/export_data.py` (→ `assets/data/nino_daily_years_tropics.csv`).
4. Interactive-only at first, exactly as the peak-intensity lollipop shipped
   (commit `38719b3`), and as non-3.4 regions already behave (no pre-rendered
   PNG → the panel falls through to the interactive figure even in static
   mode). Static PNGs can follow later if promoted (cf. `ad0d69d`).

`make_figure.py` here renders the standalone-panel variant of the same figure
from a cached extract in `data/`; `--refresh` re-extracts from the committed
CSV. No network in either mode.

## Open questions for the maintainer

1. Region-toggle option or standalone panel? The toggle is nearly free; a
   standalone panel keeps the belt visible next to Niño 3.4 rather than
   behind a click.
2. Era-relative anomalies (trend removed — comparable ENSO states, matches
   the existing figure) or the fixed 1991–2020 climatology (retains the
   warming signal — a different, also interesting story)? This mock-up uses
   era-relative for stylistic consistency.
3. Should the caption explicitly tie this series to RONI ("this is the belt
   RONI subtracts") to teach the index, or stay neutral?
4. Offer the Absolute display toggle too (seasonal cycle retained), as the
   Niño regions have?
5. Highlight set: 1997/2015 are Niño-centric choices; for the belt itself,
   would 2024 (record global SSTs) be a better highlighted reference year?
