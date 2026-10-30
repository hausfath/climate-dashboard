# Proposal 04 — Surface the forecast-skill figures

## What the panel shows

A "How good are these forecasts?" section built entirely from figures the daily
pipeline **already renders and commits** into `forecast_skill/`:

- `ec46_skill.png` — every archived ECMWF EC46 initialization replayed against
  ERA5 observed global temperature, using the same transform as the dashboard's
  46-day forecast tail — since upstream `fae45a1` a **lead-decaying bias
  correction** (replacing the earlier constant lead-0 correction), with the fit
  hardened and its provenance exposed in `2f8ebc7`; see `src/ec46_skill.py`.

  > **Proposed panel caption:** Verification of the dashboard's 46-day global
  > temperature forecasts against subsequent ERA5 observations across all
  > archived initialisations. It shows how far ahead the daily forecast has
  > earned trust.
- `enso_skill_{oni,roni}_{lines,plumes,hybrid}.png` — each frozen monthly
  multi-model ENSO plume against observed monthly-mean Niño 3.4 / RONI
  (see `src/enso_skill.py`). The mock-up proposes the **hybrid** style as the
  page default, with the other styles left as downloads.

  > **Proposed panel caption:** Verification of archived Niño 3.4 forecast
  > plumes against observed outcomes. It shows how well the multi-model
  > ensemble has anticipated events so far.

`skill-panel.html` is a static mock-up in the `mockups/` convention
(`../../mock.css`, observatory dark theme, ENSO teal accent) embedding copies
of the current PNGs from `images/`. The committed originals are untouched.

## Why it adds value

The dashboard leads with projections (annual prediction, EC46-blended monthly
projection, multi-model ENSO plume) but shows nothing about how past editions
of those forecasts fared. The verification artefacts exist, update daily, and
are committed — they are simply not linked. Surfacing them (a) answers the
first sceptical question any new reader has, (b) differentiates the dashboard
from sites that show forecasts without hindsight, and (c) costs no new data,
no new pipeline step, and no new dependencies.

## Where it sits

Proposed: a new final section on the **ENSO Forecast tab** (section 05, "How
good are these forecasts?") holding the two ENSO panels, plus the EC46 panel
appended to the Global Temperature tab's section 02 ("Where 2026 is heading").
The mock-up shows all three together on one page for review convenience.
Alternatives for the maintainer: a dedicated "Verification" tab, or a
footer-linked page.

## Data provenance

No new external data. The embedded artefacts are repo-internal derivatives:

| Field | Value |
|---|---|
| Product | EC46 skill plot + 6 ENSO plume-verification plots (PNG copies under `images/`) |
| Publisher | This repository's daily update pipeline (`run.py update` via `.github/workflows/update-data.yml`, 06:00 UTC) |
| Exact source | `forecast_skill/ec46_skill.png`, `forecast_skill/enso_skill_{oni,roni}_{lines,plumes,hybrid}.png` at commit `dc1fb33` (2026-09-06) |
| Retrieved | Copied 2026-09-07 (the EC46 plot's own title carries the live init count; archive spans 2026-05-14 → 2026-09-06 at copy time) |
| Series span | EC46: inits since 2026-05-14 (archive start). ENSO plumes: monthly archive since 2026-03 |
| Update cadence | EC46 plot: daily. ENSO plots: regenerated daily, gain a new plume monthly once the month's runs are in |
| Licence / terms | Repository MIT licence; underlying inputs are NOAA/ECMWF-derived data already used elsewhere on the dashboard |
| Type | Derived verification figures (model forecasts vs observation/reanalysis-based truths: ERA5, OISSTv2.1, ERSSTv5) |

## Implementation sketch (repo's add-a-panel pattern)

The zero-new-code option: the PNGs are already committed, so serving them only
needs them exposed via the assets pipeline and referenced by `L.panel(...)`
blocks:

1. Have the daily cron copy the three chosen PNGs into `assets/images/`
   (one-line addition where `run.py update` already regenerates images), or
   add a small Flask static route for `forecast_skill/`.
2. Add `L.panel(...)` entries — ENSO tab section 05 and Global tab section 02
   in `src/dashboard.py` — using `img_id`/`img_src` only (static-image-only
   panels, like the ridgeline: no `graph_id`, no callback, no
   `toggle_interactive_mode` outputs needed beyond image visibility).
3. Optional later upgrade, mirroring how the peak-intensity lollipop shipped
   interactive-first and gained PNGs afterwards (commits `38719b3` →
   `ad0d69d`): port the two headline figures from matplotlib to Plotly using
   the registered `climate_dark`/`climate_light` templates, giving dark/light
   parity and hover — at which point they join the standard chained-callback
   and CSV-download patterns (`src/export_data.py`).

## Open questions for the maintainer

1. **Are these figures off-dashboard deliberately?** `README.md` ("committed
   by the daily cron; not shown on the dashboard") and the module docstrings
   in `src/ec46_skill.py` / `src/enso_skill.py` ("intentionally NOT wired into
   the dashboard layout") read as a conscious choice — is that reluctance
   (clutter, audience) or just not-yet?
2. Placement: sections on the existing tabs (as mocked) or a dedicated
   verification tab/page?
3. Keep matplotlib renders (white background, no dark variant — the mock-up
   frames them in a light well) or re-render via the Plotly templates for
   theme parity before surfacing?
4. The ENSO verification archive only starts 2026-03 and EC46 2026-05 — is a
   ~6-month record enough to publish, or wait for more months to accumulate?
5. Weight: the seven PNGs total ~1.6 MB; mobile users default to static mode.
   Embed only the hybrid style (as proposed) or fewer/smaller images?
