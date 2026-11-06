# Panel proposals — a menu, not a bundle

**Every panel here stands alone.** Each numbered folder is a self-contained
proposal — its own README, provenance, build script, cached data, and rendered
figures — and any subset can be adopted without the rest. The smallest useful
contribution is **panel 01 by itself** (one new option on an existing figure,
fed by a series the repo already maintains). The tab mocked up in
`00-enso-observations-tab/` is optional packaging for panels 01–03, nothing
more: adopt one panel, three panels, or the tab, in any order.

## Why a tab (if you want one)

The dashboard's ENSO tab is a *Forecast* tab — model plumes, distributions,
probabilities. Panels 01–03 are pure *observations*, and together they tell
one physical story in the order the coupled system works:

1. **Surface** — is the tropical ocean running warm, and how does this year
   compare with every other?
2. **Atmosphere** — is the pressure see-saw tipped and are the trades
   cooperating, i.e. is the event actually coupled?
3. **Subsurface** — how much warm water is banked below the surface — the
   classic constraint on what the coming seasons can do?

Pairing an **ENSO Observations** tab with the existing **ENSO Forecast** tab
gives readers the now/next split the operational centres use. The mock-up
(`00-enso-observations-tab/enso-observations-tab.html`) shows the five-tab
topbar; the placement is a suggestion, and the three panels would work equally
well as new sections on the existing ENSO Forecast tab.

## The panels

| Folder | One-line summary (from the panel captions) |
|---|---|
| `01-tropical-sst/` | Tropical-band (20°S–20°N) daily SST anomaly vs a centred 30-year climatology, from OISST v2.1 — is the wider tropical ocean unusually warm for its era? |
| `02-soi-trade-winds/` | The station-based Southern Oscillation Index over reanalysis-derived 850 hPa trade-wind anomalies — has the pressure see-saw tipped, and is the atmosphere coupling? |
| `03-wwv-heat-content/` | PMEL's Warm Water Volume (water warmer than 20 °C, 5°N–5°S, 120°E–80°W) vs Niño 3.4 — subsurface heat leads the surface by two to three seasons. |
| `04-forecast-skill-panel/` | Proposed separately, for the ENSO **Forecast** tab: surface the EC46 and ENSO plume verification figures the pipeline already renders into `forecast_skill/`. |

## Implementation note

All three data panels follow the repo's established add-a-panel pattern:
**interactive-only first**, exactly as the peak-intensity lollipop shipped
(commit `38719b3`, static PNGs added later in `ad0d69d`) — a plot function in
`src/enso_plots.py`, an `L.panel(...)` in the layout, a chain-head callback,
and a CSV export. Adding a tab has direct precedent in the Warming Map tab
(commit `a637565`: nav item, content div, tab-switch callback entries).
Panel 01 needs no new fetcher at all; 02 and 03 each need one tiny text-file
fetcher on the existing monthly-gated pattern in `run.py`.

## Consolidated open questions

**01 — tropical SST** · Region-toggle option or standalone panel? ·
Era-relative anomalies (as mocked; upstream `cab51d4` has since adopted this
convention for the daily cards) or fixed 1991–2020 baseline? · Tie the caption
to RONI ("the belt RONI subtracts") or stay neutral? · Offer an Absolute
display too? · Better highlight years than 1997/2015 (e.g. 2024)?

**02 — SOI + trade winds** · CPC SOI (used, 1951+) or BoM's longer series
(endpoint 404s from this environment; different standardisation and licence)?
· All three wind zones or central Pacific only? · Window 2015+ or full 1979+?
· Confirm the easterly-positive sign reading against the CPC Climate
Diagnostics Bulletin before captions go live · Standardised vs raw-anomaly
SOI, or the 3-month running SOI? · Niño 3.4 overlay on by default?

**03 — warm water volume** · PMEL WWV (the literature's index) or CPC's
upper-300 m temperature anomaly (verified alternative; keeps the panel in °C)
— or both as a toggle? · Window 1990+ or full 1980+? · Lag-shifted WWV trace
instead of the correlation annotation? · West/east sub-basin split? · Is the
publish-in-arrears lag acceptable for an observations panel?

**04 — forecast skill** · Are these figures off-dashboard **deliberately**
(code comments say "intentionally NOT wired into the dashboard layout")? ·
Sections on the existing tabs or a dedicated verification page? · Keep the
matplotlib renders or re-render via the Plotly templates for theme parity? ·
Is the current archive depth enough to publish? · Image weight on mobile?
