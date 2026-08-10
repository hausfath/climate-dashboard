# Temperature Projection: Methodology

This document describes how the dashboard's annual global temperature
prediction (the "2026 prediction" on the Global Temperature tab) is
constructed, how its uncertainty is estimated, and how it is updated and
tracked through the year — and, in the final section, how the **monthly
projection** (EC46 ensemble blend) works. The companion ENSO forecast methodology — whose
multi-model forecast feeds this prediction — is documented in
[`ENSO/METHODOLOGY.md`](ENSO/METHODOLOGY.md).

## Overview

Each day, the dashboard predicts the current calendar year's annual global
mean surface temperature anomaly (relative to 1850–1900 preindustrial) using
a linear regression trained on the historical relationship between
within-year observations, ENSO state, and the eventual annual mean. The
prediction is recomputed daily as new ERA5 data arrives and a snapshot is
appended to an immutable history file, so the evolution of the forecast over
the year — and the influence of ENSO forecast revisions on it — can be
audited after the fact.

## Input Data

| Input | Source | Role |
|-------|--------|------|
| Daily global 2 m temperature anomaly | ERA5 via ECMWF Climate Pulse (updated daily, ~2-day latency) | Predictand (annual mean) and within-year predictors |
| Observed ENSO (ONI) | NOAA CPC | Observed ENSO predictor for months already completed |
| Multi-model ENSO forecast | Dashboard's own multi-model system (CFSv2, NMME, C3S, CanSIPS — see ENSO methodology) | Forecast ENSO predictor for the remaining months of the year |

### Preindustrial baseline

ERA5 anomalies are published on a 1991–2020 baseline. They are converted to
the 1850–1900 preindustrial baseline using fixed per-calendar-month offsets
(`MONTHLY_PREINDUSTRIAL_OFFSETS` in `src/models_vs_obs.py`, ranging from
+0.80 °C in July–August to +0.96 °C in January–February). The same offsets
are used across the dashboard so daily, monthly, and annual values agree
between tabs.

## The Regression Model

A six-feature ordinary least squares regression (with standardized
features) predicts the annual mean anomaly:

| Feature | Description |
|---------|-------------|
| `year` | Calendar year (captures the secular warming trend) |
| `prior_year_anomaly` | Previous year's annual mean anomaly |
| `enso_obs` | Mean ONI over months of the year already completed |
| `enso_future` | Mean ONI over the remaining months (observed for training years; the model-weighted multi-model median forecast for the current year) |
| `trailing_anomaly` | Mean daily anomaly over the trailing 30 days |
| `ytd_anomaly` | Mean daily anomaly year-to-date |

Two design points worth noting:

- **Day-of-year consistency.** For every training year, `trailing_anomaly`,
  `ytd_anomaly`, and the observed/future ENSO split are computed *as of the
  same day-of-year as today*. The regression therefore learns "given what was
  knowable on day N of the year, what did the annual mean turn out to be?" —
  the model is effectively re-trained for each calendar day, and early-year
  predictions lean on ENSO and the prior year while late-year predictions are
  increasingly pinned by `ytd_anomaly`.
- **ENSO is split into observed and future components** so that the
  forecast part (with its larger uncertainty) can be perturbed independently
  in the Monte Carlo uncertainty analysis below.

### Training set

- Years **1950 to last complete year**, requiring all six features.
- **Volcanic years excluded**: 1982–83 (El Chichón) and 1991–93 (Pinatubo),
  whose stratospheric-aerosol cooling is not representable by these features.
- The current (incomplete) year is never used in training.

## Uncertainty

Two layers:

1. **Regression residual spread.** The standard deviation of in-sample
   residuals (floored at 0.02 °C) gives the baseline 1σ uncertainty. This
   shrinks naturally through the year as the YTD features pin down the
   annual mean (≈0.09 °C in January, ≈0.04 °C by mid-year).

2. **Monte Carlo over the ENSO ensemble** (`src/annual_prediction_mc.py`).
   The central prediction uses the multi-model median ENSO trajectory, which
   collapses forecast spread. To restore it, 10,000 draws sample individual
   ENSO ensemble members (equal weight per model, so a 155-member ensemble
   doesn't drown out a 10-member one; only members covering every remaining
   month of the year are eligible), recompute `enso_future` for each draw,
   propagate it through the regression, and add Gaussian residual noise.
   The reported interval is the 2.5–97.5 percentile of the draws. If the
   ENSO ensemble is unavailable, the interval falls back to analytic ±2σ.

## Daily Update and Projection Tracking

The daily pipeline (`run.py update`, executed by the Render cron and GitHub
Actions at 6 AM UTC):

1. Fetches the latest ERA5 daily series and ENSO forecasts (monthly sources
   are re-fetched only when a new initialization is actually available,
   keyed off the `init_date` inside each source's latest file).
2. Recomputes the prediction for any calendar days missing from
   `data/projection_history_{year}.csv` and appends them. **Past rows are
   never modified** — the file is an append-only audit trail of what the
   model said on each day, with columns: `date`, `prediction`,
   `uncertainty` (1σ), `ytd_anomaly`, `trailing_30d_anomaly`,
   `days_elapsed`.
3. Snapshots the multi-model ENSO forecast. If any forecast month moved by
   ≥0.1 °C versus the previous snapshot, the date is recorded in
   `data/enso_forecast_updates.csv`.

The **Projection evolution** plot draws the prediction history with a ±2σ
band and overlays star markers on the dates ENSO forecast revisions landed —
making it visible when a change in the annual prediction was driven by new
ENSO information rather than by incoming temperature data.

Both CSVs are committed to git by the daily cron, so the full
projection-evolution record is reconstructable from repository history.

## Monthly Projection (EC46 ensemble blend)

Since August 2026 the monthly projection ("Projected August: …" panel and
card) is a **blend of month-to-date observations and the ECMWF EC46
extended-range ensemble**, replacing a purely statistical regression (which
remains as an automatic fallback). Method (`src/monthly_blend.py`):

- **Data.** Daily global-mean 2 m temperature from the 50-member EC46
  ensemble (46-day horizon), sampled via the Open-Meteo seasonal API on a
  10°×15° cos-lat-weighted grid, archived per init under
  `forecast_skill/archive_members/`. Fetch-quality gates (≥40 members,
  ≥90 % gridpoint success with every latitude band present) refuse partial
  fetches rather than publish a biased mean.
- **Estimator.** The freshest init that starts no later than the day after
  the ERA5 obs edge is chosen, so observed and forecast days tile the month
  with one coherent member set (coherence over freshness; at most ~3 days
  stale, ≤6 by hard guard). Per member, the projected monthly mean is the
  mean of observed daily anomalies plus that member's remaining-day
  anomalies. Forecast anomalies are computed against an ERA5 1991–2020
  climatology sampled **on the identical coarse grid**
  (`data/ec46_coarse_clim_doy.csv`), so the grid-sampling operator cancels,
  plus a small anomaly-space lead-0 bias correction estimated from the
  archive. The published central value is the member median.
- **Uncertainty.** Ensemble spread alone cannot represent common-mode model
  error (all members drift together), so the published 5–95 % and 25–75 %
  ranges come from a total variance with four parts: member month-mean
  spread; a **shared-per-month model-error term** estimated from verified
  months in the archive and shrunk toward the legacy regression's residual σ
  (an 85-year prior; ν₀ = 4 pseudo-months) while the archive is short; the
  lead-0 bias-correction standard error (scaled by the forecast fraction of
  the month); and ERA5T revision uncertainty (scaled by the observed
  fraction). Half-widths use a Student-t multiplier with df = ν + ν₀ − 1,
  and until **six independent months have verified** the 5–95 % width is
  hard-capped to never be narrower than half the regression's ±2σ.
- **Verification loop.** The daily cron recomputes, for every archived init
  and verified month: blend error, CRPS, PIT, and spread (member inits),
  plus the pooled ensemble-mean error by lead day
  (`forecast_skill/monthly_blend_verification.csv`,
  `monthly_blend_lead_errors.csv`), and refreshes the calibration inputs
  (`monthly_blend_calibration.csv`). Interval widths tighten only as this
  archive demonstrates they should.
- **Fallback.** If no usable init exists, members are missing, or a lead-0
  sanity gate fails (|corrected day-0 forecast − obs| > 0.15 °C), the panel
  reverts to the regression of full-month means on month-to-date means
  (1940–present) with ±2σ residual intervals; the figure caption and the
  downloadable CSV record which method produced the displayed number.

Backtesting on June–July 2026 (56 archived inits) showed the blend matching
the regression's central accuracy overall (MAE 0.020 °C both) while
providing day-0 availability and flow-dependent, properly floored
uncertainties. The design was reviewed by independent statistics and
forecast-verification passes in August 2026; the shared-offset error term,
the t-multiplier, the narrowing cap, the coarse-grid climatology fix, and
the fetch gates all originate from that review.

## Limitations

1. **Linearity.** The regression assumes linear feature–response
   relationships; nonlinear ENSO–temperature coupling (e.g. asymmetry
   between El Niño and La Niña impacts) is not captured.
2. **ENSO forecast skill.** The `enso_future` feature inherits the biases of
   the multi-model ENSO forecast (see the ENSO methodology's limitations,
   including the absence of amplitude bias correction). The Monte Carlo
   interval reflects ensemble *spread*, not structural error shared by all
   models.
3. **Volcanic and other forcing shocks.** A major eruption (or other abrupt
   forcing change) during the forecast year would invalidate the prediction;
   such years are excluded from training precisely because the features
   cannot represent them.
4. **Training-period trend assumption.** The `year` feature extrapolates the
   1950–present warming structure; a sustained acceleration or deceleration
   of the underlying trend is only absorbed gradually.
5. **ERA5 revisions.** Climate Pulse daily values for the most recent days
   are preliminary and occasionally revised; snapshots in the projection
   history are *not* recomputed when revisions occur (by design — the file
   records what was knowable at the time).
