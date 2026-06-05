# ENSO Nino3.4 Forecast Tool: Methodology

## Overview

This tool aggregates ENSO (El Nino-Southern Oscillation) Nino3.4 sea surface temperature (SST) anomaly forecasts from four major operational forecasting systems (CFSv2, NMME, C3S, and CanSIPS), combines them into a unified dataset with consistent anomaly baselines, and produces a suite of comparison visualizations. The goal is to provide a comprehensive, multi-model view of the current ENSO forecast that accounts for structural uncertainty across independent modeling centers.

## Data Sources

### 1. NOAA CFS v2 (Climate Forecast System version 2)

- **Provider**: NOAA Climate Prediction Center (CPC)
- **URL**: `https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/nino34Mon.nc`
- **Data type**: Pre-computed Nino3.4 SST anomaly index (no spatial processing needed)
- **Ensemble**: 41 members from the E3 (latest) rolling ensemble window. Empirical inspection of successive file snapshots indicates the window is ~8 days wide with ~5 new runs per day, and member indexing runs newest (index 1) → oldest (index ~40). Member 41 appears to be a stable control/reference slot that does not rotate.
- **Lead time**: 9 months beyond the initialization month
- **Anomaly baseline**: 1991-2020
- **Update frequency**: Continuously (4–5 runs per day; we use the latest E3 file)
- **Initialization detection**: The NetCDF file contains both historical verification months (all members identical, zero spread) and true forecast months (members diverge). The effective initialization date is determined automatically as one month before the first month with non-zero ensemble standard deviation.

### 2. NMME (North American Multi-Model Ensemble)

- **Provider**: NOAA CPC, hosted at `ftp.cpc.ncep.noaa.gov`
- **URL pattern**: `https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/{MODEL}/{YYYYMM}0800/{MODEL}.tmpsfc.{YYYYMM}.anom.nc`
- **Data type**: Gridded SST anomaly fields (1-degree resolution); Nino3.4 index computed as a cosine-latitude-weighted spatial mean over 5S-5N, 190-240E
- **Models and ensemble sizes** (current operational feed):
  - NCEP-CFSv2: 32 members
  - ECCC-CanESM5: 20 members
  - ECCC-GEM5.2-NEMO: 20 members
  - NCAR-CESM1: 10 members
  - NCAR-CCSM4: 10 members
  - NASA-GEOS-S2S-2: 10 members
  - Total: ~102 members across 6 models (before deduplication)
  - After dedup, only the three NCAR/NASA models are retained from NMME (see Model Deduplication below). The CFSv2 and ECCC models are sourced elsewhere.
- **Lead time**: 8–9 months beyond the initialization month
- **Anomaly baseline**: Model-specific (each model's own hindcast climatology, nominally standardized by CPC)
- **Update frequency**: Monthly (approximately the 8th of each month)
- **Time coordinate handling**: NMME uses "months since 1960-01-01" units, which xarray cannot decode automatically. Times are decoded manually.

### 3. C3S (Copernicus Climate Change Service) Seasonal Forecasts

- **Provider**: ECMWF Copernicus Climate Data Store (CDS)
- **Dataset**: `seasonal-postprocessed-single-levels`
- **API**: CDS API (`cdsapi` Python package) with user-provided API key
- **Data type**: Gridded SST anomaly fields; Nino3.4 computed as cosine-latitude-weighted spatial mean over the requested sub-region (5N-5S, 170W-120W)
- **Models and ensemble sizes** (April 2026 initialization):
  - ECMWF (system 51): 51 members
  - Meteo-France (system 9): 51 members
  - DWD (system 22): 50 members
  - CMCC (system 4): 50 members
  - ECCC (system 5): 20 members
  - NCEP (system 2): 124 members
  - BOM (system 2): 110 members
  - UKMO (system 604): 60 members
  - JMA (system 503): 155 members
  - Total: ~671 members across 9 models (before deduplication)
- **Lead time**: 6 months beyond the initialization month
- **Anomaly baseline**: 1993-2016 (C3S hindcast climatology period), adjusted to 1991-2020 (see below)
- **Update frequency**: Monthly
- **forecastMonth convention**: In C3S, `forecastMonth=1` corresponds to the initialization month itself (e.g., for a April 2026 initialization, forecastMonth=1 = April 2026, forecastMonth=2 = May 2026, etc.)
- **Availability**: UKMO and JMA are sometimes delayed relative to the other centers; they may be missing for the first few days of a new initialization cycle. When a model is absent for the current init date, it is simply excluded.

### 4. CanSIPS (Canadian Seasonal to Interannual Prediction System)

- **Provider**: Environment and Climate Change Canada (ECCC), via the MSC Datamart
- **Data type**: Gridded SST forecasts (GRIB2) distributed as per-lead-month files; a pre-computed Nino3.4 index CSV is also fetched for quality control
- **URL pattern (GRIB2)**: `https://dd.weather.gc.ca/today/model_cansips/100km/forecast/{YYYY}/{MM}/{YYYYMM}_MSC_CanSIPS_WaterTemp_Sfc_LatLon1.0_P{LL}M.grib2`
- **URL pattern (CSV)**: `https://dd.weather.gc.ca/today/ensemble/cansips/csv/indices/forecast/monthly/{YYYYMMDD}00_indices_month_{YYYYMM_START}_{YYYYMM_END}.csv`
- **Initialization convention**: End-of-month (e.g., init_date = 2026-03-31 represents the April 2026 cycle). If the most recent end-of-month file is not yet posted, the previous month's file is used.
- **Models and ensemble sizes**:
  - CanSIPS-GEM-NEMO: 20 members (members 1-20 in the source file)
  - CanSIPS-CanESM5: 20 members (members 21-40 in the source file)
  - Total: 40 members across 2 models
- **Lead time**: 12 months beyond the initialization month (longest in the aggregated dataset)
- **Anomaly baseline**: 1991-2020 (matches reference baseline; no adjustment applied)
- **Update frequency**: Monthly
- **Role**: CanSIPS is fetched directly from ECCC as a standalone source and is used in place of the NMME ECCC-CanESM5 / ECCC-GEM5.2-NEMO streams, which originate from the same physical models but use an older initialization cadence in the NMME feed.

### 5. Observed Nino3.4

- **Provider**: NOAA CPC
- **Monthly Nino3.4**: `https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices` (OISSTv2-based)
- **ONI (Oceanic Nino Index)**: `https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt` (3-month running mean, ERSSTv5-based)
- **RONI (Relative ONI)**: `https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt` (3-month running mean of Niño 3.4 SSTA minus tropical-mean (20°S-20°N) SSTA, ERSSTv5-based)
- **Anomaly baseline**: 1991-2020 (centered 30-year base period)
- **Note on file format**: The sstoi.indices file orders the Nino regions as NINO1+2, NINO3, NINO4, NINO3.4 (NINO4 comes before NINO3.4), with SST and anomaly as separate columns. The NINO3.4 anomaly is the 10th column (index 9).

## Anomaly Baseline Harmonization

Different sources compute anomalies relative to different climatological base periods:

| Source | Native baseline | Adjustment applied |
|--------|----------------|-------------------|
| CFS v2 | 1991-2020 | None (reference baseline) |
| NMME | Model-specific | None (assumed approximately consistent) |
| C3S | 1993-2016 | Adjusted to 1991-2020 |
| CanSIPS | 1991-2020 | None (reference baseline) |
| Observed | 1991-2020 | None (reference baseline) |

### C3S Baseline Adjustment Method

The C3S anomalies are adjusted from the 1993-2016 baseline to 1991-2020 using the observed Nino3.4 record:

1. For each calendar month (January through December), compute the mean of observed Nino3.4 anomalies (which are on the 1991-2020 baseline) over the 1993-2016 period
2. This mean equals `climatology_1993-2016 - climatology_1991-2020` for that month
3. Add this offset to each C3S anomaly value for the corresponding target month

The adjustment is small (approximately -0.05 to -0.09 C for most months, annual mean -0.05 C), reflecting that the 1993-2016 period was slightly cooler than 1991-2020 in the Nino3.4 region. After adjustment, C3S and CFS ensemble means agree to within ~0.01-0.12 C at overlapping forecast months, consistent with expected inter-model variability.

Residual baseline offsets of ~0.1 C may remain, particularly for NMME models whose exact baseline periods are model-specific and not independently adjusted.

## Relative ONI (rONI)

The dashboard supports a toggle between the standard Niño 3.4 anomaly (ONI) and the **Relative ONI (rONI)**, defined as

```
rONI = Niño 3.4 SST anomaly − tropical-mean (20°S–20°N) SST anomaly
```

Subtracting the tropical-mean SST anomaly removes background tropical warming so ENSO events from different decades can be compared on equal footing. As tropical mean SSTs continue to warm under climate change, ONI values are biased upward relative to the strength of the underlying ENSO event; rONI corrects for that bias and is the preferred index when judging the dynamical character of an event (see L'Heureux et al. 2024, J. Climate, DOI 10.1175/JCLI-D-23-0406; van Oldenborgh et al. 2021).

Thresholds remain ±0.5 °C — the same numerical thresholds used for ONI — since rONI is in °C and uses the same 1991–2020 baseline.

### Per-source rONI strategy

Each forecast and observed record carries three columns: `nino34_anom`, `tropical_mean_anom`, and `roni_anom`. Following L'Heureux et al. (2024, *J. Climate*), `roni_anom = (nino34_anom − tropical_mean_anom) × a(m)`, where `a(m) = σ(ONI) / σ(ONI − TropAve)` is computed from ERSSTv5 over 1950–2020 and stratified by target calendar month (monthly indices) or season-center month (3-month running means). The scaling restores the variance of the relative index to match that of ONI, so the same ±0.5/±1.5 °C classification thresholds apply. CFSv2's published `rnino34Mon.nc` is already scaled upstream by NOAA and is passed through unchanged. The strategy for obtaining the tropical mean differs by source:

| Source | rONI strategy |
|--------|--------------|
| **CFSv2** | Fetch the published rNino3.4 file `rnino34Mon.nc` from `dataInd{1,2,3}/`, parse with the same NetCDF reader as `nino34Mon.nc`, and merge by (member, target_month, init_date). Tropical mean back-computed as `nino34_anom − roni_anom`. |
| **NMME** | Compute per-member tropical-mean SST anomaly from the cached gridded NetCDFs using a cosine-latitude-weighted mean over (20°S, 20°N) × all longitudes. |
| **C3S** | Widen the CDS request from the Niño 3.4 box to the full tropical band `[20, -180, -20, 180]` and compute per-member tropical mean from the same gridded files. The 1993–2016 → 1991–2020 baseline adjustment is applied to the tropical-mean anomaly as well, using the observed tropical-mean climatology offset. |
| **CanSIPS** | Compute per-member tropical-mean SST from the cached GRIB2 (already global 1°). Anomalies are derived using the persisted observed tropical anomaly as the climatology reference, since the GRIB files contain raw SST rather than anomalies. |
| **Observed** | Use NOAA's published RONI series (ERSSTv5, 1991–2020 base) directly: `https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt`. Seasonal RONI is linearly time-interpolated to monthly to align with the monthly Niño 3.4 index. |

### Verification

- For CFSv2 the published rNino3.4 anomaly is the L'Heureux-scaled value, so `roni_anom ≈ (nino34_anom − tropical_mean_anom) × a(m)` to rounding; the equality without `a(m)` no longer holds.
- Observed `roni_anom` should track NOAA's published RONI series within the interpolation noise.
- For recent target months tropical-mean anomaly should be on the order of +0.4–0.6 °C, reflecting present-day tropical warming relative to the 1991–2020 base period.

## Model Deduplication

Several physical models appear in multiple source databases. To avoid double-counting when computing multi-model statistics, the following deduplication rules are applied. These rules are implemented as the `MEGA_PLUME_DROP` set in `enso_forecast/visualize.py`:

| Physical Model | Appears in | Kept version | Rationale |
|---------------|-----------|-------------|-----------|
| NCEP CFSv2 | CFS, NMME, C3S | **CFS** (E3, 41 members) | Most current initialization; direct Nino3.4 output |
| ECCC CanESM5 | NMME, CanSIPS | **CanSIPS** (20 members, 12 leads) | Direct feed from ECCC; longer lead window (12 vs 8 months) |
| ECCC GEM5.2-NEMO | NMME, C3S, CanSIPS | **CanSIPS** (20 members, 12 leads) | Same physical model; CanSIPS is the authoritative direct feed |
| NMME overall mean | NMME | **Dropped** | Not an independent model; derived multi-model mean |

Specifically, the `MEGA_PLUME_DROP` set excludes these (source, model) pairs:

- `("NMME", "NCEP-CFSv2")` — use CFS source instead
- `("NMME", "NMME")` — derived mean, not a distinct model
- `("NMME", "ECCC-CanESM5")` — replaced by CanSIPS direct source
- `("NMME", "ECCC-GEM5.2-NEMO")` — replaced by CanSIPS direct source
- `("C3S", "NCEP")` — duplicate of CFSv2
- `("C3S", "ECCC")` — duplicate of CanSIPS-GEM-NEMO family

After deduplication, **13 unique models** remain with a total of **~637 individual ensemble members**.

### Deduplicated Model Inventory

| Model | Source | Members | Lead (months beyond init) |
|-------|--------|---------|---------------------------|
| CFSv2 | CFS | 41 | 9 |
| ECMWF | C3S | 51 | 6 |
| Meteo-France | C3S | 51 | 6 |
| DWD | C3S | 50 | 6 |
| CMCC | C3S | 50 | 6 |
| BOM | C3S | 110 | 6 |
| UKMO | C3S | 60 | 6 |
| JMA | C3S | 155 | 6 |
| CanSIPS-CanESM5 | CanSIPS | 20 | 12 |
| CanSIPS-GEM-NEMO | CanSIPS | 20 | 12 |
| NCAR-CESM1 | NMME | 10 | 8 |
| NCAR-CCSM4 | NMME | 10 | 8 |
| NASA-GEOS-S2S-2 | NMME | 10 | 8 |

Member counts reflect the April 2026 initialization cycle; they can vary month-to-month as models add or drop realizations.

## Equal-Weight Model Statistics

When computing multi-model summary statistics (means, medians, percentiles), each model is weighted equally regardless of its ensemble size. This prevents models with large ensembles (e.g., BOM with 121 members) from dominating the statistics.

For the multi-model mean, this is computed as the simple average of per-model ensemble means (one value per model per target month).

For weighted percentiles (used in the box-and-whisker and historical context plots), all individual ensemble members are included but each member receives a weight of `1/N`, where `N` is the number of members for that model in that target month. Weighted quantiles are then computed by:

1. Sorting all member values
2. Computing cumulative normalized weights
3. Interpolating to find the value at the desired quantile

This approach preserves the full distributional information from each model's ensemble while ensuring equal inter-model influence.

## CFS Verification Data Filtering

The CFS NetCDF files contain both historical verification months (where all ensemble members report the same observed value, resulting in zero ensemble spread) and true forecast months (where members diverge). The tool automatically identifies and separates these by computing the ensemble standard deviation per target month: months with std < 0.001 C are classified as verification and excluded from forecast plots.

## Figures Produced

The dashboard ENSO tab renders three figures, generated by `generate_enso_static_images` in `src/enso_plots.py` (dark + light variants of each):

### 1. Combined Forecast Plume (`enso_mega_plume_{mode}.png`)
All ensemble members from all sources (deduplicated, 13 models, ~637 members) in a single plume plot, alongside recent observed ONI for historical context. Members are color-coded by model, with per-model means as thicker lines and the multi-model median (model-weighted) as a dashed black line. Target months with fewer than 3 distinct models reporting are dropped. This is the most comprehensive view of the full forecast distribution.

### 2. Model-Weighted Box Distribution (`enso_box_distribution_{mode}.png`)
All ensemble members as colored dots with a box-and-whisker overlay per target month. Box statistics are computed from all individual members using model-equal weighting (each member weighted by 1/N where N is that model's ensemble size in that month), so the IQR and median reflect inter-model consensus rather than being dominated by large-ensemble models.

### 3. Historical Context (`enso_historical_{mode}.png`)
Observed monthly Nino3.4 index since ~1990 with red shading for El Niño episodes (anomaly > +0.5 °C) and blue shading for La Niña episodes (anomaly < -0.5 °C). The multi-model mean forecast is overlaid as a dashed line with a model-weighted 25th-75th percentile band, letting the current forecast be compared visually against the magnitude of past events (e.g. 1997-98, 2015-16 strong El Niños).

## Software and Dependencies

- Python 3.11+
- Data fetching: `requests`, `beautifulsoup4`, `cdsapi`
- Data processing: `xarray`, `netCDF4`, `pandas`, `numpy`
- Visualization: `matplotlib`
- Testing: `pytest`, `responses` (HTTP mocking)

## Limitations and Caveats

1. **Baseline residuals**: Despite the C3S adjustment, residual inter-source offsets of ~0.1 C may remain due to differences in underlying SST products and model-specific climatologies.

2. **Model independence**: Some models share components or heritage (e.g., NCAR-CESM1 and NCAR-CCSM4 share atmospheric components). Treating them as fully independent slightly overstates the effective number of independent forecasts.

3. **Unequal lead times**: Lead coverage beyond the initialization month varies by source: C3S 6 months, NMME 8 months, CFSv2 9 months, CanSIPS 12 months. At leads 7-9 only CFSv2, NMME, and CanSIPS contribute; at leads 10-12 only CanSIPS (two models) contributes. Multi-model statistics at long leads are therefore based on fewer models, and plots filter to target months with ≥3 models reporting.

4. **Late-arriving C3S models**: UKMO and JMA can be delayed in the CDS archive by a few days after the monthly initialization date; if missing at fetch time they are simply excluded from that cycle.

5. **No amplitude bias correction**: The NMME operationally applies an amplitude bias correction (dividing forecast anomalies by the ratio of model hindcast standard deviation to observed standard deviation, per model/lead/start-month). This correction is not applied here because hindcast data is not available for all current model versions. This means some NMME models may exhibit systematically too much or too little variability.

6. **IRI models excluded**: The IRI ENSO forecast (23 models, means only) is available in the tool but excluded from the current plots because its most recent update (February 2026) predates the March 2026 initialization of the other sources. It can be included with `--include-iri`.
