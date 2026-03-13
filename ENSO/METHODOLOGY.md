# ENSO Nino3.4 Forecast Tool: Methodology

## Overview

This tool aggregates ENSO (El Nino-Southern Oscillation) Nino3.4 sea surface temperature (SST) anomaly forecasts from three major operational forecasting systems, combines them into a unified dataset with consistent anomaly baselines, and produces a suite of comparison visualizations. The goal is to provide a comprehensive, multi-model view of the current ENSO forecast that accounts for structural uncertainty across independent modeling centers.

## Data Sources

### 1. NOAA CFS v2 (Climate Forecast System version 2)

- **Provider**: NOAA Climate Prediction Center (CPC)
- **URL**: `https://www.cpc.ncep.noaa.gov/products/CFSv2/dataInd3/nino34Mon.nc`
- **Data type**: Pre-computed Nino3.4 SST anomaly index (no spatial processing needed)
- **Ensemble**: 40 members from the latest 10-day initialization window (E3)
- **Lead time**: 9 months from initialization
- **Anomaly baseline**: 1991-2020
- **Update frequency**: Continuously (4 runs per day; we use the latest E3 file)
- **Initialization detection**: The NetCDF file contains both historical verification months (all members identical, zero spread) and true forecast months (members diverge). The effective initialization date is determined automatically as one month before the first month with non-zero ensemble standard deviation.

### 2. NMME (North American Multi-Model Ensemble)

- **Provider**: NOAA CPC, hosted at `ftp.cpc.ncep.noaa.gov`
- **URL pattern**: `https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/{MODEL}/{YYYYMM}0800/{MODEL}.tmpsfc.{YYYYMM}.anom.nc`
- **Data type**: Gridded SST anomaly fields (1-degree resolution); Nino3.4 index computed as a cosine-latitude-weighted spatial mean over 5S-5N, 190-240E
- **Models and ensemble sizes**:
  - CFSv2: 32 members
  - ECCC-CanESM5: 20 members
  - ECCC-GEM5.2-NEMO: 20 members
  - NCAR-CESM1: 10 members
  - NCAR-CCSM4: 10 members
  - NASA-GEOS-S2S-2: 10 members
  - Total: 102 members across 6 models
- **Lead time**: 9 months from initialization
- **Anomaly baseline**: Model-specific (each model's own hindcast climatology, nominally standardized by CPC)
- **Update frequency**: Monthly (approximately the 8th of each month)
- **Time coordinate handling**: NMME uses "months since 1960-01-01" units, which xarray cannot decode automatically. Times are decoded manually.

### 3. C3S (Copernicus Climate Change Service) Seasonal Forecasts

- **Provider**: ECMWF Copernicus Climate Data Store (CDS)
- **Dataset**: `seasonal-postprocessed-single-levels`
- **API**: CDS API (`cdsapi` Python package) with user-provided API key
- **Data type**: Gridded SST anomaly fields; Nino3.4 computed as cosine-latitude-weighted spatial mean over the requested sub-region (5N-5S, 170W-120W)
- **Models and ensemble sizes** (for March 2026 initialization):
  - ECMWF (system 51): 51 members
  - Meteo-France (system 9): 51 members
  - DWD (system 22): 50 members
  - CMCC (system 4): 50 members
  - ECCC (system 5): 20 members
  - NCEP (system 2): 112 members
  - BOM (system 2): 121 members
  - Total: 455 members across 7 models
  - Note: UKMO and JMA were unavailable for March 2026 at the time of retrieval
- **Lead time**: 6 months from initialization
- **Anomaly baseline**: 1993-2016 (C3S hindcast climatology period), adjusted to 1991-2020 (see below)
- **Update frequency**: Monthly
- **forecastMonth convention**: In C3S, `forecastMonth=1` corresponds to the initialization month itself (e.g., for a March 2026 initialization, forecastMonth=1 = March 2026, forecastMonth=2 = April 2026, etc.)

### 4. Observed Nino3.4

- **Provider**: NOAA CPC
- **Monthly Nino3.4**: `https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices` (OISSTv2-based)
- **ONI (Oceanic Nino Index)**: `https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt` (3-month running mean, ERSSTv5-based)
- **Anomaly baseline**: 1991-2020 (centered 30-year base period)
- **Note on file format**: The sstoi.indices file orders the Nino regions as NINO1+2, NINO3, NINO4, NINO3.4 (NINO4 comes before NINO3.4), with SST and anomaly as separate columns. The NINO3.4 anomaly is the 10th column (index 9).

## Anomaly Baseline Harmonization

Different sources compute anomalies relative to different climatological base periods:

| Source | Native baseline | Adjustment applied |
|--------|----------------|-------------------|
| CFS v2 | 1991-2020 | None (reference baseline) |
| NMME | Model-specific | None (assumed approximately consistent) |
| C3S | 1993-2016 | Adjusted to 1991-2020 |
| Observed | 1991-2020 | None (reference baseline) |

### C3S Baseline Adjustment Method

The C3S anomalies are adjusted from the 1993-2016 baseline to 1991-2020 using the observed Nino3.4 record:

1. For each calendar month (January through December), compute the mean of observed Nino3.4 anomalies (which are on the 1991-2020 baseline) over the 1993-2016 period
2. This mean equals `climatology_1993-2016 - climatology_1991-2020` for that month
3. Add this offset to each C3S anomaly value for the corresponding target month

The adjustment is small (approximately -0.05 to -0.09 C for most months, annual mean -0.05 C), reflecting that the 1993-2016 period was slightly cooler than 1991-2020 in the Nino3.4 region. After adjustment, C3S and CFS ensemble means agree to within ~0.01-0.12 C at overlapping forecast months, consistent with expected inter-model variability.

Residual baseline offsets of ~0.1 C may remain, particularly for NMME models whose exact baseline periods are model-specific and not independently adjusted.

## Model Deduplication

Several physical models appear in multiple source databases. To avoid double-counting when computing multi-model statistics, the following deduplication rules are applied:

| Physical Model | Appears in | Kept version | Rationale |
|---------------|-----------|-------------|-----------|
| NCEP CFSv2 | CFS, NMME, C3S | CFS (E3, 40 members) | Most current initialization; direct Nino3.4 output |
| ECCC GEM5.2-NEMO | NMME, C3S | NMME (20 members, 9 leads) | Longer lead time than C3S (6 leads) |
| NMME overall mean | NMME | Dropped | Not an independent model; is a derived mean |

After deduplication, 11 unique models remain with a total of 433 individual ensemble members.

### Deduplicated Model Inventory

| Model | Source | Members | Lead (months) |
|-------|--------|---------|---------------|
| CFSv2 | CFS | 40 | 9 |
| ECMWF | C3S | 51 | 6 |
| Meteo-France | C3S | 51 | 6 |
| DWD | C3S | 50 | 6 |
| CMCC | C3S | 50 | 6 |
| BOM | C3S | 121 | 6 |
| ECCC-CanESM5 | NMME | 20 | 9 |
| ECCC-GEM5.2-NEMO | NMME | 20 | 9 |
| NCAR-CESM1 | NMME | 10 | 9 |
| NCAR-CCSM4 | NMME | 10 | 9 |
| NASA-GEOS-S2S-2 | NMME | 10 | 9 |

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

### 1. CFS v2 Ensemble Forecast Plume (`cfs_plume.png`)
Individual CFS ensemble members as thin semi-transparent lines, with two-tier shading (lighter for full min-max range, darker for 25th-75th percentile IQR), and the ensemble mean as a bold red line. Shows only true forecast months.

### 2. C3S Multi-System Forecast Plume (`c3s_plume.png`)
All individual C3S ensemble members color-coded by originating center, with per-center ensemble means as thicker lines and the C3S multi-system mean as a dashed black line. Shows all 7 available C3S models (455 members).

### 3. NMME Multi-Model Forecast Plume (`nmme_plume.png`)
All individual NMME ensemble members color-coded by model, with per-model means and the NMME multi-model mean. Shows all 6 NMME models (102 members).

### 4. Multi-Model Comparison (`model_comparison.png`)
Ensemble mean from each unique model as a colored line (after deduplication), with the multi-model mean as a thick dashed black line. Shows the inter-model spread and identifies which models are outliers.

### 5. Source-Level Comparison (`source_comparison.png`)
One line per source (CFS mean, NMME mean-of-means, C3S multi-system mean), with shading showing the min-max range across models within each source. Provides a high-level view of how the three forecasting systems compare.

### 6. Forecast Evolution (`forecast_evolution.png`)
For a selected source (default: CFS), shows how successive monthly forecasts have evolved over time. Each line represents one initialization date's forecast trajectory. Currently shows a single initialization; will accumulate multiple trajectories as monthly fetches are repeated.

### 7. Forecast Distribution (`forecast_distribution.png`)
Box-and-whisker plot of deduplicated model means by target month, with individual model values as colored dots. Shows inter-model consensus and disagreement at each lead time.

### 8. Combined Forecast Plume (`mega_plume.png`)
All ensemble members from all sources (deduplicated, 11 models, 433 members) in a single plume plot. Members are color-coded by model, with model means as thicker lines and the multi-model mean as a dashed black line. The most comprehensive view of the full forecast distribution.

### 9. Member Distribution with Diamonds (`member_distribution.png`)
Every ensemble member as a small colored dot, organized by target month, with bold diamond markers showing each model's ensemble mean. Models are spread horizontally within each month to reduce overlap.

### 10. Model-Weighted Box Distribution (`member_box_distribution.png`)
All ensemble members as colored dots with a box-and-whisker overlay. The box plot statistics are computed from all individual members using model-equal weighting (each member weighted by 1/N where N is ensemble size for that model). This ensures the IQR and median reflect inter-model consensus rather than being dominated by large-ensemble models.

### 11. Historical Context (`historical_context.png`)
Observed monthly Nino3.4 index since 1990, with red shading for El Nino episodes (anomaly > +0.5 C) and blue shading for La Nina episodes (anomaly < -0.5 C). The multi-model mean forecast is overlaid as a dashed line with a model-weighted 25th-75th percentile band, allowing the current forecast to be compared visually against the magnitude of past events (e.g., 1997-98, 2015-16 strong El Ninos).

## Software and Dependencies

- Python 3.11+
- Data fetching: `requests`, `beautifulsoup4`, `cdsapi`
- Data processing: `xarray`, `netCDF4`, `pandas`, `numpy`
- Visualization: `matplotlib`
- Testing: `pytest`, `responses` (HTTP mocking)

## Limitations and Caveats

1. **Baseline residuals**: Despite the C3S adjustment, residual inter-source offsets of ~0.1 C may remain due to differences in underlying SST products and model-specific climatologies.

2. **Model independence**: Some models share components or heritage (e.g., NCAR-CESM1 and NCAR-CCSM4 share atmospheric components). Treating them as fully independent slightly overstates the effective number of independent forecasts.

3. **Unequal lead times**: C3S models provide 6 months of lead time while NMME and CFS provide 9. At longer leads (months 7-9), only NMME and CFS contribute, reducing the model diversity.

4. **Missing C3S models**: UKMO and JMA were unavailable in the CDS archive for the March 2026 initialization. These models may become available with a delay after the initialization date.

5. **No amplitude bias correction**: The NMME operationally applies an amplitude bias correction (dividing forecast anomalies by the ratio of model hindcast standard deviation to observed standard deviation, per model/lead/start-month). This correction is not applied here because hindcast data is not available for all current model versions. This means some NMME models may exhibit systematically too much or too little variability.

6. **IRI models excluded**: The IRI ENSO forecast (23 models, means only) is available in the tool but excluded from the current plots because its most recent update (February 2026) predates the March 2026 initialization of the other sources. It can be included with `--include-iri`.
