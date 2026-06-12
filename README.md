# Climate Dashboard

A real-time global temperature monitoring dashboard with ENSO forecasting and climate model evaluation. Built with Plotly Dash, updated daily with ERA5 reanalysis data.

**Live at**: [climate-dashboard.onrender.com](https://climate-dashboard.onrender.com)

## Dashboard Tabs

### Global Temperature

Tracks daily global mean temperature anomalies relative to preindustrial (1850-1900) using ERA5 reanalysis data from ECMWF Climate Pulse.

**Summary cards**: Latest data date and status, current daily anomaly, monthly projection with estimated historical rank, and annual prediction with estimated historical rank.

**Plots**:
- **Time series** -- Daily anomaly since 1940 with 365-day rolling mean and 1.5C reference line
- **Daily anomalies** -- Day-of-year traces with recent years highlighted and current month shaded
- **Daily temperatures** -- Absolute temperature version of the above
- **Monthly projection** -- Historical monthly averages for the current month with regression-based projection
- **Annual prediction** -- Historical annual means and current-year prediction from a 6-feature linear regression model (year, prior year anomaly, observed ENSO, forecast ENSO, trailing 30-day anomaly, YTD anomaly). Trained on 1950-present excluding volcanic years (1982-83, 1991-93). Uncertainty interval from a Monte Carlo over the multi-model ENSO forecast ensemble. See [PROJECTION_METHODOLOGY.md](PROJECTION_METHODOLOGY.md) for full details.
- **Projection evolution** -- How the annual prediction has changed day by day, with uncertainty band and star markers for ENSO forecast updates
- **Anomaly heatmap** -- Year x day-of-year heatmap of temperature anomalies
- **Temperature heatmap** -- Year x day-of-year heatmap of absolute temperatures
- **Ridgeline plot** -- Temperature anomaly distributions by year with gradient fill

### ENSO Forecast

Multi-model ENSO (El Nino-Southern Oscillation) forecast system combining 13 distinct physical models (~650 ensemble members) from four operational sources (CFSv2, NMME, C3S, CanSIPS), with duplicate physical models de-duplicated across sources and every model weighted equally in multi-model statistics. Supports both the standard Nino3.4 anomaly (ONI) and the Relative ONI (rONI), which subtracts tropical-mean warming for fairer cross-decade comparison. Full details in [ENSO/METHODOLOGY.md](ENSO/METHODOLOGY.md).

**Summary cards**: Current ENSO state (El Nino / La Nina / Neutral with ONI value), peak forecast anomaly with model spread, and forecast source summary.

**Plots** (each in ONI and rONI variants):
- **Mega plume** -- Individual ensemble members as thin lines, per-model means as thick lines, and multi-model mean as dashed line, with observed Nino3.4 and El Nino/La Nina background shading
- **Box distribution** -- Model-weighted box plots per target month with jittered dots colored by source model
- **Historical context** -- Observed Nino3.4 since 1990 with El Nino/La Nina fills and multi-model forecast overlay
- **Strength probabilities** -- Model-weighted probability of each ENSO strength category (La Nina through Very Strong El Nino) by 3-month season

### Models vs. Observations

Compares CMIP climate model projections against five observational temperature records to assess model performance.

**Summary cards**: CMIP6 model-observation difference (last 12 months), observed warming trend (1970-present), recent trend (2011-present), and model-observation alignment percentile.

**Controls**: Model generation selector (CMIP6/5/3) and smoothing toggle (monthly or 12-month rolling average).

**Plots**:
- **Timeseries** -- Model 5-95th percentile envelope and ensemble mean vs. five observational datasets (1900-2040) with range slider
- **Trend explorer** -- Warming trends (C/decade) by start year (1970-2010) to present, model band vs. observations
- **Trend histogram grid** -- Model trend distributions for three periods (1970-present, 2001-present, 2011-present) with observed trend lines

## Data Sources

| Source | Provider | Usage |
|--------|----------|-------|
| ERA5 daily global 2m temperature | ECMWF Climate Pulse | Global temperature tab |
| CFS v2 ensemble plume | NOAA CPC | ENSO forecasts (40 rolling ensemble members) |
| NMME multi-model ensemble | NOAA CPC | ENSO forecasts (6 models) |
| C3S seasonal forecasts | Copernicus CDS | ENSO forecasts (9 models) |
| CanSIPS forecasts | ECCC MSC Datamart | ENSO forecasts (2 models, 12-month leads) |
| ONI (Oceanic Nino Index) | NOAA CPC | Observed ENSO state |
| RONI (Relative ONI) | NOAA CPC | Observed ENSO state, warming-adjusted |
| Nino3.4 SST anomaly | NOAA CPC / PSL | Observed ENSO monitoring |
| CMIP3/5/6 model ensembles | PCMDI/ESGF | Model comparison tab |
| HadCRUT5 | Met Office Hadley Centre | Observational record |
| GISTEMP v4 | NASA GISS | Observational record |
| NOAAGlobalTemp | NOAA NCEI | Observational record |
| Berkeley Earth | Berkeley Earth | Observational record |
| Copernicus/ERA5 | C3S | Observational record |

All temperature anomalies are rebaselined to 1850-1900 preindustrial.

## Key Features

- **Dark/light mode** toggle with full theme support across all plots and cards
- **Static/interactive rendering** -- each plot has a pre-rendered PNG (fast loading) and an interactive Plotly version, toggled via switch. Defaults to static on mobile.
- **Daily automated updates** via Render cron job and GitHub Actions. Monthly forecast sources are re-fetched only when a new initialization is actually available (detected from the init date inside each source's latest file)
- **Projection tracking** -- daily snapshots of the annual prediction stored in `projection_history_{year}.csv` (append-only), with ENSO forecast change markers. See [PROJECTION_METHODOLOGY.md](PROJECTION_METHODOLOGY.md)
- **Forecast verification** -- past EC46 temperature forecasts and monthly ENSO plumes are replayed against observations in `forecast_skill/` (committed by the daily cron; not shown on the dashboard)

## Methodology

- [ENSO/METHODOLOGY.md](ENSO/METHODOLOGY.md) -- the multi-model ENSO forecast system: sources, baseline harmonization, rONI scaling, model deduplication, equal-weight statistics, update detection, and verification
- [PROJECTION_METHODOLOGY.md](PROJECTION_METHODOLOGY.md) -- the annual temperature projection: regression model, training set, Monte Carlo uncertainty, and daily projection tracking

## Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

The dashboard will be available at http://127.0.0.1:8050

To update data before starting:

```bash
python run.py update --force
```

## Deployment

Deployed on [Render](https://render.com) with automated daily updates:

1. **Render web service** (starter plan) runs via Gunicorn with `render.yaml` blueprint
2. **Render cron job** runs `python run.py update --force` daily
3. **GitHub Actions** (`.github/workflows/update-data.yml`) runs daily at 6 AM UTC, commits updated data to git
4. CDS API credentials stored as GitHub secrets (`CDS_URL`, `CDS_KEY`)

## Project Structure

```
climate-dashboard/
├── app.py                          # Production entry point (Gunicorn)
├── run.py                          # CLI: data updates + dev server
├── config.py                       # Data sources, paths, settings
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment blueprint
├── src/
│   ├── dashboard.py                # Dash app: layout, callbacks, global tab plots
│   ├── scraper.py                  # ERA5 daily CSV fetch/cache
│   ├── enso.py                     # ENSO data: ONI, HadISST, IRI fetch
│   ├── enso_plots.py               # ENSO Plotly plots + cards + multi-model integration
│   ├── annual_prediction_mc.py     # Monte Carlo over ENSO ensemble for prediction CI
│   ├── ec46_skill.py               # EC46 forecast-vs-observed skill plot
│   ├── enso_skill.py               # ENSO plume forecast-vs-observed skill plots
│   ├── models_vs_obs.py            # CMIP ensemble loading, obs fetching, plots, cards
│   ├── gridded_map.py              # ERA5 spatial maps (retained for future use)
│   ├── era5_api.py                 # CDS API client for ERA5 gridded downloads
│   └── precompute_monthly_stats.py # Spatial pre-computation pipeline
├── ENSO/
│   ├── METHODOLOGY.md              # Multi-model ENSO forecast methodology
│   ├── main.py                     # CLI: fetch + plot commands
│   └── enso_forecast/
│       ├── config.py               # URLs, model metadata, dedup maps
│       ├── normalize.py            # Forecast normalization + multi-model mean
│       ├── visualize.py            # Matplotlib-based diagnostic plots
│       └── fetchers/               # CFS, NMME, C3S, CanSIPS, IRI, observed fetchers
├── assets/
│   ├── mobile.css                  # Mobile-responsive styles
│   └── images/                     # Pre-rendered static PNGs (dark + light per plot)
├── forecast_skill/                 # Forecast verification (committed, off-dashboard)
│   ├── ec46_skill.png              # EC46 inits vs observed temperature
│   ├── enso_skill_*.png            # Past ENSO plumes vs observed (ONI/rONI x 3 styles)
│   ├── enso_archive/               # Frozen monthly plume snapshots
│   └── archive/                    # Archived EC46 forecast CSVs
└── data/
    ├── cmip/                       # CMIP3/5/6 text files + cached obs CSV (committed)
    ├── era5/                       # ERA5 NetCDF + precomputed maps (git-ignored)
    ├── projection_history_*.csv    # Daily projection snapshots (committed)
    ├── enso_forecast_updates.csv   # ENSO forecast change dates (committed)
    └── *.csv                       # Runtime data files (git-ignored)
```

## License

MIT
