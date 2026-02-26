# Climate Dashboard

An interactive dashboard visualizing global temperature anomalies and spatial climate patterns using ERA5 reanalysis data.

## Features

**Global Temperature tab**
- Daily global mean temperature anomalies (relative to preindustrial baseline)
- Year-over-year comparisons
- ENSO (El Niño/La Niña) integration
- ML-powered annual temperature projections
- Monthly and annual prediction tracking
- Dark/light mode toggle

**Spatial Analysis tab**
- Pre-rendered 2D global maps: absolute temperature and anomaly views for every month since 1940
- Interactive 3D globe with WebGL rendering
- Regional statistics sidebar: per-month rankings and anomalies for continents and major countries
- Current-month maps updated daily via CDS API

## Data Sources

- **ERA5**: Daily global mean 2m temperature from ECMWF Climate Pulse (global tab); monthly gridded 2m temperature from Copernicus CDS (spatial tab)
- **ENSO**: ONI data from NOAA CPC, HadISST from PSL, forecasts from IRI

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python run.py
```

The dashboard will be available at http://127.0.0.1:8050

### Spatial Analysis setup (optional)

The Spatial Analysis tab requires pre-computed ERA5 data. To set it up locally:

1. **Configure CDS API credentials** — register at https://cds.climate.copernicus.eu and create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-PERSONAL-ACCESS-TOKEN>
   ```

2. **Download ERA5 monthly data** (requires ~1.4GB disk space):
   ```bash
   python src/era5_api.py
   ```

3. **Pre-compute statistics and maps**:
   ```bash
   python src/precompute_monthly_stats.py
   ```

## Deployment

This project is configured for deployment on [Render](https://render.com):

1. Fork/push this repository to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically detect `render.yaml` and create:
   - A **web service** (starter plan) with a 3GB persistent disk for ERA5 gridded data
   - A **cron job** for daily global temperature CSV updates
4. Add GitHub repository secrets `CDS_URL` and `CDS_KEY` for the daily spatial data update (runs via GitHub Actions)
5. After first deploy, rsync pre-computed data to the Render disk:
   ```bash
   rsync -avz data/era5/precomputed/ <render-ssh>:/opt/render/project/src/data/era5/precomputed/
   rsync -avz data/era5/era5_t2m_monthly_1940_present.nc data/era5/era5_t2m_climatology_1991_2020.nc \
     <render-ssh>:/opt/render/project/src/data/era5/
   ```

## Project Structure

```
climate-dashboard/
├── app.py                          # Production entry point (Gunicorn)
├── run.py                          # Development entry point + update pipeline
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config (web + cron)
├── src/
│   ├── dashboard.py                # Dash application (both tabs)
│   ├── scraper.py                  # Global temperature data fetching
│   ├── enso.py                     # ENSO data processing
│   ├── era5_api.py                 # CDS API client for ERA5 gridded downloads
│   ├── gridded_map.py              # 2D map and 3D globe rendering
│   └── precompute_monthly_stats.py # Pre-computation pipeline (stats + maps)
├── assets/
│   ├── globe.css                   # 3D globe scoped styles
│   └── images/                     # Static plot images (auto-generated)
└── data/
    ├── era5/
    │   ├── era5_t2m_monthly_1940_present.nc     # Full monthly gridded data (git-ignored)
    │   ├── era5_t2m_climatology_1991_2020.nc    # Climatology baseline (git-ignored)
    │   └── precomputed/
    │       ├── monthly_stats.csv                # Regional stats for all months (committed)
    │       ├── geometry_cache.npz               # 3D geometry arrays (git-ignored)
    │       └── maps/                            # Pre-rendered PNGs (git-ignored, on Render disk)
    └── *.csv                                    # Global temperature / ENSO data (git-ignored)
```

## License

MIT
