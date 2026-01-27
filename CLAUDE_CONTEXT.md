# Climate Dashboard - Development Context Summary

## Project Overview
An interactive climate dashboard visualizing global temperature anomalies using ERA5 reanalysis data, deployed on Render.com.

**Live URL**: https://climate-dashboard-cjxs.onrender.com
**GitHub Repo**: https://github.com/hausfath/climate-dashboard

## Key Features Implemented
1. **Temperature Visualizations**:
   - Time series plot of global temperature anomalies (vs preindustrial baseline)
   - Daily anomalies by day-of-year (highlighting recent years)
   - Daily absolute temperatures by day-of-year
   - Monthly and annual heatmaps

2. **ML-Powered Projections**:
   - Monthly temperature projections
   - Annual temperature projections using linear regression
   - Projection history plot showing how predictions evolve throughout the year
   - Model features: year, prior year anomaly, ENSO, **trailing 30-day anomaly**

3. **Dark Mode Toggle**: Full dark/light theme support for all plots and UI elements

4. **ENSO Integration**: Combines ONI data (NOAA), HadISST (PSL), and IRI forecasts

## Technical Architecture

### File Structure
```
climate-dashboard/
├── app.py              # Production entry point (Gunicorn) - exposes `server` for WSGI
├── run.py              # CLI entry point (update data, run dashboard)
├── config.py           # Configuration (supports env vars for production)
├── requirements.txt    # Dependencies (includes gunicorn, beautifulsoup4)
├── Procfile            # Render web service command
├── render.yaml         # Render Blueprint (web service + cron job)
├── src/
│   ├── dashboard.py    # Main Dash application (~1600 lines)
│   ├── scraper.py      # ERA5 data fetching with caching
│   ├── enso.py         # ENSO data processing and combination
│   └── annual_prediction.py
└── data/
    ├── projection_history_2026.csv  # Committed to repo (needed for deployment)
    └── *.csv (other data files gitignored - fetched at runtime)
```

### Key Code Locations in src/dashboard.py
- `THEME_CONFIG` (line ~15): Light/dark theme color definitions
- `adjust_anomalies_to_preindustrial()` (line ~70): Baseline adjustment
- `create_time_series_plot()` (line ~98): Main time series (uses Scattergl for performance)
- `create_daily_anomalies_plot()` (line ~177): Day-of-year comparison
- `create_annual_prediction_plot()` (line ~590): Annual projection with trailing 30-day predictor
- `calculate_projection_for_date()` (line ~795): Core projection calculation
- `create_projection_history_plot()` (line ~1034): Projection evolution plot
- `create_dashboard()` (line ~1306): Main dashboard factory function
- Chained callbacks (line ~1515+): Sequential graph loading

### Projection Model Details
The annual projection uses a linear regression with these features:
- Year (trend)
- Prior year anomaly
- Weighted ENSO (YTD actual + forecasts for remainder)
- **Trailing 30-day anomaly** (key feature - makes projections responsive to recent data)

For each historical year, the trailing anomaly at the same day-of-year is calculated for training.

## Deployment on Render

### Services
1. **climate-dashboard** (Web Service): Runs `gunicorn app:server`
2. **climate-data-updater** (Cron Job): Runs `python run.py update --force` daily at 6 AM UTC

### Important Limitation
**Render free tier uses ephemeral storage** - the cron job and web service have separate filesystems. Files created by the cron job are NOT visible to the web service.

**Current Workaround**: The `projection_history_2026.csv` file is committed to the repo. For ongoing updates, you need to:
- Run updates locally and push to GitHub, OR
- Upgrade to Render's persistent disk ($0.25/GB/month), OR
- Use external storage (database, S3, etc.)

### Performance Optimizations Implemented
1. **Sequential callback loading**: Graphs load top-to-bottom (chained callbacks)
2. **Loading spinners**: `dcc.Loading` components wrap each graph
3. **Scattergl**: WebGL-based rendering for large datasets (time series plot)
4. **Separate styling callback**: Dark mode toggle is instant (no plot regeneration wait)
5. **Skip heavy computation**: Projection history shows placeholder if file missing (avoids timeout)

## Known Issues / Future Improvements

### Current Issues
- **Data persistence**: Cron job can't update files visible to web service (ephemeral storage)
- **Initial load time**: Still takes several seconds for all 8 graphs to render sequentially

### Potential Improvements
1. **Database integration**: Use Supabase/PlanetScale for shared data storage
2. **Caching**: Implement Flask-Caching or Redis for plot caching
3. **Data downsampling**: Reduce data points sent to browser for faster rendering
4. **Pre-rendered plots**: Generate static images for initial load, then hydrate with interactive versions

## Commands Reference

### Local Development
```bash
cd /Users/hausfath/Desktop/Climate\ Science/Climate\ Dashboard
source venv/bin/activate
python run.py                    # Update data and run dashboard
python run.py dashboard          # Run dashboard only
python run.py update --force     # Force data refresh
```

### Git/Deployment
```bash
git add -A && git commit -m "message" && git push  # Auto-deploys on Render
gh repo view hausfath/climate-dashboard --web       # Open GitHub repo
```

### Render CLI (if needed)
```bash
# Manual deploy trigger via Render dashboard or API
# Cron job can be manually triggered via Render dashboard
```

## Data Sources
- **ERA5**: https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv
- **ONI**: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- **HadISST**: https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data
- **IRI Forecasts**: https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/

## Session History Summary
1. Built initial dashboard with ERA5 data visualizations
2. Added ENSO integration and ML projections
3. Implemented dark mode with theme toggle
4. Created projection history tracking (daily projection evolution)
5. Fixed projection model to use trailing 30-day anomaly as predictor (more responsive)
6. Set up GitHub repo and Render deployment
7. Fixed Python version for Render (needs full version like 3.11.11)
8. Diagnosed empty graphs issue (callback timeout from projection history generation)
9. Split single massive callback into chained sequential callbacks
10. Added loading spinners for better UX
