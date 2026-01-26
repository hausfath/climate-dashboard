# Climate Dashboard

An interactive dashboard visualizing global temperature anomalies using ERA5 reanalysis data.

## Features

- Daily global mean temperature anomalies (relative to preindustrial baseline)
- Year-over-year comparisons
- ENSO (El Nino/La Nina) integration
- ML-powered annual temperature projections
- Monthly and annual prediction tracking
- Dark/light mode toggle

## Data Sources

- **ERA5**: Daily global mean 2m temperature from ECMWF Climate Pulse
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

## Deployment

This project is configured for deployment on [Render](https://render.com):

1. Fork/push this repository to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically detect the `render.yaml` and create:
   - A web service for the dashboard
   - A cron job for daily data updates

## Project Structure

```
climate-dashboard/
├── app.py              # Production entry point (Gunicorn)
├── run.py              # Development entry point
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── Procfile           # Render/Heroku process file
├── render.yaml        # Render deployment config
├── src/
│   ├── dashboard.py   # Dash application
│   ├── scraper.py     # Data fetching
│   └── enso.py        # ENSO data processing
└── data/              # Data directory (gitignored)
```

## License

MIT
