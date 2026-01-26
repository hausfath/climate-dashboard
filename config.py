"""Configuration for the Climate Dashboard application."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Data sources
DATA_SOURCES = {
    "era5_global": {
        "url": "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv",
        "local_file": DATA_DIR / "era5_daily_series_2t_global.csv",
        "description": "ERA5 Daily Global Mean 2m Temperature",
    }
}

# Dashboard settings (support environment variables for production)
DASHBOARD_HOST = os.environ.get("HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.environ.get("PORT", 8050))
DEBUG_MODE = os.environ.get("DEBUG", "true").lower() == "true"

# Update schedule (24-hour format)
UPDATE_HOUR = 6  # 6 AM daily update
