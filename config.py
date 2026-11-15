"""Configuration for the Climate Dashboard application."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ERA5_GRIDDED_DIR = DATA_DIR / "era5"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
ERA5_GRIDDED_DIR.mkdir(exist_ok=True)

# Data sources
DATA_SOURCES = {
    "era5_global": {
        # Climate Pulse 2.0 path (Nov 2026); the old data/series/ files are
        # only kept for a few weeks after the release.
        "url": "https://sites.ecmwf.int/data/climatepulse/timeseries/csv/daily/timeseries_era5_t2m_daily_global.csv",
        "local_file": DATA_DIR / "timeseries_era5_t2m_daily_global.csv",
        "description": "ERA5 Daily Global Mean 2m Temperature",
    }
}

# ERA5 Gridded Data Configuration (via CDS API)
ERA5_GRIDDED_CONFIG = {
    "output_dir": ERA5_GRIDDED_DIR,
    "default_grid": "1.0/1.0",  # 1 degree resolution (reduces data volume)
    "timeout": 300,  # Request timeout in seconds
    "max_retries": 3,
    "retry_delay": 60,  # Base delay between retries
    # Common variables available
    "variables": {
        "2m_temperature": "2m_temperature",
        "sea_surface_temperature": "sea_surface_temperature",
        "total_precipitation": "total_precipitation",
        "mean_sea_level_pressure": "mean_sea_level_pressure",
    },
}

# Dashboard settings (support environment variables for production)
DASHBOARD_HOST = os.environ.get("HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.environ.get("PORT", 8050))
DEBUG_MODE = os.environ.get("DEBUG", "true").lower() == "true"

# Update schedule (24-hour format)
UPDATE_HOUR = 6  # 6 AM daily update
