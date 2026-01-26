#!/usr/bin/env python3
"""Production entry point for the Climate Dashboard.

This module creates the Dash app and exposes the Flask server for Gunicorn.
Usage: gunicorn app:server
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_SOURCES, DATA_DIR
from src.scraper import load_or_fetch_data
from src.dashboard import create_dashboard
from src.enso import create_combined_enso_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_data():
    """Load or fetch required data on startup."""
    logger.info("Initializing data...")

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Load ERA5 data
    source = DATA_SOURCES["era5_global"]
    df = load_or_fetch_data(source["url"], source["local_file"])

    # Load ENSO data
    try:
        enso_file = DATA_DIR / "enso_combined.csv"
        if not enso_file.exists():
            create_combined_enso_dataset(enso_file)
    except Exception as e:
        logger.warning(f"Could not load ENSO data: {e}")

    logger.info("Data initialization complete")
    return df


# Initialize data and create app
logger.info("Starting Climate Dashboard...")
df = initialize_data()
app = create_dashboard(df)

# Expose the Flask server for Gunicorn
server = app.server

if __name__ == "__main__":
    # For local development
    from config import DASHBOARD_HOST, DASHBOARD_PORT, DEBUG_MODE
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DEBUG_MODE)
