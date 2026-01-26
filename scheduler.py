#!/usr/bin/env python3
"""Scheduler for automatic daily data updates."""

import logging
import schedule
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_SOURCES, DATA_DIR, UPDATE_HOUR
from src.scraper import fetch_era5_data
from src.enso import create_combined_enso_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_era5_data():
    """Update ERA5 temperature data."""
    for name, source in DATA_SOURCES.items():
        try:
            logger.info(f"Updating {name}...")
            df = fetch_era5_data(source["url"], source["local_file"])
            logger.info(f"Successfully updated {name}: {len(df)} rows, latest date: {df['date'].max()}")
        except Exception as e:
            logger.error(f"Failed to update {name}: {e}")


def update_enso_data():
    """Update ENSO data (ONI, HadISST, IRI forecasts)."""
    try:
        logger.info("Updating ENSO data...")
        enso_file = DATA_DIR / "enso_combined.csv"
        df = create_combined_enso_dataset(enso_file)

        # Log summary
        historical = df[~df['is_forecast']]
        forecasts = df[df['is_forecast']]
        latest_historical = historical.iloc[-1]

        logger.info(f"Successfully updated ENSO data: {len(df)} records")
        logger.info(f"  Latest historical: {latest_historical['date'].strftime('%Y-%m')} "
                   f"ONI={latest_historical['oni']:.2f} ({latest_historical['enso_state']})")
        logger.info(f"  Forecast periods: {len(forecasts)}")
    except Exception as e:
        logger.error(f"Failed to update ENSO data: {e}")


def update_all_data():
    """Update all configured data sources."""
    logger.info(f"Starting scheduled data update at {datetime.now()}")

    # Update ERA5 temperature data
    update_era5_data()

    # Update ENSO data
    update_enso_data()

    logger.info("Scheduled update complete")


def run_scheduler():
    """Run the scheduler for daily updates."""
    logger.info(f"Starting scheduler - updates scheduled for {UPDATE_HOUR:02d}:00 daily")

    # Schedule daily update
    schedule.every().day.at(f"{UPDATE_HOUR:02d}:00").do(update_all_data)

    # Also run immediately on start
    logger.info("Running initial data update...")
    update_all_data()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    run_scheduler()
