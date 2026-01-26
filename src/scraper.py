"""Data scraper for climate datasets."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_era5_data(url: str, local_path: Path) -> pd.DataFrame:
    """
    Fetch ERA5 daily temperature data from ECMWF.

    Args:
        url: URL to the CSV data
        local_path: Path to save the downloaded data

    Returns:
        DataFrame with the temperature data
    """
    logger.info(f"Fetching ERA5 data from {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save raw data
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(response.text)
        logger.info(f"Saved data to {local_path}")

        # Parse and return DataFrame
        df = parse_era5_data(local_path)
        return df

    except requests.RequestException as e:
        logger.error(f"Failed to fetch data: {e}")
        # Try to load cached data if available
        if local_path.exists():
            logger.info("Loading cached data instead")
            return parse_era5_data(local_path)
        raise


def parse_era5_data(file_path: Path) -> pd.DataFrame:
    """
    Parse ERA5 CSV data into a DataFrame.

    Args:
        file_path: Path to the CSV file

    Returns:
        Processed DataFrame
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(file_path, comment='#')

    # Rename columns for clarity
    df = df.rename(columns={
        'date': 'date',
        '2t': 'temperature',
        'clim_91-20': 'climatology',
        'ano_91-20': 'anomaly',
        'status': 'status'
    })

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])

    # Extract useful time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    return df


def load_or_fetch_data(url: str, local_path: Path, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load data from cache or fetch fresh data.

    Args:
        url: URL to fetch data from
        local_path: Path to cached data
        force_refresh: If True, always fetch fresh data

    Returns:
        DataFrame with the data
    """
    if not force_refresh and local_path.exists():
        # Check if cache is from today
        mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
        if mtime.date() == datetime.now().date():
            logger.info("Using cached data from today")
            return parse_era5_data(local_path)

    return fetch_era5_data(url, local_path)


if __name__ == "__main__":
    # Test the scraper
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_SOURCES

    source = DATA_SOURCES["era5_global"]
    df = fetch_era5_data(source["url"], source["local_file"])
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.tail())
