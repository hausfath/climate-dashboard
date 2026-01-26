"""ENSO (El Nino Southern Oscillation) data fetching and processing."""

import logging
import re
from datetime import datetime
from pathlib import Path
from io import StringIO

import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data source URLs
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
HADISST_URL = "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
IRI_FORECAST_URL = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/?enso_tab=enso-sst_table"

# Season to month mapping (center month of 3-month period)
SEASON_TO_MONTH = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
    'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
}

# Month to season mapping
MONTH_TO_SEASON = {v: k for k, v in SEASON_TO_MONTH.items()}


def fetch_oni_data() -> pd.DataFrame:
    """
    Fetch NOAA ONI (Oceanic Nino Index) data.

    Returns:
        DataFrame with columns: year, month, season, oni_value, source
    """
    logger.info(f"Fetching ONI data from {ONI_URL}")

    response = requests.get(ONI_URL, timeout=30)
    response.raise_for_status()

    # Parse the fixed-width format
    lines = response.text.strip().split('\n')

    records = []
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 4:
            season = parts[0]
            year = int(parts[1])
            anom = float(parts[3])

            # Get center month of season
            month = SEASON_TO_MONTH.get(season)
            if month:
                # Handle year for DJF (December is previous year)
                if season == 'DJF':
                    # DJF 1950 means Dec 1949, Jan 1950, Feb 1950 -> center is Jan 1950
                    pass  # Year is already correct for center month

                records.append({
                    'year': year,
                    'month': month,
                    'season': season,
                    'oni': anom,
                    'source': 'ONI'
                })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} ONI records from {df['year'].min()} to {df['year'].max()}")
    return df


def fetch_hadisst_data() -> pd.DataFrame:
    """
    Fetch HadISST Nino 3.4 anomaly data for historical backfill.

    Returns:
        DataFrame with columns: year, month, oni_value, source
    """
    logger.info(f"Fetching HadISST data from {HADISST_URL}")

    response = requests.get(HADISST_URL, timeout=30)
    response.raise_for_status()

    lines = response.text.strip().split('\n')

    records = []
    for line in lines[1:]:  # Skip header line
        parts = line.split()
        if len(parts) >= 13:
            year = int(parts[0])
            for month_idx, value in enumerate(parts[1:13], 1):
                val = float(value)
                if val > -99:  # -99.99 is missing data
                    # Get the season for this month
                    season = MONTH_TO_SEASON.get(month_idx, '')
                    records.append({
                        'year': year,
                        'month': month_idx,
                        'season': season,
                        'oni': val,
                        'source': 'HadISST'
                    })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} HadISST records from {df['year'].min()} to {df['year'].max()}")
    return df


def fetch_iri_forecasts() -> pd.DataFrame:
    """
    Fetch IRI dynamical model ENSO forecasts.

    Returns:
        DataFrame with columns: year, month, season, oni_value, source
    """
    logger.info(f"Fetching IRI forecasts from {IRI_FORECAST_URL}")

    response = requests.get(IRI_FORECAST_URL, timeout=30)
    response.raise_for_status()

    html = response.text

    # Extract dynamical model average row
    # Find the line with "Average, Dynamical models"
    pattern = r'Average.*Dynamical.*?</th>\s*((?:<td>[^<]+</td>\s*)+)'
    match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)

    if not match:
        logger.warning("Could not find dynamical model averages in IRI page")
        return pd.DataFrame()

    # Extract values from td tags
    values_html = match.group(1)
    values = re.findall(r'<td>([^<]+)</td>', values_html)

    # The forecast seasons starting from current month
    # Determine the starting year/month based on current date
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # IRI forecasts typically start from the next 3-month period
    # The seasons are: JFM, FMA, MAM, AMJ, MJJ, JJA, JAS, ASO, SON
    forecast_seasons = ['JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON']

    records = []
    for i, (season, value) in enumerate(zip(forecast_seasons, values)):
        try:
            oni_value = float(value)
            month = SEASON_TO_MONTH[season]

            # Determine the year for this forecast
            # JFM 2026 = month 2 (Feb), etc.
            forecast_year = current_year
            if month < current_month:
                forecast_year = current_year + 1

            records.append({
                'year': forecast_year,
                'month': month,
                'season': season,
                'oni': oni_value,
                'source': 'IRI_Dynamical'
            })
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not parse forecast value: {value}, error: {e}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        logger.info(f"Loaded {len(df)} IRI forecast records")
    return df


def classify_enso_state(oni: float) -> str:
    """
    Classify ENSO state based on ONI value.

    Args:
        oni: Oceanic Nino Index value

    Returns:
        Classification: 'Strong El Nino', 'Moderate El Nino', 'Weak El Nino',
                       'Neutral', 'Weak La Nina', 'Moderate La Nina', 'Strong La Nina'
    """
    if oni >= 2.0:
        return 'Very Strong El Nino'
    elif oni >= 1.5:
        return 'Strong El Nino'
    elif oni >= 1.0:
        return 'Moderate El Nino'
    elif oni >= 0.5:
        return 'Weak El Nino'
    elif oni > -0.5:
        return 'Neutral'
    elif oni > -1.0:
        return 'Weak La Nina'
    elif oni > -1.5:
        return 'Moderate La Nina'
    elif oni > -2.0:
        return 'Strong La Nina'
    else:
        return 'Very Strong La Nina'


def fill_data_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in the time series with interpolated values.

    Identifies missing months between historical data and forecasts,
    and fills them with linear interpolation.

    Args:
        df: DataFrame with year, month, oni columns

    Returns:
        DataFrame with gaps filled
    """
    # Create a complete date range
    min_year, min_month = df['year'].min(), df['month'].min()
    max_year, max_month = df['year'].max(), df['month'].max()

    # Generate all year-month combinations
    all_periods = []
    for year in range(min_year, max_year + 1):
        for month in range(1, 13):
            if (year == min_year and month < min_month):
                continue
            if (year == max_year and month > max_month):
                continue
            all_periods.append({'year': year, 'month': month})

    all_periods_df = pd.DataFrame(all_periods)

    # Find missing periods
    df['key'] = df['year'].astype(str) + '_' + df['month'].astype(str)
    all_periods_df['key'] = all_periods_df['year'].astype(str) + '_' + all_periods_df['month'].astype(str)

    missing = all_periods_df[~all_periods_df['key'].isin(df['key'])].copy()

    if len(missing) == 0:
        df = df.drop('key', axis=1)
        return df

    logger.info(f"Found {len(missing)} gaps to fill")

    # For each missing period, interpolate from surrounding values
    new_records = []
    for _, row in missing.iterrows():
        year, month = row['year'], row['month']

        # Find previous and next values
        prev_mask = (df['year'] < year) | ((df['year'] == year) & (df['month'] < month))
        next_mask = (df['year'] > year) | ((df['year'] == year) & (df['month'] > month))

        prev_data = df[prev_mask].sort_values(['year', 'month']).tail(1)
        next_data = df[next_mask].sort_values(['year', 'month']).head(1)

        if len(prev_data) > 0 and len(next_data) > 0:
            prev_oni = prev_data['oni'].values[0]
            next_oni = next_data['oni'].values[0]

            # Calculate months between
            prev_year, prev_month = prev_data['year'].values[0], prev_data['month'].values[0]
            next_year, next_month = next_data['year'].values[0], next_data['month'].values[0]

            prev_total = prev_year * 12 + prev_month
            next_total = next_year * 12 + next_month
            curr_total = year * 12 + month

            # Linear interpolation
            fraction = (curr_total - prev_total) / (next_total - prev_total)
            interp_oni = prev_oni + fraction * (next_oni - prev_oni)

            # Get season for this month
            season = MONTH_TO_SEASON.get(month, '')

            new_records.append({
                'year': year,
                'month': month,
                'season': season,
                'oni': round(interp_oni, 3),
                'source': 'Interpolated'
            })
            logger.info(f"  Interpolated {year}-{month:02d} ({season}): {interp_oni:.3f}")

    if new_records:
        new_df = pd.DataFrame(new_records)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values(['year', 'month']).reset_index(drop=True)

    df = df.drop('key', axis=1, errors='ignore')
    return df


def create_combined_enso_dataset(output_path: Path) -> pd.DataFrame:
    """
    Create a combined ENSO dataset with historical data and forecasts.

    Priority:
    1. ONI data (most recent/authoritative)
    2. HadISST data (historical backfill)
    3. IRI forecasts (future projections)

    Args:
        output_path: Path to save the combined CSV

    Returns:
        Combined DataFrame
    """
    # Fetch all data sources
    oni_df = fetch_oni_data()
    hadisst_df = fetch_hadisst_data()
    iri_df = fetch_iri_forecasts()

    # Start with HadISST as base (oldest data)
    combined = hadisst_df.copy()

    # Update with ONI data where available (overwrite HadISST)
    if len(oni_df) > 0:
        # Create a key for merging
        combined['key'] = combined['year'].astype(str) + '_' + combined['month'].astype(str)
        oni_df['key'] = oni_df['year'].astype(str) + '_' + oni_df['month'].astype(str)

        # Remove HadISST rows that have ONI data
        combined = combined[~combined['key'].isin(oni_df['key'])]
        combined = combined.drop('key', axis=1)
        oni_df = oni_df.drop('key', axis=1)

        # Append ONI data
        combined = pd.concat([combined, oni_df], ignore_index=True)

    # Add IRI forecasts (future data)
    if len(iri_df) > 0:
        # Only add forecasts for dates not in combined
        iri_df['key'] = iri_df['year'].astype(str) + '_' + iri_df['month'].astype(str)
        combined['key'] = combined['year'].astype(str) + '_' + combined['month'].astype(str)

        new_forecasts = iri_df[~iri_df['key'].isin(combined['key'])]
        new_forecasts = new_forecasts.drop('key', axis=1)
        combined = combined.drop('key', axis=1)

        combined = pd.concat([combined, new_forecasts], ignore_index=True)

    # Sort by year and month
    combined = combined.sort_values(['year', 'month']).reset_index(drop=True)

    # Fill gaps between historical and forecast data with interpolation
    combined = fill_data_gaps(combined)

    # Add date column
    combined['date'] = pd.to_datetime(
        combined['year'].astype(str) + '-' + combined['month'].astype(str) + '-15'
    )

    # Add ENSO classification
    combined['enso_state'] = combined['oni'].apply(classify_enso_state)

    # Add is_forecast flag
    combined['is_forecast'] = combined['source'].isin(['IRI_Dynamical', 'Interpolated'])

    # Reorder columns
    combined = combined[['date', 'year', 'month', 'season', 'oni', 'enso_state', 'source', 'is_forecast']]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved combined ENSO data to {output_path}")
    logger.info(f"Total records: {len(combined)}")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    logger.info(f"Sources: {combined['source'].value_counts().to_dict()}")

    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    output_file = DATA_DIR / "enso_combined.csv"
    df = create_combined_enso_dataset(output_file)

    print("\nSample of recent historical data:")
    historical = df[~df['is_forecast']].tail(10)
    print(historical.to_string(index=False))

    print("\nForecast data:")
    forecasts = df[df['is_forecast']]
    print(forecasts.to_string(index=False))
