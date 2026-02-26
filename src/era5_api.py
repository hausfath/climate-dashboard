"""
ERA5 Gridded Data API Client.

Access ERA5 reanalysis data via the Copernicus Climate Data Store (CDS) API.
Requires a CDS account and API key configured in ~/.cdsapirc

Setup:
    1. Register at https://cds.climate.copernicus.eu
    2. Accept ERA5 license terms
    3. Get API key from https://cds.climate.copernicus.eu/how-to-api
    4. Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-PERSONAL-ACCESS-TOKEN>
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for cdsapi availability
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False
    logger.warning("cdsapi not installed. Run: pip install 'cdsapi>=0.7.7'")

# Check for xarray/netCDF4 availability
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    logger.warning("xarray not installed. Run: pip install xarray netCDF4")


@dataclass
class ERA5Request:
    """Configuration for an ERA5 data request."""

    variable: str
    year: int
    month: int
    days: list[int] = field(default_factory=lambda: list(range(1, 32)))
    hours: list[str] = field(default_factory=lambda: ['00:00', '06:00', '12:00', '18:00'])
    grid: str = '1.0/1.0'  # Coarser grid to reduce data volume
    area: Optional[list[float]] = None  # [N, W, S, E] bounds
    format: str = 'netcdf'

    def to_dict(self) -> dict:
        """Convert to CDS API request dictionary."""
        request = {
            'product_type': 'reanalysis',
            'variable': self.variable,
            'year': str(self.year),
            'month': f'{self.month:02d}',
            'day': [f'{d:02d}' for d in self.days],
            'time': self.hours,
            'grid': self.grid,
            'format': self.format,
        }
        if self.area:
            request['area'] = self.area
        return request


@dataclass
class ERA5DailyStatsRequest:
    """Configuration for ERA5 daily statistics request."""

    variable: str
    year: int
    month: int
    days: list[int] = field(default_factory=lambda: list(range(1, 32)))
    daily_statistic: str = 'daily_mean'  # daily_mean, daily_max, daily_min
    time_zone: str = 'utc+00:00'
    frequency: str = '1_hourly'

    def to_dict(self) -> dict:
        """Convert to CDS API request dictionary."""
        return {
            'product_type': 'reanalysis',
            'variable': self.variable,
            'year': str(self.year),
            'month': f'{self.month:02d}',
            'day': [f'{d:02d}' for d in self.days],
            'daily_statistic': self.daily_statistic,
            'time_zone': self.time_zone,
            'frequency': self.frequency,
        }


class ERA5Client:
    """Client for downloading ERA5 gridded data from CDS API."""

    # Available ERA5 datasets
    DATASETS = {
        'single_levels': 'reanalysis-era5-single-levels',
        'pressure_levels': 'reanalysis-era5-pressure-levels',
        'daily_stats': 'derived-era5-single-levels-daily-statistics',
        'monthly': 'reanalysis-era5-single-levels-monthly-means',
        'land': 'reanalysis-era5-land',
    }

    # Common variables for surface data
    VARIABLES = {
        '2m_temperature': '2m_temperature',
        'sea_surface_temperature': 'sea_surface_temperature',
        'total_precipitation': 'total_precipitation',
        '10m_wind_speed': '10m_wind_speed',
        'surface_pressure': 'surface_pressure',
        'mean_sea_level_pressure': 'mean_sea_level_pressure',
    }

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: int = 60,
    ):
        """
        Initialize ERA5 client.

        Args:
            output_dir: Directory for downloaded files (default: data/era5)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        if not CDSAPI_AVAILABLE:
            raise ImportError(
                "cdsapi package not installed. "
                "Run: pip install 'cdsapi>=0.7.7'"
            )

        self.output_dir = output_dir or Path(__file__).parent.parent / 'data' / 'era5'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize CDS client
        self._client = None

    @property
    def client(self) -> 'cdsapi.Client':
        """Lazy initialization of CDS client."""
        if self._client is None:
            self._client = cdsapi.Client(timeout=self.timeout)
        return self._client

    def _retrieve_with_retry(
        self,
        dataset: str,
        request: dict,
        output_path: Path,
    ) -> Path:
        """
        Retrieve data with exponential backoff retry logic.

        Args:
            dataset: CDS dataset name
            request: Request parameters
            output_path: Output file path

        Returns:
            Path to downloaded file
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {output_path.name} (attempt {attempt + 1}/{self.max_retries})")
                self.client.retrieve(dataset, request, str(output_path))
                logger.info(f"Successfully downloaded {output_path.name}")
                return output_path

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")

        raise last_error

    def download_hourly(
        self,
        variable: str,
        year: int,
        month: int,
        days: Optional[list[int]] = None,
        hours: Optional[list[str]] = None,
        grid: str = '1.0/1.0',
        area: Optional[list[float]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download ERA5 hourly data for a single month.

        Args:
            variable: Variable name (e.g., '2m_temperature')
            year: Year to download
            month: Month to download (1-12)
            days: List of days (default: all days in month)
            hours: List of hours (default: 00, 06, 12, 18)
            grid: Grid resolution (default: 1.0/1.0 degrees)
            area: Bounding box [N, W, S, E] (default: global)
            output_path: Output file path (default: auto-generated)

        Returns:
            Path to downloaded NetCDF file
        """
        request = ERA5Request(
            variable=variable,
            year=year,
            month=month,
            days=days or list(range(1, 32)),
            hours=hours or ['00:00', '06:00', '12:00', '18:00'],
            grid=grid,
            area=area,
        )

        if output_path is None:
            var_short = variable.replace('_', '')[:8]
            output_path = self.output_dir / f'era5_{var_short}_{year}_{month:02d}.nc'

        return self._retrieve_with_retry(
            self.DATASETS['single_levels'],
            request.to_dict(),
            output_path,
        )

    def download_daily_stats(
        self,
        variable: str,
        year: int,
        month: int,
        days: Optional[list[int]] = None,
        statistic: str = 'daily_mean',
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download ERA5 daily statistics (pre-computed mean/max/min).

        This is more efficient than downloading hourly data and computing
        statistics locally.

        Args:
            variable: Variable name (e.g., '2m_temperature')
            year: Year to download
            month: Month to download (1-12)
            days: List of days (default: all days in month)
            statistic: 'daily_mean', 'daily_max', or 'daily_min'
            output_path: Output file path (default: auto-generated)

        Returns:
            Path to downloaded NetCDF file
        """
        request = ERA5DailyStatsRequest(
            variable=variable,
            year=year,
            month=month,
            days=days or list(range(1, 32)),
            daily_statistic=statistic,
        )

        if output_path is None:
            var_short = variable.replace('_', '')[:8]
            stat_short = statistic.split('_')[1][:4]
            output_path = self.output_dir / f'era5_{var_short}_{stat_short}_{year}_{month:02d}.nc'

        return self._retrieve_with_retry(
            self.DATASETS['daily_stats'],
            request.to_dict(),
            output_path,
        )

    def download_monthly(
        self,
        variable: str,
        years: list[int],
        months: Optional[list[int]] = None,
        grid: str = '1.0/1.0',
        area: Optional[list[float]] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Download ERA5 monthly means.

        Args:
            variable: Variable name (e.g., '2m_temperature')
            years: List of years to download
            months: List of months (default: all 12 months)
            grid: Grid resolution (default: 1.0/1.0 degrees)
            area: Bounding box [N, W, S, E] (default: global)
            output_path: Output file path (default: auto-generated)

        Returns:
            Path to downloaded NetCDF file
        """
        request = {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': variable,
            'year': [str(y) for y in years],
            'month': [f'{m:02d}' for m in (months or range(1, 13))],
            'time': '00:00',
            'grid': grid,
            'format': 'netcdf',
        }
        if area:
            request['area'] = area

        if output_path is None:
            var_short = variable.replace('_', '')[:8]
            year_range = f'{min(years)}-{max(years)}' if len(years) > 1 else str(years[0])
            output_path = self.output_dir / f'era5_{var_short}_monthly_{year_range}.nc'

        return self._retrieve_with_retry(
            self.DATASETS['monthly'],
            request,
            output_path,
        )

    def download_all_monthly_means(
        self,
        variable: str = '2m_temperature',
        start_year: int = 1940,
        end_year: Optional[int] = None,
        grid: str = '0.25/0.25',
        output_path: Optional[Path] = None,
        chunk_size: int = 5,
    ) -> Path:
        """
        Download ERA5 monthly means from 1940-present as a consolidated file.

        Downloads in yearly chunks to respect CDS API limits, then consolidates
        into a single NetCDF file.

        Args:
            variable: Variable name (default: '2m_temperature')
            start_year: First year to download (default: 1940)
            end_year: Last year to download (default: current year)
            grid: Grid resolution (default: 0.25/0.25 degrees for full resolution)
            output_path: Final output file path (default: auto-generated)
            chunk_size: Number of years per chunk (default: 5 for 0.25deg data)

        Returns:
            Path to consolidated NetCDF file
        """
        from datetime import datetime

        if end_year is None:
            end_year = datetime.now().year

        if output_path is None:
            var_short = variable.replace('_', '')[:3]
            output_path = self.output_dir / f'era5_{var_short}_monthly_{start_year}_present.nc'

        # Check if file already exists
        if output_path.exists():
            logger.info(f"Monthly data file already exists: {output_path}")
            return output_path

        logger.info(f"Downloading ERA5 monthly means {start_year}-{end_year}...")

        # Download in chunks to avoid API limits
        temp_files = []
        years = list(range(start_year, end_year + 1))

        for i in range(0, len(years), chunk_size):
            chunk_years = years[i:i + chunk_size]
            chunk_file = self.output_dir / f'era5_monthly_chunk_{chunk_years[0]}_{chunk_years[-1]}.nc'

            if chunk_file.exists():
                logger.info(f"Chunk {chunk_years[0]}-{chunk_years[-1]} already exists, skipping")
                temp_files.append(chunk_file)
                continue

            try:
                request = {
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': variable,
                    'year': [str(y) for y in chunk_years],
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'time': '00:00',
                    'grid': grid,
                    'data_format': 'netcdf',
                }

                self._retrieve_with_retry(
                    self.DATASETS['monthly'],
                    request,
                    chunk_file,
                )
                temp_files.append(chunk_file)

            except Exception as e:
                logger.error(f"Failed to download chunk {chunk_years[0]}-{chunk_years[-1]}: {e}")
                # Continue with remaining chunks
                continue

        # Consolidate chunks into single file
        if temp_files and XARRAY_AVAILABLE:
            logger.info("Consolidating chunks into single file...")
            try:
                import xarray as xr
                datasets = [xr.open_dataset(f) for f in temp_files]
                combined = xr.concat(datasets, dim='time')
                combined = combined.sortby('time')
                combined.to_netcdf(output_path)

                # Close datasets
                for ds in datasets:
                    ds.close()

                # Remove temp chunk files
                for f in temp_files:
                    if f != output_path:
                        try:
                            f.unlink()
                        except Exception:
                            pass

                logger.info(f"Consolidated monthly data saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to consolidate chunks: {e}")
                # Return last chunk file if consolidation fails
                if temp_files:
                    return temp_files[-1]
                raise

        return output_path

    def download_year(
        self,
        variable: str,
        year: int,
        use_daily_stats: bool = True,
        statistic: str = 'daily_mean',
        grid: str = '1.0/1.0',
        area: Optional[list[float]] = None,
    ) -> list[Path]:
        """
        Download a full year of ERA5 data, month by month.

        Breaking into monthly chunks avoids CDS request limits and
        improves queue priority.

        Args:
            variable: Variable name (e.g., '2m_temperature')
            year: Year to download
            use_daily_stats: If True, use daily statistics endpoint
            statistic: Daily statistic type (if use_daily_stats=True)
            grid: Grid resolution for hourly data
            area: Bounding box [N, W, S, E]

        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []

        for month in range(1, 13):
            try:
                if use_daily_stats:
                    path = self.download_daily_stats(
                        variable=variable,
                        year=year,
                        month=month,
                        statistic=statistic,
                    )
                else:
                    path = self.download_hourly(
                        variable=variable,
                        year=year,
                        month=month,
                        grid=grid,
                        area=area,
                    )
                downloaded_files.append(path)

            except Exception as e:
                logger.error(f"Failed to download {year}-{month:02d}: {e}")
                continue

        return downloaded_files


def load_era5_netcdf(file_path: Path) -> 'xr.Dataset':
    """
    Load ERA5 NetCDF file into xarray Dataset.

    Args:
        file_path: Path to NetCDF file

    Returns:
        xarray Dataset
    """
    if not XARRAY_AVAILABLE:
        raise ImportError(
            "xarray package not installed. "
            "Run: pip install xarray netCDF4"
        )

    return xr.open_dataset(file_path)


def get_era5t_latest_date() -> datetime:
    """
    Query the CDS catalogue API for the latest available ERA5T date.

    This is a public, unauthenticated REST endpoint — no API key needed.
    Falls back to today - 5 days if the request fails for any reason.

    Returns:
        datetime of the last day for which ERA5T data is available.
    """
    import urllib.request
    import json
    from datetime import timedelta

    url = (
        'https://cds.climate.copernicus.eu/api/catalogue/v1/collections/'
        'reanalysis-era5-single-levels'
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        # extent.temporal.interval is [[start, end]]
        end_str = data['extent']['temporal']['interval'][0][1]
        # ISO 8601 with timezone offset — strip offset for naive datetime
        end_str = end_str.split('+')[0].rstrip('Z')
        latest = datetime.fromisoformat(end_str)
        logger.info(f"CDS catalogue reports ERA5T latest date: {latest.date()}")
        return latest
    except Exception as exc:
        fallback = datetime.now() - timedelta(days=5)
        logger.warning(
            f"Could not query CDS catalogue API ({exc}). "
            f"Falling back to today - 5 days: {fallback.date()}"
        )
        return fallback


def check_cds_config() -> bool:
    """
    Check if CDS API is properly configured.

    Returns:
        True if configuration is valid
    """
    config_path = Path.home() / '.cdsapirc'

    if not config_path.exists():
        logger.error(
            f"CDS configuration file not found at {config_path}\n"
            "Create it with:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <YOUR-PERSONAL-ACCESS-TOKEN>\n\n"
            "Get your key from: https://cds.climate.copernicus.eu/how-to-api"
        )
        return False

    content = config_path.read_text()
    if 'url:' not in content or 'key:' not in content:
        logger.error(
            f"Invalid CDS configuration in {config_path}\n"
            "Required format:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <YOUR-PERSONAL-ACCESS-TOKEN>"
        )
        return False

    logger.info("CDS API configuration found")
    return True


if __name__ == "__main__":
    # Example usage
    import sys

    # Check configuration
    if not check_cds_config():
        print("\nPlease configure CDS API before running.")
        print("1. Register at https://cds.climate.copernicus.eu")
        print("2. Get your API key from https://cds.climate.copernicus.eu/how-to-api")
        print("3. Create ~/.cdsapirc with your credentials")
        sys.exit(1)

    if not CDSAPI_AVAILABLE:
        print("Please install cdsapi: pip install 'cdsapi>=0.7.7'")
        sys.exit(1)

    # Initialize client
    client = ERA5Client()

    # Example: Download January 2024 daily mean temperature
    print("\nDownloading ERA5 daily mean 2m temperature for January 2024...")
    try:
        output = client.download_daily_stats(
            variable='2m_temperature',
            year=2024,
            month=1,
            statistic='daily_mean',
        )
        print(f"Downloaded: {output}")

        # Load and inspect data
        if XARRAY_AVAILABLE:
            ds = load_era5_netcdf(output)
            print(f"\nDataset variables: {list(ds.data_vars)}")
            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

    except Exception as e:
        print(f"Download failed: {e}")
        print("\nMake sure you have:")
        print("1. Accepted the ERA5 license at https://cds.climate.copernicus.eu")
        print("2. Valid API credentials in ~/.cdsapirc")
