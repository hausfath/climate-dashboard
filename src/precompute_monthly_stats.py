"""
ERA5 Monthly Statistics Pre-computation Script.

Runs once after data download to pre-compute:
1. Monthly statistics (global + continental + country) for all months
2. 3D geometry cache (coastlines, borders, graticules)
3. Pre-rendered 2D map images (both absolute and anomaly views)

Output: data/era5/precomputed/
    ├── monthly_stats.parquet        # All statistics
    ├── geometry_cache.npz           # 3D geometry arrays
    └── maps/                         # Pre-rendered 2D maps
        ├── abs/                      # Absolute temperature maps
        └── anom/                     # Anomaly maps

Usage:
    python src/precompute_monthly_stats.py
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import json
import urllib.request
from datetime import datetime
import sys
import io

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'era5'
PRECOMPUTED_DIR = DATA_DIR / 'precomputed'
MAPS_DIR = PRECOMPUTED_DIR / 'maps'
MONTHLY_DATA_FILE = DATA_DIR / 'era5_t2m_monthly_1940_present.nc'
CLIM_FILE = DATA_DIR / 'era5_t2m_climatology_1991_2020.nc'
COASTLINES_FILE = DATA_DIR / 'ne_50m_coastline.geojson'
BORDERS_FILE = DATA_DIR / 'ne_50m_admin_0_boundary_lines_land.geojson'

# Continent bounding boxes (approximate, for fallback when regionmask unavailable)
# Using 0-360 longitude format to match ERA5 data
CONTINENTS = {
    'africa': {'lon': (340, 52), 'lat': (-35, 37), 'wrap': True},  # wraps around 0
    'antarctica': {'lon': (0, 360), 'lat': (-90, -60), 'wrap': False},
    'asia': {'lon': (25, 180), 'lat': (5, 82), 'wrap': False},
    'europe': {'lon': (350, 60), 'lat': (35, 72), 'wrap': True},  # wraps around 0
    'north_america': {'lon': (190, 310), 'lat': (7, 84), 'wrap': False},  # 170W to 50W
    'oceania': {'lon': (110, 180), 'lat': (-50, 0), 'wrap': False},
    'south_america': {'lon': (278, 326), 'lat': (-56, 13), 'wrap': False},  # 82W to 34W
}

# Country bounding boxes for statistics (using 0-360 longitude)
COUNTRIES = {
    'USA': {'lon': (235, 294), 'lat': (24, 50)},  # 125W to 66W
    'China': {'lon': (73, 135), 'lat': (18, 54)},
    'India': {'lon': (68, 97), 'lat': (8, 37)},
    'Australia': {'lon': (113, 154), 'lat': (-44, -10)},
    'EU': {'lon': (350, 40), 'lat': (35, 72), 'wrap': True},  # 10W to 40E, wraps
    'Russia': {'lon': (20, 180), 'lat': (41, 82)},
    'Brazil': {'lon': (286, 326), 'lat': (-34, 5)},  # 74W to 34W
}


def ensure_directories():
    """Create output directories if they don't exist."""
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    (MAPS_DIR / 'abs').mkdir(parents=True, exist_ok=True)
    (MAPS_DIR / 'anom').mkdir(parents=True, exist_ok=True)
    print(f"Output directories ready: {PRECOMPUTED_DIR}")


def download_natural_earth_data():
    """Download Natural Earth coastlines and borders if not present."""
    base_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"

    files = [
        (COASTLINES_FILE, f"{base_url}/ne_50m_coastline.geojson"),
        (BORDERS_FILE, f"{base_url}/ne_50m_admin_0_boundary_lines_land.geojson"),
    ]

    for local_file, url in files:
        if not local_file.exists():
            print(f"Downloading {local_file.name}...")
            try:
                urllib.request.urlretrieve(url, local_file)
                print(f"  Downloaded {local_file.name}")
            except Exception as e:
                print(f"  Failed to download {local_file.name}: {e}")


def load_geojson_lines(filepath: Path, radius: float = 1.003) -> tuple:
    """
    Load GeoJSON LineString/MultiLineString features and convert to 3D coordinates.

    Returns:
        Tuple of (x_arrays, y_arrays, z_arrays) - lists of numpy arrays for each line segment
    """
    if not filepath.exists():
        return [], [], []

    with open(filepath, 'r') as f:
        data = json.load(f)

    x_arrays = []
    y_arrays = []
    z_arrays = []

    for feature in data['features']:
        geom = feature['geometry']

        def process_coords(coords):
            lons = np.array([c[0] for c in coords])
            lats = np.array([c[1] for c in coords])

            lat_rad = np.deg2rad(lats)
            lon_rad = np.deg2rad(lons)

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)

            return x, y, z

        if geom['type'] == 'LineString':
            x, y, z = process_coords(geom['coordinates'])
            x_arrays.append(x)
            y_arrays.append(y)
            z_arrays.append(z)
        elif geom['type'] == 'MultiLineString':
            for line_coords in geom['coordinates']:
                x, y, z = process_coords(line_coords)
                x_arrays.append(x)
                y_arrays.append(y)
                z_arrays.append(z)

    return x_arrays, y_arrays, z_arrays


def precompute_geometry():
    """
    Pre-compute all 3D geometry: coastlines, borders, and graticules.

    Saves to geometry_cache.npz with keys:
    - coastlines_x, coastlines_y, coastlines_z
    - borders_x, borders_y, borders_z
    - graticule_lat_x, graticule_lat_y, graticule_lat_z
    - graticule_lon_x, graticule_lon_y, graticule_lon_z
    """
    print("Pre-computing 3D geometry...")

    download_natural_earth_data()

    # Load coastlines
    print("  Loading coastlines...")
    coast_x, coast_y, coast_z = load_geojson_lines(COASTLINES_FILE, radius=1.003)

    # Load borders
    print("  Loading borders...")
    border_x, border_y, border_z = load_geojson_lines(BORDERS_FILE, radius=1.003)

    # Generate graticule lines
    print("  Generating graticules...")
    r = 1.002  # Slightly smaller radius for graticules

    # Latitude lines (every 30 degrees from -60 to 90)
    grat_lat_x = []
    grat_lat_y = []
    grat_lat_z = []

    for lat in range(-60, 91, 30):
        lat_r = np.deg2rad(lat)
        lons_line = np.linspace(0, 360, 180)
        lons_r = np.deg2rad(lons_line)

        x_line = r * np.cos(lat_r) * np.cos(lons_r)
        y_line = r * np.cos(lat_r) * np.sin(lons_r)
        z_line = r * np.sin(lat_r) * np.ones_like(lons_r)

        grat_lat_x.append(x_line)
        grat_lat_y.append(y_line)
        grat_lat_z.append(z_line)

    # Longitude lines (every 30 degrees from 0 to 330)
    grat_lon_x = []
    grat_lon_y = []
    grat_lon_z = []

    for lon in range(0, 360, 30):
        lon_r = np.deg2rad(lon)
        lats_line = np.linspace(-90, 90, 180)
        lats_r = np.deg2rad(lats_line)

        x_line = r * np.cos(lats_r) * np.cos(lon_r)
        y_line = r * np.cos(lats_r) * np.sin(lon_r)
        z_line = r * np.sin(lats_r)

        grat_lon_x.append(x_line)
        grat_lon_y.append(y_line)
        grat_lon_z.append(z_line)

    # Save to npz file using object arrays to preserve variable-length segments
    cache_file = PRECOMPUTED_DIR / 'geometry_cache.npz'
    np.savez_compressed(
        cache_file,
        coastlines_x=np.array(coast_x, dtype=object),
        coastlines_y=np.array(coast_y, dtype=object),
        coastlines_z=np.array(coast_z, dtype=object),
        borders_x=np.array(border_x, dtype=object),
        borders_y=np.array(border_y, dtype=object),
        borders_z=np.array(border_z, dtype=object),
        graticule_lat_x=np.array(grat_lat_x, dtype=object),
        graticule_lat_y=np.array(grat_lat_y, dtype=object),
        graticule_lat_z=np.array(grat_lat_z, dtype=object),
        graticule_lon_x=np.array(grat_lon_x, dtype=object),
        graticule_lon_y=np.array(grat_lon_y, dtype=object),
        graticule_lon_z=np.array(grat_lon_z, dtype=object),
    )

    print(f"  Saved geometry cache to {cache_file}")
    print(f"    Coastline segments: {len(coast_x)}")
    print(f"    Border segments: {len(border_x)}")
    print(f"    Graticule latitude lines: {len(grat_lat_x)}")
    print(f"    Graticule longitude lines: {len(grat_lon_x)}")


def calculate_regional_stats_for_month(
    temp_data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    clim_data: np.ndarray = None,
) -> dict:
    """
    Calculate temperature statistics for a single month.

    Returns dict with keys for global, continental, and country statistics.
    """
    try:
        import regionmask
    except ImportError:
        regionmask = None

    stats = {}

    # Create latitude weights for area-weighted means
    weights = np.cos(np.deg2rad(lats))
    weights_2d = np.broadcast_to(weights[:, np.newaxis], temp_data.shape)

    # Global statistics
    global_mean = np.nansum(temp_data * weights_2d) / np.nansum(weights_2d * ~np.isnan(temp_data))
    global_min = np.nanmin(temp_data)
    global_max = np.nanmax(temp_data)

    if clim_data is not None:
        clim_mean = np.nansum(clim_data * weights_2d) / np.nansum(weights_2d * ~np.isnan(clim_data))
        global_anomaly = global_mean - clim_mean
    else:
        global_anomaly = np.nan

    stats['global_mean'] = global_mean
    stats['global_min'] = global_min
    stats['global_max'] = global_max
    stats['global_anomaly'] = global_anomaly

    # Continental statistics using regionmask
    if regionmask is not None:
        try:
            continents = regionmask.defined_regions.natural_earth_v5_0_0.continents_110
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            mask = continents.mask(lon_grid, lat_grid)

            continent_keys = {
                'Africa': 'africa',
                'Antarctica': 'antarctica',
                'Asia': 'asia',
                'Europe': 'europe',
                'North America': 'north_america',
                'Oceania': 'oceania',
                'South America': 'south_america',
            }

            for i, name in enumerate(continents.names):
                if name not in continent_keys:
                    continue

                key = continent_keys[name]
                region_mask = mask == i

                if not np.any(region_mask):
                    stats[f'{key}_mean'] = np.nan
                    stats[f'{key}_min'] = np.nan
                    stats[f'{key}_max'] = np.nan
                    stats[f'{key}_anomaly'] = np.nan
                    continue

                masked_temp = np.where(region_mask, temp_data, np.nan)
                masked_weights = np.where(region_mask, weights_2d, 0)

                mean_temp = np.nansum(masked_temp * masked_weights) / np.nansum(masked_weights)
                min_temp = np.nanmin(masked_temp)
                max_temp = np.nanmax(masked_temp)

                if clim_data is not None:
                    masked_clim = np.where(region_mask, clim_data, np.nan)
                    clim_mean = np.nansum(masked_clim * masked_weights) / np.nansum(masked_weights)
                    anomaly = mean_temp - clim_mean
                else:
                    anomaly = np.nan

                stats[f'{key}_mean'] = mean_temp
                stats[f'{key}_min'] = min_temp
                stats[f'{key}_max'] = max_temp
                stats[f'{key}_anomaly'] = anomaly

        except Exception as e:
            print(f"    Warning: regionmask error: {e}, using bounding boxes")
            regionmask = None  # Fall through to bounding box method

    # Fallback: use bounding boxes for continents if regionmask unavailable
    if regionmask is None:
        for key, bounds in CONTINENTS.items():
            lon_min, lon_max = bounds['lon']
            lat_min, lat_max = bounds['lat']
            wrap = bounds.get('wrap', False)

            # Create longitude mask (handle wrapping around 0/360)
            if wrap:
                lon_mask = (lons >= lon_min) | (lons <= lon_max)
            else:
                lon_mask = (lons >= lon_min) & (lons <= lon_max)

            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            region_mask = np.outer(lat_mask, lon_mask)

            if not np.any(region_mask):
                stats[f'{key}_mean'] = np.nan
                stats[f'{key}_min'] = np.nan
                stats[f'{key}_max'] = np.nan
                stats[f'{key}_anomaly'] = np.nan
                continue

            masked_temp = np.where(region_mask, temp_data, np.nan)
            masked_weights = np.where(region_mask, weights_2d, 0)

            mean_temp = np.nansum(masked_temp * masked_weights) / np.nansum(masked_weights)
            min_temp = np.nanmin(masked_temp)
            max_temp = np.nanmax(masked_temp)

            if clim_data is not None:
                masked_clim = np.where(region_mask, clim_data, np.nan)
                clim_mean = np.nansum(masked_clim * masked_weights) / np.nansum(masked_weights)
                anomaly = mean_temp - clim_mean
            else:
                anomaly = np.nan

            stats[f'{key}_mean'] = mean_temp
            stats[f'{key}_min'] = min_temp
            stats[f'{key}_max'] = max_temp
            stats[f'{key}_anomaly'] = anomaly

    # Country statistics using bounding boxes
    country_keys = {
        'USA': 'usa',
        'China': 'china',
        'India': 'india',
        'Australia': 'australia',
        'EU': 'eu',
        'Russia': 'russia',
        'Brazil': 'brazil',
    }

    for country, bounds in COUNTRIES.items():
        key = country_keys[country]
        lon_min, lon_max = bounds['lon']
        lat_min, lat_max = bounds['lat']
        wrap = bounds.get('wrap', False)

        # Create longitude mask (handle wrapping around 0/360)
        if wrap:
            lon_mask = (lons >= lon_min) | (lons <= lon_max)
        else:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)

        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        region_mask = np.outer(lat_mask, lon_mask)

        if not np.any(region_mask):
            stats[f'{key}_mean'] = np.nan
            stats[f'{key}_min'] = np.nan
            stats[f'{key}_max'] = np.nan
            stats[f'{key}_anomaly'] = np.nan
            continue

        masked_temp = np.where(region_mask, temp_data, np.nan)
        masked_weights = np.where(region_mask, weights_2d, 0)

        mean_temp = np.nansum(masked_temp * masked_weights) / np.nansum(masked_weights)
        min_temp = np.nanmin(masked_temp)
        max_temp = np.nanmax(masked_temp)

        if clim_data is not None:
            masked_clim = np.where(region_mask, clim_data, np.nan)
            clim_mean = np.nansum(masked_clim * masked_weights) / np.nansum(masked_weights)
            anomaly = mean_temp - clim_mean
        else:
            anomaly = np.nan

        stats[f'{key}_mean'] = mean_temp
        stats[f'{key}_min'] = min_temp
        stats[f'{key}_max'] = max_temp
        stats[f'{key}_anomaly'] = anomaly

    return stats


def precompute_monthly_stats():
    """
    Pre-compute statistics for all months in the dataset.

    Saves to monthly_stats.parquet with columns:
    - year, month
    - global_mean, global_min, global_max, global_anomaly
    - {continent}_mean, {continent}_min, {continent}_max, {continent}_anomaly
    - {country}_mean, {country}_min, {country}_max, {country}_anomaly
    """
    print("Pre-computing monthly statistics...")

    # Check for data file
    if not MONTHLY_DATA_FILE.exists():
        print(f"  Error: Monthly data file not found: {MONTHLY_DATA_FILE}")
        print("  Please download ERA5 data first using the ERA5 API")
        return False

    # Load monthly dataset
    print(f"  Loading monthly data from {MONTHLY_DATA_FILE}...")
    ds = xr.open_dataset(MONTHLY_DATA_FILE)
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Load climatology
    print(f"  Loading climatology from {CLIM_FILE}...")
    if CLIM_FILE.exists():
        clim_ds = xr.open_dataset(CLIM_FILE)
        climatology = {m: clim_ds['t2m'].sel(month=m).values - 273.15 for m in range(1, 13)}
        clim_ds.close()
    else:
        print("  Warning: Climatology file not found, anomalies will be NaN")
        climatology = None

    # Get all available times
    times = ds['time'].values
    total_months = len(times)
    print(f"  Processing {total_months} months...")

    all_stats = []

    for idx, t in enumerate(times):
        dt = np.datetime64(t, 'M')
        year = int(str(dt)[:4])
        month = int(str(dt)[5:7])

        if (idx + 1) % 50 == 0 or idx == 0 or idx == total_months - 1:
            print(f"    Processing {year}-{month:02d} ({idx + 1}/{total_months})...")

        # Load temperature data for this month
        time_str = f'{year}-{month:02d}'
        try:
            month_data = ds['t2m'].sel(time=time_str, method='nearest')
        except Exception:
            target_time = np.datetime64(f'{year}-{month:02d}-01')
            month_data = ds['t2m'].sel(time=target_time, method='nearest')

        temp_c = month_data.values - 273.15

        # Get climatology for this month
        clim_data = climatology[month] if climatology else None

        # Calculate statistics
        stats = calculate_regional_stats_for_month(temp_c, lats, lons, clim_data)
        stats['year'] = year
        stats['month'] = month

        all_stats.append(stats)

    ds.close()

    # Create DataFrame and save to parquet
    df = pd.DataFrame(all_stats)

    # Reorder columns
    cols = ['year', 'month',
            'global_mean', 'global_min', 'global_max', 'global_anomaly']

    for region in ['africa', 'antarctica', 'asia', 'europe', 'north_america', 'oceania', 'south_america']:
        cols.extend([f'{region}_mean', f'{region}_min', f'{region}_max', f'{region}_anomaly'])

    for country in ['usa', 'china', 'india', 'australia', 'eu', 'russia', 'brazil']:
        cols.extend([f'{country}_mean', f'{country}_min', f'{country}_max', f'{country}_anomaly'])

    # Ensure all columns exist (in case some regions failed)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[cols]

    # Save to CSV (parquet requires pyarrow, CSV is more portable)
    stats_file = PRECOMPUTED_DIR / 'monthly_stats.csv'
    df.to_csv(stats_file, index=False)

    print(f"  Saved statistics to {stats_file}")
    print(f"    Total months: {len(df)}")
    print(f"    Date range: {df['year'].min()}-{df['month'].iloc[0]:02d} to {df['year'].max()}-{df['month'].iloc[-1]:02d}")
    print(f"    File size: {stats_file.stat().st_size / 1024:.1f} KB")

    return True


def render_map_image(
    temp_data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    year: int,
    month: int,
    show_anomaly: bool = False,
    clim_data: np.ndarray = None,
    dpi: int = 150,
) -> bytes:
    """
    Render a 2D Mollweide map as a PNG image.

    Returns PNG bytes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point

    # Create figure with Mollweide projection - consistent size for alignment
    fig = plt.figure(figsize=(10, 5.5), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())

    # Set background color
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Calculate color data
    if show_anomaly and clim_data is not None:
        color_data = temp_data - clim_data
        vmin, vmax = -10, 10
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'anomaly',
            ['#05406a', '#5ba3c0', '#ffffff', '#d4604a', '#5f100e'],
            N=256
        )
        label = 'Temperature Anomaly (°C)'
    else:
        color_data = temp_data
        vmin, vmax = -28, 40
        # ECMWF-style colormap
        colors = [
            '#800080', '#6600aa', '#4000d4', '#0000ff', '#0066ff', '#00ccff',
            '#00ffcc', '#00ff66', '#66ff00', '#ccff00', '#ffff00', '#ffcc00',
            '#ff8800', '#ff4400', '#ff0000', '#cc0000',
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list('ecmwf', colors, N=256)
        label = 'Temperature (°C)'

    # Drop the 180° longitude column.  ERA5 includes a point at exactly 180°,
    # and the pcolormesh cell centred there straddles the Mollweide projection
    # boundary (±180°) at every latitude, producing a horizontal band artifact
    # across Antarctica where meridians run nearly horizontally.  Removing that
    # single column leaves a sub-pixel (~0.125°) gap at the map edge that is
    # completely invisible at this scale.
    lon_mask = lons != 180.0
    lons_use = lons[lon_mask]          # 0°, 0.25°, …, 179.75°, 180.25°, …, 359.75°
    color_data_use = color_data[:, lon_mask]

    # Convert to −180 → 180 and sort (no point falls exactly at ±180° now)
    lons_180 = np.where(lons_use > 180, lons_use - 360, lons_use)
    sort_idx = np.argsort(lons_180)
    lons_sorted = lons_180[sort_idx]   # −179.75°, …, 179.75°

    # Trim ±90° latitude rows — poles collapse to a point in Mollweide,
    # which causes a band artifact at the very bottom of Antarctica.
    pole_mask = np.abs(lats) < 90
    lats_plot = lats[pole_mask]
    data_plot = color_data_use[:, sort_idx][pole_mask, :]

    # Plot data — no add_cyclic_point needed; the tiny sub-pixel gap at ±180°
    # is invisible and avoids any boundary-crossing cell artifact.
    lon_grid, lat_grid = np.meshgrid(lons_sorted, lats_plot)
    mesh = ax.pcolormesh(
        lon_grid, lat_grid, data_plot,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='nearest',
    )

    # Add coastlines and borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='black', alpha=0.7)

    # Set global extent
    ax.set_global()

    # Add colorbar - consistent positioning
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08, shrink=0.85, aspect=40)
    cbar.set_label(label, color='#e6edf3', fontsize=11, labelpad=8)
    cbar.ax.tick_params(colors='#e6edf3', labelsize=9)

    # Consistent tight layout with fixed padding
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)

    # Save to bytes with specified DPI
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def precompute_map_images(skip_existing: bool = True, year_range: tuple = None):
    """
    Pre-render all monthly map images (both absolute and anomaly views).

    This is optional but provides instant 2D map loading.
    """
    print("Pre-rendering map images...")

    # Check for required libraries
    try:
        import matplotlib
        import cartopy
    except ImportError as e:
        print(f"  Skipping map rendering: {e}")
        print("  Install with: pip install matplotlib cartopy")
        return False

    # Check for data file
    if not MONTHLY_DATA_FILE.exists():
        print(f"  Error: Monthly data file not found: {MONTHLY_DATA_FILE}")
        return False

    # Load datasets
    print(f"  Loading data...")
    ds = xr.open_dataset(MONTHLY_DATA_FILE)
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Load climatology
    if CLIM_FILE.exists():
        clim_ds = xr.open_dataset(CLIM_FILE)
        climatology = {m: clim_ds['t2m'].sel(month=m).values - 273.15 for m in range(1, 13)}
        clim_ds.close()
    else:
        climatology = None

    times = ds['time'].values
    total_months = len(times)
    rendered_count = 0
    skipped_count = 0

    print(f"  Rendering {total_months} months × 2 views = {total_months * 2} images...")

    for idx, t in enumerate(times):
        dt = np.datetime64(t, 'M')
        year = int(str(dt)[:4])
        month = int(str(dt)[5:7])

        # File paths
        abs_file = MAPS_DIR / 'abs' / f'{year}_{month:02d}.png'
        anom_file = MAPS_DIR / 'anom' / f'{year}_{month:02d}.png'

        # Skip if outside requested year range
        if year_range is not None and not (year_range[0] <= year <= year_range[1]):
            skipped_count += 2
            continue

        # Skip if both exist and skip_existing is True
        if skip_existing and abs_file.exists() and anom_file.exists():
            skipped_count += 2
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    Rendering {year}-{month:02d} ({idx + 1}/{total_months})...")

        # Load temperature data
        time_str = f'{year}-{month:02d}'
        try:
            month_data = ds['t2m'].sel(time=time_str, method='nearest')
        except Exception:
            target_time = np.datetime64(f'{year}-{month:02d}-01')
            month_data = ds['t2m'].sel(time=target_time, method='nearest')

        temp_c = month_data.values - 273.15
        clim_data = climatology[month] if climatology else None

        # Render absolute map
        if not abs_file.exists() or not skip_existing:
            png_bytes = render_map_image(temp_c, lats, lons, year, month,
                                         show_anomaly=False, clim_data=None)
            with open(abs_file, 'wb') as f:
                f.write(png_bytes)
            rendered_count += 1

        # Render anomaly map
        if not anom_file.exists() or not skip_existing:
            if clim_data is not None:
                png_bytes = render_map_image(temp_c, lats, lons, year, month,
                                            show_anomaly=True, clim_data=clim_data)
                with open(anom_file, 'wb') as f:
                    f.write(png_bytes)
                rendered_count += 1

    ds.close()

    print(f"  Rendered {rendered_count} new images, skipped {skipped_count} existing")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in MAPS_DIR.glob('**/*.png'))
    print(f"  Total map images size: {total_size / 1024 / 1024:.1f} MB")

    return True


def download_current_month_mtd(force: bool = False) -> Path | None:
    """
    Download and average ERA5T daily means for the current month up to the
    latest available date, as reported by the CDS catalogue API.

    Skips the download if the existing file already covers that date (unless
    force=True).  Requires cdsapi and the CDS API key in ~/.cdsapirc.

    Returns:
        Path to the saved MTD NetCDF file, or None on failure.
    """
    try:
        from src.era5_api import ERA5Client, get_era5t_latest_date
    except ImportError:
        from era5_api import ERA5Client, get_era5t_latest_date

    now = datetime.now()
    year, month = now.year, now.month
    out_file = DATA_DIR / f'era5_2mtemper_mean_{year}_{month:02d}.nc'

    # Ask CDS what the latest available day is
    latest = get_era5t_latest_date()
    # Only use days within the current month
    if latest.year != year or latest.month != month:
        # Latest date is in a previous month — nothing to do for current month
        print(f"  Latest ERA5T date is {latest.date()}, not in current month {year}-{month:02d}")
        return None
    n_days = latest.day
    print(f"  CDS latest available date: {latest.date()} → downloading days 1-{n_days}")

    # Check if existing file already matches the CDS-reported date exactly
    if not force and out_file.exists():
        try:
            ds_check = xr.open_dataset(out_file)
            vt = ds_check['valid_time'].values
            ds_check.close()
            existing_day = int(str(np.datetime64(vt.flat[0], 'D'))[8:10])
            if existing_day == n_days:
                print(f"  File already covers day {existing_day} (matches CDS). Skipping download.")
                return out_file
            print(f"  Existing file covers day {existing_day}, CDS reports day {n_days}. Refreshing.")
        except Exception:
            pass  # Re-download if we can't read the existing file

    # Download daily means day by day via CDS API
    import tempfile
    try:
        client = ERA5Client(output_dir=DATA_DIR)
    except ImportError as exc:
        print(f"  Cannot download: {exc}")
        return None
    daily_files = []
    days = list(range(1, n_days + 1))

    print(f"  Downloading {len(days)} daily means from CDS (this may take several minutes)...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for day in days:
            day_file = tmp_path / f'era5_t2m_daily_{year}_{month:02d}_{day:02d}.nc'
            try:
                client._retrieve_with_retry(
                    'derived-era5-single-levels-daily-statistics',
                    {
                        'product_type': 'reanalysis',
                        'variable': '2m_temperature',
                        'year': str(year),
                        'month': f'{month:02d}',
                        'day': [f'{day:02d}'],
                        'daily_statistic': 'daily_mean',
                        'time_zone': 'utc+00:00',
                        'frequency': '1_hourly',
                    },
                    day_file,
                )
                daily_files.append(day_file)
            except Exception as e:
                print(f"    Warning: could not download day {day}: {e}")

        if not daily_files:
            print("  No daily files downloaded. Aborting.")
            return None

        # Load all daily files, average, save as single-timestep MTD file
        print(f"  Averaging {len(daily_files)} daily means...")
        datasets = [xr.open_dataset(f) for f in daily_files]
        combined = xr.concat(datasets, dim='valid_time')
        mtd_mean = combined['t2m'].mean(dim='valid_time')
        for ds in datasets:
            ds.close()

        # Build output dataset with the last day as the reference timestamp
        last_time = np.datetime64(f'{year}-{month:02d}-{n_days:02d}', 'D')
        out_ds = xr.Dataset(
            {'t2m': mtd_mean.expand_dims('valid_time')},
            coords={
                'latitude': combined['latitude'],
                'longitude': combined['longitude'],
                'valid_time': [last_time],
            },
        )
        out_ds.to_netcdf(out_file)
        print(f"  Saved MTD mean ({len(daily_files)} days) → {out_file.name}")
        return out_file


def precompute_current_month() -> bool:
    """
    Compute statistics and render map images for the current (incomplete) month.

    Scans data/era5/ for a partial-month ERA5 file (e.g. era5_2mtemper_mean_2026_02.nc),
    computes regional statistics, upserts the row into monthly_stats.csv, and renders
    both absolute and anomaly map PNGs.

    Designed to be called from a daily cron job so the spatial tab always shows
    the most recent available data for the current month.

    Returns True on success.
    """
    now = datetime.now()
    year, month = now.year, now.month

    print(f"Precomputing current month: {year}-{month:02d}...")

    # Attempt to download / refresh the MTD data file from CDS
    download_current_month_mtd()

    # Find partial-month data file (exclude consolidated/climatology files)
    candidates = sorted(DATA_DIR.glob(f'era5_*{year}*{month:02d}*.nc'))
    candidates = [p for p in candidates
                  if 'monthly_1940' not in p.name
                  and 'clim' not in p.name
                  and 'monthly_chunk' not in p.name]
    if not candidates:
        print(f"  No partial-month data file found for {year}-{month:02d}")
        return False

    data_file = candidates[0]
    print(f"  Loading {data_file.name}...")

    ds = xr.open_dataset(data_file)
    t2m = ds['t2m'].values
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    ds.close()

    if t2m.ndim == 3:
        t2m = t2m[0]
    temp_c = t2m - 273.15

    # Load climatology
    clim_data = None
    if CLIM_FILE.exists():
        clim_ds = xr.open_dataset(CLIM_FILE)
        clim_data = clim_ds['t2m'].sel(month=month).values - 273.15
        clim_ds.close()

    # Compute stats
    print(f"  Computing statistics...")
    stats = calculate_regional_stats_for_month(temp_c, lats, lons, clim_data)
    stats['year'] = year
    stats['month'] = month

    # Upsert into monthly_stats.csv
    stats_file = PRECOMPUTED_DIR / 'monthly_stats.csv'
    cols = ['year', 'month', 'global_mean', 'global_min', 'global_max', 'global_anomaly']
    for region in ['africa', 'antarctica', 'asia', 'europe', 'north_america', 'oceania', 'south_america']:
        cols.extend([f'{region}_mean', f'{region}_min', f'{region}_max', f'{region}_anomaly'])
    for country in ['usa', 'china', 'india', 'australia', 'eu', 'russia', 'brazil']:
        cols.extend([f'{country}_mean', f'{country}_min', f'{country}_max', f'{country}_anomaly'])

    if stats_file.exists():
        df = pd.read_csv(stats_file)
        df = df[~((df['year'] == year) & (df['month'] == month))]
    else:
        df = pd.DataFrame(columns=cols)

    for col in cols:
        if col not in stats:
            stats[col] = np.nan
    new_row = pd.DataFrame([{c: stats.get(c, np.nan) for c in cols}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(stats_file, index=False)
    print(f"  Updated {stats_file.name}")

    # Render map images
    ensure_directories()
    try:
        import matplotlib
        import cartopy  # noqa: F401 — ensures cartopy is available
        abs_file = MAPS_DIR / 'abs' / f'{year}_{month:02d}.png'
        anom_file = MAPS_DIR / 'anom' / f'{year}_{month:02d}.png'

        abs_file.write_bytes(render_map_image(temp_c, lats, lons, year, month,
                                              show_anomaly=False))
        print(f"  Saved {abs_file.name}")

        if clim_data is not None:
            anom_file.write_bytes(render_map_image(temp_c, lats, lons, year, month,
                                                   show_anomaly=True, clim_data=clim_data))
            print(f"  Saved {anom_file.name}")
    except ImportError as e:
        print(f"  Skipping map rendering: {e}")

    print(f"  Done: {year}-{month:02d} precomputed.")
    return True


def precompute_all(render_maps: bool = False):
    """
    Run all pre-computations.

    Args:
        render_maps: If True, also render all 2D map images (can take 30+ minutes)
    """
    print("=" * 60)
    print("ERA5 Pre-computation Script")
    print("=" * 60)
    start_time = datetime.now()

    # Create directories
    ensure_directories()

    # Pre-compute geometry (fast)
    precompute_geometry()
    print()

    # Pre-compute statistics (moderate)
    success = precompute_monthly_stats()
    if not success:
        print("\nFailed to compute statistics. Exiting.")
        return
    print()

    # Pre-render map images (slow, optional)
    if render_maps:
        precompute_map_images(skip_existing=True)
        print()
    else:
        print("Skipping map image rendering (use --render-maps to enable)")
        print()

    elapsed = datetime.now() - start_time
    print("=" * 60)
    print(f"Pre-computation complete! Total time: {elapsed}")
    print("=" * 60)


def verify_precomputed_data():
    """Verify that pre-computed data exists and is valid."""
    print("Verifying pre-computed data...")

    errors = []

    # Check geometry cache
    geom_file = PRECOMPUTED_DIR / 'geometry_cache.npz'
    if geom_file.exists():
        try:
            data = np.load(geom_file, allow_pickle=True)
            print(f"  Geometry cache: OK ({len(data['coastlines_x'])} coastline segments)")
        except Exception as e:
            errors.append(f"Geometry cache error: {e}")
    else:
        errors.append(f"Geometry cache not found: {geom_file}")

    # Check stats parquet
    stats_file = PRECOMPUTED_DIR / 'monthly_stats.parquet'
    if stats_file.exists():
        try:
            df = pd.read_parquet(stats_file)
            print(f"  Monthly stats: OK ({len(df)} months)")
        except Exception as e:
            errors.append(f"Stats parquet error: {e}")
    else:
        errors.append(f"Stats parquet not found: {stats_file}")

    # Check map images (count)
    abs_maps = list((MAPS_DIR / 'abs').glob('*.png'))
    anom_maps = list((MAPS_DIR / 'anom').glob('*.png'))
    print(f"  Absolute maps: {len(abs_maps)} images")
    print(f"  Anomaly maps: {len(anom_maps)} images")

    if errors:
        print("\nErrors found:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\nAll pre-computed data verified successfully!")
    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pre-compute ERA5 monthly statistics and geometry')
    parser.add_argument('--render-maps', action='store_true',
                        help='Also render all 2D map images (slow, ~30+ minutes)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing pre-computed data')
    parser.add_argument('--geometry-only', action='store_true',
                        help='Only pre-compute geometry cache')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only pre-compute monthly statistics')
    parser.add_argument('--maps-only', action='store_true',
                        help='Only render map images')

    args = parser.parse_args()

    if args.verify:
        verify_precomputed_data()
    elif args.geometry_only:
        ensure_directories()
        precompute_geometry()
    elif args.stats_only:
        ensure_directories()
        precompute_monthly_stats()
    elif args.maps_only:
        ensure_directories()
        precompute_map_images(skip_existing=True)
    else:
        precompute_all(render_maps=args.render_maps)
