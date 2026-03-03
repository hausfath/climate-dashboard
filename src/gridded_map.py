"""
ERA5 Monthly Temperature Globe with Regional Statistics.

Interactive 3D globe and 2D Mollweide map showing ERA5 monthly mean temperature
data from 1940-present, with regional statistics for continents and countries.

Run locally with: python src/gridded_map.py
"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table, no_update
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import urllib.request
from datetime import datetime
import base64

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'era5'
PRECOMPUTED_DIR = DATA_DIR / 'precomputed'
MONTHLY_DATA_FILE = DATA_DIR / 'era5_t2m_monthly_1940_present.nc'
CLIM_FILE = DATA_DIR / 'era5_t2m_climatology_1991_2020.nc'
COASTLINES_FILE = DATA_DIR / 'ne_50m_coastline.geojson'
BORDERS_FILE = DATA_DIR / 'ne_50m_admin_0_boundary_lines_land.geojson'

# Pre-computed data caches
_geometry_cache = None
_stats_df = None

# Available date range
START_YEAR = 1940
END_YEAR = datetime.now().year

MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# Country bounding boxes for statistics
COUNTRIES = {
    'USA': {'lon': (-125, -66), 'lat': (24, 50)},
    'China': {'lon': (73, 135), 'lat': (18, 54)},
    'India': {'lon': (68, 97), 'lat': (8, 37)},
    'Australia': {'lon': (113, 154), 'lat': (-44, -10)},
    'EU': {'lon': (-10, 40), 'lat': (35, 72)},
    'Russia': {'lon': (20, 180), 'lat': (41, 82)},
    'Brazil': {'lon': (-74, -34), 'lat': (-34, 5)},
}

# Dark theme
THEME = {
    'bg_color': '#0d1117',
    'paper_color': '#161b22',
    'text_color': '#e6edf3',
    'card_bg': '#21262d',
    'border_color': '#30363d',
    'accent_color': '#58a6ff',
}


def get_theme() -> dict:
    """Get theme configuration."""
    return THEME


def download_natural_earth_data():
    """Download Natural Earth coastlines and borders if not present (50m resolution)."""
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


def get_geometry_cache() -> dict:
    """Load pre-computed 3D geometry from cache file."""
    global _geometry_cache
    if _geometry_cache is None:
        cache_file = PRECOMPUTED_DIR / 'geometry_cache.npz'
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            _geometry_cache = {
                'coastlines': list(zip(data['coastlines_x'], data['coastlines_y'], data['coastlines_z'])),
                'borders': list(zip(data['borders_x'], data['borders_y'], data['borders_z'])),
                'graticule_lat': list(zip(data['graticule_lat_x'], data['graticule_lat_y'], data['graticule_lat_z'])),
                'graticule_lon': list(zip(data['graticule_lon_x'], data['graticule_lon_y'], data['graticule_lon_z'])),
            }
    return _geometry_cache


def get_stats_df() -> pd.DataFrame:
    """Load pre-computed monthly statistics from CSV file."""
    global _stats_df
    if _stats_df is None:
        stats_file = PRECOMPUTED_DIR / 'monthly_stats.csv'
        if stats_file.exists():
            _stats_df = pd.read_csv(stats_file)
    return _stats_df


def get_precomputed_map_path(year: int, month: int, show_anomaly: bool) -> Path:
    """Get path to pre-rendered map image."""
    subdir = 'anom' if show_anomaly else 'abs'
    return PRECOMPUTED_DIR / 'maps' / subdir / f'{year}_{month:02d}.png'


def lookup_precomputed_stats(year: int, month: int) -> dict:
    """
    Look up pre-computed statistics for a given month.

    Returns dict with keys:
        'global': {mean, min, max, anomaly}
        'continents': {name: {mean, min, max, anomaly}}
        'countries': {name: {mean, min, max, anomaly}}

    Returns None if pre-computed stats not available.
    """
    df = get_stats_df()
    if df is None:
        return None

    row = df[(df['year'] == year) & (df['month'] == month)]
    if row.empty:
        return None

    row = row.iloc[0]

    result = {
        'global': {
            'mean': row['global_mean'],
            'min': row['global_min'],
            'max': row['global_max'],
            'anomaly': row['global_anomaly'],
        },
        'continents': {},
        'countries': {},
    }

    # Continent name mapping
    continent_keys = {
        'africa': 'Africa',
        'antarctica': 'Antarctica',
        'asia': 'Asia',
        'europe': 'Europe',
        'north_america': 'North America',
        'oceania': 'Oceania',
        'south_america': 'South America',
    }

    for key, name in continent_keys.items():
        result['continents'][name] = {
            'mean': row[f'{key}_mean'],
            'min': row[f'{key}_min'],
            'max': row[f'{key}_max'],
            'anomaly': row[f'{key}_anomaly'],
        }

    # Country name mapping
    country_keys = {
        'usa': 'USA',
        'china': 'China',
        'india': 'India',
        'australia': 'Australia',
        'eu': 'EU',
        'russia': 'Russia',
        'brazil': 'Brazil',
    }

    for key, name in country_keys.items():
        result['countries'][name] = {
            'mean': row[f'{key}_mean'],
            'min': row[f'{key}_min'],
            'max': row[f'{key}_max'],
            'anomaly': row[f'{key}_anomaly'],
        }

    return result


def load_coastlines_3d(radius: float = 1.003) -> list:
    """Load Natural Earth coastlines and convert to 3D coordinates.

    Uses pre-computed cache if available for instant loading.
    """
    # Try to use pre-computed cache
    cache = get_geometry_cache()
    if cache is not None and 'coastlines' in cache:
        return cache['coastlines']

    # Fall back to computing from GeoJSON
    if not COASTLINES_FILE.exists():
        download_natural_earth_data()

    if not COASTLINES_FILE.exists():
        return []

    with open(COASTLINES_FILE, 'r') as f:
        data = json.load(f)

    lines = []
    for feature in data['features']:
        geom = feature['geometry']
        if geom['type'] == 'LineString':
            coords = geom['coordinates']
            lons = np.array([c[0] for c in coords])
            lats = np.array([c[1] for c in coords])

            lat_rad = np.deg2rad(lats)
            lon_rad = np.deg2rad(lons)

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)

            lines.append((x, y, z))
        elif geom['type'] == 'MultiLineString':
            for line_coords in geom['coordinates']:
                lons = np.array([c[0] for c in line_coords])
                lats = np.array([c[1] for c in line_coords])

                lat_rad = np.deg2rad(lats)
                lon_rad = np.deg2rad(lons)

                x = radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = radius * np.sin(lat_rad)

                lines.append((x, y, z))

    return lines


def load_borders_3d(radius: float = 1.003) -> list:
    """Load Natural Earth country borders and convert to 3D coordinates.

    Uses pre-computed cache if available for instant loading.
    """
    # Try to use pre-computed cache
    cache = get_geometry_cache()
    if cache is not None and 'borders' in cache:
        return cache['borders']

    # Fall back to computing from GeoJSON
    if not BORDERS_FILE.exists():
        download_natural_earth_data()

    if not BORDERS_FILE.exists():
        return []

    with open(BORDERS_FILE, 'r') as f:
        data = json.load(f)

    lines = []
    for feature in data['features']:
        geom = feature['geometry']
        if geom['type'] == 'LineString':
            coords = geom['coordinates']
            lons = np.array([c[0] for c in coords])
            lats = np.array([c[1] for c in coords])

            lat_rad = np.deg2rad(lats)
            lon_rad = np.deg2rad(lons)

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)

            lines.append((x, y, z))
        elif geom['type'] == 'MultiLineString':
            for line_coords in geom['coordinates']:
                lons = np.array([c[0] for c in line_coords])
                lats = np.array([c[1] for c in line_coords])

                lat_rad = np.deg2rad(lats)
                lon_rad = np.deg2rad(lons)

                x = radius * np.cos(lat_rad) * np.cos(lon_rad)
                y = radius * np.cos(lat_rad) * np.sin(lon_rad)
                z = radius * np.sin(lat_rad)

                lines.append((x, y, z))

    return lines


# ECMWF 850 hPa temperature color palette
ECMWF_T2M_LEVELS = [-50, -44, -38, -32, -26, -20, -14, -8, -2, 4, 10, 16, 22, 28, 34, 40, 46]

ECMWF_T2M_COLORS = [
    '#800080', '#6600aa', '#4000d4', '#0000ff', '#0066ff', '#00ccff',
    '#00ffcc', '#00ff66', '#66ff00', '#ccff00', '#ffff00', '#ffcc00',
    '#ff8800', '#ff4400', '#ff0000', '#cc0000',
]


def create_ecmwf_colorscale(discrete: bool = True):
    """Create the ECMWF 850 hPa temperature colorscale."""
    vmin, vmax = -28, 40
    colorscale = []

    if discrete:
        for i, (level, color) in enumerate(zip(ECMWF_T2M_LEVELS[:-1], ECMWF_T2M_COLORS)):
            next_level = ECMWF_T2M_LEVELS[i + 1]

            pos = max(0.0, min(1.0, (level - vmin) / (vmax - vmin)))
            next_pos = max(0.0, min(1.0, (next_level - vmin) / (vmax - vmin)))

            if pos == next_pos:
                continue

            colorscale.append([pos, color])
            if next_pos > pos:
                colorscale.append([next_pos - 0.0001, color])

        colorscale.append([1.0, ECMWF_T2M_COLORS[-1]])
    else:
        for level, color in zip(ECMWF_T2M_LEVELS, ECMWF_T2M_COLORS):
            pos = (level - vmin) / (vmax - vmin)
            pos = max(0.0, min(1.0, pos))
            colorscale.append([pos, color])

        seen_positions = set()
        filtered = []
        for pos, color in colorscale:
            if pos not in seen_positions:
                filtered.append([pos, color])
                seen_positions.add(pos)
        colorscale = filtered

        if colorscale[0][0] > 0:
            colorscale.insert(0, [0.0, colorscale[0][1]])
        if colorscale[-1][0] < 1.0:
            colorscale.append([1.0, colorscale[-1][1]])

    return colorscale, vmin, vmax


# ECMWF Magics anomaly color palette
ANOMALY_LEVELS = [-10, -5, 0, 5, 10]
ANOMALY_COLORS = [
    '#05406a', '#5ba3c0', '#ffffff', '#d4604a', '#5f100e',
]


def create_anomaly_colorscale(discrete: bool = True):
    """Create the ECMWF Magics-style anomaly colorscale."""
    vmin, vmax = -10, 10
    colorscale = []

    if discrete:
        for i, (level, color) in enumerate(zip(ANOMALY_LEVELS[:-1], ANOMALY_COLORS)):
            next_level = ANOMALY_LEVELS[i + 1]

            pos = (level - vmin) / (vmax - vmin)
            next_pos = (next_level - vmin) / (vmax - vmin)

            colorscale.append([pos, color])
            colorscale.append([next_pos - 0.0001, color])

        colorscale.append([1.0, ANOMALY_COLORS[-1]])
    else:
        for level, color in zip(ANOMALY_LEVELS, ANOMALY_COLORS):
            pos = (level - vmin) / (vmax - vmin)
            colorscale.append([pos, color])

    return colorscale, vmin, vmax


# Cache for monthly data file handle
_monthly_dataset = None

# Cache for climatology data (avoid reopening file on every anomaly render)
_climatology_cache: dict = {}


def _merge_lines_to_trace(line_list: list) -> tuple:
    """
    Merge a list of (x, y, z) line segments into single NaN-separated arrays.

    Instead of creating one Scatter3d trace per segment (potentially 1000+),
    this returns three flat arrays that can be used in a single trace.
    NaN values create visual breaks between segments.
    """
    if not line_list:
        return np.array([]), np.array([]), np.array([])
    nan = np.array([np.nan])
    all_x, all_y, all_z = [], [], []
    for x, y, z in line_list:
        all_x.extend([x, nan])
        all_y.extend([y, nan])
        all_z.extend([z, nan])
    return (np.concatenate(all_x),
            np.concatenate(all_y),
            np.concatenate(all_z))


def get_monthly_dataset():
    """Get or open the monthly data file."""
    global _monthly_dataset
    if _monthly_dataset is None and MONTHLY_DATA_FILE.exists():
        _monthly_dataset = xr.open_dataset(MONTHLY_DATA_FILE)
    return _monthly_dataset


def get_available_dates() -> list:
    """Get list of available year-month combinations from the monthly file."""
    ds = get_monthly_dataset()
    if ds is None:
        return []

    times = ds['time'].values
    dates = []
    for t in times:
        dt = np.datetime64(t, 'M')
        year = int(str(dt)[:4])
        month = int(str(dt)[5:7])
        dates.append((year, month))
    return dates


def load_monthly_data(year: int, month: int) -> tuple:
    """
    Load a single month from the consolidated monthly data file.

    Returns:
        Tuple of (temperature_celsius, latitudes, longitudes)
    """
    ds = get_monthly_dataset()
    if ds is None:
        raise FileNotFoundError(f"Monthly data file not found: {MONTHLY_DATA_FILE}")

    # Select the specific month
    time_str = f'{year}-{month:02d}'
    try:
        month_data = ds['t2m'].sel(time=time_str, method='nearest')
    except Exception:
        # Try with full datetime
        target_time = np.datetime64(f'{year}-{month:02d}-01')
        month_data = ds['t2m'].sel(time=target_time, method='nearest')

    temp_k = month_data.values
    temp_c = temp_k - 273.15
    lats = ds.latitude.values
    lons = ds.longitude.values

    return temp_c, lats, lons


def load_climatology(month: int) -> np.ndarray:
    """Load ERA5 1991-2020 monthly climatology for the given month (cached in memory)."""
    if month in _climatology_cache:
        return _climatology_cache[month]

    if not CLIM_FILE.exists():
        return None

    ds = xr.open_dataset(CLIM_FILE)
    clim_k = ds['t2m'].sel(month=month).values
    clim_c = clim_k - 273.15
    ds.close()

    _climatology_cache[month] = clim_c
    return clim_c


def calculate_regional_stats(
    temp_data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    clim_data: np.ndarray = None,
) -> dict:
    """
    Calculate temperature statistics for continents and countries.

    Uses regionmask for continents and bounding boxes for countries.
    Applies latitude weighting for accurate area-weighted means.
    """
    try:
        import regionmask
    except ImportError:
        regionmask = None

    results = {'continents': {}, 'countries': {}}

    # Create latitude weights
    weights = np.cos(np.deg2rad(lats))
    weights_2d = np.broadcast_to(weights[:, np.newaxis], temp_data.shape)

    # Normalize longitude to 0-360 for regionmask compatibility
    lons_360 = np.where(lons < 0, lons + 360, lons)

    # Continental statistics using regionmask
    if regionmask is not None:
        try:
            # Get continent regions
            continents = regionmask.defined_regions.natural_earth_v5_0_0.continents_110

            # Create mask
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            mask = continents.mask(lon_grid, lat_grid)

            continent_names = [
                'Africa', 'Antarctica', 'Asia', 'Europe',
                'North America', 'Oceania', 'South America'
            ]

            for i, name in enumerate(continents.names):
                region_mask = mask == i
                if not np.any(region_mask):
                    continue

                masked_temp = np.where(region_mask, temp_data, np.nan)
                masked_weights = np.where(region_mask, weights_2d, 0)

                mean_temp = np.nansum(masked_temp * masked_weights) / np.nansum(masked_weights)
                min_temp = np.nanmin(masked_temp)
                max_temp = np.nanmax(masked_temp)

                anomaly = None
                if clim_data is not None:
                    masked_clim = np.where(region_mask, clim_data, np.nan)
                    clim_mean = np.nansum(masked_clim * masked_weights) / np.nansum(masked_weights)
                    anomaly = mean_temp - clim_mean

                results['continents'][name] = {
                    'mean': mean_temp,
                    'min': min_temp,
                    'max': max_temp,
                    'anomaly': anomaly,
                }
        except Exception as e:
            print(f"Regionmask error: {e}")

    # Country statistics using bounding boxes
    for country, bounds in COUNTRIES.items():
        lon_min, lon_max = bounds['lon']
        lat_min, lat_max = bounds['lat']

        # Create mask for bounding box
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        region_mask = np.outer(lat_mask, lon_mask)

        if not np.any(region_mask):
            continue

        masked_temp = np.where(region_mask, temp_data, np.nan)
        masked_weights = np.where(region_mask, weights_2d, 0)

        mean_temp = np.nansum(masked_temp * masked_weights) / np.nansum(masked_weights)
        min_temp = np.nanmin(masked_temp)
        max_temp = np.nanmax(masked_temp)

        anomaly = None
        if clim_data is not None:
            masked_clim = np.where(region_mask, clim_data, np.nan)
            clim_mean = np.nansum(masked_clim * masked_weights) / np.nansum(masked_weights)
            anomaly = mean_temp - clim_mean

        results['countries'][country] = {
            'mean': mean_temp,
            'min': min_temp,
            'max': max_temp,
            'anomaly': anomaly,
        }

    return results


def create_3d_globe(
    temp_data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    year: int,
    month: int,
    show_anomaly: bool = False,
    temp_unit: str = 'C',
) -> go.Figure:
    """Create an interactive 3D globe visualization of temperature data."""
    theme = get_theme()

    def c_to_f(c):
        return c * 9/5 + 32

    unit_symbol = '°F' if temp_unit == 'F' else '°C'

    # ── 2× downsample for memory efficiency ──────────────────────────────────
    # ERA5 is 721×1440 (0.25°); every-other-point gives 361×720 (0.5°)
    # which is imperceptible at typical screen sizes and cuts surface data 4×.
    step = 2
    lats_d = lats[::step]
    lons_d = lons[::step]
    temp_data_d = temp_data[::step, ::step]

    # Wrap longitude to close the seam
    lons_wrapped = np.append(lons_d, 360.0)
    temp_data_wrapped = np.column_stack([temp_data_d, temp_data_d[:, 0]])

    lon_grid, lat_grid = np.meshgrid(lons_wrapped, lats_d)

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)

    r = 1.0

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    month_name = MONTH_NAMES[month - 1]

    if show_anomaly:
        clim = load_climatology(month)
        if clim is not None:
            clim_d = clim[::step, ::step]
            clim_wrapped = np.column_stack([clim_d, clim_d[:, 0]])
            color_data = temp_data_wrapped - clim_wrapped
            title_text = f'ERA5 Temperature Anomaly (vs 1991-2020) | {month_name} {year}'
        else:
            weights = np.cos(np.deg2rad(lats_d))
            global_mean = np.average(np.nanmean(temp_data_d, axis=1), weights=weights)
            color_data = temp_data_wrapped - global_mean
            title_text = f'ERA5 Temperature Anomaly | {month_name} {year}'

        colorscale, zmin, zmax = create_anomaly_colorscale(discrete=False)
        colorbar_title = f'Anomaly ({unit_symbol})'
        tickvals_c = [-10, -5, 0, 5, 10]
        if temp_unit == 'F':
            ticktext = ['<-18', '-9', '0', '+9', '>+18']
        else:
            ticktext = ['<-10', '-5', '0', '+5', '>+10']
        tickvals = tickvals_c
    else:
        color_data = temp_data_wrapped
        colorscale, zmin, zmax = create_ecmwf_colorscale(discrete=False)
        colorbar_title = f'Temperature ({unit_symbol})'
        title_text = f'ERA5 Monthly Mean 2m Temperature | {month_name} {year}'
        tickvals_c = [-28, -20, -10, 0, 10, 20, 30, 40]
        if temp_unit == 'F':
            ticktext = [f'<{c_to_f(-28):.0f}', f'{c_to_f(-20):.0f}', f'{c_to_f(-10):.0f}',
                       f'{c_to_f(0):.0f}', f'{c_to_f(10):.0f}', f'{c_to_f(20):.0f}',
                       f'{c_to_f(30):.0f}', f'>{c_to_f(40):.0f}']
        else:
            ticktext = ['<-28', '-20', '-10', '0', '10', '20', '30', '>40']
        tickvals = tickvals_c

    # Build pre-formatted text array for hover tooltip (go.Surface doesn't
    # support customdata/surfacecolor substitution in hovertemplate).
    # Convert to Python list of lists to avoid numpy object-array serialization
    # quirks that can transpose indices when Plotly converts to JSON.
    if temp_unit == 'F':
        hover_data = color_data * 1.8 if show_anomaly else color_data * 9 / 5 + 32
    else:
        hover_data = color_data
    prefix = '+' if show_anomaly else ''
    lon_display = np.where(lon_grid > 180, lon_grid - 360, lon_grid)

    def _fmt(v, lat, lon):
        lat_str = f'{abs(lat):.1f}°{"N" if lat >= 0 else "S"}'
        lon_str = f'{abs(lon):.1f}°{"E" if lon >= 0 else "W"}'
        return f'{v:{prefix}.1f}{unit_symbol}<br>{lat_str}, {lon_str}'

    hover_text = np.vectorize(_fmt, otypes=[str])(hover_data, lat_grid, lon_display)

    globe = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=color_data,
        text=hover_text,
        colorscale=colorscale,
        cmin=zmin,
        cmax=zmax,
        colorbar=dict(
            title=dict(
                text=colorbar_title,
                font=dict(size=14, color=theme['text_color']),
            ),
            tickfont=dict(size=12, color=theme['text_color']),
            thickness=20,
            len=0.6,
            x=1.02,
            bgcolor=theme['paper_color'],
            bordercolor=theme['border_color'],
            borderwidth=1,
            outlinecolor=theme['border_color'],
            outlinewidth=1,
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        hoverinfo='text',
        lighting=dict(
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
            roughness=1.0,
            fresnel=0.0,
        ),
    )

    fig = go.Figure(data=[globe])

    # ── Coastlines, borders, graticules — each as ONE NaN-separated trace ─────
    # Replaces 1400+ individual Scatter3d traces, cutting figure size ~10×.
    coastline_color = 'rgba(0, 0, 0, 0.9)'
    border_color    = 'rgba(0, 0, 0, 0.7)'
    graticule_color = 'rgba(80, 80, 80, 0.4)'

    coastlines = load_coastlines_3d(radius=1.003)
    if coastlines:
        cx, cy, cz = _merge_lines_to_trace(coastlines)
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode='lines',
            line=dict(color=coastline_color, width=1.5),
            hoverinfo='skip', showlegend=False,
        ))

    borders = load_borders_3d(radius=1.003)
    if borders:
        bx, by, bz = _merge_lines_to_trace(borders)
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines',
            line=dict(color=border_color, width=1),
            hoverinfo='skip', showlegend=False,
        ))

    cache = get_geometry_cache()
    if cache is not None and 'graticule_lat' in cache:
        grat_lines = list(cache['graticule_lat']) + list(cache['graticule_lon'])
    else:
        # Fallback: compute graticule lines on-the-fly
        grat_lines = []
        r_g = 1.002
        for lat_deg in range(-60, 91, 30):
            lat_r = np.deg2rad(lat_deg)
            lons_r = np.deg2rad(np.linspace(0, 360, 181))
            grat_lines.append((
                r_g * np.cos(lat_r) * np.cos(lons_r),
                r_g * np.cos(lat_r) * np.sin(lons_r),
                r_g * np.sin(lat_r) * np.ones_like(lons_r),
            ))
        for lon_deg in range(0, 360, 30):
            lon_r = np.deg2rad(lon_deg)
            lats_r = np.deg2rad(np.linspace(-90, 90, 181))
            grat_lines.append((
                r_g * np.cos(lats_r) * np.cos(lon_r),
                r_g * np.cos(lats_r) * np.sin(lon_r),
                r_g * np.sin(lats_r),
            ))

    if grat_lines:
        gx, gy, gz = _merge_lines_to_trace(grat_lines)
        fig.add_trace(go.Scatter3d(
            x=gx, y=gy, z=gz,
            mode='lines',
            line=dict(color=graticule_color, width=1),
            hoverinfo='skip', showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=20, color=theme['text_color'], family='Inter, Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.95,
        ),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor=theme['bg_color'],
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=0.5, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
            dragmode='turntable',
        ),
        paper_bgcolor=theme['bg_color'],
        plot_bgcolor=theme['bg_color'],
        height=600,
        margin=dict(l=10, r=80, t=60, b=10),
        uirevision='globe',
        hoverlabel=dict(
            bgcolor=theme['card_bg'],
            bordercolor=theme['border_color'],
            font=dict(color=theme['text_color'], size=14),
        ),
    )

    return fig


def create_mollweide_map(
    temp_data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    year: int,
    month: int,
    show_anomaly: bool = False,
    temp_unit: str = 'C',
) -> go.Figure:
    """Create a static 2D map with Mollweide projection."""
    theme = get_theme()

    def c_to_f(c):
        return c * 9/5 + 32

    unit_symbol = '°F' if temp_unit == 'F' else '°C'
    month_name = MONTH_NAMES[month - 1]

    if show_anomaly:
        clim = load_climatology(month)
        if clim is not None:
            color_data = temp_data - clim
            title_text = f'ERA5 Temperature Anomaly (vs 1991-2020) | {month_name} {year}'
        else:
            weights = np.cos(np.deg2rad(lats))
            global_mean = np.average(np.nanmean(temp_data, axis=1), weights=weights)
            color_data = temp_data - global_mean
            title_text = f'ERA5 Temperature Anomaly | {month_name} {year}'

        colorscale, zmin, zmax = create_anomaly_colorscale(discrete=False)
        colorbar_title = f'Anomaly ({unit_symbol})'
    else:
        color_data = temp_data
        colorscale, zmin, zmax = create_ecmwf_colorscale(discrete=False)
        colorbar_title = f'Temperature ({unit_symbol})'
        title_text = f'ERA5 Monthly Mean 2m Temperature | {month_name} {year}'

    # Convert to -180 to 180 longitude
    lons_180 = np.where(lons > 180, lons - 360, lons)

    # Create meshgrid for heatmap
    lon_grid, lat_grid = np.meshgrid(lons_180, lats)

    # Subsample for performance (every 2nd point)
    step = 2
    lon_sub = lon_grid[::step, ::step].flatten()
    lat_sub = lat_grid[::step, ::step].flatten()
    color_sub = color_data[::step, ::step].flatten()

    # Create heatmap using Scattergeo
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=lon_sub,
        lat=lat_sub,
        mode='markers',
        marker=dict(
            size=4,
            color=color_sub,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(color=theme['text_color']),
                ),
                tickfont=dict(color=theme['text_color']),
                thickness=15,
                len=0.7,
            ),
        ),
        hovertemplate=(
            f'%{{customdata:.1f}}{unit_symbol}<br>'
            'Lat: %{lat:.1f}<br>'
            'Lon: %{lon:.1f}<extra></extra>'
        ),
        customdata=color_sub if temp_unit == 'C' else (
            color_sub * 1.8 if show_anomaly else color_sub * 1.8 + 32
        ),
    ))

    fig.update_geos(
        projection_type='mollweide',
        showcoastlines=True,
        coastlinecolor='rgba(0, 0, 0, 0.8)',
        coastlinewidth=1,
        showland=False,
        showocean=False,
        showlakes=False,
        showcountries=True,
        countrycolor='rgba(0, 0, 0, 0.5)',
        countrywidth=0.5,
        bgcolor=theme['bg_color'],
        lonaxis=dict(range=[-180, 180]),
        lataxis=dict(range=[-90, 90]),
    )

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color=theme['text_color']),
            x=0.5,
            xanchor='center',
        ),
        paper_bgcolor=theme['bg_color'],
        plot_bgcolor=theme['bg_color'],
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            bgcolor=theme['bg_color'],
        ),
    )

    return fig


def get_available_files() -> list[Path]:
    """Get list of available ERA5 NetCDF files (legacy support)."""
    if not DATA_DIR.exists():
        return []
    files = list(DATA_DIR.glob('era5_*2m*_20*.nc')) + list(DATA_DIR.glob('era5_t2m_025deg_*.nc'))
    return sorted(set(files), reverse=True)


def create_app() -> Dash:
    """Create the Dash application."""
    assets_path = Path(__file__).parent / 'assets'
    theme = get_theme()

    app = Dash(
        __name__,
        assets_folder=str(assets_path),
        external_stylesheets=[
            dbc.themes.DARKLY,
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap'
        ],
        title='ERA5 Global Temperature',
    )

    # Check for data availability
    has_monthly_data = MONTHLY_DATA_FILE.exists()
    available_files = get_available_files() if not has_monthly_data else []

    if not has_monthly_data and not available_files:
        app.layout = dbc.Container([
            html.H1("ERA5 Global Temperature", className="my-4", style={'color': theme['text_color']}),
            dbc.Alert([
                html.H4("No data files found"),
                html.P("Please download ERA5 data first using the ERA5 API:"),
                html.Pre(
                    "from src.era5_api import ERA5Client\n"
                    "client = ERA5Client()\n"
                    "client.download_all_monthly_means()",
                    style={'backgroundColor': theme['bg_color'], 'padding': '1rem'}
                ),
            ], color="warning"),
        ], fluid=True, style={'backgroundColor': theme['bg_color'], 'minHeight': '100vh'})
        return app

    # Get available dates
    if has_monthly_data:
        available_dates = get_available_dates()
        available_dates_set = set(available_dates)
        years = sorted(set(y for y, m in available_dates), reverse=True)
        initial_year = years[0] if years else END_YEAR
        # Get the latest available month for the initial year
        months_for_year = [m for y, m in available_dates if y == initial_year]
        initial_month = max(months_for_year) if months_for_year else 1
    else:
        available_dates = [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]
        available_dates_set = set(available_dates)
        years = list(range(END_YEAR, START_YEAR - 1, -1))
        initial_year = END_YEAR
        initial_month = 1

    # Create dict of available months per year for the callback
    available_months_by_year = {}
    for y, m in available_dates:
        if y not in available_months_by_year:
            available_months_by_year[y] = []
        available_months_by_year[y].append(m)

    card_style = {
        'backgroundColor': theme['card_bg'],
        'border': f"1px solid {theme['border_color']}",
        'borderRadius': '12px',
    }

    app.layout = html.Div([
        # Store available months data for callback
        dcc.Store(id='available-months-store', data=available_months_by_year),
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.Span("ERA5 ", style={'fontWeight': '400', 'color': theme['text_color']}),
                        html.Span("Global Temperature", style={'fontWeight': '600', 'color': theme['accent_color']}),
                    ], className="mt-4 mb-2", style={'fontSize': '2rem'}),
                    html.P(
                        "Interactive 3D globe and 2D map showing ERA5 monthly mean temperature (1940-present)",
                        className="mb-3",
                        style={'fontSize': '1rem', 'color': '#8b949e'}
                    ),
                ]),
            ]),

            # Controls row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Year", className="mb-1",
                                          style={'fontWeight': '500', 'fontSize': '0.9rem', 'color': '#8b949e'}),
                                dcc.Dropdown(
                                    id='year-selector',
                                    options=[{'label': str(y), 'value': y} for y in years],
                                    value=initial_year,
                                    clearable=False,
                                    className='dark-dropdown',
                                ),
                            ], md=2, sm=6),
                            dbc.Col([
                                html.Label("Month", className="mb-1",
                                          style={'fontWeight': '500', 'fontSize': '0.9rem', 'color': '#8b949e'}),
                                dcc.Dropdown(
                                    id='month-selector',
                                    options=[
                                        {
                                            'label': name,
                                            'value': i+1,
                                            'disabled': (i+1) not in available_months_by_year.get(initial_year, [])
                                        }
                                        for i, name in enumerate(MONTH_NAMES)
                                    ],
                                    value=initial_month,
                                    clearable=False,
                                    className='dark-dropdown',
                                ),
                            ], md=2, sm=6),
                            dbc.Col([
                                html.Label("View Type", className="mb-1",
                                          style={'fontWeight': '500', 'fontSize': '0.9rem', 'color': '#8b949e'}),
                                dbc.RadioItems(
                                    id='view-type',
                                    options=[
                                        {'label': 'Absolute', 'value': 'absolute'},
                                        {'label': 'Anomaly', 'value': 'anomaly'},
                                    ],
                                    value='absolute',
                                    inline=True,
                                    labelStyle={'color': theme['text_color']},
                                ),
                            ], md=3, sm=6),
                            dbc.Col([
                                html.Label("Units", className="mb-1",
                                          style={'fontWeight': '500', 'fontSize': '0.9rem', 'color': '#8b949e'}),
                                dbc.RadioItems(
                                    id='temp-unit',
                                    options=[
                                        {'label': '°C', 'value': 'C'},
                                        {'label': '°F', 'value': 'F'},
                                    ],
                                    value='C',
                                    inline=True,
                                    labelStyle={'color': theme['text_color']},
                                ),
                            ], md=2, sm=6),
                        ], className="g-3"),
                    ], style={**card_style, 'padding': '1rem', 'marginBottom': '1rem'}),
                ]),
            ]),

            # Main content: Globe and Stats side by side
            dbc.Row([
                # Left side - 3D Globe
                dbc.Col([
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(
                                id='globe-3d',
                                style={'height': '600px'},
                                config={
                                    'scrollZoom': True,
                                    'displayModeBar': True,
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                    'displaylogo': False,
                                }
                            ),
                            type='circle',
                            color=theme['accent_color'],
                        ),
                    ], style={**card_style, 'padding': '0.5rem'}),
                ], lg=8, md=12),

                # Right side - Global Statistics
                dbc.Col([
                    html.Div([
                        html.H5("Global Statistics", className="mb-3",
                               style={'fontWeight': '600', 'color': theme['text_color']}),
                        html.Div(id='stats-container'),
                    ], style={**card_style, 'padding': '1rem', 'marginBottom': '1rem'}),

                    # Instructions card
                    html.Div([
                        html.H6("Controls", className="mb-2",
                               style={'fontWeight': '600', 'color': theme['text_color']}),
                        html.Ul([
                            html.Li("Drag to rotate the globe", style={'color': '#8b949e'}),
                            html.Li("Scroll to zoom in/out", style={'color': '#8b949e'}),
                            html.Li("Right-drag to pan", style={'color': '#8b949e'}),
                        ], className="mb-0 ps-3", style={'fontSize': '0.9rem'}),
                    ], style={**card_style, 'padding': '1rem'}),
                ], lg=4, md=12),
            ], className="mb-4"),

            # 2D Maps - Side by side (Absolute and Anomaly) - Click to enlarge
            dbc.Row([
                # Absolute temperature map
                dbc.Col([
                    html.Div([
                        html.H6("Absolute Temperature", className="text-center mb-2",
                               style={'color': theme['text_color']}),
                        html.Div(
                            html.Img(
                                id='map-2d-absolute',
                                style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'borderRadius': '8px',
                                    'cursor': 'pointer',
                                },
                                title='Click to enlarge',
                            ),
                            id='map-absolute-container',
                            style={'position': 'relative'},
                        ),
                        html.P("Click to enlarge", className="text-center mt-1 mb-0",
                              style={'fontSize': '0.75rem', 'color': '#8b949e'}),
                    ], style={**card_style, 'padding': '0.75rem'}),
                ], lg=6, md=12, className="mb-3"),
                # Anomaly map
                dbc.Col([
                    html.Div([
                        html.H6("Temperature Anomaly (vs 1991-2020)", className="text-center mb-2",
                               style={'color': theme['text_color']}),
                        html.Div(
                            html.Img(
                                id='map-2d-anomaly',
                                style={
                                    'width': '100%',
                                    'height': 'auto',
                                    'borderRadius': '8px',
                                    'cursor': 'pointer',
                                },
                                title='Click to enlarge',
                            ),
                            id='map-anomaly-container',
                            style={'position': 'relative'},
                        ),
                        html.P("Click to enlarge", className="text-center mt-1 mb-0",
                              style={'fontSize': '0.75rem', 'color': '#8b949e'}),
                    ], style={**card_style, 'padding': '0.75rem'}),
                ], lg=6, md=12, className="mb-3"),
            ], className="mb-4"),

            # Modal for enlarged map view
            dbc.Modal([
                dbc.ModalHeader(
                    dbc.ModalTitle(id='map-modal-title'),
                    close_button=True,
                    style={'backgroundColor': theme['card_bg'], 'borderColor': theme['border_color']}
                ),
                dbc.ModalBody(
                    html.Img(
                        id='map-modal-image',
                        style={'width': '100%', 'height': 'auto', 'borderRadius': '8px'}
                    ),
                    style={'backgroundColor': theme['bg_color'], 'padding': '1rem'}
                ),
            ], id='map-modal', size='xl', centered=True, style={'maxWidth': '95vw'}),

            # Regional Statistics
            dbc.Row([
                # Continent Statistics
                dbc.Col([
                    html.Div([
                        html.H5("Continental Statistics", className="mb-3",
                               style={'fontWeight': '600', 'color': theme['text_color']}),
                        html.Div(id='continent-stats-container'),
                    ], style={**card_style, 'padding': '1rem'}),
                ], lg=6, md=12, className="mb-3"),

                # Country Statistics
                dbc.Col([
                    html.Div([
                        html.H5("Country Statistics", className="mb-3",
                               style={'fontWeight': '600', 'color': theme['text_color']}),
                        html.Div(id='country-stats-container'),
                    ], style={**card_style, 'padding': '1rem'}),
                ], lg=6, md=12, className="mb-3"),
            ]),

            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(className="mt-2", style={'borderColor': theme['border_color']}),
                    html.P([
                        "Data: ",
                        html.A("ERA5 Reanalysis", href="https://cds.climate.copernicus.eu",
                              target="_blank", style={'color': theme['accent_color']}),
                        " | Copernicus Climate Change Service (C3S) | Baseline: 1991-2020",
                    ], className="text-center mb-3", style={'fontSize': '0.85rem', 'color': '#8b949e'}),
                ]),
            ]),
        ], fluid=True, style={'maxWidth': '1800px'}),
    ], style={
        'backgroundColor': theme['bg_color'],
        'minHeight': '100vh',
        'color': theme['text_color'],
    })

    @callback(
        [Output('globe-3d', 'figure'),
         Output('map-2d-absolute', 'src'),
         Output('map-2d-anomaly', 'src'),
         Output('stats-container', 'children'),
         Output('continent-stats-container', 'children'),
         Output('country-stats-container', 'children')],
        [Input('year-selector', 'value'),
         Input('month-selector', 'value'),
         Input('view-type', 'value'),
         Input('temp-unit', 'value')],
    )
    def update_visualization(year, month, view_type, temp_unit):
        theme = get_theme()

        def c_to_f(c):
            return c * 9/5 + 32

        unit_symbol = '°F' if temp_unit == 'F' else '°C'
        show_anomaly = view_type == 'anomaly'

        # Try to use pre-computed stats first (instant lookup)
        precomputed_stats = lookup_precomputed_stats(year, month)

        # Check for pre-rendered static map images (both absolute and anomaly)
        def load_static_map(is_anomaly):
            map_path = get_precomputed_map_path(year, month, is_anomaly)
            if map_path.exists():
                with open(map_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    return f'data:image/png;base64,{img_data}'
            return None

        abs_map_src = load_static_map(False)
        anom_map_src = load_static_map(True)

        # Load gridded data (needed for 3D globe)
        try:
            temp_data, lats, lons = load_monthly_data(year, month)
        except FileNotFoundError:
            # Fallback to legacy file loading
            available_files = get_available_files()
            if available_files:
                ds = xr.open_dataset(available_files[0])
                temp_k = ds['t2m'].values[0]
                temp_data = temp_k - 273.15
                lats = ds.latitude.values
                lons = ds.longitude.values
                ds.close()
            else:
                # Return empty figures
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    paper_bgcolor=theme['bg_color'],
                    plot_bgcolor=theme['bg_color'],
                )
                return (empty_fig, None, None,
                        html.P("No data available"), html.P("No data"), html.P("No data"))

        # Create 3D globe visualization
        globe_fig = create_3d_globe(temp_data, lats, lons, year, month, show_anomaly, temp_unit)

        # Get statistics (from pre-computed or calculate)
        if precomputed_stats is not None:
            # Use pre-computed stats (instant)
            global_mean = precomputed_stats['global']['mean']
            min_temp = precomputed_stats['global']['min']
            max_temp = precomputed_stats['global']['max']
            anomaly_mean = precomputed_stats['global']['anomaly']

            if temp_unit == 'F':
                global_mean_display = c_to_f(global_mean)
                min_temp_display = c_to_f(min_temp)
                max_temp_display = c_to_f(max_temp)
                anomaly_display = anomaly_mean * 1.8 if not np.isnan(anomaly_mean) else 0
            else:
                global_mean_display = global_mean
                min_temp_display = min_temp
                max_temp_display = max_temp
                anomaly_display = anomaly_mean if not np.isnan(anomaly_mean) else 0

            if show_anomaly and not np.isnan(anomaly_mean):
                anomaly_str = f"{anomaly_display:+.2f}{unit_symbol}"
            else:
                anomaly_str = None

            regional_stats = precomputed_stats
        else:
            # Fall back to calculating stats (slower)
            weights = np.cos(np.deg2rad(lats))
            global_mean = np.average(np.nanmean(temp_data, axis=1), weights=weights)
            min_temp = np.nanmin(temp_data)
            max_temp = np.nanmax(temp_data)

            # Load climatology for anomaly calculation
            clim = load_climatology(month) if show_anomaly else None

            if temp_unit == 'F':
                global_mean_display = c_to_f(global_mean)
                min_temp_display = c_to_f(min_temp)
                max_temp_display = c_to_f(max_temp)
            else:
                global_mean_display = global_mean
                min_temp_display = min_temp
                max_temp_display = max_temp

            # Calculate anomaly stats
            if show_anomaly and clim is not None:
                clim_mean = np.average(np.nanmean(clim, axis=1), weights=weights)
                anomaly_mean = global_mean - clim_mean
                anomaly_display = anomaly_mean * 1.8 if temp_unit == 'F' else anomaly_mean
                anomaly_str = f"{anomaly_display:+.2f}{unit_symbol}"
            else:
                anomaly_str = None
                anomaly_mean = 0

            # Calculate regional stats
            regional_stats = calculate_regional_stats(temp_data, lats, lons, clim)

        stat_box_style = {
            'backgroundColor': theme['bg_color'],
            'borderRadius': '8px',
            'padding': '0.75rem',
            'textAlign': 'center',
            'border': f"1px solid {theme['border_color']}",
        }

        if show_anomaly and anomaly_str:
            main_stat = html.Div([
                html.Div(anomaly_str,
                        style={'fontSize': '1.8rem', 'fontWeight': '600',
                               'color': '#ff6b6b' if anomaly_mean > 0 else '#58a6ff'}),
                html.Div("Global Mean Anomaly", style={'fontSize': '0.85rem', 'color': '#8b949e'}),
            ], style={**stat_box_style, 'marginBottom': '0.75rem'})
        else:
            main_stat = html.Div([
                html.Div(f"{global_mean_display:.1f}{unit_symbol}",
                        style={'fontSize': '1.8rem', 'fontWeight': '600', 'color': theme['accent_color']}),
                html.Div("Global Mean", style={'fontSize': '0.85rem', 'color': '#8b949e'}),
            ], style={**stat_box_style, 'marginBottom': '0.75rem'})

        global_stats = html.Div([
            main_stat,
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(f"{min_temp_display:.0f}{unit_symbol}",
                                style={'fontSize': '1.3rem', 'fontWeight': '600', 'color': '#58a6ff'}),
                        html.Div("Coldest", style={'fontSize': '0.8rem', 'color': '#8b949e'}),
                    ], style=stat_box_style),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div(f"{max_temp_display:.0f}{unit_symbol}",
                                style={'fontSize': '1.3rem', 'fontWeight': '600', 'color': '#ff6b6b'}),
                        html.Div("Warmest", style={'fontSize': '0.8rem', 'color': '#8b949e'}),
                    ], style=stat_box_style),
                ], width=6),
            ], className="g-2"),
        ])

        # Create continent stats table
        continent_data = []
        for name, stats in regional_stats['continents'].items():
            mean_val = stats['mean']
            min_val = stats['min']
            max_val = stats['max']
            anom_val = stats['anomaly']

            # Skip if all values are NaN
            if np.isnan(mean_val):
                continue

            if temp_unit == 'F':
                mean_val = c_to_f(mean_val)
                min_val = c_to_f(min_val)
                max_val = c_to_f(max_val)
                if anom_val is not None and not np.isnan(anom_val):
                    anom_val = anom_val * 1.8

            if anom_val is not None and not np.isnan(anom_val):
                anom_str = f"{anom_val:+.1f}"
            else:
                anom_str = "N/A"

            continent_data.append({
                'Region': name,
                'Mean': f"{mean_val:.1f}",
                'Anomaly': anom_str,
                'Min/Max': f"{min_val:.0f}/{max_val:.0f}",
            })

        continent_table = dash_table.DataTable(
            data=continent_data,
            columns=[
                {'name': 'Region', 'id': 'Region'},
                {'name': f'Mean ({unit_symbol})', 'id': 'Mean'},
                {'name': f'Anomaly ({unit_symbol})', 'id': 'Anomaly'},
                {'name': f'Min/Max ({unit_symbol})', 'id': 'Min/Max'},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': theme['bg_color'],
                'color': theme['text_color'],
                'border': f"1px solid {theme['border_color']}",
                'textAlign': 'center',
                'padding': '8px',
            },
            style_header={
                'backgroundColor': theme['card_bg'],
                'fontWeight': '600',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Anomaly', 'filter_query': '{Anomaly} contains "+"'},
                    'color': '#ff6b6b',
                },
                {
                    'if': {'column_id': 'Anomaly', 'filter_query': '{Anomaly} contains "-"'},
                    'color': '#58a6ff',
                },
            ],
        ) if continent_data else html.P("No continent data available", style={'color': '#8b949e'})

        # Create country stats table
        country_data = []
        for name, stats in regional_stats['countries'].items():
            mean_val = stats['mean']
            min_val = stats['min']
            max_val = stats['max']
            anom_val = stats['anomaly']

            # Skip if all values are NaN
            if np.isnan(mean_val):
                continue

            if temp_unit == 'F':
                mean_val = c_to_f(mean_val)
                min_val = c_to_f(min_val)
                max_val = c_to_f(max_val)
                if anom_val is not None and not np.isnan(anom_val):
                    anom_val = anom_val * 1.8

            if anom_val is not None and not np.isnan(anom_val):
                anom_str = f"{anom_val:+.1f}"
            else:
                anom_str = "N/A"

            country_data.append({
                'Country': name,
                'Mean': f"{mean_val:.1f}",
                'Anomaly': anom_str,
                'Min/Max': f"{min_val:.0f}/{max_val:.0f}",
            })

        country_table = dash_table.DataTable(
            data=country_data,
            columns=[
                {'name': 'Country', 'id': 'Country'},
                {'name': f'Mean ({unit_symbol})', 'id': 'Mean'},
                {'name': f'Anomaly ({unit_symbol})', 'id': 'Anomaly'},
                {'name': f'Min/Max ({unit_symbol})', 'id': 'Min/Max'},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': theme['bg_color'],
                'color': theme['text_color'],
                'border': f"1px solid {theme['border_color']}",
                'textAlign': 'center',
                'padding': '8px',
            },
            style_header={
                'backgroundColor': theme['card_bg'],
                'fontWeight': '600',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Anomaly', 'filter_query': '{Anomaly} contains "+"'},
                    'color': '#ff6b6b',
                },
                {
                    'if': {'column_id': 'Anomaly', 'filter_query': '{Anomaly} contains "-"'},
                    'color': '#58a6ff',
                },
            ],
        ) if country_data else html.P("No country data available", style={'color': '#8b949e'})

        return (globe_fig, abs_map_src, anom_map_src,
                global_stats, continent_table, country_table)

    # Callback to update month dropdown when year changes
    @callback(
        [Output('month-selector', 'options'),
         Output('month-selector', 'value')],
        [Input('year-selector', 'value')],
        [State('available-months-store', 'data'),
         State('month-selector', 'value')],
    )
    def update_month_options(year, available_months_by_year, current_month):
        if not available_months_by_year or year is None:
            return no_update, no_update

        # Get available months for this year (keys are strings in JSON)
        available_months = available_months_by_year.get(str(year), [])

        # Create options with unavailable months disabled
        options = [
            {
                'label': name,
                'value': i + 1,
                'disabled': (i + 1) not in available_months
            }
            for i, name in enumerate(MONTH_NAMES)
        ]

        # If current month is not available, select the latest available month
        if current_month not in available_months:
            new_month = max(available_months) if available_months else 1
        else:
            new_month = current_month

        return options, new_month

    # Callback to open modal when clicking on map images
    @callback(
        [Output('map-modal', 'is_open'),
         Output('map-modal-image', 'src'),
         Output('map-modal-title', 'children')],
        [Input('map-absolute-container', 'n_clicks'),
         Input('map-anomaly-container', 'n_clicks')],
        [State('map-2d-absolute', 'src'),
         State('map-2d-anomaly', 'src'),
         State('year-selector', 'value'),
         State('month-selector', 'value'),
         State('map-modal', 'is_open')],
        prevent_initial_call=True,
    )
    def toggle_map_modal(abs_clicks, anom_clicks, abs_src, anom_src, year, month, is_open):
        from dash import ctx

        if not ctx.triggered_id:
            return False, no_update, no_update

        month_name = MONTH_NAMES[month - 1] if month else ''

        if ctx.triggered_id == 'map-absolute-container' and abs_src:
            title = f"Absolute Temperature - {month_name} {year}"
            return True, abs_src, title
        elif ctx.triggered_id == 'map-anomaly-container' and anom_src:
            title = f"Temperature Anomaly (vs 1991-2020) - {month_name} {year}"
            return True, anom_src, title

        return False, no_update, no_update

    return app


if __name__ == '__main__':
    print("Starting ERA5 Monthly Temperature Globe (1940-present)...")
    print("Open http://127.0.0.1:8051 in your browser")
    print("Press Ctrl+C to stop")

    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=8051)
