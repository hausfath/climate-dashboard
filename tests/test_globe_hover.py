"""
Tests for the 3D globe hover tooltip in gridded_map.py.

Uses synthetic ERA5-shaped data (721 lats × 1440 lons) with a simple
latitude-based temperature gradient so no actual data files are needed.
"""

import re
import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gridded_map import create_3d_globe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data():
    """
    Create ERA5-shaped temperature data with a realistic latitude gradient.

    Model: T(lat) = 30·cos(lat) − 25
      North/South pole (±90°): −25 °C  → clearly negative
      Equator (0°):             +5 °C  → clearly positive
    """
    lats = np.linspace(90, -90, 721)    # 90°N … 90°S, 0.25° steps
    lons = np.linspace(0, 359.75, 1440) # 0° … 359.75°E, 0.25° steps

    lat_rad = np.deg2rad(lats)
    temp_col = 30 * np.cos(lat_rad) - 25          # shape (721,)
    temp_data = np.tile(temp_col[:, np.newaxis], (1, 1440))  # (721, 1440)

    return temp_data, lats, lons


def get_hover_text(fig):
    """Return the hover text structure from the Surface trace (trace 0)."""
    surface = fig.data[0]
    assert surface.type == 'surface', f"Expected surface, got {surface.type}"
    return surface.text


def parse_temp(cell: str) -> float:
    """Extract the numeric temperature from a hover cell like '-25.0°C<br>90.0°N, 0.0°E'."""
    m = re.match(r'([+-]?\d+\.\d+)°[CF]', cell)
    assert m, f"No temperature value found in: {repr(cell)}"
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hover_shows_temperature_value():
    """Hover text must contain an actual numeric temperature, not a template literal."""
    temp_data, lats, lons = make_synthetic_data()
    with patch('src.gridded_map.download_natural_earth_data'):
        fig = create_3d_globe(temp_data, lats, lons, year=2020, month=1)

    text = get_hover_text(fig)

    assert text is not None, "Surface trace has no hover text"
    # Plotly may return tuples or lists; grab the first cell either way
    first_cell = text[0][0]
    assert '%{' not in first_cell, (
        f"Hover text still contains a template literal: {repr(first_cell)}"
    )
    assert '°C' in first_cell or '°F' in first_cell, (
        f"Hover text is missing a temperature unit: {repr(first_cell)}"
    )
    # Confirm it parses to a finite number
    val = parse_temp(first_cell)
    assert np.isfinite(val), f"Parsed temperature is not finite: {val}"


def test_pole_temperatures_are_negative():
    """Hover values at the north and south poles must be below 0 °C."""
    temp_data, lats, lons = make_synthetic_data()
    with patch('src.gridded_map.download_natural_earth_data'):
        fig = create_3d_globe(temp_data, lats, lons, year=2020, month=1)

    text = get_hover_text(fig)

    # After 2× downsampling lats_d runs 90 → −90 in 361 steps.
    # Row 0 = North Pole (90°N), row 360 = South Pole (−90°S).
    north = parse_temp(text[0][0])
    south = parse_temp(text[360][0])

    assert north < 0, f"North pole hover shows {north} °C — expected negative"
    assert south < 0, f"South pole hover shows {south} °C — expected negative"


def test_equatorial_temperature_is_positive():
    """Hover value near the equator must be above 0 °C."""
    temp_data, lats, lons = make_synthetic_data()
    with patch('src.gridded_map.download_natural_earth_data'):
        fig = create_3d_globe(temp_data, lats, lons, year=2020, month=1)

    text = get_hover_text(fig)

    # Row 180 → lats_d[180] = 90 − 180×0.5 = 0° (equator)
    equator = parse_temp(text[180][0])

    assert equator > 0, f"Equatorial hover shows {equator} °C — expected positive"


def test_hover_lat_lon_match_position():
    """
    The lat/lon embedded in the hover text must match the expected grid position.
    This catches any transposition between the text array and the x/y/z arrays.
    """
    temp_data, lats, lons = make_synthetic_data()
    with patch('src.gridded_map.download_natural_earth_data'):
        fig = create_3d_globe(temp_data, lats, lons, year=2020, month=1)

    text = get_hover_text(fig)

    # North pole: row 0, any column → should say 90.0°N
    assert '90.0°N' in text[0][0], (
        f"North pole hover should show 90.0°N, got: {repr(text[0][0])}"
    )

    # South pole: row 360 → should say 90.0°S
    assert '90.0°S' in text[360][0], (
        f"South pole hover should show 90.0°S, got: {repr(text[360][0])}"
    )

    # Prime meridian: column 0 → lons_wrapped[0] = 0°E
    assert '0.0°E' in text[0][0], (
        f"Prime meridian hover should show 0.0°E, got: {repr(text[0][0])}"
    )
