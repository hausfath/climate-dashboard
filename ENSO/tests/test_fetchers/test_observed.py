"""Tests for observed Nino3.4 fetcher."""

import pandas as pd
import pytest
import responses

from enso_forecast.config import OBSERVED_ONI_URL, OBSERVED_SSTOI_URL
from enso_forecast.fetchers.observed import fetch_monthly_nino34, fetch_oni


# Mock sstoi.indices content matching real CPC format:
# YR MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM
# Note: NINO4 comes BEFORE NINO3.4 in the actual file!
MOCK_SSTOI = """\
 YR   MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM
 2025   1   24.50  -0.80  26.10  -0.50  28.50  -0.20  27.00  -0.40
 2025   2   25.00  -0.60  26.30  -0.30  28.60  -0.10  27.20  -0.20
 2025   3   25.50  -0.40  26.50  -0.10  28.70   0.00  27.40   0.00
 2025   4   25.80  -0.20  26.80   0.10  28.80   0.10  27.60   0.20
 2025   5   24.20   0.10  27.00   0.30  28.90   0.20  27.80   0.40
 2025   6   23.50   0.30  27.20   0.50  29.00   0.30  28.00   0.60
"""

MOCK_ONI = """\
SEAS  YR  TOTAL  ANOM
DJF  2024  26.5  -0.50
JFM  2025  26.6  -0.40
FMA  2025  26.7  -0.20
MAM  2025  26.9   0.00
AMJ  2025  27.1   0.20
MJJ  2025  27.3   0.40
JJA  2025  27.5   0.60
"""


@responses.activate
def test_fetch_monthly_nino34():
    """Test parsing of sstoi.indices format."""
    responses.add(responses.GET, OBSERVED_SSTOI_URL, body=MOCK_SSTOI, status=200)

    df = fetch_monthly_nino34()

    assert len(df) == 6
    assert set(df.columns) >= {"year", "month", "nino34_anom", "date"}
    assert df["year"].iloc[0] == 2025
    assert df["month"].iloc[0] == 1
    assert df["nino34_anom"].iloc[0] == pytest.approx(-0.40)
    assert df["nino34_anom"].iloc[-1] == pytest.approx(0.60)


@responses.activate
def test_fetch_monthly_nino34_skips_missing():
    """Test that missing values (-99.9) are skipped."""
    data_with_missing = MOCK_SSTOI + " 2025   7   -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9  -99.9\n"
    responses.add(responses.GET, OBSERVED_SSTOI_URL, body=data_with_missing, status=200)

    df = fetch_monthly_nino34()
    assert len(df) == 6  # missing row skipped


@responses.activate
def test_fetch_oni():
    """Test parsing of ONI format with season-to-month mapping."""
    responses.add(responses.GET, OBSERVED_ONI_URL, body=MOCK_ONI, status=200)

    df = fetch_oni()

    assert len(df) == 7
    assert set(df.columns) >= {"year", "month", "oni", "season", "date"}

    # DJF 2024 should map to center month Jan, year 2025
    djf_row = df[df["season"] == "DJF"].iloc[0]
    assert djf_row["year"] == 2025
    assert djf_row["month"] == 1
    assert djf_row["oni"] == pytest.approx(-0.50)

    # JFM 2025 maps to center month Feb 2025
    jfm_row = df[df["season"] == "JFM"].iloc[0]
    assert jfm_row["year"] == 2025
    assert jfm_row["month"] == 2
    assert jfm_row["oni"] == pytest.approx(-0.40)


@responses.activate
def test_fetch_oni_cross_year():
    """Test that DJF season correctly adjusts year."""
    mock_data = """\
SEAS  YR  TOTAL  ANOM
DJF  2023  26.0  -0.30
DJF  2024  26.5  -0.50
"""
    responses.add(responses.GET, OBSERVED_ONI_URL, body=mock_data, status=200)

    df = fetch_oni()
    # DJF 2023 → Jan 2024, DJF 2024 → Jan 2025
    assert df["year"].tolist() == [2024, 2025]
    assert df["month"].tolist() == [1, 1]
