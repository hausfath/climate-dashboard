"""Tests for IRI ENSO forecast scraper."""

import pandas as pd
import pytest
import responses

from enso_forecast.config import IRI_URL, SEASON_TO_CENTER_MONTH
from enso_forecast.fetchers.iri import fetch_iri

# Mock IRI HTML page with a simple forecast table
MOCK_IRI_HTML = """
<html>
<body>
<h2>Forecast SST Anomalies</h2>
<table>
<tr>
  <th>Model</th>
  <th>FMA</th>
  <th>MAM</th>
  <th>AMJ</th>
  <th>MJJ</th>
</tr>
<tr>
  <td>NCEP CFSv2</td>
  <td>-0.5</td>
  <td>-0.3</td>
  <td>0.1</td>
  <td>0.4</td>
</tr>
<tr>
  <td>UKMO</td>
  <td>-0.4</td>
  <td>-0.2</td>
  <td>0.2</td>
  <td>0.5</td>
</tr>
<tr>
  <td>CSU CLIPR</td>
  <td>-0.6</td>
  <td>-0.4</td>
  <td>-0.1</td>
  <td>0.2</td>
</tr>
</table>
</body>
</html>
"""


@responses.activate
def test_fetch_iri_basic():
    """Test basic IRI table parsing with 3 models and 4 seasons."""
    responses.add(responses.GET, IRI_URL, body=MOCK_IRI_HTML, status=200)

    df = fetch_iri()

    assert len(df) > 0
    assert df["source"].unique() == ["IRI"]
    assert set(df.columns) >= {
        "source", "model", "model_type", "init_date",
        "target_month", "lead_months", "nino34_anom",
        "member_id", "temporal_resolution", "anomaly_base_period",
    }

    # Should have 3 models
    assert df["model"].nunique() == 3

    # 4 seasons per model = 12 total records
    assert len(df) == 12

    # All should be ensemble means
    assert (df["member_id"] == "mean").all()

    # Temporal resolution should be seasonal
    assert (df["temporal_resolution"] == "seasonal_3mo").all()

    # Anomaly base period for IRI is "varies"
    assert (df["anomaly_base_period"] == "varies").all()


@responses.activate
def test_iri_model_type_tagging():
    """Test that dynamical and statistical models are correctly tagged."""
    responses.add(responses.GET, IRI_URL, body=MOCK_IRI_HTML, status=200)

    df = fetch_iri()

    cfsv2 = df[df["model"] == "NCEP CFSv2"]
    assert (cfsv2["model_type"] == "dynamical").all()

    ukmo = df[df["model"] == "UKMO"]
    assert (ukmo["model_type"] == "dynamical").all()

    csu = df[df["model"] == "CSU CLIPR"]
    assert (csu["model_type"] == "statistical").all()


@responses.activate
def test_iri_value_parsing():
    """Test that numeric values are correctly parsed."""
    responses.add(responses.GET, IRI_URL, body=MOCK_IRI_HTML, status=200)

    df = fetch_iri()

    # NCEP CFSv2 first season (FMA) should be -0.5
    cfsv2_fma = df[
        (df["model"] == "NCEP CFSv2")
    ].sort_values("lead_months").iloc[0]
    assert cfsv2_fma["nino34_anom"] == pytest.approx(-0.5)


@responses.activate
def test_iri_season_mapping():
    """Test that seasons are mapped to correct target months."""
    responses.add(responses.GET, IRI_URL, body=MOCK_IRI_HTML, status=200)

    df = fetch_iri()

    # Verify target months increment chronologically
    target_months = sorted(df["target_month"].unique())
    dates = [pd.Timestamp(tm + "-01") for tm in target_months]
    for i in range(1, len(dates)):
        assert dates[i] > dates[i - 1]


@responses.activate
def test_iri_missing_values():
    """Test handling of missing values (dashes) in table."""
    html_with_missing = """
    <html><body>
    <table>
    <tr><th>Model</th><th>FMA</th><th>MAM</th><th>AMJ</th></tr>
    <tr><td>NCEP CFSv2</td><td>-0.5</td><td>-</td><td>0.1</td></tr>
    </table>
    </body></html>
    """
    responses.add(responses.GET, IRI_URL, body=html_with_missing, status=200)

    df = fetch_iri()
    # Should have 2 records (dash skipped)
    assert len(df) == 2


@responses.activate
def test_iri_skips_summary_rows():
    """Test that average/mean rows are excluded."""
    html_with_avg = """
    <html><body>
    <table>
    <tr><th>Model</th><th>FMA</th><th>MAM</th></tr>
    <tr><td>NCEP CFSv2</td><td>-0.5</td><td>-0.3</td></tr>
    <tr><td>Average</td><td>-0.5</td><td>-0.3</td></tr>
    </table>
    </body></html>
    """
    responses.add(responses.GET, IRI_URL, body=html_with_avg, status=200)

    df = fetch_iri()
    assert "Average" not in df["model"].values
