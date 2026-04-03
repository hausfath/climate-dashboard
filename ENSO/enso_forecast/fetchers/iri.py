"""Fetch ENSO forecasts from IRI (Columbia University) HTML table."""

import logging
import re
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from enso_forecast.config import (
    FORECASTS_DIR,
    IRI_DYNAMICAL_MODELS,
    IRI_STATISTICAL_MODELS,
    IRI_URL,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    SEASON_TO_CENTER_MONTH,
)

logger = logging.getLogger(__name__)


def _season_to_target_date(season: str, ref_year: int, ref_month: int) -> str | None:
    """Convert a 3-month season label to a YYYY-MM target month string.

    The center month of the season is the target. We need to figure out
    the year based on the reference date (when the forecast was issued).

    For example, if ref_month=3 (March 2026) and season=NDJ,
    that refers to Nov-Dec-Jan centered on Dec, i.e., 2026-12.
    But if season=DJF, that's Dec-Jan-Feb centered on Jan 2027.
    """
    center_month = SEASON_TO_CENTER_MONTH.get(season)
    if center_month is None:
        return None

    # Start from ref_year. Seasons progress forward from the current month.
    # The first season column should be close to ref_month.
    # We assume target months go forward from ref_month.
    year = ref_year

    # If center_month < ref_month, it's in the next year
    # (e.g., if we're in March and season is JFM with center Feb, that's next year)
    # But actually IRI forecasts start from the current season and go forward ~9 months.
    # We need to handle wrap-around carefully.

    # Simple approach: assume seasons are listed in chronological order
    # starting from the current/next season. We'll adjust year later in the
    # main parsing function based on column position.
    return center_month


def fetch_iri() -> pd.DataFrame:
    """Scrape the IRI ENSO forecast table.

    Returns DataFrame in standard forecast schema.
    """
    logger.info("Fetching IRI forecasts from %s", IRI_URL)
    resp = requests.get(IRI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Find the forecast table. The IRI page has multiple tables:
    # - Probability tables (La Nina/Neutral/El Nino percentages)
    # - The forecast SST anomalies table (model rows with season columns)
    # We identify the right one by looking for a table with many rows
    # and season headers in a header row.
    tables = soup.find_all("table")
    forecast_table = None
    season_labels = []
    data_start = 0

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            # Too small to be the forecast table
            continue

        # Search the first few rows for a header row containing seasons
        for ri in range(min(4, len(rows))):
            cells = rows[ri].find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            seasons_in_row = [t for t in cell_texts if t in SEASON_TO_CENTER_MONTH]
            if len(seasons_in_row) >= 2:
                forecast_table = table
                season_labels = seasons_in_row
                data_start = ri + 1
                break
        if forecast_table is not None:
            break

    if forecast_table is None:
        raise ValueError("Could not find the IRI forecast table on the page")

    rows = forecast_table.find_all("tr")

    if not season_labels:
        raise ValueError("Could not parse season headers from IRI table")

    logger.info("Found %d season columns: %s", len(season_labels), season_labels)

    # Detect init date from page content.
    # IRI publishes forecasts monthly; the page text includes "Published: Month DD, YYYY"
    # The init month is one month before the first season's center month.
    # E.g., if first season is FMA (center=March), init is February.
    init_date = None
    page_text = soup.get_text()

    # Try to extract published date
    import re as _re
    pub_match = _re.search(
        r"Published:\s*(\w+)\s+(\d{1,2}),?\s+(\d{4})", page_text
    )
    if pub_match:
        try:
            pub_date = datetime.strptime(
                f"{pub_match.group(1)} {pub_match.group(2)} {pub_match.group(3)}",
                "%B %d %Y"
            )
            init_date = pub_date.replace(day=1).strftime("%Y-%m-%d")
            logger.info("IRI published date: %s → init_date=%s", pub_date.strftime("%Y-%m-%d"), init_date)
        except ValueError:
            pass

    if init_date is None:
        # Fallback: derive from first season. If first season is FMA (center=Mar),
        # init is Feb. The first season center month minus 1 = init month.
        first_center = SEASON_TO_CENTER_MONTH[season_labels[0]]
        today = date.today()
        ref_year = today.year
        init_month = first_center - 1
        init_year = ref_year
        if init_month < 1:
            init_month = 12
            init_year -= 1
        init_date = f"{init_year}-{init_month:02d}-01"
        logger.info("IRI init_date derived from first season %s: %s", season_labels[0], init_date)

    ref_year = int(init_date[:4])
    ref_month = int(init_date[5:7])

    # Build chronological target months from season labels.
    # First season starts at init_month + 1; subsequent seasons progress forward.
    target_months = []
    last_center = None
    year = ref_year
    for season in season_labels:
        center = SEASON_TO_CENTER_MONTH[season]
        if last_center is not None and center <= last_center:
            year += 1
        elif last_center is None and center <= ref_month:
            # First season center is at or before init month → next year
            year += 1
        target_months.append(f"{year}-{center:02d}")
        last_center = center

    # Parse model rows
    records = []
    for row in rows[data_start:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        model_name = cells[0].get_text(strip=True)
        if not model_name or model_name.lower() in ("model", ""):
            continue

        # Skip summary rows
        if any(kw in model_name.lower() for kw in ["average", "mean", "median"]):
            continue

        # Determine model type
        if model_name in IRI_DYNAMICAL_MODELS:
            model_type = "dynamical"
        elif model_name in IRI_STATISTICAL_MODELS:
            model_type = "statistical"
        else:
            # Fuzzy match: check if any known model name is a substring
            model_type = "unknown"
            for known in IRI_DYNAMICAL_MODELS:
                if known.lower() in model_name.lower() or model_name.lower() in known.lower():
                    model_type = "dynamical"
                    break
            if model_type == "unknown":
                for known in IRI_STATISTICAL_MODELS:
                    if known.lower() in model_name.lower() or model_name.lower() in known.lower():
                        model_type = "statistical"
                        break

        # Parse values
        value_cells = cells[1:]
        for i, (cell, target_month) in enumerate(
            zip(value_cells, target_months)
        ):
            text = cell.get_text(strip=True)
            if not text or text == "-":
                continue
            try:
                value = float(text)
            except ValueError:
                continue

            # Compute lead months
            target_dt = datetime.strptime(target_month, "%Y-%m")
            init_dt = datetime.strptime(init_date, "%Y-%m-%d")
            lead = (target_dt.year - init_dt.year) * 12 + (target_dt.month - init_dt.month)

            records.append({
                "source": "IRI",
                "model": model_name,
                "model_type": model_type,
                "init_date": init_date,
                "target_month": target_month,
                "lead_months": max(lead, 0),
                "nino34_anom": value,
                "member_id": "mean",
                "temporal_resolution": "seasonal_3mo",
                "anomaly_base_period": "varies",
            })

    df = pd.DataFrame(records)
    logger.info(
        "Parsed %d forecast values from %d models",
        len(df),
        df["model"].nunique(),
    )
    return df


def save_iri(force: bool = False) -> pd.DataFrame:
    """Fetch and save IRI forecasts."""
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "IRI"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("IRI data for %s already exists, skipping (use --force)", today_str)
        return pd.read_csv(out_path)

    df = fetch_iri()
    if df.empty:
        logger.warning("IRI fetch returned no data, keeping existing file")
        return df
    df.to_csv(out_path, index=False)
    logger.info("Saved IRI forecasts to %s", out_path)
    return df
