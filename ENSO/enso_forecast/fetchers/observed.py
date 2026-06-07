"""Fetch observed Nino3.4 monthly values and ONI from NOAA CPC."""

import logging
from datetime import date

import pandas as pd
import requests

from enso_forecast.config import (
    OBSERVED_DIR,
    OBSERVED_ONI_URL,
    OBSERVED_RNINO_MONTHLY_ERSST_URL,
    OBSERVED_RNINO_MONTHLY_OISST_URL,
    OBSERVED_RONI_URL,
    OBSERVED_SSTOI_URL,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    SEASON_TO_CENTER_MONTH,
)

logger = logging.getLogger(__name__)


def fetch_monthly_nino34() -> pd.DataFrame:
    """Download and parse NOAA CPC sstoi.indices for monthly Nino3.4.

    Returns DataFrame with columns: year, month, nino34_anom, date.
    """
    logger.info("Fetching monthly Nino3.4 from %s", OBSERVED_SSTOI_URL)
    resp = requests.get(
        OBSERVED_SSTOI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")

    # The file has a header line, then data rows.
    # Columns: YR MON NINO1+2_ANOM NINO3_ANOM NINO34_ANOM NINO4_ANOM
    # (some files have SST and ANOM columns for each region)
    # Parse adaptively based on header.
    header_line = lines[0].strip()
    headers = header_line.split()

    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
        except ValueError:
            continue

        # Find the NINO3.4 anomaly column.
        # The sstoi.indices format:
        #   YR MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM
        # That's 10 columns total (yr, mon, 4 pairs of sst+anom)
        # Indices: 0=YR, 1=MON, 2=N12_SST, 3=N12_ANOM, 4=N3_SST, 5=N3_ANOM,
        #          6=N4_SST, 7=N4_ANOM, 8=N34_SST, 9=N34_ANOM
        # Note: NINO4 comes before NINO3.4 in the file!
        if len(parts) >= 10:
            try:
                nino34_anom = float(parts[9])
            except ValueError:
                continue
        elif len(parts) >= 6:
            # Fallback for condensed format
            try:
                nino34_anom = float(parts[-1])
            except ValueError:
                continue
        else:
            continue

        # Skip missing values (often coded as -99.9 or 99.9)
        if abs(nino34_anom) > 10:
            continue

        records.append({"year": year, "month": month, "nino34_anom": nino34_anom})

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
    else:
        df["date"] = pd.Series(dtype="datetime64[ns]")
    return df


def fetch_oni() -> pd.DataFrame:
    """Download and parse NOAA CPC ONI (Oceanic Nino Index).

    Returns DataFrame with columns: year, month, oni, date.
    The ONI is a 3-month running mean of Nino3.4 SST anomalies (ERSSTv5).
    Season label is mapped to center month.
    """
    logger.info("Fetching ONI from %s", OBSERVED_ONI_URL)
    resp = requests.get(
        OBSERVED_ONI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")

    records = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 4:
            continue

        # ONI format: SEAS YR TOTAL ANOM  (season first, then year)
        # Detect column order: if first field is a season name, use SEAS YR order
        season_candidate = parts[0].strip()
        if season_candidate in SEASON_TO_CENTER_MONTH:
            season = season_candidate
            try:
                year = int(parts[1])
                oni_value = float(parts[-1])
            except (ValueError, IndexError):
                continue
        else:
            # Alternative format: YR SEAS TOTAL ANOM
            try:
                year = int(parts[0])
                season = parts[1].strip()
                oni_value = float(parts[-1])
            except (ValueError, IndexError):
                continue

        if abs(oni_value) > 10:
            continue

        # Map season to center month
        center_month = SEASON_TO_CENTER_MONTH.get(season)
        if center_month is None:
            continue

        # NOAA labels each season with the year of its center month
        # (e.g. DJF 1950 = Dec 1949-Feb 1950, centered on Jan 1950).
        records.append({
            "year": year,
            "month": center_month,
            "season": season,
            "oni": oni_value,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
    else:
        df["date"] = pd.Series(dtype="datetime64[ns]")
    return df


def fetch_roni() -> pd.DataFrame:
    """Download and parse NOAA CPC RONI (Relative Oceanic Nino Index).

    Returns DataFrame with columns: year, month, season, roni, date.
    RONI is the 3-month running mean of Nino3.4 SST anomalies minus the
    tropical-mean (20S-20N) SST anomaly, on the 1991-2020 ERSSTv5 baseline.
    Same SEAS YR ANOM format as ONI.
    """
    logger.info("Fetching RONI from %s", OBSERVED_RONI_URL)
    resp = requests.get(
        OBSERVED_RONI_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")

    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue

        season_candidate = parts[0].strip()
        if season_candidate in SEASON_TO_CENTER_MONTH:
            season = season_candidate
            try:
                year = int(parts[1])
                roni_value = float(parts[-1])
            except (ValueError, IndexError):
                continue
        else:
            try:
                year = int(parts[0])
                season = parts[1].strip()
                roni_value = float(parts[-1])
            except (ValueError, IndexError):
                continue

        if abs(roni_value) > 10:
            continue

        center_month = SEASON_TO_CENTER_MONTH.get(season)
        if center_month is None:
            continue

        # NOAA labels each season with the year of its center month.
        records.append({
            "year": year,
            "month": center_month,
            "season": season,
            "roni": roni_value,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
    else:
        df["date"] = pd.Series(dtype="datetime64[ns]")
    return df


def _parse_ersst_rnino34(text: str) -> pd.DataFrame:
    """Parse NOAA's ``Rnino34.ascii.txt`` (ERSSTv5 monthly rNINO3.4).

    Format: ``YR  MTH  ANOM`` (header + numeric rows from 1949-12 onward).
    """
    records: list[dict] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            rn34 = float(parts[2])
        except ValueError:
            continue
        records.append({"year": year, "month": month, "rnino34": rn34})
    return pd.DataFrame(records)


def _parse_oisst_rel_mthsst(text: str) -> pd.DataFrame:
    """Parse NOAA's ``rel_mthsst9120.txt`` (OISSTv2.1, four Niño regions).

    We only keep rNINO3.4 (last column) for the hybrid series.
    """
    records: list[dict] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            rn34 = float(parts[5])  # rN1+2, rN3, rN4, rN3.4
        except ValueError:
            continue
        records.append({"year": year, "month": month, "rnino34": rn34})
    return pd.DataFrame(records)


def fetch_rnino_monthly() -> pd.DataFrame:
    """Return NOAA CPC's monthly rNINO3.4 series with an ERSST-primary,
    OISST-fallback strategy.

    NOAA publishes the same conceptual index (Niño 3.4 − 20°S-20°N tropical
    mean, 1991-2020 base) through two pipelines:
      - ``Rnino34.ascii.txt``     — ERSSTv5 (canonical, matches L'Heureux et
        al. 2024 and our scaling / ONI / seasonal RONI series). Updates with
        the same monthly cadence as ERSSTv5 (typically by the 5th-8th).
      - ``rel_mthsst9120.txt``    — OISSTv2.1 (real-time, occasionally
        publishes the most-recent month a few days before ERSST does).

    Strategy: take ERSSTv5 for every month it covers; for any month present
    only in OISSTv2.1 (i.e. ERSST hasn't yet caught up), backfill with the
    OISST value. A ``source`` column records the provenance per row.

    Returns DataFrame columns: year, month, rnino34, date, source.
    """
    logger.info("Fetching ERSST monthly rNINO from %s", OBSERVED_RNINO_MONTHLY_ERSST_URL)
    ersst_resp = requests.get(
        OBSERVED_RNINO_MONTHLY_ERSST_URL,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    ersst_resp.raise_for_status()
    ersst = _parse_ersst_rnino34(ersst_resp.text)
    ersst["source"] = "ERSSTv5"

    # OISST is the fallback for any latest month ERSST hasn't yet posted.
    try:
        logger.info("Fetching OISST monthly rNINO from %s", OBSERVED_RNINO_MONTHLY_OISST_URL)
        oisst_resp = requests.get(
            OBSERVED_RNINO_MONTHLY_OISST_URL,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        oisst_resp.raise_for_status()
        oisst = _parse_oisst_rel_mthsst(oisst_resp.text)
        oisst["source"] = "OISSTv2.1"
    except requests.RequestException as e:
        logger.warning("OISST monthly rNINO fetch failed (%s); proceeding with ERSST only", e)
        oisst = pd.DataFrame(columns=["year", "month", "rnino34", "source"])

    # Keep all ERSST rows; append OISST rows for (year, month) pairs ERSST lacks.
    ersst_keys = set(zip(ersst["year"], ersst["month"])) if not ersst.empty else set()
    oisst_extra = oisst[~oisst.apply(
        lambda r: (int(r["year"]), int(r["month"])) in ersst_keys, axis=1
    )] if not oisst.empty else oisst

    df = pd.concat([ersst, oisst_extra], ignore_index=True)
    if df.empty:
        df["date"] = pd.Series(dtype="datetime64[ns]")
        return df[["year", "month", "rnino34", "date", "source"]]

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-"
        + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("date").reset_index(drop=True)

    n_oisst = (df["source"] == "OISSTv2.1").sum()
    if n_oisst:
        latest_oisst_dates = df.loc[df["source"] == "OISSTv2.1", "date"].dt.strftime("%Y-%m").tolist()
        logger.info("Hybrid monthly rNINO: %d rows from ERSSTv5 + %d OISST fallback (%s)",
                    (df["source"] == "ERSSTv5").sum(), n_oisst, ", ".join(latest_oisst_dates))
    else:
        logger.info("Hybrid monthly rNINO: %d rows from ERSSTv5 (no OISST fallback needed)",
                    len(df))
    return df[["year", "month", "rnino34", "date", "source"]]


def save_observed(force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch and save both observed datasets. Returns dict of DataFrames."""
    results = {}

    monthly_path = OBSERVED_DIR / "nino34_monthly.csv"
    oni_path = OBSERVED_DIR / "oni.csv"
    roni_path = OBSERVED_DIR / "roni.csv"
    rnino_monthly_path = OBSERVED_DIR / "rnino_monthly.csv"

    if (not force and monthly_path.exists() and oni_path.exists()
            and roni_path.exists() and rnino_monthly_path.exists()):
        # Check if data is recent (within 35 days)
        existing = pd.read_csv(monthly_path)
        if len(existing) > 0:
            last_date = pd.to_datetime(existing["date"]).max()
            if (pd.Timestamp(date.today()) - last_date).days < 35:
                logger.info("Observed data is recent, skipping fetch (use --force to override)")
                results["monthly"] = existing
                results["oni"] = pd.read_csv(oni_path)
                results["roni"] = pd.read_csv(roni_path)
                results["rnino_monthly"] = pd.read_csv(rnino_monthly_path)
                return results

    monthly_df = fetch_monthly_nino34()
    monthly_df.to_csv(monthly_path, index=False)
    logger.info("Saved %d monthly Nino3.4 records to %s", len(monthly_df), monthly_path)
    results["monthly"] = monthly_df

    oni_df = fetch_oni()
    oni_df.to_csv(oni_path, index=False)
    logger.info("Saved %d ONI records to %s", len(oni_df), oni_path)
    results["oni"] = oni_df

    try:
        roni_df = fetch_roni()
        roni_df.to_csv(roni_path, index=False)
        logger.info("Saved %d RONI records to %s", len(roni_df), roni_path)
        results["roni"] = roni_df
    except Exception as e:
        logger.warning("Failed to fetch RONI: %s", e)
        if roni_path.exists():
            results["roni"] = pd.read_csv(roni_path)

    try:
        rnino_df = fetch_rnino_monthly()
        rnino_df.to_csv(rnino_monthly_path, index=False)
        logger.info("Saved %d monthly rNINO records to %s",
                    len(rnino_df), rnino_monthly_path)
        results["rnino_monthly"] = rnino_df
    except Exception as e:
        logger.warning("Failed to fetch monthly rNINO: %s", e)
        if rnino_monthly_path.exists():
            results["rnino_monthly"] = pd.read_csv(rnino_monthly_path)

    return results


def get_recent_observed(n_months: int = 24) -> pd.DataFrame:
    """Load recent observed monthly Nino3.4 with rONI and tropical-mean columns."""
    monthly_path = OBSERVED_DIR / "nino34_monthly.csv"
    if not monthly_path.exists():
        save_observed()
    df = pd.read_csv(monthly_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(n_months).reset_index(drop=True)
    # Lazy import avoids a circular dependency between fetchers and normalize.
    from enso_forecast.normalize import merge_observed_with_roni
    return merge_observed_with_roni(df)
