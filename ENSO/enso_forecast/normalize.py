"""Data normalization, validation, combining, and multi-model mean computation."""

import logging
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from enso_forecast.config import (
    DEDUP_PREFERENCE,
    FORECASTS_DIR,
    MODEL_CANONICAL_MAP,
    OBSERVED_DIR,
    VALID_NINO34_RANGE,
    WARNING_NINO34_RANGE,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "source",
    "model",
    "model_type",
    "init_date",
    "target_month",
    "lead_months",
    "nino34_anom",
    "member_id",
    "temporal_resolution",
    "anomaly_base_period",
]


def validate_forecast_df(df: pd.DataFrame) -> list[str]:
    """Validate a forecast DataFrame against the schema.

    Returns list of warning/error messages. Empty list means valid.
    """
    issues = []

    if df.empty:
        issues.append("DataFrame is empty")
        return issues

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return issues

    # Check dtypes
    for col in ["nino34_anom", "lead_months"]:
        if col in df.columns:
            try:
                pd.to_numeric(df[col])
            except (ValueError, TypeError):
                issues.append(f"Column '{col}' contains non-numeric values")

    # Range checks on nino34_anom
    anom = pd.to_numeric(df["nino34_anom"], errors="coerce")
    n_nan = anom.isna().sum()
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values in nino34_anom")

    valid = anom.dropna()
    out_of_range = valid[
        (valid < VALID_NINO34_RANGE[0]) | (valid > VALID_NINO34_RANGE[1])
    ]
    if len(out_of_range) > 0:
        issues.append(
            f"{len(out_of_range)} values outside valid range "
            f"[{VALID_NINO34_RANGE[0]}, {VALID_NINO34_RANGE[1]}]: "
            f"min={out_of_range.min():.2f}, max={out_of_range.max():.2f}"
        )

    warning_exceedances = valid[
        (valid < WARNING_NINO34_RANGE[0]) | (valid > WARNING_NINO34_RANGE[1])
    ]
    if len(warning_exceedances) > 0:
        issues.append(
            f"WARNING: {len(warning_exceedances)} values outside typical range "
            f"[{WARNING_NINO34_RANGE[0]}, {WARNING_NINO34_RANGE[1]}]"
        )

    # Check target_month is at or after init_date
    try:
        init_dates = pd.to_datetime(df["init_date"])
        target_dates = pd.to_datetime(df["target_month"] + "-01")
        before = (target_dates < init_dates).sum()
        if before > 0:
            issues.append(f"{before} rows where target_month is before init_date")
    except Exception:
        pass

    # Check lead_months range
    leads = pd.to_numeric(df["lead_months"], errors="coerce").dropna()
    bad_leads = leads[(leads < 0) | (leads > 12)]
    if len(bad_leads) > 0:
        issues.append(
            f"{len(bad_leads)} rows with lead_months outside [0, 12]: "
            f"min={bad_leads.min()}, max={bad_leads.max()}"
        )

    return issues


def compute_baseline_adjustment(
    source_period: tuple[int, int] = (1993, 2016),
    target_period: tuple[int, int] = (1991, 2020),
) -> dict[int, float]:
    """Compute monthly Nino3.4 anomaly offset to convert between baseline periods.

    Uses observed monthly Nino3.4 data (which is on the 1991-2020 baseline).
    The offset for each calendar month is:
        offset = mean(observed_anom over source_period for that month)
    Adding this offset to anomalies on the source baseline converts them to
    the target (1991-2020) baseline.

    Returns dict mapping calendar month (1-12) to offset in °C.
    """
    obs_path = OBSERVED_DIR / "nino34_monthly.csv"
    if not obs_path.exists():
        logger.warning("No observed data for baseline adjustment; returning zero offsets")
        return {m: 0.0 for m in range(1, 13)}

    obs = pd.read_csv(obs_path)
    mask = (obs["year"] >= source_period[0]) & (obs["year"] <= source_period[1])
    subset = obs[mask]

    if len(subset) == 0:
        logger.warning("Insufficient observed data for baseline adjustment")
        return {m: 0.0 for m in range(1, 13)}

    offsets = subset.groupby("month")["nino34_anom"].mean().to_dict()
    # Fill any missing months with 0
    for m in range(1, 13):
        offsets.setdefault(m, 0.0)

    logger.info(
        "Baseline adjustment %d-%d → %d-%d: mean offset = %+.4f°C",
        source_period[0], source_period[1],
        target_period[0], target_period[1],
        np.mean(list(offsets.values())),
    )
    return offsets


def adjust_c3s_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust C3S anomalies from 1993-2016 to 1991-2020 baseline.

    Only modifies rows where source == 'C3S' and anomaly_base_period == '1993-2016'.
    """
    if df.empty:
        return df

    c3s_mask = (df["source"] == "C3S") & (df["anomaly_base_period"] == "1993-2016")
    if not c3s_mask.any():
        return df

    offsets = compute_baseline_adjustment(
        source_period=(1993, 2016),
        target_period=(1991, 2020),
    )

    df = df.copy()
    for idx in df[c3s_mask].index:
        target_month_str = df.loc[idx, "target_month"]
        try:
            cal_month = int(target_month_str.split("-")[1])
        except (ValueError, IndexError):
            continue
        df.loc[idx, "nino34_anom"] += offsets.get(cal_month, 0.0)

    df.loc[c3s_mask, "anomaly_base_period"] = "1991-2020 (adjusted)"
    n_adjusted = c3s_mask.sum()
    logger.info("Adjusted %d C3S records from 1993-2016 to 1991-2020 baseline", n_adjusted)
    return df


def load_source_forecasts(source: str, fetch_date: str | None = None) -> pd.DataFrame:
    """Load forecast CSV for a specific source and date.

    Args:
        source: Source name (IRI, CFS, NMME, C3S).
        fetch_date: Date string YYYY-MM-DD. If None, uses latest available.
    """
    src_dir = FORECASTS_DIR / source
    if not src_dir.exists():
        logger.warning("No data directory for source %s", source)
        return pd.DataFrame()

    csv_files = sorted(src_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files for source %s", source)
        return pd.DataFrame()

    if fetch_date:
        target = src_dir / f"{fetch_date}.csv"
        if target.exists():
            return pd.read_csv(target)
        logger.warning("No data for %s on %s", source, fetch_date)
        return pd.DataFrame()

    # Use latest
    latest = csv_files[-1]
    logger.info("Loading %s from %s", source, latest.name)
    return pd.read_csv(latest)


def load_all_forecasts(
    fetch_date: str | None = None,
    sources: list[str] | None = None,
    init_month: str | None = None,
) -> pd.DataFrame:
    """Load and combine forecasts from specified sources.

    Args:
        fetch_date: Date string YYYY-MM-DD for the fetch to load.
        sources: List of source names to include. Default: all available.
        init_month: If set (e.g. '2026-03'), only include forecasts whose
            init_date falls in that month.
    """
    if sources is None:
        sources = ["IRI", "CFS", "NMME", "C3S", "CanSIPS"]

    dfs = []
    for src in sources:
        df = load_source_forecasts(src, fetch_date)
        if len(df) > 0:
            dfs.append(df)
            logger.info("Loaded %d records from %s", len(df), src)

    if not dfs:
        logger.warning("No forecast data found")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    combined = pd.concat(dfs, ignore_index=True)

    # Filter by init_month if requested
    if init_month is not None:
        mask = combined["init_date"].str[:7] == init_month
        n_before = len(combined)
        combined = combined[mask].reset_index(drop=True)
        logger.info(
            "Filtered to init_month=%s: %d → %d records",
            init_month, n_before, len(combined),
        )

    logger.info("Combined %d total records from %d sources", len(combined), len(dfs))

    # Adjust C3S baseline from 1993-2016 to 1991-2020
    combined = adjust_c3s_baseline(combined)

    return combined


def get_ensemble_means(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ensemble mean values only."""
    return df[df["member_id"] == "mean"].copy().reset_index(drop=True)


def _get_canonical_name(source: str, model: str) -> str:
    """Get canonical physical model name for deduplication."""
    return MODEL_CANONICAL_MAP.get((source, model), f"{source}:{model}")


def deduplicate_models(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate representations of the same physical model.

    Uses DEDUP_PREFERENCE to select the preferred source for each
    canonical model.
    """
    df = df.copy()
    df["canonical"] = df.apply(
        lambda r: _get_canonical_name(r["source"], r["model"]), axis=1
    )

    # For canonical models with a preference, keep only the preferred source
    rows_to_drop = []
    for canonical, (pref_source, pref_model) in DEDUP_PREFERENCE.items():
        mask = df["canonical"] == canonical
        if mask.sum() == 0:
            continue

        # Keep only the preferred source/model combo
        preferred_mask = mask & (df["source"] == pref_source) & (df["model"] == pref_model)
        non_preferred_mask = mask & ~preferred_mask

        if preferred_mask.sum() > 0 and non_preferred_mask.sum() > 0:
            rows_to_drop.extend(df[non_preferred_mask].index.tolist())
            logger.debug(
                "Dedup %s: keeping %s/%s, dropping %d rows",
                canonical, pref_source, pref_model, non_preferred_mask.sum(),
            )

    if rows_to_drop:
        df = df.drop(index=rows_to_drop).reset_index(drop=True)
        logger.info("Deduplication removed %d rows", len(rows_to_drop))

    return df


def compute_source_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-source mean of model means.

    Returns one row per source per target_month.
    """
    means_df = get_ensemble_means(df)
    if means_df.empty:
        return means_df

    source_means = (
        means_df.groupby(["source", "target_month"], as_index=False)
        .agg({
            "nino34_anom": "mean",
            "init_date": "first",
            "lead_months": "first",
            "temporal_resolution": "first",
            "anomaly_base_period": "first",
        })
    )
    source_means["model"] = source_means["source"] + "_mean"
    source_means["model_type"] = "multi-model"
    source_means["member_id"] = "mean"

    return source_means


def compute_multi_model_mean(
    df: pd.DataFrame,
    weight_by: str = "model",
) -> pd.DataFrame:
    """Compute overall multi-model mean after deduplication.

    Args:
        df: Combined forecast DataFrame.
        weight_by: 'model' for equal weight per deduplicated model,
                   'source' for equal weight per source.
    """
    deduped = deduplicate_models(get_ensemble_means(df))
    if deduped.empty:
        return deduped

    if weight_by == "source":
        # First compute source means, then average sources
        source_means = (
            deduped.groupby(["source", "target_month"], as_index=False)
            ["nino34_anom"].mean()
        )
        mm_mean = (
            source_means.groupby("target_month", as_index=False)
            ["nino34_anom"].mean()
        )
    else:
        # Equal weight per model
        mm_mean = (
            deduped.groupby("target_month", as_index=False)
            ["nino34_anom"].mean()
        )

    mm_mean["source"] = "multi-model"
    mm_mean["model"] = "multi-model-mean"
    mm_mean["model_type"] = "multi-model"
    mm_mean["init_date"] = deduped["init_date"].iloc[0] if len(deduped) > 0 else ""
    mm_mean["member_id"] = "mean"
    mm_mean["temporal_resolution"] = "monthly"
    mm_mean["anomaly_base_period"] = "mixed"

    # Add lead_months
    if "lead_months" not in mm_mean.columns:
        mm_mean["lead_months"] = 0

    return mm_mean


def get_model_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get one value per model per target_month (means only), deduplicated."""
    means = get_ensemble_means(df)
    deduped = deduplicate_models(means)
    return deduped


def get_baseline_caveat() -> str:
    """Return text annotation about different anomaly base periods."""
    return (
        "Note: CFS and observed use 1991\u20132020 baseline. "
        "C3S adjusted from 1993\u20132016 to 1991\u20132020. "
        "NMME: model-specific. Residual offsets of ~0.1\u00b0C may remain."
    )
