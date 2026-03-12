"""Visualization functions for ENSO forecast data."""

import logging
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from enso_forecast.config import FIGURES_DIR
from enso_forecast.normalize import (
    compute_multi_model_mean,
    compute_source_means,
    deduplicate_models,
    get_baseline_caveat,
    get_ensemble_means,
    get_model_comparison_df,
)

logger = logging.getLogger(__name__)

# Color scheme by source
SOURCE_COLORS = {
    "IRI": "#1f77b4",       # blue
    "CFS": "#ff7f0e",       # orange
    "NMME": "#2ca02c",      # green
    "C3S": "#d62728",       # red
    "multi-model": "#000000",  # black
}

# Distinct colors for individual models (cycle through for many models)
MODEL_CMAP = plt.cm.tab20


def _setup_style():
    """Set matplotlib style for all plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (12, 6),
    })


def _add_enso_shading(ax, **kwargs):
    """Add light El Nino/La Nina background shading.

    Shading extends to +-100 so it always fills to the visible axes edge
    regardless of the y-axis limits set later.
    """
    ax.axhspan(0.5, 100, color="red", alpha=0.04, zorder=0)
    ax.axhspan(-100, -0.5, color="blue", alpha=0.04, zorder=0)
    ax.axhline(0.5, color="red", linewidth=0.7, linestyle="--", alpha=0.5, label="El Ni\u00f1o threshold")
    ax.axhline(-0.5, color="blue", linewidth=0.7, linestyle="--", alpha=0.5, label="La Ni\u00f1a threshold")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)


def _add_observed(ax, obs_df: pd.DataFrame, label: str = "Observed Nino3.4"):
    """Plot observed Nino3.4 as a black line."""
    if obs_df is None or obs_df.empty:
        return
    obs_df = obs_df.copy()
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    ax.plot(
        obs_df["date"], obs_df["nino34_anom"],
        color="black", linewidth=2, zorder=5, label=label,
    )


def _add_footnote(ax):
    """No-op. Footnote removed; baseline caveats discussed in text instead."""
    pass


def _target_month_to_date(target_month: str) -> pd.Timestamp:
    """Convert 'YYYY-MM' to a Timestamp (1st of month)."""
    return pd.Timestamp(target_month + "-01")


def _format_date_axis(ax):
    """Format x-axis for date display."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")


def _set_axis_limits(
    ax,
    obs_df: pd.DataFrame | None,
    forecast_df: pd.DataFrame | None,
    obs_months: int = 3,
    fcast_months: int = 8,
    ymin_default: float = -3.0,
    ymax_default: float = 3.0,
):
    """Set x-axis to last `obs_months` of observed + next `fcast_months`, and
    y-axis to [-3, 3] unless data exceeds that range.
    """
    # Determine x-axis range
    now = pd.Timestamp(date.today().replace(day=1))
    x_start = now - pd.DateOffset(months=obs_months)

    # Find the latest forecast target month
    x_end = now + pd.DateOffset(months=fcast_months)
    if forecast_df is not None and len(forecast_df) > 0:
        if "target_month" in forecast_df.columns:
            max_target = pd.Timestamp(forecast_df["target_month"].max() + "-01")
            # Use whichever is further out
            x_end = max(x_end, max_target + pd.DateOffset(months=1))
        elif "date" in forecast_df.columns:
            max_date = forecast_df["date"].max()
            if isinstance(max_date, str):
                max_date = pd.Timestamp(max_date)
            x_end = max(x_end, max_date + pd.DateOffset(months=1))

    ax.set_xlim(x_start, x_end)

    # Determine y-axis range: default to [-3, 3], expand if data exceeds
    all_y = []
    if obs_df is not None and len(obs_df) > 0:
        obs_copy = obs_df.copy()
        obs_copy["date"] = pd.to_datetime(obs_copy["date"])
        in_range = obs_copy[(obs_copy["date"] >= x_start) & (obs_copy["date"] <= x_end)]
        if len(in_range) > 0:
            all_y.extend(in_range["nino34_anom"].dropna().tolist())

    if forecast_df is not None and len(forecast_df) > 0:
        all_y.extend(forecast_df["nino34_anom"].dropna().tolist())

    if all_y:
        data_min, data_max = min(all_y), max(all_y)
        ymin = min(ymin_default, data_min - 0.3)
        ymax = max(ymax_default, data_max + 0.3)
    else:
        ymin, ymax = ymin_default, ymax_default

    ax.set_ylim(ymin, ymax)


def _filter_cfs_forecast_months(cfs_df: pd.DataFrame) -> pd.DataFrame:
    """Filter CFS data to true forecast months only.

    CFS NetCDF files include historical verification months where all ensemble
    members have identical values (zero spread). True forecast months have
    diverging members. We identify forecast months as those where the ensemble
    standard deviation > 0 for any member group.
    """
    if cfs_df.empty:
        return cfs_df

    members = cfs_df[cfs_df["member_id"] != "mean"]
    if members.empty:
        return cfs_df

    # Compute std per target month across members
    spread = members.groupby("target_month")["nino34_anom"].std().reset_index()
    spread.columns = ["target_month", "std"]

    # Forecast months have std > 0 (members diverge)
    forecast_months = spread[spread["std"] > 0.001]["target_month"].tolist()

    if not forecast_months:
        return cfs_df

    return cfs_df[cfs_df["target_month"].isin(forecast_months)].copy()


def _get_forecast_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter combined forecast data to only true forecast months.

    For CFS, removes historical verification months.
    For other sources, keeps all data (they only contain forecasts).
    """
    non_cfs = df[df["source"] != "CFS"]
    cfs = df[df["source"] == "CFS"]

    if cfs.empty:
        return df

    cfs_filtered = _filter_cfs_forecast_months(cfs)
    return pd.concat([non_cfs, cfs_filtered], ignore_index=True)


def plot_cfs_plume(
    cfs_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "cfs_plume.png",
) -> plt.Figure:
    """Plot 1: CFS v2 ensemble plume with individual members and statistics.

    Only plots true forecast months (where ensemble spread > 0).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    # Filter to true forecast months only
    cfs_forecast = _filter_cfs_forecast_months(cfs_df)

    members = cfs_forecast[cfs_forecast["member_id"] != "mean"].copy()
    mean_data = cfs_forecast[cfs_forecast["member_id"] == "mean"].copy()

    if members.empty:
        logger.warning("No CFS ensemble members to plot")
        plt.close(fig)
        return fig

    members["date"] = members["target_month"].apply(_target_month_to_date)
    mean_data["date"] = mean_data["target_month"].apply(_target_month_to_date)

    # Compute statistics per target month
    stats = members.groupby("target_month").agg(
        mean=("nino34_anom", "mean"),
        p25=("nino34_anom", lambda x: np.percentile(x, 25)),
        p75=("nino34_anom", lambda x: np.percentile(x, 75)),
        pmin=("nino34_anom", "min"),
        pmax=("nino34_anom", "max"),
        n=("nino34_anom", "count"),
    ).reset_index()
    stats["date"] = stats["target_month"].apply(_target_month_to_date)
    stats = stats.sort_values("date")

    # Plot individual members (thin, transparent)
    for member_id, mdf in members.groupby("member_id"):
        mdf = mdf.sort_values("date")
        ax.plot(mdf["date"], mdf["nino34_anom"], color="#ff7f0e", alpha=0.15, linewidth=0.4, zorder=1)

    # Shading: full range (lighter, distinct color)
    ax.fill_between(
        stats["date"], stats["pmin"], stats["pmax"],
        color="#fdd0a2", alpha=0.5, zorder=2, label="Full range",
        edgecolor="none",
    )

    # Shading: IQR (darker, more saturated)
    ax.fill_between(
        stats["date"], stats["p25"], stats["p75"],
        color="#e6550d", alpha=0.4, zorder=3, label="25th\u201375th percentile",
        edgecolor="none",
    )

    # Ensemble mean
    if not mean_data.empty:
        mean_data = mean_data.sort_values("date")
        ax.plot(
            mean_data["date"], mean_data["nino34_anom"],
            color="#d62728", linewidth=2.5, zorder=4, label="CFS ensemble mean",
        )

    n_members = members["member_id"].nunique()
    ax.set_title(f"CFS v2 Nino3.4 Ensemble Forecast Plume ({n_members} members)")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, cfs_forecast)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved CFS plume plot to %s", out_path)

    return fig


def plot_c3s_plume(
    c3s_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "c3s_plume.png",
) -> plt.Figure:
    """Plot C3S multi-system ensemble plume (individual members from all centres).

    Similar to the official C3S Nino3.4 plume plot: all individual members as
    thin lines, per-centre ensemble means as thicker lines, and overall
    multi-system mean as a thick dashed line.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    members = c3s_df[c3s_df["member_id"] != "mean"].copy()
    means = c3s_df[c3s_df["member_id"] == "mean"].copy()

    if members.empty and means.empty:
        logger.warning("No C3S data to plot")
        plt.close(fig)
        return fig

    # Model-specific colors
    c3s_colors = {
        "ECMWF": "#d62728",
        "UKMO": "#1f77b4",
        "Meteo-France": "#2ca02c",
        "DWD": "#ff7f0e",
        "CMCC": "#9467bd",
        "JMA": "#8c564b",
        "ECCC": "#e377c2",
        "NCEP": "#7f7f7f",
        "BOM": "#bcbd22",
    }

    # Plot individual members per model (thin, transparent)
    if not members.empty:
        members["date"] = members["target_month"].apply(_target_month_to_date)
        for model in members["model"].unique():
            color = c3s_colors.get(model, "#999999")
            model_members = members[members["model"] == model]
            for member_id, mdf in model_members.groupby("member_id"):
                mdf = mdf.sort_values("date")
                ax.plot(mdf["date"], mdf["nino34_anom"],
                        color=color, alpha=0.12, linewidth=0.4, zorder=1)

    # Plot per-model ensemble means
    if not means.empty:
        means["date"] = means["target_month"].apply(_target_month_to_date)
        for model in sorted(means["model"].unique()):
            color = c3s_colors.get(model, "#999999")
            mdf = means[means["model"] == model].sort_values("date")
            n_mem = members[members["model"] == model]["member_id"].nunique() if not members.empty else 0
            label = f"{model} mean" + (f" ({n_mem} members)" if n_mem > 0 else "")
            ax.plot(mdf["date"], mdf["nino34_anom"],
                    color=color, linewidth=1.8, alpha=0.8, zorder=4, label=label)

        # Multi-system mean (average of model means)
        mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")
        ax.plot(mm["date"], mm["nino34_anom"],
                color="black", linewidth=2.5, linestyle="--", zorder=6,
                label="C3S multi-system mean")

    # Count total members across all models (member IDs overlap between models)
    n_total = sum(
        members[members["model"] == m]["member_id"].nunique()
        for m in members["model"].unique()
    ) if not members.empty else 0
    n_models = c3s_df["model"].nunique()
    ax.set_title(f"C3S Nino3.4 Multi-System Forecast Plume ({n_models} models, {n_total} members)")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, c3s_df)
    ax.legend(loc="upper left", fontsize=8, framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved C3S plume plot to %s", out_path)

    return fig


def plot_nmme_plume(
    nmme_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "nmme_plume.png",
) -> plt.Figure:
    """Plot NMME multi-model ensemble plume (individual members from all models)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    members = nmme_df[nmme_df["member_id"] != "mean"].copy()
    means = nmme_df[(nmme_df["member_id"] == "mean") & (nmme_df["model"] != "NMME")].copy()

    if members.empty and means.empty:
        logger.warning("No NMME data to plot")
        plt.close(fig)
        return fig

    nmme_colors = {
        "NCEP-CFSv2": "#d62728",
        "ECCC-CanESM5": "#1f77b4",
        "ECCC-GEM5.2-NEMO": "#2ca02c",
        "NCAR-CESM1": "#ff7f0e",
        "NCAR-CCSM4": "#9467bd",
        "NASA-GEOS-S2S-2": "#8c564b",
    }

    if not members.empty:
        members["date"] = members["target_month"].apply(_target_month_to_date)
        for model in members["model"].unique():
            color = nmme_colors.get(model, "#999999")
            model_members = members[members["model"] == model]
            for member_id, mdf in model_members.groupby("member_id"):
                mdf = mdf.sort_values("date")
                ax.plot(mdf["date"], mdf["nino34_anom"],
                        color=color, alpha=0.12, linewidth=0.4, zorder=1)

    if not means.empty:
        means["date"] = means["target_month"].apply(_target_month_to_date)
        for model in sorted(means["model"].unique()):
            color = nmme_colors.get(model, "#999999")
            mdf = means[means["model"] == model].sort_values("date")
            n_mem = members[members["model"] == model]["member_id"].nunique() if not members.empty else 0
            label = f"{model}" + (f" ({n_mem})" if n_mem > 0 else "")
            ax.plot(mdf["date"], mdf["nino34_anom"],
                    color=color, linewidth=1.8, alpha=0.8, zorder=4, label=label)

        mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")
        ax.plot(mm["date"], mm["nino34_anom"],
                color="black", linewidth=2.5, linestyle="--", zorder=6,
                label="NMME multi-model mean")

    n_total = sum(
        members[members["model"] == m]["member_id"].nunique()
        for m in members["model"].unique()
    ) if not members.empty else 0
    n_models = nmme_df[nmme_df["model"] != "NMME"]["model"].nunique()
    ax.set_title(f"NMME Nino3.4 Multi-Model Forecast Plume ({n_models} models, {n_total} members)")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, nmme_df)
    ax.legend(loc="upper left", fontsize=8, framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved NMME plume plot to %s", out_path)

    return fig


def plot_model_comparison(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    dynamical_only: bool = False,
    save: bool = True,
    filename: str = "model_comparison.png",
) -> plt.Figure:
    """Plot 2: Multi-model comparison with one line per unique model."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    # Filter to forecast-only months
    forecast_only = _get_forecast_only(forecast_df)

    comparison = get_model_comparison_df(forecast_only)
    if dynamical_only:
        comparison = comparison[comparison["model_type"] != "statistical"]

    if comparison.empty:
        logger.warning("No model data to plot")
        plt.close(fig)
        return fig

    comparison["date"] = comparison["target_month"].apply(_target_month_to_date)

    # Group by source for legend organization
    sources = comparison["source"].unique()
    color_idx = 0

    for source in sorted(sources):
        src_data = comparison[comparison["source"] == source]
        models = src_data["model"].unique()

        for model in sorted(models):
            mdf = src_data[src_data["model"] == model].sort_values("date")
            color = MODEL_CMAP(color_idx / max(1, comparison["model"].nunique()))
            color_idx += 1

            # Use markers for IRI seasonal data
            marker = "o" if mdf["temporal_resolution"].iloc[0] == "seasonal_3mo" else None
            markersize = 4 if marker else None

            ax.plot(
                mdf["date"], mdf["nino34_anom"],
                color=color, linewidth=1.2, alpha=0.7, zorder=3,
                label=f"{source}: {model}",
                marker=marker, markersize=markersize,
            )

    # Multi-model mean
    mm_mean = compute_multi_model_mean(forecast_only)
    if not mm_mean.empty:
        mm_mean["date"] = mm_mean["target_month"].apply(_target_month_to_date)
        mm_mean = mm_mean.sort_values("date")
        ax.plot(
            mm_mean["date"], mm_mean["nino34_anom"],
            color="black", linewidth=2.5, linestyle="--", zorder=6,
            label="Multi-model mean",
        )

    ax.set_title("ENSO Nino3.4 Multi-Model Forecast Comparison")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, comparison)

    # Put legend outside plot if many models
    n_models = comparison["model"].nunique()
    if n_models > 12:
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            fontsize=8, framealpha=1.0, facecolor="white", edgecolor="lightgray", ncol=1,
        )
        fig.subplots_adjust(right=0.72)
    else:
        ax.legend(loc="best", fontsize=8, framealpha=1.0, facecolor="white", edgecolor="lightgray", ncol=2)

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved model comparison plot to %s", out_path)

    return fig


def plot_source_comparison(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "source_comparison.png",
) -> plt.Figure:
    """Plot 3: Source-level comparison (one line per source with spread shading)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    # Filter to forecast-only months
    forecast_only = _get_forecast_only(forecast_df)
    means_df = get_ensemble_means(forecast_only)
    if means_df.empty:
        logger.warning("No data for source comparison")
        plt.close(fig)
        return fig

    for source in sorted(means_df["source"].unique()):
        src_data = means_df[means_df["source"] == source]
        color = SOURCE_COLORS.get(source, "#999999")

        # Compute per-target-month stats across models within this source
        stats = src_data.groupby("target_month").agg(
            mean=("nino34_anom", "mean"),
            smin=("nino34_anom", "min"),
            smax=("nino34_anom", "max"),
        ).reset_index()
        stats["date"] = stats["target_month"].apply(_target_month_to_date)
        stats = stats.sort_values("date")

        # Shading: min-max across models
        ax.fill_between(
            stats["date"], stats["smin"], stats["smax"],
            color=color, alpha=0.15, zorder=2,
        )

        # Source mean line
        ax.plot(
            stats["date"], stats["mean"],
            color=color, linewidth=2, zorder=4,
            label=f"{source} (mean of {src_data['model'].nunique()} models)",
        )

    ax.set_title("ENSO Nino3.4 Forecast by Source")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, means_df)
    ax.legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved source comparison plot to %s", out_path)

    return fig


def plot_forecast_evolution(
    forecast_df: pd.DataFrame,
    source: str = "CFS",
    model: str | None = None,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "forecast_evolution.png",
) -> plt.Figure:
    """Plot 4: How successive monthly forecasts have evolved over time.

    Each line represents one initialization date's forecast trajectory.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    # Load historical forecasts from the source directory
    src_dir = FIGURES_DIR.parent / "data" / "forecasts" / source
    if not src_dir.exists():
        logger.warning("No historical data for source %s", source)
        means = get_ensemble_means(_get_forecast_only(forecast_df))
        if model:
            means = means[means["model"] == model]
        means = means[means["source"] == source]

        if means.empty:
            plt.close(fig)
            return fig

        init_dates = sorted(means["init_date"].unique())
    else:
        csv_files = sorted(src_dir.glob("*.csv"))
        all_dfs = []
        for f in csv_files[-12:]:  # last 12 months
            try:
                df = pd.read_csv(f)
                all_dfs.append(df)
            except Exception:
                continue

        if not all_dfs:
            plt.close(fig)
            return fig

        means = pd.concat(all_dfs, ignore_index=True)
        means = means[means["member_id"] == "mean"]

        # Filter CFS to forecast months only
        if source == "CFS":
            combined = pd.concat(all_dfs, ignore_index=True)
            means = _filter_cfs_forecast_months(combined)
            means = means[means["member_id"] == "mean"]

        if model:
            means = means[means["model"] == model]
        init_dates = sorted(means["init_date"].unique())

    if not len(init_dates):
        plt.close(fig)
        return fig

    # Color gradient: older = lighter, newer = darker
    cmap = plt.cm.Blues if source == "CFS" else plt.cm.Reds
    n_init = len(init_dates)

    for idx, init_dt in enumerate(init_dates):
        idf = means[means["init_date"] == init_dt].copy()
        idf["date"] = idf["target_month"].apply(_target_month_to_date)
        idf = idf.sort_values("date")

        alpha = 0.3 + 0.7 * (idx / max(1, n_init - 1))
        color = cmap(0.3 + 0.6 * (idx / max(1, n_init - 1)))

        label = init_dt if idx in (0, n_init - 1) else None
        ax.plot(
            idf["date"], idf["nino34_anom"],
            color=color, linewidth=1.2, alpha=alpha, zorder=3,
            label=f"Init: {label}" if label else None,
        )

    title_model = model or "ensemble mean"
    ax.set_title(f"Forecast Evolution: {source} {title_model}")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, means)
    ax.legend(loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved forecast evolution plot to %s", out_path)

    return fig


def plot_distribution(
    forecast_df: pd.DataFrame,
    save: bool = True,
    filename: str = "forecast_distribution.png",
) -> plt.Figure:
    """Plot 5: Box/violin plot of forecast distribution by target month.

    Only shows true forecast months (not CFS historical verification).
    """
    _setup_style()

    # Filter to forecast-only months
    forecast_only = _get_forecast_only(forecast_df)

    comparison = get_model_comparison_df(forecast_only)
    comparison = deduplicate_models(comparison)

    if comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        logger.warning("No data for distribution plot")
        plt.close(fig)
        return fig

    comparison["date"] = comparison["target_month"].apply(_target_month_to_date)

    # Filter to relevant time window (next ~8 months from now)
    now = pd.Timestamp(date.today().replace(day=1))
    x_end = now + pd.DateOffset(months=8)
    max_target = comparison["date"].max()
    if max_target > x_end:
        x_end = max_target
    comparison = comparison[(comparison["date"] >= now - pd.DateOffset(months=1)) & (comparison["date"] <= x_end)]

    # Get unique target months in order
    target_months = sorted(comparison["target_month"].unique())
    n_months = len(target_months)

    # Fixed figure size — reasonable for typical 6-12 forecast months
    fig, ax = plt.subplots(figsize=(max(8, min(14, n_months * 1.3)), 6))

    _add_enso_shading(ax, ymin=-3, ymax=3)

    box_data = []
    positions = []
    colors = []
    tick_labels = []

    for i, tm in enumerate(target_months):
        mdf = comparison[comparison["target_month"] == tm]
        values = mdf["nino34_anom"].dropna().values
        if len(values) == 0:
            continue
        box_data.append(values)
        positions.append(i)
        tick_labels.append(pd.Timestamp(tm + "-01").strftime("%b\n%Y"))

        med = np.median(values)
        if med > 0.5:
            colors.append("#ffcccc")
        elif med < -0.5:
            colors.append("#ccccff")
        else:
            colors.append("#e0e0e0")

    if not box_data:
        plt.close(fig)
        return fig

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        medianprops=dict(color="black", linewidth=1.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual model points (jittered)
    np.random.seed(42)  # reproducible jitter
    for i, (tm, pos) in enumerate(zip(target_months, positions)):
        mdf = comparison[comparison["target_month"] == tm]
        for source in mdf["source"].unique():
            sdf = mdf[mdf["source"] == source]
            jitter = np.random.uniform(-0.15, 0.15, len(sdf))
            ax.scatter(
                pos + jitter,
                sdf["nino34_anom"],
                color=SOURCE_COLORS.get(source, "#999"),
                s=20, alpha=0.7, zorder=5, edgecolors="none",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)

    # Set y-axis: default to [-3, 3], expand if data exceeds
    all_vals = [v for bd in box_data for v in bd]
    if all_vals:
        ymin = min(-3.0, min(all_vals) - 0.3)
        ymax = max(3.0, max(all_vals) + 0.3)
    else:
        ymin, ymax = -3.0, 3.0
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_title("Distribution of Nino3.4 Forecasts by Target Month")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markersize=6, label=src)
        for src, color in SOURCE_COLORS.items()
        if src in comparison["source"].unique()
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved distribution plot to %s", out_path)

    return fig


# --- Deduplication rules for the mega plume ---
# (source, model) pairs to DROP because they duplicate another source's version
MEGA_PLUME_DROP = {
    ("NMME", "NCEP-CFSv2"),   # use CFS/CFSv2 instead
    ("NMME", "NMME"),          # overall mean, not a distinct model
    ("C3S", "NCEP"),           # CFSv2 duplicate
    ("C3S", "ECCC"),           # same as NMME/ECCC-GEM5.2-NEMO
}

# Distinct color per model for the mega plume
MEGA_COLORS = {
    "CFSv2": "#d62728",
    "ECMWF": "#1f77b4",
    "Meteo-France": "#2ca02c",
    "DWD": "#ff7f0e",
    "CMCC": "#9467bd",
    "BOM": "#e377c2",
    "ECCC-CanESM5": "#17becf",
    "ECCC-GEM5.2-NEMO": "#bcbd22",
    "NCAR-CESM1": "#8c564b",
    "NCAR-CCSM4": "#7f7f7f",
    "NASA-GEOS-S2S-2": "#aec7e8",
}


def _build_mega_df(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Build a deduplicated DataFrame for the mega plume.

    Drops duplicate physical models, keeping the preferred source version.
    """
    df = forecast_df.copy()
    mask = df.apply(lambda r: (r["source"], r["model"]) not in MEGA_PLUME_DROP, axis=1)
    deduped = df[mask].reset_index(drop=True)

    n_dropped = len(df) - len(deduped)
    if n_dropped > 0:
        logger.info("Mega plume dedup: dropped %d rows", n_dropped)

    return deduped


def plot_mega_plume(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    save: bool = True,
    filename: str = "mega_plume.png",
) -> plt.Figure:
    """Combined plume with all ensemble members from all sources, deduplicated.

    Each model's members are shown as thin lines in that model's color,
    with the model mean as a thicker line. A multi-model mean is shown as
    a thick dashed black line.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    _add_enso_shading(ax)
    _add_observed(ax, obs_df)

    # Filter to forecast-only and deduplicate
    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    means = mega[mega["member_id"] == "mean"].copy()

    if members.empty and means.empty:
        logger.warning("No data for mega plume")
        plt.close(fig)
        return fig

    # Plot individual members per model
    if not members.empty:
        members["date"] = members["target_month"].apply(_target_month_to_date)
        for model in members["model"].unique():
            color = MEGA_COLORS.get(model, "#999999")
            model_members = members[members["model"] == model]
            for member_id, mdf in model_members.groupby("member_id"):
                mdf = mdf.sort_values("date")
                ax.plot(mdf["date"], mdf["nino34_anom"],
                        color=color, alpha=0.08, linewidth=0.35, zorder=1)

    # Plot per-model means
    if not means.empty:
        means["date"] = means["target_month"].apply(_target_month_to_date)
        for model in sorted(means["model"].unique()):
            color = MEGA_COLORS.get(model, "#999999")
            mdf = means[means["model"] == model].sort_values("date")
            src = mdf["source"].iloc[0]
            n_mem = members[members["model"] == model]["member_id"].nunique() if not members.empty else 0
            label = f"{model} ({n_mem})"
            ax.plot(mdf["date"], mdf["nino34_anom"],
                    color=color, linewidth=1.8, alpha=0.85, zorder=4, label=label)

        # Multi-model mean (average of per-model means)
        mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")
        ax.plot(mm["date"], mm["nino34_anom"],
                color="black", linewidth=3, linestyle="--", zorder=6,
                label="Multi-model mean")

    n_total = sum(
        members[members["model"] == m]["member_id"].nunique()
        for m in members["model"].unique()
    ) if not members.empty else 0
    n_models = means["model"].nunique() if not means.empty else 0

    ax.set_title(f"ENSO Nino3.4 Combined Forecast Plume ({n_models} models, {n_total} members)")
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_xlabel("")
    _format_date_axis(ax)
    _set_axis_limits(ax, obs_df, mega)

    ax.legend(loc="upper left", fontsize=8, framealpha=1.0, facecolor="white", edgecolor="lightgray", ncol=2)

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved mega plume plot to %s", out_path)

    return fig


def plot_member_distribution(
    forecast_df: pd.DataFrame,
    save: bool = True,
    filename: str = "member_distribution.png",
) -> plt.Figure:
    """Distribution plot showing every ensemble member as a dot, colored by model.

    Model means are shown as larger, bold markers. Deduplicated.
    """
    _setup_style()

    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    means = mega[mega["member_id"] == "mean"].copy()

    if members.empty:
        fig, ax = plt.subplots()
        logger.warning("No member data for distribution")
        plt.close(fig)
        return fig

    members["date"] = members["target_month"].apply(_target_month_to_date)
    means["date"] = means["target_month"].apply(_target_month_to_date)

    # Filter to relevant time window
    now = pd.Timestamp(date.today().replace(day=1))
    x_end = now + pd.DateOffset(months=8)
    max_target = members["date"].max()
    if max_target > x_end:
        x_end = max_target
    members = members[(members["date"] >= now - pd.DateOffset(months=1)) & (members["date"] <= x_end)]
    means = means[(means["date"] >= now - pd.DateOffset(months=1)) & (means["date"] <= x_end)]

    target_months = sorted(members["target_month"].unique())
    n_months = len(target_months)
    models = sorted(members["model"].unique())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(10, min(16, n_months * 1.5)), 7))
    _add_enso_shading(ax, ymin=-3, ymax=3)

    # Assign horizontal offsets to spread models within each month
    model_offsets = {}
    width = 0.7  # total width for all models within one month
    for i, model in enumerate(models):
        model_offsets[model] = -width / 2 + (i + 0.5) * width / n_models

    np.random.seed(42)

    for i, tm in enumerate(target_months):
        for model in models:
            color = MEGA_COLORS.get(model, "#999999")

            # Individual members: small dots
            mdf = members[(members["target_month"] == tm) & (members["model"] == model)]
            if len(mdf) == 0:
                continue

            x_base = i + model_offsets[model]
            jitter = np.random.uniform(-0.02, 0.02, len(mdf))

            ax.scatter(
                x_base + jitter, mdf["nino34_anom"],
                color=color, s=8, alpha=0.35, zorder=3, edgecolors="none",
            )

            # Model mean: larger bold marker
            mean_row = means[(means["target_month"] == tm) & (means["model"] == model)]
            if len(mean_row) > 0:
                ax.scatter(
                    [x_base], [mean_row["nino34_anom"].iloc[0]],
                    color=color, s=80, alpha=0.95, zorder=5,
                    edgecolors="black", linewidths=0.8, marker="D",
                )

    tick_labels = [pd.Timestamp(tm + "-01").strftime("%b\n%Y") for tm in target_months]
    ax.set_xticks(range(len(target_months)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_title("Nino3.4 Forecast Distribution by Model and Target Month")

    # Y-axis
    all_vals = members["nino34_anom"].dropna().tolist()
    if all_vals:
        ymin = min(-3.0, min(all_vals) - 0.3)
        ymax = max(3.0, max(all_vals) + 0.3)
    else:
        ymin, ymax = -3.0, 3.0
    ax.set_ylim(ymin, ymax)

    # Legend: one entry per model (diamond = mean)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=MEGA_COLORS.get(m, "#999"),
               markersize=7, markeredgecolor="black", markeredgewidth=0.5, label=m)
        for m in models
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        fontsize=7, framealpha=1.0, facecolor="white", edgecolor="lightgray", ncol=2,
    )

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved member distribution plot to %s", out_path)

    return fig


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute a weighted quantile.

    Args:
        values: 1D array of data values.
        weights: 1D array of weights (same length as values).
        q: Quantile in [0, 1].
    """
    idx = np.argsort(values)
    sorted_vals = values[idx]
    sorted_weights = weights[idx]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]  # normalize to [0, 1]
    return float(np.interp(q, cum_weights, sorted_vals))


def plot_historical_context(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    start_year: int = 1990,
    save: bool = True,
    filename: str = "historical_context.png",
) -> plt.Figure:
    """Observed Nino3.4 since 1990 with El Nino/La Nina shading and forecast overlay.

    Red fill between the line and +0.5 when above +0.5 (El Nino).
    Blue fill between the line and -0.5 when below -0.5 (La Nina).
    Multi-model mean forecast appended as a dashed line with a shaded
    forecast-period indicator.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(16, 5.5))

    # Always load the full observed record from disk (not the 24-month subset
    # that other plots use for context)
    from enso_forecast.config import OBSERVED_DIR
    obs_path = OBSERVED_DIR / "nino34_monthly.csv"
    if not obs_path.exists():
        logger.warning("No observed data for historical context plot")
        plt.close(fig)
        return fig
    obs_full = pd.read_csv(obs_path)

    obs_full["date"] = pd.to_datetime(obs_full["date"])
    obs_full = obs_full[obs_full["date"] >= f"{start_year}-01-01"].sort_values("date")

    if obs_full.empty:
        plt.close(fig)
        return fig

    dates = obs_full["date"].values
    anom = obs_full["nino34_anom"].values

    # Plot observed line
    ax.plot(dates, anom, color="black", linewidth=1.2, zorder=3, label="Observed Nino3.4")

    # El Nino shading: fill between line and +0.5 where line > +0.5
    ax.fill_between(
        dates, 0.5, anom,
        where=anom > 0.5,
        color="#d62728", alpha=0.35, interpolate=True, zorder=2,
        label="El Ni\u00f1o (> +0.5\u00b0C)",
    )

    # La Nina shading: fill between line and -0.5 where line < -0.5
    ax.fill_between(
        dates, -0.5, anom,
        where=anom < -0.5,
        color="#1f77b4", alpha=0.35, interpolate=True, zorder=2,
        label="La Ni\u00f1a (< \u20130.5\u00b0C)",
    )

    # Threshold lines
    ax.axhline(0.5, color="#d62728", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.axhline(-0.5, color="#1f77b4", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.4)

    # --- Forecast overlay ---
    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)
    means = mega[mega["member_id"] == "mean"].copy()

    members = mega[mega["member_id"] != "mean"].copy()

    if not means.empty:
        # Multi-model mean (equal weight per model)
        mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")

        # Compute model-weighted 25th/75th percentiles from individual members
        target_months_sorted = sorted(mm["target_month"].unique())
        q25_vals = []
        q75_vals = []
        for tm in target_months_sorted:
            tm_members = members[members["target_month"] == tm]
            if tm_members.empty:
                q25_vals.append(np.nan)
                q75_vals.append(np.nan)
                continue
            model_counts = tm_members.groupby("model").size()
            weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
            vals = tm_members["nino34_anom"].values
            valid = ~np.isnan(vals)
            if valid.sum() == 0:
                q25_vals.append(np.nan)
                q75_vals.append(np.nan)
                continue
            q25_vals.append(_weighted_quantile(vals[valid], weights[valid], 0.25))
            q75_vals.append(_weighted_quantile(vals[valid], weights[valid], 0.75))

        # Build arrays including last observed point for smooth connection
        last_obs_date = obs_full["date"].iloc[-1]
        last_obs_val = obs_full["nino34_anom"].iloc[-1]

        fc_dates = pd.DatetimeIndex([last_obs_date] + mm["date"].tolist())
        forecast_dates = fc_dates.values
        forecast_mean = np.array([last_obs_val] + mm["nino34_anom"].tolist())
        forecast_q25 = np.array([last_obs_val] + q25_vals)
        forecast_q75 = np.array([last_obs_val] + q75_vals)

        # 25th-75th percentile shading
        ax.fill_between(
            forecast_dates, forecast_q25, forecast_q75,
            color="#ff7f0e", alpha=0.25, zorder=3,
            label="25th\u201375th percentile",
        )

        # Multi-model mean line
        ax.plot(
            forecast_dates, forecast_mean,
            color="black", linewidth=2.2, linestyle="--", zorder=4,
            label="Multi-model mean forecast",
        )

        # Forecast period indicator: light gray background band
        forecast_start = pd.Timestamp(mm["date"].min())
        forecast_end = pd.Timestamp(mm["date"].max())
        ax.axvspan(
            forecast_start, forecast_end,
            color="gray", alpha=0.07, zorder=0,
        )
        mid_forecast = forecast_start + (forecast_end - forecast_start) / 2
        ax.text(
            mid_forecast, 0.97, "Forecast",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, color="gray",
            fontstyle="italic",
        )

    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_title("ENSO Nino3.4 Index: Historical Record and Current Forecast")

    # Format x-axis for long time series
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Y-axis: accommodate the data range including forecast percentiles
    all_vals = list(anom)
    if not means.empty:
        all_vals.extend(mm["nino34_anom"].tolist())
        all_vals.extend([v for v in q75_vals if not np.isnan(v)])
        all_vals.extend([v for v in q25_vals if not np.isnan(v)])
    ymin = min(-3.0, min(all_vals) - 0.3)
    ymax = max(3.0, max(all_vals) + 0.3)
    ax.set_ylim(ymin, ymax)

    ax.legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="lightgray")

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved historical context plot to %s", out_path)

    return fig


def plot_member_box_distribution(
    forecast_df: pd.DataFrame,
    save: bool = True,
    filename: str = "member_box_distribution.png",
) -> plt.Figure:
    """Distribution plot with ensemble member dots and an equal-weight box plot.

    Each model's ensemble members are shown as small colored dots.
    A box-and-whisker plot is overlaid computed from all individual members,
    but each member is weighted by 1/(number of members for that model),
    so every model contributes equally regardless of ensemble size.
    """
    _setup_style()

    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()

    if members.empty:
        fig, ax = plt.subplots()
        logger.warning("No member data for box distribution")
        plt.close(fig)
        return fig

    members["date"] = members["target_month"].apply(_target_month_to_date)

    # Filter to relevant time window
    now = pd.Timestamp(date.today().replace(day=1))
    x_end = now + pd.DateOffset(months=8)
    max_target = members["date"].max()
    if max_target > x_end:
        x_end = max_target
    members = members[
        (members["date"] >= now - pd.DateOffset(months=1)) & (members["date"] <= x_end)
    ]

    target_months = sorted(members["target_month"].unique())
    n_months = len(target_months)
    models = sorted(members["model"].unique())

    fig, ax = plt.subplots(figsize=(max(10, min(16, n_months * 1.5)), 7))
    _add_enso_shading(ax, ymin=-3, ymax=3)

    np.random.seed(42)

    # --- Scatter: all individual members as small dots ---
    for i, tm in enumerate(target_months):
        for model in models:
            color = MEGA_COLORS.get(model, "#999999")
            mdf = members[(members["target_month"] == tm) & (members["model"] == model)]
            if len(mdf) == 0:
                continue
            jitter = np.random.uniform(-0.25, 0.25, len(mdf))
            ax.scatter(
                i + jitter, mdf["nino34_anom"],
                color=color, s=6, alpha=0.25, zorder=2, edgecolors="none",
            )

    # --- Weighted box plot from all individual members ---
    # Each member gets weight = 1/N where N = number of members for that model
    # in that target month. This ensures equal model weighting.
    from matplotlib.patches import FancyBboxPatch

    box_width = 0.5
    for i, tm in enumerate(target_months):
        tm_members = members[members["target_month"] == tm]
        if tm_members.empty:
            continue

        # Compute weights: 1/N per model
        model_counts = tm_members.groupby("model").size()
        weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
        vals = tm_members["nino34_anom"].values

        # Remove NaNs
        valid = ~np.isnan(vals)
        vals = vals[valid]
        weights = weights[valid]
        if len(vals) == 0:
            continue

        # Compute weighted statistics
        q25 = _weighted_quantile(vals, weights, 0.25)
        q50 = _weighted_quantile(vals, weights, 0.50)
        q75 = _weighted_quantile(vals, weights, 0.75)
        iqr = q75 - q25
        whisker_lo = max(vals.min(), q25 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q75 + 1.5 * iqr)

        # Box color based on median
        if q50 > 0.5:
            box_color = "#ffcccc"
        elif q50 < -0.5:
            box_color = "#ccccff"
        else:
            box_color = "#e0e0e0"

        x = i
        # Draw box (IQR)
        rect = plt.Rectangle(
            (x - box_width / 2, q25), box_width, iqr,
            facecolor=box_color, edgecolor="black",
            linewidth=1.2, alpha=0.55, zorder=4,
        )
        ax.add_patch(rect)
        # Median line
        ax.plot(
            [x - box_width / 2, x + box_width / 2], [q50, q50],
            color="black", linewidth=2, zorder=5,
        )
        # Whiskers
        ax.plot([x, x], [whisker_lo, q25], color="black", linewidth=1.2, zorder=4)
        ax.plot([x, x], [q75, whisker_hi], color="black", linewidth=1.2, zorder=4)
        # Caps
        cap_w = box_width * 0.3
        ax.plot(
            [x - cap_w, x + cap_w], [whisker_lo, whisker_lo],
            color="black", linewidth=1.2, zorder=4,
        )
        ax.plot(
            [x - cap_w, x + cap_w], [whisker_hi, whisker_hi],
            color="black", linewidth=1.2, zorder=4,
        )

    tick_labels = [pd.Timestamp(tm + "-01").strftime("%b\n%Y") for tm in target_months]
    ax.set_xticks(range(len(target_months)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Nino3.4 SST Anomaly (\u00b0C)")
    ax.set_title(
        f"Nino3.4 Forecast Distribution ({len(models)} models, model-weighted box plot)"
    )

    # Y-axis
    all_vals = members["nino34_anom"].dropna().tolist()
    if all_vals:
        ymin = min(-3.0, min(all_vals) - 0.3)
        ymax = max(3.0, max(all_vals) + 0.3)
    else:
        ymin, ymax = -3.0, 3.0
    ax.set_ylim(ymin, ymax)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MEGA_COLORS.get(m, "#999"),
               markersize=5, markeredgecolor="none", label=m)
        for m in models
    ]
    legend_elements.append(
        Patch(facecolor="#e0e0e0", edgecolor="black", alpha=0.55,
              label="IQR (model-weighted)")
    )
    ax.legend(
        handles=legend_elements, loc="upper left",
        fontsize=7, framealpha=1.0, facecolor="white", edgecolor="lightgray", ncol=2,
    )

    _add_footnote(ax)
    fig.tight_layout()

    if save:
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info("Saved member box distribution plot to %s", out_path)

    return fig


def plot_all(
    forecast_df: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    cfs_df: pd.DataFrame | None = None,
) -> list[plt.Figure]:
    """Generate all 5 standard plots."""
    figures = []

    if cfs_df is None:
        cfs_df = forecast_df[forecast_df["source"] == "CFS"].copy()

    c3s_df = forecast_df[forecast_df["source"] == "C3S"].copy()

    # Plot 1: CFS Plume
    try:
        fig = plot_cfs_plume(cfs_df, obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("CFS plume plot failed: %s", e)

    # Plot 1b: C3S Plume (if member data available)
    if not c3s_df.empty and (c3s_df["member_id"] != "mean").any():
        try:
            fig = plot_c3s_plume(c3s_df, obs_df)
            figures.append(fig)
        except Exception as e:
            logger.error("C3S plume plot failed: %s", e)

    # Plot 1c: NMME Plume (if member data available)
    nmme_df = forecast_df[forecast_df["source"] == "NMME"].copy()
    if not nmme_df.empty and (nmme_df["member_id"] != "mean").any():
        try:
            fig = plot_nmme_plume(nmme_df, obs_df)
            figures.append(fig)
        except Exception as e:
            logger.error("NMME plume plot failed: %s", e)

    # Plot 2: Multi-model comparison
    try:
        fig = plot_model_comparison(forecast_df, obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Model comparison plot failed: %s", e)

    # Plot 3: Source comparison
    try:
        fig = plot_source_comparison(forecast_df, obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Source comparison plot failed: %s", e)

    # Plot 4: Forecast evolution
    try:
        fig = plot_forecast_evolution(forecast_df, obs_df=obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Forecast evolution plot failed: %s", e)

    # Plot 5: Distribution
    try:
        fig = plot_distribution(forecast_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Distribution plot failed: %s", e)

    # Plot 6: Mega plume (all sources, deduplicated)
    try:
        fig = plot_mega_plume(forecast_df, obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Mega plume plot failed: %s", e)

    # Plot 7: Member distribution (per-model dots with model-mean diamonds)
    try:
        fig = plot_member_distribution(forecast_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Member distribution plot failed: %s", e)

    # Plot 8: Member distribution with equal-weight box plot
    try:
        fig = plot_member_box_distribution(forecast_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Member box distribution plot failed: %s", e)

    # Plot 9: Historical context (observed since 1990 + forecast)
    try:
        fig = plot_historical_context(forecast_df, obs_df)
        figures.append(fig)
    except Exception as e:
        logger.error("Historical context plot failed: %s", e)

    logger.info("Generated %d plots", len(figures))
    return figures
