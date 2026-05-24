"""Track EC46 forecast skill by comparing every past EC46 init to ERA5
observed temperatures.

Each daily cron run archives the latest EC46 percentile-summary CSV under
``forecast_skill/archive/`` (handled by ``enso_forecast.fetchers.ec46``).
This module reads every archived init, applies the same preindustrial-
anomaly conversion + observation-anchor that the live dashboard uses, and
renders a single PNG (``forecast_skill/ec46_skill.png``) showing each
forecast trajectory coloured newest→oldest with the observed series
overlaid in black. The plot is committed by the daily GitHub Actions cron
but is intentionally NOT wired into the dashboard layout.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.models_vs_obs import MONTHLY_PREINDUSTRIAL_OFFSETS
from src.scraper import parse_era5_data

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "forecast_skill" / "archive"
OUTPUT_PNG = ROOT / "forecast_skill" / "ec46_skill.png"
ERA5_CSV = ROOT / "data" / "era5_daily_series_2t_global.csv"

def _load_obs() -> pd.DataFrame:
    """Load ERA5 daily series with anomaly already on the preindustrial
    baseline."""
    obs = parse_era5_data(ERA5_CSV)
    obs["anomaly"] = obs.apply(
        lambda r: r["anomaly"] + MONTHLY_PREINDUSTRIAL_OFFSETS[r["date"].month],
        axis=1,
    )
    return obs


def _anomalize_forecast(fcst: pd.DataFrame, obs: pd.DataFrame,
                        bias_correction: float = 0.0) -> pd.DataFrame | None:
    """Convert an EC46 init's absolute t2m mean to preindustrial anomaly,
    optionally applying a constant bias correction (°C) added to every
    forecast value before subtracting climatology.

    Unlike the dashboard's daily-anomaly plot (which anchors the forecast
    to the trailing 7-day observed mean for visual continuity), the skill
    plot intentionally does *not* anchor. Instead it applies a single
    archive-wide bias correction estimated from lead-0 (model IC vs ERA5
    obs) pairings — see ``_estimate_bias_correction``. A uniform shift
    preserves the relative differences between inits while removing the
    ECMWF operational-analysis vs ERA5-reanalysis offset that otherwise
    sits as a near-constant cold tilt under every forecast line.
    """
    if fcst.empty:
        return None
    f = fcst.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["day_of_year"] = f["date"].dt.dayofyear

    doy_clim = obs.groupby("day_of_year")["climatology"].mean()
    f["clim_C"] = f["day_of_year"].map(doy_clim)
    f["pi_offset"] = f["date"].apply(lambda d: MONTHLY_PREINDUSTRIAL_OFFSETS[d.month])
    f["forecast_anom"] = (f["t2m_mean"] + bias_correction
                          - f["clim_C"] + f["pi_offset"])
    return f[["date", "forecast_anom"]]


def _load_archive() -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Return [(init_date, raw_forecast_df), ...] sorted by init_date."""
    files = sorted(ARCHIVE_DIR.glob("era5_forecast_ec46_*.csv"))
    out = []
    for fp in files:
        init_str = fp.stem.split("era5_forecast_ec46_")[-1]
        try:
            init_date = pd.Timestamp(init_str)
        except Exception:
            logger.warning("EC46 skill: skipping unparseable %s", fp.name)
            continue
        df = pd.read_csv(fp)
        out.append((init_date, df))
    return out


def _estimate_bias_correction(
    archive: list[tuple[pd.Timestamp, pd.DataFrame]],
    obs: pd.DataFrame,
) -> float:
    """Return the bias correction (°C) to add to forecast t2m so EC46
    forecasts align with ERA5 reanalysis on a global-mean basis.

    Computed as the mean of (ERA5 obs t2m − forecast t2m) at lead-day 0
    across all archived inits where the obs date is present. Lead-day 0
    is the cleanest comparison: it isolates the IC-vs-reanalysis offset
    (ECMWF operational analysis vs ERA5) from any model drift that
    accumulates with lead time. Returns 0.0 if we have no pairings yet.
    """
    diffs: list[float] = []
    obs_indexed = obs.set_index("date")["temperature"]
    for init_date, raw in archive:
        f = raw.copy()
        f["date"] = pd.to_datetime(f["date"])
        day0 = f[f["date"] == init_date]
        if day0.empty:
            continue
        try:
            obs_val = float(obs_indexed.loc[init_date])
        except KeyError:
            continue
        diffs.append(obs_val - float(day0["t2m_mean"].iloc[0]))
    if not diffs:
        return 0.0
    bias = float(np.mean(diffs))
    logger.info(
        "EC46 skill: bias correction = %+.3f °C from %d lead-0 pairings",
        bias, len(diffs),
    )
    return bias


def make_plot(output_path: Path = OUTPUT_PNG) -> Path | None:
    """Generate the EC46 forecast-vs-obs PNG. Returns the output path on
    success, None if there is nothing to plot."""
    archive = _load_archive()
    if not archive:
        logger.warning("EC46 skill: no archived forecasts under %s", ARCHIVE_DIR)
        return None

    obs = _load_obs()
    bias = _estimate_bias_correction(archive, obs)

    forecasts: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for init_date, raw in archive:
        anom = _anomalize_forecast(raw, obs, bias_correction=bias)
        if anom is not None and not anom.empty:
            forecasts.append((init_date, anom))

    if not forecasts:
        logger.warning("EC46 skill: no usable forecasts after anomaly conversion")
        return None

    forecasts.sort(key=lambda x: x[0])
    init_dates = [d for d, _ in forecasts]
    earliest_init = min(init_dates)
    latest_fc_end = max(f["date"].max() for _, f in forecasts)

    # Plot window starts a few days before the earliest archived init so
    # the obs context is visible at the left edge.
    window_start = earliest_init - pd.Timedelta(days=10)
    obs_window = obs[(obs["date"] >= window_start) & (obs["date"] <= latest_fc_end)]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)

    # Colour by init order: oldest faded, newest saturated. Using viridis
    # with a narrow value band keeps the contrast readable when only a
    # handful of inits exist.
    n = len(forecasts)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = plt.get_cmap("viridis")

    for i, (init_date, fc) in enumerate(forecasts):
        if n == 1:
            color, alpha = cmap(0.85), 1.0
        else:
            color = cmap(norm(i))
            alpha = 0.45 + 0.55 * (i / (n - 1))  # newest = most opaque
        ax.plot(
            fc["date"], fc["forecast_anom"],
            color=color, alpha=alpha, linewidth=1.6, zorder=2 + i,
        )

    ax.plot(
        obs_window["date"], obs_window["anomaly"],
        color="black", linewidth=2.2, label="ERA5 observed", zorder=100,
    )

    bias_note = f"  ·  bias correction {bias:+.2f} °C" if bias else ""
    ax.set_title(
        "ECMWF EC46 forecasts vs. ERA5 observed global temperature\n"
        f"({len(forecasts)} archived inits — {earliest_init:%Y-%m-%d} → "
        f"{init_dates[-1]:%Y-%m-%d}{bias_note})",
        fontsize=12,
    )
    ax.set_ylabel("Anomaly vs preindustrial (°C)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    # Colourbar legend mapping init order → date label.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    if n >= 2:
        tick_idx = np.linspace(0, n - 1, min(n, 6)).round().astype(int)
        cbar.set_ticks(tick_idx)
        cbar.set_ticklabels([init_dates[i].strftime("%Y-%m-%d") for i in tick_idx])
    else:
        cbar.set_ticks([0])
        cbar.set_ticklabels([init_dates[0].strftime("%Y-%m-%d")])
    cbar.set_label("Forecast init (oldest → newest)")

    ax.legend(loc="upper left", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("EC46 skill: wrote %s (%d forecasts)", output_path, len(forecasts))
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    make_plot()
