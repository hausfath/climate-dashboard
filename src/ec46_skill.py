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
# ERA5 climatology sampled on the SAME coarse grid the EC46 fetcher uses
# (10x15 deg, cos-lat weights). Anomalizing the coarse-grid forecast
# against the full-resolution climatology carried the grid's seasonally
# varying sampling bias into the anomaly (and inflated the "lead-0 bias"
# to +0.2 degC); with this file the sampling operator cancels.
COARSE_CLIM_CSV = ROOT / "data" / "ec46_coarse_clim_doy.csv"


def _load_coarse_clim() -> pd.Series | None:
    if not COARSE_CLIM_CSV.exists():
        return None
    c = pd.read_csv(COARSE_CLIM_CSV, comment="#")
    return c.set_index("day_of_year")["clim_C"]

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
                        bias_correction: float = 0.0,
                        curve: dict | None = None,
                        init_date: pd.Timestamp | None = None
                        ) -> pd.DataFrame | None:
    """Convert an EC46 init's absolute t2m mean to preindustrial anomaly.

    Bias handling, in anomaly space (the climatology term is unaffected):

    * ``curve`` given -> a **lead-dependent** correction ``c(lead)`` from
      :func:`fit_lead_bias_curve` is added to each forecast day, with the
      lead counted from ``init_date`` (or the frame's ``init_date`` column,
      else its first date). This is the production path since 2026-09-02.
    * otherwise ``bias_correction`` (a constant, °C) is added to every day
      — the legacy lead-0-only correction, kept for the lead-0 gate and
      for diagnostics.

    The forecast is intentionally *not* anchored to recent observations,
    so the skill plot, the dashboard tail and the monthly blend all see
    the identical transform.
    """
    if fcst.empty:
        return None
    f = fcst.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["day_of_year"] = f["date"].dt.dayofyear
    if curve is not None:
        if init_date is None:
            if "init_date" in f.columns:
                init_date = pd.Timestamp(pd.to_datetime(f["init_date"]).iloc[0])
            else:
                init_date = f["date"].min()
        lead = (f["date"] - pd.Timestamp(init_date)).dt.days.to_numpy()
        bias_correction = lead_bias(curve, lead)

    coarse = _load_coarse_clim()
    if coarse is not None:
        f["clim_C"] = f["day_of_year"].map(coarse)
    else:   # legacy path: full-resolution climatology (grid bias remains)
        doy_clim = obs.groupby("day_of_year")["climatology"].mean()
        f["clim_C"] = f["day_of_year"].map(doy_clim)
    f["pi_offset"] = f["date"].apply(lambda d: MONTHLY_PREINDUSTRIAL_OFFSETS[d.month])
    f["forecast_anom"] = (f["t2m_mean"] + bias_correction
                          - f["clim_C"] + f["pi_offset"])
    return f[["date", "forecast_anom"]]


# ---------------------------------------------------------------------------
# Lead-dependent bias correction
# ---------------------------------------------------------------------------
# The EC46 day-0 field sits ~0.05 °C warm of ERA5 (IFS analysis / first-day
# spin-up), and that offset decays within a few days; beyond a week the
# pooled forecast-minus-ERA5 error is ~0 across the archive. A constant
# lead-0 correction therefore pushed every lead > ~5 d about 0.05 °C too
# cold (this WAS the "cold drift" in monthly_blend_lead_errors.csv before
# 2026-09-02). The correction is now a saturating exponential
#     c(L) = a + b * (1 - exp(-L / tau))          [°C, added to forecast]
# fitted to the pooled per-lead mean of (obs - raw forecast), weighted by
# the number of pairings at each lead. Verification uses a leave-one-
# month-out fit so a month's own inits never calibrate its own errors.

LEAD_CURVE_MIN_PAIRS = 5         # pairings needed for a lead to enter the fit
LEAD_CURVE_MIN_LEADS = 4         # distinct leads needed to fit 3 parameters
LEAD_CURVE_BOUNDS = ([-1.0, -1.0, 0.5], [1.0, 1.0, 30.0])   # a, b, tau


def _sat_exp(lead, a, b, tau):
    return a + b * (1.0 - np.exp(-np.asarray(lead, dtype=float) / tau))


def _lead_pairs(archive: list[tuple[pd.Timestamp, pd.DataFrame]],
                obs: pd.DataFrame) -> pd.DataFrame:
    """One row per verified forecast day: ``init``, ``lead_day`` and
    ``diff`` = obs anomaly − raw (uncorrected) forecast anomaly."""
    obs_anom = obs.set_index("date")["anomaly"]
    parts = []
    for init_date, raw in archive:
        anom = _anomalize_forecast(raw, obs, bias_correction=0.0)
        if anom is None or anom.empty:
            continue
        a = anom.set_index("date")["forecast_anom"]
        common = a.index.intersection(obs_anom.index)
        if len(common) == 0:
            continue
        parts.append(pd.DataFrame({
            "init": init_date,
            "lead_day": (common - init_date).days,
            "diff": obs_anom.loc[common].to_numpy() - a.loc[common].to_numpy(),
        }))
    if not parts:
        return pd.DataFrame(columns=["init", "lead_day", "diff"])
    return pd.concat(parts, ignore_index=True)


def fit_lead_bias_curve(pairs: pd.DataFrame) -> dict:
    """Fit the lead-dependent correction to (obs − forecast) pairings.

    Returns a dict with the curve parameters ``a, b, tau``; ``c0`` and
    ``c_inf`` (correction at lead 0 and asymptotically); ``se_by_lead``
    (standard error of the pooled mean at each lead, °C); ``n_inits``,
    ``n_pairs``; and ``method`` (``"sat_exp"``, ``"constant"`` when there
    is too little data to fit, or ``"none"`` with no pairings at all).
    """
    empty = {"a": 0.0, "b": 0.0, "tau": 1.0, "c0": 0.0, "c_inf": 0.0,
             "se_by_lead": pd.Series(dtype=float), "emp_by_lead": None,
             "n_inits": 0, "n_pairs": 0, "method": "none", "error": None}
    if pairs is None or pairs.empty:
        return empty
    g = pairs.groupby("lead_day")["diff"].agg(["mean", "std", "count"])
    g = g[g["count"] >= LEAD_CURVE_MIN_PAIRS]
    if g.empty:
        return empty
    se = (g["std"] / np.sqrt(g["count"])).fillna(0.05)
    lead0 = float(g["mean"].loc[0]) if 0 in g.index else float(g["mean"].iloc[0])
    # Empirical fallback: pooled per-lead mean, 5-day centred smoothing.
    # Used when the parametric fit is impossible or raises, so a fit
    # failure can never silently degrade to a constant lead-0 correction
    # (that constant applied at every lead is the bug this replaces).
    emp = g["mean"].rolling(5, center=True, min_periods=1).mean()
    out = dict(empty, se_by_lead=se, emp_by_lead=emp,
               n_inits=int(pairs["init"].nunique()), n_pairs=int(len(pairs)))
    if len(g) < LEAD_CURVE_MIN_LEADS:
        out.update(a=lead0, b=0.0, tau=1.0, c0=lead0,
                   c_inf=float(emp.iloc[-1]), method="empirical")
        return out
    try:
        from scipy.optimize import curve_fit
        x = np.asarray(g.index, dtype=float)
        y = np.asarray(g["mean"], dtype=float)
        w = 1.0 / np.sqrt(np.asarray(g["count"], dtype=float))
        popt, _ = curve_fit(_sat_exp, x, y, p0=[lead0, -lead0, 3.0], sigma=w,
                            bounds=LEAD_CURVE_BOUNDS)
        a, b, tau = (float(v) for v in popt)
        out.update(a=a, b=b, tau=tau, c0=a, c_inf=a + b, method="sat_exp")
    except Exception as e:  # noqa: BLE001
        logger.warning("EC46 skill: lead-curve fit failed (%r); using the "
                       "smoothed empirical per-lead means", e)
        out.update(a=lead0, b=0.0, tau=1.0, c0=float(emp.iloc[0]),
                   c_inf=float(emp.iloc[-1]), method="empirical",
                   error=f"{type(e).__name__}: {e}")
    return out


def curve_summary(curve: dict | None) -> dict:
    """JSON-safe summary of a bias curve (for figure metadata / logging)."""
    if not curve:
        return {"method": "none"}
    return {"method": curve.get("method"), "c0": round(float(curve.get("c0", 0)), 4),
            "c_inf": round(float(curve.get("c_inf", 0)), 4),
            "tau": round(float(curve.get("tau", 0)), 2),
            "n_inits": int(curve.get("n_inits", 0)),
            "error": curve.get("error")}


def lead_bias(curve: dict | None, lead):
    """Correction (°C, add to the forecast anomaly) at the given lead(s)."""
    lead = np.clip(np.asarray(lead, dtype=float), 0, None)
    if not curve or curve.get("method") == "none":
        return np.zeros_like(lead)
    if curve.get("method") == "empirical":
        emp = curve["emp_by_lead"]
        return np.interp(lead, np.asarray(emp.index, dtype=float),
                         np.asarray(emp, dtype=float))   # clamps at the ends
    return _sat_exp(lead, curve["a"], curve["b"], curve["tau"])


def lead_bias_se(curve: dict | None, lead) -> np.ndarray:
    """Standard error of the pooled (obs − forecast) mean at each lead,
    nearest available lead; 0.05 °C where the archive has no pairings."""
    lead = np.atleast_1d(np.asarray(lead, dtype=float))
    se = curve.get("se_by_lead") if curve else None
    if se is None or len(se) == 0:
        return np.full(lead.shape, 0.05)
    idx = se.index.to_numpy(dtype=float)
    nearest = np.abs(lead[:, None] - idx[None, :]).argmin(axis=1)
    return se.to_numpy()[nearest]


def lead_curves_by_month(archive, obs) -> tuple[dict, dict]:
    """(full_curve, {init_month_period: leave-one-month-out curve})."""
    pairs = _lead_pairs(archive, obs)
    full = fit_lead_bias_curve(pairs)
    lomo = {}
    if not pairs.empty:
        months = pairs["init"].dt.to_period("M")
        for m in sorted(months.unique()):
            lomo[m] = fit_lead_bias_curve(pairs[months != m])
    return full, lomo


def estimate_lead_bias_curve() -> dict:
    """Public: the full-archive lead-dependent correction (used live by the
    monthly blend and the dashboard's forecast tail)."""
    try:
        archive = _load_archive()
        if not archive:
            return fit_lead_bias_curve(pd.DataFrame())
        return fit_lead_bias_curve(_lead_pairs(archive, _load_obs()))
    except Exception as e:  # noqa: BLE001
        logger.warning("EC46 skill: lead-bias curve failed: %s", e)
        return fit_lead_bias_curve(pd.DataFrame())


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
    bias, _, n = _estimate_bias_stats(archive, obs)
    logger.info(
        "EC46 skill: bias correction = %+.3f °C from %d lead-0 pairings",
        bias, n,
    )
    return bias


def _estimate_bias_stats(
    archive: list[tuple[pd.Timestamp, pd.DataFrame]],
    obs: pd.DataFrame,
) -> tuple[float, float, int]:
    """(bias, standard error, n) of the lead-0 offset, in ANOMALY space:
    mean of obs_anomaly − forecast_anomaly(bias=0) at each init's day 0.

    With the coarse-grid climatology present this isolates the residual
    IFS-analysis-vs-ERA5 offset (small); on the legacy full-resolution
    climatology path it reduces algebraically to the old absolute-space
    obs−forecast mean (the climatology and preindustrial terms cancel).
    """
    obs_anom = obs.set_index("date")["anomaly"]
    diffs: list[float] = []
    for init_date, raw in archive:
        f = raw.copy()
        f["date"] = pd.to_datetime(f["date"])
        if init_date not in obs_anom.index:
            continue
        anom = _anomalize_forecast(f[f["date"] == init_date], obs,
                                   bias_correction=0.0)
        if anom is None or anom.empty:
            continue
        diffs.append(float(obs_anom.loc[init_date])
                     - float(anom["forecast_anom"].iloc[0]))
    if not diffs:
        return 0.0, 0.05, 0
    d = np.asarray(diffs)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.05
    return float(d.mean()), se, len(d)


def estimate_archive_bias_stats() -> tuple[float, float, int]:
    """Public anomaly-space (bias, SE, n) — used by the monthly blend."""
    try:
        archive = _load_archive()
        if not archive:
            return 0.0, 0.05, 0
        return _estimate_bias_stats(archive, _load_obs())
    except Exception as e:  # noqa: BLE001
        logger.warning("EC46 skill: bias stats failed: %s", e)
        return 0.0, 0.05, 0


def estimate_archive_bias() -> float:
    """Public helper used by the dashboard's daily-anomaly plot. Loads
    archived EC46 inits + ERA5 obs and returns the data-driven bias
    correction (°C) to add to forecast t2m so the model lines up with
    ERA5 reanalysis on a global-mean basis. Returns 0.0 if the archive
    is empty (first-run / missing data)."""
    try:
        archive = _load_archive()
        if not archive:
            return 0.0
        return _estimate_bias_correction(archive, _load_obs())
    except Exception as e:  # noqa: BLE001
        logger.warning("EC46 skill: bias estimate failed: %s", e)
        return 0.0


def make_plot(output_path: Path = OUTPUT_PNG) -> Path | None:
    """Generate the EC46 forecast-vs-obs PNG. Returns the output path on
    success, None if there is nothing to plot."""
    archive = _load_archive()
    if not archive:
        logger.warning("EC46 skill: no archived forecasts under %s", ARCHIVE_DIR)
        return None

    obs = _load_obs()
    # Lead-dependent correction, leave-one-month-out: each init is
    # corrected with a curve fitted to the OTHER months' inits, so the
    # retrospective plot is an honest out-of-sample view.
    full_curve, lomo = lead_curves_by_month(archive, obs)

    forecasts: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for init_date, raw in archive:
        curve = lomo.get(init_date.to_period("M"), full_curve)
        anom = _anomalize_forecast(raw, obs, curve=curve, init_date=init_date)
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

    if full_curve["method"] == "sat_exp":
        bias_note = (f"  ·  lead-dependent bias correction {full_curve['c0']:+.2f} °C "
                     f"at day 0 → {full_curve['c_inf']:+.2f} °C beyond ~"
                     f"{3 * full_curve['tau']:.0f} d (leave-one-month-out)")
    elif full_curve["method"] == "empirical":
        bias_note = (f"  ·  lead-dependent bias correction {full_curve['c0']:+.2f} °C "
                     f"at day 0 → {full_curve['c_inf']:+.2f} °C (empirical, "
                     "leave-one-month-out)")
    else:
        bias_note = ""
    ax.set_title(
        "ECMWF EC46 forecasts vs. ERA5 observed global temperature\n"
        f"({len(forecasts)} archived inits — {earliest_init:%Y-%m-%d} → "
        f"{init_dates[-1]:%Y-%m-%d}{bias_note})",
        fontsize=11,
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
