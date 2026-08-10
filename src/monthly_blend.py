"""EC46-based monthly temperature projection (MTD-observed + ensemble).

Replaces the MTD regression (``calculate_monthly_prediction``) as the
dashboard's monthly projection; the regression remains the automatic
fallback whenever no usable EC46 member file exists or any sanity gate
fails.

Method (temp_files/ec46_monthly_blend_DESIGN.md, revised after the
2026-08 statistics + forecast-verification reviews):

Central estimate
  - Freshest EC46 init with init_date <= last_obs_day + 1: observed MTD
    days and forecast member days tile the month with no gap and one
    coherent 50-member set (coherence over freshness; <=3 days stale).
  - Per member: month mean = mean(observed daily anomalies + member
    anomalies for remaining days). Member anomalies use the COARSE-GRID
    day-of-year climatology (data/ec46_coarse_clim_doy.csv — ERA5
    sampled on the fetcher's own 432-point grid) so the grid-sampling
    operator cancels, plus a small anomaly-space lead-0 bias correction.
  - Published central value: member median.

Interval (member quantiles were reviewed as insufficient on their own —
ensemble spread cannot represent common-mode model error, which
dominates at these leads):
  sigma_total^2 = sigma_within^2            member month-mean spread
                + sigma_between^2           shared-per-month model error:
                                            shrinkage (nu*s_hat^2 +
                                            nu0*s0^2)/(nu + nu0) with
                                            s_hat = month-block RMS error
                                            from the verification archive,
                                            nu = # verified months,
                                            s0 = the regression's own
                                            residual sigma (85-yr prior),
                                            nu0 = 4 pseudo-months
                + (bias SE * R/D)^2 + (ERA5T revision * L/D)^2
  half-widths = t-quantile(df = nu + nu0 - 1) * sigma_total
  (posterior-predictive df of the conjugate normal-inverse-gamma scheme)
  HARD CAP: until nu >= 6 verified months, the 5-95 half-width is never
  published narrower than 0.5x the regression's 2-sigma. Widths may only
  tighten as verification accumulates; they can always widen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MEMBER_DIRS = (DATA_DIR, ROOT / "forecast_skill" / "archive_members")
COARSE_CLIM_FILE = DATA_DIR / "ec46_coarse_clim_doy.csv"
CALIBRATION_FILE = ROOT / "forecast_skill" / "monthly_blend_calibration.csv"
VERIFICATION_FILE = ROOT / "forecast_skill" / "monthly_blend_verification.csv"

MAX_INIT_LAG_DAYS = 6     # staleness guard on the chosen init
MIN_MEMBERS = 40          # fewer members -> fall back (fetch-quality gate)
LEAD0_GATE = 0.15         # degC; |corrected day-0 forecast - obs| above
                          # this means something upstream broke -> fallback
ERA5T_REVISION_SIGMA = 0.01   # 1-sigma, ERA5T->final revisions of the
                              # global-mean daily anomaly (MTD portion)
NU0 = 4                   # pseudo-months anchoring sigma_between to the
                          # regression residual sigma
MIN_MONTHS_FOR_NARROW = 6  # verified months required before the cap lifts
S0_DEFAULT = 0.10         # fallback regression sigma (~day-1 residual sigma)

_PAIRS = [(5, 95, 0.95), (25, 75, 0.75)]   # (lo%, hi%, t upper prob)


# ---------------------------------------------------------------------------
# Shared transform pieces
# ---------------------------------------------------------------------------

def _load_coarse_clim() -> pd.Series | None:
    if not COARSE_CLIM_FILE.exists():
        return None
    c = pd.read_csv(COARSE_CLIM_FILE, comment="#")
    return c.set_index("day_of_year")["clim_C"]


def _member_files() -> list[tuple[pd.Timestamp, Path]]:
    """All available member matrices, newest first, deduplicated by init."""
    seen: dict[str, Path] = {}
    for d in MEMBER_DIRS:
        if not d.exists():
            continue
        for fp in d.glob("ec46_members_*.csv"):
            seen.setdefault(fp.stem.replace("ec46_members_", ""), fp)
    out = []
    for init_str, fp in seen.items():
        try:
            out.append((pd.Timestamp(init_str), fp))
        except ValueError:
            continue
    return sorted(out, reverse=True)


def _anomalize(values, dates: pd.DatetimeIndex, clim_coarse: pd.Series,
               bias_anom: float):
    """Absolute coarse-grid t2m -> preindustrial anomaly.

    values: array-like indexed like dates (Series or DataFrame rows).
    """
    from src.models_vs_obs import MONTHLY_PREINDUSTRIAL_OFFSETS
    clim = pd.Series(dates.dayofyear, index=dates).map(clim_coarse)
    pi = pd.Series([MONTHLY_PREINDUSTRIAL_OFFSETS[d.month] for d in dates],
                   index=dates)
    if isinstance(values, pd.DataFrame):
        return values.sub(clim, axis=0).add(pi, axis=0) + bias_anom
    return values - clim + pi + bias_anom


def bias_stats_anom(clim_coarse: pd.Series | None = None
                    ) -> tuple[float, float, int]:
    """(bias, SE, n): anomaly-space lead-0 offset obs_anom - fc_anom0.
    Single source of truth lives in ec46_skill (shared with the skill
    plot and the daily-plot forecast tail); with the coarse-grid
    climatology in place this is the small residual IFS-analysis-vs-ERA5
    plus Open-Meteo interpolation offset."""
    from src.ec46_skill import estimate_archive_bias_stats
    return estimate_archive_bias_stats()


# ---------------------------------------------------------------------------
# Calibration inputs
# ---------------------------------------------------------------------------

_BUCKETS = [0, 8, 15, 22]   # min days_remaining per bucket


def _load_calibration() -> pd.DataFrame | None:
    if CALIBRATION_FILE.exists():
        try:
            c = pd.read_csv(CALIBRATION_FILE, comment="#")
            if {"min_days_remaining", "s_hat", "n_months"}.issubset(c.columns):
                return c.sort_values("min_days_remaining")
        except Exception as e:
            logger.warning("Bad calibration file: %s", e)
    return None


def _calib_for(calib: pd.DataFrame | None,
               days_remaining: int) -> tuple[float, int]:
    """(s_hat, n_months) for the bucket containing days_remaining."""
    if calib is None or not len(calib):
        return 0.0, 0
    sub = calib[calib["min_days_remaining"] <= days_remaining]
    row = sub.iloc[-1] if len(sub) else calib.iloc[0]
    return float(row["s_hat"]), int(row["n_months"])


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------

def compute_monthly_blend(df_adj: pd.DataFrame) -> dict | None:
    """Blended monthly projection for the current (latest-data) month.

    df_adj: dashboard daily ERA5 frame, 'anomaly' already preindustrial-
    adjusted, 'climatology' present. Returns None whenever any input or
    sanity gate fails — the caller falls back to the regression.
    """
    from scipy import stats as sstats

    clim_coarse = _load_coarse_clim()
    if clim_coarse is None:
        logger.warning("Monthly blend: coarse climatology missing")
        return None

    obs = df_adj.sort_values("date")
    last_obs = obs["date"].max()
    month, year = last_obs.month, last_obs.year
    m_start = pd.Timestamp(year, month, 1)
    m_end = m_start + pd.offsets.MonthEnd(0)
    n_days = int((m_end - m_start).days) + 1

    usable = [(d, fp) for d, fp in _member_files()
              if d <= last_obs + pd.Timedelta(days=1)]
    if not usable:
        logger.info("Monthly blend: no usable EC46 member file")
        return None
    init_date, mem_path = usable[0]
    if (last_obs + pd.Timedelta(days=1) - init_date).days > MAX_INIT_LAG_DAYS:
        logger.warning("Monthly blend: freshest usable init %s is stale",
                       init_date.date())
        return None

    members = pd.read_csv(mem_path, parse_dates=["date"])
    mem_cols = [c for c in members.columns if c.startswith("member")]
    if len(mem_cols) < MIN_MEMBERS:
        logger.warning("Monthly blend: only %d members in %s",
                       len(mem_cols), mem_path.name)
        return None
    members = members.set_index("date")

    bias, bias_se, n_pairs = bias_stats_anom(clim_coarse)

    # Lead-0 sanity gate on the chosen init (if its day 0 has verified)
    obs_anom = obs.set_index("date")["anomaly"]
    if init_date in members.index and init_date in obs_anom.index:
        fc0 = _anomalize(members.loc[[init_date], mem_cols],
                         pd.DatetimeIndex([init_date]), clim_coarse,
                         bias).mean(axis=1)
        if abs(float(fc0.iloc[0]) - float(obs_anom.loc[init_date])) > LEAD0_GATE:
            logger.warning("Monthly blend: lead-0 gate failed (%.2f vs %.2f)",
                           float(fc0.iloc[0]),
                           float(obs_anom.loc[init_date]))
            return None

    mtd = obs[(obs["date"] >= m_start) & (obs["date"] <= last_obs)]
    rem_days = pd.date_range(last_obs + pd.Timedelta(days=1), m_end)
    if len(mtd) + len(rem_days) != n_days:
        logger.warning("Monthly blend: day accounting mismatch")
        return None
    if len(rem_days) and not rem_days.isin(members.index).all():
        logger.info("Monthly blend: init %s does not cover through %s",
                    init_date.date(), m_end.date())
        return None

    if len(rem_days):
        rem_anom = _anomalize(members.loc[rem_days, mem_cols], rem_days,
                              clim_coarse, bias)
        member_month_means = ((mtd["anomaly"].sum() + rem_anom.sum(axis=0))
                              / n_days)
        vals = member_month_means.to_numpy(dtype=float)
    else:
        vals = np.full(len(mem_cols), mtd["anomaly"].mean())

    med = float(np.median(vals))
    sigma_within = float(np.std(vals, ddof=1))

    # Regression sigma at matched days-in: the 85-year prior for the
    # shared-per-month error component, and the basis of the width cap.
    from src.dashboard import calculate_monthly_prediction
    try:
        _, reg_2sigma, _, _ = calculate_monthly_prediction(obs, month, year)
        s0 = reg_2sigma / 2 if reg_2sigma else S0_DEFAULT
    except Exception:
        reg_2sigma, s0 = None, S0_DEFAULT

    s_hat, nu = _calib_for(_load_calibration(), len(rem_days))
    sigma_between = float(np.sqrt((nu * s_hat ** 2 + NU0 * s0 ** 2)
                                  / (nu + NU0)))

    w_fc = len(rem_days) / n_days
    w_obs = len(mtd) / n_days
    sigma_total = float(np.sqrt(sigma_within ** 2 + sigma_between ** 2
                                + (bias_se * w_fc) ** 2
                                + (ERA5T_REVISION_SIGMA * w_obs) ** 2))

    df_t = nu + NU0 - 1
    quantiles: dict[int, float] = {50: med}
    hw = {p_hi: float(sstats.t.ppf(p, df_t)) * sigma_total
          for _, p_hi, p in _PAIRS}

    # Hard cap against premature narrowing (never narrower than half the
    # regression interval until enough independent months have verified)
    if nu < MIN_MONTHS_FOR_NARROW and reg_2sigma:
        min_hw90 = 0.5 * reg_2sigma
        if hw[95] < min_hw90:
            scale = min_hw90 / hw[95]
            hw = {k: v * scale for k, v in hw.items()}

    for p_lo, p_hi, _ in _PAIRS:
        quantiles[p_lo] = med - hw[p_hi]
        quantiles[p_hi] = med + hw[p_hi]

    return {
        "method": "ec46_blend",
        "prediction": med,
        "quantiles": quantiles,
        "sigma_total": sigma_total,
        "sigma_within": sigma_within,
        "sigma_between": sigma_between,
        "t_df": df_t,
        "n_verified_months": nu,
        "mtd_avg": float(mtd["anomaly"].mean()) if len(mtd) else None,
        "days_in": int(len(mtd)),
        "days_remaining": int(len(rem_days)),
        "n_days": n_days,
        "month": month,
        "year": year,
        "init_date": init_date.strftime("%Y-%m-%d"),
        "n_members": len(mem_cols),
        "bias_correction": bias,
        "bias_n_pairs": n_pairs,
        "reg_2sigma": reg_2sigma,
    }


def rank_probabilities(result: dict, hist_monthly_means: pd.Series) -> list:
    """P(final rank = k) integrating the calibrated t predictive
    distribution against the fixed historical monthly means (same ERA5
    product/baseline, so level terms cancel)."""
    from scipy import stats as sstats
    med, s, df_t = (result["prediction"], result["sigma_total"],
                    result["t_df"])
    hist = np.sort(hist_monthly_means.to_numpy())[::-1]   # descending
    edges = np.concatenate([[np.inf], hist, [-np.inf]])
    out = []
    for k in range(1, len(edges)):
        p = (sstats.t.cdf((edges[k - 1] - med) / s, df_t)
             - sstats.t.cdf((edges[k] - med) / s, df_t))
        out.append({"rank": k, "prob": float(p)})
        if k > 8:
            break
    return out


# ---------------------------------------------------------------------------
# Verification loop (daily cron)
# ---------------------------------------------------------------------------

def update_calibration() -> pd.DataFrame | None:
    """Recompute the calibration inputs (s_hat, n_months by bucket) and
    per-init verification rows from all verified months in the archive.

    Central-estimate errors use the percentile archive (every init since
    May 2026 contributes); CRPS / PIT / spread-error are added for inits
    that have member files (accumulating from Aug 2026). Also records the
    pooled ensemble-mean error by lead day (stationary-drift detector).
    """
    from src.ec46_skill import _load_archive, _load_obs

    clim_coarse = _load_coarse_clim()
    if clim_coarse is None:
        logger.warning("Calibration: coarse climatology missing — skipping")
        return None
    archive = _load_archive()
    if not archive:
        return None
    obs = _load_obs()
    obs_anom = obs.set_index("date")["anomaly"]
    bias, _, _ = bias_stats_anom(clim_coarse)
    mem_lookup = {d.strftime("%Y-%m-%d"): fp for d, fp in _member_files()}

    rows, lead_rows = [], []
    for init_date, raw in archive:
        f = raw.copy()
        f["date"] = pd.to_datetime(f["date"])
        fdates = pd.DatetimeIndex(f["date"])
        fc_anom = _anomalize(pd.Series(f["t2m_mean"].values, index=fdates),
                             fdates, clim_coarse, bias)

        # pooled per-lead ensemble-mean error over verified forecast days
        verified = fc_anom.index.intersection(obs_anom.index)
        for d in verified:
            lead_rows.append({"lead_day": (d - init_date).days,
                              "error": float(fc_anom.loc[d]
                                             - obs_anom.loc[d])})

        m_start = pd.Timestamp(init_date.year, init_date.month, 1)
        m_end = m_start + pd.offsets.MonthEnd(0)
        month_obs = obs_anom.loc[m_start:m_end]
        if len(month_obs) < (m_end - m_start).days + 1:
            continue    # month not fully verified
        truth = float(month_obs.mean())
        mtd_days = pd.date_range(m_start, init_date - pd.Timedelta(days=1))
        rem_days = pd.date_range(init_date, m_end)
        if len(mtd_days) and not mtd_days.isin(obs_anom.index).all():
            continue
        if len(rem_days) and not rem_days.isin(fc_anom.index).all():
            continue
        parts = ([obs_anom.loc[mtd_days].to_numpy()] if len(mtd_days) else [])
        if len(rem_days):
            parts.append(fc_anom.loc[rem_days].to_numpy())
        blend = float(np.concatenate(parts).mean())
        row = {"init": init_date.strftime("%Y-%m-%d"),
               "month": f"{m_start:%Y-%m}",
               "days_remaining": len(rem_days),
               "error": blend - truth,
               "crps": np.nan, "pit": np.nan, "spread": np.nan}

        mem_fp = mem_lookup.get(init_date.strftime("%Y-%m-%d"))
        if mem_fp is not None:
            mem = pd.read_csv(mem_fp, parse_dates=["date"]).set_index("date")
            mcols = [c for c in mem.columns if c.startswith("member")]
            if len(rem_days) and rem_days.isin(mem.index).all() and mcols:
                rem_anom = _anomalize(mem.loc[rem_days, mcols], rem_days,
                                      clim_coarse, bias)
                mm = ((np.concatenate(parts[:1]).sum() if len(mtd_days) else 0)
                      + rem_anom.sum(axis=0)) / len(month_obs)
                v = np.sort(mm.to_numpy())
                # CRPS (empirical, ensemble form) and PIT of the truth
                row["crps"] = float(np.mean(np.abs(v - truth))
                                    - 0.5 * np.mean(np.abs(
                                        v[:, None] - v[None, :])))
                row["pit"] = float((v < truth).mean())
                row["spread"] = float(np.std(v, ddof=1))
        rows.append(row)

    if not rows:
        return None
    v = pd.DataFrame(rows)
    VERIFICATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    v.to_csv(VERIFICATION_FILE, index=False, float_format="%.4f")

    lead = (pd.DataFrame(lead_rows).groupby("lead_day")["error"]
            .agg(["mean", "std", "count"]).reset_index())
    lead.to_csv(VERIFICATION_FILE.with_name("monthly_blend_lead_errors.csv"),
                index=False, float_format="%.4f")

    # s_hat by bucket: month is the unit of replication — average the
    # errors of a month's inits within the bucket, then RMS across months
    edges = _BUCKETS + [99]
    out = []
    for i, lo in enumerate(_BUCKETS):
        sub = v[(v["days_remaining"] >= lo)
                & (v["days_remaining"] < edges[i + 1])]
        per_month = sub.groupby("month")["error"].mean()
        out.append({"min_days_remaining": lo,
                    "s_hat": (float(np.sqrt((per_month ** 2).mean()))
                              if len(per_month) else 0.0),
                    "n_months": int(len(per_month))})
    calib = pd.DataFrame(out)
    with CALIBRATION_FILE.open("w") as fh:
        fh.write("# Monthly-blend calibration: s_hat = month-block RMS of "
                 "blend central-estimate errors (degC) by days-remaining "
                 "bucket; n_months = verified months contributing "
                 "(the effective sample size). Auto-refreshed daily by "
                 "src.monthly_blend.update_calibration.\n")
        calib.round(4).to_csv(fh, index=False)
    logger.info("Monthly-blend calibration: %s",
                calib.to_dict("records"))
    return calib


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    logging.basicConfig(level=logging.INFO)
    if "--calibrate" in sys.argv:
        print(update_calibration())
    else:
        from config import DATA_SOURCES
        from src.scraper import load_or_fetch_data
        from src.dashboard import adjust_anomalies_to_preindustrial
        src_cfg = DATA_SOURCES["era5_global"]
        df = load_or_fetch_data(src_cfg["url"], src_cfg["local_file"])
        out = compute_monthly_blend(adjust_anomalies_to_preindustrial(df))
        if out:
            q = out["quantiles"]
            print(f"\n{out['year']}-{out['month']:02d} projection "
                  f"(method {out['method']}, init {out['init_date']}, "
                  f"{out['n_members']} members):")
            print(f"  median {out['prediction']:+.3f}  "
                  f"[{q[5]:+.3f}, {q[25]:+.3f}, {q[75]:+.3f}, {q[95]:+.3f}]")
            print(f"  sigma within/between/total: {out['sigma_within']:.3f}/"
                  f"{out['sigma_between']:.3f}/{out['sigma_total']:.3f}  "
                  f"t_df={out['t_df']}  nu={out['n_verified_months']}  "
                  f"bias={out['bias_correction']:+.3f}  "
                  f"reg 2sigma={out['reg_2sigma']}")
        else:
            print(out)
