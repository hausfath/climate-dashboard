"""ENSO Forecast Plotly visualizations for the Climate Dashboard."""

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Add ENSO project root to path so we can import enso_forecast modules
_ENSO_ROOT = Path(__file__).resolve().parent.parent / "ENSO"
if str(_ENSO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENSO_ROOT))

from enso_forecast.normalize import load_all_forecasts, get_ensemble_means
from enso_forecast.fetchers.observed import get_recent_observed
from enso_forecast.config import OBSERVED_DIR
from enso_forecast.visualize import (
    _build_mega_df,
    _get_forecast_only,
    _weighted_quantile,
    _target_month_to_date,
    MEGA_COLORS,
    MEGA_PLUME_DROP,
)

# Import theme helper from dashboard
from src.dashboard import get_theme


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_enso_forecast_data():
    """Load forecast, observed, and ONI data.

    Returns (forecast_df, obs_df, oni_df).
    """
    try:
        forecast_df = load_all_forecasts(sources=["CFS", "NMME", "C3S", "CanSIPS"])
    except Exception as e:
        logger.error(f"Failed to load ENSO forecasts: {e}")
        forecast_df = pd.DataFrame()

    try:
        obs_df = get_recent_observed(n_months=24)
    except Exception as e:
        logger.error(f"Failed to load observed Nino3.4: {e}")
        obs_df = pd.DataFrame()

    oni_path = OBSERVED_DIR / "oni.csv"
    if oni_path.exists():
        oni_df = pd.read_csv(oni_path)
        oni_df["date"] = pd.to_datetime(oni_df["date"])
    else:
        oni_df = pd.DataFrame()

    return forecast_df, obs_df, oni_df


def load_full_observed(start_year=1990):
    """Load the full observed Niño 3.4 record with rONI columns merged in."""
    obs_path = OBSERVED_DIR / "nino34_monthly.csv"
    if not obs_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(obs_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= f"{start_year}-01-01"].sort_values("date").reset_index(drop=True)
    from enso_forecast.normalize import merge_observed_with_roni
    return merge_observed_with_roni(df)


# ---------------------------------------------------------------------------
# Index-mode helpers (ONI vs rONI)
# ---------------------------------------------------------------------------

INDEX_MODES = ("oni", "roni")


def _index_meta(index_mode: str) -> dict:
    """Return label/column metadata for ONI vs rONI display modes."""
    if index_mode == "roni":
        return {
            "col": "roni_anom",
            "y_label": "Niño 3.4 minus tropical mean (rONI, \u00b0C)",
            "short": "rONI",
            "long": "Niño 3.4 Relative SST Anomaly (rONI)",
        }
    return {
        "col": "nino34_anom",
        "y_label": "Niño 3.4 SST Anomaly (\u00b0C)",
        "short": "Niño 3.4",
        "long": "Niño 3.4 SST Anomaly (ONI)",
    }


def _swap_to_nino34(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Return a copy of ``df`` with ``source_col`` renamed to ``nino34_anom``.

    Lets downstream plotting code reference a single column name regardless
    of whether the active index is ONI or rONI. If ``source_col`` is missing
    or all-NaN, returns the df unchanged so the caller can fall back.
    """
    if df is None or df.empty:
        return df
    if source_col == "nino34_anom":
        return df
    if source_col not in df.columns:
        return df
    out = df.copy()
    if "nino34_anom" in out.columns:
        out = out.drop(columns=["nino34_anom"])
    out = out.rename(columns={source_col: "nino34_anom"})
    return out


# ---------------------------------------------------------------------------
# Combined ENSO DataFrame (for annual prediction model)
# ---------------------------------------------------------------------------

def build_enso_combined(oni_df, forecast_df, obs_df=None, index_mode="oni"):
    """Build a combined observed + forecast ENSO DataFrame.

    Returns DataFrame with columns: date, year, month, <value>, is_forecast
    where <value> is ``oni`` for index_mode="oni" or ``roni`` for "roni".
    Observed data comes from NOAA's seasonal ONI / monthly Niño 3.4 (for
    ONI mode) or directly from ``obs_df.roni_anom`` (for rONI mode, which
    already merges NOAA's monthly rNINO3.4 with the seasonal RONI as
    fallback in ``merge_observed_with_roni``). Forecast portion uses the
    multi-model weighted median across the combined ensemble plume.
    """
    fc_value_col = "roni_anom" if index_mode == "roni" else "nino34_anom"
    out_col = "roni" if index_mode == "roni" else "oni"

    rows = []
    obs_dates = set()

    if index_mode == "oni":
        # Observed from NOAA's published ONI (3-month running mean)
        if not oni_df.empty and "oni" in oni_df.columns:
            for _, r in oni_df.iterrows():
                rows.append({
                    "date": r["date"],
                    "year": int(r["year"]),
                    "month": int(r["month"]),
                    out_col: r["oni"],
                    "is_forecast": False,
                })
                obs_dates.add(str(pd.Timestamp(r["date"]).to_period("M")))

        # Fill in any months with observed monthly Niño 3.4 not yet
        # covered by ONI (the seasonal ONI publication usually lags
        # monthly Niño 3.4 by 1-2 months).
        if obs_df is not None and not obs_df.empty and "nino34_anom" in obs_df.columns:
            for _, r in obs_df.iterrows():
                period = str(pd.Timestamp(r["date"]).to_period("M"))
                if period not in obs_dates:
                    rows.append({
                        "date": pd.Timestamp(r["date"]),
                        "year": int(r["year"]),
                        "month": int(r["month"]),
                        out_col: r["nino34_anom"],
                        "is_forecast": False,
                    })
                    obs_dates.add(period)
    else:
        # rONI mode: ``obs_df.roni_anom`` already merges monthly rNINO3.4
        # (preferred) with seasonal NOAA RONI (fallback for older months
        # — see ``merge_observed_with_roni``), so it's the single source
        # of truth for observed rONI.
        if obs_df is not None and not obs_df.empty and "roni_anom" in obs_df.columns:
            roni_obs = obs_df.dropna(subset=["roni_anom"]).sort_values("date")
            for _, r in roni_obs.iterrows():
                period = str(pd.Timestamp(r["date"]).to_period("M"))
                rows.append({
                    "date": pd.Timestamp(r["date"]),
                    "year": int(r["year"]),
                    "month": int(r["month"]),
                    out_col: r["roni_anom"],
                    "is_forecast": False,
                })
                obs_dates.add(period)

    # Forecast portion: multi-model weighted median (deduplicated)
    if not forecast_df.empty:
        try:
            forecast_only = _get_forecast_only(forecast_df)
            mega = _build_mega_df(forecast_only)
            members = mega[mega["member_id"] != "mean"].copy()
            if not members.empty and fc_value_col in members.columns:
                members = members.dropna(subset=[fc_value_col])
                # Only include months with ≥3 models reporting
                models_per_month = members.groupby("target_month")["model"].nunique()
                valid_months = models_per_month[models_per_month >= 3].index
                members = members[members["target_month"].isin(valid_months)]

                mm = _multimodel_weighted_median(members, value_col=fc_value_col)
                mm["date"] = mm["target_month"].apply(_target_month_to_date)
                mm = mm.sort_values("date")

                for _, r in mm.iterrows():
                    period = str(r["date"].to_period("M"))
                    if period not in obs_dates:
                        rows.append({
                            "date": r["date"],
                            "year": r["date"].year,
                            "month": r["date"].month,
                            out_col: r[fc_value_col],
                            "is_forecast": True,
                        })
        except Exception as e:
            logger.warning(f"Error building forecast portion of ENSO combined: {e}")

    if not rows:
        return pd.DataFrame(columns=["date", "year", "month", out_col, "is_forecast"])

    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    # Interpolate gaps between observed and forecast (e.g. Feb 2026
    # between observed Jan and forecast Mar).  Only fill gaps where
    # one side is observed and the other is forecast — this avoids
    # spurious interpolation past the end of the forecast period.
    filled = []
    for i in range(len(result)):
        filled.append(result.iloc[i].to_dict())
        if i < len(result) - 1:
            cur = result.iloc[i]
            nxt = result.iloc[i + 1]
            # Only interpolate across the observed→forecast boundary
            if not (cur["is_forecast"] == False and nxt["is_forecast"] == True):
                continue
            cur_period = cur["date"].to_period("M")
            nxt_period = nxt["date"].to_period("M")
            gap = (nxt_period - cur_period).n  # months between
            if gap > 1:
                for g in range(1, gap):
                    mid_period = cur_period + g
                    frac = g / gap
                    filled.append({
                        "date": mid_period.to_timestamp(),
                        "year": mid_period.year,
                        "month": mid_period.month,
                        out_col: cur[out_col] + frac * (nxt[out_col] - cur[out_col]),
                        "is_forecast": True,
                    })
    if len(filled) > len(result):
        result = pd.DataFrame(filled)
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Card computation
# ---------------------------------------------------------------------------

def compute_enso_cards(forecast_df, oni_df, obs_df=None, index_mode="oni"):
    """Compute summary card values from forecast, ONI, and observed data.

    Returns dict with keys: current_state, update_date, max_change_str,
    max_change_range.
    """
    cards = {
        "current_state": "N/A",
        "update_date": "N/A",
        "max_change_str": "N/A",
        "max_change_range": "N/A",
        "n_models": 0,
        "n_members": 0,
    }

    fc_col = "roni_anom" if index_mode == "roni" else "nino34_anom"
    obs_col = "roni_anom" if index_mode == "roni" else "nino34_anom"
    label_short = "rONI" if index_mode == "roni" else "Niño 3.4"

    # Current ENSO state from latest observed monthly value
    if obs_df is not None and not obs_df.empty and obs_col in obs_df.columns:
        obs_use = obs_df.dropna(subset=[obs_col]).sort_values("date")
        if not obs_use.empty:
            latest = obs_use.iloc[-1]
            val = latest[obs_col]
            month_label = pd.Timestamp(latest["date"]).strftime("%b %Y")
            if val >= 0.5:
                cards["current_state"] = f"El Ni\u00f1o: {val:+.1f}\u00b0C ({label_short}, {month_label})"
            elif val <= -0.5:
                cards["current_state"] = f"La Ni\u00f1a: {val:.1f}\u00b0C ({label_short}, {month_label})"
            else:
                cards["current_state"] = f"Neutral: {val:+.1f}\u00b0C ({label_short}, {month_label})"
    elif index_mode == "oni" and not oni_df.empty and "oni" in oni_df.columns:
        latest = oni_df.sort_values("date").iloc[-1]
        val = latest["oni"]
        if val >= 0.5:
            cards["current_state"] = f"El Ni\u00f1o: +{val:.1f}\u00b0C"
        elif val <= -0.5:
            cards["current_state"] = f"La Ni\u00f1a: {val:.1f}\u00b0C"
        else:
            cards["current_state"] = f"Neutral: {val:+.1f}\u00b0C"

    # Update date (month-level, since most models publish monthly)
    if not forecast_df.empty and "init_date" in forecast_df.columns:
        max_init = pd.to_datetime(forecast_df["init_date"]).max()
        cards["update_date"] = max_init.strftime("%b %Y")

    # Peak forecast anomaly over full projection period
    if not forecast_df.empty:
        try:
            forecast_only = _get_forecast_only(forecast_df)
            mega = _build_mega_df(forecast_only)
            means = mega[mega["member_id"] == "mean"].copy()
            members = mega[mega["member_id"] != "mean"].copy()

            cards["n_models"] = members["model"].nunique() if not members.empty else 0
            cards["n_members"] = sum(
                members[members["model"] == m]["member_id"].nunique()
                for m in members["model"].unique()
            ) if not members.empty else 0

            if not members.empty and fc_col in members.columns:
                members_used = members.dropna(subset=[fc_col])
                mm = _multimodel_weighted_median(members_used, value_col=fc_col)
                mm["date"] = mm["target_month"].apply(_target_month_to_date)
                mm = mm.sort_values("date")

                if not mm.empty and fc_col in mm.columns:
                    peak_idx = mm[fc_col].abs().idxmax()
                    peak_row = mm.loc[peak_idx]
                    peak_val = peak_row[fc_col]
                    peak_date = peak_row["date"]
                    cards["max_change_str"] = f"{peak_val:+.1f}\u00b0C in {peak_date.strftime('%b %Y')}"
                    # Raw values for hero/KPI binding (avoid re-parsing strings)
                    cards["peak_val"] = float(peak_val)
                    cards["peak_month_label"] = peak_date.strftime('%b %Y')

                    peak_tm = peak_row["target_month"]
                    tm_members = members_used[members_used["target_month"] == peak_tm]
                    if not tm_members.empty:
                        model_counts = tm_members.groupby("model").size()
                        weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
                        vals = tm_members[fc_col].values
                        valid = ~np.isnan(vals)
                        if valid.sum() > 0:
                            q25 = _weighted_quantile(vals[valid], weights[valid], 0.25)
                            q75 = _weighted_quantile(vals[valid], weights[valid], 0.75)
                            cards["max_change_range"] = f"25th\u201375th: {q25:+.1f} to {q75:+.1f}\u00b0C ({label_short})"
        except Exception as e:
            logger.warning(f"Error computing ENSO peak forecast: {e}")

    return cards


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _enso_threshold_shapes(theme, y_range=(-4, 4)):
    """Return shapes + annotations for +-0.5 ENSO thresholds."""
    shapes = [
        # El Nino band
        dict(type="rect", x0=0, x1=1, y0=0.5, y1=y_range[1],
             xref="paper", yref="y",
             fillcolor="red", opacity=0.04, line_width=0),
        # La Nina band
        dict(type="rect", x0=0, x1=1, y0=y_range[0], y1=-0.5,
             xref="paper", yref="y",
             fillcolor="blue", opacity=0.04, line_width=0),
    ]
    # Threshold lines at +/- 0.5
    threshold_traces = [
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color="red", width=1, dash="dash"),
                   showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[None], y=[None], mode="lines",
                   line=dict(color="blue", width=1, dash="dash"),
                   showlegend=False, hoverinfo="skip"),
    ]
    return shapes, threshold_traces


def _multimodel_weighted_median(members, target_months=None, value_col="nino34_anom"):
    """Compute model-weighted median per target month for ``value_col``.

    Each model gets equal weight regardless of ensemble size.
    """
    if target_months is None:
        target_months = sorted(members["target_month"].unique())
    rows = []
    for tm in target_months:
        tm_members = members[members["target_month"] == tm]
        if tm_members.empty:
            continue
        model_counts = tm_members.groupby("model").size()
        weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
        vals = tm_members[value_col].values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue
        median_val = _weighted_quantile(vals[valid], weights[valid], 0.50)
        rows.append({"target_month": tm, value_col: median_val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1: Mega Plume
# ---------------------------------------------------------------------------

def create_enso_mega_plume(forecast_df, obs_df, dark_mode=False, index_mode="oni"):
    """ENSO combined forecast plume — all models, deduplicated.

    ``index_mode``: "oni" (default) plots ``nino34_anom``; "roni" plots
    ``roni_anom`` (Niño 3.4 minus 20S-20N tropical-mean SSTA).
    """
    theme = get_theme(dark_mode)
    meta = _index_meta(index_mode)
    fig = go.Figure()

    # For rONI, swap the source column into nino34_anom so downstream plot
    # logic stays unchanged.
    forecast_df = _swap_to_nino34(forecast_df, meta["col"])
    obs_df = _swap_to_nino34(obs_df, meta["col"])

    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    means = mega[mega["member_id"] == "mean"].copy()

    if members.empty and means.empty:
        fig.update_layout(title="No forecast data available")
        return fig

    # Limit to target months with at least 3 distinct models reporting
    if not means.empty:
        models_per_month = means.groupby("target_month")["model"].nunique()
        valid_months = models_per_month[models_per_month >= 3].index
        means = means[means["target_month"].isin(valid_months)].copy()
        members = members[members["target_month"].isin(valid_months)].copy()

    # -- Ensemble members as thin lines per model (None-separated for efficiency) --
    if not members.empty:
        members["date"] = members["target_month"].apply(_target_month_to_date)
        for model in sorted(members["model"].unique()):
            color = MEGA_COLORS.get(model, "#999999")
            model_members = members[members["model"] == model]
            xs, ys = [], []
            for _, mdf in model_members.groupby("member_id"):
                mdf = mdf.sort_values("date")
                xs.extend(mdf["date"].tolist() + [None])
                ys.extend(mdf["nino34_anom"].tolist() + [None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=0.8),
                opacity=0.2,
                showlegend=False,
                hoverinfo="skip",
            ))

    # -- Per-model means --
    if not means.empty:
        means["date"] = means["target_month"].apply(_target_month_to_date)
        for model in sorted(means["model"].unique()):
            color = MEGA_COLORS.get(model, "#999999")
            mdf = means[means["model"] == model].sort_values("date")
            n_mem = members[members["model"] == model]["member_id"].nunique() if not members.empty else 0
            fig.add_trace(go.Scatter(
                x=mdf["date"], y=mdf["nino34_anom"],
                mode="lines",
                name=f"{model} ({n_mem})",
                line=dict(color=color, width=2),
                opacity=0.85,
                hovertemplate=f"{model}<br>%{{x|%b %Y}}: %{{y:.2f}}\u00b0C<extra></extra>",
            ))

        # Multi-model weighted median — substitute observed values where available
        mm = _multimodel_weighted_median(members)
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")
        if obs_df is not None and not obs_df.empty:
            obs_lookup = obs_df.set_index(
                pd.to_datetime(obs_df["date"]).dt.to_period("M").astype(str)
            )["nino34_anom"]
            for idx, row in mm.iterrows():
                if row["target_month"] in obs_lookup.index:
                    mm.loc[idx, "nino34_anom"] = obs_lookup[row["target_month"]]
        mm_color = "white" if dark_mode else "black"
        fig.add_trace(go.Scatter(
            x=mm["date"], y=mm["nino34_anom"],
            mode="lines",
            name="Multi-model median",
            line=dict(color=mm_color, width=3, dash="dot"),
            hovertemplate="Multi-model median<br>%{x|%b %Y}: %{y:.2f}\u00b0C<extra></extra>",
        ))

    # -- Observed (start from Dec 2025) --
    if obs_df is not None and not obs_df.empty:
        obs = obs_df.copy()
        obs["date"] = pd.to_datetime(obs["date"])
        obs = obs[obs["date"] >= "2025-12-01"]
        obs_color = "white" if dark_mode else "black"
        if not obs.empty:
            fig.add_trace(go.Scatter(
                x=obs["date"], y=obs["nino34_anom"],
                mode="lines",
                name="Observed Nino3.4",
                line=dict(color=obs_color, width=2.5),
                hovertemplate="Observed<br>%{x|%b %Y}: %{y:.2f}\u00b0C<extra></extra>",
            ))
            # Dashed connector from last observation to first forecast-only median
            if not mm.empty:
                last_obs_date = obs["date"].max()
                forecast_only = mm[mm["date"] > last_obs_date].sort_values("date")
                if not forecast_only.empty:
                    first_fcst = forecast_only.iloc[0]
                    last_obs_row = obs.loc[obs["date"].idxmax()]
                    fig.add_trace(go.Scatter(
                        x=[last_obs_date, first_fcst["date"]],
                        y=[last_obs_row["nino34_anom"], first_fcst["nino34_anom"]],
                        mode="lines",
                        line=dict(color=obs_color, width=3, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

    # -- ENSO threshold lines (labeled so the dashes are self-explanatory) --
    fig.add_hline(y=0.5, line=dict(color="red", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=-0.5, line=dict(color="blue", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5), opacity=0.4)
    fig.add_annotation(
        x=0.01, xref="paper", y=0.5, yanchor="bottom", yshift=2,
        text="El Niño ≥ +0.5°C", showarrow=False,
        font=dict(size=10.5, color="rgba(255,107,107,0.85)"),
        xanchor="left",
    )
    fig.add_annotation(
        x=0.01, xref="paper", y=-0.5, yanchor="top", yshift=-2,
        text="La Niña ≤ −0.5°C", showarrow=False,
        font=dict(size=10.5, color="rgba(84,160,255,0.85)"),
        xanchor="left",
    )

    # Counts for title
    n_models = means["model"].nunique() if not means.empty else 0
    n_total = sum(
        members[members["model"] == m]["member_id"].nunique()
        for m in members["model"].unique()
    ) if not members.empty else 0

    # Y-axis range from data
    all_vals = members["nino34_anom"].dropna().tolist() if not members.empty else []
    if not means.empty:
        all_vals.extend(means["nino34_anom"].dropna().tolist())
    if obs_df is not None and not obs_df.empty:
        all_vals.extend(obs_df["nino34_anom"].dropna().tolist())
    ymin = min(-3.0, min(all_vals) - 0.3) if all_vals else -3.0
    ymax = max(3.0, max(all_vals) + 0.3) if all_vals else 3.0

    # ENSO background shading (use computed ymin/ymax so autorange stays bounded)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=ymax,
                  xref="paper", yref="y",
                  fillcolor="red", opacity=0.04, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=ymin, y1=-0.5,
                  xref="paper", yref="y",
                  fillcolor="blue", opacity=0.04, line_width=0)

    fig.update_layout(
        yaxis_title=meta["y_label"],
        template=theme["template"],
        height=550,
        uirevision="enso-mega-plume",
        # Model roster sits outside the plot so it never covers the plume
        legend=dict(
            orientation="v",
            yanchor="top", y=1, xanchor="left", x=1.01,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10.5),
        ),
        margin=dict(l=60, r=30, t=30, b=50),
        xaxis=dict(range=["2025-12-01", None]),
        yaxis=dict(range=[ymin, ymax]),
    )

    return fig


# ---------------------------------------------------------------------------
# Plot 2: Box Distribution
# ---------------------------------------------------------------------------

def create_enso_box_distribution(forecast_df, dark_mode=False, index_mode="oni"):
    """Forecast distribution with ensemble member dots and model-weighted box plots."""
    theme = get_theme(dark_mode)
    meta = _index_meta(index_mode)
    fig = go.Figure()

    forecast_df = _swap_to_nino34(forecast_df, meta["col"])
    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    if members.empty:
        fig.update_layout(title="No member data available")
        return fig

    # Limit to target months with at least 3 distinct models reporting
    models_per_month = members.groupby("target_month")["model"].nunique()
    valid_months = models_per_month[models_per_month >= 3].index
    members = members[members["target_month"].isin(valid_months)].copy()

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
    models = sorted(members["model"].unique())

    # -- Weighted box plots via shapes --
    box_width = 0.5
    for i, tm in enumerate(target_months):
        tm_members = members[members["target_month"] == tm]
        if tm_members.empty:
            continue

        model_counts = tm_members.groupby("model").size()
        weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
        vals = tm_members["nino34_anom"].values
        valid = ~np.isnan(vals)
        vals = vals[valid]
        weights = weights[valid]
        if len(vals) == 0:
            continue

        q25 = _weighted_quantile(vals, weights, 0.25)
        q50 = _weighted_quantile(vals, weights, 0.50)
        q75 = _weighted_quantile(vals, weights, 0.75)
        iqr = q75 - q25
        whisker_lo = max(vals.min(), q25 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q75 + 1.5 * iqr)

        # Box color based on median
        if q50 > 0.5:
            box_color = "rgba(255, 100, 100, 0.4)" if dark_mode else "rgba(255, 204, 204, 0.55)"
        elif q50 < -0.5:
            box_color = "rgba(100, 100, 255, 0.4)" if dark_mode else "rgba(204, 204, 255, 0.55)"
        else:
            box_color = "rgba(180, 180, 180, 0.4)" if dark_mode else "rgba(224, 224, 224, 0.55)"

        edge_color = theme["text_color"]

        # IQR box
        fig.add_shape(type="rect",
                      x0=i - box_width / 2, x1=i + box_width / 2,
                      y0=q25, y1=q75,
                      fillcolor=box_color,
                      line=dict(color=edge_color, width=1.2))

        # Median line
        fig.add_shape(type="line",
                      x0=i - box_width / 2, x1=i + box_width / 2,
                      y0=q50, y1=q50,
                      line=dict(color=edge_color, width=2))

        # Whiskers
        fig.add_shape(type="line", x0=i, x1=i, y0=whisker_lo, y1=q25,
                      line=dict(color=edge_color, width=1.2))
        fig.add_shape(type="line", x0=i, x1=i, y0=q75, y1=whisker_hi,
                      line=dict(color=edge_color, width=1.2))

        # Caps
        cap_w = box_width * 0.3
        fig.add_shape(type="line", x0=i - cap_w, x1=i + cap_w,
                      y0=whisker_lo, y1=whisker_lo,
                      line=dict(color=edge_color, width=1.2))
        fig.add_shape(type="line", x0=i - cap_w, x1=i + cap_w,
                      y0=whisker_hi, y1=whisker_hi,
                      line=dict(color=edge_color, width=1.2))

    # ENSO threshold lines
    fig.add_hline(y=0.5, line=dict(color="red", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=-0.5, line=dict(color="blue", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5), opacity=0.4)
    fig.add_annotation(
        x=0.01, xref="paper", y=0.5, yanchor="bottom", yshift=2,
        text="El Niño ≥ +0.5°C", showarrow=False, xanchor="left",
        font=dict(size=10.5, color="rgba(255,107,107,0.85)"),
    )
    fig.add_annotation(
        x=0.01, xref="paper", y=-0.5, yanchor="top", yshift=-2,
        text="La Niña ≤ −0.5°C", showarrow=False, xanchor="left",
        font=dict(size=10.5, color="rgba(84,160,255,0.85)"),
    )

    tick_labels = [pd.Timestamp(tm + "-01").strftime("%b\n%Y") for tm in target_months]

    # Y-axis range
    all_vals = members["nino34_anom"].dropna().tolist()
    ymin = min(-3.0, min(all_vals) - 0.3) if all_vals else -3.0
    ymax = max(3.0, max(all_vals) + 0.3) if all_vals else 3.0

    # ENSO background shading (use computed ymin/ymax so autorange stays bounded)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=ymax,
                  xref="paper", yref="y",
                  fillcolor="red", opacity=0.04, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=ymin, y1=-0.5,
                  xref="paper", yref="y",
                  fillcolor="blue", opacity=0.04, line_width=0)

    fig.update_layout(
        yaxis_title=meta["y_label"],
        xaxis=dict(
            tickvals=list(range(len(target_months))),
            ticktext=tick_labels,
        ),
        yaxis=dict(range=[ymin, ymax]),
        template=theme["template"],
        height=550,
        showlegend=False,
        margin=dict(l=60, r=30, t=30, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Historical Context
# ---------------------------------------------------------------------------

def create_enso_historical_context(forecast_df, dark_mode=False, index_mode="oni"):
    """Observed Niño 3.4 (or rONI) since 1990 with threshold fills + forecast overlay."""
    theme = get_theme(dark_mode)
    meta = _index_meta(index_mode)
    fig = go.Figure()

    obs_full = load_full_observed(start_year=1990)
    obs_full = _swap_to_nino34(obs_full, meta["col"])
    forecast_df = _swap_to_nino34(forecast_df, meta["col"])
    if obs_full.empty:
        fig.update_layout(title="No observed data available")
        return fig

    # Drop any leading rows where the active index has no data (early years
    # of the rONI seasonal series may not cover the full nino34_monthly span).
    obs_full = obs_full.dropna(subset=["nino34_anom"]).reset_index(drop=True)
    if obs_full.empty:
        fig.update_layout(title=f"No observed {meta['short']} data available")
        return fig

    dates = obs_full["date"]
    anom = obs_full["nino34_anom"].values

    # -- Observed line --
    obs_color = "white" if dark_mode else "black"
    fig.add_trace(go.Scatter(
        x=dates, y=anom,
        mode="lines",
        name="Observed Nino3.4",
        line=dict(color=obs_color, width=1.2),
        hovertemplate="Observed<br>%{x|%b %Y}: %{y:.2f}\u00b0C<extra></extra>",
    ))

    # -- El Nino / La Nina fills (segment-based to avoid plotly fill artifacts) --
    # We create separate fill traces for segments above +0.5 and below -0.5
    _add_threshold_fills(fig, dates.values, anom, 0.5, "#d62728", "El Ni\u00f1o (> +0.5\u00b0C)", dark_mode)
    _add_threshold_fills(fig, dates.values, anom, -0.5, "#1f77b4", "La Ni\u00f1a (< \u20130.5\u00b0C)", dark_mode, below=True)

    # Threshold lines
    fig.add_hline(y=0.5, line=dict(color="#d62728", width=0.8, dash="dash"), opacity=0.4)
    fig.add_hline(y=-0.5, line=dict(color="#1f77b4", width=0.8, dash="dash"), opacity=0.4)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5), opacity=0.4)
    fig.add_annotation(
        x=0.01, xref="paper", y=0.5, yanchor="bottom", yshift=2,
        text="El Niño ≥ +0.5°C", showarrow=False, xanchor="left",
        font=dict(size=10.5, color="rgba(214,39,40,0.8)"),
    )
    fig.add_annotation(
        x=0.01, xref="paper", y=-0.5, yanchor="top", yshift=-2,
        text="La Niña ≤ −0.5°C", showarrow=False, xanchor="left",
        font=dict(size=10.5, color="rgba(31,119,180,0.9)"),
    )

    # -- Forecast overlay --
    if not forecast_df.empty:
        try:
            forecast_only = _get_forecast_only(forecast_df)
            mega = _build_mega_df(forecast_only)
            means = mega[mega["member_id"] == "mean"].copy()
            fc_members = mega[mega["member_id"] != "mean"].copy()

            # Limit to months with at least 3 distinct models reporting
            if not means.empty:
                models_per_month = means.groupby("target_month")["model"].nunique()
                valid_months = models_per_month[models_per_month >= 3].index
                means = means[means["target_month"].isin(valid_months)].copy()
                fc_members = fc_members[fc_members["target_month"].isin(valid_months)].copy()

            if not means.empty:
                mm = _multimodel_weighted_median(fc_members)
                mm["date"] = mm["target_month"].apply(_target_month_to_date)
                mm = mm.sort_values("date")

                # Substitute observed values where available
                obs_lookup = obs_full.set_index(
                    obs_full["date"].dt.to_period("M").astype(str)
                )["nino34_anom"]
                for idx, row in mm.iterrows():
                    if row["target_month"] in obs_lookup.index:
                        mm.loc[idx, "nino34_anom"] = obs_lookup[row["target_month"]]

                # Compute weighted 25-75th percentiles
                target_months_sorted = sorted(mm["target_month"].unique())
                q25_vals, q75_vals = [], []
                for tm in target_months_sorted:
                    # Use observed value (no spread) for months with observations
                    if tm in obs_lookup.index:
                        q25_vals.append(obs_lookup[tm])
                        q75_vals.append(obs_lookup[tm])
                        continue
                    tm_members = fc_members[fc_members["target_month"] == tm]
                    if tm_members.empty:
                        q25_vals.append(np.nan)
                        q75_vals.append(np.nan)
                        continue
                    model_counts = tm_members.groupby("model").size()
                    weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
                    vals = tm_members["nino34_anom"].values
                    valid_mask = ~np.isnan(vals)
                    if valid_mask.sum() == 0:
                        q25_vals.append(np.nan)
                        q75_vals.append(np.nan)
                        continue
                    q25_vals.append(_weighted_quantile(vals[valid_mask], weights[valid_mask], 0.25))
                    q75_vals.append(_weighted_quantile(vals[valid_mask], weights[valid_mask], 0.75))

                # Connect forecast to last observed point
                last_obs_date = obs_full["date"].iloc[-1]
                last_obs_val = obs_full["nino34_anom"].iloc[-1]

                fc_dates = [last_obs_date] + mm["date"].tolist()
                forecast_mean = [last_obs_val] + mm["nino34_anom"].tolist()
                forecast_q25 = [last_obs_val] + q25_vals
                forecast_q75 = [last_obs_val] + q75_vals

                # 25-75th percentile shading
                fig.add_trace(go.Scatter(
                    x=fc_dates + fc_dates[::-1],
                    y=forecast_q75 + forecast_q25[::-1],
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.25)",
                    line=dict(width=0),
                    name="25th\u201375th percentile",
                    hoverinfo="skip",
                ))

                # Multi-model mean
                mm_line_color = "white" if dark_mode else "black"
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=forecast_mean,
                    mode="lines",
                    name="Multi-model median forecast",
                    line=dict(color=mm_line_color, width=2.2, dash="dot"),
                    hovertemplate="Forecast<br>%{x|%b %Y}: %{y:.2f}\u00b0C<extra></extra>",
                ))

                # Forecast period background
                forecast_start = mm["date"].min()
                forecast_end = mm["date"].max()
                fig.add_vrect(
                    x0=forecast_start, x1=forecast_end,
                    fillcolor="gray", opacity=0.07,
                    line_width=0,
                )
                mid_forecast = forecast_start + (forecast_end - forecast_start) / 2
                fig.add_annotation(
                    x=mid_forecast, y=1, yref="paper",
                    text="Forecast", showarrow=False,
                    font=dict(size=12, color="gray"),
                    yanchor="top",
                )
                # Give the forecast fan breathing room at the right edge
                fig.update_xaxes(range=[
                    obs_full["date"].min(),
                    forecast_end + pd.DateOffset(months=4),
                ])
        except Exception as e:
            logger.warning(f"Error adding forecast overlay: {e}")

    # Y-axis range: pad the top to 3.5 °C so the current forecast median
    # has headroom, but keep tick labels stopping at 3 so the scale still
    # reads as a standard ONI-style chart.
    all_vals = list(anom)
    ymin = min(-3.0, min(all_vals) - 0.3)
    ymax = max(3.5, max(all_vals) + 0.3)

    fig.update_layout(
        yaxis_title=meta["y_label"],
        template=theme["template"],
        height=450,
        showlegend=False,
        yaxis=dict(
            range=[ymin, ymax],
            tickmode="array",
            tickvals=list(range(int(np.floor(ymin)), 4)),
        ),
        margin=dict(l=60, r=30, t=30, b=50),
    )

    return fig


def _add_threshold_fills(fig, dates, anom, threshold, color, name, dark_mode, below=False):
    """Add filled regions where anom crosses a threshold.

    Uses segment-based approach: for each contiguous run where the condition
    holds, we add a fill trace between the line and the threshold.
    """
    if below:
        mask = anom < threshold
    else:
        mask = anom > threshold

    # Find contiguous segments
    segments = []
    in_seg = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_seg:
            start = i
            in_seg = True
        elif not mask[i] and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask)))

    first = True
    for s, e in segments:
        # Expand by 1 on each side for interpolation at crossing
        s_ext = max(0, s - 1)
        e_ext = min(len(dates), e + 1)
        seg_dates = dates[s_ext:e_ext]
        seg_anom = anom[s_ext:e_ext]
        seg_thresh = np.full_like(seg_anom, threshold)

        # Clip anomaly to threshold for clean fill
        if below:
            fill_anom = np.minimum(seg_anom, threshold)
        else:
            fill_anom = np.maximum(seg_anom, threshold)

        fig.add_trace(go.Scatter(
            x=np.concatenate([seg_dates, seg_dates[::-1]]),
            y=np.concatenate([fill_anom, seg_thresh[::-1]]),
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)",
            mode="none",
            line=dict(width=0),
            showlegend=first,
            name=name,
            hoverinfo="skip",
        ))
        first = False


# ---------------------------------------------------------------------------
# ENSO Strength Probabilities (NOAA-CPC style stacked bars)
# ---------------------------------------------------------------------------

# (label, lower_inclusive, upper_exclusive, group, range_str, fill_color)
STRENGTH_BINS = [
    ("Very Strong El Niño",  2.0,    np.inf, "el_nino", "index ≥ 2.0°C",           "#7A0000"),
    ("Strong El Niño",       1.5,    2.0,    "el_nino", "1.5°C ≤ index < 2.0°C",   "#E60000"),
    ("Moderate El Niño",     1.0,    1.5,    "el_nino", "1.0°C ≤ index < 1.5°C",   "#F89999"),
    ("Weak El Niño",         0.5,    1.0,    "el_nino", "0.5°C ≤ index < 1.0°C",   "#FCDAD6"),
    ("Neutral",             -0.5,    0.5,    "neutral", "−0.5°C < index < 0.5°C",  "#B5B5B5"),
    ("Weak La Niña",        -1.0,   -0.5,    "la_nina", "−0.5°C ≥ index > −1.0°C", "#DCEAFA"),
    ("Moderate La Niña",    -1.5,   -1.0,    "la_nina", "−1.0°C ≥ index > −1.5°C", "#9CBDDF"),
    ("Strong La Niña",      -2.0,   -1.5,    "la_nina", "−1.5°C ≥ index > −2.0°C", "#3F71B0"),
    ("Very Strong La Niña", -np.inf, -2.0,   "la_nina", "index ≤ −2.0°C",          "#0A1F5C"),
]
STRENGTH_GROUP_EDGE = {"el_nino": "#C00000", "neutral": "#7F7F7F", "la_nina": "#1F4E79"}
# Stack order: lightest at bottom of each group, darkest on top.
EL_NINO_STACK = [3, 2, 1, 0]   # Weak, Moderate, Strong, Very Strong
LA_NINA_STACK = [5, 6, 7, 8]
NEUTRAL_IDX   = 4

SEASON_FOR_CENTER = {
    1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
    7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
}


def _classify_strength(x: float) -> int:
    """Return STRENGTH_BINS index for value x (matches src/enso.classify_enso_state)."""
    if x >= 2.0:  return 0
    if x >= 1.5:  return 1
    if x >= 1.0:  return 2
    if x >= 0.5:  return 3
    if x >  -0.5: return 4
    if x >  -1.0: return 5
    if x >  -1.5: return 6
    if x >  -2.0: return 7
    return 8


def _season_window(center: pd.Timestamp) -> list:
    """Return the three monthly Timestamps composing a 3-month season."""
    return [center - pd.DateOffset(months=1), center, center + pd.DateOffset(months=1)]


def _season_label(center: pd.Timestamp) -> str:
    return f"{SEASON_FOR_CENTER[center.month]} {center.year}"


def _build_member_pivot(mem: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot members → rows = (model, member_id), cols = month timestamp."""
    df = mem.copy()
    df["ym"] = pd.to_datetime(df["target_month"] + "-01")
    return df.pivot_table(
        index=["model", "member_id"],
        columns="ym",
        values=value_col,
        aggfunc="first",
    )


def _seasonal_means(pivot: pd.DataFrame, centers, obs_monthly: dict) -> pd.DataFrame:
    """For each (model, member, season center), compute the 3-month mean.

    Forecast values take precedence; missing months fall back to obs_monthly
    (keyed by pd.Timestamp). Cells where any month lacks both forecast and
    obs come back as NaN and are dropped from the probability calculation.
    """
    out = pd.DataFrame(index=pivot.index)
    for center in centers:
        cols = []
        for m in _season_window(center):
            col = pivot[m].copy() if m in pivot.columns else pd.Series(np.nan, index=pivot.index)
            obs_v = obs_monthly.get(m)
            if obs_v is not None and not pd.isna(obs_v):
                col = col.fillna(obs_v)
            cols.append(col)
        out[center] = pd.concat(cols, axis=1).mean(axis=1, skipna=False)
    return out


def _model_weighted_probs(season_col: pd.Series) -> np.ndarray:
    """Return percent (length-9 array) per strength category, model-weighted.

    Each model contributes total weight 1.0 regardless of ensemble size.
    """
    valid = season_col.dropna()
    if valid.empty:
        return np.zeros(len(STRENGTH_BINS))
    df = valid.reset_index()
    df.columns = ["model", "member_id", "value"]
    counts = df.groupby("model").size()
    df["weight"] = 1.0 / df["model"].map(counts)
    df["cat"] = df["value"].map(_classify_strength)
    total = df["weight"].sum()
    by_cat = df.groupby("cat")["weight"].sum() / total * 100.0
    return np.array([by_cat.get(i, 0.0) for i in range(len(STRENGTH_BINS))])


def compute_strength_probabilities(forecast_df, dark_mode_unused=None,
                                    index_mode="oni", n_seasons=9):
    """Return (centers, probs, n_models_per_season) for the strength-probs plot.

    Centers are 9 consecutive 3-month season midpoints starting at the latest
    init_date's month. probs has shape (n_seasons, 9) — one row per season,
    one column per STRENGTH_BINS entry. Members come from the canonical
    deduplicated mega plume; CFSv2's control member 41 is excluded.
    """
    if forecast_df is None or forecast_df.empty:
        return [], np.zeros((0, len(STRENGTH_BINS))), []

    meta = _index_meta(index_mode)
    value_col = meta["col"]

    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)
    mem = mega[mega["member_id"] != "mean"].copy()
    # CFSv2 ensemble window uses the latest 40 E3 runs; member 41 is the
    # stable control slot and is excluded from weighted statistics.
    mem = mem[~((mem["model"] == "CFSv2") & (mem["member_id"] == "ens_E3_041"))]
    mem = mem.dropna(subset=[value_col])
    if mem.empty:
        return [], np.zeros((0, len(STRENGTH_BINS))), []

    latest_init = pd.to_datetime(mem["init_date"]).max()
    start_center = latest_init.replace(day=1)
    centers = [start_center + pd.DateOffset(months=k) for k in range(n_seasons)]

    obs_full = load_full_observed(start_year=2000)
    if value_col in obs_full.columns:
        obs_lookup = obs_full.set_index("date")[value_col].dropna().to_dict()
    else:
        obs_lookup = {}

    pivot = _build_member_pivot(mem, value_col)
    sm = _seasonal_means(pivot, centers, obs_lookup)
    probs = np.vstack([_model_weighted_probs(sm[c]) for c in centers])
    n_models = [int(sm[c].dropna().reset_index()["model"].nunique()) for c in centers]
    return centers, probs, n_models


def create_enso_strength_probs(forecast_df, dark_mode=False, index_mode="oni"):
    """NOAA-CPC-style ENSO strength-probability bars from the multi-model plume.

    Three offset bars per season (La Niña / Neutral / El Niño), each
    internally stacked from lightest (Weak) at the bottom to darkest
    (Very Strong) on top.
    """
    theme = get_theme(dark_mode)
    meta = _index_meta(index_mode)

    fig = go.Figure()

    centers, probs, n_models = compute_strength_probabilities(
        forecast_df, index_mode=index_mode
    )
    if not centers:
        fig.update_layout(title=f"No {meta['short']} forecast data available")
        return fig

    labels = [_season_label(c) for c in centers]
    n_seasons = len(centers)

    # Build per-bin traces in stack order (lightest first → bottom).
    grid_color = theme.get("grid_color", "rgba(128,128,128,0.25)")
    text_color = theme.get("text_color", "#222")
    seen_legendgroups = set()

    # Legend order mirrors NOAA-CPC: Very Strong El Niño at top → through
    # weaker El Niño → Neutral → weakest La Niña → strongest La Niña at bottom.
    LEGEND_RANK = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}

    def _add_bin_trace(bin_idx: int, group_key: str):
        label, _, _, _, rng, fc = STRENGTH_BINS[bin_idx]
        edge = STRENGTH_GROUP_EDGE[group_key]
        y = probs[:, bin_idx]
        # Skip categories that never occur — keeps the legend to what's
        # actually on the chart.
        if np.nanmax(y) < 0.5:
            return
        # Two-line legend label (NOAA style): name + threshold range
        legend_name = f"{label}<br><span style='font-size:0.85em'>{rng}</span>"
        fig.add_trace(go.Bar(
            x=labels,
            y=y,
            name=legend_name,
            marker=dict(color=fc, line=dict(color=edge, width=0.9)),
            offsetgroup=group_key,
            legendgroup=group_key,
            legendrank=LEGEND_RANK[bin_idx],
            # Label big segments directly so shades don't have to be decoded;
            # never render a bare "100%" — these are ensemble odds.
            text=[(">99%" if v >= 99.5 else f"{v:.0f}%") if v >= 12 else ""
                  for v in y],
            textposition="inside",
            textfont=dict(size=10.5),
            insidetextanchor="middle",
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"{rng}<br>"
                "%{x}: %{y:.1f}%<extra></extra>"
            ),
            showlegend=True,
        ))

    # Order matters for stacking: bottom→top per group. Keep groups in the
    # order la_nina → neutral → el_nino so legend reads cold→warm; Plotly
    # arranges the bars left→right by trace order regardless.
    for idx in LA_NINA_STACK:
        _add_bin_trace(idx, "la_nina")
    _add_bin_trace(NEUTRAL_IDX, "neutral")
    for idx in EL_NINO_STACK:
        _add_bin_trace(idx, "el_nino")

    # n=N annotations under each season tick
    n_annotations = [
        dict(x=lbl, y=-0.09, xref="x", yref="paper",
             text=f"n={n}", showarrow=False,
             font=dict(size=10, color=theme.get("text_color", "#666")))
        for lbl, n in zip(labels, n_models)
    ]

    fig.update_layout(
        barmode="stack",
        bargap=0.2,
        bargroupgap=0.05,
        xaxis=dict(
            title="",
            tickangle=0,
            categoryorder="array",
            categoryarray=labels,
        ),
        yaxis=dict(
            title="Percent chance (%)",
            range=[0, 100],
            dtick=20,
        ),
        legend=dict(
            orientation="v",
            yanchor="top", y=1, xanchor="left", x=1.01,
            font=dict(size=10.5),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            tracegroupgap=4,
            traceorder="normal",
        ),
        annotations=n_annotations,
        margin=dict(l=60, r=30, t=30, b=70),
        template=theme.get("template", "plotly_white"),
    )
    return fig


# ---------------------------------------------------------------------------
# Static image generation
# ---------------------------------------------------------------------------

def generate_enso_static_images(forecast_df, obs_df, assets_dir):
    """Render 16 static PNGs: 4 plots × 2 themes × 2 indices (ONI / rONI)."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # (name, figure factory, height, width) — box distribution and strength
    # probs render side by side in the layout, so they export squarer.
    plot_configs = [
        ("enso_mega_plume",
         lambda dm, idx: create_enso_mega_plume(forecast_df, obs_df, dm, index_mode=idx),
         550, 1200),
        ("enso_box_distribution",
         lambda dm, idx: create_enso_box_distribution(forecast_df, dm, index_mode=idx),
         550, 780),
        ("enso_historical",
         lambda dm, idx: create_enso_historical_context(forecast_df, dm, index_mode=idx),
         450, 1200),
        ("enso_strength_probs",
         lambda dm, idx: create_enso_strength_probs(forecast_df, dm, index_mode=idx),
         550, 780),
    ]

    for index_mode in INDEX_MODES:
        for dark_mode in [True, False]:
            mode = "dark" if dark_mode else "light"
            for name, create_func, height, width in plot_configs:
                # Backwards-compat filename for ONI: keep `{name}_{mode}.png`
                # so the dashboard doesn't need to special-case the legacy
                # default. rONI gets a `_roni_{mode}.png` suffix.
                if index_mode == "oni":
                    filename = f"{name}_{mode}.png"
                else:
                    filename = f"{name}_roni_{mode}.png"
                filepath = assets_dir / filename
                try:
                    logger.info(f"Generating {filename}...")
                    fig = create_func(dark_mode, index_mode)
                    fig.write_image(str(filepath), width=width, height=height, scale=2)
                except Exception as e:
                    logger.error(f"Failed to generate {filename}: {e}")
