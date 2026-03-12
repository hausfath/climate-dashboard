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
        forecast_df = load_all_forecasts(sources=["CFS", "NMME", "C3S"])
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
    """Load the full observed Nino3.4 record from disk."""
    obs_path = OBSERVED_DIR / "nino34_monthly.csv"
    if not obs_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(obs_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= f"{start_year}-01-01"].sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Card computation
# ---------------------------------------------------------------------------

def compute_enso_cards(forecast_df, oni_df):
    """Compute summary card values from forecast and ONI data.

    Returns dict with keys: current_state, update_date, max_change_str,
    max_change_range.
    """
    cards = {
        "current_state": "N/A",
        "update_date": "N/A",
        "max_change_str": "N/A",
        "max_change_range": "N/A",
    }

    # Current ENSO state from latest ONI
    if not oni_df.empty and "oni" in oni_df.columns:
        latest = oni_df.sort_values("date").iloc[-1]
        val = latest["oni"]
        if val >= 0.5:
            cards["current_state"] = f"El Nino: +{val:.1f}\u00b0C"
        elif val <= -0.5:
            cards["current_state"] = f"La Nina: {val:.1f}\u00b0C"
        else:
            cards["current_state"] = f"Neutral: {val:+.1f}\u00b0C"

    # Update date
    if not forecast_df.empty and "init_date" in forecast_df.columns:
        max_init = pd.to_datetime(forecast_df["init_date"]).max()
        cards["update_date"] = max_init.strftime("%b %d, %Y")

    # Peak forecast anomaly over full projection period
    if not forecast_df.empty:
        try:
            forecast_only = _get_forecast_only(forecast_df)
            mega = _build_mega_df(forecast_only)
            means = mega[mega["member_id"] == "mean"].copy()
            members = mega[mega["member_id"] != "mean"].copy()

            if not means.empty:
                mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
                mm["date"] = mm["target_month"].apply(_target_month_to_date)
                mm = mm.sort_values("date")

                if not mm.empty:
                    # Find month with largest absolute anomaly
                    peak_idx = mm["nino34_anom"].abs().idxmax()
                    peak_row = mm.loc[peak_idx]
                    peak_val = peak_row["nino34_anom"]
                    peak_date = peak_row["date"]
                    cards["max_change_str"] = f"{peak_val:+.1f}\u00b0C in {peak_date.strftime('%b %Y')}"

                    # 25-75th percentile for that month
                    peak_tm = peak_row["target_month"]
                    tm_members = members[members["target_month"] == peak_tm]
                    if not tm_members.empty:
                        model_counts = tm_members.groupby("model").size()
                        weights = tm_members["model"].map(lambda m: 1.0 / model_counts[m]).values
                        vals = tm_members["nino34_anom"].values
                        valid = ~np.isnan(vals)
                        if valid.sum() > 0:
                            q25 = _weighted_quantile(vals[valid], weights[valid], 0.25)
                            q75 = _weighted_quantile(vals[valid], weights[valid], 0.75)
                            cards["max_change_range"] = f"25th\u201375th: {q25:+.1f} to {q75:+.1f}\u00b0C"
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


# ---------------------------------------------------------------------------
# Plot 1: Mega Plume
# ---------------------------------------------------------------------------

def create_enso_mega_plume(forecast_df, obs_df, dark_mode=False):
    """ENSO Nino3.4 combined forecast plume — all models, deduplicated."""
    theme = get_theme(dark_mode)
    fig = go.Figure()

    # Filter and deduplicate
    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    means = mega[mega["member_id"] == "mean"].copy()

    if members.empty and means.empty:
        fig.update_layout(title="No forecast data available")
        return fig

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

        # Multi-model mean
        mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
        mm["date"] = mm["target_month"].apply(_target_month_to_date)
        mm = mm.sort_values("date")
        mm_color = "white" if dark_mode else "black"
        fig.add_trace(go.Scatter(
            x=mm["date"], y=mm["nino34_anom"],
            mode="lines",
            name="Multi-model mean",
            line=dict(color=mm_color, width=3, dash="dash"),
            hovertemplate="Multi-model mean<br>%{x|%b %Y}: %{y:.2f}\u00b0C<extra></extra>",
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

    # -- ENSO threshold lines --
    fig.add_hline(y=0.5, line=dict(color="red", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=-0.5, line=dict(color="blue", width=1, dash="dash"), opacity=0.5)
    fig.add_hline(y=0, line=dict(color="gray", width=0.5), opacity=0.4)

    # ENSO background shading (use 100 so it always fills to axis edge)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=100,
                  xref="paper", yref="y",
                  fillcolor="red", opacity=0.04, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=-100, y1=-0.5,
                  xref="paper", yref="y",
                  fillcolor="blue", opacity=0.04, line_width=0)

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

    fig.update_layout(
        title=f"ENSO Nino3.4 Combined Forecast Plume ({n_models} models, {n_total} members)",
        yaxis_title="Nino3.4 SST Anomaly (\u00b0C)",
        template=theme["template"],
        paper_bgcolor=theme["paper_color"],
        plot_bgcolor=theme["bg_color"],
        font=dict(color=theme["text_color"]),
        height=550,
        uirevision="enso-mega-plume",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.3)" if dark_mode else "rgba(255,255,255,0.9)",
            font=dict(size=10),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(gridcolor=theme["grid_color"],
                   range=["2025-12-01", None]),
        yaxis=dict(range=[ymin, ymax], gridcolor=theme["grid_color"]),
    )

    return fig


# ---------------------------------------------------------------------------
# Plot 2: Box Distribution
# ---------------------------------------------------------------------------

def create_enso_box_distribution(forecast_df, dark_mode=False):
    """Forecast distribution with ensemble member dots and model-weighted box plots."""
    theme = get_theme(dark_mode)
    fig = go.Figure()

    forecast_only = _get_forecast_only(forecast_df)
    mega = _build_mega_df(forecast_only)

    members = mega[mega["member_id"] != "mean"].copy()
    if members.empty:
        fig.update_layout(title="No member data available")
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
    models = sorted(members["model"].unique())

    np.random.seed(42)

    # -- Scatter dots per model --
    for i, tm in enumerate(target_months):
        for model in models:
            color = MEGA_COLORS.get(model, "#999999")
            mdf = members[(members["target_month"] == tm) & (members["model"] == model)]
            if len(mdf) == 0:
                continue
            jitter = np.random.uniform(-0.25, 0.25, len(mdf))
            fig.add_trace(go.Scatter(
                x=i + jitter, y=mdf["nino34_anom"],
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.5),
                showlegend=(i == 0),  # legend entry only on first month
                name=model,
                hovertemplate=f"{model}: %{{y:.2f}}\u00b0C<extra></extra>",
            ))

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

    # ENSO background shading (use 100 so it always fills to axis edge)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.5, y1=100,
                  xref="paper", yref="y",
                  fillcolor="red", opacity=0.04, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=1, y0=-100, y1=-0.5,
                  xref="paper", yref="y",
                  fillcolor="blue", opacity=0.04, line_width=0)

    tick_labels = [pd.Timestamp(tm + "-01").strftime("%b\n%Y") for tm in target_months]

    # Y-axis range
    all_vals = members["nino34_anom"].dropna().tolist()
    ymin = min(-3.0, min(all_vals) - 0.3) if all_vals else -3.0
    ymax = max(3.0, max(all_vals) + 0.3) if all_vals else 3.0

    fig.update_layout(
        title=f"Nino3.4 Forecast Distribution ({len(models)} models, model-weighted box plot)",
        yaxis_title="Nino3.4 SST Anomaly (\u00b0C)",
        xaxis=dict(
            tickvals=list(range(len(target_months))),
            ticktext=tick_labels,
            gridcolor=theme["grid_color"],
        ),
        yaxis=dict(range=[ymin, ymax], gridcolor=theme["grid_color"]),
        template=theme["template"],
        paper_bgcolor=theme["paper_color"],
        plot_bgcolor=theme["bg_color"],
        font=dict(color=theme["text_color"]),
        height=550,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.3)" if dark_mode else "rgba(255,255,255,0.9)",
            font=dict(size=9),
        ),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Historical Context
# ---------------------------------------------------------------------------

def create_enso_historical_context(forecast_df, dark_mode=False):
    """Observed Nino3.4 since 1990 with El Nino/La Nina fills and forecast overlay."""
    theme = get_theme(dark_mode)
    fig = go.Figure()

    obs_full = load_full_observed(start_year=1990)
    if obs_full.empty:
        fig.update_layout(title="No observed data available")
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

    # -- Forecast overlay --
    if not forecast_df.empty:
        try:
            forecast_only = _get_forecast_only(forecast_df)
            mega = _build_mega_df(forecast_only)
            means = mega[mega["member_id"] == "mean"].copy()
            fc_members = mega[mega["member_id"] != "mean"].copy()

            if not means.empty:
                mm = means.groupby("target_month", as_index=False)["nino34_anom"].mean()
                mm["date"] = mm["target_month"].apply(_target_month_to_date)
                mm = mm.sort_values("date")

                # Compute weighted 25-75th percentiles
                target_months_sorted = sorted(mm["target_month"].unique())
                q25_vals, q75_vals = [], []
                for tm in target_months_sorted:
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
                    name="Multi-model mean forecast",
                    line=dict(color=mm_line_color, width=2.2, dash="dash"),
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
        except Exception as e:
            logger.warning(f"Error adding forecast overlay: {e}")

    # Y-axis range
    all_vals = list(anom)
    ymin = min(-3.0, min(all_vals) - 0.3)
    ymax = max(3.0, max(all_vals) + 0.3)

    fig.update_layout(
        title="ENSO Nino3.4 Index: Historical Record and Current Forecast",
        yaxis_title="Nino3.4 SST Anomaly (\u00b0C)",
        template=theme["template"],
        paper_bgcolor=theme["paper_color"],
        plot_bgcolor=theme["bg_color"],
        font=dict(color=theme["text_color"]),
        height=450,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.3)" if dark_mode else "rgba(255,255,255,0.9)",
            font=dict(size=10),
        ),
        xaxis=dict(gridcolor=theme["grid_color"]),
        yaxis=dict(range=[ymin, ymax], gridcolor=theme["grid_color"]),
        margin=dict(l=60, r=30, t=50, b=50),
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
# Static image generation
# ---------------------------------------------------------------------------

def generate_enso_static_images(forecast_df, obs_df, assets_dir):
    """Render 6 static PNGs (3 plots x 2 modes) for ENSO tab."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    plot_configs = [
        ("enso_mega_plume", lambda dm: create_enso_mega_plume(forecast_df, obs_df, dm), 550),
        ("enso_box_distribution", lambda dm: create_enso_box_distribution(forecast_df, dm), 550),
        ("enso_historical", lambda dm: create_enso_historical_context(forecast_df, dm), 450),
    ]

    for dark_mode in [True, False]:
        mode = "dark" if dark_mode else "light"
        for name, create_func, height in plot_configs:
            filename = f"{name}_{mode}.png"
            filepath = assets_dir / filename
            try:
                logger.info(f"Generating {filename}...")
                fig = create_func(dark_mode)
                fig.write_image(str(filepath), width=1200, height=height, scale=2)
            except Exception as e:
                logger.error(f"Failed to generate {filename}: {e}")
