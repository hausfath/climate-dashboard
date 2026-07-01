"""Climate Dashboard visualizations using Plotly Dash."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table, callback, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LinearRegression

from src.models_vs_obs import MONTHLY_PREINDUSTRIAL_OFFSETS
from src import layout as L
from src.theme import install_fonts, template_name, tokens


# Theme configurations for light and dark modes.
# Color tokens live in src/theme.py; the registered 'climate_dark' /
# 'climate_light' templates carry fonts, grids, margins, and legends.
THEME_CONFIG = {
    'light': {
        'template': 'climate_light',
        'bg_color': '#ffffff',
        'paper_color': '#ffffff',
        'text_color': '#1f2430',
        'text_dim': '#5a6172',
        'grid_color': 'rgba(31, 36, 48, 0.08)',
        'line_color': 'rgba(31, 36, 48, 0.25)',
        'rolling_color': '#d94f2e',
        'threshold_color': '#c08a1e',
        'highlight_colors': {
            2023: '#0f9d94',
            2024: '#e8890c',
            2025: '#d64541',
            2026: '#6c5ce7',
        },
        'historical_color': '#0f9d94',
        'prediction_color': '#d94f2e',
        'ytd_color': '#e8890c',
        'mtd_color': '#e8890c',
        'card_color': 'light',
        'vrect_color': 'rgba(15, 157, 148, 0.08)',
        'background_years_color': 'rgba(31, 36, 48, 0.15)',
        'enso_update_color': '#2e9cb8',
        'era_band_colors': ['rgba(46, 111, 201, 0.13)',
                            'rgba(31, 36, 48, 0.13)',
                            'rgba(232, 137, 12, 0.15)'],
        'era_line_colors': ['rgba(46, 111, 201, 0.55)',
                            'rgba(31, 36, 48, 0.45)',
                            'rgba(232, 137, 12, 0.6)'],
    },
    'dark': {
        'template': 'climate_dark',
        'bg_color': '#151a30',
        'paper_color': '#151a30',
        'text_color': '#e8eaf2',
        'text_dim': '#9aa0b8',
        'grid_color': 'rgba(232, 234, 242, 0.07)',
        'line_color': 'rgba(232, 234, 242, 0.20)',
        'rolling_color': '#ff6b4a',
        'threshold_color': '#e8b84b',
        'highlight_colors': {
            2023: '#4ecdc4',
            2024: '#ff9f43',
            2025: '#ff6b6b',
            2026: '#a29bfe',
        },
        'historical_color': '#4ecdc4',
        'prediction_color': '#ff6b4a',
        'ytd_color': '#ff9f43',
        'mtd_color': '#ff9f43',
        'card_color': 'dark',
        'vrect_color': 'rgba(78, 205, 196, 0.10)',
        'background_years_color': 'rgba(232, 234, 242, 0.13)',
        'enso_update_color': '#48cae4',
        'era_band_colors': ['rgba(84, 160, 255, 0.12)',
                            'rgba(232, 234, 242, 0.11)',
                            'rgba(255, 159, 67, 0.13)'],
        'era_line_colors': ['rgba(84, 160, 255, 0.5)',
                            'rgba(232, 234, 242, 0.4)',
                            'rgba(255, 159, 67, 0.55)'],
    }
}

# Era windows for the daily-plot background envelopes (start, end inclusive).
# The last era is open-ended and stops just before the highlighted years.
ERA_BANDS = [(1940, 1979), (1980, 1999), (2000, 2022)]


def get_theme(dark_mode: bool = False) -> dict:
    """Get theme configuration based on mode."""
    return THEME_CONFIG['dark'] if dark_mode else THEME_CONFIG['light']


def adjust_anomalies_to_preindustrial(df: pd.DataFrame, date_col: str = 'date',
                                       anomaly_col: str = 'anomaly') -> pd.DataFrame:
    """
    Adjusts temperature anomalies to show change relative to preindustrial levels.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - date_col (str): Name of the column containing the date.
    - anomaly_col (str): Name of the column containing the temperature anomaly.

    Returns:
    - DataFrame: DataFrame with adjusted temperature anomalies.
    """
    # Adjusting the anomalies (1991-2020 baseline → 1850-1900 preindustrial).
    # Source of truth lives in src/models_vs_obs.MONTHLY_PREINDUSTRIAL_OFFSETS so
    # the Models vs Obs tab applies the same per-month shift to ERA5.
    df_adjusted = df.copy()
    df_adjusted[anomaly_col] = df.apply(
        lambda row: row[anomaly_col] + MONTHLY_PREINDUSTRIAL_OFFSETS[row[date_col].month], axis=1
    )

    return df_adjusted


# ---------------------------------------------------------------------------
# ECMWF EC46 forecast tail (loaded once per render; used by Figures 1 & 2)
# ---------------------------------------------------------------------------

def _load_ec46_forecast() -> pd.DataFrame | None:
    """Read the latest EC46 percentile-summary CSV under ``data/``.
    Returns None if no forecast file is present (graceful fallback)."""
    import glob
    pattern = str(Path(__file__).resolve().parent.parent
                  / "data" / "era5_forecast_ec46_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    df = pd.read_csv(matches[-1])
    df["date"] = pd.to_datetime(df["date"])
    if "init_date" in df.columns:
        df["init_date"] = pd.to_datetime(df["init_date"])
    return df


def _ec46_anomalize(
    fcst: pd.DataFrame, obs_adj: pd.DataFrame
) -> pd.DataFrame | None:
    """Convert EC46 absolute t2m to preindustrial anomaly + apply a
    data-driven model bias correction.

    This uses exactly the same transform as the EC46 forecast-skill plot
    (src.ec46_skill._anomalize_forecast): subtract the per-day-of-year
    1991–2020 climatology, add the monthly preindustrial offset, and add
    a single archive-wide bias correction. That bias (mean of ERA5 obs −
    forecast t2m at lead-day-0 across all archived inits) removes the
    ~0.2 °C cold offset between ECMWF's operational analysis (used to
    initialise EC46) and ERA5 reanalysis.

    The forecast is intentionally *not* anchored to recent observations,
    so the dashboard tail matches the skill plot exactly. A small step
    (~0.05 °C, the former anchor delta) may therefore appear at the
    observed→forecast boundary; the dotted bridge segment spans it.

    obs_adj must already have its anomaly column preindustrial-adjusted
    (i.e. produced by adjust_anomalies_to_preindustrial). Returns a
    DataFrame with the same percentile cols, or None if alignment is
    impossible.
    """
    if fcst is None or fcst.empty or obs_adj is None or obs_adj.empty:
        return None
    obs = obs_adj.copy()
    if "day_of_year" not in obs.columns:
        obs["day_of_year"] = obs["date"].dt.dayofyear
    # Day-of-year climatology (per-DOY 1991–2020 mean) — average across
    # years to smooth leap-day noise. The production data flow renames
    # `clim_91-20` → `climatology` (src/scraper.py:parse_era5_data); accept
    # either so this helper works whether obs_adj came through that loader
    # or was read raw from disk.
    clim_col = next((c for c in ("climatology", "clim_91-20")
                     if c in obs.columns), None)
    if clim_col is None:
        return None
    doy_clim = obs.groupby("day_of_year")[clim_col].mean()

    # Data-driven IFS-vs-ERA5 bias from the EC46 archive (0.0 on first
    # run before the archive has any pairings).
    try:
        from src.ec46_skill import estimate_archive_bias
        bias = estimate_archive_bias()
    except Exception:
        bias = 0.0

    out = fcst.copy()
    out["day_of_year"] = out["date"].dt.dayofyear
    out["clim_C"] = out["day_of_year"].map(doy_clim)
    out["pi_offset"] = out["date"].apply(
        lambda d: MONTHLY_PREINDUSTRIAL_OFFSETS[d.month])
    cols = ["t2m_mean", "t2m_p5", "t2m_p25", "t2m_p50", "t2m_p75", "t2m_p95"]
    for c in cols:
        out[c] = out[c] + bias - out["clim_C"] + out["pi_offset"]

    return out.drop(columns=["clim_C", "pi_offset"])


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB or rgba() / rgb() string to a Plotly rgba(...) string
    with the given alpha override."""
    s = hex_color.strip()
    if s.startswith("#") and len(s) == 7:
        r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    if s.startswith("rgba(") and s.endswith(")"):
        parts = [p.strip() for p in s[5:-1].split(",")]
        return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    if s.startswith("rgb(") and s.endswith(")"):
        parts = [p.strip() for p in s[4:-1].split(",")]
        return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {alpha})"
    return s  # fall back to whatever we got


def create_time_series_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """Create a time series plot of global temperature anomalies relative to preindustrial."""
    theme = get_theme(dark_mode)

    # Adjust anomalies to preindustrial baseline
    df_adj = adjust_anomalies_to_preindustrial(df)

    fig = go.Figure()

    # Add anomaly line (using Scattergl for better performance with large datasets)
    fig.add_trace(go.Scattergl(
        x=df_adj['date'],
        y=df_adj['anomaly'],
        mode='lines',
        name='Daily Anomaly',
        line=dict(color=theme['line_color'], width=0.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Anomaly: %{y:.2f}°C<extra></extra>'
    ))

    # Add 365-day rolling mean
    df_rolling = df_adj.copy()
    df_rolling['rolling_365'] = df_rolling['anomaly'].rolling(window=365, center=True).mean()

    fig.add_trace(go.Scattergl(
        x=df_rolling['date'],
        y=df_rolling['rolling_365'],
        mode='lines',
        name='365-day Average',
        line=dict(color=theme['rolling_color'], width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>365-day avg: %{y:.2f}°C<extra></extra>'
    ))

    # Add 1.5°C reference line (label on the left so the right edge stays
    # free for the rolling-mean end label)
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="top left")

    # Direct label on the rolling mean's endpoint (replaces the legend)
    tail = df_rolling.dropna(subset=['rolling_365'])
    if len(tail) > 0:
        last = tail.iloc[-1]
        fig.add_annotation(
            x=last['date'], y=last['rolling_365'],
            text=f"365-day avg<br>{last['rolling_365']:.2f}°C",
            font=dict(size=12, color=theme['rolling_color']),
            showarrow=False, xanchor='left', xshift=8, align='left',
        )

    fig.update_layout(
        xaxis=dict(
            title='',
            range=[df_adj['date'].min(),
                   df_adj['date'].max() + pd.Timedelta(days=150)],
        ),
        yaxis_title='Temperature anomaly (°C)',
        template=theme['template'],
        showlegend=False,
        height=500,
        margin=dict(r=110),
    )

    return fig


def get_recent_month_bounds(df: pd.DataFrame) -> tuple:
    """Get the day-of-year bounds for the current month in the data.

    Uses the latest date's own year so the vrect aligns with the actual
    plotted day-of-year values for the current year (otherwise a leap-year
    reference shifts non-leap-year months by one day past February).
    """
    latest_date = df['date'].max()
    year = latest_date.year
    month = latest_date.month

    month_start = pd.Timestamp(f"{year}-{month:02d}-01").dayofyear
    if month == 12:
        month_end = pd.Timestamp(f"{year}-12-31").dayofyear + 1
    else:
        month_end = pd.Timestamp(f"{year}-{month+1:02d}-01").dayofyear

    return month_start, month_end


def _add_era_envelopes(fig: go.Figure, df: pd.DataFrame, value_col: str,
                       theme: dict) -> None:
    """Replace per-year background spaghetti with 5–95th percentile envelopes
    for each era in ERA_BANDS. Percentiles are computed per day-of-year and
    lightly smoothed so the bands read as clean shapes."""
    for (start, end), fill, line in zip(ERA_BANDS,
                                        theme['era_band_colors'],
                                        theme['era_line_colors']):
        era = df[(df['year'] >= start) & (df['year'] <= end)]
        if era.empty:
            continue
        grouped = era.groupby('day_of_year')[value_col]
        p05 = grouped.quantile(0.05).rolling(7, center=True, min_periods=1).mean()
        p95 = grouped.quantile(0.95).rolling(7, center=True, min_periods=1).mean()
        doy = p05.index.tolist()
        fig.add_trace(go.Scatter(
            x=doy + doy[::-1],
            y=p95.tolist() + p05.tolist()[::-1],
            fill='toself', fillcolor=fill,
            line=dict(color=line, width=0.5),
            name=f'{start}–{end}',
            hoverinfo='skip',
        ))


def create_daily_anomalies_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """
    Create a plot of daily temperature anomalies by day of year relative to preindustrial.
    Highlights years from 2023 onward and shades the most recent month.
    """
    theme = get_theme(dark_mode)

    # Adjust anomalies to preindustrial baseline
    df_adj = adjust_anomalies_to_preindustrial(df)

    fig = go.Figure()

    # Define years to highlight (2023 onward)
    years_to_highlight = [2023, 2024, 2025, 2026]

    # Get all unique years
    all_years = sorted(df_adj['year'].unique())

    # Get most recent month bounds for shading
    month_start, month_end = get_recent_month_bounds(df_adj)

    # Add shaded region for most recent month
    fig.add_vrect(
        x0=month_start, x1=month_end,
        fillcolor=theme['vrect_color'], opacity=0.5,
        layer="below", line_width=0,
    )
    fig.add_annotation(
        x=(month_start + month_end) / 2, y=1, yref='paper', yanchor='bottom',
        text='current month', showarrow=False, yshift=2,
        font=dict(size=10.5, color=theme['text_dim']),
    )

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Historical context as era percentile envelopes (was ~80 spaghetti lines)
    _add_era_envelopes(fig, df_adj, 'anomaly', theme)

    # Plot highlighted years on top
    for year in years_to_highlight:
        year_data = df_adj[df_adj['year'] == year].sort_values('day_of_year')
        if len(year_data) > 0:
            dates = year_data['date'].dt.strftime('%b %-d')
            fig.add_trace(go.Scatter(
                x=year_data['day_of_year'],
                y=year_data['anomaly'],
                mode='lines',
                name=str(year),
                line=dict(color=theme['highlight_colors'][year], width=2.5),
                customdata=dates,
                hovertemplate=f'{year}<br>%{{customdata}}<br>Anomaly: %{{y:.2f}}°C<extra></extra>'
            ))

    # ECMWF EC46 forecast tail layered onto the current-year trace
    ec46 = _load_ec46_forecast()
    if ec46 is not None and not ec46.empty:
        ec46_anom = _ec46_anomalize(ec46, df_adj)
        if ec46_anom is not None:
            current_year = int(ec46_anom['date'].iloc[0].year)
            color = theme['highlight_colors'].get(
                current_year, theme['rolling_color'])
            band_outer = _hex_to_rgba(color, 0.10)
            band_inner = _hex_to_rgba(color, 0.22)
            fc_doy = ec46_anom['date'].dt.dayofyear

            fig.add_trace(go.Scatter(
                x=list(fc_doy) + list(fc_doy[::-1]),
                y=list(ec46_anom['t2m_p95']) + list(ec46_anom['t2m_p5'][::-1]),
                fill='toself', fillcolor=band_outer,
                line=dict(width=0), hoverinfo='skip',
                name=f'EC46 5–95th %ile', showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=list(fc_doy) + list(fc_doy[::-1]),
                y=list(ec46_anom['t2m_p75']) + list(ec46_anom['t2m_p25'][::-1]),
                fill='toself', fillcolor=band_inner,
                line=dict(width=0), hoverinfo='skip',
                name=f'EC46 25–75th %ile', showlegend=False,
            ))
            # Bridge segment from last observed day to forecast day 0
            cy_obs = df_adj[df_adj['year'] == current_year].sort_values('date')
            if not cy_obs.empty:
                last_obs = cy_obs.iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[last_obs['day_of_year'], int(fc_doy.iloc[0])],
                    y=[last_obs['anomaly'], ec46_anom['t2m_mean'].iloc[0]],
                    mode='lines',
                    line=dict(color=color, width=2.5, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                ))
            fc_dates = ec46_anom['date'].dt.strftime('%b %-d')
            fig.add_trace(go.Scatter(
                x=fc_doy, y=ec46_anom['t2m_mean'],
                mode='lines',
                name=f'{current_year} forecast (EC46)',
                line=dict(color=color, width=2.5, dash='dash'),
                customdata=fc_dates,
                hovertemplate=(f'{current_year} forecast<br>%{{customdata}}'
                               '<br>Mean: %{y:.2f}°C<extra></extra>'),
            ))

    # Month labels for x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            range=[1, 366],
        ),
        yaxis=dict(title='Temperature anomaly (°C)'),
        hovermode='x',
        template=theme['template'],
        height=500,
    )

    return fig


def create_daily_absolutes_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """
    Create a plot of daily absolute temperatures by day of year.
    Highlights years from 2023 onward and shades the most recent month.
    """
    theme = get_theme(dark_mode)
    fig = go.Figure()

    # Define years to highlight (2023 onward)
    years_to_highlight = [2023, 2024, 2025, 2026]

    # Get all unique years
    all_years = sorted(df['year'].unique())

    # Get most recent month bounds for shading
    month_start, month_end = get_recent_month_bounds(df)

    # Add shaded region for most recent month
    fig.add_vrect(
        x0=month_start, x1=month_end,
        fillcolor=theme['vrect_color'], opacity=0.5,
        layer="below", line_width=0,
    )
    fig.add_annotation(
        x=(month_start + month_end) / 2, y=1, yref='paper', yanchor='bottom',
        text='current month', showarrow=False, yshift=2,
        font=dict(size=10.5, color=theme['text_dim']),
    )

    # Historical context as era percentile envelopes (was ~80 spaghetti lines)
    _add_era_envelopes(fig, df, 'temperature', theme)

    # Plot highlighted years on top
    for year in years_to_highlight:
        year_data = df[df['year'] == year].sort_values('day_of_year')
        if len(year_data) > 0:
            dates = year_data['date'].dt.strftime('%b %-d')
            fig.add_trace(go.Scatter(
                x=year_data['day_of_year'],
                y=year_data['temperature'],
                mode='lines',
                name=str(year),
                line=dict(color=theme['highlight_colors'][year], width=2.5),
                customdata=dates,
                hovertemplate=f'{year}<br>%{{customdata}}<br>Temp: %{{y:.2f}}°C<extra></extra>'
            ))

    # Month labels for x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            range=[1, 366],
        ),
        yaxis=dict(title='Temperature (°C)'),
        hovermode='x',
        template=theme['template'],
        height=500,
    )

    return fig


def calculate_monthly_prediction(df: pd.DataFrame, target_month: int, target_year: int) -> tuple:
    """
    Calculate the predicted monthly value and its error using linear regression.

    Uses month-to-date data to predict the full month average based on
    historical relationships between partial and complete months.

    Parameters:
    - df: DataFrame with anomaly data (already adjusted to preindustrial)
    - target_month: Month to predict (1-12)
    - target_year: Year for which the prediction is required

    Returns:
    - tuple: (predicted_value, error, month_to_date_avg, days_in_month_so_far)
    """
    # Filter for the target month across all years
    month_data = df[(df['date'].dt.month == target_month) & (df['date'].dt.year <= target_year)]

    # Get target year data for this month
    target_year_data = month_data[month_data['date'].dt.year == target_year]

    if len(target_year_data) == 0:
        return None, None, None, 0

    days_so_far = len(target_year_data)
    month_to_date_avg = target_year_data['anomaly'].mean()

    # Calculate full monthly averages for historical years
    monthly_avg = month_data.groupby(month_data['date'].dt.year)['anomaly'].mean()

    # Calculate month-to-date averages for historical years (same number of days as current)
    def get_mtd_avg(group):
        return group['anomaly'].iloc[:days_so_far].mean() if len(group) >= days_so_far else np.nan

    month_to_date_past = month_data.groupby(month_data['date'].dt.year).apply(get_mtd_avg, include_groups=False)

    # Remove target year and any NaN values for regression
    valid_years = month_to_date_past.dropna().index
    valid_years = [y for y in valid_years if y != target_year]

    if len(valid_years) < 5:  # Need enough data for regression
        return month_to_date_avg, 0.1, month_to_date_avg, days_so_far

    X = month_to_date_past[valid_years].values.reshape(-1, 1)
    Y = monthly_avg[valid_years].values

    # Linear regression
    regressor = LinearRegression()
    regressor.fit(X, Y)
    predicted_value = regressor.predict(np.array([[month_to_date_avg]]))[0]

    # Calculate error (2 standard deviations of residuals)
    residuals = Y - regressor.predict(X)
    std_dev = np.std(residuals)
    error = 2 * std_dev

    return predicted_value, error, month_to_date_avg, days_so_far


def create_monthly_projection_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """
    Create a plot showing historical monthly temperatures with a projection
    for where the current month will likely end up.
    """
    theme = get_theme(dark_mode)

    # Adjust anomalies to preindustrial baseline
    df_adj = adjust_anomalies_to_preindustrial(df)

    # Determine the current month and year from the data
    latest_date = df_adj['date'].max()
    target_month = latest_date.month
    target_year = latest_date.year

    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_name = month_names[target_month - 1]

    # Get prediction
    predicted_value, error, mtd_avg, days_so_far = calculate_monthly_prediction(
        df_adj, target_month, target_year
    )

    # Calculate historical monthly averages for this month
    month_data = df_adj[df_adj['date'].dt.month == target_month]
    monthly_avg = month_data.groupby(month_data['date'].dt.year)['anomaly'].mean()

    # Separate historical data from target year
    historical_years = [y for y in monthly_avg.index if y < target_year]
    historical_values = monthly_avg[historical_years]

    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_years,
        y=historical_values.values,
        mode='lines+markers',
        name='Historical',
        line=dict(color=theme['historical_color'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>Anomaly: %{y:.2f}°C<extra></extra>'
    ))

    # Add prediction with error bar
    if predicted_value is not None:
        fig.add_trace(go.Scatter(
            x=[target_year],
            y=[predicted_value],
            mode='markers',
            name=f'{month_name} {target_year} Prediction',
            marker=dict(color=theme['prediction_color'], size=12, symbol='circle'),
            error_y=dict(
                type='data',
                array=[error],
                visible=True,
                color=theme['prediction_color'],
                thickness=2,
                width=8
            ),
            hovertemplate=f'{month_name} {target_year} Prediction<br>{predicted_value:.2f}°C ±{error:.2f}°C (2σ)<extra></extra>'
        ))

        # Add month-to-date marker
        fig.add_trace(go.Scatter(
            x=[target_year],
            y=[mtd_avg],
            mode='markers',
            name=f'Month-to-date ({days_so_far} days)',
            marker=dict(color=theme['mtd_color'], size=10, symbol='diamond'),
            hovertemplate=f'Month-to-date<br>Average: {mtd_avg:.2f}°C<br>({days_so_far} days)<extra></extra>'
        ))

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Direct labels on the prediction and month-to-date markers
    if predicted_value is not None:
        fig.add_annotation(
            x=target_year, y=predicted_value + (error or 0),
            text=f"{month_name} {target_year}<br>{predicted_value:.2f} ±{error:.2f}°C",
            font=dict(size=11.5, color=theme['prediction_color']),
            showarrow=False, xanchor='right', yanchor='bottom',
            xshift=-6, align='right',
        )
        fig.add_annotation(
            x=target_year, y=mtd_avg,
            text=f"month-to-date ({days_so_far}d)",
            font=dict(size=10.5, color=theme['mtd_color']),
            showarrow=False, xanchor='right', yanchor='top',
            xshift=-8, yshift=-4,
        )

    fig.update_layout(
        xaxis=dict(title='', range=[min(historical_years) - 1, target_year + 1.5]),
        yaxis_title='Temperature anomaly (°C)',
        hovermode='x',
        template=theme['template'],
        showlegend=False,
        height=500,
    )

    return fig


def create_daily_heatmap(df: pd.DataFrame, data_type: str = 'anomaly', dark_mode: bool = False) -> go.Figure:
    """
    Create a heatmap of daily temperatures or anomalies by day of year and year.

    Parameters:
    - df: DataFrame with temperature data
    - data_type: 'anomaly' for temperature anomalies or 'temperature' for absolute temps
    - dark_mode: Whether to use dark mode styling

    Returns:
    - Plotly Figure object
    """
    theme = get_theme(dark_mode)

    # Use preindustrial-adjusted anomalies if showing anomalies
    if data_type == 'anomaly':
        data = adjust_anomalies_to_preindustrial(df)
        column_to_use = 'anomaly'
        cbar_label = 'Anomaly (°C)'
        hover_val_label = 'Anomaly'
        # Diverging scale centered on zero — blue genuinely means "below
        # preindustrial", red above.
        zmid = 0.0
    else:
        data = df.copy()
        column_to_use = 'temperature'
        cbar_label = 'Temp (°C)'
        hover_val_label = 'Temp'
        zmid = 14.5  # Approximate global mean temperature

    # Pivot data: day_of_year as rows, year as columns
    heatmap_data = data.pivot(index='day_of_year', columns='year', values=column_to_use)

    # Build date label matrix for hover (day_of_year → "Jan 1", "Jan 2", etc.)
    import datetime
    doy_to_date = {}
    for doy in heatmap_data.index:
        try:
            d = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=int(doy) - 1)  # 2024 is a leap year
            doy_to_date[doy] = d.strftime('%b %-d')
        except (ValueError, OverflowError):
            doy_to_date[doy] = f'Day {doy}'

    hover_text = [[f"Year: {col}<br>{doy_to_date.get(doy, f'Day {doy}')}<br>{hover_val_label}: {heatmap_data.loc[doy, col]:.2f}°C"
                    if pd.notna(heatmap_data.loc[doy, col]) else ""
                    for col in heatmap_data.columns]
                   for doy in heatmap_data.index]

    # Month labels for y-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu_r',
        zmid=zmid,
        colorbar=dict(title=dict(text=cbar_label, side='right'), thickness=14,
                      outlinewidth=0),
        text=hover_text,
        hoverinfo='text',
    ))

    fig.update_layout(
        xaxis=dict(title='', dtick=10, showgrid=False),
        yaxis=dict(
            title='',
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            autorange='reversed',
            showgrid=False,
        ),
        template=theme['template'],
        height=500,
    )

    return fig


def generate_ridgeline_plot(df: pd.DataFrame, output_dir: Path, dark_mode: bool = False) -> str:
    """
    Generate a ridgeline plot showing temperature anomaly distributions by year.

    Returns path to generated image.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Polygon
    import matplotlib.ticker as ticker
    import logging

    logger = logging.getLogger(__name__)

    # Adjust anomalies to preindustrial baseline
    df_plot = adjust_anomalies_to_preindustrial(df.copy())

    years = sorted(df_plot['year'].unique())
    year_min, year_max = min(years), max(years)

    # Data limits
    data_min = df_plot['anomaly'].min()
    data_max = df_plot['anomaly'].max()

    # Grid for KDE evaluation
    x_grid = np.linspace(data_min - 0.5, data_max + 0.5, 300)

    # Theme colors (match the panel background from src/theme.py)
    if dark_mode:
        bg_color = '#151a30'
        text_color = '#e8eaf2'
        line_color = 'white'
    else:
        bg_color = '#ffffff'
        text_color = '#1f2430'
        line_color = 'white'

    plt.rcParams['font.family'] = ['IBM Plex Sans', 'sans-serif']

    # Setup figure
    fig = plt.figure(figsize=(12, 16))
    fig.patch.set_facecolor(bg_color)

    gs = fig.add_gridspec(2, 1, height_ratios=[40, 1], hspace=0.03)
    ax_main = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])

    ax_main.set_facecolor(bg_color)
    ax_cbar.set_facecolor(bg_color)

    # Parameters
    step = 0.4
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(data_min, data_max)

    def plot_gradient_fill(ax, x, y, offset, year_idx, is_incomplete):
        verts = [(x[0], offset), *zip(x, y + offset), (x[-1], offset)]
        poly = Polygon(verts, facecolor='none', edgecolor='none')
        ax.add_patch(poly)

        img_data = np.atleast_2d(x)
        im = ax.imshow(img_data, extent=[x[0], x[-1], offset, offset + y.max() + 0.1],
                       aspect='auto', cmap=cmap, norm=norm, zorder=year_idx)
        im.set_clip_path(poly)

        linestyle = '--' if is_incomplete else '-'
        linewidth = 1.0 if is_incomplete else 0.5
        ax.plot(x, y + offset, color=line_color, linewidth=linewidth,
                linestyle=linestyle, zorder=year_idx + 0.1)

    # Main plotting loop
    for i, year in enumerate(years):
        data = df_plot[df_plot['year'] == year]['anomaly']

        if len(data) < 10:
            continue

        kde = gaussian_kde(data)
        y_vals = kde(x_grid)

        offset = (len(years) - 1 - i) * step
        is_incomplete = (year == year_max)

        plot_gradient_fill(ax_main, x_grid, y_vals, offset, i, is_incomplete)

        # Year labels
        if year % 5 == 0 or year == year_max or year == year_min:
            ax_main.text(data_min + 0.05, offset + 0.1, str(year),
                         verticalalignment='center', horizontalalignment='left',
                         fontsize=9, color=text_color, zorder=1000, fontweight='bold')

    # Style main plot (no title — the panel header carries it)
    ax_main.set_frame_on(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_xlim(data_min, data_max)

    # Color bar
    gradient = np.atleast_2d(np.linspace(data_min, data_max, 500))
    ax_cbar.imshow(gradient, extent=[data_min, data_max, 0, 1],
                   aspect='auto', cmap=cmap, norm=norm)

    ax_cbar.set_yticks([])
    ax_cbar.set_xlim(data_min, data_max)
    ax_cbar.set_xlabel('Temperature anomaly (°C) relative to 1850–1900',
                       fontsize=12, color=text_color)
    ax_cbar.tick_params(colors=text_color)

    ax_cbar.spines['top'].set_visible(False)
    ax_cbar.spines['left'].set_visible(False)
    ax_cbar.spines['right'].set_visible(False)
    ax_cbar.spines['bottom'].set_visible(True)
    ax_cbar.spines['bottom'].set_color(text_color)

    ax_cbar.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = 'dark' if dark_mode else 'light'
    filename = f"ridgeline_{mode_suffix}.png"
    filepath = output_dir / filename

    logger.info(f"Generating {filename}...")
    plt.savefig(filepath, dpi=150, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)

    return f"/assets/images/{filename}"


def generate_ridgeline_images(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Generate ridgeline plot images for both light and dark mode.
    """
    import logging
    logger = logging.getLogger(__name__)

    image_paths = {}

    for dark_mode in [True, False]:
        mode_suffix = 'dark' if dark_mode else 'light'
        path = generate_ridgeline_plot(df, output_dir, dark_mode)
        image_paths[mode_suffix] = path

    logger.info("Ridgeline images generated successfully")
    return image_paths


def generate_all_static_images(df: pd.DataFrame, output_dir: Path, enso_df: pd.DataFrame = None) -> None:
    """
    Generate static PNG images for all plots (both light and dark mode).
    Called during data updates to pre-render plots for fast loading.
    """
    import logging
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot configurations: (name, function, extra_args)
    plot_configs = [
        ('timeseries', lambda dm: create_time_series_plot(df, dm), {}),
        ('daily_anomalies', lambda dm: create_daily_anomalies_plot(df, dm), {}),
        ('daily_temps', lambda dm: create_daily_absolutes_plot(df, dm), {}),
        ('monthly_projections', lambda dm: create_monthly_projection_plot(df, dm), {}),
        ('annual_prediction', lambda dm: create_annual_prediction_plot(df, enso_df, dm), {}),
        ('projection_history', lambda dm: create_projection_history_plot(df, dm), {}),
        ('heatmap_anomaly', lambda dm: create_daily_heatmap(df, 'anomaly', dm), {}),
        ('heatmap_temp', lambda dm: create_daily_heatmap(df, 'temperature', dm), {}),
    ]

    for dark_mode in [True, False]:
        mode_suffix = 'dark' if dark_mode else 'light'

        for name, create_func, _ in plot_configs:
            filename = f"{name}_{mode_suffix}.png"
            filepath = output_dir / filename

            try:
                logger.info(f"Generating {filename}...")
                fig = create_func(dark_mode)

                # Adjust dimensions based on plot type. The monthly and
                # annual projection charts render side by side (duo panels),
                # so they export squarer.
                if 'heatmap' in name:
                    fig.write_image(str(filepath), width=1200, height=600, scale=2)
                elif name in ('monthly_projections', 'annual_prediction'):
                    fig.write_image(str(filepath), width=780, height=460, scale=2)
                else:
                    fig.write_image(str(filepath), width=1200, height=500, scale=2)

            except Exception as e:
                logger.error(f"Failed to generate {filename}: {e}")

    # Also generate ridgeline images
    generate_ridgeline_images(df, output_dir)

    # Generate ENSO static images
    try:
        from src.enso_plots import load_enso_forecast_data, generate_enso_static_images
        ef, eo, _ = load_enso_forecast_data()
        if not ef.empty:
            generate_enso_static_images(ef, eo, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate ENSO static images: {e}")

    logger.info("All static images generated successfully")


def create_annual_prediction_plot(df: pd.DataFrame, enso_df: pd.DataFrame = None, dark_mode: bool = False) -> go.Figure:
    """
    Create a plot showing historical annual temperatures and 2026 prediction.

    Args:
        df: ERA5 daily temperature data
        enso_df: ENSO data (optional, will load if not provided)
        dark_mode: Whether to use dark mode styling
    """
    theme = get_theme(dark_mode)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    # Load ENSO data if not provided — prefer multi-model mean forecast
    if enso_df is None:
        try:
            from src.enso_plots import load_enso_forecast_data, build_enso_combined
            ef, eo, oni = load_enso_forecast_data()
            enso_df = build_enso_combined(oni, ef, eo)
        except Exception:
            enso_file = DATA_DIR / "enso_combined.csv"
            if enso_file.exists():
                enso_df = pd.read_csv(enso_file)
                enso_df['date'] = pd.to_datetime(enso_df['date'])
            else:
                enso_df = pd.DataFrame()

    # Adjust to preindustrial
    df_adj = adjust_anomalies_to_preindustrial(df)

    # Calculate annual means
    annual = df_adj.groupby('year').agg({
        'anomaly': 'mean'
    }).reset_index()
    annual = annual.rename(columns={'anomaly': 'annual_anomaly'})

    # Add prior year anomaly
    annual['prior_year_anomaly'] = annual['annual_anomaly'].shift(1)

    current_year = df_adj['year'].max()
    latest_date = df_adj['date'].max()
    current_month = latest_date.month
    day_of_year = latest_date.dayofyear

    # Calculate split ENSO features (obs vs future) if available
    if len(enso_df) > 0:
        enso_historical = enso_df[~enso_df['is_forecast']] if 'is_forecast' in enso_df.columns else enso_df
        enso_obs_by_year = {}
        enso_future_by_year = {}
        for year in annual['year'].unique():
            yr_enso = enso_historical[enso_historical['year'] == year]
            obs_months = yr_enso[yr_enso['month'] <= current_month]
            fut_months = yr_enso[yr_enso['month'] > current_month]
            enso_obs_by_year[year] = obs_months['oni'].mean() if len(obs_months) > 0 else np.nan
            enso_future_by_year[year] = fut_months['oni'].mean() if len(fut_months) > 0 else np.nan
        annual['enso_obs'] = annual['year'].map(enso_obs_by_year)
        annual['enso_future'] = annual['year'].map(enso_future_by_year)
    else:
        annual['enso_obs'] = 0
        annual['enso_future'] = 0

    # Get current year's trailing 30-day anomaly
    ytd_data = df_adj[df_adj['year'] == current_year].sort_values('date')
    days_available = len(ytd_data)
    lookback_days = min(30, days_available)
    current_trailing_30d = ytd_data.tail(lookback_days)['anomaly'].mean() if days_available > 0 else None
    current_ytd_anomaly = ytd_data['anomaly'].mean() if days_available > 0 else None

    # For each historical year, calculate trailing anomaly and YTD mean at same day-of-year
    trailing_by_year = {}
    ytd_by_year = {}
    for year in annual['year'].unique():
        year_data = df_adj[df_adj['year'] == year].sort_values('date')
        year_data_to_doy = year_data[year_data['day_of_year'] <= day_of_year]
        if len(year_data_to_doy) > 0:
            lookback = min(30, len(year_data_to_doy))
            trailing_by_year[year] = year_data_to_doy.tail(lookback)['anomaly'].mean()
            ytd_by_year[year] = year_data_to_doy['anomaly'].mean()

    annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)
    annual['ytd_anomaly'] = annual['year'].map(ytd_by_year)

    # Build prediction model with trailing anomaly as a feature
    # Exclude volcanic years
    volcanic_years = [1982, 1983, 1991, 1992, 1993]
    train_df = annual[
        (annual['year'] >= 1950) &
        (annual['year'] < current_year) &  # Only complete years for training
        (~annual['year'].isin(volcanic_years)) &
        (annual['prior_year_anomaly'].notna()) &
        (annual['trailing_anomaly'].notna()) &
        (annual['ytd_anomaly'].notna()) &
        (annual['enso_obs'].notna()) &
        (annual['enso_future'].notna())
    ].copy()

    # Train model if we have enough data
    predictions = []
    prediction_2026 = None
    uncertainty = 0.086  # Default uncertainty

    if len(train_df) > 10:
        from sklearn.preprocessing import StandardScaler

        feature_cols = ['year', 'prior_year_anomaly', 'enso_obs', 'enso_future', 'trailing_anomaly', 'ytd_anomaly']
        X = train_df[feature_cols].values
        y = train_df['annual_anomaly'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        # Calculate uncertainty from residuals
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        uncertainty = max(np.std(residuals), 0.02)

        # Generate predictions for historical years (for plotting trend line)
        for _, row in train_df.iterrows():
            year = row['year']
            X_pred = np.array([[year, row['prior_year_anomaly'], row['enso_obs'], row['enso_future'], row['trailing_anomaly'], row['ytd_anomaly']]])
            X_pred_scaled = scaler.transform(X_pred)
            pred = model.predict(X_pred_scaled)[0]

            predictions.append({
                'year': year,
                'predicted': pred,
                'lb': pred - 2 * uncertainty,
                'ub': pred + 2 * uncertainty
            })

        # Prediction for current year using current trailing 30-day data
        if current_trailing_30d is not None and len(enso_df) > 0:
            enso_ytd = enso_df[(enso_df['year'] == current_year) &
                               (enso_df['month'] <= current_month)]
            enso_future_months = enso_df[(enso_df['year'] == current_year) &
                                         (enso_df['month'] > current_month)]

            enso_obs_val = enso_ytd['oni'].mean() if len(enso_ytd) > 0 else 0
            enso_future_val = enso_future_months['oni'].mean() if len(enso_future_months) > 0 else 0

            # Get prior year anomaly
            prior_year_row = annual[annual['year'] == current_year - 1]
            if len(prior_year_row) > 0:
                prior_year_anomaly = prior_year_row['annual_anomaly'].values[0]

                X_pred = np.array([[current_year, prior_year_anomaly, enso_obs_val, enso_future_val, current_trailing_30d, current_ytd_anomaly]])
                X_pred_scaled = scaler.transform(X_pred)
                pred = model.predict(X_pred_scaled)[0]

                # Monte Carlo over ENSO ensemble members for the CI
                lb, ub = pred - 2 * uncertainty, pred + 2 * uncertainty
                try:
                    from src.annual_prediction_mc import (
                        load_enso_future_members, monte_carlo_ci,
                    )
                    members = load_enso_future_members(int(current_year), int(current_month))
                    if not members.empty:
                        mc = monte_carlo_ci(
                            model, scaler,
                            base_features={
                                'year': current_year,
                                'prior_year_anomaly': prior_year_anomaly,
                                'enso_obs': enso_obs_val,
                                'enso_future': enso_future_val,
                                'trailing_anomaly': current_trailing_30d,
                                'ytd_anomaly': current_ytd_anomaly,
                            },
                            members=members,
                            resid_std=uncertainty,
                            feature_order=feature_cols,
                        )
                        lb, ub = mc['lb'], mc['ub']
                except Exception:
                    pass  # fall back to analytic ±2σ

                predictions.append({
                    'year': current_year,
                    'predicted': pred,
                    'lb': lb,
                    'ub': ub,
                })

                prediction_2026 = {
                    'predicted': pred,
                    'lb': lb,
                    'ub': ub,
                }

    pred_df = pd.DataFrame(predictions) if predictions else pd.DataFrame()

    # Create figure
    fig = go.Figure()

    # Historical annual temperatures (completed years only)
    historical = annual[annual['year'] < current_year]

    fig.add_trace(go.Scatter(
        x=historical['year'],
        y=historical['annual_anomaly'],
        mode='lines+markers',
        name='Observed',
        line=dict(color=theme['historical_color'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>Anomaly: %{y:.2f}°C<extra></extra>'
    ))

    # Add current year partial data point (YTD)
    current_year_data = annual[annual['year'] == current_year]
    if len(current_year_data) > 0:
        ytd_anomaly = current_year_data['annual_anomaly'].values[0]
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[ytd_anomaly],
            mode='markers',
            name=f'{current_year} YTD',
            marker=dict(color=theme['ytd_color'], size=10, symbol='diamond'),
            hovertemplate=f'{current_year} YTD<br>Anomaly: %{{y:.2f}}°C<extra></extra>'
        ))

    # Add prediction with error bar for current year
    if prediction_2026:
        up = prediction_2026['ub'] - prediction_2026['predicted']
        down = prediction_2026['predicted'] - prediction_2026['lb']
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[prediction_2026['predicted']],
            mode='markers',
            name=f'{current_year} Prediction',
            marker=dict(color=theme['prediction_color'], size=12, symbol='circle'),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[up],
                arrayminus=[down],
                visible=True,
                color=theme['prediction_color'],
                thickness=2,
                width=6
            ),
            hovertemplate=f'{current_year} Prediction<br>%{{y:.2f}}°C (+{up:.2f}/-{down:.2f}°C)<extra></extra>'
        ))

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Direct labels on the prediction and YTD markers
    if prediction_2026:
        fig.add_annotation(
            x=current_year, y=prediction_2026['ub'],
            text=(f"{current_year} projection<br>"
                  f"{prediction_2026['predicted']:.2f}°C"),
            font=dict(size=11.5, color=theme['prediction_color']),
            showarrow=False, xanchor='right', yanchor='bottom',
            xshift=-6, align='right',
        )
    if len(current_year_data) > 0:
        fig.add_annotation(
            x=current_year, y=ytd_anomaly,
            text="YTD",
            font=dict(size=10.5, color=theme['ytd_color']),
            showarrow=False, xanchor='right', yanchor='top',
            xshift=-8, yshift=-4,
        )

    # Layout
    fig.update_layout(
        xaxis=dict(title='', range=[1938, current_year + 2], dtick=10),
        yaxis=dict(title='Temperature anomaly (°C)', range=[-0.2, 1.8]),
        hovermode='x',
        template=theme['template'],
        showlegend=False,
        height=500,
    )

    return fig


def calculate_projection_for_date(df: pd.DataFrame, target_date: pd.Timestamp, enso_df: pd.DataFrame = None) -> dict:
    """
    Calculate annual projection using only data available up to target_date.

    Uses the trailing 30-day (or available) anomaly as a predictor in the regression model,
    trained on historical data where we know how early-year data related to full-year outcomes.

    Returns dict with prediction, uncertainty, ytd_anomaly, and the date.
    """
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    # Load ENSO data if not provided — prefer multi-model mean forecast
    if enso_df is None:
        try:
            from src.enso_plots import load_enso_forecast_data, build_enso_combined
            ef, eo, oni = load_enso_forecast_data()
            enso_df = build_enso_combined(oni, ef, eo)
        except Exception:
            enso_file = DATA_DIR / "enso_combined.csv"
            if enso_file.exists():
                enso_df = pd.read_csv(enso_file)
                enso_df['date'] = pd.to_datetime(enso_df['date'])
            else:
                enso_df = pd.DataFrame()

    current_year = target_date.year
    current_month = target_date.month
    day_of_year = target_date.dayofyear

    # Filter data up to target date
    df_filtered = df[df['date'] <= target_date].copy()

    if len(df_filtered[df_filtered['year'] == current_year]) == 0:
        return None

    # Adjust to preindustrial
    df_adj = adjust_anomalies_to_preindustrial(df_filtered)

    # Get YTD data for current year
    ytd_data = df_adj[df_adj['year'] == current_year]
    ytd_anomaly = ytd_data['anomaly'].mean()

    # Get trailing 30-day anomaly (or all available days if less than 30)
    ytd_data_sorted = ytd_data.sort_values('date')
    days_available = len(ytd_data_sorted)
    lookback_days = min(30, days_available)
    trailing_data = ytd_data_sorted.tail(lookback_days)
    trailing_30d_anomaly = trailing_data['anomaly'].mean()

    # Calculate annual means for complete years
    annual = df_adj.groupby('year')['anomaly'].mean().reset_index()
    annual = annual.rename(columns={'anomaly': 'annual_anomaly'})
    annual['prior_year_anomaly'] = annual['annual_anomaly'].shift(1)

    # For each historical year, calculate the trailing anomaly and YTD mean at the same day-of-year
    trailing_by_year = {}
    ytd_by_year = {}
    for year in annual['year'].unique():
        if year >= current_year:
            continue
        year_data = df_adj[df_adj['year'] == year].sort_values('date')
        # Get data up to the same day of year
        year_data_to_doy = year_data[year_data['day_of_year'] <= day_of_year]
        if len(year_data_to_doy) > 0:
            # Use trailing 30 days (or available)
            lookback = min(30, len(year_data_to_doy))
            trailing_by_year[year] = year_data_to_doy.tail(lookback)['anomaly'].mean()
            ytd_by_year[year] = year_data_to_doy['anomaly'].mean()

    annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)
    annual['ytd_anomaly'] = annual['year'].map(ytd_by_year)

    # Get prior year anomaly
    prior_year = annual[annual['year'] == current_year - 1]
    if len(prior_year) == 0:
        return None
    prior_year_anomaly = prior_year['annual_anomaly'].values[0]

    # Calculate split ENSO features (obs vs future)
    if len(enso_df) > 0:
        enso_historical = enso_df[~enso_df['is_forecast']] if 'is_forecast' in enso_df.columns else enso_df
        enso_obs_by_year = {}
        enso_future_by_year = {}
        for year in annual['year'].unique():
            yr_enso = enso_historical[enso_historical['year'] == year]
            obs_months = yr_enso[yr_enso['month'] <= current_month]
            fut_months = yr_enso[yr_enso['month'] > current_month]
            enso_obs_by_year[year] = obs_months['oni'].mean() if len(obs_months) > 0 else np.nan
            enso_future_by_year[year] = fut_months['oni'].mean() if len(fut_months) > 0 else np.nan
        annual['enso_obs'] = annual['year'].map(enso_obs_by_year)
        annual['enso_future'] = annual['year'].map(enso_future_by_year)

        # Current year ENSO values
        enso_ytd = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] <= current_month)]
        enso_future_months = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] > current_month)]
        enso_obs_val = enso_ytd['oni'].mean() if len(enso_ytd) > 0 else 0
        enso_future_val = enso_future_months['oni'].mean() if len(enso_future_months) > 0 else 0
    else:
        annual['enso_obs'] = 0
        annual['enso_future'] = 0
        enso_obs_val = 0
        enso_future_val = 0

    # Train model with trailing anomaly as a feature
    volcanic_years = [1982, 1983, 1991, 1992, 1993]
    train_df = annual[
        (annual['year'] >= 1950) &
        (annual['year'] < current_year) &
        (~annual['year'].isin(volcanic_years)) &
        (annual['prior_year_anomaly'].notna()) &
        (annual['trailing_anomaly'].notna()) &
        (annual['ytd_anomaly'].notna()) &
        (annual['enso_obs'].notna()) &
        (annual['enso_future'].notna())
    ].copy()

    if len(train_df) < 10:
        return None

    feature_cols = ['year', 'prior_year_anomaly', 'enso_obs', 'enso_future', 'trailing_anomaly', 'ytd_anomaly']
    X = train_df[feature_cols].values
    y = train_df['annual_anomaly'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Calculate uncertainty from residuals
    residuals = y - model.predict(X_scaled)
    uncertainty = max(np.std(residuals), 0.02)

    # Make prediction for current year
    X_pred = np.array([[current_year, prior_year_anomaly, enso_obs_val, enso_future_val, trailing_30d_anomaly, ytd_anomaly]])
    X_pred_scaled = scaler.transform(X_pred)
    prediction = model.predict(X_pred_scaled)[0]

    days_elapsed = len(ytd_data)

    return {
        'date': target_date,
        'prediction': prediction,
        'uncertainty': uncertainty,
        'ytd_anomaly': ytd_anomaly,
        'trailing_30d_anomaly': trailing_30d_anomaly,
        'days_elapsed': days_elapsed
    }


def load_and_update_projection_history(df: pd.DataFrame, enso_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load projection history from file and update with any missing days.

    Projections are never modified after being made - only new days are added.
    """
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    current_year = df['year'].max()
    history_file = DATA_DIR / f"projection_history_{current_year}.csv"

    # Load existing history
    if history_file.exists():
        history = pd.read_csv(history_file)
        history['date'] = pd.to_datetime(history['date'])
    else:
        history = pd.DataFrame(columns=['date', 'prediction', 'uncertainty', 'ytd_anomaly', 'trailing_30d_anomaly', 'days_elapsed'])

    # Get the latest date in data
    latest_date = df['date'].max()

    # Generate all dates from start of year to latest date
    start_of_year = pd.Timestamp(f"{current_year}-01-01")
    all_dates = pd.date_range(start=start_of_year, end=latest_date, freq='D')

    # Find missing dates
    existing_dates = set(history['date'].dt.date) if len(history) > 0 else set()

    new_projections = []
    for date in all_dates:
        if date.date() not in existing_dates:
            # Calculate projection for this date
            proj = calculate_projection_for_date(df, date, enso_df)
            if proj is not None:
                new_projections.append(proj)

    # Add new projections to history
    if new_projections:
        new_df = pd.DataFrame(new_projections)
        history = pd.concat([history, new_df], ignore_index=True)
        history = history.sort_values('date').reset_index(drop=True)

        # Save updated history
        history.to_csv(history_file, index=False)

    return history


def create_projection_history_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """
    Create a plot showing how the annual projection has evolved throughout the year.
    """
    import os
    theme = get_theme(dark_mode)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    current_year = df['year'].max()
    history_file = DATA_DIR / f"projection_history_{current_year}.csv"

    # In production, skip heavy computation if history file doesn't exist
    # The cron job will generate it
    if not history_file.exists():
        fig = go.Figure()
        fig.update_layout(
            annotations=[{
                'text': 'Projection history will be available after the daily update runs.',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5, 'showarrow': False,
                'font': {'size': 16, 'color': theme['text_color']}
            }],
            height=500,
            template=theme['template'],
        )
        return fig

    # Load existing history (don't generate new projections in web request)
    history = pd.read_csv(history_file)
    history['date'] = pd.to_datetime(history['date'])

    if len(history) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            annotations=[{
                'text': 'No projection history available',
                'xref': 'paper', 'yref': 'paper',
                'x': 0.5, 'y': 0.5, 'showarrow': False,
            }],
            template=theme['template'],
        )
        return fig

    current_year = df['year'].max()

    fig = go.Figure()

    # Add uncertainty band (accent color at low alpha)
    fig.add_trace(go.Scatter(
        x=pd.concat([history['date'], history['date'][::-1]]),
        y=pd.concat([history['prediction'] + 2*history['uncertainty'],
                     (history['prediction'] - 2*history['uncertainty'])[::-1]]),
        fill='toself',
        fillcolor=_hex_to_rgba(theme['prediction_color'], 0.13),
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI',
        hoverinfo='skip'
    ))

    # Add projection line
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['prediction'],
        mode='lines',
        name=f'{current_year} Projection',
        line=dict(color=theme['prediction_color'], width=2.5),
        hovertemplate='%{x|%b %d}<br>Projection: %{y:.2f}°C<extra></extra>'
    ))

    # Add YTD actual line
    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['ytd_anomaly'],
        mode='lines',
        name=f'{current_year} YTD',
        line=dict(color=theme['ytd_color'], width=2, dash='dot'),
        hovertemplate='%{x|%b %d}<br>YTD: %{y:.2f}°C<extra></extra>'
    ))

    # Add ENSO forecast update markers
    enso_updates_file = DATA_DIR / "enso_forecast_updates.csv"
    if enso_updates_file.exists():
        updates_df = pd.read_csv(enso_updates_file)
        updates_df['date'] = pd.to_datetime(updates_df['date'])
        history_start = history['date'].min()
        history_end = history['date'].max()
        relevant = updates_df[
            (updates_df['date'] >= history_start) &
            (updates_df['date'] <= history_end)
        ]
        if len(relevant) > 0:
            marker_x, marker_y = [], []
            for upd_date in relevant['date']:
                match = history[history['date'].dt.date == upd_date.date()]
                if len(match) > 0:
                    pred = match['prediction'].values[0]
                    # Only show marker if projection shifted >0.01°C vs prior day
                    prev = history[history['date'].dt.date == (upd_date - pd.Timedelta(days=1)).date()]
                    if len(prev) > 0 and abs(pred - prev['prediction'].values[0]) <= 0.01:
                        continue
                    marker_x.append(upd_date)
                    marker_y.append(pred)
            if marker_x:
                fig.add_trace(go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode='markers',
                    name='Major ENSO Forecast Update',
                    marker=dict(
                        symbol='star',
                        size=13,
                        color=theme['enso_update_color'],
                        line=dict(width=1, color=theme['text_color'])
                    ),
                    hovertemplate='%{x|%b %d}<br>Major ENSO Forecast Update<br>Projection: %{y:.2f}°C<extra></extra>'
                ))
                # Label the first star so the symbol is self-explanatory
                fig.add_annotation(
                    x=marker_x[0], y=marker_y[0],
                    text='ENSO forecast<br>update',
                    font=dict(size=10.5, color=theme['enso_update_color']),
                    showarrow=False, yanchor='top', yshift=-12, align='center',
                )

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Calculate y-axis range to encompass error bars with padding
    y_min = min(
        (history['prediction'] - 2*history['uncertainty']).min(),
        history['ytd_anomaly'].min(),
        1.45  # Just below 1.5°C line
    )
    y_max = max(
        (history['prediction'] + 2*history['uncertainty']).max(),
        history['ytd_anomaly'].max(),
        1.55  # Just above 1.5°C line
    )
    y_padding = (y_max - y_min) * 0.05

    # Direct end-of-line labels replace the legend
    last_row = history.sort_values('date').iloc[-1]
    fig.add_annotation(
        x=last_row['date'], y=last_row['prediction'],
        text=f"projection<br>{last_row['prediction']:.2f}°C",
        font=dict(size=11.5, color=theme['prediction_color']),
        showarrow=False, xanchor='left', xshift=8, align='left',
    )
    fig.add_annotation(
        x=last_row['date'], y=last_row['ytd_anomaly'],
        text=f"YTD<br>{last_row['ytd_anomaly']:.2f}°C",
        font=dict(size=11.5, color=theme['ytd_color']),
        showarrow=False, xanchor='left', xshift=8, align='left',
    )

    fig.update_layout(
        xaxis=dict(title='', tickformat='%b %d'),
        yaxis=dict(
            title='Temperature anomaly (°C)',
            range=[y_min - y_padding, y_max + y_padding]
        ),
        hovermode='x unified',
        template=theme['template'],
        showlegend=False,
        height=500,
        margin=dict(r=100),
    )

    return fig


def create_records_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a table of temperature records."""
    # Find record high anomalies by day of year
    current_year = df['year'].max()
    current_year_data = df[df['year'] == current_year]

    records = []
    for _, row in current_year_data.iterrows():
        doy = row['day_of_year']
        historical = df[(df['day_of_year'] == doy) & (df['year'] < current_year)]
        if len(historical) > 0:
            prev_max = historical['temperature'].max()
            prev_max_year = historical.loc[historical['temperature'].idxmax(), 'year']
            if row['temperature'] > prev_max:
                records.append({
                    'Date': row['date'].strftime('%Y-%m-%d'),
                    'Temperature': f"{row['temperature']:.2f}°C",
                    'Previous Record': f"{prev_max:.2f}°C ({int(prev_max_year)})",
                    'Margin': f"+{row['temperature'] - prev_max:.2f}°C"
                })

    return pd.DataFrame(records)


def ordinal(n: int) -> str:
    """Return ordinal string for integer n (1st, 2nd, 3rd, ...)."""
    suffix = 'th' if 11 <= (n % 100) <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def create_statistics_cards(df: pd.DataFrame) -> dict:
    """Calculate key statistics for display with preindustrial baseline."""
    # Adjust anomalies to preindustrial baseline
    df_adj = adjust_anomalies_to_preindustrial(df)

    current_year = df_adj['year'].max()
    latest_date = df_adj['date'].max()
    latest_row = df_adj[df_adj['date'] == latest_date].iloc[0]

    # Year-to-date stats (preindustrial-relative)
    ytd_data = df_adj[df_adj['year'] == current_year]
    ytd_mean_anomaly = ytd_data['anomaly'].mean()

    # Previous year comparison (preindustrial-relative)
    prev_year_data = df_adj[df_adj['year'] == current_year - 1]
    prev_year_mean = prev_year_data['anomaly'].mean()

    # Monthly prediction
    target_month = latest_date.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_name = month_names[target_month - 1]

    pred, err, mtd, days = calculate_monthly_prediction(df_adj, target_month, current_year)

    # Annual prediction using trailing 30-day anomaly as predictor
    annual_pred = None
    annual_err = 0.086  # Default uncertainty
    # Full MC draw distribution for the annual prediction. Populated when the
    # ENSO ensemble Monte Carlo runs successfully; used downstream to compute
    # rank probabilities against the historical record.
    mc_draws = None
    try:
        # Calculate annual stats for model
        annual = df_adj.groupby('year')['anomaly'].mean().reset_index()
        annual = annual.rename(columns={'anomaly': 'annual_anomaly'})
        annual['prior_year_anomaly'] = annual['annual_anomaly'].shift(1)

        # Load ENSO data — prefer multi-model mean forecast
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import DATA_DIR

        enso_df = None
        try:
            from src.enso_plots import load_enso_forecast_data, build_enso_combined
            ef, eo, oni = load_enso_forecast_data()
            enso_df = build_enso_combined(oni, ef, eo)
        except Exception:
            enso_file = DATA_DIR / "enso_combined.csv"
            if enso_file.exists():
                # NB: pandas is imported at module level; a local import here
                # would shadow `pd` for the whole function scope.
                enso_df = pd.read_csv(enso_file)
                enso_df['date'] = pd.to_datetime(enso_df['date'])

        if enso_df is not None and len(enso_df) > 0:

            # Calculate split ENSO features (obs vs future)
            current_month = latest_date.month
            enso_hist = enso_df[~enso_df['is_forecast']] if 'is_forecast' in enso_df.columns else enso_df
            enso_obs_by_year = {}
            enso_future_by_year = {}
            for year in annual['year'].unique():
                yr_enso = enso_hist[enso_hist['year'] == year]
                obs_months = yr_enso[yr_enso['month'] <= current_month]
                fut_months = yr_enso[yr_enso['month'] > current_month]
                enso_obs_by_year[year] = obs_months['oni'].mean() if len(obs_months) > 0 else np.nan
                enso_future_by_year[year] = fut_months['oni'].mean() if len(fut_months) > 0 else np.nan
            annual['enso_obs'] = annual['year'].map(enso_obs_by_year)
            annual['enso_future'] = annual['year'].map(enso_future_by_year)

            # Calculate trailing 30-day anomaly for current year
            day_of_year = latest_date.dayofyear
            ytd_data = df_adj[df_adj['year'] == current_year].sort_values('date')
            days_available = len(ytd_data)
            lookback_days = min(30, days_available)
            current_trailing_30d = ytd_data.tail(lookback_days)['anomaly'].mean() if days_available > 0 else None
            current_ytd_anomaly = ytd_data['anomaly'].mean() if days_available > 0 else None

            # Calculate trailing anomaly and YTD mean at same day-of-year for historical years
            trailing_by_year = {}
            ytd_by_year = {}
            for year in annual['year'].unique():
                if year >= current_year:
                    continue
                year_data = df_adj[df_adj['year'] == year].sort_values('date')
                year_data_to_doy = year_data[year_data['day_of_year'] <= day_of_year]
                if len(year_data_to_doy) > 0:
                    lookback = min(30, len(year_data_to_doy))
                    trailing_by_year[year] = year_data_to_doy.tail(lookback)['anomaly'].mean()
                    ytd_by_year[year] = year_data_to_doy['anomaly'].mean()

            annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)
            annual['ytd_anomaly'] = annual['year'].map(ytd_by_year)

            # Train model with trailing anomaly
            volcanic_years = [1982, 1983, 1991, 1992, 1993]
            train_df = annual[
                (annual['year'] >= 1950) &
                (annual['year'] < current_year) &
                (~annual['year'].isin(volcanic_years)) &
                (annual['prior_year_anomaly'].notna()) &
                (annual['trailing_anomaly'].notna()) &
                (annual['ytd_anomaly'].notna()) &
                (annual['enso_obs'].notna()) &
                (annual['enso_future'].notna())
            ]

            if len(train_df) > 10 and current_trailing_30d is not None:
                from sklearn.preprocessing import StandardScaler

                X = train_df[['year', 'prior_year_anomaly', 'enso_obs', 'enso_future', 'trailing_anomaly', 'ytd_anomaly']].values
                y = train_df['annual_anomaly'].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                # Calculate uncertainty
                residuals = y - model.predict(X_scaled)
                annual_err = max(np.std(residuals), 0.02)

                # Get separate ENSO values for current year
                enso_ytd = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] <= current_month)]
                enso_future_months = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] > current_month)]

                enso_obs_val = enso_ytd['oni'].mean() if len(enso_ytd) > 0 else 0
                enso_future_val = enso_future_months['oni'].mean() if len(enso_future_months) > 0 else 0

                if len(enso_ytd) > 0:
                    # Predict with trailing anomaly and ytd_anomaly
                    X_pred = np.array([[current_year, prev_year_mean, enso_obs_val, enso_future_val, current_trailing_30d, current_ytd_anomaly]])
                    X_pred_scaled = scaler.transform(X_pred)
                    annual_pred = model.predict(X_pred_scaled)[0]

                    # Monte Carlo over ENSO ensemble members → CI half-width
                    try:
                        from src.annual_prediction_mc import (
                            load_enso_future_members, monte_carlo_ci,
                        )
                        members = load_enso_future_members(int(current_year), int(current_month))
                        if not members.empty:
                            mc = monte_carlo_ci(
                                model, scaler,
                                base_features={
                                    'year': current_year,
                                    'prior_year_anomaly': prev_year_mean,
                                    'enso_obs': enso_obs_val,
                                    'enso_future': enso_future_val,
                                    'trailing_anomaly': current_trailing_30d,
                                    'ytd_anomaly': current_ytd_anomaly,
                                },
                                members=members,
                                resid_std=annual_err,
                                feature_order=['year', 'prior_year_anomaly', 'enso_obs',
                                               'enso_future', 'trailing_anomaly', 'ytd_anomaly'],
                            )
                            # Express the (possibly asymmetric) MC CI as a half-width
                            # so existing ±-style display logic stays sensible.
                            annual_err = max((mc['ub'] - mc['lb']) / 4.0, 0.02)
                            mc_draws = np.asarray(mc['draws'])
                    except Exception:
                        pass  # fall back to analytic residual std
    except Exception as e:
        pass  # Use default if prediction fails

    # Monthly ranking vs historical years
    month_rank_str = None
    month_rank_range_str = None
    if pred is not None and err is not None:
        try:
            hist_monthly = df_adj[
                (df_adj['date'].dt.month == target_month) &
                (df_adj['year'] < current_year)
            ]
            hist_month_means = hist_monthly.groupby('year')['anomaly'].mean()
            month_rank = int((hist_month_means > pred).sum()) + 1
            rank_best = int((hist_month_means > pred + err).sum()) + 1   # warmest plausible
            rank_worst = int((hist_month_means > pred - err).sum()) + 1  # coolest plausible
            month_rank_str = ordinal(month_rank)
            month_rank_range_str = (ordinal(rank_best) if rank_best == rank_worst
                                    else f"{ordinal(rank_best)}–{ordinal(rank_worst)}")
        except Exception:
            pass

    # Annual ranking vs historical years
    annual_rank_str = None
    annual_rank_range_str = None
    annual_rank_probs = []
    if annual_pred is not None:
        try:
            hist_annual_means = df_adj[df_adj['year'] < current_year].groupby('year')['anomaly'].mean()
            ci = 2 * annual_err
            annual_rank = int((hist_annual_means > annual_pred).sum()) + 1
            rank_best = int((hist_annual_means > annual_pred + ci).sum()) + 1
            rank_worst = int((hist_annual_means > annual_pred - ci).sum()) + 1
            annual_rank_str = ordinal(annual_rank)
            annual_rank_range_str = (ordinal(rank_best) if rank_best == rank_worst
                                     else f"{ordinal(rank_best)}–{ordinal(rank_worst)}")

            # Rank probabilities from the full MC distribution. For each draw,
            # rank = (# historical years warmer than the draw) + 1, so rank 1
            # is "warmest on record". We tabulate up to RANK_TABLE_DEPTH ranks
            # individually, then bundle anything cooler into a single bucket.
            if mc_draws is not None and len(mc_draws) > 0 and len(hist_annual_means) > 0:
                hist_sorted = np.sort(hist_annual_means.values)  # ascending
                # # historical years above each draw
                n_above = len(hist_sorted) - np.searchsorted(
                    hist_sorted, mc_draws, side='right')
                ranks = (n_above + 1).astype(int)
                n_draws = len(ranks)

                RANK_TABLE_DEPTH = 5
                hist_sorted_desc = np.sort(hist_annual_means.values)[::-1]
                hist_years_desc = hist_annual_means.sort_values(ascending=False).index.tolist()
                for k in range(1, RANK_TABLE_DEPTH + 1):
                    prob = float((ranks == k).mean())
                    if k - 1 < len(hist_years_desc):
                        holder_year = int(hist_years_desc[k - 1])
                        holder_val = float(hist_sorted_desc[k - 1])
                    else:
                        holder_year = None
                        holder_val = None
                    annual_rank_probs.append({
                        'rank': k,
                        'rank_label': "Warmest on record" if k == 1
                                      else f"{ordinal(k)} warmest",
                        'prob': prob,
                        'holder_year': holder_year,
                        'holder_value': holder_val,
                    })
                cooler = float((ranks > RANK_TABLE_DEPTH).mean())
                if cooler > 0.005:
                    annual_rank_probs.append({
                        'rank': RANK_TABLE_DEPTH + 1,
                        'rank_label': f"{ordinal(RANK_TABLE_DEPTH + 1)} or cooler",
                        'prob': cooler,
                        'holder_year': None,
                        'holder_value': None,
                    })
        except Exception:
            pass

    # Daily anomaly rank vs the same *calendar* day in all historical years.
    # Compare by (month, day) rather than day-of-year so leap years line up
    # correctly — DOY 136 is May 15 in a leap year but May 16 otherwise,
    # which lets the per-day climatology shift flip the ranking between
    # absolute and anomaly cards.
    daily_rank_str = None
    try:
        m, d = latest_date.month, latest_date.day
        hist_md = df_adj[(df_adj['date'].dt.month == m)
                         & (df_adj['date'].dt.day == d)
                         & (df_adj['year'] < current_year)]
        if len(hist_md) > 0:
            daily_rank = int((hist_md['anomaly'] > latest_row['anomaly']).sum()) + 1
            daily_rank_str = "Record warmest" if daily_rank == 1 else f"{ordinal(daily_rank)} warmest"
    except Exception:
        pass

    # 365-day rolling mean anomaly (current warming pulse)
    rolling_365_str = None
    try:
        daily = df_adj.sort_values('date').set_index('date')['anomaly']
        r365 = daily.rolling('365D').mean().iloc[-1]
        if pd.notna(r365):
            rolling_365_str = f"{r365:.2f}°C"
    except Exception:
        pass

    return {
        'latest_date': latest_date.strftime('%Y-%m-%d'),
        'latest_temp': f"{latest_row['temperature']:.2f}°C",
        'latest_anomaly': f"{latest_row['anomaly']:+.2f}°C",
        'rolling_365': rolling_365_str,
        'daily_rank': daily_rank_str,
        'ytd_anomaly': f"{ytd_mean_anomaly:+.2f}°C",
        'prev_year_anomaly': f"{prev_year_mean:+.2f}°C",
        'month_name': month_name,
        'month_prediction': f"{pred:+.2f}°C" if pred is not None else "N/A",
        'month_error': f"±{err:.2f}°C" if err is not None else "",
        'month_days': days,
        'month_rank': month_rank_str,
        'month_rank_range': month_rank_range_str,
        'current_year': current_year,
        'annual_prediction': f"{annual_pred:+.2f}°C" if annual_pred is not None else "N/A",
        'annual_error': f"±{2*annual_err:.2f}°C",
        'annual_rank': annual_rank_str,
        'annual_rank_range': annual_rank_range_str,
        'annual_rank_probs': annual_rank_probs,
        'data_status': latest_row['status']
    }


def _build_rank_probability_table(stats: dict, dark_mode: bool = False) -> html.Div:
    """Render the per-rank probability rows for the annual prediction.

    Reads ``stats['annual_rank_probs']`` (populated from the ENSO ensemble
    Monte Carlo). Theming is handled entirely by CSS classes (.ranktable in
    assets/theme.css), so this renders once at startup; the ``dark_mode``
    argument is kept for backward compatibility and ignored.
    """
    rows = stats.get('annual_rank_probs') or []
    if not rows:
        return html.P(
            "Rank probabilities unavailable — ENSO ensemble Monte Carlo "
            "did not run for this update.",
            style={'margin': 0},
        )

    max_prob = max((r.get('prob', 0.0) for r in rows), default=0.0)

    out_rows = []
    for entry in rows:
        prob = entry.get('prob', 0.0)
        prob_pct = prob * 100
        prob_str = f"{prob_pct:.1f}%" if prob_pct >= 0.1 else "<0.1%"
        is_max = prob == max_prob and max_prob > 0

        holder_year = entry.get('holder_year')
        holder_text = (f"{holder_year} ({entry['holder_value']:+.2f}°C)"
                       if holder_year is not None else "—")

        out_rows.append(html.Div([
            html.Div(entry['rank_label'], className="rlabel",
                     style={'fontWeight': '600'} if is_max else None),
            html.Div(holder_text, className="rholder"),
            html.Div(html.Div(className="rfill", style={
                'width': f'{max(prob_pct, 0.6):.1f}%',
                'opacity': 1.0 if is_max else 0.65,
            }), className="rtrack"),
            html.Div(prob_str, className="rpct"),
        ], className="rrow"))

    return html.Div(out_rows, className="ranktable")


def create_dashboard(df: pd.DataFrame) -> Dash:
    """Create the Dash application with dark mode support."""
    import logging
    logger = logging.getLogger(__name__)

    # Make bundled fonts available to kaleido/matplotlib before any figure
    # generation happens.
    install_fonts()

    # Assets folder is at project root, not in src/
    assets_path = Path(__file__).parent.parent / 'assets'

    app = Dash(__name__, external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ], suppress_callback_exceptions=True,
       assets_folder=str(assets_path))

    SITE_URL = "https://climate-dashboard.onrender.com"
    OG_IMAGE = f"{SITE_URL}/assets/images/annual_prediction_light.png"
    OG_TITLE = "Climate Dashboard"
    OG_DESC = (
        "Daily-updated global temperature tracker: ERA5 daily data, "
        "multi-model ENSO forecasts, annual projections, and climate "
        "models vs. observations."
    )

    app.index_string = f'''<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{OG_TITLE}</title>
        <meta property="og:type" content="website" />
        <meta property="og:url" content="{SITE_URL}" />
        <meta property="og:title" content="{OG_TITLE}" />
        <meta property="og:description" content="{OG_DESC}" />
        <meta property="og:image" content="{OG_IMAGE}" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="{OG_TITLE}" />
        <meta name="twitter:description" content="{OG_DESC}" />
        <meta name="twitter:image" content="{OG_IMAGE}" />
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>'''

    # Log data info for debugging
    logger.info(f"Creating dashboard with {len(df)} rows of data")
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Generate static plot images (for fast loading in static mode)
    assets_dir = Path(__file__).parent.parent / 'assets' / 'images'

    # Check if images exist and are recent (within last hour)
    # If not, generate them
    import os
    ridgeline_path = assets_dir / 'ridgeline_dark.png'
    should_generate = not ridgeline_path.exists()

    if not should_generate:
        # Check age of existing images
        age = datetime.now().timestamp() - os.path.getmtime(ridgeline_path)
        should_generate = age > 3600  # Regenerate if older than 1 hour

    # Update projection history with any new days (ERA5 data may be more
    # current than the last pipeline run that committed the history CSV)
    try:
        load_and_update_projection_history(df)
    except Exception as e:
        logger.warning(f"Could not update projection history: {e}")

    if should_generate:
        logger.info("Generating static plot images...")
        generate_all_static_images(df, assets_dir)
    else:
        logger.info("Using existing static plot images")

    stats = create_statistics_cards(df)

    # Store dataframe reference for callbacks (using closure)
    _df = df.copy()

    # Pre-load Models vs. Observations data (CMIP ensembles + observational records)
    _MODELS_AVAILABLE = False
    _cmip3 = pd.DataFrame()
    _cmip5 = pd.DataFrame()
    _cmip6 = pd.DataFrame()
    _obs_models = pd.DataFrame()
    _models_cards = {}
    try:
        from src.models_vs_obs import (
            load_cmip_ensemble, load_obs_data, compute_ensemble_stats,
            compute_model_obs_cards,
            create_models_vs_obs_timeseries as _create_models_timeseries,
            create_trend_explorer as _create_trend_explorer,
            create_trend_histogram_grid as _create_hist_grid,
        )
        # Only CMIP6 (the default) is loaded at startup; CMIP3/5 load on first use.
        _cmip6 = load_cmip_ensemble('cmip6')
        _obs_models = load_obs_data()
        _models_cards = compute_model_obs_cards(_cmip6, _obs_models)
        _MODELS_AVAILABLE = True
        logger.info("CMIP model data loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load CMIP data: {e}")

    # Pre-load ENSO forecast data
    _ENSO_AVAILABLE = False
    _enso_forecast_df = _enso_obs_df = _enso_oni_df = pd.DataFrame()
    _enso_combined_df = pd.DataFrame()
    _enso_cards = {}
    _enso_cards_roni = {}
    try:
        from src.enso_plots import (
            load_enso_forecast_data, compute_enso_cards, build_enso_combined,
            create_enso_mega_plume as _create_enso_mega_plume,
            create_enso_box_distribution as _create_enso_box_distribution,
            create_enso_historical_context as _create_enso_historical_context,
            create_enso_strength_probs as _create_enso_strength_probs,
        )
        _enso_forecast_df, _enso_obs_df, _enso_oni_df = load_enso_forecast_data()
        _enso_cards = compute_enso_cards(_enso_forecast_df, _enso_oni_df, _enso_obs_df)
        _enso_cards_roni = compute_enso_cards(_enso_forecast_df, _enso_oni_df, _enso_obs_df, index_mode="roni")
        _enso_combined_df = build_enso_combined(_enso_oni_df, _enso_forecast_df, _enso_obs_df)
        _ENSO_AVAILABLE = not _enso_forecast_df.empty
        if _ENSO_AVAILABLE:
            logger.info("ENSO forecast data loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load ENSO data: {e}")

    # Lazy-load registry for CMIP3/5 — populated on first use
    _cmip_lazy: dict = {}

    def _get_cmip(gen: str) -> pd.DataFrame:
        """Return CMIP DataFrame for gen, lazy-loading CMIP3/5 on first access."""
        if gen == 'cmip6':
            return _cmip6
        if gen not in _cmip_lazy:
            try:
                _cmip_lazy[gen] = load_cmip_ensemble(gen)
                logger.info(f"Lazy-loaded {gen}")
            except Exception as exc:
                logger.warning(f"Failed to lazy-load {gen}: {exc}")
                _cmip_lazy[gen] = pd.DataFrame()
        return _cmip_lazy[gen]

    # ── Hero data (computed once at startup) ────────────────────────────────
    # Odds at the latest month where all models report (not a seasonal mean —
    # seasonal means blend in observed months and drop short-horizon models).
    _enso_odds = None
    try:
        if _ENSO_AVAILABLE:
            from src.enso_plots import compute_peak_month_odds
            _enso_odds = L.enso_odds_view(compute_peak_month_odds(_enso_forecast_df))
    except Exception as e:
        logger.warning(f"Could not compute ENSO peak odds: {e}")

    _alignment = None
    try:
        if _MODELS_AVAILABLE:
            _alignment = L.models_alignment(_cmip6, _obs_models)
    except Exception as e:
        logger.warning(f"Could not compute model alignment: {e}")

    # Computed record comparison for the ENSO historical caption — always
    # qualified by the record's actual start year, never "all time".
    _enso_record_note = ""
    try:
        if _ENSO_AVAILABLE and not _enso_oni_df.empty and _enso_cards.get('peak_val'):
            oni_valid = _enso_oni_df.dropna(subset=['oni'])
            rec_max = float(oni_valid['oni'].max())
            rec_year = int(pd.to_datetime(
                oni_valid.loc[oni_valid['oni'].idxmax(), 'date']).year)
            rec_start = int(pd.to_datetime(oni_valid['date']).dt.year.min())
            peak = _enso_cards['peak_val']
            if peak > rec_max:
                _enso_record_note = (
                    f" The median peak ({peak:+.1f}°C) would exceed the highest "
                    f"ONI observed since {rec_start} ({rec_max:+.1f}°C in "
                    f"{rec_year}) — treat the upper tail with caution.")
            else:
                _enso_record_note = (
                    f" Highest ONI observed since {rec_start}: {rec_max:+.1f}°C "
                    f"({rec_year}).")
    except Exception as e:
        logger.warning(f"Could not compute ENSO record note: {e}")

    # ── Temperature tab ─────────────────────────────────────────────────────
    if stats.get('annual_prediction', 'N/A') != 'N/A':
        temp_headline = [
            "This year is on track for ",
            html.Span([
                stats['annual_prediction'].lstrip('+'), " ",
                html.Span(stats['annual_error'], className="pm"),
            ], className="num"),
            " above preindustrial.",
        ]
    else:
        temp_headline = ["Tracking global temperature, every day."]
    temp_lede = [
        "Year-to-date: ", html.Strong(stats['ytd_anomaly']), ". ",
        "The projection combines the observed year so far with the ENSO "
        "multi-model ensemble and is refreshed every morning from ERA5.",
    ]

    _enso_state_label, _enso_state_val, _enso_state_when = L.split_enso_state(
        _enso_cards.get('current_state', 'N/A'))

    temp_kpis = L.kpi_row([
        L.kpi(f"Latest day · {stats['latest_date']}", stats['latest_anomaly'],
              ([html.Span(stats['daily_rank'], className='rank'),
                " for this calendar day"]
               if stats.get('daily_rank') else f"Absolute: {stats['latest_temp']}")),
        L.kpi(f"{stats['month_name']} projection",
              [stats['month_prediction'],
               html.Small(f" {stats['month_error']}")],
              ([f"{stats['month_days']} days in · est. rank ",
                html.Span(stats['month_rank'], className='rank')] +
               ([f" ({stats['month_rank_range']})"]
                if stats.get('month_rank_range')
                and stats['month_rank_range'] != stats['month_rank'] else [])
               if stats.get('month_rank') else [f"{stats['month_days']} days in"])),
        L.kpi("365-day mean", stats.get('rolling_365') or "N/A",
              "vs preindustrial 1850–1900"),
        L.kpi("ENSO state", _enso_state_label,
              (f"Niño 3.4 at {_enso_state_val} ({_enso_state_when})"
               if _enso_state_val else "Forecast on the ENSO tab")),
    ])

    tab_global = html.Div(id='tab-content-global', children=[
        L.hero(f"{stats['current_year']} Annual Projection", temp_headline,
               temp_lede, right=L.temp_rank_strip(stats)),
        temp_kpis,
        L.section("01", "Right now", "ERA5 daily · 1940–present",
                  "Daily global mean surface temperature against every year in the "
                  "record. The 365-day running mean is the cleanest single measure "
                  "of where we stand.", [
            L.panel("Global mean anomaly vs preindustrial (1850–1900)",
                    img_id='timeseries-img',
                    img_src='/assets/images/timeseries_dark.png',
                    graph_id='timeseries-plot', graph_height=500, tag="Daily",
                    caption=[
                        "Grey: daily anomalies. The ", html.B("365-day average"),
                        " smooths out weather and the seasonal cycle.",
                    ]),
            L.panel("The year in context — recent years vs the historical range",
                    body=html.Div([
                        html.Img(id='daily-anomalies-img',
                                 src='/assets/images/daily_anomalies_dark.png',
                                 alt='Daily temperature anomalies by year',
                                 style={'width': '100%', 'height': 'auto'}),
                        html.Img(id='daily-temps-img',
                                 src='/assets/images/daily_temps_dark.png',
                                 alt='Daily absolute temperatures by year',
                                 style={'width': '100%', 'height': 'auto',
                                        'display': 'none'}),
                        dcc.Loading(id='loading-daily-anomalies-plot', type='circle',
                                    children=[dcc.Graph(
                                        id='daily-anomalies-plot',
                                        style={'height': '500px', 'display': 'none'},
                                        config={'toImageButtonOptions': {'scale': 3},
                                                'displaylogo': False})]),
                        dcc.Loading(id='loading-daily-absolutes-plot', type='circle',
                                    children=[dcc.Graph(
                                        id='daily-absolutes-plot',
                                        style={'height': '500px', 'display': 'none'},
                                        config={'toImageButtonOptions': {'scale': 3},
                                                'displaylogo': False})]),
                    ]),
                    head_extra=dbc.RadioItems(
                        id='daily-mode-toggle',
                        className='segmented btn-group',
                        inputClassName='btn-check', labelClassName='btn',
                        options=[{'label': 'Anomaly', 'value': 'anomaly'},
                                 {'label': 'Absolute', 'value': 'absolute'}],
                        value='anomaly',
                    ),
                    caption=[
                        "Recent years against era envelopes (5–95th percentile "
                        "per era). Dashed: ", html.B("EC46 46-day forecast"),
                        ". Shaded band: the current month.",
                    ]),
        ], section_id='sec-now'),
        L.section("02", f"Where {stats['current_year']} is heading",
                  "Monte Carlo · refreshed daily",
                  "Statistical projections built from year-to-date observations, "
                  "the seasonal cycle, and the ENSO forecast ensemble. Uncertainty "
                  "narrows as the year fills in.", [
            L.duo(
                L.panel(f"{stats['month_name']} {stats['current_year']} — monthly projection",
                        img_id='monthly-projections-img',
                        img_src='/assets/images/monthly_projections_dark.png',
                        graph_id='monthly-projection', graph_height=460,
                        caption=[f"Projected {stats['month_name']}: ",
                                 html.B(f"{stats['month_prediction']} {stats['month_error']}"),
                                 (f" — est. rank {stats['month_rank']}"
                                  if stats.get('month_rank') else "")]),
                L.panel(f"Annual anomaly & {stats['current_year']} prediction",
                        img_id='annual-prediction-img',
                        img_src='/assets/images/annual_prediction_dark.png',
                        graph_id='annual-prediction', graph_height=460,
                        caption=[f"Projected {stats['current_year']}: ",
                                 html.B(f"{stats['annual_prediction']} {stats['annual_error']}"),
                                 ((f" — est. rank {stats['annual_rank']}" +
                                   (f" ({stats['annual_rank_range']})"
                                    if stats.get('annual_rank_range')
                                    and stats['annual_rank_range'] != stats['annual_rank']
                                    else ""))
                                  if stats.get('annual_rank') else "")]),
            ),
            L.panel(f"{stats['current_year']} ranking probabilities",
                    body=html.Div(_build_rank_probability_table(stats),
                                  style={'padding': '14px 20px'}),
                    caption="Likelihood of each final rank against the historical "
                            "record, from a Monte Carlo over the ENSO multi-model "
                            "ensemble plus regression residuals."),
            L.panel(f"How the {stats['current_year']} projection has evolved",
                    img_id='projection-history-img',
                    img_src='/assets/images/projection_history_dark.png',
                    graph_id='projection-history', graph_height=460, tag="Daily",
                    caption=["Step changes mark ",
                             html.B("new seasonal ENSO forecasts"),
                             "; the band is the 95% interval."]),
        ], section_id='sec-projection'),
        L.section("03", "The long view", "1940–present",
                  "The full daily record in two frames: every day as a pixel, and "
                  "every year as a distribution.", [
            L.panel("Every day since 1940",
                    body=html.Div([
                        html.Img(id='heatmap-anomaly-img',
                                 src='/assets/images/heatmap_anomaly_dark.png',
                                 alt='Daily temperature anomaly heatmap',
                                 style={'width': '100%', 'height': 'auto'}),
                        html.Img(id='heatmap-temp-img',
                                 src='/assets/images/heatmap_temp_dark.png',
                                 alt='Daily absolute temperature heatmap',
                                 style={'width': '100%', 'height': 'auto',
                                        'display': 'none'}),
                        dcc.Loading(id='loading-daily-anomaly-heatmap', type='circle',
                                    children=[dcc.Graph(
                                        id='daily-anomaly-heatmap',
                                        style={'height': '500px', 'display': 'none'},
                                        config={'toImageButtonOptions': {'scale': 3},
                                                'displaylogo': False})]),
                        dcc.Loading(id='loading-daily-temp-heatmap', type='circle',
                                    children=[dcc.Graph(
                                        id='daily-temp-heatmap',
                                        style={'height': '500px', 'display': 'none'},
                                        config={'toImageButtonOptions': {'scale': 3},
                                                'displaylogo': False})]),
                    ]),
                    head_extra=dbc.RadioItems(
                        id='heatmap-mode-toggle',
                        className='segmented btn-group',
                        inputClassName='btn-check', labelClassName='btn',
                        options=[{'label': 'Anomaly', 'value': 'anomaly'},
                                 {'label': 'Absolute', 'value': 'absolute'}],
                        value='anomaly',
                    ),
                    caption="Columns are years, rows are days of the year."),
            html.Div(
                L.panel("The distribution slides warm",
                        img_id='ridgeline-img',
                        img_src='/assets/images/ridgeline_dark.png',
                        caption=["Each ridge is one year's daily anomalies; the "
                                 "current year is dashed."]),
                style={'maxWidth': '880px', 'margin': '0 auto'}),
        ], section_id='sec-history'),
    ])

    # ── ENSO tab ────────────────────────────────────────────────────────────
    _peak_val = _enso_cards.get('peak_val')
    _peak_month = _enso_cards.get('peak_month_label', '')
    if _peak_val is not None and _enso_state_label.startswith('El Niño'):
        _strength_word = ("A major" if _peak_val >= 2.0 else
                          "A moderate" if _peak_val >= 1.0 else "A weak")
        enso_headline = [
            f"{_strength_word} El Niño is under way, forecast to peak near ",
            html.Span(f"{_peak_val:+.1f} °C", className="num"),
            f" in {_peak_month}.",
        ]
    elif _peak_val is not None and _enso_state_label.startswith('La Niña'):
        enso_headline = [
            "La Niña conditions, forecast to reach ",
            html.Span(f"{_peak_val:+.1f} °C", className="num"),
            f" in {_peak_month}.",
        ]
    elif _peak_val is not None:
        enso_headline = [
            "Neutral conditions now; the ensemble points to ",
            html.Span(f"{_peak_val:+.1f} °C", className="num"),
            f" by {_peak_month}.",
        ]
    else:
        enso_headline = ["Seasonal ENSO forecasts, aggregated."]
    enso_lede = [
        "Observed Niño 3.4 is at ", html.Strong(_enso_state_val or "N/A"),
        f" ({_enso_state_when})." if _enso_state_when else ".",
        " Every seasonal forecast system's full ensemble is aggregated with "
        "equal model weighting, updated as new runs arrive.",
    ]

    enso_strip = None
    if _enso_odds:
        enso_strip = L.prob_strip(
            f"Odds for {_enso_odds['month_label']}",
            f"all {_enso_odds['n_models']} models · equal weight",
            _enso_odds['segments'], _enso_odds['legend'])

    enso_kpis = L.kpi_row([
        L.kpi("Current state", _enso_state_label,
              (f"Niño 3.4 at {_enso_state_val} ({_enso_state_when})"
               if _enso_state_val else "N/A"),
              value_id='enso-card-1-value', sub_id='enso-card-1-sub'),
        L.kpi("Forecast peak", _enso_cards.get('max_change_str', 'N/A'),
              _enso_cards.get('max_change_range', 'N/A'),
              value_id='enso-card-2-value', sub_id='enso-card-2-sub'),
        L.kpi((f"P(very strong) · {_enso_odds['month_label']}"
               if _enso_odds else "P(very strong)"),
              (L.fmt_prob(_enso_odds['p_very']) if _enso_odds else "N/A"),
              (f"index ≥ 2.0°C · latest month with all {_enso_odds['n_models']} models"
               if _enso_odds else "multi-model ensemble")),
        L.kpi("Ensemble",
              [f"{_enso_cards.get('n_models', '?')} ", html.Small("models")],
              f"{_enso_cards.get('n_members', '?')} members · CFS, NMME, C3S, CanSIPS"),
    ])

    _enso_index_ctl = [
        html.Span("Index", className="ctllabel"),
        dbc.RadioItems(
            id='enso-index-toggle',
            className='segmented btn-group',
            inputClassName='btn-check', labelClassName='btn',
            options=[{'label': 'ONI', 'value': False},
                     {'label': 'rONI', 'value': True}],
            value=False,
        ),
    ]

    tab_enso = html.Div(id='tab-content-enso', style={'display': 'none'}, children=[
        L.hero("El Niño Watch · Niño 3.4" if _enso_state_label.startswith('El Niño')
               else "La Niña Watch · Niño 3.4" if _enso_state_label.startswith('La Niña')
               else "ENSO Forecast · Niño 3.4",
               enso_headline, enso_lede, right=enso_strip),
        enso_kpis,
        L.section("01", "The forecast", _enso_index_ctl,
                  "Every seasonal forecast system's full ensemble, drawn as one "
                  "plume. The dotted line is the model-equal-weighted median.", [
            L.panel("Niño 3.4 combined forecast plume",
                    img_id='enso-mega-plume-img',
                    img_src='/assets/images/enso_mega_plume_dark.png',
                    graph_id='enso-mega-plume-plot', graph_height=550, tag="Monthly",
                    caption=["Shaded fan: model-weighted ",
                             html.B("5–95th and 25–75th percentile"),
                             " ranges across all members; colored lines are "
                             "per-model ensemble means."]),
        ], section_id='sec-plume'),
        L.section("02", "How strong, when", "Model-equal weighting",
                  "The same ensemble, sliced two ways: the monthly forecast "
                  "distribution, and the seasonal odds of each ENSO category.", [
            L.panel("Monthly forecast distribution",
                    img_id='enso-box-distribution-img',
                    img_src='/assets/images/enso_box_distribution_dark.png',
                    graph_id='enso-box-distribution-plot', graph_height=550,
                    caption="Dots: individual ensemble members, colored by "
                            "model. Boxes span the model-weighted "
                            "interquartile range."),
            L.panel("Strength probabilities by season",
                    img_id='enso-strength-probs-img',
                    img_src='/assets/images/enso_strength_probs_dark.png',
                    graph_id='enso-strength-probs-plot', graph_height=500,
                    caption="Model-weighted category odds per overlapping 3-month "
                            "season; n drops at long leads as fewer systems "
                            "forecast that far out."),
        ], section_id='sec-odds'),
        L.section("03", "In context", "1990–present",
                  "The observed ENSO record with the current forecast appended. "
                  "Red spans are El Niño events, blue La Niña.", [
            L.panel("Historical record and current forecast",
                    img_id='enso-historical-img',
                    img_src='/assets/images/enso_historical_dark.png',
                    graph_id='enso-historical-plot', graph_height=450,
                    caption=["Dotted: multi-model median forecast with the "
                             "25th–75th percentile band." + _enso_record_note]),
        ], section_id='sec-context'),
    ])

    # ── Models tab ──────────────────────────────────────────────────────────
    def _models_hero_parts(gen_label: str, scenario: str, cards: dict,
                           alignment: dict | None):
        """Kicker, headline, lede, and gauge for the Models hero — reused by
        the generation-change callback so all hero copy tracks the selector."""
        kicker = f"{gen_label} · {scenario} · five observational records"
        if alignment:
            headline = [
                "Observations are running ",
                html.Span(alignment['phrase'], className="num"),
                " of the model envelope.",
            ]
        else:
            headline = ["Climate models vs. observations."]
        lede = [
            "Since 1970 the world has warmed at ",
            html.Strong(cards.get('obs_trend_1970', 'N/A')),
            " against a model mean of ",
            html.Strong(cards.get('model_trend_1970', 'N/A')),
            "; over the last 15 years observations have run at ",
            html.Strong(cards.get('obs_trend_15', 'N/A')), ".",
        ]
        gauge = None
        if alignment:
            gauge = L.percentile_gauge(
                f"Observed trend vs {gen_label} ensemble", "1970–present",
                alignment['percentile'],
                f"obs · {ordinal(int(round(alignment['percentile'])))} percentile")
        return kicker, headline, lede, gauge

    models_kicker, models_headline, models_lede, models_gauge = \
        _models_hero_parts('CMIP6', 'SSP2-4.5', _models_cards, _alignment)

    models_kpis = L.kpi_row([
        L.kpi([html.Span("Warming vs preindustrial", id='models-card-1-title')],
              _models_cards.get('obs_warming', 'N/A'),
              (f"Models: {_models_cards.get('model_warming', 'N/A')} "
               f"({_models_cards.get('model_warming_range', 'N/A')})"),
              value_id='models-card-1-value', sub_id='models-card-1-sub'),
        L.kpi("Trend · 1970–present",
              _models_cards.get('obs_trend_1970', 'N/A'),
              (f"Models: {_models_cards.get('model_trend_1970', 'N/A')} "
               f"({_models_cards.get('model_range_1970', 'N/A')})"),
              value_id='models-card-2-value', sub_id='models-card-2-sub'),
        L.kpi([html.Span(f"Trend · {_models_cards.get('start_25', '')}–present",
                         id='models-card-3-title')],
              _models_cards.get('obs_trend_25', 'N/A'),
              (f"Models: {_models_cards.get('model_trend_25', 'N/A')} "
               f"({_models_cards.get('model_range_25', 'N/A')})"),
              value_id='models-card-3-value', sub_id='models-card-3-sub'),
        L.kpi([html.Span(f"Trend · {_models_cards.get('start_15', '')}–present",
                         id='models-card-4-title')],
              _models_cards.get('obs_trend_15', 'N/A'),
              (f"Models: {_models_cards.get('model_trend_15', 'N/A')} "
               f"({_models_cards.get('model_range_15', 'N/A')})"),
              value_id='models-card-4-value', sub_id='models-card-4-sub'),
    ])

    models_controls = html.Div(html.Div([
        html.Span([
            html.Span("Generation", className="ctllabel"),
            dbc.RadioItems(
                id='models-cmip-gen',
                className='segmented btn-group',
                inputClassName='btn-check', labelClassName='btn',
                options=[{'label': 'CMIP6', 'value': 'cmip6'},
                         {'label': 'CMIP5', 'value': 'cmip5'},
                         {'label': 'CMIP3', 'value': 'cmip3'}],
                value='cmip6',
            ),
        ], style={'display': 'inline-flex', 'alignItems': 'center'}),
        html.Span([
            html.Span("Baseline", className="ctllabel"),
            dbc.Select(
                id='models-baseline',
                options=[
                    {'label': '1850–1900 (preindustrial)', 'value': '1850-1900'},
                    {'label': '1951–1980', 'value': '1951-1980'},
                    {'label': '1961–1990', 'value': '1961-1990'},
                    {'label': '1971–2000', 'value': '1971-2000'},
                    {'label': '1981–2010', 'value': '1981-2010'},
                ],
                value='1850-1900',
            ),
        ], style={'display': 'inline-flex', 'alignItems': 'center'}),
        html.Span([
            html.Span("Smoothing", className="ctllabel"),
            dbc.RadioItems(
                id='models-smoothing',
                className='segmented btn-group',
                inputClassName='btn-check', labelClassName='btn',
                options=[{'label': 'Monthly', 'value': 'monthly'},
                         {'label': '12-month', 'value': 'rolling'}],
                value='rolling',
            ),
        ], style={'display': 'inline-flex', 'alignItems': 'center'}),
    ], className="controlbar"), className="block", style={'paddingBottom': '0'})

    tab_models = html.Div(id='tab-content-models', style={'display': 'none'}, children=[
        L.hero(models_kicker, models_headline, models_lede, right=models_gauge,
               kicker_id='models-kicker', headline_id='models-headline',
               lede_id='models-lede', right_id='models-gauge-wrap'),
        models_kpis,
        models_controls,
        L.section("01", "The scorecard",
                  html.Span("1900–2040 · SSP2-4.5", id='models-scorecard-hint'),
                  "Five independent observational records against the model "
                  "ensemble mean and its 5th–95th percentile envelope.", [
            L.panel("Climate models vs observations",
                    img_id='models-timeseries-img',
                    img_src='/assets/images/models_timeseries_dark.png',
                    graph_id='models-timeseries-plot', graph_height=520, tag="Monthly",
                    caption="All series referenced to the selected baseline."),
        ], section_id='sec-scorecard'),
        L.section("02", "Trends, however you slice them", "OLS · to present",
                  "Pick any start year: how does the observed warming rate compare "
                  "with each model's? The histograms show where observations fall "
                  "within the full ensemble for three common windows.", [
            L.panel("Warming trend by start year",
                    img_id='models-trend-explorer-img',
                    img_src='/assets/images/models_trend_explorer_dark.png',
                    graph_id='models-trend-explorer-plot', graph_height=500,
                    caption="Dashed: median observed trend, with the shaded "
                            "range across the five datasets, for every start "
                            "year through 2010."),
            L.panel("Where observations land in the model distribution",
                    img_id='models-histograms-img',
                    img_src='/assets/images/models_histograms_dark.png',
                    graph_id='models-histograms-plot', graph_height=380,
                    caption="Shaded band: range across the five observational "
                            "records; dashed line their median. Panels share "
                            "one x-scale."),
        ], section_id='sec-trends'),
    ])

    # Methodology text for the footer modal
    try:
        _methodology_md = (Path(__file__).parent.parent /
                           'PROJECTION_METHODOLOGY.md').read_text()
    except Exception:
        _methodology_md = "Methodology document unavailable."

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='is-mobile-store', data=False),
        dcc.Store(id='initial-load', data=True),
        dcc.Store(id='active-tab-store', storage_type='session', data='global'),
        html.Div(id='theme-sync', style={'display': 'none'}),

        L.topbar(stats['latest_date']),

        dbc.Tooltip("Light / dark mode", target="dark-mode-switch", placement="bottom"),
        dbc.Tooltip("Static images (fast) / interactive plots",
                    target="interactive-switch", placement="bottom"),

        tab_global,
        tab_enso,
        tab_models,

        L.footer_block(stats['latest_date']),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Projection methodology")),
            dbc.ModalBody(dcc.Markdown(_methodology_md)),
        ], id='methodology-modal', size='lg', scrollable=True, is_open=False),
    ], id='main-container')

    # Clientside callback for mobile detection - sets interactive switch based on device
    app.clientside_callback(
        """
        function(initialLoad) {
            if (!initialLoad) {
                return window.dash_clientside.no_update;
            }
            // Detect mobile based on screen width or user agent
            const isMobile = window.innerWidth < 768 ||
                /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            // Return: [is_mobile_store, interactive_switch_value, initial_load_false]
            // Mobile = static (false), Desktop = interactive (true)
            return [isMobile, !isMobile, false];
        }
        """,
        [Output('is-mobile-store', 'data'),
         Output('interactive-switch', 'value'),
         Output('initial-load', 'data')],
        [Input('initial-load', 'data')]
    )

    # Chained callbacks - graphs load sequentially from top to bottom

    # Theme: toggle body.light so CSS variables drive all chrome colors.
    app.clientside_callback(
        """
        function(dark) {
            document.body.classList.toggle('light', !dark);
            return '';
        }
        """,
        Output('theme-sync', 'children'),
        Input('dark-mode-switch', 'value'),
    )

    # Toggle-cluster icon states (active side gets the accent color)
    @app.callback(
        [Output('sun-icon', 'className'),
         Output('moon-icon', 'className'),
         Output('static-icon', 'className'),
         Output('interactive-icon', 'className')],
        [Input('dark-mode-switch', 'value'),
         Input('interactive-switch', 'value')],
    )
    def update_toggle_icons(dark_mode, interactive):
        return (
            'fas fa-sun' + ('' if dark_mode else ' tgl-on'),
            'fas fa-moon' + (' tgl-on' if dark_mode else ''),
            'fas fa-image' + ('' if interactive else ' tgl-on'),
            'fas fa-chart-line' + (' tgl-on' if interactive else ''),
        )

    # Methodology modal open/close
    @app.callback(
        Output('methodology-modal', 'is_open'),
        Input('methodology-link', 'n_clicks'),
        State('methodology-modal', 'is_open'),
        prevent_initial_call=True,
    )
    def toggle_methodology(n, is_open):
        return not is_open

    _VALID_TABS = {'global', 'enso', 'models'}
    _TAB_ACCENTS = {'global': '', 'enso': 'accent-teal', 'models': 'accent-violet'}
    _NAV_ACTIVE = {'global': 'tab-active-temp', 'enso': 'tab-active-enso',
                   'models': 'tab-active-models'}

    # Callback to switch between tab content divs
    @app.callback(
        [Output('tab-content-global', 'style'),
         Output('tab-content-enso', 'style'),
         Output('tab-content-models', 'style'),
         Output('active-tab-store', 'data'),
         Output('main-container', 'className'),
         Output('nav-global', 'className'),
         Output('nav-enso', 'className'),
         Output('nav-models', 'className'),
         Output('url', 'hash')],
        [Input('nav-global', 'n_clicks'),
         Input('nav-enso', 'n_clicks'),
         Input('nav-models', 'n_clicks'),
         Input('url', 'hash'),
         Input('active-tab-store', 'modified_timestamp')],
        [State('active-tab-store', 'data')],
    )
    def switch_tab(n_global, n_enso, n_models, url_hash, _ts, current_tab):
        triggered = callback_context.triggered[0]['prop_id'].split('.')[0]
        # An explicit nav click always wins.
        if triggered == 'nav-global':
            tab = 'global'
        elif triggered == 'nav-enso':
            tab = 'enso'
        elif triggered == 'nav-models':
            tab = 'models'
        else:
            # URL hash drives both the 'url'-triggered case and the page-load
            # tick (where active-tab-store fires first because of session
            # storage). Without this, an external #enso link is ignored when
            # the visitor has previously landed on a different tab.
            h = (url_hash or '').lstrip('#').lower()
            if h in _VALID_TABS:
                tab = h
            else:
                tab = current_tab or 'global'
        nav_cls = lambda t: ('nav-link ' + _NAV_ACTIVE[t]) if t == tab else 'nav-link'
        return (
            {} if tab == 'global' else {'display': 'none'},
            {} if tab == 'enso' else {'display': 'none'},
            {} if tab == 'models' else {'display': 'none'},
            tab,
            _TAB_ACCENTS[tab],
            nav_cls('global'),
            nav_cls('enso'),
            nav_cls('models'),
            f'#{tab}',
        )

    # Callback to toggle between static images and interactive graphs.
    # The merged panels (daily anomaly/absolute, heatmap anomaly/absolute)
    # additionally gate on their mode toggles.
    @app.callback(
        [
            # Image visibility — Global tab (8)
            Output('timeseries-img', 'style'),
            Output('daily-anomalies-img', 'style'),
            Output('daily-temps-img', 'style'),
            Output('monthly-projections-img', 'style'),
            Output('annual-prediction-img', 'style'),
            Output('projection-history-img', 'style'),
            Output('heatmap-anomaly-img', 'style'),
            Output('heatmap-temp-img', 'style'),
            # Graph visibility — Global tab (8)
            Output('timeseries-plot', 'style'),
            Output('daily-anomalies-plot', 'style'),
            Output('daily-absolutes-plot', 'style'),
            Output('monthly-projection', 'style'),
            Output('annual-prediction', 'style'),
            Output('projection-history', 'style'),
            Output('daily-anomaly-heatmap', 'style'),
            Output('daily-temp-heatmap', 'style'),
            # Image visibility — ENSO tab (4)
            Output('enso-mega-plume-img', 'style'),
            Output('enso-box-distribution-img', 'style'),
            Output('enso-historical-img', 'style'),
            Output('enso-strength-probs-img', 'style'),
            # Graph visibility — ENSO tab (4)
            Output('enso-mega-plume-plot', 'style'),
            Output('enso-box-distribution-plot', 'style'),
            Output('enso-historical-plot', 'style'),
            Output('enso-strength-probs-plot', 'style'),
            # Image visibility — Models tab (3)
            Output('models-timeseries-img', 'style'),
            Output('models-trend-explorer-img', 'style'),
            Output('models-histograms-img', 'style'),
            # Graph visibility — Models tab (3)
            Output('models-timeseries-plot', 'style'),
            Output('models-trend-explorer-plot', 'style'),
            Output('models-histograms-plot', 'style'),
        ],
        [Input('interactive-switch', 'value'),
         Input('daily-mode-toggle', 'value'),
         Input('heatmap-mode-toggle', 'value')],
    )
    def toggle_interactive_mode(interactive, daily_mode, heatmap_mode):
        daily_mode = daily_mode or 'anomaly'
        heatmap_mode = heatmap_mode or 'anomaly'

        def img(show):
            return ({'width': '100%', 'height': 'auto', 'display': 'block'}
                    if show else {'display': 'none'})

        def graph(show, height=500):
            return ({'height': f'{height}px', 'display': 'block'}
                    if show else {'display': 'none'})

        static = not interactive
        return (
            # Images — Global
            img(static),                                       # timeseries
            img(static and daily_mode == 'anomaly'),           # daily anomalies
            img(static and daily_mode == 'absolute'),          # daily temps
            img(static),                                       # monthly proj
            img(static),                                       # annual pred
            img(static),                                       # proj history
            img(static and heatmap_mode == 'anomaly'),         # heatmap anomaly
            img(static and heatmap_mode == 'absolute'),        # heatmap temp
            # Graphs — Global
            graph(interactive),                                # timeseries
            graph(interactive and daily_mode == 'anomaly'),    # daily anomalies
            graph(interactive and daily_mode == 'absolute'),   # daily absolutes
            graph(interactive, 460),                           # monthly proj
            graph(interactive, 460),                           # annual pred
            graph(interactive, 460),                           # proj history
            graph(interactive and heatmap_mode == 'anomaly'),  # heatmap anomaly
            graph(interactive and heatmap_mode == 'absolute'), # heatmap temp
            # Images — ENSO
            img(static), img(static), img(static), img(static),
            # Graphs — ENSO
            graph(interactive, 550),
            graph(interactive, 550),
            graph(interactive, 450),
            graph(interactive, 500),
            # Images — Models
            img(static), img(static), img(static),
            # Graphs — Models
            graph(interactive, 520),
            graph(interactive, 500),
            graph(interactive, 380),
        )

    # Update all static image sources based on dark mode
    @app.callback(
        [
            Output('timeseries-img', 'src'),
            Output('daily-anomalies-img', 'src'),
            Output('daily-temps-img', 'src'),
            Output('monthly-projections-img', 'src'),
            Output('annual-prediction-img', 'src'),
            Output('projection-history-img', 'src'),
            Output('heatmap-anomaly-img', 'src'),
            Output('heatmap-temp-img', 'src'),
            Output('ridgeline-img', 'src'),
            # ENSO tab images
            Output('enso-mega-plume-img', 'src'),
            Output('enso-box-distribution-img', 'src'),
            Output('enso-historical-img', 'src'),
            Output('enso-strength-probs-img', 'src'),
            # Models tab images
            Output('models-timeseries-img', 'src'),
            Output('models-trend-explorer-img', 'src'),
            Output('models-histograms-img', 'src'),
        ],
        [Input('dark-mode-switch', 'value'),
         Input('enso-index-toggle', 'value')],
    )
    def update_image_sources(dark_mode, roni_on):
        mode = 'dark' if dark_mode else 'light'
        enso_idx = 'roni_' if roni_on else ''
        return (
            f'/assets/images/timeseries_{mode}.png',
            f'/assets/images/daily_anomalies_{mode}.png',
            f'/assets/images/daily_temps_{mode}.png',
            f'/assets/images/monthly_projections_{mode}.png',
            f'/assets/images/annual_prediction_{mode}.png',
            f'/assets/images/projection_history_{mode}.png',
            f'/assets/images/heatmap_anomaly_{mode}.png',
            f'/assets/images/heatmap_temp_{mode}.png',
            f'/assets/images/ridgeline_{mode}.png',
            f'/assets/images/enso_mega_plume_{enso_idx}{mode}.png',
            f'/assets/images/enso_box_distribution_{enso_idx}{mode}.png',
            f'/assets/images/enso_historical_{enso_idx}{mode}.png',
            f'/assets/images/enso_strength_probs_{enso_idx}{mode}.png',
            f'/assets/images/models_timeseries_{mode}.png',
            f'/assets/images/models_trend_explorer_{mode}.png',
            f'/assets/images/models_histograms_{mode}.png',
        )

    # Interactive graph callbacks (only run when needed)
    from dash.exceptions import PreventUpdate

    # Graph 1: Time series
    @app.callback(
        Output('timeseries-plot', 'figure'),
        [Input('interactive-switch', 'value'), Input('dark-mode-switch', 'value')]
    )
    def update_timeseries(interactive, dark_mode):
        if not interactive:
            raise PreventUpdate
        return create_time_series_plot(_df, dark_mode)

    # Graph 2: Daily anomalies (chained)
    @app.callback(
        Output('daily-anomalies-plot', 'figure'),
        [Input('timeseries-plot', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_daily_anomalies(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_daily_anomalies_plot(_df, dark_mode)

    # Graph 3: Daily absolutes (chained)
    @app.callback(
        Output('daily-absolutes-plot', 'figure'),
        [Input('daily-anomalies-plot', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_daily_absolutes(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_daily_absolutes_plot(_df, dark_mode)

    # Graph 4: Monthly projection (chained)
    @app.callback(
        Output('monthly-projection', 'figure'),
        [Input('daily-absolutes-plot', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_monthly(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_monthly_projection_plot(_df, dark_mode)

    # Graph 5: Annual prediction (chained)
    @app.callback(
        Output('annual-prediction', 'figure'),
        [Input('monthly-projection', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_annual(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_annual_prediction_plot(_df, dark_mode=dark_mode)

    # Graph 6: Projection history (chained)
    @app.callback(
        Output('projection-history', 'figure'),
        [Input('annual-prediction', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_projection_history(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_projection_history_plot(_df, dark_mode)

    # Graph 7: Anomaly heatmap (chained)
    @app.callback(
        Output('daily-anomaly-heatmap', 'figure'),
        [Input('projection-history', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_heatmap_anomaly(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_daily_heatmap(_df, 'anomaly', dark_mode)

    # Graph 8: Temperature heatmap (chained)
    @app.callback(
        Output('daily-temp-heatmap', 'figure'),
        [Input('daily-anomaly-heatmap', 'figure')],
        [State('dark-mode-switch', 'value'), State('interactive-switch', 'value')]
    )
    def update_heatmap_temp(_, dark_mode, interactive):
        if not interactive:
            raise PreventUpdate
        return create_daily_heatmap(_df, 'temperature', dark_mode)

    # (Spatial tab removed — source files kept locally for future use)

    # ── Models vs. Observations callbacks ─────────────────────────────────────

    from dash import no_update as _no_update

    # Graph 1: Models time series (triggered by interactive switch + controls)
    @app.callback(
        Output('models-timeseries-plot', 'figure'),
        [Input('interactive-switch', 'value'),
         Input('models-cmip-gen', 'value'),
         Input('models-smoothing', 'value'),
         Input('models-baseline', 'value'),
         Input('dark-mode-switch', 'value')],
    )
    def update_models_timeseries(interactive, cmip_gen, smoothing, baseline, dark_mode):
        from dash.exceptions import PreventUpdate
        from src.models_vs_obs import scenario_for
        if not interactive:
            raise PreventUpdate
        if not _MODELS_AVAILABLE or _cmip6.empty:
            return go.Figure()
        try:
            gen = cmip_gen or 'cmip6'
            cmip_df = _get_cmip(gen)
            gen_label = {'cmip3': 'CMIP3', 'cmip5': 'CMIP5', 'cmip6': 'CMIP6'}.get(gen, 'CMIP6')
            rolling = (smoothing == 'rolling')
            bl = baseline or '1850-1900'
            return _create_models_timeseries(cmip_df, _obs_models, rolling, dark_mode,
                                             gen_label, bl, scenario=scenario_for(gen))
        except Exception as e:
            logger.error(f"Models timeseries error: {e}")
            return go.Figure()

    # Graph 2: Trend explorer (chained from timeseries)
    @app.callback(
        Output('models-trend-explorer-plot', 'figure'),
        [Input('models-timeseries-plot', 'figure')],
        [State('models-cmip-gen', 'value'),
         State('dark-mode-switch', 'value'),
         State('interactive-switch', 'value')],
    )
    def update_models_trend_explorer(_, cmip_gen, dark_mode, interactive):
        from dash.exceptions import PreventUpdate
        from src.models_vs_obs import scenario_for
        if not interactive:
            raise PreventUpdate
        if not _MODELS_AVAILABLE or _cmip6.empty:
            return go.Figure()
        try:
            gen = cmip_gen or 'cmip6'
            cmip_df = _get_cmip(gen)
            gen_label = {'cmip3': 'CMIP3', 'cmip5': 'CMIP5', 'cmip6': 'CMIP6'}.get(gen, 'CMIP6')
            return _create_trend_explorer(cmip_df, _obs_models, dark_mode, gen_label,
                                          scenario=scenario_for(gen))
        except Exception as e:
            logger.error(f"Models trend explorer error: {e}")
            return go.Figure()

    # Graph 3: Histogram grid (chained from trend explorer, uses selected gen)
    @app.callback(
        Output('models-histograms-plot', 'figure'),
        [Input('models-trend-explorer-plot', 'figure')],
        [State('dark-mode-switch', 'value'),
         State('interactive-switch', 'value'),
         State('models-cmip-gen', 'value')],
    )
    def update_models_histograms(_, dark_mode, interactive, cmip_gen):
        from dash.exceptions import PreventUpdate
        from src.models_vs_obs import scenario_for
        if not interactive:
            raise PreventUpdate
        if not _MODELS_AVAILABLE or _cmip6.empty:
            return go.Figure()
        try:
            gen = cmip_gen or 'cmip6'
            cmip_df = _get_cmip(gen)
            gen_label = {'cmip3': 'CMIP3', 'cmip5': 'CMIP5', 'cmip6': 'CMIP6'}.get(gen, 'CMIP6')
            return _create_hist_grid(cmip_df, _obs_models, dark_mode, gen_label,
                                     scenario=scenario_for(gen))
        except Exception as e:
            logger.error(f"Models histograms error: {e}")
            return go.Figure()

    # Update cards + hero copy + gauge when CMIP generation changes
    @app.callback(
        [Output('models-card-1-title', 'children'),
         Output('models-card-1-value', 'children'),
         Output('models-card-1-sub', 'children'),
         Output('models-card-2-value', 'children'),
         Output('models-card-2-sub', 'children'),
         Output('models-card-3-title', 'children'),
         Output('models-card-3-value', 'children'),
         Output('models-card-3-sub', 'children'),
         Output('models-card-4-title', 'children'),
         Output('models-card-4-value', 'children'),
         Output('models-card-4-sub', 'children'),
         Output('models-kicker', 'children'),
         Output('models-headline', 'children'),
         Output('models-lede', 'children'),
         Output('models-gauge-wrap', 'children'),
         Output('models-scorecard-hint', 'children')],
        [Input('models-cmip-gen', 'value')],
        prevent_initial_call=True,
    )
    def update_models_cards_from_gen(cmip_gen):
        from dash.exceptions import PreventUpdate
        if not _MODELS_AVAILABLE:
            raise PreventUpdate
        try:
            from src.models_vs_obs import scenario_for
            gen = cmip_gen or 'cmip6'
            label = gen.upper()
            scenario = scenario_for(gen)
            cmip_df = _get_cmip(gen)
            cards = compute_model_obs_cards(cmip_df, _obs_models, cmip_label=label)
            alignment = L.models_alignment(cmip_df, _obs_models)
            kicker, headline, lede, gauge = _models_hero_parts(
                label, scenario, cards, alignment)
            return (
                f"{label} ({scenario}) vs Observed",
                cards.get('obs_warming', 'N/A'),
                f"Models: {cards.get('model_warming', 'N/A')} ({cards.get('model_warming_range', 'N/A')})",
                cards.get('obs_trend_1970', 'N/A'),
                f"Models: {cards.get('model_trend_1970', 'N/A')} ({cards.get('model_range_1970', 'N/A')})",
                f"{cards.get('start_25', '')}–Present Trend",
                cards.get('obs_trend_25', 'N/A'),
                f"Models: {cards.get('model_trend_25', 'N/A')} ({cards.get('model_range_25', 'N/A')})",
                f"{cards.get('start_15', '')}–Present Trend",
                cards.get('obs_trend_15', 'N/A'),
                f"Models: {cards.get('model_trend_15', 'N/A')} ({cards.get('model_range_15', 'N/A')})",
                kicker,
                headline,
                lede,
                gauge,
                f"1900–2040 · {scenario}",
            )
        except Exception as e:
            logger.error(f"Models cards update error: {e}")
            raise PreventUpdate

    # ── ENSO Forecast callbacks ─────────────────────────────────────────────

    # ENSO card values follow the rONI toggle
    @app.callback(
        [Output('enso-card-1-value', 'children'),
         Output('enso-card-1-sub', 'children'),
         Output('enso-card-2-value', 'children'),
         Output('enso-card-2-sub', 'children')],
        [Input('enso-index-toggle', 'value')],
    )
    def update_enso_card_values(roni_on):
        cards = _enso_cards_roni if roni_on else _enso_cards
        label, val, when = L.split_enso_state(cards.get('current_state', 'N/A'))
        idx_name = 'rONI' if roni_on else 'Niño 3.4'
        sub = f"{idx_name} at {val} ({when})" if val else "N/A"
        return (
            label,
            sub,
            cards.get('max_change_str', 'N/A'),
            cards.get('max_change_range', 'N/A'),
        )

    # ENSO Graph 1: Mega Plume (triggered by interactive switch + dark mode)
    @app.callback(
        Output('enso-mega-plume-plot', 'figure'),
        [Input('interactive-switch', 'value'),
         Input('dark-mode-switch', 'value'),
         Input('enso-index-toggle', 'value')],
    )
    def update_enso_mega_plume(interactive, dark_mode, roni_on):
        from dash.exceptions import PreventUpdate
        if not interactive:
            raise PreventUpdate
        if not _ENSO_AVAILABLE:
            return go.Figure()
        try:
            index_mode = 'roni' if roni_on else 'oni'
            return _create_enso_mega_plume(_enso_forecast_df, _enso_obs_df, dark_mode, index_mode=index_mode)
        except Exception as e:
            logger.error(f"ENSO mega plume error: {e}")
            return go.Figure()

    # ENSO Graph 2: Box Distribution (chained from mega plume)
    @app.callback(
        Output('enso-box-distribution-plot', 'figure'),
        [Input('enso-mega-plume-plot', 'figure')],
        [State('dark-mode-switch', 'value'),
         State('interactive-switch', 'value'),
         State('enso-index-toggle', 'value')],
    )
    def update_enso_box_distribution(_, dark_mode, interactive, roni_on):
        from dash.exceptions import PreventUpdate
        if not interactive:
            raise PreventUpdate
        if not _ENSO_AVAILABLE:
            return go.Figure()
        try:
            index_mode = 'roni' if roni_on else 'oni'
            return _create_enso_box_distribution(_enso_forecast_df, dark_mode, index_mode=index_mode)
        except Exception as e:
            logger.error(f"ENSO box distribution error: {e}")
            return go.Figure()

    # ENSO Graph 3: Historical Context (chained from box distribution)
    @app.callback(
        Output('enso-historical-plot', 'figure'),
        [Input('enso-box-distribution-plot', 'figure')],
        [State('dark-mode-switch', 'value'),
         State('interactive-switch', 'value'),
         State('enso-index-toggle', 'value')],
    )
    def update_enso_historical(_, dark_mode, interactive, roni_on):
        from dash.exceptions import PreventUpdate
        if not interactive:
            raise PreventUpdate
        if not _ENSO_AVAILABLE:
            return go.Figure()
        try:
            index_mode = 'roni' if roni_on else 'oni'
            return _create_enso_historical_context(_enso_forecast_df, dark_mode, index_mode=index_mode)
        except Exception as e:
            logger.error(f"ENSO historical context error: {e}")
            return go.Figure()

    # ENSO Graph 4: Strength Probabilities (chained from historical context)
    @app.callback(
        Output('enso-strength-probs-plot', 'figure'),
        [Input('enso-historical-plot', 'figure')],
        [State('dark-mode-switch', 'value'),
         State('interactive-switch', 'value'),
         State('enso-index-toggle', 'value')],
    )
    def update_enso_strength_probs(_, dark_mode, interactive, roni_on):
        from dash.exceptions import PreventUpdate
        if not interactive:
            raise PreventUpdate
        if not _ENSO_AVAILABLE:
            return go.Figure()
        try:
            index_mode = 'roni' if roni_on else 'oni'
            return _create_enso_strength_probs(_enso_forecast_df, dark_mode, index_mode=index_mode)
        except Exception as e:
            logger.error(f"ENSO strength probs error: {e}")
            return go.Figure()

    return app


if __name__ == "__main__":
    # Test the dashboard
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_SOURCES
    from scraper import load_or_fetch_data

    source = DATA_SOURCES["era5_global"]
    df = load_or_fetch_data(source["url"], source["local_file"])

    app = create_dashboard(df)
    app.run(debug=True)
