"""Climate Dashboard visualizations using Plotly Dash."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


# Theme configurations for light and dark modes
THEME_CONFIG = {
    'light': {
        'template': 'plotly_white',
        'bg_color': 'white',
        'paper_color': 'white',
        'text_color': '#2c3e50',
        'grid_color': 'lightgray',
        'line_color': 'rgba(100, 100, 100, 0.3)',
        'rolling_color': '#d62728',
        'threshold_color': 'orange',
        'highlight_colors': {
            2023: '#1f77b4',
            2024: '#ff7f0e',
            2025: '#d62728',
            2026: '#9467bd',
        },
        'historical_color': '#1f77b4',
        'prediction_color': '#d62728',
        'ytd_color': '#ff7f0e',
        'mtd_color': '#ff7f0e',
        'card_color': 'light',
        'vrect_color': 'lightblue',
        'background_years_color': 'lightgrey',
    },
    'dark': {
        'template': 'plotly_dark',
        'bg_color': '#1a1a2e',
        'paper_color': '#1a1a2e',
        'text_color': '#eaeaea',
        'grid_color': 'rgba(255, 255, 255, 0.1)',
        'line_color': 'rgba(200, 200, 200, 0.2)',
        'rolling_color': '#ff6b6b',
        'threshold_color': '#feca57',
        'highlight_colors': {
            2023: '#4ecdc4',
            2024: '#ff9f43',
            2025: '#ff6b6b',
            2026: '#a29bfe',
        },
        'historical_color': '#4ecdc4',
        'prediction_color': '#ff6b6b',
        'ytd_color': '#ff9f43',
        'mtd_color': '#ff9f43',
        'card_color': 'dark',
        'vrect_color': 'rgba(78, 205, 196, 0.2)',
        'background_years_color': 'rgba(255, 255, 255, 0.15)',
    }
}


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
    # Monthly adjustment values (1991-2020 baseline to preindustrial)
    monthly_adjustments = {
        1: 0.96, 2: 0.96, 3: 0.95, 4: 0.91, 5: 0.87, 6: 0.83,
        7: 0.80, 8: 0.80, 9: 0.81, 10: 0.85, 11: 0.89, 12: 0.93
    }

    # Adjusting the anomalies
    df_adjusted = df.copy()
    df_adjusted[anomaly_col] = df.apply(
        lambda row: row[anomaly_col] + monthly_adjustments[row[date_col].month], axis=1
    )

    return df_adjusted


def create_time_series_plot(df: pd.DataFrame, dark_mode: bool = False) -> go.Figure:
    """Create a time series plot of global temperature anomalies relative to preindustrial."""
    theme = get_theme(dark_mode)

    # Adjust anomalies to preindustrial baseline
    df_adj = adjust_anomalies_to_preindustrial(df)

    fig = go.Figure()

    # Add anomaly line
    fig.add_trace(go.Scatter(
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

    fig.add_trace(go.Scatter(
        x=df_rolling['date'],
        y=df_rolling['rolling_365'],
        mode='lines',
        name='365-day Average',
        line=dict(color=theme['rolling_color'], width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>365-day avg: %{y:.2f}°C<extra></extra>'
    ))

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    fig.update_layout(
        title=dict(
            text='Global Mean Temperature Anomaly vs Preindustrial',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis_title='',
        yaxis_title='Temperature Anomaly (°C)',
        hovermode='x unified',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
    )

    return fig


def get_recent_month_bounds(df: pd.DataFrame) -> tuple:
    """Get the day-of-year bounds for the most recent complete month in the data."""
    latest_date = df['date'].max()
    # Use the previous month if we're early in the current month
    if latest_date.day < 15:
        # Go to previous month
        if latest_date.month == 1:
            month = 12
        else:
            month = latest_date.month - 1
    else:
        month = latest_date.month

    # Calculate day of year for month start and end
    year = 2024  # Use a leap year for day-of-year calculation
    month_start = pd.Timestamp(f"{year}-{month:02d}-01").dayofyear
    if month == 12:
        month_end = 366  # End of year
    else:
        month_end = pd.Timestamp(f"{year}-{month+1:02d}-01").dayofyear

    return month_start, month_end


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

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Plot background years (not highlighted) first
    for year in all_years:
        if year not in years_to_highlight:
            year_data = df_adj[df_adj['year'] == year].sort_values('day_of_year')
            if len(year_data) > 0:
                fig.add_trace(go.Scatter(
                    x=year_data['day_of_year'],
                    y=year_data['anomaly'],
                    mode='lines',
                    name=str(year),
                    line=dict(color=theme['background_years_color'], width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # Plot highlighted years on top
    for year in years_to_highlight:
        year_data = df_adj[df_adj['year'] == year].sort_values('day_of_year')
        if len(year_data) > 0:
            fig.add_trace(go.Scatter(
                x=year_data['day_of_year'],
                y=year_data['anomaly'],
                mode='lines',
                name=str(year),
                line=dict(color=theme['highlight_colors'][year], width=2.5),
                hovertemplate=f'{year}<br>Day %{{x}}<br>Anomaly: %{{y:.2f}}°C<extra></extra>'
            ))

    # Month labels for x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.update_layout(
        title=dict(
            text='Daily Temperature Anomalies vs Preindustrial',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            range=[1, 366],
            gridcolor=theme['grid_color'],
            gridwidth=0.5,
        ),
        yaxis=dict(
            title='Temperature Anomaly (°C)',
            gridcolor=theme['grid_color'],
            gridwidth=0.5,
        ),
        hovermode='x',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
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

    # Plot background years (not highlighted) first
    for year in all_years:
        if year not in years_to_highlight:
            year_data = df[df['year'] == year].sort_values('day_of_year')
            if len(year_data) > 0:
                fig.add_trace(go.Scatter(
                    x=year_data['day_of_year'],
                    y=year_data['temperature'],
                    mode='lines',
                    name=str(year),
                    line=dict(color=theme['background_years_color'], width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # Plot highlighted years on top
    for year in years_to_highlight:
        year_data = df[df['year'] == year].sort_values('day_of_year')
        if len(year_data) > 0:
            fig.add_trace(go.Scatter(
                x=year_data['day_of_year'],
                y=year_data['temperature'],
                mode='lines',
                name=str(year),
                line=dict(color=theme['highlight_colors'][year], width=2.5),
                hovertemplate=f'{year}<br>Day %{{x}}<br>Temp: %{{y:.2f}}°C<extra></extra>'
            ))

    # Month labels for x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig.update_layout(
        title=dict(
            text='Daily Global Mean Temperature',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            range=[1, 366],
            gridcolor=theme['grid_color'],
            gridwidth=0.5,
        ),
        yaxis=dict(
            title='Temperature (°C)',
            gridcolor=theme['grid_color'],
            gridwidth=0.5,
        ),
        hovermode='x',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
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
            name=f'{target_year} Projection',
            marker=dict(color=theme['prediction_color'], size=12, symbol='circle'),
            error_y=dict(
                type='data',
                array=[error],
                visible=True,
                color=theme['prediction_color'],
                thickness=2,
                width=8
            ),
            hovertemplate=f'{target_year} Projection<br>Predicted: {predicted_value:.2f}°C<br>±{error:.2f}°C (2σ)<extra></extra>'
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

    fig.update_layout(
        title=dict(
            text=f'{month_name} Temperature Anomaly vs Preindustrial',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis_title='Year',
        yaxis_title='Temperature Anomaly (°C)',
        hovermode='x',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
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
        cbar_label = 'Temperature Anomaly (°C)'
        title = 'Daily Temperature Anomaly vs Preindustrial'
        hover_template = 'Year: %{x}<br>Day: %{y}<br>Anomaly: %{z:.2f}°C<extra></extra>'
        zmid = 1.0  # Center around ~1°C warming
    else:
        data = df.copy()
        column_to_use = 'temperature'
        cbar_label = 'Temperature (°C)'
        title = 'Daily Global Mean Temperature'
        hover_template = 'Year: %{x}<br>Day: %{y}<br>Temp: %{z:.2f}°C<extra></extra>'
        zmid = 14.5  # Approximate global mean temperature

    # Pivot data: day_of_year as rows, year as columns
    heatmap_data = data.pivot(index='day_of_year', columns='year', values=column_to_use)

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
        colorbar=dict(title=cbar_label, tickfont=dict(color=theme['text_color'])),
        hovertemplate=hover_template
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis=dict(
            title='Year',
            dtick=10
        ),
        yaxis=dict(
            title='',
            tickmode='array',
            tickvals=month_starts,
            ticktext=month_names,
            autorange='reversed'
        ),
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
    )

    return fig


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

    # Load ENSO data if not provided
    if enso_df is None:
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

    # Calculate annual ENSO if available
    if len(enso_df) > 0:
        enso_historical = enso_df[~enso_df['is_forecast']] if 'is_forecast' in enso_df.columns else enso_df
        annual_enso = enso_historical.groupby('year')['oni'].mean().reset_index()
        annual_enso = annual_enso.rename(columns={'oni': 'annual_enso'})
        annual = annual.merge(annual_enso, on='year', how='left')
    else:
        annual['annual_enso'] = 0

    current_year = df_adj['year'].max()
    latest_date = df_adj['date'].max()
    day_of_year = latest_date.dayofyear

    # Get current year's trailing 30-day anomaly
    ytd_data = df_adj[df_adj['year'] == current_year].sort_values('date')
    days_available = len(ytd_data)
    lookback_days = min(30, days_available)
    current_trailing_30d = ytd_data.tail(lookback_days)['anomaly'].mean() if days_available > 0 else None

    # For each historical year, calculate trailing anomaly at same day-of-year
    trailing_by_year = {}
    for year in annual['year'].unique():
        year_data = df_adj[df_adj['year'] == year].sort_values('date')
        year_data_to_doy = year_data[year_data['day_of_year'] <= day_of_year]
        if len(year_data_to_doy) > 0:
            lookback = min(30, len(year_data_to_doy))
            trailing_by_year[year] = year_data_to_doy.tail(lookback)['anomaly'].mean()

    annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)

    # Build prediction model with trailing anomaly as a feature
    # Exclude volcanic years
    volcanic_years = [1982, 1983, 1991, 1992, 1993]
    train_df = annual[
        (annual['year'] >= 1950) &
        (annual['year'] < current_year) &  # Only complete years for training
        (~annual['year'].isin(volcanic_years)) &
        (annual['prior_year_anomaly'].notna()) &
        (annual['trailing_anomaly'].notna()) &
        (annual['annual_enso'].notna())
    ].copy()

    # Train model if we have enough data
    predictions = []
    prediction_2026 = None
    uncertainty = 0.086  # Default uncertainty

    if len(train_df) > 10:
        from sklearn.preprocessing import StandardScaler

        feature_cols = ['year', 'prior_year_anomaly', 'annual_enso', 'trailing_anomaly']
        X = train_df[feature_cols].values
        y = train_df['annual_anomaly'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        # Calculate uncertainty from residuals
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        uncertainty = np.std(residuals)

        # Generate predictions for historical years (for plotting trend line)
        for _, row in train_df.iterrows():
            year = row['year']
            X_pred = np.array([[year, row['prior_year_anomaly'], row['annual_enso'], row['trailing_anomaly']]])
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
            current_month = latest_date.month

            enso_ytd = enso_df[(enso_df['year'] == current_year) &
                               (enso_df['month'] <= current_month)]
            enso_future = enso_df[(enso_df['year'] == current_year) &
                                  (enso_df['month'] > current_month)]

            if len(enso_ytd) > 0 and len(enso_future) > 0:
                fraction = current_month / 12
                enso_val = fraction * enso_ytd['oni'].mean() + (1-fraction) * enso_future['oni'].mean()
            elif len(enso_ytd) > 0:
                enso_val = enso_ytd['oni'].mean()
            else:
                enso_val = 0

            # Get prior year anomaly
            prior_year_row = annual[annual['year'] == current_year - 1]
            if len(prior_year_row) > 0:
                prior_year_anomaly = prior_year_row['annual_anomaly'].values[0]

                X_pred = np.array([[current_year, prior_year_anomaly, enso_val, current_trailing_30d]])
                X_pred_scaled = scaler.transform(X_pred)
                pred = model.predict(X_pred_scaled)[0]

                # Uncertainty decreases as we have more actual data
                days_in_year = 366 if (current_year % 4 == 0 and (current_year % 100 != 0 or current_year % 400 == 0)) else 365
                ytd_fraction = days_available / days_in_year
                adjusted_uncertainty = uncertainty * (1 - ytd_fraction * 0.5)

                predictions.append({
                    'year': current_year,
                    'predicted': pred,
                    'lb': pred - 2 * adjusted_uncertainty,
                    'ub': pred + 2 * adjusted_uncertainty
                })

                prediction_2026 = {
                    'predicted': pred,
                    'lb': pred - 2 * adjusted_uncertainty,
                    'ub': pred + 2 * adjusted_uncertainty
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
        fig.add_trace(go.Scatter(
            x=[current_year],
            y=[prediction_2026['predicted']],
            mode='markers',
            name=f'{current_year} Prediction',
            marker=dict(color=theme['prediction_color'], size=12, symbol='circle'),
            error_y=dict(
                type='data',
                array=[2 * uncertainty],
                visible=True,
                color=theme['prediction_color'],
                thickness=2,
                width=6
            ),
            hovertemplate=f'{current_year} Prediction<br>%{{y:.2f}}°C (±{2*uncertainty:.2f}°C)<extra></extra>'
        ))

    # Add 1.5°C reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color=theme['threshold_color'], opacity=0.7,
                  annotation_text="1.5°C", annotation_position="right")

    # Layout
    fig.update_layout(
        title=dict(
            text='Annual Global Temperature Anomaly vs Preindustrial',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis=dict(
            title='Year',
            range=[1938, current_year + 2],
            dtick=10
        ),
        yaxis=dict(
            title='Temperature Anomaly (°C)',
            range=[-0.2, 1.8]
        ),
        hovermode='x',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
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

    # Load ENSO data if not provided
    if enso_df is None:
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

    # For each historical year, calculate the trailing anomaly at the same day-of-year
    # This becomes a feature in the model
    trailing_by_year = {}
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

    annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)

    # Get prior year anomaly
    prior_year = annual[annual['year'] == current_year - 1]
    if len(prior_year) == 0:
        return None
    prior_year_anomaly = prior_year['annual_anomaly'].values[0]

    # Calculate annual ENSO
    if len(enso_df) > 0:
        enso_historical = enso_df[~enso_df['is_forecast']] if 'is_forecast' in enso_df.columns else enso_df
        annual_enso = enso_historical.groupby('year')['oni'].mean().reset_index()
        annual = annual.merge(annual_enso, on='year', how='left')

        # Calculate weighted ENSO for current year
        enso_ytd = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] <= current_month)]
        enso_future = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] > current_month)]

        if len(enso_ytd) > 0:
            fraction = current_month / 12
            if len(enso_future) > 0:
                weighted_enso = fraction * enso_ytd['oni'].mean() + (1-fraction) * enso_future['oni'].mean()
            else:
                weighted_enso = enso_ytd['oni'].mean()
        else:
            weighted_enso = 0
    else:
        annual['oni'] = 0
        weighted_enso = 0

    # Train model with trailing anomaly as a feature
    volcanic_years = [1982, 1983, 1991, 1992, 1993]
    train_df = annual[
        (annual['year'] >= 1950) &
        (annual['year'] < current_year) &
        (~annual['year'].isin(volcanic_years)) &
        (annual['prior_year_anomaly'].notna()) &
        (annual['trailing_anomaly'].notna()) &
        (annual['oni'].notna() if 'oni' in annual.columns else True)
    ].copy()

    if len(train_df) < 10:
        return None

    # Include trailing_anomaly as a predictor
    if 'oni' in train_df.columns:
        X = train_df[['year', 'prior_year_anomaly', 'oni', 'trailing_anomaly']].values
    else:
        X = train_df[['year', 'prior_year_anomaly', 'trailing_anomaly']].values
    y = train_df['annual_anomaly'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Calculate uncertainty from residuals
    residuals = y - model.predict(X_scaled)
    uncertainty = np.std(residuals)

    # Make prediction for current year using trailing_30d_anomaly as the trailing feature
    if 'oni' in train_df.columns:
        X_pred = np.array([[current_year, prior_year_anomaly, weighted_enso, trailing_30d_anomaly]])
    else:
        X_pred = np.array([[current_year, prior_year_anomaly, trailing_30d_anomaly]])
    X_pred_scaled = scaler.transform(X_pred)
    prediction = model.predict(X_pred_scaled)[0]

    # Uncertainty decreases as we have more actual data through the year
    days_elapsed = len(ytd_data)
    days_in_year = 366 if (current_year % 4 == 0 and (current_year % 100 != 0 or current_year % 400 == 0)) else 365
    ytd_fraction = days_elapsed / days_in_year
    adjusted_uncertainty = uncertainty * (1 - ytd_fraction * 0.5)  # Uncertainty decreases but doesn't go to zero

    return {
        'date': target_date,
        'prediction': prediction,
        'uncertainty': adjusted_uncertainty,
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
    theme = get_theme(dark_mode)

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    # Load ENSO data
    enso_file = DATA_DIR / "enso_combined.csv"
    if enso_file.exists():
        enso_df = pd.read_csv(enso_file)
        enso_df['date'] = pd.to_datetime(enso_df['date'])
    else:
        enso_df = None

    # Load and update projection history
    history = load_and_update_projection_history(df, enso_df)

    if len(history) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No projection history available",
            template=theme['template']
        )
        return fig

    current_year = df['year'].max()

    fig = go.Figure()

    # Add uncertainty band
    fig.add_trace(go.Scatter(
        x=pd.concat([history['date'], history['date'][::-1]]),
        y=pd.concat([history['prediction'] + 2*history['uncertainty'],
                     (history['prediction'] - 2*history['uncertainty'])[::-1]]),
        fill='toself',
        fillcolor=theme['prediction_color'].replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in theme['prediction_color'] else f"rgba(255, 107, 107, 0.2)",
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

    fig.update_layout(
        title=dict(
            text=f'{current_year} Annual Projection Evolution',
            font=dict(size=20, color=theme['text_color'])
        ),
        xaxis=dict(
            title='Date',
            tickformat='%b %d'
        ),
        yaxis=dict(
            title='Temperature Anomaly (°C)',
            range=[y_min - y_padding, y_max + y_padding]
        ),
        hovermode='x unified',
        template=theme['template'],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500,
        paper_bgcolor=theme['paper_color'],
        plot_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'])
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
    try:
        # Calculate annual stats for model
        annual = df_adj.groupby('year')['anomaly'].mean().reset_index()
        annual = annual.rename(columns={'anomaly': 'annual_anomaly'})
        annual['prior_year_anomaly'] = annual['annual_anomaly'].shift(1)

        # Load ENSO data
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import DATA_DIR

        enso_file = DATA_DIR / "enso_combined.csv"
        if enso_file.exists():
            import pandas as pd
            enso_df = pd.read_csv(enso_file)
            enso_df['date'] = pd.to_datetime(enso_df['date'])

            # Get annual ENSO
            enso_hist = enso_df[~enso_df['is_forecast']]
            annual_enso = enso_hist.groupby('year')['oni'].mean().reset_index()
            annual = annual.merge(annual_enso, on='year', how='left')

            # Calculate trailing 30-day anomaly for current year
            day_of_year = latest_date.dayofyear
            ytd_data = df_adj[df_adj['year'] == current_year].sort_values('date')
            days_available = len(ytd_data)
            lookback_days = min(30, days_available)
            current_trailing_30d = ytd_data.tail(lookback_days)['anomaly'].mean() if days_available > 0 else None

            # Calculate trailing anomaly at same day-of-year for historical years
            trailing_by_year = {}
            for year in annual['year'].unique():
                if year >= current_year:
                    continue
                year_data = df_adj[df_adj['year'] == year].sort_values('date')
                year_data_to_doy = year_data[year_data['day_of_year'] <= day_of_year]
                if len(year_data_to_doy) > 0:
                    lookback = min(30, len(year_data_to_doy))
                    trailing_by_year[year] = year_data_to_doy.tail(lookback)['anomaly'].mean()

            annual['trailing_anomaly'] = annual['year'].map(trailing_by_year)

            # Train model with trailing anomaly
            volcanic_years = [1982, 1983, 1991, 1992, 1993]
            train_df = annual[
                (annual['year'] >= 1950) &
                (annual['year'] < current_year) &
                (~annual['year'].isin(volcanic_years)) &
                (annual['prior_year_anomaly'].notna()) &
                (annual['trailing_anomaly'].notna()) &
                (annual['oni'].notna())
            ]

            if len(train_df) > 10 and current_trailing_30d is not None:
                from sklearn.preprocessing import StandardScaler

                X = train_df[['year', 'prior_year_anomaly', 'oni', 'trailing_anomaly']].values
                y = train_df['annual_anomaly'].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LinearRegression()
                model.fit(X_scaled, y)

                # Calculate uncertainty
                residuals = y - model.predict(X_scaled)
                annual_err = np.std(residuals)

                # Get weighted ENSO for current year
                current_month = latest_date.month
                enso_ytd = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] <= current_month)]
                enso_future = enso_df[(enso_df['year'] == current_year) & (enso_df['month'] > current_month)]

                if len(enso_ytd) > 0:
                    fraction = current_month / 12
                    if len(enso_future) > 0:
                        weighted_enso = fraction * enso_ytd['oni'].mean() + (1-fraction) * enso_future['oni'].mean()
                    else:
                        weighted_enso = enso_ytd['oni'].mean()

                    # Predict with trailing anomaly
                    X_pred = np.array([[current_year, prev_year_mean, weighted_enso, current_trailing_30d]])
                    X_pred_scaled = scaler.transform(X_pred)
                    annual_pred = model.predict(X_pred_scaled)[0]

                    # Adjust uncertainty based on days elapsed
                    days_in_year = 366 if (current_year % 4 == 0 and (current_year % 100 != 0 or current_year % 400 == 0)) else 365
                    ytd_fraction = days_available / days_in_year
                    annual_err = annual_err * (1 - ytd_fraction * 0.5)
    except Exception as e:
        pass  # Use default if prediction fails

    return {
        'latest_date': latest_date.strftime('%Y-%m-%d'),
        'latest_temp': f"{latest_row['temperature']:.2f}°C",
        'latest_anomaly': f"{latest_row['anomaly']:+.2f}°C",
        'ytd_anomaly': f"{ytd_mean_anomaly:+.2f}°C",
        'prev_year_anomaly': f"{prev_year_mean:+.2f}°C",
        'month_name': month_name,
        'month_prediction': f"{pred:+.2f}°C" if pred is not None else "N/A",
        'month_error': f"±{err:.2f}°C" if err is not None else "",
        'month_days': days,
        'current_year': current_year,
        'annual_prediction': f"{annual_pred:+.2f}°C" if annual_pred is not None else "N/A",
        'annual_error': f"±{2*annual_err:.2f}°C",
        'data_status': latest_row['status']
    }


def create_dashboard(df: pd.DataFrame) -> Dash:
    """Create the Dash application with dark mode support."""
    import logging
    logger = logging.getLogger(__name__)

    app = Dash(__name__, external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ], suppress_callback_exceptions=True)

    # Log data info for debugging
    logger.info(f"Creating dashboard with {len(df)} rows of data")
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    stats = create_statistics_cards(df)

    # Store dataframe reference for callbacks (using closure)
    _df = df.copy()

    app.layout = dbc.Container([
        # Store for dark mode state
        dcc.Store(id='dark-mode-store', data=False),

        # Header with dark mode toggle
        dbc.Row([
            dbc.Col([
                html.H1("Global Temperature Dashboard", className="text-center my-4", id='main-title'),
                html.P("ERA5 Daily Global Mean 2m Temperature",
                       className="text-center text-muted", id='subtitle'),
            ], md=10),
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-sun me-2", id='sun-icon'),
                    dbc.Switch(
                        id='dark-mode-switch',
                        value=False,
                        className="d-inline-block",
                        style={'transform': 'scale(1.3)'}
                    ),
                    html.I(className="fas fa-moon ms-2", id='moon-icon'),
                ], className="d-flex align-items-center justify-content-end mt-4")
            ], md=2),
        ]),

        # Statistics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Latest Data", className="card-title", id='card-1-title'),
                        html.P(stats['latest_date'], className="card-text fs-5", id='card-1-value'),
                        html.Small(f"Status: {stats['data_status']}", id='card-1-sub')
                    ], id='card-1-body')
                ], id='card-1')
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Latest Anomaly", className="card-title", id='card-2-title'),
                        html.P(stats['latest_anomaly'], className="card-text fs-5", id='card-2-value'),
                        html.Small(f"Absolute: {stats['latest_temp']}", id='card-2-sub')
                    ], id='card-2-body')
                ], id='card-2')
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['month_name']} Projection", className="card-title", id='card-3-title'),
                        html.P(f"{stats['month_prediction']} {stats['month_error']}", className="card-text fs-5", id='card-3-value'),
                        html.Small(f"{stats['month_days']} days of data", id='card-3-sub')
                    ], id='card-3-body')
                ], id='card-3')
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['current_year']} Projection", className="card-title", id='card-4-title'),
                        html.P(f"{stats['annual_prediction']} {stats['annual_error']}", className="card-text fs-5", id='card-4-value'),
                        html.Small(f"YTD: {stats['ytd_anomaly']}", id='card-4-sub')
                    ], id='card-4-body')
                ], id='card-4')
            ], md=3),
        ], className="mb-4"),

        # Main time series plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='timeseries-plot')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Daily anomalies plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='daily-anomalies-plot')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Daily absolutes plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='daily-absolutes-plot')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Monthly projection plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='monthly-projection')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Annual prediction plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='annual-prediction')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Annual projection evolution plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='projection-history')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Daily anomaly heatmap
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='daily-anomaly-heatmap')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Daily temperature heatmap
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='daily-temp-heatmap')
            ], md={'size': 10, 'offset': 1})
        ], className="mb-4"),

        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(id='footer-hr'),
                html.P([
                    "Data source: ",
                    html.A("ECMWF ERA5 Climate Pulse",
                           href="https://climate.copernicus.eu/climate-pulse",
                           target="_blank",
                           id='footer-link'),
                    f" | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ], className="text-center", id='footer-text')
            ])
        ])

    ], fluid=True, id='main-container')

    # Callback to update all plots and styling based on dark mode
    @app.callback(
        [
            Output('timeseries-plot', 'figure'),
            Output('daily-anomalies-plot', 'figure'),
            Output('daily-absolutes-plot', 'figure'),
            Output('monthly-projection', 'figure'),
            Output('annual-prediction', 'figure'),
            Output('projection-history', 'figure'),
            Output('daily-anomaly-heatmap', 'figure'),
            Output('daily-temp-heatmap', 'figure'),
            Output('main-container', 'style'),
            Output('main-title', 'style'),
            Output('subtitle', 'style'),
            Output('card-1', 'color'),
            Output('card-2', 'color'),
            Output('card-3', 'color'),
            Output('card-4', 'color'),
            Output('card-1', 'style'),
            Output('card-2', 'style'),
            Output('card-3', 'style'),
            Output('card-4', 'style'),
            Output('card-1-title', 'style'),
            Output('card-2-title', 'style'),
            Output('card-3-title', 'style'),
            Output('card-4-title', 'style'),
            Output('card-1-value', 'style'),
            Output('card-2-value', 'style'),
            Output('card-3-value', 'style'),
            Output('card-4-value', 'style'),
            Output('card-1-sub', 'style'),
            Output('card-2-sub', 'style'),
            Output('card-3-sub', 'style'),
            Output('card-4-sub', 'style'),
            Output('footer-text', 'style'),
            Output('sun-icon', 'style'),
            Output('moon-icon', 'style'),
        ],
        [Input('dark-mode-switch', 'value')]
    )
    def update_theme(dark_mode):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"update_theme callback triggered with dark_mode={dark_mode}")

        # Use the dataframe from closure
        df_cb = _df

        theme = get_theme(dark_mode)

        # Generate all figures with the current theme, with error handling
        try:
            logger.info("Creating timeseries plot...")
            fig_timeseries = create_time_series_plot(df_cb, dark_mode)
            logger.info("Creating daily anomalies plot...")
            fig_daily_anomalies = create_daily_anomalies_plot(df_cb, dark_mode)
            logger.info("Creating daily absolutes plot...")
            fig_daily_absolutes = create_daily_absolutes_plot(df_cb, dark_mode)
            logger.info("Creating monthly projection plot...")
            fig_monthly = create_monthly_projection_plot(df_cb, dark_mode)
            logger.info("Creating annual prediction plot...")
            fig_annual = create_annual_prediction_plot(df_cb, dark_mode=dark_mode)
            logger.info("Creating projection history plot...")
            fig_projection_history = create_projection_history_plot(df_cb, dark_mode)
            logger.info("Creating heatmap plots...")
            fig_heatmap_anomaly = create_daily_heatmap(df_cb, 'anomaly', dark_mode)
            fig_heatmap_temp = create_daily_heatmap(df_cb, 'temperature', dark_mode)
            logger.info("All plots created successfully")
        except Exception as e:
            logger.error(f"Error creating plots: {e}", exc_info=True)
            # Create empty figures as fallback
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Error loading plot")
            fig_timeseries = empty_fig
            fig_daily_anomalies = empty_fig
            fig_daily_absolutes = empty_fig
            fig_monthly = empty_fig
            fig_annual = empty_fig
            fig_projection_history = empty_fig
            fig_heatmap_anomaly = empty_fig
            fig_heatmap_temp = empty_fig

        # Container style
        container_style = {
            'backgroundColor': theme['bg_color'],
            'minHeight': '100vh',
            'paddingBottom': '20px'
        }

        # Title styles
        title_style = {'color': theme['text_color']}
        subtitle_style = {'color': theme['text_color'], 'opacity': '0.7'}

        # Card color and style
        card_color = theme['card_color']
        card_style = {'backgroundColor': theme['card_color'] if dark_mode else None}
        card_title_style = {'color': theme['text_color']}
        card_value_style = {'color': theme['text_color']}
        card_sub_style = {'color': theme['text_color'], 'opacity': '0.6'}

        # Footer style
        footer_style = {'color': theme['text_color'], 'opacity': '0.7'}

        # Icon styles
        sun_style = {'color': '#feca57' if not dark_mode else theme['text_color'], 'opacity': 0.5 if dark_mode else 1}
        moon_style = {'color': '#a29bfe' if dark_mode else theme['text_color'], 'opacity': 1 if dark_mode else 0.5}

        return (
            fig_timeseries,
            fig_daily_anomalies,
            fig_daily_absolutes,
            fig_monthly,
            fig_annual,
            fig_projection_history,
            fig_heatmap_anomaly,
            fig_heatmap_temp,
            container_style,
            title_style,
            subtitle_style,
            card_color,
            card_color,
            card_color,
            card_color,
            card_style,
            card_style,
            card_style,
            card_style,
            card_title_style,
            card_title_style,
            card_title_style,
            card_title_style,
            card_value_style,
            card_value_style,
            card_value_style,
            card_value_style,
            card_sub_style,
            card_sub_style,
            card_sub_style,
            card_sub_style,
            footer_style,
            sun_style,
            moon_style,
        )

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
