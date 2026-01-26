"""
Annual temperature prediction model.

Predicts where the current year's annual temperature will end up using:
- Long-term trend (year)
- Prior year's anomaly (persistence)
- Recent temperature anomaly (past 30 days or available)
- ENSO state year-to-date
- Projected ENSO for remainder of year
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Years to exclude from training (years following major volcanic eruptions)
# El Chichón: March-April 1982 -> affects 1982-1983
# Pinatubo: June 1991 -> affects 1991-1993
VOLCANIC_EXCLUSION_YEARS = [1982, 1983, 1991, 1992, 1993]


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ERA5 temperature and ENSO data."""
    era5_file = data_dir / "era5_daily_series_2t_global.csv"
    enso_file = data_dir / "enso_combined.csv"

    # Load ERA5 data
    era5_df = pd.read_csv(era5_file, comment='#')
    era5_df['date'] = pd.to_datetime(era5_df['date'])
    era5_df['year'] = era5_df['date'].dt.year
    era5_df['month'] = era5_df['date'].dt.month

    # Rename columns
    era5_df = era5_df.rename(columns={
        '2t': 'temperature',
        'clim_91-20': 'climatology',
        'ano_91-20': 'anomaly'
    })

    # Load ENSO data
    enso_df = pd.read_csv(enso_file)
    enso_df['date'] = pd.to_datetime(enso_df['date'])

    return era5_df, enso_df


def apply_preindustrial_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust anomalies to preindustrial baseline."""
    monthly_adjustments = {
        1: 0.96, 2: 0.96, 3: 0.95, 4: 0.91, 5: 0.87, 6: 0.83,
        7: 0.80, 8: 0.80, 9: 0.81, 10: 0.85, 11: 0.89, 12: 0.93
    }

    df = df.copy()
    df['anomaly_pi'] = df.apply(
        lambda row: row['anomaly'] + monthly_adjustments[row['month']], axis=1
    )
    return df


def calculate_annual_stats(era5_df: pd.DataFrame, enso_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annual statistics needed for the model.

    For each year, calculates:
    - Annual mean temperature anomaly (preindustrial-relative)
    - Prior year's anomaly
    - Mean ENSO (ONI) for the year
    """
    # Apply preindustrial adjustment
    era5_df = apply_preindustrial_adjustment(era5_df)

    # Calculate annual means
    annual = era5_df.groupby('year').agg({
        'anomaly_pi': 'mean',
        'temperature': 'mean'
    }).reset_index()
    annual = annual.rename(columns={'anomaly_pi': 'annual_anomaly'})

    # Add prior year anomaly
    annual['prior_year_anomaly'] = annual['annual_anomaly'].shift(1)

    # Calculate annual ENSO mean (excluding forecasts for historical years)
    enso_historical = enso_df[~enso_df['is_forecast']]
    annual_enso = enso_historical.groupby('year')['oni'].mean().reset_index()
    annual_enso = annual_enso.rename(columns={'oni': 'annual_enso'})

    # Merge ENSO data
    annual = annual.merge(annual_enso, on='year', how='left')

    # Add prior year ENSO
    annual['prior_year_enso'] = annual['annual_enso'].shift(1)

    return annual


def calculate_ytd_features(era5_df: pd.DataFrame, enso_df: pd.DataFrame,
                           target_year: int, as_of_date: Optional[datetime] = None) -> dict:
    """
    Calculate year-to-date features for prediction.

    Args:
        era5_df: ERA5 temperature data
        enso_df: ENSO data (historical + forecasts)
        target_year: Year to predict
        as_of_date: Date to use as "current" (defaults to latest in data)

    Returns:
        Dictionary of features for prediction
    """
    # Apply preindustrial adjustment
    era5_df = apply_preindustrial_adjustment(era5_df)

    # Determine the current date from data if not specified
    if as_of_date is None:
        as_of_date = era5_df['date'].max()

    current_doy = as_of_date.timetuple().tm_yday
    current_month = as_of_date.month

    # Get year-to-date temperature data
    ytd_temp = era5_df[(era5_df['year'] == target_year) &
                       (era5_df['date'] <= as_of_date)]

    if len(ytd_temp) == 0:
        raise ValueError(f"No temperature data available for {target_year}")

    # Calculate recent anomaly (past 30 days or available)
    recent_days = min(30, len(ytd_temp))
    recent_anomaly = ytd_temp.tail(recent_days)['anomaly_pi'].mean()

    # Calculate YTD anomaly
    ytd_anomaly = ytd_temp['anomaly_pi'].mean()

    # Get prior year's annual anomaly
    prior_year = target_year - 1
    prior_year_data = era5_df[era5_df['year'] == prior_year]
    prior_year_anomaly = prior_year_data['anomaly_pi'].mean() if len(prior_year_data) > 0 else np.nan

    # Calculate ENSO YTD (historical data for months that have passed)
    enso_ytd = enso_df[(enso_df['year'] == target_year) &
                       (enso_df['month'] <= current_month) &
                       (~enso_df['is_forecast'])]

    # If no historical ENSO data for current year yet, use interpolated/forecast
    if len(enso_ytd) == 0:
        enso_ytd = enso_df[(enso_df['year'] == target_year) &
                          (enso_df['month'] <= current_month)]

    enso_ytd_mean = enso_ytd['oni'].mean() if len(enso_ytd) > 0 else 0

    # Calculate projected ENSO for remainder of year
    remaining_months = list(range(current_month + 1, 13))
    enso_forecast = enso_df[(enso_df['year'] == target_year) &
                           (enso_df['month'].isin(remaining_months))]

    enso_forecast_mean = enso_forecast['oni'].mean() if len(enso_forecast) > 0 else enso_ytd_mean

    # Weight ENSO by fraction of year
    fraction_complete = current_doy / 365
    weighted_enso = (fraction_complete * enso_ytd_mean +
                    (1 - fraction_complete) * enso_forecast_mean)

    return {
        'year': target_year,
        'days_available': len(ytd_temp),
        'fraction_complete': fraction_complete,
        'recent_anomaly': recent_anomaly,
        'ytd_anomaly': ytd_anomaly,
        'prior_year_anomaly': prior_year_anomaly,
        'enso_ytd': enso_ytd_mean,
        'enso_forecast': enso_forecast_mean,
        'weighted_enso': weighted_enso
    }


def build_prediction_model(annual_df: pd.DataFrame,
                           exclude_years: list = VOLCANIC_EXCLUSION_YEARS) -> dict:
    """
    Build the linear regression model for annual temperature prediction.

    Args:
        annual_df: DataFrame with annual statistics
        exclude_years: Years to exclude from training (e.g., volcanic years)

    Returns:
        Dictionary containing model, scaler, and diagnostics
    """
    # Filter training data
    train_df = annual_df[
        (annual_df['year'] >= 1950) &  # Start from 1950 for reliable data
        (~annual_df['year'].isin(exclude_years)) &
        (annual_df['prior_year_anomaly'].notna()) &
        (annual_df['annual_enso'].notna())
    ].copy()

    logger.info(f"Training on {len(train_df)} years (excluding volcanic years: {exclude_years})")

    # Features: year, prior_year_anomaly, annual_enso
    feature_cols = ['year', 'prior_year_anomaly', 'annual_enso']
    X = train_df[feature_cols].values
    y = train_df['annual_anomaly'].values

    # Standardize features for better interpretation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Calculate predictions and residuals
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    # Calculate metrics
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    rmse = np.sqrt(np.mean(residuals ** 2))
    std_residuals = np.std(residuals)

    # Feature importance (standardized coefficients)
    feature_importance = dict(zip(feature_cols, model.coef_))

    logger.info(f"Model R² = {r_squared:.4f}, RMSE = {rmse:.4f}°C")
    logger.info(f"Feature coefficients (standardized): {feature_importance}")

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'r_squared': r_squared,
        'rmse': rmse,
        'std_residuals': std_residuals,
        'feature_importance': feature_importance,
        'training_years': train_df['year'].tolist(),
        'residuals': residuals
    }


def predict_annual_temperature(model_dict: dict,
                               ytd_features: dict,
                               annual_df: pd.DataFrame) -> dict:
    """
    Predict the annual temperature for a target year.

    Uses a two-stage approach:
    1. Model prediction based on year, prior year, and ENSO
    2. Adjustment based on year-to-date observations

    Args:
        model_dict: Output from build_prediction_model
        ytd_features: Output from calculate_ytd_features
        annual_df: Historical annual data for context

    Returns:
        Dictionary with prediction and uncertainty estimates
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_cols = model_dict['feature_cols']
    std_residuals = model_dict['std_residuals']

    target_year = ytd_features['year']
    fraction_complete = ytd_features['fraction_complete']

    # Stage 1: Base model prediction
    # Use weighted ENSO (YTD + forecast for remainder)
    X_pred = np.array([[
        target_year,
        ytd_features['prior_year_anomaly'],
        ytd_features['weighted_enso']
    ]])
    X_pred_scaled = scaler.transform(X_pred)
    base_prediction = model.predict(X_pred_scaled)[0]

    # Stage 2: Blend with YTD observations
    # As more of the year is complete, weight observations more heavily
    ytd_anomaly = ytd_features['ytd_anomaly']

    # Blending weight: start trusting observations more as year progresses
    # Use sqrt to give more weight to observations early
    obs_weight = np.sqrt(fraction_complete)
    model_weight = 1 - obs_weight

    blended_prediction = (obs_weight * ytd_anomaly + model_weight * base_prediction)

    # Uncertainty estimation
    # Base uncertainty from model residuals
    base_uncertainty = std_residuals

    # Uncertainty decreases as more of the year is observed
    # Assume remaining months have similar variance to model error
    remaining_fraction = 1 - fraction_complete
    observation_uncertainty = base_uncertainty * np.sqrt(remaining_fraction)

    # Combined uncertainty (simplified)
    total_uncertainty = np.sqrt(
        (model_weight * base_uncertainty) ** 2 +
        (obs_weight * observation_uncertainty) ** 2
    )

    # 2-sigma bounds
    lower_bound = blended_prediction - 2 * total_uncertainty
    upper_bound = blended_prediction + 2 * total_uncertainty

    # Compare to recent years for context
    recent_years = annual_df[annual_df['year'].isin([2023, 2024, 2025])]

    return {
        'target_year': target_year,
        'prediction': blended_prediction,
        'uncertainty_1sigma': total_uncertainty,
        'uncertainty_2sigma': 2 * total_uncertainty,
        'lower_bound_2sigma': lower_bound,
        'upper_bound_2sigma': upper_bound,
        'base_model_prediction': base_prediction,
        'ytd_anomaly': ytd_anomaly,
        'fraction_complete': fraction_complete,
        'obs_weight': obs_weight,
        'model_weight': model_weight,
        'days_available': ytd_features['days_available'],
        'enso_ytd': ytd_features['enso_ytd'],
        'enso_forecast': ytd_features['enso_forecast'],
        'prior_year_anomaly': ytd_features['prior_year_anomaly'],
        'model_r_squared': model_dict['r_squared'],
        'model_rmse': model_dict['rmse']
    }


def run_prediction(data_dir: Path) -> dict:
    """
    Run the full prediction pipeline.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with prediction results
    """
    # Load data
    logger.info("Loading data...")
    era5_df, enso_df = load_data(data_dir)

    # Calculate annual statistics
    logger.info("Calculating annual statistics...")
    annual_df = calculate_annual_stats(era5_df, enso_df)

    # Build model
    logger.info("Building prediction model...")
    model_dict = build_prediction_model(annual_df)

    # Get current year
    current_year = era5_df['date'].max().year

    # Calculate YTD features
    logger.info(f"Calculating YTD features for {current_year}...")
    ytd_features = calculate_ytd_features(era5_df, enso_df, current_year)

    # Make prediction
    logger.info("Making prediction...")
    prediction = predict_annual_temperature(model_dict, ytd_features, annual_df)

    return {
        'prediction': prediction,
        'model': model_dict,
        'annual_data': annual_df,
        'ytd_features': ytd_features
    }


def print_prediction_report(results: dict) -> None:
    """Print a formatted prediction report."""
    pred = results['prediction']
    model = results['model']
    ytd = results['ytd_features']

    print("\n" + "=" * 60)
    print(f"ANNUAL TEMPERATURE PREDICTION FOR {pred['target_year']}")
    print("=" * 60)

    print(f"\n📊 Model Performance:")
    print(f"   R² = {model['r_squared']:.4f}")
    print(f"   RMSE = {model['rmse']:.3f}°C")
    print(f"   Training years: {len(model['training_years'])} (excl. volcanic)")

    print(f"\n📅 Data Status:")
    print(f"   Days available: {pred['days_available']}")
    print(f"   Year {pred['fraction_complete']*100:.1f}% complete")

    print(f"\n🌡️  Current Observations:")
    print(f"   YTD anomaly: {pred['ytd_anomaly']:+.3f}°C (vs preindustrial)")
    print(f"   Prior year ({pred['target_year']-1}): {pred['prior_year_anomaly']:+.3f}°C")

    print(f"\n🌊 ENSO Status:")
    print(f"   YTD average ONI: {pred['enso_ytd']:+.2f}")
    print(f"   Forecast ONI (remainder): {pred['enso_forecast']:+.2f}")

    print(f"\n🎯 PREDICTION:")
    print(f"   Expected annual anomaly: {pred['prediction']:+.3f}°C")
    print(f"   Uncertainty (±2σ): ±{pred['uncertainty_2sigma']:.3f}°C")
    print(f"   Range: {pred['lower_bound_2sigma']:+.3f}°C to {pred['upper_bound_2sigma']:+.3f}°C")

    print(f"\n📈 Prediction Components:")
    print(f"   Base model prediction: {pred['base_model_prediction']:+.3f}°C")
    print(f"   YTD observation weight: {pred['obs_weight']*100:.1f}%")
    print(f"   Model weight: {pred['model_weight']*100:.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATA_DIR

    results = run_prediction(DATA_DIR)
    print_prediction_report(results)

    # Also show recent years for context
    print("\n📊 Recent Annual Anomalies (vs preindustrial):")
    annual = results['annual_data']
    recent = annual[annual['year'] >= 2020].sort_values('year')
    for _, row in recent.iterrows():
        enso_str = f"ONI={row['annual_enso']:+.2f}" if pd.notna(row['annual_enso']) else "ONI=N/A"
        print(f"   {int(row['year'])}: {row['annual_anomaly']:+.3f}°C ({enso_str})")
