"""Monte Carlo over the multi-model ENSO ensemble for the dashboard's
annual temperature prediction.

The dashboard regression takes two ENSO features:
- ``enso_obs``    — mean ONI for months already observed in the current year
- ``enso_future`` — mean ONI for the remaining months of the year

`enso_obs` is observation-based and has no ensemble uncertainty. `enso_future`
is the model-weighted multi-model median, which collapses the ensemble down
to a single trajectory and therefore omits forecast spread. This module
re-introduces that spread by sampling individual ensemble members (with
equal weight per model so a 130-member forecast doesn't drown out a
10-member one) and propagating each draw through the regression.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DASHBOARD = Path(__file__).resolve().parent.parent
SOURCES = ["CFS", "NMME", "C3S", "CanSIPS"]


def load_enso_future_members(year: int, current_month: int) -> pd.DataFrame:
    """Per-member mean ONI over months ``> current_month`` of ``year``.

    Returns a DataFrame with columns ``model``, ``member_id``, ``enso_future``.
    Only members with a forecast for *every* future month are kept, so the
    per-member mean is comparable across the ensemble.
    """
    sys.path.insert(0, str(DASHBOARD))
    sys.path.insert(0, str(DASHBOARD / "ENSO"))
    from enso_forecast.normalize import load_all_forecasts
    from enso_forecast.visualize import _build_mega_df, _get_forecast_only

    forecast_df = load_all_forecasts(sources=SOURCES)
    mega = _build_mega_df(_get_forecast_only(forecast_df))
    members = mega[mega["member_id"] != "mean"].copy()
    members["period"] = pd.to_datetime(members["target_month"]).dt.to_period("M")

    future_periods = [
        pd.Period(f"{year}-{m:02d}", "M")
        for m in range(current_month + 1, 13)
    ]
    if not future_periods:
        return pd.DataFrame(columns=["model", "member_id", "enso_future"])

    members = members[members["period"].isin(future_periods)]
    members = members.dropna(subset=["nino34_anom"])
    pivot = members.pivot_table(
        index=["model", "member_id"], columns="period",
        values="nino34_anom", aggfunc="first",
    )
    full = pivot.dropna(how="any")
    if full.empty:
        return pd.DataFrame(columns=["model", "member_id", "enso_future"])
    out = full.mean(axis=1).rename("enso_future").reset_index()
    return out


def _model_weights(members: pd.DataFrame) -> np.ndarray:
    """Equal weight per model (1/n_models), split evenly across that model's
    members."""
    counts = members.groupby("model").size()
    n_models = len(counts)
    w = np.array([1.0 / (n_models * counts[m]) for m in members["model"]])
    return w / w.sum()


def monte_carlo_ci(
    model,
    scaler,
    base_features: dict,
    members: pd.DataFrame,
    resid_std: float,
    *,
    feature_order: list[str],
    n_draws: int = 10000,
    seed: int = 0,
) -> dict:
    """Run Monte Carlo over ENSO members + Gaussian regression residuals.

    Parameters
    ----------
    model, scaler
        Fitted sklearn LinearRegression and StandardScaler.
    base_features : dict
        All non-ENSO-future features, keyed by name (e.g. ``year``,
        ``prior_year_anomaly``, ``enso_obs``, ``trailing_anomaly``,
        ``ytd_anomaly``).
    members : DataFrame
        Output of ``load_enso_future_members`` — one row per ensemble
        member with column ``enso_future``.
    resid_std : float
        In-sample residual std of the regression (used as the σ of the
        Gaussian residual draw).
    feature_order : list[str]
        Column ordering expected by ``scaler``/``model``.
    n_draws : int
        Total Monte Carlo draws.

    Returns
    -------
    dict with keys ``median``, ``lb`` (2.5%), ``ub`` (97.5%), ``draws``,
    ``n_members``, ``n_models``.
    """
    if members is None or members.empty:
        raise ValueError("No ENSO members available for Monte Carlo.")

    weights = _model_weights(members)
    enso_future_vals = members["enso_future"].values
    n_models = members["model"].nunique()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(members), size=n_draws, replace=True, p=weights)
    sampled_enso = enso_future_vals[idx]

    # Build feature matrix with sampled enso_future, holding all others fixed
    X = np.tile(
        np.array([base_features[c] for c in feature_order], dtype=float),
        (n_draws, 1),
    )
    enso_idx = feature_order.index("enso_future")
    X[:, enso_idx] = sampled_enso

    X_scaled = scaler.transform(X)
    point = model.predict(X_scaled)
    residuals = rng.normal(0.0, resid_std, size=n_draws)
    draws = point + residuals

    return {
        "median": float(np.median(draws)),
        "lb": float(np.quantile(draws, 0.025)),
        "ub": float(np.quantile(draws, 0.975)),
        "draws": draws,
        "n_members": len(members),
        "n_models": int(n_models),
    }
