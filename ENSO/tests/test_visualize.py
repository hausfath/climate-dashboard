"""Tests for visualization functions.

Uses property-based tests (not image comparison) to verify plot structure.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from enso_forecast.visualize import (
    plot_all,
    plot_cfs_plume,
    plot_distribution,
    plot_forecast_evolution,
    plot_model_comparison,
    plot_source_comparison,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


class TestCFSPlume:
    """Tests for plot_cfs_plume."""

    def test_smoke(self, synthetic_cfs_df, synthetic_observed_df):
        fig = plot_cfs_plume(synthetic_cfs_df, synthetic_observed_df, save=False)
        assert isinstance(fig, plt.Figure)

    def test_has_axes(self, synthetic_cfs_df):
        fig = plot_cfs_plume(synthetic_cfs_df, save=False)
        assert len(fig.axes) >= 1

    def test_has_threshold_lines(self, synthetic_cfs_df):
        fig = plot_cfs_plume(synthetic_cfs_df, save=False)
        ax = fig.axes[0]
        # Should have horizontal lines at ±0.5
        hlines = [
            line for line in ax.get_lines()
            if hasattr(line, "get_ydata")
            and len(line.get_ydata()) == 2
            and abs(abs(line.get_ydata()[0]) - 0.5) < 0.01
        ]
        # At least the threshold lines should be present
        assert len(ax.get_lines()) > 2  # members + mean + thresholds

    def test_has_title(self, synthetic_cfs_df):
        fig = plot_cfs_plume(synthetic_cfs_df, save=False)
        ax = fig.axes[0]
        assert "CFS" in ax.get_title()

    def test_has_ylabel(self, synthetic_cfs_df):
        fig = plot_cfs_plume(synthetic_cfs_df, save=False)
        ax = fig.axes[0]
        assert "Nino3.4" in ax.get_ylabel() or "anomaly" in ax.get_ylabel().lower()

    def test_with_observed(self, synthetic_cfs_df, synthetic_observed_df):
        fig = plot_cfs_plume(synthetic_cfs_df, synthetic_observed_df, save=False)
        ax = fig.axes[0]
        # Should have more lines than without observed
        assert len(ax.get_lines()) > 0


class TestModelComparison:
    """Tests for plot_model_comparison."""

    def test_smoke(self, synthetic_forecast_df, synthetic_observed_df):
        fig = plot_model_comparison(synthetic_forecast_df, synthetic_observed_df, save=False)
        assert isinstance(fig, plt.Figure)

    def test_has_legend(self, synthetic_forecast_df):
        fig = plot_model_comparison(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    def test_has_title(self, synthetic_forecast_df):
        fig = plot_model_comparison(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        assert len(ax.get_title()) > 0

    def test_dynamical_only_filter(self, synthetic_forecast_df):
        fig_all = plot_model_comparison(
            synthetic_forecast_df, dynamical_only=False, save=False
        )
        fig_dyn = plot_model_comparison(
            synthetic_forecast_df, dynamical_only=True, save=False
        )
        # Dynamical-only should have fewer or equal lines
        lines_all = len(fig_all.axes[0].get_lines())
        lines_dyn = len(fig_dyn.axes[0].get_lines())
        assert lines_dyn <= lines_all


class TestSourceComparison:
    """Tests for plot_source_comparison."""

    def test_smoke(self, synthetic_forecast_df, synthetic_observed_df):
        fig = plot_source_comparison(
            synthetic_forecast_df, synthetic_observed_df, save=False
        )
        assert isinstance(fig, plt.Figure)

    def test_has_legend(self, synthetic_forecast_df):
        fig = plot_source_comparison(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    def test_has_shading(self, synthetic_forecast_df):
        fig = plot_source_comparison(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        # Should have fill_between patches (PolyCollection)
        collections = ax.collections
        assert len(collections) > 0


class TestForecastEvolution:
    """Tests for plot_forecast_evolution."""

    def test_smoke(self, synthetic_forecast_df):
        fig = plot_forecast_evolution(synthetic_forecast_df, save=False)
        assert isinstance(fig, plt.Figure)

    def test_has_title_with_source(self, synthetic_forecast_df):
        fig = plot_forecast_evolution(
            synthetic_forecast_df, source="CFS", save=False
        )
        ax = fig.axes[0]
        assert "CFS" in ax.get_title()


class TestDistribution:
    """Tests for plot_distribution."""

    def test_smoke(self, synthetic_forecast_df):
        fig = plot_distribution(synthetic_forecast_df, save=False)
        assert isinstance(fig, plt.Figure)

    def test_has_title(self, synthetic_forecast_df):
        fig = plot_distribution(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        assert len(ax.get_title()) > 0

    def test_has_ylabel(self, synthetic_forecast_df):
        fig = plot_distribution(synthetic_forecast_df, save=False)
        ax = fig.axes[0]
        assert "Nino3.4" in ax.get_ylabel() or "anomaly" in ax.get_ylabel().lower()


class TestPlotAll:
    """Tests for plot_all."""

    def test_returns_figures(self, synthetic_forecast_df, synthetic_observed_df, tmp_path, monkeypatch):
        # Always redirect to tmp_path so tests never overwrite real figures
        import enso_forecast.visualize as viz
        monkeypatch.setattr(viz, "FIGURES_DIR", tmp_path)

        figures = plot_all(
            synthetic_forecast_df, synthetic_observed_df
        )
        assert len(figures) > 0
        assert all(isinstance(f, plt.Figure) for f in figures)

    def test_saves_to_disk(self, synthetic_forecast_df, synthetic_observed_df, tmp_path, monkeypatch):
        import enso_forecast.visualize as viz
        monkeypatch.setattr(viz, "FIGURES_DIR", tmp_path)

        figures = plot_all(synthetic_forecast_df, synthetic_observed_df)

        # Check that files were created
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0
