#!/usr/bin/env python3
"""CLI entry point for ENSO Forecast Scraping & Visualization Tool."""

import argparse
import logging
import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from enso_forecast.config import FIGURES_DIR
from enso_forecast.normalize import (
    load_all_forecasts,
    load_source_forecasts,
    validate_forecast_df,
)

logger = logging.getLogger("enso_forecast")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_fetch(args):
    """Fetch forecast data from one or all sources."""
    sources = [args.source] if args.source else ["observed", "iri", "cfs", "nmme", "c3s"]
    results = {}
    errors = {}

    for source in sources:
        try:
            if source == "observed":
                from enso_forecast.fetchers.observed import save_observed
                logger.info("=== Fetching observed Nino3.4 ===")
                results["observed"] = save_observed(force=args.force)
                print(f"  Observed: OK")

            elif source == "iri":
                from enso_forecast.fetchers.iri import save_iri
                logger.info("=== Fetching IRI forecasts ===")
                df = save_iri(force=args.force)
                results["iri"] = df
                print(f"  IRI: {df['model'].nunique()} models, {len(df)} records")

            elif source == "cfs":
                from enso_forecast.fetchers.cfs import save_cfs
                logger.info("=== Fetching CFS v2 ensemble ===")
                df = save_cfs(force=args.force)
                results["cfs"] = df
                n_members = df[df["member_id"] != "mean"]["member_id"].nunique() if len(df) > 0 else 0
                print(f"  CFS: {n_members} ensemble members, {len(df)} records")

            elif source == "nmme":
                from enso_forecast.fetchers.nmme import save_nmme
                logger.info("=== Fetching NMME data ===")
                df = save_nmme(force=args.force)
                results["nmme"] = df
                print(f"  NMME: {df['model'].nunique() if len(df) > 0 else 0} models, {len(df)} records")

            elif source == "c3s":
                from enso_forecast.fetchers.c3s import save_c3s
                logger.info("=== Fetching C3S/Copernicus data ===")
                df = save_c3s(force=args.force)
                results["c3s"] = df
                print(f"  C3S: {df['model'].nunique() if len(df) > 0 else 0} models, {len(df)} records")

            else:
                logger.warning("Unknown source: %s", source)

        except Exception as e:
            errors[source] = str(e)
            logger.error("Failed to fetch %s: %s", source, e, exc_info=True)
            print(f"  {source.upper()}: FAILED - {e}")

    # Summary
    print(f"\nFetch complete: {len(results)} succeeded, {len(errors)} failed")
    if errors:
        print("Failures:")
        for src, err in errors.items():
            print(f"  {src}: {err}")

    # Validate
    for name, data in results.items():
        if isinstance(data, dict):
            continue  # skip observed (dict of DataFrames)
        if len(data) > 0:
            issues = validate_forecast_df(data)
            if issues:
                print(f"\nValidation issues for {name}:")
                for issue in issues:
                    print(f"  - {issue}")


def cmd_plot(args):
    """Generate forecast plots."""
    from enso_forecast.fetchers.observed import get_recent_observed
    from enso_forecast.visualize import (
        plot_all,
        plot_c3s_plume,
        plot_cfs_plume,
        plot_distribution,
        plot_forecast_evolution,
        plot_model_comparison,
        plot_source_comparison,
    )

    fetch_date = args.date

    # Determine which sources to include
    plot_sources = [s for s in ["CFS", "NMME", "C3S"] if not args.include_iri]
    if args.include_iri:
        plot_sources = ["IRI", "CFS", "NMME", "C3S"]

    # Load data
    print("Loading forecast data...")
    forecast_df = load_all_forecasts(
        fetch_date,
        sources=plot_sources,
        init_month=args.init_month,
    )
    if forecast_df.empty:
        print("No forecast data available. Run 'python main.py fetch' first.")
        sys.exit(1)

    print(f"Loaded {len(forecast_df)} records from {forecast_df['source'].nunique()} sources")

    # Load observed
    try:
        obs_df = get_recent_observed(n_months=24)
        print(f"Loaded {len(obs_df)} months of observed data")
    except Exception as e:
        logger.warning("Could not load observed data: %s", e)
        obs_df = None

    # Generate plots
    plot_type = args.type

    if plot_type is None or plot_type == "all":
        print("Generating all plots...")
        cfs_df = forecast_df[forecast_df["source"] == "CFS"].copy()
        figures = plot_all(forecast_df, obs_df, cfs_df)
        print(f"Generated {len(figures)} plots in {FIGURES_DIR}")
    else:
        if plot_type == "plume":
            cfs_df = forecast_df[forecast_df["source"] == "CFS"].copy()
            if cfs_df.empty:
                print("No CFS data for plume plot")
                sys.exit(1)
            plot_cfs_plume(cfs_df, obs_df)
            print("Generated CFS plume plot")

        elif plot_type == "c3s_plume":
            c3s_df = forecast_df[forecast_df["source"] == "C3S"].copy()
            if c3s_df.empty:
                print("No C3S data for plume plot")
                sys.exit(1)
            plot_c3s_plume(c3s_df, obs_df)
            print("Generated C3S plume plot")

        elif plot_type == "comparison":
            plot_model_comparison(forecast_df, obs_df)
            print("Generated model comparison plot")

        elif plot_type == "sources":
            plot_source_comparison(forecast_df, obs_df)
            print("Generated source comparison plot")

        elif plot_type == "evolution":
            plot_forecast_evolution(forecast_df, obs_df=obs_df)
            print("Generated forecast evolution plot")

        elif plot_type == "distribution":
            plot_distribution(forecast_df)
            print("Generated distribution plot")

        else:
            print(f"Unknown plot type: {plot_type}")
            print("Available: plume, comparison, sources, evolution, distribution")
            sys.exit(1)

    import matplotlib.pyplot as plt
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="ENSO Forecast Scraping & Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py fetch                    Fetch from all sources
  python main.py fetch --source iri       Fetch only IRI data
  python main.py fetch --force            Re-download even if data exists
  python main.py plot                     Generate all plots
  python main.py plot --type plume        Generate only CFS plume plot
  python main.py plot --date 2026-02-01   Plot using data from specific date
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # fetch subcommand
    fetch_parser = subparsers.add_parser("fetch", help="Fetch latest forecast data")
    fetch_parser.add_argument(
        "--source",
        choices=["observed", "iri", "cfs", "nmme", "c3s"],
        help="Fetch from specific source only",
    )
    fetch_parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if data exists for this month",
    )

    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate forecast plots")
    plot_parser.add_argument(
        "--type",
        choices=["all", "plume", "c3s_plume", "comparison", "sources", "evolution", "distribution"],
        default=None,
        help="Type of plot to generate (default: all)",
    )
    plot_parser.add_argument(
        "--date",
        help="Use data from specific fetch date (YYYY-MM-DD)",
    )
    plot_parser.add_argument(
        "--init-month",
        help="Only include forecasts initialized in this month (YYYY-MM)",
    )
    plot_parser.add_argument(
        "--include-iri",
        action="store_true",
        help="Include IRI data (excluded by default since it lags by ~1 month)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
