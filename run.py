#!/usr/bin/env python3
"""Main entry point for the Climate Dashboard application."""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_SOURCES, DATA_DIR, DASHBOARD_HOST, DASHBOARD_PORT, DEBUG_MODE
from src.scraper import load_or_fetch_data, fetch_era5_data
from src.dashboard import create_dashboard
from src.enso import create_combined_enso_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_enso_forecasts(force: bool = False) -> None:
    """Fetch multi-model ENSO forecasts (CFS, NMME, C3S, IRI) and observed data."""
    enso_root = str(Path(__file__).parent / "ENSO")
    if enso_root not in sys.path:
        sys.path.insert(0, enso_root)

    from enso_forecast.fetchers.observed import save_observed
    from enso_forecast.fetchers.cfs import save_cfs
    from enso_forecast.fetchers.nmme import save_nmme
    from enso_forecast.fetchers.c3s import save_c3s
    from enso_forecast.fetchers.iri import save_iri
    from enso_forecast.config import FORECASTS_DIR

    sources = [
        ("observed", save_observed),
        ("CFS", save_cfs),
        ("NMME", save_nmme),
        ("C3S", save_c3s),
        ("IRI", save_iri),
    ]

    for name, fetch_fn in sources:
        try:
            # Monthly sources: skip if we already have a file from the current month
            # (C3S in particular takes ~30 min via CDS API)
            if name in ("NMME", "C3S", "IRI") and not force:
                src_dir = FORECASTS_DIR / name
                if src_dir.exists():
                    current_month_prefix = date.today().strftime("%Y-%m")
                    existing = [f.stem for f in src_dir.glob("*.csv")]
                    if any(f.startswith(current_month_prefix) for f in existing):
                        logger.info(f"{name} already fetched this month, skipping")
                        continue

            logger.info(f"Fetching {name} ENSO data...")
            fetch_fn(force=force)

            # Clean up old dated CSVs, keep only the latest
            if name != "observed":  # observed uses fixed filenames
                src_dir = FORECASTS_DIR / name
                csvs = sorted(src_dir.glob("*.csv"))
                if len(csvs) > 1:
                    for old in csvs[:-1]:
                        old.unlink()
                        logger.info(f"Removed old {name} file: {old.name}")

        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")


def update_data(force: bool = False) -> None:
    """Update all data sources."""
    logger.info("Updating data sources...")

    # Update ERA5 data
    for name, source in DATA_SOURCES.items():
        logger.info(f"Updating {name}...")
        if force:
            fetch_era5_data(source["url"], source["local_file"])
        else:
            load_or_fetch_data(source["url"], source["local_file"])

    # Fetch multi-model ENSO forecasts
    logger.info("Fetching ENSO multi-model forecasts...")
    try:
        update_enso_forecasts(force=force)
    except Exception as e:
        logger.error(f"ENSO forecast fetch failed: {e}")

    # Update ENSO data
    logger.info("Updating ENSO data...")
    try:
        import pandas as pd
        enso_file = DATA_DIR / "enso_combined.csv"
        snapshot_file = DATA_DIR / "last_enso_forecast_snapshot.csv"

        create_combined_enso_dataset(enso_file)

        # Build multi-model mean forecast (same data the prediction model uses)
        from src.enso_plots import load_enso_forecast_data, build_enso_combined
        ef, _, oni = load_enso_forecast_data()
        enso_combined = build_enso_combined(oni, ef)

        # Extract forecast-only rows as the snapshot
        new_forecast = enso_combined[enso_combined['is_forecast']][['year', 'month', 'oni']].copy()
        new_forecast['oni'] = new_forecast['oni'].round(3)
        new_forecast = new_forecast.sort_values(['year', 'month']).reset_index(drop=True)

        # Load previous snapshot
        old_snapshot = None
        if snapshot_file.exists():
            old_snapshot = pd.read_csv(snapshot_file)
            old_snapshot['oni'] = old_snapshot['oni'].round(3)
            old_snapshot = old_snapshot.sort_values(['year', 'month']).reset_index(drop=True)

        if old_snapshot is None:
            new_forecast.to_csv(snapshot_file, index=False)
            logger.info("Multi-model ENSO forecast snapshot initialised (first run)")
        else:
            # Compare forecast values; use month order to handle year rollovers
            if len(old_snapshot) == len(new_forecast) and len(new_forecast) > 0:
                old_vals = old_snapshot.sort_values('month')['oni'].reset_index(drop=True)
                new_vals = new_forecast.sort_values('month')['oni'].reset_index(drop=True)
                forecast_changed = not old_vals.equals(new_vals)
            else:
                forecast_changed = True
            if forecast_changed:
                new_forecast.to_csv(snapshot_file, index=False)
                updates_file = DATA_DIR / "enso_forecast_updates.csv"
                today_str = datetime.now().strftime('%Y-%m-%d')
                if updates_file.exists():
                    updates_df = pd.read_csv(updates_file)
                    if today_str not in updates_df['date'].values:
                        updates_df = pd.concat(
                            [updates_df, pd.DataFrame({'date': [today_str]})],
                            ignore_index=True
                        )
                        updates_df.to_csv(updates_file, index=False)
                        logger.info(f"ENSO multi-model forecast changed — recorded update on {today_str}")
                else:
                    pd.DataFrame({'date': [today_str]}).to_csv(updates_file, index=False)
                    logger.info(f"ENSO multi-model forecast changed — recorded update on {today_str}")
            else:
                logger.info("Multi-model ENSO forecast unchanged")
    except Exception as e:
        logger.error(f"Failed to update ENSO data: {e}")

    # Update projection history
    logger.info("Updating projection history...")
    try:
        from src.dashboard import load_and_update_projection_history
        import pandas as pd

        # Load the ERA5 data
        source = DATA_SOURCES["era5_global"]
        df = load_or_fetch_data(source["url"], source["local_file"])

        # Use multi-model mean ENSO forecast (not the IRI-based enso_combined.csv)
        # so projections are consistent with the annual prediction plot
        enso_proj = None
        try:
            from src.enso_plots import load_enso_forecast_data, build_enso_combined
            ef, _, oni = load_enso_forecast_data()
            enso_proj = build_enso_combined(oni, ef)
        except Exception as enso_err:
            logger.warning(f"Could not load multi-model ENSO, falling back to combined: {enso_err}")
            if enso_file.exists():
                enso_proj = pd.read_csv(enso_file)
                enso_proj['date'] = pd.to_datetime(enso_proj['date'])

        # Generate/update projection history
        load_and_update_projection_history(df, enso_proj)
        logger.info("Projection history updated")
    except Exception as e:
        logger.error(f"Failed to update projection history: {e}")

    # Regenerate static plot images
    logger.info("Regenerating static plot images...")
    try:
        from src.dashboard import generate_all_static_images
        import pandas as pd

        # Load the ERA5 data
        source = DATA_SOURCES["era5_global"]
        df = load_or_fetch_data(source["url"], source["local_file"])

        # Load ENSO data
        enso_df = None
        enso_file = DATA_DIR / "enso_combined.csv"
        if enso_file.exists():
            enso_df = pd.read_csv(enso_file)
            enso_df['date'] = pd.to_datetime(enso_df['date'])

        # Generate all static images
        assets_dir = Path(__file__).parent / 'assets' / 'images'
        generate_all_static_images(df, assets_dir, enso_df)
        logger.info("Static images regenerated")
    except Exception as e:
        logger.error(f"Failed to regenerate static images: {e}")

    # Update observational temperature records (for Models vs. Obs tab)
    update_obs_data(force=force)

    logger.info("Data update complete")


def update_obs_data(force: bool = False) -> None:
    """Fetch and cache the 5 observational temperature datasets."""
    logger.info("Updating observational temperature records...")
    try:
        from src.models_vs_obs import fetch_obs_data
        from pathlib import Path
        cache_path = DATA_DIR / 'cmip' / 'combined_obs_1981_2010.csv'
        fetch_obs_data(cache_path=cache_path, force=force)
        logger.info("Observational data updated")
    except Exception as e:
        logger.error(f"Failed to update observational data: {e}")


def run_dashboard(host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT,
                  debug: bool = DEBUG_MODE) -> None:
    """Run the dashboard server."""
    logger.info(f"Starting dashboard at http://{host}:{port}")

    # Load data
    source = DATA_SOURCES["era5_global"]
    df = load_or_fetch_data(source["url"], source["local_file"])

    # Create and run dashboard
    app = create_dashboard(df)
    app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="Climate Dashboard Application")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the dashboard server')
    dashboard_parser.add_argument('--host', default=DASHBOARD_HOST, help='Host to bind to')
    dashboard_parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help='Port to bind to')
    dashboard_parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update data from sources')
    update_parser.add_argument('--force', action='store_true', help='Force refresh even if cache is current')

    # Run command (default - update and run dashboard)
    run_parser = subparsers.add_parser('run', help='Update data and run dashboard')
    run_parser.add_argument('--host', default=DASHBOARD_HOST, help='Host to bind to')
    run_parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help='Port to bind to')

    args = parser.parse_args()

    if args.command == 'dashboard':
        run_dashboard(args.host, args.port, not args.no_debug)
    elif args.command == 'update':
        update_data(args.force)
    elif args.command == 'run' or args.command is None:
        update_data()
        host = getattr(args, 'host', DASHBOARD_HOST)
        port = getattr(args, 'port', DASHBOARD_PORT)
        run_dashboard(host, port, DEBUG_MODE)


if __name__ == "__main__":
    main()
