#!/usr/bin/env python3
"""Main entry point for the Climate Dashboard application."""

import argparse
import logging
import sys
from datetime import datetime
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

    # Update ENSO data
    logger.info("Updating ENSO data...")
    try:
        import pandas as pd
        enso_file = DATA_DIR / "enso_combined.csv"
        snapshot_file = DATA_DIR / "last_iri_snapshot.csv"

        # Load persisted IRI snapshot (committed to git, survives deploys)
        old_snapshot = None
        if snapshot_file.exists():
            old_snapshot = pd.read_csv(snapshot_file)
            old_snapshot['oni'] = old_snapshot['oni'].round(2)
            old_snapshot = old_snapshot.sort_values(['year', 'month']).reset_index(drop=True)

        create_combined_enso_dataset(enso_file)

        # Extract fresh IRI values from the newly written file
        new_enso = pd.read_csv(enso_file)
        mask = new_enso['source'] == 'IRI_Dynamical'
        new_iri = new_enso[mask][['year', 'month', 'oni']].copy()
        new_iri['oni'] = new_iri['oni'].round(2)
        new_iri = new_iri.sort_values(['year', 'month']).reset_index(drop=True)

        if old_snapshot is None:
            # First run — initialise snapshot without flagging a change
            new_iri.to_csv(snapshot_file, index=False)
            logger.info("IRI snapshot initialised (first run)")
        else:
            # Compare by month order only, not by (year, month).
            # The IRI always publishes the same 9 seasons (center months 2–10).
            # When current_month rolls over (e.g. Feb→Mar), the year assigned to
            # JFM shifts from the current year to next year, making old_snapshot
            # and new_iri differ even when the forecast values haven't changed.
            if len(old_snapshot) == len(new_iri) and len(new_iri) > 0:
                old_vals = old_snapshot.sort_values('month')['oni'].reset_index(drop=True)
                new_vals = new_iri.sort_values('month')['oni'].reset_index(drop=True)
                iri_changed = not old_vals.equals(new_vals)
            else:
                # Different row count → structural change in the IRI page
                iri_changed = True
            if iri_changed:
                # Update snapshot and log the update date
                new_iri.to_csv(snapshot_file, index=False)
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
                        logger.info(f"ENSO forecast changed — recorded update on {today_str}")
                else:
                    pd.DataFrame({'date': [today_str]}).to_csv(updates_file, index=False)
                    logger.info(f"ENSO forecast changed — recorded update on {today_str}")
            else:
                logger.info("IRI forecast unchanged")
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

        # Load ENSO data
        enso_df = None
        if enso_file.exists():
            enso_df = pd.read_csv(enso_file)
            enso_df['date'] = pd.to_datetime(enso_df['date'])

        # Generate/update projection history
        load_and_update_projection_history(df, enso_df)
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
