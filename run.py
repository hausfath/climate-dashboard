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


def update_ec46_forecast(force: bool = False) -> None:
    """Fetch ECMWF EC46 (extended-range) global-mean 2m temperature forecast.

    Skips quietly on failure so a transient Open-Meteo outage doesn't
    crash the daily cron — the dashboard already degrades gracefully if
    the forecast CSV is missing.
    """
    enso_root = str(Path(__file__).parent / "ENSO")
    if enso_root not in sys.path:
        sys.path.insert(0, enso_root)

    try:
        from enso_forecast.fetchers.ec46 import save_ec46
        save_ec46(force=force)
    except Exception as e:
        logger.error(f"Failed to fetch EC46 forecast: {e}")


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
    from enso_forecast.fetchers.cansips import save_cansips
    from enso_forecast.config import FORECASTS_DIR

    sources = [
        ("observed", save_observed),
        ("CFS", save_cfs),
        ("NMME", save_nmme),
        ("C3S", save_c3s),
        ("CanSIPS", save_cansips),
        ("IRI", save_iri),
    ]

    for name, fetch_fn in sources:
        try:
            # Monthly sources (NMME, C3S, CanSIPS, IRI): fetch only when a new
            # run is actually available, judged by the init_date INSIDE the
            # latest CSV — not its fetch-date filename, and deliberately NOT
            # gated on --force.
            #
            # Keying on the filename was wrong: a file fetched this calendar
            # month but holding last month's run (the new run wasn't posted yet
            # when the cron last ran) suppressed fetching for the rest of the
            # month, so the new run was never picked up once it appeared.
            #
            # Ignoring --force is intentional. The daily cron runs
            # `update --force` so ERA5 re-downloads in its ephemeral container,
            # but that flag used to also force these monthly fetchers every day
            # — re-pulling C3S (~30 min via CDS) needlessly, since these sources
            # only change monthly. Keying the skip on init_date instead picks up
            # a new monthly run within a day of posting while keeping same-month
            # days cheap. (CFS/observed aren't here, so they still refresh daily;
            # ERA5 still honors --force above.)
            #
            # Skip only when the latest CSV is already on the current month's
            # initialization. Convention-agnostic — NMME/C3S/IRI init = the 1st,
            # CanSIPS = end-of-month — because the prior month's run is always <
            # the first of the current month; "any model still on a prior init"
            # keeps re-fetching while C3S's panel trickles in over ~10 days. To
            # force-repull a current-month run (e.g. after a bad fetch), delete
            # the source's latest CSV so this guard falls through.
            if name in ("NMME", "C3S", "CanSIPS", "IRI"):
                src_dir = FORECASTS_DIR / name
                existing = sorted(src_dir.glob("*.csv")) if src_dir.exists() else []
                if existing:
                    import pandas as pd
                    try:
                        latest = pd.read_csv(existing[-1])
                    except Exception:
                        latest = None
                    if latest is not None and not latest.empty and "init_date" in latest.columns:
                        current_init = f"{date.today().strftime('%Y-%m')}-01"
                        inits = latest["init_date"].astype(str)
                        if not (inits < current_init).any():
                            logger.info(
                                f"{name} already on current-month init "
                                f"({inits.max()}), skipping"
                            )
                            continue
                        logger.info(
                            f"{name} latest init {inits.min()} predates "
                            f"{current_init} — fetching for new run"
                        )
                    # No init_date / empty / unreadable file: fall through and fetch.

            logger.info(f"Fetching {name} ENSO data...")
            fetch_fn(force=force)

            # Clean up old dated CSVs, keep only the latest
            # Only clean up if the latest file has real data (> 10 bytes)
            if name != "observed":  # observed uses fixed filenames
                src_dir = FORECASTS_DIR / name
                csvs = sorted(src_dir.glob("*.csv"))
                if len(csvs) > 1 and csvs[-1].stat().st_size > 10:
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

    # Fetch ECMWF EC46 global-mean GSAT forecast (drives the daily-anomaly
    # plot's 46-day forecast tail). Cheap and idempotent on cron repeats.
    logger.info("Fetching ECMWF EC46 GSAT forecast...")
    update_ec46_forecast(force=force)

    # Refresh the EC46 skill-tracking plot (past forecasts vs. observed).
    # Output lives under forecast_skill/ and is committed by the daily
    # cron; not surfaced in the dashboard layout.
    logger.info("Rebuilding EC46 forecast-skill plot...")
    try:
        from src.ec46_skill import make_plot as make_ec46_skill_plot
        make_ec46_skill_plot()
    except Exception as e:
        logger.error(f"EC46 skill plot failed: {e}")

    # Update ENSO data
    logger.info("Updating ENSO data...")
    try:
        import pandas as pd
        enso_file = DATA_DIR / "enso_combined.csv"
        snapshot_file = DATA_DIR / "last_enso_forecast_snapshot.csv"
        snapshot_file_roni = DATA_DIR / "last_enso_forecast_snapshot_roni.csv"

        create_combined_enso_dataset(enso_file)

        # Build multi-model mean forecast (same data the prediction model uses)
        from src.enso_plots import load_enso_forecast_data, build_enso_combined
        ef, eo, oni = load_enso_forecast_data()
        enso_combined = build_enso_combined(oni, ef, eo)
        enso_combined_roni = build_enso_combined(oni, ef, eo, index_mode="roni")

        # Extract forecast-only rows as the snapshot for each index
        new_forecast = enso_combined[enso_combined['is_forecast']][['year', 'month', 'oni']].copy()
        new_forecast['oni'] = new_forecast['oni'].round(3)
        new_forecast = new_forecast.sort_values(['year', 'month']).reset_index(drop=True)

        new_forecast_roni = enso_combined_roni[enso_combined_roni['is_forecast']][['year', 'month', 'roni']].copy()
        new_forecast_roni['roni'] = new_forecast_roni['roni'].round(3)
        new_forecast_roni = new_forecast_roni.sort_values(['year', 'month']).reset_index(drop=True)

        # Guard: don't save empty snapshots (fetch failure)
        if new_forecast.empty:
            logger.warning("ENSO forecast snapshot is empty — skipping snapshot update")
        else:
            # Load previous snapshot
            old_snapshot = None
            if snapshot_file.exists():
                old_snapshot = pd.read_csv(snapshot_file)
                if old_snapshot.empty or 'oni' not in old_snapshot.columns:
                    old_snapshot = None
                else:
                    old_snapshot['oni'] = old_snapshot['oni'].round(3)
                    old_snapshot = old_snapshot.sort_values(['year', 'month']).reset_index(drop=True)

            if old_snapshot is None:
                new_forecast.to_csv(snapshot_file, index=False)
                logger.info("Multi-model ENSO forecast snapshot initialised (first run)")
            else:
                # Compare forecast values on months present in both snapshots;
                # only flag a change if the largest monthly shift exceeds 0.1 °C
                # (avoids spurious stars from daily numerical drift).
                # Only consider months with ≥3 models (build_enso_combined
                # already filters to these, but guard against edge cases).
                merged = old_snapshot.merge(new_forecast, on=['year', 'month'],
                                           suffixes=('_old', '_new'))
                if len(merged) > 0:
                    max_diff = (merged['oni_new'] - merged['oni_old']).abs().max()
                    forecast_changed = max_diff >= 0.1
                else:
                    # No overlapping months — likely a major composition change
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

        # rONI snapshot: unconditionally refresh each cron run (no
        # change-detection logic — that lives on the ONI snapshot since
        # the dashboard's "forecast updated" indicator is driven there).
        if new_forecast_roni.empty:
            logger.warning("rONI ENSO forecast snapshot is empty — skipping")
        else:
            new_forecast_roni.to_csv(snapshot_file_roni, index=False)
            logger.info("Multi-model rONI ENSO forecast snapshot saved (%d rows)",
                        len(new_forecast_roni))
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
            ef, eo, oni = load_enso_forecast_data()
            enso_proj = build_enso_combined(oni, ef, eo)
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

        # Load ENSO data from live multi-model forecast
        enso_df = None
        try:
            from src.enso_plots import load_enso_forecast_data, build_enso_combined
            ef, eo, oni = load_enso_forecast_data()
            enso_df = build_enso_combined(oni, ef, eo)
        except Exception as enso_err:
            logger.warning(f"Could not load multi-model ENSO for images: {enso_err}")
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
