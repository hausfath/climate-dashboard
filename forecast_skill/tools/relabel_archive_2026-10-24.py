"""One-off (2026-10-24): relabel the EC46 archive by model RUN date.

Until 2026-10-24 the fetcher named files and set ``init_date`` to the
fetch date. Open-Meteo rolls the same-day EC46 run out at ~20:30 UTC, so:

* every fetch before 1 Oct 2026 (06:00 UTC cron, or local fetches around
  04 UTC) retrieved the PREVIOUS day's run  -> run = fetch - 1 day;
* from 1 Oct the 16:30 UTC cron fired ~20 UTC and its 35-min fetch
  straddled the rollout. Run dates below come from the Actions start
  times (same-day when the fetch ran after ~20:30 UTC), the file span
  (46 rows = same-day run with its last day complete), the narrow day-0
  ensemble width of a true lead-0 day, and the "dropping ... (N/432 pts)"
  log lines that count how many gridpoints were already on the new run.

Files whose run is already held by an earlier file (the fetch re-served
the same run) are deleted; the 10-23 fetch (143/432 points on the Oct 23
run, the rest on Oct 22) is a mixed-run file and is deleted from the
archive, to be replaced by a clean fetch with the new fetcher.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRS = {
    "era5_forecast_ec46_": [ROOT / "forecast_skill" / "archive", ROOT / "data"],
    "ec46_members_": [ROOT / "forecast_skill" / "archive_members", ROOT / "data"],
}
OCTOBER = {                       # fetch date -> run date (None = delete)
    "2026-10-01": "2026-10-01",  # Actions 21:12 UTC, span 45
    "2026-10-02": "2026-10-02",  # 20:52, span 45
    "2026-10-03": None,          # 19:20 -> Oct 2 run again (byte-identical minus day 0)
    "2026-10-04": "2026-10-03",  # 19:45 -> previous day's run
    "2026-10-05": "2026-10-05",  # 22:50, narrow day-0 width (last day truncated earlier)
    "2026-10-06": None,          # local 04:27 UTC -> Oct 5 run again (day-0 diff +0.004)
    "2026-10-07": "2026-10-07",  # 21:28, span 45
    "2026-10-08": "2026-10-08",  # 21:26, span 45
    "2026-10-09": "2026-10-09",  # 21:07, span 45
    "2026-10-11": "2026-10-10",  # 17:38 -> previous day's run
    "2026-10-12": "2026-10-12",  # 23:08, span 45
    "2026-10-13": "2026-10-13",  # 21:25, span 45
    "2026-10-14": "2026-10-14",  # 20:17 fetch to ~20:55, span 45
    "2026-10-15": None,          # 19:58 -> Oct 14 run again (byte-identical minus day 0)
    "2026-10-16": "2026-10-15",  # 19:38 -> previous day's run
    "2026-10-17": "2026-10-16",  # 18:43
    "2026-10-18": "2026-10-17",  # 18:47
    "2026-10-19": "2026-10-19",  # 21:11, span 45
    "2026-10-20": "2026-10-20",  # 20:04; 236/432 points on the Oct 20 run (majority)
    "2026-10-21": "2026-10-21",  # 20:18; 372/432 on the Oct 21 run
    "2026-10-22": "2026-10-22",  # 20:14; 317/432 on the Oct 22 run
    "2026-10-23": None,          # 20:01; only 143/432 on the Oct 23 run -> mixed, drop
}


def target(fetch: str) -> str | None:
    if fetch >= "2026-10-01":
        if fetch not in OCTOBER:
            raise SystemExit(f"no run-date rule for {fetch}")
        return OCTOBER[fetch]
    return (pd.Timestamp(fetch) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def main(apply: bool) -> None:
    for prefix, dirs in DIRS.items():
        for d in dirs:
            files = sorted(d.glob(f"{prefix}*.csv"))
            plan = []
            for fp in files:
                fetch = fp.stem.replace(prefix, "")
                plan.append((fetch, target(fetch), fp))
            # deletions first, then renames in ascending order (each target
            # date is freed by the preceding rename)
            for fetch, run, fp in plan:
                if run is None:
                    print(f"DELETE {fp.relative_to(ROOT)}")
                    if apply:
                        fp.unlink()
            runs = [run for _, run, _ in plan if run is not None]
            if len(set(runs)) != len(runs):
                dups = sorted({r for r in runs if runs.count(r) > 1})
                raise SystemExit(f"two files map to the same run in {d}: {dups}")
            for fetch, run, fp in plan:
                if run is None or run == fetch:
                    if run == fetch:
                        print(f"KEEP   {fp.relative_to(ROOT)}")
                    continue
                new = fp.with_name(f"{prefix}{run}.csv")
                # ascending order: the target date was freed by the previous
                # rename (or belongs to a deleted file)
                print(f"RENAME {fp.relative_to(ROOT)} -> {new.name}")
                if apply:
                    df = pd.read_csv(fp)
                    if "init_date" in df.columns:
                        df["init_date"] = run
                    df.to_csv(new, index=False)
                    fp.unlink()
    # rewrite init_date on kept files too (fetch date == run date there)
    if apply:
        for prefix, dirs in DIRS.items():
            for d in dirs:
                for fp in d.glob(f"{prefix}*.csv"):
                    df = pd.read_csv(fp)
                    run = fp.stem.replace(prefix, "")
                    if "init_date" in df.columns and (df["init_date"].astype(str) != run).any():
                        df["init_date"] = run
                        df.to_csv(fp, index=False)


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
