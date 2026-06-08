"""Track multi-model ENSO forecast skill by replaying our past plumes against
observed Niño3.4 / rNiño3.4.

The live ENSO pipeline only keeps the latest dated CSV per source on disk, so
past plumes are recovered two ways:

* **Backfill (one-time, local):** git history is our archive — every
  daily/monthly data-update commit captured that day's forecast CSVs.
  ``backfill_from_git`` walks the log, picks the most mature commit for each
  past NMME init-month, reconstructs the combined multi-model plume (reusing
  the dashboard's dedup + C3S baseline + rONI scaling + model-weighted
  quantiles), and writes one archive CSV per month.

* **Going forward (cron):** ``update_enso_skill`` reads the *live* on-disk
  forecasts each run, archives the current month's plume once its runs are in
  (maturity auto-detected, with a day-of-month fallback), and regenerates the
  figures. Runtime never touches git history, so it works under the shallow
  ``actions/checkout`` used by the GitHub Actions cron — mirroring how
  ``ec46_skill.py`` reads its own committed archive.

Renders, per (index, style), to ``forecast_skill/``:
    enso_skill_{oni,roni}_{lines,plumes,hybrid}.png
Committed to git but intentionally NOT wired into the dashboard layout.

Run directly:  python -m src.enso_skill            # backfill (if needed) + render
               python -m src.enso_skill --backfill # force git backfill
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
ENSO_ROOT = ROOT / "ENSO"
if str(ENSO_ROOT) not in sys.path:
    sys.path.insert(0, str(ENSO_ROOT))

from enso_forecast.normalize import (  # noqa: E402
    adjust_c3s_baseline,
    apply_roni_scaling,
    load_all_forecasts,
)
from enso_forecast.visualize import (  # noqa: E402
    _build_mega_df,
    _get_forecast_only,
    _weighted_quantile,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "forecast_skill"
ARCHIVE_DIR = OUTPUT_DIR / "enso_archive"
OBSERVED_DIR = ENSO_ROOT / "data" / "observed"
FORECAST_REL = "ENSO/data/forecasts"               # repo-relative forecast root
SOURCES = ["CFS", "NMME", "C3S", "CanSIPS"]         # IRI excluded (stale; not in central est.)
CORE_MONTHLY = ["NMME", "C3S", "CanSIPS"]           # sources whose new run defines a "month"
MIN_MODELS = 3                                       # match dashboard's ≥3-model floor
MATURITY_FALLBACK_DAY = 20                            # archive the month by here even if a source lags
STYLES = ("lines", "plumes", "hybrid")

QUANTILE_COLS = [
    "oni_q25", "oni_q50", "oni_q75",
    "roni_q25", "roni_q50", "roni_q75",
]

# Per-index metadata: forecast column + observed CSV + observed value column.
INDEX_META = {
    "oni": {
        "col": "nino34_anom",
        "obs_file": "nino34_monthly.csv",
        "obs_col": "nino34_anom",
        "label": "Niño 3.4 SST anomaly (ONI basis, °C)",
        "title": "Niño 3.4 (ONI)",
    },
    "roni": {
        "col": "roni_anom",
        "obs_file": "rnino_monthly.csv",
        "obs_col": "rnino34",
        "label": "Niño 3.4 relative SST anomaly (rONI, °C)",
        "title": "Relative Niño 3.4 (rONI)",
    },
}


# ---------------------------------------------------------------------------
# Plume computation (shared by git backfill and live update)
# ---------------------------------------------------------------------------

def _plume_from_combined(combined: pd.DataFrame) -> pd.DataFrame:
    """From a baseline-adjusted, rONI-scaled combined forecast frame, compute
    the model-weighted 25/50/75 quantiles per target month for ONI and rONI.

    Returns columns: target_month, n_models, {oni,roni}_{q25,q50,q75}.
    """
    if combined.empty:
        return pd.DataFrame()
    mega = _build_mega_df(_get_forecast_only(combined))
    members = mega[mega["member_id"] != "mean"].copy()
    if members.empty:
        return pd.DataFrame()

    rows = []
    for tm in sorted(members["target_month"].unique()):
        sub = members[members["target_month"] == tm]
        if sub["model"].nunique() < MIN_MODELS:
            continue
        counts = sub.groupby("model").size()
        w = sub["model"].map(lambda m: 1.0 / counts[m]).values
        row = {"target_month": tm, "n_models": int(sub["model"].nunique())}
        for idx, meta in INDEX_META.items():
            if meta["col"] not in sub.columns:
                row[f"{idx}_q25"] = row[f"{idx}_q50"] = row[f"{idx}_q75"] = np.nan
                continue
            vals = sub[meta["col"]].values.astype(float)
            ok = ~np.isnan(vals)
            if ok.sum() == 0:
                row[f"{idx}_q25"] = row[f"{idx}_q50"] = row[f"{idx}_q75"] = np.nan
                continue
            row[f"{idx}_q25"] = _weighted_quantile(vals[ok], w[ok], 0.25)
            row[f"{idx}_q50"] = _weighted_quantile(vals[ok], w[ok], 0.50)
            row[f"{idx}_q75"] = _weighted_quantile(vals[ok], w[ok], 0.75)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Git-history reconstruction (one-time backfill)
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _read_source_at_commit(commit: str, source: str, usecols=None) -> pd.DataFrame:
    """Extract the forecast CSV present for ``source`` at ``commit`` from git."""
    listing = _git(
        "ls-tree", "-r", "--name-only", commit, f"{FORECAST_REL}/{source}/"
    ).strip().splitlines()
    csvs = [p for p in listing if p.endswith(".csv")]
    if not csvs:
        return pd.DataFrame()
    try:
        blob = _git("show", f"{commit}:{sorted(csvs)[-1]}")
    except subprocess.CalledProcessError:
        return pd.DataFrame()
    if not blob.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(blob), usecols=usecols)
    except Exception:
        return pd.DataFrame()


def _nmme_init_month_at(commit: str) -> str | None:
    df = _read_source_at_commit(commit, "NMME", usecols=["init_date"])
    if df.empty:
        return None
    return str(df["init_date"].astype(str).max())[:7]


def _select_init_month_commits() -> list[tuple[str, str]]:
    """Return [(init_month 'YYYY-MM', commit_sha), ...]: the most recent commit
    for each distinct NMME init-month (i.e. the most mature copy of that
    month's forecast)."""
    shas = _git("log", "--format=%H", "--", f"{FORECAST_REL}/NMME").strip().splitlines()
    seen: dict[str, str] = {}  # init_month -> first (newest) commit seen
    for sha in shas:
        im = _nmme_init_month_at(sha)
        if im and im not in seen:
            seen[im] = sha
    return sorted(seen.items())


def _combined_at_commit(commit: str) -> pd.DataFrame:
    """Reconstruct load_all_forecasts() output as of ``commit`` from git."""
    frames = [
        df for src in SOURCES
        if not (df := _read_source_at_commit(commit, src)).empty
    ]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = adjust_c3s_baseline(combined)
    combined = apply_roni_scaling(combined)
    return combined


def backfill_from_git(archive_dir: Path = ARCHIVE_DIR, force: bool = False) -> list[str]:
    """Reconstruct each past NMME init-month's mature plume from git history and
    write archive CSVs. Skips the current calendar month (the live path owns it)
    and any month already archived (unless ``force``)."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    current_month = date.today().strftime("%Y-%m")
    written = []
    for im, sha in _select_init_month_commits():
        if im == current_month:
            continue  # live update owns the in-progress month
        path = archive_dir / f"enso_plume_{im}.csv"
        if path.exists() and not force:
            continue
        plume = _plume_from_combined(_combined_at_commit(sha))
        if plume.empty:
            logger.warning("ENSO skill: empty backfill plume for %s (%s)", im, sha[:8])
            continue
        plume["mature"] = True  # past months are complete
        _write_archive(path, plume)
        written.append(im)
        logger.info("ENSO skill: backfilled %s (%d target months, %s)", im, len(plume), sha[:8])
    return written


# ---------------------------------------------------------------------------
# Live current-month plume + maturity
# ---------------------------------------------------------------------------

def _current_plume_live() -> tuple[pd.DataFrame, str | None, bool]:
    """Compute the current combined plume from the live on-disk forecasts.

    Returns (plume_df, init_month, mature). ``mature`` is True once every
    present core monthly source (NMME/C3S/CanSIPS) carries the newest
    init-month — i.e. the new month's runs are all in.
    """
    combined = load_all_forecasts(sources=SOURCES)
    if combined.empty:
        return pd.DataFrame(), None, False

    months = {}
    for src in CORE_MONTHLY:
        sub = combined[combined["source"] == src]
        if not sub.empty:
            months[src] = str(sub["init_date"].astype(str).max())[:7]
    if not months:
        return pd.DataFrame(), None, False
    init_month = max(months.values())
    mature = all(m == init_month for m in months.values())

    plume = _plume_from_combined(combined)
    return plume, init_month, mature


# ---------------------------------------------------------------------------
# Archive I/O
# ---------------------------------------------------------------------------

def _write_archive(path: Path, plume: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["target_month", "n_models", *QUANTILE_COLS, "mature"]
    plume = plume.copy()
    for c in cols:
        if c not in plume.columns:
            plume[c] = np.nan
    plume[cols].to_csv(path, index=False)


def load_archived_forecasts(archive_dir: Path = ARCHIVE_DIR) -> list[tuple[str, pd.DataFrame]]:
    """Return [(init_month, plume_df), ...] sorted by init_month."""
    out = []
    for fp in sorted(archive_dir.glob("enso_plume_*.csv")):
        im = fp.stem.replace("enso_plume_", "")
        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["target_month"].astype(str) + "-01")
        out.append((im, df.sort_values("date").reset_index(drop=True)))
    return out


def _archived_is_mature(archive_dir: Path, init_month: str) -> bool:
    path = archive_dir / f"enso_plume_{init_month}.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return bool(df["mature"].iloc[0]) if "mature" in df.columns and len(df) else False


# ---------------------------------------------------------------------------
# Observed truth
# ---------------------------------------------------------------------------

def _load_observed(index_mode: str, start: str = "2025-09-01") -> pd.DataFrame:
    meta = INDEX_META[index_mode]
    path = OBSERVED_DIR / meta["obs_file"]
    if not path.exists():
        logger.warning("ENSO skill: observed file missing: %s", path)
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start)].copy()
    df = df.rename(columns={meta["obs_col"]: "value"})
    return df[["date", "value"]].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(index_mode: str, style: str, forecasts, output_path: Path) -> Path | None:
    """Render one verification figure. ``style`` ∈ {lines, plumes, hybrid}."""
    meta = INDEX_META[index_mode]
    med, q25, q75 = f"{index_mode}_q50", f"{index_mode}_q25", f"{index_mode}_q75"

    # Keep only inits that actually have data for this index (e.g. rONI didn't
    # exist in the earliest months) so the colorbar/title stay honest.
    usable = [(lbl, fc) for lbl, fc in forecasts if fc[med].notna().any()]
    if not usable:
        logger.warning("ENSO skill: no inits with %s data for %s", index_mode, style)
        return None

    obs = _load_observed(index_mode)
    n = len(usable)
    latest_idx = n - 1
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=140)

    for i, (init_label, fc) in enumerate(usable):
        f = fc[fc[med].notna()]
        if f.empty:
            continue
        color = cmap(0.85) if n == 1 else cmap(norm(i))
        show_band = style == "plumes" or (style == "hybrid" and i == latest_idx)
        if show_band:
            ax.fill_between(f["date"], f[q25], f[q75],
                            color=color, alpha=0.16, linewidth=0, zorder=2 + i)
        lw = 2.6 if (style == "hybrid" and i == latest_idx) else 1.8
        alpha = 0.55 + 0.45 * (i / (n - 1)) if n > 1 else 1.0
        ax.plot(f["date"], f[med], color=color, alpha=alpha, linewidth=lw,
                zorder=5 + i, marker="o", markersize=2.5)

    if not obs.empty:
        ax.plot(obs["date"], obs["value"], color="black", linewidth=2.6,
                marker="o", markersize=4, label="Observed", zorder=100)
        ax.axvline(obs["date"].max(), color="0.4", linestyle="--", linewidth=1, zorder=1)
        ax.text(obs["date"].max(), ax.get_ylim()[1], "  obs frontier",
                color="0.4", fontsize=8, va="top", ha="left")

    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    style_label = {"lines": "central lines",
                   "plumes": "full plumes (25–75 band)",
                   "hybrid": "latest plume + prior central lines"}[style]
    labels = [lbl for lbl, _ in usable]
    ax.set_title(
        f"ENSO forecast verification — {meta['title']}\n"
        f"{n} monthly plumes ({labels[0]} → {labels[-1]}) vs. observed"
        f"  ·  {style_label}",
        fontsize=12,
    )
    ax.set_ylabel(meta["label"])
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_ticks(list(range(n)))
    cbar.set_ticklabels(labels)
    cbar.set_label("Forecast init (oldest → newest)")
    ax.legend(loc="upper left", framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("ENSO skill: wrote %s", output_path)
    return output_path


def generate_all(archive_dir: Path = ARCHIVE_DIR, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Render all index × style figures from the archived plumes."""
    forecasts = load_archived_forecasts(archive_dir)
    if not forecasts:
        logger.warning("ENSO skill: no archived plumes under %s; nothing to render", archive_dir)
        return []
    written = []
    for index_mode in ("oni", "roni"):
        for style in STYLES:
            out = output_dir / f"enso_skill_{index_mode}_{style}.png"
            if make_plot(index_mode, style, forecasts, out):
                written.append(out)
    return written


# ---------------------------------------------------------------------------
# Cron entry point
# ---------------------------------------------------------------------------

def update_enso_skill(today: date | None = None, force: bool = False,
                      archive_dir: Path = ARCHIVE_DIR,
                      output_dir: Path = OUTPUT_DIR) -> bool:
    """Archive the current month's plume (once mature / past the fallback day)
    and regenerate the figures. Cheap no-op on days with nothing new.

    Returns True if it (re)archived a month and rebuilt the figures.

    Detection: the current month is archived & frozen once all core monthly
    sources carry its init-month (auto-detected), with a day-of-month fallback
    so a lagging source can't stall the update indefinitely. A month already
    archived as *mature* is left frozen; the figures then change only when a
    new month arrives — i.e. a monthly cadence.
    """
    today = today or date.today()
    plume, init_month, mature = _current_plume_live()
    if plume.empty or init_month is None:
        logger.warning("ENSO skill: no live plume; skipping update")
        return False

    already_mature = _archived_is_mature(archive_dir, init_month)
    fallback = today.day >= MATURITY_FALLBACK_DAY
    figures_missing = not any(output_dir.glob("enso_skill_*.png"))

    if already_mature and not force and not figures_missing:
        logger.info("ENSO skill: %s already archived (mature); nothing to do", init_month)
        return False

    # The maturity flag always reflects reality (whether the month's runs are
    # in / past the fallback day). ``force`` only bypasses the skip-guard to
    # trigger a re-archive + regen — it must not falsely freeze an immature
    # month, or the current month would never refresh as its runs arrive.
    plume = plume.copy()
    plume["mature"] = bool(mature or fallback)
    _write_archive(archive_dir / f"enso_plume_{init_month}.csv", plume)
    paths = generate_all(archive_dir, output_dir)
    logger.info(
        "ENSO skill: archived %s (mature=%s, fallback=%s) and rebuilt %d figures",
        init_month, mature, fallback, len(paths),
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force_backfill = "--backfill" in sys.argv
    backfill_from_git(force=force_backfill)
    update_enso_skill(force=True)
    print("ENSO skill figures written to", OUTPUT_DIR)
