#!/usr/bin/env python3
"""Tropical-mean (20degS-20degN) daily SST year-lines -- proposal mock-up.

Renders figure_dark.png / figure_light.png in the style of the dashboard's
daily Nino year-lines figure (src/enso_plots.py::create_nino34_daily_years),
applied to the OISSTv2.1 tropical-belt series the repo already maintains.

Data: the committed daily box-mean history data/nino34_daily_history.csv
("tropics" column: 20S-20N, all longitudes, cos-weighted mean on the same
strided OISSTv2.1 subsample as the Nino boxes -- see TROPICS_BOX in
src/nino_daily.py). The default run reads the cached subset in ./data/;
--refresh re-extracts it from the repo's committed CSV. Neither mode
touches the network: the upstream fetch path (CoastWatch ERDDAP, NCEI
fallback) already lives in src/nino_daily.py and is deliberately not
duplicated here.

Anomalies use the dashboard's own era-relative convention, imported
read-only from src.nino_daily.era_relative_anomalies: centered 30-year
day-of-year climatology per year, windows clamped at the record edges,
current year excluded from its own baseline, 15-day circular smooth.

Usage:
    python make_figure.py            # render from the cached extract
    python make_figure.py --refresh  # re-extract the cache first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True  # keep the repo free of __pycache__

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

# Read-only imports from the repo: template registration + the exact
# anomaly convention the existing year-lines figure uses.
from src.theme import template_name, tokens  # noqa: E402  (registers templates)
from src.nino_daily import era_relative_anomalies  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

DATA_DIR = HERE / "data"
CACHE = DATA_DIR / "tropics_daily_sst.csv"
MANIFEST = DATA_DIR / "MANIFEST.json"
SOURCE = REPO / "data" / "nino34_daily_history.csv"

# Styling constants mirrored from create_nino34_daily_years. Copied, not
# imported: importing src.enso_plots would drag in the full dash stack.
GRAY = {"dark": "#3d434b", "light": "#d3cec4"}
CUR_COLOR = {"dark": "#e4572e", "light": "#d94f25"}
FG_SOFT = {"dark": "#9a958d", "light": "#8a857c"}
HIGHLIGHTS = {
    2015: {"light": "#2a7fa8", "dark": "#7cc7e8", "label_doy": 330},
    1997: {"light": "#6c5ce7", "dark": "#a29bfe", "label_doy": 285},
}
MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
          "Aug", "Sep", "Oct", "Nov", "Dec"]

EXPORT = dict(width=1200, height=500, scale=2)  # repo standard full-width


def refresh_cache() -> None:
    """Extract (date, tropics) from the repo's committed history CSV."""
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"{SOURCE} missing -- run from a checkout that carries the "
            "committed daily history CSV")
    df = pd.read_csv(SOURCE)
    expected = {"date", "nino34", "tropics"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{SOURCE.name}: expected columns {sorted(expected)},"
                         f" got {sorted(df.columns)}")
    sub = df[["date", "tropics"]].copy()
    sub["date"] = pd.to_datetime(sub["date"], format="%Y-%m-%d")
    if sub["tropics"].isna().any():
        raise ValueError(f"{SOURCE.name}: "
                         f"{int(sub['tropics'].isna().sum())} missing tropics values")
    lo, hi = float(sub["tropics"].min()), float(sub["tropics"].max())
    if not (20.0 < lo and hi < 32.0):
        raise ValueError(f"tropics SST outside plausible range: {lo:.2f}..{hi:.2f} degC")
    if len(sub) < 15000 or not sub["date"].is_monotonic_increasing:
        raise ValueError(f"unexpected series shape: {len(sub)} rows")

    DATA_DIR.mkdir(exist_ok=True)
    sub.to_csv(CACHE, index=False, date_format="%Y-%m-%d", float_format="%.4f")
    MANIFEST.write_text(json.dumps({
        "cache": CACHE.name,
        "source": "data/nino34_daily_history.csv (committed; maintained by the"
                  " daily cron from NOAA OISSTv2.1 via CoastWatch ERDDAP with"
                  " NCEI fallback -- see src/nino_daily.py)",
        "series": "tropics: 20S-20N all-longitude cos-weighted SST box mean",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "rows": len(sub),
        "first": str(sub["date"].min().date()),
        "last": str(sub["date"].max().date()),
        "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"cache refreshed: {len(sub)} rows through {sub['date'].max().date()}")


def load_cache() -> pd.DataFrame:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing -- run with --refresh first")
    df = pd.read_csv(CACHE, parse_dates=["date"])
    if list(df.columns) != ["date", "tropics"] or df.empty:
        raise ValueError(f"{CACHE.name}: unexpected columns {list(df.columns)}")
    if df["tropics"].isna().any() or not df["date"].is_monotonic_increasing:
        raise ValueError(f"{CACHE.name}: NaNs or unsorted dates")
    return df


def build_figure(anom: pd.DataFrame, dark_mode: bool) -> go.Figure:
    """Year-lines figure, mirroring create_nino34_daily_years for a
    non-Nino-3.4 region (no ONI category bands -- those thresholds are
    defined on Nino 3.4 only, matching the existing figure's convention)."""
    mode = "dark" if dark_mode else "light"
    t = tokens(dark_mode)
    fig = go.Figure()

    cur_year = int(anom["year"].max())

    fig.add_hline(y=0, line_dash="dash", line_color=FG_SOFT[mode],
                  opacity=0.5, line_width=1)

    for yr, g in anom.groupby("year"):
        if yr == cur_year:
            continue
        hl = HIGHLIGHTS.get(yr)
        fig.add_trace(go.Scatter(
            x=g["doy"], y=g["anom"], mode="lines",
            name=str(yr),
            line=dict(color=hl[mode] if hl else GRAY[mode],
                      width=1.8 if hl else 0.8),
            opacity=0.95 if hl else 0.75,
            hoverinfo="skip",
        ))

    g_cur = anom[anom["year"] == cur_year]
    fig.add_trace(go.Scatter(
        x=g_cur["doy"], y=g_cur["anom"], mode="lines",
        name=str(cur_year),
        line=dict(color=CUR_COLOR[mode], width=3.2),
        hoverinfo="skip",
    ))

    cur_val = float(g_cur["anom"].iloc[-1])
    fig.add_annotation(
        x=float(g_cur["doy"].iloc[-1]), y=cur_val,
        text=f"{cur_year}  {cur_val:+.2f}°C",
        showarrow=False, xanchor="left", xshift=8, yshift=8,
        font=dict(size=13, color=CUR_COLOR[mode], weight="bold"),
        bgcolor=("rgba(21, 24, 28, 0.75)" if dark_mode
                 else "rgba(251, 250, 247, 0.75)"))
    for yr, hl in HIGHLIGHTS.items():
        g = anom[anom["year"] == yr]
        if g.empty:
            continue
        i = int((g["doy"].values > hl["label_doy"]).argmax())
        fig.add_annotation(
            x=float(g["doy"].iloc[i]), y=float(g["anom"].iloc[i]),
            text=str(yr), showarrow=False, yshift=10,
            font=dict(size=11, color=hl[mode], weight="bold"))

    fig.update_layout(
        xaxis=dict(title="", tickvals=MONTH_STARTS, ticktext=MONTHS,
                   range=[1, 366], showgrid=False),
        yaxis=dict(title="20°S–20°N mean SST anomaly (°C, "
                         "vs centered 30-yr climatology)",
                   range=[float(anom["anom"].min()) - 0.1,
                          float(anom["anom"].max()) + 0.1]),
        template=template_name(dark_mode),
        showlegend=False,
        hoverlabel=dict(bgcolor=t["panel"], bordercolor=t["grid"],
                        font=dict(color=t["text"])),
        margin=dict(l=60, r=30, t=30, b=40),
    )
    return fig


def _assert_png(path: Path, w: int = 2400, h: int = 1000,
                min_bytes: int = 20 * 1024) -> None:
    """Fail loudly if the export is missing, tiny, or the wrong size."""
    b = path.read_bytes()
    if len(b) <= min_bytes:
        raise AssertionError(f"{path.name}: only {len(b)} bytes")
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        raise AssertionError(f"{path.name}: not a PNG")
    ww, hh = struct.unpack(">II", b[16:24])
    if (ww, hh) != (w, h):
        raise AssertionError(f"{path.name}: {ww}x{hh}px, expected {w}x{h}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract the cache from the repo's committed CSV")
    args = ap.parse_args()

    if args.refresh:
        refresh_cache()
    hist = load_cache()

    # The dashboard's own anomaly convention, applied to the tropics belt.
    anom = era_relative_anomalies(hist, index_mode="oni", region="tropics")
    if anom.empty:
        raise ValueError("era_relative_anomalies returned no rows")

    for dark in (True, False):
        out = HERE / f"figure_{'dark' if dark else 'light'}.png"
        fig = build_figure(anom, dark)
        fig.write_image(str(out), **EXPORT)
        _assert_png(out)
        print(f"wrote {out.name} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
