#!/usr/bin/env python3
"""SOI + equatorial 850 hPa trade-wind anomalies -- proposal mock-up.

Two-row monthly figure:
  row 1  Southern Oscillation Index (standardized Tahiti - Darwin SLP),
         as bars -- station-based observation (NOAA CPC).
  row 2  Equatorial Pacific 850 hPa trade-wind anomalies for the three
         CPC-published longitude zones -- reanalysis-derived indices
         (CDAS/NCEP-NCAR, 1981-2010 base; NOAA CPC) -- with the observed
         Nino 3.4 anomaly overlaid from the repo's committed CSV.

Index convention (verified against the files: 1997/2015/2026 El Nino years
are strongly negative, 2010/2020 La Nina years positive, and the ORIGINAL
blocks are positive ~5-11 m/s): the wind indices report easterly trade-wind
speed, so a POSITIVE anomaly = stronger-than-normal trades and a NEGATIVE
anomaly = weakened trades / westerly anomalies (El Nino-ish).

Sources are cached raw and verbatim under ./data/ (with MANIFEST.json
recording URL, sha256, and retrieval time). Default run is offline from the
cache; --refresh re-downloads. CPC index files are fixed-width -- YEAR in
columns 0-3 then twelve 6-character fields -- and are parsed as such
because -999.9 sentinels collide with neighbouring values when
whitespace-split (e.g. "-6.1-999.9" in the current year's row).

Usage:
    python make_figure.py            # offline, from ./data/
    python make_figure.py --refresh  # re-download sources first
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

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

# Registers the climate_dark / climate_light Plotly templates (read-only).
from src.theme import template_name, tokens  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

DATA_DIR = HERE / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"
NINO34_CSV = REPO / "ENSO" / "data" / "observed" / "nino34_monthly.csv"

SOURCES = {
    # name: (url, content marker that must appear in the payload)
    "soi": ("https://www.cpc.ncep.noaa.gov/data/indices/soi",
            "(STAND TAHITI - STAND DARWIN)"),
    "wpac850": ("https://www.cpc.ncep.noaa.gov/data/indices/wpac850",
                "(135E-180W)"),
    "cpac850": ("https://www.cpc.ncep.noaa.gov/data/indices/cpac850",
                "(175W-140W)"),
    "epac850": ("https://www.cpc.ncep.noaa.gov/data/indices/epac850",
                "(135W-120W)"),
}

WINDOW_START = "2015-01-01"
SENTINEL = -999.0          # values <= this are CPC missing-data markers
EXPORT = dict(width=1200, height=500, scale=2)  # repo standard full-width

UA = {"User-Agent": "climate-dashboard-proposal/0.1 (panel mock-up)"}


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

def refresh_cache() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    manifest = {}
    for name, (url, marker) in SOURCES.items():
        resp = requests.get(url, headers=UA, timeout=60)
        resp.raise_for_status()
        text = resp.text
        if len(text) < 1000:
            raise ValueError(f"{name}: implausibly short payload "
                             f"({len(text)} bytes) from {url}")
        if marker not in text:
            raise ValueError(f"{name}: expected marker {marker!r} not found "
                             f"-- source layout may have changed ({url})")
        path = DATA_DIR / f"{name}.txt"
        path.write_text(text, encoding="utf-8", newline="")  # verbatim
        manifest[name] = {
            "url": url,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        }
        print(f"fetched {name}: {len(text)} bytes")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8")


# ---------------------------------------------------------------------------
# Strict fixed-width parsing of CPC index files
# ---------------------------------------------------------------------------

def parse_cpc_blocks(text: str, name: str) -> dict[str, pd.DataFrame]:
    """Split a CPC index file into blocks keyed by normalized subtitle
    ('ANOMALY', 'STANDARDIZED DATA', 'ORIGINAL DATA').

    Layout per block: title line, subtitle line, blank, 'YEAR JAN ... DEC'
    header, then fixed-width rows (YEAR cols 0-3, twelve 6-char fields).
    """
    lines = text.split("\n")
    blocks: dict[str, pd.DataFrame] = {}
    current_subtitle: str | None = None
    rows: list[tuple[int, list[float]]] = []

    def flush() -> None:
        nonlocal rows, current_subtitle
        if current_subtitle is None:
            if rows:
                raise ValueError(f"{name}: data rows before any subtitle")
            return
        if not rows:
            raise ValueError(f"{name}: block {current_subtitle!r} is empty")
        years = [y for y, _ in rows]
        if years != sorted(set(years)):
            raise ValueError(f"{name}: {current_subtitle!r} years not "
                             "strictly increasing")
        df = pd.DataFrame([v for _, v in rows], index=years,
                          columns=range(1, 13))
        if current_subtitle in blocks:
            raise ValueError(f"{name}: duplicate block {current_subtitle!r}")
        blocks[current_subtitle] = df
        rows = []

    pending_subtitle = False
    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = " ".join(line.split())
        if not stripped:
            continue
        if stripped.startswith("YEAR "):
            pending_subtitle = False
            continue
        # data row: 4-digit year then fixed-width fields
        if len(line) >= 10 and line[:4].isdigit():
            year = int(line[:4])
            if not 1900 <= year <= 2100:
                raise ValueError(f"{name}: implausible year {year}")
            padded = line.ljust(4 + 6 * 12)
            vals: list[float] = []
            for i in range(12):
                field = padded[4 + 6 * i: 10 + 6 * i].strip()
                if not field:
                    raise ValueError(f"{name}: empty field {i + 1} in "
                                     f"year {year}")
                try:
                    v = float(field)
                except ValueError as e:
                    raise ValueError(f"{name}: non-numeric field {field!r} "
                                     f"in year {year}") from e
                vals.append(np.nan if v <= SENTINEL else v)
            rows.append((year, vals))
            continue
        # otherwise: a title or subtitle line
        if pending_subtitle:
            # second consecutive text line = the subtitle; a new block begins
            flush()
            current_subtitle = stripped
            pending_subtitle = False
        else:
            pending_subtitle = True
    flush()
    return blocks


def block_to_series(block: pd.DataFrame, name: str,
                    max_abs: float) -> pd.Series:
    """Wide (year x month) block -> monthly Series with a strict tail rule:
    NaNs (sentinels) may only appear as a contiguous tail."""
    long = block.stack(future_stack=True)
    idx = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in long.index])
    s = pd.Series(long.values, index=idx, name=name).sort_index()
    non_nan = s.dropna()
    if non_nan.empty:
        raise ValueError(f"{name}: no valid values")
    last_valid = non_nan.index[-1]
    if s.loc[:last_valid].isna().any():
        gaps = s.loc[:last_valid][s.loc[:last_valid].isna()].index
        raise ValueError(f"{name}: interior missing values at "
                         f"{[str(d.date()) for d in gaps[:5]]}")
    s = s.loc[:last_valid]
    worst = float(s.abs().max())
    if worst > max_abs:
        raise ValueError(f"{name}: |value| {worst} exceeds sanity bound "
                         f"{max_abs}")
    full = pd.date_range(s.index[0], s.index[-1], freq="MS")
    if len(full) != len(s):
        raise ValueError(f"{name}: month gaps in series")
    return s


def load_series() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name, (_, marker) in SOURCES.items():
        path = DATA_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run with --refresh")
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            raise ValueError(f"{name}: cached file lacks marker {marker!r}")
        blocks = parse_cpc_blocks(text, name)
        want = "STANDARDIZED DATA" if name == "soi" else "ANOMALY"
        if want not in blocks:
            raise ValueError(f"{name}: block {want!r} not found "
                             f"(have {sorted(blocks)})")
        out[name] = block_to_series(blocks[want], name,
                                    max_abs=6.0 if name == "soi" else 15.0)
    return out


def load_nino34() -> pd.Series:
    """Observed monthly Nino 3.4 anomaly from the repo's committed CSV."""
    if not NINO34_CSV.exists():
        raise FileNotFoundError(f"{NINO34_CSV} missing")
    df = pd.read_csv(NINO34_CSV)
    need = {"year", "month", "nino34_anom"}
    if not need.issubset(df.columns):
        raise ValueError(f"{NINO34_CSV.name}: expected columns {sorted(need)}")
    if df["nino34_anom"].abs().max() > 4.0:
        raise ValueError(f"{NINO34_CSV.name}: implausible Nino 3.4 values")
    idx = pd.to_datetime(df["year"].astype(str) + "-"
                         + df["month"].astype(str).str.zfill(2) + "-01")
    return pd.Series(df["nino34_anom"].values, index=idx,
                     name="nino34").sort_index()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

WIND_META = [
    ("wpac850", "W Pac (135°E–180°)", "blue"),
    ("cpac850", "C Pac (175°W–140°W)", "orange"),
    ("epac850", "E Pac (135°W–120°W)", "violet"),
]


def build_figure(series: dict[str, pd.Series], nino34: pd.Series,
                 dark_mode: bool) -> go.Figure:
    t = tokens(dark_mode)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        row_heights=[0.42, 0.58], specs=[[{}], [{"secondary_y": True}]])

    soi = series["soi"].loc[WINDOW_START:]
    if soi.empty:
        raise ValueError("SOI window is empty")
    colors = [t["ember"] if v < 0 else t["teal"] for v in soi.values]
    fig.add_trace(go.Bar(
        x=soi.index, y=soi.values, marker_color=colors, marker_line_width=0,
        name="SOI (standardized)", showlegend=False,
        hovertemplate="%{x|%b %Y}  SOI %{y:.1f}<extra></extra>"), row=1, col=1)

    for name, label, color_key in WIND_META:
        s = series[name].loc[WINDOW_START:]
        if s.empty:
            raise ValueError(f"{name} window is empty")
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(color=t[color_key], width=1.8),
            hovertemplate="%{x|%b %Y}  " + label
                          + " %{y:.1f} m/s<extra></extra>"),
            row=2, col=1, secondary_y=False)

    n34 = nino34.loc[WINDOW_START:]
    fig.add_trace(go.Scatter(
        x=n34.index, y=n34.values, mode="lines",
        name="Niño 3.4 (°C, right)",
        line=dict(color=t["red"], width=1.6, dash="dash"), opacity=0.85,
        hovertemplate="%{x|%b %Y}  Niño 3.4 %{y:.2f}°C<extra></extra>"),
        row=2, col=1, secondary_y=True)

    for r in (1, 2):
        fig.add_hline(y=0, line_width=1, line_color=t["zeroline"],
                      row=r, col=1)

    fig.update_yaxes(title_text="SOI (standardized)", row=1, col=1)
    fig.update_yaxes(title_text="Trade-wind anomaly (m/s)",
                     row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Niño 3.4 anomaly (°C)", row=2, col=1,
                     secondary_y=True, showgrid=False)

    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.16, xanchor="left",
        showarrow=False, align="left",
        font=dict(size=10.5, color=t["text_dim"]),
        text=("SOI: station-based observation — standardized Tahiti − Darwin "
              "sea-level pressure (NOAA CPC).  Winds: reanalysis-derived — "
              "CDAS/NCEP–NCAR 850 hPa trade-wind indices, base 1981–2010 "
              "(NOAA CPC).<br>Positive wind anomaly = stronger easterly "
              "trades; negative SOI and weakened trades lean El Niño."))

    fig.update_layout(
        template=template_name(dark_mode),
        bargap=0.15,
        margin=dict(l=64, r=56, t=28, b=86),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0),
    )
    return fig


def _assert_png(path: Path, w: int = 2400, h: int = 1000,
                min_bytes: int = 20 * 1024) -> None:
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
                    help="re-download the CPC source files first")
    args = ap.parse_args()

    if args.refresh:
        refresh_cache()
    series = load_series()
    nino34 = load_nino34()

    for dark in (True, False):
        out = HERE / f"figure_{'dark' if dark else 'light'}.png"
        fig = build_figure(series, nino34, dark)
        fig.write_image(str(out), **EXPORT)
        _assert_png(out)
        print(f"wrote {out.name} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
