#!/usr/bin/env python3
"""Warm Water Volume vs Nino 3.4 -- ENSO precursor proposal mock-up.

Single-panel monthly figure: PMEL's equatorial Pacific Warm Water Volume
anomaly (volume of water warmer than 20 degC, 5N-5S, 120E-80W -- derived
from ocean analyses, not a direct observation) against the observed
Nino 3.4 SST anomaly from the repo's committed CSV. Upper-ocean heat
content leads eastern-Pacific SST by roughly two to three seasons
(Meinen & McPhaden 2000, J. Climate 13, 3551-3559); the figure annotates
the lag correlation computed from the plotted window so the lead
relationship is visible rather than asserted.

The raw wwv.dat download is cached verbatim under ./data/ (MANIFEST.json
records URL, sha256, retrieval time). Default run is offline from the
cache; --refresh re-downloads. Parsing is strict: header sentence checked,
months must be contiguous, volumes and anomalies must fall in physical
ranges -- any surprise raises instead of substituting.

Usage:
    python make_figure.py            # offline, from ./data/
    python make_figure.py --refresh  # re-download wwv.dat first
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
WWV_FILE = DATA_DIR / "wwv.dat"
MANIFEST = DATA_DIR / "MANIFEST.json"
NINO34_CSV = REPO / "ENSO" / "data" / "observed" / "nino34_monthly.csv"

WWV_URL = "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat"
WWV_MARKER = "Warm Water Volume"

WINDOW_START = "1990-01-01"
MAX_LAG_MONTHS = 12
EXPORT = dict(width=1200, height=500, scale=2)  # repo standard full-width

UA = {"User-Agent": "climate-dashboard-proposal/0.1 (panel mock-up)"}


def refresh_cache() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    resp = requests.get(WWV_URL, headers=UA, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if len(text) < 5000:
        raise ValueError(f"wwv.dat: implausibly short payload ({len(text)})")
    if WWV_MARKER not in text.split("\n", 1)[0]:
        raise ValueError("wwv.dat: expected header marker "
                         f"{WWV_MARKER!r} not found -- layout changed?")
    WWV_FILE.write_text(text, encoding="utf-8", newline="")  # verbatim
    MANIFEST.write_text(json.dumps({
        "wwv.dat": {
            "url": WWV_URL,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        }
    }, indent=2) + "\n", encoding="utf-8")
    print(f"fetched wwv.dat: {len(text)} bytes")


def load_wwv() -> pd.Series:
    """WWV anomaly as a monthly Series in units of 1e14 m^3."""
    if not WWV_FILE.exists():
        raise FileNotFoundError(f"{WWV_FILE} missing -- run with --refresh")
    lines = WWV_FILE.read_text(encoding="utf-8").split("\n")
    if WWV_MARKER not in lines[0]:
        raise ValueError("cached wwv.dat lacks the expected header")
    try:
        head = next(i for i, l in enumerate(lines)
                    if l.split()[:3] == ["date", "Volume", "Anomaly"])
    except StopIteration:
        raise ValueError("wwv.dat: 'date Volume Anomaly' header not found")

    dates, anoms = [], []
    for ln in lines[head + 1:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"wwv.dat: expected 3 columns, got {ln!r}")
        ym, vol_s, anom_s = parts
        if len(ym) != 6 or not ym.isdigit():
            raise ValueError(f"wwv.dat: bad date field {ym!r}")
        vol, anom = float(vol_s), float(anom_s)
        if not 1.5e15 < vol < 4.0e15:
            raise ValueError(f"wwv.dat: volume {vol:.3e} outside physical "
                             f"range at {ym}")
        if abs(anom) > 1.5e15:
            raise ValueError(f"wwv.dat: anomaly {anom:.3e} implausible at {ym}")
        dates.append(pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01"))
        anoms.append(anom / 1e14)

    s = pd.Series(anoms, index=pd.DatetimeIndex(dates), name="wwv").sort_index()
    if len(s) < 400:
        raise ValueError(f"wwv.dat: only {len(s)} rows -- truncated download?")
    full = pd.date_range(s.index[0], s.index[-1], freq="MS")
    if len(full) != len(s) or s.index.has_duplicates:
        raise ValueError("wwv.dat: month gaps or duplicates")
    return s


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


def lead_correlation(wwv: pd.Series, nino34: pd.Series) -> tuple[float, int]:
    """Max Pearson r of corr(WWV(t), Nino3.4(t+k)) over k = 0..MAX_LAG."""
    best_r, best_k = -np.inf, 0
    for k in range(MAX_LAG_MONTHS + 1):
        shifted = nino34.copy()
        shifted.index = shifted.index - pd.DateOffset(months=k)
        joint = pd.concat([wwv, shifted], axis=1, join="inner").dropna()
        if len(joint) < 60:
            raise ValueError(f"lag {k}: only {len(joint)} overlapping months")
        r = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
        if r > best_r:
            best_r, best_k = r, k
    return best_r, best_k


def build_figure(wwv: pd.Series, nino34: pd.Series, r: float, k: int,
                 dark_mode: bool) -> go.Figure:
    t = tokens(dark_mode)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    w = wwv.loc[WINDOW_START:]
    n = nino34.loc[WINDOW_START:]
    if w.empty or n.empty:
        raise ValueError("plot window is empty")

    fig.add_trace(go.Scatter(
        x=w.index, y=w.values, mode="lines",
        name="WWV anomaly (right)",
        line=dict(color=t["teal"], width=2),
        hovertemplate="%{x|%b %Y}  WWV %{y:+.1f}×10¹⁴ m³<extra></extra>"),
        secondary_y=True)
    fig.add_trace(go.Scatter(
        x=n.index, y=n.values, mode="lines",
        name="Niño 3.4 anomaly",
        line=dict(color=t["ember"], width=2),
        hovertemplate="%{x|%b %Y}  Niño 3.4 %{y:+.2f}°C<extra></extra>"),
        secondary_y=False)

    fig.add_hline(y=0, line_width=1, line_color=t["zeroline"])

    # Symmetric ranges so the two zero lines coincide.
    n_max = float(n.abs().max()) * 1.15
    w_max = float(w.abs().max()) * 1.15
    fig.update_yaxes(title_text="Niño 3.4 SST anomaly (°C)",
                     range=[-n_max, n_max], secondary_y=False)
    fig.update_yaxes(title_text="WWV anomaly (10¹⁴ m³)",
                     range=[-w_max, w_max], secondary_y=True, showgrid=False)

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98, xanchor="left",
        showarrow=False, align="left",
        font=dict(size=12, color=t["text"]),
        bgcolor=t["panel"], opacity=0.85,
        text=(f"Warm water builds first: max r = {r:.2f} with Niño 3.4 "
              f"lagged {k} months (this window)"))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.16, xanchor="left",
        showarrow=False, align="left",
        font=dict(size=10.5, color=t["text_dim"]),
        text=("WWV: volume of water warmer than 20 °C, 5°N–5°S, 120°E–80°W "
              "(NOAA/PMEL, derived from ocean analyses — not a direct "
              "observation).  Niño 3.4: observed, NOAA CPC via the repo's "
              "committed series.<br>Precursor relationship: Meinen & "
              "McPhaden (2000), J. Climate 13, 3551–3559."))

    fig.update_layout(
        template=template_name(dark_mode),
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
                    help="re-download wwv.dat first")
    args = ap.parse_args()

    if args.refresh:
        refresh_cache()
    wwv = load_wwv()
    nino34 = load_nino34()

    r, k = lead_correlation(wwv.loc[WINDOW_START:], nino34)
    print(f"lead correlation: r={r:.3f} at Niño 3.4 lag +{k} months")

    for dark in (True, False):
        out = HERE / f"figure_{'dark' if dark else 'light'}.png"
        fig = build_figure(wwv, nino34, r, k, dark)
        fig.write_image(str(out), **EXPORT)
        _assert_png(out)
        print(f"wrote {out.name} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
