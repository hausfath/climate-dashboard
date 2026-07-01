"""Layout builders for the Climate Dashboard chrome.

Pure presentation helpers: topbar, hero blocks, KPI cards, numbered
sections, and figure panels. All theming is CSS-variable driven
(assets/theme.css); these builders only emit class names, never inline
theme styles.
"""

import re

import numpy as np
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc


# ---------------------------------------------------------------------------
# Chrome
# ---------------------------------------------------------------------------

def topbar(last_updated: str) -> html.Header:
    return html.Header([
        html.Div([
            "CLIMATE", html.Span("·", className="dot"), "DASHBOARD",
            html.Span("ERA5 · daily", className="brand-sub"),
        ], className="brand"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Temperature", id='nav-global', href='#global', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("ENSO Forecast", id='nav-enso', href='#enso', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Models vs Obs", id='nav-models', href='#models', n_clicks=0)),
        ], id='main-nav'),
        html.Div([
            html.Div([
                html.Span(className="pulse"),
                f"Updated {last_updated}",
            ], className="live"),
            html.Div([
                html.I(className="fas fa-sun", id='sun-icon'),
                dbc.Switch(id='dark-mode-switch', value=True, className="mb-0"),
                html.I(className="fas fa-moon tgl-on", id='moon-icon'),
            ], className="toggle-cluster"),
            html.Div([
                html.I(className="fas fa-image", id='static-icon'),
                dbc.Switch(id='interactive-switch', value=True, className="mb-0"),
                html.I(className="fas fa-chart-line tgl-on", id='interactive-icon'),
            ], className="toggle-cluster"),
        ], className="topbar-right"),
    ], className="topbar")


def hero(kicker: str, headline, lede, right=None,
         kicker_id: str = None, headline_id: str = None,
         lede_id: str = None, right_id: str = None) -> html.Div:
    kw = lambda i: {'id': i} if i else {}
    left = html.Div([
        html.Div(kicker, className="kicker", **kw(kicker_id)),
        html.H1(headline, **kw(headline_id)),
        html.P(lede, className="lede", **kw(lede_id)),
    ])
    right_el = html.Div(right, **kw(right_id)) if right_id else right
    children = [left] + ([right_el] if right_el is not None else [])
    return html.Div(children, className="hero")


def fmt_prob(p: float) -> str:
    """Format a probability for display, avoiding false certainty at the
    edges (0.997 → '>99%', 0.002 → '<1%')."""
    pct = p * 100
    if pct >= 99.5:
        return ">99%"
    if 0 < pct < 0.5:
        return "<1%"
    return f"{pct:.0f}%"


def prob_strip(label_left: str, label_right: str, segments, legend) -> html.Div:
    """segments: list of (fraction 0-1, bg, fg, show_label). legend: [(color, text)]."""
    segs = []
    for frac, bg, fg, show in segments:
        pct = max(frac, 0) * 100
        # Never display a bare "100%" — ensemble odds are not certainties.
        label = fmt_prob(frac) if show and pct >= 8 else ""
        segs.append(html.Div(
            label,
            className="seg",
            style={'width': f'{pct:.1f}%', 'background': bg, 'color': fg},
        ))
    return html.Div([
        html.Div([html.Span(label_left), html.B(label_right)], className="strip-label"),
        html.Div(segs, className="probbar"),
        html.Div([
            html.Span([html.Span(className="swatch", style={'background': c}), t])
            for c, t in legend
        ], className="striplegend"),
    ], className="herostrip")


def percentile_gauge(label_left: str, label_right: str, percentile: float,
                     mid_label: str) -> html.Div:
    pct = float(np.clip(percentile, 2, 98))
    return html.Div([
        html.Div([html.Span(label_left), html.B(label_right)], className="strip-label"),
        html.Div([
            html.Div(className="gauge-tick", style={'left': '50%'}),
            html.Div(className="gauge-marker", style={'left': f'{pct:.0f}%'}),
        ], className="gauge-track"),
        html.Div([
            html.Span("Cooler than models"),
            html.Span(mid_label, className="mid"),
            html.Span("Warmer than models"),
        ], className="gauge-ends"),
    ], className="herostrip")


def kpi(label, value, sub, value_id=None, sub_id=None, label_id=None) -> html.Div:
    kw = lambda i: {'id': i} if i else {}
    return html.Div([
        html.Div(label, className="k-label", **kw(label_id)),
        html.Div(value, className="k-value", **kw(value_id)),
        html.Div(sub, className="k-sub", **kw(sub_id)),
    ], className="kpi")


def kpi_row(cards) -> html.Div:
    return html.Div(cards, className="kpis")


def section(no: str, title: str, hint, desc, children, section_id=None) -> html.Div:
    kw = {'id': section_id} if section_id else {}
    return html.Div([
        html.Div([
            html.Span(no, className="no"),
            html.H2(title),
            html.Div(hint, className="hint"),
        ], className="sechead"),
        html.P(desc, className="secdesc"),
        *children,
    ], className="block", **kw)


def panel(title: str, img_id: str = None, img_src: str = None,
          graph_id: str = None, graph_height: int = 500,
          tag: str = None, caption=None, body=None,
          head_extra=None, alt: str = None) -> html.Div:
    head = [html.H3(title)]
    if tag:
        head.append(html.Span(tag, className="ptag"))
    if head_extra is not None:
        head.append(html.Div(head_extra, style={'marginLeft': 'auto'}))

    content = []
    if img_id:
        content.append(html.Img(id=img_id, src=img_src, alt=alt or title,
                                style={'width': '100%', 'height': 'auto'}))
    if graph_id:
        content.append(dcc.Loading(
            id=f"loading-{graph_id}", type="circle",
            children=[dcc.Graph(
                id=graph_id,
                style={'height': f'{graph_height}px', 'display': 'none'},
                config={'toImageButtonOptions': {'scale': 3},
                        'displaylogo': False},
            )],
        ))
    if body is not None:
        content.append(body)

    children = [
        html.Div(head, className="phead"),
        html.Div(content, className="pbody"),
    ]
    if caption is not None:
        children.append(html.Div(caption, className="pcap"))
    return html.Div(children, className="panel")


def duo(*panels) -> html.Div:
    return html.Div(list(panels), className="duo")


def footer_block(last_updated: str) -> html.Footer:
    return html.Footer([
        html.Span(["DATA · ",
                   html.A("ECMWF ERA5", href="https://pulse.climate.copernicus.eu/",
                          target="_blank"),
                   " · NOAA CPC · NMME · C3S · CanSIPS · CMIP"]),
        html.Span(["METHOD · ",
                   html.A("projection methodology", id='methodology-link',
                          href='#', n_clicks=0)]),
        html.Span(f"Updated {last_updated}"),
        html.Span("CLIMATE·DASHBOARD — rebuilt nightly 06:00 UTC", className="right"),
    ], className="footer")


# ---------------------------------------------------------------------------
# Hero data helpers
# ---------------------------------------------------------------------------

# Rank-strip palette (works on both themes)
RANK_COLORS = [('#b5432c', '#ffe3da'), ('#7d3a2e', '#f4c9bd'), ('#3c2f3d', '#b8a8be')]
ENSO_ODDS_COLORS = [('#8f1d1d', '#ffd9d4'), ('#c94b3a', '#ffe9e4'), ('#4a3040', '#c6b3c8')]


def temp_rank_strip(stats: dict) -> html.Div | None:
    """Bucket annual rank probabilities into warmest / 2nd / 3rd-or-lower."""
    probs = stats.get('annual_rank_probs') or []
    if not probs:
        return None
    p1 = sum(p['prob'] for p in probs if p['rank'] == 1)
    p2 = sum(p['prob'] for p in probs if p['rank'] == 2)
    p3 = max(1.0 - p1 - p2, 0.0)
    segments = [
        (p1, *RANK_COLORS[0], True),
        (p2, *RANK_COLORS[1], True),
        (p3, *RANK_COLORS[2], True),
    ]
    legend = [(RANK_COLORS[0][0], "Warmest on record"),
              (RANK_COLORS[1][0], "2nd warmest"),
              (RANK_COLORS[2][0], "3rd or lower")]
    return prob_strip(f"Where {stats['current_year']} likely ranks",
                      "Monte Carlo · ENSO ensemble", segments, legend)


def split_enso_state(state_str: str) -> tuple:
    """'El Niño: +1.5°C (Niño 3.4, Jun 2026)' → ('El Niño', '+1.5°C', 'Jun 2026')."""
    if not state_str or state_str == 'N/A':
        return 'N/A', '', ''
    m = re.match(r'([^:]+):\s*([+\-−]?[\d.]+°C)(?:\s*\(([^)]*)\))?', state_str)
    if not m:
        return state_str, '', ''
    label, val, paren = m.group(1).strip(), m.group(2), m.group(3) or ''
    when = paren.split(',')[-1].strip() if ',' in paren else paren
    return label, val, when


def enso_odds_view(pm: dict) -> dict | None:
    """Build hero-strip segments/legend from ``compute_peak_month_odds``
    output (odds at the latest month where all models report)."""
    if not pm:
        return None
    p_very, p_strong = pm['p_very'], pm['p_strong']
    p_rest = max(1.0 - p_very - p_strong, 0.0)
    segments = [
        (p_very, *ENSO_ODDS_COLORS[0], True),
        (p_strong, *ENSO_ODDS_COLORS[1], True),
        (p_rest, *ENSO_ODDS_COLORS[2], False),
    ]
    legend = [(ENSO_ODDS_COLORS[0][0], "Very strong (≥2.0°C)"),
              (ENSO_ODDS_COLORS[1][0], "Strong (1.5–2.0)"),
              (ENSO_ODDS_COLORS[2][0], "Weaker")]
    return {
        'month_label': pm['month_label'],
        'kind': pm['kind'],
        'segments': segments,
        'legend': legend,
        'p_very': p_very,
        'n_models': pm['n_models'],
    }


def models_alignment(cmip_df, obs_df) -> dict | None:
    """Percentile of the mean observed 1970–present trend within the CMIP
    member trend distribution."""
    try:
        from src.models_vs_obs import calculate_member_trends, calculate_obs_trends
        end = str(obs_df['date'].max().date())
        member_trends = calculate_member_trends(cmip_df, '1970-01-01', end)
        obs_trends = calculate_obs_trends(obs_df, '1970-01-01', end)
        obs_vals = [v for v in obs_trends.values() if v is not None]
        if len(member_trends) == 0 or not obs_vals:
            return None
        obs_mean = float(np.mean(obs_vals))
        pct = float((member_trends < obs_mean).mean() * 100)
        if pct < 25:
            phrase = "on the cool side"
        elif pct > 75:
            phrase = "on the warm side"
        else:
            phrase = "through the middle"
        return {'percentile': pct, 'phrase': phrase, 'obs_trend': obs_mean}
    except Exception:
        return None
