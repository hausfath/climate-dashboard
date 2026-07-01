"""Design system for the Climate Dashboard.

Owns the color tokens shared by the page CSS (assets/theme.css) and the
Plotly figures, the registered Plotly templates (``climate_dark`` /
``climate_light``), and font installation so kaleido (static PNG export)
and matplotlib (ridgeline) render the same faces the browser does.
"""

import logging
import platform
import shutil
import subprocess
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

logger = logging.getLogger(__name__)

FONTS_DIR = Path(__file__).parent.parent / 'assets' / 'fonts'

FONT_BODY = 'IBM Plex Sans'
FONT_MONO = 'IBM Plex Mono'
FONT_DISPLAY = 'Bricolage Grotesque'

_BODY_STACK = f'{FONT_BODY}, -apple-system, Segoe UI, sans-serif'
_MONO_STACK = f'{FONT_MONO}, SFMono-Regular, Menlo, monospace'

# ---------------------------------------------------------------------------
# Color tokens — keep in sync with assets/theme.css custom properties
# ---------------------------------------------------------------------------

DARK = {
    'bg': '#0b0e1a',           # page background
    'panel': '#151a30',        # figure well / raised panels
    'panel_raised': '#12162a', # KPI cards
    'text': '#e8eaf2',
    'text_dim': '#9aa0b8',
    'text_faint': '#61677f',
    'line': 'rgba(232, 234, 242, 0.08)',
    'grid': 'rgba(232, 234, 242, 0.07)',
    'zeroline': 'rgba(232, 234, 242, 0.18)',
    # accents
    'ember': '#ff6b4a',
    'teal': '#4ecdc4',
    'violet': '#a29bfe',
    'gold': '#e8b84b',
    'orange': '#ff9f43',
    'red': '#ff6b6b',
    'blue': '#54a0ff',
}

LIGHT = {
    'bg': '#faf9f6',
    'panel': '#ffffff',
    'panel_raised': '#ffffff',
    'text': '#1f2430',
    'text_dim': '#5a6172',
    'text_faint': '#9198a8',
    'line': 'rgba(31, 36, 48, 0.10)',
    'grid': 'rgba(31, 36, 48, 0.08)',
    'zeroline': 'rgba(31, 36, 48, 0.22)',
    'ember': '#d94f2e',
    'teal': '#0f9d94',
    'violet': '#6c5ce7',
    'gold': '#c08a1e',
    'orange': '#e8890c',
    'red': '#d64541',
    'blue': '#2e6fc9',
}


def tokens(dark_mode: bool = True) -> dict:
    return DARK if dark_mode else LIGHT


# ---------------------------------------------------------------------------
# Plotly templates
# ---------------------------------------------------------------------------

def _build_template(t: dict) -> go.layout.Template:
    axis_common = dict(
        gridcolor=t['grid'],
        gridwidth=1,
        zerolinecolor=t['zeroline'],
        zerolinewidth=1,
        linecolor=t['line'],
        showline=False,
        ticks='',
        tickfont=dict(family=_MONO_STACK, size=11.5, color=t['text_dim']),
        title=dict(font=dict(family=_BODY_STACK, size=12.5, color=t['text_dim'])),
        automargin=True,
    )
    return go.layout.Template(layout=go.Layout(
        font=dict(family=_BODY_STACK, size=13, color=t['text']),
        paper_bgcolor=t['panel'],
        plot_bgcolor=t['panel'],
        margin=dict(l=64, r=28, t=28, b=48),
        xaxis=axis_common,
        yaxis=axis_common,
        colorway=[t['teal'], t['orange'], t['red'], t['violet'],
                  t['gold'], t['blue'], t['ember']],
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='left', x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(family=_BODY_STACK, size=12, color=t['text_dim']),
        ),
        hoverlabel=dict(
            bgcolor=t['panel_raised'] if t is DARK else '#ffffff',
            bordercolor=t['line'],
            font=dict(family=_MONO_STACK, size=12, color=t['text']),
        ),
        hovermode='x unified',
    ))


pio.templates['climate_dark'] = _build_template(DARK)
pio.templates['climate_light'] = _build_template(LIGHT)


def template_name(dark_mode: bool = True) -> str:
    return 'climate_dark' if dark_mode else 'climate_light'


def end_label(fig: go.Figure, x, y, text: str, color: str,
              size: float = 12, xshift: int = 8) -> None:
    """Direct line-end label — replaces a legend entry."""
    fig.add_annotation(
        x=x, y=y, text=text,
        font=dict(family=_MONO_STACK, size=size, color=color),
        showarrow=False, xanchor='left', xshift=xshift,
    )


# ---------------------------------------------------------------------------
# Fonts — make the bundled faces visible to kaleido and matplotlib
# ---------------------------------------------------------------------------

_fonts_installed = False


def install_fonts() -> None:
    """Copy bundled fonts to the user font dir (for kaleido's chromium) and
    register them with matplotlib. Safe to call repeatedly."""
    global _fonts_installed
    if _fonts_installed:
        return
    _fonts_installed = True

    if not FONTS_DIR.exists():
        logger.warning(f'Font directory missing: {FONTS_DIR}')
        return

    if platform.system() == 'Darwin':
        user_font_dir = Path.home() / 'Library' / 'Fonts'
    else:
        user_font_dir = Path.home() / '.fonts'
    try:
        user_font_dir.mkdir(parents=True, exist_ok=True)
        copied = False
        for src in FONTS_DIR.glob('*.ttf'):
            dst = user_font_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied = True
        if copied and platform.system() == 'Linux':
            subprocess.run(['fc-cache', '-f'], capture_output=True, timeout=30)
    except Exception as e:
        logger.warning(f'Could not install fonts for static export: {e}')

    try:
        from matplotlib import font_manager
        for src in FONTS_DIR.glob('*.ttf'):
            font_manager.fontManager.addfont(str(src))
    except Exception as e:
        logger.warning(f'Could not register fonts with matplotlib: {e}')
