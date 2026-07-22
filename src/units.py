"""°C → °F conversion for the dashboard's Plotly figures.

Every figure in the dashboard is authored in °C. When the topbar °F
toggle is on, each figure callback passes its finished figure through
``convert_figure_units()`` just before returning it — plot functions
never need to know about units.

Conversion is axis-aware: only data plotted against an axis whose title
mentions °C is rescaled, so date/month/probability/count axes pass
through untouched (this matters for the trend-histogram grid, where the
temperature quantity lives on the x axes).

Two kinds of quantity:
  - anomalies / trends (the default): scale by 9/5, no offset;
  - absolute temperatures (``absolute=True`` — the daily-absolutes plot
    and the temperature heatmap): scale by 9/5 and add 32.
Values embedded in text (annotations, trace names, hover templates,
prebuilt hover strings, category labels) are converted by regex;
±-prefixed values are uncertainty half-widths and always scale-only.

The dashboard chrome (KPI cards, heroes, captions) is converted
separately, client-side, by assets/units.js.
"""

import re

import numpy as np

# "+1.5°C", "−0.5 °C", "±0.16°C", "29.43°C" — sign captured separately.
_NUM_C = re.compile(r'([+\-−±]?)(\d+(?:\.\d+)?)\s*°C')

# "29.43 ±0.16°C" — the leading value carries no °C of its own; convert it
# in a pre-pass (the ± half-width is handled by _NUM_C afterwards).
_VAL_BEFORE_PM = re.compile(r'([+\-−]?\d+(?:\.\d+)?)(?=\s*±\s*\d+(?:\.\d+)?\s*°C)')


def _decode_bdata(values):
    """Plotly ≥6 may hand arrays back as {'dtype', 'bdata'[, 'shape']}
    base64 blobs (e.g. after a figure has been JSON round-tripped);
    decode those to ndarrays, pass everything else through."""
    if isinstance(values, dict) and 'bdata' in values:
        import base64
        arr = np.frombuffer(base64.b64decode(values['bdata']),
                            dtype=np.dtype(values.get('dtype', 'f8')))
        if values.get('shape'):
            arr = arr.reshape([int(s) for s in str(values['shape']).split(',')])
        return arr
    return values


def _c2f_array(values, absolute: bool):
    arr = np.asarray(_decode_bdata(values), dtype=float)
    return arr * 1.8 + (32.0 if absolute else 0.0)


def _c2f_scalar(v, absolute: bool) -> float:
    return float(v) * 1.8 + (32.0 if absolute else 0.0)


def convert_text(s, absolute: bool = False):
    """Convert numeric °C mentions in a string to °F, then relabel any
    remaining bare '°C' (axis titles like 'Trend (°C/decade)')."""
    if not isinstance(s, str) or '°C' not in s:
        return s

    def repl(m):
        sign, num = m.group(1), m.group(2)
        dec = len(num.split('.')[1]) if '.' in num else 0
        val = -float(num) if sign in ('-', '−') else float(num)
        # ± marks an uncertainty half-width: scale only, never offset.
        if sign == '±':
            return f"±{abs(val) * 1.8:.{dec}f}°F"
        f = _c2f_scalar(val, absolute)
        if sign == '+' and f >= 0:
            return f"+{f:.{dec}f}°F"
        return f"{f:.{dec}f}°F"

    def repl_pm(m):
        num = m.group(1).replace('−', '-')
        dec = len(num.split('.')[1]) if '.' in num else 0
        f = _c2f_scalar(float(num), absolute)
        sign = '+' if num.startswith('+') and f >= 0 else ''
        return f"{sign}{f:.{dec}f}"

    s = _VAL_BEFORE_PM.sub(repl_pm, s)
    return _NUM_C.sub(repl, s).replace('°C', '°F')


def _convert_str_seq(seq, absolute: bool):
    """Elementwise convert_text over a (possibly nested) sequence, leaving
    non-strings alone. Returns None if nothing needed conversion."""
    changed = False

    def walk(item):
        nonlocal changed
        if isinstance(item, str):
            new = convert_text(item, absolute)
            changed = changed or (new is not item and new != item)
            return new
        if isinstance(item, (list, tuple, np.ndarray)):
            return [walk(v) for v in item]
        return item

    out = walk(list(seq))
    return out if changed else None


def _numeric(seq) -> bool:
    """True if seq looks like a numeric array (data on a temp axis)."""
    try:
        arr = np.asarray(_decode_bdata(seq), dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.size > 0


def _temp_axes(fig, absolute: bool):
    """Find axes whose titles mention °C; convert those titles (and any
    explicit ranges/tickvals) and return the axis ids, e.g. {'x','x3'}."""
    temp_x, temp_y = set(), set()
    layout_json = fig.layout.to_plotly_json()
    for key, val in layout_json.items():
        if not (key.startswith('xaxis') or key.startswith('yaxis')):
            continue
        title = val.get('title')
        text = title.get('text') if isinstance(title, dict) else title
        if not (isinstance(text, str) and '°C' in text):
            continue
        axis_id = ('x' if key[0] == 'x' else 'y') + key[5:]
        (temp_x if key[0] == 'x' else temp_y).add(axis_id)
        fig.layout[key].title.text = convert_text(text, absolute)
        if val.get('range') is not None:
            fig.layout[key].range = [
                _c2f_scalar(r, absolute) for r in val['range']]
        if val.get('tickvals') is not None:
            fig.layout[key].tickvals = _c2f_array(val['tickvals'], absolute)
    return temp_x, temp_y


def _convert_trace(tr, temp_x, temp_y, absolute):
    xid = getattr(tr, 'xaxis', None) or 'x'
    yid = getattr(tr, 'yaxis', None) or 'y'

    # Heatmaps carry temperature in z, flagged by the colorbar title.
    if tr.type in ('heatmap', 'contour'):
        cb = getattr(tr, 'colorbar', None)
        cb_text = cb.title.text if cb is not None and cb.title is not None else None
        if isinstance(cb_text, str) and '°C' in cb_text:
            if tr.z is not None:
                tr.z = _c2f_array(tr.z, absolute)
            tr.colorbar.title.text = convert_text(cb_text, absolute)
            for attr in ('zmin', 'zmax'):
                v = getattr(tr, attr, None)
                if v is not None:
                    setattr(tr, attr, _c2f_scalar(v, absolute))
    else:
        for dim, temp_ids, axis in (('y', temp_y, yid), ('x', temp_x, xid)):
            data = getattr(tr, dim, None)
            if axis in temp_ids and data is not None and _numeric(data):
                setattr(tr, dim, _c2f_array(data, absolute))
            # error bars are half-widths: scale only
            err = getattr(tr, f'error_{dim}', None)
            if err is not None and axis in temp_ids:
                for eattr in ('array', 'arrayminus'):
                    ev = getattr(err, eattr, None)
                    if ev is not None:
                        setattr(err, eattr, _c2f_array(ev, False))
                if getattr(err, 'value', None) is not None:
                    err.value = float(err.value) * 1.8

    # Text payloads: names, hover templates, prebuilt hover/label strings,
    # and categorical axis values that embed °C thresholds.
    if isinstance(getattr(tr, 'name', None), str):
        tr.name = convert_text(tr.name, absolute)
    if isinstance(getattr(tr, 'hovertemplate', None), str):
        tr.hovertemplate = convert_text(tr.hovertemplate, absolute)
    for attr in ('text', 'hovertext'):
        v = getattr(tr, attr, None)
        if isinstance(v, str):
            setattr(tr, attr, convert_text(v, absolute))
        elif v is not None:
            new = _convert_str_seq(v, absolute)
            if new is not None:
                setattr(tr, attr, new)
    for dim in ('x', 'y'):
        v = getattr(tr, dim, None)
        if v is not None and not _numeric(v):
            new = _convert_str_seq(v, absolute)
            if new is not None:
                setattr(tr, dim, new)


def convert_figure_units(fig, fahrenheit: bool, absolute: bool = False):
    """Convert a °C figure to °F in place (no-op when ``fahrenheit`` is
    falsy). ``absolute=True`` for figures showing absolute temperatures."""
    if not fahrenheit or fig is None:
        return fig

    temp_x, temp_y = _temp_axes(fig, absolute)

    for tr in fig.data:
        _convert_trace(tr, temp_x, temp_y, absolute)

    # An unset x/yref means the default 'x'/'y' axis.
    for ann in fig.layout.annotations or ():
        if isinstance(ann.text, str):
            ann.text = convert_text(ann.text, absolute)
        if (ann.yref or 'y') in temp_y and isinstance(ann.y, (int, float)):
            ann.y = _c2f_scalar(ann.y, absolute)
        if (ann.xref or 'x') in temp_x and isinstance(ann.x, (int, float)):
            ann.x = _c2f_scalar(ann.x, absolute)

    for shape in fig.layout.shapes or ():
        if (shape.yref or 'y') in temp_y:
            for attr in ('y0', 'y1'):
                v = getattr(shape, attr, None)
                if isinstance(v, (int, float)):
                    setattr(shape, attr, _c2f_scalar(v, absolute))
        if (shape.xref or 'x') in temp_x:
            for attr in ('x0', 'x1'):
                v = getattr(shape, attr, None)
                if isinstance(v, (int, float)):
                    setattr(shape, attr, _c2f_scalar(v, absolute))

    title = fig.layout.title
    if title is not None and isinstance(title.text, str):
        fig.layout.title.text = convert_text(title.text, absolute)

    # Category orderings must convert in lockstep with categorical values.
    layout_json = fig.layout.to_plotly_json()
    for key, val in layout_json.items():
        if (key.startswith('xaxis') or key.startswith('yaxis')) and \
                val.get('categoryarray') is not None:
            new = _convert_str_seq(val['categoryarray'], absolute)
            if new is not None:
                fig.layout[key].categoryarray = new

    return fig
