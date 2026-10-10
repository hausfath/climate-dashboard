"""Which forecast initialization the pipeline runs on, and the season bookkeeping that follows from it.

INIT: `ELNINO_INIT=YYYYMM` in the environment, else the newest raw/nmme_YYYYMM folder.
Seasons are defined by calendar month; their lead indices follow from the init month, so a September start
has SON..MAM and an October start OND..MAM (SON is dropped because it begins before the start month).
"""
import os
from pathlib import Path

import xarray as xr

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
MNAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NMME = ["CFSv2", "CanESM5", "GEM5.2_NEMO", "NASA_GEOS5v2", "NCAR_CCSM4", "NCAR_CESM1"]
C3S_IND = ["ecmwf", "ukmo", "meteo_france", "dwd", "cmcc", "jma", "bom"]   # centres not already in NMME
C3S_DUP = ["ncep", "eccc"]                                                   # = CFSv2 / CanSIPS: consistency check only

# calendar months of each season (12 = Dec; the following months roll into the next year)
SEASON_DEF = {"SON": [9, 10, 11], "OND": [10, 11, 12], "DJF": [12, 1, 2], "JF": [1, 2], "JFM": [1, 2, 3], "MAM": [3, 4, 5]}
SEASON_ORDER = ["SON", "OND", "DJF", "JF", "JFM", "MAM"]


def _resolve_init():
    ym = os.environ.get("ELNINO_INIT")
    if not ym:
        runs = sorted(RAW.glob("nmme_??????"))
        if not runs:
            raise SystemExit("no raw/nmme_YYYYMM folder: run model_patterns/fetch_nmme.sh YYYYMM first "
                             "(and fetch_cds.py YYYYMM), or set ELNINO_INIT")
        ym = runs[-1].name[5:]
    return int(ym[:4]), int(ym[4:])


INIT_YEAR, INIT_MONTH = _resolve_init()
INIT_YM = f"{INIT_YEAR}{INIT_MONTH:02d}"
INIT_LABEL = f"{MNAME[INIT_MONTH - 1]} {INIT_YEAR}"           # "Oct 2026": export_dashboard parses this form
NMME_DIR, C3S_DIR, SST_DIR = RAW / f"nmme_{INIT_YM}", RAW / f"c3s_{INIT_YM}", RAW / f"nmme_{INIT_YM}_sst"
PROB_FILE = RAW / "nmme_prob" / f"prate.{INIT_YM}.prob.adj.seas.nc"


def _leads():
    nm = next(NMME_DIR.glob("*.prate.*.nc"), None)
    c3 = C3S_DIR / "ecmwf_tprate.nc"
    if nm is None or not c3.exists():
        raise SystemExit(f"raw forecast files for {INIT_YM} missing under {RAW} (need nmme_{INIT_YM}/ and c3s_{INIT_YM}/)")
    n = xr.open_dataset(nm, decode_times=False).fcst.shape[0]
    c = int(xr.open_dataset(c3).forecastMonth.size)
    return n, c


NMME_LEADS, C3S_LEADS = _leads()


def season_offsets(s):
    """Lead offsets (0 = start month) of a season's months, or None if it starts before the start month."""
    first = (SEASON_DEF[s][0] - INIT_MONTH) % 12
    return [first + k for k in range(len(SEASON_DEF[s]))]


def season_abs_months(s):
    """Absolute month indices (year*12 + month-1) of the season for this event."""
    return [INIT_YEAR * 12 + (INIT_MONTH - 1) + o for o in season_offsets(s)]


def season_target_time(s):
    """Decimal year at the season's midpoint (used for the temperature trend adjustment)."""
    m = season_abs_months(s)
    return m[0] // 12 + ((m[0] % 12) + len(m) / 2) / 12


# seasons the forecast covers: name -> (NMME lead indices, C3S forecastMonth list or None, target time)
SEASONS = {}
for _s in SEASON_ORDER:
    _off = season_offsets(_s)
    if _off[-1] < NMME_LEADS:                           # within the NMME horizon (a season that began before the start wraps past it)
        _c3s = [o + 1 for o in _off] if _off[-1] < C3S_LEADS else None
        SEASONS[_s] = (_off, _c3s, season_target_time(_s))
ALL_MODEL_SEASONS = [s for s, v in SEASONS.items() if v[1] is not None]   # 13-system seasons
MAP_SEASONS = [s for s in SEASONS if len(SEASON_DEF[s]) == 3]               # 3-month seasons shown on the map
PROB_LEAD = {s: SEASONS[s][0][0] for s in SEASONS}   # NMME tercile file: its first target code = the season starting at lead 0
                                                       # (codes were 800.. for the Sep 2026 file and 801.. for Oct 2026; process.py check [3] verifies)


def ens_for(s):
    """Models with a forecast for season s: 13 where C3S covers it, else the 6 NMME."""
    return NMME + C3S_IND if SEASONS[s][1] is not None else NMME


def season_or_fallback(s):
    """A requested season, or the first later season the forecast covers (SON -> OND for an October start)."""
    if s in SEASONS:
        return s
    for t in SEASON_ORDER[SEASON_ORDER.index(s) + 1:]:
        if t in SEASONS and len(SEASON_DEF[t]) == len(SEASON_DEF[s]):
            return t
    raise KeyError(s)


def season_label(s):
    m = season_abs_months(s)
    ya, yb = m[0] // 12, m[-1] // 12
    a, b = MNAME[m[0] % 12], MNAME[m[-1] % 12]
    return f"{a}–{b} {ya}" if ya == yb else f"{a} {ya}–{b} {yb}"


if __name__ == "__main__":
    print(f"init {INIT_LABEL} ({INIT_YM}); NMME leads {NMME_LEADS}, C3S leads {C3S_LEADS}")
    for s, (n, c, t) in SEASONS.items():
        print(f"  {s:4s} {season_label(s):18s} nmme {n} c3s {c} target {t:.3f} prob {PROB_TARGET[s]} models {len(ens_for(s))}")
