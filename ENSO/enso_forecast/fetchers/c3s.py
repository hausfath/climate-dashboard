"""Fetch C3S/Copernicus seasonal SST anomaly forecasts via CDS API."""

import logging
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from enso_forecast.config import (
    C3S_ANOMALY_BASE_PERIOD,
    C3S_DATASET,
    C3S_MODELS,
    C3S_VARIABLE,
    CDS_API_KEY,
    CDS_API_URL,
    FORECASTS_DIR,
    NINO34_LAT_BOUNDS,
    NINO34_LON_BOUNDS_180,
    NINO34_LON_BOUNDS_360,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# Cache for constraints lookups {(centre, year, month): response dict}
_constraints_cache: dict[tuple[str, int, int], dict] = {}

# Per-retrieve timeout. A CDS job can sit in the queue for hours when MARS is
# restricted or backed up, eating the workflow timeout — so we poll with a
# deadline and CANCEL the job server-side when we give up. Abandoning the job
# (the old thread-timeout approach) left it queued on CDS, counting against
# the per-user/dataset queue quota; enough zombies and CDS rejects every new
# submission with "Number queued requests for this dataset is temporarily
# limited" until they drain.
C3S_RETRIEVE_TIMEOUT_SEC = 900  # 15 minutes
_POLL_INTERVAL_SEC = 10


def _cds_headers() -> dict:
    return {"PRIVATE-TOKEN": CDS_API_KEY}


def _job_rejection_reason(job_id: str) -> str:
    """Best-effort extraction of why a CDS job was rejected/failed.

    The job-status payload carries an empty log for rejections; the real
    reason (e.g. queue-limit throttling) is in the results body.
    """
    try:
        r = requests.get(
            f"{CDS_API_URL}/retrieve/v1/jobs/{job_id}/results",
            headers=_cds_headers(), timeout=30,
        )
        body = r.json()
        return body.get("traceback") or body.get("title") or str(body)[:300]
    except Exception:
        return "(no reason available)"


def _retrieve_direct(request: dict, target: str, timeout_sec: float) -> None:
    """Submit a CDS retrieve, poll to completion, download the result.

    Replaces cdsapi.Client.retrieve so we hold the job ID: on deadline we
    DELETE the job server-side instead of leaving a zombie in the queue, and
    on rejection we surface CDS's actual reason instead of a bare 400.
    """
    base = f"{CDS_API_URL}/retrieve/v1"
    r = requests.post(
        f"{base}/processes/{C3S_DATASET}/execution",
        headers=_cds_headers(), json={"inputs": request}, timeout=60,
    )
    r.raise_for_status()
    job_id = r.json()["jobID"]

    deadline = time.monotonic() + timeout_sec
    status = r.json().get("status", "accepted")
    while status in ("accepted", "running"):
        if time.monotonic() >= deadline:
            try:
                requests.delete(f"{base}/jobs/{job_id}",
                                headers=_cds_headers(), timeout=30)
                logger.info("C3S: cancelled timed-out CDS job %s", job_id)
            except Exception as e:  # noqa: BLE001 — cancellation is best-effort
                logger.warning("C3S: failed to cancel job %s: %s", job_id, e)
            raise TimeoutError(
                f"CDS retrieve exceeded {timeout_sec:.0f}s for {C3S_DATASET} "
                f"(job {job_id} cancelled)"
            )
        time.sleep(_POLL_INTERVAL_SEC)
        s = requests.get(f"{base}/jobs/{job_id}",
                         headers=_cds_headers(), timeout=30)
        s.raise_for_status()
        status = s.json().get("status")

    if status != "successful":
        raise RuntimeError(
            f"CDS job {job_id} {status}: {_job_rejection_reason(job_id)}"
        )

    res = requests.get(f"{base}/jobs/{job_id}/results",
                       headers=_cds_headers(), timeout=60)
    res.raise_for_status()

    def _find_href(node):
        if isinstance(node, dict):
            if "href" in node:
                return node["href"]
            for v in node.values():
                if (h := _find_href(v)) is not None:
                    return h
        elif isinstance(node, list):
            for v in node:
                if (h := _find_href(v)) is not None:
                    return h
        return None

    href = _find_href(res.json())
    if not href:
        raise RuntimeError(f"CDS job {job_id}: no download href in results")
    with requests.get(href, headers=_cds_headers(), stream=True, timeout=300) as dl:
        dl.raise_for_status()
        with open(target, "wb") as f:
            for chunk in dl.iter_content(chunk_size=65536):
                f.write(chunk)


def _resolve_constraints(centre: str, year: int, month: int) -> dict:
    """Return the CDS constraints payload (valid systems, leads, …) for a
    centre/year/month combo. Cached for the session; {} on lookup failure."""
    key = (centre, year, month)
    if key in _constraints_cache:
        return _constraints_cache[key]
    data: dict = {}
    try:
        url = f"{CDS_API_URL}/retrieve/v1/processes/{C3S_DATASET}/constraints"
        r = requests.post(url, headers=_cds_headers(), json={
            "inputs": {
                "originating_centre": [centre],
                "variable": [C3S_VARIABLE],
                "year": [str(year)],
                "month": [f"{month:02d}"],
            }
        }, timeout=30)
        if r.status_code == 200:
            data = r.json()
    except Exception as e:
        logger.debug("Constraints lookup failed for %s: %s", centre, e)
    _constraints_cache[key] = data
    return data


def _resolve_system(centre: str, year: int, month: int, configured_system: str) -> str:
    """Return the correct system version for a centre/year/month combo.

    Uses the CDS constraints API to find the valid system when the
    configured default doesn't match (centres periodically bump versions).
    """
    systems = _resolve_constraints(centre, year, month).get("system", [])
    if configured_system in systems or not systems:
        return configured_system
    # Use the highest-numbered system (latest version)
    best = max(systems, key=lambda s: int(s) if s.isdigit() else 0)
    logger.info(
        "C3S %s: system %s not valid for %d-%02d, using system %s",
        centre, configured_system, year, month, best,
    )
    return best


def _resolve_leads(centre: str, year: int, month: int, configured_max: int) -> list[str]:
    """Return the leadtime_month list to request, clamped to what the CDS
    constraints say is valid — asking for an out-of-range lead (e.g. 7 when
    the postprocessed-anomalies dataset only carries 1-6) rejects the whole
    job. Falls back to the configured range if the lookup fails."""
    valid = _resolve_constraints(centre, year, month).get("leadtime_month", [])
    valid_ints = sorted(int(v) for v in valid if str(v).isdigit())
    if not valid_ints:
        return [str(i) for i in range(1, configured_max + 1)]
    max_lead = min(configured_max, valid_ints[-1])
    if max_lead < configured_max:
        logger.info(
            "C3S %s: clamping leads to 1-%d (configured 1-%d; CDS allows max %d)",
            centre, max_lead, configured_max, valid_ints[-1],
        )
    return [str(i) for i in valid_ints if i <= max_lead]


TROPICAL_LAT_BOUNDS = (-20.0, 20.0)


def _compute_area_mean(
    ds: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float] | None,
) -> xr.DataArray:
    """Cosine-latitude-weighted area mean over a lat/lon box.

    Preserves non-spatial dimensions (number/member, forecastMonth, etc.).
    When ``lon_bounds`` is None, averages over all longitudes (used for the
    20S-20N tropical band).
    """
    var_name = None
    for v in ds.data_vars:
        if "sst" in v.lower() or "anom" in v.lower() or "temperature" in v.lower():
            var_name = v
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

    lat_name = lon_name = None
    for dim in data.dims:
        dl = dim.lower()
        if "lat" in dl:
            lat_name = dim
        elif "lon" in dl:
            lon_name = dim

    if lat_name is None or lon_name is None:
        keep = {"number", "forecastmonth", "forecastMonth", "leadtime_month"}
        spatial = [d for d in data.dims if d.lower() not in {k.lower() for k in keep}]
        if spatial:
            return data.mean(dim=spatial)
        return data

    lat = ds.coords[lat_name]

    squeeze_dims = []
    for dim in data.dims:
        if dim in (lat_name, lon_name):
            continue
        if data.sizes[dim] == 1 and dim.lower() not in ("number", "forecastmonth"):
            squeeze_dims.append(dim)
    if squeeze_dims:
        data = data.squeeze(dim=squeeze_dims, drop=True)

    lat_min, lat_max = lat_bounds
    lat_slice = slice(lat_max, lat_min) if float(lat[0]) > float(lat[-1]) else slice(lat_min, lat_max)
    sel = {lat_name: lat_slice}
    if lon_bounds is not None:
        lon_min, lon_max = lon_bounds
        sel[lon_name] = slice(lon_min, lon_max)
    region = data.sel(sel)

    if region.sizes[lat_name] == 0 or region.sizes[lon_name] == 0:
        raise ValueError(f"Empty area selection for lat={lat_bounds} lon={lon_bounds}")

    weights = np.cos(np.deg2rad(region.coords[lat_name]))
    return region.weighted(weights).mean(dim=[lat_name, lon_name])


def _compute_nino34_area_mean(ds: xr.Dataset) -> xr.DataArray:
    """Cosine-latitude-weighted Niño 3.4 area mean (back-compat wrapper)."""
    return _compute_area_mean(ds, NINO34_LAT_BOUNDS, NINO34_LON_BOUNDS_180)


def _compute_tropical_mean(ds: xr.Dataset) -> xr.DataArray:
    """Cosine-latitude-weighted 20S-20N tropical mean (all longitudes)."""
    return _compute_area_mean(ds, TROPICAL_LAT_BOUNDS, lon_bounds=None)


def _extract_records(
    nino34: xr.DataArray,
    model_name: str,
    init_date: str,
    tropical: xr.DataArray | None = None,
) -> list[dict]:
    """Extract forecast records, attaching tropical-mean and rONI when available."""
    records = []

    fm_dim = None
    for dim in nino34.dims:
        if "forecast" in dim.lower() or "lead" in dim.lower():
            fm_dim = dim
            break
    if fm_dim is None and len(nino34.dims) == 1:
        fm_dim = nino34.dims[0]

    member_dim = None
    for dim in nino34.dims:
        if "number" in dim.lower() or "member" in dim.lower():
            member_dim = dim
            break

    init_dt = datetime.strptime(init_date, "%Y-%m-%d")

    if member_dim is not None:
        n_members = nino34.sizes[member_dim]
        member_coords = nino34.coords[member_dim].values
    else:
        n_members = 0
        member_coords = []

    def _add_records(n34_1d, trop_1d, member_id: str):
        if fm_dim is None:
            val = float(n34_1d.values)
            if np.isnan(val):
                return
            trop_val = float(trop_1d.values) if trop_1d is not None else float("nan")
            roni = val - trop_val if not np.isnan(trop_val) else float("nan")
            records.append({
                "source": "C3S",
                "model": model_name,
                "model_type": "dynamical",
                "init_date": init_date,
                "target_month": pd.Timestamp(init_date).strftime("%Y-%m"),
                "lead_months": 0,
                "nino34_anom": round(val, 4),
                "tropical_mean_anom": round(trop_val, 4) if not np.isnan(trop_val) else float("nan"),
                "roni_anom": round(roni, 4) if not np.isnan(roni) else float("nan"),
                "member_id": member_id,
                "temporal_resolution": "monthly",
                "anomaly_base_period": C3S_ANOMALY_BASE_PERIOD,
            })
            return

        fm_coords = nino34.coords.get(fm_dim)
        for t_idx in range(n34_1d.sizes[fm_dim]):
            val = float(n34_1d.isel({fm_dim: t_idx}).values)
            if np.isnan(val):
                continue
            trop_val = float(trop_1d.isel({fm_dim: t_idx}).values) if trop_1d is not None else float("nan")
            roni = val - trop_val if not np.isnan(trop_val) else float("nan")

            if fm_coords is not None:
                forecast_month_num = int(fm_coords.values[t_idx])
            else:
                forecast_month_num = t_idx + 1

            lead = forecast_month_num - 1
            target_dt = init_dt + pd.DateOffset(months=lead)
            records.append({
                "source": "C3S",
                "model": model_name,
                "model_type": "dynamical",
                "init_date": init_date,
                "target_month": target_dt.strftime("%Y-%m"),
                "lead_months": lead,
                "nino34_anom": round(val, 4),
                "tropical_mean_anom": round(trop_val, 4) if not np.isnan(trop_val) else float("nan"),
                "roni_anom": round(roni, 4) if not np.isnan(roni) else float("nan"),
                "member_id": member_id,
                "temporal_resolution": "monthly",
                "anomaly_base_period": C3S_ANOMALY_BASE_PERIOD,
            })

    trop_member_dim = None
    if tropical is not None:
        for dim in tropical.dims:
            if "number" in dim.lower() or "member" in dim.lower():
                trop_member_dim = dim
                break

    if member_dim is not None:
        for m_idx in range(n_members):
            m_num = int(member_coords[m_idx])
            member_id = f"ens_{m_num:03d}"
            n34_m = nino34.isel({member_dim: m_idx})
            trop_m = tropical.isel({trop_member_dim: m_idx}) if (tropical is not None and trop_member_dim) else tropical
            _add_records(n34_m, trop_m, member_id)

        ens_mean_n34 = nino34.mean(dim=member_dim)
        ens_mean_trop = tropical.mean(dim=trop_member_dim) if (tropical is not None and trop_member_dim) else tropical
        _add_records(ens_mean_n34, ens_mean_trop, "mean")
    else:
        _add_records(nino34, tropical, "mean")

    return records


def _fetch_c3s_month(
    year: int,
    month: int,
    include_members: bool,
    only_models: set[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Fetch C3S data for a specific init month.

    Args:
        only_models: If provided, only fetch these model names.

    Returns:
        (records, succeeded_models) — the fetched records and the set of
        model names that returned data.
    """
    init_date = f"{year}-{month:02d}-01"
    c3s_raw = RAW_DIR / "c3s" / f"{year}{month:02d}"
    c3s_raw.mkdir(parents=True, exist_ok=True)

    product_type = "monthly_mean" if include_members else "ensemble_mean"
    all_records = []
    succeeded_models: set[str] = set()

    for model_name, model_info in C3S_MODELS.items():
        if only_models is not None and model_name not in only_models:
            continue

        centre = model_info["originating_centre"]
        system = _resolve_system(centre, year, month, model_info["system"])
        logger.info("Fetching C3S %s (system %s, %s)...",
                     model_name, system, product_type)

        suffix = "members" if include_members else "ensmean"
        nc_path = c3s_raw / f"{centre}_{year}{month:02d}_{suffix}.nc"

        try:
            _retrieve_direct(
                {
                    "originating_centre": centre,
                    "system": system,
                    "variable": C3S_VARIABLE,
                    "product_type": product_type,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "leadtime_month": _resolve_leads(
                        centre, year, month, model_info.get("max_lead_months", 6)
                    ),
                    "area": [20, -180, -20, 180],  # N, W, S, E — full tropical band for rONI
                    "data_format": "netcdf",
                },
                str(nc_path),
                C3S_RETRIEVE_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning("Failed to fetch C3S %s: %s", model_name, e)
            continue

        try:
            ds = xr.open_dataset(nc_path)
            nino34 = _compute_nino34_area_mean(ds)
            try:
                tropical = _compute_tropical_mean(ds)
            except Exception as e:
                logger.warning("C3S %s: tropical-mean failed (%s); rONI will be NaN", model_name, e)
                tropical = None
            logger.info("C3S %s: Nino3.4 dims=%s, shape=%s",
                        model_name, nino34.dims, nino34.shape)

            model_records = _extract_records(nino34, model_name, init_date, tropical=tropical)
            all_records.extend(model_records)
            succeeded_models.add(model_name)

            n_members = 0
            for dim in nino34.dims:
                if "number" in dim.lower():
                    n_members = nino34.sizes[dim]
            n_leads = max(1, nino34.sizes.get("forecastMonth", 1))

            if n_members > 0:
                logger.info("C3S %s: %d members x %d lead times + mean",
                            model_name, n_members, n_leads)
            else:
                logger.info("C3S %s: %d lead times (ensemble mean)",
                            model_name, n_leads)

            ds.close()

        except Exception as e:
            logger.warning("Error processing C3S %s: %s", model_name, e)
            continue

    return all_records, succeeded_models


def fetch_c3s(
    year: int | None = None,
    month: int | None = None,
    include_members: bool = True,
) -> pd.DataFrame:
    """Fetch C3S seasonal forecasts for all configured models.

    Tries the current month first for each model; for any models whose
    data is not yet available, falls back to the previous month.

    Returns DataFrame in standard forecast schema.
    """
    if not CDS_API_KEY:
        raise ValueError(
            "CDS API key not configured. Set CDS_API_KEY environment variable "
            "or edit enso_forecast/config.py. Get a key at "
            "https://cds.climate.copernicus.eu/"
        )

    if year is None:
        year = date.today().year
    if month is None:
        month = date.today().month

    # Try current month for all models
    all_records, succeeded = _fetch_c3s_month(year, month, include_members)

    # Fall back to previous month for any models that didn't return data
    missing = set(C3S_MODELS.keys()) - succeeded
    if missing:
        prev = pd.Timestamp(f"{year}-{month:02d}-01") - pd.DateOffset(months=1)
        logger.info(
            "C3S: %d models missing for %d-%02d (%s), falling back to %d-%02d",
            len(missing), year, month, ", ".join(sorted(missing)),
            prev.year, prev.month,
        )
        prev_records, prev_succeeded = _fetch_c3s_month(
            prev.year, prev.month, include_members, only_models=missing,
        )
        all_records.extend(prev_records)
        succeeded |= prev_succeeded

        still_missing = missing - prev_succeeded
        if still_missing:
            logger.warning("C3S: no data for models: %s", ", ".join(sorted(still_missing)))

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        n_models = df["model"].nunique()
        n_members = df[df["member_id"] != "mean"]["member_id"].nunique()
        logger.info("C3S: %d records from %d models (%d individual members)",
                     len(df), n_models, n_members)
    else:
        logger.info("C3S: 0 records")
    return df


def save_c3s(force: bool = False) -> pd.DataFrame:
    """Fetch and save C3S data.

    Protects against regressions: a new fetch is only saved if it has at
    least as many current-month-init models as the previous file.  This
    prevents transient CDS API failures from overwriting a good file.
    """
    today_str = date.today().isoformat()
    out_dir = FORECASTS_DIR / "C3S"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today_str}.csv"

    if not force and out_path.exists():
        logger.info("C3S data for %s already exists, skipping", today_str)
        return pd.read_csv(out_path)

    df = fetch_c3s(include_members=True)
    if df.empty:
        logger.warning("C3S fetch returned no data, keeping existing file")
        return df

    # Guard: don't regress — compare with the best existing file
    existing_csvs = sorted(out_dir.glob("*.csv"))
    if existing_csvs:
        prev = pd.read_csv(existing_csvs[-1])
        if "init_date" in prev.columns and "init_date" in df.columns:
            current_init = f"{date.today().strftime('%Y-%m')}-01"
            prev_current = prev[prev["init_date"] >= current_init]["model"].nunique()
            new_current = df[df["init_date"] >= current_init]["model"].nunique()
            prev_models = prev["model"].nunique()
            new_models = df["model"].nunique()
            if new_current < prev_current or new_models < prev_models:
                logger.warning(
                    "C3S fetch regressed: %d/%d models on current init (was %d/%d) "
                    "— keeping previous file %s",
                    new_current, new_models, prev_current, prev_models,
                    existing_csvs[-1].name,
                )
                return prev

    df.to_csv(out_path, index=False)
    logger.info("Saved C3S data to %s (%d models)", out_path, df["model"].nunique())
    return df
