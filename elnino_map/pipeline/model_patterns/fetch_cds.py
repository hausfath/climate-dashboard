#!/usr/bin/env python3.13
"""Fetch C3S seasonal anomalies (init YYYYMM from argv, default Sep 2026) and ERA5 monthly T2m from the CDS.

C3S: seasonal-postprocessed-single-levels, ensemble_mean, 2m T + precip anomalies,
leadtime months 1-6 (Sep 2026 - Feb 2027), global 1 deg. Anomalies are relative to
each system's 1993-2016 hindcast climatology.
Centres: the 7 not already represented in NMME (NCEP = CFSv2 and ECCC = CanSIPS
models are taken from NMME instead; they are fetched too, for a cross-source
consistency check only).

ERA5: reanalysis-era5-single-levels-monthly-means, 2m T, 1979-2025, 1 deg grid,
for the trend adjustment and interannual SD of temperature.

Job handling follows the Climate Dashboard fetcher: poll with a deadline and
cancel server-side on timeout so abandoned jobs don't clog the CDS queue.
"""
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

txt = open(os.path.expanduser("~/.cdsapirc")).read()
KEY = re.search(r"key:\s*(\S+)", txt).group(1)
URL = re.search(r"url:\s*(\S+)", txt).group(1)
H = {"PRIVATE-TOKEN": KEY}
RAW = Path(__file__).parent / "raw"
TIMEOUT = int(os.environ.get("CDS_TIMEOUT", 3600))   # seconds a job may wait in the CDS queue
YM = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "202609"   # init YYYYMM
YEAR, MONTH = YM[:4], YM[4:]

C3S = {"ecmwf": "51", "ukmo": "610", "meteo_france": "9", "dwd": "22", "cmcc": "4",
       "jma": "4", "bom": "2", "ncep": "2", "eccc": "5"}
C3S_VARS = {"t2m": "2m_temperature_anomaly",
            "tprate": "total_precipitation_anomalous_rate_of_accumulation"}


def _existing_job(dataset, request):
    """(jobID, status) of the newest job on the account with exactly this request that is
    queued, running or successful, else (None, None)."""
    base = f"{URL}/retrieve/v1"
    try:
        jobs = requests.get(f"{base}/jobs?limit=100&sortby=-created", headers=H, timeout=90).json().get("jobs", [])
        for j in jobs:
            if j.get("processID") != dataset or j.get("status") not in ("accepted", "running", "successful"):
                continue
            d = requests.get(f"{base}/jobs/{j['jobID']}?request=true", headers=H, timeout=90).json()
            if d.get("metadata", {}).get("request", {}).get("ids") == request:
                return j["jobID"], j["status"]
    except requests.RequestException:
        pass
    return None, None


def retrieve(dataset, request, target):
    target = Path(target)
    if target.exists() and target.stat().st_size > 0:
        print("have", target.name, flush=True)
        return
    base = f"{URL}/retrieve/v1"
    deadline = time.monotonic() + TIMEOUT
    # Adopt an identical job already on the account (queued, running or done) before submitting:
    # when the CDS is congested a POST can time out client-side after the server has created the
    # job, and those orphans count against the small per-dataset queue limit (17 Nov 2026).
    job, status = _existing_job(dataset, request)
    if job:
        print("adopt", target.name, job[:8], status, flush=True)
    while job is None:
        try:
            r = requests.post(f"{base}/processes/{dataset}/execution", headers=H,
                              json={"inputs": request}, timeout=120)
            r.raise_for_status()
            job, status = r.json()["jobID"], r.json().get("status", "accepted")
        except requests.RequestException as e:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{target.name}: could not submit within {TIMEOUT}s ({e})")
            time.sleep(120)
            job, status = _existing_job(dataset, request)     # the timed-out POST may have landed
    while status in ("accepted", "running", "rejected"):
        if status == "rejected":
            try:
                body = requests.get(f"{base}/jobs/{job}/results", headers=H, timeout=60).text
            except requests.RequestException:
                body = "Maximum number queued (results unreadable; assume queue full)"
            if "Maximum number queued" not in body or time.monotonic() > deadline:
                raise RuntimeError(f"{target.name}: job {job} rejected: {body[:300]}")
            # queue full: resubmit later, slowly (ECMWF's Oct 2026 protection measures ask for sequential
            # requests and may suspend accounts that resubmit too often)
            time.sleep(int(os.environ.get("CDS_REQUEUE_WAIT", 600)))
            try:
                r = requests.post(f"{base}/processes/{dataset}/execution", headers=H,
                                  json={"inputs": request}, timeout=120)
                r.raise_for_status()
                job, status = r.json()["jobID"], r.json().get("status", "accepted")
            except requests.RequestException:
                job2, status2 = _existing_job(dataset, request)   # the timed-out POST may have landed
                if job2:
                    job, status = job2, status2
            continue
        if time.monotonic() > deadline:
            requests.delete(f"{base}/jobs/{job}", headers=H, timeout=30)
            raise TimeoutError(f"{target.name}: job {job} cancelled after {TIMEOUT}s")
        time.sleep(15)
        try:
            status = requests.get(f"{base}/jobs/{job}", headers=H, timeout=60).json().get("status")
        except requests.RequestException:
            pass                                             # transient; keep polling
    if status != "successful":
        body = requests.get(f"{base}/jobs/{job}/results", headers=H, timeout=30).text[:400]
        raise RuntimeError(f"{target.name}: job {job} {status}: {body}")
    res = requests.get(f"{base}/jobs/{job}/results", headers=H, timeout=60).json()
    href = res["asset"]["value"]["href"]
    tmp = target.with_suffix(".part")
    with requests.get(href, stream=True, timeout=600) as dl:
        dl.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in dl.iter_content(1 << 16):
                f.write(chunk)
    tmp.rename(target)
    print("got", target.name, target.stat().st_size // 1024, "KB", flush=True)


def c3s_job(centre, short):
    out = RAW / f"c3s_{YM}" / f"{centre}_{short}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    retrieve("seasonal-postprocessed-single-levels", {
        "originating_centre": centre, "system": C3S[centre], "variable": [C3S_VARS[short]],
        "product_type": ["ensemble_mean"], "year": [YEAR], "month": [MONTH],
        "leadtime_month": ["1", "2", "3", "4", "5", "6"], "data_format": "netcdf",
    }, out)


def era5_job():
    out = RAW / "era5" / "era5_t2m_monthly_1979_2025_1deg.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"], "variable": ["2m_temperature"],
        "year": [str(y) for y in range(1979, 2026)],
        "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"],
        "grid": ["1.0", "1.0"], "data_format": "netcdf",
    }, out)


if __name__ == "__main__":
    jobs = [(c3s_job, (c, v)) for c in C3S for v in C3S_VARS] + [(era5_job, ())]
    errors = []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("CDS_WORKERS", 1))) as ex:   # CDS asks for sequential requests
        futs = {ex.submit(f, *a): (f.__name__, a) for f, a in jobs}
        for fu, lab in futs.items():
            try:
                fu.result()
            except Exception as e:  # report and continue
                errors.append(f"{lab}: {e}")
                print("ERROR", lab, e, flush=True)
    print("done;", len(errors), "errors")
    sys.exit(1 if errors else 0)
