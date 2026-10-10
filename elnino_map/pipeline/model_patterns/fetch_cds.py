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
TIMEOUT = 3600
YM = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "202609"   # init YYYYMM
YEAR, MONTH = YM[:4], YM[4:]

C3S = {"ecmwf": "51", "ukmo": "610", "meteo_france": "9", "dwd": "22", "cmcc": "4",
       "jma": "4", "bom": "2", "ncep": "2", "eccc": "5"}
C3S_VARS = {"t2m": "2m_temperature_anomaly",
            "tprate": "total_precipitation_anomalous_rate_of_accumulation"}


def retrieve(dataset, request, target):
    target = Path(target)
    if target.exists() and target.stat().st_size > 0:
        print("have", target.name, flush=True)
        return
    base = f"{URL}/retrieve/v1"
    r = requests.post(f"{base}/processes/{dataset}/execution", headers=H,
                      json={"inputs": request}, timeout=60)
    r.raise_for_status()
    job = r.json()["jobID"]
    deadline = time.monotonic() + TIMEOUT
    status = r.json().get("status", "accepted")
    while status in ("accepted", "running"):
        if time.monotonic() > deadline:
            requests.delete(f"{base}/jobs/{job}", headers=H, timeout=30)
            raise TimeoutError(f"{target.name}: job {job} cancelled after {TIMEOUT}s")
        time.sleep(15)
        status = requests.get(f"{base}/jobs/{job}", headers=H, timeout=30).json().get("status")
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
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(f, *a): (f.__name__, a) for f, a in jobs}
        for fu, lab in futs.items():
            try:
                fu.result()
            except Exception as e:  # report and continue
                errors.append(f"{lab}: {e}")
                print("ERROR", lab, e, flush=True)
    print("done;", len(errors), "errors")
    sys.exit(1 if errors else 0)
