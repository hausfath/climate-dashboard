"""Diagnostic (2026-10-23/24): fetch every EC46 gridpoint at 6-hourly resolution for the first days
of the served run; argv[1] = output CSV. Add past_days=1 to the query to capture the run's own lead-0
day after 00 UTC. Used to show the lead-0 warm offset sits in the 00Z analysis step.
"""
import json, urllib.request, numpy as np, pandas as pd, time, sys
sys.path.insert(0,'.'); sys.path.insert(0,'ENSO')
from enso_forecast.fetchers.ec46 import _grid, LAT_STEP_DEFAULT, LON_STEP_DEFAULT, API, N_MEMBERS
pts=_grid(LAT_STEP_DEFAULT, LON_STEP_DEFAULT)
rows=[]
for i,(lat,lon) in enumerate(pts):
    url=(f"{API}?latitude={lat:.3f}&longitude={lon:.3f}&models=ecmwf_ec46&hourly=temperature_2m"
         f"&forecast_days=5&ensemble_members={N_MEMBERS}")
    for a in range(3):
        try:
            p=json.load(urllib.request.urlopen(url,timeout=30)); break
        except Exception as e:
            p=None; time.sleep(1)
    if p is None: print("fail",lat,lon); continue
    h=p['hourly']; t=h['time']
    mk=sorted(k for k in h if k.startswith('temperature_2m_member'))
    M=np.array([h[k] for k in mk],dtype=float)
    ctrl=np.array(h.get('temperature_2m',[np.nan]*len(t)),dtype=float)
    for j,ts in enumerate(t):
        rows.append(dict(lat=lat,lon=lon,elev=p.get('elevation'),time=ts,mean=np.nanmean(M[:,j]),sd=np.nanstd(M[:,j]),ctrl=ctrl[j],n=int((~np.isnan(M[:,j])).sum())))
    if i%50==0: print(i,flush=True)
    time.sleep(0.05)
pd.DataFrame(rows).to_csv(sys.argv[1],index=False); print("done",len(rows))
