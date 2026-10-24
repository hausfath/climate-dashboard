"""Compare the 23 Oct 2026 run's lead-0 steps with the 22 Oct run at the same valid hours (probe CSVs).
"""
import pandas as pd, numpy as np
S = "forecast_skill/tools/probe_out"  # folder holding the three probe CSVs
def load(f):
    d=pd.read_csv(f); d['time']=pd.to_datetime(d['time']); d['w']=np.cos(np.deg2rad(d.lat)); d['land']=d.elev>0; return d
old=load(S+"/ec46_steps_2026-10-23.csv")          # Oct 22 run (Oct 23 = lead 24-42 h)
new=load(S+"/ec46_steps_2026-10-24_pastday.csv")  # past_days=1 at ~01 UTC Oct 24 (Oct 23 = Oct 23 run lead 0-18 h?)
r23=load(S+"/ec46_steps_2026-10-23_run23.csv")    # Oct 23 run, axis from Oct 24
def gm(d):
    return d.groupby('time').apply(lambda g: np.average(g['mean'],weights=g.w), include_groups=False)
a=gm(new); b=gm(r23); common=a.index.intersection(b.index)
print("past-day probe vs run-23 probe on shared steps: max |diff| = %.4f over %d steps"%((a[common]-b[common]).abs().max(), len(common)))
steps=[t for t in a.index if t.date()==pd.Timestamp('2026-10-23').date()]
o=gm(old)
print("\nOct 23 valid times: Oct-23-run (lead 0..18h) minus Oct-22-run (lead 24..42h); cos-weighted global, land, ocean; member sd")
for t in steps:
    n_=new[new.time==t]; o_=old[old.time==t]
    dl=np.average(n_[n_.land]['mean'],weights=n_[n_.land].w)-np.average(o_[o_.land]['mean'],weights=o_[o_.land].w)
    do=np.average(n_[~n_.land]['mean'],weights=n_[~n_.land].w)-np.average(o_[~o_.land]['mean'],weights=o_[~o_.land].w)
    print(f"  {t:%m-%d %Hz}  global {a[t]-o[t]:+.3f}   land {dl:+.3f}  ocean {do:+.3f}   member sd {np.average(n_['sd'],weights=n_.w):.3f} vs {np.average(o_['sd'],weights=o_.w):.3f}")
print("  day mean  global %+.3f"%(a[steps].mean()-o[steps].mean()))
steps24=[t for t in a.index if t.date()==pd.Timestamp('2026-10-24').date() and t in o.index]
print("\nOct 24 valid times (Oct-23-run lead 24-42h minus Oct-22-run lead 48-66h): day mean global %+.3f"%(a[steps24].mean()-o[steps24].mean()))
new23=new[new.time.isin(steps)]; old23=old[old.time.isin(steps)]
m=new23.groupby('lat')['mean'].mean()-old23.groupby('lat')['mean'].mean()
print("\nlead-0 day minus lead-1 day by latitude band (°C):"); print(m.round(2).to_string())
