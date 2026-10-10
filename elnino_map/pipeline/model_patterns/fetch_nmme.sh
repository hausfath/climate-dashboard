#!/bin/sh
# NMME realtime ensemble-mean anomalies (per model + the NMME mean), the CPC calibrated tercile probabilities and
# the per-model SST anomalies for one initialization. usage: ./fetch_nmme.sh 202610 [2026100800]
# CPC posts the month's files in a folder named by the issue date (usually the 8th): list
# ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/ to find it.
set -e
YM=${1:?usage: fetch_nmme.sh YYYYMM [issue folder, default YYYYMM0800]}
ISSUE=${2:-${YM}0800}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE="ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/$ISSUE"
mkdir -p "$HERE/raw/nmme_$YM" "$HERE/raw/nmme_${YM}_sst" "$HERE/raw/nmme_prob"
for m in CanESM5 CFSv2 GEM5.2_NEMO NASA_GEOS5v2 NCAR_CCSM4 NCAR_CESM1 NMME; do
  for v in prate tmp2m; do
    f="$m.$v.$YM.ENSMEAN.anom.nc"
    [ -s "$HERE/raw/nmme_$YM/$f" ] || curl -sf --max-time 300 -o "$HERE/raw/nmme_$YM/$f" "$BASE/$f"
  done
  [ "$m" = NMME ] || [ -s "$HERE/raw/nmme_${YM}_sst/$m.tmpsfc.anom.nc" ] || \
    curl -sf --max-time 300 -o "$HERE/raw/nmme_${YM}_sst/$m.tmpsfc.anom.nc" "$BASE/$m.tmpsfc.$YM.ENSMEAN.anom.nc"
done
f="prate.$YM.prob.adj.seas.nc"
[ -s "$HERE/raw/nmme_prob/$f" ] || curl -sf --max-time 300 -o "$HERE/raw/nmme_prob/$f" "ftp://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/$f"
ls -la "$HERE/raw/nmme_$YM" | wc -l
