#!/bin/sh
# Run the whole refresh for one initialization: ./run_refresh.sh 202610  (raw files must already be fetched;
# see README.md). Each step is the same command the README lists, so any one can be rerun on its own.
set -e
export ELNINO_INIT=${1:?usage: run_refresh.sh YYYYMM}
cd "$(dirname "$0")"
echo "== process.py"; (cd model_patterns && python3.13 process.py 2>&1 | grep -v Warning | tail -3)
echo "== regions.py"; (cd model_patterns && python3.13 regions.py 2>&1 | grep -v Warning | tail -1)
echo "== model_counts.py"; (cd hit_rates && python3.13 model_counts.py 2>&1 | grep -v Warning | tail -1)
echo "== window_tests.py"; (cd model_patterns && python3.13 window_tests.py 2>&1 | grep -v Warning | grep -i "step 1\|changed\|Traceback\|Error\|wrote" | head -8)
echo "== export_data.py"; (cd export && python3.13 export_data.py 2>&1 | grep -v Warning | tail -5)
echo "== export_dashboard.py"; (cd export && python3.13 export_dashboard.py 2>&1 | grep -v Warning)
echo "== DONE"
