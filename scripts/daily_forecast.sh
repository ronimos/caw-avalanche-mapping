#!/usr/bin/env bash
# Daily 2026-27 forecast: pull the latest SNOWPACK output for the model
# stations, then rebuild and publish the forecast map.  Run from cron, e.g.
#
#   30 11 * * * /home/ops/post-processing/caw-avalanche-mapping/scripts/daily_forecast.sh
#
# Station set: nearest station to every WMP point (data/FOCUS_Ava.kml and
# data/WMP_Locations-2024.csv) and every station within 8 km of the Pamir
# Highway, all aspects, plus/minus the Add/Remove edits saved from the focus map editor.
set -euo pipefail

REPO=/home/ops/post-processing/caw-avalanche-mapping
UV=/home/ops/.local/bin/uv
SEASON=2026-2027
SOURCE=/mnt/HC_Volume_106760790/model/snowpack/runs/output.caw.2026
SELECTION=/var/www/html/nwp/caw/test/model_locations/model_locations.json
SITE_DIR=/var/www/html/nwp/caw/test/forecast
LOG=/home/ops/rtsys/log/caw_forecast.log

exec >>"$LOG" 2>&1
exec 9>/tmp/caw_daily_forecast.lock
if ! flock -n 9; then
    echo "$(date -u +%FT%TZ) previous run still active — skipping"
    exit 0
fi

trap 'echo "=== $(date -u +%FT%TZ) daily forecast FAILED (line $LINENO, exit $?)"' ERR
echo "=== $(date -u +%FT%TZ) daily forecast start"
cd "$REPO"
export PYTHONWARNINGS=ignore   # sklearn feature-name warnings flood the log

sync_args=(--season "$SEASON" --source "$SOURCE" --focus data/FOCUS_Ava.kml --focus data/WMP_Locations-2024.csv --highway)
if [[ -s "$SELECTION" ]]; then
    sync_args+=(--selection "$SELECTION" --prune)
fi
"$UV" run python scripts/sync_pro_files.py "${sync_args[@]}" | grep -v '^\s*$' | tail -8

"$UV" run python src/main.py --forecast-only --site-dir "$SITE_DIR" \
    | grep -E 'Forecast date|Model used|→ Map|→ Charts|WARNING|Error'

echo "=== $(date -u +%FT%TZ) daily forecast done"
