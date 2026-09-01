#!/usr/bin/env bash
# What is running, what is done, what is queued. No arguments, no guessing.
#
#   bash scripts/status.sh          one snapshot
#   bash scripts/status.sh -w       refresh every 30s until nothing is running
#
# Answers "which run is up right now and how far along is it" without you needing to know
# the run id -- it reads the live process, then that run's own log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURVEY="$(dirname "$HERE")"
RESULTS="$SURVEY/results/v1.1"

snapshot () {
  echo "=================================================================="
  date "+ %Y-%m-%d %H:%M:%S"
  echo "=================================================================="

  # ---- what is alive (there can be more than one)
  local n=0
  while IFS= read -r proc; do
    [[ -z "$proc" ]] && continue
    n=$((n+1))
    local pid=${proc%% *}
    local id
    id=$(sed -n 's/.*--run-id \([^ ]*\).*/\1/p' <<<"$proc")
    echo
    echo "RUNNING  ${id:-?}   (pid $pid)"
    local log="$RESULTS/$id/run.log"
    if [[ -f "$log" ]]; then
      local last best
      last=$(grep -E "^  epoch " "$log" | tail -1)
      best=$(grep -E "^  epoch " "$log" | sed -E 's/.*top1 ([0-9.]+).*/\1/' | sort -rn | head -1)
      echo "  latest : ${last:-(no epoch finished yet)}"
      [[ -n "$best" ]] && echo "  best   : top1 $best so far"
      echo "  log    : tail -f $log"
    else
      echo "  log    : none in $RESULTS/$id/ -- started before runs self-logged;"
      echo "           try  tail -f /tmp/$id.out"
    fi
  done < <(
    # real trainer processes only: a python running 10_train.py with a --run-id.
    # Deduped by run id, because `poetry run python ...` shows up alongside the python it
    # spawned, and a bare `pgrep -fl` also matches this script and any grep of it.
    ps -eo pid=,command= 2>/dev/null \
      | grep -E "[Pp]ython[^ ]* .*10_train\.py .*--run-id " \
      | grep -v "status.sh" \
      | awk '{ id=""; for (i=1;i<=NF;i++) if ($i=="--run-id") id=$(i+1);
               if (id != "" && !(id in seen)) { seen[id]=1; print } }'
  )
  if [[ $n -eq 0 ]]; then
    echo
    echo "RUNNING  nothing"
  fi

  # ---- a chained batch waiting its turn
  local waiter
  waiter=$(pgrep -fl "run_batch" 2>/dev/null | grep -v "pgrep" | head -1)
  [[ -n "$waiter" ]] && echo "  batch  : ${waiter#* } (queues the remaining runs)"

  # ---- everything that has reported
  echo
  echo "REPORTED"
  python3 - "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
res = Path(sys.argv[1])
rows = []
for d in sorted(res.iterdir()):
    f = d / "result.json"
    if not f.exists():
        continue
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    rows.append((r["metrics"]["top1"], d.name, r.get("arch", "?"), r.get("stage", "?")))
if not rows:
    print("  (none yet)")
for top1, name, arch, stage in sorted(rows, reverse=True):
    print(f"  {top1:.3f}  {name:<14} {arch:<9} stage {stage}")
PY

  # ---- started but unfinished
  echo
  echo "PARTIAL (has a checkpoint, no result.json -- resumes on next launch)"
  local any=0
  for d in "$RESULTS"/*/; do
    [[ -f "$d/result.json" ]] && continue
    [[ -f "$d/state.pt" ]] || continue
    local id n log
    id=$(basename "$d")
    log="$d/run.log"; [[ -f "$log" ]] || log="/tmp/$id.out"
    if [[ -f "$log" ]]; then
      n=$(grep -cE "^  epoch " "$log" 2>/dev/null || echo 0)
      echo "  $id  ($n epochs done, per $log)"
    else
      echo "  $id  (has state.pt; no log found)"
    fi
    any=1
  done
  [[ $any -eq 0 ]] && echo "  (none)"
  echo
}

if [[ "${1:-}" == "-w" ]]; then
  while true; do
    clear; snapshot
    pgrep -f "10_train.py" >/dev/null || pgrep -f "run_batch" >/dev/null || { echo "all done."; break; }
    sleep 30
  done
else
  snapshot
fi
