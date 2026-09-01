#!/usr/bin/env bash
# Shared preamble for the batch scripts. Not run directly.
#
# Every batch is: a list of `run <run-id> <flags...>` lines, executed in order, each one
# logging to results/v1.1/<run-id>/run.log and skipping itself if that run already produced
# a result.json. So a batch that dies halfway -- or that you Ctrl-C -- resumes by being
# started again, and finished runs are never redone.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SURVEY="$(dirname "$HERE")"
REPO="$(dirname "$(dirname "$SURVEY")")"
RESULTS="$SURVEY/results/v1.1"

run () {
  local id="$1"; shift
  local dir="$RESULTS/$id"
  if [[ -f "$dir/result.json" && "${FORCE:-0}" != "1" ]]; then
    echo "== SKIP $id (result.json exists; FORCE=1 to redo)"
    return 0
  fi
  mkdir -p "$dir"
  echo "== RUN  $id  $*"
  echo "   log: $dir/run.log"
  local t0=$SECONDS
  # 10_train.py writes "$dir/run.log" itself now, so no tee here -- teeing as well would
  # double every line in that file. This batch's own stdout is wherever you redirected it.
  ( cd "$REPO" && poetry run python "$SURVEY/scripts/10_train.py" --run-id "$id" "$@" ) 2>&1
  local rc=$?
  echo "== $( [[ $rc -eq 0 ]] && echo DONE || echo FAILED\(rc=$rc\) ) $id in $(( (SECONDS-t0)/60 )) min"
  echo
}

report () {
  ( cd "$SURVEY" && poetry run python scripts/90_report.py --write )
}
