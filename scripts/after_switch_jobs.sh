#!/usr/bin/env bash
# Wait until no SWITCH generation process is running, then run explore_math.sh with the given args.
set -uo pipefail
while pgrep -f "scripts/explore_math.py" >/dev/null; do echo "[queue] $(date -u +%T) waiting"; sleep 30; done
exec bash scripts/explore_math.sh "$@"
