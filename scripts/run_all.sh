#!/usr/bin/env bash
# Run configs in order, each in its own process. Close heavy apps first:
# unified memory is shared with the rest of the system.
#   scripts/run_all.sh configs/r2_*.yaml configs/r3_*.yaml
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 0 ] && set -- configs/*.yaml
for cfg in "$@"; do
  python scripts/run.py "$cfg"
done
