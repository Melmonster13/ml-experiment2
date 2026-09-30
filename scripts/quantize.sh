#!/usr/bin/env bash
# Build each quantized base model once. Pass settings to build a subset:
#   scripts/quantize.sh q4_g64 q8_g64
set -euo pipefail
cd "$(dirname "$0")/.."

HF_MODEL=bigcode/starcoder2-3b
SETTINGS=("${@:-q4_g32 q4_g64 q4_g128 q8_g64}")

for s in ${SETTINGS[@]}; do
  bits=${s#q}; bits=${bits%_g*}
  group=${s#*_g}
  out=models/$s
  if [ -d "$out" ]; then
    echo "skip $out (exists)"
    continue
  fi
  echo "== $out: ${bits}-bit, group size $group"
  python -m mlx_lm convert --hf-path "$HF_MODEL" --mlx-path "$out" \
    -q --q-bits "$bits" --q-group-size "$group" --q-mode affine
done
