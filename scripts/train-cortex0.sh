#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STEPS="${CORTEX_TRAIN_STEPS:-20000}"
echo "Cortex-0 developmental training — $STEPS steps per cortex"
for cortex in visual perceptual memory associative language; do
  echo
  echo "=== $cortex ==="
  CORTEX_TRAIN_STEPS="$STEPS" cargo run -p cortex-node --bin cortex-train -- "$cortex"
done
echo
echo "Cortex-0 checkpoints:"
ls -lh weights/cortex-0
echo "Runtime will load these automatically on the next ./scripts/run-seed0.sh"
