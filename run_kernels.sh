#!/usr/bin/env bash
set -euo pipefail

# where to save per-kernel outputs
OUTDIR=results
mkdir -p "$OUTDIR"

# list of kernel IDs to run
for k in {0..5}; do
  echo ">>> running kernel $k"
  ./build/gemm "$k" 2>&1 | tee "${OUTDIR}/kernel_${k}.log"
done
