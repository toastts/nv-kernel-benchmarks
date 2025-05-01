#!/usr/bin/env bash
set -euo pipefail

OUTDIR=profiler
mkdir -p "$OUTDIR"

# report metrics list
METRICS=\
"sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on,\
sm__sass_thread_inst_executed_op_fadd.sum,\
sm__sass_thread_inst_executed_op_fmul.sum,\
sm__sass_thread_inst_executed_op_ffma.sum,\
sm__throughput.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_bytes.sum,\
sm__warps_launched.sum"

# for k in {0..5}; do
#   echo ">>> profiling kernel $k"
#   /opt/cuda/nsight_compute/ncu \
#     --metrics "$METRICS" \
#     --csv \
#     --log-file "${OUTDIR}/profile_${k}.csv" \
#     ./build/gemm "$k"
# done

for k in {0..4}; do
  echo ">>> profiling kernel $k"
  /opt/cuda/nsight_compute/ncu \
    --metrics "$METRICS" \
    --csv \
    --log-file "${OUTDIR}/profile_${k}.csv" \
    ./build/gemm "$k"
done

echo "done – all reports are in $OUTDIR/"
