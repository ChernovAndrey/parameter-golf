#!/bin/bash
# Launcher for "Gated_EW on parallel layers only + MLP 3.25x targeted shrink".
# 2x H100, 40 min training (eval ~9 min more).
#
# Architecture (vs PR #1493 SOTA, 1.0810 BPB):
#   Layers 0-6 (sequential residual)
#       NO gate, MLP 4.0x (h=2048)  — unchanged from SOTA
#       Includes the looped layers 3, 4, 5 (visited 3x per pass) at full capacity.
#   Layers 7-10 (parallel residual)
#       + Qwen G1 elementwise gate: y = sigmoid(W_g x) * Attn(x), W_g [512->512]
#         4 gates x 262K = 1.05M params (~+0.50 MB after int6 GPTQ + Brotli)
#       - MLP_MULT 4.0 -> 3.25 (h=2048 -> 1664)
#         Saves 4 x 2 x 512 x 384 = 1.57M params (~-0.77 MB)
#         h=1664 = 13x128, so all GPTQ blocks divide cleanly (no partial blocks).
#       Per-token MLP volume in this zone: 4 x 1664 = 6656, vs FatBlock's
#       validated regime of 1 x 6144 (~8%).
#   = Net artifact: ~15.79 MB (predicted, ~210 KB headroom)
#     Fallback: drop PARALLEL_MLP_MULT to 3.0 (~490 KB headroom) if over budget.
#
# Variants:
#   smoke                     — 30 s sanity (tiny batch, no sliding eval)
#   gated_ew_parallel         — main run [default]: gate + MLP 3.1x on layers 7-10
#   baseline_no_gate          — control: no gate, same MLP shrink (isolates gate effect)
#   gated_ew_parallel_safe    — fallback: PARALLEL_MLP_MULT=3.0 (more headroom)
#
# Examples:
#   ./run.sh                       # gated_ew_parallel, seed 42
#   ./run.sh gated_ew_parallel 314
#   ./run.sh smoke
#   ./run.sh baseline_no_gate
#   ./run.sh gated_ew_parallel_safe
#
# Logs land in logs/<variant>_s<seed>.log.

set -e
set -o pipefail

# Defensively unset any inherited vars that could leak into the run.
unset GATED_ATTN_ENABLED GATED_ATTN_MODE MLP_MULT PARALLEL_MLP_MULT \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED SLIDING_WINDOW_ENABLED SEED DATA_DIR \
      ITERATIONS VAL_LOSS_EVERY COMPRESSOR \
      MUON_WD MATRIX_LR MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      PARALLEL_RESIDUAL_START

VARIANT=${1:-gated_ew_parallel}
SEED=${2:-42}

mkdir -p logs

# Common settings (per-line export so they propagate to torchrun children).
export DATA_DIR=../../../data
export TTT_ENABLED=0
export SEED=$SEED

case "$VARIANT" in
    smoke)
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.25
        export MAX_WALLCLOCK_SECONDS=30
        export TRAIN_BATCH_TOKENS=65536
        export WARMUP_STEPS=1
        export SLIDING_WINDOW_ENABLED=0
        ;;
    gated_ew_parallel)
        # Main run: elementwise gate + MLP 3.1x on parallel-residual layers (7-10).
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.25
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    baseline_no_gate)
        # Control: same targeted MLP shrink, no gate. Isolates the gate's contribution
        # from the MLP-shrink cost. If gated_ew_parallel - baseline_no_gate ~= 0, the
        # gate is not pulling its weight; if Δ > 0.001 BPB, the gate adds real value.
        export GATED_ATTN_ENABLED=0
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.25
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    gated_ew_parallel_safe)
        # Fallback if the 3.1x main run lands over the 16 MB cap.
        # Drops parallel-layer MLP to 3.0x (h=1536) for ~490 KB headroom.
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.0
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke gated_ew_parallel baseline_no_gate gated_ew_parallel_safe" >&2
        exit 1
        ;;
esac

LOGFILE="logs/${VARIANT}_s${SEED}.log"
echo "=============================================="
echo "  Variant: $VARIANT   |   Seed: $SEED"
echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED  MODE=${GATED_ATTN_MODE:-n/a}"
echo "  MLP_MULT=$MLP_MULT  PARALLEL_MLP_MULT=$PARALLEL_MLP_MULT"
echo "  Log: $LOGFILE"
echo "=============================================="

torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOGFILE"

echo "=============================================="
echo "  DONE: $VARIANT seed=$SEED"
SLIDING_LINE=$(grep 'quantized_sliding_window' "$LOGFILE" | tail -1)
QUANT_LINE=$(grep '^quantized val_loss' "$LOGFILE" | tail -1)
ARTIFACT_LINE=$(grep 'Total submission size' "$LOGFILE" | tail -1)
echo "  Sliding:   ${SLIDING_LINE:-not found (smoke test? or failed)}"
echo "  Quantized: ${QUANT_LINE:-not found}"
echo "  Artifact:  ${ARTIFACT_LINE:-not found}"
echo "=============================================="
