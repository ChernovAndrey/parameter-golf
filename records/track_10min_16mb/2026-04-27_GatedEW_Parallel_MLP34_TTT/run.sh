#!/bin/bash
# Launcher for "Gated_EW on parallel layers + MLP 3.4x targeted shrink + Legal TTT".
# 2x H100, 40 min training + ~25 min eval (sliding + TTT) ≈ 65 min total.
#
# Architecture (vs PR #1493 SOTA, 1.0810 BPB w/TTT, 1.0827 sliding-only):
#   Layers 0-6 (sequential residual)
#       NO gate, MLP 4.0x (h=2048)  — unchanged from SOTA
#       Includes the looped layers 3, 4, 5 (visited 3x per pass) at full capacity.
#   Layers 7-10 (parallel residual)
#       + Qwen G1 elementwise gate: y = sigmoid(W_g x) * Attn(x), W_g [512->512]
#         4 gates x 262K = 1.05M params (~+0.51 MB after int6 GPTQ + Brotli-11)
#       - MLP_MULT 4.0 -> 3.4 (h=2048 -> 1740)
#         Saves 4 x 2 x 512 x 308 = 1.26M params (~-0.62 MB)
#       Per-token MLP volume in this zone: 4 x 1740 = 6960
#       (vs FatBlock's validated 6144, ~13% wider).
#   + Legal Score-First TTT (Issue #1017 Track B, same as PR #1493 SOTA)
#       lr=0.005, momentum=0.9, epochs=3, chunk_tokens=32768
#       Score-before-update per chunk; eval-time only (zero artifact cost).
#       Expected: -0.002 BPB beyond sliding-only.
#   = Net artifact: ~15.96 MB (predicted, ~40 KB headroom — SOTA-level tightness)
#     Fallback: drop PARALLEL_MLP_MULT to 3.375 (~70 KB headroom).
#
# Predecessor result (no TTT, MLP 3.25x): sliding val_bpb 1.08226 at seed 42,
# artifact 15.80 MB. See doc/experiment_gated_ew_parallel_results.md.
#
# Variants:
#   smoke                            — 30 s sanity (TTT off, no sliding eval)
#   gated_ew_parallel_ttt            — main: gates + MLP 3.4x + TTT  [default]
#   gated_ew_parallel_ttt_safe       — fallback: PARALLEL_MLP_MULT=3.375
#   gated_ew_parallel_no_ttt         — same arch, TTT off (isolates TTT contribution)
#
# Examples:
#   ./run.sh                              # main run, seed 42
#   ./run.sh gated_ew_parallel_ttt 314
#   ./run.sh smoke
#   ./run.sh gated_ew_parallel_ttt_safe   # if main lands over 16 MB
#
# Logs land in logs/<variant>_s<seed>.log.

set -e
set -o pipefail

# Defensively unset any inherited vars that could leak into the run.
unset GATED_ATTN_ENABLED GATED_ATTN_MODE MLP_MULT PARALLEL_MLP_MULT \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED TTT_LR TTT_EPOCHS TTT_MOMENTUM TTT_CHUNK_TOKENS \
      SLIDING_WINDOW_ENABLED SEED DATA_DIR \
      ITERATIONS VAL_LOSS_EVERY COMPRESSOR \
      MUON_WD MATRIX_LR MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      PARALLEL_RESIDUAL_START

VARIANT=${1:-gated_ew_parallel_ttt}
SEED=${2:-42}

mkdir -p logs

# Common settings (per-line export so they propagate to torchrun children).
export DATA_DIR=../../../data
export SEED=$SEED

case "$VARIANT" in
    smoke)
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.4
        export TTT_ENABLED=0
        export MAX_WALLCLOCK_SECONDS=30
        export TRAIN_BATCH_TOKENS=65536
        export WARMUP_STEPS=1
        export SLIDING_WINDOW_ENABLED=0
        ;;
    gated_ew_parallel_ttt)
        # Main run: 4 elementwise gates on layers 7-10 + MLP 3.4x there + Legal TTT.
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.4
        export TTT_ENABLED=1
        export TTT_LR=0.005
        export TTT_EPOCHS=3
        export TTT_MOMENTUM=0.9
        export TTT_CHUNK_TOKENS=32768
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    gated_ew_parallel_ttt_safe)
        # Fallback if main lands over 16 MB. Drops PARALLEL_MLP_MULT 3.4 -> 3.375
        # for ~70 KB headroom (vs ~40 KB main).
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.375
        export TTT_ENABLED=1
        export TTT_LR=0.005
        export TTT_EPOCHS=3
        export TTT_MOMENTUM=0.9
        export TTT_CHUNK_TOKENS=32768
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    gated_ew_parallel_no_ttt)
        # Same architecture as main, but TTT off. Isolates TTT's contribution
        # vs the prior MLP3.25 result (which was also TTT-off).
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=4.0
        export PARALLEL_MLP_MULT=3.4
        export TTT_ENABLED=0
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke gated_ew_parallel_ttt gated_ew_parallel_ttt_safe gated_ew_parallel_no_ttt" >&2
        exit 1
        ;;
esac

LOGFILE="logs/${VARIANT}_s${SEED}.log"
echo "=============================================="
echo "  Variant: $VARIANT   |   Seed: $SEED"
echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED  MODE=${GATED_ATTN_MODE:-n/a}"
echo "  MLP_MULT=$MLP_MULT  PARALLEL_MLP_MULT=$PARALLEL_MLP_MULT"
echo "  TTT_ENABLED=$TTT_ENABLED  ${TTT_ENABLED:+(lr=${TTT_LR:-} epochs=${TTT_EPOCHS:-} chunk=${TTT_CHUNK_TOKENS:-})}"
echo "  Log: $LOGFILE"
echo "=============================================="

torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOGFILE"

echo "=============================================="
echo "  DONE: $VARIANT seed=$SEED"
SLIDING_LINE=$(grep 'quantized_sliding_window' "$LOGFILE" | tail -1)
TTT_LINE=$(grep 'quantized_ttt' "$LOGFILE" | tail -1)
QUANT_LINE=$(grep '^quantized val_loss' "$LOGFILE" | tail -1)
ARTIFACT_LINE=$(grep 'Total submission size' "$LOGFILE" | tail -1)
echo "  Sliding:   ${SLIDING_LINE:-not found (smoke or failed)}"
echo "  TTT:       ${TTT_LINE:-not found (TTT off?)}"
echo "  Quantized: ${QUANT_LINE:-not found}"
echo "  Artifact:  ${ARTIFACT_LINE:-not found}"
echo "=============================================="
