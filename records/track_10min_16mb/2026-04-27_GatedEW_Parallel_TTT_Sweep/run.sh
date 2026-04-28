#!/bin/bash
# MLP-ratio sweep for the parallel-only-gates architecture, all with Legal TTT.
# 2x H100, 40 min training + ~25 min eval (sliding + TTT) = ~65 min per variant.
#
# Hypothesis under test: smaller per-token MLP volume in the parallel zone gives
# better quantization (lower entropy weights) without losing val_bpb capacity,
# and may match or beat MLP 3.25's sliding-only result of 1.08226.
#
# Per-token MLP volume in parallel zone (4 layers, h = round(mult * 512)):
#   MLP 3.0    -> h=1536, vol = 6144  (= FatBlock validated 1x6144 EXACTLY)
#   MLP 3.25   -> h=1664, vol = 6656  (8% over FatBlock; sliding 1.08226 last run)
#   MLP 3.4    -> h=1740, vol = 6960  (13% over; sliding 1.08601 — backfired)
#
# Variants:
#   smoke        — 30 s sanity (TTT off, MLP 3.25)
#   mlp325_ttt   — main: MLP 3.25 + TTT [default]; predicted artifact ~15.85 MB
#   mlp30_ttt    — hypothesis test: MLP 3.0 + TTT; predicted artifact ~15.58 MB
#   mlp325_no_ttt — control: reproduce previous 1.08226 (sliding only, no TTT)
#
# Examples:
#   ./run.sh                           # mlp325_ttt, seed 42
#   ./run.sh mlp30_ttt 42
#   ./run.sh smoke
#
# Logs land in logs/<variant>_s<seed>.log.

set -e
set -o pipefail

# Ensure brotli is installed (project default COMPRESSOR=brotli; H100 container
# does NOT have it pre-installed — and a missing brotli only fails 40 min in
# during serialize, after training is otherwise complete).
if ! python3 -c "import brotli" 2>/dev/null; then
    echo "[setup] brotli not installed — installing now..."
    pip install brotli
fi

# Defensively unset any inherited vars that could leak into the run.
unset GATED_ATTN_ENABLED GATED_ATTN_MODE MLP_MULT PARALLEL_MLP_MULT \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED TTT_LR TTT_EPOCHS TTT_MOMENTUM TTT_CHUNK_TOKENS \
      SLIDING_WINDOW_ENABLED SEED DATA_DIR \
      ITERATIONS VAL_LOSS_EVERY COMPRESSOR \
      MUON_WD MATRIX_LR MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      PARALLEL_RESIDUAL_START

VARIANT=${1:-mlp325_ttt}
SEED=${2:-42}

mkdir -p logs

# Common settings.
export DATA_DIR=../../../data
export SEED=$SEED
export GATED_ATTN_ENABLED=1
export GATED_ATTN_MODE=elementwise
export MLP_MULT=4.0

case "$VARIANT" in
    smoke)
        export PARALLEL_MLP_MULT=3.25
        export TTT_ENABLED=0
        export MAX_WALLCLOCK_SECONDS=30
        export TRAIN_BATCH_TOKENS=65536
        export WARMUP_STEPS=1
        export SLIDING_WINDOW_ENABLED=0
        ;;
    mlp325_ttt)
        # Highest-confidence sub-SOTA bet. MLP 3.25 already produced sliding
        # 1.08226 (sub-SOTA-by-0.0006); TTT typically subtracts ~0.002 BPB.
        # Projected TTT: ~1.080.
        export PARALLEL_MLP_MULT=3.25
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
    mlp30_ttt)
        # Hypothesis: per-token MLP volume = 6144 EXACTLY matches FatBlock,
        # easier to quantize, more headroom, more steps per wall-clock.
        # Predicted artifact ~15.58 MB (~420 KB headroom).
        export PARALLEL_MLP_MULT=3.0
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
    mlp325_no_ttt)
        # Sanity-reproduce the prior MLP 3.25 result (sliding 1.08226).
        export PARALLEL_MLP_MULT=3.25
        export TTT_ENABLED=0
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke mlp325_ttt mlp30_ttt mlp325_no_ttt" >&2
        exit 1
        ;;
esac

LOGFILE="logs/${VARIANT}_s${SEED}.log"
echo "=============================================="
echo "  Variant: $VARIANT   |   Seed: $SEED"
echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED  MODE=$GATED_ATTN_MODE"
echo "  MLP_MULT=$MLP_MULT  PARALLEL_MLP_MULT=$PARALLEL_MLP_MULT"
echo "  TTT_ENABLED=$TTT_ENABLED  ${TTT_ENABLED:+(lr=${TTT_LR:-} epochs=${TTT_EPOCHS:-} chunk=${TTT_CHUNK_TOKENS:-})}"
echo "  Log: $LOGFILE"
echo "=============================================="

torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOGFILE"

echo "=============================================="
echo "  DONE: $VARIANT seed=$SEED"
ARTIFACT_LINE=$(grep 'Total submission size' "$LOGFILE" | tail -1)
QUANT_LINE=$(grep '^quantized val_loss' "$LOGFILE" | tail -1)
SLIDING_LINE=$(grep 'quantized_sliding_window' "$LOGFILE" | tail -1)
TTT_LINE=$(grep 'quantized_ttt' "$LOGFILE" | tail -1)
echo "  Artifact:  ${ARTIFACT_LINE:-not found}"
echo "  Quantized: ${QUANT_LINE:-not found}"
echo "  Sliding:   ${SLIDING_LINE:-not found}"
echo "  TTT:       ${TTT_LINE:-not found}    <-- HEADLINE: target <= 1.0810"
echo "=============================================="
