#!/bin/bash
# Launcher for the LoopShare-MLP / Unique-Attn experiments on 2x H100, 40 min train each.
#
# All variants build on top of SOTA-equivalent base (FAT_BLOCK_ENABLED=0, no TTT).
# Modifications confined to the looped core (blocks loop_start..loop_end, default 3..5).
#
# Variants:
#     smoke_v1               — 30 s sanity for V1 (unique attns + thin shared MLP)
#     smoke_v2               — 30 s sanity for V2 (shared fat MLP)
#     shared_fat_mlp         — V2: 1 shared fat MLP (h=6144) across blocks 3, 4, 5
#     unique_attn_thin_mlp   — V1: 9 unique attns (3 per block × 3 visits) +
#                                  1 thin shared MLP (h=1280) across blocks 3, 4, 5
#
# Defaults: seed=42. Log lands in logs/<variant>_s<seed>.log.
#
# Examples:
#     ./run.sh smoke_v1                 # quick sanity check for V1
#     ./run.sh smoke_v2                 # quick sanity check for V2
#     ./run.sh shared_fat_mlp           # V2 full run
#     ./run.sh unique_attn_thin_mlp     # V1 full run

set -e
# pipefail propagates the torchrun exit code through the `| tee` pipeline.
# Without this, training crashes are masked by tee's 0 exit and the script
# pretends the run succeeded (which then poisons any wrapper's success check).
set -o pipefail

# Defensively unset any shell-exported vars that could leak into the training run.
# (Needed because earlier smoke-test sessions may have `export`ed these, and
#  `env VAR=val command` does NOT unset unlisted inherited vars.)
unset FAT_BLOCK_ENABLED FAT_BLOCK_NUM_ATTNS FAT_BLOCK_MLP_HIDDEN \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED SLIDING_WINDOW_ENABLED SEED \
      GATED_ATTN GATED_ATTN_MODE GLU_V DATA_DIR \
      MUON_WD MATRIX_LR COMPRESSOR ITERATIONS VAL_LOSS_EVERY \
      MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      FAT_ATTN_HEAD_DIM FAT_BLOCK_MLP_ENABLED FAT_BLOCK_MLP_MODE ATTN_OUTPUT_ACTIVATION \
      LOOP_SHARED_MLP LOOP_SHARED_MLP_HIDDEN LOOP_UNIQUE_ATTN

VARIANT=${1:-shared_fat_mlp}
SEED=${2:-42}

case "$VARIANT" in
    smoke_v1)
        # 30 s sanity: V1 with reduced training, sliding eval off
        EXTRA="LOOP_UNIQUE_ATTN=1 LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=1280"
        SMOKE=1
        ;;
    smoke_v2)
        # 30 s sanity: V2 with reduced training, sliding eval off
        EXTRA="LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=6144"
        SMOKE=1
        ;;
    shared_fat_mlp)
        # V2: SOTA + 1 shared fat MLP (h=6144) across blocks 3, 4, 5.
        # Param count ~equal to SOTA (3 × h=2048 = h=6144 effective).
        EXTRA="LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=6144"
        ;;
    unique_attn_thin_mlp)
        # V1: 9 unique attentions (3 per looped block × 3 visits) + 1 thin
        # shared MLP (h=1280) across blocks 3, 4, 5. Tests "attention
        # specialization across visits matters; MLP can be shared."
        EXTRA="LOOP_UNIQUE_ATTN=1 LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=1280"
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke_v1 smoke_v2 shared_fat_mlp unique_attn_thin_mlp" >&2
        exit 1
        ;;
esac

mkdir -p logs

# Base settings — SOTA-equivalent (FAT_BLOCK_ENABLED=0, no TTT).
BASE="FAT_BLOCK_ENABLED=0 DATA_DIR=../../../data TTT_ENABLED=0 SEED=$SEED"

if [ "${SMOKE:-0}" = "1" ]; then
    # Smoke: tiny batch, 30 s training cap, no sliding eval.
    BASE="$BASE MAX_WALLCLOCK_SECONDS=30 TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=1 SLIDING_WINDOW_ENABLED=0"
else
    # Full training: 40 min wall-clock cap (training-only; eval runs after).
    BASE="$BASE MAX_WALLCLOCK_SECONDS=2400 TRAIN_BATCH_TOKENS=786432 WARMUP_STEPS=20 SLIDING_WINDOW_ENABLED=1"
fi

LOGFILE="logs/${VARIANT}_s${SEED}.log"

echo "=============================================="
echo "  Variant: $VARIANT   |   Seed: $SEED"
echo "  Log: $LOGFILE"
echo "=============================================="

env $BASE $EXTRA torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee "$LOGFILE"

echo "=============================================="
echo "  DONE: $VARIANT seed=$SEED"
SLIDING_LINE=$(grep 'quantized_sliding_window' "$LOGFILE" | tail -1)
QUANT_LINE=$(grep '^quantized val_loss' "$LOGFILE" | tail -1)
ARTIFACT_LINE=$(grep 'Total submission size' "$LOGFILE" | tail -1)
echo "  Sliding:   ${SLIDING_LINE:-not found (smoke test? or failed)}"
echo "  Quantized: ${QUANT_LINE:-not found}"
echo "  Artifact:  ${ARTIFACT_LINE:-not found}"
echo "=============================================="
