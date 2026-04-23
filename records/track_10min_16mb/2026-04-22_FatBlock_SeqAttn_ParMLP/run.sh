#!/bin/bash
# Launcher for the fat-block attention-variants sweep on 2x H100, 40 min train each.
#
# Usage:
#     ./run.sh <variant> [seed]
#
# Variants:
#     smoke     — 30 s architecture sanity check (tiny batch, no sliding eval)
#     vanilla   — fat block only, no nonlinearity addons
#     gated_hw  — fat block + Qwen G1 gated attention (headwise sigmoid)
#     gated_ew  — fat block + Qwen G1 gated attention (elementwise sigmoid)
#     glu_v     — fat block + GLU on V projection
#     both      — fat block + headwise gate + GLU-V
#
# Defaults: seed=42. Log lands in logs/<variant>_s<seed>.log.
#
# Examples:
#     ./run.sh smoke                # quick sanity check
#     ./run.sh vanilla              # main variant 1, seed 42
#     ./run.sh both 314             # stackability variant at a different seed
#     for V in vanilla gated_hw gated_ew glu_v both; do ./run.sh $V; done

set -e

# Defensively unset any shell-exported vars that could leak into the training run.
# (Needed because earlier smoke-test sessions may have `export`ed these, and
#  `env VAR=val command` does NOT unset unlisted inherited vars.)
unset FAT_BLOCK_ENABLED FAT_BLOCK_NUM_ATTNS FAT_BLOCK_MLP_HIDDEN \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED SLIDING_WINDOW_ENABLED SEED \
      GATED_ATTN GATED_ATTN_MODE GLU_V DATA_DIR \
      MUON_WD MATRIX_LR COMPRESSOR ITERATIONS VAL_LOSS_EVERY \
      MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS

VARIANT=${1:-vanilla}
SEED=${2:-42}

case "$VARIANT" in
    smoke)
        EXTRA="GATED_ATTN=0 GLU_V=0"
        SMOKE=1
        ;;
    vanilla)
        EXTRA="GATED_ATTN=0 GLU_V=0"
        ;;
    gated_hw)
        EXTRA="GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0"
        ;;
    gated_ew)
        EXTRA="GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0"
        ;;
    glu_v)
        EXTRA="GATED_ATTN=0 GLU_V=1"
        ;;
    both)
        EXTRA="GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1"
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke vanilla gated_hw gated_ew glu_v both" >&2
        exit 1
        ;;
esac

mkdir -p logs

# Shared settings across all full-training variants.
BASE="FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144"
BASE="$BASE DATA_DIR=../../../data TTT_ENABLED=0 SEED=$SEED"

if [ "${SMOKE:-0}" = "1" ]; then
    # Smoke test: tiny batch, 30 s training cap, no sliding eval.
    BASE="$BASE MAX_WALLCLOCK_SECONDS=30 TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=1 SLIDING_WINDOW_ENABLED=0"
else
    # Full training: 40 min wall-clock cap (training-only; eval runs after).
    # Explicitly set full-training values for the smoke-only overrides in case
    # they leaked from a prior shell session.
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
