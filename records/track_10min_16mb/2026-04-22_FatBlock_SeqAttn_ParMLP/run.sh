#!/bin/bash
# Launcher for the fat-block attention-variants sweep on 2x H100, 40 min train each.
#
# Usage:
#     ./run.sh <variant> [seed]
#
# Variants (sweep 1 — completed 2026-04-23):
#     smoke     — 30 s architecture sanity check (tiny batch, no sliding eval)
#     vanilla   — fat block only, no nonlinearity addons
#     gated_hw  — fat block + Qwen G1 gated attention (headwise sigmoid)
#     gated_ew  — fat block + Qwen G1 gated attention (elementwise sigmoid)  [WINNER, 1.08292]
#     glu_v     — fat block + GLU on V projection
#     both      — fat block + headwise gate + GLU-V
#     both_ew   — fat block + elementwise gate + GLU-V
#
# Variants (sweep 2 — architectural follow-ups, built on gated_ew):
#     delete_mlp_widen — [A1] delete fat MLP + widen fat attns (head_dim 64->96)
#     mlp_sequential   — [A2] big MLP reads z instead of x_in
#     leaky_attn       — [A3] LeakyReLU^2 on SDPA output before gate+proj
#
# Defaults: seed=42. Log lands in logs/<variant>_s<seed>.log.
#
# Examples:
#     ./run.sh smoke                # quick sanity check
#     ./run.sh gated_ew             # current winner
#     ./run.sh delete_mlp_widen     # top A1 bet
#     for V in delete_mlp_widen mlp_sequential leaky_attn; do ./run.sh $V; done

set -e

# Defensively unset any shell-exported vars that could leak into the training run.
# (Needed because earlier smoke-test sessions may have `export`ed these, and
#  `env VAR=val command` does NOT unset unlisted inherited vars.)
unset FAT_BLOCK_ENABLED FAT_BLOCK_NUM_ATTNS FAT_BLOCK_MLP_HIDDEN \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED SLIDING_WINDOW_ENABLED SEED \
      GATED_ATTN GATED_ATTN_MODE GLU_V DATA_DIR \
      MUON_WD MATRIX_LR COMPRESSOR ITERATIONS VAL_LOSS_EVERY \
      MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      FAT_ATTN_HEAD_DIM FAT_BLOCK_MLP_ENABLED FAT_BLOCK_MLP_MODE ATTN_OUTPUT_ACTIVATION

VARIANT=${1:-vanilla}
SEED=${2:-42}

# Base for follow-up variants: start from gated_ew (the sweep-1 winner) and
# toggle one architectural change per variant so each is a clean ablation.
GATED_EW_BASE="GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0"

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
        EXTRA="$GATED_EW_BASE"
        ;;
    glu_v)
        EXTRA="GATED_ATTN=0 GLU_V=1"
        ;;
    both)
        EXTRA="GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1"
        ;;
    both_ew)
        EXTRA="GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=1"
        ;;
    # ------------------------------------------------------------------
    # Sweep 2: architectural follow-ups (gated_ew + one modification each)
    # ------------------------------------------------------------------
    delete_mlp_widen)
        # A1: delete the fat-block MLP and widen its 4 attentions (head_dim 64->96).
        # Frees ~2.3 MB of artifact (big MLP gone); reinvests ~0.7 MB into wider attns.
        # Net: smaller artifact (~14.3 MB), ~13% more compute per step.
        EXTRA="$GATED_EW_BASE FAT_BLOCK_MLP_ENABLED=0 FAT_ATTN_HEAD_DIM=96"
        ;;
    mlp_sequential)
        # A2: big MLP reads post-attention-chain state z instead of fat-block
        # input x_in. Same architecture otherwise. Zero param change.
        EXTRA="$GATED_EW_BASE FAT_BLOCK_MLP_MODE=sequential"
        ;;
    leaky_attn)
        # A3: apply leaky_relu(y, 0.5).square() on SDPA output, before gate+proj,
        # in EVERY attention (all 11 blocks). Zero param change.
        EXTRA="$GATED_EW_BASE ATTN_OUTPUT_ACTIVATION=leaky_relu_sq"
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke vanilla gated_hw gated_ew glu_v both both_ew" >&2
        echo "                 delete_mlp_widen mlp_sequential leaky_attn" >&2
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
