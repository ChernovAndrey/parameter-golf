#!/bin/bash
# Launcher for "Gated_EW everywhere + MLP 3.6x" experiment.
# 2x H100, 40 min training (eval ~9 min more).
#
# Architecture (vs PR #1493 SOTA, 1.0810 BPB):
#   + Qwen G1 elementwise gated attention on ALL 11 layers
#       y = sigmoid(W_g x) * Attn(x), per-layer W_g of shape [dim, dim]
#       Adds 11 x 262K = 2.88M params (~+1.0 MB after int6 GPTQ + Brotli-11)
#   - MLP_MULT 4.0 -> 3.6  (hidden 2048 -> 1843)
#       Saves 11 x 2 x 512 x 205 = 2.31M params (~-1.13 MB)
#       Principled: gates inject per-token nonlinearity into attention,
#       so the MLP can shed some capacity without losing total expressivity.
#   = Net artifact: ~15.89 MB (within budget; ~110 KB headroom)
#     Fallback: drop MLP_MULT to 3.5 (~390 KB headroom) if a run lands over.
#
# Variants:
#   smoke           — 30 s CPU-side sanity check (tiny batch, no sliding eval)
#   gated_ew        — main run: gated_ew on every layer + MLP 3.6x  [default]
#   gated_ew_safe   — same as gated_ew but MLP 3.5x (more headroom, less capacity)
#   gated_hw        — ablation: cheap headwise gate everywhere (~+0 MB) + MLP 4.0x
#   baseline_mlp36  — ablation: no gate, just MLP 3.6x (separates the two effects)
#
# Examples:
#   ./run.sh                    # gated_ew, seed 42
#   ./run.sh gated_ew 314
#   ./run.sh smoke
#   ./run.sh baseline_mlp35
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
unset GATED_ATTN_ENABLED GATED_ATTN_MODE MLP_MULT \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED SLIDING_WINDOW_ENABLED SEED DATA_DIR \
      ITERATIONS VAL_LOSS_EVERY COMPRESSOR \
      MUON_WD MATRIX_LR MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS

VARIANT=${1:-gated_ew}
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
        export MLP_MULT=3.6
        export MAX_WALLCLOCK_SECONDS=30
        export TRAIN_BATCH_TOKENS=65536
        export WARMUP_STEPS=1
        export SLIDING_WINDOW_ENABLED=0
        ;;
    gated_ew)
        # Main run: elementwise gate on every attention layer, MLP 3.6x.
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=3.6
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    gated_ew_safe)
        # Same as gated_ew but with MLP 3.5x — use if gated_ew lands > 16 MB.
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=elementwise
        export MLP_MULT=3.5
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    gated_hw)
        # Cheap headwise variant: keeps SOTA's MLP 4.0x (gate is ~free).
        export GATED_ATTN_ENABLED=1
        export GATED_ATTN_MODE=headwise
        export MLP_MULT=4.0
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    baseline_mlp36)
        # Isolates the MLP_MULT=3.6 effect from the gate effect.
        export GATED_ATTN_ENABLED=0
        export MLP_MULT=3.6
        export MAX_WALLCLOCK_SECONDS=2400
        export TRAIN_BATCH_TOKENS=786432
        export WARMUP_STEPS=20
        export SLIDING_WINDOW_ENABLED=1
        ;;
    *)
        echo "Unknown variant: $VARIANT" >&2
        echo "Valid variants: smoke gated_ew gated_ew_safe gated_hw baseline_mlp36" >&2
        exit 1
        ;;
esac

LOGFILE="logs/${VARIANT}_s${SEED}.log"
echo "=============================================="
echo "  Variant: $VARIANT   |   Seed: $SEED"
echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED  MODE=${GATED_ATTN_MODE:-n/a}  MLP_MULT=$MLP_MULT"
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
