#!/bin/bash
# Recovery launcher — skips training, reuses final_model.pt written before the
# brotli ImportError, runs only the post-training pipeline (GPTQ + serialize +
# quantized eval + sliding window + TTT). Saves ~40 min vs a full re-run.
#
# Prereq: pip install brotli   (the original failure was because brotli was
# not installed in this container's Python env)
#
# Usage:
#   ./recover.sh                # default: seed 42
#   ./recover.sh 314             # other seed (must match the saved final_model.pt)
#
# Log: logs/recover_s<seed>.log

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

SEED=${1:-42}

# Ensure brotli is installed; if not, install it now.
if ! python3 -c "import brotli" 2>/dev/null; then
    echo "[recover] brotli not installed — installing now..."
    pip install brotli
fi

# Sanity: final_model.pt must exist
if [ ! -f final_model.pt ]; then
    echo "[recover] ERROR: final_model.pt not found in $(pwd)" >&2
    echo "[recover] The original run must have saved this checkpoint before the brotli failure." >&2
    exit 1
fi

mkdir -p logs

# Same arch + TTT settings as the main run (must match what produced final_model.pt).
export DATA_DIR=../../../data
export SEED=$SEED
export GATED_ATTN_ENABLED=1
export GATED_ATTN_MODE=elementwise
export MLP_MULT=4.0
export PARALLEL_MLP_MULT=3.4
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_CHUNK_TOKENS=32768
export SLIDING_WINDOW_ENABLED=1

LOGFILE="logs/recover_s${SEED}.log"

echo "=============================================="
echo "  RECOVERY  |  Seed: $SEED"
echo "  Loading final_model.pt, skipping training."
echo "  Will run: GPTQ + serialize + quantized eval + sliding + TTT"
echo "  Log: $LOGFILE"
echo "=============================================="

torchrun --standalone --nproc_per_node=2 recover_eval.py 2>&1 | tee "$LOGFILE"

echo "=============================================="
echo "  RECOVERY DONE: seed=$SEED"
ARTIFACT_LINE=$(grep 'Total submission size' "$LOGFILE" | tail -1)
QUANT_LINE=$(grep '^quantized val_loss' "$LOGFILE" | tail -1)
SLIDING_LINE=$(grep 'quantized_sliding_window' "$LOGFILE" | tail -1)
TTT_LINE=$(grep 'quantized_ttt' "$LOGFILE" | tail -1)
echo "  Artifact:  ${ARTIFACT_LINE:-not found}"
echo "  Quantized: ${QUANT_LINE:-not found}"
echo "  Sliding:   ${SLIDING_LINE:-not found}"
echo "  TTT:       ${TTT_LINE:-not found}"
echo "=============================================="
