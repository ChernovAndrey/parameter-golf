#!/bin/bash
# Post-hoc TTT eval for the MLP 3.25 + parallel-only-gates checkpoint that was
# trained without TTT (final sliding 1.08226). Loads final_model.pt and runs
# only the eval pipeline with TTT_ENABLED=1.
#
# This does NOT retrain — uses the EMA-weighted checkpoint already on disk.
# Same architecture (PARALLEL_MLP_MULT=3.25, gates only on layers 7-10) so
# the resulting `quantized_ttt val_bpb` is what the original run WOULD have
# produced if TTT had been enabled at the time.
#
# Prereq: final_model.pt must exist in this folder (saved during the original
# completed run, before post-train eval finished).
#
# Usage:
#   ./recover.sh                # default: seed 42
#   ./recover.sh 314             # other seed (must match saved final_model.pt)
#
# Log: logs/recover_ttt_s<seed>.log

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

SEED=${1:-42}

if [ ! -f final_model.pt ]; then
    echo "[recover] ERROR: final_model.pt not found in $(pwd)" >&2
    echo "[recover] The original gated_ew_parallel s$SEED run must have saved this checkpoint." >&2
    exit 1
fi

mkdir -p logs

# MUST match the architecture of the saved checkpoint:
# - 4 elementwise gates on parallel layers (7-10)
# - MLP 4.0x on 0-6, 3.25x on 7-10
# - SP8192 vocab, 11 physical layers, etc. (defaults in train_gpt.py)
export DATA_DIR=../../../data
export SEED=$SEED
export GATED_ATTN_ENABLED=1
export GATED_ATTN_MODE=elementwise
export MLP_MULT=4.0
export PARALLEL_MLP_MULT=3.25
# The eval-only changes vs the original run:
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_CHUNK_TOKENS=32768
export SLIDING_WINDOW_ENABLED=1

LOGFILE="logs/recover_ttt_s${SEED}.log"

echo "=============================================="
echo "  POST-HOC TTT EVAL  |  Seed: $SEED"
echo "  Architecture: parallel-only gates + MLP 3.25x (=> sliding 1.08226 was the result without TTT)"
echo "  Loading final_model.pt, skipping training, running eval with TTT enabled."
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
echo "  TTT:       ${TTT_LINE:-not found}    <-- HEADLINE: target <= 1.0810"
echo "=============================================="
