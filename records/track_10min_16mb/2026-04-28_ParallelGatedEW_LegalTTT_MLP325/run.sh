#!/bin/bash
# Submission-grade launcher: 3 seeds back-to-back, 8x H100 / 600s training
# (the official competition spec — same as PR #1493 SOTA used).
#
# Per-seed timing (8x H100): 10 min train (600s wall-clock cap) + ~8 min eval
# (sliding ~2 min + TTT ~6 min) = ~18 min/seed. Three seeds ≈ 55 minutes total.
# COSTLY in $$ — every safeguard below is intentional.
#
# Architecture (vs PR #1493 SOTA, 1.0810 BPB):
#   Layers 0-6: MLP 4.0x, no gate (unchanged from SOTA)
#   Layers 7-10: MLP 3.25x (h=1664) + Qwen G1 elementwise gated attention
#   + Legal Score-First TTT (Issue #1017 Track B; same params as SOTA)
#
# Single-seed reference (no TTT, 2x H100 / 2400s, 2026-04-27):
#   sliding val_bpb 1.08226, artifact 15,804,975 bytes (195 KB headroom).
#
# Robustness features:
#   - Pre-flight: GPU count == 8, modules importable, data present
#   - Per-seed artifact preservation: final_model.pt and .int6.ptz are renamed
#     to seed{N}_*.pt|.ptz immediately after each seed completes, so a later
#     failure doesn't overwrite earlier seeds' results.
#   - Idempotent: skips seeds whose seed{N}_final_model.int6.ptz already exists.
#   - brotli auto-install at the top.
#
# Usage:
#   ./run.sh                  # all 3 seeds (skips already-done ones)
#   ./run.sh 42               # just seed 42
#   ./run.sh 314
#   ./run.sh 999
#   ./run.sh smoke            # 30s, 8 GPU, no sliding, no TTT — pre-flight test
#   ./run.sh check            # only run the pre-flight checks, exit 0 on pass

set -e
set -o pipefail

# ============================================================
# 0) brotli auto-install (project default COMPRESSOR=brotli;
#    H100 container historically does NOT have it pre-installed).
# ============================================================
if ! python3 -c "import brotli" 2>/dev/null; then
    echo "[setup] brotli not installed — installing now..."
    pip install brotli
fi

# ============================================================
# 1) Pre-flight checks (cheap, run before any GPU work).
# ============================================================
preflight () {
    echo "[preflight] running checks..."
    # 1a) Required Python modules importable
    for mod in torch flash_attn_interface sentencepiece numpy brotli; do
        python3 -c "import $mod" 2>/dev/null || { echo "[preflight] FAIL: cannot import $mod" >&2; exit 1; }
    done
    echo "[preflight] OK: torch, flash_attn_interface, sentencepiece, numpy, brotli importable"

    # 1b) Exactly 8 visible GPUs (the competition spec)
    local n_gpu=$(python3 -c "import torch; print(torch.cuda.device_count())")
    if [ "$n_gpu" != "8" ]; then
        echo "[preflight] WARN: torch.cuda.device_count() = $n_gpu (expected 8 per competition spec)" >&2
        echo "[preflight] Override OK_DIFFERENT_GPU_COUNT=1 to proceed anyway." >&2
        if [ "${OK_DIFFERENT_GPU_COUNT:-0}" != "1" ]; then exit 1; fi
    fi
    echo "[preflight] OK: $n_gpu GPUs visible"

    # 1c) Tokenizer + at least one train shard present
    local tokenizer="../../../data/tokenizers/fineweb_8192_bpe.model"
    local train_glob="../../../data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin"
    [ -f "$tokenizer" ] || { echo "[preflight] FAIL: $tokenizer not found" >&2; exit 1; }
    [ -f "$train_glob" ] || { echo "[preflight] FAIL: training shard 000000 not found at $train_glob" >&2; exit 1; }
    echo "[preflight] OK: tokenizer + train shard present"

    # 1d) Print versions
    python3 -c "import torch; print(f'[preflight] torch={torch.__version__}  cuda={torch.version.cuda}')"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1 | awk '{print "[preflight] GPU:", $0}'
}
preflight

# Allow ./run.sh check to exit after preflight (cheap dry-run before paying).
if [ "${1:-}" = "check" ]; then
    echo "[preflight] all checks passed — exiting before any GPU work."
    exit 0
fi

# ============================================================
# 2) Defensively unset env vars; set the architecture config.
# ============================================================
unset GATED_ATTN_ENABLED GATED_ATTN_MODE MLP_MULT PARALLEL_MLP_MULT \
      TRAIN_BATCH_TOKENS WARMUP_STEPS MAX_WALLCLOCK_SECONDS \
      TTT_ENABLED TTT_LR TTT_EPOCHS TTT_MOMENTUM TTT_CHUNK_TOKENS \
      SLIDING_WINDOW_ENABLED SEED DATA_DIR \
      ITERATIONS VAL_LOSS_EVERY COMPRESSOR \
      MUON_WD MATRIX_LR MATRIX_CLIP_SIGMAS EMBED_CLIP_SIGMAS \
      PARALLEL_RESIDUAL_START

ARG=${1:-all}

# Architecture (must NOT change between seeds — judging requires identical arch)
export DATA_DIR=../../../data
export GATED_ATTN_ENABLED=1
export GATED_ATTN_MODE=elementwise
export MLP_MULT=4.0
export PARALLEL_MLP_MULT=3.25
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_MOMENTUM=0.9
export TTT_CHUNK_TOKENS=32768
export SLIDING_WINDOW_ENABLED=1

# ============================================================
# 3) Smoke variant: 30s, full 8-GPU stack, no sliding, no TTT.
#    Confirms the architecture builds + data loads + GPTQ works
#    + artifact size lands under 16 MB before paying for full run.
# ============================================================
if [ "$ARG" = "smoke" ]; then
    export SEED=42
    export TTT_ENABLED=0
    export MAX_WALLCLOCK_SECONDS=30
    export TRAIN_BATCH_TOKENS=65536
    export WARMUP_STEPS=1
    export SLIDING_WINDOW_ENABLED=0
    LOGFILE=train_smoke.log
    echo "=============================================="
    echo "  SMOKE  |  8 GPU  |  30s  |  no sliding  |  no TTT"
    echo "=============================================="
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "=============================================="
    echo "  SMOKE artifact line:"
    grep 'Total submission size' "$LOGFILE" | tail -1 || echo "  (no artifact line — smoke may have failed before serialize)"
    # Smoke leaves final_model* on disk; do NOT preserve them as seed42 artifacts.
    rm -f final_model.pt final_model.int6.ptz
    echo "=============================================="
    exit 0
fi

# ============================================================
# 4) Full run: 8 GPU, 600s training, sliding+TTT eval.
# ============================================================
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=786432
export WARMUP_STEPS=20

run_one_seed () {
    local seed=$1
    local LOGFILE="train_seed${seed}.log"
    local SEED_PT="seed${seed}_final_model.pt"
    local SEED_PTZ="seed${seed}_final_model.int6.ptz"

    # Idempotency: if this seed's artifact already exists, skip.
    if [ -f "$SEED_PTZ" ] && [ -f "$LOGFILE" ]; then
        echo "[run] seed=$seed already completed (have $SEED_PTZ) — skipping. Delete to re-run."
        return 0
    fi

    export SEED=$seed
    # Clear any leftover working files from a prior partial run.
    rm -f final_model.pt final_model.int6.ptz

    echo "=============================================="
    echo "  SUBMISSION RUN  |  Seed: $seed  |  8 GPU  |  600s + eval"
    echo "  GATED_ATTN_ENABLED=$GATED_ATTN_ENABLED  PARALLEL_MLP_MULT=$PARALLEL_MLP_MULT"
    echo "  TTT_ENABLED=$TTT_ENABLED (lr=$TTT_LR epochs=$TTT_EPOCHS chunk=$TTT_CHUNK_TOKENS)"
    echo "  Log: $LOGFILE"
    echo "=============================================="

    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOGFILE"

    # Preserve per-seed artifacts immediately after the run, so the next seed
    # can't overwrite them. final_model.pt is the EMA-weighted full-precision
    # checkpoint (recoverable if a later step blows up); .int6.ptz is the
    # actual submission blob.
    if [ -f final_model.pt ]; then mv final_model.pt "$SEED_PT"; fi
    if [ -f final_model.int6.ptz ]; then mv final_model.int6.ptz "$SEED_PTZ"; fi

    echo "=============================================="
    echo "  DONE: seed=$seed  artifacts saved as $SEED_PT, $SEED_PTZ"
    grep 'Total submission size' "$LOGFILE" | tail -1
    grep 'quantized_sliding_window' "$LOGFILE" | tail -1
    grep 'quantized_ttt' "$LOGFILE" | tail -1
    echo "=============================================="
}

if [ "$ARG" = "all" ]; then
    for s in 42 314 999; do
        run_one_seed "$s"
    done
elif [[ "$ARG" =~ ^[0-9]+$ ]]; then
    run_one_seed "$ARG"
else
    echo "Unknown arg: $ARG" >&2
    echo "Usage: $0 [all|42|314|999|smoke|check]" >&2
    exit 1
fi

# ============================================================
# 5) Final summary across all seeds (whatever's present on disk).
# ============================================================
echo "=============================================="
echo "  ALL DONE — summary"
echo "=============================================="
for s in 42 314 999; do
    f="train_seed${s}.log"
    if [ -f "$f" ]; then
        echo "--- seed $s ---"
        grep 'Total submission size' "$f" | tail -1
        grep 'quantized_sliding_window' "$f" | tail -1
        grep 'quantized_ttt' "$f" | tail -1
    fi
done
echo "=============================================="
echo "  Per-seed checkpoints saved as seed{42,314,999}_final_model{,.int6.ptz}"
echo "  Submit by filling submission.json + README.md tables, then committing"
echo "  this folder. The .log files are part of the submission."
echo "=============================================="
