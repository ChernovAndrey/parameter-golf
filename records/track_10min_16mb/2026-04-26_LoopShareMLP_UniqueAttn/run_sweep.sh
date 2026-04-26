#!/bin/bash
# Sequential launcher for the LoopShare-MLP / Unique-Attn experiment.
#
# Runs both architectural variants back-to-back on the same 2xH100 node:
#     shared_fat_mlp        (V2) — 1 shared fat MLP (h=6144) across blocks 3, 4, 5
#     unique_attn_thin_mlp  (V1) — 9 unique attns (3 per block × 3 visits) +
#                                  1 thin shared MLP (h=1280) across blocks 3, 4, 5
#
# Each variant uses the 40-min training cap baked into run.sh
# (MAX_WALLCLOCK_SECONDS=2400). Per-variant logs land in logs/<variant>_s<seed>.log;
# per-variant compressed artifacts are preserved under artifacts/.
#
# Usage:
#     ./run_sweep.sh [seed]            # defaults to seed=42
#
# Behaviour:
#     * On a variant failure, record it and continue with the next variant.
#     * Keeps only the compressed <variant>_s<seed>_final_model.int6.ptz under
#       artifacts/; the ~135 MB final_model.pt is deleted after each run.
#     * Prints a compact results table at the end.

set -u  # NOT set -e: we want to continue past variant failures.

SEED=${1:-42}
VARIANTS=(shared_fat_mlp unique_attn_thin_mlp)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Ensure brotli is available — the trainer's serialize() step imports it at the
# very end of training, so a missing module silently wastes the full ~50 min run
# (training succeeds, compression crashes, no artifact). Hit on a fresh pod
# during sweep-2 variant 1 — install up-front.
python3 -c "import brotli" 2>/dev/null || pip install -q brotli

mkdir -p artifacts logs
SWEEP_LOG="logs/sweep_s${SEED}.log"

declare -A STATUS

SWEEP_START_TS=$(date -Iseconds 2>/dev/null || date)
{
    echo "=============================================="
    echo "  LoopShare sweep start: $SWEEP_START_TS"
    echo "  Seed:          $SEED"
    echo "  Variants:      ${VARIANTS[*]}"
    echo "  Working dir:   $SCRIPT_DIR"
    echo "=============================================="
} | tee "$SWEEP_LOG"

for V in "${VARIANTS[@]}"; do
    TS_START=$(date -Iseconds 2>/dev/null || date)
    echo "### START $V seed=$SEED $TS_START" | tee -a "$SWEEP_LOG"

    # Make sure no stale artifact from a previous crashed variant leaks in.
    rm -f final_model.pt final_model.int6.ptz

    RC=0
    ./run.sh "$V" "$SEED" || RC=$?
    if [ "$RC" -eq 0 ] && [ -f final_model.int6.ptz ]; then
        mv final_model.int6.ptz "artifacts/${V}_s${SEED}_final_model.int6.ptz"
        STATUS[$V]="ok"
    elif [ "$RC" -ne 0 ]; then
        STATUS[$V]="FAILED(rc=${RC})"
    else
        # run.sh returned 0 but the artifact was never produced — treat as failure.
        STATUS[$V]="FAILED(no_artifact)"
    fi
    rm -f final_model.pt final_model.int6.ptz

    TS_END=$(date -Iseconds 2>/dev/null || date)
    echo "### END   $V seed=$SEED $TS_END  [${STATUS[$V]}]" | tee -a "$SWEEP_LOG"
done

# -------- Final summary --------
{
    echo ""
    echo "=============================================="
    echo "  LoopShare sweep summary (seed=$SEED)"
    echo "=============================================="
    printf "%-22s %-18s %-18s %-18s %-14s\n" \
        "variant" "status" "sliding_val_bpb" "quantized_val_bpb" "artifact_bytes"
    printf "%-22s %-18s %-18s %-18s %-14s\n" \
        "-------" "------" "---------------" "-----------------" "--------------"
    for V in "${VARIANTS[@]}"; do
        LOG="logs/${V}_s${SEED}.log"
        SLIDE="-"
        QUANT="-"
        ART="-"
        if [ -f "$LOG" ]; then
            LINE=$(grep 'quantized_sliding_window val_loss' "$LOG" 2>/dev/null | tail -1)
            [ -n "$LINE" ] && SLIDE=$(echo "$LINE" | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
            LINE=$(grep '^quantized val_loss' "$LOG" 2>/dev/null | tail -1)
            [ -n "$LINE" ] && QUANT=$(echo "$LINE" | grep -oE 'val_bpb:[0-9.]+' | head -1 | cut -d: -f2)
            LINE=$(grep 'Total submission size' "$LOG" 2>/dev/null | tail -1)
            [ -n "$LINE" ] && ART=$(echo "$LINE" | awk '{print $(NF-1)}')
        fi
        printf "%-22s %-18s %-18s %-18s %-14s\n" \
            "$V" "${STATUS[$V]:-?}" "$SLIDE" "$QUANT" "$ART"
    done
    echo "=============================================="
    echo "  Artifacts:        $(ls artifacts/ 2>/dev/null | wc -l | tr -d ' ') file(s) under artifacts/"
    echo "  Per-variant logs: logs/<variant>_s${SEED}.log"
    echo "  Sweep log:        $SWEEP_LOG"
    echo "=============================================="
} | tee -a "$SWEEP_LOG"
