#!/bin/bash
# Pull the per-seed artifacts (logs + model weights) from the RunPod box back
# to this local folder, after the 3-seed run completes.
#
# What gets pulled:
#   train_seed{42,314,999}.log              # full training+eval logs (REQUIRED for submission)
#   seed{42,314,999}_final_model.pt         # full-precision EMA checkpoint (~133 MB ea, BACKUP)
#   seed{42,314,999}_final_model.int6.ptz   # the int6+brotli artifact (~16 MB ea, the actual submission blob)
#   train_smoke.log                          # if it exists
#
# Run this from inside the submission folder on your LOCAL machine.
#
# Connection: provide the same `ssh ...` line RunPod gave you (the "Direct TCP"
# connection or "SSH over Exposed TCP"). Either pass it as $POD or fill the
# defaults below.
#
# Usage:
#   POD="root@123.45.67.89 -p 12345" ./pull_from_runpod.sh
#   POD="root@ssh.runpod.io -p 12345 -i ~/.ssh/id_ed25519" ./pull_from_runpod.sh
#   ./pull_from_runpod.sh                             # uses POD_USER/POD_HOST/POD_PORT/POD_KEY env vars
#
# Environment variables (any of these can be set instead of $POD):
#   POD_USER   default: root
#   POD_HOST   no default (required)
#   POD_PORT   default: 22
#   POD_KEY    default: (none — uses ssh-agent or default key)
#   POD_PATH   default: /workspace/parameter-golf/parameter-golf/records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325

set -euo pipefail

# ============================================================
# Resolve connection params — either parse $POD or read env vars
# ============================================================
if [ -n "${POD:-}" ]; then
    # POD looks like "root@1.2.3.4 -p 12345 [-i ~/.ssh/key]"
    SSH_TARGET=$(echo "$POD" | awk '{print $1}')
    SSH_OPTS=$(echo "$POD" | cut -d' ' -f2-)
else
    POD_USER="${POD_USER:-root}"
    POD_HOST="${POD_HOST:-}"
    POD_PORT="${POD_PORT:-22}"
    POD_KEY="${POD_KEY:-}"
    if [ -z "$POD_HOST" ]; then
        echo "ERROR: must provide either \$POD or \$POD_HOST." >&2
        echo "  Example: POD=\"root@ssh.runpod.io -p 12345\" ./pull_from_runpod.sh" >&2
        echo "  Or:      POD_HOST=ssh.runpod.io POD_PORT=12345 ./pull_from_runpod.sh" >&2
        exit 1
    fi
    SSH_TARGET="${POD_USER}@${POD_HOST}"
    SSH_OPTS="-p ${POD_PORT}"
    [ -n "$POD_KEY" ] && SSH_OPTS="$SSH_OPTS -i $POD_KEY"
fi

POD_PATH="${POD_PATH:-/workspace/parameter-golf/parameter-golf/records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325}"
LOCAL_DEST="$(pwd)"

echo "=============================================="
echo "  Pulling from: $SSH_TARGET   (ssh opts: $SSH_OPTS)"
echo "  Remote path:  $POD_PATH/"
echo "  Local dest:   $LOCAL_DEST/"
echo "=============================================="

# ============================================================
# Step 1 — quick remote inventory before pulling
# ============================================================
echo "[1/3] checking what exists on the pod..."
ssh $SSH_OPTS "$SSH_TARGET" "ls -lah $POD_PATH/ | grep -E 'train_seed|seed.*_final|train_smoke' || echo '(no matching files yet)'" || {
    echo "ERROR: could not connect or path missing. Verify SSH connection works:" >&2
    echo "  ssh $SSH_OPTS $SSH_TARGET ls $POD_PATH/" >&2
    exit 1
}

# ============================================================
# Step 2 — pull files via rsync (resumable, shows progress, only the files we care about)
# ============================================================
echo "[2/3] pulling logs + checkpoints (rsync, resumable)..."
rsync -avzh --partial --progress -e "ssh $SSH_OPTS" \
    --include='train_seed*.log' \
    --include='seed*_final_model.pt' \
    --include='seed*_final_model.int6.ptz' \
    --include='train_smoke.log' \
    --exclude='*' \
    "$SSH_TARGET:$POD_PATH/" "$LOCAL_DEST/"

# ============================================================
# Step 3 — verify what we got and print the headline numbers
# ============================================================
echo "[3/3] local summary after pull:"
echo "--- log files ---"
ls -lh "$LOCAL_DEST"/train_seed*.log "$LOCAL_DEST"/train_smoke.log 2>/dev/null || echo "(no log files)"
echo "--- per-seed checkpoints ---"
ls -lh "$LOCAL_DEST"/seed*_final_model* 2>/dev/null || echo "(no checkpoints)"
echo
echo "--- headline per-seed numbers ---"
for s in 42 314 999; do
    f="$LOCAL_DEST/train_seed${s}.log"
    if [ -f "$f" ]; then
        echo "seed $s:"
        grep -E 'Total submission size|quantized_sliding_window|quantized_ttt' "$f" | tail -3 | sed 's/^/    /'
    fi
done
echo "=============================================="
echo "Pull complete."
echo
echo "Next steps to finalize the submission:"
echo "  1. Fill submission.json with the 3 seed_results + mean/std (use the headline numbers above)."
echo "  2. Fill the 3-seed table at the top of README.md."
echo "  3. git add train_seed{42,314,999}.log README.md submission.json train_gpt.py run.sh"
echo "  4. (Optional) DO NOT commit the .pt/.ptz files — submission convention is logs-only."
echo "     They're kept locally as a safety backup."
echo "=============================================="
