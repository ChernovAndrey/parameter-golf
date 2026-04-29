#!/bin/bash
# Run this ON THE POD (web terminal is fine — no SSH needed) after the 3-seed
# run completes. Pushes the submission-relevant files back to GitHub via git;
# you then `git pull` locally to get them.
#
# What gets pushed:
#   - train_seed{42,314,999}.log              (small text — REQUIRED for submission)
#   - seed{42,314,999}_final_model.int6.ptz   (~16 MB each — the actual artifact blobs)
#   - README.md, submission.json, run.sh, train_gpt.py (the rest of the folder)
#
# What does NOT get pushed automatically:
#   - seed{42,314,999}_final_model.pt   (~133 MB each — exceeds GitHub's 100 MB
#                                        per-file limit. Options below.)
#
# Usage:
#   ./push_from_pod.sh                          # commit + push logs + .ptz to current branch
#   ./push_from_pod.sh --include-pt             # ALSO try to push .pt files (will fail on GitHub
#                                                if any single file > 100 MB; safer to use
#                                                an external host — see notes below)
#   ./push_from_pod.sh --upload-pt-to-fileio    # upload .pt files to file.io and print URLs
#                                                (no git, just URLs you paste/save)
#   ./push_from_pod.sh --serve-pt               # start `python3 -m http.server` so you can
#                                                download via RunPod's port forwarding

set -e
set -o pipefail

INCLUDE_PT=0
UPLOAD_PT=0
SERVE_PT=0
for arg in "$@"; do
    case "$arg" in
        --include-pt) INCLUDE_PT=1 ;;
        --upload-pt-to-fileio) UPLOAD_PT=1 ;;
        --serve-pt) SERVE_PT=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Sanity: must be in a git repo
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "ERROR: not inside a git repo. cd into the cloned parameter-golf directory first." >&2
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[push] branch: $CURRENT_BRANCH"

# ============================================================
# 1) Show what we have locally and what's missing.
# ============================================================
echo "[push] inventory:"
for s in 42 314 999; do
    log="train_seed${s}.log"
    pt="seed${s}_final_model.pt"
    ptz="seed${s}_final_model.int6.ptz"
    printf "  seed %-3s: " "$s"
    [ -f "$log" ] && printf "log(%s) " "$(du -h "$log" | cut -f1)" || printf "log(MISSING) "
    [ -f "$ptz" ] && printf "ptz(%s) " "$(du -h "$ptz" | cut -f1)" || printf "ptz(MISSING) "
    [ -f "$pt" ]  && printf "pt(%s)\n" "$(du -h "$pt" | cut -f1)" || printf "pt(MISSING)\n"
done

# ============================================================
# 2) --serve-pt mode: start an HTTP server in this folder.
#    Use RunPod's "Expose HTTP Port" feature to make port 8000
#    visible, then download via browser.
# ============================================================
if [ "$SERVE_PT" = "1" ]; then
    echo
    echo "[serve] starting python3 -m http.server 8000 in $(pwd)"
    echo "[serve] On RunPod: open the pod's settings, expose HTTP port 8000,"
    echo "[serve] then visit https://<pod-id>-8000.proxy.runpod.net/"
    echo "[serve] You'll see seed*_final_model.pt as clickable links — right-click → save as."
    echo "[serve] Ctrl+C this when done."
    exec python3 -m http.server 8000
fi

# ============================================================
# 3) --upload-pt-to-fileio mode: push large .pt files to file.io
#    (free, expires after first download — ideal for one-shot transfer).
# ============================================================
if [ "$UPLOAD_PT" = "1" ]; then
    echo
    echo "[upload] uploading seed*_final_model.pt to file.io (one-shot links)..."
    for s in 42 314 999; do
        f="seed${s}_final_model.pt"
        if [ -f "$f" ]; then
            echo "[upload] $f ($(du -h "$f" | cut -f1)) → file.io ..."
            url=$(curl -s -F "file=@$f" https://file.io/?expires=1d | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('link', d))")
            echo "[upload]   $f  →  $url   (expires in 1 day, single download)"
        fi
    done
    echo "[upload] save those URLs — each link works for ONE download then dies."
    exit 0
fi

# ============================================================
# 4) Default mode: git add + commit + push (logs + .ptz + folder files).
# ============================================================
git add train_seed*.log 2>/dev/null || true
git add seed*_final_model.int6.ptz 2>/dev/null || true
git add README.md submission.json run.sh pull_from_runpod.sh push_from_pod.sh train_gpt.py 2>/dev/null || true

if [ "$INCLUDE_PT" = "1" ]; then
    echo "[push] including .pt files (133 MB each — will be rejected by GitHub if any file > 100 MB)"
    git add seed*_final_model.pt 2>/dev/null || true
fi

# Show staged size
echo
echo "[push] staged for commit:"
git diff --cached --stat || true
TOTAL_STAGED_KB=$(git diff --cached --numstat | awk '{print $1+$2}' | wc -l)

if ! git diff --cached --quiet; then
    echo
    read -p "[push] proceed with commit + push to origin/$CURRENT_BRANCH? [y/N] " ans
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
        echo "[push] aborted (changes are staged but not committed)."
        exit 0
    fi
    git commit -m "submission: 3-seed run results (parallel-zone gated_ew + MLP 3.25x + Legal TTT)"
    git push origin "$CURRENT_BRANCH"
    echo
    echo "[push] DONE. On your local machine:"
    echo "    cd <repo>"
    echo "    git pull origin $CURRENT_BRANCH"
else
    echo "[push] nothing staged — all files are already up to date in git, or none exist."
fi

# ============================================================
# 5) Reminder about .pt files (safety backup, not in submission).
# ============================================================
echo
if [ "$INCLUDE_PT" != "1" ]; then
    pt_count=$(ls seed*_final_model.pt 2>/dev/null | wc -l | tr -d ' ')
    if [ "$pt_count" != "0" ]; then
        echo "[note] $pt_count .pt file(s) NOT pushed (too big for git). They live on the pod at:"
        echo "    $(pwd)/seed*_final_model.pt"
        echo "    To grab them anyway, use:"
        echo "      ./push_from_pod.sh --upload-pt-to-fileio    (one-shot links via file.io)"
        echo "      ./push_from_pod.sh --serve-pt               (HTTP server + RunPod port forward)"
        echo "    Or just leave them on the pod — they're recovery backups, not part of the submission."
    fi
fi
