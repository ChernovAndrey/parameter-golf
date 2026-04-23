# Remote Machine Instructions (2× H100)

Instructions for running the Fat-Block Seq-Attn + Parallel MLP experiment on a remote 2× H100 pod. Wall-clock budget **40 minutes per variant**.

This is the operational companion to the architecture doc `doc/experiment_fatblock_2026-04-22.md`. Read that first for what the experiment actually is.

---

## Context

We are testing a new architecture that replaces the 4 parallel-residual blocks (old layers 7–10 of PR #1493) with a **single FatBlock** containing:
- **4 sequential attention sublayers** (each refines the previous)
- **1 big shared MLP** (hidden = 6144) running **in parallel** with the attention chain

Plus two optional attention-nonlinearity flags (toggled via env vars):
- `GATED_ATTN` — Qwen NeurIPS 2025 G1 sigmoid gate on SDPA output
- `GLU_V` — SwiGLU-style gated value projection

**Baseline to beat**: PR #1493 no-TTT sliding eval ≈ **1.0827 BPB** (SOTA with TTT is 1.0810; we're running without TTT to stay in budget, so the fair comparison is 1.0827).

The code is at: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/train_gpt.py`

---

## Experiment matrix

| # | Label | Flags | Env |
|---|---|---|---|
| 1 | `vanilla` | fat-block only | `GATED_ATTN=0 GLU_V=0` |
| 2 | `gated_hw` | + Qwen headwise gate | `GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0` |
| 3 | `gated_ew` | + Qwen elementwise gate | `GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0` |
| 4 | `glu_v` | + GLU on V | `GATED_ATTN=0 GLU_V=1` |
| 5 | `both` | headwise gate + GLU-V | `GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1` |

All 5 share `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144`.

---

## Pre-flight checks

Run these **before** burning training compute.

### 1. GPU check

```bash
nvidia-smi
# Expect: 2× H100 (SXM or PCIe, 80 GB each)
# Verify: both GPUs visible, no other processes using memory
```

### 2. Dependencies

```bash
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
from flash_attn_interface import flash_attn_func; print('FlashAttn3: OK')
import sentencepiece; print('SentencePiece: OK')
import brotli; print('Brotli: OK')
import numpy; print(f'NumPy: {numpy.__version__}')
"
```

All must pass. PyTorch must be **2.11+ with CUDA 13** (SOTA's validated combo). If anything fails:

```bash
# PyTorch + CUDA 13
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Flash Attention 3 (Hopper-only)
pip install --no-cache-dir \
  https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl

# Rest
pip install brotli sentencepiece numpy
```

### 3. Data check

Our trainer uses the **SP8192** variant (vocab = 8192), not SP1024. Kevin Clark's HF repo hosts it; the default `willdepueoai/parameter-golf` repo does NOT.

```bash
ls ./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l
# Expect: 128 (or at least 80 — we specify --train-shards 128 below)

ls ./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin 2>/dev/null | wc -l
# Expect: >= 1 validation shard

ls ./data/tokenizers/fineweb_8192_bpe.model
# Expect: ~100 KB tokenizer file
```

If any check fails, download the SP8192 data (takes ~5 min on a typical pod):

```bash
cd /workspace/parameter-golf    # or wherever the repo is cloned
rm -f data/manifest.json        # drop any stale manifest from a prior variant
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

**Do NOT** use `--variant sp1024` — that's the old 1024-vocab data and would fail with `VOCAB_SIZE=8192 does not match tokenizer vocab_size=1024` on launch.

### 4. Working directory

```bash
cd /workspace/parameter-golf    # repo root for data paths to resolve
pwd
```

All subsequent commands in this doc assume you're in the repo root.

---

## Phase 1: Syntax + smoke (no GPU, ~5 s)

Confirm the new trainer parses and the model instantiates with correct shapes **before** loading any data.

```bash
python3 -m py_compile records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/train_gpt.py && echo "✓ syntax OK"
```

```bash
python3 -c "
import os, sys, types
# Stub out FA3 (not needed for CPU-only check)
sys.modules['flash_attn_interface'] = types.ModuleType('flash_attn_interface')
sys.modules['flash_attn_interface'].flash_attn_func = lambda *a, **k: None
os.environ.update(WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                  FAT_BLOCK_ENABLED='1', FAT_BLOCK_MLP_HIDDEN='6144')
sys.path.insert(0, 'records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP')
import train_gpt
m = train_gpt.GPT(train_gpt.Hyperparameters())
nparams = sum(p.numel() for p in m.parameters())
print(f'params:           {nparams:,} ({nparams/1e6:.2f}M)')
print(f'physical blocks:  {len(m.blocks)} (7 Block + 1 FatBlock)')
print(f'encoder_indices:  {m.encoder_indices}')
print(f'decoder_indices:  {m.decoder_indices}')
fat = m.blocks[-1]
print(f'FatBlock:         {fat.num_attns} attns, MLP hidden={fat.mlp.fc.out_features}')
"
```

**Expected:**
- ~33.8M params
- 8 physical blocks, last one is FatBlock
- encoder_indices: `[0, 1, 2, 3, 4, 5, 3]`
- decoder_indices: `[4, 5, 3, 4, 5, 6, 7]`
- FatBlock has 4 attns and MLP hidden = 6144

---

## Phase 2: 1-step forward/backward on GPU (~60 s, dominated by FA3 compile)

Single step at a tiny batch to verify no NaN / Inf / shape bug. Uses `MAX_WALLCLOCK_SECONDS=30` to abort quickly if something's stuck.

```bash
cd records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP

env FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 \
    TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=1 \
    MAX_WALLCLOCK_SECONDS=30 TTT_ENABLED=0 SLIDING_WINDOW_ENABLED=0 \
    SEED=42 DATA_DIR=../../../data \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/smoke.log
```

**Check:**
- [ ] No `nan`, `inf` in loss
- [ ] `model_params:` line shows ~33.8M
- [ ] `warmup_step: 1/1` logged
- [ ] Artifact bytes < 16,000,000 at serialization
- [ ] Process exits cleanly

If this passes, the architecture + data pipeline are verified. Move to the full variants.

---

## Phase 3: Run the 5 variants (5 × ~40 min = ~3h20m)

**Training budget per run**: `MAX_WALLCLOCK_SECONDS=2400` → 40 min training (cap is training-only; eval runs uncapped after). Post-training eval (GPTQ + quantized + sliding-window) adds ~9 min on 2 H100, so expect ~50 min of pod time per variant.

Run each variant and `tee` its log:

```bash
cd records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP
mkdir -p logs

COMMON=(
    FAT_BLOCK_ENABLED=1
    FAT_BLOCK_NUM_ATTNS=4
    FAT_BLOCK_MLP_HIDDEN=6144
    MAX_WALLCLOCK_SECONDS=2400
    TTT_ENABLED=0
    DATA_DIR=../../../data
)

# 1. vanilla fat block
env "${COMMON[@]}" SEED=42 GATED_ATTN=0 GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/vanilla_s42.log

# 2. gated attention, headwise
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/gated_hw_s42.log

# 3. gated attention, elementwise
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/gated_ew_s42.log

# 4. GLU-V only
env "${COMMON[@]}" SEED=42 GATED_ATTN=0 GLU_V=1 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/glu_v_s42.log

# 5. both (headwise gate + GLU-V)
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/both_s42.log
```

**Per-variant checks** (repeat for each log file):
- [ ] Both GPUs utilized (`nvidia-smi` during training)
- [ ] `model_params:` at start matches expected for the variant (±0.5M, see below)
- [ ] Training reaches step count close to expected (~3000–3500 steps)
- [ ] No NaN/Inf mid-training
- [ ] `layer_loop:enabled` message appears at step ~1225 (35% of effective training)
- [ ] `Total submission size quantized+brotli:` < 16,000,000 bytes
- [ ] `quantized_sliding_window val_bpb:` logged at the end — **record this number**

**Expected total params per variant** (from pre-run smoke test):

| Variant | Total params |
|---|---:|
| vanilla | 33.84M |
| gated_hw | 33.86M |
| gated_ew | 34.89M |
| glu_v | 34.37M |
| both | 34.38M |

---

## Phase 4: Read & compare results

Extract the final BPB from each log:

```bash
for f in logs/vanilla_s42 logs/gated_hw_s42 logs/gated_ew_s42 logs/glu_v_s42 logs/both_s42; do
  bpb=$(grep 'quantized_sliding_window val_bpb' ${f}.log | tail -1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/')
  artifact=$(grep 'Total submission size' ${f}.log | tail -1 | sed -E 's/.* ([0-9]+) bytes.*/\1/')
  echo "${f##*/}   BPB=$bpb   artifact=$artifact bytes"
done
```

**Baseline**: PR #1493 no-TTT ≈ **1.0827 BPB**. Expected variant ordering (from the papers):

- `vanilla` within ±0.01 of 1.0827 → architecture works
- `gated_hw` ≤ `vanilla` (Qwen paper: monotone improvement)
- `gated_ew` ≤ `gated_hw` (elementwise marginally better)
- `glu_v` ≤ `vanilla` (GLU Attention paper: monotone improvement)
- `both` ≤ min(`gated_hw`, `glu_v`) (stackable per Qwen paper)

**Pick a winner** (lowest BPB) for Phase 5 seed confirmation.

---

## Phase 5: 3-seed confirmation on the winner (2 × ~40 min = ~1h20m)

Single-seed deltas < 0.001 BPB aren't distinguishable (SOTA std is 0.0002 across 3 seeds). Re-run the winner at 2 more seeds to confirm the signal clears noise.

Say the winner is `both`:

```bash
WINNER_FLAGS=(GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1)

for SEED in 314 999; do
    env "${COMMON[@]}" "${WINNER_FLAGS[@]}" SEED=$SEED \
        torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/winner_s${SEED}.log
    echo "=== seed $SEED done ==="
done
```

Compute 3-seed mean and std:

```bash
python3 <<'EOF'
import re, glob, statistics
bpbs = []
for f in sorted(glob.glob('logs/*winner*.log') + glob.glob('logs/*_s42.log')):
    with open(f) as fh: text = fh.read()
    m = re.search(r'quantized_sliding_window val_bpb:([0-9.]+)', text)
    if m:
        bpb = float(m.group(1))
        bpbs.append(bpb)
        print(f'{f:<50} {bpb:.6f}')
if len(bpbs) >= 3:
    print(f'\nmean: {statistics.mean(bpbs):.6f}')
    print(f'std:  {statistics.stdev(bpbs):.6f}')
    print(f'vs baseline (1.0827): delta = {statistics.mean(bpbs) - 1.0827:+.6f}')
EOF
```

**Signal criteria:**
- 3-seed mean beats 1.0827 by at least 3× std (SOTA std is 0.0002, so ≥ 0.0006 BPB improvement)
- Artifact stays under 16 MB on all 3 seeds

---

## Troubleshooting

### `VOCAB_SIZE=8192 does not match tokenizer vocab_size=1024`
You downloaded SP1024. Re-run data prep with `--variant sp8192` and `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`.

### `No files found for pattern: ./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin`
Either `DATA_DIR` isn't pointing at the right folder (should be `../../../data` when running from the experiment folder), or data prep failed. Re-check Phase 0 step 3.

### NaN/Inf loss in first 100 steps
Fat-block may be unstable at init. Try `FAT_BLOCK_SKIP_MODE=drop` (disables the skip-aggregation gate).
If that also fails, reduce `MATRIX_LR` from 0.022 to 0.015.

### OOM (Out of Memory) on 2× H100
The SOTA was tuned for 8× H100 with each GPU seeing 48 sequences/step. On 2 GPUs with `grad_accum_steps=4`, per-GPU micro-batch is still 48, so memory footprint should be identical. If OOM:
- Reduce `TRAIN_BATCH_TOKENS=393216` (halves effective batch; compute stays same, but noisier gradients)
- Check `peak memory allocated` in logs — should be < 75 GB on H100 80 GB

### Artifact > 16 MB after GPTQ
Tight margin on `both` variant. Try:
- `FAT_BLOCK_MLP_HIDDEN=5120` (saves ~0.5 MB)
- Or `MUON_WD=0.10` (tighter weight distribution → better brotli)
- Confirm `COMPRESSOR=brotli` (not lzma) in the log

### Step time >> 520 ms/step
Check `nvidia-smi` for contention. Wider MLP (6144) + GLU-V may push step time up. If > 700 ms/step, the 40-min training budget still produces a valid result (just fewer steps), so no action needed unless the pod has a hard total-time limit.

### `layer_loop:enabled` never prints
Recurrence activation requires `frac >= 0.35`, where `frac = elapsed_ms / max_wallclock_ms`. If training aborts before 35% of budget (e.g. via wall-clock reached_cap), the loop never turns on. Check that training isn't being cut short by something other than `MAX_WALLCLOCK_SECONDS`.

### Sliding-window eval runs too long (> 600 s on 2 GPUs)
`EVAL_STRIDE=64` is SOTA's default. For a faster but coarser eval, bump to `EVAL_STRIDE=128` — halves eval time with ~0.005 BPB noise penalty. Or skip sliding entirely with `SLIDING_WINDOW_ENABLED=0` and rely on `quantized val_bpb` (non-sliding, ~50 s).

### Flash Attention 3 kernel hangs on first step
Compilation takes 30–60 s the first time. Patience. If it truly hangs (> 3 min), verify your torch and flash_attn_3 versions are compatible (torch 2.11 + cu130 is the combo SOTA validated).

---

## Post-experiment

**Save logs to the experiment folder** (for future reference, not submission):

```bash
# Logs already in records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/
ls records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/
```

**Report back** with:
- 5-variant BPB table (single seed at 42)
- Winner's 3-seed mean + std
- Any anomalies / aborted runs
- Actual peak memory + step time per variant
- Which pitfalls (if any) from the troubleshooting section above were hit

---

## Summary of key differences from old `remote_instructions.md`

| | Old (shared-MLP experiment) | New (fat-block experiment) |
|---|---|---|
| Hardware | 8× H100 | **2× H100** |
| Wall-clock budget | 600 s (competition) | **40 min per run** |
| Data variant | SP1024 (`--variant sp1024`) | **SP8192 (`--variant sp8192`)** |
| Data source | default repo | `kevclark/parameter-golf` |
| Baseline to beat | 1.1147 BPB (PR #1019) | **1.0827 BPB (PR #1493 no-TTT)** |
| Architecture folder | `2026-04-02_SharedMLP_Rank100_16L` | `2026-04-22_FatBlock_SeqAttn_ParMLP` |
| Number of runs | 3 seeds of one config | **5 variants + 2 seed-repeats on winner** |
