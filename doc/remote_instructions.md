# Remote Machine Instructions (8xH100)

Instructions for Claude Code running on the remote GPU machine to validate
and run the shared MLP + low-rank adapter submission.

---

## Context

We are testing a modified Parameter Golf submission that uses:
- **16 layers** (up from 11 in SOTA)
- **3 shared MLP bases** + **rank-100 per-layer adapters** (replaces per-layer MLP banks)
- Everything else identical to current SOTA (PR #1019)

The goal is to beat 1.1147 BPB by at least 0.005 nats (~0.003 BPB).

The code is at: `records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py`

---

## Pre-Flight Checks

Before running ANY training, verify the environment:

### 1. GPU check
```bash
nvidia-smi
# Expect: 8x H100 SXM 80GB
# Verify: all 8 GPUs visible, no processes using memory
```

### 2. Dependencies
```bash
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
from flash_attn_interface import flash_attn_func; print('FlashAttn3: OK')
import sentencepiece; print('SentencePiece: OK')
import zstandard; print('Zstandard: OK')
import numpy; print(f'NumPy: {numpy.__version__}')
"
# All must pass. If flash_attn_interface fails:
# pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

### 3. Data check
```bash
ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l
# Expect: 80 training shards

ls ./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l
# Expect: validation shards present

ls ./data/tokenizers/fineweb_1024_bpe.model
# Expect: tokenizer file exists
```

If data missing:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 4. Working directory
```bash
cd /workspace/parameter-golf  # or wherever the repo is cloned
pwd
# Must be at repo root for data paths to resolve
```

---

## Phase 1: Smoke Test (backward compatibility)

Verify the modified code works identically to SOTA when adapters are disabled:

```bash
cd /workspace/parameter-golf

ADAPTER_RANK=0 NUM_SHARED_MLPS=0 NUM_LAYERS=11 \
XSA_LAST_N=11 VE_LAYERS=9,10 WARMDOWN_ITERS=4000 \
MAX_WALLCLOCK_SECONDS=60 ITERATIONS=200 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py
```

**Check:**
- [ ] No errors
- [ ] Train loss decreases
- [ ] `model_params` matches SOTA (~27M)

---

## Phase 2: Architecture Test (1 GPU, 3 min)

```bash
NUM_LAYERS=16 ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
VE_LAYERS=14,15 XSA_LAST_N=16 \
WARMDOWN_ITERS=2800 \
MAX_WALLCLOCK_SECONDS=180 ITERATIONS=9000 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py
```

**Check:**
- [ ] No errors / OOM
- [ ] Train loss decreases (should reach ~2.0-2.1 in 3 min on 1 GPU)
- [ ] `model_params` shows ~24.7M (less than SOTA's 27M despite more layers)
- [ ] Step time ~400-600ms on 1 GPU (will be ~125ms on 8 GPUs)
- [ ] Logs show: `XSA:last_16`, correct layer count, adapter info

---

## Phase 3: Full Training + Quantization (1 GPU, ~20 min)

```bash
NUM_LAYERS=16 ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
VE_LAYERS=14,15 XSA_LAST_N=16 \
WARMDOWN_ITERS=2800 TARGET_MB=15.7 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=9000 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py
```

**Check:**
- [ ] Training completes within wallclock
- [ ] GPTQ quantization runs (shared bases + adapters)
- [ ] `Serialized model int6+lzma: XXXXX bytes` — must be < 16,000,000
- [ ] `Total submission size int6+lzma: XXXXX bytes` — must be < 16,000,000
- [ ] `final_int6_roundtrip val_bpb` shows a reasonable number
- [ ] Quantization gap (pre-quant BPB - post-quant BPB) < 0.01
- [ ] No NaN or inf in loss

---

## Phase 4: Multi-GPU Test (8 GPU, single seed)

```bash
NUM_LAYERS=16 ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
VE_LAYERS=14,15 XSA_LAST_N=16 \
WARMDOWN_ITERS=2800 TARGET_MB=15.7 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=314 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py \
  2>&1 | tee train_seed314.log
```

**Check:**
- [ ] All 8 GPUs utilized (check nvidia-smi during training)
- [ ] Step time ~120-135ms (log: `step_avg`)
- [ ] Training completes in < 600s
- [ ] Steps completed: ~4500-5000
- [ ] Artifact size < 16,000,000 bytes
- [ ] `final_int6_sliding_window val_bpb` — record this number
- [ ] Eval completes in < 600s (total eval time in logs)

**If BPB is not competitive** (> 1.12):
- Try increasing warmdown: `WARMDOWN_ITERS=3200`
- Try lower adapter rank: `ADAPTER_RANK=64` (allows more steps per second)
- Check if quantization gap is the issue (compare pre/post quant BPB)

---

## Phase 5: Final Evaluation (8 GPU, 3 seeds)

Run all 3 seeds:

```bash
for SEED in 314 42 999; do
  NUM_LAYERS=16 ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
  VE_LAYERS=14,15 XSA_LAST_N=16 \
  WARMDOWN_ITERS=2800 TARGET_MB=15.7 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py \
    2>&1 | tee train_seed${SEED}.log
  echo "=== Seed $SEED complete ==="
done
```

**After all 3 seeds complete, verify:**

```python
import numpy as np
from scipy import stats

# Our scores (read from logs: final_int6_sliding_window_exact val_bpb)
ours = np.array([SEED_314_BPB, SEED_42_BPB, SEED_999_BPB])

# SOTA scores (from PR #1019)
sota = np.array([1.11508120, 1.11437394, 1.11475014])

print(f"Our mean:  {ours.mean():.8f} (std {ours.std():.6f})")
print(f"SOTA mean: {sota.mean():.8f}")
print(f"Delta BPB: {ours.mean() - sota.mean():.6f}")

# Convert to nats for the 0.005 threshold
delta_nats = (ours.mean() - sota.mean()) * np.log(2) / (1.0)  # approximate
# Actually: val_loss is already in nats. BPB = bits_per_token * tokens_per_byte
# The 0.005 nat threshold is on val_loss, not BPB.
# Read val_loss from logs instead for exact nat comparison.

t_stat, p_value = stats.ttest_ind(ours, sota, equal_var=False)
print(f"Welch's t: {t_stat:.4f}, p-value: {p_value:.6f}")
print(f"Passes p<0.01: {p_value < 0.01}")
```

**Final checklist:**
- [ ] All 3 artifacts < 16,000,000 bytes
- [ ] All 3 training times < 600s
- [ ] All 3 eval times < 600s
- [ ] Mean BPB < 1.1147 (beats SOTA)
- [ ] Improvement ≥ 0.005 nats (on val_loss, not BPB)
- [ ] p < 0.01 on Welch's t-test

---

## Submission Preparation

### Copy log files to records folder
```bash
cp train_seed314.log records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/
cp train_seed42.log records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/
cp train_seed999.log records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/
```

### Create submission.json
```json
{
  "author": "YOUR_NAME",
  "github_id": "YOUR_GITHUB",
  "name": "Shared MLP Bases + Rank-100 Adapters (16L)",
  "val_bpb": MEAN_BPB,
  "val_loss": MEAN_VAL_LOSS,
  "num_seeds": 3,
  "artifact_bytes": MAX_ARTIFACT_BYTES,
  "train_time_seconds": 600,
  "hardware": "8xH100 SXM 80GB"
}
```

### Verify PR scope
```bash
git add records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/
git diff --cached --stat
# Should show ONLY files in the new folder:
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/README.md
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/submission.json
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_gpt.py
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_seed314.log
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_seed42.log
#   records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/train_seed999.log
# NO other files should be modified
```

---

## Troubleshooting

### OOM (Out of Memory)
- Reduce `TRAIN_BATCH_TOKENS` (default 786432, try 524288)
- Check `peak memory allocated` in logs — should be < 70 GB

### Step time too slow (> 140ms)
- Check GPU utilization with `nvidia-smi`
- Ensure no other processes on GPUs
- Try reducing BIGRAM_VOCAB_SIZE to 2048 (minor BPB impact, saves compute)

### Artifact too large (> 16MB)
- Reduce TARGET_MB to 15.5
- Try ADAPTER_RANK=96 (saves ~200 KB)
- Check if selective pruning activated in logs

### Quantization gap too large (> 0.005 BPB)
- This means the shared base + adapter separate quantization loses too much
- Add Noisy QAT for shared bases (see doc/shared_mlp_proposal.md section on Noisy QAT)
- Or try ADAPTER_RANK=64 (fewer params to quantize)

### BPB not competitive (> 1.115)
- Increase warmdown: WARMDOWN_ITERS=3200 or 3500
- Adjust learning rate: MATRIX_LR=0.03 or 0.02
- Try 2 shared MLPs instead of 3: NUM_SHARED_MLPS=2 (allows rank-100 at 16L with more headroom)
- Check train loss curve — if still decreasing at end, need more steps (slower step time is the constraint)