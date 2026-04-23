# Fat-Block Experiment — 4 Sequential Attentions + 1 Parallel Fat MLP

**Status:** experimental, not a leaderboard submission.
**Target hardware:** 2× H100 (SXM or PCIe) for ~40 minutes per run.
**Base:** PR #1493 (`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/`), 1.0810 BPB SOTA.

---

## Architecture

Replaces the 4 parallel-residual blocks (old layers 7–10) with a single **FatBlock** containing 4 sequential attentions + 1 big MLP running in parallel:

```python
# FatBlock.forward(x_in)
# Attention chain (each attention reads the updated hidden state)
z = x_in
for i in range(4):
    z = z + attn_scales[i] * attn_i(norm(z))

# Big MLP runs in parallel on x_in (not on z)
mlp_out = big_MLP(norm(x_in))

# Merge both paths
x_out = z + mlp_scale * mlp_out
```

- **4 sequential attention sublayers** — each one refines the previous one's output. This preserves `attn_to_attn` depth, which PR #1204's learned-routing data showed is the strongest information-flow signal in deep layers.
- **1 big shared MLP** (hidden=6144 by default, vs 2048 per-layer in original SOTA) running **in parallel** with the attention chain. It reads the fat-block input `x_in`, not the refined `z`, so attn and MLP are decoupled.
- **`attn_scales`** is a learnable `[num_attns=4, dim=512]` tensor; **`mlp_scale`** is a learnable `[dim=512]` vector. Both init to all-ones. Inherited convention from the SOTA `Block`: per-feature learnable gains on attn/MLP contributions before they enter the residual stream.
- **Depth trade**: model goes from 17 virtual evals → 14, loses 3 MLP duplications, gains one fat MLP that compresses the 4 old MLPs into one.

Layers 0–6 are unchanged Block instances (sequential attn→MLP). Depth recurrence on layers 3/4/5 still runs (each visited 3×). SP8192 vocab, Partial RoPE 16/64, GQA-4, LeakyReLU(0.5)², MuonEq-R, SDClip GPTQ, Brotli-11 — all inherited from PR #1493.

### Encoder / decoder recurrence sequence — what the log's unsorted indices mean

When you see this in the training log:

```
layer_loop:enabled step:~2200 frac:0.350
  encoder:[0, 1, 2, 3, 4, 5, 3]
  decoder:[4, 5, 3, 4, 5, 6, 7]
```

these are the **physical block indices visited in order** during the forward pass. The model has 8 physical blocks (7 regular Blocks at indices 0–6, plus the FatBlock at index 7). With `NUM_LOOPS=2 LOOP_START=3 LOOP_END=5`, layers 3, 4, 5 are each visited **3 times per step**; other blocks are visited once.

Visits per physical block (encoder + decoder):

| Block | Encoder | Decoder | Total |
|---:|---:|---:|---:|
| 0 | 1 | 0 | **1** |
| 1 | 1 | 0 | **1** |
| 2 | 1 | 0 | **1** |
| 3 | 2 | 1 | **3** ← looped |
| 4 | 1 | 2 | **3** ← looped |
| 5 | 1 | 2 | **3** ← looped |
| 6 | 0 | 1 | **1** |
| 7 (FatBlock) | 0 | 1 | **1** |
| | | | **14 virtual evals** |

**How the sequence is built** (from `GPT.__init__`):

```python
loop_seg = [3, 4, 5]                       # layers that get looped
all_indices = [0, 1, 2]                    # pre-loop
for _ in range(num_loops + 1):             # 3 iterations
    all_indices.extend([3, 4, 5])
# all_indices == [0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]
all_indices.extend([6, 7])                 # post-loop (last reg block + FatBlock)
# all_indices == [0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7]   (14 items)

num_enc = 14 // 2 = 7
encoder_indices = all_indices[:7]          # [0, 1, 2, 3, 4, 5, 3]
decoder_indices = all_indices[7:]          # [4, 5, 3, 4, 5, 6, 7]
```

The loop straddles the encoder/decoder split on purpose — this makes the U-Net skip structure work cleanly. Encoder visits save their outputs into a `skips` list; decoder visits pop them in reverse order and add them back with learnable `skip_weights[i]` and sigmoid `skip_gates[i]`. With 7 encoder and 7 decoder positions, `num_skip_weights = 7` and each decoder position receives exactly one skip.

Visually:

```
                        ENCODER (saves 7 outputs)          |          DECODER (reads 7 skips in reverse)
visit  →  block    0    1    2    3    4    5    3        |    4    5    3    4    5    6    7
                   ↓    ↓    ↓    ↓    ↓    ↓    ↓        |    ↓    ↓    ↓    ↓    ↓    ↓    ↓
state             h0 → h1 → h2 → h3 → h4 → h5 → h6  (=s0..s6)     +s6  +s5  +s4  +s3  +s2  +s1  +s0
                                                           |     (each skip-gated by sigmoid(skip_gates[i]))
```

Each `+sk` means "blend in the corresponding saved encoder output" via `x = lerp(skip, x, sigmoid(skip_gates[i]))`. That's the U-Net shortcut, preserved from SOTA.

**Non-looping (warmup) version** — during the first 20 warmup steps, `looping_active=False` and the sequence is the plain range:

```
encoder: [0, 1, 2, 3]
decoder: [4, 5, 6, 7]
```

8 virtual evals, each block visited once. The looping sequence activates when `elapsed_ms / MAX_WALLCLOCK_SECONDS ≥ ENABLE_LOOPING_AT` (default 0.35 → ~14 min into a 40-min run).

### Optional attention nonlinearity variants (apply INSIDE FatBlock only)

Two independent flags that can be combined:

- **`GATED_ATTN=1` / `GATED_ATTN_MODE={headwise,elementwise}`** — Qwen G1 gated attention (NeurIPS 2025 Best Paper, [arXiv 2505.06708](https://arxiv.org/abs/2505.06708)). Sigmoid gate on SDPA output before `W_o`:
  ```
  y = FlashAttn3(Q, K, V)
  y = sigmoid(W_g @ x) * y       # headwise: W_g is [dim -> n_heads]; elementwise: [dim -> dim]
  return W_o @ y
  ```
  Deployed in Qwen3-Next-80B; proven stackable with other mods.

- **`GLU_V=1`** — GLU on value projection ([arXiv 2507.00022](https://arxiv.org/abs/2507.00022), July 2025). Replaces the V projection with SwiGLU-style gating:
  ```
  V = silu(W_v1 @ x) * (W_v2 @ x)   # element-wise, before attention mixes V linearly
  ```
  FA3 takes whatever V you hand it — no kernel changes.

---

## Experiment matrix (5 runs × ~40 min each)

| # | Label | Flags | Notes |
|---|---|---|---|
| 1 | `vanilla` | `GATED_ATTN=0 GLU_V=0` | Baseline fat block, no nonlinearity additions |
| 2 | `gated_hw` | `GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0` | Cheap gate (Qwen paper default) |
| 3 | `gated_ew` | `GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0` | Full-width gate (~60× more gate params) |
| 4 | `glu_v` | `GATED_ATTN=0 GLU_V=1` | GLU on V alone |
| 5 | `both` | `GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1` | Stacked; headwise gate keeps it cheap |

**Design**: same `FAT_BLOCK_MLP_HIDDEN=6144` across all 5 runs. Differences reflect only the attention modification. All fit with ~1.0–1.4 MB of headroom under the 16 MB artifact cap.

**Seed plan**: run each variant at `SEED=42` first (~40 min each = 3h20m total). Then repeat the winner at `SEED=314` and `SEED=999` to confirm (~80 min more). Grand total ~4h40m.

---

## Estimated artifact budgets

From the CPU-only smoke test (`python3 smoke_test.py`, compression ratios from `doc/sota_profile.md`):

| Variant | Total params | Est. artifact | Est. headroom |
|---|---:|---:|---:|
| vanilla | 33.84M | 14.58 MB | +1.39 MB |
| gated_hw | 33.86M | 14.60 MB | +1.37 MB |
| gated_ew | 34.89M | 14.94 MB | +1.03 MB |
| glu_v | 34.37M | 14.76 MB | +1.21 MB |
| both | 34.38M | 14.78 MB | +1.19 MB |

(Add ~58 KB per variant for the un-wrapped code size vs SOTA's LZMA-wrapped code. All still comfortably under 16 MB.)

---

## Prerequisites

```bash
# PyTorch 2.11 + CUDA 13
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Flash Attention 3 (Hopper-only)
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"

# Brotli, SentencePiece, NumPy
pip install brotli sentencepiece numpy
```

## Data prep (one-time, ~5 min)

From the repo root:

```bash
cd parameter-golf
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

This downloads pre-tokenized FineWeb into `data/datasets/fineweb10B_sp8192/` and the SP8192 tokenizer into `data/tokenizers/fineweb_8192_bpe.model`.

---

## Launch commands — 5 variants at SEED=42

**Hardware**: 2× H100. **Training time**: 40 min (`MAX_WALLCLOCK_SECONDS=2400`). **Eval runs after training** and takes ~9 min more, so expect ~50 min of pod time per variant.

### Key settings

- **`MAX_WALLCLOCK_SECONDS=2400`** — caps **training only** (40 min). Eval happens uncapped after the cap hits.
- **`--nproc_per_node=2`** — use only 2 GPUs. The SOTA trainer auto-adjusts `grad_accum_steps = 8 // world_size = 4`, preserving the 786K-token effective batch.
- **No `ITERATIONS`** needed — with wall-clock set, LR warmdown (`WARMDOWN_FRAC=0.72`) and recurrence activation (`ENABLE_LOOPING_AT=0.35`) are both driven by `elapsed_ms / MAX_WALLCLOCK_SECONDS`. The default `ITERATIONS=20000` is a safety cap far beyond what 40 min of compute reaches.
- **`TTT_ENABLED=0`** — skip test-time training. TTT eval alone would add ~20 min on 2 H100 and isn't needed for architecture comparison.

### What runs after training (not capped)

~9 min total, automatic:
- Post-training `eval_val` on the full-precision model (~40 s)
- GPTQ Hessian collection + quantization + serialization (~40 s)
- Dequantize + `eval_val` on the quantized model (~50 s)
- Sliding-window eval — this is the `quantized_sliding_window val_bpb` we compare against SOTA (~550 s on 2 H100, scales ~4× from SOTA's ~130 s on 8 H100)

### Commands

All commands assume you `cd` into the experiment folder first. `DATA_DIR=../../../data` walks back to `parameter-golf/data/`.

```bash
cd records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP
mkdir -p logs

# --- Shared settings (constant across variants) ---
# MAX_WALLCLOCK_SECONDS=2400 → 40 min training only; eval runs after for ~9 min more
COMMON=(
    FAT_BLOCK_ENABLED=1
    FAT_BLOCK_NUM_ATTNS=4
    FAT_BLOCK_MLP_HIDDEN=6144
    MAX_WALLCLOCK_SECONDS=2400
    TTT_ENABLED=0
    DATA_DIR=../../../data
)

# Variant 1: vanilla fat block
env "${COMMON[@]}" SEED=42 GATED_ATTN=0 GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/vanilla_s42.log

# Variant 2: gated attention, headwise
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/gated_hw_s42.log

# Variant 3: gated attention, elementwise
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/gated_ew_s42.log

# Variant 4: GLU-V only
env "${COMMON[@]}" SEED=42 GATED_ATTN=0 GLU_V=1 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/glu_v_s42.log

# Variant 5: both (headwise gate + GLU-V)
env "${COMMON[@]}" SEED=42 GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1 \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/both_s42.log
```

**Tip:** you can run these 5 variants back-to-back from a single bash script by pasting the block above; each run creates its own log file.

## Seed repeats on the winner

Once you know which variant wins at SEED=42 (call it `WINNER=both`), repeat at two more seeds:

```bash
WINNER_FLAGS=(GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=1)   # adjust to match the winning variant

for SEED in 314 999; do
  env "${COMMON[@]}" "${WINNER_FLAGS[@]}" SEED=$SEED \
      torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | tee logs/winner_s${SEED}.log
done
```

---

## Pre-launch sanity checks

Run these before burning compute:

### 1. Syntax parse (instant)
```bash
python3 -m py_compile records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/train_gpt.py && echo OK
```

### 2. Param count + artifact-budget estimate (seconds, no GPU)
```bash
python3 -c "
import os, sys, types
# Stub FA3 (not installed on CPU)
sys.modules['flash_attn_interface'] = types.ModuleType('flash_attn_interface')
sys.modules['flash_attn_interface'].flash_attn_func = lambda *a,**k: None
os.environ.update(WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
sys.path.insert(0, 'records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP')
import train_gpt
m = train_gpt.GPT(train_gpt.Hyperparameters())
print('params:', sum(p.numel() for p in m.parameters()))
print('encoder:', m.encoder_indices)
print('decoder:', m.decoder_indices)
print('blocks:', [type(b).__name__ for b in m.blocks])
"
```
Expected: ~34M params, 7 Block + 1 FatBlock, encoder `[0,1,2,3,4,5,3]`, decoder `[4,5,3,4,5,6,7]`.

### 3. One-step forward/backward (needs a GPU + FA3, ~30 s)
```bash
env FAT_BLOCK_ENABLED=1 ITERATIONS=1 TRAIN_BATCH_TOKENS=65536 WARMUP_STEPS=1 \
    TTT_ENABLED=0 DATA_DIR=../../../data \
    torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | head -30
```
Expected: no NaN/Inf, finishes with a quantized artifact under 16 MB.

---

## Monitoring during training

Look for these landmarks in the log:

| Step | Expected train_loss | Interpretation |
|---|---|---|
| 10 (warmup) | ~11.2 | Random init, 8192 vocab |
| 500 | ~3.0 | Early learning |
| 1000 | ~2.3 | Past the "memorization" phase |
| 1500 | `layer_loop:enabled` message appears at step ~1225 (35% of 3500) | Recurrence activates |
| 2500 | ~2.0 | Mid-training |
| 3500 (final) | ~1.9 | End of training |
| Final `val_bpb` | ~1.07–1.10 | Compare vs PR #1493 no-TTT baseline of **1.0827** |

**Abort conditions:**
- `train_loss` is `nan` or `inf` at any point → bug in architecture (try `FAT_BLOCK_SKIP_MODE=drop` as fallback)
- `train_loss` plateaus above 3.0 past step 1500 → architecture is saturating early
- Artifact size > 16 MB after GPTQ → SDClip assumption broke (try `MUON_WD=0.10` and re-run)

---

## Interpreting results

The final line to read in the log is:

```
quantized_sliding_window val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms
```

**Baseline to beat**: PR #1493 sliding-only (no TTT) = **1.0827 BPB** (3-seed mean).

**Expected variant ordering** (from the papers):

| Variant | Expected BPB |
|---|---|
| vanilla | within ±0.01 of 1.0827 if fat-block survives; worse than 1.10 means the structure is broken |
| gated_hw | ≤ vanilla (Qwen paper: monotone improvement) |
| gated_ew | ≤ gated_hw (Qwen paper: small elementwise > headwise gain) |
| glu_v | ≤ vanilla (GLU Attention paper: monotone improvement) |
| both | ≤ min(gated_hw, glu_v) (Qwen paper: stackable) |

**Noise floor**: PR #1493 is 0.0002 std across 3 seeds. A 1-seed delta < 0.001 BPB is not distinguishable — that's why the winner needs a 3-seed confirmation.

If `both > gated_hw` or `both > glu_v` → the two mechanisms interact negatively. Report and do not combine in future work.

---

## Pitfalls & mitigations (reference)

| Pitfall | Mitigation |
|---|---|
| `train_loss` NaN early (step < 100) | Try `FAT_BLOCK_SKIP_MODE=drop` — the aggregate skip-gate may be injecting pathological values at init |
| `Artifact > 16 MB` after GPTQ | Reduce `FAT_BLOCK_MLP_HIDDEN` to 5120 (saves ~0.5 MB); or raise `MUON_WD` to 0.10 (tighter weight distribution) |
| FA3 kernel compile hangs first step | Normal — first step can take 30–60 s. Not counted against `MAX_WALLCLOCK_SECONDS` (training timer starts AFTER warmup). Just wait it out. |
| OOM on 2× H100 80GB | Lower `TRAIN_BATCH_TOKENS` to 393,216 (effective batch halved; will auto-adjust grad_accum) |
| Step time >> 520 ms/step | Wider MLP or GLU-V may push latency up. Use `ITERATIONS=3000` instead of 3500 to ensure convergence before wall clock |
| Recurrence never activates | Check log for `layer_loop:enabled step:N frac:0.35` — must appear by step ~1225 (35% of 3500). If missing, `ENABLE_LOOPING_AT` isn't being reached |
| `fat_block` exists but SOTA-baseline comparison run shows 11 blocks | Set `FAT_BLOCK_ENABLED=0` — defaults to 1 in this trainer |
| Tokenizer mismatch error | `VOCAB_SIZE` (default 8192) must match the tokenizer model file `fineweb_8192_bpe.model`. Re-run data prep if switching vocabs |

---

## References

- **Gated Attention**: Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025 Best Paper. [arXiv 2505.06708](https://arxiv.org/abs/2505.06708) · [GitHub](https://github.com/qiuzh20/gated_attention) · deployed in Qwen3-Next-80B.
- **GLU Attention**: Zichen Zhang, *GLU Attention Improve Transformer*, [arXiv 2507.00022](https://arxiv.org/abs/2507.00022), July 2025.
- **Parallel Residuals motivation**: `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/README.md` — Marko Sisovic's learned-routing table showing `mlp_to_attn ≈ 0` in deep layers.
- **Base trainer**: `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md` — PR #1493 SOTA details.

## Changelog

- **2026-04-22**: Initial version.
