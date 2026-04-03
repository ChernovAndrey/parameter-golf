# Experiment: Pure MLP Sharing (No Adapters)

## The Idea

Share MLP weights between pairs of adjacent layers — no low-rank adapters,
no per-layer specialization. The saved parameters go entirely to wider model
capacity (full MHA + bigger MLP hidden).

```
Layer 0 ─┐
         ├── share MLP_base_0 (full-rank, 1792×512 + 512×1792)
Layer 1 ─┘

Layer 2 ─┐
         ├── share MLP_base_1
Layer 3 ─┘

Layer 4 ─┐
         ├── share MLP_base_2
Layer 5 ─┘

Layer 6 ─┐
         ├── share MLP_base_3
Layer 7 ─┘

Layer 8 ─┐
         ├── share MLP_base_4
Layer 9 ─┘

Layer 10 ──── MLP_base_5 (unique, no sharing partner)
```

6 MLP bases for 11 layers. Most layers share with exactly 1 neighbor.
Layer 10 is unique. Attention is fully independent per layer (never shared).

---

## Motivation

### Why drop the adapters?

The 16L experiment (shared + rank-100 adapters) revealed three problems
caused by the adapters:

1. **Quantization gap 4x worse than SOTA** (0.012 vs 0.003 BPB) — separate
   quantization of shared bases + adapter matrices introduces more error
   than single-matrix GPTQ
2. **LZMA compression ratio worse** (1.58x vs 1.73x) — adapter A matrices
   have high entropy from kaiming initialization, compress poorly
3. **Training overhead** — 32 extra matmuls per forward pass (A@B for each
   layer × up/down), adapter initialization delay (B starts at zero,
   needs 2 steps to activate)

Without adapters:
- Quantization is standard GPTQ on shared bases — same proven pipeline as SOTA
- No high-entropy adapter weights to compress — LZMA ratio should be closer to 1.73x
- No extra matmuls — faster step time → more training steps
- No initialization issues — shared bases are orthogonal-init, immediately functional

### Why this is NOT PR #363 (depth recurrence)

PR #363 shared **entire transformer blocks** (attention + MLP) and looped
them 3 times. That caused:
- 900x quantization error amplification (same quantized weights used 3× in sequence)
- Each loop step adds 32ms step time overhead

Our approach shares **ONLY MLP**, and only between adjacent pairs:
- Attention is fully independent per layer → provides per-layer identity
- MLP is shared between 2 adjacent layers, not looped 3 times
- The attention between shared-MLP layers transforms the input differently,
  so the same MLP weight operates on different data each time

### Budget math: what we gain

```
SOTA MLP:           11 independent × 1.57M = 17.30M params
Our MLP (6 bases):   6 shared     × 1.57M = 9.44M params  (at 3x)
                     6 shared     × 1.84M = 11.01M params (at 3.5x)
Savings:            6-8M params freed → spent on:
```

Where the savings go:
- **Full MHA** (KV heads 4→8): +2.9M attention params
- **Wider MLP** (3x→3.5x, hidden 1536→1792): +1.6M shared MLP params
- **Remaining headroom**: comfortable artifact margin

---

## Configuration

### Parameters

```bash
export NUM_LAYERS=11
export NUM_KV_HEADS=8        # Full MHA (was 4 = GQA)
export MLP_MULT=3.5           # 1792 hidden (was 3.0 = 1536)
export ADAPTER_RANK=0         # No adapters (pure sharing)
export NUM_SHARED_MLPS=6      # 6 shared bases for 11 layers
export XSA_LAST_N=11          # XSA all layers
export VE_LAYERS=9,10         # Value embedding on last 2
export WARMDOWN_ITERS=4000    # Same as SOTA
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export TARGET_MB=14.5         # Conservative target
export SEED=1337
```

### Run command (2xH100 test, 40 min)

```bash
MAX_WALLCLOCK_SECONDS=2400 torchrun --standalone --nproc_per_node=2 \
  records/track_10min_16mb/2026-04-02_SharedMLP_Rank100_16L/train_gpt.py
```

### Run command (8xH100 final)

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-02_SharedMLP_Rank100_16L/train_gpt.py
```

---

## Expected Numbers

### Model

| Metric | SOTA | This experiment |
|--------|------|-----------------|
| Layers | 11 | 11 |
| model_dim | 512 | 512 |
| Heads / KV heads | 8 / 4 (GQA) | 8 / 8 (MHA) |
| MLP hidden | 1536 (3x) | 1792 (3.5x) |
| MLP structure | 11 independent | 6 shared bases |
| Adapters | None | None |
| Total params | 27.1M | ~23.7M |

### Layer-to-group mapping

```
layer_to_group = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]

Group 0: layers 0, 1   (share MLP)
Group 1: layers 2, 3   (share MLP)
Group 2: layers 4, 5   (share MLP)
Group 3: layers 6, 7   (share MLP)
Group 4: layers 8, 9   (share MLP)
Group 5: layer 10      (unique)
```

### Artifact budget

| Component | SOTA | This experiment |
|-----------|------|-----------------|
| Attention (int6) | 8.3 MB | 11.6 MB (+3.3, from MHA) |
| MLP (int6) | 10.0 MB | 6.8 MB (-3.2, from 6 bases vs 11) |
| Other | 1.2 MB | 1.2 MB |
| **Before LZMA** | **~26 MB** | **~24.2 MB** |
| **After LZMA (1.65x)** | **~15.9 MB** | **~14.7 MB** |
| **After LZMA (1.73x)** | **~15.1 MB** | **~14.1 MB** |
| **Headroom (conservative)** | **~100 KB** | **~1.2 MB** |

Comfortable margin. No risk of going over budget.

### Training performance

| Metric | SOTA | This experiment |
|--------|------|-----------------|
| Step time (8xH100) | 86.7ms | ~96ms est. (wider MLP + MHA) |
| Steps in 600s | 6,927 | ~6,228 est. |
| Step count difference | — | -10% (much better than 16L's -31%) |

### Quantization

| Metric | SOTA | This experiment |
|--------|------|-----------------|
| Quantization method | GPTQ int6 per-layer | GPTQ int6 on shared bases |
| Hessians | Per-layer | Aggregated per base (avg of 2 layers) |
| Expected quant gap | 0.003 BPB | ~0.003-0.005 BPB (simpler than adapters) |

---

## What Could Go Wrong

### 1. Adjacent layers might need different MLPs

Layers in a sharing pair have IDENTICAL MLP behavior. If layer 0 needs
to detect "syntax features" and layer 1 needs "semantic features",
sharing hurts. But attention provides per-layer identity, and adjacent
layers likely learn very similar features anyway.

### 2. Quantization error applied twice

The same quantized weight is used by 2 layers. The quantization error
`Q(W) - W` is the same for both. This is NOT error compounding (the
error doesn't amplify), but it means both layers suffer the same
quantization artifacts — they can't compensate for each other.

### 3. Gradient accumulation for shared bases

Each base gets gradients from 2 layers (or 1 for layer 10). With Muon's
Newton-Schulz normalization, the 2x gradient magnitude is normalized away.
This should be fine.

---

## Comparison: All Three Approaches

| | SOTA (independent) | 16L (shared+adapter) | Pure sharing |
|---|---|---|---|
| MLP independence | Full | Adapter-modified | None (within pairs) |
| Extra depth | No | +5 layers | No |
| Extra width | No | No | MHA + 3.5x MLP |
| Steps in 600s | 6,927 | ~4,720 | ~6,228 |
| Quant complexity | Standard | Complex (2 sources) | Standard |
| Artifact risk | Low | High (was over budget) | Low |
| Code complexity | Baseline | High | Low |

---

## Success Criteria

1. `model_params` should show ~23.7M
2. Train loss at step 500 should be close to or below SOTA's 2.3787
3. Pre-quant val_bpb should be ≤ 1.135 (SOTA level)
4. Quant gap should be ≤ 0.005 BPB
5. Artifact < 16,000,000 bytes
6. Final sliding window BPB should beat 1.1147 (SOTA)