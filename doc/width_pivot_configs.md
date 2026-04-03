# Width Pivot: Configuration Analysis

All configs use 11 layers, 3 shared MLP bases, shared+adapter architecture.
LZMA ratio estimated conservatively at 1.65x (measured 1.58x in 16L experiment,
SOTA achieves 1.73x — reality is likely between).

---

## Config Comparison

| Config | dim | H | KV | MLP | Rank | Params | Artifact | Headroom | Step | Steps/600s |
|--------|-----|---|----|----|------|--------|----------|----------|------|-----------|
| SOTA (reference) | 512 | 8 | 4 | 3x indep | — | 27.1M | 15.86M | 140KB | 87ms | 6,927 |
| **A: MHA only** | 512 | 8 | **8** | 3x | 100 | 21.9M | 14.4M | 1594KB | 92ms | 6,556 |
| **B: MHA + 3.5x** | 512 | 8 | **8** | **3.5x** | 100 | 23.3M | 15.2M | 791KB | 96ms | 6,228 |
| C: MHA + 4x | 512 | 8 | 8 | 4x | 100 | 24.6M | 16.0M | -12KB | 101ms | 5,931 |
| **D: MHA + 4x r80** | 512 | 8 | **8** | **4x** | **80** | 23.5M | 15.2M | 789KB | 101ms | 5,931 |
| E: MHA + 4x r64 | 512 | 8 | 8 | 4x | 64 | 22.6M | 14.5M | 1429KB | 101ms | 5,931 |
| F: 576d MHA | 576 | 9 | 9 | 3x | 64 | 25.1M | 16.1M | -98KB | 104ms | 5,752 |
| G: 640d GQA-5 | 640 | 10 | 5 | 3x | 64 | 25.8M | 16.6M | -569KB | 111ms | 5,401 |
| I: 576d GQA-3 | 576 | 9 | 3 | 3x | 100 | 22.0M | 14.5M | 1499KB | 96ms | 6,238 |

## Why Not Widen model_dim?

Widening from 512 to 576 or 640 is tempting but:

1. **Everything scales**: Attention is O(dim²), MLP is O(dim × mlp_dim). Going
   512→640 increases total FLOPs by ~56%, adding ~25ms to step time.
2. **Budget pressure**: 576d MHA (Config F) is OVER budget at conservative LZMA.
   640d (Config G) is way over.
3. **Step count drops**: 5,400-5,750 steps vs SOTA's 6,927. We already proved
   fewer steps hurts.
4. **Diminishing returns**: The H100 parallelizes width well, but there's still
   overhead from larger memory transfers.

Staying at 512 dim is the safe choice — minimal step time increase, maximum
training steps.

## Why Full MHA (GQA-4 → MHA-8)?

Going from 4 KV heads to 8 (full multi-head attention):

- **Each head gets unique K,V**: No more sharing K,V between head pairs.
  Currently heads 0-1 share K₀,V₀ and heads 2-3 share K₁,V₁. With MHA,
  every head has its own perspective on what to attend to.
- **Cost**: +2.9M attention params, ~5ms step time → ~400 fewer steps
- **Historical context**: Most of the competition uses GQA-4 because attention
  budget was limited. With shared MLP freeing 8M params, we can afford MHA.

## Why Wider MLP (3x → 3.5x or 4x)?

Going from MLP hidden 1536 to 1792 or 2048:

- **More feature detectors**: The MLP is the main "thinking" layer. More hidden
  neurons = more features the model can detect and combine.
- **Historical evidence**: Going 2x→3x MLP was worth ~0.02 BPB in competition
  history (one of the biggest single improvements).
- **Cost**: 3.5x adds ~1.3M shared + ~0.6M adapter params. 4x adds ~2.6M + ~1.1M.
  Step time: +10-15ms (MLP matmuls parallelize on GPU).

---

## Recommended Config

### Primary: Config B — 512d, MHA, 3.5x MLP, rank 100

```
model_dim=512, heads=8, kv_heads=8, mlp_dim=1792, 3 shared bases, rank=100
```

| Metric | Value | vs SOTA |
|--------|-------|---------|
| Params | 23.3M | -3.8M |
| Artifact (conservative) | 15.2M | -0.7M |
| Headroom | 791 KB | Safe |
| Est step time (8xH100) | ~96ms | +9ms |
| Est steps in 600s | ~6,228 | -699 (10% fewer) |

**Why this config:**
1. **Both improvements**: Full MHA AND wider MLP. Two quality levers.
2. **Comfortable budget**: 791 KB headroom at conservative LZMA. Even with
   seed variance (~120 KB) and pruning compression loss, we fit.
3. **Minimal step time hit**: ~96ms vs SOTA's 87ms. Only 10% fewer steps,
   much better than the 16L experiment's 31% fewer.
4. **Rank 100 preserved**: Full adapter specialization capacity.

### Fallback: Config A — 512d, MHA, 3x MLP, rank 100

If Config B's artifact is too close to the limit:

```
model_dim=512, heads=8, kv_heads=8, mlp_dim=1536, 3 shared bases, rank=100
```

| Metric | Value | vs SOTA |
|--------|-------|---------|
| Artifact (conservative) | 14.4M | -1.5M |
| Headroom | 1594 KB | Very safe |
| Est steps | ~6,556 | Only 5% fewer than SOTA |

Safest possible — just MHA, same MLP width. Lots of headroom. If MHA alone
gives 0.005+ BPB improvement, this wins without any risk.

### Aggressive: Config D — 512d, MHA, 4x MLP, rank 80

If Config B works and we want to push further:

```
model_dim=512, heads=8, kv_heads=8, mlp_dim=2048, 3 shared bases, rank=80
```

| Metric | Value | vs SOTA |
|--------|-------|---------|
| Artifact (conservative) | 15.2M | Same as B |
| Est steps | ~5,931 | 14% fewer |

Wider MLP (4x) but lower adapter rank (80). Trades per-layer specialization
for more raw MLP capacity. Higher risk — 14% fewer steps is significant.

---

## What To Change in Code

All configs use the same code (our existing `train_gpt.py`). Just change env vars:

**Config B (recommended):**
```bash
NUM_LAYERS=11 NUM_KV_HEADS=8 MLP_MULT=3.5 \
ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
WARMDOWN_ITERS=4000 XSA_LAST_N=11 VE_LAYERS=9,10 \
TARGET_MB=15.2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Config A (safe fallback):**
```bash
NUM_LAYERS=11 NUM_KV_HEADS=8 \
ADAPTER_RANK=100 NUM_SHARED_MLPS=3 \
WARMDOWN_ITERS=4000 XSA_LAST_N=11 VE_LAYERS=9,10 \
TARGET_MB=14.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: NUM_LAYERS=11, XSA_LAST_N=11, VE_LAYERS=9,10, WARMDOWN_ITERS=4000 are
reverted to SOTA defaults since we're back to 11 layers.