# Ideas Backlog

Potential configurations and ideas to test, ordered by priority.

---

## Untested Configurations

### 1. GQA-4 + 5x MLP (pure sharing, 6 bases)

Trade MHA for much wider MLP. Keep original GQA-4 attention.

```bash
export NUM_LAYERS=11 NUM_KV_HEADS=4 MLP_MULT=5 XSA_LAST_N=11 VE_LAYERS=9,10
export WARMDOWN_ITERS=4000 ADAPTER_RANK=0 NUM_SHARED_MLPS=6
export BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 TARGET_MB=15.9 SEED=1337
```

| Metric | Value |
|--------|-------|
| MLP hidden | 2560 (vs SOTA 1536, +67%) |
| KV heads | 4 (same as SOTA) |
| Params | 25.5M |
| Artifact | ~15.4M (621 KB headroom) |
| Est steps (8xH100) | ~5,662 |

**Rationale**: MLP expansion 2x→3x was ~0.02 BPB gain. Going 3x→5x
could be similarly impactful. GQA-4 already captures ~98% of MHA quality
per the original GQA paper. The 25% extra MLP width (2560 vs 2048) might
matter more than the MHA upgrade.

### 2. GQA-4 + 4.5x MLP (safer version of #1)

```bash
export NUM_KV_HEADS=4 MLP_MULT=4.5 NUM_SHARED_MLPS=6 ADAPTER_RANK=0
```

| Metric | Value |
|--------|-------|
| MLP hidden | 2304 |
| Artifact | ~14.4M (1.5 MB headroom) |
| Est steps | ~5,931 |

More headroom, slightly less MLP width. Good fallback if #1 is over budget.

### 3. MHA-8 + 4x MLP + 4 bases (more sharing, wider MLP)

Current experiment uses 6 bases. With 4 bases, each group is 3 layers
(more sharing but fewer base params → room for wider MLP or higher rank).

```bash
export NUM_KV_HEADS=8 MLP_MULT=4 NUM_SHARED_MLPS=4 ADAPTER_RANK=0
```

| Metric | Value |
|--------|-------|
| MLP hidden | 2048 |
| Sharing | [3,3,3,2] layers per base |
| Params | ~21.1M |
| Artifact | ~13.2M (2.8 MB headroom) |

### 4. GQA-4 + 5x MLP + 4 bases (max MLP, min sharing overhead)

```bash
export NUM_KV_HEADS=4 MLP_MULT=5 NUM_SHARED_MLPS=4 ADAPTER_RANK=0
```

| Metric | Value |
|--------|-------|
| MLP hidden | 2560 |
| Sharing | [3,3,3,2] layers per base |
| Params | ~20.2M |
| Artifact | ~12.3M (3.6 MB headroom) |

Lots of headroom. Could push MLP even wider (5.5x?) or add adapters back.

### 5. Pure sharing + TTT (test-time training)

If pure sharing gets close but doesn't beat SOTA, add TTT back.
TTT gave -0.0025 BPB on the previous SOTA stack (PR #549).

Would need to verify TTT works with shared MLP architecture.
Budget concern: TTT adds ~410s eval time. Must fit in 600s eval window.

### 6. Hybrid: 6 bases + small rank-16 adapters

Minimal adapters (rank 16 instead of 100) for slight per-layer
specialization without the heavy quant/compute overhead.

```bash
export NUM_SHARED_MLPS=6 ADAPTER_RANK=16 MLP_MULT=4 NUM_KV_HEADS=8
```

Adapter overhead is tiny: 16 × (2048×16 + 16×512 + 512×16 + 16×2048) = 1.2M params.
Almost no step time impact. Might capture the most important per-layer differences.

---

## Ideas Not Yet Explored

### 7. Asymmetric sharing
Not all layer pairs need to share equally. Maybe:
- Shallow layers (0-3): share aggressively (2 bases for 4 layers)
- Deep layers (7-10): each gets own MLP (4 independent bases)
- Middle layers (4-6): moderate sharing

Deep layers do more specialized work → need more independence.

### 8. Progressive unfreezing
Start with all layers sharing 1 base, then gradually "unfreeze" to more
bases during training. Early training learns shared features, late training
specializes.

### 9. Larger vocab + factored embedding
Current: 1024 vocab, 512 dim embedding (524K params, ~0.5 MB).
Larger vocab captures more subwords → fewer tokens per text → more
context per position. But embedding cost grows.
Factored embedding (like ternary submission): 8192 vocab × 256 bottleneck
+ projection. Fits in similar budget.

### 10. Shared attention too (risky)
Share K,V projections between adjacent layers (not Q,O).
This is closer to PR #363 territory but less aggressive — only K,V shared,
Q,O independent. Each layer still computes unique attention patterns.
High risk of quality loss.