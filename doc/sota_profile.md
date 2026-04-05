# SOTA Architecture Profile (PR #1019, 1.1147 BPB)

## Compute Breakdown (per step, 8xH100)

```
                              GFLOP/layer    % of compute
MLP (up + down projections):     3,401.6       58.0%     ← dominates
Attention projections (Q+K+V+O): 1,700.8       29.0%
Flash Attention 3:                 510.2        8.7%     ← cheap
Overhead (norms, scales, etc):     255.1        4.3%
```

MLP is **58% of compute**. The two matmuls ([B×T, 512] × [512, 1536]
and [B×T, 1536] × [512]) dominate step time. Flash Attention 3 is only
8.7% — highly optimized on H100 Hopper.

Total per step: 17.60 TFLOP (forward + backward)
Step time: 86.7ms on 8xH100 → 6,927 steps in 600s

## Artifact Size Breakdown (15.86 MB actual)

```
                          Params      Before LZMA    After LZMA (est.)    % of artifact
MLP (11 independent):     17.30M      17.35 MB       ~10.0 MB             63%
Attention (Q+K+V+O):       8.65M       8.68 MB       ~5.0 MB              31%
Other (emb, bigram, etc):   1.12M       1.29 MB       ~0.7 MB              5%
Code:                       —           —              0.10 MB              1%
```

MLP is **63% of artifact size** AND **58% of compute**. It dominates both.

## Why MLP Dominates Both

Each layer has two MLP matrices:
- Up projection: [1536, 512] = 786,432 params
- Down projection: [512, 1536] = 786,432 params
- Total: 1,572,864 params/layer × 11 layers = 17.30M

For comparison, attention per layer:
- Q: [512, 512] = 262,144
- K: [256, 512] = 131,072  (GQA-4: only 4 KV heads)
- V: [256, 512] = 131,072
- O: [512, 512] = 262,144
- Total: 786,432 params/layer × 11 layers = 8.65M

MLP is 2x the params of attention per layer because of the 3x expansion.

## Implications for Architecture Optimization

### Saving MLP params saves both artifact AND compute
Sharing MLP bases directly reduces:
1. Artifact: fewer unique matrices to store
2. Compute: no change (same matmuls during forward pass)

Wait — compute doesn't change with sharing. The same [B×T, 512] × [512, 1536]
matmul runs regardless of whether the weight is shared or independent. Only the
artifact benefits from sharing.

### What actually affects step time
The step time is determined by the FORWARD/BACKWARD computation, which depends on:
- Matrix dimensions (dim, mlp_dim, kv_dim)
- Number of layers
- Sequence length × batch size

Sharing MLP weights does NOT change step time — the matmuls are the same size.
What changes step time:
- Wider MLP (larger matmuls) → slower
- More KV heads (MHA vs GQA) → slightly slower
- More layers → proportionally slower
- Adapter A@B materialization → ~0.1% overhead (negligible)

### The key tradeoff
```
Wider MLP  → better per-step quality, larger artifact, slower steps
Shared MLP → smaller artifact, same compute, loss of per-layer independence
Adapters   → restores some independence, tiny compute cost, artifact cost depends on rank
```

## How Our Experiments Compared

| Config | Compute vs SOTA | Steps vs SOTA | Pre-quant BPB |
|--------|:---:|:---:|:---:|
| SOTA (3x, independent) | 1.00x | 6,927 | 1.1354 |
| MHA+4x, 6 shared, no adapt | 1.33x | 5,931 (-14%) | 1.1443 (+0.009) |
| GQA+5x, 6 shared, no adapt | 1.41x | 5,612 (-19%) | 1.1472 (+0.012) |
| GQA+2.5x, 6 shared, r51 | 0.90x | 7,289 (+5%) | 1.1714 (+0.036) |
| **GQA+3x, 6 shared, r42** | **1.00x** | **~6,920** | **untested** |

The GQA+3x+r42 config is unique: **identical compute to SOTA** (same MLP width,
same GQA, same step count) but with shared MLP + lossless rank-42 adapters.
The only question is whether shared+adapter quality matches independent MLP quality.