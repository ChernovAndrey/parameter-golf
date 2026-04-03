# Experiment Results: Width Pivot (MHA + 3.5x MLP + Adapters)

## Configuration

```
Layers: 11 (same as SOTA)
model_dim: 512
Heads: 8, KV heads: 8 (Full MHA — was GQA-4 in SOTA)
MLP: 3.5x (1792 hidden — was 3x/1536 in SOTA)
MLP structure: 3 shared bases + rank-100 per-layer adapters
Training: 2xH100, 2400s wallclock
```

## Results

| Metric | SOTA (PR #1019) | 16L experiment | Width experiment |
|--------|:---:|:---:|:---:|
| Layers | 11 | 16 | 11 |
| KV heads | 4 (GQA) | 4 (GQA) | 8 (MHA) |
| MLP hidden | 1536 (3x) | 1536 (3x) | 1792 (3.5x) |
| MLP structure | Independent | 3 shared + r100 adapter | 3 shared + r100 adapter |
| Params | 27.07M | 24.98M | 23.26M |
| Steps | 6,927 | 4,720 | 5,780 |
| Step time (2xH100) | — | 508ms | 415ms |
| Step time (8xH100 est.) | 86.7ms | ~127ms | ~104ms |
| Pre-quant val_bpb | 1.1354 | 1.1721 | 1.1653 |
| Post-quant val_bpb | 1.1386 | 1.1841 | 1.1717 |
| Sliding window val_bpb | 1.1151 | 1.1598 | **1.1475** |
| Quant gap | 0.003 | 0.012 | 0.006 |
| Artifact | 15.86 MB | 16.67 MB (OVER) | 15.16 MB |
| Peak memory | 22.9 GB | 32.9 GB | 26.7 GB |
| Gap from SOTA (sliding) | — | +0.045 | +0.032 |

## Train Loss Comparison

```
Step     SOTA      16L exp     Width exp     Width vs SOTA
500      2.3787    2.4503      2.4246        +0.046
1000     2.2509    2.3432      2.3334        +0.083
1500     2.1982    2.2279      2.2283        +0.030
2000     2.0412    2.1261      2.1195        +0.078
2500     2.1464    2.2099      2.2092        +0.063
3000     2.1423    2.1081      2.1109        -0.031
3500     2.1495    2.0921      2.1022        -0.047
4000     1.9433    —           2.1091        +0.166 (different LR phase)
4000 val 1.2051    1.2012      1.2176        +0.013
4500     2.0982    2.0797      2.1222        +0.024
5000     2.0805    —           2.0124        -0.068
5500     1.9939    —           2.0205        +0.027
5780     —         —           1.1653 (bpb)  Final (wallclock stop)
6000     1.9209    —           —
6927     1.1354    —           —             SOTA final
```

## Key Observations

### Improvements over 16L experiment

1. **More training steps**: 5,780 vs 4,720 (+22%). Width adds less
   step time than depth.
2. **Better quantization gap**: 0.006 vs 0.012 BPB. Still 2x worse
   than SOTA but much better than 16L.
3. **Artifact fits**: 15.16 MB with 840 KB headroom. No pruning needed.
4. **Better final BPB**: 1.1475 vs 1.1598 (-0.012 BPB improvement).
5. **Lower memory**: 26.7 GB vs 32.9 GB (11L vs 16L).

### Persistent problems

1. **Still 0.032 behind SOTA** (1.1475 vs 1.1147). The shared MLP
   approach loses ~0.03 BPB regardless of how we spend the savings.

2. **Width (MHA + 3.5x) didn't help vs depth (16L)**:
   - At step 4000: Width val_bpb 1.2176 vs 16L val_bpb 1.2012
   - Width is WORSE per step despite MHA + wider MLP
   - Width only wins overall because of more steps (5,780 vs 4,720)

3. **The 0.03 gap is consistent**: Both experiments show ~0.03-0.05 gap
   from SOTA at comparable training points. This suggests the shared MLP
   itself costs ~0.03 BPB vs independent MLPs.

## Analysis: Where Does the 0.03 BPB Come From?

### Hypothesis 1: Shared MLP lacks per-layer specialization
Even with rank-100 adapters, the effective weight is constrained to
`W_shared + rank-100 perturbation`. Independent MLPs have full rank-512
freedom per layer. The adapter captures the most important 100 directions
of specialization but misses the remaining 412.

Evidence: Width and 16L experiments have adapters but both show ~0.03 gap.

### Hypothesis 2: Adapter quantization overhead
The 0.006 quant gap (vs SOTA 0.003) loses 0.003 BPB. That's ~10% of
the total 0.032 gap.

### Hypothesis 3: Fewer training steps
Width got 5,780 steps vs SOTA 6,927 — 17% fewer. At step 4000, our
val_bpb (1.2176) was behind SOTA (1.2051) by 0.013. The remaining
1,147 step deficit costs additional BPB in final convergence.

### Estimated gap breakdown
```
Shared MLP specialization loss:   ~0.015-0.020 BPB
Fewer training steps (17%):       ~0.008-0.010 BPB
Quantization overhead:            ~0.003 BPB
Total estimated:                  ~0.026-0.033 BPP
Actual:                            0.032 BPB ← matches
```

## Motivation for Next Experiment: Pure Sharing (No Adapters)

The analysis suggests two things to try:
1. Remove adapters → eliminates quant overhead (saves 0.003 BPB)
   + faster steps → more training steps
2. Use 6 bases instead of 3 → less sharing (pairs vs groups of 4)
   + allows 4x MLP (wider hidden, more features)

Pure sharing (no adapters) trades per-layer specialization for:
- **Simpler quantization** (standard GPTQ, no adapter overhead)
- **Faster step time** (no A@B matmuls)
- **More MLP capacity** (4x = 2048 hidden vs 3.5x = 1792)
- **Less sharing** (6 bases = pairs, vs 3 bases = groups of 4)

The bet: 4x MLP width + better quant + more steps > per-layer adapters.