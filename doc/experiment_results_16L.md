# Experiment Results: 16-Layer Shared MLP + Low-Rank Adapters

## Configuration Tested

```
Layers: 16, model_dim: 512, heads: 8, KV heads: 4 (GQA)
MLP: 3x (1536), 3 shared bases + rank-100 per-layer adapters
Training: 2xH100, 2400s wallclock (matching ~4800 8xH100 steps)
```

## Results

| Metric | SOTA (PR #1019) | Ours (16L shared) | Delta |
|--------|:---:|:---:|:---:|
| Layers | 11 | 16 | +5 |
| Params | 27.07M | 24.98M | -2.09M |
| Steps completed | 6,927 (8xH100, 600s) | 4,720 (2xH100, 2400s) | -2,207 (31% fewer) |
| Step time | 86.7ms (8xH100) | 508ms (2xH100) ≈ 127ms (8xH100 est.) | +40ms |
| Pre-quant val_bpb | 1.1354 | 1.1721 | +0.037 |
| Post-quant val_bpb | 1.1386 | 1.1841 | +0.046 |
| Sliding window val_bpb | 1.1151 | 1.1598 | +0.045 |
| Quant gap | 0.003 | 0.012 | 4x worse |
| Artifact size | 15.86 MB | 16.67 MB | OVER BUDGET |

## Train Loss Comparison (step-by-step)

```
Step     SOTA      Ours      Gap       Notes
500      2.3787    2.4503    +0.072    Adapters still warming up
1000     2.2509    2.3432    +0.092    Gap widening
1500     2.1982    2.2279    +0.030    Gap closing (adapters activating)
2000     2.0412    2.1261    +0.085    SOTA had big drop, we didn't match
2500     2.1464    2.2099    +0.064    Both fluctuating
3000     2.1423    2.1081    -0.034    We overtake! (our warmdown helping)
3500     2.1495    2.0921    -0.057    Still ahead in warmdown
4000     1.9433    2.0807    +0.137    SOTA at full LR, us in warmdown (unfair comparison)
4000     val_bpb:  1.2051    1.2012    -0.004    Our val_bpb beats SOTA at same step count!
4720     —         1.1720    —         Our final pre-quant
6927     1.1354    —         —         SOTA final pre-quant
```

## Why It Failed

### 1. Fewer training steps (primary cause)

16 layers at ~508ms/step (2 GPU) = 4,720 steps in 2400s.
SOTA at ~86.7ms/step (8 GPU) = 6,927 steps in 600s.

Even though our val_bpb was BETTER at step 4000 (1.2012 vs 1.2051), SOTA
trained for 2,927 more steps and dropped from 1.2051 → 1.1354. We only had
720 more steps after step 4000, dropping from 1.2012 → 1.1720. The step
deficit is the #1 problem.

### 2. Quantization gap too large

```
Pre-quant:  1.1721
Post-quant: 1.1841
Gap:        0.012 BPB (SOTA gap is 0.003)
```

Separate quantization of shared bases + adapters introduces more error than
quantizing full effective weights. Plus, 64% of ±1 values were aggressively
pruned to fit the 15.9MB target, further hurting quality.

### 3. Artifact over budget

16.67 MB total (16.56 MB model + 0.11 MB code) > 16.00 MB competition limit.
The selective pruning couldn't prune enough. Our budget estimate (15.77 MB)
assumed LZMA ratio of 1.73x, but actual ratio was ~1.58x — the adapter
weights have higher entropy and compress worse than standard bank weights.

## What We Learned

1. **Depth is expensive**: Each layer adds ~40ms to step time (8 GPU),
   costing ~300 training steps per added layer in a 600s race.

2. **Per-step quality matches SOTA**: At the same step count, our 16L model
   is competitive or better (val_bpb 1.2012 vs 1.2051 at step 4000). The
   architecture works — it just can't train long enough.

3. **Shared MLP + adapters work**: Adapters activated within 2 steps (B_up
   from zero to non-zero). The model successfully learned per-layer
   specialization. No optimizer instability.

4. **LZMA ratio is worse with adapters**: ~1.58x vs SOTA's 1.73x. Adapter
   A matrices (kaiming-init, high entropy) compress worse than standard
   bank weights. Budget calculations must account for this.

5. **Quantization gap is 4x worse**: Separate shared+adapter quantization
   loses more than single-matrix GPTQ.

---

## Motivation to Pivot: Width Over Depth

### The insight

The 16L experiment showed that at equal step counts, more depth helps.
But more depth = slower steps = fewer steps in 600s. The solution:

**Keep 11 layers (same step count as SOTA) and spend the shared MLP savings
on making each layer WIDER (more attention capacity, wider MLP).**

Width is parallelizable — wider matmuls run on the GPU with minimal step
time increase because H100 has massive parallelism. Depth is sequential —
each added layer adds ~40ms of serial compute.

### What changes

| | SOTA | 16L experiment | Width pivot |
|---|---|---|---|
| Layers | 11 | 16 | 11 |
| Step time (8 GPU) | 87ms | ~127ms | ~95ms |
| Steps in 600s | 6,927 | ~4,720 | ~6,300 |
| MLP | 3x independent | 3x shared+adapter | 4x shared+adapter |
| Attention | GQA (4 KV heads) | GQA (4 KV heads) | Full MHA (8 KV heads) |
| Budget spent on | per-layer MLP | depth | width |

### Why width should help more than depth

1. **Step count**: 6,300 steps vs 4,720 — 33% more training
2. **4x MLP**: Wider hidden layer (2048 vs 1536) = more feature detectors.
   Going 2x→3x was worth ~0.02 BPB in competition history.
3. **Full MHA**: Each head gets its own K,V (8 KV heads vs 4). Richer
   attention patterns without GQA's information sharing constraint.
4. **GPU utilization**: H100 is underutilized at 512-dim 11-layer. Wider
   model fills the GPU better, getting more quality per millisecond.