# Experiment Results: GQA-4 + 3x MLP + Rank-42 Lossless Adapters

## Configuration

```
Layers: 11, model_dim: 512, heads: 8, KV heads: 4 (GQA — same as SOTA)
MLP: 3x (1536 hidden — same as SOTA), 6 shared bases, rank-42 adapters (all fp16)
Training: 2xH100, 2400s wallclock
Seed: 1337
```

## Results

| Metric | SOTA | MHA+4x pure (best) | **GQA+3x r42** |
|--------|:---:|:---:|:---:|
| KV heads | 4 | 8 | 4 |
| MLP hidden | 1536 (3x) | 2048 (4x) | 1536 (3x) |
| Adapters | None | None | rank-42 fp16 |
| Params | 27.07M | 25.27M | 21.10M |
| Steps | 6,927 | 6,178 | 6,244 |
| Step time (2xH100) | — | 389ms | 384ms |
| Pre-quant val_bpb | 1.1354 | 1.1443 | **1.1670** |
| Post-quant val_bpb | 1.1386 | crashed | **1.1710** |
| Artifact | 15.86 MB | — | **15.07 MB** |
| Quant gap | 0.003 | — | **0.004** |
| Peak memory | 22.9 GB | 27.7 GB | **22.9 GB** |
| Gap from SOTA | — | +0.009 | **+0.032** |

## Train Loss Comparison

```
Step     SOTA      MHA+4x pure   GQA+3x r42
500      2.3787    2.3710        2.4063
1000     2.2509    2.2861        2.3208
1500     2.1982    2.1697        2.2073
2000     2.0412    2.0645        2.1076
2500     2.1464    2.1694        2.2086
3000     2.1423    2.0761        2.1151
3500     2.1495    2.0761        2.1138
4000 val 1.2051    1.2004        1.2262
5000     2.0805    1.9891        2.0325
6000     1.9209    —             1.9610
6244     —         —             1.1670 (bpb, final)
```

## Key Findings

### 1. Adapters didn't help: worse than pure sharing at same MLP width

This config matches SOTA's MLP width (3x/1536) and nearly matches its
step count (6,244 vs 6,927), adding rank-42 lossless adapters for
per-layer specialization. But the result (1.1670) is much worse than
our MHA+4x pure sharing (1.1443) despite similar step counts.

### 2. Shared 3x MLP is too weak even with adapters

The rank-42 adapters provide per-layer specialization but can't
compensate for sharing the MLP weights. At step 4000, val_bpb was
1.2262 — 0.021 behind SOTA (1.2051) and 0.026 behind MHA+4x (1.2004).

### 3. Good quantization gap confirms lossless adapters work

Quant gap: 0.004 BPP (vs 0.003 for SOTA). All rank-42 adapter
matrices stayed under 65K elements → fp16 passthrough → near-zero
adapter quantization error. The lossless adapter concept is validated.

### 4. Artifact well under budget

15.07 MB with 930 KB headroom. No pruning needed.

## Conclusion

**Adapters cannot recover the quality lost from MLP sharing.** Even with
rank-42 lossless adapters giving per-layer specialization, shared 3x MLP
underperforms wider 4x MLP without adapters. The width of the MLP matters
more than per-layer adapter specialization.

Updated ranking of all experiments:

```
Config                    MLP    Adapt  Steps   Pre-quant   Gap
MHA+4x pure share        2048   none   6,178   1.1443      +0.009  ← BEST
GQA+5x pure share        2560   none   5,612   1.1472      +0.012
MHA+3.5x r100 adapter    1792   r100   5,780   1.1653      +0.030
GQA+3x r42 adapter       1536   r42    6,244   1.1670      +0.032
GQA+2.5x r51 adapter     1280   r51    7,388   1.1714      +0.036
16L shared+adapter        1536   r100   4,720   1.1721      +0.037
```

**The pattern: wider MLP (without adapters) > narrower MLP (with adapters).**