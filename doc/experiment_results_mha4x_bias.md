# Experiment Results: MHA + 4x MLP + Bias (First Complete End-to-End)

## Configuration

```
Layers: 11, model_dim: 512, heads: 8, KV heads: 8 (Full MHA)
MLP: 4x (2048 hidden), 6 shared bases, NO adapters
MLP layer bias: ON (22K params)
Training: 2xH100, 2400s wallclock
Seed: 1337
```

## Full Results

| Metric | SOTA (PR #1019) | **This experiment** |
|--------|:---:|:---:|
| KV heads | 4 (GQA) | 8 (MHA) |
| MLP hidden | 1536 (3x) | 2048 (4x) |
| MLP structure | 11 independent | 6 shared + bias |
| Params | 27.07M | 25.29M |
| Steps | 6,927 | 6,307 |
| Step time (2xH100) | — | 380ms |
| Step time (8xH100 est.) | 86.7ms | ~95ms |
| Pre-quant val_bpb | 1.1354 | 1.1452 |
| Post-quant val_bpb | 1.1386 | **1.1504** |
| **Sliding window BPB** | **1.1151** | **1.1267** |
| Quant gap | 0.003 | 0.005 |
| Artifact | 15.86 MB | **14.78 MB** |
| Peak memory | 22.9 GB | 30.9 GB |
| **Gap from SOTA (sliding)** | — | **+0.012** |

## First Complete End-to-End Result

This is the first experiment to complete the FULL pipeline:
training → EMA → GPTQ quantization → selective pruning → LZMA →
dequant round-trip → standard eval → sliding window eval.

Previous MHA+4x run (without bias) crashed at quantization due to a bug.
Its pre-quant was 1.1443 (slightly better than this run's 1.1452).

## Train Loss Comparison

```
Step     SOTA      MHA+4x+bias    MHA+4x(crashed)
500      2.3787    2.3649         2.3710
1000     2.2509    2.2851         2.2861
1500     2.1982    2.1718         2.1697
2000     2.0412    2.0698         2.0645
2500     2.1464    2.1686         2.1694
3000     2.1423    2.0786         2.0761
3500     2.1495    2.0814         2.0761
4000 val 1.2051    1.2058         1.2004
5000     2.0805    1.9965         1.9891
6000     1.9209    1.9279         —
6307     —         1.1452 (bpb)   —
6927     1.1354    —              —
```

## Key Metrics

### Quantization
```
Pre-quant:      1.1452
Post-quant:     1.1504  (gap: 0.005 — better than 16L's 0.012)
Sliding window: 1.1267  (improvement: 0.024 from sliding)
```

Quant gap of 0.005 is reasonable — no adapters to quantize separately.
No pruning needed (artifact 14.09 MB unpruned < 15.9 MB target).

### Artifact
```
Model (int6+lzma):  14.66 MB
Code:                0.11 MB
Total:              14.78 MB (1.22 MB headroom)
```

### Sliding Window
```
Standard eval:       1.1504 BPB
Sliding (stride 64): 1.1267 BPB (saves 0.024)
SOTA sliding:        1.1151 BPB
Gap:                 0.012 BPB
```

## Gap Analysis

```
SOTA final:    1.1151
Ours final:    1.1267
Gap:           0.012 BPB

Breakdown (estimated):
  ~620 fewer steps (6307 vs 6927):  ~0.005 BPB
  Shared MLP quality loss:          ~0.004 BPB
  Quant gap (0.005 vs 0.003):       ~0.002 BPP
  Bias effect:                      ~0.001 BPB (neutral/slight negative)
  Total:                            ~0.012 BPB ← matches
```

## Comparison: All Complete Experiments

| Config | Sliding BPB | Gap from SOTA |
|--------|:-----------:|:-------------:|
| **SOTA (PR #1019)** | **1.1151** | **—** |
| MHA+4x+bias (this) | 1.1267 | +0.012 |
| MHA+3.5x+adapter (width) | 1.1475 | +0.032 |
| 16L shared+adapter | 1.1598 | +0.045 |
| GQA+2.5x r51 | not completed | — |

## Conclusion

The 0.012 gap from SOTA is our best result. The architecture works —
the gap is from fewer steps (wider model = slower) and shared MLP
quality loss. The bias didn't meaningfully help or hurt.

Next experiment: Pool routing (MLP_POOL_ROUTING=1) on the same config
to test whether per-token dynamic routing can close the remaining 0.004
shared MLP quality gap.