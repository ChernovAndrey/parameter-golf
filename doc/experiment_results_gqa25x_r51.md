# Experiment Results: GQA-4 + 2.5x MLP + Rank-51 Lossless Adapters

## Configuration

```
Layers: 11, model_dim: 512, heads: 8, KV heads: 4 (GQA — same as SOTA)
MLP: 2.5x (1280 hidden), 6 shared bases, rank-51 adapters (all fp16, zero quant error)
Training: 2xH100, 2400s wallclock
Seed: 1337
```

## Results

| Metric | SOTA | MHA+4x pure | GQA+5x pure | **GQA+2.5x r51** |
|--------|:---:|:---:|:---:|:---:|
| KV heads | 4 (GQA) | 8 (MHA) | 4 (GQA) | 4 (GQA) |
| MLP hidden | 1536 (3x) | 2048 (4x) | 2560 (5x) | **1280 (2.5x)** |
| Adapters | None | None | None | **rank-51 fp16** |
| Params | 27.07M | 25.27M | 25.49M | **19.64M** |
| Steps | 6,927 | 6,178 | 5,612 | **7,388** |
| Step time (2xH100) | — | 389ms | 428ms | **325ms** |
| Step time (8xH100 est.) | 86.7ms | ~97ms | ~107ms | **~81ms** |
| Pre-quant val_bpb | 1.1354 | 1.1443 | 1.1472 | **1.1714** |
| Peak memory | 22.9 GB | 27.7 GB | 27.1 GB | **21.8 GB** |
| Gap from SOTA | — | +0.009 | +0.012 | **+0.036** |

## Train Loss Comparison

```
Step     SOTA      MHA+4x     GQA+5x     GQA+2.5x r51
500      2.3787    2.3710     2.3737     2.4158 (+0.037)
1000     2.2509    2.2861     2.2844     2.3319 (+0.081)
1500     2.1982    2.1697     2.1638     2.2217 (+0.024)
2000     2.0412    2.0645     2.0564     2.1199 (+0.079)
2500     2.1464    2.1694     2.1504     2.2275 (+0.081)
3000     2.1423    2.0761     2.0592     2.1485 (+0.006)
3500     2.1495    2.0761     2.0629     2.1565 (+0.007)
4000 val 1.2051    1.2004     1.1917     1.2533 (+0.048)
5000     2.0805    1.9891     1.9689     2.0815
6000     1.9209    —          1.9218     2.0178
7000     —         —          —          1.9249
7388     —         —          —          1.1714 (bpb, final)
```

## Key Findings

### 1. Most training steps of any experiment — but worst BPB

7,388 steps — 461 more than SOTA (6,927). But pre-quant val_bpb 1.1714
is the worst of all our shared MLP experiments. The extra steps couldn't
compensate for the weaker MLP.

### 2. 2.5x MLP (1280 hidden) is too narrow

At step 4000, val_bpb was 1.2533 — **0.048 behind SOTA's 1.2051**. This
is the largest per-step gap of any experiment. Even with 3,388 more steps
after step 4000, the model only improved to 1.1714.

Compare with MHA+4x which was 0.005 AHEAD of SOTA at step 4000 with
only 2,178 remaining steps and reached 1.1443.

### 3. Fastest step time but doesn't matter

325ms on 2 GPUs (~81ms on 8 GPUs) — fastest of all experiments. But
speed without per-step quality is worthless:

```
Config         Step time    Steps    val_bpb
MHA+4x pure    389ms        6,178    1.1443   ← best result
GQA+5x pure    428ms        5,612    1.1472
GQA+2.5x r51   325ms        7,388    1.1714   ← worst result despite most steps
```

### 4. Rank-51 lossless adapters worked correctly

All adapter matrices stayed under 65K elements → fp16 passthrough →
zero quantization error on adapters. The adapter mechanism is sound;
the problem is purely the narrow MLP base.

### 5. Lowest memory usage

21.8 GB peak — smallest footprint. Not useful if quality suffers.

## Conclusion: MLP Width Has a Floor

There's a minimum MLP width below which extra steps can't compensate.
That floor is somewhere between 2.5x (1280, too narrow) and 3x (1536,
SOTA's choice). Going below SOTA's 3x MLP width loses more quality
per step than faster training can recover.

This validates the next experiment: **GQA-4, 3x MLP (1536), 6 bases,
rank-42 adapters** — match SOTA's MLP width AND step count, add only
lossless per-layer specialization.

## Optimal MLP Width Summary (from all experiments)

```
MLP width    per-step quality    step count    final BPB
2.5x (1280)  worst               best (7388)   worst (1.1714)
3x (1536)    baseline (SOTA)     baseline      baseline (1.1354)
3.5x (1792)  good                ok (5780)     ok (1.1653 pre)
4x (2048)    great               ok- (6178)    best (1.1443 pre)
5x (2560)    best                worst (5612)  ok (1.1472 pre)
```

Sweet spot: 3x-4x MLP. Below 3x loses too much per-step quality.
Above 4x the step count loss outweighs the width gain.