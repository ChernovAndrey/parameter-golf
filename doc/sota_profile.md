# SOTA Architecture Profile (PR #1493, 1.0810 BPB)

**Submission**: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT
**Author**: bigbag · **Date**: 2026-04-09 · **Artifact ~15.99 MB**
**Hardware**: 8× H100 SXM 80 GB · **Train ~588 s · Eval ~500 s**

Replaces the PR #1019 (1.1147) profile. The old profile is retained at the bottom as an appendix.

> Per-record and per-technique discussion lives in `leaderboard_update_2026-04-21.md`; the model card lives in `sota_architecture.md`. All numbers below are reproducible via `python3 doc/profile_calculator.py --preset new_sota` (or `--all` to compare old vs new side-by-side).

---

## Contribution to Memory vs Latency

Side-by-side: share of artifact size (16 MB budget) vs share of per-step compute (loop-active). Computed with `profile_calculator.py` at batch = 786,432 tokens/step, T = 98,304 tokens/GPU, Flash Attn 3 ≈ 0.30× proj cost, fwd+bwd = 3× fwd.

| Component | **Memory %** | **Latency %** | What it is |
|---|---:|---:|---|
| **MLP (up+down, 4× × 11 layers)** | **72.5 %** | **59.7 %** | Dominant on both axes. 4× expansion (hidden 2048) + recurrence multiplies the compute share on top. |
| **Attention (Q+K+V+O × 11 layers)** | **19.3 %** | **22.4 %** | Compute share slightly exceeds memory share → attention is "compute-dense" per byte. |
| **Token embedding (8192 × 512, tied)** | **7.1 %** | **7.0 %** | Tied to LM head, so one matrix does both input embed lookup and output projection. SP8192 is the reason this row grew. |
| **Flash Attention 3** | — | **6.7 %** | Pure compute (no stored params). Cheap on Hopper. |
| **Other (norms/scales/skip gates/code/tokenizer)** | **1.2 %** | **4.1 %** | LayerNorm scales, q_gain, attn/mlp/skip scalars, tokenizer model, LZMA-wrapped code. |
| **TOTAL** | **100 %** | **100 %** |  |

Absolute numbers behind the percentages:
- **Artifact**: 15.63 MB estimated, 15.99 MB observed (difference is the overhead constant and rounding).
- **Compute**: 35.2 TFLOP per step (fwd+bwd) when the 3-layer recurrence is active; 23.6 TFLOP per step pre-recurrence (first 35 % of training).

Quick visual:

```
Memory share                         Latency share (loop on)
──────────────────────────           ─────────────────────────────
MLP              ████████████  72%   MLP               █████████  60%
Attention        ███            19%  Attention         ████       22%
Token embed      █               7%  Token embed/head  █           7%
Other            ▏               1%  Flash Attention   █           7%
                                     Overhead          ▏           4%
```

**Reading these numbers for MoE planning.** The asymmetry MLP = 72 % memory vs 60 % latency (and attention = 19 % memory vs 22 % latency) says **MLP is limited more by artifact than by wall time**. If you route MLP experts with **bit-sharing or rank-factorization** (cutting memory without much compute cost), you trade against the stingier axis; if you route them with full-rank experts (cutting compute per token but duplicating memory), you trade against the already-saturated axis.

---

## Per-Step Compute Breakdown (8× H100, loop-active)

MLP expansion is 4× (hidden 2048, was 3× = 1536). Three-layer recurrence means layers 3, 4, 5 are each evaluated **three times per step** while the loop is active (activated at 35 % of training). Encoder virtual sequence `[0,1,2,3,4,5,3,4]` + decoder `[5,3,4,5,6,7,8,9,10]` = 17 virtual evals.

```
                                     TFLOP (fwd+bwd)    % of step compute
MLP  (up 512→2048 + down 2048→512):        21.02           59.7%   ← dominates
Attention projections (Q/K/V/O):            7.89           22.4%
LM head / embedding (8192 vocab, tied):     2.47            7.0%   ← bigger with SP8192
Flash Attention 3 (causal):                 2.36            6.7%
Overhead (norms, scales):                   1.44            4.1%
──────────────────────────────────────────────────────────────────
TOTAL per step (looping active):           35.20 TFLOP     100%
TOTAL per step (pre-recurrence):           23.65 TFLOP
Step time (observed, avg):                  ~129 ms  →  ~4,550 steps in 588 s
```

Pre-recurrence (first 35 % of training) the step is 11 evals × 641.7 GFLOP/eval forward ≈ 21 GFLOP/step forward; loop-active regime is 17 evals × 641.7 = 10,907 GFLOP/step forward. Fwd+bwd adds ~3× on top.

Key shifts vs the PR #1019 profile:
- MLP fraction grew from 57.0 % → 59.7 %. The 4× widening + 3× recurrence on layers 3–5 more than offsets the bigger denominator (SP8192 LM head).
- Attention-projection fraction fell (28.5 % → 22.4 %) as MLP and LM head grew.
- Flash Attention 3 stayed cheap (8.5 % → 6.7 %) — same matrices, bigger denominator.
- LM head / embedding fraction grew dramatically (1.7 % → 7.0 %) — SP8192 multiplies the final `[T, 512] × [512, 8192]` projection by 8×.
- Step time went from 86.7 ms (6,927 steps) to ~129 ms (4,550 steps) — recurrence is not free, but it buys effective depth at zero artifact cost.

---

## Artifact Size Breakdown (~15.99 MB)

After SDClip Full-Hessian GPTQ (int6 matrices, int8 embeddings), byte-shuffle + Brotli-11, plus the LZMA-wrapped source code. Per-component compression ratios (compressed / int-raw) are **MLP 0.65**, **Attention 0.46**, **Embedding 0.26** — attention compresses best because the matrices sit in the sharp-tail regime where SDClip's `clip = k·σ` is most entropy-efficient.

```
                               Raw params   int6/int8 raw   After Brotli-11    % of artifact
MLP bank (up + down, 4× × 11L):  23.07M      ~17.3 MB (int6)    ~11.3 MB          72.5%
Attention (Q+K+V+O, 11L):         8.65M       ~6.5 MB (int6)     ~3.0 MB          19.3%
Token embedding (8192 × 512):     4.19M       ~4.19 MB (int8)    ~1.1 MB           7.1%
Small params + tokenizer + LZMA-code:   ~0.5 MB raw              ~0.19 MB          1.2%
──────────────────────────────────────────────────────────────────────────────
Total artifact (estimated):                                      ~15.63 MB        100%
Total artifact (observed):                                       ~15.99 MB
```

Per-record numbers from the seed tables: 15,991,930 / 15,992,919 / 15,993,232 bytes. The ~0.36 MB gap between estimate and observed is a combination of (a) the 180 KB overhead constant in the calculator, (b) scale-metadata per row that varies by matrix, and (c) rounding inside Brotli blocks.

Notes:
- MLP is ~71 % of artifact at MLP 4× (was 63 % at MLP 3×). Artifact savings scale more with compression than with per-layer pruning now.
- Token embedding jumped from ~0.3 MB (SP1024, int6 RTN, LZMA) to ~1.1 MB (SP8192, int8 GPTQ SDClip, brotli). Bigger table, but the extra context/step from SP8192 pays back multiplicatively.
- Selective pruning to {−1, 0, +1} was dropped — SDClip fits natively under 16 MB.
- The `train_gpt.py` source is itself LZMA+base85 wrapped inside an `exec(…)` — saves ~43 KB vs plain source.

---

## Why Depth Recurrence Is So Compelling Here

3-layer recurrence on layers 3, 4, 5 gives you **17 virtual layers from 11 physical params**. Per-layer cost:

```
Virtual depth gained:          +6 layers
Artifact cost of the gain:       0 bytes  (shared weights)
Compute cost of the gain:     +6 × per-block TFLOPs during looping region
```

Given that MLP is ~71 % of the artifact and artifact is the binding constraint, *zero-artifact virtual depth is the highest leverage technique in the stack*. The ~0.005 BPB measured gain from 3-layer recurrence is worth more than any single-record simplification because it costs nothing against the 16 MB budget. The only cost is step time.

Step-time accounting: PR #1493 drops from ~6,927 steps (no recurrence) to ~4,550 steps (recurrence on). That is roughly a 34 % reduction in steps in exchange for 6 extra virtual layers on ~65 % of the training schedule. The tradeoff is net-positive.

---

## Where the Budget Actually Lives Now

```
Compute budget (600 s train):
  Looping region (65 %):  ~4,550 steps × ~129 ms → dominates
  Pre-loop region (35 %): ~same step count on equivalent non-recurrent compute

Artifact budget (16 MB):
  MLP bank:               71 %  ← biggest single lever
  Attention Q/K/V/O:      19 %
  Token embedding:         7 %  ← grew with SP8192
  Everything else:         3 %

Eval budget (600 s):
  Sliding window:  ~130 s
  Legal TTT:       ~370 s   ← ~0.002 BPB purchase
  Headroom:         ~100 s
```

---

## What Changes the Tradeoff Curve

| Lever | Acts on | Artifact cost | Compute cost | Quality cost |
|---|---|---|---|---|
| Wider MLP (4×→5×) | Capacity | Large (+) | Moderate (+) | Better → but unclear if worth the artifact |
| More layers | Capacity | Large (+) | Large (+) | Historically strong lever, but already saturated |
| More recurrence depth | Virtual depth | 0 | Large (+) | Diminishing past 3 layers; step time matters |
| Parallel residuals | Representation | 0 | ~0 | +0.002 to 0.003 BPB, single-shot |
| MoE experts | Capacity | Depends | Mostly routing | High-variance; must beat shared-weight recurrence net of routing overhead |
| Bit-level compression (ternary, 1-bit) | Artifact headroom | −− | 0 (post-train) | Unlocks more params/features, but the non-record track shows it's not cheap inside 10 min |
| Longer eval TTT | Eval quality | 0 | 0 (within eval budget) | ~0.002 BPB, already near ceiling |
| SDClip tuning (per-group k) | Artifact headroom | 0 | 0 | Flagged as an explicit future direction in PR #1412 |

### Implications for a MoE baseline

- Any MoE design must budget against an **SP8192-scale embedding (~7 % of artifact)** and an **MLP bank at ~71 % of artifact**. Routed experts will cost more than a shared MLP unless experts are heavily rank-factored or bit-shared.
- Depth recurrence (layers 3–5, shared params) currently delivers ~−0.005 BPB for zero artifact cost. MoE must beat this **after** routing overhead and load-balancing losses are included, not just match the raw depth increase.
- Parallel residuals (layer 7+) is an almost-free representational win. It composes with MoE: routed MLP in the parallel-residual region is a natural first experiment.

---

## Appendix — Previous Profile (PR #1019, 1.1147 BPB)

Retained for reference. **Do not benchmark against this** — the 0.0337 BPB gap makes it an outdated baseline. Kept because the user's earlier experiment folders (`2026-04-02_SharedMLP_Rank100_16L/`, etc.) were scaled against these numbers.

### Compute breakdown (per step, 8× H100)

```
                              GFLOP/layer    % of compute
MLP (up + down projections):     3,401.6       58.0%     ← dominates
Attention projections (Q+K+V+O): 1,700.8       29.0%
Flash Attention 3:                 510.2        8.7%     ← cheap
Overhead (norms, scales, etc):     255.1        4.3%
```

Total per step: 17.60 TFLOP (fwd + bwd). Step time 86.7 ms on 8× H100 → 6,927 steps in 600 s.

### Artifact breakdown (15.86 MB actual)

```
                          Params      Before LZMA    After LZMA (est.)    % of artifact
MLP (11 independent):     17.30M      17.35 MB       ~10.0 MB             63%
Attention (Q+K+V+O):       8.65M       8.68 MB       ~5.0 MB              31%
Other (emb, bigram, etc):   1.12M       1.29 MB       ~0.7 MB              5%
Code:                       —           —              0.10 MB              1%
```

### Why MLP dominated (and still dominates)

- Per-layer MLP (3×): up [1536, 512] = 786 432 + down [512, 1536] = 786 432 = 1 572 864 params/layer × 11 = 17.30 M.
- Per-layer attention: Q [512, 512] + K [256, 512] + V [256, 512] + O [512, 512] = 786 432 params/layer × 11 = 8.65 M.
- MLP was 2× attention params/layer at 3× expansion. At 4× (current SOTA), MLP per layer is ~2.10 M, attention stays at 0.79 M — the ratio widens to ~2.7×.

### Our earlier experiment comparison (obsolete, kept for reference)

| Config | Compute vs old SOTA | Steps vs old SOTA | Pre-quant BPB |
|--------|:---:|:---:|:---:|
| old SOTA (3×, independent) | 1.00× | 6,927 | 1.1354 |
| MHA+4×, 6 shared, no adapt | 1.33× | 5,931 (−14 %) | 1.1443 (+0.009) |
| GQA+5×, 6 shared, no adapt | 1.41× | 5,612 (−19 %) | 1.1472 (+0.012) |
| GQA+2.5×, 6 shared, r51 | 0.90× | 7,289 (+5 %) | 1.1714 (+0.036) |
| GQA+3×, 6 shared, r42 | 1.00× | ~6,920 | untested |

Reminder: the old SOTA baseline (1.1354 pre-quant) is now ~0.05 BPB behind the current post-TTT SOTA (1.0810). Any new experiment should be benchmarked against the PR #1493 stack, not these numbers.
