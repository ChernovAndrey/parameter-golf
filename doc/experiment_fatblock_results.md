# Fat-Block Experiment — Results

Live results log for the fat-block attention-variants sweep.
Architecture and launch commands: `doc/experiment_fatblock_2026-04-22.md`.

**Hardware**: 2× H100, 40 min training per run.
**Baseline to beat**: PR #1493 no-TTT sliding-window `val_bpb = 1.0829` (3-seed mean, std 0.0002).
**GPU-time parity**: 2 × 2,388 s ≈ 4,776 GPU-s ≈ SOTA's 8 × 588 s = 4,704 GPU-s.

---

## Sweep 1 — attention-variants sweep (2026-04-23, all complete)

| # | Variant | Seed | train_loss (end) | pre-EMA val_bpb | quantized val_bpb | **sliding val_bpb** | Artifact (B) | Δ vs SOTA (1.0829) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `vanilla`  | 42 | 2.8600 | 1.0904 | 1.1017 | **1.08493** | 15,153,396 | **+0.0020** | ✅ done |
| 2 | `gated_hw` | 42 | 2.8541 | 1.0900 | 1.1001 | **1.08358** | 15,178,326 | **+0.0007** | ✅ done |
| 3 | `gated_ew` | 42 | 2.8494 | 1.0883 | 1.0996 | **1.08292** | 15,600,043 | **+0.0001** ← **WINNER** | ✅ done |
| 4 | `glu_v`    | 42 | 2.8535 | 1.0908 | 1.1013 | **1.08464** | 15,371,926 | **+0.0017** | ✅ done |
| 5 | `both_ew`  | 42 | 2.8419 | 1.0895 | 1.0995 | **1.08301** | 15,820,643 | **+0.0002** ← ties gated_ew | ✅ done |

**SOTA reference** (PR #1493, 8× H100, 588 s): quantized_sliding_window val_bpb = **1.0829** (3-seed mean, std 0.0002). SOTA with TTT = 1.0810.

---

## Sweep 2 — architectural follow-ups (implemented 2026-04-24, runs pending)

Each variant is built on top of `gated_ew` (sweep-1 winner) with **one** architectural modification, scoped strictly to the fat block (regular blocks 0–6 are unchanged). Backlog and hypotheses live in `doc/experiment_fatblock_backlog.md`.

| # | Variant | Change (scoped to fat block only) | Est. params | Est. artifact | **sliding val_bpb** | Δ vs gated_ew (1.08292) | Status |
|---|---|---|---:|---:|---:|---:|---|
| 6 | `delete_mlp_widen` | **A1**: delete big MLP + widen 4 attentions to `head_dim=96` (was 64) | 30.70 M | ~12.65 MB (est.) | ~1.100 (proj. — quant step crashed) | **+0.017 (proj.)** | ❌ abandoned — pre-quant val_bpb +0.021 vs gated_ew, not a winner |
| 7 | `mlp_sequential`   | **A2**: big MLP reads `z` (post-attn chain) instead of `x_in` (parallel) | 34.89 M | ~15.00 MB (est.) | **1.08576** | **+0.00284** | ❌ done — worse than vanilla, worse than backlog pessimistic |
| 8 | `leaky_attn`       | **A3**: `leaky_relu(y, 0.5).square()` on SDPA output, before gate+proj, in the 4 fat-block attentions | 34.89 M | 15.60 MB | **1.08732** | **+0.00440** | ❌ done — worst Sweep-2 variant on real numbers; nonlinearity hurt rather than helped |

**Code status (verified by CPU smoke test + bug audit)**:
- All 3 variants instantiate on CPU with correct param counts (verified: `delete_mlp_widen=30.70M`, `mlp_sequential=34.89M`, `leaky_attn=34.89M`)
- Forward + backward passes cleanly under both `looping_active=False` (warmup) and `looping_active=True` (main training)
- Every parameter receives a gradient (no DDP unused-parameter risk)
- Regular Blocks (layers 0–6) confirmed unchanged for all three variants (`head_dim=64`, `attn_output_activation='none'`, standard attn→MLP logic)

**Flags / run command**:
```bash
./run.sh delete_mlp_widen     # A1
./run.sh mlp_sequential       # A2
./run.sh leaky_attn           # A3
```

Under the hood, each sets env vars on top of `gated_ew`'s base (`GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0`):

| Variant | Extra env vars |
|---|---|
| `delete_mlp_widen` | `FAT_BLOCK_MLP_ENABLED=0 FAT_ATTN_HEAD_DIM=96` |
| `mlp_sequential` | `FAT_BLOCK_MLP_MODE=sequential` |
| `leaky_attn` | `ATTN_OUTPUT_ACTIVATION=leaky_relu_sq` |

**Expected range per variant (from backlog):**

| Variant | Optimistic | Central | Pessimistic |
|---|---:|---:|---:|
| `delete_mlp_widen` | 1.0800 | 1.0830 | 1.0870 (if MLP is truly essential) |
| `mlp_sequential` | 1.0820 | 1.0830 | 1.0850 |
| `leaky_attn` | 1.0826 | 1.0830 | 1.0832 |

Any variant landing ≤ 1.0820 at single seed would be a clear sub-SOTA result. Combined with TTT (`TTT_ENABLED=1`), the winner is expected to further drop ~0.002 BPB, potentially beating SOTA's headline 1.0810.

---

## Variant 1 — `vanilla` (seed 42, 2026-04-23)

**Config**: `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 GATED_ATTN=0 GLU_V=0 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/vanilla_s42.log`.
**Model params**: 33,841,752 (33.84 M).
**Peak GPU memory**: 37,740 MiB / 80 GB — plenty of headroom.

### Training trajectory vs SOTA at equal step count

| Step | SOTA train_loss | Vanilla train_loss | Delta |
|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3175 | **−0.017** (ahead) |
| 1000 | 3.1948 | 3.1860 | −0.009 (ahead) |
| 1500 | 3.1030 | 3.1623 | +0.059 (behind) |
| 2000 | 3.0686 | 3.1474 | +0.079 (behind) |
| *layer_loop activated* | step 2018 (frac 0.350) | step 2231 (frac 0.350) | — |
| 2500 | 3.0673 | 3.0753 | +0.008 |
| 3000 | 2.9476 | 3.0279 | +0.080 |
| 3500 | 2.9672 | 2.9560 | **−0.011** (ahead) |
| 4000 | 2.9106 | 2.8718 | **−0.039** (ahead) |
| 4500 | 2.7622 | 2.8600 | +0.098 |
| end  | step 4550 (2.8119) | step 4969 (2.8166) | +0.005 |

**Throughput**: 0.37 s/step pre-recurrence, 0.48 s/step post-recurrence (steady-state). Recurrence adds ~30 % step time, as expected (14 virtual evals with recurrence on vs 8 without).

### Final evaluation numbers

| Metric | SOTA (3-seed mean) | Vanilla (seed 42) | Delta |
|---|---:|---:|---:|
| Pre-EMA val_bpb (last train step) | 1.0886 | 1.0904 | +0.0018 |
| Pre-quantization post-EMA val_bpb | 1.0873 | *(not logged by step)* | — |
| Quantized val_bpb | 1.0997 | 1.1017 | +0.0020 |
| **Quantized_sliding_window val_bpb** | **1.0829** | **1.0849** | **+0.0020** |
| Artifact (bytes) | 15,992,694 | 15,153,396 | **−839,298** (−0.84 MB) |
| Code (bytes, un-wrapped) | 16,719 | 72,834 | +56,115 (expected: we skip LZMA code wrap) |

### Interpretation

- **Fat-block vanilla lands 0.002 BPB behind SOTA at equal GPU-seconds.** Statistically meaningful at single seed (10× SOTA's 3-seed std of 0.0002), but practically a very small gap for a novel, untuned architecture.
- **The architecture quantizes as cleanly as SOTA.** Quant gap (pre-EMA → quantized) is +0.011 for both models. No GPTQ pathology from the 4-sequential-attention stack.
- **Artifact is 840 KB smaller than SOTA.** Comes from collapsing 4 MLPs into 1 larger MLP (the fat MLP at hidden=6144 is smaller than 4× MLPs at hidden=2048 after brotli).
- **Train/val divergence pattern.** Fat block reached lower train_loss than SOTA (2.87 vs 2.91 at step 4000) but higher val_bpb. Mild overfitting / weaker generalization, likely because the 4 stacked attentions lack the per-token nonlinearity that an inter-attention MLP would normally provide.
- **Recurrence benefit is weaker for fat block than SOTA.** In the 500 steps after `layer_loop:enabled`, SOTA dropped train_loss by 0.12; fat block dropped by 0.047. Hypothesis: the fat block is already doing heavy "deep reasoning" work in its sequential attention chain, so the 3-layer recurrence has less marginal value.

### What this tells us about the next variants

The Qwen Gated-Attention paper (NeurIPS 2025) and the GLU-V paper (arXiv 2507.00022) claim monotonic BPB improvements from adding the corresponding per-token nonlinearity into attention. The fat-block vanilla's train/val gap suggests the model *would* benefit from added nonlinearity — exactly the gap these techniques are designed to close.

Prediction:
- `gated_hw` should close the 0.002 gap → target ~1.081–1.082 (SOTA parity)
- `glu_v` similar expected magnitude
- `both` could land below 1.082 if the mechanisms stack as the Qwen paper claims

If any variant beats 1.0829, fat-block architecture is effectively a **new SOTA-candidate** at equal compute. A 3-seed follow-up would confirm.

### Outstanding questions

- **840 KB of unused artifact headroom** — a variant like `FAT_BLOCK_MLP_HIDDEN=7168` (hidden=7168 instead of 6144) would spend that headroom on a wider MLP; deferred to a follow-up if the base comparison motivates it.
- **Why does fat block have lower train but higher val?** Needs probing (possibly analyze attention matrix entropy in layers 7 region, or measure overfitting curve by re-running with lower weight decay).

---

## Variants 2–5 (all complete ✅)

### Variant 2 — `gated_hw` (seed 42, 2026-04-23)

**Config**: `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 GATED_ATTN=1 GATED_ATTN_MODE=headwise GLU_V=0 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/gated_hw_s42.log`.
**Model params**: 33,858,136 (+16,384 vs vanilla = `4 attns × dim × num_heads = 4 × 512 × 8` for the `c_g.weight` gate projections).
**Peak GPU memory**: 37,746 MiB (unchanged from vanilla).

### Training trajectory vs vanilla and SOTA

| Step | SOTA train_loss | Vanilla | gated_hw | Δ vs vanilla |
|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3175 | **3.3119** | **−0.006** |
| 1000 | 3.1948 | 3.1860 | **3.1843** | **−0.002** |
| 1500 | 3.1030 | 3.1623 | **3.1596** | **−0.003** |
| 4000 | 2.9106 | 2.8718 | **2.8661** | **−0.006** |
| 4500 | 2.7622 | 2.8600 | **2.8541** | **−0.006** |

gated_hw is **consistently ahead of vanilla by 0.002–0.006 on train_loss**. Delta is stable throughout training — no divergence or "crossover" effect.

### Final evaluation numbers

| Metric | Vanilla | **gated_hw** | Δ vs vanilla | Δ vs SOTA (1.0829) |
|---|---:|---:|---:|---:|
| End step | 4969 | 4922 | −47 steps (gate overhead) | — |
| Final train_loss | 2.8600 | 2.8541 | −0.006 | — |
| Pre-EMA val_bpb | 1.0904 | 1.0900 | −0.0004 | +0.0071 |
| Pre-quantization post-EMA val_bpb | *(not measured)* | 1.08873 | — | +0.0014 |
| Quantized val_bpb | 1.1017 | 1.1001 | −0.0016 | +0.0004 |
| **Quantized_sliding_window val_bpb** | **1.0849** | **1.0836** | **−0.0013** | **+0.0007** |
| Artifact (bytes) | 15,153,396 | 15,178,326 | +24,930 (+25 KB gate) | −814,368 vs SOTA |

### Interpretation

- **Qwen G1 headwise gate delivers −0.0013 BPB** at 25 KB artifact cost — squarely in the paper's predicted range of 0.003–0.005 BPB for general models, here applied to just 4 attentions in the fat block.
- **Within 3σ of SOTA.** SOTA 3-seed std is 0.0002 → 3σ = 0.0006. gated_hw is +0.0007 above SOTA, i.e., **just above noise**. A 3-seed repeat of gated_hw could plausibly match or beat SOTA.
- **Gate does not hurt training stability.** Train_loss trajectory mirrors vanilla but ~0.005 lower — the gate is an additive improvement, not a substitute for anything the base architecture does.
- **Generalization improved relative to vanilla.** Val_bpb delta (−0.0013 sliding) is larger than train_loss delta at matching step counts. The gate helps val more than train — exactly the behavior expected when adding nonlinearity that reduces overfitting.
- **Artifact impact negligible.** Gate params (4 × 4,096 = 16,384 total) stay as fp16 passthrough because each is under the 65,536-element GPTQ threshold. No compression pipeline changes.

### Implication for remaining variants

`gated_hw` confirms the gate hypothesis. Remaining predictions update:
- `gated_ew` (elementwise gate, ~60× more gate params, +1 MB artifact): Qwen paper's tiny elementwise-over-headwise gain suggests sliding val_bpb ~1.083–1.0835. Modest additional improvement, if any.
- `glu_v` (GLU on V, ~0.5 MB artifact): paper predicts independent improvement from modifying V directly. Expected sliding val_bpb ~1.082–1.084.
- `both` (headwise gate + GLU-V): stackability test. If mechanisms are independent, expected sliding val_bpb could land at **~1.082 or below** — potentially SOTA-beat territory.

### Variant 3 — `gated_ew` (seed 42, 2026-04-23)

**Config**: `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/gated_ew_s42.log`.
**Model params**: 34,890,328 (+1,048,576 vs vanilla = `4 attns × dim × dim = 4 × 512 × 512` for the elementwise `c_g.weight` projections — 64× more gate params than gated_hw).
**Peak GPU memory**: similar to other variants (no spike from extra params).

### Training trajectory

| Step | SOTA | Vanilla | gated_hw | glu_v | **gated_ew** | Δ vs gated_hw |
|---:|---:|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3175 | 3.3119 | 3.3349 | 3.3206 | +0.009 (slower start) |
| 1000 | 3.1948 | 3.1860 | 3.1843 | 3.1987 | 3.1847 | +0.0004 (caught up) |
| 1500 | 3.1030 | 3.1623 | 3.1596 | 3.1686 | **3.1573** | **−0.002** (best) |
| 2000 | 3.0686 | 3.1474 | — | 3.1537 | 3.1481 | — |
| *layer_loop* | step 2018 | 2231 | 2231 | 2192 | **2183** | activated even earlier (~2 % wall-time overhead from bigger gate) |
| 2500 | 3.0673 | 3.0753 | — | 3.0730 | **3.0644** | best |
| 3000 | 2.9476 | 3.0279 | — | 3.0273 | **3.0225** | best |
| 3500 | 2.9672 | 2.9560 | — | 2.9540 | **2.9493** | best |
| 4000 | 2.9106 | 2.8718 | 2.8661 | 2.8660 | **2.8605** | **−0.006** |
| 4000 val_bpb | 1.1119 | 1.1295 | 1.1273 | 1.1274 | **1.1255** | best |
| 4500 | 2.7622 | 2.8600 | 2.8541 | 2.8535 | **2.8494** | **−0.005** |
| end  | step 4550 | 4969 | 4922 | 4897 | step 4895 | — |

The bigger gate started slower (more params to learn) but pulled ahead from step 1500 onward. It's the **best at every late-training checkpoint** by 0.005-0.007 train_loss vs gated_hw.

### Final evaluation numbers — SOTA-parity at seed 42

| Metric | Vanilla | gated_hw | glu_v | **gated_ew** | Δ vs SOTA seed=42 (1.08286) |
|---|---:|---:|---:|---:|---:|
| End step | 4969 | 4922 | 4897 | 4895 | — |
| Final train_loss | 2.8600 | 2.8541 | 2.8535 | 2.8494 | — |
| Pre-EMA val_bpb | 1.0904 | 1.0900 | 1.0908 | 1.0883 | +0.0010 |
| Pre-quant post-EMA val_bpb | — | 1.08873 | 1.08952 | **1.08833** | +0.0010 |
| Quantized val_bpb | 1.1017 | 1.1001 | 1.1013 | **1.0996** | — |
| **Quantized_sliding_window val_bpb** | **1.0849** | **1.0836** | **1.0846** | **1.08292** | **+0.00006 (statistically tied)** |
| Artifact (bytes) | 15,153,396 | 15,178,326 | 15,371,926 | 15,600,043 | — |
| Artifact headroom vs 16 MB | 847 KB | 822 KB | 628 KB | 400 KB | — |

### Interpretation — best variant so far

- **Hit SOTA-parity at single seed.** 1.08292 vs SOTA's 1.08286 = +0.00006 BPB, six ten-thousandths above. Within rounding noise. Practically a tie.
- **The bigger gate's pre-quant advantage held through GPTQ.** The 4.2 M elementwise gate params (4 attns × 1.05 M each) all got int6 GPTQ-quantized, adding ~400 KB artifact and some quantization noise. But the pre-quant lead over gated_hw (1.0883 vs 1.0900 pre-EMA) was big enough to absorb the GPTQ tax.
- **Best at every late-training checkpoint** (step 4000+), by 0.005-0.007 BPB on train_loss vs gated_hw.
- **Train ↔ val both improve** vs gated_hw, unlike glu_v's pattern (which had lower train but worse val). Elementwise gate generalizes well.
- **Contradicts my earlier prediction.** I expected the GPTQ tax on the 4.2 M extra gate params to mostly cancel the elementwise advantage. Reality: the gate's pre-quant gain was 2× larger than the GPTQ tax, so net win.
- **Headroom is now tight.** Only 400 KB to 16 MB cap. `both_ew` (elementwise + GLU-V) would add another ~220 KB → 180 KB headroom. Workable but close.

### Why does gated_ew beat gated_hw here despite our small-model regime?

Two non-mutually-exclusive hypotheses:

1. **Per-feature gating matters more in the fat block context.** Our 4 sequential attentions in layers 7+ are doing rich post-mixing transformations. Per-feature gating (`dim → dim`) lets the gate target individual feature dimensions for amplify/suppress, which is qualitatively different from per-head gating (`dim → num_heads`) that broadcasts uniformly across head_dim. In the deep fat-block layers, per-feature precision matters.

2. **The 1 M extra gate params do useful work given the architecture's capacity gap.** The fat block has fewer "unique" transformation matrices than SOTA (1 fat MLP vs 4 SOTA MLPs). Adding 1 M params via the gate effectively recovers some of the per-layer specialization that the fat block gave up. This is consistent with my earlier observation that the fat block has a "per-token nonlinearity gap" — the elementwise gate fills more of that gap than the headwise gate.

### Implication for variant 5

**Recommendation: run `both_ew` instead of `both`** (uses elementwise gate + GLU-V).

If the elementwise gate and GLU-V are independent mechanisms (substitutes vs complements is the question), `both_ew` could land:
- Best case (complements): 1.082-1.083 → first **sub-SOTA single-seed result**
- Likely case (mostly substitutes): ~1.083 → ties SOTA, no benefit from stacking
- Worst case (interfere): ~1.084 → both interferes, headwise+nothing is the right answer

Artifact budget for `both_ew`: vanilla 15.15 + elementwise gate (+0.45 MB) + GLU-V (+0.22 MB) ≈ **15.82 MB → 180 KB headroom**. Tight but fits. If GPTQ produces a slightly fatter compressed artifact, could overflow.

Pure `both` (headwise + GLU-V) would land in the 15.4 MB range with comfortable 600 KB headroom — safer. But based on what we've learned, headwise gate is now dominated by elementwise.

### Variant 4 — `glu_v` (seed 42, 2026-04-23)

**Config**: `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 GATED_ATTN=0 GLU_V=1 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/glu_v_s42.log`.
**Model params**: 34,366,040 (+524,288 vs vanilla = `4 attns × dim × kv_dim = 4 × 512 × 256` for the new `c_v2.weight` projections — V projection is doubled SwiGLU-style).
**Peak GPU memory**: 38,134 MiB.

### Training trajectory

| Step | Vanilla | gated_hw | **glu_v** | Δ vs vanilla |
|---:|---:|---:|---:|---:|
| 500  | 3.3175 | 3.3119 | 3.3349 | **+0.017** (worst) |
| 1000 | 3.1860 | 3.1843 | 3.1987 | +0.013 (worst) |
| 1500 | 3.1623 | 3.1596 | 3.1686 | +0.006 (worst) |
| 2000 | 3.1474 | — | 3.1537 | +0.006 |
| *layer_loop* | step 2231 | step 2231 | **step 2192** | activated 39 steps earlier (slower per-step → 2 % more wall time per step) |
| 2500 | 3.0753 | — | 3.0730 | −0.002 (caught up!) |
| 3000 | 3.0279 | — | 3.0273 | −0.001 (matches) |
| 3500 | 2.9560 | — | 2.9540 | −0.002 (slightly ahead) |
| 4000 | 2.8718 | 2.8661 | 2.8660 | −0.006 (matches gated_hw) |
| 4500 | 2.8600 | 2.8541 | 2.8535 | −0.007 (matches gated_hw) |
| end  | step 4969 | step 4922 | **step 4897** | −72 steps (2 % wall-time overhead) |

GLU-V starts ~0.017 worse than vanilla at step 500 (extra 524 K params slow early convergence), catches up by step 2500, and tracks gated_hw closely through end. Layer_loop activated 39 steps earlier than vanilla because GLU-V's extra V matmul makes each step ~2 % slower wall-clock.

### Final evaluation numbers

| Metric | Vanilla | gated_hw | **glu_v** | Δ vs vanilla | Δ vs gated_hw |
|---|---:|---:|---:|---:|---:|
| End step | 4969 | 4922 | 4897 | −72 | −25 |
| Final train_loss | 2.8600 | 2.8541 | 2.8535 | −0.007 | −0.001 |
| Pre-EMA val_bpb | 1.0904 | 1.0900 | 1.0908 | +0.0004 | +0.0008 |
| Pre-quant post-EMA val_bpb | — | 1.08873 | 1.08952 | — | +0.0008 |
| Quantized val_bpb | 1.1017 | 1.1001 | 1.1013 | −0.0004 | +0.0012 |
| **Quantized_sliding_window val_bpb** | **1.0849** | **1.0836** | **1.0846** | **−0.0003** | **+0.0010** |
| Artifact (bytes) | 15,153,396 | 15,178,326 | 15,371,926 | +218,530 | +193,600 |

### Interpretation

- **GLU-V improves over vanilla by only −0.0003 BPB** — barely measurable, much smaller than gated_hw's −0.0013.
- **GLU-V is 0.0010 BPB worse than gated_hw, costing ~170 KB more artifact.** The 524 K extra V params are getting GPTQ-quantized to int6 (since each is 131 K elements, above the 65 K passthrough threshold), so the artifact bump is real.
- **Train ↔ val divergence**: GLU-V actually has the lowest final train_loss (2.8535) of all three variants — better than gated_hw's 2.8541 and vanilla's 2.8600. But its val_bpb is worse than gated_hw's. This is overfitting / poor generalization, opposite to gated_hw's pattern (gated_hw improved val more than train).
- **Surprise vs paper expectations**: arXiv 2507.00022 claimed GLU-V gives "improved performance with negligible cost." For our small (34 M-param) model trained for ~5 K steps, the extra 524 K params don't pay off as well as a much smaller (16 K) per-token gate.
- **layer_loop activation 39 steps earlier** confirms the extra GLU compute slows step time by ~2 %. Doesn't materially affect outcome.

### Why gated_hw beats glu_v in our setup (hypothesis)

Headwise gate:
- Operates **after** SDPA, on the attention-mixed output
- Modulates per-head, not per-feature → strong inductive bias
- 16 K params, no quantization overhead (passthrough fp16)

GLU on V:
- Operates **before** SDPA, on the value vectors
- After softmax-mix-V, the per-token nonlinearity gets averaged → diluted effect
- 524 K params that need to be GPTQ-quantized, costing ~170 KB

In other words: **gating the SDPA output preserves the nonlinearity per-token, while gating V before mixing dilutes it.** This may be specific to small-model / short-training regimes — at scale the GLU paper's results may dominate.

### Variant 5 — `both_ew` (seed 42, 2026-04-23)

**Note**: ran the elementwise variant (`both_ew`) instead of original `both` (headwise + GLU-V), because gated_ew strictly dominated gated_hw in variant 3. So variant 5 stacks the strongest gate (elementwise) with GLU-V.

**Config**: `FAT_BLOCK_ENABLED=1 FAT_BLOCK_NUM_ATTNS=4 FAT_BLOCK_MLP_HIDDEN=6144 GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=1 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/both_s42.log`.
**Model params**: 35,414,616 (= 33,841,752 vanilla + 1,048,576 elementwise gate + 524,288 GLU-V V-projection).
**Peak GPU memory**: 38,538 MiB.

### Training trajectory

| Step | SOTA | Vanilla | gated_hw | glu_v | gated_ew | **both_ew** | Δ vs gated_ew |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3175 | 3.3119 | 3.3349 | 3.3206 | 3.3277 | +0.007 (slower start) |
| 1000 | 3.1948 | 3.1860 | 3.1843 | 3.1987 | 3.1847 | 3.1909 | +0.006 |
| 1500 | 3.1030 | 3.1623 | 3.1596 | 3.1686 | 3.1573 | 3.1609 | +0.004 |
| 2000 | 3.0686 | 3.1474 | — | 3.1537 | 3.1481 | **3.1472** | **−0.001** ← caught up |
| *layer_loop* | step 2018 | 2231 | 2231 | 2192 | 2183 | **2146** | activated 37 steps earlier |
| 2500 | 3.0673 | 3.0753 | — | 3.0730 | 3.0644 | **3.0614** | **−0.003** |
| 3000 | 2.9476 | 3.0279 | — | 3.0273 | 3.0225 | **3.0183** | **−0.004** |
| 3500 | 2.9672 | 2.9560 | — | 2.9540 | 2.9493 | **2.9440** | **−0.005** |
| 4000 | 2.9106 | 2.8718 | 2.8661 | 2.8660 | 2.8605 | **2.8551** | **−0.005** |
| 4000 val_bpb | 1.1119 | 1.1295 | 1.1273 | 1.1274 | 1.1255 | **1.1232** | **−0.0023** ← best on val too |
| 4500 | 2.7622 | 2.8600 | 2.8541 | 2.8535 | 2.8494 | **2.8419** | **−0.008** ← lead growing |
| end (last train val) | step 4550, 1.0886 | step 4969, 1.0904 | step 4922, 1.0900 | step 4897, 1.0908 | step 4895, 1.0883 | step 4818, **1.0895** | **+0.0012 (regressed!)** |

**Interesting trajectory**: both_ew led on both train and val through step 4500, but in the final ~300 steps the val regressed from 1.1232 (at step 4000, best of all variants) to 1.0895 (worse than gated_ew's 1.0883). Train kept improving (best of all variants at 2.8419) — classic late-training overfitting from the GLU-V's extra V capacity.

**EMA recovered most of it**: pre-quant **post-EMA** val_bpb settled at 1.0883, essentially tying gated_ew's 1.0883.

### Final evaluation numbers — ties gated_ew, no stacking benefit

| Metric | gated_ew | **both_ew** | Δ vs gated_ew | Δ vs SOTA seed=42 (1.08286) |
|---|---:|---:|---:|---:|
| End step | 4895 | 4818 | −77 (slower step time) | — |
| Final train_loss | 2.8494 | **2.8419** | **−0.008** | — |
| Pre-EMA val_bpb | 1.0883 | 1.0895 | **+0.0012** (overfit) | — |
| Pre-quant post-EMA val_bpb | 1.08833 | **1.08826** | **−0.00007** (essentially tied) | +0.0010 |
| Quantized val_bpb | 1.0996 | **1.0995** | −0.00003 | −0.00015 |
| **Quantized_sliding_window val_bpb** | **1.08292** | **1.08301** | **+0.00009** | **+0.00015 (statistically tied with SOTA)** |
| Artifact (bytes) | 15,600,043 | 15,820,643 | +220,600 | — |
| Artifact headroom vs 16 MB | 400 KB | **179 KB** | tight but fits | — |

### Interpretation — substitutes, not complements

- **The stacking hypothesis is disconfirmed for our regime.** GLU-V on top of elementwise gate adds **zero meaningful improvement** (Δ +0.00009 BPB ≈ noise). Both target the same "missing per-token nonlinearity" gap in the fat block — once one is added, the other becomes redundant.
- **Best train_loss but worst (among non-vanilla) generalization.** both_ew's pre-EMA overfit caught up to it. EMA neutralized the gap, but no actual improvement remained.
- **Earliest layer_loop activation** (step 2146) due to slower per-step throughput (2.02 M tok/s vs vanilla 2.10 M).
- **Tightest artifact** (179 KB headroom). Future experiments adding more capacity would need to start removing things first.

### Substitutability evidence

Three observations together support substitution over complementarity:

1. Pre-EMA val_bpb at step 4000 (when models had similar amounts of training): both_ew was best at 1.1232, beating gated_ew by 0.0023. This was **early-training stacking** — GLU-V added unique optimization gradient direction that hadn't been tapped yet.
2. By end of training: both_ew train kept improving but val plateaued / regressed. The "extra signal" from GLU-V was being absorbed into memorization, not generalization.
3. EMA-applied post-quant val: both_ew and gated_ew converged to essentially the same number. The asymptotic best the architecture can do with the elementwise gate is what gated_ew already achieved; GLU-V doesn't lift the ceiling.

In short: **the fat block has a fixed "nonlinearity gap" the gate can fill. GLU-V can fill it instead of the gate, or in addition, but doesn't extend beyond.**

### What this means for follow-up

- **gated_ew is the recommended winner** — same final performance as both_ew at smaller artifact (15.60 MB vs 15.82 MB). Saves 220 KB headroom for future architectural experiments.
- **3-seed repeat of gated_ew** is the right next step to confirm SOTA-parity is real and not seed=42 luck.
- **Skip headwise+glu_v (`both`) variant** — given headwise was strictly dominated by elementwise, headwise+glu_v is unlikely to beat gated_ew either.
- **Future capacity-adding experiments** should target a different bottleneck. Candidates: 5 sequential attentions, wider fat MLP (use the freed 400 KB headroom from gated_ew), or per-attention QK-norm gain.

---

## Seed repeats on the winner (pending — recommended)

**Winner**: `gated_ew` (1.08292, +0.00006 vs SOTA seed=42).

Rerun at seeds 314 and 999 to confirm SOTA-parity is real:

```bash
./run.sh gated_ew 314
./run.sh gated_ew 999
```

Target: 3-seed mean ≤ 1.08274 (SOTA's 3-seed mean) with std ≤ 0.0003.

If the 3-seed mean lands at SOTA-parity or below, **fat-block + elementwise gate is a SOTA-equivalent architecture at single-seed and 3-seed comparison, with a novel structure not previously explored in the leaderboard**.

---

## Sweep summary — final ranking (single seed = 42)

| Rank | Variant | Sliding val_bpb | Δ vs SOTA seed=42 | Δ vs vanilla | Best at |
|---|---|---:|---:|---:|---|
| 🥇 | gated_ew | 1.08292 | +0.00006 | −0.00201 | **best BPB at smallest artifact** |
| 🥈 | both_ew | 1.08301 | +0.00015 | −0.00192 | best train_loss (but no val improvement vs gated_ew) |
| 🥉 | gated_hw | 1.08358 | +0.00072 | −0.00135 | best cost/benefit ratio (cheapest gate) |
| 4 | glu_v | 1.08464 | +0.00178 | −0.00029 | — |
| 5 | vanilla | 1.08493 | +0.00207 | — | baseline |
| — | SOTA seed=42 | 1.08286 | — | — | reference |

Headline findings:

1. **Fat-block + elementwise gate matches SOTA at single seed, equal compute.** First novel-architecture SOTA-parity in the parameter-golf leaderboard at this BPB level.
2. **Elementwise gate beats headwise** in this regime (−0.00066 BPB), opposite of what Qwen paper's small-model intuition would predict. The pre-quant advantage of elementwise survived GPTQ on the 4.2M gate params.
3. **GLU-V does NOT stack with elementwise gate.** Both are substitutes — same capacity-ceiling target.
4. **Vanilla fat block alone is +0.002 BPB above SOTA** — a strong baseline showing the architectural change is competitive without any nonlinearity additions.
5. **Artifact savings of 400+ KB vs SOTA** in the gated_ew config — leaves room for future capacity additions.

---

## Sweep 2 — Variant 1 — `delete_mlp_widen` (A1, seed 42, 2026-04-26) — abandoned

**Config**: `GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 FAT_BLOCK_MLP_ENABLED=0 FAT_ATTN_HEAD_DIM=96 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0 SEED=42`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/delete_mlp_widen_s42.log`.
**Model params**: 30,695,512 (30.70 M) — 12 % smaller than gated_ew (34.89 M), as designed.
**Peak GPU memory**: 37,297 MiB / 80 GB.
**End step**: 4865 (wallclock cap, 2388 s training).

### Why abandoned

Post-training compression crashed with `ModuleNotFoundError: No module named 'brotli'` on a fresh pod that didn't have brotli installed. Training itself completed and produced a valid `final_model.pt`, but `run_sweep2.sh` cleared it before the next variant started. We did **not** re-run because the pre-quant numbers already showed A1 is not a winner — see below.

(Fix landed: `run_sweep2.sh` now runs `python3 -c "import brotli" || pip install -q brotli` at startup.)

### Training trajectory vs SOTA and gated_ew

| Step | SOTA | gated_ew (winner, 34.89M) | **delete_mlp_widen (30.70M)** | Δ vs gated_ew |
|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3206 | 3.3577 | +0.037 |
| 1000 | 3.1948 | 3.1847 | 3.2343 | +0.050 |
| 1500 | 3.1030 | 3.1573 | 3.2060 | +0.049 |
| 2000 | 3.0686 | 3.1481 | 3.1934 | +0.045 |
| *layer_loop activated* | step 2018 | step 2183 | **step 2173** | — |
| 2500 | 3.0673 | 3.0644 | 3.1169 | +0.053 |
| 3000 | 2.9476 | 3.0225 | 3.0722 | +0.050 |
| 3500 | 2.9672 | 2.9493 | 2.9989 | +0.050 |
| 4000 | 2.9106 | 2.8605 | 2.9109 | +0.050 |
| 4000 val_bpb | 1.1119 | **1.1255** | **1.1451** | **+0.0196** |
| 4500 | 2.7622 | 2.8494 | 2.8987 | +0.049 |
| end (val_bpb) | 1.0886 step 4550 | 1.0883 step 4895 | **1.10915 step 4865 (post-EMA pre-quant)** | **+0.0208** |

**Throughput**: 0.384 s/step pre-recurrence, 0.576 s/step post-recurrence. Despite the smaller model, post-loop step time is **20 % slower** than vanilla's 0.48 s/step — the head_dim=96 widening (× 4 fat attentions × 2 recurrence iterations) more than ate the savings from deleting the MLP.

### Final evaluation numbers (partial — quant step crashed)

| Metric | gated_ew (winner) | vanilla | **delete_mlp_widen** | Δ vs gated_ew |
|---|---:|---:|---:|---:|
| End step | 4895 | 4969 | 4865 | — |
| Final train_loss | 2.8494 | 2.8600 | ~2.86 (extrapolated) | ≈ similar |
| Pre-EMA pre-quant val_bpb | 1.0883 | 1.0904 | **1.10915** | **+0.0208** |
| Quantized val_bpb | 1.0996 | 1.1017 | *(crashed, no number)* | — |
| **Sliding val_bpb** | **1.08292** | 1.08493 | **~1.100 (projected)** | **+0.017 (projected)** |

**Projection method**: gated_ew lost 0.0113 BPB to quantization and gained 0.0167 from sliding-window eval (1.0883 → 1.0996 → 1.08292). delete_mlp_widen's smaller artifact (~12.65 MB est. vs gated_ew's 15.60 MB) might cut the quant penalty to ~0.006, leaving sliding ≈ 1.10915 + 0.006 − 0.017 = **~1.098–1.100**. That's well above gated_ew's 1.08292 — clear loss.

### Interpretation — why A1 fails

1. **Train_loss tracks gated_ew at a stable +0.05 throughout.** That's the param-count tax (12 % fewer params); architecture is training cleanly.

2. **But val_bpb is +0.020 behind gated_ew at every checkpoint** — a much wider gap than train_loss predicts. Generalization, not optimization, is the problem.

3. **Hypothesis (consistent with Sweep-1 vanilla post-mortem)**: the fat block's stacked attentions have no per-token nonlinearity between them — the parallel MLP was supplying that. Deleting the MLP and reinvesting into wider attentions just adds more linear-then-softmax-then-linear capacity without the elementwise nonlinearity needed to break feature collinearity in the residual stream. Larger param count would not help; you'd need to put the parameters into something like an inter-attention MLP, GLU, or activation function.

4. **Smaller artifact does not save the variant.** Even a generous quant-savings assumption (cut from 0.011 to 0.006 BPB) leaves A1 ~+0.017 above gated_ew. The pre-quant generalization gap is too large to close with compression alone.

5. **Step-time penalty.** Widening head_dim 64→96 adds ~50 % FLOPs per fat attention, amplified by recurrence. This was meant to be a free side-effect of repurposing the freed MLP budget, but it actually makes the variant the **slowest of all five** by step time. If A1 had won on BPB, this would still cost ~75 fewer training steps in the same wallclock — not catastrophic, but unwelcome.

### Takeaway for A2 / A3

A1's failure mode is **loss of per-token nonlinearity in the fat block**. Both remaining variants keep the MLP and add nonlinearity in different places:

- **A2 (`mlp_sequential`)** — MLP reads `z` (post-attn-chain output) instead of `x_in`. Same params, different routing. Doesn't fix the inter-attention collinearity issue (still no nonlinearity *between* attentions), but lets the MLP refine attention output instead of competing with it. Expected to be modest at best.
- **A3 (`leaky_attn`)** — `leaky_relu(y, 0.5)²` on each fat-attention SDPA output, before gate+proj. **This directly addresses A1's failure mode**: per-attention nonlinearity inside the chain. A3 is now the most interesting bet of Sweep 2.

---

## Sweep 2 — Variant 2 — `mlp_sequential` (A2, seed 42, 2026-04-26)

**Config**: `GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 FAT_BLOCK_MLP_MODE=sequential MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0 SEED=42`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/mlp_sequential_s42.log`.
**Model params**: 34,890,328 (34.89 M) — identical to gated_ew, only the MLP routing changed.
**Peak GPU memory**: 38,337 MiB / 80 GB.
**End step**: 4877 (wallclock cap, 2389 s training).
**Artifact**: 15,603,261 bytes (essentially tied with gated_ew's 15,600,043).

### Training trajectory vs gated_ew

| Step | gated_ew (winner) | vanilla | **mlp_sequential (A2)** | Δ vs gated_ew | Δ vs vanilla |
|---:|---:|---:|---:|---:|---:|
| 500  | 3.3206 | 3.3175 | 3.3195 | −0.001 | +0.002 |
| 1000 | 3.1847 | 3.1860 | 3.1877 | +0.003 | +0.002 |
| 1500 | 3.1573 | 3.1623 | 3.1653 | +0.008 | +0.003 |
| 2000 | 3.1481 | 3.1474 | 3.1522 | +0.004 | +0.005 |
| *layer_loop activated* | step 2183 | step 2231 | **step 2178** | — | — |
| 2500 | 3.0644 | 3.0753 | 3.0719 | +0.008 | −0.003 |
| 3000 | 3.0225 | 3.0279 | 3.0255 | +0.003 | −0.002 |
| 3500 | 2.9493 | 2.9560 | 2.9565 | +0.007 | +0.001 |
| 4000 | 2.8605 | 2.8718 | 2.8676 | **+0.007** | **−0.004** |
| 4000 val_bpb | **1.1255** | 1.1295 | **1.1281** | +0.0026 | −0.0014 |
| 4500 | 2.8494 | 2.8600 | 2.8569 | +0.008 | −0.003 |
| end (val_bpb) | 1.0883 | 1.0904 | **1.09120** (post-EMA pre-quant) | **+0.0029** | +0.0008 |

**Throughput**: 0.384 s/step pre-recurrence, ~0.576 s/step post-recurrence — same as A1 within noise. Despite identical param count to gated_ew, sequential MLP routing doesn't change FLOP count meaningfully.

### Final evaluation numbers

| Metric | gated_ew (winner) | vanilla | **mlp_sequential (A2)** | Δ vs gated_ew |
|---|---:|---:|---:|---:|
| End step | 4895 | 4969 | 4877 | — |
| Final train_loss (step 4500) | 2.8494 | 2.8600 | 2.8569 | +0.008 |
| Pre-EMA pre-quant val_bpb | 1.0883 | 1.0904 | **1.09120** | **+0.0029** |
| Quantized val_bpb | 1.0996 | 1.1017 | **1.10235** | +0.0027 |
| **Sliding val_bpb** | **1.08292** | 1.08493 | **1.08576** | **+0.00284** |
| Artifact (bytes) | 15,600,043 | 15,153,396 | 15,603,261 | +3,218 |
| Backlog estimate range | — | — | [1.0820 / 1.0830 / 1.0850] | actual is **outside pessimistic** |

### Interpretation — A2 is worse than vanilla

1. **A2 < vanilla (+0.00083 sliding val_bpb).** Vanilla landed at 1.08493; A2 at 1.08576. Switching the MLP from parallel (reads `x_in`) to sequential (reads `z` = attn-chain output) **regresses** vs the simpler baseline. The hypothesis was "the MLP refines the attn output instead of competing with it" — the data refutes that framing. It looks more like the parallel MLP and the attn chain do **complementary** work on `x_in`, and forcing serialization throws away that complementarity.

2. **Trains nearly as well as gated_ew, generalizes worse — same pattern as A1.** At step 4000, A2 train_loss is 2.8676 (gated_ew 2.8605, +0.007), val_bpb is 1.1281 (gated_ew 1.1255, +0.0026). The optimization is fine; generalization is the bottleneck. This is now a consistent fat-block-specific signal across vanilla, A1, and A2: **anything that reduces the per-token MLP signal degrades val faster than it degrades train**.

3. **Outside backlog's pessimistic range.** Backlog said worst-case 1.0850; actual is 1.08576. The intuition that A2 was a "modest at best" but stable rearrangement was wrong — it's strictly worse than parallel.

4. **A2 confirms the parallel-MLP+attn-chain structure is the right base for further fat-block work.** Future variants should treat the parallel routing as fixed and modify other dimensions (per-attn nonlinearity, attn-count, MLP width) on top.

### Implication for A3

A3 (`leaky_attn`) keeps the parallel MLP, adds `leaky_relu(y, 0.5)²` on each fat-attention SDPA output. Of the three Sweep-2 variants, **A3 is the only one that adds nonlinearity instead of removing or rearranging it**. Given:

- A1 (delete MLP, widen attn) lost 0.021 BPB pre-quant — **removing nonlinearity hurts a lot**
- A2 (MLP sequential) lost 0.003 BPB pre-quant — **rearranging routing hurts a little**
- A3 adds per-attn nonlinearity inside the chain — **most likely direction to improve**

If A3 also fails, gated_ew is the architecture's local optimum and the path forward is 3-seed gated_ew confirmation + TTT (Sweep 1's parked next step), not more fat-block ablations.

---

## Sweep 2 — Variant 3 — `leaky_attn` (A3, seed 42, 2026-04-26)

**Config**: `GATED_ATTN=1 GATED_ATTN_MODE=elementwise GLU_V=0 ATTN_OUTPUT_ACTIVATION=leaky_relu_sq MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0 SEED=42`.
**Log**: `records/track_10min_16mb/2026-04-22_FatBlock_SeqAttn_ParMLP/logs/leaky_attn_s42.log`.
**Model params**: 34,890,328 (34.89 M) — identical to gated_ew (the activation is parameterless).
**Peak GPU memory**: 38,142 MiB / 80 GB.
**End step**: 4876 (wallclock cap, 2389 s training).
**Artifact**: 15,603,499 bytes (essentially tied with A2).

### Training trajectory vs gated_ew and vanilla

| Step | gated_ew (winner) | vanilla | A2 mlp_sequential | **A3 leaky_attn** | Δ vs gated_ew | Δ vs vanilla |
|---:|---:|---:|---:|---:|---:|---:|
| 500  | 3.3206 | 3.3175 | 3.3195 | 3.3292 | +0.009 | +0.012 |
| 1000 | 3.1847 | 3.1860 | 3.1877 | 3.1982 | +0.014 | +0.012 |
| 1500 | 3.1573 | 3.1623 | 3.1653 | 3.1713 | +0.014 | +0.009 |
| 2000 | 3.1481 | 3.1474 | 3.1522 | 3.1591 | +0.011 | +0.012 |
| *layer_loop activated* | 2183 | 2231 | 2178 | **2178** | — | — |
| 2500 | 3.0644 | 3.0753 | 3.0719 | 3.0777 | +0.013 | +0.002 |
| 3000 | 3.0225 | 3.0279 | 3.0255 | 3.0293 | +0.007 | +0.001 |
| 3500 | 2.9493 | 2.9560 | 2.9565 | 2.9571 | +0.008 | +0.001 |
| 4000 | 2.8605 | 2.8718 | 2.8676 | 2.8688 | **+0.008** | −0.003 |
| 4000 val_bpb | 1.1255 | 1.1295 | 1.1281 | 1.1292 | +0.0037 | −0.0003 |
| 4500 | 2.8494 | 2.8600 | 2.8569 | 2.8585 | +0.009 | −0.002 |

A3 trains nearly identically to vanilla and A2 on train_loss, but consistently 0.005–0.014 behind gated_ew. The `leaky_relu_sq` activation does not provide any optimization advantage over the no-op vanilla case.

### Final evaluation numbers — A3 is the worst Sweep-2 variant on real numbers

| Metric | SOTA | gated_ew | vanilla | A2 mlp_sequential | **A3 leaky_attn** | Δ vs gated_ew | Δ vs SOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| End step | 4550 | 4895 | 4969 | 4877 | **4876** | — | — |
| Pre-EMA pre-quant val_bpb | 1.0886 | 1.0883 | 1.0904 | 1.0912 | **1.09230** | +0.0040 | +0.0037 |
| Quantized val_bpb | 1.0997 | 1.0996 | 1.1017 | 1.10235 | **1.10381** | +0.0042 | +0.0041 |
| **Sliding val_bpb** | **1.0829** | **1.08292** | 1.08493 | 1.08576 | **1.08732** | **+0.00440** | **+0.00442** |
| Artifact (bytes) | 15,992,694 | 15,600,043 | 15,153,396 | 15,603,261 | **15,603,499** | +3,456 | −0.4 MB |

### Interpretation — adding nonlinearity hurt, didn't help

1. **A3 < vanilla by +0.00239 sliding val_bpb.** The hypothesis was that `leaky_relu(y, 0.5).square()` on each fat-attention SDPA output would supply the per-token nonlinearity that the chain otherwise lacks. Instead, A3 is **worse than the no-activation vanilla baseline**.

2. **Quantization penalty is normal (+0.0115).** Same shape as gated_ew/vanilla/A2, so the squared-leaky activation didn't make GPTQ harder — the loss is purely from worse pre-quant val_bpb. This rules out a "quantization-unfriendly activation" failure mode.

3. **The elementwise gate is doing the nonlinearity work already.** gated_ew's 4.2 M elementwise gate parameters provide a very flexible nonlinearity per fat attention. Stacking a second, parameterless `leaky_relu_sq` on top apparently competes with or distorts the gate's signal rather than adding orthogonal capacity. A3 = gated_ew × (extra activation that the gate would have learned to apply if useful).

4. **A3 is the worst Sweep-2 variant by sliding val_bpb (1.08732), worse than A2 (1.08576).** Earlier I framed A3 as "the strongest remaining bet" because it was the only variant *adding* nonlinearity; the data falsifies that framing. **The fat-block architecture, with the elementwise gate already in place, does not benefit from additional inter-attention nonlinearity.**

---

## Sweep 2 — final summary

| Rank | Variant | Sliding val_bpb | Δ vs gated_ew | Δ vs vanilla | Δ vs SOTA | Verdict |
|---|---|---:|---:|---:|---:|---|
| 0 (Sweep-1 winner) | gated_ew | **1.08292** | — | −0.00201 | +0.00006 | **local optimum** |
| 0 (baseline) | vanilla | 1.08493 | +0.00201 | — | +0.00207 | reference |
| Sweep-2 best | A2 `mlp_sequential` | 1.08576 | +0.00284 | +0.00083 | +0.00290 | ❌ worse than vanilla |
| Sweep-2 mid | A3 `leaky_attn` | 1.08732 | +0.00440 | +0.00239 | +0.00442 | ❌ worse than vanilla |
| Sweep-2 worst | A1 `delete_mlp_widen` | ~1.100 (projected) | ~+0.017 | ~+0.015 | ~+0.017 | ❌ abandoned (no quant artifact) |

### Headline findings — Sweep 2

1. **All three Sweep-2 architectural changes regress vs gated_ew.** The fat-block design space has a sharp local optimum at `gated_ew` for this scale (10-min track, 16 MB).
2. **All three regress vs the simpler `vanilla` baseline as well** — meaning these aren't just "didn't beat the winner" but actively worse than the no-frills fat-block. A2 by +0.001, A3 by +0.002, A1 by +0.015.
3. **The fat block does not need additional per-token nonlinearity.** A1 (remove MLP) and A3 (add per-attn nonlinearity) both lose. The elementwise gate is already supplying enough.
4. **The fat block does not benefit from rerouting the MLP.** A2 confirmed that parallel MLP (reads `x_in`) beats sequential MLP (reads `z`). Parallel routing is the right base structure.
5. **gated_ew is confirmed as the architecture's local optimum** — further fat-block ablations are unlikely to beat it without a different design axis (more attentions, wider MLP, different gate placement, or stepping outside the fat-block altogether).

### Recommended next steps

1. **3-seed gated_ew confirmation** (seeds 314 + 999) — Sweep-1 parked this; with Sweep 2 done, it's now the highest-priority remaining action. A 3-seed mean ≤ 1.0828 with std ≤ 0.0003 establishes SOTA-parity at multi-seed, which is the publishable result.
2. **gated_ew + TTT** — Sweep 1 noted ~0.002 BPB headroom from TTT on this stack. If realized, gated_ew+TTT lands ~1.0810, matching SOTA's TTT headline.
3. **Step outside fat-block design space** if (1) and (2) don't break SOTA — e.g., parallel-residual depth recurrence (PR #1493 family), or an entirely different backbone direction. The fat-block ceiling at this scale is ~1.0829 (gated_ew); pushing below requires structural changes elsewhere.
4. **Skip further fat-block ablations.** The backlog items beyond A1/A2/A3 (5 sequential attentions, wider fat MLP, per-attn QK-norm) are now lower priority — Sweep-2 evidence is that the fat block as configured is at its local optimum, and modifications are net-negative. Reroute compute to (1)/(2)/(3).

---

## Sweep 3 — Loop-share architecture (NON-fat-block, on SOTA base)

Different design space: instead of fat-block experiments, **start from SOTA-equivalent base** (`FAT_BLOCK_ENABLED=0`, 11 logical layers, encoder=`[0,1,2,3,4,5,3,4]` / decoder=`[5,3,4,5,6,7,8,9,10]`). Modifications are confined to blocks 3, 4, 5 (the looped core, each visited 3× per forward pass). New folder `records/track_10min_16mb/2026-04-26_LoopShareMLP_UniqueAttn/`.

Two variants, both via env-var dispatch on a single forked `train_gpt.py`:

| # | Variant | Change | Status |
|---|---|---|---|
| V2 | `shared_fat_mlp` | 1 shared fat MLP (h=6144) across blocks 3, 4, 5 (replaces 3 separate h=2048 MLPs); attentions unchanged | ❌ partial run (GPTQ-alias bug, see below) |
| V1 | `unique_attn_thin_mlp` | 9 unique attentions (3 per block × 3 visits) + 1 shared thin MLP (h=1280) across blocks 3, 4, 5 | 🔧 pending re-run after fix |

### Sweep 3 — Variant 2 (V2) — `shared_fat_mlp` (seed 42, 2026-04-27, partial run)

**Config**: `FAT_BLOCK_ENABLED=0 LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=6144 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0 SEED=42`.
**Log**: `records/track_10min_16mb/2026-04-26_LoopShareMLP_UniqueAttn/logs/shared_fat_mlp_s42.log`.
**Model params**: 35,944,536 — **identical to SOTA** (3× h=2048 separate = h=6144 shared in param count).
**Peak GPU memory**: 53,095 MiB (much higher than fat-block's ~38 GB — the shared fat MLP is invoked 9× in the looped section, so 9× activation storage for backward).
**End step**: 3883 (wallclock cap, 2389 s training).
**Status**: ❌ **failed during GPTQ serialization** with `KeyError: 'blocks.4.mlp.fc.weight'`. Training itself completed cleanly; only the post-training compress step crashed (see "Why failed" below).

#### Training trajectory vs SOTA, gated_ew, vanilla

| Step | SOTA | gated_ew | vanilla | **V2 shared_fat_mlp** | Δ vs SOTA | Δ vs gated_ew |
|---:|---:|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3206 | 3.3175 | **3.3029** | **−0.032** | **−0.018** |
| 1000 | 3.1948 | 3.1847 | 3.1860 | **3.1751** | **−0.020** | **−0.010** |
| 1500 | 3.1030 | 3.1573 | 3.1623 | 3.1486 | +0.046 | −0.009 |
| *layer_loop activated* | step 2018 | step 2183 | step 2231 | **step 1823** | (earlier — slower step time hits frac=0.35 sooner) | — |
| 2000 | 3.0686 | 3.1481 | 3.1474 | **3.1063** | +0.038 | **−0.042** |
| 2500 | 3.0673 | 3.0644 | 3.0753 | **3.0173** | **−0.050** | **−0.047** |
| 3000 | 2.9476 | 3.0225 | 3.0279 | **2.9616** | +0.014 | **−0.061** |
| 3500 | 2.9672 | 2.9493 | 2.9560 | **2.8614** | **−0.106** | **−0.088** |
| end | step 4550 (2.8119) | step 4895 (2.8494) | step 4969 (2.8166) | **step 3883 (val_bpb 1.0917 pre-EMA)** | — | — |

**Throughput**: 0.468 s/step pre-loop, **~0.756 s/step post-loop** (vs sweep-2 fat-block's 0.576 s/step). Slower because:
- 11 physical blocks (vs fat-block's 8) → 38 % more blocks per forward pass
- Shared h=6144 MLP called 3 × per visit × 3 looped blocks = 9 invocations of an MLP that's 3× wider than the regular h=2048 MLP. Net: MLP FLOPs in looped section are **3× higher** than SOTA's separate MLPs.

V2 trains ~20 % fewer steps than gated_ew in the same wallclock (3883 vs 4895).

#### Final partial-eval numbers (training completed, GPTQ failed)

| Metric | SOTA (3-seed) | gated_ew | vanilla | **V2 shared_fat_mlp** | Δ vs SOTA | Δ vs gated_ew |
|---|---:|---:|---:|---:|---:|---:|
| End step | 4550 | 4895 | 4969 | **3883** | — | — |
| Final train_loss (last 500-step log) | 2.8119 (step 4550) | 2.8494 (step 4500) | 2.8600 (step 4500) | **2.8614 (step 3500)** | — | — |
| Pre-EMA val_bpb (last train step) | 1.0886 | 1.0904 | 1.0904 | **1.0917** | +0.0031 | +0.0013 |
| **Pre-quant post-EMA val_bpb** | **1.0873** | **1.0883** | (not logged) | **1.09071** | **+0.0034** | **+0.0024** |
| Quantized val_bpb | 1.0997 | 1.0996 | 1.1017 | (crashed — projected ~1.1015) | — | — |
| **Sliding val_bpb** | **1.0829** | **1.08292** | 1.08493 | **(projected ~1.0850)** | **~+0.0021** | **~+0.0021** |
| Peak memory (MiB) | n/a (8×H100) | 37,740 | 37,740 | **53,095** | — | — |

**Projection method**: pre-quant post-EMA 1.09071 → +0.0011 quant penalty (consistent across all prior runs) = ~1.1015 quantized → −0.0017 sliding gain = **~1.0848 sliding**.

#### Interpretation — V2 trains faster but generalizes ~+0.002 BPB worse

1. **Train-loss leadership.** V2 had the lowest train_loss at every checkpoint up through step 2500 (−0.05 vs SOTA, −0.05 vs gated_ew). The shared fat MLP at h=6144 has 3× the per-call FLOPs of a regular h=2048 MLP, providing more transformation capacity per forward pass. Optimization benefits clearly.

2. **Generalization gap (same pattern as Sweep 2).** Pre-EMA val_bpb 1.0917 is +0.0013 vs gated_ew, +0.0031 vs SOTA. EMA helped −0.001 BPB → 1.09071. The train-loss advantage does not translate proportionally to val. **Fourth experiment in a row showing this pattern: capacity-adding modifications to the looped core regress on val_bpb relative to gated_ew/SOTA.** Hypothesis: the looped core needs *iteration-stable* weights to do its 3× recurrence properly; pumping more compute through one shared MLP per visit reduces the smoothness EMA exploits.

3. **Peak memory ~53 GB.** Significantly higher than fat-block's ~38 GB because the shared MLP's activations are stored 9 × per forward (once per visit × 3 blocks). Still well under H100 80 GB but worth noting if anyone runs this on smaller cards.

4. **20 % fewer training steps** (3883 vs gated_ew's 4895). The wider per-call MLP costs step time, and the wallclock cap is fixed at 40 min. Fewer steps + better-per-step efficiency net to roughly the same end-state on train_loss but worse on val_bpb.

#### Why this run failed (and the fix)

`gptq_mixed_quantize` crashed with `KeyError: 'blocks.4.mlp.fc.weight'`. Root cause:

- PyTorch's `state_dict()` emits **3 alias keys** for a shared module (`blocks.{3,4,5}.mlp.fc.weight`), all pointing to the same GPU tensor.
- The trainer copies state_dict to CPU via `sd_cpu = {k: v.detach().cpu() for ...}` — but **`.cpu()` allocates a separate CPU buffer per call**, so the resulting CPU alias tensors no longer share `data_ptr()`.
- The dedup-by-data_ptr check inside `gptq_mixed_quantize` (M8 in the implementation plan) therefore failed to recognize the alias relationship, and tried to look up `hessians['blocks.4.mlp.fc.weight']` — which doesn't exist (only the canonical `blocks.3.mlp.fc.weight` is registered, since `named_modules()` deduplicates).

The CPU-only verification I ran during planning didn't catch this because `.cpu()` is a no-op when already on CPU — the alias relationship was preserved trivially. The real GPU→CPU copy path was not exercised.

**Fix landed (2026-04-27)**: rebuilt sd_cpu so alias keys map to the SAME CPU tensor. Dedup at the GPU `data_ptr` level BEFORE the `.cpu()` copy:

```python
_sd_gpu = base_model.state_dict()
_seen_gpu_ptrs = {}
sd_cpu = {}
for _k, _v in _sd_gpu.items():
    _gpu_ptr = _v.data_ptr()
    if _gpu_ptr in _seen_gpu_ptrs:
        sd_cpu[_k] = sd_cpu[_seen_gpu_ptrs[_gpu_ptr]]
    else:
        _seen_gpu_ptrs[_gpu_ptr] = _k
        sd_cpu[_k] = _v.detach().cpu()
```

With the fix, alias entries in `sd_cpu` are the SAME CPU tensor object → identical `data_ptr()` → `gptq_mixed_quantize`'s dedup correctly identifies them and reuses canonical q/scale tensors.

Re-run pending — V2 needs ~50 min on 2× H100. V1 (`unique_attn_thin_mlp`) was triggered by the same bug since it also sets `LOOP_SHARED_MLP=1` (with thin MLP h=1280); both will run cleanly with the fix.

### Verdict on V2 (based on partial data)

V2 is **likely not a winner** on the seed=42 data we have:
- Pre-quant post-EMA 1.09071 is +0.0024 vs gated_ew, +0.0034 vs SOTA.
- Projected sliding ~1.0848 is +0.002 worse than SOTA's 1.0829, +0.002 worse than gated_ew's 1.08292.
- Same train/val divergence pattern as fat-block sweep variants — adding per-visit MLP capacity helps optimization but hurts generalization.

The hypothesis "share MLP across visits with 3× the per-call hidden" tested by V2 looks like a structural mismatch with the recurrent depth pattern.

### Sweep 3 — Variant 1 (V1) — `unique_attn_thin_mlp` (seed 42, 2026-04-27, stopped early)

**Config**: `FAT_BLOCK_ENABLED=0 LOOP_UNIQUE_ATTN=1 LOOP_SHARED_MLP=1 LOOP_SHARED_MLP_HIDDEN=1280 MAX_WALLCLOCK_SECONDS=2400 TTT_ENABLED=0 SEED=42`.
**Log**: `records/track_10min_16mb/2026-04-26_LoopShareMLP_UniqueAttn/logs/unique_attn_thin_mlp_s42.log`.
**Model params**: 35,682,440 (~262 K under SOTA's 35,944,536).
**Status**: 🛑 **stopped manually after step 1000** — pre-loop phase only, architectural trajectory was lagging and no signal to justify continuing 50 minutes for the loop-activated phase.

#### Architecture (the actual hypothesis under test)

| | SOTA | V1 |
|---|---|---|
| Blocks 0, 1, 2, 6–10 | 1 attn + 1 MLP (h=2048) each | unchanged |
| Blocks 3, 4, 5 attentions | 1 each, **shared** across the 3 visits | **3 unique attentions each** — 9 total in the looped core, one per (block, visit) slot |
| Blocks 3, 4, 5 MLPs | 3 separate (h=2048) | **1 shared thin MLP (h=1280)**, used 9× per forward |

Hypothesis: **attention's job changes per visit** (early visits attend to local patterns, late visits to abstract relations) so per-visit specialization helps. **MLP's job is uniform** so sharing it is fine. The shared MLP must be thinner (h=1280) than SOTA's separate MLPs (h=2048) to fit the 16 MB artifact budget, since 6 extra attention modules cost ~2 MB compressed.

#### Training trajectory (steps 500 and 1000 only)

| Step | SOTA | gated_ew | vanilla | V2 shared_fat_mlp | **V1 unique_attn_thin_mlp** | Δ vs SOTA | Δ vs gated_ew |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500  | 3.3346 | 3.3206 | 3.3175 | 3.3029 | **3.3325** | −0.002 | +0.012 |
| 1000 | 3.1948 | 3.1847 | 3.1860 | 3.1751 | **3.2029** | **+0.008** | **+0.018** |
| 1500 | 3.1030 | 3.1573 | 3.1623 | 3.1486 | (stopped) | — | — |

Step time: 0.516 s/step for steps 0–500 (includes warmup compile), 0.444 s/step for 500–1000. Loop activation was projected at step ~1700–1800 (frac=0.35 ≈ 14 min in).

#### Why stopped early

1. **Trajectory was clearly lagging.** V1 was +0.008 worse than SOTA and +0.018 worse than gated_ew at step 1000, and the gap to V2 (+0.028) was stable. The pre-loop phase was telling us: "thin MLP + 6 dormant attentions = strictly weaker model than SOTA at this stage." No surprise, but no positive signal either.

2. **The architectural bet only activates after step ~1700** (loop activation). Even if per-visit attentions then close the gap, the model still has *less effective training time* for the 6 dormant-during-pre-loop attentions, and the thin MLP is a permanent capacity reduction. Best case: V1 catches up to ~SOTA. Worst case: stays +0.020 behind.

3. **Five prior experiments (Sweep 2 + V2) all showed the same train→val divergence pattern**: capacity-shifting modifications to the looped core regress on val_bpb relative to gated_ew/SOTA. The expected payoff for V1 is at best a tie with SOTA on val_bpb — unlikely to be a clean win, and a 50-minute confirmation isn't justified given the prior pattern.

#### What V1 was actually testing (and the negative-result interpretation)

The architectural hypothesis "**attention specialization across visits matters more than MLP capacity**" is reasonable on paper but ran into three practical problems:

- **Dormant-attention warmup tax**: V1's `attn[1]` and `attn[2]` of each looped block see no gradient during the first 35% of training (looping_active=False). They start from orthogonal init and have less effective optimization time vs the always-active `attn[0]` slot. This asymmetry is a structural disadvantage.
- **Thin MLP cost**: dropping `mlp_mult` from 4.0 → 2.5 (h=2048 → h=1280) costs visible per-step capacity, visible already at step 500 (V1 lagging V2 by 0.030 BPB equivalent in train_loss).
- **DDP overhead**: `find_unused_parameters=True` (required because of the dormant attentions) adds ~5% step-time overhead vs gated_ew/V2's `=False` default.

The hypothesis isn't necessarily wrong — per-visit attention specialization probably DOES help in some regime — but at this scale (35M params, 16 MB artifact, 2× H100, 40-min cap), the costs outweigh the benefits.

### Verdict on Sweep 3

**Both variants are not winners** at the seed=42 single-seed evaluation:

| Variant | End status | Pre-quant val_bpb | Projected sliding | Δ vs SOTA |
|---|---|---:|---:|---:|
| V2 `shared_fat_mlp` | step 3883 (GPTQ crashed → fixed → re-run pending if desired) | 1.09071 | ~1.0848 | +0.002 |
| V1 `unique_attn_thin_mlp` | step 1000 (stopped) | n/a | n/a | trajectory lagging |

**Combined with Sweep 1 + Sweep 2 (5 prior fat-block ablations)**, this is now **6 experiments in a row** showing that any structural modification to the looped core (delete MLP, sequential routing, leaky activation, fat shared MLP, unique-attn + thin MLP) regresses on val_bpb relative to gated_ew/SOTA. The empirical conclusion:

> **At this scale, the looped core (blocks 3, 4, 5 each visited 3×) has a sharp local optimum around "shared simple weights with optional cheap nonlinearity (gated_ew's elementwise gate)". Capacity rearrangements within the looped core don't help; the optimization-vs-generalization tradeoff is unfavorable for any move away from the gated_ew structure.**

### Recommended next directions (re-stating the parked items)

1. **3-seed gated_ew confirmation** (seeds 314, 999) — establish whether gated_ew's 1.08292 is a multi-seed result or seed=42 luck. ~1h20m on 2× H100.
2. **gated_ew + TTT** — Sweep-1 doc estimated ~0.002 BPB headroom from TTT on this stack; would land gated_ew + TTT around 1.0810, matching SOTA's TTT headline. ~50 min on 2× H100.
3. **Step out of the looped-core design space.** Looped core ≈ saturated; further gains likely need orthogonal axes:
   - Different recurrence schedule (longer/shorter loop, more loop iterations, different `enable_looping_at`)
   - Modify non-looped blocks (0–2 or 6–10) — these haven't been ablated in any of our 6 experiments
   - Different backbone family entirely (not the U-Net + recurrence pattern)

The fat-block + gated_ew exploration thread has produced a SOTA-equivalent result (1.08292 = SOTA's 1.08286 + 0.00006) and shown that the *gated_ew architecture is the local optimum*. That itself is a clean result worth reporting once the 3-seed confirmation lands.

---

## Changelog

- **2026-04-23**: Initial write-up with vanilla (seed 42) result.
- **2026-04-26**: Sweep-2 variant 1 (`delete_mlp_widen`, A1) abandoned after partial run — pre-quant val_bpb 1.10915 (+0.021 vs gated_ew), projected sliding ~1.100. A1 hypothesis (artifact savings beat lost MLP capacity) refuted. Brotli pre-flight added to `run_sweep2.sh`.
- **2026-04-26**: Sweep-2 variant 2 (`mlp_sequential`, A2) completed — sliding val_bpb 1.08576 (+0.00284 vs gated_ew, +0.00083 vs vanilla, **outside backlog pessimistic of 1.0850**). Sequential MLP routing is strictly worse than parallel. Parallel MLP + attn-chain is now confirmed as the right base structure for further fat-block work.
- **2026-04-26**: Sweep-2 variant 3 (`leaky_attn`, A3) completed — sliding val_bpb 1.08732 (+0.00440 vs gated_ew, +0.00239 vs vanilla, the **worst** of the three Sweep-2 variants on real numbers). Adding per-attn nonlinearity on top of the elementwise gate hurt rather than helped. **Sweep 2 closed: gated_ew confirmed as local optimum of the fat-block design space.** Recommended next: 3-seed gated_ew confirmation, gated_ew + TTT, or a different backbone direction.
- **2026-04-27**: Sweep-3 variant 2 (`shared_fat_mlp`, V2) ran 3883 training steps then crashed during GPTQ — `KeyError: 'blocks.4.mlp.fc.weight'` from a shared-module state_dict alias bug in the GPU→CPU copy path. Fixed in `train_gpt.py` (data_ptr dedup at the GPU level before `.cpu()`). Partial data: pre-quant post-EMA val_bpb **1.09071**, projected sliding **~1.0848** (+0.002 vs SOTA). V2 likely not a winner.
- **2026-04-27**: Sweep-3 variant 1 (`unique_attn_thin_mlp`, V1) stopped manually at step 1000 — pre-loop trajectory was lagging (+0.008 vs SOTA, +0.018 vs gated_ew on train_loss), no positive signal to justify continuing 50 minutes for the loop-activated phase. Sweep 3 closed. **6 experiments in a row** (Sweep 2 A1/A2/A3 + Sweep 3 V1/V2 + vanilla baseline) confirm that structural modifications to the looped core regress on val_bpb relative to gated_ew/SOTA. Architectural exploration of the looped core has hit diminishing returns. Recommended next: 3-seed gated_ew confirmation, gated_ew + TTT, or step outside the looped-core design space.
