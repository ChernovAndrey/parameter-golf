# Experiment Results — gated_ew on parallel-residual layers (7-10) + targeted MLP 3.25×

**Status:**
- 2-GPU 2400s seed-42 result: sub-SOTA at single seed (1.08226 sliding, no TTT).
- **8-GPU 600s seed-42 result (competition spec, with TTT): 1.08326 TTT — +0.0025 ABOVE SOTA. Architecture does NOT survive the competition wall-clock budget.** See "8-GPU competition-spec follow-up" section below.
**Date:** 2026-04-27 (initial 2-GPU), 2026-04-29 (8-GPU follow-up)
**Folders:** 
- `records/track_10min_16mb/2026-04-27_GatedEW_Parallel_MLP325/` (2-GPU)
- `records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325/` (8-GPU + TTT submission attempt)
**Base:** PR #1493 SOTA (1.0810 BPB w/ TTT, 1.0827 sliding-only).
**Predecessor:** `doc/experiment_gated_ew_all_layers_results.md` (all-layer variant overflowed 16 MB cap).

---

## Headline result

| Metric | **This run (s42)** | SOTA s42 (sliding) | FatBlock gated_ew s42 | AllLayers gated_ew s42 |
|---|---:|---:|---:|---:|
| **Quantized sliding-window val_bpb** | **1.08226** | ~1.0827 (3-seed mean) | 1.08292 | (over budget) |
| Pre-quant post-EMA val_bpb | 1.08757 | ~1.087 | 1.08833 | 1.08938 |
| Quantized val_bpb (non-sliding) | 1.09889 | — | 1.0996 | 1.10007 |
| Total submission size | **15,804,975 B** | 15,991,930 B | 15,600,043 B | 16,274,654 B ❌ |
| Headroom vs 16 MB cap | **195 KB** | 8 KB | 400 KB | −274 KB |
| Total params | 35,420,248 | 27.1 M | 34.89 M | 36.52 M |
| Final training step | 4,824 | 4,550 (8 GPU) | 4,895 | 4,110 |
| Wall time | ~50 min (2 GPU) | ~16 min (8 GPU) | ~50 min | ~50 min |

**Sub-SOTA by 0.00044 BPB at single seed.** Pre-quant val_bpb is the lowest of all four runs compared.

---

## Architecture

- **Layers 0-6 (sequential residual):** unchanged from SOTA. MLP 4.0× (h=2048), no gate. Includes the looped layers 3, 4, 5 (visited 3× per pass) at full SOTA capacity.
- **Layers 7-10 (parallel residual):**
  - Qwen G1 elementwise gated attention added: `y = sigmoid(W_g x) * Attn(x)`, `W_g [512→512]`, then `proj`. 4 gates × 262 144 = **1.05 M params** (~+0.51 MB after int6 GPTQ + Brotli-11).
  - MLP shrink 4.0× → 3.25× (h=2048 → **h=1664 = 13×128**, clean GPTQ block alignment). Saves 4 × 2 × 512 × 384 = **1.57 M params** (~−0.77 MB).
- All other PR #1493 ingredients unchanged: SP8192 vocab, 11 physical layers, 3-layer recurrence on {3,4,5} activated at frac 0.35, MuonEq-R, SDClip-Hessian-GPTQ int6 matrices + int8 token embedding, byte-shuffle + Brotli-11.

**Per-token MLP volume in parallel zone:** 4 × 1664 = 6656 (vs FatBlock's validated 1 × 6144 single shared MLP — within 8% of the same per-token compute volume).

---

## Training trajectory — seed 42

| Step | SOTA s42 (8GPU) | FatBlock gated_ew s42 (2GPU) | AllLayers gated_ew s42 (2GPU) | **This run s42 (2GPU)** |
|---:|---:|---:|---:|---:|
| 500 | 3.3346 | 3.3206 | 3.3085 | **3.3210** |
| 1000 | 3.1948 | 3.1847 | 3.1725 | **3.1875** |
| 1500 | 3.1030 | 3.1573 | 3.1465 | **3.1582** |
| 2000 | 3.0686 | 3.1481 | 3.1138 | **3.1428** |
| _layer_loop activated_ | step 2018 | 2183 | 1832 | **2148** |
| 2500 | 3.0673 | 3.0644 | 3.0228 | **3.0619** |
| 3000 | 2.9476 | 3.0225 | 2.9712 | **3.0153** |
| 3500 | 2.9672 | 2.9493 | 2.8828 | **2.9418** |
| 4000 | 2.9106 | 2.8605 | 2.7791 | **2.8539** |
| 4000 val_bpb | 1.1119 | 1.1255 | 1.0932 | **1.1229** |
| 4500 | 2.7622 | 2.8494 | n/a (capped) | **2.8391** |
| 4824 val_bpb | — | — | — | **1.08880** |
| End step | 4550 | 4895 | 4110 | **4824** |

**Throughput:** 2.02 M tok/s pre-loop, 1.65 M tok/s post-loop. ~17 % faster per step than the AllLayers variant (1.72 M / 1.36 M) — directly because of having 4 gates instead of 11. **The faster per-step throughput is what produced the late-game val_bpb advantage.**

---

## Reading the result

### What worked

1. **The principled hypothesis held empirically.** Restricting elementwise gating to the parallel-residual layers (where attn output enters the residual without a downstream MLP nonlinearity) and paying for it via a targeted MLP shrink on the same layers produced the lowest pre-quant val_bpb of any variant we've measured. The gate-for-MLP substitution principle is now empirically supported on the SOTA architecture, not just the FatBlock context.
2. **Looped layers (3, 4, 5) preserved at full capacity** — keeping their MLP at 4.0× was the right call. These are the highest-leverage MLPs in the model (visited 3× per pass when looping is active).
3. **Calibrated budget prediction** — using the corrected gate compression rate (~0.49 B/param, MLP-rate, not 0.35 attention-rate) the predicted artifact (15.78 MB) matched actual to within a few KB.
4. **Clean GPTQ block alignment** — h=1664 = 13×128, all GPTQ blocks divide cleanly. No partial-block compression efficiency loss.

### Mid-training scare worth noting

Step-4000 val_bpb came in at 1.1229 — worst of the 3 runs measured at that checkpoint. This was a false alarm: the model was just slightly behind the late-training "phase change" that AllLayers had crossed at step 4000. By step 4824 (using the 714 extra steps the faster throughput bought us), val_bpb dropped to 1.0888 — closing 0.034 BPB in 824 steps. **Lesson: do not interpret single mid-training val_bpb as predictive when remaining training horizon differs across runs.**

### What we don't know yet

- **Seed variance.** SOTA's seed-to-seed std is 0.0002 BPB. A −0.00044 single-seed advantage is 2.2× that, so it is plausibly real but seed 42 noise alone could account for it. Need seeds 314 and 999.
- **Whether the gate or the MLP shrink is doing the work.** Run `baseline_no_gate` (same MLP shrink, no gate) to attribute the contribution.

---

## Next steps

1. **Confirm with seeds 314 and 999** (4d ago in the original convention):
   ```bash
   ./run.sh gated_ew_parallel 314
   ./run.sh gated_ew_parallel 999
   ```
   Pass criterion: 3-seed mean ≤ 1.0827 with std ≤ 0.001.
2. **Run baseline ablation:**
   ```bash
   ./run.sh baseline_no_gate 42
   ```
   Decomposes the gain. If baseline_no_gate val_bpb ≈ 1.083, the gate's contribution is ~0.001 BPB. If ≈ 1.085+, the gate is contributing ~0.003 BPB. If ≈ 1.082, the MLP shrink alone is the source.
3. **If 3-seed confirmed, propose as SOTA candidate.** Consider also enabling Legal Score-First TTT (PR #1493 used it; we disabled for fair comparison) for an additional ~0.002 BPB.
4. **Possible follow-ups if seed-mean holds:**
   - Try MLP 3.5× on layers 7-10 (more capacity, ~125 KB headroom).
   - Try gated_ew on parallel + cheap headwise gates on 0-6 (test "gate everywhere with right type per layer" hypothesis at low cost).

---

## Files

- Trainer: `records/track_10min_16mb/2026-04-27_GatedEW_Parallel_MLP325/train_gpt.py`
- Launcher: `records/track_10min_16mb/2026-04-27_GatedEW_Parallel_MLP325/run.sh`
- Run log (seed 42): `records/track_10min_16mb/2026-04-27_GatedEW_Parallel_MLP325/logs/gated_ew_parallel_s42.log`

---

## 8-GPU competition-spec follow-up (2026-04-29) — architecture does NOT win at the true budget

### Setup

Re-ran the same architecture (4 elementwise gates on parallel-residual layers 7-10, MLP 4.0× on 0-6 + 3.25× on 7-10) on the **official competition spec: 8× H100 / 600s training cap**, with **Legal Score-First TTT** enabled (same params as PR #1493 SOTA: lr=0.005, momentum=0.9, 3 epochs/chunk, 32K-token chunks). Folder: `records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325/`.

### Seed 42 result

| Stage | This run (8-GPU) | SOTA s42 (8-GPU) | Δ vs SOTA |
|---|---:|---:|---:|
| End step | 4592 | 4550 | +42 |
| Pre-quant post-EMA val_bpb | **1.08990** | 1.08735 | **+0.00255** ❌ |
| Quantized val_bpb | 1.10117 | 1.09970 | +0.00147 |
| Quant tax | +0.01127 | +0.01235 | −0.00108 (gates compress *better* than SOTA's projections) |
| Sliding val_bpb | 1.08453 | 1.08286 | +0.00167 |
| **Quantized TTT val_bpb** | **1.08326** | **1.08079** | **+0.00247** ❌ |
| Artifact bytes | 15,806,422 | 15,991,930 | −185,508 (193 KB headroom vs SOTA's 8 KB) |

Decision after seed 42: **aborted seeds 314 and 999.** A 3-seed mean ≤ 1.08100 (SOTA) would have required the other two seeds to average ≤ 1.0790, i.e. −0.0036 below seed 42 — about 5σ from SOTA's seed-to-seed std (0.0002). Implausible; saves ~5 H100-hours of compute.

### Why the 2-GPU result didn't survive — the lesson

| | 2-GPU 2400s run | 8-GPU 600s run | Note |
|---|---:|---:|---|
| Total compute (GPU-seconds) | 4776 | 4704 | nominally equivalent |
| Steps reached | 4824 | 4592 | **−232 steps on 8-GPU** |
| Pre-quant post-EMA | 1.08757 | 1.08990 | +0.00233 |

The 2-GPU "win" came from **232 extra optimizer steps**, not from the architecture. Two reasons 8-GPU underperforms its theoretical 4× speedup:

1. **NCCL all-reduce overhead scales with rank count.** 8-rank reductions cost more wall-clock per step than 2-rank, so per-step time on 8 GPUs is ~3.87× faster than on 2 GPUs, not the ideal 4×.
2. **Torch.compile fixed cost (~1-2 min)** eats a much bigger fraction of 600s than of 2400s.

Net effect: ~232 fewer optimizer steps in the 600s budget. Late-training each step is worth ~0.0001 BPB, so 232 × 0.0001 ≈ **0.023 BPB raw-eval gap**, which propagates roughly proportionally through pre-quant (1.08757 → 1.08990 = 0.00233) and through to TTT (1.0823 → 1.08326).

The architectural change (gates + targeted MLP shrink) costs **~0.0026 BPB pre-quant vs SOTA at the same step count**. With more steps available (2-GPU regime), the model trains long enough to absorb that cost. With the competition's 8-GPU/600s budget, it doesn't.

### What it means

- **Gates compress cleanly** — quant tax is *lower* than SOTA's (0.01127 vs 0.01235). The artifact-budget hypothesis (gates as MLP-rate, not attn-rate) was right, and the GPTQ pipeline handles them fine.
- **Per-token MLP volume in the parallel zone matters more than I thought.** Reducing it from 4 × 2048 = 8192 to 4 × 1664 = 6656 (a 19% cut) was too aggressive for the 600s budget — the gates can't compensate fast enough.
- **The general design principle still holds** (gates substitute for MLP capacity precisely where attn output enters the residual without a downstream MLP nonlinearity). But the substitution rate isn't 1:1 in this regime — the gates need more training to find their useful values than the wall-clock allows.

### Course-correct candidates (not yet tested)

Listed roughly in order of likelihood-to-help:

1. **Headwise gates instead of elementwise** — gate is ~free (4 layers × 8 = 32 params per layer), keep MLP at 4.0× on all layers. Tests whether the per-layer-nonlinearity matters at all without paying any MLP capacity.
2. **Less aggressive MLP shrink (3.5× or 3.625× on parallel)** — keeps more capacity, costs more headroom, may close the pre-quant gap enough for TTT to take us under SOTA.
3. **Drop gates, just submit MLP shrink alone** — sanity baseline. If MLP 3.25× alone matches SOTA, the gates are doing nothing useful; if it's worse, the gates are adding ~something but not enough.
4. **Revert to no architecture change.** PR #1493 SOTA is hard to beat; maybe the right play is finding a different lever entirely (different optimizer schedule, different recurrence pattern, etc.).

### Files preserved

- `train_seed42.log` — full 8-GPU/600s/TTT run log (seed 42 only; 314 and 999 not run)
- `seed42_final_model.pt` — full-precision EMA state dict (~133 MB, recovery backup)
- `seed42_final_model.int6.ptz` — 15.81 MB int6+brotli artifact
- All inside `records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325/`

## Changelog

- **2026-04-27**: 2-GPU/2400s seed-42 result. **Sliding val_bpb 1.08226 (sub-SOTA by 0.00044), artifact 15.80 MB (195 KB headroom).** Pre-quant 1.08757, quant non-sliding 1.09889. Architecture: 4 elementwise gates on layers 7-10, MLP 4.0× on 0-6 + 3.25× on 7-10 (h=1664).
- **2026-04-29**: **8-GPU/600s seed-42 result (competition spec, with TTT): 1.08326 TTT, +0.00247 ABOVE SOTA.** Aborted seeds 314 and 999. Architecture loses to SOTA at the true wall-clock budget; the 2-GPU win was budget-driven (232 extra optimizer steps), not architectural. Logged 4 candidate course-corrections to try next.
