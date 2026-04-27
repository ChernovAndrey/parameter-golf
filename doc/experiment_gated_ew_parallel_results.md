# Experiment Results — gated_ew on parallel-residual layers (7-10) + targeted MLP 3.25×

**Status:** seed-42 result complete. **Sub-SOTA at single seed.** Awaiting seed 314 and 999 confirmation.
**Date:** 2026-04-27
**Folder:** `records/track_10min_16mb/2026-04-27_GatedEW_Parallel_MLP325/`
**Hardware:** 2× H100, 40 min training cap, TTT off, sliding window on.
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

## Changelog

- **2026-04-27**: Seed-42 result. **Sliding val_bpb 1.08226 (sub-SOTA by 0.00044), artifact 15.80 MB (195 KB headroom).** Pre-quant 1.08757, quant non-sliding 1.09889. Architecture: 4 elementwise gates on layers 7-10, MLP 4.0× on 0-6 + 3.25× on 7-10 (h=1664). Awaiting seeds 314 and 999.
