# Fat-Block Experiment — Idea Backlog

Tracked ideas for pushing gated_ew (current best at 1.08292 val_bpb sliding) below SOTA. Results log lives in `doc/experiment_fatblock_results.md`; architecture walk-through lives in `doc/experiment_fatblock_2026-04-22.md`.

Baseline reference: `gated_ew` at seed=42 → sliding val_bpb **1.08292**, artifact 15.60 MB, 400 KB headroom under 16 MB cap.

---

## Active ideas (ready to prototype, in priority order)

### A1 — Delete fat-block big MLP + widen the 4 attentions in the fat block ⭐ top pick

**Goal**: Reallocate the ~3 MB currently locked up in the big MLP toward making the 4 sequential attentions *wider* (more per-head capacity), rather than adding more attention layers (which would spike step time).

**Why it's interesting**:
- Ties back to what the sweep already told us: gated_ew's gap vs SOTA is NOT about more layers — it's about per-token capacity inside each attention. The elementwise gate already provides per-token nonlinearity; widening attention gives it more channels to modulate.
- Deleting the big MLP saves **3.07 MB of artifact**; widening the 4 existing attentions reinvests ~1.5–2.5 MB of that back. Net: frees artifact AND makes attention richer.
- Step-time impact is modest if we widen by 1.5× (head_dim 64 → 96) instead of doubling.

**Concrete config to prototype**:

```
Fat block:
  4 gated attentions, head_dim=96 (was 64), num_heads=8, kv_heads=4
  NO big MLP
```

Per-attention param count with elementwise gate at head_dim=96:
- Q: 512 × (8×96) = 393 K
- K: 512 × (4×96) = 196 K
- V: 196 K
- O: (8×96) × 512 = 393 K
- `c_g` gate (dim → attn_dim=768): 393 K
- Total per attention: ~1.57 M params → ~0.54 MB compressed (int6+brotli)

Artifact accounting:
- 4 widened attentions: 4 × 0.54 = **~2.17 MB**
- Big MLP: **0.00 MB** (deleted)
- Current fat block cost: 1.44 MB (4 attns) + 3.07 MB (MLP) = 4.51 MB
- **New fat block cost: 2.17 MB** → frees **2.34 MB of artifact**
- Total artifact estimate: ~13.3 MB (plenty of headroom)

Compute impact:
- Current fat block forward: ~3,384 GFLOP (4 attns + 1 MLP)
- New fat block forward: ~3,840 GFLOP (4 widened attns, no MLP) = **+13% compute per step**
- Step time: ~0.48s → ~0.54s. Total steps in 2400s: ~4,900 → ~4,400

**Hypothesis**: with head_dim=96 each attention can store more nuanced "who-attends-to-whom" patterns. The elementwise gate is already providing per-token nonlinearity. Removing the fat MLP forces the 4 attentions to carry the per-token work too.

**Expected gain**: **−0.001 to −0.004 BPB vs gated_ew** (high variance — this is a qualitative architectural change, could also regress).

**Required changes**:
- New hyperparameter: `FAT_ATTN_HEAD_DIM` (default 64; set 96 for experiment)
- New hyperparameter: `FAT_BLOCK_MLP_ENABLED` (default 1; set 0 to delete big MLP)
- `CausalSelfAttention` needs to support `internal_dim != model_dim` (add `attn_dim` arg; Q/O projections go model_dim ↔ attn_dim; K/V go to `kv_heads × head_dim`)
- `FatBlock.__init__` instantiates attentions with widened head_dim only when `FAT_ATTN_HEAD_DIM != head_dim of normal blocks`
- `FatBlock.forward` skips the MLP path when `FAT_BLOCK_MLP_ENABLED=0`

**Follow-ups if it works**:
- Try head_dim=128 (2× original) — bigger step-time hit but more capacity
- Try adding ONE inter-attention tiny MLP back (hybrid config — best of both)

---

### A2 — Big MLP reads `z` (post-attention) instead of `x_in` (parallel → sequential)

**Goal**: Test whether our single-big-MLP fat block should be sequential (MLP processes attention output) rather than parallel (MLP processes pre-attention input).

**Context**: The "parallel" design came from PR #1204's learned-routing data showing `mlp_to_attn ≈ 0` when there are **multiple small MLPs**. With **one big MLP**, reading stale `x_in` may waste MLP capacity on information that the 4 attentions have already enriched in `z`.

**Change**: One-line edit in `FatBlock.forward`:
```python
mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
                                   ^^^^
# Change to: self.mlp_norm(z)
```

**Cost**: 0 params, 0 artifact change.

**Expected gain**: **±0.002 BPB** (genuine coin flip).

**Risk**: Low (zero-param, easily reversible).

**Required changes**:
- New hyperparameter: `FAT_BLOCK_MLP_MODE` (default `'parallel'`; `'sequential'` reads `z` instead of `x_in`)
- Single conditional in `FatBlock.forward`

**Priority**: Low-effort, medium-EV free experiment. Good to run in parallel with A1 (different compute pod if available).

---

### A3 — LeakyReLU² on attention output before gate + projection

**Goal**: Add zero-param per-token nonlinearity INSIDE each attention, matching the MLP activation style.

**Change**: In `CausalSelfAttention.forward`, after FA3 output and before the elementwise gate + `proj`:
```python
y = flash_attn_3_func(q, k, v, causal=True)
y = F.leaky_relu(y, negative_slope=0.5).square()  # ← NEW
# gate, reshape, proj...
```

**Cost**: 0 params, negligible compute (one activation).

**Expected gain**: **0 to −0.001 BPB**.

**Risk**: Could interact with the elementwise gate (which also operates on `y`). If the gate + activation combined clip too aggressively, loss might increase slightly.

**Required changes**:
- New hyperparameter: `ATTN_OUTPUT_ACTIVATION` (default `'none'`; `'leaky_relu_sq'` enables)
- Two-line change in `CausalSelfAttention.forward`

**Priority**: Zero-cost add-on. Best stacked on top of A1 or A2 winner.

---

## Future / lower-priority ideas (triaged, on the backlog)

### B1 — Inter-attention tiny MLP (shared)

Add one shared `MLP(dim, hidden=256)` applied between each pair of sequential attentions in the fat block. Directly fills the "missing per-token transformation between attentions" gap.

- **Cost**: +262 K params, ~128 KB artifact.
- **Expected**: −0.001 to −0.003 BPB.
- **Status**: Parked for now — user prefers more radical changes first. Revisit if A1/A2/A3 underperform.

### B2 — Enable Legal TTT at eval

Set `TTT_ENABLED=1`. Known −0.002 BPB from SOTA's measurement. Non-architectural but mandatory for "beat SOTA headline 1.0810".

- **Cost**: 0 params. Eval time ~20 min longer → total run ~70 min instead of ~50 min.
- **Expected**: −0.002 BPB on top of whatever architectural base.
- **Status**: Mandatory as a final step once architecture is settled. Not an architectural idea per se.

### B3 — Higher MUON_WD (0.095 → 0.105 or 0.110)

Follows SOTA's WD evolution arc (0.04 → 0.085 → 0.090 → 0.095). Higher WD → smaller weights → better brotli → more artifact headroom, which frees capacity to invest elsewhere.

- **Cost**: 0 params.
- **Expected**: −0.0005 to −0.0015 BPB.
- **Status**: Good hyperparameter-tuning pass. Low priority vs architectural changes.

### B4 — Delete ALL MLPs (nuclear option)

Delete every MLP in the model (layers 0–6 regular blocks + fat block). Frees ~10.2 MB. Reinvest in ~28 additional attentions or drastically wider existing ones.

- **Cost**: Radical re-architecture.
- **Expected**: Wide range (from +0.01 to −0.005 BPB).
- **Status**: Follow-up if A1 works. Too speculative to try first.

### B5 — 5 sequential attentions in fat block

Keep big MLP; add 1 more attention in the sequential chain (4 → 5).

- **Cost**: +270 KB artifact (fits in 400 KB headroom), +25 % fat-block compute.
- **Expected**: 0 to −0.0005 BPB (diminishing returns from recurrence suggest depth past 4 helps less).
- **Status**: Low priority — evidence from recurrence behavior suggests depth alone isn't the bottleneck.

### B6 — Post-attention RMSNorm (NormFormer-style)

Add a RMSNorm on each attention's contribution before residual add in the fat block:

```python
z = z + attn_scales[i] * F.rms_norm(a, (a.size(-1),))
```

- **Cost**: 0 params.
- **Expected**: 0 to −0.0005 BPB. Stabilizes residual magnitude across 4 sequential attentions.
- **Status**: Free, but modest expected gain. Try if nothing else works.

### B7 — Differential attention (ICLR 2025)

Compute two softmax passes with different Q projections and subtract. Reduces attention noise.

- **Cost**: +1.5 MB artifact (2× Q params), +50 % attention compute, needs non-standard kernel.
- **Expected**: −0.002 to −0.005 BPB in large-model settings; unknown for small.
- **Status**: Expensive to prototype. Defer indefinitely.

### B8 — Sigmoid attention (replace softmax)

Use FlashSigmoid kernel (17 % faster than softmax FA2 per paper). Per-element nonlinearity replaces row-normalized softmax.

- **Cost**: Kernel dependency (FlashSigmoid not in our current install).
- **Expected**: Neutral-to-positive on our scale.
- **Status**: Kernel porting cost too high for the marginal gain. Skip.

### B9 — Dropout / stochastic depth

Add dropout on attention output or stochastic depth across the 4 sequential attentions, with the same training budget.

- **Cost**: 0 params.
- **Expected**: Unclear — regularization may help or hurt in our compute-constrained regime.
- **Status**: Speculative. Worth a cheap run if all A-tier options fail.

---

## Experiment execution recommendations

**Preferred sequence** (assuming each run is ~50 min and A-tier results inform next steps):

1. **A1** (delete MLP + widen attentions) — highest-EV radical bet
2. **A2** (`FAT_BLOCK_MLP_MODE=sequential`) — orthogonal free test, can run in parallel
3. **A3** (LeakyReLU² on attn output) — if A1 or A2 wins, stack this on top
4. **B2** (TTT) — enable on the best architectural winner
5. **3-seed confirmation** of the final winner

If A1 fails (net regression):
- Fall back to **B1** (inter-attention tiny MLP) as an iterative improvement
- Then B3 (higher WD) to explore capacity via regularization

---

## Status tracking

Update this section as experiments complete.

| Idea | Status | Variant name | Result |
|---|---|---|---|
| A1 | 🔧 code ready (2026-04-24), run pending | `delete_mlp_widen` | — |
| A2 | 🔧 code ready (2026-04-24), run pending | `mlp_sequential`   | — |
| A3 | 🔧 code ready (2026-04-24), run pending | `leaky_attn` — scoped to fat block only | — |
| B1 | deferred (user preferred radical changes first) | — | — |
| B2 | pending (after A-tier winner)                  | — (would set `TTT_ENABLED=1` on winner) | — |
| B3 | deferred                                       | — | — |
| B4 | deferred                                       | — | — |
| B5 | deferred                                       | — | — |
| B6 | deferred                                       | — | — |
| B7 | deferred (infeasible — kernel cost)            | — | — |
| B8 | deferred (kernel cost)                         | — | — |
| B9 | deferred                                       | — | — |

**Implementation notes (2026-04-24)**:
- A1/A2/A3 code added to `train_gpt.py` via four new hyperparameters:
  - `FAT_ATTN_HEAD_DIM` — overrides head_dim for fat-block attentions only (A1)
  - `FAT_BLOCK_MLP_ENABLED` — toggles the big MLP in the fat block (A1)
  - `FAT_BLOCK_MLP_MODE` — `parallel`/`sequential` toggles where the big MLP reads from (A2)
  - `ATTN_OUTPUT_ACTIVATION` — applied only inside fat-block attentions; `none` / `leaky_relu_sq` (A3)
- **Regular Blocks (layers 0–6) are NOT modified** by any A-tier flag. Only the fat block (layer 7) sees A1/A2/A3 effects. This is a clean ablation against `gated_ew`.
- CPU smoke tests + a bug audit covering 8 edge cases (RoPE with widened head_dim, gate shape mismatch, GLU-V + widening combo, DDP unused-parameter safety, optimizer routing, encoder/decoder indices, invalid value handling, missing MLP attributes) all pass.
- Launcher `run.sh` has three new variants: `delete_mlp_widen`, `mlp_sequential`, `leaky_attn`.

---

## Open questions

- **How aggressively to widen in A1?** head_dim=96 (1.5×) is the default above; head_dim=128 (2×) gives more capacity at higher compute cost. Could sweep.
- **Should widening apply to the 7 regular blocks too?** Currently only proposes widening the 4 fat-block attentions. Widening all 11 attentions would be a much bigger capacity bump (and compute bump).
- **Keep the elementwise gate during A1?** Assumed yes (it's part of gated_ew's definition). Testing "widen attn, drop gate" is a separate ablation worth considering.
- **Do we have enough pod time for all 3 A-tier runs + TTT + 3-seed confirm?** Approximate budget: 3 × 50 min + 1 × 70 min + 2 × 70 min = ~7 hours serial. Can parallelize if multi-pod.
