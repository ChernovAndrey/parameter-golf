# Technique Impact Rankings — Parameter Golf SOTA

> **Current SOTA (2026-04-09, PR #1493): 1.0810 BPB.** This file has two parts:
> 1. The **new-stack techniques** (everything introduced between 2026-03-26 and 2026-04-21) — use this for planning new work.
> 2. The **legacy 1.1147-BPB table** (historical) — kept as reference for the earlier stack.
>
> See `leaderboard_update_2026-04-21.md` for per-record walkthrough and `sota_architecture.md` for the current model card.

---

## Part 1 — New-Stack Techniques (SOTA 1.0810 BPB, snapshot 2026-04-21)

Approximate BPB impact of each technique introduced since 2026-03-25, sourced from the record READMEs in `records/track_10min_16mb/2026-03-31…2026-04-09`.

### Big Winners (0.008+ BPB)

| Technique | Approx. BPB gain | Source | Why it's big |
|-----------|------------------:|---|---|
| **SP4096 → SP8192** | ~−0.008 BPB | PR #1394 | Bigger tokenizer = more context per step + net-positive after SDClip |
| **SP1024 → SP4096** | ~−0.009 BPB | PR #1218 | Same mechanism, first half of the vocab pivot |
| **Depth recurrence (3-layer)** | ~−0.005 BPB | PR #1437 (3-layer), PR #1204 (mini) | 17 virtual layers from 11 physical, zero artifact cost |
| **MLP 3× → 4× + WD stratification (0.085/0.02)** | bundled, part of ~−0.017 BPB pivot | PR #1218 | More capacity + WD-compression synergy |

### Medium Winners (0.002–0.005 BPB)

| Technique | Approx. BPB gain | Source | Why |
|-----------|------------------:|---|---|
| **Parallel residuals (layer 7+)** | ~−0.002 to −0.003 BPB | PR #1204, PR #1412, PR #1477 | Attention/MLP operate on separate lanes; tightens quant gap |
| **Legal Score-First TTT** | ~−0.002 BPB | PR #549 framework, PR #1413 on SP8192 | 32K chunks × 3 epochs × SGD lr=0.005; eval-time adaptation |
| **SDClip `c = k · σ` vs quantile search** | ~−0.001 to −0.002 BPB | PR #1394 | Principled rate-distortion; R² ≈ 0.995 RMS ↔ compression |
| **QK-Gain 1.5 → 5.25** | monotonic, small per step | PR #1125, PR #1217, PR #1445 | Learnable per-head Q scaling |
| **WD-quantization synergy (0.085 → 0.090 → 0.095)** | ~−0.002 BPB | PR #1285, PR #1445 | Higher WD → smaller weights → better brotli → all-int6 possible |
| **GPTQ on embeddings (vs RTN)** | ~−0.001 BPB | PR #1394 | Full-Hessian quant on the token embedding matrix |

### Marginal Winners (< 0.002 BPB)

| Technique | Approx. BPB gain | Source | Why |
|-----------|------------------:|---|---|
| MuonEq-R (row-norm pre-NS5) | ~−0.001 BPB, zero-byte | PR #1260 | Improves conditioning for deep narrow models |
| Skip gates (sigmoid-gated U-Net skips) | ~−0.001 BPB | PR #1089 | Replaces hard skip add with learned gate |
| Fractional warmdown (`WARMDOWN_FRAC=0.72`) | tuning | PR #1445 | Schedule flexibility; linear decay to 0 over last 72 % |
| EMA-only (drop SWA, `EMA_DECAY=0.9965`) | simplification | PR #1218 lineage | Same/better than SWA snapshots on the new stack |
| Byte-shuffle + Brotli-11 | artifact headroom | PR #1089 | Replaces LZMA preset=9 on the main weight banks |
| LZMA code wrapper | ~−43 KB artifact | PR #1493 | `exec(lzma.decompress(base85_blob))` wraps train_gpt.py |
| Hessian-aware SDClip (λ=0.175) | 0.0002 | PR #1412 (non-record) | Modest zero-cost tweak; group-level Hessian traces stable |
| Progressive recurrence (two-phase activation) | small | PR #1412 (non-record) | Avoids sharp loss spike when all loops activate at once |

### Superseded / Removed (don't use on new-stack builds)

| Dropped technique | Replaced by | Reason |
|-------------------|-------------|--------|
| BigramHash (3072×112) | SP8192 tokenizer | Larger vocab makes bigram hashing redundant |
| SmearGate | — | Neutral on new stack after vocab expansion |
| Value embeddings (VE128, layers 9–10) | — | Neutral on new stack |
| SWA (50-step snapshots) | EMA 0.9965 | Simplification, same/better |
| QAT (int6 STE at warmdown > 15 %) | Post-training SDClip-GPTQ | Post-training is enough; QAT was marginal |
| Mixed int5/int6 quantization | All-int6 (66/66 layers) | WD sweep freed the headroom |
| Selective pruning → {−1, 0, +1} | — | SDClip fits natively under 16 MB |
| Parameter banking + distributed Muon | Plain Muon + DDP | Removed for simplicity in PR #1218 |
| Non-legal TTT / SLOT | Legal Score-First TTT | Compliance with Issue #1017 Track B |
| Coprime-stride data loader | ShuffledSequenceLoader | Simpler, no measurable regression |
| LZMA preset=9 on weights | Byte-shuffle + Brotli-11 | Brotli-11 wins on the new weight distribution |
| AR self-generated GPTQ calibration | Training-time Hessian calibration (64 batches) | Faster and cleaner inside the 600 s budget |

### New-Stack Key Insight

The big gains since 2026-03-25 were not architectural novelty — they were **three simplifications compounding**:
1. **Bigger tokenizer** (SP1024 → SP8192): reduced sequence pressure, unlocked everything downstream.
2. **WD-compression synergy**: treating weight decay as an artifact-budget lever (not just regularization) unlocks deeper quantization.
3. **Shared-weight virtual depth**: 3-layer recurrence buys effective depth at zero artifact cost.

The Apr 9 SOTA then adds **Legal Score-First TTT** (eval-time adaptation) and **parallel residuals** (representation routing) on top, for another ~0.004 BPB.

Saturation note: the last three records gained 0.001–0.002 BPB each. Within this architectural template the easy wins are gone.

### New-Stack Artifact Budget (~15.99 MB)

| Component | Raw (float32) | After SDClip-GPTQ + Brotli | % of artifact |
|-----------|--------------:|---------------------------:|--------------:|
| MLP bank (up + down, 4× × 11L) | 92.3 MB | ~11.3 MB | ~71 % |
| Attention Q+K+V+O (11L) | 34.6 MB | ~3.0 MB | ~19 % |
| Token embedding (8192 × 512) | 16.8 MB | ~1.1 MB | ~7 % |
| Small params + norms + gates | ~2.0 MB | ~0.2 MB | ~1 % |
| Tokenizer model + LZMA code wrapper | — | ~0.12 MB | ~1 % |

MLP's share grew from 63 % (old SOTA at 3× expansion) to ~71 % (new SOTA at 4× expansion). Any capacity-adding change (including MoE) will be measured against how well it uses that 71 %.

---

## Part 2 — Legacy Table (1.1147 BPB — PR #1019, 2026-03-25)

> Historical reference. Use Part 1 for planning new work.

Approximate BPB impact of each technique, estimated from ablations and
submission history across the leaderboard as of 2026-03-25.

---

## Big Winners (0.01+ BPB each)

| Technique | Approx. BPB gain | Why it's big |
|-----------|-------------------|---|
| More layers (9 to 11) | ~0.02-0.03 | More depth = more capacity, biggest single knob |
| 3x MLP expansion (512 to 1536) | ~0.02 | Massive capacity increase, 63% of artifact budget |
| Full Hessian GPTQ | ~0.015-0.02 | Better quantization = more effective params in 16MB |
| Sliding window eval (stride 64) | ~0.015 | Free BPB from overlapping eval windows, zero training cost |
| Longer sequence (512 to 2048) | ~0.015 | More context for prediction |
| Int6 quantization (vs int8) | ~0.01-0.015 | 6 bits/param vs 8 = ~37% more params in 16MB |

## Medium Winners (0.003-0.01 BPB)

| Technique | Approx. BPB gain | Why |
|-----------|-------------------|---|
| XSA on all layers | ~0.005-0.008 | Forces heads to carry cross-position info, zero params |
| EMA + SWA weight averaging | ~0.005-0.007 | Smooths the loss landscape |
| LeakyReLU(0.5) squared | ~0.003-0.005 | Eliminates dead neurons in MLP |
| Muon optimizer (Newton-Schulz) | ~0.003-0.005 | Better optimization trajectory |
| Weight decay tuning (0.04) | ~0.003 | Better regularization |
| Late QAT (STE at warmdown > 15%) | ~0.003 | Training adapts to quantization noise |

## Marginal Winners (< 0.003 BPB)

| Technique | Approx. BPB gain | Why |
|-----------|-------------------|---|
| BigramHash (3072 x 112) | ~0.002-0.003 | Cheap bigram context at input |
| SmearGate | ~0.001-0.002 | Learnable local blending with previous token |
| Partial RoPE (16/64 dims) | ~0.001-0.002 | More content-matching capacity in attention |
| LN Scale 1/sqrt(layer+1) | ~0.001-0.002 | Stabilizes deep narrow transformers |
| Value Embedding VE128 (layers 9-10) | ~0.001-0.002 | Re-injects token identity at deep layers |
| U-Net skip connections | ~0.001-0.002 | Shortcut for shallow features to deep layers |
| Logit softcap (30.0) | ~0.001 | Prevents logit explosion |
| Selective pruning (to -1, 0, +1) | ~0.001 | Better LZMA compression |
| LZMA preset=9 | < 0.001 | Max compression squeezes extra KB |

---

## Key Insight

The competition is won on **quantization and systems engineering**, not creative architecture:

1. **Cramming more effective params into 16MB** (GPTQ, int6, LZMA, pruning) is the #1 lever
2. **Using those params efficiently** (more layers, wider MLP, longer context) is #2
3. **Free eval tricks** (sliding window, longer eval context) are #3
4. **Architectural novelties** (XSA, BigramHash, SmearGate, etc.) are collectively ~0.01-0.015 BPB combined but individually marginal

---

## Artifact Budget Breakdown (~15.91 MB)

Where the 16MB goes after int6 GPTQ + LZMA:

| Component | Raw (float32) | After quant+compress | % of artifact |
|-----------|---------------|---------------------|---------------|
| MLP banks (up+down) | 69.2 MB | ~10.0 MB | 63% |
| QO bank (Q+Out projections) | 23.1 MB | ~3.3 MB | 21% |
| KV bank (K+V projections) | 11.5 MB | ~1.7 MB | 11% |
| Token embedding | 2.1 MB | ~0.3 MB | 2% |
| BigramHash | 1.6 MB | ~0.2 MB | 1% |
| Small params + code | ~1.0 MB | ~0.3 MB | 2% |
