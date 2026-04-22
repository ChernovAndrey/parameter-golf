# Parameter-Golf Leaderboard Update — What Changed Since 2026-03-25

**Pulled at:** 2026-03-25 (last-seen SOTA was `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`, **1.1147 BPB**).
**Re-surveyed:** 2026-04-21. Current SOTA is `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`, **1.0810 BPB** (3-seed mean, std 0.0002, author: bigbag, PR #1493).

**Delta: −0.0337 BPB in 27 days.** About ~3% relative improvement, crossing the 0.005-nat record threshold **six times** in that window.

The field has shifted into a single dominant stack: **SP8192 tokenizer + Parallel Residuals + Depth Recurrence + MuonEq-R + SDClip GPTQ + Legal Score-First TTT**. The previous 1.1147 recipe (SmearGate, BigramHash, value embeddings, SWA, QAT, hash embeddings, non-legal TTT) has been stripped out piece by piece and replaced.

---

## TL;DR

| | 2026-03-25 snapshot (PR #1019) | 2026-04-09 snapshot (PR #1493) |
|---|---|---|
| val_bpb | 1.1147 | **1.0810** |
| Vocab | 1024 BPE | **8192 BPE** (passed through 4096) |
| MLP expansion | 3× | **4×** |
| Depth | 11L (U-Net, independent) | **11L + 3-layer recurrence** (17 virtual layers) |
| Residual structure | Sequential (attn → MLP) | **Parallel from layer 7+** (GPT-J style) |
| Optimizer | Parallel Muon | **MuonEq-R** (row-norm pre-NS5) |
| Weight decay | 0.04 (muon) | **0.095 (muon) / 0.085 (embed) / 0.02 (adam)** |
| QK gain | 1.5 | **5.25 (learnable per-head)** |
| GPTQ | Full-Hessian, RTN embeds | **SDClip `c=k·σ`** (k=12.85 matrices, k=20 embeds), **GPTQ-on-embeds** |
| Activation | LeakyReLU(0.5)² | LeakyReLU(0.5)² (unchanged) |
| Test-time | Dropped on prior stack | **Legal Score-First TTT** (SGD lr=0.005, 3 epochs × 32K chunks) |
| Compression | LZMA preset=9 | **Byte-shuffle + Brotli-11 + LZMA code wrapper** (~43 KB savings) |
| Bigram / hash embeds | `BigramHash 3072×112` | **Removed** |
| Value embeddings | Layers 9–10 | **Removed** |
| SWA + QAT | Yes | **Removed** — replaced by EMA=0.9965 + post-training SDClip-GPTQ |
| SmearGate | Yes | **Removed** |
| Fractional warmdown | No (iter-count) | **Yes (`WARMDOWN_FRAC=0.72`)** |
| Artifact | ~15.91 MB | ~15.99 MB |
| Step time / steps | 86.7 ms / 6,927 steps | ~129 ms / 4,550 steps (3-layer recurrence burns step time) |

Headline: the leaderboard rotated through **two independent unlocks** in sequence.
1. **Vocabulary scale + regularization** (Apr 1 PR #1218): 1024→4096 BPE + MLP 4× + WD 0.085, with TTT and most "tricks" deleted. Dropped BPB 1.1147 → 1.0979 in a single jump.
2. **SP8192 + SDClip + Recurrence + Parallel Residuals + Legal TTT** (Apr 5–9): the current stack assembled over five merged PRs.

---

## Leaderboard delta table (chronological)

| # | Date | Record | Author | BPB | Single incremental change | Base |
|---|---|---|---|---|---|---|
| — | 2026-03-25 | ValCalib GPTQ + XSA-all + BigramHash 3072×112 | abaybektursun | 1.1147 | *(baseline — last pulled)* | PR #549 |
| 1 | 2026-03-31 | Parallel Residuals + Mini Depth Recurrence | Marko Sisovic | 1.1063 | Parallel residuals + loop layers 4–5 activated at step 3000 | PR #1179 (1.1105) |
| 2 | 2026-04-01 | Vocab4096 + MLPMult4 + WD085 | Kevin Clark (clarkkev) | 1.0979 | **Pivot**: SP4096, MLP 4×, WD 0.085/0.02, drop TTT/BigramHash/SmearGate/value-res/QAT | PR #549 stack, simplified |
| 3 | 2026-04-03 | MuonEq-R + DepthRecurrence + WD090 + AllInt6 | dexhunter | 1.0912 | WD 0.085→0.090 + MuonEq-R + layers 4–5 recurrence, all-int6 GPTQ | PR #1218 |
| 4 | 2026-04-04 | SP4096 + DepthRecurrence + ParResid + MuonEqR | aryanbhosale | 1.0897 | Parallel residuals (layer 7+) + QK-Gain 5.0 on top of PR #1285/1218 | PR #1218 |
| 5 | 2026-04-05 | **SP8192** + GPTQ-Embeddings + SDClip + Loop45×2 | Kevin Clark | **1.0856** | **SP4096 → SP8192**, GPTQ-on-embeds, SDClip `c=k·σ`, loop layers 4–5 **twice**, coprime-stride loader dropped | PR #1218 |
| — | 2026-04-06 | SP8192 + Hessian-SDClip + ProgressiveRecurrence *(non-record)* | Robby Sneiderman | 1.0835 | Parallel residuals + λ·Hessian-weighted SDClip + split recurrence 50%/65% | PR #1394 |
| 6 | 2026-04-06 | SP8192 + QK5 + Legal-TTT | dexhunter | 1.0828 | **QK-Gain 4.0 → 5.0** + **legal score-first TTT** (32K chunks, 3 ep, lr=0.005) | PR #1394 |
| 7 | 2026-04-08 | SP8192 + ParResid + Score-First TTT | aryanbhosale | 1.0822 | Parallel residuals **combined** with legal TTT on SP8192 | PR #1413 + PR #1412 |
| 8 | 2026-04-09 | **SP8192 + 3-Layer Recur + ParResid + QK5.25 + Legal-TTT** | bigbag | **1.0810** | Loop layers **3,4,5** (17 virtual from 11 physical) + QK 5.0→5.25 + WD 0.095, EMA 0.9965, warmdown 0.72 | PR #1413 + PR #1437 + PR #1445 |

BPB over time (mini-chart):

```
1.1147 |███████████████████████████████████████████  (2026-03-25 baseline)
1.1063 |█████████████████████████████████████████   (Apr 1, ParResid + mini DR)
1.0979 |████████████████████████████████████        (Apr 1, Vocab4096 pivot)
1.0912 |███████████████████████████████             (Apr 3, WD 0.090 + MuonEq-R)
1.0897 |███████████████████████████████             (Apr 4, SP4096 + everything)
1.0856 |█████████████████████████████               (Apr 5, SP8192 ← CLIFF)
1.0828 |████████████████████████████                (Apr 6, + Legal TTT)
1.0822 |████████████████████████████                (Apr 8, + ParResid)
1.0810 |███████████████████████████                 (Apr 9, + 3-layer recurrence)  ← CURRENT SOTA
```

---

## Per-record details

### 1. 2026-03-31 · Parallel Residuals + Mini Depth Recurrence — **1.1063 BPB** (Marko Sisovic, PR #1204)

Built from PR #1179. Two ideas, both ported from modded-nanogpt lineage:
- **Parallel residuals from layer 7**: attention and MLP read from separate residual lanes, each sublayer learns how strongly to write back into both lanes. Measured routing is asymmetric — MLP barely writes to attention's lane (e.g. `mlp_to_attn = 0.006–0.084` in deeper layers).
- **Mini depth recurrence**: only layers 4–5 repeated once, activated at `RECUR_START_STEP=3000`. Delayed activation matters — "always on" gave ~1.1163, delayed gave ~1.1153.

Also carried: mixed int5/int6 quantization + AR self-generated GPTQ calibration from PR #1105. Full seed table shows mean 1.1063, std 0.0017. No TTT.

→ `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/README.md`

### 2. 2026-04-01 · Vocab4096 + MLPMult4 + WD085 — **1.0979 BPB** (Kevin Clark, PR #1218) — MAJOR PIVOT

Built back off PR #549 (not the prior SOTA). Subtractive rewrite:
- **Removed**: TTT, QAT, BigramHash, SmearGate, value residuals, parameter banking + distributed-muon boilerplate.
- **Added**: 1024→4096 BPE (existing tokenizer script), MLP 3×→4×, muon WD 0.04→0.085, added embed WD 0.085, adam WD 0.04→0.02, LR 0.025→0.020, coprime-stride loader (PR #726), full-Hessian GPTQ (PR #1060), byte-shuffle + brotli compression (PR #1089), sigmoid-gated U-Net skip connections, QK-Gain 1.5→4.0 (PR #1125). Also fixed a sliding-eval bug that overcounted end-of-set tokens.

**Key finding**: `RMS(weight_matrix)` correlates with compressed-MB / raw-MB with **R² ≈ 0.99**, so weight decay is a direct lever on artifact size. This is the foundation of the SDClip story that lands a week later.

Author's own words: "The main changes are to use a bigger but more strongly regularized model."

→ `records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/README.md`

### 3. 2026-04-03 · MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ — **1.0912 BPB** (dexhunter, PR #1285)

Built on PR #1218 with three synergistic tweaks.
- **WD 0.085 → 0.090**: smaller weights compress ~5% better under brotli-11. Frees ~280 KB artifact headroom.
- **All-int6 GPTQ**: with that headroom, ALL 66 weight layers go to int6 (clip_range=31); no layers demoted to int5.
- **MuonEq-R**: row-normalize gradient rows before Newton-Schulz orthogonalization (zero-byte cost).
- **Depth recurrence (layers 4–5)**: shared-param loop, zero extra params.

Compression table from the record:
```
PR #1260   WD=0.085  60 int6 layers  15,981 KB  1.09217 BPB
PR #1279   WD=0.085  61 int6 layers  15,997 KB  1.09170 BPB
This       WD=0.090  66 int6 layers  15,967 KB  1.09057 BPB
```

→ `records/track_10min_16mb/2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/README.md`

### 4. 2026-04-04 · SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R — **1.0897 BPB** (aryanbhosale, PR #1334)

First record that **stacks all previously separate unlocks**: SP4096 + WD 0.090 + MLP 4× + Depth Recurrence (layers 4,5) + Parallel Residuals (layer 7+) + MuonEq-R + QK-Gain 5.0 + full GPTQ int6 + Brotli + LZMA code wrapper (~24 KB).

No TTT; Track-A compliant (standard sliding eval). Demonstrates the stack is strictly additive — each piece ported cleanly from a different PR. This is the baseline the Apr 5+ records build from.

→ `records/track_10min_16mb/2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR/README.md`

### 5. 2026-04-05 · SP8192 + GPTQ Embeddings + SDClip + Loop45×2 — **1.0856 BPB** (Kevin Clark, PR #1394) — SP8192 CLIFF

This is the **biggest single-record jump** in the window (−0.0041 BPB). Five changes on PR #1218:
- **Vocab 4096 → 8192** (bigger tokenizer, more context per step).
- **GPTQ-quantize the embedding matrix** — previously RTN. Reduces pre/post-quant gap.
- **Remove value embeddings**.
- **Simpler data loader** (`ShuffledSequenceLoader`) instead of coprime-stride.
- **Loop layers 4–5 twice** (share params; simpler implementation than PR #1204).
- **Row-normalized Muon** from PR #1217.
- **SDClip** — `clip = k · σ(row)` instead of quantile search. Principled rate-distortion: entropy of quantized weights `H(q) ≈ H(clip_level)` and raising `k` reduces entropy more effectively than reducing bitwidth. Uses `MATRIX_CLIP_SIGMAS=12.85` and `EMBED_CLIP_SIGMAS=20.0`.

5-seed mean: 1.08563, std 0.0007.

→ `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/README.md`

### 6. 2026-04-06 *(non-record)* · SP8192 + Parallel Residuals + Hessian-Aware SDClip + Progressive Recurrence — 1.0835 BPB (Robby Sneiderman, PR #1412)

Zero-cost variants on PR #1394. 3-seed (too small for statistical claim):
- **Parallel residuals (layers 7+)** tightens the quantization gap even when pre-quant BPB is flat.
- **Hessian-weighted SDClip**: `c_i = k · σ_i · [1 + λ(r_i − 1)]`, `λ = 0.175`. Higher λ reduces rounding error but increases entropy → worse brotli. Group-level Hessian traces are very stable across seeds (r=0.997); per-row importance is noisy (r=0.12).
- **Progressive recurrence**: two loop phases activated at 50% / 65% of training.

Not a record submission, but the parallel-residual + SP8192 combination seeds the Apr 8 record.

→ `records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/README.md`

### 7. 2026-04-06 · SP8192 + QK-Gain 5 + Legal Score-First TTT — **1.0828 BPB** (dexhunter, PR #1413)

**Single-knob change + TTT.** On PR #1394 base, just raise `QK_GAIN_INIT` from 4.0 → 5.0 and add legal TTT:

```python
for chunk in chunks:
    # Phase 1: SCORE (no grad, no update)
    with torch.inference_mode():
        nll = model.forward_logits(batch).cross_entropy(targets)
        loss_sum += nll.sum()
    # Phase 2: TRAIN on just-scored chunk
    if not is_last_chunk:
        for _ in range(TTT_EPOCHS):
            for x, y in chunk_seqs:
                (model(x, y)).backward()
                optimizer.step()
```

TTT settings: 3 epochs, SGD lr=0.005, momentum=0.9, cosine LR decay across chunks, 32K-token chunks. TTT time ~293 s (within 600s eval budget); ~588 s training. 3-seed mean post-TTT 1.08279 (std ~0.0005), pre-TTT sliding 1.08486 — TTT alone is worth about 0.002 BPB.

**Compliance (Issue #1017 Track B):**
1. Causality — strictly causal sliding-window eval.
2. Normalized — standard softmax over full vocab, no n-gram cache, no logit biasing.
3. Score-before-update — each chunk fully scored under `inference_mode` before any SGD update.
4. Single pass — each token scored exactly once.

→ `records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/README.md`

### 8. 2026-04-08 · SP8192 + Parallel Residuals + Score-First TTT — **1.0822 BPB** (aryanbhosale, PR #1477)

Synthesis of PR #1413 (SP8192 + TTT, 1.0828) and PR #1412 (SP8192 + parallel residuals, 1.0835). **1.0822 beats either alone** — the techniques are separable. A learned `lane_merge` scalar (init 0.5) blends lanes after the final layer.

→ `records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/README.md`

### 9. 2026-04-09 · SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT — **1.0810 BPB** (bigbag, PR #1493) — CURRENT SOTA

Deepens recurrence and tightens hyperparameters:
- **3-layer recurrence (layers 3,4,5)**: encoder `[0,1,2,3,4,5,3,4]` / decoder `[5,3,4,5,6,7,8,9,10]` — **17 virtual layers from 11 physical**. Activated at `ENABLE_LOOPING_AT=0.35` (35% of training).
- **QK-Gain 5.0 → 5.25** (monotonic improvement from 4.0 → 5.0 → 5.25).
- **Tuned hyperparams** (from PR #1445): `MUON_WD=0.095`, MLR=0.022, `EMA_DECAY=0.9965`, `WARMDOWN_FRAC=0.72`.
- **Logit softcap = 30.0** (`logits = 30 · tanh(logits/30)`).
- **LZMA code wrapper** — ~16.6 KB code, saves ~43 KB vs uncompressed source.

3-seed results: 42 → 1.0808, 314 → 1.0810, 999 → 1.0812. Mean 1.0810, std 0.0002. Artifact ~15.99 MB. Train 588s, eval (sliding + TTT) ~500 s.

→ `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md`

---

## New techniques catalog

Grouped by kind. Each bullet lists: **what it is** · provenance PR · approximate gain · code anchor.

### Architecture

- **SP8192 tokenizer** · PR #1394 · **~−0.008 BPB vs SP4096, ~−0.017 BPB vs SP1024** · `VOCAB_SIZE=8192`, tokenizer model ~100 KB, built via `data/download_hf_docs_and_tokenize.py`. Data available at `kevclark/parameter-golf`. Works because larger vocab = better tokenization entropy, more context per fixed sequence length. Net-positive after SDClip even though the embedding table grows.
- **Depth recurrence — 3-layer** · PR #1437, PR #1331 · **~−0.005 BPB over no recurrence** · `NUM_LOOPS`, `LOOP_LAYERS=3`, `ENABLE_LOOPING_AT=0.35`. Reconstructs encoder/decoder as `[0,1,2,3,4,5,3,4]` + `[5,3,4,5,6,7,8,9,10]`. Activation timing matters — too early burns steps, too late doesn't help. Author swept beyond 3-layer and observed diminishing returns.
- **Parallel residuals (layer 7+)** · PR #1204, PR #1412 · **~−0.002 to −0.003 BPB** · `PARALLEL_RESIDUAL_START=7`. GPT-J style: `x_out = x + attn_scale·Attn(x) + mlp_scale·MLP(x)`. Routing is asymmetric in practice (MLP barely writes to attention lane in deep layers). Tightens post-quantization gap.
- **Skip gates on U-Net** · PR #1089 · `SKIP_GATES_ENABLED=1` · Sigmoid-gated skip connections replace hard skip addition: `x = lerp(skip, x, sigmoid(gate))`.
- **QK-Gain scaling** · PR #1217, PR #1445 · **monotonic** · `QK_GAIN_INIT=5.25`. Learnable per-head Q scaling; baseline 1.5 → 4.0 (Apr 1) → 5.0 (Apr 6) → 5.25 (Apr 9).
- **Logit softcap 30.0** · `logits = 30 · tanh(logits / 30)`. Prevents logit explosion.

### Optimizer

- **MuonEq-R** · PR #1260 · **~−0.001 BPB, zero-byte** · `MUON_ROW_NORMALIZE=1`. Row-normalize gradient matrix before Newton-Schulz-5 orthogonalization. Improves conditioning for deep narrow models.
- **Stratified weight decay** · PR #1218, PR #1285, PR #1445 · **WD-compression synergy** · `MUON_WD=0.095`, `EMBED_WD=0.085`, `ADAM_WD=0.02`. Higher WD → smaller weights → better brotli compression → more artifact headroom → deeper quantization allowed. This feedback loop is the mechanism behind R² ≈ 0.99 RMS↔compressibility.
- **EMA instead of SWA** · `EMA_DECAY=0.9965`. Drop-in replacement for snapshot-based SWA; cheaper and more stable under the new stack.
- **Fractional warmdown** · PR #1445 · `WARMDOWN_FRAC=0.72`. Linear warmdown to LR=0 over the final 72% of training (vs previous fixed iter count of 3500/4000).
- **Learning rate split** · `embed_lr`, `head_lr`, `tied_embed_lr`, `matrix_lr` (MLR=0.022), `scalar_lr` — more granular per-param-group LR.

### Quantization

- **SDClip** · PR #1394 · **principled rate-distortion** · `MATRIX_CLIP_SIGMAS=12.85`, `EMBED_CLIP_SIGMAS=20.0`. Quantile-search replaced by `clip = k · σ(row)`. Key identity: post-quantization entropy `H(q)` — which drives compressed size — scales with clip level, so raising `k` is a cheaper lever than reducing bitwidth. R² ≈ 0.995 between matrix RMS and compression ratio.
- **GPTQ on embeddings** · PR #1394 · **~−0.001 BPB** · `MATRIX_BITS=6`, `EMBED_BITS=8`. Previously embeddings used RTN; now Full-Hessian GPTQ.
- **All-int6 across 66 weight layers** · PR #1285 · possible because WD 0.085→0.090 freed enough headroom that no layer needs int5.
- **Byte-shuffle + Brotli-11 + LZMA code wrapper** · PR #1089 · `CODE_WRAPPER`. ~16.6 KB of LZMA+base85 wrapped source, saves ~43 KB vs uncompressed code.

### Test-Time Training (Legal Score-First)

- Full TTT module gated behind `TTT_ENABLED=1`.
- **Chunking**: `TTT_CHUNK_TOKENS=32768`. Each chunk is first fully scored under `torch.inference_mode()`, then SGD-updated on that chunk's tokens.
- **Optimizer**: SGD with `TTT_LR=0.005`, `TTT_MOMENTUM=0.9`, gradient clip at 1.0, cosine LR decay across chunks.
- **Epochs per chunk**: `TTT_EPOCHS=3`.
- **Gain**: ~0.002 BPB on SP8192 stack (Apr 6). Gain held across subsequent records.
- **Compliance** (Issue #1017 Track B, conditions 1–4): causality, full-distribution softmax, score-before-update, single pass. No SLOT, no pre-quant TTT on val data, no n-gram cache, no eval-time logit bias.

### Data / Eval

- **Simpler shuffled-sequence loader** replaced the coprime-stride loader (PR #1394).
- Validation eval still sliding-window, standard causal, stride 64. Total eval budget 600s (sliding + TTT).

---

## Trainer code delta (`train_gpt.py` 2026-03-25 → 2026-04-09)

The newest trainer is LZMA+base85 wrapped (~16.6 KB), so diffs are against the decompressed source. Hyperparameters below are readable in each record's README or reproduction command.

| Category | 2026-03-25 (PR #1019) | 2026-04-09 (PR #1493) |
|---|---|---|
| **Vocab / tokenizer** | SentencePiece BPE 1024 | **SentencePiece BPE 8192** |
| **Train seq len** | 2048 | 2048 |
| **Model dim** | 512 | 512 |
| **Embed dim** | 512 (tied to model_dim) | 512 (`EMBEDDING_DIM` decoupled) |
| **Layers (physical)** | 11 U-Net | 11 U-Net |
| **Layers (virtual)** | 11 | **17** (3-layer recurrence on layers 3,4,5) |
| **MLP expansion** | 3× (hidden=1536) | **4×** (hidden=2048) |
| **Attention** | 8H / 4KV GQA, Partial RoPE 16/64 | (same) |
| **XSA** | All 11 layers | All 11 layers |
| **Parallel residuals** | No | **Yes, `PARALLEL_RESIDUAL_START=7`** |
| **Recurrence** | No | **Yes, `LOOP_LAYERS=3,4,5`, `ENABLE_LOOPING_AT=0.35`** |
| **Skip gates** | No | **Yes, `SKIP_GATES_ENABLED=1`** |
| **Activation** | LeakyReLU(0.5)² | LeakyReLU(0.5)² |
| **QK gain init** | 1.5 | **5.25** (`QK_GAIN_INIT`) |
| **Logit softcap** | 30 | 30 |
| **Optimizer** | Parallel Muon + AdamW | **MuonEq-R (`MUON_ROW_NORMALIZE=1`) + AdamW** |
| **Muon WD** | 0.04 | **0.095** (`MUON_WD`) |
| **Embed WD** | ~0 (implicit) | **0.085** (`EMBED_WD`) |
| **Adam WD** | 0.04 | **0.02** (`ADAM_WD`) |
| **MLR** | 0.025 | 0.022 |
| **Warmdown** | Iteration-based (≈4000) | **Fractional, `WARMDOWN_FRAC=0.72`** |
| **EMA / SWA** | SWA snapshots + EMA 0.997 | **EMA only, `EMA_DECAY=0.9965`** |
| **BigramHash** | 3072×112 | **Removed** |
| **SmearGate** | Yes | **Removed** |
| **Value embeddings** | Layers 9–10 | **Removed** |
| **QAT** | Int6 STE at warmdown > 15% | **Removed** (post-training GPTQ only) |
| **GPTQ calibration** | AR self-gen, 64×2048 | **Training-time Hessian (PR #1060 lineage), 64 batches** |
| **GPTQ method** | Full-Hessian int6 + RTN on embeds | **SDClip Full-Hessian: `c=k·σ`, int6 matrices k=12.85, int8 embeds k=20** |
| **Selective pruning** | Yes (prune to −1,0,+1) | **Not needed** (SDClip fits natively under 16 MB) |
| **Compression** | LZMA preset=9 | **Byte-shuffle + Brotli-11 + LZMA code wrapper** |
| **MATRIX_BITS / EMBED_BITS** | Hard-coded 6/8 | **Configurable** (`MATRIX_BITS=6`, `EMBED_BITS=8`) |
| **TTT** | Dropped (neutral on prior stack) | **Legal Score-First TTT module** (disabled-by-default, enabled via `TTT_ENABLED=1`) |
| **TTT settings** | — | `TTT_CHUNK_TOKENS=32768`, `TTT_LR=0.005`, `TTT_EPOCHS=3`, `TTT_MOMENTUM=0.9`, cosine LR decay |
| **Sliding eval stride** | 64 | 64 |
| **Step time** | ~86.7 ms | ~129 ms |
| **Steps in 600 s** | ~6,927 | ~4,550 |
| **Code size** | plain source | **LZMA+base85 wrapped (~16.6 KB)** |

---

## Trend arc — what worked, what died

**Clear winners** (monotonic across the window, confirmed in ≥2 records):

| Technique | Confidence | Evidence |
|---|---|---|
| SP8192 tokenizer | **Very high** | PR #1394 step-function drop, carried by all subsequent records |
| Depth recurrence | **High** | Worked at 1-layer, 2-layer, 3-layer; step-time budget limits further depth |
| Parallel residuals (layer 7+) | **High** | Separately proven (PR #1204, PR #1412), additive with TTT (PR #1477) |
| SDClip `c=k·σ` | **High** | Principled + R²≈0.995 correlation; no subsequent record has reverted |
| GPTQ on embeddings | **High** | Carried through every SP8192 record |
| Legal score-first TTT | **High** | Three records used it; ~0.002 BPB consistent gain |
| QK-Gain monotonic scaling | **High** | 1.5 → 4.0 → 5.0 → 5.25, each a small gain |
| Higher muon/embed WD | **High** | 0.04 → 0.085 → 0.090 → 0.095, synergistic with compression |
| Byte-shuffle + Brotli-11 | **High** | Standard across the stack |
| MuonEq-R | Medium | ~0.001 BPB gain, zero cost |
| Fractional warmdown | Medium | Used in PR #1445 tuning |

**Died / removed** (present at 2026-03-25, explicitly stripped in 2026-04-09):

| Dropped technique | Reason |
|---|---|
| Non-legal TTT / SLOT | Compliance (Issue #1017); replaced by legal score-first TTT |
| BigramHash | Made obsolete by SP8192's larger vocab |
| SmearGate | Made obsolete by larger vocab + parallel residuals |
| Value embeddings (VE128 on layers 9–10) | Removed in PR #1394, no regression |
| SWA snapshots | Replaced by EMA 0.9965 |
| QAT (int6 STE) | Post-training SDClip-GPTQ is enough |
| Mixed int5/int6 | All-int6 possible once WD sweep freed headroom |
| Selective pruning to {−1, 0, +1} | SDClip fits natively under 16 MB |
| Parameter banking + distributed Muon | Removed for simplicity in PR #1218; not reintroduced |
| Coprime-stride data loader | Replaced by `ShuffledSequenceLoader` in PR #1394 |
| Pre-quant TTT on val data | Compliance |

**Saturation.** The current stack sits at 1.0810. The last few records gained only 0.001–0.002 BPB each. Within the existing architectural template (dense 11L × 512d + recurrence + parallel residuals + SDClip-GPTQ + Legal TTT), the easy wins are gone. Further progress likely needs one of:
- A **fundamentally different backbone** (MoE, SSMs / mamba-like, selective pruning, megakernel fusion, hybrid attention).
- A **bigger tokenizer experiment** (SP16384 tradeoff vs embedding bloat).
- **Non-standard TTT** — within compliance, richer adaptation objectives.
- **Improved compression** beyond brotli (e.g. per-group clip allocation, output-Hessian row importance — flagged in PR #1412's future-work section).

The non-record track shows 1-bit / ternary quantization at 1.1239 / 1.1570 BPB — unlocks exist outside the 10-min budget but are not competitive within it yet.

---

## Implications for the user's MoE direction

The user's own experiment (`records/track_10min_16mb/2026-04-02_SharedMLP_Rank100_16L/`, shared-MLP + rank-100 adapters on 16L) was built on the PR #1019 (1.1147) baseline and did not beat SOTA. The user is now pursuing **sparse MoE routing**.

**Recommended baseline for the next iteration:** fork `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`, **not** the PR #1019 baseline. The 0.0337 BPB gap vs the old baseline is not something MoE alone will close.

**Orthogonal to MoE (import into the MoE baseline):**
- SP8192 tokenizer + GPTQ-on-embeds + SDClip (`k=12.85` matrices, `k=20` embeds).
- MuonEq-R optimizer, fractional warmdown, stratified WD 0.095/0.085/0.02, EMA 0.9965.
- Byte-shuffle + Brotli-11 + LZMA code wrapper.
- Legal Score-First TTT at eval (2× BPB win "for free" once compliance scaffolding is in place).

**Competes with MoE for the same budget (pick carefully):**
- **Depth recurrence vs MoE-over-layers**: both use shared parameters to manufacture virtual depth. 3-layer recurrence is already worth ~0.005 BPB; MoE needs to beat that *net of its routing overhead and load-balancing loss*, not just match raw recurrence.
- **Parallel residuals vs MoE routing structure**: parallel residuals already separate attention/MLP streams into specialized lanes; a routed-MLP expert system is a generalization of this. If MoE routes the *MLP only* from layer 7+, it's most additive; if it routes attention too, it's more aggressive and more risky.
- **MLP 4× vs expert dimensioning**: with 4× MLP, MLP is ~70% of the artifact. MoE pays for this many times over unless experts are heavily compressed, bit-shared, or use rank factorization. Budgeting is load-bearing.

**Early signs to watch for in the MoE baseline:**
- Does pre-quant BPB match PR #1493 within ~0.005 BPB? If not, the routing is leaking capacity.
- Does the artifact stay under 16 MB after SDClip-GPTQ + brotli, or does the expert bank explode compression entropy?
- Does Legal TTT still move BPB by 0.002? (It will if the router is stable; it won't if TTT destabilizes routing.)

Prior project note (from `memory/project_shared_mlp_adapters_failed.md`): shared-MLP with rank-100 adapters did not work. Likely because rank-100 adapters + shared MLP burn 4 MB+ on adapters vs ~0 for depth recurrence, which achieves similar effective-depth benefits at zero parameter cost. Any MoE design should keep the **effective-depth vs artifact-cost** ratio better than the current SOTA's zero-cost recurrence.

---

## Pointers

Record folders (all under `/Users/andreichernov/Documents/Personal/parameter-golf/records/track_10min_16mb/`):

- `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/` — previous SOTA (last pulled baseline)
- `2026-03-31_ParallelResiduals_MiniDepthRecurrence/`
- `2026-04-01_Vocab4096_MLPMult4_WD085/`
- `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/`
- `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR/`
- `2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/`
- `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/` *(non-record)*
- `2026-04-06_SP8192_QK5_LegalTTT_1.0828/`
- `2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/`
- `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/` — **current SOTA**

Newest trainer (decompress via `lzma -d` on the base85 blob to read it): `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`.

Leaderboard (always current): top of `/Users/andreichernov/Documents/Personal/parameter-golf/README.md`.

Refreshed companion docs (same snapshot, different views):
- `doc/sota_architecture.md` — model card keyed to PR #1493.
- `doc/sota_profile.md` — compute + artifact breakdown for the new stack.
- `doc/techniques_impact.md` — per-technique BPB deltas (amended).
