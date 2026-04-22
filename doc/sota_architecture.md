# Current SOTA Architecture — Parameter Golf Leaderboard

**Submission**: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT
**Score**: 1.0810 BPB (3-seed mean, std 0.0002)
**Author**: bigbag (PR #1493) | **Date**: 2026-04-09
**Artifact Size**: ~15.99 MB (limit: 16 MB)
**Previous SOTA**: 1.0822 BPB (PR #1477, aryanbhosale) | **Improvement**: −0.0012 BPB
**Last-pulled-before-update SOTA**: 1.1147 BPB (PR #1019) | **Cumulative improvement since last pull**: −0.0337 BPB

> A companion doc, `leaderboard_update_2026-04-21.md`, walks through all 9 records merged between the last pull and this snapshot. A compute-and-artifact breakdown lives in `sota_profile.md`. The per-technique BPB delta table lives in `techniques_impact.md`.

---

## Model Card

| Property | Value |
|----------|-------|
| Type | Autoregressive causal LM (next-token prediction) |
| Context length (train) | 2048 tokens |
| Context length (eval) | 2048 tokens (sliding window, stride 64) |
| Vocabulary | **8192 BPE tokens (SentencePiece)** |
| Tokenization | SentencePiece BPE, tokenizer model ~100 KB |
| Model dimension | 512 |
| Embedding dimension | 512 (`EMBEDDING_DIM`, decoupled from model_dim) |
| Layers (physical) | 11 (U-Net: 5 encoder + 6 decoder) |
| Layers (virtual) | **17** (3-layer recurrence on layers 3, 4, 5) |
| Attention heads | 8 (head_dim = 64) |
| KV heads | 4 (Grouped Query Attention) |
| XSA | All 11 layers |
| MLP expansion | **4× (hidden = 2048)** |
| Tied embeddings | Yes (tok_emb reused as lm_head) |
| Positional encoding | Partial RoPE (16 of 64 dims) |
| QK scaling | Learnable per-head q_gain, init **5.25** |
| Residual structure | **Sequential layers 0–6, Parallel (GPT-J style) layers 7–10** |
| Skip connections | U-Net + **sigmoid skip gates** (`SKIP_GATES_ENABLED=1`) |
| Activation | LeakyReLU(0.5)² |
| Logit softcap | 30.0 |
| Attention impl | Flash Attention 3 (causal) |
| Precision | BF16 training, **SDClip Full-Hessian GPTQ**: int6 matrices (k=12.85), int8 embeddings (k=20.0); **byte-shuffle + Brotli-11**; LZMA code wrapper (~16.6 KB code) |
| Training hardware | 8× H100 SXM 80 GB |
| Training time | ~588 s (~4,550 steps) |
| Eval time | ~500 s total (sliding + TTT) |
| TTT | **Legal Score-First TTT** (Issue #1017 Track B): 32K-token chunks, 3 epochs × SGD lr=0.005 momentum=0.9 with cosine LR decay; score-before-update per chunk |

---

## Architecture Overview

### Depth Recurrence — Virtual-Layer Construction

The 11 physical blocks run as a 17-virtual-layer sequence once looping is active (activated at `ENABLE_LOOPING_AT = 0.35`, i.e. 35 % of training):

```
Encoder virtual sequence  :  0  1  2  3  4  5  3  4
Decoder virtual sequence  :  5  3  4  5  6  7  8  9  10
(Encoder length 8, Decoder length 9 — 17 virtual layers total.)
```

Layers 3, 4, 5 are each visited twice inside the loop. Physical-param count is unchanged; effective depth roughly matches a 17-layer dense network on the forward/backward paths where recurrence is active.

### Parallel Residuals — Layer 7+

Layers 0–6 use the classic sequential residual: `h = x + attn_scale · Attn(norm(x)); x_out = h + mlp_scale · MLP(norm(h))`.

Layers 7–10 compute attention and MLP **in parallel** from the same input (GPT-J style):

```
x_out = x + attn_scale · Attn(norm(x)) + mlp_scale · MLP(norm(x))
```

Attention and MLP see the same pre-block input, and their outputs are summed into a single residual. A learned `lane_merge` scalar (init 0.5) blends the attention/MLP lanes in the submissions that carry the two-lane variant (PR #1477); in PR #1493 the recipe is single-lane parallel residual.

### Forward Pass (high-level)

```
ids  : [B, T]                                           (T = 2048, V = 8192)
e    = TokEmbed(ids)                                     [B, T, 512]      ← int8 SDClip GPTQ
                                                             no BigramHash, no SmearGate, no VE128

e, state = loop_active ? VirtualLayerSequence(e) : DenseLayerSequence(e)
           where each block is:
             z     = RMSNorm(x) * LN_scale
             attn  = FlashAttn3( Partial-RoPE-Q(z)*q_gain, K(z), V(z) )
             (attention projections go through SDClip-int6 Q/K/V/O)
             if layer >= PARALLEL_RESIDUAL_START (=7):
                 mlp   = MLP( RMSNorm(x) * LN_scale )
                 x_out = x + attn_scale * attn + mlp_scale * mlp
             else:
                 x_out = x + attn_scale * attn
                 x_out = x_out + mlp_scale * MLP( RMSNorm(x_out) * LN_scale )

logits = 30 * tanh( Linear(RMSNorm(e)) / 30 )            Linear weights tied to TokEmbed
```

### U-Net Skip Gates

The encoder/decoder pairing is preserved, but each skip is now gated:
`x = lerp(skip, x, sigmoid(skip_gate))`
`skip_gate` is a small per-feature learnable parameter (`SKIP_GATES_ENABLED=1`). This replaces the hard additive skip (`x = x + w_i ⊙ skip_i`) of the PR #1019 era.

---

## Training Recipe

### Optimizer

- **MuonEq-R** on weight matrices: row-normalize the gradient rows, then Newton–Schulz-5 orthogonalization, then apply update (`MUON_ROW_NORMALIZE = 1`).
- **AdamW** for embeddings and scalars.
- **Stratified weight decay**: `MUON_WD = 0.095`, `EMBED_WD = 0.085`, `ADAM_WD = 0.02`.
- **Stratified learning rates**: `matrix_lr (MLR) = 0.022`, plus separate embed / tied-embed / head / scalar LRs.
- **Fractional warmdown**: linear decay to LR = 0 over the final `WARMDOWN_FRAC = 0.72` of training.
- **EMA only** (no SWA): `EMA_DECAY = 0.9965`.

### Depth Recurrence Activation

- `LOOP_LAYERS = 3, 4, 5`
- `NUM_LOOPS = 1` extra pass (i.e. each of {3,4,5} is visited twice)
- `ENABLE_LOOPING_AT = 0.35` (activate looping at 35 % of the total step schedule)

Running without recurrence first keeps step time cheap early, then absorbs the recurrence cost after the model has a reasonable initial representation.

### Quantization (post-training)

**SDClip** replaces quantile clip search:

```
σ_row    = rowwise std of weight matrix
clip     = k · σ_row
bits     = MATRIX_BITS (=6) for matrices, EMBED_BITS (=8) for token embeddings
k        = MATRIX_CLIP_SIGMAS (=12.85) for matrices, EMBED_CLIP_SIGMAS (=20.0) for embeddings
```

Why: the compressed size of a weight matrix is dominated by the entropy of the quantized values `H(q)`. `H(q)` is controlled by the clip level more than by the bitwidth, and `σ_row` is a cheap principled proxy. Empirically, `RMS(weight) ↔ compressed_ratio` has R² ≈ 0.995 in this regime.

Full-Hessian GPTQ runs on all matrices *including* the token embedding (PR #1394 change — embeddings used to be RTN-quantized). 64 calibration batches; Hessian computed in-training within the 600 s budget.

Artifact pipeline: `SDClip-GPTQ → byte-shuffle → Brotli-11 → final artifact`. No selective pruning is needed (the model fits under 16 MB natively). The `train_gpt.py` source itself is wrapped as `exec(lzma.decompress(base85_blob, …))`, saving ~43 KB vs plain source.

### Test-Time Training — Legal Score-First

Module gated behind `TTT_ENABLED = 1`. Loop per validation chunk:

```python
for chunk in chunks:
    # Phase 1 — SCORE (frozen model)
    with torch.inference_mode():
        nll = cross_entropy(model(batch), targets)
    loss_sum += nll.sum()

    # Phase 2 — TRAIN on just-scored chunk
    if not is_last_chunk:
        for _ in range(TTT_EPOCHS):           # = 3
            for x, y in chunk_seqs:
                (model(x, y)).backward()
                sgd_step(lr = cosine(TTT_LR = 0.005),
                         momentum = TTT_MOMENTUM = 0.9,
                         clip_norm = 1.0)
```

Chunk size `TTT_CHUNK_TOKENS = 32768`. Each chunk fully scored before it is trained on; no rescoring; no updates affect tokens that have already been scored.

**Compliance** (Issue #1017 Track B — legal eval-time adaptation):
1. Causality — strictly causal sliding-window eval.
2. Normalized distribution — standard softmax over the full 8192-token vocab, no n-gram cache, no logit biasing.
3. Score-before-update — every token scored under `inference_mode` before any gradient update.
4. Single pass — each token scored exactly once.
5. No SLOT, no pre-quant TTT on val data, no eval-time logit bias (ETLB).

Eval budget 600 s (sliding + TTT); actual ~500 s on all seeds.

---

## 3-Seed Results (PR #1493)

| Seed | Sliding BPB | **TTT BPB** | Artifact bytes |
|---|---|---|---|
| 42  | 1.0829 | **1.0808** | 15,991,930 |
| 314 | 1.0827 | **1.0810** | 15,992,919 |
| 999 | 1.0826 | **1.0812** | 15,993,232 |
| **Mean** | **1.0827** | **1.0810** | **15,992,694** |
| **Std** | 0.0002 | 0.0002 | |

Pre-TTT → post-TTT delta: ~0.002 BPB.

---

## What Changed from the Last-Pulled SOTA (PR #1019, 1.1147 BPB)

| Change | 2026-03-25 (PR #1019) | 2026-04-09 (PR #1493) | Approx. impact |
|--------|----|----|----|
| Vocab | 1024 BPE | **8192 BPE** | **~−0.016 BPB** (cumulative across the window) |
| MLP expansion | 3× | **4×** | part of the Apr 1 pivot |
| Virtual depth | 11 | **17** (3-layer recurrence) | ~−0.005 BPB |
| Parallel residuals | — | **Yes, from layer 7** | ~−0.002 to −0.003 BPB |
| Optimizer | Parallel Muon + AdamW | **MuonEq-R + AdamW** | ~−0.001 BPB |
| Weight decay | 0.04 (muon, lumped) | **0.095 muon / 0.085 embed / 0.02 adam** | compression headroom |
| QK gain init | 1.5 | **5.25** (monotonic 1.5→4.0→5.0→5.25) | small, monotonic |
| Quantization | Full-Hessian GPTQ, RTN on embeds | **SDClip `c=k·σ`**; GPTQ on embeds | ~−0.001 BPB + artifact headroom |
| BigramHash, SmearGate, value-emb, QAT | Yes | **All removed** | null/negative on new stack |
| SWA | Yes | **EMA only, 0.9965** | simplification |
| Selective pruning | Yes (−1, 0, +1) | **Not needed** | SDClip fits natively |
| Compression | LZMA preset=9 | **Byte-shuffle + Brotli-11 + LZMA code wrapper** | −43 KB from code wrapper |
| TTT | Dropped | **Legal Score-First TTT** (Track B compliant) | ~−0.002 BPB |

Lineage:

```
PR #1019 (1.1147)
 └── PR #1204 (Apr 1, 1.1063) — parallel residuals + mini depth recurrence
 └── PR #1218 (Apr 1, 1.0979) — SP4096 pivot: MLP 4×, WD 0.085, strip tricks, SDClip precursors
       └── PR #1285 (Apr 3, 1.0912) — WD 0.090, MuonEq-R, all-int6, DR layers 4–5
            └── PR #1334 (Apr 4, 1.0897) — + parallel residuals, QK-Gain 5.0 (still SP4096)
       └── PR #1394 (Apr 5, 1.0856) — SP8192, GPTQ-on-embeds, SDClip, loop 4–5 twice
            ├── PR #1412 (Apr 6, non-rec 1.0835) — Hessian-SDClip, progressive recurrence
            ├── PR #1413 (Apr 6, 1.0828) — QK-Gain 5.0 + Legal Score-First TTT
            └── PR #1477 (Apr 8, 1.0822) — + parallel residuals on TTT stack
                 └── PR #1437 + PR #1445 + PR #1493 (Apr 9, 1.0810) — 3-layer recurrence, QK 5.25, WD 0.095, fractional warmdown
```

---

## Appendix — Previous SOTA Model Card (PR #1019, 1.1147 BPB, 2026-03-25)

Retained here as a historical reference. This was the snapshot the user's earlier experiment folders were built on. **Do not use as a baseline for new work** — the gap to current SOTA is −0.0337 BPB.

| Property | Value |
|----------|-------|
| Submission | AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112 |
| Score | 1.1147 BPB (3-seed mean, std 0.0004) |
| Author | abaybektursun (PR #1019) | Date 2026-03-25 |
| Artifact | ~15.91 MB |
| Vocab | 1024 BPE (SentencePiece) |
| Context | 2048 train / 2048 eval sliding stride 64 |
| Layers | 11 U-Net (5 enc + 6 dec, with value embeddings on layers 9–10) |
| Dim / MLP | 512 dim, 3× MLP (hidden 1536) |
| Attention | 8 heads, 4 KV (GQA), Partial RoPE (16/64), XSA on all 11 layers |
| Activation | LeakyReLU(0.5)² |
| Input extras | TokEmbed + BigramHash(3072×112→512), then SmearGate(RMSNorm(·)) |
| Optimizer | Parallel Muon + AdamW, WD=0.04 |
| Weight averaging | EMA decay 0.997 + SWA snapshot every 50 steps |
| Quantization | Late QAT (int6 STE at warmdown > 15 %) + Full-Hessian GPTQ int6 (AR self-gen calibration, 64×2048 temp=0.8) + selective pruning {−1, 0, +1} |
| Compression | LZMA preset=9 |
| Training | ~6,927 steps at 86.7 ms/step in 600 s on 8× H100 SXM |
| Eval | Sliding window only (TTT dropped as neutral/negative on this stack) |

The PR #1019 recipe is superseded. Its individual pieces that survived into PR #1493: 11L × 512d × GQA(8H/4KV), Partial RoPE 16/64, LeakyReLU(0.5)², tied embeddings, logit softcap 30.0, XSA on all layers, Full-Hessian GPTQ on matrices. Everything else (BigramHash, SmearGate, VE128, QAT, SWA, selective pruning, AR self-gen calibration, parameter banking + distributed-muon boilerplate) was removed.
