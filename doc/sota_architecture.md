# Current SOTA Architecture — Parameter Golf Leaderboard

**Submission**: AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112
**Score**: 1.1147 BPB (3-seed mean, std 0.0004)
**Author**: abaybektursun | **Date**: 2026-03-25
**Artifact Size**: ~15.91 MB (limit: 16 MB)
**Previous SOTA**: 1.1194 BPB (PR #549, same author) | **Improvement**: -0.0046 BPB

---

## Model Card

| Property | Value |
|----------|-------|
| Type | Autoregressive causal LM (next-token prediction) |
| Context length (train) | 2048 tokens |
| Context length (eval) | 2048 tokens (sliding window, stride 64) |
| Vocabulary | 1024 BPE tokens (SentencePiece) |
| Tokenization | Byte-pair encoding via SentencePiece |
| Model dimension | 512 |
| Layers | 11 (U-Net: 5 encoder + 6 decoder) |
| Attention heads | 8 (head_dim = 64) |
| KV heads | 4 (Grouped Query Attention) |
| MLP expansion | 3x (hidden = 1536) |
| Tied embeddings | Yes (tok_emb reused as lm_head) |
| Positional encoding | Partial RoPE (16 of 64 dims) |
| Attention impl | Flash Attention 3 (causal) |
| Precision | BF16 training, int6 QAT, Full Hessian GPTQ int6 + LZMA preset=9 |
| Training hardware | 8x H100 SXM 80GB |
| Training time | 600s (~6,922 steps at 86.7ms/step) |
| Eval time | Standard sliding window only (no TTT) |
| TTT | Dropped (neutral/negative on this stack) |

---

## Architecture Overview

### Forward Pass Formulas

**Input Processing:**
```
x = TokenEmbed(ids) + BigramHash(ids)
x = SmearGate(RMSNorm(x))
x₀ = x                                ← saved, fed into every block via residual mixing
```

**Encoder (layers 0–4) — each layer saves its output:**
```
h₀ = Block₀(x, x₀)
h₁ = Block₁(h₀, x₀)
h₂ = Block₂(h₁, x₀)
h₃ = Block₃(h₂, x₀)
h₄ = Block₄(h₃, x₀)
```

**Decoder (layers 5–10) — skip connections added in reverse order:**
```
x  = Block₅( h₄ + w₀⊙h₄,  x₀)       ← skip from layer 4
x  = Block₆(  x + w₁⊙h₃,  x₀)       ← skip from layer 3
x  = Block₇(  x + w₂⊙h₂,  x₀)       ← skip from layer 2
x  = Block₈(  x + w₃⊙h₁,  x₀)       ← skip from layer 1
x  = Block₉(  x + w₄⊙h₀,  x₀)       ← skip from layer 0
x  = Block₁₀( x,           x₀)       ← no skip (6 decoders > 5 encoders)
```

**Output:**
```
logits = 30 · tanh( Linear(RMSNorm(x)) / 30 )       ← Linear reuses tok_emb weights
```

`w₀..w₄` are learnable 512-dim vectors (per-feature skip scaling).
`x₀` enters every block via residual mixing: `x_in = mix₀·x + mix₁·x₀`.

### U-Net Skip Connection Structure

The encoder and decoder layers are paired in reverse order, forming a U-shape.
Deep decoder layers receive shallow encoder features, and vice versa:

```
Encoder                                                    Decoder
───────                                                    ───────
Layer 0  ─── h₀ ──────────────────────────────── +w₄⊙h₀ → Layer 9
  ↓                                                          ↑
Layer 1  ─── h₁ ────────────────────────── +w₃⊙h₁ ────→ Layer 8
  ↓                                                          ↑
Layer 2  ─── h₂ ──────────────────── +w₂⊙h₂ ──────→ Layer 7
  ↓                                                          ↑
Layer 3  ─── h₃ ────────────── +w₁⊙h₃ ────────→ Layer 6
  ↓                                                          ↑
Layer 4  ─── h₄ ──────── +w₀⊙h₄ ──────────→ Layer 5
  ↓                ↑
  └──── main path ─┘                          Layer 10 (no skip)
                                                   ↓
                                                Output
```

### Full Architecture (Mermaid)

```mermaid
graph TD
    subgraph Input
        A[Token IDs<br/>seq_len = 2048] --> B[Token Embedding<br/>1024 × 512]
        A --> C[BigramHash<br/>3072 x 112 to 512]
        B --> D((+))
        C --> D
        D --> E[RMSNorm]
        E --> F[SmearGate]
    end

    F --> X0["x₀ (saved for all blocks)"]
    X0 --> ENC0

    subgraph Encoder
        ENC0[Block 0] --> ENC1[Block 1]
        ENC1 --> ENC2[Block 2]
        ENC2 --> ENC3[Block 3]
        ENC3 --> ENC4[Block 4]
    end

    ENC4 --> SKIP4["+ w₀ ⊙ h₄"]
    SKIP4 --> DEC0

    subgraph Decoder
        DEC0[Block 5] --> SKIP3["+ w₁ ⊙ h₃"]
        SKIP3 --> DEC1[Block 6]
        DEC1 --> SKIP2["+ w₂ ⊙ h₂"]
        SKIP2 --> DEC2[Block 7]
        DEC2 --> SKIP1["+ w₃ ⊙ h₁"]
        SKIP1 --> DEC3[Block 8]
        DEC3 --> SKIP0["+ w₄ ⊙ h₀"]
        SKIP0 --> DEC4[Block 9]
        DEC4 --> DEC5[Block 10<br/>no skip]
    end

    ENC4 -.->|h₄| SKIP4
    ENC3 -.->|h₃| SKIP3
    ENC2 -.->|h₂| SKIP2
    ENC1 -.->|h₁| SKIP1
    ENC0 -.->|h₀| SKIP0

    DEC5 --> FN[RMSNorm]
    FN --> LM["Tied LM Head<br/>reuses tok_emb weights"]
    LM --> SC["Logit Softcap<br/>30 · tanh(logits / 30)"]
    SC --> OUT[Next-Token Probabilities]

    style Input fill:#1a1a2e,stroke:#e94560,color:#fff
    style Encoder fill:#16213e,stroke:#0f3460,color:#fff
    style Decoder fill:#16213e,stroke:#533483,color:#fff
```

---

## Transformer Block Detail

```mermaid
graph TD
    subgraph Block["Transformer Block (each of 11 layers)"]
        X[x residual] --> MIX["Residual Mix<br/>x_in = mix0 * x + mix1 * x0"]
        X0[x0 original embed] --> MIX

        MIX --> AN[RMSNorm × LN_scale<br/>scale = 1/√ layer+1]

        AN --> ATTN

        subgraph ATTN["Causal Self-Attention (GQA)"]
            direction TB
            QKV["Q: 512→512 | K: 512→256 | V: 512→256<br/>(weights from Parameter Banks)"]
            QKV --> QKNORM["QK RMSNorm"]
            QKNORM --> ROPE["Partial RoPE<br/>16/64 dims rotated"]
            ROPE --> QGAIN["Q × learnable q_gain per head"]
            QGAIN --> FA3["Flash Attention 3<br/>(causal)"]
            FA3 --> XSAOP["XSA: all 11 layers<br/>y = y - proj_v(y)"]
            XSAOP --> OUTPROJ["Out projection 512 to 512"]
        end

        ATTN --> ASCALE["× attn_scale"]
        MIX --> ARES((+))
        ASCALE --> ARES

        ARES --> MN[RMSNorm × LN_scale]

        subgraph MLP_BLOCK["MLP (3x expansion)"]
            direction TB
            UP["Linear 512 → 1536<br/>(from mlp_up_bank)"]
            UP --> ACT["LeakyReLU 0.5 → square<br/>leaky_relu(x, 0.5)²"]
            ACT --> DOWN["Linear 1536 → 512<br/>(from mlp_down_bank)"]
        end

        MN --> MLP_BLOCK
        MLP_BLOCK --> MSCALE["× mlp_scale"]
        ARES --> MRES((+))
        MSCALE --> MRES
        MRES --> XOUT[x output]
    end

    style Block fill:#0a0a23,stroke:#e94560,color:#fff
    style ATTN fill:#1a1a2e,stroke:#0f3460,color:#fff
    style MLP_BLOCK fill:#1a1a2e,stroke:#533483,color:#fff
```

---

## Value Embedding Injection (Layers 9-10 only)

```mermaid
graph LR
    TID[Token IDs] --> VE[Shared ValueEmbedding<br/>1024 → 128d → 256d]
    VE --> SCALE["× per-layer scale"]
    SCALE --> VADD((+ added to V<br/>before attention))
    V_PROJ["V projection output"] --> VADD

    style VE fill:#1a1a2e,stroke:#e94560,color:#fff
```

---

## Tokenization Pipeline

```mermaid
graph LR
    RAW[Raw UTF-8 Text] --> SP["SentencePiece BPE<br/>vocab = 1024 tokens"]
    SP --> IDS[Token IDs ∈ 0..1023]
    IDS --> EMB["Token Embedding<br/>1024 × 512"]
    IDS --> BH["BigramHash<br/>XOR hash of adjacent token pairs<br/>3072-entry table, 112d to 512d"]
    EMB --> ADD((+))
    BH --> ADD
    ADD --> MODEL[Into Transformer]

    style SP fill:#16213e,stroke:#0f3460,color:#fff
```

---

## Parameter Banks & Parallel Muon Optimizer

```mermaid
graph TD
    subgraph Banks["4 Parameter Banks (contiguous 3D tensors)"]
        QO["qo_bank<br/>[22, 512, 512]<br/>Q + Out for 11 layers"]
        KV["kv_bank<br/>[22, 256, 512]<br/>K + V for 11 layers"]
        UP["mlp_up_bank<br/>[11, 1536, 512]"]
        DN["mlp_down_bank<br/>[11, 512, 1536]"]
    end

    subgraph Muon["Parallel Muon Optimizer Pipeline"]
        direction TB
        BW["backward()"] --> RS["Async Reduce-Scatter<br/>(biggest banks first)"]
        RS --> ADAM["Meanwhile: Adam steps on<br/>small params (scales, gates, embeds)"]
        RS --> WAIT["Wait for RS"]
        WAIT --> NS5["Local Newton-Schulz 5-step<br/>orthogonalization on gradient shard"]
        NS5 --> AG["Async All-Gather"]
        AG --> APPLY["Apply update:<br/>w -= lr × scale × NS5(grad)"]
    end

    Banks --> Muon

    style Banks fill:#16213e,stroke:#0f3460,color:#fff
    style Muon fill:#1a1a2e,stroke:#533483,color:#fff
```

---

## Weight Averaging & Quantization

```mermaid
graph LR
    subgraph Training
        STEP[Training Step] --> EMA["EMA decay=0.997<br/>shadow weights"]
        EMA --> SWA["SWA snapshot every 50 steps<br/>averaged with prior snapshots"]
    end

    subgraph Quantization
        SWA --> LQAT["Late QAT<br/>int6 STE fake-quant<br/>kicks in at warmdown > 15%"]
        LQAT --> SELFGEN["AR Self-Generation<br/>64 seqs x 2048 tokens<br/>temp=0.8, fixed seed"]
        SELFGEN --> GPTQ["Full Hessian GPTQ int6<br/>Cholesky error compensation<br/>column reordering<br/>H = X^T X from self-gen data"]
        GPTQ --> PRUNE["Selective Pruning<br/>prune values to -1, 0, +1<br/>by reconstruction error"]
        PRUNE --> LZMA["LZMA preset=9"]
        LZMA --> ART["Artifact ~15.91 MB"]
    end

    style Training fill:#16213e,stroke:#0f3460,color:#fff
    style Quantization fill:#1a1a2e,stroke:#e94560,color:#fff
```

### AR Self-Generated GPTQ (key innovation)

The previous SOTA used GPTQ-lite (diagonal Hessian approximation). This submission uses **Full Hessian GPTQ** — a strictly better quantizer with Cholesky error compensation and column reordering.

The problem: Full Hessian GPTQ needs calibration data to compute `H = X^T X`. Prior attempts used training data, which was ruled illegal after the 600s training window.

The solution: **the model generates its own calibration data**. After training completes, it autoregressively generates 64 sequences of 2048 tokens (temperature=0.8). No validation data, no training data accessed during quantization. This is fully self-contained.

### TTT: Dropped

The previous SOTA (PR #549) used Test-Time Training for -0.0025 BPB. On this new stack, TTT was tested 25+ times and found **neutral or negative**. The Full Hessian GPTQ improvement more than compensates.

---

## What Changed from Previous SOTA (PR #549, 1.1194 BPB)

| Change | Previous (1.1194) | Current (1.1147) | Impact |
|--------|-------------------|-------------------|--------|
| **Quantization** | GPTQ-lite (diagonal Hessian) | Full Hessian GPTQ with AR self-gen calibration | Major improvement |
| **XSA** | Last 4 layers only | All 11 layers | Free quality (zero params) |
| **BigramHash** | 1536 x 128 | 3072 x 112 | Wider table, slightly narrower dim |
| **TTT** | Score-first TTT (-0.0025 BPB) | Dropped (neutral on this stack) | Simplifies eval |
| **Selective pruning** | No | Yes (prune to -1, 0, +1 by reconstruction error) | Better compression |
| **Warmdown** | 3500 iters | 4000 iters | Slightly longer cooldown |
| **Compression** | LZMA | LZMA preset=9 | Max compression |

### Lineage

```
PR #549 (1.1194) — Parallel Muon + LeakyReLU² + Legal TTT
    └── PR #1019 (1.1147) — This work adds:
        ├── AR self-gen Full Hessian GPTQ (no external data during quantization)
        ├── BigramHash 3072 x 112
        ├── XSA on all 11 layers (from PR #478)
        ├── Selective pruning to -1, 0, +1 (from PR #609)
        ├── Warmdown 4000, LZMA preset=9
        └── Dropped TTT (25+ failed experiments, PR #756)
```