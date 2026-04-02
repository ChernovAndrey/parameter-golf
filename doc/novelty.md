# XSA — Exclusive Self Attention

**Paper**: [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) (Shuangfei Zhai, 2026)
**Adopted in**: OpenAI Parameter Golf challenge (PRs #198, #478, #609, #1019)
**Used in current SOTA**: Yes, applied to all 11 layers (1.1147 BPB)

---

## The Problem XSA Solves

In standard self-attention, each token's output is a weighted sum of all value vectors:

```
y[t] = sum_s  alpha[t,s] * v[s]
```

where `alpha[t,s] = softmax(q[t] . k[s])`.

Nothing prevents `alpha[t,t]` (self-attention weight) from dominating. When it does, the output collapses to:

```
y[t] ≈ alpha[t,t] * v[t]    (mostly just copying the token's own value)
```

This is wasteful — the head spends its capacity repeating information the token already has, instead of aggregating context from other positions. The original paper calls this **self-attention bias** and shows it increases in deeper layers.

---

## The XSA Solution

After computing standard attention output, **subtract the component that points along the token's own value direction**:

```
v̂[t] = v[t] / ||v[t]||                          (normalize)

y[t] = y[t] - (y[t] . v̂[t]) * v̂[t]             (remove self-value projection)
```

This is a vector projection removal. Geometrically:

```
                    y[t] (original attention output)
                   /
                  /
                 / ) theta
                /___________  v̂[t] (self-value direction)

    After XSA:

                    y'[t] (orthogonal to v̂)
                   |
                   |
                   |___________  v̂[t]
```

The output `y'[t]` is now **orthogonal to the token's own value**, meaning it can only contain information aggregated from **other** positions.

---

## Why It Helps Small Models

In a large model (e.g. 96 heads), heads naturally specialize — some attend to self, others to context. In a tiny model (8 heads, 4 KV heads), every head matters. XSA forces all heads to carry cross-position information:

| Without XSA | With XSA |
|-------------|----------|
| Head can "cheat" by attending to self | Self-value component removed |
| Some heads redundantly copy values | All heads must aggregate context |
| Wastes capacity in parameter-constrained models | Maximizes information per head |

The constraint is strict: zero parameter cost, applied as a post-attention projection.

---

## Mathematical Formulation

### Standard Attention

```
Q = X * W_q     ∈ R^{B x T x H x D}
K = X * W_k     ∈ R^{B x T x H_kv x D}
V = X * W_v     ∈ R^{B x T x H_kv x D}

y = softmax(Q K^T / sqrt(d)) V       ∈ R^{B x T x H x D}
```

### XSA Post-Processing

For each position `t` and each KV head group:

```
v̂[t] = v[t] / ||v[t]||_2

y'[t] = y[t] - (y[t] . v̂[t]) v̂[t]
```

This is equivalent to multiplying by a projection matrix:

```
y'[t] = (I - v̂[t] v̂[t]^T) y[t]
```

where `I - v̂ v̂^T` is the **orthogonal projector** onto the subspace perpendicular to `v̂`.

### Properties

- **Idempotent**: Applying XSA twice gives the same result (it's a projection)
- **Zero parameters**: No learnable weights added
- **Output is orthogonal to v**: `y'[t] . v[t] = 0` (by construction)
- **Preserves all non-self information**: Components from other positions' values are kept

---

## GQA-Aware Efficient Implementation

The challenge models use **Grouped Query Attention** (8 query heads, 4 KV heads). A naive XSA implementation would require `repeat_interleave` to expand `v` from 4 to 8 heads, doubling memory.

### Naive (expensive)

```python
# v: [B, T, 4, D]  →  expand to [B, T, 8, D]
v_expanded = v.repeat_interleave(2, dim=-2)      # allocates 2x memory
vn = F.normalize(v_expanded, dim=-1)
proj = (y * vn).sum(-1, keepdim=True) * vn
y = y - proj
```

### Efficient (zero allocation)

```python
# y: [B, T, 8, D]  →  reshape into KV head groups: [B, T, 4, 2, D]
B, T, H, D = y.shape
Hkv = v.size(-2)                                  # 4
group = H // Hkv                                  # 2

y_g = y.reshape(B, T, Hkv, group, D)             # [B, T, 4, 2, D]  (free view)
vn = F.normalize(v, dim=-1).unsqueeze(-2)         # [B, T, 4, 1, D]  (broadcast)
proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn  # broadcast across group dim
y = (y_g - proj).reshape(B, T, H, D)
```

**Key insight**: Reshape `y` from `[B, T, 8, D]` to `[B, T, 4, 2, D]` — grouping the 2 query heads that share each KV head. Then `vn` with shape `[B, T, 4, 1, D]` broadcasts naturally across the group dimension. No memory duplication.

**Performance**: ~2ms/step overhead (down from ~7ms naive) at 11 layers with GQA (8H/4KV).

---

## Application Strategies in the Challenge

The competition explored different strategies for how many layers to apply XSA to:

| Submission | Layers with XSA | Score (BPB) | Rationale |
|------------|----------------|-------------|-----------|
| PR #198 (2026-03-20) | Last 3 of 11 | 1.1307 | Self-attention bias is highest in deep layers |
| PR #287 (2026-03-21) | Last 4 of 11 | 1.1248 | Slightly wider coverage |
| PR #549 (2026-03-23) | Last 4 of 11 | 1.1194 | Same as above + other improvements |
| PR #1019 (2026-03-25) | All 11 layers | 1.1147 | Full coverage, zero param cost |

The progression shows that applying XSA to **all** layers works best, despite the original paper's finding that self-attention bias concentrates in deep layers. In this parameter-constrained setting, even shallow layers benefit from the constraint.

---

## Where XSA Sits in the Forward Pass

```
          Q, K, V projections
                │
        Flash Attention 3 (causal)
                │
           y = Attn(Q,K,V)
                │
          ┌─────┴─────┐
          │   XSA:     │
          │   remove   │
          │   self-v   │
          │   projection│
          └─────┬─────┘
                │
           y' (orthogonal to v)
                │
          Out projection
                │
          + residual
```

XSA operates between Flash Attention output and the output projection. It's a simple linear operation (projection onto orthogonal complement) that requires no additional parameters or gradient computation beyond the existing `v` tensor.

---

## Limitations

- **Information loss**: By removing the self-value component, the model loses direct access to the token's own value representation through attention. This is compensated by the residual connection (which bypasses attention entirely).
- **Fixed constraint**: Unlike a learnable gate, XSA always fully removes the self-value component. A softer version (partial removal) might be better in some settings.
- **Assumes self-value redundancy**: If the self-value contains genuinely useful information not available through the residual, XSA hurts. This appears not to be the case in practice.

---

## XSA References

- **Original paper**: Shuangfei Zhai, *Exclusive Self Attention*, arXiv:2603.09078, 2026
- **Efficient GQA implementation**: PR #198 by @unnir (2026-03-20)
- **All-layer extension**: PR #478 by @gowtham0992 (2026-03-23)
- **Current SOTA integration**: PR #1019 by @abaybektursun (2026-03-25)

---
---

# BigramHash — Cheap N-gram Context Without Attention

**Introduced in**: PR #162 by @raahilshah (concept), iterated in PRs #549, #609, #1019
**Used in current SOTA**: Yes, 3072 buckets x 112-dim
**Parameter cost**: ~0.38M (tiny vs ~26M total model)

---

## The Problem BigramHash Solves

The model uses a **1024-token BPE vocabulary** — extremely small. For comparison, GPT-2 has 50,257 tokens. With so few tokens, individual tokens are very short fragments (often 1-3 characters). A single token like `"un"` or `"at"` carries almost no meaning on its own.

**Context is critical**: `"un"` followed by `"do"` means "undo". `"un"` followed by `"it"` means "unit". The model needs bigram (token-pair) context to disambiguate.

Standard self-attention can learn this, but it's expensive:
- Requires a full Q/K/V computation + softmax + output projection
- For the very first layer, the model sees tokens with no context at all

BigramHash injects **free bigram context at the input**, before any attention computation.

---

## The Algorithm Step by Step

### Step 1: Hash adjacent token pairs

For a sequence of token IDs `[t₀, t₁, t₂, t₃, ...]`, compute a hash for each adjacent pair:

```
Position 0:  no previous token → use special index (mod - 1 = 3071)
Position 1:  hash(t₁, t₀) = XOR(36313 * t₁, 27191 * t₀) mod 3071
Position 2:  hash(t₂, t₁) = XOR(36313 * t₂, 27191 * t₁) mod 3071
Position 3:  hash(t₃, t₂) = XOR(36313 * t₃, 27191 * t₂) mod 3071
...
```

In code:
```python
def bigram_hash(self, tokens):
    t = tokens.to(torch.int32)
    mod = self.bigram_vocab_size - 1       # 3071
    out = torch.empty_like(t)
    out[..., 0] = mod                       # position 0: no bigram, use last bucket
    out[..., 1:] = torch.bitwise_xor(
        36313 * t[..., 1:],                # current token * prime
        27191 * t[..., :-1]                 # previous token * different prime
    ) % mod
    return out.long()
```

**Why these specific numbers?**
- `36313` and `27191` are **primes** — multiplication by distinct primes followed by XOR creates a good hash distribution
- `mod 3071` maps all possible token pairs (1024 x 1024 = 1M combinations) into 3071 buckets
- This means ~341 token pairs share each bucket on average — some collision, but the embedding learns to capture the most useful shared patterns

### Step 2: Look up embeddings

Each hash index maps to a **112-dimensional learned embedding**:

```
hash_indices:  [3071,  1842,  507,   2931,  ...]
                 ↓       ↓      ↓      ↓
embeddings:   [e₀,    e₁,    e₂,    e₃,   ...]    each ∈ R^112
```

The embedding table has shape `[3072, 112]` — that's 3072 * 112 = ~344K parameters.

### Step 3: Project to model dimension

The 112-dim embeddings are projected to the model's 512-dim via a linear layer:

```
h = Embedding[hash_index]           ∈ R^112
h = Linear_proj(h)                  ∈ R^512    (weight: 112 x 512 = ~57K params)
h = h * scale                       scale is a learnable scalar (init: 0.05)
```

### Step 4: Add to token embedding

The projected bigram embedding is simply added to the standard token embedding:

```
x = TokenEmbed(token_id) + BigramHash(token_id, prev_token_id)
```

This happens **before** RMSNorm and SmearGate, at the very start of the forward pass.

---

## Concrete Example

Given text `"the cat"` tokenized as `[t_"th", t_"e", t_" c", t_"at"]`:

```
Position 0: "th"  → hash = 3071 (no previous)     → Embed[3071] → project → add
Position 1: "e"   → hash("e", "th")  = XOR(36313*e_id, 27191*th_id) % 3071
                                      = some bucket, say 1842
                                      → Embed[1842] → project → add
Position 2: " c"  → hash(" c", "e")  = XOR(36313*c_id, 27191*e_id) % 3071
                                      = some bucket, say 507
                                      → Embed[507] → project → add
Position 3: "at"  → hash("at", " c") = XOR(36313*at_id, 27191*c_id) % 3071
                                      = some bucket, say 2931
                                      → Embed[2931] → project → add
```

Each position now carries information about **which two tokens appeared together**, before any attention is computed.

---

## Full Data Flow

```
Token IDs: [t₀, t₁, t₂, ..., t_T]
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
TokenEmbed             BigramHash
1024 x 512             ┌──────────────────────────┐
    │                  │ 1. Hash adjacent pairs     │
    │                  │    XOR(36313*t, 27191*t₋₁) │
    │                  │    mod 3071                 │
    │                  │                            │
    │                  │ 2. Lookup: 3072 x 112      │
    │                  │                            │
    │                  │ 3. Project: 112 → 512      │
    │                  │                            │
    │                  │ 4. Scale by learned scalar  │
    │                  │    (init 0.05)             │
    │                  └─────────┬──────────────────┘
    │                            │
    └──────────┬─────────────────┘
               ▼
              (+)  element-wise add
               │
               ▼
           RMSNorm
               │
               ▼
           SmearGate
               │
               ▼
        Into Transformer blocks
```

---

## Why Hash Instead of a Full Bigram Table?

A full bigram embedding table would be `1024 x 1024 x dim` — over 1M entries. At 112-dim, that's ~117M parameters, more than the entire model.

Hashing compresses 1M pairs into 3072 buckets:
- **3072 x 112 = 344K params** (vs 117M for full table)
- Collisions are tolerable — the embedding learns the average pattern across colliding pairs
- The hash is **order-sensitive**: `hash(A, B) != hash(B, A)` because the primes differ

---

## Trigram Extension (implemented but not used in SOTA)

The codebase also includes a trigram hash that captures 3-token context:

```python
def trigram_hash(self, tokens):
    t = tokens.to(torch.int32)
    mod = self.bigram_vocab_size - 1
    out = torch.empty_like(t)
    out[..., :2] = mod                                              # first 2 positions: no trigram
    out[..., 2:] = (36313 * t[..., 2:]                             # current
                   ^ 27191 * t[..., 1:-1]                           # previous
                   ^ 51497 * t[..., :-2]) % mod                     # two back
    return out.long()
```

When enabled, bigram and trigram embeddings are **added together**, reusing the same embedding table (zero extra parameters). The current SOTA does not enable trigrams (`TRIGRAM=0`).

---

## Evolution Across Submissions

| Submission | Buckets | Dim | Params | BPB Impact |
|------------|---------|-----|--------|------------|
| PR #162 (concept) | initial | - | - | Introduced the idea |
| PR #549 (prev SOTA) | 1536 | 128 | ~230K | Baseline |
| PR #609 | 2048 | 128 | ~295K | Wider table |
| PR #1019 (current SOTA) | 3072 | 112 | ~380K | Wider table, narrower dim |

The trend: **more buckets** (fewer hash collisions) matters more than **wider embeddings** (diminishing returns per dimension). The narrowing from 128 to 112 freed parameter budget for the wider table while staying under 16MB.

---

## Key Design Decisions

1. **Zero initialization**: Both the embedding table and projection are initialized to zeros. The model starts as if BigramHash doesn't exist and gradually learns to use it. The scale parameter starts at 0.05, meaning even after learning, bigram information is a small additive signal.

2. **Additive injection**: BigramHash is *added* to the token embedding, not concatenated. This means it doesn't increase the model dimension, and the model can learn to ignore it for positions where bigram context isn't helpful.

3. **Position 0 special case**: The first token has no previous token, so it gets a fixed bucket index (3071). This is effectively a learned "start-of-sequence" bigram embedding.

4. **No learned hash**: The hash function is fixed (not learned). This is deliberate — a learned hash would require backprop through a discrete operation. The fixed hash with a learned embedding table is a clean separation of concerns.