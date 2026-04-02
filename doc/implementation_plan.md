# Implementation Plan: Shared MLP Bases + Low-Rank Adapters

## 1. What We Are Building

### Current SOTA Architecture (11 layers, 1.1147 BPB)

```
Input → TokenEmbed + BigramHash → RMSNorm → SmearGate → x₀

Encoder (layers 0-4):
  For each layer i:
    up_w  = mlp_up_bank[i]       ← independent full-rank [1536, 512]
    down_w = mlp_down_bank[i]    ← independent full-rank [512, 1536]
    x = Block(x, x₀, attn_weights, up_w, down_w)
    save x to skip stack

Decoder (layers 5-10):
  For each layer i:
    x += skip_weight * skip_stack.pop()
    up_w  = mlp_up_bank[i]       ← independent full-rank
    down_w = mlp_down_bank[i]    ← independent full-rank
    x = Block(x, x₀, attn_weights, up_w, down_w)

Output → RMSNorm → Tied LM Head → Logit Softcap → Loss
```

11 layers. Each layer has its OWN full-rank MLP weights. MLP eats 63% of the
16MB artifact. Only room for 11 layers total.

### Our Proposed Architecture (16 layers, target < 1.112 BPB)

```
Input → TokenEmbed + BigramHash → RMSNorm → SmearGate → x₀

Encoder (layers 0-7):
  For each layer i:
    g = layer_group(i)           ← 0, 1, or 2
    up_w  = mlp_shared_up[g]  + adapter_up_A[i]  @ adapter_up_B[i]
    down_w = mlp_shared_down[g] + adapter_down_A[i] @ adapter_down_B[i]
    x = Block(x, x₀, attn_weights, up_w, down_w)
    save x to skip stack

Decoder (layers 8-15):
  For each layer i:
    x += skip_weight * skip_stack.pop()
    g = layer_group(i)
    up_w  = mlp_shared_up[g]  + adapter_up_A[i]  @ adapter_up_B[i]
    down_w = mlp_shared_down[g] + adapter_down_A[i] @ adapter_down_B[i]
    x = Block(x, x₀, attn_weights, up_w, down_w)

Output → RMSNorm → Tied LM Head → Logit Softcap → Loss
```

16 layers. 3 shared MLP bases + rank-100 adapters. Each layer has UNIQUE
effective weights. Block and MLP classes are UNCHANGED — they still receive
2D weight tensors. The only difference is HOW those tensors are computed.

---

## 2. U-Net Structure: SOTA vs Ours

### SOTA (11 layers)

```
Encoder: 5 layers (0-4)     Decoder: 6 layers (5-10)     Skips: 5

Layer 0 ──── h₀ ──────────────────────── +w₄⊙h₀ → Layer 9
Layer 1 ──── h₁ ────────────────── +w₃⊙h₁ ───→ Layer 8
Layer 2 ──── h₂ ──────────── +w₂⊙h₂ ─────→ Layer 7
Layer 3 ──── h₃ ────── +w₁⊙h₃ ───────→ Layer 6
Layer 4 ──── h₄ ── +w₀⊙h₄ ─────────→ Layer 5
                                        Layer 10 (no skip)
```

### Ours (16 layers)

```
Encoder: 8 layers (0-7)     Decoder: 8 layers (8-15)     Skips: 8

Layer 0  ──── h₀ ──────────────────────────────── +w₇⊙h₀ → Layer 15
Layer 1  ──── h₁ ────────────────────────── +w₆⊙h₁ ───→ Layer 14
Layer 2  ──── h₂ ──────────────────── +w₅⊙h₂ ─────→ Layer 13
Layer 3  ──── h₃ ────────────── +w₄⊙h₃ ───────→ Layer 12
Layer 4  ──── h₄ ──────── +w₃⊙h₄ ─────────→ Layer 11
Layer 5  ──── h₅ ────── +w₂⊙h₅ ───────────→ Layer 10
Layer 6  ──── h₆ ── +w₁⊙h₆ ─────────────→ Layer 9
Layer 7  ──── h₇ +w₀⊙h₇ ───────────────→ Layer 8
```

More skip connections (8 vs 5). Deeper U, richer cross-depth feature transfer.
The code auto-adjusts: `num_encoder = num_layers // 2`, no manual changes needed.

### MLP Group Assignment

```
Group 0 (shallow):  layers 0-4   → share mlp_shared_up[0], mlp_shared_down[0]
Group 1 (middle):   layers 5-10  → share mlp_shared_up[1], mlp_shared_down[1]
Group 2 (deep):     layers 11-15 → share mlp_shared_up[2], mlp_shared_down[2]
```

Each layer ALSO has its own rank-100 adapter: `A[i] @ B[i]` added to the shared base.
So the effective weight is full-rank (rank 512), with per-layer specialization.

---

## 3. What Changes in Each Component

### 3.1 Architecture Changes

| Component | SOTA (11L) | Ours (16L) | Changed? |
|-----------|-----------|------------|----------|
| Layers | 11 | 16 | YES |
| Encoder/Decoder | 5/6 | 8/8 | Auto (code unchanged) |
| Skip connections | 5 | 8 | Auto (code unchanged) |
| MLP weights | Per-layer bank [11, 1536, 512] | 3 shared bases + 16 adapters | YES |
| Attention weights | Per-layer bank [22, 512, 512] | Per-layer bank [32, 512, 512] | Size only (code unchanged) |
| Block class | Receives up_w, down_w | Same | NO |
| MLP class | F.linear(x, up_w) | Same | NO |
| Attention class | Full implementation | Same | NO |
| XSA | All 11 layers | All 16 layers | Config only |
| VE layers | 9, 10 | 14, 15 | Config only |
| BigramHash | 3072 x 112 | Same | NO |
| SmearGate | dim=512 | Same | NO |
| RoPE | Partial 16/64 | Same | NO |
| Tied embeddings | Yes | Same | NO |
| Logit softcap | 30.0 | Same | NO |

### 3.2 Optimizer Changes

| Component | SOTA | Ours | Changed? |
|-----------|------|------|----------|
| Muon optimizer class | Unchanged | Unchanged | NO |
| Newton-Schulz | 5-step batched | Same | NO |
| Muon bank params | 4 banks (qo, kv, mlp_up, mlp_down) | 8 banks (qo, kv, shared_up, shared_down, 4 adapter banks) | YES (param list only) |
| Adam for embeddings | Same | Same | NO |
| Adam for scalars | Same | Same | NO |
| Learning rates | matrix_lr=0.025 | Same | NO |
| Weight decay | 0.04 | Same | NO |
| Muon momentum | 0.99 (warmup from 0.92) | Same | NO |
| Gradient clipping | 0.3 | Same | NO |

The only optimizer change is which parameters go into the Muon param list.
Muon handles any 3D bank shape — verified that B=3 (shared bases, padded to 8
for 8 GPUs) works correctly with the existing reduce-scatter/all-gather pipeline.

### 3.3 Quantization Changes

| Step | SOTA | Ours | Changed? |
|------|------|------|----------|
| Late QAT during training | STE on CastedLinear (not on bank weights) | Same | NO |
| EMA + SWA | Auto-tracks all params | Auto-tracks new params | NO (code unchanged) |
| Unbanking for Hessian | mlp_bank → per-layer 2D weights | shared+adapter → materialized effective weights | YES (new path) |
| _HessianGPT model | Standard CastedLinear per layer | Same (loads materialized effective weights) | NO |
| AR self-gen calibration | 64 seqs x 2048 tokens | Same | NO |
| Hessian collection | Per-layer H = X^T X | Same, plus aggregation for shared bases | YES (add aggregation) |
| Quantization format | Per-layer int6 GPTQ | Shared bases: int6 GPTQ, Adapters: int6/fp16 by size | YES (new format) |
| Selective pruning | Prune ±1 by reconstruction error | Same (operates on all int6 tensors) | NO |
| LZMA compression | preset=9 | Same | NO |
| Eval dequantization | Dequant → rebank → standard GPT | Dequant → compute effective weights → rebank → standard GPT | YES (add effective weight computation) |
| Eval model | Standard banked GPT | Same (adapter_rank=0) | NO |

### 3.4 Compression / Artifact Changes

| Component | SOTA (bytes) | Ours (bytes) | Changed? |
|-----------|-------------|-------------|----------|
| Format | int6 GPTQ + LZMA | Same | NO |
| LZMA preset | 9 | 9 | NO |
| TARGET_MB | 15.9 | 15.7 (more margin) | Config only |
| **What's stored in artifact** | Per-layer MLP weights (int6) | Shared bases (int6 GPTQ) + Adapters (int6/fp16) | YES |

The artifact stores FEWER total bytes because shared bases (3 matrices) + adapters
(compact) < 16 full-rank MLP matrices. This is the core space saving.

### 3.5 Training Time Impact

```
SOTA:  11 layers × 86.7ms/step ≈ 6,920 steps in 600s
Ours:  16 layers × ~125ms/step ≈ 4,800 steps in 600s
                                  ─────
                                  ~31% fewer training steps
```

Adapter overhead per step: 32 small matmuls ([1536,100]@[100,512]) ≈ 2ms total.
Main cost is 5 extra layers of attention + MLP forward/backward.

### 3.6 Eval Time Impact

Eval uses the standard banked GPT with materialized effective weights.
16 layers vs 11 → ~45% more eval time.

```
SOTA:  standard eval ~120s + no TTT = ~120s total
Ours:  standard eval ~175s + no TTT = ~175s total
Budget: 600s for eval → well within limit
```

---

## 4. Quantization Detail: What Gets Stored How

### SOTA artifact contents (per-layer MLP)

```
blocks.0.mlp.fc.weight     [1536, 512]   int6 GPTQ   ← full-rank per layer
blocks.0.mlp.proj.weight   [512, 1536]   int6 GPTQ
blocks.1.mlp.fc.weight     [1536, 512]   int6 GPTQ
...                         (11 layers × 2 = 22 MLP matrices)
blocks.0.attn.c_q.weight   [512, 512]    int6 GPTQ
...                         (11 layers × 4 = 44 attention matrices)
tok_emb.weight             [1024, 512]   int8
bigram.embed.weight        [3072, 112]   int8
(small params)                           fp16/fp32
```

### Our artifact contents (shared + adapters)

```
shared_mlp.0.up            [1536, 512]   int6 GPTQ (aggregated Hessian)
shared_mlp.0.down          [512, 1536]   int6 GPTQ
shared_mlp.1.up            [1536, 512]   int6 GPTQ
shared_mlp.1.down          [512, 1536]   int6 GPTQ
shared_mlp.2.up            [1536, 512]   int6 GPTQ
shared_mlp.2.down          [512, 1536]   int6 GPTQ
                           ↑ 6 shared matrices (was 22 per-layer matrices)

blocks.0.adapter.up_A      [1536, 100]   int6 per-row (> 65K elements)
blocks.0.adapter.up_B      [100, 512]    fp16 pass   (< 65K elements)
blocks.0.adapter.down_A    [512, 100]    fp16 pass   (< 65K elements)
blocks.0.adapter.down_B    [100, 1536]   int6 per-row (> 65K elements)
...                        (16 layers × 4 = 64 adapter matrices)

blocks.0.attn.c_q.weight   [512, 512]   int6 GPTQ
...                        (16 layers × 4 = 64 attention matrices)

tok_emb.weight             [1024, 512]   int8
bigram.embed.weight        [3072, 112]   int8
(small params)                           fp16/fp32
```

### Why this saves space

```
SOTA MLP:  22 matrices × [1536,512] or [512,1536] ≈ 16.5 MB before LZMA
Ours MLP:  6 shared + 64 adapters (32 large int6 + 32 small fp16) ≈ 12.1 MB before LZMA
Savings:   ~4.4 MB → spent on 5 extra layers of attention
```

---

## 5. How Evaluation Works (unchanged eval code)

```
1. Decompress LZMA → torch.load → quant_state dict
2. Dequantize:
   - shared_mlp.0.up → float32
   - blocks.0.adapter.up_A → float32
   - blocks.0.adapter.up_B → float32  (from fp16)
   - etc.
3. Compute effective weights:
   For each layer i:
     g = layer_to_group[i]
     mlp_up_eff[i] = deQ(shared_mlp.{g}.up) + deQ(adapter.up_A[i]) @ deQ(adapter.up_B[i])
     mlp_down_eff[i] = deQ(shared_mlp.{g}.down) + deQ(adapter.down_A[i]) @ deQ(adapter.down_B[i])
4. Stack into standard banks:
   mlp_up_bank = stack([mlp_up_eff[0], ..., mlp_up_eff[15]])
   mlp_down_bank = stack([mlp_down_eff[0], ..., mlp_down_eff[15]])
5. Load into eval GPT model (standard bank architecture, adapter_rank=0)
6. Run eval_val / eval_val_sliding — ZERO changes to eval code
```

---

## 6. Why This Is NOT Depth Recurrence (PR #363)

| | Depth Recurrence (failed) | Our Approach |
|---|---|---|
| **Effective weights** | Identical across loops | Unique per layer: `W_shared + A[i]@B[i]` |
| **At inference** | Same weight matrix reused N times | Different materialized weight per layer |
| **Quantization** | Error compounds: `Q(W) @ Q(W) @ Q(W) @ x` = 900x amplification | Each layer quantized independently, no compounding |
| **In the artifact** | One copy of weights, reused | Shared bases + per-layer adapters stored separately |
| **PR #363 result** | +0.014 BPB worse than flat 11L | Not yet tested |

---

## 7. Validation Checks (Competition Requirements)

### 7.1 Artifact Size

- **Requirement**: Total artifact (compressed model + code) < 16,000,000 bytes
- **Our estimate**: ~15.77 MB (226 KB headroom)
- **Check**: After quantization, log `total_submission_size` and assert < 16,000,000
- **Safety margin**: TARGET_MB=15.7 (300 KB below budget)

### 7.2 Training Time

- **Requirement**: Training must complete in < 600 seconds on 8xH100 SXM
- **Our estimate**: ~125ms/step × ~4800 steps = 600s (wallclock-capped)
- **Check**: `MAX_WALLCLOCK_SECONDS=600` enforced by existing code (line 44)
- **Risk**: If step time > expected, fewer steps. Model still trains — just fewer steps.

### 7.3 Evaluation Time

- **Requirement**: Evaluation must complete in < 600 seconds on 8xH100 SXM
- **Our estimate**: Standard eval ~175s + sliding window ~175s = ~350s total
- **Check**: Log eval time (already done in existing code)
- **No TTT**: We're not using test-time training, so eval is simpler

### 7.4 Statistical Significance

- **Requirement**: Beat SOTA by ≥ 0.005 nats at p < 0.01 (Welch's t-test)
- **SOTA scores**: seeds [1.11508, 1.11437, 1.11475] → mean 1.11473 BPB
- **Target**: mean BPB < ~1.112 (0.003 BPB improvement ≈ 0.005 nats)
- **Check**: Run 3 seeds, compute Welch's t-test

### 7.5 Submission Files

- **Requirement**: PR adds ONLY a new folder under `records/track_10min_16mb/`
- **Our folder**: `records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/`
- **Contents** (ONLY these files):
  ```
  README.md              # architecture explanation
  submission.json        # metadata
  train_gpt.py           # self-contained, must run from this folder
  train_seed314.log
  train_seed42.log
  train_seed999.log
  ```
- **DO NOT modify**: Any existing file in the repo. No changes to base train_gpt.py,
  README.md, data/, or any other records/ folder.
- **Check**: `git diff` before submitting should show only additions in our new folder

### 7.6 Code Self-Containment

- **Requirement**: `train_gpt.py` must compile and run from the records folder
- **Check**: The script imports only standard libraries + torch + flash_attn + sentencepiece + zstandard
- **No new dependencies**: Our changes use only existing torch operations (nn.Parameter, F.linear, @)

### 7.7 Reproducibility

- **Requirement**: Results must be reproducible with the provided seed
- **Check**: Fixed seed controls all randomness. Run same seed twice, verify identical results.
- **Existing code**: Already sets `torch.manual_seed(seed)`, `random.seed(seed)`, `np.random.seed(seed)`

### 7.8 No External Data During Evaluation

- **Requirement**: Cannot access training data during evaluation
- **Check**: Our eval uses standard eval_val / eval_val_sliding on validation tokens only
- **No TTT**: We dropped test-time training, so no data access concerns

---

## 8. Implementation Order

1. Create new records folder
2. Copy SOTA train_gpt.py to new folder
3. Implement changes 1-10 (architecture + optimizer + training)
4. Smoke test with ADAPTER_RANK=0 (should match SOTA behavior)
5. Test with ADAPTER_RANK=100 on 1 GPU, short run
6. Implement changes 11-17 (quantization pipeline)
7. Test full quantization round-trip on 1 GPU
8. Verify artifact size < 16 MB
9. Test on 8 GPUs, single seed
10. Run 3 seeds, verify statistical significance
11. Prepare README.md and submission.json
12. Verify git diff shows only new folder additions