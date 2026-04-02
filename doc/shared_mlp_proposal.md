# Proposal: Shared MLP Bases + Low-Rank Per-Layer Adapters

**Target**: Beat current SOTA (1.1147 BPB) on the Parameter Golf leaderboard
**Config**: 16 layers, 3 shared MLP bases, rank-100 per-layer adapters
**Estimated artifact**: 15.77 MB (226 KB headroom within 16 MB budget)

---

## Motivation

### The bottleneck: MLP eats 63% of the artifact

The current SOTA spends its 16 MB budget as:

```
MLP weights:      ~10.0 MB   (63%)
Attention weights: ~4.6 MB   (31%)
Everything else:   ~1.2 MB   (6%)
```

The MLP is expensive because of the 3x expansion: each layer has two matrices
`[1536, 512]` and `[512, 1536]` = 1.57M params per layer, times 11 layers = 17.3M params.
This leaves room for only 11 layers total.

### The hypothesis: depth matters more than per-layer MLP expressiveness

Leaderboard history shows consistent gains from adding layers:

```
9 layers (baseline):  1.2244 BPB
10 layers:            1.1748 BPB   (~0.05 BPB gain)
11 layers:            1.1307 BPB   (~0.04 BPB gain)
```

Each layer adds ~0.04-0.05 BPB. If we could add 5 more layers, even at
reduced per-layer capacity, the potential gain is significant.

### The bet

We trade **per-layer MLP independence** for **more layers** by sharing MLP
base weights across groups of layers, with small per-layer adapters for
specialization.

```
Current SOTA:  11 layers × full-rank MLP   = 17.30M MLP params, 8.65M attn params
Proposed:      16 layers × shared+adapter   = 11.01M MLP params, 12.58M attn params
```

We spend LESS on MLP but MORE on attention (5 extra full-rank attention layers).
Total artifact is smaller (15.77 MB vs ~15.86 MB) despite having 5 more layers.

---

## Architecture Detail

### Layer structure: 16 layers, U-Net

```
Encoder (layers 0-7):   8 layers, each saves output for skip connections
Decoder (layers 8-15):  8 layers, receives skips in reverse order

Layer 0  ─── h₀ ────────────────────────────────── +w₇⊙h₀ → Layer 15
Layer 1  ─── h₁ ──────────────────────────── +w₆⊙h₁ ───→ Layer 14
Layer 2  ─── h₂ ────────────────────── +w₅⊙h₂ ─────→ Layer 13
Layer 3  ─── h₃ ──────────────── +w₄⊙h₃ ───────→ Layer 12
Layer 4  ─── h₄ ────────── +w₃⊙h₄ ─────────→ Layer 11
Layer 5  ─── h₅ ────── +w₂⊙h₅ ───────────→ Layer 10
Layer 6  ─── h₆ ── +w₁⊙h₆ ─────────────→ Layer 9
Layer 7  ─── h₇ +w₀⊙h₇ ───────────────→ Layer 8
```

8 skip connections (up from 5 in the current SOTA).

### MLP: 3 shared bases + rank-100 adapters

Three independent full-rank MLP bases, assigned by depth:

```
Base 0 (shallow):  layers 0-4      (5 layers)
Base 1 (middle):   layers 5-10     (6 layers)
Base 2 (deep):     layers 11-15    (5 layers)
```

Each layer computes its effective MLP weight as:

```
Up_eff[i]   = Up_shared[group(i)]   + A_up[i]   @ B_up[i]
Down_eff[i] = Down_shared[group(i)] + A_down[i] @ B_down[i]
```

Where:
- `Up_shared[g]`:   [1536, 512]  — full-rank shared base (3 of these)
- `Down_shared[g]`: [512, 1536]  — full-rank shared base (3 of these)
- `A_up[i]`:  [1536, 100]  — per-layer adapter left factor
- `B_up[i]`:  [100, 512]   — per-layer adapter right factor
- `A_down[i]`: [512, 100]  — per-layer adapter left factor
- `B_down[i]`: [100, 1536] — per-layer adapter right factor

### Why 3 bases (not 1 or 2)

- **1 base**: Maximum layer budget (up to 23L with r=32), but all layers
  share ONE MLP function. Shallow syntax layers and deep semantic layers
  forced to share — too constraining.

- **2 bases**: Good compromise (up to 20L with r=64), but the middle layers
  must belong to either "shallow" or "deep" — awkward split.

- **3 bases**: Natural three-way split (shallow/middle/deep). Costs 4.72M
  shared params but gives each depth range its own full-rank MLP. Fits 16
  layers at rank 100. This is our choice.

- **4 bases**: Only 17L at r=64. The extra base costs 1.57M params for
  one fewer layer. Not worth it.

### Why rank 100

- **LoRA literature**: Rank 4-16 is sufficient for fine-tuning billion-param
  models. Rank 100 is generous for a 512-dim model.
- **Budget**: Rank 100 at 16 layers leaves 226 KB headroom. Rank 104 leaves
  only 41 KB (too tight given ~120 KB seed-to-seed LZMA variance).
- **Expressiveness**: Rank 100 means each layer can modify 100 independent
  directions in the MLP weight space. The full weight matrix is rank 512,
  so the adapter can alter ~20% of the spectral energy per layer.

### Effective rank per layer

Each layer's effective MLP weight is:

```
W_eff = W_shared + A @ B
```

`W_shared` is full-rank (rank 512). Adding a rank-100 perturbation
doesn't change the rank — `W_eff` is still full-rank. Every neuron in
the MLP hidden layer can fire independently. This is fundamentally
different from a pure low-rank factored MLP (which would be rank-100).

---

## Why This Is NOT Depth Recurrence

PR #363 extensively tested depth recurrence and found it fails. Our
approach is structurally different.

### Depth recurrence (PR #363, failed)

```
for repeat in range(3):
    x = SharedBlock(x)     ← EXACT same weights, 3 times
```

Problem: After int6 quantization, the quantization error passes through
the same computation 3 times:

```
y = Q(W) @ (Q(W) @ (Q(W) @ x))
```

Each pass amplifies the error. PR #363 measured **900x error amplification**
through 3 cycles. Even with Noisy QAT (which reduced the gap 185x), the
looped model lost to a flat 11L model by +0.014 BPB.

### Our approach (shared base + per-layer adapter)

```
x = Block_0(x, W_shared[0] + A[0]@B[0])    ← unique effective weight
x = Block_1(x, W_shared[0] + A[1]@B[1])    ← different effective weight
x = Block_2(x, W_shared[0] + A[2]@B[2])    ← different effective weight
...
```

Each layer has **unique effective weights**. After quantization:

```
y = Q(W_shared[0] + A[0]@B[0]) @ x₀      ← independent quant error
y = Q(W_shared[0] + A[1]@B[1]) @ x₁      ← independent quant error
y = Q(W_shared[0] + A[2]@B[2]) @ x₂      ← independent quant error
```

The quantization errors are independent per layer — they don't compound.
The shared base has a single quantization error that affects all layers in
its group, but this is an additive constant error, not a multiplicative
amplification.

### Key structural differences

| | Depth recurrence | Our approach |
|---|---|---|
| Effective weights | Identical across loops | Unique per layer |
| Quantization | Errors compound through loops | Errors independent per layer |
| Gradient flow | Same weight updated by all loops | Shared base + independent adapters |
| At inference | Same weight matrix reused | Different materialized weight per layer |
| PR #363 result | Failed (900x error amplification) | Not yet tested |

---

## Forward Pass

### MLP forward

```python
def forward(self, x, layer_idx):
    group = self.layer_to_group[layer_idx]

    # Materialize effective weights (one GEMM to compute adapter, then add)
    up_eff = self.up_shared[group] + self.A_up[layer_idx] @ self.B_up[layer_idx]
    down_eff = self.down_shared[group] + self.A_down[layer_idx] @ self.B_down[layer_idx]

    # Standard MLP forward with LeakyReLU(0.5)²
    h = F.leaky_relu(F.linear(x, up_eff), negative_slope=0.5).square()
    return F.linear(h, down_eff)
```

Alternative (avoid materializing full weight):

```python
def forward(self, x, layer_idx):
    group = self.layer_to_group[layer_idx]

    # Shared path + adapter path (two matmuls, no materialization)
    h = F.linear(x, self.up_shared[group]) + F.linear(F.linear(x, self.B_up[layer_idx]), self.A_up[layer_idx])
    h = F.leaky_relu(h, negative_slope=0.5).square()
    return F.linear(h, self.down_shared[group]) + F.linear(F.linear(h, self.B_down[layer_idx]), self.A_down[layer_idx])
```

The materialized approach (first version) is preferred for GPU efficiency:
one large GEMM is faster than two small ones. The extra memory for the
materialized weight ([1536, 512] = 3 MB in bf16) is acceptable.

---

## Optimizer Strategy

### Shared MLP bases: Adam

- Lower learning rate (0.01 vs 0.025 for other matrices)
- Adam is more stable for parameters that receive aggregated gradients
  from multiple layers
- The shared base should change slowly — it represents the "common MLP
  function" across a group of layers
- Weight decay: 0.04 (same as current)

### Adapter banks: Muon

- Same learning rate as current matrix params (0.025)
- Stored as 4 parameter banks for batched Newton-Schulz:
  ```
  adapter_up_A_bank:    [16, 1536, 100]
  adapter_up_B_bank:    [16, 100, 512]
  adapter_down_A_bank:  [16, 512, 100]
  adapter_down_B_bank:  [16, 100, 1536]
  ```
- Newton-Schulz handles non-square matrices (transposes if rows > cols)
- Weight decay: 0.04

### Attention banks: Muon (unchanged)

- QO bank: [32, 512, 512] (was [22, ...] — now 16 layers)
- KV bank: [32, 256, 512] (was [22, ...])
- Same learning rate and momentum as current

### Concern: Adapter bank shapes are very non-square

The adapter matrices like [1536, 100] have aspect ratio 15:1. Newton-Schulz
orthogonalization finds the nearest orthogonal matrix, which for a [100, 1536]
matrix (after transpose) means 100 orthogonal rows in 1536-dimensional space.
This is well-conditioned and should work fine.

If Muon struggles with these shapes, fallback is Adam for adapter banks too.

---

## Initialization

```python
# Shared bases: standard init (same as current MLP banks)
for s in range(3):
    nn.init.orthogonal_(up_shared[s], gain=1.0)
    nn.init.zeros_(down_shared[s])  # zero-init output projection
    down_shared[s].mul_(1.0 / math.sqrt(2 * n_layers))  # projection scale

# Adapters: zero init
# Model starts as if adapters don't exist → pure shared MLP
# Then gradually learns per-layer specialization
for i in range(16):
    nn.init.zeros_(A_up[i])
    nn.init.zeros_(B_up[i])
    nn.init.zeros_(A_down[i])
    nn.init.zeros_(B_down[i])
```

Zero-initializing adapters is deliberate:
- At step 0, the model behaves like a 3-MLP depth-recurrent model
- The shared bases carry all the initial learning
- As training progresses, adapters activate and specialize each layer
- This gives the shared bases a head start before adapters introduce noise

---

## Quantization Plan

### Storage format (in artifact)

```
Shared bases (3):   int6 GPTQ per-row   (large matrices, benefit from Hessian)
Adapter A matrices: int6 per-row         (> 65536 elements for [1536,100])
Adapter B matrices: fp16 passthrough     (< 65536 elements for [100,512])
Attention banks:    int6 GPTQ per-row    (same as current)
Embeddings:         int8 / fp16          (same as current)
```

### Eval-time dequantization

```
1. Dequantize shared bases:  Q(Up_shared[g]) → Up_shared_f32[g]
2. Dequantize adapter A:     Q(A_up[i]) → A_up_f32[i]
3. Load adapter B (fp16):    B_up[i] → B_up_f32[i]
4. Compute effective weight: W_eff = Up_shared_f32[g] + A_up_f32[i] @ B_up_f32[i]
5. Run inference with W_eff
```

### Quantization error analysis

Two sources of error per layer:
```
Error = Q(W_shared) + Q(A)@Q(B) - (W_shared + A@B)
      = [Q(W_shared) - W_shared] + [Q(A)@Q(B) - A@B]
        ─────────────────────────   ───────────────────
        shared base quant error     adapter quant error
        (same for all layers in     (independent per layer)
         the group)
```

The shared base error is a constant offset for all layers in a group.
This is NOT amplification — it's the same error added once, not
multiplied through loops. The adapter errors are independent per layer.

Total error per layer ≈ base_error + adapter_error. Neither compounds.

### Noisy QAT for shared bases (if needed)

If quantization gap is too large (> 0.005 BPB), apply Noisy QAT from
PR #363 to the shared bases during training:

```python
# During forward pass, inject quantization-calibrated noise to shared bases
with torch.no_grad():
    amax = shared_weight.float().abs().amax(dim=1, keepdim=True)
    step_size = amax / 31.0  # int6 step size
noise = (torch.rand_like(w) - 0.5) * step_size
w_noisy = w + noise  # differentiable, trains robustness to quantization
```

This trains the model to be robust to int6 rounding of shared bases.
Only apply to shared bases (not adapters, which are already per-layer).

---

## Training Estimates

### Step time

```
Current SOTA:  11 layers × 86.7 ms/step
Proposed:      16 layers × ~126 ms/step  (16/11 × 86.7, rough estimate)
                          + ~5 ms adapter overhead (materializing A@B per layer)
                          ≈ ~131 ms/step
```

### Training steps in 600s

```
Current SOTA:  600,000 ms / 86.7 ms ≈ 6,920 steps
Proposed:      600,000 ms / 131 ms  ≈ 4,580 steps
                                      ─────
                                      ~34% fewer training steps
```

### Warmdown adjustment

Current SOTA uses warmdown=4000 out of ~6,920 effective steps (58%).
Proportionally: 4,580 × 0.58 ≈ 2,650 warmdown iters.

Round to 2700 for safety.

### Hyperparameters (proposed starting point)

| Parameter | Current SOTA | Proposed | Rationale |
|-----------|:---:|:---:|---|
| NUM_LAYERS | 11 | 16 | +5 layers |
| Shared MLP bases | — | 3 | Shallow/middle/deep |
| Adapter rank | — | 100 | Budget-optimized |
| WARMDOWN_ITERS | 4000 | 2700 | Proportional to fewer steps |
| WARMUP_STEPS | 20 | 20 | Keep same |
| MATRIX_LR (Muon, adapters) | 0.025 | 0.025 | Same |
| SHARED_BASE_LR (Adam) | — | 0.01 | Lower for stability |
| MUON_MOMENTUM | 0.99 | 0.99 | Keep same |
| MUON_WD | 0.04 | 0.04 | Keep same |
| XSA_LAST_N | 11 (all) | 16 (all) | Keep XSA on all layers |
| VE_LAYERS | 9,10 | 14,15 | Last 2 layers |
| EMA_DECAY | 0.997 | 0.997 | Keep same |
| SWA_EVERY | 50 | 50 | Keep same |
| BIGRAM_VOCAB_SIZE | 3072 | 3072 | Keep same |
| BIGRAM_DIM | 112 | 112 | Keep same |

---

## Testing Plan

### Phase 1: Architecture validation (1x H100, ~30 min, ~$1.50)

1. Implement architecture changes in train_gpt.py
2. Run short training: MAX_WALLCLOCK_SECONDS=180, 1 GPU
3. Verify:
   - [ ] Training converges (loss drops comparable to baseline)
   - [ ] Step time is acceptable (expect ~500ms on 1 GPU)
   - [ ] No OOM (expect ~25 GB peak memory)
   - [ ] Adapters activate (monitor adapter weight norms over training)

### Phase 2: Quantization validation (1x H100, ~20 min, ~$1.00)

1. Run full 10-minute training on 1 GPU
2. Quantize shared bases + adapters separately
3. Verify:
   - [ ] Artifact size < 16 MB
   - [ ] Quantization gap < 0.005 BPB
   - [ ] Dequantization + effective weight computation works correctly

If quantization gap too large:
   - [ ] Enable Noisy QAT for shared bases
   - [ ] Re-run and verify gap improves

### Phase 3: Multi-GPU validation (8x H100, ~20 min, ~$7.00)

1. Run single seed on 8x H100
2. Verify:
   - [ ] DDP / Parallel Muon works with new bank structure
   - [ ] Step time ~131ms (not significantly worse)
   - [ ] val_bpb competitive with SOTA pre-quantization

### Phase 4: Full evaluation (8x H100, ~60 min, ~$20.00)

1. Run 3 seeds: SEED=314, 42, 999
2. Record pre-quant BPB, post-quant BPB, sliding window BPB
3. Compute Welch's t-test vs SOTA's 3 seeds
4. Acceptance criteria:
   - [ ] Mean BPB < 1.1147 (beats SOTA)
   - [ ] Improvement ≥ 0.005 nats (~0.003 BPB)
   - [ ] p < 0.01 on Welch's t-test
   - [ ] All 3 seeds: artifact < 16 MB
   - [ ] All 3 seeds: training < 600s
   - [ ] All 3 seeds: eval < 600s

---

## Submission Preparation

### Required files

```
records/track_10min_16mb/YYYY-MM-DD_SharedMLP_Rank100_16L/
├── README.md            # architecture, motivation, ablations, comparison to PR #363
├── submission.json      # {"author": "...", "val_bpb": ..., ...}
├── train_gpt.py         # self-contained, must run from this folder
├── train_seed314.log
├── train_seed42.log
└── train_seed999.log
```

### README must address

1. **Clear explanation** of shared MLP + adapter architecture
2. **Why this is NOT depth recurrence** (unique effective weights, no error compounding)
3. **Ablation table**: shared+adapter vs full-rank at same layer count
4. **Statistical significance**: Welch's t-test results
5. **Lineage**: Built on PR #1019 (current SOTA) stack

### Risks to address in README

- Acknowledge PR #363 (depth recurrence failure) and explain structural differences
- Show quantization gap is controlled (base + adapter errors don't compound)
- Show adapter utilization (they actually learn different things per layer)

---

## Risk Assessment

### Risk 1: 34% fewer training steps (HIGH likelihood, HIGH impact)

16 layers at ~131ms/step means ~4,580 steps vs ~6,920 current. This is
the single biggest risk. The depth gain must compensate.

**Mitigation**: Can't avoid this. Must verify empirically that depth wins.
Historical evidence (9L→11L gave ~0.04 BPB per layer despite fewer steps)
suggests it's possible. If step time is worse than expected, could try:
- Reducing adapter rank to speed up materialization
- Reducing to 15 layers
- Using approach B (two matmuls) if materialization is slow

### Risk 2: Quantization gap (MEDIUM likelihood, HIGH impact)

Quantizing shared bases and adapters separately may introduce more error
than quantizing full effective weights.

**Mitigation**: Noisy QAT for shared bases. Late QAT for all weights.
If gap is still too large, quantize materialized effective weights instead
(but this uses the same artifact size as 16 independent MLPs — defeats
the purpose).

### Risk 3: Not meeting 0.005 nats threshold (UNKNOWN likelihood)

Must beat 1.1147 BPB by ~0.003 BPB → target ~1.112 BPB. This is a
significant improvement that requires the depth to meaningfully help.

**Mitigation**: If close but not enough, can try:
- Enabling trigram hash (already implemented, zero extra params)
- Adjusting adapter rank up or down
- Tuning learning rates for shared vs adapter
- Adding TTT back (if it works on this stack)

### Risk 4: Optimizer instability (LOW likelihood, MEDIUM impact)

Shared bases receive aggregated gradients from 5-6 layers. Could cause
unstable updates.

**Mitigation**: Adam with low LR (0.01) for shared bases. If unstable,
reduce further or freeze shared bases after initial training phase.

---

## Budget Verification

### Parameter count

```
Attention (Q/K/V/Out):    16 layers × 786,432    = 12,583,M
MLP shared bases:          3 bases  × 1,572,864  =  4,719M
MLP adapters:             16 layers × 393,600    =  6,298M
Token embedding:          1024 × 512              =    524K
BigramHash:               3072 × 112 + 512 × 112 =    401K
Value Embedding:          1024 × 128 + 256 × 128  =    164K
Small/control params:     ~40K
                                                   ─────────
TOTAL:                                              ~24.72M
```

### Artifact size (estimated)

```
Before LZMA:     ~25.57 MB
After LZMA:      ~14.75 MB  (÷1.733)
Code:            ~0.11 MB
                 ──────────
TOTAL ARTIFACT:  ~15.77 MB
BUDGET:           16.00 MB
HEADROOM:          0.23 MB (226 KB)   ✓
```

### Comparison with current SOTA

```
                    SOTA (11L)     Proposed (16L)     Delta
Layers              11             16                 +5
Total params        27.07M         24.72M             -2.35M
MLP params          17.30M         11.01M             -6.29M
Attention params     8.65M         12.58M             +3.93M
Artifact            ~15.86 MB      ~15.77 MB          -0.09 MB
Headroom            ~30 KB         ~226 KB            +196 KB
```