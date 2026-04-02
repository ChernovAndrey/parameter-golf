# Monitoring Checklist: What to Watch During Training & Eval

Things that could go wrong, why, and how to detect and fix them.

---

## 1. Adapter Activation (CRITICAL — first thing to verify)

### What to check
The adapter B matrices must become non-zero within the first ~50-100 steps.
If they stay near-zero, the model is running as a pure shared MLP with no
per-layer specialization — defeating the entire purpose of the architecture.

### Why this could fail
We initialize A with `kaiming_uniform_` and B with zeros. This means:
```
Step 0:  A@B = A@0 = 0   (no adapter contribution, correct)
Step 1:  dL/dB = A^T @ grad  (non-zero, B starts moving)
Step 2:  dL/dA = grad @ B^T  (B is now non-zero, A starts moving too)
```

If something prevents B from getting gradients (e.g., gradient clipping
kills the small signal, Muon's NS5 normalizes away the tiny gradient),
adapters stay dead.

### How to detect
Add this logging after ~100 steps:
```python
if step == 100 and rank == 0:
    for name in ["adapter_up_B", "adapter_down_B"]:
        p = getattr(base_model, name)
        print(f"{name}: norm={p.data.norm():.6f} max={p.data.abs().max():.6f}")
```

- **Healthy**: B norm > 0.01, growing over time
- **Broken**: B norm ≈ 0 after 100+ steps → adapters are dead

### How to fix if broken
1. Initialize B with small random values too (e.g., `normal_(std=0.01)`).
   This breaks the `A@B = 0` init property, but at least adapters are alive.
2. Or use a warmup: freeze shared bases for first 200 steps, only train adapters.

---

## 2. Muon Scale Asymmetry for Non-Square Adapters

### What to check
The Muon optimizer applies a scale factor based on matrix aspect ratio:
`scale = max(1, rows/cols)^0.5`. For our adapter banks:

| Bank | Shape per slice | Aspect | Scale | Effective LR |
|------|----------------|--------|-------|-------------|
| adapter_up_A | [1536, 100] | 15.4:1 | 3.92 | 0.098 |
| adapter_up_B | [100, 512] | 1:5.1 | 1.0 | 0.025 |
| adapter_down_A | [512, 100] | 5.1:1 | 2.26 | 0.057 |
| adapter_down_B | [100, 1536] | 1:15.4 | 1.0 | 0.025 |

The A matrices move ~2-4x faster than B matrices. This asymmetry is
Muon's intended behavior for non-square matrices, but could cause instability
with such extreme aspect ratios (the current SOTA banks are roughly square).

### Why this could fail
If A matrices update too aggressively relative to B, the adapter product
`A@B` can oscillate — A moves, then B compensates, then A compensates back.
This wastes training steps on oscillation rather than progress.

### How to detect
- Train loss oscillates or increases after initial drop
- `step_avg` is normal but loss curve is noisy compared to baseline
- A matrix norms grow much faster than B matrix norms

### How to fix if broken
1. Lower MATRIX_LR from 0.025 to 0.02 or 0.018 (affects all Muon banks)
2. Or move adapter banks to Adam instead of Muon (separate optimizer with
   lower LR, eliminates the NS5 scale asymmetry)
3. Or use a separate Muon instance for adapters with lower LR

---

## 3. Shared Base Gradient Magnitude

### What to check
Each shared MLP base receives accumulated gradients from 5-6 layers.
A per-layer attention bank gets gradient from 1 layer. The raw gradient
magnitude for shared bases is ~5-6x larger.

### Why this is probably OK
Muon's Newton-Schulz normalization orthogonalizes the gradient and produces
unit-norm updates. The 5-6x larger raw gradient gets normalized away —
the update magnitude is the same as for per-layer banks. Only the direction
is affected (it's the average direction across layers in the group).

### When this could fail
If one layer in the group dominates the gradient (e.g., layer 0 has much
larger gradients than layer 4), the shared base will be pulled mostly toward
that layer's needs, hurting the others.

### How to detect
- Compare train loss to a baseline run with `ADAPTER_RANK=0 NUM_LAYERS=11`
- If the 16L model has WORSE pre-quant loss than the 11L model, the
  optimization is broken (shared bases aren't learning well)

### How to fix if broken
1. Use Adam for shared bases with lower LR (e.g., 0.01)
2. Add gradient accumulation normalization (divide shared base grad by
   number of layers in its group before Muon)

---

## 4. Quantization Gap

### What to check
Compare `DIAGNOSTIC post_ema val_bpb` (pre-quant) vs `final_int6_roundtrip val_bpb`
(post-quant). The difference is the quantization gap.

### Acceptable gap
- SOTA gap: ~0.003 BPB (1.1344 pre-quant → 1.1386 post-quant for seed 314)
- Our target: gap < 0.005 BPB
- If gap > 0.01 BPB: quantization is seriously broken

### Why our gap could be larger
We quantize shared bases and adapters SEPARATELY, then reconstruct
effective weights at eval:
```
W_eff = deQ(W_shared) + deQ(A) @ deQ(B)
```

Two sources of quantization error combine:
1. Shared base error: `deQ(W_shared) - W_shared`
2. Adapter error: `deQ(A)@deQ(B) - A@B`

The SOTA quantizes the full effective weight as one matrix — single
source of error. Our approach has two error sources that add up.

Additionally: adapter B matrices are stored in fp16 (< 65K elements),
which loses some precision. And adapter A matrices use per-row int6
WITHOUT GPTQ Hessian info (just percentile search), which is less optimal.

### How to detect
```
DIAGNOSTIC post_ema val_bpb: X.XXXX
final_int6_roundtrip val_bpb: Y.YYYY
Gap = Y.YYYY - X.XXXX
```

### How to fix if gap too large
1. **Noisy QAT for shared bases**: During training, inject noise calibrated
   to int6 step size on the shared base weights. This trains the model to
   be robust to quantization of the shared component:
   ```python
   with torch.no_grad():
       amax = shared_weight.float().abs().amax(dim=1, keepdim=True)
       step_size = amax / 31.0
   noise = (torch.rand_like(w) - 0.5) * step_size
   w_noisy = w + noise
   ```
2. **Quantize effective weights instead**: Materialize `W_shared + A@B`
   for each layer, then quantize those 16 full-rank matrices. This uses
   the same artifact space as 16 independent MLPs and DEFEATS the purpose.
   Only use as a last resort.
3. **Lower adapter rank**: Rank 64 instead of 100. Smaller adapters =
   less quantization error from adapters.

---

## 5. Artifact Size

### What to check
Log line: `Total submission size int6+lzma: XXXXX bytes`
Must be < 16,000,000 bytes.

### Our estimate: ~15.77 MB (226 KB headroom)

### Why it could be wrong
- LZMA compression ratio varies by ~120 KB across seeds (observed in SOTA)
- Our adapter A matrices are initialized with kaiming_uniform_ (not zero),
  so they have more entropy after training → potentially harder to compress
- If adapter weights don't converge well, they might have higher entropy
  than the shared bases → worse compression ratio

### How to detect
Check all 3 seeds. The worst-case seed must still be < 16 MB.

### How to fix if over budget
1. Lower TARGET_MB to 15.5 (more aggressive selective pruning)
2. Lower adapter rank to 96 (saves ~200 KB)
3. Check if selective pruning is activating (log line `selective_prune:`)

---

## 6. Step Time

### What to check
Log line: `step_avg:XXms`

### Acceptable range
- Expected: ~120-135ms on 8xH100
- SOTA: 86.7ms (11 layers)
- Ratio: 16/11 * 86.7 ≈ 126ms, plus ~2-5ms adapter overhead

### Why it could be worse than expected
- Adapter matmul `A[i]@B[i]` is computed 32 times per forward pass
  (16 layers × 2 for up/down), adding small overhead
- torch.compile may not optimize the adapter matmuls as well as the
  main MLP matmuls (different shapes, not fused)
- More parameters in Muon (8 banks vs 4) means more reduce-scatter
  and all-gather operations during the optimizer step

### How to detect
Compare step_avg to expected ~126ms. If > 140ms, something is wrong.

### How to fix if too slow
1. Check nvidia-smi for competing processes
2. Try reducing BIGRAM_VOCAB_SIZE to 2048 (saves ~1ms)
3. Profile: is the overhead in forward, backward, or optimizer?
4. If adapter matmuls are the bottleneck: pre-compute A@B once per step
   instead of inside the forward (save in a buffer, reuse in forward_logits)

---

## 7. Training Steps (consequence of step time)

### What to check
Total steps completed before wallclock cap.

### Acceptable
- Expected: ~4,500-5,000 steps in 600s
- SOTA: ~6,920 steps

### Why fewer steps could hurt
~31% fewer steps means less training. Each step trains the same batch size
but the model has more parameters to learn (5 extra attention layers).
The depth gain must compensate for the training deficit.

### How to detect
- `stopping_early: wallclock_cap train_time:600XXXms step:XXXX/20000`
- If steps < 4,000: step time is too high, architecture is not competitive

### How to fix
1. Accept the tradeoff (if BPB still beats SOTA)
2. Reduce adapter rank (smaller matmuls → faster steps)
3. Reduce warmdown (WARMDOWN_ITERS=2500) to give more full-LR steps
4. Lower to 15 layers instead of 16

---

## 8. Train Loss Curve Shape

### What to check
The train_loss logged every 500 steps should decrease smoothly.
Compare to SOTA's log as a reference.

### Warning signs
- **Loss increases after step ~100**: Optimization instability. Likely
  caused by Muon scale asymmetry on adapters or shared base gradient issues.
- **Loss plateaus early**: Model isn't learning from the extra depth.
  Adapters might be dead or the shared bases dominate too much.
- **Loss decreases then spikes at SWA/QAT activation**: Weight averaging
  or QAT interacts badly with the adapter structure.

### SOTA reference (seed 314)
```
step:500   train_loss:2.3787
step:1000  train_loss:2.2509
step:1500  train_loss:2.1982
step:2000  train_loss:2.0412
step:3000  train_loss:2.1423
step:4000  train_loss:1.9433
step:5000  train_loss:2.0805
step:6000  train_loss:1.9209
step:6927  val_bpb:1.1354 (pre-quant)
```

Our curve will have fewer total steps but should show similar or better
loss at each step count. If loss at step 2000 is worse than SOTA's 2.04,
the model is behind and may not recover.

---

## 9. Eval Time

### What to check
Eval must complete in < 600s on 8xH100.

### Expected: ~350s total
- Standard eval: ~175s (16 layers vs 11, ~45% more)
- Sliding window: ~175s
- No TTT

### Why it could be worse
16 layers means 45% more compute per eval token. If the sliding window
eval is memory-bound rather than compute-bound, the impact is smaller.

### How to detect
```
final_int6_roundtrip eval_time:XXXXXms
final_int6_sliding_window eval_time:XXXXXms
```
Sum must be < 600,000ms.

### How to fix if over 600s
Not expected to happen (~350s << 600s). If it does:
1. Increase eval stride (EVAL_STRIDE=128 instead of 64)
2. Reduce eval_seq_len

---

## Quick Reference: What to Look For in Logs

```
# After ~100 steps: verify adapters are alive
adapter_up_B: norm=X.XXXXXX max=X.XXXXXX   ← must be > 0.01

# Training:
step_avg:XXms          ← expect 120-135ms
train_loss:X.XXXX      ← should decrease smoothly

# After training:
model_params:XXXXX     ← expect ~24.7M
stopping_early step:XXXX  ← expect 4500-5000

# Quantization:
DIAGNOSTIC post_ema val_bpb:X.XXXX     ← pre-quant BPB
Total submission size int6+lzma: XXXXX  ← must be < 16,000,000
final_int6_roundtrip val_bpb:X.XXXX    ← gap from pre-quant should be < 0.005

# Final score:
final_int6_sliding_window val_bpb:X.XXXX  ← must be < 1.1147 to beat SOTA
```