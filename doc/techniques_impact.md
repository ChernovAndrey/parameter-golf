# Technique Impact Rankings — Parameter Golf SOTA (1.1147 BPB)

Approximate BPB impact of each technique, estimated from ablations and
submission history across the leaderboard.

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
