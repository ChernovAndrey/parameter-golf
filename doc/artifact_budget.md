# Artifact Budget — Parameter Golf SOTA (1.1147 BPB)

Total artifact = compressed model bytes + code bytes.
Hard cap: 16,000,000 bytes (decimal 16 MB).

---

## Raw Model (before any quantization)

Serialized float32 state dict: **106,289,590 bytes (~101 MB)**

| Component | Shape | Params | Raw bytes (float32) | % of raw |
|-----------|-------|--------|--------------------:|:--------:|
| MLP up bank | [11, 1536, 512] | 8,650,752 | 34,603,008 | 32.6% |
| MLP down bank | [11, 512, 1536] | 8,650,752 | 34,603,008 | 32.6% |
| QO bank (Q + Out) | [22, 512, 512] | 5,767,168 | 23,068,672 | 21.7% |
| KV bank (K + V) | [22, 256, 512] | 2,883,584 | 11,534,336 | 10.9% |
| Token embedding | [1024, 512] | 524,288 | 2,097,152 | 2.0% |
| BigramHash embed | [3072, 112] | 344,064 | 1,376,256 | 1.3% |
| BigramHash proj | [512, 112] | 57,344 | 229,376 | 0.2% |
| VE embed | [1024, 128] | 131,072 | 524,288 | 0.5% |
| VE proj | [256, 128] | 32,768 | 131,072 | 0.1% |
| Small params (norms, scales, gates, skip_weights, q_gain, etc.) | various | ~50K | ~200,000 | 0.2% |
| **Total** | | **~27.1M** | **~108 MB** | **100%** |

---

## Quantization Pipeline

```
Float32 (101 MB)
    │
    ▼
Late QAT (during training)
    int6 STE fake-quantization when warmdown > 15%
    Trains model to be robust to int6 rounding
    │
    ▼
EMA + SWA (weight averaging)
    EMA(0.997) shadow weights applied
    SWA snapshots averaged (every 50 steps)
    │
    ▼
AR Self-Generated Calibration
    Model generates 64 seqs x 2048 tokens (temp=0.8)
    No train/val data accessed
    │
    ▼
Full Hessian GPTQ (int6)
    H = X^T X from self-generated sequences
    Cholesky error compensation
    Column reordering for minimal reconstruction error
    Large tensors (> 65K elements): int6 per-row quantization
    Small tensors (< 65K elements): kept in float16/float32
    │
    ▼
Selective Pruning
    Values near -1, 0, +1 snapped to exactly -1, 0, +1
    Decision based on per-row reconstruction error
    Creates more repeated values for better compression
    │
    ▼
LZMA preset=9 compression
    Maximum compression level
    Exploits repeated patterns from quantization + pruning
    │
    ▼
Final artifact: ~15.76 MB compressed model + ~0.10 MB code = ~15.86 MB
```

---

## After Each Stage (estimated)

| Stage | Size | Compression ratio |
|-------|-----:|:-----------------:|
| Raw float32 state dict | 101 MB | 1.0x |
| After int8 quantization (baseline approach) | ~27 MB | 3.7x |
| After int6 quantization | ~20 MB | 5.0x |
| After Full Hessian GPTQ int6 | ~18 MB | 5.6x |
| After selective pruning | ~17.5 MB | 5.8x |
| After LZMA preset=9 | ~15.76 MB | 6.4x |
| + code (train_gpt.py) | ~15.86 MB | 6.4x |

---

## Per-Component Artifact Breakdown (after int6 GPTQ + LZMA)

| Component | Raw (float32) | After quant + compress | % of artifact | Notes |
|-----------|:-------------:|:----------------------:|:-------------:|-------|
| MLP banks (up + down) | 69.2 MB | ~10.0 MB | 63% | Biggest cost, 3x expansion |
| QO bank (Q + Out) | 23.1 MB | ~3.3 MB | 21% | 22 matrices of 512x512 |
| KV bank (K + V) | 11.5 MB | ~1.7 MB | 11% | 22 matrices of 256x512 |
| Token embedding | 2.1 MB | ~0.3 MB | 2% | Kept in float16 (< 65K elem threshold) |
| BigramHash | 1.6 MB | ~0.2 MB | 1% | Embed + projection |
| VE + small params | ~0.9 MB | ~0.2 MB | 1% | Norms, scales, gates in float32 |
| Code | — | ~0.1 MB | 1% | train_gpt.py UTF-8 |
| **Total** | **~108 MB** | **~15.86 MB** | **100%** | |

---

## What Gets Quantized vs Kept Float

| Category | Quantization | Storage | Why |
|----------|-------------|---------|-----|
| MLP banks | int6 GPTQ per-row | Compressed | Large matrices, biggest savings |
| QO bank | int6 GPTQ per-row | Compressed | Large matrices |
| KV bank | int6 GPTQ per-row | Compressed | Large matrices |
| Token embedding | float16 passthrough | Compressed | < 65K elements, tied with lm_head |
| BigramHash embed | float16 passthrough | Compressed | < 65K elements |
| Control tensors (attn_scale, mlp_scale, resid_mix, q_gain, skip_weights, smear gate, VE scales) | float32 passthrough | Compressed | Must be precise, tiny param count |

The 65K element threshold is the cutoff: tensors below this are too small for int6/int8
quantization to help (overhead of storing per-row scales outweighs savings).

---

## Budget Pressure Points

| If you want to... | Cost in artifact | What it displaces |
|---|---|---|
| Add 1 layer | ~1.0 MB (attn + MLP banks grow) | Must shrink something else |
| Go 4x MLP (1536 to 2048) | ~3.3 MB more | Blows budget without compensating cuts |
| Double BigramHash (3072 to 6144) | ~0.2 MB more | Easily fits |
| Add 1024 vocab tokens | ~0.3 MB (embedding grows) | Marginal |
| Switch from int6 to int5 | Saves ~2.5 MB | Quality drop from coarser quantization |
| Switch from int6 to int8 | Costs ~5.0 MB more | Way over budget |

---

## Per-Seed Artifact Sizes (from logs)

| Seed | Raw model | Int6+LZMA model | Code | Total artifact |
|------|----------:|----------------:|-----:|---------------:|
| 314 | 106,289,590 | 15,761,428 | 101,850 | 15,863,278 |
| 42 | 106,289,590 | 15,883,000 | 101,850 | 15,984,850 |
| 999 | 106,289,590 | 15,774,460 | 101,850 | 15,876,310 |

Note: raw model size is identical across seeds (same architecture).
Compressed size varies by ~120KB due to different weight distributions from different
random seeds affecting LZMA compressibility.
