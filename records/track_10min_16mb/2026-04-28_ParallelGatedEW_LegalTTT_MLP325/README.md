# Record: Parallel-Zone Gated Attention + Targeted MLP 3.25× + Legal Score-First TTT

**val_bpb = TODO_FILL_AFTER_RUNS** (3-seed mean, std TODO) | **TODO MB** | 8× H100 80GB SXM, 600s training cap

> Built on PR #1493 SOTA. Adds Qwen G1 elementwise gated attention restricted to the parallel-residual layers (7-10) and re-distributes the artifact budget by shrinking the MLPs on those *same* layers (4.0× → 3.25×). Layers 0-6 (including looped layers 3-5) remain unchanged from SOTA.

## 3-Seed Results

| Seed | Sliding val_bpb | **TTT val_bpb** | Artifact bytes |
|------|---|---|---|
| 42   | TODO | **TODO** | TODO |
| 314  | TODO | **TODO** | TODO |
| 999  | TODO | **TODO** | TODO |
| **Mean** | **TODO** | **TODO** | **TODO** |
| **Std**  | TODO | TODO | |

Single-seed reference (no TTT, run on 2026-04-27, same architecture):
- seed 42 sliding val_bpb = 1.08226 (sub-SOTA-by-0.0006 vs SOTA's 1.08274 sliding mean)
- artifact 15,804,975 bytes (195 KB headroom under 16 MB cap)

Previous SOTA: PR #1493 = **1.0810 BPB** (TTT, 3-seed mean, std 0.0002).

## Key Techniques

Inherited from PR #1493 SOTA:
1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0) (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** on layers 3, 4, 5; 17 virtual layers from 11 physical (PR #1331, #1437 @dexhunter)
3. **Parallel Residuals** from layer 7 — GPT-J style (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25** — learnable per-head query scaling
5. **Legal Score-First TTT** — SGD lr=0.005, momentum=0.9, 3 epochs/chunk, 32K-token chunks, score-before-update (PR #549 @abaybektursun, PR #1413 @dexhunter)
6. **Tuned hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)

New in this submission:

7. **Parallel-zone gated attention (Qwen G1, elementwise)** — sigmoid gate applied to attention output before W_o, restricted to layers 7-10 only:
   ```
   y = sigmoid(W_g @ x) * Attn(x)
   ```
   `W_g` is per-layer `[dim → attn_dim]` = `[512 → 512]`. 4 gates total × 262,144 params = 1.05 M new params. Gates are placed *only* where the residual structure is parallel — i.e., where the attention output enters the residual stream without a downstream MLP nonlinearity. In sequential layers 0-6, attn → MLP, so the MLP's LeakyReLU² already supplies a per-token nonlinearity; in parallel layers 7-10, attn ‖ MLP are summed linearly into the residual, leaving attn without its own nonlinearity — gating fills that gap. (Validated in our FatBlock experiment, doc/experiment_fatblock_results.md.)
   Reference: Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025 best paper, arXiv:2505.06708.

8. **Targeted MLP 3.25× on parallel layers only** — the MLPs at layers 7-10 shrink from `h=2048` → `h=1664` (= 13×128, clean GPTQ block alignment). Layers 0-6 keep `h=2048`. The MLP shrink offsets the gate's artifact cost while preserving the high-leverage looped MLPs (layers 3, 4, 5 are visited 3× per pass when looping is active). Per-token MLP volume in the parallel zone (4 × 1664 = 6656) sits ~8% above FatBlock's validated regime (1 × 6144).

## Architecture

11 physical layers × 512 dim, 8 heads / 4 KV heads (GQA). Partial RoPE (16/64), LeakyReLU(0.5)², layerwise LN scale, tied embeddings, logit softcap 30.0. SP8192 BPE vocab.

- Layers 0-6 (sequential residual): MLP 4.0× (h=2048), no gate.
- Layers 7-10 (parallel residual, GPT-J style): **MLP 3.25× (h=1664), elementwise gated attention.**
- Depth recurrence: encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]` (loops layers 3-5, activated at frac=0.35).
- Skip-gated U-Net connections.

Total params: **35,420,248** (vs SOTA 27.1 M; the +8.3 M is mostly the 4 elementwise gates plus skipgate + recurrence overhead, partially offset by the 4-layer MLP shrink).

## Training

MuonEq-R optimizer (row-normalized Muon, NS-5), AdamW for embeddings/scalars. **8× H100 SXM, 600s wall-clock** (`MAX_WALLCLOCK_SECONDS=600`) — competition spec, same as PR #1493 SOTA. Linear warmdown to LR=0 over final 72% of training. EMA decay 0.9965.

## Quantization

Full-Hessian GPTQ with SDClip (`clip = k · σ_row`). int6 for matrices, int8 for token embeddings. Byte-shuffle + Brotli-11 compression. No selective pruning needed.

## TTT (Test-Time Training)

Identical to PR #1493 SOTA: chunk-based SGD adaptation at eval time, score-first ordering, 32K-token chunks, 3 epochs per chunk, cosine LR decay, gradient clipping at 1.0, distributed all-reduce.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):
- **Causality:** strictly causal sliding-window eval.
- **Normalized distribution:** standard softmax over full vocab; no n-gram cache, no logit biasing.
- **Score before update:** every chunk fully scored under `torch.inference_mode()` BEFORE any SGD update.
- **Single pass:** each token scored exactly once.
- No SLOT, no pre-quant TTT on val data, no ETLB.
- No use of validation data for any architecture/hyperparameter tuning beyond what PR #1493 already did.

## Reproducibility

```bash
cd records/track_10min_16mb/2026-04-28_ParallelGatedEW_LegalTTT_MLP325
./run.sh check     # ~5s pre-flight (8 GPU + modules + data)
./run.sh smoke     # 30s sanity (artifact-size check)
./run.sh           # all 3 seeds back-to-back, ~55 min on 8x H100
# or per-seed:
./run.sh 42
./run.sh 314
./run.sh 999
```

Logs land in `train_seed{42,314,999}.log`. The launcher auto-installs brotli if missing.

## Files

- `train_gpt.py` — trainer (50,755 bytes UTF-8)
- `run.sh` — 3-seed launcher
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log` — full training+eval logs
- `submission.json` — machine-readable metadata
- `README.md` — this file
