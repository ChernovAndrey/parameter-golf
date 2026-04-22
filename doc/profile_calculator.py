#!/usr/bin/env python3
"""
Profile calculator for Parameter Golf architectures.
Estimates per-component compute, artifact size, and training steps.

v1 (2026-03): 11L × 512d × MLP 3×, SP1024, LZMA compression, no recurrence.
v2 (2026-04): adds SP8192, MLP 4×, depth recurrence, parallel residuals,
              SDClip + byte-shuffle + Brotli-11 per-component compression,
              int8 embeddings with GPTQ, LZMA code wrapper.

Usage:
  python3 doc/profile_calculator.py                         # default = new SOTA (PR #1493)
  python3 doc/profile_calculator.py --preset old_sota       # old SOTA (PR #1019)
  python3 doc/profile_calculator.py --preset new_sota
  python3 doc/profile_calculator.py --all                   # run preset comparison
  python3 doc/profile_calculator.py --mlp 2.5 --kv 4 --rank 51 --bases 6   # custom

Backward-compatible: all v1 kwargs still work with the same defaults when
preset="old_sota" is chosen. New kwargs default to no-op for v1 calls.
"""
import argparse
import math


# ──────────────────────────────────────────────────────────────────────────
# Presets — the two snapshots the doc talks about
# ──────────────────────────────────────────────────────────────────────────

PRESETS = {
    # 2026-03-25 SOTA (last-pulled baseline): AR self-gen GPTQ + XSA + BigramHash
    "old_sota": dict(
        dim=512, heads=8, kv_heads=4, mlp_mult=3.0, n_layers=11,
        vocab=1024, bigram_vocab=3072, bigram_dim=112, ve_dim=128,
        smeargate=True,
        loop_layers=None, num_loops=0, parallel_start=None,
        matrix_bits=6, embed_bits=8,
        # Observed LZMA preset=9 compression on the old weight distribution.
        # Single ratio across the artifact (no per-component brotli tuning).
        mlp_compress=10.0 / 17.35,     # ≈ 0.577
        attn_compress=5.0 / 6.50,      # ≈ 0.769 (from old profile 8.68→~5.0 MB)
        embed_compress=0.3 / 2.1,      # ≈ 0.143 — SP1024 was tiny
        code_bytes=115_000,
        calibration_step_ms=86.7, calibration_vocab=1024, calibration_mlp=3.0,
    ),
    # 2026-04-09 SOTA (current): SP8192 + 3-Layer Recurrence + ParResid + QK5.25 + Legal TTT
    # Encoder virtual seq [0,1,2,3,4,5,3,4] + Decoder [5,3,4,5,6,7,8,9,10] = 17 evals.
    # Each of layers 3/4/5 is visited 3× total (base + 2 extra), so num_loops=2.
    "new_sota": dict(
        dim=512, heads=8, kv_heads=4, mlp_mult=4.0, n_layers=11,
        vocab=8192, bigram_vocab=0, bigram_dim=0, ve_dim=0, smeargate=False,
        loop_layers=[3, 4, 5], num_loops=2, enable_fraction=0.35,
        parallel_start=7,
        matrix_bits=6, embed_bits=8,
        # Observed per-component Brotli-11 + byte-shuffle + SDClip ratios,
        # derived from PR #1493 artifact (15.99 MB breakdown in sota_profile.md).
        mlp_compress=11.3 / 17.30,     # ≈ 0.653
        attn_compress=3.0 / 6.50,      # ≈ 0.462
        embed_compress=1.1 / 4.20,     # ≈ 0.262   (SP8192 × int8)
        code_bytes=17_000,             # LZMA-wrapped train_gpt.py
        calibration_step_ms=129.0, calibration_vocab=8192, calibration_mlp=4.0,
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# Main profile calculator
# ──────────────────────────────────────────────────────────────────────────

def calc_profile(
    # Core architecture
    dim=512, heads=8, kv_heads=4, mlp_mult=3.0, n_layers=11,
    # Shared-MLP / adapter experiment params (v1)
    n_bases=0, adapter_rank=0,
    # Legacy components (default off for new SOTA)
    bigram_vocab=3072, bigram_dim=112, ve_dim=128, smeargate=True,
    vocab=1024, seq_len=2048, batch_tokens=786432, world_size=8,
    # Compression:
    #   lzma_ratio is the v1 single-global-ratio (old SOTA).
    #   mlp_compress / attn_compress / embed_compress are per-component ratios
    #   (v2). If any are given, they override lzma_ratio for that component.
    lzma_ratio=1.70,
    mlp_compress=None, attn_compress=None, embed_compress=None,
    matrix_bits=6, embed_bits=8,
    # Recurrence (v2): loop_layers is a list of physical layer indices that
    # get re-visited. num_loops is the number of EXTRA visits (so num_loops=1
    # means each looped layer is visited twice total inside the recurrence).
    loop_layers=None, num_loops=0, enable_fraction=0.35,
    # Parallel residuals (v2): first physical layer index using GPT-J style
    # parallel residual (None = sequential everywhere). FLOPs identical; this
    # is carried for reporting only.
    parallel_start=None,
    # Code overhead
    code_bytes=115_000,
    # Step-time calibration (v2)
    calibration_step_ms=86.7, calibration_vocab=1024, calibration_mlp=3.0,
    target_step_ms=None,
):
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    mlp_dim = int(mlp_mult * dim)
    tokens_per_gpu = batch_tokens // world_size

    # ================================================================
    # PARAMETER COUNT
    # ================================================================
    attn_q = n_layers * dim * dim
    attn_k = n_layers * kv_dim * dim
    attn_v = n_layers * kv_dim * dim
    attn_o = n_layers * dim * dim
    attn_total = attn_q + attn_k + attn_v + attn_o

    if n_bases > 0:
        mlp_shared = n_bases * 2 * mlp_dim * dim
        if adapter_rank > 0:
            mlp_adapt = n_layers * (mlp_dim * adapter_rank + adapter_rank * dim +
                                     dim * adapter_rank + adapter_rank * mlp_dim)
        else:
            mlp_adapt = 0
        mlp_indep = 0
    else:
        mlp_shared = 0
        mlp_adapt = 0
        mlp_indep = n_layers * 2 * mlp_dim * dim

    mlp_total = mlp_shared + mlp_adapt + mlp_indep

    # Other (emb, legacy bigram/VE/smeargate, skip weights, per-block scalars)
    tok_emb = vocab * dim
    bigram_emb = bigram_vocab * bigram_dim
    bigram_proj = dim * bigram_dim if bigram_dim > 0 else 0
    ve_emb = vocab * ve_dim if ve_dim > 0 else 0
    ve_proj = kv_dim * ve_dim if ve_dim > 0 else 0
    n_enc = n_layers // 2
    n_dec = n_layers - n_enc
    skip_w = min(n_enc, n_dec) * dim
    per_block = n_layers * (dim + dim + 2 * dim + heads)  # LN scales, resid_mix, q_gain
    smear = dim if smeargate else 0
    other_total = (tok_emb + bigram_emb + bigram_proj + ve_emb + ve_proj
                   + skip_w + per_block + smear)

    total_params = attn_total + mlp_total + other_total

    # ================================================================
    # ARTIFACT SIZE (bytes)
    # ================================================================
    # Bits-per-param for matrix vs embedding
    matrix_byte_factor = matrix_bits / 8.0
    embed_byte_factor = embed_bits / 8.0

    # Attention: matrix_bits with per-row scale overhead (2 bytes/row)
    attn_rows = n_layers * (dim + kv_dim + kv_dim + dim)
    attn_raw = attn_total * matrix_byte_factor + attn_rows * 2

    # MLP shared
    if n_bases > 0:
        mlp_shared_rows = n_bases * (mlp_dim + dim)
        mlp_shared_raw = mlp_shared * matrix_byte_factor + mlp_shared_rows * 2
    else:
        mlp_shared_raw = 0

    # MLP independent
    if mlp_indep > 0:
        mlp_indep_rows = n_layers * (mlp_dim + dim)
        mlp_indep_raw = mlp_indep * matrix_byte_factor + mlp_indep_rows * 2
    else:
        mlp_indep_raw = 0

    # Adapters (fp16 vs int6 threshold, carried from v1 for MoE/shared experiments)
    max_lossless_rank = 65536 // mlp_dim if mlp_dim > 0 else 0
    adapters_all_fp16 = adapter_rank <= max_lossless_rank
    if adapter_rank > 0:
        adapter_matrices = [
            (mlp_dim * adapter_rank, mlp_dim),   # A_up
            (adapter_rank * dim, adapter_rank),    # B_up
            (dim * adapter_rank, dim),             # A_down
            (adapter_rank * mlp_dim, adapter_rank), # B_down
        ]
        adapt_raw = 0
        adapt_fp16_count = 0
        adapt_int6_count = 0
        for elems, rows in adapter_matrices:
            if elems <= 65536:
                adapt_raw += n_layers * elems * 2   # fp16
                adapt_fp16_count += 1
            else:
                adapt_raw += n_layers * (elems * matrix_byte_factor + rows * 2)
                adapt_int6_count += 1
    else:
        adapt_raw = 0
        adapt_fp16_count = 0
        adapt_int6_count = 0

    # Embeddings / legacy other
    tok_raw = tok_emb * embed_byte_factor + vocab * 2       # int8 (or configurable)
    bigram_raw = (bigram_emb + bigram_vocab * 2 + bigram_proj * 2) if bigram_vocab > 0 else 0
    ve_raw = (ve_emb + vocab * 2 + ve_proj * 2) if ve_dim > 0 else 0
    small_raw = per_block * 4 + skip_w * 4 + smear * 4 + 8  # fp32 scalars
    other_raw = tok_raw + bigram_raw + ve_raw + small_raw
    overhead = 180_000  # tokenizer model + headers + misc

    # Per-component compression. If per-component ratios given, use them;
    # otherwise fall back to the legacy single lzma_ratio (1/lzma_ratio keep).
    def _compress(raw_bytes, per_comp_ratio):
        if per_comp_ratio is None:
            return raw_bytes / lzma_ratio
        return raw_bytes * per_comp_ratio

    attn_bytes = _compress(attn_raw, attn_compress)
    mlp_indep_bytes = _compress(mlp_indep_raw, mlp_compress)
    mlp_shared_bytes = _compress(mlp_shared_raw, mlp_compress)
    adapt_bytes = _compress(adapt_raw, mlp_compress)  # adapters use MLP ratio
    tok_bytes = _compress(tok_raw, embed_compress)
    # Legacy components + small scalars keep the bulk lzma ratio (not separately tuned)
    legacy_other_raw = bigram_raw + ve_raw + small_raw + overhead
    legacy_other_bytes = legacy_other_raw / lzma_ratio

    total_artifact_weights = (attn_bytes + mlp_indep_bytes + mlp_shared_bytes
                              + adapt_bytes + tok_bytes + legacy_other_bytes)
    total_artifact = total_artifact_weights + code_bytes
    headroom = 16_000_000 - total_artifact

    # ================================================================
    # COMPUTE (FLOPs per step, forward-only per layer-evaluation)
    # ================================================================
    T = tokens_per_gpu

    # Per-layer-evaluation forward-only FLOPs
    flops_q = 2 * T * dim * dim
    flops_k = 2 * T * dim * kv_dim
    flops_v = 2 * T * dim * kv_dim
    flops_o = 2 * T * dim * dim
    flops_attn_proj = flops_q + flops_k + flops_v + flops_o

    # Flash Attention 3: empirically ~30% of projection cost on H100
    flops_flash = 0.30 * flops_attn_proj

    flops_mlp_up = 2 * T * dim * mlp_dim
    flops_mlp_down = 2 * T * mlp_dim * dim
    flops_mlp = flops_mlp_up + flops_mlp_down

    if adapter_rank > 0:
        flops_adapt = 2 * (mlp_dim * adapter_rank * dim + dim * adapter_rank * mlp_dim)
    else:
        flops_adapt = 0

    flops_overhead = 0.05 * (flops_attn_proj + flops_mlp)

    flops_per_layer_eval = (flops_attn_proj + flops_flash + flops_mlp
                            + flops_adapt + flops_overhead)

    # Layer-evaluations per step (accounts for recurrence)
    # Each physical layer is visited once by default; looped layers get extra visits.
    per_layer_evals = [1] * n_layers
    if loop_layers:
        for li in loop_layers:
            if 0 <= li < n_layers:
                per_layer_evals[li] += num_loops
    total_evals_loop_on = sum(per_layer_evals)
    total_evals_loop_off = n_layers

    # Token embedding / tied LM head forward FLOPs (once per step, not per-layer)
    # Embedding lookup is a gather (~0 FLOP); LM head is [B*T, dim] × [dim, vocab].
    flops_lm_head = 2 * T * dim * vocab

    # Per-step totals (forward only)
    flops_fwd_loop_on = total_evals_loop_on * flops_per_layer_eval + flops_lm_head
    flops_fwd_loop_off = total_evals_loop_off * flops_per_layer_eval + flops_lm_head

    # Fwd + bwd ≈ 3× forward
    flops_step_loop_on = flops_fwd_loop_on * 3
    flops_step_loop_off = flops_fwd_loop_off * 3

    # ================================================================
    # COMPONENT BREAKDOWN (absolute + %) for loop-active step
    # ================================================================
    def _pct(x, total):
        return 100.0 * x / total if total > 0 else 0.0

    # Latency % (loop-active, fwd+bwd)
    comp_mlp_flops   = total_evals_loop_on * flops_mlp * 3
    comp_attn_flops  = total_evals_loop_on * flops_attn_proj * 3
    comp_flash_flops = total_evals_loop_on * flops_flash * 3
    comp_over_flops  = total_evals_loop_on * flops_overhead * 3
    comp_adapt_flops = total_evals_loop_on * flops_adapt * 3
    comp_lm_flops    = flops_lm_head * 3
    total_step = flops_step_loop_on
    latency_pct = {
        "MLP":              _pct(comp_mlp_flops, total_step),
        "Attention proj.":  _pct(comp_attn_flops, total_step),
        "Flash Attention":  _pct(comp_flash_flops, total_step),
        "LM head / emb":    _pct(comp_lm_flops, total_step),
        "Overhead":         _pct(comp_over_flops, total_step),
        "Adapter A@B":      _pct(comp_adapt_flops, total_step),
    }

    # Memory % (artifact)
    comp_mem = {
        "MLP":              mlp_indep_bytes + mlp_shared_bytes + adapt_bytes,
        "Attention":        attn_bytes,
        "Token embedding":  tok_bytes,
        "Other + code":     legacy_other_bytes + code_bytes,
    }
    memory_pct = {k: _pct(v, total_artifact) for k, v in comp_mem.items()}

    # ================================================================
    # STEP TIME ESTIMATE
    # ================================================================
    # Use the calibration snapshot (old SOTA: 86.7 ms, new SOTA: 129 ms avg).
    # Compute a ratio using the same model as the calibration point, then
    # apply sqrt-like wall-time scaling. For loop-based models we report two
    # numbers (loop-on and loop-off) and a weighted average.
    cal_dim = 512; cal_heads = 8; cal_kv = 4; cal_layers = 11
    cal_mlp = int(calibration_mlp * cal_dim)
    cal_T = T
    cal_flops_per_layer = (
        2 * cal_T * cal_dim * cal_dim +            # Q
        2 * cal_T * cal_dim * (cal_kv * cal_dim // cal_heads) +  # K
        2 * cal_T * cal_dim * (cal_kv * cal_dim // cal_heads) +  # V
        2 * cal_T * cal_dim * cal_dim +            # O
        0.30 * (2 * cal_T * cal_dim * cal_dim * 2 +
                2 * cal_T * cal_dim * (cal_kv * cal_dim // cal_heads) * 2) +
        2 * cal_T * cal_dim * cal_mlp +            # MLP up
        2 * cal_T * cal_mlp * cal_dim              # MLP down
    )
    # Include 5% overhead and LM head for calibration
    cal_flops_per_layer *= 1.05
    cal_lm = 2 * cal_T * cal_dim * calibration_vocab
    cal_total = 3 * (cal_layers * cal_flops_per_layer + cal_lm)

    ratio_loop_on = flops_step_loop_on / cal_total
    ratio_loop_off = flops_step_loop_off / cal_total
    # sqrt-like: wider matmuls parallelize, half the FLOP bump → wall time
    est_step_ms_loop_on = calibration_step_ms * (1.0 + 0.5 * (ratio_loop_on - 1.0))
    est_step_ms_loop_off = calibration_step_ms * (1.0 + 0.5 * (ratio_loop_off - 1.0))
    # Weighted: enable_fraction of steps are loop-off, rest are loop-on
    frac = enable_fraction if loop_layers else 1.0
    avg_step_ms = frac * est_step_ms_loop_off + (1.0 - frac) * est_step_ms_loop_on
    est_steps_600s = int(600_000 / avg_step_ms)

    # ================================================================
    # RETURN DICT (programmatic access)
    # ================================================================
    stats = {
        "params":            total_params,
        "attn_params":       attn_total,
        "mlp_params":        mlp_total,
        "other_params":      other_total,
        "artifact":          total_artifact,
        "artifact_weights":  total_artifact_weights,
        "headroom":          headroom,
        "memory_pct":        memory_pct,
        "latency_pct":       latency_pct,
        "step_ms_loop_on":   est_step_ms_loop_on,
        "step_ms_loop_off":  est_step_ms_loop_off,
        "step_ms_avg":       avg_step_ms,
        "steps_600s":        est_steps_600s,
        "flops_step_loop_on":  flops_step_loop_on,
        "flops_step_loop_off": flops_step_loop_off,
        "flops_ratio_loop_on": ratio_loop_on,
        "per_layer_evals":   per_layer_evals,
        "total_evals_loop_on": total_evals_loop_on,
    }
    return stats, {
        "attn_bytes":        attn_bytes,
        "mlp_indep_bytes":   mlp_indep_bytes,
        "mlp_shared_bytes":  mlp_shared_bytes,
        "adapt_bytes":       adapt_bytes,
        "tok_bytes":         tok_bytes,
        "legacy_other_bytes": legacy_other_bytes,
        "code_bytes":        code_bytes,
        "adapters_all_fp16": adapters_all_fp16,
        "adapt_fp16_count":  adapt_fp16_count,
        "adapt_int6_count":  adapt_int6_count,
        "max_lossless_rank": max_lossless_rank,
        "dim": dim, "heads": heads, "kv_heads": kv_heads, "mlp_dim": mlp_dim,
        "mlp_mult": mlp_mult, "n_layers": n_layers, "vocab": vocab,
        "n_bases": n_bases, "adapter_rank": adapter_rank,
        "loop_layers": loop_layers, "num_loops": num_loops,
        "parallel_start": parallel_start,
        "flops_per_layer_eval": flops_per_layer_eval,
        "flops_attn_proj": flops_attn_proj, "flops_flash": flops_flash,
        "flops_mlp": flops_mlp, "flops_overhead": flops_overhead,
        "flops_adapt": flops_adapt, "flops_lm_head": flops_lm_head,
    }


# ──────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────

def print_report(stats, extra, label=None):
    dim = extra["dim"]; heads = extra["heads"]; kv_heads = extra["kv_heads"]
    mlp_mult = extra["mlp_mult"]; mlp_dim = extra["mlp_dim"]
    n_layers = extra["n_layers"]; vocab = extra["vocab"]
    n_bases = extra["n_bases"]; adapter_rank = extra["adapter_rank"]
    loop_layers = extra["loop_layers"]; num_loops = extra["num_loops"]
    parallel_start = extra["parallel_start"]

    kv_str = f"MHA-{kv_heads}" if kv_heads == heads else f"GQA-{kv_heads}"
    adapt_str = f"rank-{adapter_rank}" if adapter_rank > 0 else "none"
    base_str = f"{n_bases} shared" if n_bases > 0 else f"{n_layers} independent"
    loop_str = (f"loop layers {loop_layers} ×{1 + num_loops}"
                if loop_layers else "no recurrence")
    par_str = f"parallel resid @ layer {parallel_start}" if parallel_start is not None else "sequential resid"

    header = f"{dim}d, {heads}H, {kv_str}, {mlp_mult}× MLP ({mlp_dim}), {n_layers}L, SP{vocab}"
    print(f"\n{'=' * 78}")
    if label:
        print(f"  [{label}]  {header}")
    else:
        print(f"  {header}")
    print(f"  MLP: {base_str}, adapters: {adapt_str}")
    print(f"  {loop_str}, {par_str}")
    print(f"{'=' * 78}")

    # Parameters
    print(f"\n  PARAMETERS")
    print(f"  {'Attention (Q+K+V+O):':<30} {stats['attn_params']:>12,} "
          f"({stats['attn_params']/1e6:.2f}M)")
    print(f"  {'MLP:':<30} {stats['mlp_params']:>12,} "
          f"({stats['mlp_params']/1e6:.2f}M)")
    print(f"  {'Other (emb + small):':<30} {stats['other_params']:>12,} "
          f"({stats['other_params']/1e6:.2f}M)")
    print(f"  {'TOTAL:':<30} {stats['params']:>12,} "
          f"({stats['params']/1e6:.2f}M)")

    # Artifact (compressed)
    print(f"\n  ARTIFACT (bytes, compressed)")
    print(f"  {'Attention:':<30} {int(extra['attn_bytes']):>12,}")
    if extra['mlp_indep_bytes'] > 0:
        print(f"  {'MLP (independent):':<30} {int(extra['mlp_indep_bytes']):>12,}")
    if extra['mlp_shared_bytes'] > 0:
        print(f"  {'MLP (shared bases):':<30} {int(extra['mlp_shared_bytes']):>12,}")
    if extra['adapt_bytes'] > 0:
        qual = "ALL fp16" if extra['adapters_all_fp16'] else \
               f"{extra['adapt_fp16_count']} fp16 + {extra['adapt_int6_count']} int6"
        print(f"  {'Adapters (' + qual + '):':<30} {int(extra['adapt_bytes']):>12,}")
    print(f"  {'Token embedding:':<30} {int(extra['tok_bytes']):>12,}")
    print(f"  {'Legacy + overhead:':<30} {int(extra['legacy_other_bytes']):>12,}")
    print(f"  {'Code:':<30} {int(extra['code_bytes']):>12,}")
    print(f"  {'TOTAL ARTIFACT:':<30} {int(stats['artifact']):>12,} "
          f"({stats['artifact']/1e6:.2f} MB)")
    print(f"  {'Headroom:':<30} {int(stats['headroom']):>12,} "
          f"({stats['headroom']/1024:+.0f} KB) "
          f"{'✓' if stats['headroom'] > 0 else '✗ OVER'}")

    # Memory % breakdown
    print(f"\n  MEMORY SHARE (% of artifact)")
    for k, v in sorted(stats['memory_pct'].items(), key=lambda kv: -kv[1]):
        bar = "█" * int(v / 2)
        print(f"  {k:<22} {v:5.1f}%  {bar}")

    # Latency % breakdown (loop-active)
    print(f"\n  LATENCY SHARE (% of step FLOPs, loop-active)")
    for k, v in sorted(stats['latency_pct'].items(), key=lambda kv: -kv[1]):
        if v < 0.05:
            continue
        bar = "█" * int(v / 2)
        print(f"  {k:<22} {v:5.1f}%  {bar}")

    # Step time
    print(f"\n  COMPUTE & STEP TIME")
    print(f"  {'Per-layer-eval (fwd):':<32} "
          f"{extra['flops_per_layer_eval']/1e9:>7.1f} GFLOP")
    print(f"  {'LM head (fwd):':<32} "
          f"{extra['flops_lm_head']/1e9:>7.1f} GFLOP")
    print(f"  {'Total evals/step (loop-on):':<32} {stats['total_evals_loop_on']:>7d}")
    print(f"  {'Step TFLOPs (fwd+bwd, loop-on):':<32} "
          f"{stats['flops_step_loop_on']/1e12:>7.2f}")
    print(f"  {'Step TFLOPs (fwd+bwd, loop-off):':<32} "
          f"{stats['flops_step_loop_off']/1e12:>7.2f}")
    print(f"  {'Step time est (loop-on):':<32} "
          f"{stats['step_ms_loop_on']:>7.1f} ms")
    print(f"  {'Step time est (loop-off):':<32} "
          f"{stats['step_ms_loop_off']:>7.1f} ms")
    print(f"  {'Step time est (avg):':<32} "
          f"{stats['step_ms_avg']:>7.1f} ms")
    print(f"  {'Steps in 600s:':<32} {stats['steps_600s']:>7,}")


def calc_and_print(preset=None, label=None, **overrides):
    """Convenience: resolve preset, apply overrides, compute, print."""
    kwargs = dict(PRESETS[preset]) if preset else {}
    kwargs.update(overrides)
    stats, extra = calc_profile(**kwargs)
    print_report(stats, extra, label=label or preset)
    return stats, extra


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Parameter Golf architecture")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="new_sota",
                        help="Baseline preset (default: new_sota = PR #1493)")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv", type=int, default=None)
    parser.add_argument("--mlp", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=None)
    parser.add_argument("--bases", type=int, default=None,
                        help="0 = independent MLP per layer")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--loop-layers", type=str, default=None,
                        help="Comma-separated physical-layer indices, e.g. '3,4,5'")
    parser.add_argument("--num-loops", type=int, default=None,
                        help="Extra visits per looped layer")
    parser.add_argument("--parallel-start", type=int, default=None,
                        help="First layer with GPT-J parallel residual")
    parser.add_argument("--all", action="store_true",
                        help="Run old_sota vs new_sota preset comparison")
    args = parser.parse_args()

    if args.all:
        # Snapshot comparison
        calc_and_print(preset="old_sota", label="OLD SOTA · PR #1019 · 1.1147 BPB")
        calc_and_print(preset="new_sota", label="NEW SOTA · PR #1493 · 1.0810 BPB")
        # A couple of MoE-exploratory variants on the new baseline
        print("\n\n" + "─" * 78)
        print("  ABLATION 1: new SOTA but no recurrence (baseline for MoE delta)")
        print("─" * 78)
        calc_and_print(preset="new_sota", label="new_sota, no-loop",
                       loop_layers=None, num_loops=0)
        print("\n\n" + "─" * 78)
        print("  ABLATION 2: new SOTA with SP4096 (shows cost of vocab doubling)")
        print("─" * 78)
        calc_and_print(preset="new_sota", label="new_sota, SP4096",
                       vocab=4096)
    else:
        overrides = {}
        if args.dim is not None:          overrides["dim"] = args.dim
        if args.heads is not None:        overrides["heads"] = args.heads
        if args.kv is not None:           overrides["kv_heads"] = args.kv
        if args.mlp is not None:          overrides["mlp_mult"] = args.mlp
        if args.layers is not None:       overrides["n_layers"] = args.layers
        if args.vocab is not None:        overrides["vocab"] = args.vocab
        if args.bases is not None:        overrides["n_bases"] = args.bases
        if args.rank is not None:         overrides["adapter_rank"] = args.rank
        if args.loop_layers is not None:
            overrides["loop_layers"] = [int(x) for x in args.loop_layers.split(",") if x]
        if args.num_loops is not None:    overrides["num_loops"] = args.num_loops
        if args.parallel_start is not None: overrides["parallel_start"] = args.parallel_start

        calc_and_print(preset=args.preset, label=args.preset, **overrides)
