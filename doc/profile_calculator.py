#!/usr/bin/env python3
"""
Profile calculator for Parameter Golf architectures.
Estimates per-component compute, artifact size, and training steps.

Usage:
  python3 doc/profile_calculator.py
  python3 doc/profile_calculator.py --mlp 2.5 --kv 4 --rank 51 --bases 6
"""
import argparse
import math


def calc_profile(
    dim=512, heads=8, kv_heads=4, mlp_mult=3.0, n_layers=11,
    n_bases=0, adapter_rank=0, bigram_vocab=3072, bigram_dim=112,
    vocab=1024, ve_dim=128, seq_len=2048, batch_tokens=786432,
    world_size=8, lzma_ratio=1.70, code_bytes=115_000,
    target_step_ms=None,
):
    head_dim = dim // heads
    kv_dim = kv_heads * head_dim
    mlp_dim = int(mlp_mult * dim)
    tokens_per_gpu = batch_tokens // world_size

    # ================================================================
    # PARAMETER COUNT
    # ================================================================
    # Attention (always independent per layer)
    attn_q = n_layers * dim * dim
    attn_k = n_layers * kv_dim * dim
    attn_v = n_layers * kv_dim * dim
    attn_o = n_layers * dim * dim
    attn_total = attn_q + attn_k + attn_v + attn_o

    # MLP
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

    # Other
    tok_emb = vocab * dim
    bigram_emb = bigram_vocab * bigram_dim
    bigram_proj = dim * bigram_dim
    ve_emb = vocab * ve_dim
    ve_proj = kv_dim * ve_dim
    n_enc = n_layers // 2
    n_dec = n_layers - n_enc
    skip_w = min(n_enc, n_dec) * dim
    per_block = n_layers * (dim + dim + 2 * dim + heads)  # scales, resid_mix, q_gain
    smear = dim
    other_total = tok_emb + bigram_emb + bigram_proj + ve_emb + ve_proj + skip_w + per_block + smear

    total_params = attn_total + mlp_total + other_total

    # ================================================================
    # ARTIFACT SIZE (bytes)
    # ================================================================
    # Attention: int6 (1 byte/param + 2 bytes/row for scale)
    attn_rows = n_layers * (dim + kv_dim + kv_dim + dim)
    attn_bytes = attn_total + attn_rows * 2

    # MLP shared: int6
    if n_bases > 0:
        mlp_shared_rows = n_bases * (mlp_dim + dim)
        mlp_shared_bytes = mlp_shared + mlp_shared_rows * 2
    else:
        mlp_shared_bytes = 0

    # MLP independent: int6
    if mlp_indep > 0:
        mlp_indep_rows = n_layers * (mlp_dim + dim)
        mlp_indep_bytes = mlp_indep + mlp_indep_rows * 2
    else:
        mlp_indep_bytes = 0

    # Adapters: check fp16 vs int6 threshold
    max_lossless_rank = 65536 // mlp_dim if mlp_dim > 0 else 0
    adapters_all_fp16 = adapter_rank <= max_lossless_rank
    if adapter_rank > 0:
        adapter_matrices = [
            (mlp_dim * adapter_rank, mlp_dim),   # A_up
            (adapter_rank * dim, adapter_rank),    # B_up
            (dim * adapter_rank, dim),             # A_down
            (adapter_rank * mlp_dim, adapter_rank), # B_down
        ]
        adapt_bytes = 0
        adapt_fp16_count = 0
        adapt_int6_count = 0
        for elems, rows in adapter_matrices:
            if elems <= 65536:
                adapt_bytes += n_layers * elems * 2  # fp16
                adapt_fp16_count += 1
            else:
                adapt_bytes += n_layers * (elems + rows * 2)  # int6
                adapt_int6_count += 1
    else:
        adapt_bytes = 0
        adapt_fp16_count = 0
        adapt_int6_count = 0

    # Other: mixed fp16/fp32/int8
    tok_bytes = tok_emb + vocab * 2  # int8
    bigram_bytes = bigram_emb + bigram_vocab * 2 + bigram_proj * 2  # int8 + fp16
    ve_bytes = ve_emb + vocab * 2 + kv_dim * ve_dim * 2  # int8 + fp16
    small_bytes = per_block * 4 + skip_w * 4 + smear * 4 + 8  # fp32
    other_bytes = tok_bytes + bigram_bytes + ve_bytes + small_bytes
    overhead = 180_000

    total_before_lzma = attn_bytes + mlp_shared_bytes + mlp_indep_bytes + adapt_bytes + other_bytes + overhead
    total_after_lzma = total_before_lzma / lzma_ratio
    total_artifact = total_after_lzma + code_bytes
    headroom = 16_000_000 - total_artifact

    # ================================================================
    # COMPUTE (FLOPs per step, forward only)
    # ================================================================
    T = tokens_per_gpu

    # Per-layer attention projections
    flops_q = 2 * T * dim * dim
    flops_k = 2 * T * dim * kv_dim
    flops_v = 2 * T * dim * kv_dim
    flops_o = 2 * T * dim * dim
    flops_attn_proj = flops_q + flops_k + flops_v + flops_o

    # Flash Attention: approximate as O(T * T * H * D) but very optimized
    # Empirically ~25-35% of projection cost on H100 with Flash Attn 3
    flops_flash = 0.30 * flops_attn_proj

    # MLP
    flops_mlp_up = 2 * T * dim * mlp_dim
    flops_mlp_down = 2 * T * mlp_dim * dim
    flops_mlp = flops_mlp_up + flops_mlp_down

    # Adapter materialization (NOT per-token, just matrix multiply)
    if adapter_rank > 0:
        flops_adapt = 2 * (mlp_dim * adapter_rank * dim + dim * adapter_rank * mlp_dim)
    else:
        flops_adapt = 0

    # Norms, activations, residuals (~5% of main compute)
    flops_overhead = 0.05 * (flops_attn_proj + flops_mlp)

    flops_per_layer = flops_attn_proj + flops_flash + flops_mlp + flops_adapt + flops_overhead
    flops_fwd = n_layers * flops_per_layer
    flops_fwd_bwd = flops_fwd * 3  # backward ≈ 2x forward

    # Step time estimate
    # Calibrate from SOTA: 86.7ms at known FLOPs
    sota_flops_per_layer = (2 * T * 512 * 512 + 2 * T * 512 * 256 * 2 + 2 * T * 512 * 512 +
                            0.30 * (2 * T * 512 * 512 + 2 * T * 512 * 256 * 2 + 2 * T * 512 * 512) +
                            2 * T * 512 * 1536 + 2 * T * 1536 * 512 +
                            0.05 * (2 * T * 512 * 512 + 2 * T * 512 * 256 * 2 + 2 * T * 512 * 512 + 2 * T * 512 * 1536 + 2 * T * 1536 * 512))
    sota_total = 11 * sota_flops_per_layer * 3
    our_total = flops_fwd_bwd

    # GPU doesn't scale perfectly linearly — use sqrt model for width changes
    # (wider matmuls parallelize, so half the FLOP increase hits wall time)
    ratio = our_total / sota_total
    est_step_ms = 86.7 * (1.0 + 0.5 * (ratio - 1.0))
    est_steps_600s = int(600_000 / est_step_ms)

    # ================================================================
    # PRINT REPORT
    # ================================================================
    kv_str = f"MHA-{kv_heads}" if kv_heads == heads else f"GQA-{kv_heads}"
    adapt_str = f"rank-{adapter_rank}" if adapter_rank > 0 else "none"
    base_str = f"{n_bases} shared" if n_bases > 0 else f"{n_layers} independent"

    print(f"\n{'=' * 70}")
    print(f"  {dim}d, {heads}H, {kv_str}, {mlp_mult}x MLP ({mlp_dim}), {n_layers}L")
    print(f"  MLP: {base_str}, adapters: {adapt_str}")
    print(f"{'=' * 70}")

    print(f"\n  PARAMETERS")
    print(f"  {'Attention (Q+K+V+O):':<30} {attn_total:>12,} ({attn_total/1e6:.2f}M)")
    if mlp_shared > 0:
        print(f"  {'MLP shared bases:':<30} {mlp_shared:>12,} ({mlp_shared/1e6:.2f}M)")
    if mlp_adapt > 0:
        print(f"  {'MLP adapters:':<30} {mlp_adapt:>12,} ({mlp_adapt/1e6:.2f}M)")
    if mlp_indep > 0:
        print(f"  {'MLP independent:':<30} {mlp_indep:>12,} ({mlp_indep/1e6:.2f}M)")
    print(f"  {'Other:':<30} {other_total:>12,} ({other_total/1e6:.2f}M)")
    print(f"  {'TOTAL:':<30} {total_params:>12,} ({total_params/1e6:.2f}M)")

    print(f"\n  ARTIFACT (bytes)")
    print(f"  {'Attention (int6):':<30} {attn_bytes:>12,}")
    if mlp_shared_bytes > 0:
        print(f"  {'MLP shared (int6):':<30} {mlp_shared_bytes:>12,}")
    if mlp_indep_bytes > 0:
        print(f"  {'MLP independent (int6):':<30} {mlp_indep_bytes:>12,}")
    if adapt_bytes > 0:
        qual = "ALL fp16" if adapters_all_fp16 else f"{adapt_fp16_count} fp16 + {adapt_int6_count} int6"
        print(f"  {'Adapters (' + qual + '):':<30} {adapt_bytes:>12,}")
    if adapter_rank > 0:
        print(f"  {'Max lossless rank:':<30} {max_lossless_rank:>12}")
    print(f"  {'Other:':<30} {other_bytes:>12,}")
    print(f"  {'Overhead:':<30} {overhead:>12,}")
    print(f"  {'Before LZMA:':<30} {total_before_lzma:>12,} ({total_before_lzma/1e6:.2f}M)")
    print(f"  {'After LZMA (÷{lzma_ratio}):':<30} {total_after_lzma:>12,.0f} ({total_after_lzma/1e6:.2f}M)")
    print(f"  {'+ Code:':<30} {code_bytes:>12,}")
    print(f"  {'TOTAL ARTIFACT:':<30} {total_artifact:>12,.0f} ({total_artifact/1e6:.2f}M)")
    print(f"  {'Headroom:':<30} {headroom:>12,.0f} ({headroom/1024:.0f} KB) {'✓' if headroom > 0 else '✗ OVER'}")

    print(f"\n  COMPUTE (per step, {world_size} GPUs)")
    print(f"  Per layer (forward):")
    print(f"    {'Attn projections (Q+K+V+O):':<32} {n_layers*flops_attn_proj/1e9:>8.1f} GFLOP ({flops_attn_proj/flops_per_layer*100:.1f}%)")
    print(f"    {'Flash Attention:':<32} {n_layers*flops_flash/1e9:>8.1f} GFLOP ({flops_flash/flops_per_layer*100:.1f}%)")
    print(f"    {'MLP (up + down):':<32} {n_layers*flops_mlp/1e9:>8.1f} GFLOP ({flops_mlp/flops_per_layer*100:.1f}%)")
    if flops_adapt > 0:
        print(f"    {'Adapter A@B:':<32} {n_layers*flops_adapt/1e9:>8.1f} GFLOP ({flops_adapt/flops_per_layer*100:.1f}%)")
    print(f"    {'Overhead (norms, etc):':<32} {n_layers*flops_overhead/1e9:>8.1f} GFLOP ({flops_overhead/flops_per_layer*100:.1f}%)")
    print(f"  {'Total (fwd+bwd):':<34} {flops_fwd_bwd/1e12:>8.2f} TFLOP")
    print(f"  {'vs SOTA ratio:':<34} {ratio:>8.2f}x")

    print(f"\n  TRAINING ESTIMATE ({world_size}x H100, 600s)")
    print(f"  {'Est step time:':<30} {est_step_ms:>8.1f} ms")
    print(f"  {'Est steps in 600s:':<30} {est_steps_600s:>8,}")
    print(f"  {'SOTA steps:':<30} {6927:>8,}")
    print(f"  {'Step difference:':<30} {est_steps_600s - 6927:>+8,} ({(est_steps_600s-6927)/6927*100:+.1f}%)")

    return {
        'params': total_params, 'artifact': total_artifact, 'headroom': headroom,
        'step_ms': est_step_ms, 'steps': est_steps_600s, 'flops_ratio': ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Parameter Golf architecture")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv", type=int, default=4)
    parser.add_argument("--mlp", type=float, default=3.0)
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--bases", type=int, default=0, help="0 = independent MLP per layer")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--lzma", type=float, default=1.70)
    parser.add_argument("--all", action="store_true", help="Run all key configs")
    args = parser.parse_args()

    if args.all:
        configs = [
            dict(kv_heads=4, mlp_mult=3.0, n_bases=0, adapter_rank=0),    # SOTA
            dict(kv_heads=8, mlp_mult=4.0, n_bases=6, adapter_rank=0),    # MHA+4x pure
            dict(kv_heads=4, mlp_mult=5.0, n_bases=6, adapter_rank=0),    # GQA+5x pure
            dict(kv_heads=4, mlp_mult=2.5, n_bases=6, adapter_rank=51),   # GQA+2.5x r51
            dict(kv_heads=4, mlp_mult=3.0, n_bases=6, adapter_rank=42),   # GQA+3x r42
            dict(kv_heads=8, mlp_mult=3.5, n_bases=3, adapter_rank=100),  # MHA+3.5x r100
        ]
        for cfg in configs:
            calc_profile(**cfg)
    else:
        calc_profile(
            dim=args.dim, heads=args.heads, kv_heads=args.kv,
            mlp_mult=args.mlp, n_layers=args.layers,
            n_bases=args.bases, adapter_rank=args.rank,
            lzma_ratio=args.lzma,
        )