"""
Recovery script: skip training, reuse `final_model.pt` written before the
brotli ImportError, run only the post-training pipeline (GPTQ + serialize +
quantized eval + sliding window + TTT). Same env vars as the main run.

Usage:
    pip install brotli            # MUST run on each rank's machine first
    torchrun --standalone --nproc_per_node=2 recover_eval.py 2>&1 | tee logs/recover_s42.log
"""
import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.distributed as dist

import train_gpt as t


def main():
    # Match exactly what main() does in train_gpt.py
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = t.Hyperparameters()
    t.set_logging_hparams(h)

    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        t.log("=" * 100, console=False)
        t.log("RECOVERY RUN — skipping training, reusing final_model.pt", console=True)
        t.log(f"  TTT_ENABLED={h.ttt_enabled}  COMPRESSOR={h.compressor}", console=True)
        t.log(f"  PARALLEL_MLP_MULT={h.parallel_mlp_mult}  GATED_ATTN_ENABLED={h.gated_attn_enabled}", console=True)
        t.log("=" * 100, console=False)

    # Same RNG seeding as train_and_eval — ensures Hessian calibration uses the
    # same data ordering that the original (interrupted) run would have used.
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)

    val_data = t.ValidationData(h, device)

    # Build a fresh model and load EMA weights from the checkpoint that was
    # saved INSIDE serialize() right before the brotli failure.
    if not os.path.exists(h.model_path):
        raise FileNotFoundError(
            f"Expected to find {h.model_path} in cwd. cwd={os.getcwd()}. "
            "Run this script from the experiment folder where the original run was launched."
        )

    base_model = t.GPT(h).to(device).bfloat16()
    t.restore_fp32_params(base_model)
    state = torch.load(h.model_path, map_location="cpu")
    # The state was saved as float32 by torch.save inside serialize. load_state_dict
    # handles dtype conversion automatically since base_model already has its
    # control tensors in fp32 via restore_fp32_params.
    base_model.load_state_dict(state, strict=True)

    if h.is_main_process:
        n_params = sum(p.numel() for p in base_model.parameters())
        t.log(f"loaded final_model.pt — params: {n_params}")

    # Now do the same flow as serialize() + post-serialize evals.
    code = open("train_gpt.py", "r", encoding="utf-8").read()
    bytes_total, quant_file_bytes = t.serialize(h, base_model, code)
    if h.is_main_process:
        t.log(f"recovery:artifact bytes_total={bytes_total} quant_blob={quant_file_bytes}")

    if h.distributed:
        dist.barrier()

    eval_model = t.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True

    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    t.timed_eval("quantized", t.eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        t.timed_eval(
            "quantized_sliding_window",
            t.eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
        )

    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = t.deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        t.timed_eval(
            "quantized_ttt",
            t.eval_val_ttt,
            h,
            device,
            val_data,
            ttt_model,
        )
        del ttt_model

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
