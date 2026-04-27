"""
Fat-Block experimental trainer for Parameter Golf.

Replaces the 4 parallel-residual blocks (layers 7-10) of PR #1493 with a single
"fat block" that runs 4 sequential attention sublayers IN PARALLEL with one
big shared MLP. Optional attention nonlinearities:
  - Gated Attention (Qwen NeurIPS 2025, G1 variant): sigmoid gate on SDPA output
  - GLU on V projection (arXiv 2507.00022, July 2025): V = silu(W_v1 x) * (W_v2 x)

Base: decompressed source of
records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py

This file is NOT LZMA-wrapped (unlike the submission-format trainer). It's for
local experimentation on 2x H100 over 40 min wall-clock. Launch via torchrun
(see README.md for commands).
"""
import collections, copy, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func


class Hyperparameters:
    # ------------------------------------------------------------------
    # Standard hyperparameters (unchanged from PR #1493 defaults)
    # ------------------------------------------------------------------
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', .72))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 6e2))
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 3e1))
    rope_base = float(os.environ.get('ROPE_BASE', 1e4))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.))
    num_loops = int(os.environ.get('NUM_LOOPS', 2))
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 5))
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', .35))
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))
    min_lr = float(os.environ.get('MIN_LR', .0))
    embed_lr = float(os.environ.get('EMBED_LR', .6))
    head_lr = float(os.environ.get('HEAD_LR', .008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', .03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', .005))
    matrix_lr = float(os.environ.get('MATRIX_LR', .022))
    scalar_lr = float(os.environ.get('SCALAR_LR', .02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', .99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', .92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', .9))
    beta2 = float(os.environ.get('BETA2', .95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-08))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', .3))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', .95))
    adam_wd = float(os.environ.get('ADAM_WD', .02))
    muon_wd = float(os.environ.get('MUON_WD', .095))
    embed_wd = float(os.environ.get('EMBED_WD', .085))
    ema_decay = float(os.environ.get('EMA_DECAY', .9965))
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '0')))
    ttt_lr = float(os.environ.get('TTT_LR', .005))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', .9))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
    etlb_enabled = bool(int(os.environ.get('ETLB_ENABLED', '0')))
    etlb_lr = float(os.environ.get('ETLB_LR', .05))
    etlb_steps = int(os.environ.get('ETLB_STEPS', 5))
    etlb_clip = float(os.environ.get('ETLB_CLIP', 3.))
    compressor = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 2e1))

    # ------------------------------------------------------------------
    # NEW: fat-block architecture hyperparameters
    # ------------------------------------------------------------------
    # If enabled, physical blocks 0..parallel_residual_start-1 (i.e., 0..6) are
    # regular Block instances and layers 7..num_layers-1 are replaced by a
    # single FatBlock containing fat_block_num_attns sequential attentions + 1
    # big shared MLP running in parallel with the attention chain.
    fat_block_enabled = bool(int(os.environ.get('FAT_BLOCK_ENABLED', '1')))
    fat_block_num_attns = int(os.environ.get('FAT_BLOCK_NUM_ATTNS', 4))
    fat_block_mlp_hidden = int(os.environ.get('FAT_BLOCK_MLP_HIDDEN', 6144))
    # How to handle U-Net skip connections that used to land on old layers
    # 7..9 but now have no decoder counterpart:
    #   'aggregate' - sum the skip-gated inputs into the FatBlock input
    #   'drop'      - silently ignore those skips
    fat_block_skip_mode = os.environ.get('FAT_BLOCK_SKIP_MODE', 'aggregate')

    # ------------------------------------------------------------------
    # NEW: attention nonlinearity variants (applied inside FatBlock's attentions only)
    # ------------------------------------------------------------------
    # Qwen NeurIPS 2025 "Gated Attention" (G1): sigmoid gate on SDPA output,
    # before W_o. headwise uses a [dim, n_heads] gate; elementwise uses [dim, dim].
    gated_attn_enabled = bool(int(os.environ.get('GATED_ATTN', '0')))
    gated_attn_mode = os.environ.get('GATED_ATTN_MODE', 'headwise')  # headwise | elementwise
    # GLU Attention (arXiv 2507.00022): V = silu(W_v1 x) * (W_v2 x).
    # Adds per-token nonlinearity before attention mixes V linearly.
    glu_v_enabled = bool(int(os.environ.get('GLU_V', '0')))

    # ------------------------------------------------------------------
    # Architectural explorations (A1 / A2 / A3 in the backlog)
    # ------------------------------------------------------------------
    # A1 — widen the attentions inside the fat block. When > 0, overrides
    # head_dim JUST for FatBlock's attentions (regular blocks 0..6 stay at
    # model_dim // num_heads = 64). E.g. FAT_ATTN_HEAD_DIM=96 makes each of
    # the fat block's 4 attentions internally 8 heads x 96 dim = 768 attn_dim
    # before projecting back to model_dim=512. If 0 (default), use the
    # standard head_dim = model_dim / num_heads.
    fat_attn_head_dim = int(os.environ.get('FAT_ATTN_HEAD_DIM', '0'))
    # A1 — toggle whether the fat block has its big MLP at all. When 0, the
    # fat block is 4 attentions with no MLP; the residual-stream MLP work
    # is carried entirely by the (wider) attentions.
    fat_block_mlp_enabled = bool(int(os.environ.get('FAT_BLOCK_MLP_ENABLED', '1')))
    # A2 — change where the big MLP reads from. 'parallel' (default) reads
    # the fat-block input x_in (GPT-J style). 'sequential' reads the
    # post-attention-chain state z (standard transformer style).
    fat_block_mlp_mode = os.environ.get('FAT_BLOCK_MLP_MODE', 'parallel')
    # A3 — optional per-token nonlinearity applied to the SDPA output,
    # before the optional gate and the output projection.
    #   'none'          (default)  — no activation
    #   'leaky_relu_sq' — y = leaky_relu(y, 0.5).square()  (matches MLP style)
    attn_output_activation = os.environ.get('ATTN_OUTPUT_ACTIVATION', 'none')

    # ------------------------------------------------------------------
    # NEW: looped-section sharing & specialization (V1 / V2 in the backlog)
    # ------------------------------------------------------------------
    # Apply only to the looped blocks loop_start..loop_end (default 3..5).
    # Both flags require fat_block_enabled=0 (work on top of SOTA-equivalent base).
    #
    # V2 — share one MLP across blocks loop_start..loop_end. Replaces the
    # 3 separate per-block MLPs with one shared MLP module of hidden size
    # LOOP_SHARED_MLP_HIDDEN. Default 6144 makes total looped-MLP params
    # equal to SOTA's 3 × h=2048 = 6144 effective hidden.
    loop_shared_mlp = bool(int(os.environ.get('LOOP_SHARED_MLP', '0')))
    loop_shared_mlp_hidden = int(os.environ.get('LOOP_SHARED_MLP_HIDDEN', '6144'))
    # V1 — give each looped block its own per-visit attention modules.
    # With num_loops=2, each looped block is visited 3 times per forward
    # pass; this flag replaces the single shared attention with a
    # ModuleList of (num_loops + 1) unique attentions, dispatched by
    # visit_count in Block.forward.
    loop_unique_attn = bool(int(os.environ.get('LOOP_UNIQUE_ATTN', '0')))

    # ------------------------------------------------------------------
    # Distributed / derived settings (unchanged from PR #1493)
    # ------------------------------------------------------------------
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, 'datasets', f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f"fineweb_{vocab_size}_bpe.model")
    logfile = f"logs/{run_id}.txt"
    model_path = 'final_model.pt'
    quantized_model_path = 'final_model.int6.ptz'


_logger_hparams = None
def set_logging_hparams(h): global _logger_hparams; _logger_hparams = h
def log(msg, console=True):
    if _logger_hparams is None: print(msg); return
    if _logger_hparams.is_main_process:
        if console: print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f: print(msg, file=f)


# ======================================================================
# Data loading (unchanged from PR #1493)
# ======================================================================
class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size: raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id('▁') != sp.unk_id(), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith('▁'): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0: raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[:usable + 1]


def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    token_bytes = np.dtype('<u2').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens: raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}
def _read_num_tokens(file):
    key = str(file); cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None: return cached
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2]); _SHARD_NTOKENS_CACHE[key] = n; return n

def _get_shard_memmap(file):
    key = str(file); mm = _MMAP_CACHE.get(key)
    if mm is not None: return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode='r', dtype='<u2', offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm; return mm


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size; self.seq_len = h.train_seq_len; self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files: raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)): self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)): self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64); total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop(); remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]; y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ======================================================================
# Core modules (RMSNorm, CastedLinear, Rotary) — unchanged
# ======================================================================
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1. / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1. / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


# ======================================================================
# CausalSelfAttention — MODIFIED to support gated attention (Qwen G1)
#                       and GLU on V (arXiv 2507.00022)
# ======================================================================
class CausalSelfAttention(nn.Module):
    """
    Standard GQA causal self-attention with Partial RoPE + QK-Norm + learnable q_gain.
    Optional nonlinearity extensions (opt-in per attention instance):

      gated_mode in {None, 'headwise', 'elementwise'}:
        Applies sigmoid gate to SDPA output before W_o (Qwen G1):
          gate = sigmoid(W_g @ x)
          y = y * gate
          y = W_o @ y
        'headwise':    W_g: [dim -> num_heads]    (broadcast gate over head_dim)
        'elementwise': W_g: [dim -> attn_dim]     (per-feature gate, ~60x more params)

      glu_v: bool
        Replaces V projection with SwiGLU-style gated value:
          V = silu(W_v1 @ x) * (W_v2 @ x)
        FA3 receives the gated V. Compatible with GQA.

      head_dim: int | None
        Override the default head_dim = dim / num_heads. When provided, the
        attention operates internally at num_heads * head_dim (the "attn_dim"),
        with Q projecting dim -> attn_dim, K/V projecting dim -> kv_dim, and
        the output projection going attn_dim -> dim to match the residual
        stream. Used to widen attention capacity in the fat block without
        changing the model_dim of the residual stream.

      attn_output_activation in {'none', 'leaky_relu_sq'}:
        Optional per-token nonlinearity on SDPA output, applied BEFORE the
        gate and output projection. 'leaky_relu_sq' matches the MLP activation
        style (`leaky_relu(y, 0.5).square()`).
    """
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
                 gated_mode=None, glu_v=False,
                 head_dim=None, attn_output_activation='none'):
        super().__init__()
        if num_heads % num_kv_heads != 0: raise ValueError('num_heads must be divisible by num_kv_heads')
        # Resolve head_dim (defaults to dim / num_heads, i.e. attn_dim == dim).
        if head_dim is None or head_dim <= 0:
            if dim % num_heads != 0: raise ValueError('model_dim must be divisible by num_heads when head_dim is not provided')
            head_dim = dim // num_heads
        if head_dim % 2 != 0: raise ValueError('head_dim must be even for RoPE')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim           # internal attention dim (pre-proj)
        kv_dim = num_kv_heads * head_dim
        # Q projects into the internal attention dim; K/V into the grouped-kv dim.
        self.c_q = CastedLinear(dim, self.attn_dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.glu_v = glu_v
        if glu_v:
            self.c_v1 = CastedLinear(dim, kv_dim, bias=False)
            self.c_v2 = CastedLinear(dim, kv_dim, bias=False)
            self.c_v = None
        else:
            self.c_v = CastedLinear(dim, kv_dim, bias=False)
            self.c_v1 = self.c_v2 = None
        # Output projection always brings the attention output back to the residual-stream dim.
        self.proj = CastedLinear(self.attn_dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

        # Qwen G1 gated attention. Gate is applied on the SDPA output (pre-proj),
        # which has shape [B, T, attn_dim]. For 'elementwise' mode the gate
        # projection must therefore map `dim -> attn_dim` (not `dim -> dim`).
        self.gated_mode = gated_mode
        if gated_mode == 'headwise':
            self.c_g = CastedLinear(dim, num_heads, bias=False)
        elif gated_mode == 'elementwise':
            self.c_g = CastedLinear(dim, self.attn_dim, bias=False)
        else:
            self.c_g = None

        # A3: optional per-token activation on SDPA output (before gate + proj).
        if attn_output_activation not in ('none', 'leaky_relu_sq'):
            raise ValueError(f"Unknown attn_output_activation: {attn_output_activation!r}")
        self.attn_output_activation = attn_output_activation

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        # V (optionally GLU'd)
        if self.glu_v:
            v = F.silu(self.c_v1(x)) * self.c_v2(x)
        else:
            v = self.c_v(x)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        # QK-Norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        # Learnable per-head Q gain
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # SDPA via FA3
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        # A3: optional per-token activation on SDPA output (before gate + proj)
        if self.attn_output_activation == 'leaky_relu_sq':
            y = F.leaky_relu(y, negative_slope=0.5).square()
        # Qwen G1 gated attention: sigmoid gate on SDPA output, BEFORE W_o
        if self.gated_mode == 'headwise':
            # gate shape: [B, T, num_heads], broadcast over head_dim
            gate = torch.sigmoid(self.c_g(x)).to(dtype=y.dtype)
            y = y * gate.unsqueeze(-1)
        elif self.gated_mode == 'elementwise':
            # gate shape: [B, T, attn_dim]
            gate = torch.sigmoid(self.c_g(x)).to(dtype=y.dtype)
            y = y.reshape(bsz, seqlen, self.attn_dim) * gate
            y = y.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        # Output projection (attn_dim -> model_dim)
        y = y.reshape(bsz, seqlen, self.attn_dim)
        return self.proj(y)


# ======================================================================
# MLP — unchanged
# ======================================================================
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult=None, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=.5).square())


# ======================================================================
# Block — handles sequential & parallel residual modes; optional per-visit attentions
# ======================================================================
class Block(nn.Module):
    # NOTE: The A3 `ATTN_OUTPUT_ACTIVATION` flag is intentionally NOT forwarded here —
    # it applies to the FatBlock's attentions only (see FatBlock.__init__). Regular
    # blocks (layers 0-6) already have an MLP after their attention providing
    # per-token nonlinearity, so the activation would be redundant there and it
    # would also confound the architectural ablation against gated_ew.
    #
    # `num_visits` (V1 in the loop-share backlog): if > 1, replaces the single shared
    # attention with a ModuleList of `num_visits` independent attentions. Block.forward
    # accepts a `visit_count` arg dispatched by GPT.forward_logits to select which
    # per-visit attention to run. attn_norm stays single (LN is just a normalizer).
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 train_seq_len, layer_idx=0, ln_scale=False, num_visits=1):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        if num_visits > 1:
            self.attn = nn.ModuleList([
                CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
                for _ in range(num_visits)
            ])
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult=mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1. / math.sqrt(layer_idx + 1) if ln_scale else 1.
        self.parallel = False
        self.num_visits = num_visits

    def forward(self, x, x0, visit_count=0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_module = self.attn[visit_count] if isinstance(self.attn, nn.ModuleList) else self.attn
        attn_out = attn_module(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


# ======================================================================
# FatBlock — NEW: 4 sequential attentions in parallel with 1 big MLP
# ======================================================================
class FatBlock(nn.Module):
    """
    Replaces the 4 parallel-residual blocks (old layers 7-10) with a single
    block that has:

      - num_attns sequential attention sublayers (each refines the previous
        via its own residual + pre-norm). Attention depth is preserved.
      - One big MLP reading the fat-block INPUT (x_in), running in parallel
        with the attention chain. Both paths sum into the residual.

    Structure:

          x_in (after resid_mix + optional aggregated skip)
              │
              ├─► attn₁ ─► + (residual) ─► attn₂ ─► + ─► ... ─► attn_N ─► + ─► z
              │
              └─► big_MLP(norm(x_in)) ─► mlp_out
                         │
              x_out = z + mlp_scale * mlp_out

    Motivation: PR #1204 learned-routing data showed mlp_to_attn ≈ 0 and
    mlp_to_mlp very weak in deep layers, while attn_to_attn is the strongest
    signal (layer 10). Fat block keeps attention depth, drops MLP duplication,
    reinvests the artifact savings in a bigger single MLP.
    """
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 train_seq_len, layer_idx=0, ln_scale=False,
                 num_attns=4, mlp_hidden=6144,
                 gated_mode=None, glu_v=False,
                 fat_head_dim=None,          # A1: override head_dim for attentions in this block
                 mlp_enabled=True,           # A1: toggle the big MLP entirely
                 mlp_mode='parallel',        # A2: 'parallel' (reads x_in) or 'sequential' (reads z)
                 attn_output_activation='none'):  # A3: forwarded to CausalSelfAttention
        super().__init__()
        if mlp_mode not in ('parallel', 'sequential'):
            raise ValueError(f"fat_block mlp_mode must be 'parallel' or 'sequential', got {mlp_mode!r}")
        self.num_attns = num_attns
        self.mlp_enabled = bool(mlp_enabled)
        self.mlp_mode = mlp_mode
        # One pre-norm per attention in the chain
        self.attn_norms = nn.ModuleList([RMSNorm() for _ in range(num_attns)])
        # num_attns sequential attentions, each optionally widened/gated/GLU-V'd/activated
        self.attns = nn.ModuleList([
            CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
                                gated_mode=gated_mode, glu_v=glu_v,
                                head_dim=fat_head_dim,
                                attn_output_activation=attn_output_activation)
            for _ in range(num_attns)
        ])
        # Optional big MLP path. When disabled, the fat block is attentions-only.
        if self.mlp_enabled:
            self.mlp_norm = RMSNorm()
            self.mlp = MLP(dim, hidden=mlp_hidden)
            self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        # Per-sublayer attention residual scales
        self.attn_scales = nn.Parameter(torch.ones(num_attns, dim, dtype=torch.float32))
        # Residual mix (same convention as Block): blend with x0 (initial embedding)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # LN scale factor (uses the layer_idx of the FIRST attention in the chain for consistency)
        self.ln_scale_factor = 1. / math.sqrt(layer_idx + 1) if ln_scale else 1.
        # Note: with the balanced encoder/decoder index layout in GPT
        # (min(len(enc), len(dec)) skip weights, one per decoder position),
        # the fat block always gets a regular skip via GPT.skip_weights/skip_gates.
        # We don't create a separate aggregated_skip_gate here — that would be
        # a dead parameter under the current recurrence config and would trip
        # DDP's unused-parameter check.

    def forward(self, x, x0, visit_count=0):
        # visit_count accepted for signature compatibility with Block (so that
        # GPT.forward_logits can pass it uniformly); FatBlock has no per-visit
        # specialization and the value is ignored. (Avoiding `del` to keep
        # the bytecode dynamo-friendly under fullgraph compilation.)
        _ = visit_count  # noqa: F841 — intentionally unused
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # --- Attention chain: sequential refinement ---
        z = x_in
        for i in range(self.num_attns):
            a = self.attns[i](self.attn_norms[i](z) * self.ln_scale_factor)
            scale_i = self.attn_scales[i].to(dtype=z.dtype)[None, None, :]
            z = z + scale_i * a
        # --- Optional big MLP path ---
        if self.mlp_enabled:
            # A2: 'parallel' reads fat-block input x_in (GPT-J style),
            #     'sequential' reads post-attention-chain state z (standard transformer).
            mlp_input = z if self.mlp_mode == 'sequential' else x_in
            mlp_out = self.mlp(self.mlp_norm(mlp_input) * self.ln_scale_factor)
            return z + self.mlp_scale.to(dtype=z.dtype)[None, None, :] * mlp_out
        # Attentions-only fat block: residual stream goes through just the attn chain.
        return z


# ======================================================================
# GPT — MODIFIED __init__ and forward_logits to handle FatBlock
# ======================================================================
class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= .0: raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None; self.head_proj = None

        self.fat_block_enabled = h.fat_block_enabled
        if h.fat_block_enabled:
            # Build 0..parallel_residual_start-1 regular blocks + 1 FatBlock at index
            # `parallel_residual_start`. We DROP the remaining physical blocks
            # (old layers parallel_residual_start+1 .. num_layers-1) in favor of the
            # fat block's internal attention chain.
            regular_block_count = h.parallel_residual_start  # e.g. 7
            blocks = [
                Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                      h.rope_base, h.qk_gain_init, h.train_seq_len,
                      layer_idx=i, ln_scale=h.ln_scale)
                for i in range(regular_block_count)
            ]
            gated_mode = h.gated_attn_mode if h.gated_attn_enabled else None
            # FAT_ATTN_HEAD_DIM=0 means "use default head_dim" (same as model_dim / num_heads)
            fat_head_dim = h.fat_attn_head_dim if h.fat_attn_head_dim > 0 else None
            fat = FatBlock(
                h.model_dim, h.num_heads, h.num_kv_heads,
                h.rope_base, h.qk_gain_init, h.train_seq_len,
                layer_idx=regular_block_count, ln_scale=h.ln_scale,
                num_attns=h.fat_block_num_attns,
                mlp_hidden=h.fat_block_mlp_hidden,
                gated_mode=gated_mode,
                glu_v=h.glu_v_enabled,
                fat_head_dim=fat_head_dim,
                mlp_enabled=h.fat_block_mlp_enabled,
                mlp_mode=h.fat_block_mlp_mode,
                attn_output_activation=h.attn_output_activation,
            )
            blocks.append(fat)
            self.blocks = nn.ModuleList(blocks)
            self.fat_block_idx = regular_block_count  # the physical index of the fat block
            # Effective num_layers (for encoder/decoder index math) = regular_count + 1
            effective_num_layers = regular_block_count + 1
        else:
            # Fallback: original PR #1493 architecture
            self.blocks = nn.ModuleList([
                Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                      h.rope_base, h.qk_gain_init, h.train_seq_len,
                      layer_idx=i, ln_scale=h.ln_scale)
                for i in range(h.num_layers)
            ])
            self.fat_block_idx = None
            effective_num_layers = h.num_layers
        # encoder/decoder split uses the EFFECTIVE physical layer count
        self.num_encoder_layers = effective_num_layers // 2
        self.num_decoder_layers = effective_num_layers - self.num_encoder_layers

        # ------------------------------------------------------------------
        # Loop-share modifications (V1 / V2). Run BEFORE rope_dims/XSA init
        # so those loops (patched below) iterate over the FINAL block list.
        # Both flags only apply when fat_block is OFF (work on top of SOTA-equivalent).
        # ------------------------------------------------------------------
        if h.loop_unique_attn and not h.fat_block_enabled:
            # Each looped block (loop_start..loop_end) is visited (num_loops + 1)
            # times per forward pass. Replace its single attention with a
            # ModuleList of (num_loops + 1) independent attentions, dispatched
            # by visit_count in Block.forward.
            num_visits = h.num_loops + 1
            for i in range(h.loop_start, h.loop_end + 1):
                if i < len(self.blocks):
                    self.blocks[i] = Block(
                        h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                        h.rope_base, h.qk_gain_init, h.train_seq_len,
                        layer_idx=i, ln_scale=h.ln_scale, num_visits=num_visits,
                    )
            # parallel flag is set by the parallel_residual loop below, which
            # iterates self.blocks again — the new block will get the right flag.

        if h.loop_shared_mlp and not h.fat_block_enabled:
            # Build one shared MLP and assign to blocks loop_start..loop_end.
            # PyTorch's named_parameters/named_modules dedup by default, so the
            # optimizer / Hessian collection / state_dict serialization see one
            # canonical instance. The state_dict will emit alias keys for each
            # block path; gptq_mixed_quantize handles those via data_ptr dedup.
            shared_mlp = MLP(h.model_dim, hidden=h.loop_shared_mlp_hidden)
            for i in range(h.loop_start, h.loop_end + 1):
                if i < len(self.blocks):
                    self.blocks[i].mlp = shared_mlp

        if h.rope_dims > 0:
            # Use each attention's own head_dim when rebuilding the Rotary cache.
            # FatBlock attentions may have been widened via FAT_ATTN_HEAD_DIM (A1),
            # so hard-coding `h.model_dim // h.num_heads` would mis-size the
            # rotary frequencies for those attentions. Per-visit attentions
            # (V1 LOOP_UNIQUE_ATTN) are stored as a ModuleList; iterate them too.
            for block in self.blocks:
                if isinstance(block, FatBlock):
                    for a in block.attns:
                        a.rope_dims = h.rope_dims
                        a.rotary = Rotary(a.head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
                else:
                    attns_to_init = block.attn if isinstance(block.attn, nn.ModuleList) else [block.attn]
                    for a in attns_to_init:
                        a.rope_dims = h.rope_dims
                        a.rotary = Rotary(a.head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)

        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True

        # XSA on the last xsa_last_n layers
        if h.xsa_last_n > 0:
            num_physical = len(self.blocks)
            for i in range(max(0, num_physical - h.xsa_last_n), num_physical):
                block = self.blocks[i]
                if isinstance(block, FatBlock):
                    for a in block.attns: a.use_xsa = True
                else:
                    attns_to_init = block.attn if isinstance(block.attn, nn.ModuleList) else [block.attn]
                    for a in attns_to_init: a.use_xsa = True

        # Parallel residual applies to non-fat-block layers from parallel_residual_start.
        # In fat-block mode, the fat block REPLACES this region, so there are no
        # parallel-residual Block instances to flag (blocks 0..parallel_residual_start-1
        # remain sequential).
        if h.parallel_residual_start >= 0 and not h.fat_block_enabled:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True

        # ------------------------------------------------------------------
        # Encoder / decoder virtual sequence (with optional recurrence)
        # ------------------------------------------------------------------
        # When fat-block is ON, "layer indices" refer to physical indices in
        # self.blocks (range effective_num_layers=8). The fat block sits at index
        # parallel_residual_start (=7). All recurrence math uses physical indices.
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, effective_num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            num_enc = effective_num_layers // 2
            self.encoder_indices = list(range(num_enc))
            self.decoder_indices = list(range(num_enc, effective_num_layers))

        # Precompute per-position visit counts. For each position in the
        # combined encoder+decoder index list, how many times has the
        # corresponding physical block been entered prior (=visit slot).
        # Block.forward dispatches per-visit attentions (V1) using these.
        # Non-looped blocks get visit 0 always.
        combined = list(self.encoder_indices) + list(self.decoder_indices)
        _counts = {}
        _visits = []
        for _idx in combined:
            _vc = _counts.get(_idx, 0)
            _visits.append(_vc)
            _counts[_idx] = _vc + 1
        self.encoder_visits = _visits[:len(self.encoder_indices)]
        self.decoder_visits = _visits[len(self.encoder_indices):]
        # Non-looping path: each block visited exactly once → vc=0 everywhere.
        self.encoder_visits_flat = [0] * self.num_encoder_layers
        self.decoder_visits_flat = [0] * self.num_decoder_layers

        # Skip weights: one per skip-pair between encoder and decoder
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=.0, std=self.tied_embed_init_std)
        for (name, module) in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.)

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None: x = self.embed_proj(x)
        x0 = x
        skips = []
        num_physical = len(self.blocks)
        if self.looping_active:
            enc_iter = self.encoder_indices
            dec_iter = self.decoder_indices
            enc_visits = self.encoder_visits
            dec_visits = self.decoder_visits
        else:
            enc_iter = list(range(self.num_encoder_layers))
            dec_iter = list(range(self.num_encoder_layers, num_physical))
            enc_visits = self.encoder_visits_flat
            dec_visits = self.decoder_visits_flat
        # Encoder
        for (i, vc) in zip(enc_iter, enc_visits):
            x = self.blocks[i](x, x0, visit_count=vc)
            skips.append(x)
        # Decoder with skip connections (identical logic for Block and FatBlock —
        # both take (x, x0) and return x; FatBlock just happens to contain more
        # sublayers internally).
        for (skip_idx, (i, vc)) in enumerate(zip(dec_iter, dec_visits)):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0, visit_count=vc)
        x = self.final_norm(x)
        if self.head_proj is not None: x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction='mean')


# ======================================================================
# Optimizer / training / eval / serialization — copied verbatim from PR #1493
# ======================================================================
def classify_param(name):
    if 'tok_emb' in name or 'lm_head' in name: return 'embed'
    if '.mlp.' in name: return 'mlp'
    if '.attn.' in name or '.attns.' in name or ('.proj.' in name and '.mlp.' not in name): return 'attn'
    return 'other'


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group['params']
            if not params: continue
            lr = group['lr']; momentum = group['momentum']; backend_steps = group['backend_steps']; nesterov = group['nesterov']
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for (i, p) in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    if group.get('row_normalize', False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07); g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** .5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get('weight_decay', .0); curr = 0
            for p in params:
                if wd > .0: p.data.mul_(1. - lr * wd)
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype); p.add_(g, alpha=-lr); curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(pattern for pattern in os.environ.get(
    'CONTROL_TENSOR_NAME_PATTERNS',
    'attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates'
).split(',') if pattern)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [p for (name, p) in block_named_params
                         if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
        scalar_params = [p for (name, p) in block_named_params
                         if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
        if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0: scalar_params.append(base_model.skip_gates)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{'params': [base_model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}]
        self.optimizer_tok = torch.optim.AdamW(tok_params, betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups: group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam([{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self): return iter(self.optimizers)
    def zero_grad_all(self):
        for opt in self.optimizers: opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.optimizers: opt.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear): module.float()
    for (name, param) in model.named_parameters():
        if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}; hooks = []
    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return hook_fn
    for (name, module) in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + '.weight')
            if cat in ('mlp', 'attn'):
                hooks.append(module.register_forward_hook(make_hook(name + '.weight')))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm
        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks: hook.remove()
    for name in hessians: hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_quantize_weight(w, H, clip_sigmas=3., clip_range=63, block_size=128):
    W_orig = w.float().clone(); rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1
    damp = .01 * H.diag().mean(); H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True); invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone(); W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]; d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols: W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}; meta = {}
    # Shared modules (e.g., LOOP_SHARED_MLP) cause state_dict to emit multiple
    # alias keys (one per parent path) all referring to the same tensor. But
    # `hessians` only has the canonical key (since named_modules deduplicates).
    # Track tensors by their underlying storage pointer so alias keys reuse the
    # already-computed q/scale instead of looking up a missing hessian entry.
    seen_ptrs = {}  # data_ptr -> canonical state_dict key
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            canonical = seen_ptrs[ptr]
            result[name + '.q'] = result[canonical + '.q']
            result[name + '.scale'] = result[canonical + '.scale']
            meta[name] = f"shared_with:{canonical}"
            continue
        seen_ptrs[ptr] = name
        cs = h.embed_clip_sigmas if 'tok_emb' in name else h.matrix_clip_sigmas
        bits = h.embed_bits if 'tok_emb' in name else h.matrix_bits
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1)
        result[name + '.q'] = q
        result[name + '.scale'] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub('\\.\\d+$', '', re.sub('blocks\\.\\d+', 'blocks', name))
        categories[cat].add(short)
    log('Quantized weights:')
    for cat in sorted(categories): log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if 'passthrough' in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16): t = t.to(orig_dtype)
            out[name] = t; continue
        q, s = result[name + '.q'], result[name + '.scale']
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b'BSHF'
def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8); n = len(src)
    out = np.empty(n, dtype=np.uint8); dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC: return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5); n = len(payload)
    out = np.empty(n, dtype=np.uint8); src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == 'lzma': return lzma.compress(data, preset=6)
    elif compressor == 'brotli': import brotli; return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")

def _decompress(data, compressor):
    if compressor == 'lzma': raw = lzma.decompress(data)
    elif compressor == 'brotli': import brotli; raw = brotli.decompress(data)
    else: raise ValueError(f"Unknown compressor: {compressor!r}")
    raw = _byte_unshuffle(raw); return raw


def serialize(h, base_model, code):
    code_bytes = len(code.encode('utf-8'))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
    # Build sd_cpu while preserving shared-parameter aliasing. PyTorch's state_dict
    # emits multiple alias keys for shared modules (LOOP_SHARED_MLP) — all pointing
    # at the same GPU tensor. A naive `.cpu()` per-key creates separate CPU buffers,
    # which breaks the downstream data_ptr-based alias dedup in gptq_mixed_quantize
    # (canonical hessian key lookup fails for the alias keys with KeyError).
    # Dedup at the GPU level FIRST so alias keys map to the SAME CPU tensor.
    _sd_gpu = base_model.state_dict()
    _seen_gpu_ptrs = {}  # GPU data_ptr -> canonical state_dict key
    sd_cpu = {}
    for _k, _v in _sd_gpu.items():
        _gpu_ptr = _v.data_ptr()
        if _gpu_ptr in _seen_gpu_ptrs:
            sd_cpu[_k] = sd_cpu[_seen_gpu_ptrs[_gpu_ptr]]
        else:
            _seen_gpu_ptrs[_gpu_ptr] = _k
            sd_cpu[_k] = _v.detach().cpu()
    device = torch.device('cuda', h.local_rank)
    log('GPTQ:collecting Hessians from calibration data...')
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device, n_calibration_batches=h.gptq_calibration_batches)
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f: f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16(); restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for (k, v) in eval_model.state_dict().items()}
    with open(h.quantized_model_path, 'rb') as f: quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location='cpu')
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len: raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}")
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train(); return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for (i, ws) in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens); wlen = we - ws; wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
            for (i, ws) in enumerate(batch_ws):
                wlen = wlens[i]; s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum(); token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train(); return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    rank = h.rank; world_size = h.world_size
    seq_len = h.eval_seq_len; stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}")
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params: p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows: continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for (i, ws) in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens); wlen = we - ws; wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]; y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
                for (i, ws) in enumerate(batch_ws):
                    wlen = wlens[i]; s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum(); token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * .5 * (1. + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups: pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel(): continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.)
                        optimizer.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters(): p.requires_grad_(True)
    base_model.eval(); return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize(); elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16(); restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        # V1 (loop_unique_attn) leaves attn[1], attn[2] of looped blocks unused
        # whenever looping_active=False (warmup phase 1, first 35% of main training).
        # DDP raises a runtime error in that regime with the default
        # find_unused_parameters=False; enable it conditionally to allow training.
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False,
                    find_unused_parameters=h.loop_unique_attn)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None: return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0: return 1.
        if frac >= 1. - h.warmdown_frac: return max((1. - frac) / h.warmdown_frac, h.min_lr)
        return 1.

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all(); train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed: model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach(); (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.) if h.muon_momentum_warmup_steps > 0 else 1.
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups: group['momentum'] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group['lr'] = group['base_lr'] * lr_scale
        if h.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step(); return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for (name, tensor) in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for (opt, state) in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed: model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    ema_state = {name: t.detach().float().clone() for (name, t) in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = .0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms); scale = lr_mul(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for (name, t) in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1. - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    log(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
    log('ema:applying EMA weights')
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed); np.random.seed(h.seed); torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
    if h.distributed: dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0: eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval('quantized', eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval('quantized_sliding_window', eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model; torch._dynamo.reset(); torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0: ttt_model.looping_active = True
        timed_eval('quantized_ttt', eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model


def main():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    if world_size <= 0: raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    device = torch.device('cuda', local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend='nccl', device_id=device); dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters(); set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log(100 * '=', console=False); log('Hyperparameters:', console=True)
        for (k, v) in sorted(vars(type(h)).items()):
            if not k.startswith('_'): log(f"  {k}: {v}", console=True)
        log('=' * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
        log('=' * 100, console=False)
    train_and_eval(h, device)
    if distributed: dist.destroy_process_group()


if __name__ == '__main__':
    main()
