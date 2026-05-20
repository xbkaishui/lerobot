#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Estimate (or empirically measure) GPU memory usage of a LeRobot policy.

Two modes:

* ``--mode analytical``  纯解析估算。不依赖 GPU、不下载预训练权重。
  对 pi0 / pi05 使用细粒度公式（VLM + Action Expert + Vision Tower + KV Cache），
  对其它 policy 退化为「在 CPU 上实例化并统计参数」+ 经验激活系数。

* ``--mode empirical``   真实测量。把模型放到 ``--device`` 上跑一次
  forward（``--train`` 时再跑 backward + Adam step），用
  ``torch.cuda.max_memory_allocated()`` 取峰值。

示例：
    # 估算 pi05 在 batch=4, 224x224 单相机下的训练显存（含 AdamW 状态）
    python examples/estimate_policy_memory.py \\
        --policy_type pi05 --batch_size 4 --num_views 1 --train

    # 实测 pi0 推理峰值
    python examples/estimate_policy_memory.py \\
        --policy_type pi0 --batch_size 1 --mode empirical --device cuda

    # 估算 ACT
    python examples/estimate_policy_memory.py \\
        --policy_type act --batch_size 16
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
}


def _bytes_to_gb(n: int | float) -> float:
    return float(n) / (1024**3)


def _human(n_bytes: int | float) -> str:
    return f"{_bytes_to_gb(n_bytes):8.3f} GB"


@dataclass
class MemoryReport:
    policy_type: str
    batch_size: int
    seq_len: int | None = None
    prefix_len: int | None = None
    image_tokens_per_view: int | None = None
    num_views: int | None = None

    # bytes
    weights: int = 0
    weights_fp32_part: int = 0
    weights_low_precision_part: int = 0

    activations: int = 0
    kv_cache: int = 0
    gradients: int = 0
    optimizer_states: int = 0
    compile_overhead: int = 0  # torch.compile 图捕获 + 融合缓冲 + autotuning
    cuda_overhead: int = int(1.2 * 1024**3)  # 经验：CUDA context + workspace

    notes: list[str] = field(default_factory=list)

    @property
    def inference_total(self) -> int:
        return self.weights + self.activations + self.kv_cache + self.compile_overhead + self.cuda_overhead

    @property
    def training_total(self) -> int:
        return (
            self.weights
            + self.activations
            + self.gradients
            + self.optimizer_states
            + self.compile_overhead
            + self.cuda_overhead
        )

    def pretty(self, training: bool = False) -> str:
        lines = []
        lines.append("=" * 68)
        lines.append(f"Policy        : {self.policy_type}")
        lines.append(f"Batch size    : {self.batch_size}")
        if self.num_views is not None:
            lines.append(f"# camera views: {self.num_views}")
        if self.image_tokens_per_view is not None:
            lines.append(f"img tokens/view: {self.image_tokens_per_view}")
        if self.prefix_len is not None:
            lines.append(f"prefix tokens : {self.prefix_len}")
        if self.seq_len is not None:
            lines.append(f"seq_len       : {self.seq_len}")
        lines.append("-" * 68)
        lines.append(f"  Weights total       : {_human(self.weights)}")
        if self.weights_fp32_part:
            lines.append(
                f"    └─ fp32 part      : {_human(self.weights_fp32_part)}"
            )
            lines.append(
                f"    └─ bf16/fp16 part : {_human(self.weights_low_precision_part)}"
            )
        lines.append(f"  Activations         : {_human(self.activations)}")
        lines.append(f"  KV cache (infer)    : {_human(self.kv_cache)}")
        if training:
            lines.append(f"  Gradients           : {_human(self.gradients)}")
            lines.append(f"  Optimizer (AdamW)   : {_human(self.optimizer_states)}")
        if self.compile_overhead:
            lines.append(f"  torch.compile extra : {_human(self.compile_overhead)}")
        lines.append(f"  CUDA overhead       : {_human(self.cuda_overhead)}")
        lines.append("-" * 68)
        if training:
            lines.append(f"  >> Train peak (est) : {_human(self.training_total)}")
        else:
            lines.append(f"  >> Infer peak (est) : {_human(self.inference_total)}")
        lines.append("=" * 68)
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# torch.compile overhead estimation
# ---------------------------------------------------------------------------

# 经验系数：torch.compile 的额外显存来自图捕获缓冲、融合 kernel workspace、
# autotuning 候选缓存。大小与模型结构相关：
#   - 规则的堆叠 Transformer（如 pi0/pi05）：编译后复用率高，开销较低 (~15% of activations)
#   - 含分支/动态逻辑的模型（ACT decoder、diffusion UNet）：graph break 多，开销较高 (~20%)
#   - mode="reduce-overhead" (CUDA Graphs)：额外固定 ~200MB 图录制开销
_COMPILE_FACTOR = {
    "pi0": 0.15,
    "pi05": 0.15,
    "act": 0.20,
    "diffusion": 0.25,  # UNet 的 skip connections 导致更多中间 buffer
    "smolvla": 0.18,
    "vqbet": 0.20,
}
_COMPILE_FACTOR_DEFAULT = 0.20
_CUDA_GRAPHS_FIXED_OVERHEAD = int(200 * 1024**2)  # ~200 MB


def _estimate_compile_overhead(
    policy_type: str,
    activations: int,
    weights: int,
    compile_mode: str = "default",
) -> int:
    """估算 torch.compile 引入的额外显存。

    开销主要与 activations 相关（图捕获时需保留中间张量副本用于 shape propagation
    和 kernel fusion workspace），同时受 compile_mode 影响：
      - "default": 仅 Inductor 图融合，开销 = factor * activations
      - "reduce-overhead": 启用 CUDA Graphs，额外锁定 ~200MB 静态显存
      - "max-autotune": Inductor + triton autotuning，开销再加 ~5%（候选 kernel 缓存）
    """
    factor = _COMPILE_FACTOR.get(policy_type, _COMPILE_FACTOR_DEFAULT)

    # 基础开销：与激活成比例
    overhead = int(activations * factor)

    # CUDA Graphs 模式额外固定开销
    if compile_mode == "reduce-overhead":
        overhead += _CUDA_GRAPHS_FIXED_OVERHEAD

    # max-autotune 会在编译阶段缓存更多 kernel 变体
    if compile_mode == "max-autotune":
        overhead += int(activations * 0.05)

    return overhead


# ---------------------------------------------------------------------------
# Analytical formulas for the Gemma family (pi0 / pi05)
# ---------------------------------------------------------------------------

# (width, depth, mlp_dim, num_heads, num_kv_heads, head_dim)
_GEMMA_VARIANTS = {
    "gemma_300m": (1024, 18, 4096, 8, 1, 256),
    "gemma_2b": (2048, 18, 16_384, 8, 1, 256),
}
_GEMMA_VOCAB = 257_152
# SigLIP-So400m used by PaliGemma-3B.  Patch=14 → image_tokens = (H/14)*(W/14)
_SIGLIP_PATCH = 14
_SIGLIP_PARAMS = 400_000_000  # ≈ 400M, kept fp32 in pi05 (modeling_pi05.to_bfloat16_for_selected_params)
_PALIGEMMA_MM_PROJECTOR_PARAMS = 4_300_000  # multi_modal_projector ≈ 4M


def _gemma_layer_params(width, mlp_dim, num_heads, num_kv_heads, head_dim, use_adarms=False):
    """Trainable params per Gemma transformer block (q/k/v/o + gate/up/down + 2x norm)."""
    qkv = width * (num_heads + 2 * num_kv_heads) * head_dim
    o = num_heads * head_dim * width
    mlp = 3 * width * mlp_dim
    norm = 2 * width
    if use_adarms:
        # PiGemmaRMSNorm with cond_dim adds a Linear(cond_dim, dim*3) per norm.
        # cond_dim == width on the action expert side.
        norm += 2 * (width * width * 3 + width * 3)
    return qkv + o + mlp + norm


def _gemma_total_params(variant, with_embed=True, use_adarms=False):
    width, depth, mlp_dim, num_heads, num_kv_heads, head_dim = _GEMMA_VARIANTS[variant]
    p = depth * _gemma_layer_params(
        width, mlp_dim, num_heads, num_kv_heads, head_dim, use_adarms=use_adarms
    )
    p += width  # final norm
    if with_embed:
        p += _GEMMA_VOCAB * width
    return p


def _estimate_pi_family(policy_type: str, cfg, batch_size: int) -> MemoryReport:
    """Architectural estimate for pi0 / pi05 (PaliGemma + Action Expert)."""
    dt_bytes = DTYPE_BYTES[cfg.dtype]
    is_pi05 = policy_type == "pi05"

    # ---- 1. Sizes ---------------------------------------------------------
    width_v, depth_v, _, _, kv_heads_v, head_dim_v = _GEMMA_VARIANTS[cfg.paligemma_variant]
    width_e, depth_e, _, _, kv_heads_e, head_dim_e = _GEMMA_VARIANTS[cfg.action_expert_variant]

    H, W = cfg.image_resolution
    img_tokens_per_view = (H // _SIGLIP_PATCH) * (W // _SIGLIP_PATCH)
    # 默认假定 1 个相机 + (empty_cameras), 调用方可通过 input_features 注入
    num_views = max(1, len(cfg.image_features) + cfg.empty_cameras)
    prefix_len = num_views * img_tokens_per_view + cfg.tokenizer_max_length
    suffix_len = cfg.chunk_size
    seq_len = prefix_len + suffix_len

    # ---- 2. Parameter counts ---------------------------------------------
    vlm_params = _gemma_total_params(
        cfg.paligemma_variant, with_embed=True, use_adarms=False
    )
    expert_params = _gemma_total_params(
        cfg.action_expert_variant, with_embed=False, use_adarms=is_pi05  # pi05 uses adaRMS on expert
    )
    siglip_params = _SIGLIP_PARAMS + _PALIGEMMA_MM_PROJECTOR_PARAMS
    proj_params = (
        cfg.max_action_dim * width_e  # action_in_proj
        + width_e * cfg.max_action_dim  # action_out_proj
        + width_e * width_e             # time_mlp_in
        + width_e * width_e             # time_mlp_out
    )
    if not is_pi05:
        # pi0 also has state_proj (max_state_dim -> width_e)
        proj_params += cfg.max_state_dim * width_e

    total_params = vlm_params + expert_params + siglip_params + proj_params

    # ---- 3. Weight memory -------------------------------------------------
    # pi05 keeps vision_tower + multi_modal_projector + all layernorms in fp32
    # even when dtype=bf16 (see modeling_pi05.to_bfloat16_for_selected_params).
    # For pi0 we use a single dtype for everything.
    if is_pi05 and dt_bytes < 4:
        # ~3 layernorms per Gemma block of (2*width) + final norm; that's tiny but we add it.
        norms_v = depth_v * 2 * width_v + width_v
        norms_e = depth_e * 2 * width_e + width_e
        fp32_params = siglip_params + norms_v + norms_e
        low_params = total_params - fp32_params
        weights_fp32 = fp32_params * 4
        weights_low = low_params * dt_bytes
    else:
        fp32_params = 0
        low_params = total_params
        weights_fp32 = 0 if dt_bytes == 4 else 0
        weights_low = total_params * dt_bytes

    weights_total = weights_fp32 + weights_low if is_pi05 else total_params * dt_bytes

    # ---- 4. KV cache (only the VLM caches; the expert recomputes suffix) -
    # 2 (K+V) * B * prefix_len * depth * kv_heads * head_dim * dtype
    kv_cache = (
        2 * batch_size * prefix_len * depth_v * kv_heads_v * head_dim_v * dt_bytes
    )

    # ---- 5. Activations (training, no gradient checkpointing) -------------
    # Dominant terms: hidden states + 2 * attention scores (B*heads*seq_len^2).
    # Empirical multiplier 1.4x to cover misc scratch tensors.
    width_avg = (width_v + width_e) / 2
    depth = depth_v  # pi0 / pi05 share depth
    hidden_act = depth * batch_size * seq_len * (4 * width_avg) * dt_bytes
    attn_act = depth * batch_size * 8 * seq_len * seq_len * dt_bytes  # 8 = num_heads
    siglip_act = batch_size * num_views * img_tokens_per_view * 1152 * 4 * 2  # SigLIP fp32
    activations = int(1.4 * (hidden_act + attn_act + siglip_act))

    if cfg.gradient_checkpointing:
        activations = int(activations / max(1.0, math.sqrt(depth)))

    # ---- 6. Gradients + AdamW -------------------------------------------
    # Trainable parameter count (account for freezing flags)
    trainable_params = total_params
    if cfg.freeze_vision_encoder:
        trainable_params -= siglip_params
    if cfg.train_expert_only:
        trainable_params = expert_params + proj_params

    # Gradients sit in the same dtype as the parameters; AdamW (fp32 m & v).
    grad_bytes = trainable_params * dt_bytes
    optim_bytes = trainable_params * 2 * 4  # m, v in fp32

    rep = MemoryReport(
        policy_type=policy_type,
        batch_size=batch_size,
        seq_len=seq_len,
        prefix_len=prefix_len,
        image_tokens_per_view=img_tokens_per_view,
        num_views=num_views,
        weights=weights_total,
        weights_fp32_part=weights_fp32,
        weights_low_precision_part=weights_low,
        activations=activations,
        kv_cache=kv_cache,
        gradients=grad_bytes,
        optimizer_states=optim_bytes,
    )
    rep.notes.append(
        f"params: VLM≈{vlm_params/1e9:.2f}B  Expert≈{expert_params/1e9:.2f}B "
        f"SigLIP≈{siglip_params/1e6:.0f}M  proj≈{proj_params/1e6:.1f}M  "
        f"total≈{total_params/1e9:.2f}B  trainable≈{trainable_params/1e9:.2f}B"
    )
    if is_pi05 and dt_bytes < 4:
        rep.notes.append(
            "pi05 keeps vision_tower + multi_modal_projector + layernorms in fp32"
        )
    if cfg.gradient_checkpointing:
        rep.notes.append("gradient_checkpointing=True → activations divided by √depth")
    return rep


# ---------------------------------------------------------------------------
# Generic estimate for any other policy (instantiate on CPU + heuristic)
# ---------------------------------------------------------------------------


def _generic_estimate(policy_type: str, cfg, batch_size: int) -> MemoryReport:
    """Fallback: build the policy on CPU, count parameters, apply heuristics."""
    cls = get_policy_class(policy_type)
    cfg.device = "cpu"
    policy = cls(cfg)
    policy.eval()

    weights_bytes = sum(
        p.numel() * p.element_size() for p in policy.parameters()
    ) + sum(b.numel() * b.element_size() for b in policy.buffers())

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    # Heuristic activation memory: 2 × weights × batch_size / 8
    # (purely a rule-of-thumb for small policies; user should run --mode empirical
    #  for accurate numbers).
    activations = int(weights_bytes * batch_size * 0.25)

    # Gradients in fp32; AdamW state 8B/param
    grad_bytes = n_trainable * 4
    optim_bytes = n_trainable * 8

    rep = MemoryReport(
        policy_type=policy_type,
        batch_size=batch_size,
        weights=weights_bytes,
        activations=activations,
        kv_cache=0,
        gradients=grad_bytes,
        optimizer_states=optim_bytes,
    )
    rep.notes.append(
        f"params total={n_params/1e6:.1f}M  trainable={n_trainable/1e6:.1f}M"
    )
    rep.notes.append(
        "generic estimate uses a rough heuristic for activations; "
        "use --mode empirical for accuracy."
    )
    return rep


# ---------------------------------------------------------------------------
# Empirical measurement
# ---------------------------------------------------------------------------


def _build_dummy_batch(cfg, batch_size: int, device: torch.device) -> dict:
    """Best-effort dummy batch for a forward pass."""
    batch: dict = {}
    # images
    for key, ft in cfg.image_features.items():
        batch[key] = torch.rand(batch_size, *ft.shape, device=device)
    # state
    if cfg.robot_state_feature is not None:
        batch[OBS_STATE] = torch.zeros(
            batch_size, *cfg.robot_state_feature.shape, device=device
        )
    # action (for forward/loss)
    if cfg.action_feature is not None:
        chunk = getattr(cfg, "chunk_size", None) or getattr(cfg, "horizon", None) or 1
        batch[ACTION] = torch.zeros(
            batch_size, chunk, *cfg.action_feature.shape, device=device
        )
    # task / language tokens (only for VLA policies)
    batch["task"] = ["pick up the cube"] * batch_size
    return batch


def _empirical_measure(policy_type: str, cfg, batch_size: int, train: bool) -> MemoryReport:
    if not torch.cuda.is_available():
        raise RuntimeError("--mode empirical requires a CUDA device.")

    device = torch.device(cfg.device or "cuda")
    cls = get_policy_class(policy_type)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy = cls(cfg).to(device)
    policy.train(train)
    weights_peak = torch.cuda.memory_allocated()

    # If the policy ships a processor (e.g. tokenizer), users would normally
    # run it before forward.  Skip here and assume forward can take raw batch.
    batch = _build_dummy_batch(cfg, batch_size, device)

    torch.cuda.reset_peak_memory_stats()
    optim_bytes = 0
    if train:
        loss, _ = policy.forward(batch)
        loss.backward()
        # Mimic AdamW state allocation
        opt = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-5
        )
        opt.step()
        optim_bytes = (
            sum(s["exp_avg"].numel() * s["exp_avg"].element_size()
                + s["exp_avg_sq"].numel() * s["exp_avg_sq"].element_size()
                for g in opt.param_groups for p in g["params"]
                for s in [opt.state[p]] if "exp_avg" in s)
        )
    else:
        with torch.no_grad():
            policy.predict_action_chunk(batch)

    peak = torch.cuda.max_memory_allocated()

    rep = MemoryReport(
        policy_type=policy_type,
        batch_size=batch_size,
        weights=weights_peak,
        activations=max(0, peak - weights_peak - optim_bytes),
        kv_cache=0,
        gradients=0,
        optimizer_states=optim_bytes,
        cuda_overhead=0,
    )
    rep.notes.append(f"measured peak (cuda) = {_human(peak)}")
    return rep


# ---------------------------------------------------------------------------
# Config preparation
# ---------------------------------------------------------------------------


def _prepare_config(args: argparse.Namespace):
    """Build a PreTrainedConfig with sensible defaults + image features."""
    overrides = {}
    if args.policy_type in {"pi0", "pi05"}:
        overrides["dtype"] = args.dtype
        overrides["chunk_size"] = args.chunk_size
        overrides["n_action_steps"] = args.chunk_size
        overrides["image_resolution"] = (args.image_resolution, args.image_resolution)
        # We will inject `num_views` real image features below, so no "empty" cameras.
        overrides["empty_cameras"] = 0
        overrides["gradient_checkpointing"] = args.gradient_checkpointing
        overrides["freeze_vision_encoder"] = args.freeze_vision_encoder
        overrides["train_expert_only"] = args.train_expert_only
    overrides["device"] = args.device
    cfg = make_policy_config(args.policy_type, **overrides)

    # Inject input/output features so cfg.image_features works in the analytical path
    img_shape = (3, args.image_resolution, args.image_resolution)
    if not cfg.input_features:
        cfg.input_features = {}
    if not cfg.output_features:
        cfg.output_features = {}
    for i in range(args.num_views):
        key = f"{OBS_IMAGES}.cam_{i}" if i > 0 else f"{OBS_IMAGES}.main"
        cfg.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=img_shape)

    state_dim = args.state_dim
    cfg.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    action_dim = args.action_dim
    cfg.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))

    cfg.validate_features()
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate or measure GPU memory of a LeRobot policy."
    )
    parser.add_argument("--policy_type", required=True,
                        help="e.g. pi05, pi0, act, diffusion, smolvla, ...")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mode", choices=["analytical", "empirical"], default="analytical")
    parser.add_argument("--train", action="store_true",
                        help="Include gradient + AdamW optimizer state in the report.")

    # input shape knobs
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--state_dim", type=int, default=14)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--chunk_size", type=int, default=50)

    # pi-family knobs
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float32", "fp16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--train_expert_only", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # torch.compile knobs
    parser.add_argument("--compile", action="store_true",
                        help="Include torch.compile overhead in estimate.")
    parser.add_argument("--compile_mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. Affects overhead estimate.")

    args = parser.parse_args()

    cfg = _prepare_config(args)

    if args.mode == "empirical":
        report = _empirical_measure(args.policy_type, cfg, args.batch_size, args.train)
    else:
        if args.policy_type in {"pi0", "pi05"}:
            report = _estimate_pi_family(args.policy_type, cfg, args.batch_size)
        else:
            report = _generic_estimate(args.policy_type, cfg, args.batch_size)

    # Apply torch.compile overhead if requested
    if args.compile:
        compile_extra = _estimate_compile_overhead(
            args.policy_type,
            activations=report.activations,
            weights=report.weights,
            compile_mode=args.compile_mode,
        )
        report.compile_overhead = compile_extra
        report.notes.append(
            f"torch.compile(mode='{args.compile_mode}') → "
            f"extra {_human(compile_extra)} "
            f"(factor={_COMPILE_FACTOR.get(args.policy_type, _COMPILE_FACTOR_DEFAULT):.0%} of activations)"
        )

    print(report.pretty(training=args.train))


if __name__ == "__main__":
    main()
