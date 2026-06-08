#!/usr/bin/env python3
"""Precompute and cache text embeddings for all tasks in a LeRobot dataset.

This script reads the task list from a dataset's `meta/tasks.parquet`,
applies the prompt_template, encodes each task via the T5 text encoder,
and persists the results to disk so that training can skip text encoder loading.

Usage:
    python scripts/precompute_prompt_cache.py \
        --dataset_root /root/autodl-fs/datasets/libero \
        --model_id /root/autodl-fs/ckpts/models/Wan-AI/Wan2.2-TI2V-5B \
        --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Must match the default in FastWAMConfig.prompt_cache_dir
DEFAULT_DISK_CACHE_DIR = "/root/autodl-fs/ckpts/fast_wam/text_embeding_lerobot_cache/libero"

# Default prompt template (same as FastWAMConfig.prompt_template)
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def get_cache_path(prompt: str, cache_dir: str = DEFAULT_DISK_CACHE_DIR) -> Path:
    """Generate a stable file path for a prompt's disk cache (SHA-256 hash)."""
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.pt"


def load_tasks_from_dataset(dataset_root: str | Path) -> list[str]:
    """Read all unique task descriptions from dataset meta/tasks.parquet."""
    tasks_parquet = Path(dataset_root) / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        raise FileNotFoundError(f"Tasks parquet not found: {tasks_parquet}")

    df = pd.read_parquet(tasks_parquet)
    # The index contains the task text, task_index is the column
    tasks = df.index.tolist()
    logger.info("Loaded %d tasks from %s", len(tasks), tasks_parquet)
    return tasks


def build_text_encoder(model_id: str, device: str, torch_dtype: torch.dtype):
    """Load only the T5 text encoder and tokenizer (no DiT/VAE needed)."""
    from lerobot.policies.fastwam.wan_components import (
        load_wan_text_encoder,
        load_wan_tokenizer,
        resolve_wan_checkpoint_dir,
        resolve_wan_checkpoint_paths,
    )

    checkpoint_dir = resolve_wan_checkpoint_dir(model_id)
    paths = resolve_wan_checkpoint_paths(
        checkpoint_dir,
        tokenizer_dir=checkpoint_dir,
        load_dit=False,
        load_text_encoder=True,
    )

    if paths.text_encoder is None or paths.tokenizer is None:
        raise FileNotFoundError("Text encoder or tokenizer not found in checkpoint dir.")

    logger.info("Loading T5 text encoder from %s ...", paths.text_encoder)
    t0 = time.time()
    text_encoder = load_wan_text_encoder(
        paths.text_encoder, torch_dtype=torch_dtype, device=device
    )
    logger.info("  Text encoder loaded in %.2f s", time.time() - t0)

    logger.info("Loading tokenizer from %s ...", paths.tokenizer)
    tokenizer = load_wan_tokenizer(paths.tokenizer, tokenizer_max_len=512)

    return text_encoder, tokenizer


def encode_prompt(
    text_encoder: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of prompts using the T5 text encoder.

    Mirrors the logic in modular_fastwam.py FastWAM.encode_prompt.
    """
    # Tokenize
    ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device=device)
    mask = mask.to(device=device)

    # Encode
    with torch.no_grad():
        seq_lens = mask.gt(0).long().sum(dim=1)
        context = text_encoder(ids, mask)
        # Zero out invalid positions
        for i, seq_len in enumerate(seq_lens):
            context[i, seq_len:] = 0
    # Return all-ones mask (invalid embeddings already zeroed)
    context_mask = torch.ones(context.shape[0], context.shape[1], device=device)
    return context, context_mask


def main():
    parser = argparse.ArgumentParser(description="Precompute prompt cache for FastWAM")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/autodl-fs/datasets/libero",
        help="Path to the LeRobot dataset root",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/root/autodl-fs/ckpts/models/Wan-AI/Wan2.2-TI2V-5B",
        help="Path to Wan2.2-TI2V-5B model checkpoint directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template with {task} placeholder",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_DISK_CACHE_DIR,
        help="Directory to store prompt embedding cache files",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    # 1. Load task list
    raw_tasks = load_tasks_from_dataset(args.dataset_root)

    # 2. Apply prompt template
    prompts = [args.prompt_template.format(task=task) for task in raw_tasks]
    logger.info("Generated %d prompts with template", len(prompts))

    # 3. Check which are already cached
    uncached_prompts = []
    cached_count = 0
    for p in prompts:
        cache_path = get_cache_path(p, args.cache_dir)
        if cache_path.exists():
            cached_count += 1
        else:
            uncached_prompts.append(p)

    logger.info("Already cached: %d, need encoding: %d", cached_count, len(uncached_prompts))

    if not uncached_prompts:
        logger.info("All prompts already cached on disk. Nothing to do.")
        return

    # 4. Load text encoder
    text_encoder, tokenizer = build_text_encoder(args.model_id, args.device, torch_dtype)

    # 5. Encode in batches and save to disk
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_encoded = 0
    t_start = time.time()

    for batch_start in range(0, len(uncached_prompts), args.batch_size):
        batch = uncached_prompts[batch_start : batch_start + args.batch_size]
        context, context_mask = encode_prompt(text_encoder, tokenizer, batch, args.device)

        for j, prompt in enumerate(batch):
            ctx_j = context[j : j + 1].cpu()
            mask_j = context_mask[j : j + 1].cpu()
            cache_path = get_cache_path(prompt, args.cache_dir)
            torch.save(
                {"context": ctx_j, "context_mask": mask_j, "prompt": prompt},
                cache_path,
            )
            total_encoded += 1

        logger.info(
            "  Encoded batch [%d-%d] / %d",
            batch_start,
            batch_start + len(batch),
            len(uncached_prompts),
        )

    elapsed = time.time() - t_start
    logger.info(
        "Done! Encoded %d prompts in %.2f s. Cache dir: %s",
        total_encoded, elapsed, args.cache_dir,
    )

    # 6. Print summary
    logger.info("\n=== Cache Summary ===")
    all_files = list(cache_dir.glob("*.pt"))
    logger.info("Total cached files: %d", len(all_files))
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
    logger.info("Total cache size: %.2f MB", total_size_mb)


if __name__ == "__main__":
    main()
