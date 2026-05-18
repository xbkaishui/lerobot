export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1
rm -rf /root/autodl-tmp/ckpts/pi05/libero_pi05_0518_model

accelerate launch  \
--multi_gpu  \
--num_processes=2  \
--mixed_precision=bf16  \
/root/miniconda3/bin/lerobot-train \
--dataset.repo_id=libero_pi05_0518 \
--dataset.root=/root/autodl-fs/datasets/libero \
--policy.type=pi05  \
--output_dir=/root/autodl-tmp/ckpts/pi05/libero_pi05_0518_model  \
--job_name=libero_pi05_0518_model  \
--policy.device=cuda  \
--wandb.enable=true  \
--policy.push_to_hub=false  \
--steps=800  \
--batch_size=64 \
--save_freq=500  \
--keep_last_n_checkpoints=3 \
--policy.pretrained_path=/root/autodl-fs/ckpts/models/lerobot/pi05_base  \
--policy.compile_model=true  \
--policy.gradient_checkpointing=true  \
--policy.dtype=bfloat16  \
--num_workers=8  \
--policy.tokenizer_max_length=64  \
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'