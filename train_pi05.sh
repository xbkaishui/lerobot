# trap '' HUP
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1
# try uninstall numpy
# python3 -m pip uninstall -y numpy && rm -rf /root/miniconda3/lib/python3.12/site-packages/numpy*
# 输出目录配置项（可通过环境变量 OUTPUT_DIR 覆盖）
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/ckpts/pi05/libero_pi05_0519_finetune_test}"
# OUTPUT_DIR="/root/autodl-fs/ckpts/pi05/libero_pi05_0519_finetune_test"

rm -rf "${OUTPUT_DIR}"


accelerate launch  \
--multi_gpu  \
--num_processes=4  \
--mixed_precision=bf16  \
/root/miniconda3/bin/lerobot-train \
--dataset.repo_id=libero_pi05_0518 \
--dataset.root=/root/autodl-fs/datasets/libero \
--policy.type=pi05  \
--output_dir="${OUTPUT_DIR}"  \
--job_name=libero_pi05_0519_finetune_test  \
--policy.device=cuda  \
--wandb.enable=true  \
--policy.push_to_hub=false  \
--steps=10000  \
--batch_size=64 \
--save_freq=2000  \
--keep_last_n_checkpoints=3 \
--policy.pretrained_path=/root/autodl-fs/ckpts/models/lerobot/pi05-libero  \
--policy.compile_model=true  \
--policy.gradient_checkpointing=true  \
--policy.dtype=bfloat16  \
--num_workers=16  \
--policy.tokenizer_max_length=64 \
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'  \