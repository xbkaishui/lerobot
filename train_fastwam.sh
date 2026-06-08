# trap '' HUP
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_DATASETS_IN_MEMORY=1
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1
# try uninstall numpy
# python3 -m pip uninstall -y numpy && rm -rf /root/miniconda3/lib/python3.12/site-packages/numpy*
# 输出目录配置项（可通过环境变量 OUTPUT_DIR 覆盖）
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/ckpts/fastwam/libero_fastwam_0608_finetune_test}"
# OUTPUT_DIR="/root/autodl-fs/ckpts/pi05/libero_pi05_0519_finetune_test"

rm -rf "${OUTPUT_DIR}"
# 数据集最好放到 tmp里面去， /root/autodl-tmp/

# accelerate launch  \
# --multi_gpu  \
accelerate launch  \
--num_processes=1  \
--mixed_precision=bf16  \
/root/miniconda3/bin/lerobot-train \
--dataset.repo_id=libero_pi05_0518 \
--dataset.root=/root/autodl-fs/datasets/libero \
--policy.type=fastwam  \
--output_dir="${OUTPUT_DIR}"  \
--job_name=libero_fastwam_0608_finetune_test  \
--policy.device=cuda  \
--wandb.enable=true  \
--policy.push_to_hub=false  \
--steps=20000  \
--batch_size=1 \
--save_freq=2000  \
--keep_last_n_checkpoints=3 \
--policy.load_text_encoder=false \
--policy.model_id=/root/autodl-fs/ckpts/models/Wan-AI/Wan2.2-TI2V-5B  \
--policy.tokenizer_model_id=/root/autodl-fs/ckpts/models/Wan-AI/Wan2.2-TI2V-5B  \
--num_workers=16
# --policy.tokenizer_max_length=64 \
# --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'  \