#!/bin/bash

# ==============================
# AIXueRLHF 训练脚本
# ==============================

set -e  # 遇到错误立即退出
set -o pipefail # 任何管道失败都退出，避免误报“训练完成”

export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO

# 设置目录
DATASET_PATH="datasets/aixue_rlhf_dataset/bad_cases.json"
PRETRAINED_MODEL_PATH="/root/group-shared/models/base_models/Qwen3-8B"
OUTPUT_DIR="models/trained_models/Qwen3-8B-AIXueRLHF"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs/training"

# 创建必要的目录
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo -e "✅ 开始训练\n模型: ${PRETRAINED_MODEL_PATH}\n数据: ${DATASET_PATH}\n输出: ${OUTPUT_DIR}" | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

accelerate launch \
    --config_file configs/accelerate_configs/deepspeed_zero3_offload.yaml \
    --main_process_port 29504 \
    src/aixuerlhf/aixue.py \
    --dataset_name "${DATASET_PATH}" \
    --model_name_or_path "${PRETRAINED_MODEL_PATH}" \
    --sft_model_path "${PRETRAINED_MODEL_PATH}" \
    --learning_rate 1e-6 \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1024 \
    --num_ppo_epochs 4 \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --save_safetensors true \
    --attn_implementation flash_attention_2 \
    --use_liger_loss false \
    --kl_estimator k3 \
    --temperature 1.0 \
    --whiten_advantages true \
    --kl_coef 0.04 \
    --report_to tensorboard \
    --save_total_limit 3 \
    --save_steps 100 \
    2>&1 | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"

# -e 选项用于使 echo 支持转义字符（如 \n 实现换行），否则 \n 会被当作普通字符输出
echo -e "✅ 训练完成！\n模型保存在: ${OUTPUT_DIR}\n训练日志: ${LOG_DIR}/train_${TIMESTAMP}.log" | tee -a "${LOG_DIR}/train_${TIMESTAMP}.log"