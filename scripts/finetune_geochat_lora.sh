#!/bin/bash

################## FastVLM (LLaVA: FastViTHD + Qwen2 LLM) ##################
# Same recipe as GeoChat/scripts/finetune_lora.sh
PROMPT_VERSION=qwen_2
MODEL_VERSION="llava-fastvithd_1.5b_stage3"
################## FastVLM ##################

# GeoChat_Instruct data (https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct); override with DATA_DIR=...
DATA_DIR=${DATA_DIR:-/mnt/d/GeoChat_data}

export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_PROJECT=geochat

 deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --max_samples 10000\
    --version $PROMPT_VERSION \
    --data_path $DATA_DIR/GeoChat_Instruct.json \
    --image_folder $DATA_DIR/share/softwares/kartik/GeoChat_finetuning/final_images_llava \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/${MODEL_VERSION}-geochat-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name fastvlm-geochat-lora-$(date +%m%d-%H%M)
