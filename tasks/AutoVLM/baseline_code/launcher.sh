export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export MASTER_PORT=${MASTER_PORT:-34229}
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OPENAI_API_KEY=''


source activate moce_vlm

pip install -e .


############### Pretrain ################

GPUS=${GPUS:-8}
PRETRAIN_BATCH_SIZE=${BATCH_SIZE:-256}
PRETRAIN_PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
PRETRAIN_GRADIENT_ACC=$((PRETRAIN_BATCH_SIZE / WORLD_SIZE / GPUS / PRETRAIN_PER_DEVICE_BATCH_SIZE))
LLM_VERSION="/hug_ckpts/Qwen2.5-Math-7B-Instruct" 
VISION_MODEL_VERSION="/hug_ckpts/siglip-so400m-patch14-384"
PRETRAIN_OUTPUT_DIR='checkpoints/pretrain'
FINETUNE_OUTPUT_DIR='checkpoints/finetune'
PROMPT_VERSION="qwen_1_5"

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nnodes=${WORLD_SIZE}  \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /datasets/URSA-MATH/Moce_data/selected_pretrain_data.json \
    --image_folder /datasets/ \
    --mm_tunable_parts="mm_mlp_adapter" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${PRETRAIN_OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PRETRAIN_PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${PRETRAIN_GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \



############### Finetune ################

FINETUNE_BATCH_SIZE=${BATCH_SIZE:-128}
FINETUNE_PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
FINETUNE_GRADIENT_ACC=$((FINETUNE_BATCH_SIZE / WORLD_SIZE / GPUS / FINETUNE_PER_DEVICE_BATCH_SIZE))
ADAPTER_PATH="${PRETRAIN_OUTPUT_DIR}/mm_projector.bin"

ACCELERATE_CPU_AFFINITY=1 torchrun \
    --nnodes=${WORLD_SIZE}  \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version $PROMPT_VERSION \
    --data_path /datasets/URSA-MATH/Moce_data/selected_finetune_data_300k.json \
    --image_folder /datasets/ \
    --pretrain_mm_mlp_adapter=${ADAPTER_PATH} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ${FINETUNE_OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${FINETUNE_PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${FINETUNE_GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \



############### Evaluation ################

# Script to run the evaluation of the model on the MathVista Geometry & Function set

# FILE_NAME="Geo_Fun"
# python llava/evaluate/evaluation/generate_response.py \
#     --model_path "${FINETUNE_OUTPUT_DIR}" \
#     --output_dir "llava/evaluate/results/" \
#     --output_file "output_${FILE_NAME}.json" \
#     --dataset_name "/datasets/MathVista"

# python llava/evaluate/evaluation/extract_answer_parallel.py \
#     --output_dir "llava/evaluate/results/" \
#     --output_file "output_${FILE_NAME}.json"

# python llava/evaluate/evaluation/calculate_score.py \
#     --dataset_name "/datasets/MathVista" \
#     --output_dir "llava/evaluate/results/" \
#     --output_file "output_${FILE_NAME}.json" \
#     --score_file "scores_${FILE_NAME}.json"
