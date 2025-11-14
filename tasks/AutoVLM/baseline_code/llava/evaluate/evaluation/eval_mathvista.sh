
MODEL_PATH="LLaVA-NeXT/checkpoints/finetune_300k_qwen"
FILE_NAME="Geo_Fun"

python llava/evaluate/evaluation/generate_response.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "llava/evaluate/results/" \
    --output_file "output_${FILE_NAME}.json" \
    --dataset_name "/datasets/MathVista"

python llava/evaluate/evaluation/extract_answer_parallel.py \
    --output_dir "llava/evaluate/results/" \
    --output_file "output_${FILE_NAME}.json"

python llava/evaluate/evaluation/calculate_score.py \
    --dataset_name "/datasets/MathVista" \
    --output_dir "llava/evaluate/results/" \
    --output_file "output_${FILE_NAME}.json" \
    --score_file "scores_${FILE_NAME}.json"




