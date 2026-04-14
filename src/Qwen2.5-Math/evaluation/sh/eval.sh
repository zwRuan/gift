set -ex


MODEL_NAME_OR_PATH=checkpoints/gift/llama31-8b_gift_2k
OUTPUT_DIR=outputs/gift/llama31-8b_gift_2k
PROMPT_TYPE=llama31-8b
n_sampling=16
temperature=1

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="math_oai,minerva_math,olympiadbench,aime24,amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u src/Qwen2.5-Math/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${temperature} \
    --n_sampling ${n_sampling} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm
