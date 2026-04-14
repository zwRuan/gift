#!/bin/bash

DATA_PATH=data/training_data/numina_cot_2k_llama31-8b.jsonl
OUTPUT_DIR=checkpoints/lora_adapter/llama31-8b_gift_2k
MODE=gift
LEARNING_RATE=2e-4
MODEL_PATH=meta-llama/Llama-3.1-8B
INSTRUCT_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
MODEL_MAX_LENGTH=2048
GLOBAL_BATCH_SIZE=256
NUM_TRAIN_EPOCHS=1

python src/python_scripts/llama31-8b_gift_2k.py \
    --mode $MODE \
    --model_max_length $MODEL_MAX_LENGTH \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR

# Merge Instruct Model
ADAPTER_MODEL=$OUTPUT_DIR
FILENAME=$(basename "$OUTPUT_DIR")
MODEL_NAME_OR_PATH=checkpoints/gift/${FILENAME}
python src/python_scripts/merge_adapter_to_base_model.py --base_model $INSTRUCT_MODEL_PATH --adapter $ADAPTER_MODEL --output_path $MODEL_NAME_OR_PATH
