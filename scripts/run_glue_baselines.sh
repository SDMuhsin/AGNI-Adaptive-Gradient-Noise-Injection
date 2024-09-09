#!/bin/bash

# Define constants
OUTPUT_DIR="saves/tmp"
MODELS=("bert-base-uncased")#("albert/albert-base-v1" "squeezebert/squeezebert-uncased" "facebook/bart-base" "bert-base-uncased" "google-t5/t5-base")  # Add your model names here
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32
NUM_EPOCHS=3
LEARNING_RATE=2e-5


export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE
export NUM_EPOCHS
export LEARNING_RATE
# Run the script for each model, task, and seed
for MODEL in "${MODELS[@]}"; do
    for TASK in qqp mnli qnli; do


		export MODEL TASK
		 parallel -j 1 -u 'echo "Running for model: $MODEL, task: $TASK, with seed: {}"; \
		    python3 ./source/run_glue_baselines.py \
		    --output_dir $OUTPUT_DIR \
		    --model_name_or_path $MODEL \
		    --seed {} \
		    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
		    --num_train_epochs $NUM_EPOCHS \
		    --learning_rate $LEARNING_RATE \
		    --job_id baselines \
		    --split_train n \
		    --just_download n \
		    --overwrite_saves n \
		    --task_name $TASK' ::: 41
    done
done
