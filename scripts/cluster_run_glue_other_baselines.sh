#!/bin/bash

OUTPUT_DIR="saves/tmp"
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32
NUM_EPOCHS=3
LEARNING_RATE=2e-5

# Set default model and allow override via command-line argument
MODEL=${1:-"bert-base-uncased"}

export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE
export NUM_EPOCHS
export LEARNING_RATE
export MODEL

# Run the script for the specified model, each task, and seed
for TASK in cola stsb rte mrpc sst2 qnli mnli qqp; do
    echo "Running for model: $MODEL, task: $TASK"
    parallel --jobs 2 -u "echo 'Running for model: $MODEL, task: $TASK, with seed: {}'; \
    conda activate double_env_6 && \
    python3 ./source/run_glue_other_baselines.py \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL \
        --seed {} \
        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --job_id baselines \
	--split_train n \
        --task_name $TASK \
	--optimizer adagrad \
        --overwrite_saves n" ::: 41 42 43 44 45
done
