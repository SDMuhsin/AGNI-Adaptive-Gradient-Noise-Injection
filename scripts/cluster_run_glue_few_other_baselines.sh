#!/bin/bash

OUTPUT_DIR="saves/tmp"
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32

# Set default model and allow override via command-line argument
MODEL=${1:-"bert-base-uncased"}

export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE
export MODEL

# Define arrays for the hyperparameters
OPTIMIZERS=("sgd" "asgd" "adam" "adamw" "adadelta" "adagrad" "rmsprop")
LEARNING_RATES=("2e-5" "2e-4" "2e-3" "2e-2" "2e-6")
NUM_EPOCHS=(2 4)

# Run the script for the specified model, each task, and seed
for TASK in stsb rte; do
    echo "Running for model: $MODEL, task: $TASK"
    for OPTIMIZER in "${OPTIMIZERS[@]}"; do
        for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
            for NUM_EPOCH in "${NUM_EPOCHS[@]}"; do
                echo "Using optimizer: $OPTIMIZER, learning rate: $LEARNING_RATE, epochs: $NUM_EPOCH"
                parallel --jobs 1 -u "echo 'Running for model: $MODEL, task: $TASK, optimizer: $OPTIMIZER, learning rate: $LEARNING_RATE, epochs: $NUM_EPOCH, with seed: {}'; \
                conda activate double_env_6 && \
                python3 ./source/run_glue_few_other_baselines.py \
                    --output_dir $OUTPUT_DIR \
                    --model_name_or_path $MODEL \
                    --seed {} \
                    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
                    --num_train_epochs $NUM_EPOCH \
                    --learning_rate $LEARNING_RATE \
                    --job_id few_other_baselines \
                    --split_train n \
                    --task_name $TASK \
                    --optimizer $OPTIMIZER \
                    --overwrite_saves n" ::: 41 42 43
            done
        done
    done
done
