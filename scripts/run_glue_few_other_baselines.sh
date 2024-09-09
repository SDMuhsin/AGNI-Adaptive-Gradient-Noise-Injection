#!/bin/bash

# Define constants
OUTPUT_DIR="saves/tmp"
MODELS=("bert-base-uncased")  # Add your model names here
MAX_SEQ_LENGTH=128
TRAIN_BATCH_SIZE=32

# Define the parameter ranges
OPTIMIZERS=("sgd" "asgd" "adam" "adamw" "adadelta" "adagrad" "rmsprop")
LEARNING_RATES=(2e-5)
NUM_EPOCHS_LIST=(3)

export OUTPUT_DIR
export MAX_SEQ_LENGTH
export TRAIN_BATCH_SIZE

# Run the script for each model and task
for MODEL in "${MODELS[@]}"; do
    for TASK in rte; do
        for OPTIMIZER in "${OPTIMIZERS[@]}"; do
            for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                for NUM_EPOCHS in "${NUM_EPOCHS_LIST[@]}"; do

                    export MODEL TASK OPTIMIZER LEARNING_RATE NUM_EPOCHS

                    parallel -j 1 -u 'echo "Running for model: $MODEL, task: $TASK, optimizer: $OPTIMIZER, learning rate: $LEARNING_RATE, num epochs: $NUM_EPOCHS with seed: {}"; \
                        python3 ./source/run_glue_few_other_baselines.py \
                        --output_dir $OUTPUT_DIR \
                        --model_name_or_path $MODEL \
                        --seed {} \
                        --per_device_train_batch_size $TRAIN_BATCH_SIZE \
                        --num_train_epochs $NUM_EPOCHS \
                        --learning_rate $LEARNING_RATE \
                        --job_id few_other_baselines \
                        --split_train n \
                        --just_download n \
                        --overwrite_saves n \
                        --optimizer $OPTIMIZER \
                        --task_name $TASK' ::: 41

                done
            done
        done
    done
done
