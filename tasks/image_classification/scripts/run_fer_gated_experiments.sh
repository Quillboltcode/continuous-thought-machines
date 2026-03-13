#!/bin/bash

# Configuration
DATA_ROOT="/kaggle/input/datasets/tunminhhunh/fer-plus/FERPlus/fer_plus"
DEVICE=0
TRAIN_ITERS=5001
BATCH_SIZE=64
LR=5e-4

# Base Model Params
BASE_PARAMS="--model ctm_gated \
    --convert_grayscale_to_rgb \
    --d_model 512 \
    --d_input 256 \
    --synapse_depth 4 \
    --heads 8 \
    --n_synch_out 256 \
    --n_synch_action 256 \
    --iterations 50 \
    --memory_length 10 \
    --deep_memory \
    --memory_hidden_dims 16 \
    --dropout 0.1 \
    --backbone_type resnet18-4 \
    --pretrained_backbone imagenet \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --training_iterations $TRAIN_ITERS \
    --img_size 112 \
    --use_scheduler \
    --scheduler_type cosine \
    --dataset FerPlusPlus \
    --data_root $DATA_ROOT \
    --use_amp \
    --save_best_model \
    --device $DEVICE"

echo "Starting FER Gated Experiments Comparison..."

# 1. Baseline: No Early Exit
echo "Running Baseline: No Early Exit..."
python -m tasks.image_classification.train $BASE_PARAMS \
    --exit_strategy none --loss_type standard

# 2. Certainty Strategy with Different Thresholds and Losses
for thr in 0.7 0.85; do
    for loss in "standard" "loop"; do
        echo "Running: Strategy=certainty, Threshold=$thr, Loss=$loss"
        python -m tasks.image_classification.train $BASE_PARAMS \
            --exit_strategy certainty --exit_threshold $thr --loss_type $loss --min_steps 5
    done
done

# 3. Ponder Strategy (PonderNet style)
echo "Running: Strategy=ponder, Loss=ponder"
python -m tasks.image_classification.train $BASE_PARAMS \
    --exit_strategy ponder --loss_type ponder --lambda_p 0.2 --beta 0.01 --min_steps 3

# 4. Learned Strategy (Learnable halting probability)
for loss in "standard" "loop"; do
    echo "Running: Strategy=learned, Loss=$loss"
    python -m tasks.image_classification.train $BASE_PARAMS \
        --exit_strategy learned --loss_type $loss --min_steps 5
done

# 5. Normal Strategy (Time-based decay exit)
echo "Running: Strategy=normal, Loss=standard"
python -m tasks.image_classification.train $BASE_PARAMS \
    --exit_strategy normal --loss_type standard --min_steps 1

echo "All experiments completed!"
