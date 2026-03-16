#!/bin/bash

# Ablation study: test each component in isolation first
# This follows a more systematic approach: baseline -> individual -> combinations

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HNE_CONFIGS_FILE="$SCRIPT_DIR/hne_configs.json"

BASE_ARGS="
  --d_model 512
  --d_input 256
  --synapse_depth 4
  --heads 8
  --n_synch_out 256
  --n_synch_action 256
  --n_random_pairing_self 0
  --neuron_select_type random
  --iterations 50
  --memory_length 5
  --deep_memory
  --memory_hidden_dims 16
  --dropout 0.2
  --no-do_normalisation
  --positional_embedding_type none
  --backbone_type resnet18-4
  --pretrained_backbone imagenet
  --batch_size 64
  --batch_size_test 32
  --lr 5e-4
  --training_iterations 5001
  --warmup_steps 1000
  --use_scheduler
  --scheduler_type cosine
  --weight_decay 0.0
  --img_size 224
  --dataset RAFDB
  --data_root /kaggle/input/raf-db-dataset/DATASET
  --save_every 250
  --track_every 125
  --seed 42
  --n_test_batches 50
  --use_amp
  --sanp_init_top_k 1000
  --hne_group_configs_file $HNE_CONFIGS_FILE
"

# Exp 1: Baseline - standard CTM (no innovations)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm \
  --log_dir logs-lambda/rafdb/ablations/exp1_baseline_ctm \
  --device 0 &

# Exp 2: GSH only (Gated Synchronization Highway)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp2_gsh_only \
  --device 1 \
  --use_gsh &

# Exp 3: HNE only (Hierarchical NLM Ensemble)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp3_hne_only \
  --device 2 \
  --use_hne &

# Exp 4: SANP only (Sparse Adaptive Neuron Pairing)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp4_sanp_only \
  --device 3 \
  --use_sanp &

wait
echo "Phase 1 complete! Individual components tested."

# Phase 2: Two-component combinations
echo "Starting Phase 2: Two-component combinations..."

# Exp 5: GSH + HNE
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp5_gsh_hne \
  --device 0 \
  --use_gsh --use_hne &

# Exp 6: GSH + SANP
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp6_gsh_sanp \
  --device 1 \
  --use_gsh --use_sanp &

# Exp 7: HNE + SANP
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp7_hne_sanp \
  --device 2 \
  --use_hne --use_sanp &

wait
echo "Phase 2 complete! Two-component combinations tested."

# Phase 3: Full model with all three components
echo "Starting Phase 3: Full model..."

# Exp 8: Full model (GSH + HNE + SANP)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp8_full_model \
  --device 3 \
  --use_gsh --use_hne --use_sanp &

wait
echo "All ablation experiments complete!"
