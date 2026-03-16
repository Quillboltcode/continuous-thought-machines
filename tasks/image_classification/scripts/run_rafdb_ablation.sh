#!/bin/bash

# Ablation study: 4 experiments
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
  --hne_group_configs \"[{'n_neurons': 170, 'memory': 2, 'hidden': 8}, {'n_neurons': 171, 'memory': 3, 'hidden': 16}, {'n_neurons': 171, 'memory': 5, 'hidden': 32}]\"
  --sanp_init_top_k 1000
  --use_psl
  --lambda_psl 0.1
  --use_ctcs
  --lambda_ctcs 0.1
"

# Exp 1: Full model (all innovations)
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp1_full \
  --device 0 \
  --use_gsh --use_hne --use_sanp &

# Exp 2: No GSH
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp2_no_gsh \
  --device 1 \
  --use_hne --use_sanp &

# Exp 3: No HNE
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp3_no_hne \
  --device 2 \
  --use_gsh --use_sanp &

# Exp 4: No SANP
python -m tasks.image_classification.train \
  $BASE_ARGS \
  --model ctm_with_innovations \
  --log_dir logs-lambda/rafdb/ablations/exp4_no_sanp \
  --device 3 \
  --use_gsh --use_hne &

wait
echo "All experiments complete!"
