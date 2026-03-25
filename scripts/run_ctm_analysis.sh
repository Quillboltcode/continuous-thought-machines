#!/bin/bash

# CTM Analysis Script
# Usage: ./scripts/run_ctm_analysis.sh <checkpoint_path> <dataset> <data_root> <output_name> [actions]

set -e

CHECKPOINT=$1
DATASET=${2:-rafdb}
DATA_ROOT=$3
OUTPUT_NAME=${4:-ctm_analysis}
ACTIONS=${5:-"classification_report"}

# Default output dir
OUTPUT_DIR="tasks/image_classification/analysis/outputs/${OUTPUT_NAME}"
CHECKPOINT_DIR="analysis_checkpoints/${DATASET}"

echo "=== CTM Analysis Script ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Data root: ${DATA_ROOT}"
echo "Output name: ${OUTPUT_NAME}"
echo "Actions: ${ACTIONS}"
echo ""

# Create checkpoint directory if not exists
mkdir -p ${CHECKPOINT_DIR}

# Run analysis
echo "Running analysis..."
python -m tasks.image_classification.analysis.run_imagenet_analysis \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --no-debug \
    --device 0 \
    --actions ${ACTIONS} \
    --inference_iterations 30 \
    --N_to_viz 5

# Zip outputs
echo "Zipping outputs..."
ZIP_NAME="${OUTPUT_NAME}.zip"
cd tasks/image_classification/analysis/outputs
zip -r ${ZIP_NAME} ${OUTPUT_NAME}/

# Move to checkpoint dir
echo "Moving to ${CHECKPOINT_DIR}..."
mv ${ZIP_NAME} ../../../../${CHECKPOINT_DIR}/

# Cleanup
echo "Cleaning up..."
rm -rf ${OUTPUT_NAME}

echo ""
echo "=== Done! ==="
echo "Output: ${CHECKPOINT_DIR}/${ZIP_NAME}"
