# 🕰️ The Continuous Thought Machine (CTM)

**Paper:** [Technical Report (arXiv 2505.05522)](https://arxiv.org/abs/2505.05522) | **Blog:** [sakana.ai/ctm/](https://sakana.ai/ctm/) | **Demo:** [Interactive Website](https://pub.sakana.ai/ctm) | **Tutorial:** [examples/01_mnist.ipynb](examples/01_mnist.ipynb)

![Activations](assets/activations.gif)

---

> **Note from Fork**: This is a fork of [sakana-ai/continuous-thought-machines](https://github.com/sakana-ai/continuous-thought-machines) with enhancements for RAF-DB facial emotion recognition, improved memory write/read mechanisms, and additional analysis tools.

---

We present the Continuous Thought Machine (CTM), a model designed to unfold and leverage neural activity as the underlying mechanism for observation and action. Our contributions are:

1. **Internal temporal axis**, decoupled from any input data, that enables neuron activity to unfold over multiple "thought ticks."
2. **Neuron-level temporal processing**, where each neuron uses unique weight parameters to process a history of incoming signals, enabling fine-grained temporal dynamics.
3. **Neural synchronisation**, employed as a direct latent representation for modulating data and producing outputs, thus encoding information in the timing of neural activity.

The CTM demonstrates strong performance and versatility across challenging tasks including ImageNet classification, solving 2D mazes, sorting, parity computation, question-answering, and reinforcement learning (RL).

## Key Features of This Fork

- **RAF-DB Facial Emotion Recognition**: Full training and analysis pipeline for Radial FERPlus (RAF-DB) dataset
- **Enhanced Memory Mechanisms**: Configurable memory write methods (`fifo` and `attention`-based)
- **Comprehensive Analysis Tools**: GFLOPS calculation, temporal evaluation, latent space visualization
- **Multiple Backbone Support**: ResNet-18/34/50, ViT-Tiny, CLIP adapters, shallow-wide convnets
- **Pre-trained Checkpoints**: Best model achieves 85.10% accuracy on RAF-DB test set

## Quick Start

### Installation

```bash
conda create --name=ctm python=3.12
conda activate ctm
pip install -r requirements.txt
```

If there are PyTorch/CUDA issues:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Train on RAF-DB

```bash
# Basic training with ResNet-18 backbone
python -m tasks.image_classification.train --config rafdb_resnet18

# With CTM innovations enabled
python -m tasks.image_classification.train --config rafdb_ctm_innovations
```

### Run Analysis

```bash
# Generate classification reports, confusion matrices, latent visualizations
python tasks/image_classification/analysis/run_imagenet_analysis.py \
  --dataset rafdb \
  --checkpoint analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt \
  --actions plots classification_report latent_viz
```

### Calculate GFLOPS

```bash
python calculate_gflops.py
```

See [GFLOPS_FINAL_SUMMARY.txt](GFLOPS_FINAL_SUMMARY.txt) for detailed analysis.

## Memory Write Mechanisms

The CTM supports two memory write strategies, controlled by `memory_write_type`:

### 1. FIFO (Default)
- Sliding window over the last `memory_length` states
- Minimal computational overhead
- Suitable for tasks requiring recent history

### 2. Attention-Based
- Learns to attend over historical memory states
- Computes: `attended = softmax(Q·K^T/√d) · V`
- Replace last timestep with attended summary
- Better for long-range dependencies, slightly more compute

```python
model = ContinuousThoughtMachine(
    ...
    memory_write_type='attention',  # or 'fifo'
    memory_length=5
)
```

## RAF-DB Training Results

| Model | Backbone | Accuracy | Checkpoint |
|-------|----------|----------|------------|
| **CTM (Ours)** | ResNet-18 (scale-4) | **85.10%** | `r18-4_i50_d512_m20/checkpoint.pt` |
| CTM + Innovations | ResNet-18 | 83.42% | `r18-4_i50_d512_m20_frob_norm/checkpoint.pt` |
| ViT-Tiny | ViT-Tiny | 78.35% | `vittiny_i50_d512_m30_sd6_mhd32/checkpoint.pt` |

**Training Details:**
- Dataset: RAF-DB (29,672 images, 7 emotion classes)
- Input size: 224×224 (grayscale converted to RGB)
- Optimizer: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 32
- Epochs: 100 (early stopping on validation)
- Augmentations: Random horizontal flip, color jitter, mixup (α=0.2)

## Repo Structure

```
├── tasks/
│   ├── image_classification/
│   │   ├── train.py                          # Main training script
│   │   ├── finetune.py                       # Fine-tuning utilities
│   │   ├── plotting.py                       # Visualization utilities
│   │   ├── analysis/
│   │   │   ├── run_imagenet_analysis.py      # Classification reports, latent viz
│   │   │   └── outputs/                      # Generated analyses
│   │   └── scripts/
│   │       ├── train_rafdb_ctm_innovations.sh
│   │       └── run_rafdb_ablation.sh
│   ├── parity/                               # Parity computation task
│   ├── qamnist/                             # Quantized MNIST QA
│   ├── mazes/                                # 2D maze solving
│   ├── sort/                                 # List sorting
│   └── rl/                                   # Reinforcement learning
├── models/
│   ├── ctm.py                               # Main CTM model
│   ├── modules.py                           # NLM, Synapse UNET, etc.
│   ├── resnet.py                            # ResNet backbone wrapper
│   ├── vittiny.py                           # ViT-Tiny backbone
│   └── clip_adapter.py                      # CLIP feature extractor
├── calculate_gflops.py                      # GFLOPS analysis tool
├── calculate_gflops_detailed.py             # Per-component breakdown
├── analysis_checkpoints/                    # Pre-trained checkpoints
│   └── rafdb/
│       ├── r18-4_i50_d512_m20/
│       │   └── checkpoint.pt               # 85.10% accuracy
│       └── ...
├── GFLOPS_*.txt/.md                         # Computational analysis reports
└── utils/
    ├── losses.py                           # Loss functions
    └── evaluation.py                       # Evaluation metrics
```

## Model Configuration (RAF-DB)

```python
ContinuousThoughtMachine(
    iterations=50,           # Internal "thought" ticks
    d_model=512,             # Latent dimension
    d_input=256,             # Attention projection dimension
    heads=8,                 # Attention heads
    n_synch_out=256,         # Output synchronization neurons
    n_synch_action=256,      # Action synchronization neurons
    synapse_depth=4,         # U-Net depth for synapse layer
    memory_length=5,         # History length for NLMs
    deep_nlms=True,          # Use 2-layer MLPs for NLMs
    backbone_type='resnet18-4',  # Feature extractor
    pretrained_backbone='imagenet',
    out_dims=7,              # Number of emotion classes
    neuron_select_type='random-pairing',
    memory_write_type='fifo',  # or 'attention'
)
```

## Computing Requirements

**Per-iteration FLOPs (RAF-DB config, 224×224 input):**
- CTM internal loop: 2.79 MFLOPs
- Backbone (ResNet-18): 13.74 GFLOPs
- Total per iteration: 13.74 GFLOPs
- Total for 50 iterations: 13.88 GFLOPs

See [GFLOPS_FINAL_SUMMARY.txt](GFLOPS_FINAL_SUMMARY.txt) for detailed breakdown.

## Analysis Tools

### Latent Space Visualization

```bash
python tasks/image_classification/analysis/run_imagenet_analysis.py \
  --actions latent_viz \
  --checkpoint analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt
```

Generates PCA/UMAP trajectories showing how the CTM's latent state evolves over iterations.

### Synchronization Matrix

Visualizes pairwise neuron synchrony at different thought ticks:

![Synchronization Diagram](synchronization_diagram.png)

### Temporal Accuracy

Evaluates how predictions change across internal iterations:

```bash
python evaluate_temporal_accuracy.py --checkpoint <path>
```

## Key Hyperparameters

### CTM Architecture
- `d_model`: 512 (latent dimension)
- `synapse_depth`: 4 (U-Net layers)
- `memory_length`: 5 (NLM history)
- `iterations`: 50 (thought ticks)

### Training
- Learning rate: 1e-4 (AdamW)
- Weight decay: 0.05
- Batch size: 32
- Label smoothing: 0.1
- Mixup α: 0.2

### Data Augmentation
- Random resized crop (224×224)
- Random horizontal flip
- Color jitter (brightness=0.4, contrast=0.4, saturation=0.2)
- Cutout (for regularization)

## Pretrained Models

Download checkpoints from the [analysis_checkpoints/](analysis_checkpoints/) directory or train from scratch:

```bash
# RAF-DB (ResNet-18)
python -m tasks.image_classification.train \
  --dataset rafdb \
  --backbone_type resnet18-4 \
  --epochs 100

# With CTM innovations (Frobenius norm regularization)
python -m tasks.image_classification.train \
  --dataset rafdb \
  --backbone_type resnet18-4 \
  --ctm_innovations True
```

## Citation

If you use this code, please cite the original CTM paper:

```bibtex
@article{ctm2025,
  title={Continuous Thought Machines},
  author={Sakana AI},
  journal={arXiv preprint arXiv:2505.05522},
  year={2025}
}
```

## License

This repository is released under the [MIT License](LICENSE).

---

**Additional Resources:**
- [Technical Report](https://arxiv.org/abs/2505.05522)
- [Interactive Demo](https://pub.sakana.ai/ctm/)
- [Blog Post](https://sakana.ai/ctm/)
- [Tutorial Notebook](examples/01_mnist.ipynb)

