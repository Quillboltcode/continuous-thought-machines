# CTM Model GFLOPS Analysis Report

## Executive Summary

This report analyzes the floating-point operations (FLOPs) per internal loop (thought tick) for the Continuous Thought Machine (CTM) model, with and without backbone feature extraction.

### Key Findings

| Configuration | GFLOPs/loop | GFLOPs/Total (50 iters) |
|---------------|-------------|-------------------------|
| **CTM (no backbone)** | **0.0028 G** (2.79 M) | **0.140 G** (139.69 M) |
| **CTM + ResNet-18** (112px) | **3.437 G** | **175.30 G** |
| Ratio (with/no) | **~1,230x** | **~1,256x** |

- **Per-loop FLOPs (no backbone):** 2.79 MFLOPs (0.0028 GFLOPs)
- **Per-loop FLOPs (with ResNet-18):** 3.44 GFLOPs
- **Backbone share:** 99.9% of per-loop FLOPs (3.43 GFLOPs / 3.44 GFLOPs)

---

## 1. Model Configuration

```
d_model:              512
d_input:              256
heads:                8
n_synch_out:          256
n_synch_action:       256
synapse_depth:        4
memory_length:        5
iterations:           50
backbone_type:        resnet18-4
img_size:             112 (RAF-DB typical)
```

---

## 2. Per-Loop Component Breakdown (With Backbone)

| Component | GFLOPs | % of Non-Backbone |
|-----------|--------|-------------------|
| **Backbone (ResNet-18)** | **3,434.45 M** | - |
| Synchronization (action) | 0.003 M | 0.1% |
| **Attention computation** | **0.328 M** | **11.7%** |
| - Query projection | 0.262 M | - |
| - Q @ K^T | 0.031 M | - |
| - Softmax | 0.0003 M | - |
| - Softmax @ V | 0.031 M | - |
| **Synapse layer (depth=4)** | **1.662 M** | **59.5%** |
| **Neuron-Level Models** | **0.794 M** | **28.4%** |
| - Deep NLMs (2-layer MLP per neuron) | - | - |
| Synchronization (output) | 0.003 M | 0.1% |
| Output projection | 0.004 M | 0.1% |
| **Total per loop** | **3,437.24 M** | **100%** |

---

## 3. Detailed Component Analysis

### 3.1 Backbone (ResNet-18, scale-4)
- **FLOPs:** 3.43 GFLOPs per forward pass
- **Architecture:** 
  - Conv1: 3→64, 3×3, stride 1
  - MaxPool: 3×3, stride 2
  - Layer1 (2 blocks): 64→64, no downsampling
  - Layer2 (2 blocks): 64→128, stride 2
  - Layer3 (2 blocks): 128→256, stride 2
  - Layer4 (2 blocks): 256→512, stride 2
- **Output:** 512 channels × 7 × 7 (for 112px input)
- **Share:** 99.9% of total per-loop FLOPs

### 3.2 Attention Computation
- **Query projection:** Linear(d_model=512 → d_input=256)
  - FLOPs: 262,400 (0.262 M)
- **Q @ K^T:** (1, 256) @ (256, 64) where N_kv ≈ 64 spatial positions
  - FLOPs: 32,768 (0.031 M)
- **Softmax:** Over 64 positions
  - FLOPs: 320 (0.0003 M)
- **Softmax @ V:** (1, 64) @ (64, 256)
  - FLOPs: 32,768 (0.031 M)
- **Total:** 327,968 FLOPs (0.328 M)

### 3.3 Synapse Layer (f_theta1)
- **Type:** U-Net with depth=4
- **Structure:**
  - First projection: (d_input + d_model=768) → d_model=512
  - Down blocks: 512→320→192→112
  - Up blocks: 112→192→320→512
  - Skip connections with LayerNorm
- **FLOPs:** 1.662 M (59.5% of non-backbone)
- **Purpose:** Information sharing between neurons (synaptic connections)

### 3.4 Neuron-Level Models (NLMs)
- **Configuration:** Single NLM for all 512 neurons
- **Architecture (deep):**
  - Layer 1: Linear(memory_length=5 → 2×hidden=32) + GLU
    - FLOPs: 542,720 (0.543 M)
  - Layer 2: Linear(hidden=16 → 2) + GLU
    - FLOPs: 250,880 (0.251 M)
- **Total:** 793,600 FLOPs (0.794 M, 28.4% of non-backbone)
- **Purpose:** Per-neuron temporal processing of activation history

### 3.5 Synchronization
- **Action sync:** 3,328 FLOPs (0.003 M, 0.1%)
  - Pairwise multiply + exponential decay recurrence
- **Output sync:** 3,328 FLOPs (0.003 M, 0.1%)
  - Same computation for different neuron subset
- **Purpose:** Compute dot products between neuron activation traces

### 3.6 Output Projection
- **Linear:** synch_representation_size=256 → out_dims=7 (RAF-DB classes)
- **FLOPs:** 3,591 (0.004 M, 0.1%)

---

## 4. Comparison: With vs. Without Backbone

| Metric | With Backbone | Without Backbone | Ratio |
|--------|---------------|------------------|-------|
| **Per-loop FLOPs** | 3.437 G | 2.79 M | **1,230×** |
| **Total FLOPs (50 iters)** | 175.30 G | 139.69 M | **1,256×** |
| **Backbone contribution** | 99.9% | 0% | - |

### Key Insight
The backbone (feature extractor) dominates the computational cost, accounting for **>99.9%** of FLOPs in the full CTM model. The CTM's internal recurrent computation (without backbone) is extremely lightweight at **2.79 MFLOPs per iteration**.

---

## 5. Computational Complexity by Component Type

| Component Type | FLOPs (M) | % of CTM-only (no backbone) |
|----------------|-----------|------------------------------|
| Synapse layer (recurrent) | 1.662 | 59.5% |
| Neuron-Level Models | 0.794 | 28.4% |
| Attention | 0.328 | 11.7% |
| Synchronization | 0.007 | 0.3% |
| Output projection | 0.004 | 0.1% |
| **CTM Total** | **2.791** | **100%** |

---

## 6. Implications

### 6.1 Efficiency Considerations
- **CTM core is efficient:** 2.79 MFLOPs/iteration is minimal compared to modern vision backbones
- **Backbone choice matters:** Different backbones dramatically affect total cost:
  - ResNet-18: ~3.4 GFLOPs (per forward pass)
  - ResNet-50: ~8.2 GFLOPs (2.4× more)
  - ViT-Base: ~17.6 GFLOPs (5.2× more)
- **Iterations scale linearly:** 50 iterations × 2.79 MFLOPs = 139.7 MFLOPs (negligible vs. backbone)

### 6.2 Bottlenecks
1. **Backbone feature extraction** (dominant: 99.9%)
2. **Synapse layer** (largest CTM component: 59.5% of CTM-only)
3. **Neuron-Level Models** (second largest: 28.4% of CTM-only)

### 6.3 Optimization Opportunities
- Use lighter backbones (e.g., MobileNet, EfficientNet-lite) for edge deployment
- Reduce synapse depth (depth=2 instead of 4 cuts ~40% of CTM FLOPs)
- Reduce d_model or memory_length
- Use shallow NLMs (linear instead of deep) for faster inference
- Reduce number of iterations (trade-off with reasoning quality)

---

## 7. Conclusion

The CTM's computational profile is dominated by the backbone feature extractor (99.9% of FLOPs), while the recurrent "thought" mechanism itself is remarkably lightweight at **2.79 MFLOPs per iteration**. This makes CTM an efficient reasoning layer that can be paired with various backbones depending on the accuracy-computation trade-off required for the task.

For the RAF-DB configuration (ResNet-18, 112px, 50 iterations):
- **Total per-image FLOPs:** 175.3 GFLOPs
- **Backbone:** 175.0 GFLOPs (99.8%)
- **CTM reasoning:** 0.3 GFLOPs (0.2%)

The CTM adds minimal overhead while enabling iterative reasoning, making it an efficient approach for tasks requiring temporal processing and multi-step inference.
