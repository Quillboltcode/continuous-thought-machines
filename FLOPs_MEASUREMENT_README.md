# CTM FLOPs Measurement Script

This script measures and compares FLOPs (floating point operations) for Continuous Thought Machine (CTM) models using two methods:

1. **Analytical** - Formula-based calculation from `models/utils.py:compute_ctm_flops`
2. **fvcore** - Actual operator-level measurement using Facebook's fvcore library

## Usage

```bash
# Basic usage with small model
python measure_flops.py --iterations 5 --d_model 64 --backbone_type none

# With ResNet backbone
python measure_flops.py --iterations 5 --d_model 128 --backbone_type resnet18-2 --batch_size 4

# Save results to JSON
python measure_flops.py --save_results results.json

# Quiet mode (compact output)
python measure_flops.py --quiet

# Use CUDA if available
python measure_flops.py --device cuda
```

## Output

The script produces:

1. **Model parameter count** - Total and trainable parameters
2. **Analytical FLOPs** - Calculated based on model architecture formulas
3. **fvcore FLOPs** - Actual measured FLOPs including all tensor operations
4. **Ratio** - Comparison between methods
5. **Timing** - Average forward pass time
6. **JSON export** - Optional structured results for further analysis

## Understanding the Results

- **Analytical**: Estimates FLOPs based on matrix multiplication sizes and loop iterations. May exclude small operations (indexing, reshapes, element-wise ops).
- **fvcore**: Measures FLOPs at the operator level during actual execution. More comprehensive and accurate.
- **Typical ratio**: fvcore is often 20-500× higher because it counts every operation, not just major matrix multiplies.

## Requirements

- PyTorch (already in project)
- fvcore (for operator-level profiling): `pip install fvcore`

## Example Output

```
Device: cpu

Configuration:
  T=5, D=128, d_in=256, heads=4
  D_out=32, D_action=32
  synapse_depth=1, M=10
  backbone=resnet18-2, neuron_select=random-pairing

Parameters: total=1,278,040, trainable=1,278,040

============================================================
FLOPs MEASUREMENT
============================================================

Analytical Calculation Results:
  Total FLOPs: 1.10e+07
  Per iteration: 2.03e+06
  Loop total (T=5): 1.02e+07

fvcore Results:
  Total FLOPs: 2.40e+08
  Top ops: linear(9.05e+07), conv(1.46e+08), ...

============================================================
COMPARISON
============================================================
  Analytical: 1.10e+07
  fvcore:     2.40e+08  (x21.75)
```

## Notes

- The script initializes lazy modules with a dummy forward pass before measurement
- For backbone models, analytical FLOPs include rough estimates for ResNet/ViT
- Timing is averaged over 100 forward passes after 10 warmup iterations
- Results saved to JSON include full config and metrics for reproducibility
