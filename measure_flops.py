#!/usr/bin/env python3
"""
FLOPs Measurement Script for Continuous Thought Machine (CTM)

This script:
1. Instantiates a CTM model with specified parameters
2. Runs a forward pass with synthetic data
3. Measures actual FLOPs using fvcore profiler
4. Compares against analytical estimates from compute_ctm_flops
"""

import torch
import argparse
import time
import json
from models.ctm import ContinuousThoughtMachine
from models.utils import compute_ctm_flops, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='Measure CTM FLOPs', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  # Small model
  python measure_flops.py --iterations 5 --d_model 64 --backbone_type none
  
  # With ResNet backbone
  python measure_flops.py --iterations 5 --d_model 128 --backbone_type resnet18-2
  
  # Save results
  python measure_flops.py --save_results results.json
""")
    
    # Model architecture parameters
    parser.add_argument('--iterations', type=int, default=10, help='Number of internal thought ticks (T) [default: 10]')
    parser.add_argument('--d_model', type=int, default=128, help='Core dimensionality of CTM latent space (D) [default: 128]')
    parser.add_argument('--d_input', type=int, default=256, help='Dimensionality of projected attention outputs [default: 256]')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads [default: 4]')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons for output synchronisation (D_out) [default: 32]')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons for action synchronisation (D_action) [default: 32]')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of synapse model (1=MLP, >1=U-Net) [default: 1]')
    parser.add_argument('--memory_length', type=int, default=10, help='History length for Neuron-Level Models (M) [default: 10]')
    parser.add_argument('--deep_nlms', action='store_true', default=True, help='Use deeper (2-layer) NLMs [default: True]')
    parser.add_argument('--no_deep_nlms', action='store_false', dest='deep_nlms', help='Disable deep NLMs')
    parser.add_argument('--memory_hidden_dims', type=int, default=64, help='Hidden dimension size for deep NLMs [default: 64]')
    parser.add_argument('--do_layernorm_nlm', action='store_true', default=False, help='Apply LayerNorm within NLMs [default: False]')
    parser.add_argument('--backbone_type', type=str, default='none',
                       help='Backbone type: none, resnet18-2, resnet34-2, resnet50-2, shallow-wide, parity_backbone, vit-tiny [default: none]')
    parser.add_argument('--pretrained_backbone', type=str, default='none', help='Pretrained backbone: none, imagenet [default: none]')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                       help='Positional embedding: none, learnable-fourier, multi-learnable-fourier, custom-rotational [default: none]')
    parser.add_argument('--out_dims', type=int, default=10, help='Output dimension size [default: 10]')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate [default: 0.0]')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing',
                       help='Neuron selection: first-last, random, random-pairing [default: random-pairing]')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Self-to-self pairs for random-pairing [default: 0]')
    
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size [default: 4]')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels [default: 3]')
    parser.add_argument('--input_height', type=int, default=32, help='Input height [default: 32]')
    parser.add_argument('--input_width', type=int, default=32, help='Input width [default: 32]')
    
    # Measurement parameters
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: cuda or cpu [default: auto]')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Optional path to save results as JSON')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, only show final comparison')
    parser.add_argument('--use_fvcore_for_analytical', action='store_true',
                       help='Use fvcore to measure backbone/linears in analytical calculation')
    
    return parser.parse_args()


def count_flops_fvcore(model, input_tensor):
    """Count FLOPs using fvcore's FlopCountAnalysis."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("  fvcore not available. Install with: pip install fvcore")
        return None
    
    model.eval()
    with torch.no_grad():
        flop_analysis = FlopCountAnalysis(model, input_tensor)
        total_flops = flop_analysis.total()
    
    print(f"\nfvcore Results:")
    print(f"  Total FLOPs: {total_flops:.2e}")
    
    try:
        flop_table = flop_analysis.by_operator()
        if hasattr(flop_table, 'sort'):
            sorted_ops = list(flop_table.sort(reverse=True).items())[:10]
        else:
            sorted_ops = sorted(flop_table.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n  Top operations by FLOPs:")
        for op, count in sorted_ops:
            print(f"    {op}: {count:.2e}")
    except Exception:
        pass
    
    return total_flops


def count_flops_analytical(model, input_tensor, use_fvcore_for_components=False):
    """Count FLOPs using analytical formula from compute_ctm_flops."""
    input_shape = tuple(input_tensor.shape[1:])
    flops_dict = compute_ctm_flops(model, input_shape, batch_size=input_tensor.shape[0], use_fvcore=use_fvcore_for_components)
    
    print(f"\nAnalytical Calculation Results:")
    print(f"  Total FLOPs: {flops_dict['total']:.2e}")
    print(f"  Per iteration: {flops_dict['per_iteration']:.2e}")
    print(f"  Loop total (T={model.iterations}): {flops_dict['loop_total']:.2e}")
    print(f"  Breakdown:")
    print(f"    Outside loop: {flops_dict['breakdown']['outside_loop']:.2e}")
    print(f"    Inside loop (per iter): {flops_dict['breakdown']['inside_loop_per_iter']:.2e}")
    print(f"    Inside loop (total): {flops_dict['breakdown']['inside_loop_total']:.2e}")
    
    return flops_dict['total']


def main():
    args = parse_args()
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not args.quiet:
        print(f"Device: {device}")
        print("\nConfiguration:")
        print(f"  T={args.iterations}, D={args.d_model}, d_in={args.d_input}, heads={args.heads}")
        print(f"  D_out={args.n_synch_out}, D_action={args.n_synch_action}")
        print(f"  synapse_depth={args.synapse_depth}, M={args.memory_length}")
        print(f"  backbone={args.backbone_type}, neuron_select={args.neuron_select_type}")
    
    # Create model
    if not args.quiet:
        print("\nInstantiating CTM...")
    model = ContinuousThoughtMachine(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_nlms,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_layernorm_nlm,
        backbone_type=args.backbone_type,
        pretrained_backbone=args.pretrained_backbone,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=args.out_dims,
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    )
    model = model.to(device)
    
    # Create synthetic input
    input_shape = (args.batch_size, args.input_channels, args.input_height, args.input_width)
    if not args.quiet:
        print(f"\nSynthetic input: {input_shape}")
    x = torch.randn(input_shape).to(device)
    
    # Initialize lazy modules
    if not args.quiet:
        print("Initializing lazy modules...")
    model.eval()
    with torch.no_grad():
        _ = model(x)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    if not args.quiet:
        print(f"\nParameters: total={total_params:,}, trainable={trainable_params:,}")
    
    # Measure FLOPs
    if not args.quiet:
        print("\n" + "="*60)
        print("FLOPs MEASUREMENT")
        print("="*60)
    
    analytical = count_flops_analytical(model, x, use_fvcore_for_components=args.use_fvcore_for_analytical)
    fvcore = count_flops_fvcore(model, x)
    
    # Comparison
    if not args.quiet:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"  Analytical: {analytical:.2e}")
        if fvcore is not None:
            print(f"  fvcore:     {fvcore:.2e}  (x{fvcore/analytical:.2f})")
            print()
            comp_method = "for components" if args.use_fvcore_for_analytical else "only for core ops"
            print(f"  Note: analytical {comp_method};")
            print("        fvcore counts all operations including indexing, reshapes, layer norm, etc.")
    
    # Timing
    if not args.quiet:
        print("\n" + "="*60)
        print("TIMING")
        print("="*60)
    
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(100):
            predictions, certainties, synch_out = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()
    
    avg_ms = (end - start) / 100 * 1000
    if not args.quiet:
        print(f"  Avg forward: {avg_ms:.2f} ms")
        print(f"  Per iteration: {avg_ms/args.iterations:.2f} ms")
    
    # Save results
    if args.save_results:
        results = {
            'config': {
                'iterations': args.iterations,
                'd_model': args.d_model,
                'd_input': args.d_input,
                'heads': args.heads,
                'n_synch_out': args.n_synch_out,
                'n_synch_action': args.n_synch_action,
                'synapse_depth': args.synapse_depth,
                'memory_length': args.memory_length,
                'deep_nlms': args.deep_nlms,
                'memory_hidden_dims': args.memory_hidden_dims,
                'backbone_type': args.backbone_type,
                'out_dims': args.out_dims,
                'batch_size': args.batch_size,
            },
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
            },
            'flops': {
                'analytical': analytical,
                'fvcore': fvcore,
            },
            'timing_ms': {
                'avg_forward': avg_ms,
                'per_iteration': avg_ms / args.iterations,
            },
        }
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"\nResults saved to: {args.save_results}")
    
    if args.quiet:
        print(f"Analytical: {analytical:.2e}")
        if fvcore is not None:
            print(f"fvcore: {fvcore:.2e}")
        else:
            print("fvcore: N/A")
        print(f"Time per forward: {avg_ms:.2f} ms")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
