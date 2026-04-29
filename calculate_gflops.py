#!/usr/bin/env python3
"""
GFLOPS calculation for CTM (Continuous Thought Machine) model.
Calculates floating point operations per 'loop' (one internal iteration/tick).

Both with and without backbone configurations are analyzed.
"""

import torch
import torch.nn as nn
import math
from models.ctm import ContinuousThoughtMachine

def count_linear_flops(in_features, out_features, batch_size=1, num_instances=1):
    """Count FLOPs for a linear layer: y = xW + b
    
    Args:
        in_features: input dimension
        out_features: output dimension  
        batch_size: batch size
        num_instances: number of parallel instances (e.g., number of neurons for SuperLinear)
    
    Returns:
        Total FLOPs (multiply-add counted as 2 operations)
    """
    # Each output element requires in_features multiply-adds = 2*in_features FLOPs
    # Plus 1 FLOP for bias addition per output element
    # Total: batch_size * num_instances * out_features * (2*in_features + 1)
    return batch_size * num_instances * out_features * (2 * in_features + 1)

def count_matmul_flops(m, n, k, batch_size=1):
    """Count FLOPs for matrix multiplication of (B,m,n) @ (B,n,k) -> (B,m,k)
    
    Returns:
        Total FLOPs (multiply-add counted as 2 operations)
    """
    return batch_size * m * n * k * 2

def count_conv2d_flops(in_channels, out_channels, kernel_h, kernel_w, out_h, out_w, batch_size=1):
    """Count FLOPs for a Conv2d layer.
    
    Returns:
        Total FLOPs (multiply-add counted as 2 operations)
    """
    return batch_size * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2

def count_layernorm_flops(normalized_shape, batch_size=1, num_instances=1):
    """Count FLOPs for LayerNorm.
    
    Includes: mean (1 FLOP per element), variance (2 FLOPs per element),
    normalization (2 FLOPs per element), scale+shift (2 FLOPs per element)
    Approximately 7 FLOPs per element.
    """
    if isinstance(normalized_shape, int):
        elements = normalized_shape
    else:
        elements = 1
        for s in normalized_shape:
            elements *= s
    return batch_size * num_instances * elements * 7

def count_gelu_flops(features, batch_size=1, num_instances=1):
    """Count FLOPs for GELU activation (approximated with tanh)."""
    # GELU involves: exp, division, multiplication, tanh, etc.
    # Roughly ~20-30 FLOPs per element, using 25 as estimate
    return batch_size * num_instances * features * 25

def count_linear_with_glu_flops(in_features, out_dims, batch_size=1, num_instances=1):
    """Count FLOPs for Linear + GLU (Gated Linear Unit).
    
    GLU: out = A * sigmoid(B) where A and B are halves of linear output
    """
    # Linear: input * (2*out_dims + 1) FLOPs per output element
    linear_flops = count_linear_flops(in_features, 2*out_dims, batch_size, num_instances)
    # GLU: sigmoid on half (~20 FLOPs per element) + multiply (~1 FLOP per element)
    glu_flops = batch_size * num_instances * out_dims * 21
    return linear_flops + glu_flops


def calculate_ctm_loop_with_backbone(model, d_input, d_model, memory_length, n_synch_action, 
                                      n_synch_out, heads, synapse_depth, iterations, 
                                      backbone_type='none', backbone_flops_per_loop=0,
                                      batch_size=1, group_count=0, memory_write_type='fifo'):
    """
    Calculate GFLOPS for one CTM internal loop (iteration) WITH backbone.
    
    This calculates FLOPs for one "thought tick" or internal recurrence step.
    """
    total_flops = 0
    
    # --- Backbone feature extraction (once per loop, operates on same features) ---
    # Note: backbone runs once per input, but if called in loop it would be this many flops
    total_flops += backbone_flops_per_loop
    
    # --- Compute synchronisation for action (attn interaction) ---
    # compute_synchronisation for 'action' type
    # For random-pairing (default): element-wise multiply of selected neurons
    # n_synch_action pairs, each: multiply (1 FLOP)
    synch_action_mult_flops = batch_size * n_synch_action * 1  # multiply
    
    # decay computation: r * decay + pairwise_product (2 FLOPs per element: mult+add)
    # plus division and sqrt for final sync (approx 10 FLOPs per element)
    synch_action_recur_flops = batch_size * n_synch_action * (2 + 10)
    
    total_flops += synch_action_mult_flops + synch_action_recur_flops
    
    # --- Attention computation ---
    # q_proj: Linear(d_model -> d_input)
    total_flops += count_linear_flops(d_model, d_input, batch_size, 1)
    
    # MultiheadAttention: query @ key^T and then @ value
    # q: (B, 1, d_input), k: (B, N_kv, d_input), v: (B, N_kv, d_input)
    # For attention, N_kv depends on backbone. Estimate from typical feature map.
    # With ResNet18-4 on 224x224: feature map is 7x7 = 49 positions, C=512
    # After projection to d_input: N_kv = 49 (spatial positions)
    # But d_input varies. Let's use a conservative estimate of N_kv = 64
    N_kv = 64  # approximate number of key-value pairs from backbone features
    
    # Q @ K^T: (B, 1, d_input) @ (B, d_input, N_kv) -> (B, 1, N_kv)
    total_flops += count_matmul_flops(1, d_input, N_kv, batch_size)
    # Softmax on attention weights: ~5*N_kv FLOPs per query position
    total_flops += batch_size * 1 * N_kv * 5
    # Attn_out = softmax @ V: (B, 1, N_kv) @ (B, N_kv, d_input) -> (B, 1, d_input)
    total_flops += count_matmul_flops(1, N_kv, d_input, batch_size)
    
    attn_out_dim = d_input
    
    # --- Synapse layer (f_theta1) ---
    # Input: concatenated(attn_out, activated_state) = d_input + d_model
    synapse_input_dim = attn_out_dim + d_model
    
    if synapse_depth == 1:
        # Simple: Dropout + Linear + GLU + LayerNorm
        # Linear: (d_input + d_model) -> 2*d_model, then GLU -> d_model
        total_flops += count_linear_with_glu_flops(synapse_input_dim, d_model, batch_size, 1)
        total_flops += count_layernorm_flops(d_model, batch_size, 1)
    else:
        # U-Net synapse: more complex
        # Minimum width 16, depth layers
        # Approximate with multiple linear layers
        # First projection: (d_input+d_model) -> d_model
        total_flops += count_linear_flops(synapse_input_dim, d_model, batch_size, 1)
        total_flops += count_layernorm_flops(d_model, batch_size, 1)
        
        # Down and up blocks: each does dropout+linear+layernorm+glu
        width = d_model
        min_width = 16
        for i in range(synapse_depth - 1):
            next_width = max(min_width, int(width * (synapse_depth - 2 - i) / (synapse_depth - 1)))
            if next_width < 1:
                next_width = min_width
            # Down block
            total_flops += count_linear_flops(width, next_width, batch_size, 1)
            total_flops += count_layernorm_flops(next_width, batch_size, 1)
            # Up block (in reverse)
            total_flops += count_linear_flops(next_width, width, batch_size, 1)
            total_flops += count_layernorm_flops(width, batch_size, 1)
            width = next_width
    
    # --- Memory write ---
    if memory_write_type == 'attention':
        # memory_query_proj: Linear(d_model -> d_model)
        total_flops += count_linear_flops(d_model, d_model, batch_size, 1)
        # memory_key_proj: Linear(d_model -> d_model) 
        total_flops += count_linear_flops(d_model, d_model, batch_size, memory_length)
        # memory_value_proj: Linear(d_model -> d_model)
        total_flops += count_linear_flops(d_model, d_model, batch_size, memory_length)
        
        # Attention over memory: (B, 1, d_model) @ (B, d_model, T) -> (B, 1, T)
        # Then @ (B, T, d_model) -> (B, 1, d_model)
        total_flops += count_matmul_flops(1, d_model, memory_length, batch_size)
        total_flops += batch_size * 1 * memory_length * 5  # softmax
        total_flops += count_matmul_flops(1, memory_length, d_model, batch_size)
    else:
        # FIFO: just concatenation and slicing, negligible FLOPs
        pass
    
    # --- Neuron Level Models (NLMs) ---
    if group_count > 0:
        group_size = d_model // group_count
        # Each group: SuperLinear (deep or shallow)
        # Deep: Linear(memory_length -> 2*hidden) + GLU + Linear(hidden -> 2) + GLU
        # Shallow: Linear(memory_length -> 2) + GLU
        
        # Assume deep NLMs (as per paper)
        memory_hidden_dims = 16  # default from model
        
        # First linear: (memory_length -> 2*memory_hidden_dims)
        total_flops += group_count * count_linear_with_glu_flops(
            memory_length, 2*memory_hidden_dims, batch_size, group_size
        )
        # Second linear: (memory_hidden_dims -> 2)
        total_flops += group_count * count_linear_with_glu_flops(
            memory_hidden_dims, 2, batch_size, group_size
        )
    else:
        # Single NLM for all neurons
        # Deep: two layers with GLU
        memory_hidden_dims = 16  # typical value
        total_flops += count_linear_with_glu_flops(
            memory_length, 2*memory_hidden_dims, batch_size, d_model
        )
        total_flops += count_linear_with_glu_flops(
            memory_hidden_dims, 2, batch_size, d_model
        )
    
    # --- Compute synchronisation for output ---
    # Similar to action sync but for n_synch_out
    synch_out_mult_flops = batch_size * n_synch_out * 1
    synch_out_recur_flops = batch_size * n_synch_out * (2 + 10)
    total_flops += synch_out_mult_flops + synch_out_recur_flops
    
    # --- Output projection ---
    # Linear(synch_representation_size -> out_dims)
    # Using random-pairing (the default in CTM paper)
    synch_out_size = n_synch_out  # random-pairing
    
    total_flops += count_linear_flops(synch_out_size, 7, batch_size, 1)  # 7 classes for RAF-DB
    
    return total_flops


def calculate_ctm_loop_no_backbone(d_model, d_input, memory_length, n_synch_action, 
                                   n_synch_out, heads, synapse_depth, 
                                   neuron_select_type='random-pairing',
                                   group_count=0, memory_write_type='fifo',
                                   batch_size=1):
    """Calculate FLOPs for one CTM loop WITHOUT backbone."""
    total_flops = 0
    
    # --- Synchronisation for action ---
    if neuron_select_type == 'random-pairing':
        synch_action_mult = batch_size * n_synch_action * 1
    else:
        # first-last or random: need full pairwise
        raise NotImplementedError("Only random-pairing supported in this calc")
    
    synch_action_recur = batch_size * n_synch_action * 12  # mult+add + sqrt/div
    total_flops += synch_action_mult + synch_action_recur
    
    # --- Attention ---
    total_flops += count_linear_flops(d_model, d_input, batch_size, 1)
    
    N_kv = 64
    total_flops += count_matmul_flops(1, d_input, N_kv, batch_size)
    total_flops += batch_size * 1 * N_kv * 5
    total_flops += count_matmul_flops(1, N_kv, d_input, batch_size)
    
    attn_out_dim = d_input
    synapse_input_dim = attn_out_dim + d_model
    
    # --- Synapse ---
    if synapse_depth == 1:
        total_flops += count_linear_with_glu_flops(synapse_input_dim, d_model, batch_size, 1)
        total_flops += count_layernorm_flops(d_model, batch_size, 1)
    else:
        total_flops += count_linear_flops(synapse_input_dim, d_model, batch_size, 1)
        total_flops += count_layernorm_flops(d_model, batch_size, 1)
        min_width = 16
        width = d_model
        for i in range(synapse_depth - 1):
            next_width = max(min_width, int(width * (synapse_depth - 2 - i) / (synapse_depth - 1)))
            if next_width < 1:
                next_width = min_width
            total_flops += count_linear_flops(width, next_width, batch_size, 1)
            total_flops += count_layernorm_flops(next_width, batch_size, 1)
            total_flops += count_linear_flops(next_width, width, batch_size, 1)
            total_flops += count_layernorm_flops(width, batch_size, 1)
            width = next_width
    
    # --- Memory write ---
    if memory_write_type == 'attention':
        total_flops += count_linear_flops(d_model, d_model, batch_size, 1)
        total_flops += count_linear_flops(d_model, d_model, batch_size, memory_length)
        total_flops += count_linear_flops(d_model, d_model, batch_size, memory_length)
        total_flops += count_matmul_flops(1, d_model, memory_length, batch_size)
        total_flops += batch_size * 1 * memory_length * 5
        total_flops += count_matmul_flops(1, memory_length, d_model, batch_size)
    
    # --- NLMs ---
    if group_count > 0:
        group_size = d_model // group_count
        memory_hidden_dims = 16
        total_flops += group_count * count_linear_with_glu_flops(memory_length, 2*memory_hidden_dims, batch_size, group_size)
        total_flops += group_count * count_linear_with_glu_flops(memory_hidden_dims, 2, batch_size, group_size)
    else:
        memory_hidden_dims = 16
        total_flops += count_linear_with_glu_flops(memory_length, 2*memory_hidden_dims, batch_size, d_model)
        total_flops += count_linear_with_glu_flops(memory_hidden_dims, 2, batch_size, d_model)
    
    # --- Synchronisation for output ---
    if neuron_select_type == 'random-pairing':
        synch_out_mult = batch_size * n_synch_out * 1
        synch_out_size = n_synch_out
    else:
        raise NotImplementedError("Only random-pairing supported in this calc")
    
    synch_out_recur = batch_size * n_synch_out * 12
    total_flops += synch_out_mult + synch_out_recur
    
    # --- Output projection ---
    total_flops += count_linear_flops(synch_out_size, 7, batch_size, 1)
    
    return total_flops


def count_resnet18_backbone_flops(img_size=224, num_classes=1000, output_scales=[4]):
    """
    Count FLOPs for ResNet-18 backbone (upto layer4 or specified scale).
    Uses kernel_size=3, stride=1 for conv1 (as per this project's ResNet).
    
    Returns FLOPs for processing one image through the backbone.
    """
    total_flops = 0
    
    H = W = img_size
    
    # conv1: 3->64, 3x3, stride 1, padding 1 (per project's ResNet)
    out_h, out_w = H, W  # 224 (no downsampling)
    total_flops += count_conv2d_flops(3, 64, 3, 3, out_h, out_w, 1)
    
    # maxpool 3x3 stride 2
    out_h, out_w = out_h//2, out_w//2  # 112
    
    # layer1: 2 BasicBlock, 64->64 (expansion=1)
    # Each block: 2 convs (3x3), no downsampling in layer1
    total_flops += 4 * count_conv2d_flops(64, 64, 3, 3, out_h, out_w, 1)
    
    # layer2: 2 blocks, 64->128, stride 2 on first block's conv1
    # After layer2, spatial dims are halved
    if 2 in output_scales or 3 in output_scales or 4 in output_scales:
        prev_h, prev_w = out_h, out_w
        out_h, out_w = out_h//2, out_w//2  # 56
        # Block1: conv1 64->128 stride2, conv2 128->128
        total_flops += count_conv2d_flops(64, 128, 3, 3, out_h, out_w, 1)  # conv1 stride2
        total_flops += count_conv2d_flops(128, 128, 3, 3, out_h, out_w, 1)  # conv2
        # downsample conv: 64->128 on reduced spatial dims
        total_flops += count_conv2d_flops(64, 128, 1, 1, out_h, out_w, 1)
        # Block2: conv1 128->128, conv2 128->128
        total_flops += 2 * count_conv2d_flops(128, 128, 3, 3, out_h, out_w, 1)
    
    # layer3: 2 blocks, 128->256
    if 3 in output_scales or 4 in output_scales:
        prev_h, prev_w = out_h, out_w
        out_h, out_w = out_h//2, out_w//2  # 28
        total_flops += count_conv2d_flops(128, 256, 3, 3, out_h, out_w, 1)  # conv1 stride2
        total_flops += count_conv2d_flops(256, 256, 3, 3, out_h, out_w, 1)  # conv2
        total_flops += count_conv2d_flops(128, 256, 1, 1, out_h, out_w, 1)  # downsample
        total_flops += 2 * count_conv2d_flops(256, 256, 3, 3, out_h, out_w, 1)  # block2
    
    # layer4: 2 blocks, 256->512
    if 4 in output_scales:
        prev_h, prev_w = out_h, out_w
        out_h, out_w = out_h//2, out_w//2  # 14 (for 224 input) or 7 (for 112)
        total_flops += count_conv2d_flops(256, 512, 3, 3, out_h, out_w, 1)  # conv1 stride2
        total_flops += count_conv2d_flops(512, 512, 3, 3, out_h, out_w, 1)  # conv2
        total_flops += count_conv2d_flops(256, 512, 1, 1, out_h, out_w, 1)  # downsample
        total_flops += 2 * count_conv2d_flops(512, 512, 3, 3, out_h, out_w, 1)  # block2
    
    return total_flops


def count_vit_tiny_backbone_flops(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3):
    """
    Count FLOPs for ViT-Tiny backbone.
    """
    total_flops = 0
    
    H = W = img_size
    num_patches = (H // patch_size) * (W // patch_size)  # 196 for 224/16
    
    # Patch embedding: conv 16x16, 3->192, stride 16
    total_flops += count_conv2d_flops(3, embed_dim, patch_size, patch_size, 
                                       H//patch_size, W//patch_size, 1)
    
    # Position embedding: negligible (just addition)
    
    # Transformer blocks
    seq_len = num_patches + 1  # +1 for class token
    
    for _ in range(depth):
        # LayerNorm
        total_flops += count_layernorm_flops(embed_dim, 1, seq_len)
        
        # Attention
        # Q, K, V projections: each Linear(embed_dim -> embed_dim)
        total_flops += 3 * count_linear_flops(embed_dim, embed_dim, 1, seq_len)
        
        # Attention: Q @ K^T -> (seq_len, seq_len)
        total_flops += count_matmul_flops(seq_len, embed_dim, seq_len, 1)
        # Softmax
        total_flops += seq_len * seq_len * 5
        # @ V
        total_flops += count_matmul_flops(seq_len, seq_len, embed_dim, 1)
        
        # Output projection
        total_flops += count_linear_flops(embed_dim, embed_dim, 1, seq_len)
        
        # MLP
        total_flops += count_linear_flops(embed_dim, embed_dim * 4, 1, seq_len)
        # GELU activation (approx)
        total_flops += seq_len * embed_dim * 4 * 25
        total_flops += count_linear_flops(embed_dim * 4, embed_dim, 1, seq_len)
        
        # LayerNorm after MLP
        total_flops += count_layernorm_flops(embed_dim, 1, seq_len)
    
    return total_flops


def get_backbone_flops(backbone_type, img_size=224):
    """
    Get FLOPs per forward pass for different backbones.
    """
    if backbone_type == 'none':
        return 0
    elif 'resnet18' in backbone_type:
        # Extract scale from backbone_type like 'resnet18-4'
        return count_resnet18_backbone_flops(img_size=img_size)
    elif backbone_type == 'vit-tiny':
        return count_vit_tiny_backbone_flops(img_size=img_size)
    elif 'shallow-wide' in backbone_type:
        # ShallowWide: conv 3->4096 (3x3), GLU, BN, conv 4096->4096 (3x3) groups=32, GLU, BN
        # For 224x224 input:
        # After first conv+stride2: 112x112, 4096 channels
        out_h, out_w = 112, 112
        total = count_conv2d_flops(3, 4096, 3, 3, out_h, out_w, 1)
        total += count_conv2d_flops(4096, 4096, 3, 3, out_h, out_w, 1, groups=32)
        return total
    elif 'parity_backbone' in backbone_type:
        # Parity backbone is just embedding, negligible for image tasks
        return 0
    else:
        print(f"Warning: Unknown backbone type '{backbone_type}', using 0 FLOPs")
        return 0


def format_flops(flops):
    """Format FLOPs as readable string."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} K"
    else:
        return f"{flops:.0f}"


def main():
    print("=" * 80)
    print("CTM Model FLOPS Analysis")
    print("=" * 80)
    print()
    
    # Model parameters from analysis checkpoint
    d_model = 512
    d_input = 256
    heads = 8
    n_synch_out = 256
    n_synch_action = 256
    synapse_depth = 4
    memory_length = 5
    iterations = 50
    backbone_type = 'resnet18-4'
    img_size = 224  # RAfDB images are resized to 224×224
    batch_size = 1
    group_count = 0
    memory_write_type = 'fifo'
    
    print("Model Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  d_input: {d_input}")
    print(f"  heads: {heads}")
    print(f"  n_synch_out: {n_synch_out}")
    print(f"  n_synch_action: {n_synch_action}")
    print(f"  synapse_depth: {synapse_depth}")
    print(f"  memory_length: {memory_length}")
    print(f"  iterations: {iterations}")
    print(f"  backbone_type: {backbone_type}")
    print(f"  img_size: {img_size}")
    print()
    
    # Calculate backbone FLOPs per forward pass
    backbone_flops = get_backbone_flops(backbone_type, img_size=img_size)
    print(f"Backbone FLOPs per forward pass: {format_flops(backbone_flops)} FLOPs ({backbone_flops:.0f})")
    print()
    
    # --- WITH BACKBONE ---
    print("-" * 80)
    print("WITH BACKBONE (resnet18-4)")
    print("-" * 80)
    
    loop_flops_with_backbone = calculate_ctm_loop_with_backbone(
        model=None, d_input=d_input, d_model=d_model, 
        memory_length=memory_length, 
        n_synch_action=n_synch_action, n_synch_out=n_synch_out,
        heads=heads, synapse_depth=synapse_depth,
        iterations=iterations, backbone_type=backbone_type,
        backbone_flops_per_loop=backbone_flops,
        batch_size=batch_size, group_count=group_count,
        memory_write_type=memory_write_type
    )
    
    # Per loop (single iteration)
    print(f"\nFLOPs per loop (single iteration/tick): {format_flops(loop_flops_with_backbone)} FLOPs")
    print(f"  = {loop_flops_with_backbone:.0f} FLOPs")
    print(f"  = {loop_flops_with_backbone/1e9:.4f} GFLOPs")
    
    # Per iteration (full forward pass with iterations loops)
    total_flops_with_backbone = backbone_flops + iterations * loop_flops_with_backbone
    print(f"\nTotal FLOPs per forward pass (backbone + {iterations} loops):")
    print(f"  Backbone: {format_flops(backbone_flops)} FLOPs ({backbone_flops/1e9:.4f} G)")
    print(f"  {iterations} loops: {format_flops(iterations * loop_flops_with_backbone)} FLOPs")
    print(f"  Total: {format_flops(total_flops_with_backbone)} FLOPs ({total_flops_with_backbone/1e9:.4f} G)")
    print()
    
    # --- WITHOUT BACKBONE ---
    print("-" * 80)
    print("WITHOUT BACKBONE (backbone_type='none')")
    print("-" * 80)
    
    loop_flops_no_backbone = calculate_ctm_loop_no_backbone(
        d_model=d_model, d_input=d_input,
        memory_length=memory_length,
        n_synch_action=n_synch_action, n_synch_out=n_synch_out,
        heads=heads, synapse_depth=synapse_depth,
        neuron_select_type='random-pairing',
        group_count=group_count,
        memory_write_type=memory_write_type,
        batch_size=batch_size
    )
    
    print(f"\nFLOPs per loop (single iteration/tick): {format_flops(loop_flops_no_backbone)} FLOPs")
    print(f"  = {loop_flops_no_backbone:.0f} FLOPs")
    print(f"  = {loop_flops_no_backbone/1e9:.4f} GFLOPs")
    
    total_flops_no_backbone = iterations * loop_flops_no_backbone
    print(f"\nTotal FLOPs for {iterations} loops (no backbone):")
    print(f"  = {format_flops(total_flops_no_backbone)} FLOPs ({total_flops_no_backbone/1e9:.4f} G)")
    print()
    
    # With backbone: backbone runs ONCE, then iterations loops
    # First iteration includes backbone feature extraction
    first_iter = backbone_flops + loop_flops_no_backbone
    # Subsequent iterations are just CTM loops
    subsequent_iters = loop_flops_no_backbone
    # Total forward pass
    total_with_backbone = backbone_flops + iterations * loop_flops_no_backbone
    
    print(f"With backbone (resnet18-4) - total forward pass ({iterations} iterations):")
    print(f"  Backbone (once):     {format_flops(backbone_flops)} ({backbone_flops/1e9:.4f} G)")
    print(f"  {iterations} CTM loops:     {format_flops(iterations * loop_flops_no_backbone)} ({iterations * loop_flops_no_backbone/1e9:.4f} G)")
    print(f"  Total:               {format_flops(total_with_backbone)} ({total_with_backbone/1e9:.4f} G)")
    print()
    
    # --- Comparison ---
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\nPer-loop (single iteration) FLOPs:")
    print(f"  No backbone:         {loop_flops_no_backbone/1e9:.4f} G ({format_flops(loop_flops_no_backbone)})")
    print(f"  First iter (w/ BB):  {first_iter/1e9:.4f} G ({format_flops(first_iter)})")
    print(f"  Ratio (first/no):    {first_iter/loop_flops_no_backbone:.1f}x")
    print(f"  Backbone share (first iter): {backbone_flops/first_iter*100:.1f}%")
    print()
    print(f"Total forward pass FLOPs ({iterations} iterations):")
    print(f"  Without backbone:    {total_flops_no_backbone/1e9:.4f} G ({format_flops(total_flops_no_backbone)})")
    print(f"  With backbone:       {total_with_backbone/1e9:.4f} G ({format_flops(total_with_backbone)})")
    print(f"  Ratio:               {total_with_backbone/total_flops_no_backbone:.1f}x")
    print()
    
    # Additional summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Metric':<40} {'Value':>15}")
    print("-" * 55)
    print(f"{'CTM loop (no backbone)':<40} {loop_flops_no_backbone/1e6:>10.2f} MFLOPs")
    print(f"{'First iteration (w/ backbone)':<40} {first_iter/1e9:>10.4f} GFLOPs")
    print(f"{'Total forward pass ({iterations} iters)':<40} {total_with_backbone/1e9:>10.4f} GFLOPs")
    print(f"{'Backbone FLOPs':<40} {backbone_flops/1e9:>10.4f} GFLOPs")
    print(f"{'CTM-only FLOPs ({iterations} iters)':<40} {iterations * loop_flops_no_backbone/1e9:>10.4f} GFLOPs")
    print(f"{'Backbone % of first iter':<40} {backbone_flops/first_iter*100:>9.1f}%")
    print(f"{'Backbone % of total ({iterations} iters)':<40} {backbone_flops/total_with_backbone*100:>9.1f}%")
    print()
    
    # Component percentages of CTM-only loop
    print("=" * 80)
    print("CTM-ONLY LOOP COMPONENT BREAKDOWN")
    print("=" * 80)
    print("Note: Excludes backbone, shows per-iteration CTM computation")
    print()

    # Calculate component FLOPs (same as detailed breakdown)
    from calculate_gflops_detailed import detailed_breakdown
    detailed_breakdown(
        d_model=d_model, d_input=d_input, memory_length=memory_length,
        n_synch_action=n_synch_action, n_synch_out=n_synch_out,
        heads=heads, synapse_depth=synapse_depth,
        backbone_flops=0, group_count=group_count,
        memory_write_type=memory_write_type, batch_size=batch_size
    )


if __name__ == '__main__':
    main()
