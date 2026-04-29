#!/usr/bin/env python3
"""
Detailed GFLOPS breakdown for CTM model per loop.
"""

def count_linear_flops(in_f, out_f, batch=1, num_inst=1):
    """FLOPs for Linear: multiply-add (2) + bias (1) per output element."""
    return batch * num_inst * out_f * (2 * in_f + 1)

def count_matmul_flops(m, n, k, batch=1):
    """FLOPs for matmul (m,n) @ (n,k): 2*m*n*k multiply-adds."""
    return batch * m * n * k * 2

def count_layernorm_flops(shape, batch=1, num_inst=1):
    """LayerNorm: ~7 FLOPs per element (mean, var, normalize, scale, shift)."""
    if isinstance(shape, int):
        elems = shape
    else:
        elems = 1
        for s in shape:
            elems *= s
    return batch * num_inst * elems * 7

def count_linear_glu_flops(in_f, out_f, batch=1, num_inst=1):
    """FLOPs for Linear(in_f->2*out_f) + GLU."""
    linear = count_linear_flops(in_f, 2*out_f, batch, num_inst)
    # GLU: sigmoid (~20 FLOPs) + multiply (1 FLOP) per output element
    glu = batch * num_inst * out_f * 21
    return linear + glu

def detailed_breakdown(d_model=512, d_input=256, memory_length=5,
                       n_synch_action=256, n_synch_out=256,
                       heads=8, synapse_depth=4,
                       backbone_flops=0, group_count=0,
                       memory_write_type='fifo', batch_size=1):
    """
    Print detailed FLOPs breakdown per CTM internal loop.
    """
    synapse_in = d_input + d_model  # concatenated input
    
    print(f"\nPer-Loop FLOPs Breakdown (batch_size={batch_size}):")
    print("-" * 60)
    
    total = 0
    
    # 1. Backbone feature extraction
    b = backbone_flops
    total += b
    print(f"1. Backbone (ResNet-18 scale-4): {b/1e6:.2f} M ({b:.0f})")
    
    # 2. Synchronization computation (action)
    # Element-wise multiply for n_synch_action pairs
    sync_action_mult = batch_size * n_synch_action * 1
    # Recurrence: r * decay + x (2 FLOPs) + sqrt/div (10 FLOPs approx)
    sync_action_recur = batch_size * n_synch_action * 12
    sync_action = sync_action_mult + sync_action_recur
    total += sync_action
    print(f"2. Synchronization (action):       {sync_action/1e6:.2f} M ({sync_action:.0f})")
    print(f"   - Pairwise multiply:             {sync_action_mult/1e6:.2f} M")
    print(f"   - Recurrence (decay):            {sync_action_recur/1e6:.2f} M")
    
    # 3. Attention computation
    # q_proj: Linear(d_model -> d_input)
    q_proj = count_linear_flops(d_model, d_input, batch_size, 1)
    total += q_proj
    print(f"3. Query projection (attn):         {q_proj/1e6:.2f} M ({q_proj:.0f})")
    
    # Attention over backbone features
    N_kv = 64  # estimated key-value pairs (e.g., 8x8 spatial grid from backbone)
    # Q @ K^T: (1, d_input) @ (d_input, N_kv)
    qk_matmul = count_matmul_flops(1, d_input, N_kv, batch_size)
    total += qk_matmul
    print(f"4. Attention Q @ K^T:               {qk_matmul/1e6:.2f} M ({qk_matmul:.0f})")
    
    # Softmax over N_kv
    attn_softmax = batch_size * 1 * N_kv * 5
    total += attn_softmax
    print(f"5. Attention softmax:               {attn_softmax/1e6:.2f} M ({attn_softmax:.0f})")
    
    # Attn_out = softmax @ V: (1, N_kv) @ (N_kv, d_input)
    attn_v_matmul = count_matmul_flops(1, N_kv, d_input, batch_size)
    total += attn_v_matmul
    print(f"6. Attention softmax @ V:           {attn_v_matmul/1e6:.2f} M ({attn_v_matmul:.0f})")
    
    # 7. Synapse layer (f_theta1)
    if synapse_depth == 1:
        # Linear + GLU + LayerNorm
        synapse = count_linear_glu_flops(synapse_in, d_model, batch_size, 1)
        synapse += count_layernorm_flops(d_model, batch_size, 1)
    else:
        # U-Net synapse
        # First projection
        synapse = count_linear_flops(synapse_in, d_model, batch_size, 1)
        synapse += count_layernorm_flops(d_model, batch_size, 1)
        
        min_width = 16
        width = d_model
        for i in range(synapse_depth - 1):
            next_w = max(min_width, int(width * (synapse_depth - 2 - i) / (synapse_depth - 1)))
            if next_w < 1:
                next_w = min_width
            # Down block
            synapse += count_linear_flops(width, next_w, batch_size, 1)
            synapse += count_layernorm_flops(next_w, batch_size, 1)
            # Up block
            synapse += count_linear_flops(next_w, width, batch_size, 1)
            synapse += count_layernorm_flops(width, batch_size, 1)
            width = next_w
    
    total += synapse
    print(f"7. Synapse layer (depth={synapse_depth}): {synapse/1e6:.2f} M ({synapse:.0f})")
    
    # 8. Memory write
    if memory_write_type == 'attention':
        # Q, K, V projections
        mem_q = count_linear_flops(d_model, d_model, batch_size, 1)
        mem_k = count_linear_flops(d_model, d_model, batch_size, memory_length)
        mem_v = count_linear_flops(d_model, d_model, batch_size, memory_length)
        mem_attn1 = count_matmul_flops(1, d_model, memory_length, batch_size)
        mem_softmax = batch_size * 1 * memory_length * 5
        mem_attn2 = count_matmul_flops(1, memory_length, d_model, batch_size)
        mem_write = mem_q + mem_k + mem_v + mem_attn1 + mem_softmax + mem_attn2
    else:
        mem_write = 0  # FIFO: negligible FLOPs
    
    total += mem_write
    print(f"8. Memory write ({memory_write_type}):   {mem_write/1e6:.2f} M ({mem_write:.0f})")
    
    # 9. Neuron Level Models (NLMs)
    if group_count > 0:
        group_size = d_model // group_count
        # Deep NLMs: 2-layer MLP per group
        mem_hidden = 16
        nlm = group_count * count_linear_glu_flops(memory_length, 2*mem_hidden, batch_size, group_size)
        nlm += group_count * count_linear_glu_flops(mem_hidden, 2, batch_size, group_size)
    else:
        # Single NLM for all d_model neurons
        mem_hidden = 16
        nlm = count_linear_glu_flops(memory_length, 2*mem_hidden, batch_size, d_model)
        nlm += count_linear_glu_flops(mem_hidden, 2, batch_size, d_model)
    
    total += nlm
    print(f"9. Neuron-Level Models (NLMs):      {nlm/1e6:.2f} M ({nlm:.0f})")
    print(f"   (d_model={d_model}, memory_length={memory_length}, hidden={mem_hidden})")
    
    # 10. Synchronization (output)
    sync_out_mult = batch_size * n_synch_out * 1
    sync_out_recur = batch_size * n_synch_out * 12
    sync_out = sync_out_mult + sync_out_recur
    total += sync_out
    print(f"10. Synchronization (output):       {sync_out/1e6:.2f} M ({sync_out:.0f})")
    
    # 11. Output projection
    if 'random-pairing' in 'random-pairing':  # default
        sync_out_size = n_synch_out
    else:
        sync_out_size = n_synch_out * (n_synch_out + 1) // 2
    
    out_proj = count_linear_flops(sync_out_size, 7, batch_size, 1)  # 7 classes
    total += out_proj
    print(f"11. Output projection:              {out_proj/1e6:.2f} M ({out_proj:.0f})")
    
    print("-" * 60)
    print(f"TOTAL per loop:                     {total/1e6:.2f} M ({total:.0f})")
    print(f"                                   = {total/1e9:.4f} G FLOPs")
    print()
    
    # Component percentages (excluding backbone for per-loop)
    print("Component % of per-loop FLOPs (excluding backbone):")
    loop_only = total - b
    print(f"  Synchronization (action):  {sync_action/loop_only*100:5.1f}%")
    print(f"  Attention computation:     {(q_proj+qk_matmul+attn_softmax+attn_v_matmul)/loop_only*100:5.1f}%")
    print(f"  Synapse layer:             {synapse/loop_only*100:5.1f}%")
    print(f"  Memory write:              {mem_write/loop_only*100:5.1f}%")
    print(f"  Neuron-Level Models:       {nlm/loop_only*100:5.1f}%")
    print(f"  Synchronization (output):  {sync_out/loop_only*100:5.1f}%")
    print(f"  Output projection:         {out_proj/loop_only*100:5.1f}%")
    print()
    
    return total

if __name__ == '__main__':
    detailed_breakdown()
