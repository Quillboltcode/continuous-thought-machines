import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import os

def compute_decay(T, params, clamp_lims=(0, 15)):
    """
    This function computes exponential decays for learnable synchronisation 
    interactions between pairs of neurons. 
    """
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'
    
    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.

    Args:
        tensor: A PyTorch tensor of shape (B, D, H, W).

    Returns:
        A PyTorch tensor of shape (B, D, H, W, 2) with the last dimension
        representing the 2D coordinates within the HW dimensions.
    """
    B, H, W = x.shape
    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the 
    final dimension.

    Args:
      logits: A PyTorch tensor of logits. 

    Returns:
      A PyTorch tensor containing the normalized entropy values.
    """

    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy
    
def reshape_predictions(predictions, prediction_reshaper):
    B, T = predictions.size(0), predictions.size(-1)
    new_shape = [B] + prediction_reshaper + [T]
    rehaped_predictions = predictions.reshape(new_shape)
    return rehaped_predictions

def get_all_log_dirs(root_dir):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".pt") for f in filenames):
            folders.append(dirpath)
    return folders

def get_latest_checkpoint(log_dir):
    files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
    return os.path.join(log_dir, max(files, key=lambda f: int(re.search(r'\d+', f).group()))) if files else None

def get_latest_checkpoint_file(filepath, limit=300000):
    checkpoint_files = get_checkpoint_files(filepath)
    checkpoint_files = [
        f for f in checkpoint_files if int(re.search(r'checkpoint_(\d+)\.pt', f).group(1)) <= limit
    ]
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]

def get_checkpoint_files(filepath):
    regex = r'checkpoint_(\d+)\.pt'
    files = [f for f in os.listdir(filepath) if re.match(regex, f)]
    files = sorted(files, key=lambda f: int(re.search(regex, f).group(1)))
    return [os.path.join(filepath, f) for f in files]

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def get_model_args_from_checkpoint(checkpoint):
    if "args" in checkpoint:
        return(checkpoint["args"])
    else:
        raise ValueError("Checkpoint does not contain saved args.")

def get_accuracy_and_loss_from_checkpoint(checkpoint, device="cpu"):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies

def compute_flop(model, input_shape):
    """
    Compute FLOPs (floating point operations) for a PyTorch model using fvcore.
    
    Args:
        model: PyTorch model
        input_shape: Tuple defining input shape (e.g., (1, 3, 224, 224))
    
    Returns:
        float: Total FLOPs
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(*input_shape)
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
    except ImportError:
        raise ImportError("fvcore is required for FLOP counting. Install with: pip install fvcore")


def compute_ctm_flops(model, input_shape, batch_size=1, use_fvcore=True):
    """
    Compute FLOPs (floating point operations) for a ContinuousThoughtMachine model.
    Uses fvcore for component-level FLOPs where possible, but manually accounts for
    the iterative forward pass loop.
    
    Args:
        model: ContinuousThoughtMachine model instance
        input_shape: Tuple defining input shape excluding batch dimension (e.g., (3, 224, 224))
        batch_size: Batch size for FLOP calculation (default: 1)
        use_fvcore: If True, use fvcore for component-level FLOPs (default: True)
    
    Returns:
        dict: Dictionary containing total FLOPs and breakdown by component
    """
    B = batch_size
    T = model.iterations  # Number of iterations in the loop
    
    # Try to use fvcore if available
    fvcore_available = False
    if use_fvcore:
        try:
            from fvcore.nn import FlopCountAnalysis
            fvcore_available = True
        except ImportError:
            fvcore_available = False
    
    # Get model dimensions
    d_model = model.d_model
    d_input = model.d_input
    # Check if attention is used (heads > 0)
    heads = model.attention.num_heads if model.attention is not None else 0
    n_synch_out = model.n_synch_out
    n_synch_action = model.n_synch_action
    memory_length = model.memory_length
    out_dims = model.out_dims
    
    # Extract memory_hidden_dims from trace_processor if deep_nlms is used
    # Check if trace_processor has multiple SuperLinear layers (deep_nlms)
    memory_hidden_dims = None
    if hasattr(model, 'trace_processor') and model.trace_processor is not None:
        # Check if it's a deep NLM by looking at the structure
        try:
            # Deep NLMs have 2 SuperLinear layers, the first one has out_dims = 2 * memory_hidden_dims
            first_layer = model.trace_processor[0][0]
            if hasattr(first_layer, 'w1') and first_layer.w1 is not None:
                # SuperLinear: w1 shape is (in_dims, out_dims, N)
                # For deep NLMs: out_dims = 2 * memory_hidden_dims (due to GLU)
                if first_layer.w1.shape[1] > 2:  # More than 2 output dims suggests deep NLM
                    memory_hidden_dims = first_layer.w1.shape[1] // 2  # Divide by 2 for GLU
        except (AttributeError, IndexError):
            pass
    
    # If we couldn't extract it, use a default (this shouldn't happen in practice)
    if memory_hidden_dims is None:
        memory_hidden_dims = 64  # Default fallback
    
    # Calculate synchronisation representation sizes
    synch_size_action = model.synch_representation_size_action
    synch_size_out = model.synch_representation_size_out
    
    flops = {}
    total_flops = 0
    
    # ========== Operations OUTSIDE the loop (one-time) ==========
    
    # 1. Initial RGB conversion (if backbone is resnet)
    if 'resnet' in model.backbone_type:
        # Conv2d: kernel_size^2 * in_channels * out_channels * output_h * output_w * batch_size
        # Assuming 1x1 conv, output shape same as input
        if len(input_shape) == 3:
            C_in, H, W = input_shape
            C_out = 3
            flops['initial_rgb'] = 1 * 1 * C_in * C_out * H * W * B
            total_flops += flops['initial_rgb']
        else:
            flops['initial_rgb'] = 0
    
    # 2. Backbone - use fvcore if available, otherwise manual estimate
    if model.backbone_type == 'none':
        flops['backbone'] = 0
        kv_seq_len = 1  # For no backbone, assume single token
        d_backbone = d_input
    elif fvcore_available and hasattr(model, 'backbone') and model.backbone is not None:
        # Use fvcore to measure backbone FLOPs
        try:
            if len(input_shape) == 3:
                C, H, W = input_shape
                backbone_input = torch.randn(B, C, H, W)
                if hasattr(model, 'initial_rgb') and model.initial_rgb is not None:
                    backbone_input = model.initial_rgb(backbone_input)
                flop_analysis = FlopCountAnalysis(model.backbone, backbone_input)
                flops['backbone'] = flop_analysis.total()
                # Estimate kv_seq_len from backbone output shape
                with torch.no_grad():
                    backbone_out = model.backbone(backbone_input)
                    if isinstance(backbone_out, torch.Tensor):
                        if len(backbone_out.shape) == 4:  # (B, C, H, W)
                            kv_seq_len = backbone_out.shape[2] * backbone_out.shape[3]
                            d_backbone = backbone_out.shape[1]
                        else:
                            kv_seq_len = 1
                            d_backbone = backbone_out.shape[-1] if len(backbone_out.shape) > 1 else d_input
                    else:
                        kv_seq_len = 1
                        d_backbone = model.get_d_backbone()
            else:
                kv_seq_len = 1
                d_backbone = d_input
                flops['backbone'] = 0
        except Exception:
            # Fallback to manual estimation
            if len(input_shape) == 3:
                _, H, W = input_shape
                feat_h, feat_w = H // 4, W // 4
                kv_seq_len = feat_h * feat_w
                d_backbone = model.get_d_backbone()
                flops['backbone'] = 1000000 * B  # Rough estimate
            else:
                kv_seq_len = 1
                d_backbone = d_input
                flops['backbone'] = 0
    elif 'resnet' in model.backbone_type:
        # Manual estimation for ResNet
        if len(input_shape) == 3:
            _, H, W = input_shape
            feat_h, feat_w = H // 4, W // 4
            kv_seq_len = feat_h * feat_w
            d_backbone = model.get_d_backbone()
            flops['backbone'] = 1000000 * B  # Rough estimate
        else:
            kv_seq_len = 1
            d_backbone = d_input
            flops['backbone'] = 0
    else:
        # Other backbones - simplified
        kv_seq_len = 1
        d_backbone = d_input
        flops['backbone'] = 0
    
    total_flops += flops['backbone']
    
    # 3. Positional embedding (usually element-wise operations)
    if model.positional_embedding_type != 'none':
        # Element-wise operations: roughly kv_seq_len * d_backbone * B
        flops['positional_embedding'] = kv_seq_len * d_backbone * B
    else:
        flops['positional_embedding'] = 0
    total_flops += flops['positional_embedding']
    
    # 4. KV projection (Linear layer) - use fvcore if available
    if heads > 0:
        if fvcore_available and model.kv_proj is not None:
            try:
                kv_input_size = d_backbone if model.backbone_type != 'none' else d_input
                kv_input = torch.randn(B, kv_seq_len, kv_input_size)
                flop_analysis = FlopCountAnalysis(model.kv_proj, kv_input)
                flops['kv_proj'] = flop_analysis.total()
            except Exception:
                # Fallback to manual calculation
                kv_input_size = d_backbone if model.backbone_type != 'none' else d_input
                flops['kv_proj'] = kv_input_size * d_input * B * kv_seq_len
                # Add LayerNorm FLOPs if present (fvcore may skip this)
                if not fvcore_available and isinstance(model.kv_proj, nn.Sequential):
                    for module in model.kv_proj:
                        if isinstance(module, nn.LayerNorm):
                            # LayerNorm: mean, variance, normalize = 3 * num_elements
                            flops['kv_proj'] += d_input * B * kv_seq_len * 3
        else:
            # Manual calculation
            kv_input_size = d_backbone if model.backbone_type != 'none' else d_input
            flops['kv_proj'] = kv_input_size * d_input * B * kv_seq_len
            # Add LayerNorm FLOPs if present
            if isinstance(model.kv_proj, nn.Sequential):
                for module in model.kv_proj:
                    if isinstance(module, nn.LayerNorm):
                        flops['kv_proj'] += d_input * B * kv_seq_len * 3
        total_flops += flops['kv_proj']
    else:
        flops['kv_proj'] = 0
    
    # 5. Initial synchronisation computation (once before loop)
    # Synchronisation involves element-wise operations and dot products
    # For random-pairing: n_synch multiplications
    # For first-last/random: n_synch * (n_synch + 1) / 2 multiplications
    if model.neuron_select_type == 'random-pairing':
        synch_ops = n_synch_out * B
    else:
        synch_ops = (n_synch_out * (n_synch_out + 1)) // 2 * B
    flops['initial_synch'] = synch_ops
    total_flops += flops['initial_synch']
    
    # ========== Operations INSIDE the loop (multiplied by T) ==========
    
    # Per-iteration FLOPs
    per_iter_flops = 0
    
    # 1. Synchronisation computation (action) - 2 times per iteration
    if synch_size_action > 0:
        if model.neuron_select_type == 'random-pairing':
            synch_action_ops = n_synch_action * B
        else:
            synch_action_ops = (n_synch_action * (n_synch_action + 1)) // 2 * B
        # Recurrent update: element-wise operations
        synch_action_ops += synch_size_action * B * 3  # multiply, add, divide
        per_iter_flops += synch_action_ops
    
    # 2. Q projection (Linear) - use fvcore if available
    if heads > 0:
        if fvcore_available and model.q_proj is not None:
            try:
                q_input = torch.randn(B, synch_size_action)
                flop_analysis = FlopCountAnalysis(model.q_proj, q_input)
                flops_q_proj = flop_analysis.total()
            except Exception:
                # Fallback to manual calculation
                flops_q_proj = synch_size_action * d_input * B
        else:
            # Manual calculation
            flops_q_proj = synch_size_action * d_input * B
        per_iter_flops += flops_q_proj
    
    # 3. Attention mechanism
    if heads > 0:
        # QK^T: (B, 1, d_input) @ (B, kv_seq_len, d_input)^T -> (B, 1, kv_seq_len)
        # FLOPs: B * 1 * d_input * kv_seq_len
        flops_qkt = B * 1 * d_input * kv_seq_len
        # Softmax: B * 1 * kv_seq_len (approximate)
        flops_softmax = B * 1 * kv_seq_len * 3  # exp, sum, divide
        # Attention @ V: (B, 1, kv_seq_len) @ (B, kv_seq_len, d_input) -> (B, 1, d_input)
        # FLOPs: B * 1 * kv_seq_len * d_input
        flops_attn_v = B * 1 * kv_seq_len * d_input
        # Multi-head: multiply by number of heads
        flops_attention = (flops_qkt + flops_softmax + flops_attn_v) * heads
        per_iter_flops += flops_attention
    
    # 4. Synapses - use fvcore if available
    synapse_input_size = d_input + d_model if heads > 0 else d_model
    if fvcore_available and hasattr(model, 'synapses') and model.synapses is not None:
        try:
            synapse_input = torch.randn(B, synapse_input_size)
            flop_analysis = FlopCountAnalysis(model.synapses, synapse_input)
            flops_synapses = flop_analysis.total()
            per_iter_flops += flops_synapses
        except Exception:
            # Fallback to manual calculation
            synapse_depth = 1
            if hasattr(model.synapses, 'n_deep'):
                synapse_depth = model.synapses.n_deep
            
            if synapse_depth == 1:
                flops_synapses = synapse_input_size * d_model * 2 * B  # GLU doubles output
            else:
                flops_synapses = synapse_input_size * d_model * synapse_depth * 16 * B
            per_iter_flops += flops_synapses
    else:
        # Manual calculation
        synapse_depth = 1
        if hasattr(model, 'synapses') and model.synapses is not None:
            if hasattr(model.synapses, 'n_deep'):
                synapse_depth = model.synapses.n_deep
        
        if synapse_depth == 1:
            flops_synapses = synapse_input_size * d_model * 2 * B
        else:
            flops_synapses = synapse_input_size * d_model * synapse_depth * 16 * B
        per_iter_flops += flops_synapses
    
    # 5. Neuron-Level Models (NLMs)
    # Check if deep_nlms by examining trace_processor structure
    is_deep_nlm = False
    has_layernorm_nlm = False
    if hasattr(model, 'trace_processor') and model.trace_processor is not None:
        try:
            # Deep NLMs have multiple SuperLinear layers
            num_superlinear = sum(1 for layer in model.trace_processor[0] if hasattr(layer, 'w1'))
            is_deep_nlm = num_superlinear > 1
            # Check if LayerNorm is enabled in NLMs
            first_layer = model.trace_processor[0][0]
            if hasattr(first_layer, 'layernorm') and first_layer.layernorm is not None:
                has_layernorm_nlm = True
        except:
            pass
    
    if is_deep_nlm:
        # First SuperLinear: memory_length * (2 * memory_hidden_dims) * d_model * B
        flops_nlm1 = memory_length * (2 * memory_hidden_dims) * d_model * B
        # LayerNorm: 2 * memory_hidden_dims * d_model * B (mean, variance, normalize)
        if has_layernorm_nlm and not fvcore_available:
            flops_nlm1 += 2 * memory_hidden_dims * d_model * B * 2  # Applied per SuperLinear
        # GLU: element-wise, negligible
        # Second SuperLinear: memory_hidden_dims * 2 * d_model * B
        flops_nlm2 = memory_hidden_dims * 2 * d_model * B
        if has_layernorm_nlm and not fvcore_available:
            flops_nlm2 += memory_hidden_dims * d_model * B * 2  # LayerNorm for second layer
        flops_nlm = flops_nlm1 + flops_nlm2
    else:
        # Single SuperLinear: memory_length * 2 * d_model * B
        flops_nlm = memory_length * 2 * d_model * B
        # LayerNorm: memory_length * d_model * B * 2 (mean, variance, normalize)
        if has_layernorm_nlm and not fvcore_available:
            flops_nlm += memory_length * d_model * B * 2
    per_iter_flops += flops_nlm
    
    # 6. Synchronisation computation (out)
    if model.neuron_select_type == 'random-pairing':
        synch_out_ops = n_synch_out * B
    else:
        synch_out_ops = (n_synch_out * (n_synch_out + 1)) // 2 * B
    # Recurrent update: element-wise operations
    synch_out_ops += synch_size_out * B * 3  # multiply, add, divide
    per_iter_flops += synch_out_ops
    
    # 7. Output projection (Linear) - use fvcore if available
    if fvcore_available and hasattr(model, 'output_projector') and model.output_projector is not None:
        try:
            output_input = torch.randn(B, synch_size_out)
            flop_analysis = FlopCountAnalysis(model.output_projector, output_input)
            flops_output_proj = flop_analysis.total()
        except Exception:
            flops_output_proj = synch_size_out * out_dims * B
    else:
        flops_output_proj = synch_size_out * out_dims * B
    per_iter_flops += flops_output_proj
    
    # 8. Certainty computation (entropy calculation)
    # Softmax: out_dims * B * 3 (exp, sum, divide)
    flops_softmax_certainty = out_dims * B * 3
    # Entropy: out_dims * B (log and multiply)
    flops_entropy = out_dims * B * 2
    flops_certainty = flops_softmax_certainty + flops_entropy
    per_iter_flops += flops_certainty
    
    # 9. Halting head - use fvcore if available
    if fvcore_available and hasattr(model, 'halting_head') and model.halting_head is not None:
        try:
            halt_input = torch.randn(B, synch_size_out)
            flop_analysis = FlopCountAnalysis(model.halting_head, halt_input)
            flops_halting = flop_analysis.total()
            per_iter_flops += flops_halting
        except Exception:
            # Fallback to manual calculation
            flops_halt1 = synch_size_out * 64 * B
            flops_gelu = 64 * B * 3
            flops_halt2 = 64 * 1 * B
            flops_halting = flops_halt1 + flops_gelu + flops_halt2
            per_iter_flops += flops_halting
    else:
        # Manual calculation
        flops_halt1 = synch_size_out * 64 * B
        flops_gelu = 64 * B * 3
        flops_halt2 = 64 * 1 * B
        flops_halting = flops_halt1 + flops_gelu + flops_halt2
        per_iter_flops += flops_halting
    
    # Multiply per-iteration FLOPs by number of iterations
    flops['per_iteration'] = per_iter_flops
    flops['loop_total'] = per_iter_flops * T
    total_flops += flops['loop_total']
    
    # Store breakdown
    flops['breakdown'] = {
        'outside_loop': flops.get('initial_rgb', 0) + flops.get('backbone', 0) + 
                        flops.get('positional_embedding', 0) + flops.get('kv_proj', 0) + 
                        flops.get('initial_synch', 0),
        'inside_loop_per_iter': per_iter_flops,
        'inside_loop_total': per_iter_flops * T,
    }
    
    flops['total'] = total_flops
    flops['fvcore_used'] = fvcore_available
    
    return flops

def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params



class PonderNetLoss(nn.Module):
    def __init__(self, lambda_p=0.2, beta=0.01):
        """
        Args:
            lambda_p: Success probability for Geometric prior (higher = shorter thought).
            beta: Regularization weight (higher = stronger push toward the prior).
        """
        super().__init__()
        self.lambda_p = lambda_p
        self.beta = beta

    def forward(self, predictions, halt_probs, targets):
        """
        predictions: [Batch, Classes, Steps]
        halt_probs:  [Batch, Steps] (Sigmoid output)
        targets:     [Batch] (Class indices)
        """
        B, C, T = predictions.size()
        # Handle different input shapes for halt_probs
        if halt_probs.dim() == 3:
            if halt_probs.size(1) == 1:
                halt_probs = halt_probs.squeeze(1) # [B, T]
            elif halt_probs.size(1) == 2:
                # Assume the first dimension is the halting probability
                halt_probs = halt_probs[:, 0:1, :] # [B, 1, T] -> [B, T]
                halt_probs = halt_probs.squeeze(1)
            else:
                raise ValueError(f"Expected halt_probs shape [B, 1, T], [B, 2, T], or [B, T], got {halt_probs.shape}")
        elif halt_probs.dim() != 2:
            raise ValueError(f"Expected halt_probs to be 2D or 3D, got {halt_probs.dim()}D with shape {halt_probs.shape}")

        # --- 1. Compute Halting Distribution (p_t) ---
        # p_t is the probability of stopping exactly at step t
        ones = torch.ones(B, 1, device=predictions.device)
        # Prob of NOT halting in all steps before current
        cum_not_halted = torch.cumprod(torch.cat((ones, 1.0 - halt_probs[:, :-1]), dim=1), dim=1)
        p_t = halt_probs * cum_not_halted
        
        # PonderNet Trick: The remaining probability is dumped into the last step
        p_t[:, -1] = 1.0 - p_t[:, :-1].sum(dim=1)

        # --- 2. Calculate Reconstruction Loss (Cross-Entropy) ---
        # Expand targets to match (Batch, Steps)
        expanded_targets = targets.unsqueeze(1).expand(-1, T)
        
        # Calculate CE for every step: [B, T]
        ce_per_step = F.cross_entropy(predictions, expanded_targets, reduction='none')
        
        # Expectation of CE over the halting distribution
        loss_rec = (p_t * ce_per_step).sum(dim=1).mean()

        # --- 3. Calculate Regularization (KL Divergence) ---
        # Geometric Prior: P(L=k) = (1-p)^(k-1) * p
        steps = torch.arange(1, T + 1, device=predictions.device).float()
        geo_prior = torch.pow(1 - self.lambda_p, steps - 1) * self.lambda_p
        
        # Re-normalize prior to sum to 1 over finite steps T
        geo_prior = geo_prior / geo_prior.sum()
        
        # KL Divergence: sum(p * log(p/q))
        # We add 1e-8 for numerical stability in the log
        kl_div = (p_t * (torch.log(p_t + 1e-8) - torch.log(geo_prior + 1e-8))).sum(dim=1).mean()

        total_loss = loss_rec + self.beta * kl_div
        
        return total_loss, p_t
