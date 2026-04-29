import torch
import numpy as np
from models.ctm import ContinuousThoughtMachine

# Set up model parameters (minimal config for inference)
model_args = {
    'iterations': 10,  # Number of internal thought ticks
    'd_model': 128,    # Latent space dimensionality
    'd_input': 64,     # Input feature dimensionality
    'heads': 4,        # Attention heads
    'n_synch_out': 64, # Output synchronization neurons
    'n_synch_action': 64,  # Action synchronization neurons
    'synapse_depth': 1,    # Synapse depth (1 = MLP, >1 = U-Net)
    'memory_length': 5,    # Memory length for NLMs
    'deep_nlms': True,     # Use deep NLMs
    'memory_hidden_dims': 16,  # Hidden dims for deep NLMs
    'do_layernorm_nlm': False,  # No LayerNorm in NLMs
    'backbone_type': 'none',    # No backbone (raw input)
    'positional_embedding_type': 'none',  # No positional embedding
    'out_dims': 10,      # Output classes (arbitrary for demo)
    'neuron_select_type': 'random-pairing',  # Neuron selection
    'n_random_pairing_self': 0,  # No self-pairing
    'group_count': 0,    # No neuron grouping
    'memory_write_type': 'fifo',  # FIFO memory
}

# Instantiate model
print("Creating CTM model...")
model = ContinuousThoughtMachine(**model_args)
model.eval()

# Create dummy input (batch_size=2, assume sequence length for 'none' backbone)
# For 'none' backbone, input is passed through identity, so shape can be flexible
batch_size = 2
seq_len = 32  # Arbitrary sequence length
input_features = model_args['d_input']

# Create random input tensor
x = torch.randn(batch_size, seq_len, input_features)
print(f"Input shape: {x.shape}")

# Run inference
print("Running inference...")
with torch.no_grad():
    predictions, certainties, synchronisation_out = model(x)
    # Also run with tracking for detailed analysis
    predictions_tracked, certainties_tracked, synch_out_tracked, pre_activations, post_activations, attention_weights = model(x, track=True)

# Print results
print(f"\nPredictions shape: {predictions.shape}")  # (B, out_dims, iterations)
print(f"Certainties shape: {certainties.shape}")    # (B, 2, iterations)
print(f"Synchronisation out shape: {synchronisation_out.shape}")  # (B, synch_size)

print(f"\nFinal predictions (last iteration): {predictions[:, :, -1]}")
print(f"Final certainties (last iteration): {certainties[:, :, -1]}")

# Analysis: Evolution over iterations
print("\n=== Analysis over iterations ===")
for i in range(model_args['iterations']):
    pred_entropy = torch.distributions.Categorical(logits=predictions[:, :, i]).entropy().mean()
    certainty = certainties[:, 1, i].mean()  # Certainty is 1 - normalized_entropy
    print(f"Iteration {i+1}: Entropy={pred_entropy:.4f}, Certainty={certainty:.4f}")

# Synchronization analysis
print("\n=== Synchronization Analysis ===")
print(f"Synchronisation values (sample 0, last iteration): {synchronisation_out[0]}")
print(f"Mean synchronisation magnitude: {torch.abs(synchronisation_out).mean().item():.4f}")
print(f"Max synchronisation magnitude: {torch.abs(synchronisation_out).max().item():.4f}")

# Neuron activation analysis (from tracking)
print("\n=== Neuron Activation Analysis ===")
print(f"Post-activations shape: {post_activations.shape}")  # (iterations, B, d_model)
neuron_variances = torch.var(post_activations, dim=[0, 1])  # Variance across iterations and batch
print(f"Neuron activation variances (top 10): {neuron_variances.topk(10).values}")

# Attention analysis (if heads > 0)
if model_args['heads'] > 0:
    print("\n=== Attention Analysis ===")
    print(f"Attention weights shape: {attention_weights.shape}")  # (iterations, B, 1, seq_len)
    avg_attention = attention_weights.mean(dim=[0, 1, 2])  # Average over iterations, batch, heads
    print(f"Average attention weights: {avg_attention}")

print("\nInference completed successfully!")