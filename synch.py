import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example dimensions
B = 2  # Batch size
n_synch_out = 3 # Number of neurons selected for output sync
t = 50  # Current internal tick (history length), made longer for a better plot

# --- Create Example Time Series Data ---
# Let's create some clear examples instead of random noise.
# We'll make one batch (B=1) for simplicity in visualization.

# Time axis
time = torch.linspace(0, 4 * np.pi, t)

# Neuron 0: A sine wave
neuron_0_series = torch.sin(time)

# Neuron 1: A cosine wave (out of phase with sine)
neuron_1_series = torch.cos(time)

# Neuron 2: Another sine wave, but slightly shifted and smaller (mostly in phase with neuron 0)
neuron_2_series = 0.7 * torch.sin(time - 0.5)

# Stack them into the history tensor for one batch item
selected_Z_history = torch.stack([neuron_0_series, neuron_1_series, neuron_2_series], dim=0).unsqueeze(0)

print(f"selected_Z_history shape: {selected_Z_history.shape}")
# e.g., selected_Z_history = tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
#                                   [[1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4]]])

# Example pairing indices (e.g., pair 0 with 1, pair 1 with 2, pair 2 with 0)
left_indices = torch.tensor([0, 1, 2]) # Indices of the first neuron in each pair
right_indices = torch.tensor([1, 2, 0]) # Indices of the second neuron in each pair

# Select the history for the left and right neurons in each pair
left_history = selected_Z_history[:, left_indices, :] # Shape: (B, n_synch_out, t)
right_history = selected_Z_history[:, right_indices, :] # Shape: (B, n_synch_out, t)

print(f"left_history shape: {left_history.shape}")
print(f"right_history shape: {right_history.shape}")

# Compute the dot product along the time dimension (t)
sync_values = torch.sum(left_history * right_history, dim=-1) # Shape: (B, n_synch_out) for each pair
print(f"sync_values shape (Random-Pairing S^t_out): {sync_values.shape}")
print(f"Calculated Sync Values: {sync_values.squeeze(0).numpy()}")
# This 'sync_values' is the synchronization representation S^t_out used for output.


# --- Visualization ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(n_synch_out, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Visualization of Neuron Pairing and Synchronization Score", fontsize=16)


# We look at the first item in the batch (B=0)
batch_idx = 0

for i in range(n_synch_out):
    left_neuron_idx = left_indices[i]
    right_neuron_idx = right_indices[i]
    
    # Get the time series for the pair
    ts_left = left_history[batch_idx, i, :].numpy()
    ts_right = right_history[batch_idx, i, :].numpy()
    
    # Get the calculated sync score for this pair
    score = sync_values[batch_idx, i].item()
    
    axes[i].plot(ts_left, label=f"Neuron {left_neuron_idx}", color='blue')
    axes[i].plot(ts_right, label=f"Neuron {right_neuron_idx}", color='red', linestyle='--')
    axes[i].set_title(f"Pair {i}: Neuron {left_neuron_idx} vs Neuron {right_neuron_idx} | Sync Score: {score:.2f}")
    axes[i].legend(loc='upper right')
    axes[i].set_ylabel("Activation")

plt.xlabel("Time (t)")
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure to a file instead of trying to display it interactively
plt.savefig("synchronization_plot.png")
print("\nPlot saved to synchronization_plot.png")


# Continuing from selected_Z_history (B, n_synch_out, t)

# Compute the full matrix of pairwise dot products (using batch matrix multiplication)
# Z.T @ Z (where @ is matrix multiplication)
full_sync_matrix = torch.bmm(selected_Z_history, selected_Z_history.transpose(1, 2))
# Shape: (B, n_synch_out, n_synch_out)

print(f"full_sync_matrix shape: {full_sync_matrix.shape}")

# --- Visualization of the full synchronization matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    full_sync_matrix[0].numpy(), # Use the first item in the batch
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=[f"Neuron {i}" for i in range(n_synch_out)],
    yticklabels=[f"Neuron {i}" for i in range(n_synch_out)]
)
plt.title("Full Pairwise Synchronization Matrix (Heatmap)")
plt.xlabel("Neuron j")
plt.ylabel("Neuron i")
plt.tight_layout()
plt.savefig("full_sync_matrix_heatmap.png")
print("\nFull sync matrix heatmap saved to full_sync_matrix_heatmap.png")

# Extract the upper triangle (including diagonal) to form the synchronization vector
# PyTorch doesn't have a direct batched triu_indices, so we do it element-wise or use masking
n = n_synch_out
sync_vector_size = (n * (n + 1)) // 2 # e.g., 3*(3+1)//2 = 6
sync_vector = torch.zeros(B, sync_vector_size)

# Example using manual indexing (inefficient for large n, but clear)
idx = 0
for i in range(n):
    for j in range(i, n): # Upper triangle including diagonal
        sync_vector[:, idx] = full_sync_matrix[:, i, j]
        idx += 1

print(f"sync_vector shape (Random S^t_out): {sync_vector.shape}")
# This 'sync_vector' is the synchronization representation S^t_out used for output.