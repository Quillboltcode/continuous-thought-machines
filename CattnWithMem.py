import torch
import torch.nn as nn
import torch.func as func # Requires torch >= 2.0
import math

# 1. Define the operation for a single neuron on its history
# This function operates on tensors where the first dimension is the batch dimension.
# Here, 'history' is expected to have shape (B, M), where B is the batch size.
# 'weight' and 'bias' are specific parameters for *one* neuron, shaped (M, out_dims) and (out_dims,).
def single_neuron_nlm(history, weight, bias): # history: (B, M), weight: (M, out_dims), bias: (out_dims,)
    # Perform the linear transformation: (B, M) @ (M, out_dims) -> (B, out_dims)
    result = torch.matmul(history, weight) + bias # Output: (B, out_dims)
    return result

class VmapNLM(nn.Module):
    """
    NLM using torch.vmap for vectorization.
    Defines the operation for one neuron and maps it across N neurons.
    """
    def __init__(self, in_dims, out_dims, N, dropout=0.0, do_norm=False):
        super().__init__()
        self.N = N # Number of neurons (d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm = nn.LayerNorm(in_dims) if do_norm else nn.Identity()
        self.do_norm = do_norm

        # Parameters: shape (N, in_dims, out_dims) and (N, out_dims)
        # These are the parameters for *all N neurons*, stacked along the first dimension.
        self.weight = nn.Parameter(torch.empty(N, in_dims, out_dims))
        self.bias = nn.Parameter(torch.empty(N, out_dims))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(self.weight.shape[1] + self.weight.shape[2]),
                          1/math.sqrt(self.weight.shape[1] + self.weight.shape[2]))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: (B, N, M) where B=batch, N=d_model (number of neurons), M=memory_length
        B, N, M = x.shape

        out = self.dropout(x) # (B, N, M)
        out = self.layernorm(out) # (B, N, M)

        # 2. Prepare inputs for vmap
        # We want to apply the single neuron function to each of the N neurons.
        # The input 'history' for vmap needs to be the part that changes for each neuron.
        # The parameters 'weight' and 'bias' also need to be the part that changes for each neuron.
        # The 'single_neuron_nlm' function expects history: (B, M) and weight: (M, out_dims), bias: (out_dims,)

        # To apply it across the N neurons, we need to align the dimensions correctly.
        # We transpose 'out' from (B, N, M) to (N, B, M).
        # Now, the first dimension (dim=0) is the "N" dimension.
        # When vmap iterates, it will take slices of shape (B, M) from this transposed tensor.
        out = out.transpose(0, 1) # Shape: (N, B, M)

        # The parameters 'self.weight' and 'self.bias' already have the N dimension as the first dim:
        # self.weight: (N, in_dims, out_dims)
        # self.bias:   (N, out_dims)

        # 3. Apply vmap
        # func.vmap takes the function to be vectorized ('single_neuron_nlm').
        # The 'in_dims' argument tells vmap which dimension of each input argument
        # corresponds to the "thing" being mapped over (in this case, the N neurons).
        # Here, dim 0 of 'out', 'self.weight', and 'self.bias' all correspond to the neuron index.
        vmapped_func = func.vmap(single_neuron_nlm, in_dims=(0, 0, 0)) # Apply over dim 0 of inputs

        # Execute the vectorized function
        # Input to vmapped_func:
        # - out: (N, B, M) -> vmap takes (B, M) slices along dim 0
        # - self.weight: (N, in_dims, out_dims) -> vmap takes (in_dims, out_dims) slices along dim 0
        # - self.bias: (N, out_dims) -> vmap takes (out_dims,) slices along dim 0
        # Output: (N, B, out_dims) -> Each of the N neuron results stacked
        out = vmapped_func(out, self.weight, self.bias) # Output: (N, B, out_dims)

        # 4. Reshape back to the desired output format
        # The result is currently (N, B, out_dims), but we want (B, N, out_dims)
        out = out.transpose(0, 1) # Shape: (B, N, out_dims)

        # If out_dims is 1, squeeze it (to match the pattern of other NLM implementations)
        if out.size(-1) == 1:
            out = out.squeeze(-1) # Shape: (B, N)

        return out

# Instead of:
# SuperLinear → GLU → SuperLinear → GLU → Squeeze

# Use a FourierKAN layer:
class FourierKAN(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable frequency scaling
        self.freqs = nn.Parameter(torch.randn(out_features, in_features, num_frequencies))
        self.weights = nn.Parameter(torch.randn(out_features, in_features, num_frequencies))

    def forward(self, x):
        # x: (B, N, M)
        B, N, M = x.shape
        x_expanded = x.unsqueeze(-1)  # (B, N, M, 1)
        freqs = self.freqs.unsqueeze(0)  # (1, N, M, K)
        weights = self.weights.unsqueeze(0)  # (1, N, M, K)
        # Compute Fourier basis: sin(w*x) and cos(w*x)
        sin_term = torch.sin(x_expanded * freqs)
        cos_term = torch.cos(x_expanded * freqs)
        # Combine with learnable weights
        out = (weights * sin_term + weights * cos_term).sum(-1)  # (B, N, M)
        return out.mean(dim=-1)  # (B, N) — scalar per neuron

if __name__ == "__main__":
    # Test VmapNLM
    from models.modules import SuperLinear
    B, N, M, out_dims = 4, 8, 16, 1
    x = torch.randn(B, N, M)
    model = VmapNLM(in_dims=M, out_dims=out_dims, N=N, dropout=0.1, do_norm=True)
    out = model(x)
    print("VmapNLM output shape:", out.shape) # Expected: (B, N)

    # Test FourierKAN
    fourier_model = FourierKAN(in_features=M, out_features=N, num_frequencies=5)
    fourier_out = fourier_model(x)
    print("FourierKAN output shape:", fourier_out.shape) # Expected: (B, N)
    # Test SuperLinear for comparison
    super_linear = SuperLinear(in_dims=M, out_dims=out_dims, dropout=0.1, do_norm=True, N=N)
    super_linear_out = super_linear(x) # (B, N, out_dims)
    print("SuperLinear output shape:", super_linear_out.shape) # Expected: (B, N)
    
    # Compare three way outputs using torch equallity checks


    print("SuperLinear diff to Vmap:", torch.abs(super_linear_out - out).mean().item())
    print("SuperLinear diff to FourierKAN:", torch.abs(super_linear_out - fourier_out).mean().item())
    print("FourierKAN diff to Vmap:", torch.abs(fourier_out - out).mean().item())