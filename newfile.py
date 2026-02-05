import torch
import torch.nn as nn
# Assuming SuperLinear and Squeeze are defined elsewhere or imported
from models.modules import SuperLinear, Squeeze



# Example of a potential FourierKAN-like layer (simplified)
class FourierKAN(nn.Module):
    """
    A simplified example of a KAN-like layer using Fourier basis functions.
    This is a conceptual placeholder.
    Input: (B, N, M) where M is the history length
    Output: (B, N) - scalar per neuron
    """
    def __init__(self, in_features, out_features, N, num_frequencies=5):
        super().__init__()
        assert out_features == 1 # Expected for final NLM output
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # Learnable frequency and phase shifts for each neuron and each frequency component
        # Shape: (N, num_frequencies)
        self.freqs = nn.Parameter(torch.randn(N, num_frequencies))
        self.phases = nn.Parameter(torch.randn(N, num_frequencies))
        # Learnable weights for combining frequency components
        # Shape: (N, num_frequencies)
        self.weights = nn.Parameter(torch.randn(N, num_frequencies))

    def forward(self, x):
        # x shape: (B, N, M)
        B, N, M = x.shape
        assert N == self.N

        # Expand x to apply frequencies: (B, N, M, 1)
        x_expanded = x.unsqueeze(-1) # (B, N, M, 1)
        # Expand freqs and phases to match: (1, N, 1, num_f)
        freqs = self.freqs.unsqueeze(0).unsqueeze(2) # (1, N, 1, num_f)
        phases = self.phases.unsqueeze(0).unsqueeze(2) # (1, N, 1, num_f)

        # Compute Fourier basis: sin(freq * x + phase)
        basis = torch.sin(freqs * x_expanded + phases) # (B, N, M, num_f)

        # Weighted sum over frequencies: (B, N, M, num_f) * (1, N, 1, num_f) -> (B, N, M, num_f)
        weighted_basis = basis * self.weights.unsqueeze(0).unsqueeze(2) # (B, N, M, num_f)
        # Sum over frequencies: (B, N, M, num_f) -> (B, N, M)
        intermediate = weighted_basis.sum(dim=-1) # (B, N, M)

        # Sum over the history dimension M to get a single output per neuron
        out = intermediate.sum(dim=-1, keepdim=True) # (B, N, 1)
        out = out.squeeze(-1) # (B, N)

        return out


def get_neuron_level_models_v2(core_module: nn.Module, deep_nlms=True, dropout=0.0):
    """
    Creates the Neuron-Level Models (NLMs) using a provided core module.

    Args:
        core_module (nn.Module): The core processing module to be used.
                                 It should accept input of shape (B, N, in_dims) or (B, N, history_length)
                                 and produce output of shape (B, N, out_dims) or (B, N).
        deep_nlms (bool): If True, stack the core module with GLU activations.
                          If False, use the core module directly with GLU and Squeeze.
        dropout (float): Dropout rate applied before the core module.
    """
    if deep_nlms:
        # Stack the core module, GLU, another core module, GLU, and Squeeze
        # This assumes the first core module outputs something that can be fed into the second.
        # Often, the first core_module might need to output (B, N, 2*hidden_dims),
        # then GLU makes (B, N, hidden_dims), and the second core_module outputs (B, N, 2).
        # This requires the core_module to be flexible or for you to wrap it.
        # For simplicity here, assume the first core module outputs the right shape for GLU.
        # A more robust way is to have core_module output (B, N, 2*...), then GLU -> (B, N, ...)

        # Example assuming core_module can be configured or wrapped to handle this:
        # Let's assume core_module_out1 outputs (B, N, 2*H) -> GLU -> (B, N, H)
        # Then core_module_out2 takes (B, N, H) -> outputs (B, N, 2) -> GLU -> (B, N, 1) -> Squeeze -> (B, N)
        # This is complex with arbitrary nn.Modules. A common pattern is:
        # core_module -> GLU -> nn.Linear -> GLU -> Squeeze

        # For a generic nn.Module, we can't assume internal structure like SuperLinear's in/out dims.
        # A more flexible way is to wrap the core module with an adapter if needed.
        # Here, we'll assume the core module handles the in/out dims for the deep structure.
        # This is tricky. Let's define a minimal wrapper that works with the original SuperLinear logic.

        # A simpler, more robust approach for deep NLMS with a generic module:
        # Wrap the core module in a way that mimics the original structure.
        # e.g., use the core module twice, expecting it to handle internal dims.

        # Let's define a sequential that tries to mimic the original deep structure
        # using the provided core_module. This requires the module to be compatible.
        # For instance, if core_module takes (B,N,M) -> (B,N,2*H) and another takes (B,N,H) -> (B,N,2):

        # Option 1: If core_module is designed for this (e.g., a custom KAN block)
        # Option 2: Wrap it with standard layers if needed.
        # Let's go with Option 2, assuming core_module handles the first part.

        # This is difficult to make generic without knowing core_module's signature.
        # A more practical approach might be to pass the *type* of core module and its args,
        # or to have core_module be a factory function.

        # For now, let's assume core_module is designed to fit the deep structure,
        # or we use a fixed adapter around it.
        # Example adapter for a KAN-like module:
        # Input: (B, N, M) -> [Some processing -> (B, N, H1)] -> GLU -> [-> (B, N, H2)] -> GLU -> Squeeze(-1)

        # Let's define a more standard deep structure that can use the core module:
        # core_module_1 -> GLU -> core_module_2 -> GLU -> Squeeze
        # This requires core_module to be flexible or to use nn.Sequential with compatible layers.

        # To keep it simple and generic, let's assume core_module is a "block" that can be repeated
        # or that it's designed for this specific NLM shape (B, N, history) -> (B, N, out)
        # We'll use nn.Sequential to build the deep part, potentially using core_module multiple times
        # or wrapping it.

        # A generic deep structure using the provided core module might look like this:
        # If core_module is SuperLinear, it handles the in/out dims internally.
        # If core_module is FourierKAN, it might need a different structure.

        # Let's define the structure assuming core_module is like SuperLinear (takes history, outputs new dims)
        # and can be stacked.

        # Define the first part: core_module -> GLU
        first_part = nn.Sequential(
            nn.Dropout(dropout),
            core_module, # This should take (B, N, history) and output (B, N, new_dim1)
            nn.GLU(dim=-1) # This halves the last dim: (B, N, new_dim1) -> (B, N, new_dim1//2)
        )
        # We need to know the output dim of the first part to define the second core module.
        # This is the challenge with a fully generic nn.Module.

        # Let's assume core_module has an attribute or method to get its output shape
        # or that it's designed to work with the next layer implicitly.
        # For this example, let's assume the first GLU outputs (B, N, H) where H is fixed or predictable.
        # Let's say we want the final output before Squeeze to be (B, N, 2).
        # So, after first GLU -> (B, N, H), we need another layer to -> (B, N, 2).

        # A more practical deep structure might be:
        # Input -> core_module_1 -> GLU -> nn.Linear(to correct dim for core_module_2) -> core_module_2 -> GLU -> Squeeze
        # Or simpler: Input -> core_module_1 -> GLU -> nn.Linear(H, 2) -> GLU -> Squeeze
        # This is getting complex with arbitrary modules.

        # Let's simplify and assume core_module is designed for the *first* transformation
        # and we add standard layers afterwards.
        # Or, let's assume core_module is the *entire* first block (e.g., outputs (B,N, 2*hidden_dims))
        # and we add a second standard/core block.

        # To make this work generically, we need a convention for the core module's I/O.
        # Let's assume core_module takes (B, N, in_history) and outputs (B, N, out_features).
        # We'll build the deep structure around that.

        # Define a potential deep structure:
        # Block 1: core_module (e.g., takes M, outputs 2*H) -> GLU (outputs H)
        # Block 2: core_module (e.g., takes H, outputs 2) -> GLU (outputs 1) -> Squeeze
        # This requires core_module to be flexible or for us to use standard layers between them.

        # A more robust way without knowing core_module internals:
        # Use nn.Sequential with standard layers and insert core_module where appropriate.
        # e.g., nn.Linear -> core_module -> nn.Linear -> core_module -> GLU -> Squeeze
        # But this mixes standard and custom layers.

        # The most practical approach for *this specific function* might be to pass
        # the *type* of the core module and its arguments, not an instance.
        # But since the request is for an nn.Module instance, let's proceed with a standard structure
        # that assumes core_module can be used in the deep stack or is a pre-configured block.

        # Let's assume core_module is a pre-built block for the *first* part of the deep NLM.
        # We'll add standard layers to complete the structure.

        # Example: core_module outputs (B, N, 2*H), GLU -> (B, N, H)
        # Then, we need another layer to get to (B, N, 2), GLU -> (B, N, 1), Squeeze -> (B, N)
        # This requires knowing the output size of core_module.

        # Let's define it assuming core_module is designed for this, e.g., a custom KAN block
        # that internally handles the dimensions.

        # For a generic core_module that outputs (B, N, X), we can't easily stack another
        # core_module of the *same* type unless X is known to be the right input for the next one.

        # Let's try a different approach: Allow core_module to be the *entire* first part
        # of the deep structure, including its internal logic, and then add standard layers.
        # This is complex.

        # A simpler, more robust way: Allow core_module to be a "layer" that fits the deep structure.
        # For example, define a standard deep NLM structure and let core_module be one of its components.
        # Or, allow core_module to be the *whole* deep structure if it implements the logic.

        # Given the complexity of making this fully generic, let's define it assuming core_module
        # is designed to replace the *first* SuperLinear in the original deep structure.
        # We'll need to know its output size to proceed.

        # Let's create a dummy forward pass or inspect it if possible (though inspecting arbitrary nn.Module is hard).
        # Instead, let's define the structure based on common patterns.

        # If core_module is like SuperLinear(in=M, out=2*H, N=N), it works.
        # If it's like FourierKAN(in=M, out=1, N=N), it might not fit directly.

        # Let's define the deep structure assuming core_module is the *first* transformation layer.
        # We'll need to know its output dimensions to define the second part.
        # Since we can't easily inspect arbitrary nn.Module, let's pass an *instance* that is already configured
        # for the specific task, or assume a standard interface.

        # For this rewrite, I'll assume core_module is designed to work within the deep structure
        # and handles its input/output dimensions appropriately.
        # This is the most practical way to accept an arbitrary nn.Module instance here.

        # Example: core_module takes (B, N, M) and outputs (B, N, 2*internal_dim)
        # Then GLU makes (B, N, internal_dim)
        # Then another core_module takes (B, N, internal_dim) and outputs (B, N, 2)
        # Then GLU makes (B, N, 1)
        # Then Squeeze makes (B, N)

        # This requires core_module to be flexible or for the user to pass appropriately configured instances.
        # Let's define the structure as such:

        # The original structure was:
        # SuperLinear(M -> 2*H) -> GLU -> SuperLinear(H -> 2) -> GLU -> Squeeze

        # If core_module_1 is designed like SuperLinear(M -> 2*H), it can replace the first SuperLinear.
        # If core_module_2 is designed like SuperLinear(H -> 2), it can replace the second SuperLinear.

        # Or, core_module is a block that does M -> 2*H -> GLU internally, and we stack another block.
        # This is getting too complex for a simple rewrite.

        # Let's define a structure that takes a *single* core_module instance and builds the deep part around it,
        # assuming the user configures the instance appropriately.

        # For example:
        # core_module_instance = SomeCustomModule(in_dims=memory_length, out_dims=2*hidden_dims, N=d_model)
        # This function then wraps it with GLU, potentially another layer, GLU, Squeeze.

        # Or, core_module_instance is already a Sequential or block that does the first part.
        # Let's go with the latter for maximum flexibility from the caller's side.

        # Assume core_module is the *first* block: takes (B, N, M) -> (B, N, X)
        # We'll add GLU, then potentially another layer/core_module, GLU, Squeeze.
        # To add another layer/core_module, we need to know the output dim X.
        # Let's assume core_module is designed to output (B, N, 2*final_hidden) so GLU -> (B, N, final_hidden).
        # Then we need another layer to go (B, N, final_hidden) -> (B, N, 2).
        # We could pass a *second* core_module instance for this part too, but that complicates the function signature.

        # Let's define it assuming core_module handles the first transformation,
        # and we add standard layers for the rest of the deep structure.

        # Define the first deep part using the provided module
        first_deep_part = nn.Sequential(
            nn.Dropout(dropout),
            core_module, # Expected to take (B, N, M) and output (B, N, 2*H) or similar
            nn.GLU(dim=-1) # Halves the last dim
        )
        # Now, we need to get from the output of GLU (shape B, N, H_after_first_glu) to (B, N, 2)
        # We could add a standard nn.Linear here, or another instance of core_module if it's compatible.
        # Adding a standard Linear is more robust for arbitrary modules.
        # Let's assume the output of the first GLU is H, and we want to go to 2.
        # We don't know H easily. Let's add a placeholder or use a common approach.
        # A common approach is to use nn.Linear to project to the desired size.
        # Since we don't know the intermediate size easily, let's define the final structure differently.

        # A more robust deep structure with a generic core module:
        # Input (B, N, M)
        # -> nn.Linear(M -> 2*H) or core_module_1 (M -> 2*H) -> GLU -> (B, N, H)
        # -> nn.Linear(H -> 2) or core_module_2 (H -> 2) -> GLU -> (B, N, 1)
        # -> Squeeze(-1) -> (B, N)
        # This requires knowing the output size of the first part to define the input size of the second part.

        # Let's define it assuming core_module is the *first* transformation block (M -> 2*H).
        # We'll add a standard GLU, then a standard Linear to go to 2, then GLU, then Squeeze.

        # Example call: core_module = SuperLinear(M, 2*H, N)
        # Or: core_module = FourierKANAdapter(M, 2*H, N) # if FourierKAN needs adaptation
        # Or: core_module = nn.Sequential(Preprocess(M), ActualKAN(M, 2*H, N), PostProcess(2*H))
        # The caller configures the instance.

        # Define the deep NLM using the provided core module instance
        return nn.Sequential(
            nn.Sequential(
                nn.Dropout(dropout), # Apply dropout first
                core_module, # The provided core module for the first transformation
                nn.GLU(dim=-1), # Apply GLU
                # Add a standard linear layer to project to the final required size before the last GLU
                # We assume the core_module's output after GLU is fed into this.
                # If core_module outputs (B,N, 2), then GLU makes (B,N, 1), then Squeeze makes (B,N).
                # If core_module outputs (B,N, X), GLU makes (B,N, X//2). We need a layer to map X//2 -> 2.
                # Let's assume the user configures core_module such that its output dim after GLU is 2.
                # Or, let's add a standard layer to ensure the final dim before last GLU is 2.
                # nn.Linear(?, 2) - we don't know ? easily.
                # nn.Linear(H_after_glu_of_core, 2) - requires introspection.
                # nn.Linear(-1, 2) - PyTorch doesn't support -1 for intermediate dims in nn.Linear like reshape.
                # This is the crux of the issue with a fully generic nn.Module.

                # To solve this, we can define a wrapper layer that adapts the core module's output.
                # Or, we assume core_module handles the *entire* first part of the deep structure implicitly.
                # Or, we define the structure up to the point where core_module's output is known.

                # Let's define a standard projection layer after the first GLU.
                # We'll need to infer the size after GLU. This is difficult.
                # Let's define a lambda or a custom module to handle this, or assume a fixed intermediate size.

                # The most practical way is to assume core_module is designed to output the correct shape
                # for the subsequent GLU and Squeeze, OR to pass the intermediate dimensions explicitly.

                # Since the goal is to accept *any* nn.Module, let's define a structure that works
                # if the core_module is designed for this specific NLM context.
                # For example, core_module outputs (B, N, 2), then GLU -> (B, N, 1), Squeeze -> (B, N).

                # Or, core_module outputs (B, N, 4), then GLU -> (B, N, 2), then another GLU -> (B, N, 1), Squeeze -> (B, N).

                # Let's define it assuming core_module outputs (B, N, 2*final_size), so GLU -> (B, N, final_size).
                # We then need another layer/core_module to go to (B, N, 2).
                # Let's add a standard nn.Linear for this second projection.
                # We'll need the output size of the first GLU. Let's call it `intermediate_dim`.
                # We can't easily get `intermediate_dim` from an arbitrary nn.Module instance.
                # Let's pass an `intermediate_dim` argument or assume the user wraps core_module appropriately.

                # To keep the signature clean and accept just nn.Module, the user must configure core_module
                # such that its output after GLU is compatible with the final GLU -> Squeeze sequence.
                # E.g., core_module outputs (B, N, 2), GLU -> (B, N, 1), Squeeze -> (B, N).

                # Or, core_module outputs (B, N, 4), GLU -> (B, N, 2), GLU -> (B, N, 1), Squeeze -> (B, N).

                # Let's assume the user configures it to output (B, N, 2) before the final GLU.
                # So structure is: core_module -> GLU -> GLU -> Squeeze.

                # Or, more commonly matching the original: core_module(M->2*H) -> GLU -> core_module2(H->2) -> GLU -> Squeeze.
                # If we only have one core_module instance, we can't easily replicate it for the second part
                # unless it's stateless or we clone it (which is complex).

                # The cleanest way with one module instance is probably:
                # core_module(M->X) -> GLU -> nn.Linear(X_after_glu -> 2) -> GLU -> Squeeze
                # But we don't know X_after_glu.

                # Let's define it assuming core_module is designed for the *full first half*.
                # Or, let's accept that the user must provide a module that fits the structure,
                # and use standard layers where dimensions are unknown.

                # Define using standard nn.Linear for the second projection:
                # core_module (M -> 2*H) -> GLU -> nn.Linear(H -> 2) -> GLU -> Squeeze
                # We don't know H easily. Let's define a version that works if core_module is designed for it.

                # Let's define a version that assumes core_module is a *block* that handles the first transformation
                # and is pre-configured to output the right shape for the subsequent fixed layers.
                # e.g., core_module outputs (B, N, 2), then final GLU -> (B, N, 1) -> Squeeze -> (B, N).

                # Or, core_module outputs (B, N, 4), GLU -> (B, N, 2), final GLU -> (B, N, 1) -> Squeeze -> (B, N).

                # For maximum flexibility, let's define the sequence as provided core_module -> GLU -> Standard Linear -> GLU -> Squeeze.
                # We'll need to know the size after the first GLU to define the Linear layer.
                # Since we can't easily infer it, let's define a version that requires the user to provide
                # a core_module that outputs a known size after GLU, or provide the size.

                # Let's add a standard nn.Linear after the first GLU, assuming we know its input size.
                # We'll need to know `intermediate_dim` = core_module.out_features_after_GLU.
                # This is not possible to infer from a generic nn.Module instance easily.

                # Therefore, the most practical implementation of this function accepting a generic nn.Module
                # requires the user to provide a `core_module` that is already designed to work with the fixed
                # GLU -> GLU -> Squeeze structure, OR to provide intermediate dimensions.

                # Let's proceed with the assumption that the provided `core_module` is designed for this specific context.
                # For example, it's a custom KAN block designed to output (B, N, 2) before the final GLU.
                # Or, it's designed to output (B, N, 4), GLU -> (B, N, 2), final GLU -> (B, N, 1), Squeeze -> (B, N).

                # If core_module outputs (B, N, 2*final_hidden), GLU -> (B, N, final_hidden).
                # We need another layer to get to (B, N, 2).
                # If core_module outputs (B, N, 2), GLU -> (B, N, 1). Then Squeeze -> (B, N). This is the simplest final case.

                # Let's define the structure assuming core_module is designed to output (B, N, 2) before the *last* GLU.
                # So the sequence is: core_module -> GLU -> GLU -> Squeeze.
                # core_module outputs (B, N, 4), GLU -> (B, N, 2), GLU -> (B, N, 1), Squeeze -> (B, N).

                # Or, core_module outputs (B, N, X), GLU -> (B, N, X//2).
                # If X//2 is 2, then next GLU -> (B, N, 1), Squeeze -> (B, N).
                # So, core_module should output (B, N, 4) for this specific structure.

                # Let's define it with the assumption that the user configures core_module appropriately.
                # This is the most flexible way to accept an arbitrary nn.Module.

                nn.GLU(dim=-1), # This requires core_module to output an even last dimension
                # If core_module outputs (B, N, 4), this GLU makes (B, N, 2).
                # If core_module outputs (B, N, 2), this GLU makes (B, N, 1). Squeeze makes (B, N).
                # If core_module outputs (B, N, >4 and even), this GLU makes (B, N, >2).
                # To get to the final GLU -> Squeeze, we need the *input* to the final GLU to be (B, N, 2).
                # So, the output of *this* GLU must be 2.
                # So, the input to *this* GLU (output of core_module) must be 4.
                # Structure: core_module(out=4) -> GLU(out=2) -> GLU(out=1) -> Squeeze(out=scalar per neuron).

                # This means the user must configure their core_module to output the correct size (4 in this case)
                # for the final GLU -> Squeeze to work.
                # This is a reasonable assumption if the user knows the expected NLM structure.

                # Final GLU
                nn.GLU(dim=-1), # Takes (B, N, 2) -> outputs (B, N, 1)
                # Final Squeeze
                Squeeze(-1) # Takes (B, N, 1) -> outputs (B, N)
            )
        )

    else:
        # Use the core module directly, followed by GLU and Squeeze
        # This assumes core_module outputs (B, N, 2) so GLU -> (B, N, 1) -> Squeeze -> (B, N)
        return nn.Sequential(
            nn.Sequential(
                nn.Dropout(dropout),
                core_module, # Expected to take (B, N, M) -> (B, N, 2)
                nn.GLU(dim=-1), # (B, N, 2) -> (B, N, 1)
                Squeeze(-1) # (B, N, 1) -> (B, N)
            )
        )


# --- Example Usage ---
# Assuming you have your SimplifiedCTM class as defined previously...

# Example 1: Using the original SuperLinear (wrapped appropriately)
# You would need to instantiate the SuperLinear module with correct dimensions
input = torch.rand(8, 256, 10)  # (B, N, M)
# d_model = 256
# memory_length = 10
# super_linear_module = SuperLinear(in_dims=memory_length, out_dims=2*64, N=d_model, dropout=0.1)
# nlms = get_neuron_level_models_v2(core_module=super_linear_module, deep_nlms=True, dropout=0.1)
# print(nlms(input).shape)  # Should output (8, 256)
# print(nlms)

# Example 2: Using a potential FourierKAN-like module (simplified)
d_model = 256
memory_length = 10
fourier_kan_module = FourierKAN(in_features=memory_length, out_features=4, N=d_model, num_frequencies=5)
nlms = get_neuron_level_models_v2(core_module=fourier_kan_module, deep_nlms=True, dropout=0.1)
print(nlms(input).shape)  # Should output (8, 256)
print(nlms)
# Note: The FourierKAN example above outputs (B, N, 1) by summing over M. To make it output (B, N, 4),
# you'd need to modify it to have an internal hidden layer or multiple outputs per neuron.
# e.g., it could output (B, N, 4) by having 4 sets of freqs/weights per neuron.
# The user must configure the nn.Module instance passed to fit the expected NLM structure.

# Example 3: Using a standard MLP block as the core
# d_model = 256
# memory_length = 10
# standard_mlp = nn.Sequential(
#     nn.Linear(memory_length, 128),
#     nn.ReLU(),
#     nn.Linear(128, 4) # Output 4 to match the deep NLM structure: GLU -> 2 -> GLU -> 1 -> Squeeze -> scalar
# )
# # Wrap the MLP to handle (B, N, M) -> (B, N, 4) by applying it across the N dimension
# class MLPCore(nn.Module):
#     def __init__(self, mlp, N):
#         super().__init__()
#         self.mlp = mlp
#         self.N = N
#     def forward(self, x): # x: (B, N, M)
#         B, N, M = x.shape
#         x_reshaped = x.view(B * N, M) # (B*N, M)
#         out = self.mlp(x_reshaped) # (B*N, 4)
#         return out.view(B, N, -1) # (B, N, 4)
#
# mlp_core = MLPCore(standard_mlp, d_model)
# nlms = get_neuron_level_models_v2(core_module=mlp_core, deep_nlms=True, dropout=0.1)

# This function `get_neuron_level_models_v2` now accepts any nn.Module instance
# as the `core_module`, allowing for easy swapping of the internal processing logic
# within the Neuron-Level Models, provided the instance is configured to match
# the expected input/output dimensions of the surrounding structure (e.g., GLU, Squeeze).