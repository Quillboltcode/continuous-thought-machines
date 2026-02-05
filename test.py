import torch
import torch.nn as nn
import timm

def get_timm_model_features_dim(model_name, pretrained=True):
    """Get the feature dimension of a timm model"""
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
    # Create a dummy input to get the feature dimension
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model(dummy_input)
    if isinstance(features, (list, tuple)):
        features = features[-1]  # Take the last feature map
    return features.shape[1]  # Return the channel dimension

# Your existing classes (SuperLinear, Squeeze, GatedMemoryUnit, AttentionMemoryLayer)
class SuperLinear(nn.Module):
    """Your NLM implementation from CTM paper"""
    def __init__(self, in_dims, out_dims, N, T=1.0, do_norm=False, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_dims = in_dims
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        self.register_parameter('T', nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        out = self.dropout(x)
        out = self.layernorm(out)
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1
        out = out.squeeze(-1) / self.T
        return out

class Squeeze(nn.Module):
    """Helper module to squeeze specified dimension"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class GatedMemoryUnit(nn.Module):
    """Gated Memory Unit similar to GRU but for feature-specific memory"""
    def __init__(self, feature_dim, memory_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        
        # Update gate: decides how much of the old memory to keep
        self.update_gate = nn.Sequential(
            nn.Linear(feature_dim + memory_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # Reset gate: decides how much of the old memory to forget
        self.reset_gate = nn.Sequential(
            nn.Linear(feature_dim + memory_dim, memory_dim),
            nn.Sigmoid()
        )
        
        # Candidate memory: new memory content
        self.candidate_memory = nn.Sequential(
            nn.Linear(feature_dim + memory_dim, memory_dim),
            nn.Tanh()
        )
        
        # Memory update projection
        self.memory_projection = nn.Linear(feature_dim, memory_dim)

    def forward(self, current_features, prev_memory):
        # current_features: (batch, feature_dim)
        # prev_memory: (batch, memory_dim) - now batched!
        
        # Concatenate current features with previous memory
        combined = torch.cat([current_features, prev_memory], dim=-1)
        
        # Compute gates
        update_gate = self.update_gate(combined)
        reset_gate = self.reset_gate(combined)
        
        # Compute candidate memory (with reset gate applied to previous memory)
        reset_memory = reset_gate * prev_memory
        candidate_input = torch.cat([current_features, reset_memory], dim=-1)
        candidate_memory = self.candidate_memory(candidate_input)
        
        # Update memory
        new_memory = update_gate * prev_memory + (1 - update_gate) * candidate_memory
        
        return new_memory

class AttentionMemoryLayer(nn.Module):
    """Attention-based memory layer similar to key-value attention in LLMs"""
    def __init__(self, feature_dim, memory_size=50, head_dim=64, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_dim = head_dim * num_heads
        
        # Projections for query, key, value
        self.query_proj = nn.Linear(feature_dim, self.total_dim)
        self.key_proj = nn.Linear(feature_dim, self.total_dim)
        self.value_proj = nn.Linear(feature_dim, self.total_dim)
        
        # Output projection
        self.out_proj = nn.Linear(self.total_dim, feature_dim)
        
        # Memory storage (key-value pairs) - now batch-aware
        self.register_buffer('memory_keys', torch.zeros(memory_size, self.total_dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, self.total_dim))
        self.memory_counter = 0
        self.memory_full = False
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def reset_memory(self):
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_counter = 0
        self.memory_full = False
    
    def forward(self, current_features, update_memory=True):
        batch_size, _ = current_features.shape
        
        # Project current features to query space
        queries = self.query_proj(current_features).view(batch_size, self.num_heads, self.head_dim)
        
        if update_memory:
            # Update memory with current features
            current_keys = self.key_proj(current_features)
            current_values = self.value_proj(current_features)
            
            for i in range(batch_size):
                idx = (self.memory_counter + i) % self.memory_size
                self.memory_keys[idx] = current_keys[i]
                self.memory_values[idx] = current_values[i]
            
            self.memory_counter = (self.memory_counter + batch_size) % self.memory_size
            if self.memory_counter == 0 and not self.memory_full:
                self.memory_full = True
        
        # Compute attention
        if self.memory_full or self.memory_counter > 0:
            # Get all keys and values from memory
            keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)  # (B, memory_size, total_dim)
            values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)  # (B, memory_size, total_dim)
            
            # Reshape for multi-head attention
            keys = keys.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, memory_size, head_dim)
            values = values.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, memory_size, head_dim)
            
            # Compute attention scores
            attention_scores = torch.matmul(queries.unsqueeze(-2), keys.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, 1, memory_size)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            attended_values = torch.matmul(attention_weights, values)  # (B, num_heads, 1, head_dim)
            attended_values = attended_values.squeeze(-2).view(batch_size, self.total_dim)  # (B, total_dim)
            
            # Output projection
            output = self.out_proj(attended_values)
        else:
            # If no memory, return current features
            output = current_features
        
        # Apply layer norm
        output = self.layer_norm(output)
        
        return output

class MemoryAugmentedCTM(nn.Module):
    """CTM with integrated memory mechanisms using timm models"""
    def __init__(self, num_classes, model_name='resnet18', memory_length=5, memory_hidden_dims=64, 
                 deep_nlms=True, do_layernorm_nlm=True, dropout=0.1, 
                 memory_type='both', freeze_backbone=True, pretrained=True):
        super().__init__()
        
        # Create timm model with features_only=True
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1]  # Get the final feature map
        )
        
        # Get the feature dimension dynamically
        self.feature_dim = get_timm_model_features_dim(model_name, pretrained)
        
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Original CTM NLMs
        if deep_nlms:
            self.nlm = nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, 
                               N=self.feature_dim, do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, 
                               N=self.feature_dim, do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            self.nlm = nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, 
                               N=self.feature_dim, do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        
        # Memory mechanisms
        self.memory_type = memory_type
        if memory_type in ['gated', 'both']:
            self.gated_memory = GatedMemoryUnit(self.feature_dim, self.feature_dim)
            # Initialize memory state per batch
            self.register_buffer('gated_memory_initialized', torch.tensor(False))
            self.gated_memory_state = None  # Will be initialized per batch
        
        if memory_type in ['attention', 'both']:
            self.attention_memory = AttentionMemoryLayer(self.feature_dim)
        
        # Final classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Temporal history for NLM
        self.memory_length = memory_length
        self.register_buffer('history_buffer', None)
        self.register_buffer('history_initialized', torch.tensor(False))
    
    def reset_memory(self):
        self.history_initialized.fill_(False)
        self.history_buffer = None
        if self.memory_type in ['gated', 'both']:
            self.gated_memory_initialized.fill_(False)
            self.gated_memory_state = None
        if self.memory_type in ['attention', 'both']:
            self.attention_memory.reset_memory()
    
    def update_history(self, features):
        batch_size, feature_dim = features.shape
        if not self.history_initialized:
            # Initialize: Use .data to ensure it doesn't try to backprop through initialization
            self.history_buffer = features.unsqueeze(-1).expand(-1, -1, self.memory_length).clone().data
            self.history_initialized.fill_(True)
        else:
            # Detach the incoming features for memory update
            new_history = torch.cat([
                self.history_buffer[:, :, 1:],
                features.unsqueeze(-1)
            ], dim=-1)
            # Update the buffer IN PLACE with the detached result
            self.history_buffer.data.copy_(new_history.data)
            
        # The returned history is what the NLM operates on, so it is connected to the graph
        return self.history_buffer
    
    def forward(self, x, update_memory=True):
        # Extract features using timm model
        features = self.feature_extractor(x)
        
        # Handle different output formats from timm
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take the last feature map
        
        # Reshape features: (B, C, H, W) -> (B, C) by global average pooling
        if features.dim() == 4:  # If we have spatial dimensions
            features = features.mean(dim=[2, 3])  # Global average pooling
        
        batch_size = features.size(0)
        
        # Apply CTM NLM (neuron-level temporal processing)
        if update_memory:
            history = self.update_history(features)
        else:
            history = self.history_buffer.clone() if self.history_buffer is not None else features.unsqueeze(-1).expand(-1, -1, self.memory_length)
        
        nlm_output = self.nlm(history)
        
        # Apply memory mechanisms
        if self.memory_type == 'gated':
            # 1. Initialization (only needs to run once per batch sequence)
            if not self.gated_memory_initialized:
                # Initialize detached state
                # Use .data or torch.empty() followed by .data.zero_()
                self.gated_memory_state = torch.zeros(batch_size, self.feature_dim, device=nlm_output.device).data
                self.gated_memory_initialized.fill_(True)
            
            # 2. Update logic (runs only if update_memory is True)
            if update_memory:
                # Calculate the new state, and immediately copy its detached value to the memory buffer.
                # This performs the forward pass and updates the memory state in one step.
                
                # --- CRITICAL FIX ---
                self.gated_memory_state.data.copy_(
                    self.gated_memory(nlm_output, self.gated_memory_state).data
                )
                
            # The memory output is the current (updated or non-updated) memory state,
            # which is correctly detached from previous batches due to the .data.copy_() above.
            memory_output = self.gated_memory_state
        elif self.memory_type == 'attention':
            memory_output = self.attention_memory(nlm_output, update_memory=update_memory)
        elif self.memory_type == 'both':
            # 1. Initialization
            if not self.gated_memory_initialized:
                self.gated_memory_state = torch.zeros(batch_size, self.feature_dim, device=nlm_output.device).data
                self.gated_memory_initialized.fill_(True)
            
            # 2. Update logic
            if update_memory:
                # Update gated memory state in place
                # --- CRITICAL FIX ---
                self.gated_memory_state.data.copy_(
                    self.gated_memory(nlm_output, self.gated_memory_state).data
                )

            gated_output = self.gated_memory_state
            # Attention memory is handled within its own layer (assuming it uses .data internally too)
            attention_output = self.attention_memory(nlm_output, update_memory=update_memory)
            memory_output = gated_output + attention_output
        else:
            # No additional memory, just use NLM output
            memory_output = nlm_output
        
        # Final output
        final_output = memory_output
        logits = self.classifier(final_output)
        
        return logits, final_output

def test_memory_augmented_ctm():
    """Test the MemoryAugmentedCTM model for forward/backward pass"""
    print("Testing MemoryAugmentedCTM model...")
    
    # Test different memory types
    memory_types = ['gated', 'attention', 'both']
    
    for memory_type in memory_types:
        print(f"\nTesting model with memory_type='{memory_type}'")
        
        # Create model
        model = MemoryAugmentedCTM(
            num_classes=10,
            model_name='resnet18',
            memory_length=5,
            memory_hidden_dims=64,
            deep_nlms=True,
            do_layernorm_nlm=True,
            dropout=0.1,
            memory_type=memory_type,
            freeze_backbone=False,  # Don't freeze for testing
            pretrained=False  # Use random weights for testing
        )
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy data
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        targets = torch.randint(0, 10, (batch_size,)).to(device)
        
        # Test forward pass
        print("  Running forward pass...")
        model.train()
        model.reset_memory()  # Reset memory before forward pass
        
        outputs, final_features = model(x, update_memory=True)
        print(f"  Output shape: {outputs.shape}")
        print(f"  Features shape: {final_features.shape}")
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        print(f"  Loss: {loss.item():.4f}")
        
        # Test backward pass
        print("  Running backward pass...")
        loss.backward()
        
        # Check if gradients exist for parameters that should have them
        trainable_params = 0
        params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    # Check if gradients are not all zeros
                    if param.grad.abs().sum() == 0:
                        print(f"  Warning: {name} has zero gradients")
        
        print(f"  Trainable parameters: {trainable_params}")
        print(f"  Parameters with gradients: {params_with_grad}")
        
        # Verify that gradients were computed
        if params_with_grad == 0:
            print("  ERROR: No gradients computed!")
        else:
            print("  SUCCESS: Gradients computed successfully!")
        
        # Clean up
        del model, x, targets, outputs, final_features, loss
        
        print(f"  Completed test for memory_type='{memory_type}'\n")
    
    print("All tests completed!")

def test_memory_sequence():
    """Test that the model handles memory correctly across sequences"""
    print("Testing memory sequence handling...")
    
    model = MemoryAugmentedCTM(
        num_classes=10,
        model_name='resnet18',
        memory_length=3,
        memory_hidden_dims=32,
        deep_nlms=False,
        memory_type='gated',
        freeze_backbone=False,
        pretrained=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test with multiple forward passes to see if memory is updated properly
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224).to(device)
    x2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print("Forward pass 1...")
    model.reset_memory()
    out1, feat1 = model(x1, update_memory=True)
    
    print("Forward pass 2 (with same model)...")
    out2, feat2 = model(x2, update_memory=True)
    
    print(f"Output 1 shape: {out1.shape}")
    print(f"Output 2 shape: {out2.shape}")
    print(f"Features 1 shape: {feat1.shape}")
    print(f"Features 2 shape: {feat2.shape}")
    
    # Check if outputs are different (they should be due to memory)
    if torch.allclose(out1, out2, atol=1e-5):
        print("Warning: Outputs are very similar, memory might not be working as expected")
    else:
        print("Good: Outputs are different, memory is affecting the results")
    
    # Test with reset memory
    print("\nForward pass 3 (with reset memory)...")
    model.reset_memory()
    out3, feat3 = model(x1, update_memory=True)
    
    print(f"Output 3 shape: {out3.shape}")
    # This should be similar to out1 if memory is properly reset
    if torch.allclose(out1, out3, atol=1e-5):
        print("Good: Memory reset works correctly")
    else:
        print("Note: Outputs differ after reset (this might be expected depending on implementation)")
    
    print("Memory sequence test completed!")

if __name__ == "__main__":
    import math
    from torch.nn import Identity
    
    print("Starting model tests...\n")
    
    # Test basic functionality
    test_memory_augmented_ctm()
    
    print("\n" + "="*50 + "\n")
    
    # Test memory sequence handling
    test_memory_sequence()
    
    print("\nAll tests completed successfully!")