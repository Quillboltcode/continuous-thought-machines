import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import math
from torch.nn import Identity
from tqdm import tqdm
import timm
import wandb
import matplotlib.pyplot as plt

from NLMonly import create_resnet18_backbone,set_seed
from get_dataloader import get_train_val_dataloaders, get_test_dataloader


class SuperLinear(nn.Module):
    """Your NLM implementation from CTM paper"""
    def __init__(self, in_dims, out_dims, N, T=1.0, do_norm=False, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
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


def get_timm_model_features_dim(model_name, pretrained=True):
    """Get the feature dimension of a timm model"""
    model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
    # Create a dummy input to get the feature dimension
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model(dummy_input)
    if isinstance(features, (list, tuple)):
        features = features[-1]  # Take the last feature map
    return features.shape[1]  # Return the channel dimension

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
        # Re-initialize if not initialized or if batch size changes
        if not self.history_initialized or (self.history_buffer is not None and self.history_buffer.size(0) != batch_size):
            # Initialize: Use .data to ensure it doesn't try to backprop through initialization
            self.history_buffer = features.unsqueeze(-1).expand(-1, -1, self.memory_length).clone().data
            self.history_initialized.fill_(True)
        else:
            # Detach the incoming features for memory update
            new_history = torch.cat([
                # If batch size changes, this could fail.
                # The check above handles re-initialization, but as a safeguard,
                # we ensure we only use a slice that matches the new batch size.
                # This part is now safer due to the re-initialization logic.
                self.history_buffer[:batch_size, :, 1:],
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
            # Re-initialize if not initialized or if batch size changes (e.g., last batch)
            if not self.gated_memory_initialized or self.gated_memory_state is None or self.gated_memory_state.size(0) != batch_size:
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
            # Re-initialize if not initialized or if batch size changes
            if not self.gated_memory_initialized or self.gated_memory_state is None or self.gated_memory_state.size(0) != batch_size:
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


class CTMClassifier(nn.Module):
    """Original CTM for comparison using timm models"""
    def __init__(self, num_classes, model_name='resnet18', memory_length=5, memory_hidden_dims=64, 
                 deep_nlms=True, do_layernorm_nlm=True, dropout=0.1, freeze_backbone=False, pretrained=True):
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
        
        # CTM NLMs: Each neuron processes its own temporal history
        if deep_nlms:
            self.nlm = nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=self.feature_dim,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=self.feature_dim,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            self.nlm = nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=self.feature_dim,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.memory_length = memory_length
        self.register_buffer('history_buffer', None)
        self.register_buffer('history_initialized', torch.tensor(False))
    
    def reset_history(self):
        self.history_initialized.fill_(False)
        self.history_buffer = None
    
    def update_history(self, features):
        batch_size, feature_dim = features.shape
        if not self.history_initialized:
            self.history_buffer = features.unsqueeze(-1).expand(-1, -1, self.memory_length)
            self.history_initialized.fill_(True)
        else:
            self.history_buffer = torch.cat([
                self.history_buffer[:, :, 1:].clone(),
                features.unsqueeze(-1)
            ], dim=-1)
        return self.history_buffer.clone()
    
    def forward(self, x, update_history=True):
        # Extract features using timm model
        features = self.feature_extractor(x)
        
        # Handle different output formats from timm
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take the last feature map
        
        # Reshape features: (B, C, H, W) -> (B, C) by global average pooling
        if features.dim() == 4:  # If we have spatial dimensions
            features = features.mean(dim=[2, 3])  # Global average pooling
        
        if update_history:
            history = self.update_history(features)
        else:
            history = self.history_buffer.clone() if self.history_buffer is not None else features.unsqueeze(-1).expand(-1, -1, self.memory_length)
        
        # CTM: Each feature has its own temporal processing (NLM)
        nlm_output = self.nlm(history)
        logits = self.classifier(nlm_output)
        return logits, nlm_output


def evaluate_model(model, dataloader, device, model_name='ctm', num_batches=None):
    """Evaluate model performance"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Handle different model types
            if model_name in ['ctm', 'memory_ctm']:
                outputs = model(images, update_memory=False)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:  # fallback
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / (i + 1)
    return accuracy, avg_loss


def run_memory_enhanced_ctm_comparison(dataset='cifar10', data_dir='./data', num_classes=10, 
                                     batch_size=16, epochs=5, lr=0.001, memory_length=5, 
                                     memory_hidden_dims=64, deep_nlms=True, do_layernorm_nlm=True,
                                     dropout=0.1, project_name='memory_ctm_comparison', 
                                     run_name='experiment_1', backbone_name='resnet18',
                                     models_to_run=['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']):
    """
    Compare Original CTM vs Memory-Enhanced CTM with timm models
    backbone_name: name of the backbone model (e.g., 'resnet18', 'efficientnet_b0')
    models_to_run: list of model types to run ['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']
    """
    # Initialize wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'dataset': dataset,
            'num_classes': num_classes,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'memory_length': memory_length,
            'memory_hidden_dims': memory_hidden_dims,
            'deep_nlms': deep_nlms,
            'do_layernorm_nlm': do_layernorm_nlm,
            'dropout': dropout,
            'backbone_name': backbone_name,
            'models_to_run': models_to_run,
        },
        reinit=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders using your new functions
    if dataset == 'cifar10':
        train_loader, val_loader = get_train_val_dataloaders(dataset_name=dataset, data_dir=data_dir, batch_size=batch_size)
        test_loader = get_test_dataloader(dataset_name=dataset, data_dir=data_dir, batch_size=batch_size)
        num_classes = 10
    else:
        train_loader, val_loader = get_train_val_dataloaders(dataset_name=dataset, data_dir=data_dir, batch_size=batch_size)
        test_loader = get_test_dataloader(dataset_name=dataset, data_dir=data_dir, batch_size=batch_size)
        num_classes = num_classes
    
    # Initialize models with timm backbone
    all_models_dict = {
        'original_ctm': CTMClassifier(
            num_classes, 
            model_name=backbone_name,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            deep_nlms=deep_nlms,
            do_layernorm_nlm=do_layernorm_nlm,
            dropout=dropout,
            freeze_backbone=False  # Train the whole model
        ),
        'memory_ctm_gated': MemoryAugmentedCTM(
            num_classes, 
            model_name=backbone_name,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            deep_nlms=deep_nlms,
            do_layernorm_nlm=do_layernorm_nlm,
            dropout=dropout,
            memory_type='gated',
            freeze_backbone=False  # Train the whole model
        ),
        'memory_ctm_attention': MemoryAugmentedCTM(
            num_classes, 
            model_name=backbone_name,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            deep_nlms=deep_nlms,
            do_layernorm_nlm=do_layernorm_nlm,
            dropout=dropout,
            memory_type='attention',
            freeze_backbone=False  # Train the whole model
        ),
        'memory_ctm_both': MemoryAugmentedCTM(
            num_classes, 
            model_name=backbone_name,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            deep_nlms=deep_nlms,
            do_layernorm_nlm=do_layernorm_nlm,
            dropout=dropout,
            memory_type='both',
            freeze_backbone=False  # Train the whole model
        )
    }
    
    # Filter models based on models_to_run
    models_dict = {name: model for name, model in all_models_dict.items() if name in models_to_run}
    
    results = {}
    all_metrics = {}
    
    print(f"\n{'='*80}")
    print(f"COMPARING CTM MODELS WITH BACKBONE: {backbone_name}")
    print(f"Models to run: {models_to_run}")
    print(f"{'='*80}\n")
    
    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} model...")
        print(f"Using backbone: {backbone_name}")
        print(f"{'='*50}")
        
        model = model.to(device)
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Use CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0
            
            # Reset memory for temporal models at the start of each epoch
            if name in ['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']:
                if hasattr(model, 'reset_history'):
                    model.reset_history()
                elif hasattr(model, 'reset_memory'):
                    model.reset_memory()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass - create a fresh computation graph each time
                # Reset memory for each batch to avoid gradient graph issues
                if batch_idx % 50 == 0:  # Reset every 50 batches to avoid too much temporal dependency
                    if name in ['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']:
                        if hasattr(model, 'reset_history'):
                            model.reset_history()
                        elif hasattr(model, 'reset_memory'):
                            model.reset_memory()
                
                if name == 'original_ctm':
                    output, _ = model(data, update_history=True)
                else:  # memory-enhanced models
                    output, _ = model(data, update_memory=True)
                
                loss = criterion(output, target)
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total_train_correct += (predicted == target).sum().item()
                total_train_samples += target.size(0)
                
                # Update progress bar
                train_acc = total_train_correct / total_train_samples
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_acc:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_acc = total_train_correct / total_train_samples
            
            # Validation - reset memory first
            model.eval()
            if name in ['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']:
                if hasattr(model, 'reset_history'):
                    model.reset_history()
                elif hasattr(model, 'reset_memory'):
                    model.reset_memory()
            
            val_acc, val_loss = evaluate_model(model, val_loader, device, model_name='memory_ctm' if 'memory' in name else 'ctm')
            
            # Update learning rate
            scheduler.step()
            
            # Store metrics
            train_accuracies.append(avg_train_acc)
            val_accuracies.append(val_acc)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f'{name}/epoch': epoch + 1,
                    f'{name}/train_loss': avg_train_loss,
                    f'{name}/train_acc': avg_train_acc,
                    f'{name}/val_loss': val_loss,
                    f'{name}/val_acc': val_acc,
                    f'{name}/lr': scheduler.get_last_lr()[0],
                })
            
            print(f"Epoch {epoch+1}: Train Acc: {avg_train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save a copy of the model state dict to avoid gradient graph issues
                model_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
                torch.save(model_state_dict, f'best_{name}_{dataset}.pth')
        
        # Final evaluation on test set using the best model
        print(f"Loading best model for final test evaluation...")
        # Load the saved model state dict
        best_model_state = torch.load(f'best_{name}_{dataset}.pth')
        model.load_state_dict(best_model_state)
        
        # Reset memory for final test evaluation
        if name in ['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both']:
            if hasattr(model, 'reset_history'):
                model.reset_history()
            elif hasattr(model, 'reset_memory'):
                model.reset_memory()
        
        # Final test evaluation
        test_acc, test_loss = evaluate_model(model, test_loader, device, model_name='memory_ctm' if 'memory' in name else 'ctm')
        
        results[name] = {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss
        }
        all_metrics[name] = {
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"Final Results - {name}: Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Log final results
    for name, metrics in results.items():
        wandb.run.summary[f'final_{name}_val_accuracy'] = metrics['val_acc']
        wandb.run.summary[f'final_{name}_test_accuracy'] = metrics['test_acc']
        wandb.run.summary[f'final_{name}_test_loss'] = metrics['test_loss']
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    print("Model Performance Comparison:")
    print(f"{'Model Name':25s} {'Val Acc':<10s} {'Test Acc':<10s}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name.upper():25s} {metrics['val_acc']:<10.4f} {metrics['test_acc']:<10.4f}")
    print(f"{'='*80}")
    
    # Create comparison plots only if there are models to plot
    if models_dict:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training accuracy
        for name in models_dict.keys():
            axes[0, 0].plot(all_metrics[name]['train_accuracies'], label=f'{name}', linewidth=2)
            axes[0, 1].plot(all_metrics[name]['val_accuracies'], label=f'{name}', linewidth=2)
            axes[1, 0].plot(all_metrics[name]['train_losses'], label=f'{name}', linewidth=2)
            axes[1, 1].plot(all_metrics[name]['val_losses'], label=f'{name}', linewidth=2)
        
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary bar plot - Test Accuracy
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(results.keys())
        test_accuracies = [results[name]['test_acc'] for name in names]
        val_accuracies = [results[name]['val_acc'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
        
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Validation vs Test Accuracy Comparison: {backbone_name} Backbone')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    # Finish wandb run
    wandb.finish()
    
    return results, all_metrics


# Run the comparison
if __name__ == "__main__":
    print("Memory-Enhanced CTM Comparison Functions Loaded")
    print("Use run_memory_enhanced_ctm_comparison() to start training")
    print("\nComparison includes:")
    print("- Original CTM: Neuron-level temporal processing only")
    print("- Memory CTM (Gated): + Gated Memory Unit (GRU-like)")
    print("- Memory CTM (Attention): + Attention-based memory (LLM-like)")
    print("- Memory CTM (Both): + Both memory mechanisms combined")
    # model = MemoryAugmentedCTM(num_classes=10, memory_type='both')
    # input = torch.randn(16,3 ,224,224)
    # output = model(input)
    # print("logits shape:",output[0].shape)
    # print("final_output shape:",output[1].shape)
    # Example usage:
    # Run only original CTM
    results, metrics = run_memory_enhanced_ctm_comparison(
        models_to_run=['memory_ctm_gated'],
        backbone_name='resnet18',
        epochs=2,
        batch_size =128
        )
    


    # results, metrics = run_memory_enhanced_ctm_comparison(
    #     models_to_run=['memory_ctm_gated'],
    #     run_name='experiment_rafdb_gated_1',
    #     backbone_name='resnet18',
    #     dataset='imagefolder',
    #     data_dir='./kaggle/dataset',
    #     num_classes=7,
    #     batch_size=16,
    #     epochs=5,
    #     lr=1e-3,
    #     memory_length=5,
    #     memory_hidden_dims=64,
    #     deep_nlms=True,
    #     do_layernorm_nlm=True,
    #     dropout=0.1,
    # )

    # Run original and one memory-enhanced version
    # results, metrics = run_memory_enhanced_ctm_comparison(
    #     models_to_run=['original_ctm', 'memory_ctm_gated'],
    #     backbone_name='efficientnet_b0'
    # )

    # Run all models
    # results, metrics = run_memory_enhanced_ctm_comparison(
    #     models_to_run=['original_ctm', 'memory_ctm_gated', 'memory_ctm_attention', 'memory_ctm_both'],
    #     backbone_name='resnet50'
    # )