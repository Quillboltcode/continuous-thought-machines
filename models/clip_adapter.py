import torch
import torch.nn as nn
import open_clip
import torch.nn.functional as F

from models.ctm import ContinuousThoughtMachine


class CLIPBackboneAdapter(nn.Module):
    """
    CLIP-Adapter style backbone with frozen CLIP visual encoder and trainable bottleneck adapter.
    
    Architecture:
        CLIP vision encoder (frozen) → trainable bottleneck adapter → CTM
        
    The adapter uses a learnable alpha parameter to blend:
        adapted = alpha * adapter(tokens) + (1 - alpha) * tokens
    
    Args:
        clip_model: OpenCLIP model with .visual attribute
        reduction: Bottleneck reduction ratio (default: 4)
        alpha_init: Initial value for alpha blend parameter (default: 0.5)
    """
    def __init__(self, clip_model, reduction=4, alpha_init=0.5):
        super().__init__()
        self.clip_visual = clip_model.visual
        d = self.clip_visual.output_dim
        
        for param in self.clip_visual.parameters():
            param.requires_grad = False
        
        self.adapter = nn.Sequential(
            nn.Linear(d, d // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(d // reduction, d, bias=False),
            nn.ReLU()
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        self.output_dim = d
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    
    def _extract_patch_tokens(self, x):
        """Extract patch tokens from frozen CLIP visual encoder."""
        B = x.shape[0]
        
        mean = self.mean.to(x.device).view(1, 3, 1, 1)
        std = self.std.to(x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std
        
        patch_tokens = self.clip_visual.conv1(x_norm)
        H = W = self.clip_visual.grid_size[0]
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], patch_tokens.shape[1], H * W).permute(0, 2, 1)
        
        cls_token = self.clip_visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.clip_visual.positional_embedding
        
        tokens = self.clip_visual.ln_pre(tokens)
        tokens = self.clip_visual.transformer(tokens)
        tokens = self.clip_visual.ln_post(tokens)
        
        tokens = tokens @ self.clip_visual.proj
        
        patch_tokens = tokens[:, 1:, :]
        return patch_tokens
    
    def forward(self, x):
        with torch.no_grad():
            tokens = self._extract_patch_tokens(x)
        
        adapted = self.alpha * self.adapter(tokens) + (1 - self.alpha) * tokens
        return adapted


class CLIPAdapterBaseline(nn.Module):
    """
    CLIP-Adapter baseline model without CTM.
    
    Architecture:
        CLIP vision encoder (frozen) 
            → CLIPBackboneAdapter (trainable adapter + alpha)
                → Pooled features (mean over patches)
                    → Linear classifier (trainable)
    
    This serves as a baseline to compare with CLIP-CTM.
    Uses standard cross-entropy loss for training.
    
    Args:
        clip_model_name: OpenCLIP model name (e.g., 'ViT-B-32', 'ViT-L-14')
        pretrained: Pretrained dataset (e.g., 'laion2b_s34b_b79k', 'openai')
        adapter_reduction: Bottleneck reduction ratio for adapter (default: 4)
        alpha_init: Initial alpha value for adapter blend (default: 0.5)
        num_classes: Number of output classes
        use_text_prompts: Whether to concatenate text prompt features
        text_prompts: Dict with 'templates' and 'classes' for text features
    """
    
    def __init__(self,
                 clip_model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k",
                 adapter_reduction=4,
                 alpha_init=0.5,
                 num_classes=7,
                 use_text_prompts=False,
                 text_prompts=None):
        
        super().__init__()
        
        self.clip_model_name = clip_model_name
        self.use_text_prompts = use_text_prompts
        self.num_classes = num_classes
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=pretrained
        )
        self.clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_dim = self.clip_model.visual.output_dim
        
        self.backbone_adapter = CLIPBackboneAdapter(
            self.clip_model, 
            reduction=adapter_reduction, 
            alpha_init=alpha_init
        )
        
        self.text_tokens = None
        self.text_proj = None
        
        # Process text prompts if provided
        if use_text_prompts and text_prompts is not None:
            self._setup_text_prompts(text_prompts)
        
        # Compute input dim: CLIP features only (text prompts not used for classification)
        input_dim = self.clip_dim
        
        # Linear classifier (linear probe style)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _setup_text_prompts(self, text_prompts):
        """Precompute text features from templates and classes."""
        templates = text_prompts.get('templates', ["a photo of a {}"])
        classes = text_prompts.get('classes', [])
        
        if not classes:
            return
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        
        # Encode text prompts
        text_features = []
        for cls in classes:
            for template in templates:
                text = template.format(cls)
                tokens = self.tokenizer([text]).to(next(self.parameters()).device)
                with torch.no_grad():
                    emb = self.clip_model.encode_text(tokens)
                    text_features.append(emb)
        
        self.text_tokens = torch.stack(text_features).mean(0)  # (num_classes, clip_dim)
        
        if self.text_tokens.shape[-1] != self.clip_dim:
            self.text_proj = nn.Linear(self.text_tokens.shape[-1], self.clip_dim, bias=False)
    
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, return features + logits
        
        Returns:
            logits: (B, num_classes) if return_features=False
            (features, logits): if return_features=True
        """
        B = x.size(0)
        
        # Get adapted visual features
        visual_features = self.backbone_adapter(x)  # (B, n_patches, clip_dim)
        
        # Pool visual features (mean over patches)
        pooled_visual = visual_features.mean(dim=1)  # (B, clip_dim)
        
        # Use linear classifier
        logits = self.classifier(pooled_visual)
        
        if return_features:
            return pooled_visual, logits
        return logits


def create_clip_adapter_baseline(clip_model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", **kwargs):
    """
    Factory function to create CLIP-Adapter baseline model.
    """
    return CLIPAdapterBaseline(
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        **kwargs
    )