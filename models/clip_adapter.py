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
        self.clip_model = clip_model
        self.clip_visual = clip_model.visual
        self.clip_text = clip_model.transformer
        
        d_visual = self.clip_visual.output_dim  # 512 (projected CLS dim)
        d_text = self.clip_text.width  # 512
        
        for param in self.clip_visual.parameters():
            param.requires_grad = False
        for param in self.clip_text.parameters():
            param.requires_grad = False
        
        # Visual adapter (operates on projected CLS like trung_adapter)
        self.adapter_visual = nn.Sequential(
            nn.Linear(d_visual, d_visual // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(d_visual // reduction, d_visual, bias=False),
            nn.ReLU()
        )
        
        # Text adapter (uses text transformer width)
        self.adapter_text = nn.Sequential(
            nn.Linear(d_text, d_text // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(d_text // reduction, d_text, bias=False),
            nn.ReLU()
        )
        
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))
        
        self.output_dim = d_visual
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    
    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)
    
    def _extract_patch_tokens(self, x):
        """Extract CLS token from frozen CLIP visual encoder."""
        B = x.shape[0]
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        
        patch_tokens = self.clip_visual.conv1(x_norm)
        H = W = self.clip_visual.grid_size[0]
        patch_tokens = patch_tokens.reshape(patch_tokens.shape[0], patch_tokens.shape[1], H * W).permute(0, 2, 1)
        
        cls_token = self.clip_visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.clip_visual.positional_embedding
        
        tokens = self.clip_visual.ln_pre(tokens)
        tokens = self.clip_visual.transformer(tokens)
        
        # Use CLS token (like trung_adapter - already projected)
        cls_token_out = self.clip_visual.ln_post(tokens[:, 0, :])
        cls_token_out = cls_token_out @ self.clip_visual.proj
        
        return cls_token_out.unsqueeze(1)  # (B, 1, 512)
    
    def _extract_text_tokens(self, tokens):
        """Extract CLS token from frozen CLIP text encoder."""
        B = tokens.shape[0]
        
        x = self.clip_model.token_embedding(tokens).float()
        x = x + self.clip_model.positional_embedding[:tokens.shape[1]]
        x = x.permute(1, 0, 2)
        x = self.clip_text(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        cls_token = x[:, 0, :]
        
        return cls_token
    
    def forward(self, x, text_tokens=None):
        """Forward pass for visual features, or both visual and text if text_tokens provided."""
        if text_tokens is None:
            # Visual only - CLS token
            with torch.no_grad():
                tokens = self._extract_patch_tokens(x)  # (B, 1, 512)
            adapted = self.alpha * self.adapter_visual(tokens) + (1 - self.alpha) * tokens
            return adapted.squeeze(1)  # (B, 512)
        else:
            # Both visual and text
            with torch.no_grad():
                visual_tokens = self._extract_patch_tokens(x)
                text_raw = self._extract_text_tokens(text_tokens)
            
            visual_adapted = self.alpha * self.adapter_visual(visual_tokens) + (1 - self.alpha) * visual_tokens
            
            text_adapted = self.alpha * self.adapter_text(text_raw.unsqueeze(1)) + (1 - self.alpha) * text_raw.unsqueeze(1)
            
            return visual_adapted.squeeze(1), text_adapted.squeeze(1)


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
        
        self.backbone_adapter = CLIPBackboneAdapter(
            self.clip_model, 
            reduction=adapter_reduction, 
            alpha_init=alpha_init
        )
        
        self.clip_dim = self.backbone_adapter.output_dim  # projected dim (512)
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        if use_text_prompts and text_prompts is not None:
            self._setup_text_prompts(text_prompts)
        
        self.classifier = nn.Linear(self.clip_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _setup_text_prompts(self, text_prompts):
        templates = text_prompts.get('templates', ["a photo of a {}"])
        classes = text_prompts.get('classes', [])
        
        if not classes:
            return
        
        text_tokens_list = []
        for cls in classes:
            for template in templates:
                text_tokens_list.append(template.format(cls))
        
        tokens = self.tokenizer(text_tokens_list).to(next(self.parameters()).device)
        with torch.no_grad():
            all_embeds = self.clip_model.encode_text(tokens)
        
        # Store raw text features (not adapted) - allow gradients to flow to visual adapter
        all_embeds = all_embeds.reshape(len(classes), len(templates), -1)
        self._raw_text_tokens = all_embeds.mean(dim=1)  # (num_classes, 512)
    
    def forward(self, x, return_features=False, return_text_features=False):
        # Get CLS feature (already single token, no pooling needed)
        visual_features = self.backbone_adapter(x)  # (B, 512)
        
        if self.use_text_prompts and hasattr(self, '_raw_text_tokens') and self._raw_text_tokens is not None:
            # Use raw CLIP text features
            text_features_raw = self._raw_text_tokens.to(visual_features.device)
            
            # Normalize BOTH before similarity (like trung_adapter)
            pooled = F.normalize(visual_features, dim=-1)
            text_features = F.normalize(text_features_raw, dim=-1)
            
            if return_text_features:
                return pooled, text_features
            logits = pooled @ text_features.T * self.clip_model.logit_scale.exp()
        else:
            if return_text_features:
                raise ValueError("return_text_features=True but text prompts not enabled")
            pooled = F.normalize(visual_features, dim=-1)
            logits = self.classifier(pooled)
        
        if return_features:
            return pooled, logits
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