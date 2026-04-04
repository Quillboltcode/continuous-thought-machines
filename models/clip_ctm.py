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


class CLIPCTM(ContinuousThoughtMachine):
    """
    CLIP-CTM using OpenCLIP vision encoder with adapter.
    
    Uses a trainable bottleneck adapter that maps frozen CLIP vision features 
    to CTM's latent reasoning space. The CLIP model remains completely frozen 
    during training, while only the adapter and CTM internals are trained.
    
    Architecture:
        CLIP vision encoder (frozen) 
            → CLIPBackboneAdapter (trainable adapter + alpha)
                → CTM kv_proj (trainable)
                    → CTM reasoning loop (trainable)
    
    Supports optional text prototypes for zero-shot classification:
        templates: List of prompt templates (e.g., "a photo of a {} expression")
        classes: List of class names (e.g., ["happy", "sad", "angry"])
        Text features are precomputed once and concatenated to visual features.
    
    Args:
        clip_model_name: OpenCLIP model name (e.g., 'ViT-B-32', 'ViT-L-14')
        pretrained: Pretrained dataset (e.g., 'laion2b_s34b_b79k', 'openai')
        adapter_reduction: Bottleneck reduction ratio for adapter (default: 4)
        alpha_init: Initial alpha value for adapter blend (default: 0.5)
        use_intermediate_layer: Add learnable per-patch positional embedding
        text_prompts: Optional dict with 'templates' and 'classes' for text prototypes
        **ctm_kwargs: Forwarded to ContinuousThoughtMachine
    """
    
    def __init__(self,
                 clip_model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k",
                 adapter_reduction=4,
                 alpha_init=0.5,
                 use_intermediate_layer=False,
                 text_prompts=None,
                 **ctm_kwargs):
        ctm_kwargs['backbone_type'] = 'none'
        ctm_kwargs['positional_embedding_type'] = 'none'
        ctm_kwargs['pretrained_backbone'] = None
        
        d_input = ctm_kwargs['d_input']
        
        super().__init__(**ctm_kwargs)
        
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        self.use_intermediate_layer = use_intermediate_layer
        self.text_prompts = text_prompts
        
        self.clip_model, self.clip_transform, self.tokenizer = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=pretrained
        )
        self.clip_model.eval()
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.n_patches = self._get_n_patches()
        self.clip_dim = self.clip_model.visual.output_dim
        
        self.backbone_adapter = CLIPBackboneAdapter(
            self.clip_model, 
            reduction=adapter_reduction, 
            alpha_init=alpha_init
        )
        
        self.patch_pe = None
        if use_intermediate_layer:
            self.patch_pe = nn.Embedding(self.n_patches, d_input)
        
        self.text_tokens = None
        self.text_proj = None
        if text_prompts is not None:
            self._setup_text_prompts(text_prompts)
        
        with torch.no_grad():
            n_tok = self.n_patches
            if self.text_tokens is not None:
                n_tok += self.text_tokens.shape[0]
            dummy_tokens = torch.zeros(1, n_tok, self.clip_dim)
            self.kv_proj(dummy_tokens)
    
    def _get_n_patches(self):
        if hasattr(self.clip_model.visual, 'patch_size'):
            img_size = getattr(self.clip_model.visual.image_size, 'val', 
                             getattr(self.clip_model.visual.image_size, 'grid_size', [7, 7]))
            if isinstance(img_size, (list, tuple)):
                return img_size[0] * img_size[1]
            return img_size ** 2
        return 196
    
    def _setup_text_prompts(self, text_prompts):
        """Precompute text prototypes from templates and classes."""
        templates = text_prompts.get('templates', ["a photo of a {}"])
        classes = text_prompts.get('classes', [])
        
        if not classes:
            return
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        
        text_prototypes = []
        for cls in classes:
            embeddings = []
            for template in templates:
                text = template.format(cls)
                tokens = self.tokenizer([text]).to(next(self.parameters()).device)
                with torch.no_grad():
                    emb = self.clip_model.encode_text(tokens)
                    emb = F.normalize(emb, dim=-1)
                    embeddings.append(emb)
            text_prototypes.append(torch.stack(embeddings).mean(0))
        
        text_prototypes = torch.stack(text_prototypes)
        text_prototypes = F.normalize(text_prototypes, dim=-1)
        
        self.register_buffer('text_prototypes', text_prototypes)
        
        tokens_list = []
        for cls in classes:
            for template in templates:
                text = template.format(cls)
                tokens_list.append(text)
        
        tokens = self.tokenizer(tokens_list).to(next(self.parameters()).device)
        with torch.no_grad():
            self.text_tokens = self.clip_model.encode_text(tokens)
        
        if self.text_tokens.shape[-1] != self.clip_dim:
            self.text_proj = nn.Linear(self.text_tokens.shape[-1], self.clip_dim, bias=False)
    
    def compute_features(self, x):
        """Extract frozen CLIP tokens and apply adapter."""
        B = x.size(0)
        
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        
        tokens = self.backbone_adapter(x)
        
        if self.text_tokens is not None:
            text = self.text_tokens.to(x.device).unsqueeze(0).expand(B, -1, -1).float()
            if self.text_proj is not None:
                text = self.text_proj(text)
            tokens = torch.cat([tokens, text], dim=1)
        
        if self.patch_pe is not None:
            positions = torch.arange(tokens.size(1), device=tokens.device)
            tokens = tokens + self.patch_pe(positions).unsqueeze(0)
        
        return tokens


def create_clip_ctm(clip_model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", **kwargs):
    """
    Factory function to create CLIP-CTM model.
    
    Example usage:
        # Basic model
        model = create_clip_ctm('ViT-B-32', 'laion2b_s34b_b79k', d_input=512, d_model=512, iterations=5)
        
        # With text prototypes for zero-shot classification
        model = create_clip_ctm(
            'ViT-B-32', 'laion2b_s34b_b79k',
            d_input=512, d_model=512, iterations=5,
            text_prompts={
                'templates': ["a photo of a {} expression", "a face showing {} emotion"],
                'classes': ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
            }
        )
    
    Example models:
        - 'ViT-B-32' with 'laion2b_s34b_b79k' or 'openai'
        - 'ViT-L-14' with 'laion2b_s32b_b82k' or 'openai'
        - 'ViT-H-14' with 'laion2b_s32b_b82k'
    """
    return CLIPCTM(clip_model_name=clip_model_name, pretrained=pretrained, **kwargs)