import torch
import torch.nn as nn
import open_clip
import torch.nn.functional as F

from models.ctm import ContinuousThoughtMachine


class CLIPBackboneAdapter(nn.Module):
    def __init__(self, clip_model, num_classes=7, reduction=4, 
                 use_text=False, text_prompts=None, alpha_init=0.5):
        super().__init__()
        self.clip_model = clip_model
        self.clip_visual = clip_model.visual
        
        self.use_text = use_text
        self.num_classes = num_classes
        
        d = self.clip_visual.output_dim
        
        for param in self.clip_visual.parameters():
            param.requires_grad = False
        
        self.adapter = nn.Sequential(
            nn.Linear(d, d // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(d // reduction, d, bias=False),
            nn.ReLU()
        )
        
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        
        self.output_dim = d
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        
        if use_text:
            self.visual_type_emb = nn.Parameter(torch.zeros(1, 1, d))
            self.text_type_emb   = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.normal_(self.visual_type_emb, std=0.02)
            nn.init.normal_(self.text_type_emb,   std=0.02)
            
            model_name = getattr(clip_model, 'name', 'ViT-B-32')
            if not isinstance(model_name, str):
                model_name = 'ViT-B-32'
            protos = self._compute_text_prototypes(
                clip_model, text_prompts, num_classes, model_name)
            self.register_buffer('text_prototypes', protos)
            
            self.text_adapter = nn.Sequential(
                nn.Linear(d, d // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(d // reduction, d, bias=False),
                nn.ReLU()
            )
            self.text_alpha = nn.Parameter(torch.tensor(0.0))
    
    @staticmethod
    def _compute_text_prototypes(clip_model, text_prompts, num_classes, model_name='ViT-B-32'):
        templates = text_prompts.get('templates', ["a photo of a {} face"])
        classes   = text_prompts.get('classes',
                    ["angry","disgust","fear","happy","sad","surprise","neutral"])
        tokenizer = open_clip.get_tokenizer(model_name)

        protos = []
        with torch.no_grad():
            for cls in classes:
                embs = []
                for t in templates:
                    tok = tokenizer([t.format(cls)])
                    embs.append(F.normalize(
                        clip_model.encode_text(tok), dim=-1))
                protos.append(torch.stack(embs).mean(0))
        return torch.cat(protos, dim=0)
    
    def _extract_patch_tokens(self, x):
        B = x.shape[0]
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        
        patch_tokens = self.clip_visual.conv1(x_norm)
        H, W = self.clip_visual.grid_size
        patch_tokens = patch_tokens.reshape(B, patch_tokens.shape[1], H * W).permute(0, 2, 1)
        
        cls_token = self.clip_visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.clip_visual.positional_embedding
        
        tokens = self.clip_visual.ln_pre(tokens)
        tokens = self.clip_visual.transformer(tokens)
        
        patch_tokens = tokens[:, 1:, :]
        patch_tokens = self.clip_visual.ln_post(patch_tokens)
        patch_tokens = patch_tokens @ self.clip_visual.proj
        
        return patch_tokens
    
    def forward(self, x):
        with torch.no_grad():
            visual_tokens = self._extract_patch_tokens(x)

        v_alpha = torch.sigmoid(self.alpha)
        adapted_visual = (v_alpha * self.adapter(visual_tokens) 
                         + (1 - v_alpha) * visual_tokens)

        if not self.use_text:
            return adapted_visual

        B = x.size(0)

        text_tokens = self.text_prototypes.unsqueeze(0).expand(B, -1, -1)
        t_alpha = torch.sigmoid(self.text_alpha)
        adapted_text = (t_alpha * self.text_adapter(text_tokens)
                       + (1 - t_alpha) * text_tokens)

        adapted_visual = adapted_visual + self.visual_type_emb
        adapted_text   = adapted_text   + self.text_type_emb

        combined = torch.cat([adapted_visual, adapted_text], dim=1)
        return combined
    
    def train(self, mode=True):
        super().train(mode)
        self.clip_visual.eval()
        return self


class CLIPAdapterBaseline(nn.Module):
    def __init__(self,
                 clip_model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k",
                 adapter_reduction=4,
                 alpha_init=0.2,
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
            alpha_init=alpha_init,
            use_text=use_text_prompts,
            text_prompts=text_prompts
        )
        
        self.clip_dim = self.backbone_adapter.output_dim
        
        self.classifier = nn.Linear(self.clip_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        if use_text_prompts and text_prompts is not None:
            self._setup_text_prompts(text_prompts)
    
    def _setup_text_prompts(self, text_prompts):
        templates = text_prompts.get('templates', ["a photo of a {}"])
        classes = text_prompts.get('classes', [])
        
        if not classes:
            return
        
        tokens_per_class = []
        for cls in classes:
            embs = []
            for template in templates:
                tok = self.tokenizer([template.format(cls)])
                with torch.no_grad():
                    emb = F.normalize(self.clip_model.encode_text(tok), dim=-1)
                embs.append(emb)
            tokens_per_class.append(torch.stack(embs).mean(0))
        
        protos = torch.cat(tokens_per_class, dim=0)
        self.register_buffer('text_prototypes_baseline', protos)
    
    def configure_optimizers(self, lr=1e-3, alpha_lr=1e-4, weight_decay=0.01):
        param_groups = [
            {'params': self.backbone_adapter.adapter.parameters(),
             'lr': lr, 'weight_decay': weight_decay},
            {'params': [self.backbone_adapter.alpha],
             'lr': alpha_lr, 'weight_decay': 0.0},
            {'params': self.classifier.parameters(),
             'lr': lr, 'weight_decay': weight_decay},
        ]
        if self.backbone_adapter.use_text:
            param_groups += [
                {'params': self.backbone_adapter.text_adapter.parameters(),
                 'lr': lr, 'weight_decay': weight_decay},
                {'params': [self.backbone_adapter.text_alpha],
                 'lr': alpha_lr, 'weight_decay': 0.0},
                {'params': [self.backbone_adapter.visual_type_emb,
                            self.backbone_adapter.text_type_emb],
                 'lr': lr, 'weight_decay': weight_decay},
            ]
        return param_groups
    
    def forward(self, x, return_features=False):
        visual_features = self.backbone_adapter(x)
        
        pooled = visual_features.mean(dim=1)
        
        if self.use_text_prompts and hasattr(self, 'text_prototypes_baseline'):
            pooled_norm = F.normalize(pooled, dim=-1)
            proto_norm  = F.normalize(self.text_prototypes_baseline, dim=-1)
            scale = self.clip_model.logit_scale.exp().clamp(max=100)
            logits = pooled_norm @ proto_norm.T * scale
        else:
            logits = self.classifier(pooled)
        
        if return_features:
            return pooled, logits
        return logits
    
    def train(self, mode=True):
        super().train(mode)
        self.clip_model.visual.eval()
        return self


def create_clip_adapter_baseline(clip_model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", **kwargs):
    """
    Factory function to create CLIP-Adapter baseline model.
    """
    return CLIPAdapterBaseline(
        clip_model_name=clip_model_name,
        pretrained=pretrained,
        **kwargs
    )