import torch
import torch.nn as nn
import timm
from transformers import CLIPVisionModel, CLIPVisionConfig


class ViTBackbone(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True, grayscale=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.grayscale = grayscale
        self.out_dim = self.model.embed_dim
        
    def forward(self, x):
        if self.grayscale:
            x = x.repeat(1, 3, 1, 1)
        features = self.model.forward_features(x)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        return features


class CLIPBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", pretrained=True, grayscale=False):
        super().__init__()
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = CLIPVisionConfig()
            self.model = CLIPVisionModel(config)
        
        self.grayscale = grayscale
        self.out_dim = self.model.config.hidden_size
        
    def forward(self, x):
        if self.grayscale:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.model.vision_model(pixel_values=x)
        features = outputs.last_hidden_state
        return features


def prepare_vit_backbone(backbone_type, pretrained="imagenet", grayscale=False):
    model_map = {
        "vit-tiny": "vit_tiny_patch16_224",
        "vit-small": "vit_small_patch16_224",
        "vit-base": "vit_base_patch16_224",
        "vit-large": "vit_large_patch16_224",
    }
    if backbone_type not in model_map:
        raise ValueError(f"Unsupported ViT backbone: {backbone_type}. Choose from: {list(model_map.keys())}")
    return ViTBackbone(model_name=model_map[backbone_type], pretrained=(pretrained == "imagenet"), grayscale=grayscale)


def prepare_clip_backbone(backbone_type, pretrained="imagenet", grayscale=False):
    if backbone_type == "clip-base":
        model_name = "openai/clip-vit-base-patch32"
    elif backbone_type == "clip-large":
        model_name = "openai/clip-vit-large-patch14"
    elif backbone_type == "clip-small":
        model_name = "openai/clip-vit-small-patch16"
    else:
        raise ValueError(f"Unsupported CLIP backbone: {backbone_type}")
    
    return CLIPBackbone(model_name=model_name, pretrained=(pretrained == "imagenet"), grayscale=grayscale)