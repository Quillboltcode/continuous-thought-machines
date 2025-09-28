import torch.nn as nn

# Local imports (Assuming these contain necessary custom modules)
from models.modules import *
from models.resnet import prepare_resnet_backbone


class FFBaseline(nn.Module):
    """
    LSTM Baseline.

    Wrapper that lets us use the same backbone as the CTM and LSTM baselines, with a 


    Args:
        d_model (int): workaround that projects final layer to this space so that parameter-matching is plausible.
        backbone_type (str): Type of feature extraction backbone (e.g., 'resnet18-2', 'none').
        out_dims (int): Dimensionality of the final output projection.
        dropout (float): dropout in last layer
    """

    def __init__(self,
                 d_model,
                 backbone_type,
                 out_dims,
                 dropout=0,
                 ):
        super(FFBaseline, self).__init__()

        # --- Core Parameters ---
        self.d_model = d_model
        self.backbone_type = backbone_type
        self.out_dims = out_dims

        # --- Input Assertions ---
        assert backbone_type in ['resnet18-1', 'resnet18-2', 'resnet18-3', 'resnet18-4',
                                 'resnet34-1', 'resnet34-2', 'resnet34-3', 'resnet34-4',
                                 'resnet50-1', 'resnet50-2', 'resnet50-3', 'resnet50-4',
                                 'resnet101-1', 'resnet101-2', 'resnet101-3', 'resnet101-4',
                                 'resnet152-1', 'resnet152-2', 'resnet152-3', 'resnet152-4',
                                 'none', 'shallow-wide', 'parity_backbone'], f"Invalid backbone_type: {backbone_type}"

        # --- Backbone / Feature Extraction ---
        self.initial_rgb = Identity() # Placeholder, potentially replaced if using ResNet

        
        self.backbone = prepare_resnet_backbone(self.backbone_type, pretrained=False)


        # At this point we will have a 4D tensor of features: [B, C, H, W]
        # The following lets us scale up the resnet with d_model until it matches the CTM
        self.output_projector = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Squeeze(-1), Squeeze(-1), nn.LazyLinear(d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, out_dims))


    def forward(self, x):
        return self.output_projector(self.backbone(x))
