import torch
import torch.nn as nn
import os
from models.modules import Identity
import torchvision.models as models

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        
        # activation = None
        # activation = out.detach().cpu().numpy()
        out = self.relu(out)
        # return out, activation

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_scales,
        stride,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        do_initial_max_pool=True,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # NOTE: Important!
        # This has changed from a kernel size of 7 (padding=3) to a kernel of 3 (padding=1)
        # The reason for this was to limit the receptive field to constrain models to 
        # "Looking around" to gather information.

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        ) if in_channels in [1, 3] else nn.LazyConv2d(
            self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if do_initial_max_pool else Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_scales = feature_scales
        if 2 in feature_scales:
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=stride, dilate=replace_stride_with_dilation[0]
            )
            if 3 in feature_scales:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=stride, dilate=replace_stride_with_dilation[1]
                )
                if 4 in feature_scales:
                    self.layer4 = self._make_layer(
                        block, 512, layers[3], stride=stride, dilate=replace_stride_with_dilation[2]
                    )

        # NOTE: Commented this out as it is not used anymore for this work, kept it for reference
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # if return_activations: activations.append(torch.clone(x))
        x = self.layer1(x)

        if 2 in self.feature_scales:
            x = self.layer2(x)
            if 3 in self.feature_scales:
                x = self.layer3(x)
                if 4 in self.feature_scales:
                    x = self.layer4(x)
        return x


def _resnet(in_channels, feature_scales, stride, arch, block, layers, pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs):
    if pretrained:
        if pretrained_dataset.lower() == 'imagenet':
            # Load the standard 3-channel pretrained model
            pretrained_model = models.__dict__[arch](pretrained=True)
            pretrained_state_dict = pretrained_model.state_dict()

            # Create our new model with the desired number of input channels
            model = ResNet(in_channels, feature_scales, stride, block, layers, do_initial_max_pool=do_initial_max_pool, **kwargs)
            
            # If the input channels are not 3, we need to adapt the first convolutional layer's weights
            if in_channels != 3:
                # Get the weights of the first conv layer from the pretrained model
                conv1_weights = pretrained_state_dict['conv1.weight']
                # Average the weights across the 3 input channels
                avg_weights = conv1_weights.mean(dim=1, keepdim=True)
                # Repeat the averaged weights for the new number of input channels
                new_conv1_weights = avg_weights.repeat(1, in_channels, 1, 1)
                # Update the state dict with the new conv1 weights
                pretrained_state_dict['conv1.weight'] = new_conv1_weights
            
            # Load the (potentially modified) state dict. strict=False is important.
            model.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained_dataset.lower() == 'celeba':
            model = ResNet(in_channels, feature_scales, stride, block, layers, do_initial_max_pool=do_initial_max_pool, **kwargs)
            script_dir = os.path.dirname(__file__)
            state_dict_path = os.path.join(script_dir, 'state_dicts', f'{arch}_celeba.pt')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"CelebA pretrained weights not found at {state_dict_path}")
        else:
            raise ValueError("pretrained_dataset must be 'imagenet' or 'celeba'")
    else:
        # If not pretrained, just create the model directly
        model = ResNet(in_channels, feature_scales, stride, block, layers, do_initial_max_pool=do_initial_max_pool, **kwargs)
    return model


def resnet18(in_channels, feature_scales, stride=2, pretrained=False, pretrained_dataset='imagenet', progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs
    )


def resnet34(in_channels, feature_scales, stride=2, pretrained=False, pretrained_dataset='imagenet', progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs
    )


def resnet50(in_channels, feature_scales, stride=2, pretrained=False, pretrained_dataset='imagenet', progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs
    )


def resnet101(in_channels, feature_scales, stride=2, pretrained=False, pretrained_dataset='imagenet', progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs
    )


def resnet152(in_channels, feature_scales, stride=2, pretrained=False, pretrained_dataset='imagenet', progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet152", Bottleneck, [3, 4, 36, 3], pretrained, pretrained_dataset, progress, device, do_initial_max_pool, **kwargs
    )

def prepare_resnet_backbone(backbone_type, pretrained='none'):
      
    resnet_family = resnet18 # Default
    if '34' in backbone_type: resnet_family = resnet34
    if '50' in backbone_type: resnet_family = resnet50
    if '101' in backbone_type: resnet_family = resnet101
    if '152' in backbone_type: resnet_family = resnet152
    
    # Check for pretrained options with 'imagenet' or 'celeba' or 'none' in pretrained
    if pretrained not in ['none', 'imagenet', 'celeba', False]:
        raise ValueError("pretrained must be one of 'none', 'imagenet', or 'celeba'")

    # Determine which ResNet blocks to keep
    # Filter out 'pretrained' before splitting to get the block number
    block_num_str = backbone_type.replace('-pretrained', '').split('-')[-1]
    hyper_blocks_to_keep = list(range(1, int(block_num_str) + 1)) if block_num_str.isdigit() else [1, 2, 3, 4]
    # Create the ResNet backbone
    if pretrained == 'imagenet':
        backbone = resnet_family(
            7,
            hyper_blocks_to_keep,
            stride=2,
            pretrained=True,
            pretrained_dataset='imagenet',
            progress=True,
            device="cpu",
            do_initial_max_pool=True,
        )
    elif pretrained == 'celeba':
        backbone = resnet_family(
            3,
            hyper_blocks_to_keep,
            stride=2,
            pretrained=True,
            pretrained_dataset='ms-celeba',
            progress=True,
            device="cpu",
            do_initial_max_pool=True,
        )
    else:
        backbone = resnet_family(
            3,
            hyper_blocks_to_keep,
            stride=2,
            pretrained=False, # Explicitly set to False for 'none' case
            progress=True,
            device="cpu",
            do_initial_max_pool=True,
        )

    return backbone

if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It allows for testing the functions in this file.

    # --- Fix for local testing ---
    # Add the project root to the Python path to allow absolute imports
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    print("Current Python Path:", sys.path)
    print("--- Running ResNet Test ---")

    # --- Test Case 1: Create a non-pretrained resnet18 ---
    print("\n1. Testing non-pretrained resnet18 creation...")
    try:
        model_no_pretrain = resnet18(in_channels=3, feature_scales=[1, 2, 3, 4], do_initial_max_pool=True)
        print("   ✅ Success: Created non-pretrained resnet18.")
        # print(model_no_pretrain)
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # --- Test Case 2: Load ImageNet pretrained weights ---
    # This tests the `pretrained=True` and `pretrained_dataset='imagenet'` path
    print("\n2. Testing ImageNet pretrained resnet18 loading...")
    try: # Now testing with 7 input channels
        model_imagenet = _resnet(
            in_channels=7, feature_scales=[1, 2, 3, 4], stride=2,
            arch="resnet18", block=BasicBlock, layers=[2, 2, 2, 2],
            pretrained=True, pretrained_dataset='imagenet', progress=True,
            device="cpu", do_initial_max_pool=True
        )
        print("   ✅ Success: Loaded ImageNet pretrained weights into a 7-channel resnet18.")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # --- Test Case 3: Attempt to load CelebA weights (will likely fail if file doesn't exist) ---
    # This tests the `pretrained_dataset='celeba'` path
    print("\n3. Testing CelebA pretrained resnet18 loading...")
    try:
        model_celeba = _resnet(
            in_channels=3, feature_scales=[1, 2, 3, 4], stride=2,
            arch="resnet18", block=BasicBlock, layers=[2, 2, 2, 2],
            pretrained=True, pretrained_dataset='celeba', progress=True,
            device="cpu", do_initial_max_pool=True
        )
        print("   ✅ Success: Loaded CelebA pretrained weights (file was found).")
    except FileNotFoundError as e:
        print(f"   ✅ Success (as expected): Correctly raised FileNotFoundError because weights file is missing.")
        print(f"      Details: {e}")
    except Exception as e:
        print(f"   ❌ Failed with an unexpected error: {e}")

    print("\n--- Running prepare_resnet_backbone Test ---")

    # --- Test Case 4: Create a non-pretrained resnet18-4 ---
    print("\n4. Testing non-pretrained resnet18-4 creation...")
    try:
        model_prepared_no_pretrain = prepare_resnet_backbone('resnet18-4', pretrained='none')
        print("   ✅ Success: Created non-pretrained resnet18-4.")
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model_prepared_no_pretrain(dummy_input)
        print(f"   ✅ Success: Forward pass completed with output shape: {output.shape}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # --- Test Case 5: Create an ImageNet-pretrained resnet50-3 ---
    print("\n5. Testing ImageNet-pretrained resnet50-3 creation...")
    try:
        model_prepared_imagenet = prepare_resnet_backbone('resnet50-3', pretrained='imagenet')
        print("   ✅ Success: Created ImageNet-pretrained resnet50-3.")
        # Test forward pass (note: pretrained on imagenet expects 7 channels in this implementation)
        dummy_input = torch.randn(2, 7, 224, 224)
        output = model_prepared_imagenet(dummy_input)
        print(f"   ✅ Success: Forward pass completed with output shape: {output.shape}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # --- Test Case 6: Test invalid pretrained argument ---
    print("\n6. Testing invalid 'pretrained' argument...")
    try:
        prepare_resnet_backbone('resnet18-1', pretrained='invalid_option')
        print(f"   ❌ Failed: Did not raise ValueError for invalid 'pretrained' argument.")
    except ValueError as e:
        print(f"   ✅ Success: Correctly raised ValueError.")
        print(f"      Details: {e}")
    except Exception as e:
        print(f"   ❌ Failed with an unexpected error: {e}")


    print("\n--- Test Complete ---")