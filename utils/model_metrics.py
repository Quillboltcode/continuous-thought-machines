import torch


def compute_model_metrics(model, model_type, dataset, input_channels=None, batch_size=1):
    """
    Compute FLOPs and parameter count for a model.

    Args:
        model: PyTorch model
        model_type: Model type string ('ctm', 'ctm_gated', 'lstm', 'ff')
        dataset: Dataset name for determining input shape
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        batch_size: Batch size for FLOP calculation

    Returns:
        dict: Dictionary containing 'total_params', 'trainable_params', 'total_flops', and 'flops_breakdown'
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = {}
    total_flops = 0

    if model_type in ["ctm", "ctm_gated"]:
        if input_channels is None:
            input_channels = 3

        input_shape = (
            (input_channels, 224, 224)
            if dataset == "imagenet"
            else (input_channels, 32, 32)
        )

        from models.utils import compute_ctm_flops
        flops = compute_ctm_flops(model, input_shape, batch_size=batch_size)
        total_flops = flops["total"]

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_flops": total_flops,
        "flops_breakdown": flops,
    }
