import argparse
import json
import os
import random
import sys
import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
import torch

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
import torch.nn as nn
from tqdm.auto import tqdm

from tasks.image_classification.plotting import (
    plot_neural_dynamics,
    make_classification_gif,
)
from utils.housekeeping import set_seed, zip_python_code
from utils.dataset_utils import (
    get_dataset,
    compute_dataset_stats,
    compute_class_distribution,
    log_class_histogram_wandb,
)
from models.factories import create_model
from models.ctm_with_innovations import (
    PredictiveSynchronyLoss,
    CrossTickContrastiveLoss,
)
from models.utils import PonderNetLoss as PonderLoss
from utils.losses import image_classification_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

from autoclip.torch import QuantileClip

import gc
import torchvision

torchvision.disable_beta_transforms_warning()


import warnings

warnings.filterwarnings(
    "ignore", message="using precomputed metric; inverse_transform will be unavailable"
)
warnings.filterwarnings(
    "ignore", message="divide by zero encountered in power", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    "Corrupt EXIF data",
    UserWarning,
    r"^PIL\.TiffImagePlugin$",  # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Metadata Warning",
    UserWarning,
    r"^PIL\.TiffImagePlugin$",  # Using a regular expression to match the module.
)
warnings.filterwarnings(
    "ignore",
    "UserWarning: Truncated File Read",
    UserWarning,
    r"^PIL\.TiffImagePlugin$",  # Using a regular expression to match the module.
)


def flatten_config(cfg):
    """Placeholder for backwards compatibility - now just returns cfg as dict."""
    return cfg if cfg else {}


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument(
        "--model",
        type=str,
        default="ctm",
        choices=[
            "ctm",
            "ctm_gated",
            "ctm_with_innovations",
            "lstm",
            "ff",
            "clip_adapter",
        ],
        help="Model type to train.",
    )

    # Model Architecture
    # Common
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model."
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="resnet18-4",
        help="Type of backbone featureiser.",
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=str,
        default="none",
        help="Use a pretrained backbone one of none, imagenet, ms-celeba",
        choices=["none", "imagenet", "ms-celeba"],
    )
    parser.add_argument(
        "--grayscale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use grayscale images.",
    )
    parser.add_argument(
        "--convert_grayscale_to_rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert grayscale images to 3-channel RGB by duplicating.",
    )
    # CTM / LSTM specific
    parser.add_argument(
        "--d_input", type=int, default=128, help="Dimension of the input (CTM, LSTM)."
    )
    parser.add_argument(
        "--heads", type=int, default=4, help="Number of attention heads (CTM, LSTM)."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=75,
        help="Number of internal ticks (CTM, LSTM).",
    )
    parser.add_argument(
        "--positional_embedding_type",
        type=str,
        default="none",
        help="Type of positional embedding (CTM, LSTM).",
        choices=[
            "none",
            "learnable-fourier",
            "multi-learnable-fourier",
            "custom-rotational",
        ],
    )
    # CTM specific
    parser.add_argument(
        "--use_ponder_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PonderLoss for CTM.",
    )
    parser.add_argument(
        "--lambda_p",
        type=float,
        default=0.2,
        help="Lambda for PonderLoss geometric distribution.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Beta weight for KL regularization in ponder/loop loss.",
    )
    # CTM-Gated specific
    parser.add_argument(
        "--exit_strategy",
        type=str,
        default="certainty",
        choices=["certainty", "ponder", "normal", "learned", "none"],
        help="Exit strategy for CTM-Gated (certainty threshold, ponder, normal, learned, none).",
    )
    parser.add_argument(
        "--exit_threshold",
        type=float,
        default=0.9,
        help="Certainty threshold for early exit (for certainty strategy).",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=1,
        help="Minimum steps before early exit allowed for CTM-Gated.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="standard",
        choices=["standard", "ponder", "loop"],
        help="Loss type for CTM-Gated (standard, ponder, loop).",
    )
    parser.add_argument(
        "--synapse_depth",
        type=int,
        default=4,
        help="Depth of U-NET model for synapse. 1=linear, no unet (CTM only).",
    )
    parser.add_argument(
        "--n_synch_out",
        type=int,
        default=512,
        help="Number of neurons to use for output synch (CTM only).",
    )
    parser.add_argument(
        "--n_synch_action",
        type=int,
        default=512,
        help="Number of neurons to use for observation/action synch (CTM only).",
    )
    parser.add_argument(
        "--neuron_select_type",
        type=str,
        default="random-pairing",
        help="Protocol for selecting neuron subset (CTM only).",
    )
    parser.add_argument(
        "--n_random_pairing_self",
        type=int,
        default=0,
        help="Number of neurons paired self-to-self for synch (CTM only).",
    )
    parser.add_argument(
        "--memory_length",
        type=int,
        default=25,
        help="Length of the pre-activation history for NLMS (CTM only).",
    )
    parser.add_argument(
        "--deep_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deep memory (CTM only).",
    )
    parser.add_argument(
        "--memory_hidden_dims",
        type=int,
        default=4,
        help="Hidden dimensions of the memory if using deep memory (CTM only).",
    )
    parser.add_argument(
        "--dropout_nlm",
        type=float,
        default=None,
        help="Dropout rate for NLMs specifically. Unset to match dropout on the rest of the model (CTM only).",
    )
    parser.add_argument(
        "--do_normalisation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply normalization in NLMs (CTM only).",
    )
    # LSTM specific
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM stacked layers (LSTM only).",
    )
    # CTM-Innovations specific
    parser.add_argument(
        "--use_gsh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Gated Synchronization Highway (CTM-Innovations only).",
    )
    parser.add_argument(
        "--use_hne",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Hierarchical NLM Ensembles (CTM-Innovations only).",
    )
    parser.add_argument(
        "--use_sanp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Sparse Adaptive Neuron Pairing (CTM-Innovations only).",
    )
    parser.add_argument(
        "--hne_group_configs",
        type=str,
        default=None,
        help="JSON string for HNE group configs (CTM-Innovations only).",
    )
    parser.add_argument(
        "--hne_group_configs_file",
        type=str,
        default=None,
        help="Path to JSON file with HNE group configs (alternative to --hne_group_configs).",
    )
    parser.add_argument(
        "--sanp_init_top_k",
        type=int,
        default=1000,
        help="Number of candidate pairs for SANP (CTM-Innovations only).",
    )
    parser.add_argument(
        "--use_psl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Predictive Synchrony Loss (CTM-Innovations only).",
    )
    parser.add_argument(
        "--lambda_psl",
        type=float,
        default=0.1,
        help="Weight for PSL (CTM-Innovations only).",
    )
    parser.add_argument(
        "--use_ctcs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Cross-Tick Contrastive Synchronization (CTM-Innovations only).",
    )
    parser.add_argument(
        "--lambda_ctcs",
        type=float,
        default=0.1,
        help="Weight for CTCS (CTM-Innovations only).",
    )
    # CLIP-Adapter specific
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="ViT-B-32",
        help="CLIP model name (CLIP-Adapter only).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="CLIP pretrained weights (CLIP-Adapter only).",
    )
    parser.add_argument(
        "--adapter_reduction",
        type=int,
        default=4,
        help="Adapter bottleneck reduction (CLIP-Adapter only).",
    )
    parser.add_argument(
        "--alpha_init",
        type=float,
        default=0.5,
        help="Initial alpha for adapter blend (CLIP-Adapter only).",
    )
    parser.add_argument(
        "--use_text_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use text prompts for CLIP-Adapter (CLIP-Adapter only).",
    )

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--batch_size_test", type=int, default=32, help="Batch size for testing."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the model."
    )
    parser.add_argument(
        "--training_iterations",
        type=int,
        default=100001,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=5000, help="Number of warmup steps."
    )
    parser.add_argument(
        "--use_scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a learning rate scheduler.",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["multistep", "cosine"],
        help="Type of learning rate scheduler.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        default=[8000, 15000, 20000],
        nargs="+",
        help="Learning rate scheduler milestones.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Learning rate scheduler gamma for multistep.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay factor."
    )
    parser.add_argument(
        "--weight_decay_exclusion_list",
        type=str,
        nargs="+",
        default=[],
        help="List to exclude from weight decay. Typically good: bn, ln, bias, start",
    )
    parser.add_argument(
        "--gradient_clipping",
        type=float,
        default=-1,
        help="Gradient quantile clipping value (-1 to disable).",
    )
    parser.add_argument(
        "--do_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try to compile model components (backbone, synapses if CTM).",
    )
    parser.add_argument(
        "--num_workers_train", type=int, default=1, help="Num workers training."
    )
    parser.add_argument(
        "--img_size", type=int, default=100, help="Image size to resize to (square)."
    )

    # Housekeeping
    parser.add_argument(
        "--log_dir", type=str, default="logs/scratch", help="Directory for logging."
    )
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use wandb for logging.",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ctm", help=" wandb project name."
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="wandb entity team."
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="Dataset to use."
    )
    parser.add_argument(
        "--data_root", type=str, default="data/", help="Where to save/get dataset."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoints every this many iterations.",
    )
    parser.add_argument("--seed", type=int, default=412, help="Random seed.")
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload from disk?",
    )
    parser.add_argument(
        "--reload_model_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload only the model from disk?",
    )
    parser.add_argument(
        "--strict_reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Should use strict reload for model weights.",
    )  # Added back
    parser.add_argument(
        "--track_every",
        type=int,
        default=1000,
        help="Track metrics every this many iterations.",
    )
    parser.add_argument(
        "--n_test_batches",
        type=int,
        default=20,
        help="How many minibatches to approx metrics. Set to -1 for full eval",
    )
    parser.add_argument(
        "--eval_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate on full validation set after each epoch. Automatically sets n_test_batches=-1 and calculates track_every based on dataset size.",
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[-1],
        help="List of GPU(s) to use. Set to -1 to use CPU.",
    )
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="AMP autocast.",
    )
    # Train/Val/Test split options
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of training data to use for validation (0.0-1.0).",
    )
    parser.add_argument(
        "--use_test_as_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use original test set as validation instead of splitting train.",
    )
    parser.add_argument(
        "--final_test_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run final evaluation on test set after training.",
    )
    parser.add_argument(
        "--save_best_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save best model based on validation performance.",
    )
    parser.add_argument(
        "--compute_flops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute and log FLOPs for the model.",
    )
    parser.add_argument(
        "--log_dataset_distribution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log dataset class distribution to wandb.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args_dict = vars(args)
    args = type("Args", (), args_dict)()

    # Parse hne_group_configs if provided
    if hasattr(args, "hne_group_configs") and args.hne_group_configs is not None:
        args.hne_group_configs = json.loads(args.hne_group_configs)
    elif (
        hasattr(args, "hne_group_configs_file")
        and args.hne_group_configs_file is not None
    ):
        with open(args.hne_group_configs_file, "r") as f:
            args.hne_group_configs = json.load(f)
    else:
        args.hne_group_configs = None

    # Set up descriptive run name for experiments
    run_name = f"{args.model}_{args.dataset}_bs{args.batch_size}_lr{args.lr}"
    if args.model == "ctm_gated":
        run_name += f"_exit={args.exit_strategy}"
        if args.exit_strategy == "certainty":
            run_name += f"-thr{args.exit_threshold}"
        elif args.exit_strategy in ("ponder", "normal", "learned"):
            run_name += f"_loss={args.loss_type}"

    # Update log_dir to be unique based on run_name if it's the default or generic
    if "scratch" in args.log_dir or args.log_dir == "logs":
        args.log_dir = os.path.join("logs", run_name)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logging
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=args.log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            reinit=True,
        )
        wandb.log({"Logging initialized": True})

    assert args.dataset in [
        "cifar10",
        "cifar100",
        "imagenet",
        "RAFDB",
        "FerPlusPlus",
        "affectnet",
    ], (
        f"Need to be one of cifar10, cifar100, imagenet, RAFDB, FerPlusPlus, affectnet, got {args.dataset}"
    )

    # Data
    train_data, val_data, test_data, class_labels, dataset_mean, dataset_std = (
        get_dataset(
            args.dataset,
            args.data_root,
            args.val_split_ratio,
            args.use_test_as_val,
            args.grayscale,
            args.convert_grayscale_to_rgb,
            args.img_size,
        )
    )

    # Log class distribution to wandb
    if args.log_dataset_distribution:
        train_class_dist = compute_class_distribution(train_data)
        print(f"[INFO] Train class distribution: {dict(train_class_dist)}")
        log_class_histogram_wandb(train_class_dist, class_labels, split="train")
        if val_data is not None:
            val_class_dist = compute_class_distribution(val_data)
            print(f"[INFO] Val class distribution: {dict(val_class_dist)}")
            log_class_histogram_wandb(val_class_dist, class_labels, split="val")
        if test_data is not None:
            test_class_dist = compute_class_distribution(test_data)
            print(f"[INFO] Test class distribution: {dict(test_class_dist)}")
            log_class_histogram_wandb(test_class_dist, class_labels, split="test")

    first_sample, first_label = train_data[0]
    print(
        f"[DEBUG] First sample shape: {first_sample.shape}, mean: {first_sample.mean().item():.4f}, label: {first_label}"
    )

    num_workers_test = 1  # Defaulting to 1, change if needed
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers_train,
    )

    # Handle validation set
    if val_data is not None:
        valloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=num_workers_test,
            drop_last=False,
        )
    else:
        valloader = None

    # Handle test set (if available)
    if test_data is not None:
        testloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=num_workers_test,
            drop_last=False,
        )
    else:
        testloader = None

    # Auto-calculate epoch-based validation if requested
    if args.eval_per_epoch:
        iterations_per_epoch = len(train_data) // args.batch_size
        args.track_every = iterations_per_epoch
        args.n_test_batches = -1
        print(
            f"[INFO] Eval per epoch enabled: track_every={args.track_every} (iterations/epoch), n_test_batches={args.n_test_batches} (full validation)"
        )

    prediction_reshaper = [-1]  # Problem specific
    args.out_dims = len(class_labels)

    # For total reproducibility
    zip_python_code(f"{args.log_dir}/repo_state.zip")
    with open(f"{args.log_dir}/args.txt", "w") as f:
        print(args, file=f)

    # Configure device string (support MPS on macOS)
    if args.device[0] != -1:
        device = f"cuda:{args.device[0]}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Running model {args.model} on {device}")

    model, loss_fn = create_model(args, prediction_reshaper)
    model = model.to(device)

    if loss_fn is not None:
        ctm_gated_loss = loss_fn
    else:
        ctm_gated_loss = None

    if args.model == "ctm" and args.use_ponder_loss:
        ponder_loss = PonderLoss(lambda_p=args.lambda_p, beta=0.01).to(device)
    else:
        ponder_loss = None

    psl_loss = None
    ctcs_loss = None
    if args.model == "ctm_with_innovations":
        if args.use_psl:
            psl_loss = PredictiveSynchronyLoss(synch_size=args.n_synch_out).to(device)
        if args.use_ctcs:
            ctcs_loss = CrossTickContrastiveLoss().to(device)

    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)

    model.train()

    # Always log params, FLOPs is optional
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    if args.use_wandb:
        wandb.log({"Total Parameters": sum(p.numel() for p in model.parameters())})

    if args.compute_flops:
        input_channels = (
            1 if args.grayscale and not args.convert_grayscale_to_rgb else 3
        )
        metrics = compute_model_metrics(
            model, args.model, args.dataset, input_channels=input_channels
        )
        if metrics["total_flops"] > 0:
            print(f"Total FLOPs: {metrics['total_flops']:,}")
            if args.use_wandb:
                wandb.log(
                    {
                        "Total FLOPs": metrics["total_flops"],
                        "FLOPs Breakdown": metrics["flops_breakdown"],
                    }
                )

    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip parameters that don't require gradients
        if any(
            exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list
        ):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
    if len(no_decay_names):
        print(f"WARNING, excluding: {no_decay_names}")

    # Collect additional params from auxiliary losses
    aux_params = []
    if psl_loss is not None:
        aux_params.append(
            {"params": psl_loss.parameters(), "weight_decay": args.weight_decay}
        )
    if ctcs_loss is not None:
        aux_params.append(
            {"params": ctcs_loss.parameters(), "weight_decay": args.weight_decay}
        )

    # Optimizer and scheduler (Common setup)
    if len(no_decay_names) and args.weight_decay != 0:
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0},
            ]
            + aux_params,
            lr=args.lr,
            eps=1e-8 if not args.use_amp else 1e-6,
        )
    else:
        param_groups = [
            {"params": model.parameters(), "weight_decay": args.weight_decay}
        ] + aux_params
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            eps=1e-8 if not args.use_amp else 1e-6,
        )

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_schedule.step
    )
    if args.use_scheduler:
        if args.scheduler_type == "multistep":
            scheduler = WarmupMultiStepLR(
                optimizer,
                warmup_steps=args.warmup_steps,
                milestones=args.milestones,
                gamma=args.gamma,
            )
        elif args.scheduler_type == "cosine":
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                args.warmup_steps,
                args.training_iterations,
                warmup_start_lr=1e-20,
                eta_min=1e-7,
            )
        else:
            raise NotImplementedError

    # Metrics tracking
    start_iter = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    iters = []
    # Conditional metrics for CTM/LSTM/CTM-Gated
    train_accuracies_most_certain = (
        []
        if args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]
        else None
    )
    test_accuracies_most_certain = (
        []
        if args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]
        else None
    )
    # CTM-Gated specific: track exit steps and certainty
    train_exit_steps = [] if args.model == "ctm_gated" else None
    test_exit_steps = [] if args.model == "ctm_gated" else None
    train_certainty_at_exit = [] if args.model == "ctm_gated" else None
    test_certainty_at_exit = [] if args.model == "ctm_gated" else None
    train_certainty_per_step = [] if args.model == "ctm_gated" else None
    test_certainty_per_step = [] if args.model == "ctm_gated" else None

    # Best model tracking
    best_val_acc = 0.0
    best_checkpoint_path = f"{args.log_dir}/best_model.pt"

    scaler = torch.amp.GradScaler(
        "cuda" if "cuda" in device else "cpu", enabled=args.use_amp
    )

    # Reloading logic
    if args.reload:
        checkpoint_path = f"{args.log_dir}/checkpoint.pt"
        if os.path.isfile(checkpoint_path):
            print(f"Reloading from: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if not args.strict_reload:
                print("WARNING: not using strict reload for model weights!")
            load_result = model.load_state_dict(
                checkpoint["model_state_dict"], strict=args.strict_reload
            )
            print(
                f" Loaded state_dict. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}"
            )

            if not args.reload_model_only:
                print("Reloading optimizer etc.")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                start_iter = checkpoint["iteration"]
                # Load common metrics
                train_losses = checkpoint["train_losses"]
                test_losses = checkpoint["test_losses"]
                train_accuracies = checkpoint["train_accuracies"]
                test_accuracies = checkpoint["test_accuracies"]
                iters = checkpoint["iters"]
                if "best_val_acc" in checkpoint:
                    best_val_acc = checkpoint["best_val_acc"]

                # Load conditional metrics if they exist in checkpoint and are expected for current model
                if args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]:
                    train_accuracies_most_certain = checkpoint[
                        "train_accuracies_most_certain"
                    ]
                    test_accuracies_most_certain = checkpoint[
                        "test_accuracies_most_certain"
                    ]
                if args.model == "ctm_gated" and "train_exit_steps" in checkpoint:
                    train_exit_steps = checkpoint["train_exit_steps"]
                    test_exit_steps = checkpoint["test_exit_steps"]

            else:
                print("Only reloading model!")

            if "torch_rng_state" in checkpoint:
                # Reset seeds
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu().byte())
                np.random.set_state(checkpoint["numpy_rng_state"])
                random.setstate(checkpoint["random_rng_state"])

            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Conditional Compilation
    if args.do_compile:
        print("Compiling...")
        if hasattr(model, "backbone"):
            model.backbone = torch.compile(
                model.backbone, mode="reduce-overhead", fullgraph=True
            )

        # Compile synapses only for CTM/CTM-Gated/CTM-Innovations
        if args.model in ["ctm", "ctm_with_innovations", "ctm_gated"]:
            model.synapses = torch.compile(
                model.synapses, mode="reduce-overhead", fullgraph=True
            )

    # Training
    iterator = iter(trainloader)

    with tqdm(
        total=args.training_iterations,
        initial=start_iter,
        leave=False,
        position=0,
        dynamic_ncols=True,
    ) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]["lr"]

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)

            loss = None
            accuracy = None
            # Model-specific forward and loss calculation
            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                if args.do_compile:  # CUDAGraph marking for clean compile
                    torch.compiler.cudagraph_mark_step_begin()

                if args.model == "ctm":
                    predictions, certainties, synchronisation = model(inputs)
                    if args.use_ponder_loss:
                        loss, p_t = ponder_loss(predictions, certainties, targets)
                        where_to_eval = p_t.argmax(-1)
                        accuracy = (
                            (
                                predictions.argmax(1)[
                                    torch.arange(
                                        predictions.size(0), device=predictions.device
                                    ),
                                    where_to_eval,
                                ]
                                == targets
                            )
                            .float()
                            .mean()
                            .item()
                        )
                        avg_steps = (
                            (p_t * torch.arange(1, p_t.shape[1] + 1, device=p_t.device))
                            .sum(-1)
                            .mean()
                            .item()
                        )
                        pbar_desc = f"CTM PonderLoss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Avg_steps={avg_steps:0.2f}"
                    else:
                        loss, where_most_certain = image_classification_loss(
                            predictions, certainties, targets, use_most_certain=True
                        )
                        accuracy = (
                            (
                                predictions.argmax(1)[
                                    torch.arange(
                                        predictions.size(0), device=predictions.device
                                    ),
                                    where_most_certain,
                                ]
                                == targets
                            )
                            .float()
                            .mean()
                            .item()
                        )
                        pbar_desc = f"CTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})"

                elif args.model == "ctm_with_innovations":
                    predictions, certainties, synchronisation, synch_out_seq = model(
                        inputs
                    )
                    loss, where_most_certain = image_classification_loss(
                        predictions, certainties, targets, use_most_certain=True
                    )
                    aux_loss = 0
                    if psl_loss is not None and synch_out_seq is not None:
                        aux_loss = aux_loss + args.lambda_psl * psl_loss(synch_out_seq)
                    if ctcs_loss is not None and synch_out_seq is not None:
                        aux_loss = aux_loss + args.lambda_ctcs * ctcs_loss(
                            synch_out_seq
                        )
                    loss = loss + aux_loss
                    accuracy = (
                        (
                            predictions.argmax(1)[
                                torch.arange(
                                    predictions.size(0), device=predictions.device
                                ),
                                where_most_certain,
                            ]
                            == targets
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    pbar_desc = f"CTM-Innovations Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})"

                elif args.model == "ctm_gated":
                    predictions, certainties, synchronisation, exit_steps, _ = model(
                        inputs
                    )
                    loss, where_most_certain = ctm_gated_loss(
                        predictions, certainties, targets, exit_steps
                    )
                    accuracy = (
                        (
                            predictions.argmax(1)[
                                torch.arange(
                                    predictions.size(0), device=predictions.device
                                ),
                                where_most_certain,
                            ]
                            == targets
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    avg_exit_step = exit_steps.float().mean().item()

                    # Compute certainty statistics
                    avg_certainty_per_step = certainties[:, 1, :].mean(dim=0)  # [T]
                    certainty_at_exit = (
                        certainties[:, 1, :].gather(1, exit_steps.unsqueeze(1)).mean()
                    )
                    min_certainty_at_exit = (
                        certainties[:, 1, :].gather(1, exit_steps.unsqueeze(1)).min()
                    )
                    max_certainty_at_exit = (
                        certainties[:, 1, :].gather(1, exit_steps.unsqueeze(1)).max()
                    )

                    pbar_desc = f"CTM-Gated Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Exit_step={avg_exit_step:0.2f}. Cert_exit={certainty_at_exit.item():0.3f}"

                elif args.model == "lstm":
                    predictions, certainties, synchronisation = model(inputs)
                    loss, where_most_certain = image_classification_loss(
                        predictions, certainties, targets, use_most_certain=True
                    )
                    # LSTM where_most_certain will just be -1 because use_most_certain is False owing to stability issues with LSTM training
                    accuracy = (
                        (
                            predictions.argmax(1)[
                                torch.arange(
                                    predictions.size(0), device=predictions.device
                                ),
                                where_most_certain,
                            ]
                            == targets
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    pbar_desc = f"LSTM Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d})"

                elif args.model == "ff":
                    predictions = model(inputs)
                    loss = nn.CrossEntropyLoss()(predictions, targets)
                    accuracy = (predictions.argmax(1) == targets).float().mean().item()
                    pbar_desc = f"FF Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}"

                elif args.model == "clip_adapter":
                    logits = model(inputs)
                    loss = nn.CrossEntropyLoss()(logits, targets)
                    accuracy = (logits.argmax(1) == targets).float().mean().item()
                    pbar_desc = f"CLIP-Adapter Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}"

            scaler.scale(loss).backward()

            if args.gradient_clipping != -1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.gradient_clipping
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            pbar.set_description(
                f"Dataset={args.dataset}. Model={args.model}. {pbar_desc}"
            )

            # Metrics tracking and plotting (conditional logic needed)
            if (bi % args.track_every == 0 or bi == args.warmup_steps) and (
                bi != 0 or args.reload_model_only
            ):
                iters.append(bi)
                current_train_losses = []
                current_test_losses = []
                current_train_accuracies = []  # Holds list of accuracies per tick for CTM/LSTM, single value for FF
                current_test_accuracies = []  # Holds list of accuracies per tick for CTM/LSTM, single value for FF
                current_train_accuracies_most_certain = []  # Only for CTM/LSTM
                current_test_accuracies_most_certain = []  # Only for CTM/LSTM

                # Reset BN stats using train mode
                pbar.set_description("Resetting BN")
                model.train()
                for module in model.modules():
                    if isinstance(
                        module,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    ):
                        module.reset_running_stats()

                pbar.set_description("Tracking: Computing TRAIN metrics")
                with (
                    torch.no_grad()
                ):  # Should use inference_mode? CTM/LSTM scripts used no_grad
                    loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=args.batch_size_test,
                        shuffle=True,
                        num_workers=num_workers_test,
                    )
                    all_targets_list = []
                    all_predictions_list = []  # List to store raw predictions (B, C, T) or (B, C)
                    all_predictions_most_certain_list = []  # Only for CTM/LSTM
                    all_where_most_certain_list = []  # Store tick indices where most certain
                    all_losses = []

                    with tqdm(
                        total=len(loader),
                        initial=0,
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                    ) as pbar_inner:
                        for inferi, (inputs, targets) in enumerate(loader):
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            all_targets_list.append(targets.detach().cpu().numpy())

                            # Model-specific forward and loss for evaluation
                            if args.model == "ctm":
                                these_predictions, certainties, _ = model(inputs)
                                if args.use_ponder_loss:
                                    loss, p_t = ponder_loss(
                                        these_predictions,
                                        certainties.unsqueeze(1),
                                        targets,
                                    )
                                    where_to_eval = p_t.argmax(-1)
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )  # Shape (B, T)
                                    all_predictions_most_certain_list.append(
                                        these_predictions.argmax(1)[
                                            torch.arange(
                                                these_predictions.size(0),
                                                device=these_predictions.device,
                                            ),
                                            where_to_eval,
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )  # Shape (B,)
                                    all_where_most_certain_list.append(
                                        where_to_eval.detach().cpu().numpy()
                                    )
                                else:
                                    loss, where_most_certain = (
                                        image_classification_loss(
                                            these_predictions,
                                            certainties,
                                            targets,
                                            use_most_certain=True,
                                        )
                                    )
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )  # Shape (B, T)
                                    all_predictions_most_certain_list.append(
                                        these_predictions.argmax(1)[
                                            torch.arange(
                                                these_predictions.size(0),
                                                device=these_predictions.device,
                                            ),
                                            where_most_certain,
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )  # Shape (B,)
                                    all_where_most_certain_list.append(
                                        where_most_certain.detach().cpu().numpy()
                                    )

                            elif args.model == "ctm_gated":
                                (
                                    these_predictions,
                                    certainties,
                                    _,
                                    these_exit_steps,
                                    _,
                                ) = model(inputs, use_early_exit=False)
                                loss, where_most_certain = ctm_gated_loss(
                                    these_predictions,
                                    certainties,
                                    targets,
                                    these_exit_steps,
                                )
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
                                )  # Shape (B, T)
                                all_predictions_most_certain_list.append(
                                    these_predictions.argmax(1)[
                                        torch.arange(
                                            these_predictions.size(0),
                                            device=these_predictions.device,
                                        ),
                                        where_most_certain,
                                    ]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                all_where_most_certain_list.append(
                                    where_most_certain.detach().cpu().numpy()
                                )
                                if train_exit_steps is not None:
                                    train_exit_steps.append(
                                        these_exit_steps.float().mean().item()
                                    )
                                if train_certainty_at_exit is not None:
                                    # certainties shape: [B, 2, T] where [:, 1, :] is certainty
                                    certainty_at_exit = (
                                        certainties[:, 1, :]
                                        .gather(1, these_exit_steps.unsqueeze(1))
                                        .mean()
                                        .item()
                                    )
                                    train_certainty_at_exit.append(certainty_at_exit)
                                if train_certainty_per_step is not None:
                                    # Store average certainty per step
                                    train_certainty_per_step.append(
                                        certainties[:, 1, :].mean(dim=0).cpu().numpy()
                                    )

                            elif args.model == "lstm":
                                these_predictions, certainties, _ = model(inputs)
                                loss, where_most_certain = image_classification_loss(
                                    these_predictions,
                                    certainties,
                                    targets,
                                    use_most_certain=True,
                                )
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
                                )  # Shape (B, T)
                                all_predictions_most_certain_list.append(
                                    these_predictions.argmax(1)[
                                        torch.arange(
                                            these_predictions.size(0),
                                            device=these_predictions.device,
                                        ),
                                        where_most_certain,
                                    ]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )  # Shape (B,)
                                all_where_most_certain_list.append(
                                    where_most_certain.detach().cpu().numpy()
                                )

                            elif args.model == "ctm_with_innovations":
                                these_predictions, certainties, _, _ = model(inputs)
                                loss, where_most_certain = image_classification_loss(
                                    these_predictions,
                                    certainties,
                                    targets,
                                    use_most_certain=True,
                                )
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
                                )  # Shape (B, T)
                                all_predictions_most_certain_list.append(
                                    these_predictions.argmax(1)[
                                        torch.arange(
                                            these_predictions.size(0),
                                            device=these_predictions.device,
                                        ),
                                        where_most_certain,
                                    ]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )  # Shape (B,)
                                all_where_most_certain_list.append(
                                    where_most_certain.detach().cpu().numpy()
                                )

                            elif args.model == "ff":
                                these_predictions = model(inputs)
                                loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
                                )  # Shape (B,)

                            elif args.model == "clip_adapter":
                                logits = model(inputs)
                                loss = nn.CrossEntropyLoss()(logits, targets)
                                all_predictions_list.append(
                                    logits.argmax(1).detach().cpu().numpy()
                                )  # Shape (B,)

                            all_losses.append(loss.item())

                            if (
                                args.n_test_batches != -1
                                and inferi >= args.n_test_batches - 1
                            ):
                                break  # Check condition >= N-1
                            pbar_inner.set_description(
                                f"Computing metrics for train (Batch {inferi + 1})"
                            )
                            pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(
                        all_predictions_list
                    )  # Shape (N, T) or (N,)
                    train_losses.append(np.mean(all_losses))

                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        # Accuracies per tick for CTM/LSTM
                        current_train_accuracies = np.mean(
                            all_predictions == all_targets[..., np.newaxis], axis=0
                        )  # Mean over batch dim -> Shape (T,)
                        train_accuracies.append(current_train_accuracies)
                        # Most certain accuracy
                        all_predictions_most_certain = np.concatenate(
                            all_predictions_most_certain_list
                        )
                        current_train_accuracies_most_certain = (
                            all_targets == all_predictions_most_certain
                        ).mean()
                        train_accuracies_most_certain.append(
                            current_train_accuracies_most_certain
                        )
                    else:  # FF
                        current_train_accuracies = (
                            all_targets == all_predictions
                        ).mean()  # Shape scalar
                        train_accuracies.append(current_train_accuracies)

                    log_dict = {
                        "Train Loss": train_losses[-1],
                        "Learning Rate": current_lr,
                    }

                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        log_dict["Train Accuracy (Most Certain)"] = (
                            current_train_accuracies_most_certain
                        )
                        # Log where_most_certain statistics
                        if all_where_most_certain_list:
                            all_where_most_certain = np.concatenate(
                                all_where_most_certain_list
                            )
                            log_dict["Train Most Certain Tick (Mean)"] = (
                                all_where_most_certain.mean()
                            )
                            log_dict["Train Most Certain Tick (Median)"] = np.median(
                                all_where_most_certain
                            )
                            log_dict["Train Most Certain Tick (Min)"] = (
                                all_where_most_certain.min()
                            )
                            log_dict["Train Most Certain Tick (Max)"] = (
                                all_where_most_certain.max()
                            )
                            log_dict["Train Most Certain Tick (Std)"] = (
                                all_where_most_certain.std()
                            )
                            log_dict["Train Most Certain Tick Distribution"] = (
                                wandb.Histogram(all_where_most_certain)
                            )
                        for i, acc in enumerate(current_train_accuracies):
                            log_dict[f"Train Accuracy (Tick {i})"] = acc
                        if args.model == "ctm_gated" and train_exit_steps:
                            log_dict["Train Avg Exit Steps"] = np.mean(train_exit_steps)
                            if train_certainty_at_exit:
                                log_dict["Train Certainty at Exit"] = np.mean(
                                    train_certainty_at_exit
                                )
                            if train_certainty_per_step:
                                # Log certainty at specific steps
                                avg_certainty_per_step = np.mean(
                                    train_certainty_per_step, axis=0
                                )
                                log_dict["Train Certainty (Step 5)"] = (
                                    avg_certainty_per_step[4]
                                    if len(avg_certainty_per_step) > 4
                                    else 0
                                )
                                log_dict["Train Certainty (Step 10)"] = (
                                    avg_certainty_per_step[9]
                                    if len(avg_certainty_per_step) > 9
                                    else 0
                                )
                                log_dict["Train Certainty (Final)"] = (
                                    avg_certainty_per_step[-1]
                                )
                                # Log full trajectory as histogram
                                log_dict["Train Certainty Trajectory"] = (
                                    wandb.Histogram(avg_certainty_per_step)
                                )
                    else:  # FF
                        log_dict["Train Accuracy"] = current_train_accuracies

                    wandb.log(log_dict, step=bi)

                del these_predictions

                # Switch to eval mode for test metrics (fixed BN stats)
                model.eval()
                with torch.inference_mode():  # Use inference_mode for test eval
                    if test_data is None:
                        pbar.set_description(
                            "Tracking: No test data, skipping TEST metrics"
                        )
                    else:
                        pbar.set_description(
                            f"Tracking: Computing TEST metrics (iter {bi})"
                        )
                        loader = torch.utils.data.DataLoader(
                            test_data,
                            batch_size=args.batch_size_test,
                            shuffle=True,
                            num_workers=num_workers_test,
                        )
                        all_targets_list = []
                        all_predictions_list = []
                        all_predictions_most_certain_list = []  # Only for CTM/LSTM
                        all_where_most_certain_list = []  # Store tick indices where most certain
                        all_losses = []

                        with tqdm(
                            total=len(loader),
                            initial=0,
                            leave=False,
                            position=1,
                            dynamic_ncols=True,
                        ) as pbar_inner:
                            for inferi, (inputs, targets) in enumerate(loader):
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                all_targets_list.append(targets.detach().cpu().numpy())

                                # Model-specific forward and loss for evaluation
                                if args.model == "ctm":
                                    these_predictions, certainties, _ = model(inputs)
                                    if args.use_ponder_loss:
                                        loss, p_t = ponder_loss(
                                            these_predictions,
                                            certainties.unsqueeze(1),
                                            targets,
                                        )
                                        where_to_eval = p_t.argmax(-1)
                                        all_predictions_list.append(
                                            these_predictions.argmax(1)
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                        all_predictions_most_certain_list.append(
                                            these_predictions.argmax(1)[
                                                torch.arange(
                                                    these_predictions.size(0),
                                                    device=these_predictions.device,
                                                ),
                                                where_to_eval,
                                            ]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                        all_where_most_certain_list.append(
                                            where_to_eval.detach().cpu().numpy()
                                        )
                                    else:
                                        loss, where_most_certain = (
                                            image_classification_loss(
                                                these_predictions,
                                                certainties,
                                                targets,
                                                use_most_certain=True,
                                            )
                                        )
                                        all_predictions_list.append(
                                            these_predictions.argmax(1)
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                        all_predictions_most_certain_list.append(
                                            these_predictions.argmax(1)[
                                                torch.arange(
                                                    these_predictions.size(0),
                                                    device=these_predictions.device,
                                                ),
                                                where_most_certain,
                                            ]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                        all_where_most_certain_list.append(
                                            where_most_certain.detach().cpu().numpy()
                                        )

                                elif args.model == "ctm_gated":
                                    (
                                        these_predictions,
                                        certainties,
                                        _,
                                        these_exit_steps,
                                        _,
                                    ) = model(inputs, use_early_exit=False)

                                    loss, where_most_certain = ctm_gated_loss(
                                        these_predictions,
                                        certainties,
                                        targets,
                                        these_exit_steps,
                                    )
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_predictions_most_certain_list.append(
                                        these_predictions.argmax(1)[
                                            torch.arange(
                                                these_predictions.size(0),
                                                device=these_predictions.device,
                                            ),
                                            where_most_certain,
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_where_most_certain_list.append(
                                        where_most_certain.detach().cpu().numpy()
                                    )
                                    if test_exit_steps is not None:
                                        test_exit_steps.append(
                                            these_exit_steps.float().mean().item()
                                        )
                                    if test_certainty_at_exit is not None:
                                        certainty_at_exit = (
                                            certainties[:, 1, :]
                                            .gather(1, these_exit_steps.unsqueeze(1))
                                            .mean()
                                            .item()
                                        )
                                        test_certainty_at_exit.append(certainty_at_exit)
                                    if test_certainty_per_step is not None:
                                        test_certainty_per_step.append(
                                            certainties[:, 1, :]
                                            .mean(dim=0)
                                            .cpu()
                                            .numpy()
                                        )

                                elif args.model == "ctm_with_innovations":
                                    these_predictions, certainties, _, _ = model(inputs)
                                    loss, where_most_certain = (
                                        image_classification_loss(
                                            these_predictions,
                                            certainties,
                                            targets,
                                            use_most_certain=True,
                                        )
                                    )
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_predictions_most_certain_list.append(
                                        these_predictions.argmax(1)[
                                            torch.arange(
                                                these_predictions.size(0),
                                                device=these_predictions.device,
                                            ),
                                            where_most_certain,
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_where_most_certain_list.append(
                                        where_most_certain.detach().cpu().numpy()
                                    )

                                elif args.model == "lstm":
                                    these_predictions, certainties, _ = model(inputs)
                                    loss, where_most_certain = (
                                        image_classification_loss(
                                            these_predictions,
                                            certainties,
                                            targets,
                                            use_most_certain=True,
                                        )
                                    )
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_predictions_most_certain_list.append(
                                        these_predictions.argmax(1)[
                                            torch.arange(
                                                these_predictions.size(0),
                                                device=these_predictions.device,
                                            ),
                                            where_most_certain,
                                        ]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                    all_where_most_certain_list.append(
                                        where_most_certain.detach().cpu().numpy()
                                    )

                                elif args.model == "ff":
                                    these_predictions = model(inputs)
                                    loss = nn.CrossEntropyLoss()(
                                        these_predictions, targets
                                    )
                                    all_predictions_list.append(
                                        these_predictions.argmax(1)
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )

                                elif args.model == "clip_adapter":
                                    logits = model(inputs)
                                    loss = nn.CrossEntropyLoss()(logits, targets)
                                    all_predictions_list.append(
                                        logits.argmax(1).detach().cpu().numpy()
                                    )

                                all_losses.append(loss.item())

                                if (
                                    args.n_test_batches != -1
                                    and inferi >= args.n_test_batches - 1
                                ):
                                    break
                                pbar_inner.set_description(
                                    f"Computing metrics for test (Batch {inferi + 1})"
                                )
                                pbar_inner.update(1)

                    all_targets = np.concatenate(all_targets_list)
                    all_predictions = np.concatenate(all_predictions_list)
                    test_losses.append(np.mean(all_losses))

                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        current_test_accuracies = np.mean(
                            all_predictions == all_targets[..., np.newaxis], axis=0
                        )
                        test_accuracies.append(current_test_accuracies)
                        all_predictions_most_certain = np.concatenate(
                            all_predictions_most_certain_list
                        )
                        current_test_accuracies_most_certain = (
                            all_targets == all_predictions_most_certain
                        ).mean()
                        test_accuracies_most_certain.append(
                            current_test_accuracies_most_certain
                        )
                    else:  # FF
                        current_test_accuracies = (
                            all_targets == all_predictions
                        ).mean()
                        test_accuracies.append(current_test_accuracies)

                    log_dict = {
                        "Test Loss": test_losses[-1],
                    }

                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        log_dict["Test Accuracy (Most Certain)"] = (
                            current_test_accuracies_most_certain
                        )
                        # Log where_most_certain statistics
                        if all_where_most_certain_list:
                            all_where_most_certain = np.concatenate(
                                all_where_most_certain_list
                            )
                            log_dict["Test Most Certain Tick (Mean)"] = (
                                all_where_most_certain.mean()
                            )
                            log_dict["Test Most Certain Tick (Median)"] = np.median(
                                all_where_most_certain
                            )
                            log_dict["Test Most Certain Tick (Min)"] = (
                                all_where_most_certain.min()
                            )
                            log_dict["Test Most Certain Tick (Max)"] = (
                                all_where_most_certain.max()
                            )
                            log_dict["Test Most Certain Tick (Std)"] = (
                                all_where_most_certain.std()
                            )
                            log_dict["Test Most Certain Tick Distribution"] = (
                                wandb.Histogram(all_where_most_certain)
                            )
                        for i, acc in enumerate(current_test_accuracies):
                            log_dict[f"Test Accuracy (Tick {i})"] = acc
                        if args.model == "ctm_gated" and test_exit_steps:
                            log_dict["Test Avg Exit Steps"] = np.mean(test_exit_steps)
                            if test_certainty_at_exit:
                                log_dict["Test Certainty at Exit"] = np.mean(
                                    test_certainty_at_exit
                                )
                            if test_certainty_per_step:
                                # Log certainty at specific steps
                                avg_certainty_per_step = np.mean(
                                    test_certainty_per_step, axis=0
                                )
                                log_dict["Test Certainty (Step 5)"] = (
                                    avg_certainty_per_step[4]
                                    if len(avg_certainty_per_step) > 4
                                    else 0
                                )
                                log_dict["Test Certainty (Step 10)"] = (
                                    avg_certainty_per_step[9]
                                    if len(avg_certainty_per_step) > 9
                                    else 0
                                )
                                log_dict["Test Certainty (Final)"] = (
                                    avg_certainty_per_step[-1]
                                )
                                # Log full trajectory as histogram
                                log_dict["Test Certainty Trajectory"] = wandb.Histogram(
                                    avg_certainty_per_step
                                )
                    else:  # FF
                        log_dict["Test Accuracy"] = current_test_accuracies

                    wandb.log(log_dict, step=bi)

                # Save best model checkpoint based on validation (test) accuracy
                if args.save_best_model and valloader is not None:
                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        current_val_acc = current_test_accuracies_most_certain
                    else:
                        current_val_acc = current_test_accuracies

                    if current_val_acc > best_val_acc:
                        best_val_acc = current_val_acc
                        best_checkpoint = {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "iteration": bi,
                            "train_losses": train_losses,
                            "test_losses": test_losses,
                            "train_accuracies": train_accuracies,
                            "test_accuracies": test_accuracies,
                            "iters": iters,
                            "best_val_acc": best_val_acc,
                        }
                        if args.model in [
                            "ctm",
                            "ctm_with_innovations",
                            "lstm",
                            "ctm_gated",
                        ]:
                            best_checkpoint["train_accuracies_most_certain"] = (
                                train_accuracies_most_certain
                            )
                            best_checkpoint["test_accuracies_most_certain"] = (
                                test_accuracies_most_certain
                            )
                        torch.save(best_checkpoint, best_checkpoint_path)
                        print(
                            f"Saved best checkpoint with val_acc={best_val_acc:.4f} at iteration {bi}"
                        )

                # Plotting (conditional)
                if test_data is None:
                    print("Skipping plotting as test_data is None")
                else:
                    figacc = plt.figure(figsize=(10, 10))
                    axacc_train = figacc.add_subplot(211)
                    axacc_test = figacc.add_subplot(212)
                    cm = sns.color_palette("viridis", as_cmap=True)

                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        # Plot per-tick accuracy for CTM/LSTM
                        train_acc_arr = np.array(train_accuracies)  # Shape (N_iters, T)
                        test_acc_arr = np.array(test_accuracies)  # Shape (N_iters, T)
                        num_ticks = train_acc_arr.shape[1]
                        for ti in range(num_ticks):
                            axacc_train.plot(
                                iters,
                                train_acc_arr[:, ti],
                                color=cm(ti / num_ticks),
                                alpha=0.3,
                            )
                            axacc_test.plot(
                                iters,
                                test_acc_arr[:, ti],
                                color=cm(ti / num_ticks),
                                alpha=0.3,
                            )
                        # Plot most certain accuracy
                        axacc_train.plot(
                            iters,
                            train_accuracies_most_certain,
                            "k--",
                            alpha=0.7,
                            label="Most certain",
                        )
                        axacc_test.plot(
                            iters,
                            test_accuracies_most_certain,
                            "k--",
                            alpha=0.7,
                            label="Most certain",
                        )
                    else:  # FF
                        axacc_train.plot(
                            iters, train_accuracies, "k-", alpha=0.7, label="Accuracy"
                        )
                        axacc_test.plot(
                            iters, test_accuracies, "k-", alpha=0.7, label="Accuracy"
                        )

                    axacc_train.set_title("Train Accuracy")
                    axacc_test.set_title("Test Accuracy")
                    axacc_train.legend(loc="lower right")
                    axacc_test.legend(loc="lower right")
                    axacc_train.set_xlim([0, args.training_iterations])
                    axacc_test.set_xlim([0, args.training_iterations])
                    if args.dataset == "cifar10":
                        axacc_train.set_ylim([0.75, 1])
                        axacc_test.set_ylim([0.75, 1])

                    figacc.tight_layout()
                    figacc.savefig(f"{args.log_dir}/accuracies.png", dpi=150)
                    plt.close(figacc)

                    figloss = plt.figure(figsize=(10, 5))
                    axloss = figloss.add_subplot(111)
                    axloss.plot(
                        iters,
                        train_losses,
                        "b-",
                        linewidth=1,
                        alpha=0.8,
                        label=f"Train: {train_losses[-1]:.4f}",
                    )
                    axloss.plot(
                        iters,
                        test_losses,
                        "r-",
                        linewidth=1,
                        alpha=0.8,
                        label=f"Test: {test_losses[-1]:.4f}",
                    )
                    axloss.legend(loc="upper right")
                    axloss.set_xlim([0, args.training_iterations])
                    axloss.set_ylim(bottom=0)

                    figloss.tight_layout()
                    figloss.savefig(f"{args.log_dir}/losses.png", dpi=150)
                    plt.close(figloss)

                # Conditional Visualization (Only for CTM/LSTM)
                if (
                    args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]
                    and testloader is not None
                ):
                    try:  # For safety
                        inputs_viz, targets_viz = next(
                            iter(testloader)
                        )  # Get a fresh batch
                        inputs_viz = inputs_viz.to(device)
                        targets_viz = targets_viz.to(device)

                        pbar.set_description("Tracking: Processing test data for viz")
                        if args.model == "ctm_gated":
                            (
                                predictions_viz,
                                certainties_viz,
                                _,
                                pre_activations_viz,
                                post_activations_viz,
                                attention_tracking_viz,
                                _,
                            ) = model(inputs_viz, track=True)
                        else:
                            (
                                predictions_viz,
                                certainties_viz,
                                _,
                                pre_activations_viz,
                                post_activations_viz,
                                attention_tracking_viz,
                            ) = model(inputs_viz, track=True)

                        att_shape = (
                            model.kv_features.shape[2],
                            model.kv_features.shape[3],
                        )
                        attention_tracking_viz = attention_tracking_viz.reshape(
                            attention_tracking_viz.shape[0],
                            attention_tracking_viz.shape[1],
                            -1,
                            att_shape[0],
                            att_shape[1],
                        )

                        pbar.set_description("Tracking: Neural dynamics plot")
                        plot_neural_dynamics(
                            post_activations_viz, 100, args.log_dir, axis_snap=True
                        )

                        imgi = 0  # Visualize the first image in the batch
                        img_to_gif = np.moveaxis(
                            np.clip(
                                inputs_viz[imgi].detach().cpu().numpy()
                                * np.array(dataset_std).reshape(len(dataset_std), 1, 1)
                                + np.array(dataset_mean).reshape(
                                    len(dataset_mean), 1, 1
                                ),
                                0,
                                1,
                            ),
                            0,
                            -1,
                        )

                        pbar.set_description("Tracking: Producing attention gif")
                        make_classification_gif(
                            img_to_gif,
                            targets_viz[imgi].item(),
                            predictions_viz[imgi].detach().cpu().numpy(),
                            certainties_viz[imgi].detach().cpu().numpy(),
                            post_activations_viz[:, imgi],
                            attention_tracking_viz[:, imgi],
                            class_labels,
                            f"{args.log_dir}/{imgi}_attention.gif",
                        )
                        del (
                            predictions_viz,
                            certainties_viz,
                            pre_activations_viz,
                            post_activations_viz,
                            attention_tracking_viz,
                        )
                    except Exception as e:
                        print(f"Visualization failed for model {args.model}: {e}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model.train()  # Switch back to train mode

            # Save model checkpoint (conditional metrics)
            if (
                bi % args.save_every == 0 or bi == args.training_iterations - 1
            ) and bi != start_iter:
                pbar.set_description("Saving model checkpoint...")
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "iteration": bi,
                    # Always save these
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_accuracies": train_accuracies,  # This is list of scalars for FF, list of arrays for CTM/LSTM
                    "test_accuracies": test_accuracies,  # This is list of scalars for FF, list of arrays for CTM/LSTM
                    "iters": iters,
                    "best_val_acc": best_val_acc,
                    "args": args,  # Save args used for this run
                    # RNG states
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "random_rng_state": random.getstate(),
                }
                # Conditionally add metrics specific to CTM/LSTM/CTM-Gated
                if args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]:
                    checkpoint_data["train_accuracies_most_certain"] = (
                        train_accuracies_most_certain
                    )
                    checkpoint_data["test_accuracies_most_certain"] = (
                        test_accuracies_most_certain
                    )
                # CTM-Gated specific
                if args.model == "ctm_gated":
                    checkpoint_data["train_exit_steps"] = train_exit_steps
                    checkpoint_data["test_exit_steps"] = test_exit_steps

                torch.save(checkpoint_data, f"{args.log_dir}/checkpoint.pt")

            pbar.update(1)

    print("Training complete!")

    if args.final_test_eval and testloader is not None and args.save_best_model:
        if os.path.isfile(best_checkpoint_path):
            print(f"Loading best checkpoint from: {best_checkpoint_path}")
            checkpoint = torch.load(
                best_checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model with val_acc={checkpoint.get('best_val_acc', 'N/A'):.4f}"
            )

        print("Running final evaluation on test set...")
        model.eval()
        final_test_acc = 0
        final_test_loss = 0
        final_test_count = 0
        with torch.inference_mode():
            for i, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(
                    device_type="cuda" if "cuda" in device else "cpu",
                    dtype=torch.float16,
                    enabled=args.use_amp,
                ):
                    if args.model in [
                        "ctm",
                        "ctm_with_innovations",
                        "lstm",
                        "ctm_gated",
                    ]:
                        if args.model == "ctm_gated":
                            predictions, certainties, _, _, _ = model(inputs)
                        else:
                            predictions, certainties, _ = model(inputs)
                        loss, where_most_certain = image_classification_loss(
                            predictions, certainties, targets, use_most_certain=True
                        )
                        outputs = predictions[
                            torch.arange(
                                predictions.size(0), device=predictions.device
                            ),
                            :,
                            where_most_certain,
                        ]
                    else:
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, targets)

                if args.model in ["ctm", "ctm_with_innovations", "lstm", "ctm_gated"]:
                    final_test_acc += (
                        (outputs.argmax(1) == targets).float().sum().item()
                    )
                else:
                    final_test_acc += (
                        (outputs.argmax(1) == targets).float().sum().item()
                    )

                final_test_loss += loss.item() * inputs.size(0)
                final_test_count += inputs.size(0)
                if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                    break

        final_test_acc /= final_test_count
        final_test_loss /= final_test_count
        print(
            f"Final Test Accuracy: {final_test_acc:.4f}, Test Loss: {final_test_loss:.4f}"
        )
        if args.use_wandb:
            wandb.log(
                {
                    "Final Test Accuracy": final_test_acc,
                    "Final Test Loss": final_test_loss,
                },
                step=args.training_iterations,
            )
            wandb.finish()


if __name__ == "__main__":
    main()
