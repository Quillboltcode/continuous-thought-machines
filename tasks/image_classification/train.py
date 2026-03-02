import argparse
import os
import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
import torch

if torch.cuda.is_available():
    # For faster
    torch.set_float32_matmul_precision("high")
import torch.nn as nn
from tqdm.auto import tqdm

from data.custom_datasets import ImageNet
from torchvision import datasets
from torchvision import transforms
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES
from models.ctm import ContinuousThoughtMachine
from models.ctm_gated import CTMGated, CTMGatedLoss
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from tasks.image_classification.plotting import (
    plot_neural_dynamics,
    make_classification_gif,
)
from utils.housekeeping import set_seed, zip_python_code
from models.utils import PonderNetLoss as PonderLoss
from models.utils import compute_ctm_flops, count_parameters
from utils.losses import image_classification_loss  # Used by CTM, LSTM
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


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument(
        "--model",
        type=str,
        default="ctm",
        choices=["ctm", "ctm_gated", "lstm", "ff"],
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

    # Housekeeping
    parser.add_argument(
        "--log_dir", type=str, default="logs/scratch", help="Directory for logging."
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

    args = parser.parse_args()
    return args


def get_dataset(
    dataset, root, val_split_ratio=0.0, use_test_as_val=False, grayscale=False
):
    if dataset == "imagenet":
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        class_labels = list(IMAGENET2012_CLASSES.values())

        train_data = ImageNet(which_split="train", transform=train_transform)
        test_data = ImageNet(which_split="validation", transform=test_transform)
    elif dataset == "cifar10":
        dataset_mean = [0.49139968, 0.48215827, 0.44653124]
        dataset_std = [0.24703233, 0.24348505, 0.26158768]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        train_transform = transforms.Compose(
            [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        full_train_data = datasets.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = datasets.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        class_labels = [
            "air",
            "auto",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        if use_test_as_val:
            train_data = full_train_data
            val_data = test_data
            test_data = None
        elif val_split_ratio > 0.0:
            train_size = int((1.0 - val_split_ratio) * len(full_train_data))
            val_size = len(full_train_data) - train_size
            train_data, val_data = torch.utils.data.random_split(
                full_train_data,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(412),
            )
            test_data = test_data
        else:
            train_data = full_train_data
            val_data = None
            test_data = test_data
    elif dataset == "cifar100":
        dataset_mean = [0.5070751592371341, 0.48654887331495067, 0.4409178433670344]
        dataset_std = [0.2673342858792403, 0.2564384629170882, 0.27615047132568393]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        train_transform = transforms.Compose(
            [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_data = datasets.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = datasets.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        idx_order = np.argsort(np.array(list(train_data.class_to_idx.values())))
        class_labels = list(np.array(list(train_data.class_to_idx.keys()))[idx_order])
    elif dataset == "FerPlusPlus":
        if grayscale:
            dataset_mean = [0.5]
            dataset_std = [0.5]
        else:
            dataset_mean = [0.485, 0.456, 0.406]
            dataset_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        if grayscale:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize((100, 100)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        train_data = datasets.ImageFolder(
            os.path.join(root, "train"), transform=train_transform
        )
        val_data = datasets.ImageFolder(
            os.path.join(root, "val"), transform=test_transform
        )
        test_data = datasets.ImageFolder(
            os.path.join(root, "test"), transform=test_transform
        )
        # ImageFolder assigns class indices alphabetically based on folder names
        class_labels = sorted(train_data.classes)

        if use_test_as_val:
            val_data = test_data
            test_data = None
        elif val_split_ratio > 0.0:
            train_size = int((1.0 - val_split_ratio) * len(train_data))
            val_size = len(train_data) - train_size
            train_data, val_data = torch.utils.data.random_split(
                train_data,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(412),
            )
        else:
            test_data = test_data
    elif dataset == "RAFDB":
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        train_transform = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_data = datasets.ImageFolder(
            os.path.join(root, "train"), transform=train_transform
        )
        
        # RAFDB may only have train and test folders (no val folder)
        val_path = os.path.join(root, "val")
        test_path = os.path.join(root, "test")
        
        if os.path.exists(val_path):
            val_data = datasets.ImageFolder(val_path, transform=test_transform)
            test_data = datasets.ImageFolder(test_path, transform=test_transform) if os.path.exists(test_path) else None
        else:
            # No val folder - use test as validation or split train
            if use_test_as_val:
                val_data = datasets.ImageFolder(test_path, transform=test_transform) if os.path.exists(test_path) else None
                test_data = None
            elif val_split_ratio > 0.0:
                train_size = int((1.0 - val_split_ratio) * len(train_data))
                val_size = len(train_data) - train_size
                train_data, val_data = torch.utils.data.random_split(
                    train_data,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(412),
                )
                test_data = datasets.ImageFolder(test_path, transform=test_transform) if os.path.exists(test_path) else None
            else:
                val_data = None
                test_data = datasets.ImageFolder(test_path, transform=test_transform) if os.path.exists(test_path) else None
        
        class_labels = [
            "Surprise",
            "Fear",
            "Disgust",
            "Happiness",
            "Sadness",
            "Anger",
            "Neutral",
        ]
    else:
        raise NotImplementedError

    return train_data, val_data, test_data, class_labels, dataset_mean, dataset_std


if __name__ == "__main__":
    # Hosuekeeping
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Set up logging
    wandb.init(
        project="continuous-thought-machines-fer",
        dir=args.log_dir,
        config=vars(args),
        name=f"{args.model}_{args.dataset}_bs{args.batch_size}_lr{args.lr}_iters{args.training_iterations}",
        reinit=True,
    )
    wandb.log({"Logging initialized": True})
    assert args.dataset in [
        "cifar10",
        "cifar100",
        "imagenet",
        "RAFDB",
        "FerPlusPlus",
    ], (
        f"Need to be one of cifar10, cifar100, imagenet, RAFDB, FerPlusPlus, got {args.dataset}"
    )

    # Data
    train_data, val_data, test_data, class_labels, dataset_mean, dataset_std = (
        get_dataset(
            args.dataset,
            args.data_root,
            args.val_split_ratio,
            args.use_test_as_val,
            args.grayscale,
        )
    )

    first_sample, first_label = train_data[0]
    print(f"[DEBUG] First sample shape: {first_sample.shape}, mean: {first_sample.mean().item():.4f}, label: {first_label}")

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

    # Conditionally create PonderLoss
    if args.model == "ctm" and args.use_ponder_loss:
        ponder_loss = PonderLoss(lambda_p=args.lambda_p, beta=0.01).to(device)

    # Build model conditionally
    model = None
    if args.model == "ctm":
        model = ContinuousThoughtMachine(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            pretrained_backbone=args.pretrained_backbone,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            grayscale=args.grayscale,
        ).to(device)
    elif args.model == "ctm_gated":
        model = CTMGated(
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            n_synch_out=args.n_synch_out,
            n_synch_action=args.n_synch_action,
            synapse_depth=args.synapse_depth,
            memory_length=args.memory_length,
            deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims,
            do_layernorm_nlm=args.do_normalisation,
            backbone_type=args.backbone_type,
            pretrained_backbone=args.pretrained_backbone,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
            dropout_nlm=args.dropout_nlm,
            neuron_select_type=args.neuron_select_type,
            n_random_pairing_self=args.n_random_pairing_self,
            grayscale=args.grayscale,
            exit_strategy=args.exit_strategy,
            exit_threshold=args.exit_threshold,
            lambda_p=args.lambda_p,
            beta=0.01,
            min_steps=args.min_steps,
        ).to(device)
        ctm_gated_loss = CTMGatedLoss(
            loss_type=args.loss_type,
            lambda_p=args.lambda_p,
            beta=0.01,
            use_most_certain=True,
            max_steps=args.iterations
        ).to(device)
    elif args.model == "lstm":
        model = LSTMBaseline(
            num_layers=args.num_layers,
            iterations=args.iterations,
            d_model=args.d_model,
            d_input=args.d_input,
            heads=args.heads,
            backbone_type=args.backbone_type,
            positional_embedding_type=args.positional_embedding_type,
            out_dims=args.out_dims,
            prediction_reshaper=prediction_reshaper,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "ff":
        model = FFBaseline(
            d_model=args.d_model,
            backbone_type=args.backbone_type,
            out_dims=args.out_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # For lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)

    model.train()

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")

    # Compute and log FLOPs
    if args.model in ["ctm", "ctm_gated"]:
        input_channels = 1 if args.grayscale else 3
        input_shape = (
            (input_channels, 224, 224)
            if args.dataset == "imagenet"
            else (input_channels, 32, 32)
        )
        flops = compute_ctm_flops(model, input_shape, batch_size=1)
        total_flops = flops["total"]
        print(f"Total FLOPs: {total_flops:,}")
        wandb.log({"Total FLOPs": total_flops, "FLOPs Breakdown": flops})

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

    # Optimizer and scheduler (Common setup)
    if len(no_decay_names) and args.weight_decay != 0:
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0},
            ],
            lr=args.lr,
            eps=1e-8 if not args.use_amp else 1e-6,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            eps=1e-8 if not args.use_amp else 1e-6,
            weight_decay=args.weight_decay,
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
    train_accuracies_most_certain = [] if args.model in ["ctm", "lstm", "ctm_gated"] else None
    test_accuracies_most_certain = [] if args.model in ["ctm", "lstm", "ctm_gated"] else None
    # CTM-Gated specific: track exit steps
    train_exit_steps = [] if args.model == "ctm_gated" else None
    test_exit_steps = [] if args.model == "ctm_gated" else None

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
                if args.model in ["ctm", "lstm", "ctm_gated"]:
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

        # Compile synapses only for CTM/CTM-Gated
        if args.model in ["ctm", "ctm_gated"]:
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

                elif args.model == "ctm_gated":
                    predictions, certainties, synchronisation, exit_steps = model(inputs)
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
                    pbar_desc = f"CTM-Gated Loss={loss.item():0.3f}. Acc={accuracy:0.3f}. LR={current_lr:0.6f}. Exit_step={avg_exit_step:0.2f}"

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

                            elif args.model == "ctm_gated":
                                these_predictions, certainties, _, these_exit_steps = model(inputs, use_early_exit=False)
                                loss, where_most_certain = ctm_gated_loss(
                                    these_predictions, certainties, targets, these_exit_steps
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
                                )
                                if train_exit_steps is not None:
                                    train_exit_steps.append(these_exit_steps.float().mean().item())

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

                            elif args.model == "ff":
                                these_predictions = model(inputs)
                                loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
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

                    if args.model in ["ctm", "lstm", "ctm_gated"]:
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

                    if args.model in ["ctm", "lstm", "ctm_gated"]:
                        log_dict["Train Accuracy (Most Certain)"] = (
                            current_train_accuracies_most_certain
                        )
                        for i, acc in enumerate(current_train_accuracies):
                            log_dict[f"Train Accuracy (Tick {i})"] = acc
                        if args.model == "ctm_gated" and train_exit_steps:
                            log_dict["Train Avg Exit Steps"] = np.mean(train_exit_steps)
                    else:  # FF
                        log_dict["Train Accuracy"] = current_train_accuracies

                    wandb.log(log_dict, step=bi)

                del these_predictions

                # Switch to eval mode for test metrics (fixed BN stats)
                model.eval()
                pbar.set_description("Tracking: Computing TEST metrics")
                with torch.inference_mode():  # Use inference_mode for test eval
                    if test_data is None:
                        pbar.set_description("Tracking: No test data, skipping TEST metrics")
                        test_losses.append(0.0)
                        if args.model in ["ctm", "lstm", "ctm_gated"]:
                            test_accuracies.append(np.zeros(args.iterations))
                            test_accuracies_most_certain.append(0.0)
                        else:
                            test_accuracies.append(0.0)
                        wandb.log({"Test Loss": 0.0, "Test Accuracy": 0.0}, step=bi)
                    else:
                        loader = torch.utils.data.DataLoader(
                            test_data,
                            batch_size=args.batch_size_test,
                            shuffle=True,
                            num_workers=num_workers_test,
                        )
                        all_targets_list = []
                        all_predictions_list = []
                        all_predictions_most_certain_list = []  # Only for CTM/LSTM
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

                            elif args.model == "ctm_gated":
                                these_predictions, certainties, _, these_exit_steps = model(inputs, use_early_exit=False)
                                loss, where_most_certain = ctm_gated_loss(
                                    these_predictions, certainties, targets, these_exit_steps
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
                                if test_exit_steps is not None:
                                    test_exit_steps.append(these_exit_steps.float().mean().item())

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

                            elif args.model == "ff":
                                these_predictions = model(inputs)
                                loss = nn.CrossEntropyLoss()(these_predictions, targets)
                                all_predictions_list.append(
                                    these_predictions.argmax(1).detach().cpu().numpy()
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

                    if args.model in ["ctm", "lstm", "ctm_gated"]:
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

                    if args.model in ["ctm", "lstm", "ctm_gated"]:
                        log_dict["Test Accuracy (Most Certain)"] = (
                            current_test_accuracies_most_certain
                        )
                        for i, acc in enumerate(current_test_accuracies):
                            log_dict[f"Test Accuracy (Tick {i})"] = acc
                        if args.model == "ctm_gated" and test_exit_steps:
                            log_dict["Test Avg Exit Steps"] = np.mean(test_exit_steps)
                    else:  # FF
                        log_dict["Test Accuracy"] = current_test_accuracies

                    wandb.log(log_dict, step=bi)

                # Save best model checkpoint based on validation (test) accuracy
                if args.save_best_model and valloader is not None:
                    if args.model in ["ctm", "lstm", "ctm_gated"]:
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
                        if args.model in ["ctm", "lstm", "ctm_gated"]:
                            best_checkpoint["train_accuracies_most_certain"] = train_accuracies_most_certain
                            best_checkpoint["test_accuracies_most_certain"] = test_accuracies_most_certain
                        torch.save(best_checkpoint, best_checkpoint_path)
                        print(f"Saved best checkpoint with val_acc={best_val_acc:.4f} at iteration {bi}")

                # Plotting (conditional)
                if test_data is None:
                    print("Skipping plotting as test_data is None")
                else:
                    figacc = plt.figure(figsize=(10, 10))
                    axacc_train = figacc.add_subplot(211)
                    axacc_test = figacc.add_subplot(212)
                    cm = sns.color_palette("viridis", as_cmap=True)

                    if args.model in ["ctm", "lstm", "ctm_gated"]:
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
                if args.model in ["ctm", "lstm", "ctm_gated"] and testloader is not None:
                    try:  # For safety
                        inputs_viz, targets_viz = next(
                            iter(testloader)
                        )  # Get a fresh batch
                        inputs_viz = inputs_viz.to(device)
                        targets_viz = targets_viz.to(device)

                        pbar.set_description("Tracking: Processing test data for viz")
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
                if args.model in ["ctm", "lstm", "ctm_gated"]:
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
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model with val_acc={checkpoint.get('best_val_acc', 'N/A'):.4f}")
        
        print("Running final evaluation on test set...")
        model.eval()
        final_test_acc = 0
        final_test_loss = 0
        final_test_count = 0
        with torch.inference_mode():
            for i, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                    if args.model in ["ctm", "lstm", "ctm_gated"]:
                        predictions, certainties, _ = model(inputs)
                        loss, where_most_certain = image_classification_loss(
                            predictions, certainties, targets, use_most_certain=True
                        )
                        outputs = predictions[
                            torch.arange(predictions.size(0), device=predictions.device),
                            :,
                            where_most_certain
                        ]
                    else:
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, targets)
                
                if args.model in ["ctm", "lstm", "ctm_gated"]:
                    final_test_acc += (outputs.argmax(1) == targets).float().sum().item()
                else:
                    final_test_acc += (outputs.argmax(1) == targets).float().sum().item()
                
                final_test_loss += loss.item() * inputs.size(0)
                final_test_count += inputs.size(0)
                if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                    break
        
        final_test_acc /= final_test_count
        final_test_loss /= final_test_count
        print(f"Final Test Accuracy: {final_test_acc:.4f}, Test Loss: {final_test_loss:.4f}")
        wandb.log({
            "Final Test Accuracy": final_test_acc,
            "Final Test Loss": final_test_loss,
        }, step=args.training_iterations)
