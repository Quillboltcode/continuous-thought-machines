import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
from tqdm.auto import tqdm
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from utils.housekeeping import set_seed, zip_python_code
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import warnings
warnings.filterwarnings("ignore")

import gc
import torchvision
torchvision.disable_beta_transforms_warning()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    parser.add_argument("--grayscale", action="store_true", default=False, help="Use grayscale images")
    parser.add_argument("--conv1_channels", type=int, default=3, help="Number of input channels for conv1 (1 for grayscale, 3 for RGB)")
    parser.add_argument("--conv1_kernel_size", type=int, default=7, help="Kernel size for conv1")
    parser.add_argument("--convert_grayscale_to_rgb", action="store_true", default=False, help="Convert grayscale images to 3-channel RGB")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--batch_size_test", type=int, default=32, help="Batch size for testing.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the model.")
    parser.add_argument("--training_iterations", type=int, default=5001, help="Number of training iterations.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps.")
    parser.add_argument("--use_scheduler", action="store_true", default=True, help="Use a learning rate scheduler.")
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["multistep", "cosine"], help="Type of learning rate scheduler.")
    parser.add_argument("--milestones", type=int, default=[8000, 15000, 20000], nargs="+", help="Learning rate scheduler milestones.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate scheduler gamma for multistep.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay factor.")
    parser.add_argument("--gradient_clipping", type=float, default=-1, help="Gradient quantile clipping value (-1 to disable).")
    parser.add_argument("--num_workers_train", type=int, default=1, help="Num workers training.")

    parser.add_argument("--log_dir", type=str, default="logs/finetune", help="Directory for logging.")
    parser.add_argument("--dataset", type=str, default="FerPlusPlus", help="Dataset to use.")
    parser.add_argument("--data_root", type=str, default="data/", help="Where to save/get dataset.")
    parser.add_argument("--save_every", type=int, default=250, help="Save checkpoints every this many iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--reload", action="store_true", default=False, help="Reload from disk?")
    parser.add_argument("--track_every", type=int, default=125, help="Track metrics every this many iterations.")
    parser.add_argument("--n_test_batches", type=int, default=50, help="How many minibatches to approx metrics. Set to -1 for full eval")
    parser.add_argument("--device", type=int, nargs="+", default=[0], help="List of GPU(s) to use. Set to -1 to use CPU.")
    parser.add_argument("--use_amp", action="store_true", default=False, help="AMP autocast.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Ratio of training data to use for validation (0.0-1.0).")
    parser.add_argument("--use_test_as_val", action="store_true", default=False, help="Use test set as validation.")
    parser.add_argument("--final_test_eval", action="store_true", default=False, help="Run final evaluation on test set after training.")

    args = parser.parse_args()
    return args


class FinetuneModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True, grayscale=False, conv1_channels=3, conv1_kernel_size=7):
        super().__init__()
        self.grayscale = grayscale
        
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        in_features = self.backbone.fc.in_features
        
        if conv1_channels == 1 or grayscale:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=conv1_kernel_size, stride=2, padding=conv1_kernel_size//2, bias=False)
            if pretrained:
                self.backbone.conv1.weight.data = self.backbone.conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=conv1_kernel_size, stride=2, padding=conv1_kernel_size//2, bias=False)
        
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def get_dataset(dataset, root, val_split_ratio=0.0, use_test_as_val=False, grayscale=False, convert_grayscale_to_rgb=False):
    if dataset == "FerPlusPlus":
        if grayscale and not convert_grayscale_to_rgb:
            dataset_mean = [0.5]
            dataset_std = [0.5]
        else:
            dataset_mean = [0.485, 0.456, 0.406]
            dataset_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        if grayscale and not convert_grayscale_to_rgb:
            train_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                normalize,
            ])
        elif convert_grayscale_to_rgb:
            train_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                normalize,
            ])

        train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
        val_data = datasets.ImageFolder(os.path.join(root, "val"), transform=test_transform)
        test_data = datasets.ImageFolder(os.path.join(root, "test"), transform=test_transform)
        class_labels = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

        if use_test_as_val:
            val_data = test_data
            test_data = None
        elif val_split_ratio > 0.0:
            train_size = int((1.0 - val_split_ratio) * len(train_data))
            val_size = len(train_data) - train_size
            train_data, _ = random_split(
                train_data, [train_size, val_size],
                generator=torch.Generator().manual_seed(412),
            )
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return train_data, val_data, test_data, class_labels


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed, False)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    wandb.init(
        project="continuous-thought-machines-fer",
        dir=args.log_dir,
        config=vars(args),
        name=f"finetune_{args.backbone}_grayscale={args.grayscale}_rgb_conv={args.convert_grayscale_to_rgb}",
        reinit=True,
    )

    train_data, val_data, test_data, class_labels = get_dataset(
        args.dataset, args.data_root, args.val_split_ratio, args.use_test_as_val, args.grayscale, args.convert_grayscale_to_rgb
    )

    first_sample, first_label = train_data[0]
    print(f"[DEBUG] First sample shape: {first_sample.shape}, mean: {first_sample.mean().item():.4f}, label: {first_label}")

    num_classes = len(class_labels)
    print(f"Dataset: {args.dataset}, Classes: {num_classes}")

    trainloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train
    )

    if val_data is not None:
        valloader = DataLoader(val_data, batch_size=args.batch_size_test, shuffle=False, num_workers=1)
    else:
        valloader = None

    if test_data is not None:
        testloader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=1)
    else:
        testloader = None

    device = f"cuda:{args.device[0]}" if args.device[0] >= 0 else "cpu"
    print(f"Running on {device}")

    # If converting grayscale to RGB, ensure conv1_channels is 3
    if args.convert_grayscale_to_rgb:
        args.conv1_channels = 3
    
    model = FinetuneModel(args.backbone, num_classes, pretrained=args.pretrained, grayscale=args.grayscale, conv1_channels=args.conv1_channels, conv1_kernel_size=args.conv1_kernel_size).to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == "multistep":
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler_type == "cosine":
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    iters = []
    best_val_acc = 0.0
    best_checkpoint_path = f"{args.log_dir}/best_checkpoint.pt"

    start_iter = 0
    if args.reload:
        checkpoint_path = f"{args.log_dir}/checkpoint.pt"
        if os.path.isfile(checkpoint_path):
            print(f"Reloading from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_iter = checkpoint["iteration"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            train_accuracies = checkpoint["train_accuracies"]
            test_accuracies = checkpoint["test_accuracies"]
            iters = checkpoint["iters"]
            best_val_acc = checkpoint.get("best_val_acc", 0.0)

    iterator = iter(trainloader)

    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]["lr"]

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                accuracy = (outputs.argmax(1) == targets).float().mean().item()

            scaler.scale(loss).backward()

            if args.gradient_clipping != -1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            pbar.set_description(f"Loss={loss.item():0.4f}. Acc={accuracy:0.4f}. LR={current_lr:0.6f}")
            pbar.update(1)

            if bi % args.track_every == 0 and bi > 0:
                iters.append(bi)

                model.eval()
                with torch.inference_mode():
                    train_acc = 0
                    train_loss = 0
                    train_count = 0
                    loader = DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=1)
                    for i, (inputs, targets) in enumerate(loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                            outputs = model(inputs)
                            loss = nn.CrossEntropyLoss()(outputs, targets)
                        train_acc += (outputs.argmax(1) == targets).float().sum().item()
                        train_loss += loss.item() * inputs.size(0)
                        train_count += inputs.size(0)
                        if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                            break
                    train_acc /= train_count
                    train_loss /= train_count

                    val_acc = 0
                    val_loss = 0
                    val_count = 0
                    if valloader is not None:
                        for i, (inputs, targets) in enumerate(valloader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                                outputs = model(inputs)
                                loss = nn.CrossEntropyLoss()(outputs, targets)
                            val_acc += (outputs.argmax(1) == targets).float().sum().item()
                            val_loss += loss.item() * inputs.size(0)
                            val_count += inputs.size(0)
                            if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                                break
                        val_acc /= val_count
                        val_loss /= val_count

                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                if valloader is not None:
                    test_losses.append(val_loss)
                    test_accuracies.append(val_acc)

                wandb.log({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Val Loss": val_loss if valloader else 0,
                    "Val Accuracy": val_acc if valloader else 0,
                    "Learning Rate": current_lr,
                }, step=bi)

                if valloader is not None and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_checkpoint = {
                        "iteration": bi + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_losses": train_losses,
                        "test_losses": test_losses,
                        "train_accuracies": train_accuracies,
                        "test_accuracies": test_accuracies,
                        "iters": iters,
                        "best_val_acc": best_val_acc,
                    }
                    torch.save(best_checkpoint, best_checkpoint_path)
                    print(f"Saved best checkpoint with val_acc={best_val_acc:.4f} at iteration {bi + 1}")

                model.train()

            if bi % args.save_every == 0 and bi > 0:
                checkpoint = {
                    "iteration": bi + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_accuracies": train_accuracies,
                    "test_accuracies": test_accuracies,
                    "iters": iters,
                    "best_val_acc": best_val_acc,
                }
                torch.save(checkpoint, f"{args.log_dir}/checkpoint.pt")

    print("Training complete!")

    if args.final_test_eval and testloader is not None:
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
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
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
