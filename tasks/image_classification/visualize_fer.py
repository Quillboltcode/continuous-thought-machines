import argparse
import os
import random
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms

FER_PLUS_PLUS_CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/", help="Where to get dataset.")
    parser.add_argument("--grayscale", action="store_true", default=False, help="Use grayscale images")
    parser.add_argument("--convert_grayscale_to_rgb", action="store_true", default=True, help="Convert grayscale images to 3-channel RGB")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--log_to_wandb", action="store_true", default=True, help="Log to wandb")
    return parser.parse_args()


def get_base_transforms(grayscale, convert_grayscale_to_rgb):
    if convert_grayscale_to_rgb:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    elif grayscale:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])


def get_full_transforms(grayscale, convert_grayscale_to_rgb):
    if grayscale and not convert_grayscale_to_rgb:
        dataset_mean = [0.5]
        dataset_std = [0.5]
    else:
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
    
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    
    if grayscale and not convert_grayscale_to_rgb:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,
        ])
    elif convert_grayscale_to_rgb:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            normalize,
        ])


def get_normalize_params(grayscale, convert_grayscale_to_rgb):
    if grayscale and not convert_grayscale_to_rgb:
        return [0.5], [0.5]
    else:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def visualize_comparison(data_root, split, grayscale, convert_grayscale_to_rgb, num_samples, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    
    base_transform = get_base_transforms(grayscale, convert_grayscale_to_rgb)
    full_transform = get_full_transforms(grayscale, convert_grayscale_to_rgb)
    mean, std = get_normalize_params(grayscale, convert_grayscale_to_rgb)
    
    base_dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=base_transform)
    norm_dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=full_transform)
    
    indices = random.sample(range(len(base_dataset)), min(num_samples, len(base_dataset)))
    
    figures = []
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        img_base, label = base_dataset[idx]
        img_norm, _ = norm_dataset[idx]
        
        img_denorm = denormalize(img_norm, mean, std)
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        if img_base.shape[0] == 1:
            img_base_vis = img_base.squeeze(0)
            axes[0].imshow(img_base_vis, cmap="gray")
        else:
            img_base_vis = img_base.permute(1, 2, 0).numpy()
            axes[0].imshow(img_base_vis)
        
        axes[0].set_title("Before Normalize", fontsize=12)
        axes[0].axis("off")
        
        if img_denorm.shape[0] == 1:
            img_denorm_vis = img_denorm.squeeze(0)
            axes[1].imshow(img_denorm_vis, cmap="gray")
        else:
            img_denorm_vis = img_denorm.permute(1, 2, 0).numpy()
            axes[1].imshow(img_denorm_vis)
        
        axes[1].set_title("After Normalize", fontsize=12)
        axes[1].axis("off")
        
        fig.suptitle(f"Sample {i+1}: {FER_PLUS_PLUS_CLASSES[label]}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        figures.append(fig)
    
    return figures, indices, base_dataset, FER_PLUS_PLUS_CLASSES


def main():
    args = parse_args()
    
    if args.log_to_wandb:
        wandb.init(
            project="fer-visualization",
            name=f"fer_comparison_{args.split}_rgb_conv={args.convert_grayscale_to_rgb}",
            reinit=True,
        )
    
    print(f"Visualizing Fer++ dataset with:")
    print(f"  - split: {args.split}")
    print(f"  - grayscale: {args.grayscale}")
    print(f"  - convert_grayscale_to_rgb: {args.convert_grayscale_to_rgb}")
    print(f"  - num_samples: {args.num_samples}")
    
    figures, indices, dataset, class_names = visualize_comparison(
        args.data_root,
        args.split,
        args.grayscale,
        args.convert_grayscale_to_rgb,
        args.num_samples,
        args.seed,
    )
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.log_to_wandb:
        for i, fig in enumerate(figures):
            _, label = dataset[indices[i]]
            wandb.log({f"sample_{i+1}_before_after": wandb.Image(fig)})
        
        wandb.log({
            "metadata": {
                "split": args.split,
                "grayscale": args.grayscale,
                "convert_grayscale_to_rgb": args.convert_grayscale_to_rgb,
                "num_samples": len(figures),
                "seed": args.seed,
                "class_names": class_names,
            }
        })
        
        artifact = wandb.Artifact("fer-normalize-comparison", type="visualization")
        
        for i, fig in enumerate(figures):
            filename = f"fer_sample_{i+1}_comparison.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
            artifact.add_file(os.path.join(output_dir, filename))
        
        wandb.log_artifact(artifact)
        print(f"Logged {len(figures)} figures to wandb")
    else:
        for i, fig in enumerate(figures):
            filename = f"fer_sample_{i+1}_comparison.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    
    print(f"Saved {len(figures)} figures to {output_dir}/")
    
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
