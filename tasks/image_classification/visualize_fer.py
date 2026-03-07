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


def get_transforms(grayscale, convert_grayscale_to_rgb, split="train"):
    if grayscale and not convert_grayscale_to_rgb:
        dataset_mean = [0.5]
        dataset_std = [0.5]
    else:
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
    
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    
    if grayscale and not convert_grayscale_to_rgb:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,
        ])
    elif convert_grayscale_to_rgb:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def visualize_samples(data_root, split, grayscale, convert_grayscale_to_rgb, num_samples, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    
    transform = get_transforms(grayscale, convert_grayscale_to_rgb, split)
    
    dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=transform)
    
    if convert_grayscale_to_rgb:
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]
    else:
        dataset_mean = [0.5]
        dataset_std = [0.5]
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        img, label = dataset[idx]
        
        img_denorm = denormalize(img, dataset_mean, dataset_std)
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        if img_denorm.shape[0] == 1:
            img_denorm = img_denorm.squeeze(0)
            ax.imshow(img_denorm, cmap="gray")
        else:
            img_denorm = img_denorm.permute(1, 2, 0).numpy()
            ax.imshow(img_denorm)
        
        ax.set_title(f"Label: {FER_PLUS_PLUS_CLASSES[label]}", fontsize=10)
        ax.axis("off")
    
    plt.suptitle(f"Fer++ {split.upper()} - After Transform\n(convert_grayscale_to_rgb={convert_grayscale_to_rgb}, grayscale={grayscale})", fontsize=14)
    plt.tight_layout()
    
    return fig, indices, dataset, FER_PLUS_PLUS_CLASSES


def main():
    args = parse_args()
    
    if args.log_to_wandb:
        wandb.init(
            project="fer-visualization",
            name=f"fer_visualize_{args.split}_rgb_conv={args.convert_grayscale_to_rgb}",
            reinit=True,
        )
    
    print(f"Visualizing Fer++ dataset with:")
    print(f"  - split: {args.split}")
    print(f"  - grayscale: {args.grayscale}")
    print(f"  - convert_grayscale_to_rgb: {args.convert_grayscale_to_rgb}")
    print(f"  - num_samples: {args.num_samples}")
    
    fig, indices, dataset, class_names = visualize_samples(
        args.data_root,
        args.split,
        args.grayscale,
        args.convert_grayscale_to_rgb,
        args.num_samples,
        args.seed,
    )
    
    if args.log_to_wandb:
        wandb.log({"fer_transformed_samples": wandb.Image(fig)})
        wandb.log({
            "metadata": {
                "split": args.split,
                "grayscale": args.grayscale,
                "convert_grayscale_to_rgb": args.convert_grayscale_to_rgb,
                "num_samples": len(indices),
                "seed": args.seed,
                "class_names": class_names,
            }
        })
        
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            wandb.log({f"sample_{i}_label": class_names[label]})
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fer_{args.split}_rgb_conv={args.convert_grayscale_to_rgb}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_dir}/{filename}")
    
    if args.log_to_wandb:
        artifact = wandb.Artifact("fer-transformed-samples", type="visualization")
        artifact.add_file(os.path.join(output_dir, filename))
        wandb.log_artifact(artifact)
    
    plt.close(fig)


if __name__ == "__main__":
    main()
