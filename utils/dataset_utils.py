import torch
from collections import Counter


def compute_dataset_stats(dataset, num_samples=None):
    """
    Compute mean and std of a dataset.
    
    Args:
        dataset: PyTorch dataset with ToTensor transforms applied
        num_samples: Number of samples to compute stats from (None = all)
    
    Returns:
        mean: List of mean values per channel
        std: List of std values per channel
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
        
        if num_samples and total_samples >= num_samples:
            break
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


def compute_class_distribution(dataset):
    """
    Compute class distribution in a dataset.
    
    Args:
        dataset: PyTorch dataset with __getitem__ returning (image, label)
    
    Returns:
        Counter: Dictionary mapping class index to count
    """
    labels = []
    for _, label in dataset:
        labels.append(label)
    return Counter(labels)


def log_class_histogram_wandb(class_counts, class_names, split="train"):
    """
    Log class distribution histogram to wandb.
    
    Args:
        class_counts: Counter object or dict mapping class index to count
        class_names: List of class names
        split: Dataset split name (train/val/test)
    """
    import wandb
    
    if isinstance(class_counts, Counter):
        class_counts = dict(class_counts)
    
    sorted_counts = {class_names[k]: class_counts.get(k, 0) for k in sorted(class_counts.keys())}
    
    wandb.log({
        f"class_counts/{split}": sorted_counts,
    })
