import os
import torch
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    """Uniform interface for dataset splits."""
    train: Any
    val: Optional[Any]
    test: Optional[Any]
    class_labels: List[str]
    mean: List[float]
    std: List[float]


def _create_transforms(mean: List[float], std: List[float], img_size: int = 100, 
                       grayscale: bool = False, convert_grayscale_to_rgb: bool = False,
                       train: bool = True, use_augmentation: bool = True):
    """Create transform pipeline based on configuration."""
    if grayscale and not convert_grayscale_to_rgb:
        num_channels = 1
    elif convert_grayscale_to_rgb:
        num_channels = 3
    else:
        num_channels = 3

    transform_list = []

    if train and use_augmentation:
        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=num_channels))
        transform_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=num_channels))
        transform_list.append(transforms.Resize((img_size, img_size)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)


def _apply_val_split(train_data, val_split_ratio: float, use_test_as_val: bool = False, 
                      test_data=None):
    """Apply validation split logic."""
    if use_test_as_val:
        return train_data, test_data, None
    elif val_split_ratio > 0.0:
        train_size = int((1.0 - val_split_ratio) * len(train_data))
        val_size = len(train_data) - train_size
        train_split, val_split = torch.utils.data.random_split(
            train_data,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(412),
        )
        return train_split, val_split, test_data
    else:
        return train_data, None, test_data


def load_cifar10(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False):
    """Load CIFAR-10 dataset."""
    dataset_mean = [0.49139968, 0.48215827, 0.44653124]
    dataset_std = [0.24703233, 0.24348505, 0.26158768]
    
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    full_train_data = datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(root, train=False, transform=test_transform, download=True)
    
    class_labels = ["air", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    train_data, val_data, test_data = _apply_val_split(full_train_data, val_split_ratio, use_test_as_val, test_data)

    return DatasetBundle(train=train_data, val=val_data, test=test_data, 
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


def load_cifar100(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False):
    """Load CIFAR-100 dataset."""
    dataset_mean = [0.5070751592371341, 0.48654887331495067, 0.4409178433670344]
    dataset_std = [0.2673342858792403, 0.2564384629170882, 0.27615047132568393]
    
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_data = datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(root, train=False, transform=test_transform, download=True)
    
    idx_order = np.argsort(np.array(list(train_data.class_to_idx.values())))
    class_labels = list(np.array(list(train_data.class_to_idx.keys()))[idx_order])

    train_data, val_data, test_data = _apply_val_split(train_data, val_split_ratio, use_test_as_val, test_data)

    return DatasetBundle(train=train_data, val=val_data, test=test_data,
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


def load_imagenet(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False):
    """Load ImageNet dataset."""
    from data.custom_datasets import ImageNet
    from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES

    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = ImageNet(which_split="train", transform=train_transform)
    test_data = ImageNet(which_split="validation", transform=test_transform)
    val_data = None

    class_labels = list(IMAGENET2012_CLASSES.values())

    return DatasetBundle(train=train_data, val=val_data, test=test_data,
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


def load_ferplusplus(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False,
                      grayscale: bool = False, convert_grayscale_to_rgb: bool = False, img_size: int = 100):
    """Load FERPlusPlus dataset."""
    if grayscale and not convert_grayscale_to_rgb:
        dataset_mean = [0.5]
        dataset_std = [0.5]
    else:
        dataset_mean = [0.485, 0.456, 0.406]
        dataset_std = [0.229, 0.224, 0.225]

    train_transform = _create_transforms(dataset_mean, dataset_std, img_size, grayscale, convert_grayscale_to_rgb, train=True)
    test_transform = _create_transforms(dataset_mean, dataset_std, img_size, grayscale, convert_grayscale_to_rgb, train=False)

    train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(root, "val"), transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(root, "test"), transform=test_transform)
    class_labels = sorted(train_data.classes)

    train_data, val_data, test_data = _apply_val_split(train_data, val_split_ratio, use_test_as_val, test_data)

    return DatasetBundle(train=train_data, val=val_data, test=test_data,
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


def load_rafdb(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False, img_size: int = 100):
    """Load RAF-DB dataset."""
    dataset_mean = [0.485, 0.456, 0.406]
    dataset_std = [0.229, 0.224, 0.225]

    train_transform = _create_transforms(dataset_mean, dataset_std, img_size, train=True)
    test_transform = _create_transforms(dataset_mean, dataset_std, img_size, train=False)

    train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    
    val_path = os.path.join(root, "val")
    test_path = os.path.join(root, "test")
    
    print(f"[RAFDB] Checking dataset folders in: {root}")
    print(f"[RAFDB] train folder exists: {os.path.exists(os.path.join(root, 'train'))}")
    print(f"[RAFDB] val folder exists: {os.path.exists(val_path)}")
    print(f"[RAFDB] test folder exists: {os.path.exists(test_path)}")
    
    if os.path.exists(val_path):
        val_data = datasets.ImageFolder(val_path, transform=test_transform)
        test_data = datasets.ImageFolder(test_path, transform=test_transform) if os.path.exists(test_path) else None
    else:
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
    
    class_labels = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    return DatasetBundle(train=train_data, val=val_data, test=test_data,
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


def load_affectnet(root: str, val_split_ratio: float = 0.0, use_test_as_val: bool = False, img_size: int = 100):
    """Load AffectNet dataset."""
    temp_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    temp_train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=temp_transform)
    
    print("[INFO] Computing dataset mean and std from training set...")
    dataset_mean, dataset_std = compute_dataset_stats(temp_train_data)
    print(f"[INFO] AffectNet computed stats - mean: {dataset_mean}, std: {dataset_std}")
    
    train_transform = _create_transforms(dataset_mean, dataset_std, img_size, train=True)
    test_transform = _create_transforms(dataset_mean, dataset_std, img_size, train=False)
    
    train_data = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(root, "val"), transform=test_transform)
    test_data = datasets.ImageFolder(os.path.join(root, "test"), transform=test_transform) if os.path.exists(os.path.join(root, "test")) else None
    
    affectnet_idx_to_label = {
        "0": "Neutral", "1": "Happy", "2": "Sad", "3": "Surprise",
        "4": "Fear", "5": "Disgust", "6": "Anger", "7": "Contempt",
    }
    class_labels = [affectnet_idx_to_label[i] for i in train_data.classes]
    
    train_data, val_data, test_data = _apply_val_split(train_data, val_split_ratio, use_test_as_val, test_data)

    return DatasetBundle(train=train_data, val=val_data, test=test_data,
                         class_labels=class_labels, mean=dataset_mean, std=dataset_std)


DATASET_REGISTRY = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "imagenet": load_imagenet,
    "ferplusplus": load_ferplusplus,
    "rafdb": load_rafdb,
    "affectnet": load_affectnet,
}


def get_dataset(
    dataset: str,
    root: str,
    val_split_ratio: float = 0.0,
    use_test_as_val: bool = False,
    grayscale: bool = False,
    convert_grayscale_to_rgb: bool = False,
    img_size: int = 100,
):
    """
    Load and return train, validation, and test datasets.

    Args:
        dataset: Dataset name ('imagenet', 'cifar10', 'cifar100', 'ferplusplus', 'rafdb', 'affectnet')
        root: Root directory for the dataset
        val_split_ratio: Ratio of training data to use for validation
        use_test_as_val: Use test set as validation
        grayscale: Use grayscale images
        convert_grayscale_to_rgb: Convert grayscale to 3-channel RGB
        img_size: Image size to resize to

    Returns:
        train_data, val_data, test_data, class_labels, dataset_mean, dataset_std
    """
    dataset = dataset.lower()
    
    if dataset not in DATASET_REGISTRY:
        raise NotImplementedError(f"Dataset {dataset} not supported. Available: {list(DATASET_REGISTRY.keys())}")
    
    loader_fn = DATASET_REGISTRY[dataset]
    
    if dataset in ["ferplusplus"]:
        bundle = loader_fn(root, val_split_ratio, use_test_as_val, grayscale, convert_grayscale_to_rgb, img_size)
    elif dataset in ["rafdb", "affectnet"]:
        bundle = loader_fn(root, val_split_ratio, use_test_as_val, img_size)
    elif dataset in ["cifar10", "cifar100"]:
        bundle = loader_fn(root, val_split_ratio, use_test_as_val)
    elif dataset == "imagenet":
        bundle = loader_fn(root, val_split_ratio, use_test_as_val)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    
    return bundle.train, bundle.val, bundle.test, bundle.class_labels, bundle.mean, bundle.std


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
