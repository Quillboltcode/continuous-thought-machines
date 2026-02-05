import os
import torch
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as T

# --- Configuration for Standard Datasets ---
# The logic for these mean/std values is kept outside for cleaner function bodies.
# CIFAR-10 mean/std for Tensors
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# ImageNet mean/std (commonly used for ImageFolder/RAF-DB) for Tensors
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(dataset_name, train):
    """
    Defines and returns the V2 transform pipeline for a given dataset and phase.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        # CIFAR-10 generally does not use extensive pre-normalization augmentations
        if train:
            # T.AutoAugment is often used for SOTA CIFAR-10 but keeping it simple here
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    
    elif dataset_name in ["raf-db", "imagefolder"]:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        
        # --- TRAINING (RAF-DB / ImageFolder) ---
        if train:
            # Note: V2 Compose handles PIL -> Tensor -> PIL -> Tensor conversion implicitly
            transform = T.Compose([
                # Augmentations applied on PIL/Tensor image
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                
                # Preprocessing
                T.Resize(256),
                T.CenterCrop(224),
                
                # Final conversion and normalization (T.ToTensor is now T.ToImage and T.ToDtype)
                T.ToImage(),  # Converts to T.Image (internal tensor representation)
                T.ToDtype(torch.float32, scale=True), # Converts to float, scales to [0, 1]
                T.Normalize(mean, std),
            ])
        
        # --- VALIDATION/TESTING (RAF-DB / ImageFolder) ---
        else:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224), # Common practice is CenterCrop 224 from 256 for validation/test
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean, std),
            ])
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    return transform

# ----------------------------------------------------------------------
# Cleaned-up Data Loader Functions
# ----------------------------------------------------------------------

def get_train_val_dataloaders(dataset_name, data_dir, batch_size=64, val_ratio=0.2, num_workers=4):
    """
    Returns the train and validation dataloaders for the specified dataset.
    Splits the 'train' portion into train and validation sets.
    """
    train_transform = get_transforms(dataset_name, train=True)
    val_transform = get_transforms(dataset_name, train=False)

    if dataset_name.lower() == "cifar10":
        # Get the full train dataset with *train* transforms
        full_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
    else:
        # For other datasets, load from the 'train' subdirectory with *train* transforms
        full_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=train_transform
        )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = math.ceil(total_size * val_ratio) # Use ceil to ensure val_size >= 1
    train_size = total_size - val_size
    
    # Split the dataset (maintaining the train transforms on both splits initially)
    # NOTE: The *proper* way to assign different transforms to the splits is more complex
    # but the simplest approach is to modify the dataset transform after splitting.
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # --- IMPORTANT: Assign validation transforms to the val_dataset ---
    # This ensures no augmentation is used on the validation split.
    # We must access the underlying dataset's transform attribute.
    # This pattern only works if full_dataset is ImageFolder/CIFAR10/etc.
    if hasattr(val_dataset.dataset, 'transform'):
        # For the validation set, we use the validation transforms
        val_dataset.dataset.transform = val_transform


    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


def get_test_dataloader(dataset_name, data_dir, batch_size=64, num_workers=4):
    """
    Returns the test dataloader for the specified dataset.
    """
    test_transform = get_transforms(dataset_name, train=False)
    
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
    else:
        dataset = datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform=test_transform
        )
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader