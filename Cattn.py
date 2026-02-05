import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from timm.data import resolve_data_config
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import wandb
import matplotlib.pyplot as plt
import timm


def create_resnet18_backbone(pretrained=True, features_only=True, out_indices=(4,)):
    return timm.create_model(
        "resnet18",
        pretrained=pretrained,
        features_only=features_only,
        out_indices=out_indices,
    )

def set_seed(seed=42, deterministic=True):
    """
    ... and the answer is ... 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

class SimplifiedCTM(nn.Module):
    """
    A minimal model implementing the core recurrence from the CTM,
    matching the provided diagram.
    """

    def __init__(
        self,
        d_model=256,
        d_attn=256,
        d_backbone=512,
        iterations=5,
        dropout=0.1,
        num_heads=4,
        num_classes=10,
        pretrained_backbone=True,
    ):
        super(SimplifiedCTM, self).__init__()

        self.iterations = iterations
        self.d_model = d_model
        self.d_attn = d_attn  # Dimension for attention
        self.d_backbone = d_backbone  # Dimension from backbone
        self.dropout = dropout
        self.pretrained_backbone = pretrained_backbone
        self.num_classes = num_classes  # Default for CIFAR-10; can be modified as needed

        # Backbone (to get features from input)
        # ResNet18's stage 4 output has 512 features
        self.backbone = create_resnet18_backbone(
            pretrained=self.pretrained_backbone, features_only=True, out_indices=(4,)
        )

        # Cross Attention
        self.q_proj = nn.Linear(d_model, d_attn)
        self.kv_proj = nn.Linear(d_backbone, d_attn)  # Project from 512 -> 256
        self.attention = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=num_heads, batch_first=True
        )

        # "GAP" / Processing Block (Synapse Model)
        self.synapse = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_attn + d_model, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )

        # "Neuron-Level Model" (Simplified: just a linear layer for update)
        self.nlms = nn.Linear(d_model, d_model)  # Simulates the NLM

        # Initial state as a learnable parameter of zeros (maybe random or 1/sqrt could be better)
        self.start_state = nn.Parameter(torch.zeros(d_model))

        # Last Classifier
        self.classifier = nn.Linear(d_model, self.num_classes)

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Featurise Input Data ---
        # backbone returns a list of feature maps, even if there's only one.
        features_map = self.backbone(x)[0]  # Shape: (B, C, H, W)

        # Global Average Pooling to flatten spatial dimensions
        pooled_features = features_map.mean(dim=(-2, -1))  # Shape: (B, C)
        kv = self.kv_proj(pooled_features).unsqueeze(1)  # Shape: (B, 1, D_input)

        # --- Initialise Recurrent State ---
        z_t = self.start_state.unsqueeze(0).expand(B, -1)  # Shape: (B, D_model)

        # --- Tracking ---
        states = []
        if track:
            states.append(z_t.detach().cpu().numpy())

        # --- Recurrent Loop ---
        for t in range(self.iterations):
            # --- Cross Attention ---
            q = self.q_proj(z_t).unsqueeze(1)  # Shape: (B, 1, D_input)
            attn_out, _ = self.attention(
                q, kv, kv, need_weights=False
            )  # Shape: (B, 1, D_input)
            attn_out = attn_out.squeeze(1)  # Shape: (B, D_input)

            # --- Concatenate with z_t ("Could be concat") ---
            concat_input = torch.cat(
                [attn_out, z_t], dim=1
            )  # Shape: (B, D_input + D_model)

            # --- Apply Synapse ("GAP" / Processing) ---
            x_prime = self.synapse(concat_input)  # Shape: (B, D_model)

            # --- Update z_t to z_{t+1} (via "NLM") ---
            z_t = self.nlms(x_prime)  # Shape: (B, D_model)

            # --- Track ---
            if track:
                states.append(z_t.detach().cpu().numpy())

        # --- Final Classification ---
        logits = self.classifier(z_t)
        return (logits, states) if track else logits





def get_dataloader(dataset_name, data_dir, batch_size=64, train=True, val_ratio=0.2):
    if dataset_name.lower() == "cifar10":
        # CIFAR-10 specific logic remains unchanged
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    
    else:
        # --- RAF-DB / ImageFolder Logic ---
        
        # 1. Define Augmentations for TRAINING (train=True)
        if train:
            transform = transforms.Compose(
                [
                    # --- Augmentations from RAF-DB papers ---
                    transforms.RandomHorizontalFlip(), # Flips the image with p=0.5
                    transforms.RandomRotation(10),     # Rotates up to +/- 10 degrees
                    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Simple color adjustment

                    # --- Standard Preprocessing ---
                    transforms.Resize(256),            # Resize to a larger size
                    transforms.CenterCrop(224),        # Crop the final 224x224 (for consistency)
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        
        # 2. Define NO Augmentations for TESTING (train=False)
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),       # Resize to a larger size
                    transforms.CenterCrop(224),   # Center crop the final image area
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        
        # Load the full train dataset (since we only have train and test folders)
        full_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=transform
        )
        
        # If train=True, split the train dataset into train and validation
        if train:
            # Calculate split sizes
            total_size = len(full_dataset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            
            # Split the dataset
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Return the appropriate split based on the train parameter
            # We need to handle this differently - let's modify the approach
            return train_dataset, val_dataset
        else:
            # For test set, return the test dataset as before
            dataset = datasets.ImageFolder(
                os.path.join(data_dir, "test"), transform=transform
            )
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            return loader

def get_train_val_dataloaders(dataset_name, data_dir, batch_size=64, val_ratio=0.2):
    """New function to return both train and validation dataloaders"""
    import torch
    
    if dataset_name.lower() == "cifar10":
        # CIFAR-10 specific logic
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        full_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
    else:
        # For other datasets (like RAF-DB), use train transforms for training data
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        
        full_dataset = datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=train_transform
        )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_dataloader(dataset_name, data_dir, batch_size=64):
    """Function to get test dataloader"""
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        
        dataset = datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform=transform
        )
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return loader

# Updated main function that maintains backward compatibility
def get_dataloader(dataset_name, data_dir, batch_size=64, train=True, val_ratio=0.2, return_val=False):
    if return_val:
        return get_train_val_dataloaders(dataset_name, data_dir, batch_size, val_ratio)
    
    if dataset_name.lower() == "cifar10":
        # CIFAR-10 specific logic remains unchanged
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    
    else:
        # --- RAF-DB / ImageFolder Logic ---
        
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            # Load the full train dataset and split it
            full_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "train"), transform=transform
            )
            
            total_size = len(full_dataset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Return train dataset if train=True
            dataset = train_dataset
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            
            dataset = datasets.ImageFolder(
                os.path.join(data_dir, "test"), transform=transform
            )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True
    )
    return loader


def main():
    # Parameters - set directly for notebook use
    dataset = "imagefolder"  # 'cifar10' or 'imagefolder'
    data_dir = "/kaggle/input/raf-db-dataset/DATASET"
    batch_size = 128
    epochs = 100
    lr = 1e-4
    warmup_epochs = 5
    thought_steps = 10
    seed = 42
    train_val_split = 0.2
    set_seed(seed)

    # Create a simple 'args' object
    class Args:
        def __init__(
            self,
            dataset,
            data_dir,
            batch_size,
            epochs,
            lr,
            warmup_epochs,
            thought_steps,
            train_val_split,
            seed
        ):
            self.dataset = dataset
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.warmup_epochs = warmup_epochs
            self.thought_steps = thought_steps
            self.train_val_split = train_val_split
            self.seed = seed

    args = Args(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        warmup_epochs=warmup_epochs,
        thought_steps=thought_steps,
        train_val_split=train_val_split,
        seed=seed

    )

    # Initialize wandb
    wandb.init(
        project="ctm-simplified-training",  # Replace with your project name
        config=vars(args)  # Log all arguments as hyperparameters
    )
    # Optionally, update wandb.config with other relevant parameters not in args
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_train_val_dataloaders(
        args.dataset, args.data_dir, args.batch_size, val_ratio=0.2
    )
    test_loader = get_test_dataloader(
        args.dataset, args.data_dir, args.batch_size
    )

    # Model
    num_classes = 10 if args.dataset == "cifar10" else len(os.listdir(os.path.join(args.data_dir, "train")))
    wandb.config.update({"seed": seed, "num_classes": num_classes})
    model = SimplifiedCTM(
        d_model=256,
        d_attn=256,
        d_backbone=512,  # Explicitly set backbone feature dimension
        iterations=args.thought_steps,
        dropout=0.1,
        pretrained_backbone=True,
        num_classes=num_classes,
        num_heads=4,
    )

    model = model.to(device)
    
    # Watch model for gradients and parameters
    wandb.watch(model, log="all")

    # Optimizer & LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup + Cosine scheduler (timm style)
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        decay_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        cooldown_epochs=0,
        min_lr=1e-6,
    )

    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics for plotting
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
    
    # Variables to track best model
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        # Training phase
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # Now returns only logits
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        scheduler.step(epoch + 1)  # timm schedulers expect epoch count starting at 1

        # Validation phase
        model.eval()
        val_correct = 0
        val_total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate and store metrics for this epoch
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / len(train_loader.dataset)
        val_acc = 100.0 * val_correct / len(val_loader.dataset)
        val_loss = val_total_loss / len(val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            torch.save(best_model_state, "best_model.pth")
            wandb.log({
                "best_val_accuracy": best_val_acc
            }, step=epoch + 1)
            print(f"  -> New best model saved! Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Load the best model for final evaluation
    model.load_state_dict(best_model_state)
    print(f"Loaded best model from epoch {best_epoch} for final evaluation.")

    # Final evaluation on test set using the best model
    print("Evaluating best model on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)

    test_acc = 100.0 * test_correct / test_total
    print(f"Final Test Accuracy with best model: {test_acc:.2f}%")
    
    # Log final test accuracy to wandb
    wandb.log({"final_test_accuracy": test_acc})
    wandb.finish() # End the wandb run

    # --- Visualization of Results ---
    epochs_range = range(1, args.epochs + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Training and Validation Metrics")

    # Plot Training and Validation Accuracy
    ax1.plot(epochs_range, history["train_acc"], "o-", label="Training Accuracy")
    ax1.plot(epochs_range, history["val_acc"], "o-", label="Validation Accuracy")
    ax1.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Val Acc: {best_val_acc:.2f}%')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="lower right")
    ax1.set_title("Training and Validation Accuracy")

    # Plot Training Loss
    ax2.plot(epochs_range, history["train_loss"], "o-", label="Training Loss")
    ax2.plot(epochs_range, history["val_loss"], "o-", label="Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.set_title("Training and Validation Loss")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Print summary
    print(f"\n=== Model Selection Summary ===")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Validation-test gap: {best_val_acc - test_acc:.2f}%")
    
    return model, history, best_val_acc, test_acc