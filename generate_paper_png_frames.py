import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import patheffects
from scipy.special import softmax
from sklearn.decomposition import PCA

# Import CTM components
from models.ctm import ContinuousThoughtMachine
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PNG frames for CTM paper visualization")
    parser.add_argument('--checkpoint', type=str,
                       default='analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt',
                       help="Path to CTM checkpoint")
    parser.add_argument('--data_indices', type=int, nargs='+', default=[0, 1, 2],
                       help="Sample indices to visualize")
    parser.add_argument('--output_dir', type=str, default='ctm_paper_png_frames',
                       help="Directory for PNG frames")
    parser.add_argument('--inference_iterations', type=int, default=30,
                       help="Iterations to run")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size for processing")
    return parser.parse_args()

def create_visualization_frame(image, target, predictions, certainties, post_activations,
                              stepi, total_steps, class_labels, sample_idx, save_path):
    """Create a single PNG frame showing CTM state at a specific iteration"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Input image
    axes[0,0].imshow(image)
    axes[0,0].set_title(f'Sample {sample_idx}\nGround Truth: {class_labels[target]}', fontsize=12)
    axes[0,0].axis('off')

    # Plot 2: Certainty evolution
    certainties_history = certainties[1, :stepi+1]
    x_vals = np.arange(len(certainties_history))
    axes[0,1].plot(x_vals, certainties_history, 'b-', linewidth=3, alpha=0.8)

    # Color background based on correctness at each step
    for ii, (x, cert) in enumerate(zip(x_vals, certainties_history)):
        if ii < len(predictions[0]):
            pred_class = predictions[:, ii].argmax()
            is_correct = pred_class == target
            color = 'lightgreen' if is_correct else 'lightcoral'
            axes[0,1].axvspan(ii, ii + 1, facecolor=color, edgecolor=None, lw=0, alpha=0.3)

    axes[0,1].set_xlim([0, total_steps])
    axes[0,1].set_ylim([-0.05, 1.05])
    axes[0,1].set_xlabel('Thought Iteration', fontsize=10)
    axes[0,1].set_ylabel('Model Certainty', fontsize=10)
    axes[0,1].set_title('Certainty Evolution Over Time', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Current predictions
    if stepi < len(predictions[0]):
        ps = softmax(predictions[:, stepi])
        k = min(7, len(class_labels))
        topk_indices = np.argsort(ps)[::-1][:k]
        topk_probs = ps[topk_indices]
        top_classes = [class_labels[i] for i in topk_indices]

        bars = axes[0,2].bar(range(k), topk_probs, color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
        # Highlight ground truth
        gt_idx = np.where(topk_indices == target)[0]
        if len(gt_idx) > 0:
            bars[gt_idx[0]].set_color('darkorange')
            bars[gt_idx[0]].set_edgecolor('darkred')

        axes[0,2].set_xticks(range(k))
        axes[0,2].set_xticklabels([cls[:8] for cls in top_classes], rotation=45, ha='right', fontsize=9)
        axes[0,2].set_ylabel('Predicted Probability', fontsize=10)
        axes[0,2].set_title(f'Top-{k} Predictions (Step {stepi+1})', fontsize=12)
        axes[0,2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Neuron activation patterns
    if stepi < len(post_activations):
        neuron_acts = post_activations[stepi]

        # Create a meaningful 2D representation
        n_neurons = len(neuron_acts)
        grid_size = int(np.sqrt(n_neurons))
        if grid_size * grid_size < n_neurons:
            grid_size += 1

        # Create square grid and fill with activations
        act_grid = np.full((grid_size, grid_size), np.nan)  # Use NaN for empty spots
        act_grid.flat[:n_neurons] = neuron_acts

        # Replace NaN with mean for better visualization
        act_grid = np.nan_to_num(act_grid, nan=np.mean(neuron_acts))

        im = axes[1,0].imshow(act_grid, cmap='RdYlBu_r', vmin=-2, vmax=2, aspect='auto')
        axes[1,0].set_title(f'Neuron Activation Patterns\n(Step {stepi+1})', fontsize=12)
        axes[1,0].axis('off')
        cbar = plt.colorbar(im, ax=axes[1,0], fraction=0.046, shrink=0.8)
        cbar.set_label('Activation Strength', fontsize=10)

    # Plot 5: Latent space trajectory (PCA)
    if len(post_activations) > 1:
        # Use all activations up to current step for PCA
        acts_for_pca = np.array(post_activations[:stepi+1])  # (steps, neurons)

        if acts_for_pca.shape[0] > 2:  # Need at least 3 points for meaningful PCA
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(acts_for_pca)

            # Plot trajectory
            axes[1,1].plot(pca_coords[:, 0], pca_coords[:, 1], 'b-', alpha=0.7, linewidth=2)
            axes[1,1].scatter(pca_coords[:, 0], pca_coords[:, 1], c=range(len(pca_coords)),
                             cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)

            # Highlight current step
            if stepi < len(pca_coords):
                axes[1,1].scatter(pca_coords[stepi, 0], pca_coords[stepi, 1], c='red', s=150,
                                 marker='*', edgecolors='black', linewidth=2, zorder=10)

            axes[1,1].set_xlabel('PC1', fontsize=10)
            axes[1,1].set_ylabel('PC2', fontsize=10)
            axes[1,1].set_title('Latent Space Trajectory', fontsize=12)
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Not enough data\nfor trajectory',
                          ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Latent Space Trajectory', fontsize=12)

    # Plot 6: Synchronization analysis
    if len(post_activations) > 0:
        # Compute average activation strength over time
        sync_strengths = [np.mean(np.abs(acts)) for acts in post_activations[:stepi+1]]
        x_sync = np.arange(len(sync_strengths))

        axes[1,2].plot(x_sync, sync_strengths, 'g-', linewidth=3, alpha=0.8)
        axes[1,2].fill_between(x_sync, sync_strengths, alpha=0.3, color='green')

        # Mark current step
        if stepi < len(sync_strengths):
            axes[1,2].scatter([stepi], [sync_strengths[stepi]], c='red', s=100,
                             marker='o', edgecolors='black', linewidth=2, zorder=10)

        axes[1,2].set_xlim([0, total_steps])
        axes[1,2].set_xlabel('Thought Iteration', fontsize=10)
        axes[1,2].set_ylabel('Avg |Neuron Activation|', fontsize=10)
        axes[1,2].set_title('Neural Synchronization Strength', fontsize=12)
        axes[1,2].grid(True, alpha=0.3)

    plt.suptitle(f'Continuous Thought Machine Reasoning Process\nSample {sample_idx} - Iteration {stepi+1}/{total_steps}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Extract model args
    model_args = checkpoint.get('args')
    if model_args is None:
        print("Warning: No args in checkpoint, using defaults")
        class ModelArgs:
            d_model = 512
            d_input = 256
            heads = 8
            n_synch_out = 256
            n_synch_action = 256
            neuron_select_type = 'random'
            iterations = args.inference_iterations
            memory_length = 20
            deep_memory = True
            memory_hidden_dims = 32
            do_normalisation = False
            positional_embedding_type = 'none'
            backbone_type = 'resnet18-4'
            out_dims = 7
            grayscale = False
        model_args = ModelArgs()

    # Create model
    model = ContinuousThoughtMachine(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.d_input,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=model_args.n_synch_action,
        synapse_depth=4,
        memory_length=model_args.memory_length,
        deep_nlms=model_args.deep_memory,
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=model_args.do_normalisation,
        backbone_type=model_args.backbone_type,
        pretrained_backbone=getattr(model_args, 'pretrained_backbone', None),
        positional_embedding_type=model_args.positional_embedding_type,
        out_dims=model_args.out_dims,
        neuron_select_type=model_args.neuron_select_type,
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Setup test data from ~/Downloads/DATASET/test
    data_root = os.path.expanduser('~/Downloads/DATASET/test')
    if not os.path.exists(data_root):
        print(f"Warning: Data not found at {data_root}, using CIFAR-10 for demo")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        # Custom dataset setup (assuming RAF-DB format with 7 classes)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dataset = datasets.ImageFolder(root=data_root, transform=transform)
        # RAF-DB emotion classes
        class_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        print(f"Using dataset from {data_root} with {len(test_dataset)} samples and {len(class_labels)} classes")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating PNG frames for {len(args.data_indices)} samples...")

    # Process each sample
    for idx, sample_idx in enumerate(args.data_indices):
        if sample_idx >= len(test_dataset):
            print(f"Warning: Sample {sample_idx} out of range, using sample {sample_idx % len(test_dataset)}")
            sample_idx = sample_idx % len(test_dataset)

        print(f"\nProcessing sample {sample_idx} ({idx+1}/{len(args.data_indices)})")

        # Get sample
        image, target = test_dataset[sample_idx]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize for display

        # Add batch dimension and run model
        input_tensor = image.unsqueeze(0)
        with torch.no_grad():
            predictions, certainties, synchronisation_tracking, pre_activations, post_activations, attention_tracking = model(input_tensor, track=True)

        # Convert to numpy for processing
        predictions = predictions.squeeze(0).numpy()  # (out_dims, iterations)
        certainties = certainties.squeeze(0).numpy()  # (2, iterations)
        # post_activations is (iterations, batch_size, d_model) - squeeze batch dimension
        post_activations = post_activations.squeeze(1)  # Now (iterations, d_model)
        print(f"Debug: predictions shape {predictions.shape}, certainties shape {certainties.shape}, post_activations shape {post_activations.shape}")

        # Create sample directory
        sample_dir = os.path.join(args.output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)

        # Generate frames (every few iterations to keep reasonable number)
        frame_step = max(1, args.inference_iterations // 10)  # About 10 frames per sample
        frames_saved = []

        for stepi in range(0, args.inference_iterations, frame_step):
            frame_path = os.path.join(sample_dir, f'iteration_{stepi+1:03d}.png')
            create_visualization_frame(
                image_np, target, predictions, certainties, post_activations,
                stepi, args.inference_iterations, class_labels, sample_idx, frame_path
            )
            frames_saved.append(frame_path)

        print(f"  Saved {len(frames_saved)} PNG frames to {sample_dir}")

    print(f"\nAll PNG frames generated in {args.output_dir}")
    print("Each sample has frames showing the CTM's reasoning evolution over iterations.")
    print("You can now select the most informative frames for your paper!")

if __name__ == '__main__':
    main()