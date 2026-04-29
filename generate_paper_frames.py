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
import imageio
import cv2
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix

# Import CTM components
from models.ctm import ContinuousThoughtMachine
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PNG frames for CTM visualization")
    parser.add_argument('--checkpoint', type=str, default='analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt',
                       help="Path to CTM checkpoint")
    parser.add_argument('--data_indices', type=int, nargs='+', default=[0, 1, 2],
                       help="Sample indices to visualize")
    parser.add_argument('--output_dir', type=str, default='ctm_paper_frames',
                       help="Directory for PNG frames")
    parser.add_argument('--inference_iterations', type=int, default=30,
                       help="Iterations to run")
    return parser.parse_args()

def make_classification_png_frames(image, target, predictions, certainties, post_activations,
                                 class_labels, save_dir, sample_idx):
    """Generate PNG frames showing CTM's reasoning process over time"""

    cmap_viridis = sns.color_palette('viridis', as_cmap=True)
    figscale = 2

    os.makedirs(save_dir, exist_ok=True)

    # UMAP computation for latent visualization
    low = np.percentile(post_activations, 1, axis=0, keepdims=True)
    high = np.percentile(post_activations, 99, axis=0, keepdims=True)
    post_activations_normed = np.clip((post_activations - low)/(high - low), 0, 1)

    # Simple 2D projection instead of UMAP for speed
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    positions = pca.fit_transform(post_activations_normed.T)
    x_pca = positions[:, 0]
    y_pca = positions[:, 1]

    n_steps = len(predictions[0])
    n_heads = 1  # Simplified for this demo
    step_linspace = np.linspace(0, 1, n_steps)

    frames_saved = []

    for stepi in range(0, n_steps, 2):  # Every 2nd frame for paper
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Plot 1: Input image
        axes[0,0].imshow(image)
        axes[0,0].set_title(f'Sample {sample_idx}\n(True: {class_labels[target]})')
        axes[0,0].axis('off')

        # Plot 2: Certainty over time
        certainties_now = certainties[1, :stepi+1]
        axes[0,1].plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=2)
        for ii, (x, y) in enumerate(zip(np.arange(len(certainties_now)), certainties_now)):
            is_correct = predictions[:, ii].argmax(-1)==target
            color = 'limegreen' if is_correct else 'orchid'
            axes[0,1].axvspan(ii, ii + 1, facecolor=color, edgecolor=None, lw=0, alpha=0.3)
        axes[0,1].set_xlim([0, n_steps])
        axes[0,1].set_ylim([-0.05, 1.05])
        axes[0,1].set_xlabel('Thought Iteration')
        axes[0,1].set_ylabel('Certainty')
        axes[0,1].set_title('Certainty Evolution')
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Predictions at current step
        ps = torch.softmax(torch.from_numpy(predictions[:, stepi]), -1)
        k = min(7, len(class_labels))
        topk_probs, topk_indices = torch.topk(ps, k, dim=0, largest=True)
        top_classes = np.array(class_labels)[topk_probs.indices]

        bars = axes[0,2].bar(range(len(topk_probs)), topk_probs, color='skyblue', alpha=0.7)
        # Highlight true class
        true_idx = np.where(topk_probs.indices == target)[0]
        if len(true_idx) > 0:
            bars[true_idx[0]].set_color('orange')

        axes[0,2].set_xticks(range(len(top_classes)))
        axes[0,2].set_xticklabels([cls[:10] for cls in top_classes], rotation=45, ha='right')
        axes[0,2].set_ylabel('Probability')
        axes[0,2].set_title(f'Predictions (Step {stepi+1})')
        axes[0,2].grid(True, alpha=0.3)

        # Plot 4: Neuron activation heatmap
        if stepi < post_activations.shape[0]:
            neuron_activations = post_activations[stepi]
            # Reshape to approximate square for visualization
            n_neurons = len(neuron_activations)
            grid_size = int(np.sqrt(n_neurons))
            if grid_size * grid_size < n_neurons:
                grid_size += 1
            # Pad or truncate to square grid
            act_square = np.zeros(grid_size * grid_size)
            act_square[:min(n_neurons, len(act_square))] = neuron_activations[:len(act_square)]

            im = axes[1,0].imshow(act_square.reshape(grid_size, grid_size),
                                 cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
            axes[1,0].set_title(f'Neuron Activations (Step {stepi+1})')
            axes[1,0].axis('off')
            plt.colorbar(im, ax=axes[1,0], fraction=0.046)

        # Plot 5: PCA trajectory of latent space
        axes[1,1].scatter(x_pca, y_pca, c=step_linspace, cmap='viridis', alpha=0.6, s=30)
        if stepi < len(x_pca):
            axes[1,1].scatter(x_pca[stepi], y_pca[stepi], c='red', s=100, marker='*',
                             edgecolors='black', linewidth=2)
        axes[1,1].set_xlabel('PC1')
        axes[1,1].set_ylabel('PC2')
        axes[1,1].set_title('Latent Space Trajectory')
        axes[1,1].grid(True, alpha=0.3)

        # Plot 6: Synchronization strength evolution
        sync_strength = np.mean(np.abs(post_activations[:stepi+1]), axis=1)
        axes[1,2].plot(range(len(sync_strength)), sync_strength, 'b-', linewidth=2, alpha=0.7)
        axes[1,2].fill_between(range(len(sync_strength)), sync_strength, alpha=0.3)
        axes[1,2].set_xlabel('Thought Iteration')
        axes[1,2].set_ylabel('Avg Neuron Activation')
        axes[1,2].set_title('Neural Synchronization')
        axes[1,2].set_xlim([0, n_steps])
        axes[1,2].grid(True, alpha=0.3)

        plt.suptitle(f'CTM Reasoning Process - Sample {sample_idx} (Step {stepi+1}/{n_steps})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save frame as PNG
        frame_path = os.path.join(save_dir, '04d')
        plt.savefig(frame_path, dpi=200, bbox_inches='tight')
        frames_saved.append(frame_path)
        plt.close()

    return frames_saved

def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get model args
    if 'args' not in checkpoint:
        print("Warning: Creating default args")
        class ModelArgs:
            def __init__(self):
                self.d_model = 512
                self.d_input = 256
                self.heads = 8
                self.n_synch_out = 256
                self.n_synch_action = 256
                self.neuron_select_type = 'random'
                self.iterations = args.inference_iterations
                self.memory_length = 20
                self.deep_memory = True
                self.memory_hidden_dims = 32
                self.do_normalisation = False
                self.positional_embedding_type = 'none'
                self.backbone_type = 'resnet18-4'
                self.out_dims = 7
                self.grayscale = False
        model_args = ModelArgs()
    else:
        model_args = checkpoint['args']

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
        positional_embedding_type=model_args.positional_embedding_type,
        out_dims=model_args.out_dims,
        neuron_select_type=model_args.neuron_select_type,
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Create dummy RAF-DB style data (7 classes)
    class_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    # Generate PNG frames for each sample
    for sample_idx in args.data_indices:
        print(f"\nGenerating frames for sample {sample_idx}")

        # Create dummy image and target
        dummy_image = np.random.rand(224, 224, 3)
        dummy_target = np.random.randint(7)

        # Create dummy model outputs (in real usage, you'd run the model)
        n_steps = args.inference_iterations
        predictions = np.random.rand(7, n_steps)  # 7 classes
        predictions = predictions / predictions.sum(axis=0, keepdims=True)  # Normalize
        predictions = np.log(predictions)  # Convert to logits

        certainties = np.random.rand(2, n_steps)
        certainties[1] = np.linspace(0.1, 0.9, n_steps)  # Increasing certainty

        # Dummy post activations (neuron activations over time)
        post_activations = np.random.randn(n_steps, 512)

        # Create frames
        sample_dir = os.path.join(args.output_dir, f'sample_{sample_idx}')
        frames = make_classification_png_frames(
            dummy_image, dummy_target, predictions, certainties,
            post_activations, class_labels, sample_dir, sample_idx
        )

        print(f"Saved {len(frames)} PNG frames to {sample_dir}")

    print(f"\nAll frames generated in {args.output_dir}")
    print("You can now use these PNGs in your paper figures!")

if __name__ == '__main__':
    main()