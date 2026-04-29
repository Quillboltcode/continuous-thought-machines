import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
sns.set_style('darkgrid')

# Import CTM components
from models.ctm import ContinuousThoughtMachine
from torchvision import datasets, transforms

def main():
    # Hardcoded parameters for 2 samples from ~/Downloads/DATASET/test
    checkpoint_path = 'analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt'
    data_indices = [0, 1]
    output_dir = 'outputs/2_samples_png_analysis'
    inference_iterations = 50  # Match the model's trained iterations

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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
            iterations = inference_iterations
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

    # Setup RAF-DB test data from ~/Downloads/DATASET/test
    data_root = os.path.expanduser('~/Downloads/DATASET/test')
    if not os.path.exists(data_root):
        print(f"Error: Data not found at {data_root}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    class_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    print(f"Using dataset with {len(test_dataset)} samples and {len(class_labels)} classes")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating PNG analysis for {len(data_indices)} samples...")

    # Process only the specified samples
    for idx, sample_idx in enumerate(data_indices):
        if sample_idx >= len(test_dataset):
            print(f"Warning: Sample {sample_idx} out of range, skipping")
            continue

        print(f"\nProcessing sample {sample_idx} ({idx+1}/{len(data_indices)})")

        # Get sample
        image, target = test_dataset[sample_idx]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize for display

        # Run model
        input_tensor = image.unsqueeze(0)
        with torch.no_grad():
            predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking = model(input_tensor, track=True)

        # Convert to numpy for processing
        predictions = predictions.squeeze(0).numpy()  # (out_dims, iterations)
        certainties = certainties.squeeze(0).numpy()  # (2, iterations)
        post_activations = post_activations.squeeze(1)  # (iterations, d_model)

        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Certainties shape: {certainties.shape}")
        print(f"  Post-activations shape: {post_activations.shape}")

        # Create sample directory
        sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_dir, exist_ok=True)

        # Generate key analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Input image
        axes[0,0].imshow(image_np)
        axes[0,0].set_title(f'Sample {sample_idx}\nGround Truth: {class_labels[target]}', fontsize=14)
        axes[0,0].axis('off')

        # Plot 2: Certainty evolution
        certainties_history = certainties[1, :]
        x_vals = np.arange(len(certainties_history))
        axes[0,1].plot(x_vals, certainties_history, 'b-', linewidth=3, alpha=0.8)

        # Color background based on correctness at each step
        for ii, (x, cert) in enumerate(zip(x_vals, certainties_history)):
            if ii < len(predictions[0]):
                pred_class = predictions[:, ii].argmax()
                is_correct = pred_class == target
                color = 'lightgreen' if is_correct else 'lightcoral'
                axes[0,1].axvspan(ii, ii + 1, facecolor=color, edgecolor=None, lw=0, alpha=0.3)

        axes[0,1].set_xlim([0, inference_iterations])
        axes[0,1].set_ylim([-0.05, 1.05])
        axes[0,1].set_xlabel('Thought Iteration', fontsize=12)
        axes[0,1].set_ylabel('Model Certainty', fontsize=12)
        axes[0,1].set_title('Certainty Evolution Over Time', fontsize=14)
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Final predictions
        final_preds = predictions[:, -1]  # Last iteration
        ps = np.exp(final_preds) / np.sum(np.exp(final_preds))  # Softmax
        k = min(7, len(class_labels))
        topk_indices = np.argsort(ps)[::-1][:k]
        top_classes = [class_labels[i] for i in topk_indices]
        top_probs = ps[topk_indices]

        bars = axes[1,0].bar(range(k), top_probs, color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
        # Highlight ground truth
        gt_idx = np.where(topk_indices == target)[0]
        if len(gt_idx) > 0:
            bars[gt_idx[0]].set_color('darkorange')
            bars[gt_idx[0]].set_edgecolor('darkred')

        axes[1,0].set_xticks(range(k))
        axes[1,0].set_xticklabels([cls[:8] for cls in top_classes], rotation=45, ha='right', fontsize=10)
        axes[1,0].set_ylabel('Predicted Probability', fontsize=12)
        axes[1,0].set_title('Final Predictions (Last Iteration)', fontsize=14)
        axes[1,0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Neuron activation heatmap at final step
        if len(post_activations) > 0:
            neuron_acts = post_activations[-1]  # Final step activations

            # Create a 2D grid for visualization
            n_neurons = len(neuron_acts)
            grid_size = int(np.sqrt(n_neurons))
            if grid_size * grid_size < n_neurons:
                grid_size += 1

            # Create square grid and fill with activations
            act_grid = np.full((grid_size, grid_size), np.nan)
            act_grid.flat[:n_neurons] = neuron_acts
            act_grid = np.nan_to_num(act_grid, nan=np.mean(neuron_acts))

            im = axes[1,1].imshow(act_grid, cmap='RdYlBu_r', vmin=-2, vmax=2, aspect='auto')
            axes[1,1].set_title(f'Final Neuron Activations\n({n_neurons} neurons)', fontsize=14)
            axes[1,1].axis('off')
            cbar = plt.colorbar(im, ax=axes[1,1], fraction=0.046, shrink=0.8)
            cbar.set_label('Activation Strength', fontsize=10)

        plt.suptitle(f'CTM Analysis: Sample {sample_idx} - {inference_iterations} Thought Iterations',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save the analysis plot
        analysis_path = os.path.join(sample_dir, 'analysis.png')
        plt.savefig(analysis_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  Saved analysis plot to {analysis_path}")

        # Also save individual evolution plots for key metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Certainty evolution
        axes[0].plot(range(inference_iterations), certainties[1, :], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Certainty')
        axes[0].set_title('Certainty Evolution')
        axes[0].grid(True)

        # Prediction stability (argmax over time)
        pred_classes = predictions.argmax(axis=0)
        axes[1].plot(range(inference_iterations), pred_classes, 'r.-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Predicted Class')
        axes[1].set_title('Prediction Stability')
        axes[1].set_yticks(range(len(class_labels)))
        axes[1].set_yticklabels([cls[:8] for cls in class_labels])
        axes[1].grid(True)

        # Neuron activation variance over time
        neuron_vars = [np.var(acts) for acts in post_activations]
        axes[2].plot(range(inference_iterations), neuron_vars, 'g-', linewidth=2)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Neuron Activation Variance')
        axes[2].set_title('Neural Dynamics')
        axes[2].grid(True)

        plt.suptitle(f'Evolution Metrics - Sample {sample_idx}')
        plt.tight_layout()

        metrics_path = os.path.join(sample_dir, 'evolution_metrics.png')
        plt.savefig(metrics_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"  Saved evolution metrics to {metrics_path}")

    print(f"\nAll PNG analysis generated in {output_dir}")
    print("You can now use these plots in your paper!")

if __name__ == '__main__':
    main()