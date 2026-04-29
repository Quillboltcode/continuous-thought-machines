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
    # Load the best model
    checkpoint_path = 'analysis_checkpoints/rafdb/r18-4_i50_d512_m20/checkpoint.pt'
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
            iterations = 50
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

    # Setup RAF-DB test data
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Using dataset with {len(test_dataset)} samples")
    print("Evaluating accuracy at timesteps: 1, 5, 10, 15, 20, 25, 30")

    # Initialize accuracy tracking for each timestep
    timesteps_to_evaluate = [0, 4, 9, 14, 19, 24, 29]  # 0-indexed (tick 1, 5, 10, 15, 20, 25, 30)
    accuracy_at_timesteps = {f'tick_{i+1}': [] for i in [1, 5, 10, 15, 20, 25, 30]}

    # Process samples in batches
    model.eval()
    total_samples = 0
    correct_at_each_timestep = {f'tick_{i+1}': 0 for i in [1, 5, 10, 15, 20, 25, 30]}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to('cpu')  # Use CPU since we're not using GPU
            targets = targets.to('cpu')

            batch_size = inputs.size(0)

            # Run model inference
            predictions, certainties, _ = model(inputs)

            # predictions shape: (batch_size, out_dims, iterations)
            # We need to evaluate at specific timesteps
            for i, timestep_idx in enumerate(timesteps_to_evaluate):
                tick_name = f'tick_{timestep_idx+1}'

                # Get predictions at this timestep
                timestep_predictions = predictions[:, :, timestep_idx]  # (batch_size, out_dims)

                # Get predicted classes
                pred_classes = timestep_predictions.argmax(dim=1)  # (batch_size,)

                # Count correct predictions
                correct = (pred_classes == targets).sum().item()
                correct_at_each_timestep[tick_name] += correct

            total_samples += batch_size

            if batch_idx % 10 == 0:
                print(f"Processed {total_samples}/{len(test_dataset)} samples")

            # Limit to first 1000 samples for faster evaluation (can be increased)
            if total_samples >= 1000:
                break

    # Calculate accuracies
    print("
Final Results:")
    print(f"Evaluated on {total_samples} samples")
    print("\nAccuracy at each timestep:")

    results = []
    for tick_name in ['tick_1', 'tick_5', 'tick_10', 'tick_15', 'tick_20', 'tick_25', 'tick_30']:
        accuracy = correct_at_each_timestep[tick_name] / total_samples * 100
        results.append(accuracy)
        print(f"{tick_name}: {accuracy:.2f}%")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    timesteps = [1, 5, 10, 15, 20, 25, 30]
    ax.plot(timesteps, results, 'bo-', linewidth=3, markersize=8, alpha=0.8)

    ax.set_xlabel('Thought Iteration (Tick)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CTM Accuracy Evolution Over Time\n(RAF-DB Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(timesteps)
    ax.set_ylim(min(results) - 2, max(results) + 2)
    ax.grid(True, alpha=0.3)

    # Add value labels on points
    for x, y in zip(timesteps, results):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save the plot
    output_dir = 'outputs/temporal_accuracy_analysis'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'accuracy_over_time.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'accuracy_over_time.pdf'), dpi=200, bbox_inches='tight')

    print(f"\nPlot saved to {output_dir}/accuracy_over_time.png")

    # Save numerical results
    with open(os.path.join(output_dir, 'accuracy_results.txt'), 'w') as f:
        f.write("CTM Accuracy at Different Timesteps\n")
        f.write("==================================\n\n")
        f.write(f"Dataset: RAF-DB Test Set\n")
        f.write(f"Samples evaluated: {total_samples}\n")
        f.write(f"Model: {checkpoint_path}\n\n")
        f.write("Results:\n")
        for tick, acc in zip([1, 5, 10, 15, 20, 25, 30], results):
            f.write(f"Tick {tick}: {acc:.2f}%\n")

    print(f"Numerical results saved to {output_dir}/accuracy_results.txt")

if __name__ == '__main__':
    main()