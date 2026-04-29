import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

# Import CTM components
from models.ctm import ContinuousThoughtMachine
from torchvision import datasets, transforms
from tasks.image_classification.plotting import make_classification_gif

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

    # Setup dataset
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

    # Process sample 0
    sample_idx = 0
    print(f"\nProcessing sample {sample_idx} for GIF generation")

    # Get sample
    image, target = test_dataset[sample_idx]
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize for display

    print(f"Ground truth emotion: {class_labels[target]}")

    # Run model with tracking
    input_tensor = image.unsqueeze(0)
    with torch.no_grad():
        predictions, certainties, synchronisation, pre_activations, post_activations, attention_tracking = model(input_tensor, track=True)

    # Convert to numpy
    predictions = predictions.squeeze(0).numpy()
    certainties = certainties.squeeze(0).numpy()
    post_activations = post_activations.squeeze(1)
    attention_tracking = attention_tracking.squeeze(1)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Certainties shape: {certainties.shape}")
    print(f"Post-activations shape: {post_activations.shape}")
    print(f"Attention tracking shape: {attention_tracking.shape}")

    # Create output directory
    output_dir = 'outputs/gif_generation'
    os.makedirs(output_dir, exist_ok=True)

    # Generate GIF using make_classification_gif
    print("\nGenerating GIF with attention evolution...")
    gif_path = os.path.join(output_dir, f'sample_{sample_idx}_attention.gif')

    make_classification_gif(
        image_np, target, predictions, certainties,
        post_activations, attention_tracking, class_labels, gif_path
    )

    print(f"\nSuccess! GIF saved to {gif_path}")

    # Also show final prediction
    final_pred = predictions[:, -1].argmax()
    final_prob = torch.softmax(torch.from_numpy(predictions[:, -1]), -1).max().item()
    final_certainty = certainties[1, -1]

    print("\nFinal Results:")
    print(f"Predicted emotion: {class_labels[final_pred]}")
    print(f"Confidence: {final_prob:.3f}")
    print(f"Model certainty: {final_certainty:.3f}")
    print(f"Correct: {'✓' if final_pred == target else '✗'}")

if __name__ == '__main__':
    main()