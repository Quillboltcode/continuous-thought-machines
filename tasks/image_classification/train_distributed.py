import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torch.nn as nn 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.samplers import FastRandomDistributedSampler 
from tqdm.auto import tqdm

from tasks.image_classification.train import get_dataset
from models.factories import create_model
from models.ctm_with_innovations import PredictiveSynchronyLoss, CrossTickContrastiveLoss

# Plotting/Utils Imports
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup

import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore", message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings('ignore', message='divide by zero encountered in power', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="UserWarning: Metadata Warning, tag 274 had too many entries: 4, expected 1")
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning, r"^PIL\.TiffImagePlugin$")
warnings.filterwarnings("ignore", "UserWarning: Metadata Warning", UserWarning, r"^PIL\.TiffImagePlugin$")
warnings.filterwarnings("ignore", "UserWarning: Truncated File Read", UserWarning, r"^PIL\.TiffImagePlugin$")


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['ctm', 'ctm_gated', 'ctm_with_innovations', 'clip_ctm', 'ctm_fwpkm', 'lstm', 'ff'],
                        help='Model type to train.')

    parser.add_argument('--adapter_reduction', type=int, default=4,
                        help='Bottleneck reduction ratio for adapters.')

    # CLIP-CTM specific
    parser.add_argument('--clip_model_name', type=str, default='ViT-B-32',
                        help='OpenCLIP model name for clip_ctm (e.g., ViT-B-32, ViT-L-14).')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b79k',
                        help='OpenCLIP pretrained dataset for clip_ctm (e.g., laion2b_s34b_b79k, openai).')
    parser.add_argument('--alpha_init', type=float, default=0.5,
                        help='Initial alpha value for CLIP-Adapter blend.')
    parser.add_argument('--text_prompts', type=str, default=None,
                        help='JSON string for text prompts with templates and classes (clip_ctm).')

    # CTM-FwPKM specific
    parser.add_argument('--fwpkm_d_key', type=int, default=64, help='FwPKM key dimension (ctm_fwpkm).')
    parser.add_argument('--fwpkm_d_val', type=int, default=128, help='FwPKM value dimension (ctm_fwpkm).')
    parser.add_argument('--fwpkm_n_mem', type=int, default=1024, help='FwPKM memory slots, must be perfect square (ctm_fwpkm).')
    parser.add_argument('--fwpkm_top_k', type=int, default=8, help='FwPKM top-k retrieval (ctm_fwpkm).')
    parser.add_argument('--fwpkm_lr', type=float, default=0.1, help='FwPKM fast-weight learning rate (ctm_fwpkm).')
    parser.add_argument('--fwpkm_chunk_size', type=int, default=16, help='FwPKM write chunk size (ctm_fwpkm).')

    # Model Architecture - Common
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--backbone_type', type=str, default='resnet18-4', help='Type of backbone featureiser.')
    parser.add_argument('--pretrained_backbone', type=str, default='none', 
                        choices=['none', 'imagenet', 'ms-celeba'], help='Use a pretrained backbone.')
    parser.add_argument('--grayscale', action=argparse.BooleanOptionalAction, default=False, help='Use grayscale images.')
    parser.add_argument('--convert_grayscale_to_rgb', action=argparse.BooleanOptionalAction, default=False, 
                        help='Convert grayscale images to 3-channel RGB.')

    # CTM / LSTM specific
    parser.add_argument('--d_input', type=int, default=512, help='Dimension of the input (CTM, LSTM). Must match CLIP output dim (512 for ViT-B-32).')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads (CTM, LSTM).') 
    parser.add_argument('--iterations', type=int, default=30, help='Number of internal ticks (CTM, LSTM).') 
    parser.add_argument('--positional_embedding_type', type=str, default='none', 
                        choices=['none', 'learnable-fourier', 'multi-learnable-fourier', 'custom-rotational', 'custom-rotational-1d'],
                        help='Type of positional embedding.')

    # CTM specific
    parser.add_argument('--use_ponder_loss', action=argparse.BooleanOptionalAction, default=False, help='Use PonderLoss for CTM.')
    parser.add_argument('--lambda_p', type=float, default=0.2, help='Lambda for PonderLoss.')
    parser.add_argument('--beta', type=float, default=0.01, help='Beta for ponder/loop loss.')
    parser.add_argument('--synapse_depth', type=int, default=4, help='Depth of U-NET for synapse.')
    parser.add_argument('--n_synch_out', type=int, default=128, help='Number of neurons for output synch.')
    parser.add_argument('--n_synch_action', type=int, default=128, help='Number of neurons for action synch.')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing', 
                        choices=['first-last', 'random', 'random-pairing'],
                        help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=0, help='Self-to-self synch pairs.')
    parser.add_argument('--memory_length', type=int, default=20, help='Pre-activation history length.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory.')
    parser.add_argument('--memory_hidden_dims', type=int, default=16, help='Hidden dims for deep memory.')
    parser.add_argument('--dropout_nlm', type=float, default=None, help='Dropout for NLMs.')
    parser.add_argument('--do_normalisation', type=lambda x: x.lower() == 'true', default=False,
                        help='Apply normalization in NLMs. Use --do_normalisation true or --do_normalisation false.')

    # CTM-Gated specific
    parser.add_argument('--exit_strategy', type=str, default='certainty',
                        choices=['certainty', 'ponder', 'normal', 'learned', 'none'],
                        help='Exit strategy for CTM-Gated.')
    parser.add_argument('--exit_threshold', type=float, default=0.9, help='Certainty threshold.')
    parser.add_argument('--min_steps', type=int, default=1, help='Minimum steps before early exit.')
    parser.add_argument('--loss_type', type=str, default='standard', choices=['standard', 'ponder', 'loop'],
                        help='Loss type for CTM-Gated.')

    # CTM-Innovations specific
    parser.add_argument('--use_gsh', action=argparse.BooleanOptionalAction, default=False, 
                        help='Enable Gated Synchronization Highway.')
    parser.add_argument('--use_hne', action=argparse.BooleanOptionalAction, default=False, 
                        help='Enable Hierarchical NLM Ensembles.')
    parser.add_argument('--use_sanp', action=argparse.BooleanOptionalAction, default=False, 
                        help='Enable Sparse Adaptive Neuron Pairing.')
    parser.add_argument('--hne_group_configs', type=str, default=None, help='JSON string for HNE group configs.')
    parser.add_argument('--hne_group_configs_file', type=str, default=None, help='Path to JSON file with HNE configs.')
    parser.add_argument('--sanp_init_top_k', type=int, default=1000, help='Candidate pairs for SANP.')
    parser.add_argument('--use_psl', action=argparse.BooleanOptionalAction, default=False, 
                        help='Use Predictive Synchrony Loss.')
    parser.add_argument('--lambda_psl', type=float, default=0.1, help='Weight for PSL.')
    parser.add_argument('--use_ctcs', action=argparse.BooleanOptionalAction, default=False, 
                        help='Use Cross-Tick Contrastive Synchronization.')
    parser.add_argument('--lambda_ctcs', type=float, default=0.1, help='Weight for CTCS.')

    # LSTM specific
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--start_type', type=str, default='zeros', choices=['zeros', 'random'], 
                        help='Initial hidden state type.')

    # Training 
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (per GPU).')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=20001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use LR scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[])
    parser.add_argument('--gradient_clipping', type=float, default=-1)
    parser.add_argument('--num_workers_train', type=int, default=1)
    parser.add_argument('--use_custom_sampler', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False)

    # Housekeeping 
    parser.add_argument('--log_dir', type=str, default='logs/scratch')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'RAFDB', 'FerPlusPlus', 'affectnet'])
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=412)
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--strict_reload', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ignore_metrics_when_reloading', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--val_split_ratio', type=float, default=0.1)
    parser.add_argument('--use_test_as_val', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_best_model', action=argparse.BooleanOptionalAction, default=False)

    # Tracking 
    parser.add_argument('--track_every', type=int, default=1000)
    parser.add_argument('--n_test_batches', type=int, default=20)
    parser.add_argument('--eval_per_epoch', action=argparse.BooleanOptionalAction, default=False,
                        help="Evaluate on full validation set after each epoch. Automatically sets n_test_batches=-1 and calculates track_every based on dataset size.")
    parser.add_argument('--plot_indices', type=int, default=[0], nargs='+')
    parser.add_argument('--compute_flops', action=argparse.BooleanOptionalAction, default=False)

    # Precision
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False)

    # Logging
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=False, help='Use Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='continuous-thought-machines-fer', help='W&B project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team name.')

    args = parser.parse_args()
    return args

# --- DDP Setup Functions ---
def setup_ddp():
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['LOCAL_RANK'] = '0'
        print("Running in non-distributed mode (simulated DDP setup).")
        if not torch.cuda.is_available() or int(os.environ['WORLD_SIZE']) == 1:
            dist.init_process_group(backend='gloo')
            print("Initialized process group with Gloo backend.")
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            return rank, world_size, local_rank

    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank} setup on GPU {local_rank}")
    else:
        print(f"Rank {rank} setup on CPU")
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP cleanup complete.")

def is_main_process(rank):
    return rank == 0
# --- End DDP Setup ---


if __name__=='__main__':
    args = parse_args()

    # Parse hne_group_configs and text_prompts
    if args.hne_group_configs is not None:
        args.hne_group_configs = json.loads(args.hne_group_configs)
    elif args.hne_group_configs_file is not None:
        with open(args.hne_group_configs_file, 'r') as f:
            args.hne_group_configs = json.load(f)
    
    if args.text_prompts is not None:
        args.text_prompts = json.loads(args.text_prompts)

    rank, world_size, local_rank = setup_ddp()

    set_seed(args.seed, False)

    # Rank 0 handles directory creation
    if is_main_process(rank):
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        zip_python_code(f'{args.log_dir}/repo_state.zip')
        with open(f'{args.log_dir}/args.txt', 'w') as f:
            print(args, file=f)
        
        # Initialize W&B
        if args.use_wandb:
            run_name = f"{args.model}_{args.dataset}_bs{args.batch_size}_lr{args.lr}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                dir=args.log_dir,
                config=vars(args),
                name=run_name,
                reinit=True,
            )
            wandb.log({"Logging initialized": True})
    if world_size > 1: dist.barrier()

    # Data Loading
    train_data, val_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(
        args.dataset, args.data_root, args.val_split_ratio, args.use_test_as_val,
        args.grayscale, args.convert_grayscale_to_rgb, args.img_size
    )

    # Setup Samplers
    train_sampler = (FastRandomDistributedSampler(train_data, num_replicas=world_size, rank=rank, seed=args.seed, epoch_steps=int(10e10))
                     if args.use_custom_sampler else
                     DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed))
    
    # Use val_data for validation if available, else test_data
    eval_data = val_data if val_data is not None else test_data
    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)

    # Auto-calculate epoch-based validation if requested
    if args.eval_per_epoch and is_main_process(rank):
        iterations_per_epoch = len(train_data) // args.batch_size
        args.track_every = iterations_per_epoch
        args.n_test_batches = -1
        print(f"[INFO] Eval per epoch enabled: track_every={args.track_every} (iterations/epoch), n_test_batches={args.n_test_batches} (full validation)")

    # Setup DataLoaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                                              num_workers=args.num_workers_train, pin_memory=True, drop_last=True)
    evalloader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size_test, sampler=eval_sampler,
                                             num_workers=1, pin_memory=True, drop_last=False)

    prediction_reshaper = [-1]
    args.out_dims = len(class_labels)

    # Setup Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        if world_size > 1:
            warnings.warn("Running DDP on CPU is not recommended.")
    if is_main_process(rank):
        print(f'Main process (Rank {rank}): Using device {device}. World size: {world_size}. Model: {args.model}')

    # --- Model Creation via Factory ---
    model_base, ctm_gated_loss = create_model(args, prediction_reshaper)
    model_base = model_base.to(device)

    # Initialize lazy modules with proper input size
    # Must run multiple times to ensure all lazy modules are initialized
    if args.model == 'clip_ctm':
        try:
            print("Initializing lazy modules for clip_ctm...")
            for _ in range(3):
                pseudo_inputs = torch.randn(2, 3, args.img_size, args.img_size).to(device)
                with torch.no_grad():
                    _ = model_base(pseudo_inputs)
            print("Lazy module initialization complete.")
        except Exception as e:
            print(f"Warning: Pseudo forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    # Wrap model with DDP
    find_unused = args.model in ('ctm_fwpkm', 'clip_ctm')  # These models may have unused parameters
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model_base, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    elif device.type == 'cpu' and world_size > 1:
        model = DDP(model_base, find_unused_parameters=find_unused)
    else:
        model = model_base

    # Auxiliary losses for CTM-Innovations
    psl_loss = None
    ctcs_loss = None
    if args.model == 'ctm_with_innovations':
        if args.use_psl:
            psl_loss = PredictiveSynchronyLoss(synch_size=args.n_synch_out).to(device)
        if args.use_ctcs:
            ctcs_loss = CrossTickContrastiveLoss().to(device)

    if is_main_process(rank):
        param_count = sum(p.numel() for p in model.module.parameters() if p.requires_grad) if world_size > 1 else sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total trainable params: {param_count:,}')
        
        # Debug: print trainable adapter params for clip_ctm
        if args.model == 'clip_ctm':
            model_for_debug = model.module if world_size > 1 else model
            adapter_params = [p.numel() for n, p in model_for_debug.named_parameters() if 'adapter' in n and p.requires_grad]
            alpha_params = [p.numel() for n, p in model_for_debug.named_parameters() if 'alpha' in n and p.requires_grad]
            kv_proj_params = [p.numel() for n, p in model_for_debug.named_parameters() if 'kv_proj' in n and p.requires_grad]
            print(f'  Adapter params: {sum(adapter_params):,}')
            print(f'  Alpha params: {sum(alpha_params):,}')
            print(f'  kv_proj params: {sum(kv_proj_params):,}')
        
        if args.use_wandb:
            wandb.log({"Total Parameters": param_count})

    # Optimizer
    decay_params = []
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(exclusion_str in name for exclusion_str in args.weight_decay_exclusion_list):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)

    # Add auxiliary losses
    aux_params = []
    if psl_loss is not None:
        aux_params.append({"params": psl_loss.parameters(), "weight_decay": args.weight_decay})
    if ctcs_loss is not None:
        aux_params.append({"params": ctcs_loss.parameters(), "weight_decay": args.weight_decay})

    if len(no_decay_names) and is_main_process(rank):
        print(f'WARNING, excluding: {no_decay_names}')

    if len(no_decay_names) and args.weight_decay != 0:
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0}
        ] + aux_params, lr=args.lr, eps=1e-8 if not args.use_amp else 1e-6)
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": args.weight_decay}] + aux_params
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8 if not args.use_amp else 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup(args.warmup_steps).step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, args.warmup_steps, args.milestones, args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)

    # Metrics
    start_iter = 0
    train_losses, test_losses = [], []
    train_accuracies_most_certain, test_accuracies_most_certain = [], []
    iters = []
    best_val_acc = 0.0

    scaler = torch.amp.GradScaler("cuda" if device.type == 'cuda' else "cpu", enabled=args.use_amp)

    # Reloading
    if args.reload:
        map_location = device
        chkpt_path = f'{args.log_dir}/checkpoint.pt'
        if os.path.isfile(chkpt_path):
            print(f'Rank {rank}: Reloading from: {chkpt_path}')
            checkpoint = torch.load(chkpt_path, map_location=map_location, weights_only=False)
            model_to_load = model.module if isinstance(model, DDP) else model
            
            state_dict = checkpoint['model_state_dict']
            has_module_prefix = all(k.startswith('module.') for k in state_dict)
            is_wrapped = isinstance(model, DDP)
            
            if has_module_prefix and not is_wrapped:
                state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
            elif not has_module_prefix and is_wrapped:
                load_result = model_to_load.load_state_dict(state_dict, strict=args.strict_reload)
                print(f" Loaded state_dict. Missing: {load_result.missing_keys}")
                state_dict = None
            
            if state_dict is not None:
                load_result = model_to_load.load_state_dict(state_dict, strict=args.strict_reload)
                print(f" Loaded state_dict. Missing: {load_result.missing_keys}")

            if not args.reload_model_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_iter = checkpoint['iteration']
                if is_main_process(rank) and not args.ignore_metrics_when_reloading:
                    iters = checkpoint['iters']
                    train_losses = checkpoint['train_losses']
                    test_losses = checkpoint['test_losses']
                    if 'train_accuracies_most_certain' in checkpoint:
                        train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
                        test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                    if 'best_val_acc' in checkpoint:
                        best_val_acc = checkpoint['best_val_acc']
            
            del checkpoint
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"Rank {rank}: Reload finished, starting from iteration {start_iter}")
        if world_size > 1: dist.barrier()

    # Compilation
    if args.do_compile:
        if is_main_process(rank): print('Compiling model components...')
        model_to_compile = model.module if isinstance(model, DDP) else model
        if hasattr(model_to_compile, 'backbone'):
            model_to_compile.backbone = torch.compile(model_to_compile.backbone, mode='reduce-overhead', fullgraph=True)
        if args.model in ['ctm', 'ctm_gated', 'ctm_with_innovations']:
            if hasattr(model_to_compile, 'synapses'):
                model_to_compile.synapses = torch.compile(model_to_compile.synapses, mode='reduce-overhead', fullgraph=True)
        if world_size > 1: dist.barrier()
        if is_main_process(rank): print('Compilation finished.')

    # --- Training Loop ---
    model.train()
    pbar = tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True, disable=not is_main_process(rank))
    iterator = iter(trainloader)

    for bi in range(start_iter, args.training_iterations):
        if not args.use_custom_sampler and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(bi)

        current_lr = optimizer.param_groups[-1]['lr']

        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(trainloader)
            inputs, targets = next(iterator)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        loss = None
        with torch.autocast(device_type="cuda" if device.type == 'cuda' else "cpu", dtype=torch.float16, enabled=args.use_amp):
            if args.do_compile:
                torch.compiler.cudagraph_mark_step_begin()

            if args.model == 'ctm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'ctm_gated':
                predictions, certainties, synchronisation, exit_steps, _ = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'ctm_with_innovations':
                predictions, certainties, synchronisation, synch_out_seq = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                aux_loss = 0
                if psl_loss is not None and synch_out_seq is not None:
                    aux_loss = aux_loss + args.lambda_psl * psl_loss(synch_out_seq)
                if ctcs_loss is not None and synch_out_seq is not None:
                    aux_loss = aux_loss + args.lambda_ctcs * ctcs_loss(synch_out_seq)
                loss = loss + aux_loss
            elif args.model == 'lstm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'clip_ctm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model in ['clip_ctm_adapter_a', 'clip_ctm_adapter_b', 'clip_ctm_adapter_c', 'clip_ctm_adapter_d', 'clip_ctm_adapter_e', 'clip_ctm_adapter_f']:
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'ctm_fwpkm':
                predictions, certainties, synchronisation = model(inputs)
                loss, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
            elif args.model == 'ff':
                predictions = model(inputs)
                loss = nn.CrossEntropyLoss()(predictions, targets)
                where_most_certain = None
        
        scaler.scale(loss).backward()

        if args.gradient_clipping != -1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clipping)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Aggregation
        loss_log = loss.detach()
        if world_size > 1: dist.all_reduce(loss_log, op=dist.ReduceOp.AVG)

        if is_main_process(rank):
            accuracy_local = 0.0
            if args.model in ['ctm', 'ctm_gated', 'ctm_with_innovations', 'clip_ctm', 'clip_ctm_adapter_a', 'clip_ctm_adapter_b', 'clip_ctm_adapter_c', 'clip_ctm_adapter_d', 'clip_ctm_adapter_e', 'clip_ctm_adapter_f', 'ctm_fwpkm', 'lstm']:
                accuracy_local = (predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain] == targets).float().mean().item()
                pbar_desc = f'Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_local:.3f} LR={current_lr:.6f}'
                
                # Log metrics for clip_ctm
                if args.model == 'clip_ctm':
                    if bi % 500 == 0:
                        model_for_log = model.module if world_size > 1 else model
                        if hasattr(model_for_log, 'backbone_adapter') and hasattr(model_for_log.backbone_adapter, 'alpha'):
                            alpha_val = model_for_log.backbone_adapter.alpha.data.item()
                            print(f"  [iter {bi}] alpha = {alpha_val:.4f}, loss = {loss_log.item():.4f}")
                            if args.use_wandb:
                                wandb.log({"alpha": alpha_val, "iteration": bi})
                    if args.use_wandb and bi % 100 == 0:
                        wandb.log({
                            "Train Loss": loss_log.item(),
                            "Train Accuracy": accuracy_local,
                            "Learning Rate": current_lr,
                            "Iteration": bi,
                        })
            elif args.model == 'ff':
                accuracy_local = (predictions.argmax(1) == targets).float().mean().item()
                pbar_desc = f'Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_local:.3f} LR={current_lr:.6f}'
            pbar.set_description(f'{args.model.upper()} {pbar_desc}')

        # Evaluation
        if bi % args.track_every == 0 and (bi != 0 or args.reload_model_only):
            model.eval()
            with torch.inference_mode():
                iters.append(bi)
                
                total_eval_loss = torch.tensor(0.0, device=device)
                total_eval_correct = torch.tensor(0.0, device=device)
                total_eval_samples = torch.tensor(0.0, device=device)

                pbar_inner_desc = 'Eval (Rank 0)' if is_main_process(rank) else None
                with tqdm(total=len(evalloader), desc=pbar_inner_desc, leave=False, position=1, dynamic_ncols=True, disable=not is_main_process(rank)) as pbar_inner:
                    for inferi, (inputs, targets) in enumerate(evalloader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)

                        if args.model == 'ctm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'ctm_gated':
                            predictions, certainties, _, exit_steps, _ = model(inputs, use_early_exit=False)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'ctm_with_innovations':
                            predictions, certainties, _, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'lstm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'clip_ctm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model in ['clip_ctm_adapter_a', 'clip_ctm_adapter_b', 'clip_ctm_adapter_c', 'clip_ctm_adapter_d', 'clip_ctm_adapter_e', 'clip_ctm_adapter_f']:
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'ctm_fwpkm':
                            predictions, certainties, _ = model(inputs)
                            loss_eval, where_most_certain = image_classification_loss(predictions, certainties, targets, use_most_certain=True)
                            preds_eval = predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain]
                        elif args.model == 'ff':
                            predictions = model(inputs)
                            loss_eval = nn.CrossEntropyLoss()(predictions, targets)
                            preds_eval = predictions.argmax(1)

                        total_eval_loss += loss_eval * inputs.size(0)
                        total_eval_correct += (preds_eval == targets).sum()
                        total_eval_samples += inputs.size(0)

                        if args.n_test_batches != -1 and inferi >= args.n_test_batches - 1: break
                        pbar_inner.update(1)

                if world_size > 1:
                    dist.all_reduce(total_eval_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_eval_correct, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_eval_samples, op=dist.ReduceOp.SUM)

                if is_main_process(rank) and total_eval_samples > 0:
                    avg_eval_loss = total_eval_loss.item() / total_eval_samples.item()
                    eval_acc = total_eval_correct.item() / total_eval_samples.item()
                    test_losses.append(avg_eval_loss)
                    test_accuracies_most_certain.append(eval_acc)
                    
                    print(f"Iter {bi} Eval: Loss={avg_eval_loss:.4f}, Acc={eval_acc:.4f}")
                    
                    # Log to W&B
                    if args.use_wandb:
                        wandb.log({
                            "Eval Loss": avg_eval_loss,
                            "Eval Accuracy": eval_acc,
                            "Iteration": bi,
                        })

                    # Save best model
                    if eval_acc > best_val_acc:
                        best_val_acc = eval_acc
                        if args.save_best_model:
                            save_path = f'{args.log_dir}/best_model.pt'
                            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                            torch.save({'model_state_dict': model_state_to_save, 'args': args}, save_path)
                            print(f"Best model saved: {save_path} (acc={eval_acc:.4f})")
                        if args.use_wandb:
                            wandb.run.summary["best_val_acc"] = best_val_acc

                    # Plotting
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111)
                    ax.plot(iters, test_accuracies_most_certain, 'k-', alpha=0.9, label=f'Acc ({test_accuracies_most_certain[-1]:.3f})')
                    ax.set_title('Eval Accuracy')
                    ax.legend()
                    ax.set_xlim([0, args.training_iterations])
                    fig.tight_layout()
                    fig.savefig(f'{args.log_dir}/eval_accuracies.png', dpi=150)
                    plt.close(fig)

            if world_size > 1: dist.barrier()
            model.train()

        # Checkpointing
        if bi % args.save_every == 0 and bi != start_iter and is_main_process(rank):
            save_path = f'{args.log_dir}/checkpoint.pt'
            model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            save_dict = {
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'iteration': bi,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'iters': iters,
                'args': args,
                'train_accuracies_most_certain': train_accuracies_most_certain,
                'test_accuracies_most_certain': test_accuracies_most_certain,
                'best_val_acc': best_val_acc,
            }
            torch.save(save_dict, save_path)
            pbar.set_description(f"Checkpoint saved to {save_path}")

        if world_size > 1: dist.barrier()
        if is_main_process(rank):
            pbar.update(1)

    if is_main_process(rank):
        pbar.close()
        if args.use_wandb:
            wandb.finish()

    cleanup_ddp()
