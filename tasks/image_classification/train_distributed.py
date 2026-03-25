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

import hydra
from omegaconf import DictConfig, OmegaConf

from tasks.image_classification.train import get_dataset
from models.factories import create_model
from models.ctm_with_innovations import PredictiveSynchronyLoss, CrossTickContrastiveLoss

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


def flatten_config(cfg: DictConfig, prefix: str = '') -> dict:
    """Flatten nested config into flat dict with underscore-separated keys."""
    result = {}
    for key, value in cfg.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, DictConfig):
            result.update(flatten_config(value, new_key))
        else:
            result[new_key] = value
    return result

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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    args = flatten_config(cfg)
    args = type('Args', (), args)()
    
    if hasattr(cfg.model, 'innovations') and cfg.model.innovations.get('hne_group_configs_file'):
        with open(cfg.model.innovations.hne_group_configs_file, 'r') as f:
            args.hne_group_configs = json.load(f)
    elif hasattr(cfg.model, 'innovations') and cfg.model.innovations.get('hne_group_configs'):
        args.hne_group_configs = cfg.model.innovations.hne_group_configs
    else:
        args.hne_group_configs = None

    rank, world_size, local_rank = setup_ddp()

    set_seed(args.seed, False)

    if is_main_process(rank):
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        zip_python_code(f'{args.log_dir}/repo_state.zip')
        with open(f'{args.log_dir}/args.txt', 'w') as f:
            print(OmegaConf.to_yaml(cfg), file=f)
        
        if args.use_wandb:
            run_name = f"{args.model}_{args.dataset}_bs{args.batch_size}_lr{args.lr}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                dir=args.log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=run_name,
                reinit=True,
            )
            wandb.log({"Logging initialized": True})
    if world_size > 1: dist.barrier()

    train_data, val_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(
        args.dataset, args.data_root, args.val_split_ratio, args.use_test_as_val,
        args.grayscale, args.convert_grayscale_to_rgb, args.img_size
    )

    train_sampler = (FastRandomDistributedSampler(train_data, num_replicas=world_size, rank=rank, seed=args.seed, epoch_steps=int(10e10))
                     if args.use_custom_sampler else
                     DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed))
    
    eval_data = val_data if val_data is not None else test_data
    eval_sampler = DistributedSampler(eval_data, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)

    if args.eval_per_epoch and is_main_process(rank):
        iterations_per_epoch = len(train_data) // args.batch_size
        args.track_every = iterations_per_epoch
        args.n_test_batches = -1
        print(f"[INFO] Eval per epoch enabled: track_every={args.track_every} (iterations/epoch), n_test_batches={args.n_test_batches} (full validation)")

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

    # Initialize lazy modules
    try:
        pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
        model_base(pseudo_inputs)
    except Exception as e:
        print(f"Warning: Pseudo forward pass failed: {e}")

    # Wrap model with DDP
    if device.type == 'cuda' and world_size > 1:
        model = DDP(model_base, device_ids=[local_rank], output_device=local_rank)
    elif device.type == 'cpu' and world_size > 1:
        model = DDP(model_base)
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
            if args.model in ['ctm', 'ctm_gated', 'ctm_with_innovations', 'lstm']:
                accuracy_local = (predictions.argmax(1)[torch.arange(predictions.size(0), device=device), where_most_certain] == targets).float().mean().item()
                pbar_desc = f'Loss(avg)={loss_log.item():.3f} Acc(loc)={accuracy_local:.3f} LR={current_lr:.6f}'
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
