#!/usr/bin/env python3
"""
Script to load training results from Weights & Biases and save to TSV file.

Usage:
    python scripts/load_wandb_to_tsv.py --run_path "user/project/run_name" --output results/run_name.tsv
    python scripts/load_wandb_to_tsv.py --run_id "abc123" --project "continuous-thought-machines-fer" --output results/run.tsv
    python scripts/load_wandb_to_tsv.py --all --project "continuous-thought-machines-fer" --output results/all_runs.tsv
"""

import argparse
import os
import pandas as pd
import wandb
from datetime import datetime


def load_run_history(run_path: str = None, run_id: str = None, project: str = None):
    """Load history from a single W&B run."""
    if run_path:
        api = wandb.Api()
        run = api.run(run_path)
    elif run_id and project:
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
    else:
        raise ValueError("Must provide either --run_path or --run_id and --project")
    
    # Get history data
    history = run.history()
    
    # Get summary metrics
    summary = run.summary
    
    # Get config
    config = run.config
    
    return history, summary, config, run.name, run.id


def load_all_runs(project: str, entity: str = None):
    """Load history from all runs in a project."""
    api = wandb.Api()
    
    if entity:
        runs = api.runs(f"{entity}/{project}")
    else:
        runs = api.runs(project)
    
    all_history = []
    run_info = []
    
    for run in runs:
        try:
            history = run.history()
            if len(history) > 0:
                history['_run_name'] = run.name
                history['_run_id'] = run.id
                history['_run_state'] = run.state
                all_history.append(history)
                run_info.append({
                    'name': run.name,
                    'id': run.id,
                    'state': run.state,
                    'created_at': run.created_at,
                    'config': run.config
                })
        except Exception as e:
            print(f"Warning: Could not load run {run.id}: {e}")
    
    return all_history, run_info


def save_to_tsv(df: pd.DataFrame, output_path: str):
    """Save DataFrame to TSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load W&B results to TSV")
    parser.add_argument('--run_path', type=str, help='Full W&B run path (entity/project/run_id)')
    parser.add_argument('--run_id', type=str, help='W&B run ID')
    parser.add_argument('--project', type=str, help='W&B project name')
    parser.add_argument('--entity', type=str, help='W&B entity/username (optional)')
    parser.add_argument('--output', type=str, default='results/wandb_export.tsv', help='Output TSV path')
    parser.add_argument('--all', action='store_true', help='Load all runs from project')
    parser.add_argument('--include_config', action='store_true', help='Include config as columns')
    
    args = parser.parse_args()
    
    if args.all:
        if not args.project:
            raise ValueError("--project required when using --all")
        
        print(f"Loading all runs from project: {args.project}")
        all_history, run_info = load_all_runs(args.project, args.entity)
        
        if all_history:
            # Concatenate all histories
            combined_df = pd.concat(all_history, ignore_index=True)
            
            # Add run info as columns
            run_info_df = pd.DataFrame(run_info)
            
            # Save combined history
            save_to_tsv(combined_df, args.output)
            
            # Also save run summary info
            summary_path = args.output.replace('.tsv', '_runs.tsv')
            save_to_tsv(run_info_df, summary_path)
            
            print(f"Loaded {len(run_info)} runs with {len(combined_df)} total rows")
        else:
            print("No runs found")
    
    else:
        # Load single run
        history, summary, config, run_name, run_id = load_run_history(
            args.run_path, args.run_id, args.project
        )
        
        print(f"Loaded run: {run_name} (ID: {run_id})")
        print(f"Total rows: {len(history)}")
        
        # Add metadata columns
        history['_run_name'] = run_name
        history['_run_id'] = run_id
        
        if args.include_config:
            # Flatten config and add as columns
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    history[f'config_{key}'] = value
        
        save_to_tsv(history, args.output)
        
        # Save summary as separate file
        summary_path = args.output.replace('.tsv', '_summary.tsv')
        summary_df = pd.DataFrame([summary])
        summary_df['_run_name'] = run_name
        summary_df['_run_id'] = run_id
        save_to_tsv(summary_df, summary_path)
        
        print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
