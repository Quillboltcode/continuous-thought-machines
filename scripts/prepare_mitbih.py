#!/usr/bin/env python3
"""
MIT-BIH Arrhythmia Dataset Downloader and Preprocessor

This script downloads and prepares the MIT-BIH Arrhythmia Database for training.
The dataset can be used in two modes:
1. Image mode: ECG signals converted to images for use with standard vision models
2. Raw mode: Direct WFDB records for sequence models

Usage:
    python scripts/prepare_mitbih.py --data_root data/mitbih --download
    
    python scripts/prepare_mitbih.py --data_root data/mitbih --convert_to_images --img_size 224
"""

import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

CLASS_MAPPING = {
    'N': 'Normal',
    'L': 'LeftBundleBranchBlock',
    'R': 'RightBundleBranchBlock',
    'A': 'AtrialPremature',
    'V': 'PrematureVentricular',
    'F': 'Fusion',
    'f': 'FusionPaced',
    'Q': 'Unclassifiable',
}


def download_physionet(output_dir):
    """Download MIT-BIH database from PhysioNet."""
    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import wfdb
        print("Using wfdb to download records...")
        
        for record in tqdm(MITBIH_RECORDS, desc="Downloading records"):
            try:
                wfdb.rdsamp(record, pbdir='mitdb', pndir='physionet.org/files/mitdb/1.0.0',
                           download=True, return_res=16)
            except Exception as e:
                print(f"Warning: Could not download {record}: {e}")
        
        print(f"Downloaded to {output_dir}")
        return True
    except ImportError:
        print("wfdb not installed. Installing...")
        subprocess.run(['pip', 'install', 'wfdb'], check=True)
        
        for record in tqdm(MITBIH_RECORDS, desc="Downloading records"):
            try:
                import wfdb
                wfdb.rdsamp(record, pbdir='mitdb', pndir='physionet.org/files/mitdb/1.0.0',
                           download=True, return_res=16)
            except Exception as e:
                print(f"Warning: Could not download {record}: {e}")
        
        return True


def convert_to_images(data_dir, output_dir, img_size=224):
    """Convert ECG signals to images."""
    print(f"Converting ECG signals to images (size: {img_size}x{img_size})...")
    
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    import wfdb
    from PIL import Image
    
    all_samples = []
    
    for record in tqdm(MITBIH_RECORDS, desc="Processing records"):
        try:
            signals, fields = wfdb.rdsamp(record, pbdir='mitdb', pndir='physionet.org/files/mitdb/1.0.0')
            annotation = wfdb.rdann(record, 'atr', pbdir='mitdb', pndir='physionet.org/files/mitdb/1.0.0')
            
            signal = signals[:, 0]
            
            for i, symbol in enumerate(annotation.symbol):
                if symbol in CLASS_MAPPING:
                    sample_idx = annotation.sample[i]
                    
                    start = max(0, sample_idx - img_size)
                    end = min(len(signal), sample_idx + img_size)
                    segment = signal[start:end]
                    
                    if len(segment) < img_size:
                        segment = np.pad(segment, (0, img_size - len(segment)), mode='constant')
                    elif len(segment) > img_size:
                        segment = segment[:img_size]
                    
                    segment = ((segment - segment.min()) / (segment.max() - segment.min() + 1e-8) * 255).astype(np.uint8)
                    
                    img = Image.fromarray(segment, mode='L')
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    
                    class_name = CLASS_MAPPING[symbol]
                    all_samples.append((img, class_name, record, i))
                    
        except Exception as e:
            print(f"Warning: Could not process {record}: {e}")
            continue
    
    print(f"Total samples: {len(all_samples)}")
    
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    train_size = int(0.7 * len(all_samples))
    val_size = int(0.15 * len(all_samples))
    
    splits = {
        'train': all_samples[:train_size],
        'val': all_samples[train_size:train_size + val_size],
        'test': all_samples[train_size + val_size:]
    }
    
    for split, samples in splits.items():
        for img, class_name, record, idx in tqdm(samples, desc=f"Saving {split}"):
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            filename = f"{record}_{idx}.png"
            img.save(os.path.join(class_dir, filename))
    
    print(f"Converted dataset saved to {output_dir}")
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        for class_name in CLASS_MAPPING.values():
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                count = len(os.listdir(class_path))
                print(f"  {split}/{class_name}: {count} samples")


def create_class_summary(output_dir):
    """Print summary of class distribution."""
    print("\n=== Dataset Class Distribution ===")
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        split_path = os.path.join(output_dir, split)
        
        if not os.path.exists(split_path):
            continue
            
        total = 0
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                total += count
                print(f"  {class_name}: {count}")
        
        print(f"  Total: {total}")


def main():
    parser = argparse.ArgumentParser(description='Prepare MIT-BIH Arrhythmia Dataset')
    parser.add_argument('--data_root', type=str, default='data/mitbih',
                       help='Root directory for dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download dataset from PhysioNet')
    parser.add_argument('--convert_to_images', action='store_true',
                       help='Convert raw signals to images')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for converted images')
    
    args = parser.parse_args()
    
    if args.download:
        download_physionet(args.data_root)
    
    if args.convert_to_images:
        convert_to_images(args.data_root, args.data_root + '_images', args.img_size)
        create_class_summary(args.data_root + '_images')


if __name__ == '__main__':
    main()