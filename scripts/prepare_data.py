#!/usr/bin/env python3
"""
GammaFold Data Preparation Script

Downloads protein sequences from UniProt, filters them, and creates train/val/test splits.
Run this script to prepare data for training.

Usage:
    python scripts/prepare_data.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Organisms to download (diverse set for protein language modeling)
ORGANISMS = [
    # Mammals
    "human",
    "mouse", 
    "rat",
    "bovine",
    "pig",
    # Model organisms
    "yeast",
    "ecoli",
    "drosophila",
    "zebrafish",
    "arabidopsis",
    # Additional bacteria
    "bacillus",
    "pseudomonas",
    # Worms
    "celegans",
]

# Filtering parameters
MIN_LENGTH = 50
MAX_LENGTH = 512


def run_cmd(cmd, desc=None):
    """Run a command and print status."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        return False
    return True


def download_all():
    """Download data from all organisms."""
    print("\n" + "="*60)
    print("  STEP 1: DOWNLOADING DATA FROM UNIPROT")
    print("="*60)
    
    for organism in ORGANISMS:
        output_dir = RAW_DIR / organism
        if (output_dir / f"{organism}_swissprot.fasta").exists():
            print(f"\n[SKIP] {organism} already downloaded")
            continue
        
        cmd = f"gammafold download --source uniprot --organism {organism} --output {output_dir}"
        if not run_cmd(cmd, f"Downloading {organism}"):
            return False
    
    return True


def count_sequences(fasta_path):
    """Count sequences in a FASTA file."""
    count = 0
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def combine_fasta_files():
    """Combine all downloaded FASTA files into one."""
    print("\n" + "="*60)
    print("  STEP 2: COMBINING FASTA FILES")
    print("="*60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = PROCESSED_DIR / "all_sequences.fasta"
    
    total_seqs = 0
    with open(combined_path, 'w') as out_file:
        for organism in ORGANISMS:
            fasta_path = RAW_DIR / organism / f"{organism}_swissprot.fasta"
            if not fasta_path.exists():
                print(f"[WARN] {fasta_path} not found, skipping")
                continue
            
            count = count_sequences(fasta_path)
            total_seqs += count
            print(f"  Adding {organism}: {count:,} sequences")
            
            with open(fasta_path, 'r') as in_file:
                out_file.write(in_file.read())
    
    print(f"\n  Total: {total_seqs:,} sequences")
    print(f"  Output: {combined_path}")
    return combined_path


def filter_sequences(input_path):
    """Filter sequences by length."""
    print("\n" + "="*60)
    print("  STEP 3: FILTERING SEQUENCES")
    print("="*60)
    
    output_path = PROCESSED_DIR / "filtered_sequences.fasta"
    
    kept = 0
    skipped_short = 0
    skipped_long = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        header = None
        sequence = ""
        
        for line in infile:
            if line.startswith('>'):
                # Process previous sequence
                if header and sequence:
                    seq_len = len(sequence)
                    if seq_len < MIN_LENGTH:
                        skipped_short += 1
                    elif seq_len > MAX_LENGTH:
                        skipped_long += 1
                    else:
                        outfile.write(header)
                        outfile.write(sequence + "\n")
                        kept += 1
                
                header = line
                sequence = ""
            else:
                sequence += line.strip()
        
        # Process last sequence
        if header and sequence:
            seq_len = len(sequence)
            if MIN_LENGTH <= seq_len <= MAX_LENGTH:
                outfile.write(header)
                outfile.write(sequence + "\n")
                kept += 1
    
    print(f"  Kept: {kept:,} sequences")
    print(f"  Skipped (too short): {skipped_short:,}")
    print(f"  Skipped (too long): {skipped_long:,}")
    print(f"  Output: {output_path}")
    
    return output_path


def create_splits(input_path):
    """Create train/val/test splits."""
    print("\n" + "="*60)
    print("  STEP 4: CREATING TRAIN/VAL/TEST SPLITS")
    print("="*60)
    
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = f"gammafold split --input {input_path} --output {SPLITS_DIR} --train 0.9 --val 0.05 --test 0.05 --seed 42"
    if not run_cmd(cmd):
        return False
    
    # Verify splits
    for split in ["train", "val", "test"]:
        split_path = SPLITS_DIR / f"{split}.fasta"
        if split_path.exists():
            count = count_sequences(split_path)
            print(f"  {split}: {count:,} sequences")
    
    return True


def verify_data():
    """Verify the prepared data."""
    print("\n" + "="*60)
    print("  VERIFICATION")
    print("="*60)
    
    train_path = SPLITS_DIR / "train.fasta"
    val_path = SPLITS_DIR / "val.fasta"
    
    if not train_path.exists():
        print("  ERROR: train.fasta not found")
        return False
    
    if not val_path.exists():
        print("  ERROR: val.fasta not found")
        return False
    
    train_count = count_sequences(train_path)
    val_count = count_sequences(val_path)
    
    print(f"  Train sequences: {train_count:,}")
    print(f"  Val sequences: {val_count:,}")
    
    if train_count < 1000:
        print("  WARNING: Training set is small (<1000)")
    
    print("\n  ✓ Data preparation complete!")
    print(f"\n  Ready to train with:")
    print(f"    gammafold train --data {SPLITS_DIR}/train.fasta --size small")
    
    return True


def main():
    print("\n" + "#"*60)
    print("  GAMMAFOLD DATA PREPARATION")
    print("#"*60)
    
    # Step 1: Download
    if not download_all():
        print("Download failed!")
        sys.exit(1)
    
    # Step 2: Combine
    combined_path = combine_fasta_files()
    
    # Step 3: Filter
    filtered_path = filter_sequences(combined_path)
    
    # Step 4: Split
    if not create_splits(filtered_path):
        print("Split failed!")
        sys.exit(1)
    
    # Verify
    if not verify_data():
        print("Verification failed!")
        sys.exit(1)
    
    print("\n" + "#"*60)
    print("  DATA PREPARATION COMPLETE!")
    print("#"*60)
    print()


if __name__ == "__main__":
    main()
