"""
GammaFold CLI - Command Line Interface.

Commands:
- download: Download training data
- prepare: Filter and prepare data
- train: Train models
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='gammafold',
        description='GammaFold: Protein Structure Prediction'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download training data')
    download_parser.add_argument('--source', choices=['uniprot', 'pdb', 'alphafold'], 
                                 default='uniprot', help='Data source')
    download_parser.add_argument('--organism', default='human', help='Organism name')
    download_parser.add_argument('--output', '-o', default='data/raw', help='Output directory')
    download_parser.add_argument('--resolution', type=float, default=2.5, 
                                 help='Max PDB resolution (Angstroms)')
    download_parser.add_argument('--limit', type=int, default=1000, 
                                 help='Maximum number of structures')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Filter and prepare data')
    prepare_parser.add_argument('--input', '-i', required=True, help='Input data directory')
    prepare_parser.add_argument('--output', '-o', default='data/processed', help='Output directory')
    prepare_parser.add_argument('--min-length', type=int, default=50, help='Minimum sequence length')
    prepare_parser.add_argument('--max-length', type=int, default=1024, help='Maximum sequence length')
    prepare_parser.add_argument('--max-resolution', type=float, default=3.0, 
                                help='Maximum PDB resolution')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Create train/val/test splits')
    split_parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    split_parser.add_argument('--output', '-o', default='data/splits', help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.9, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.05, help='Validation ratio')
    split_parser.add_argument('--test', type=float, default=0.05, help='Test ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    split_parser.add_argument('--cluster-based', action='store_true', 
                              help='Use cluster-based splitting')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', '-c', help='Path to config YAML file')
    train_parser.add_argument('--model', choices=['original', 'multimodal'], 
                              default='original', help='Model type')
    train_parser.add_argument('--size', choices=['small', 'medium', 'large'], 
                              default='small', help='Model size')
    train_parser.add_argument('--data', '-d', default='data/train', help='Training data path')
    train_parser.add_argument('--output', '-o', default='outputs', help='Output directory')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--steps', type=int, default=100000, help='Training steps')
    train_parser.add_argument('--resume', help='Resume from checkpoint')
    train_parser.add_argument('--smoke-test', action='store_true', 
                              help='Run smoke test (10 steps)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'download':
        from gammafold.cli.download import run_download
        return run_download(args)
    
    elif args.command == 'prepare':
        from gammafold.cli.prepare import run_prepare
        return run_prepare(args)
    
    elif args.command == 'split':
        from gammafold.cli.prepare import run_split
        return run_split(args)
    
    elif args.command == 'train':
        from gammafold.cli.train import run_train
        return run_train(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
