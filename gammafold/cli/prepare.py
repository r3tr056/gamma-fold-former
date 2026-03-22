"""
Data preparation CLI for GammaFold.

Usage:
    gammafold prepare --input data/raw --output data/processed
    gammafold split --input data/processed/sequences.fasta --output data/splits
"""

import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_prepare(args: argparse.Namespace) -> int:
    """
    Run prepare command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    import os
    from gammafold.data.filtering import FilterConfig, prepare_dataset
    
    try:
        config = FilterConfig(
            min_length=args.min_length,
            max_length=args.max_length,
            max_pdb_resolution=args.max_resolution
        )
        
        # Find input files
        sequence_files = []
        pdb_dir = None
        
        for item in os.listdir(args.input):
            path = os.path.join(args.input, item)
            if item.endswith('.fasta') or item.endswith('.fa'):
                sequence_files.append(path)
            elif os.path.isdir(path) and any(f.endswith('.pdb') for f in os.listdir(path)):
                pdb_dir = path
        
        if not sequence_files:
            # Check for FASTA files in subdirectories
            for root, dirs, files in os.walk(args.input):
                for f in files:
                    if f.endswith('.fasta') or f.endswith('.fa'):
                        sequence_files.append(os.path.join(root, f))
        
        logger.info(f"Found {len(sequence_files)} FASTA files")
        if pdb_dir:
            logger.info(f"Found PDB directory: {pdb_dir}")
        
        # Run preparation
        stats = prepare_dataset(
            sequence_files=sequence_files,
            pdb_dir=pdb_dir,
            output_dir=args.output,
            config=config
        )
        
        logger.info(f"Preparation complete: {stats}")
        return 0
        
    except Exception as e:
        logger.error(f"Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_split(args: argparse.Namespace) -> int:
    """
    Run split command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from gammafold.data.splits import SplitConfig, split_fasta
    
    try:
        config = SplitConfig(
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            cluster_based=args.cluster_based
        )
        
        logger.info(f"Splitting {args.input}")
        logger.info(f"Ratios: train={args.train}, val={args.val}, test={args.test}")
        
        split_info = split_fasta(
            input_path=args.input,
            output_dir=args.output,
            config=config
        )
        
        logger.info(f"Split complete:")
        logger.info(f"  Train: {len(split_info.train_indices)} sequences")
        logger.info(f"  Val: {len(split_info.val_indices)} sequences")
        logger.info(f"  Test: {len(split_info.test_indices)} sequences")
        return 0
        
    except Exception as e:
        logger.error(f"Split failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Prepare protein data')
    subparsers = parser.add_subparsers(dest='command')
    
    # Prepare
    prep = subparsers.add_parser('prepare', help='Filter and prepare data')
    prep.add_argument('--input', '-i', required=True, help='Input directory')
    prep.add_argument('--output', '-o', default='data/processed', help='Output directory')
    prep.add_argument('--min-length', type=int, default=50)
    prep.add_argument('--max-length', type=int, default=1024)
    prep.add_argument('--max-resolution', type=float, default=3.0)
    
    # Split
    split = subparsers.add_parser('split', help='Create splits')
    split.add_argument('--input', '-i', required=True)
    split.add_argument('--output', '-o', default='data/splits')
    split.add_argument('--train', type=float, default=0.9)
    split.add_argument('--val', type=float, default=0.05)
    split.add_argument('--test', type=float, default=0.05)
    split.add_argument('--seed', type=int, default=42)
    split.add_argument('--cluster-based', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        return run_prepare(args)
    elif args.command == 'split':
        return run_split(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit(main())
