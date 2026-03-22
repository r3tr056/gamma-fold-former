"""
Download CLI for GammaFold.

Usage:
    gammafold download --source uniprot --organism human --output data/raw
    gammafold download --source pdb --resolution 2.5 --limit 1000
"""

import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_download(args: argparse.Namespace) -> int:
    """
    Run download command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    from gammafold.data.download import (
        DownloadConfig,
        UniProtDownloader,
        PDBDownloader,
        AlphaFoldDownloader
    )
    
    config = DownloadConfig(output_dir=args.output)
    
    try:
        if args.source == 'uniprot':
            logger.info(f"Downloading from UniProt: organism={args.organism}")
            downloader = UniProtDownloader(config)
            path = downloader.download_by_organism(
                organism=args.organism,
                reviewed_only=True
            )
            logger.info(f"Downloaded: {path}")
            
        elif args.source == 'pdb':
            logger.info(f"Downloading from PDB: resolution<={args.resolution}Å, limit={args.limit}")
            downloader = PDBDownloader(config)
            paths = downloader.download_by_resolution(
                max_resolution=args.resolution,
                limit=args.limit
            )
            logger.info(f"Downloaded {len(paths)} PDB files")
            
        elif args.source == 'alphafold':
            logger.info(f"Downloading from AlphaFold: organism={args.organism}")
            downloader = AlphaFoldDownloader(config)
            path = downloader.download_proteome(organism=args.organism)
            logger.info(f"Downloaded: {path}")
        
        logger.info("Download complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Download protein data')
    parser.add_argument('--source', choices=['uniprot', 'pdb', 'alphafold'], 
                        default='uniprot', help='Data source')
    parser.add_argument('--organism', default='human', help='Organism name')
    parser.add_argument('--output', '-o', default='data/raw', help='Output directory')
    parser.add_argument('--resolution', type=float, default=2.5, 
                        help='Max PDB resolution')
    parser.add_argument('--limit', type=int, default=1000, 
                        help='Maximum structures')
    
    args = parser.parse_args()
    return run_download(args)


if __name__ == '__main__':
    exit(main())
