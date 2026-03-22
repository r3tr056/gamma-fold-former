"""
Data filtering utilities for GammaFold.

Filters and cleans protein data:
- Sequence length filtering
- Non-standard amino acid removal
- PDB resolution filtering
- Redundancy removal via clustering
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Set, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

from gammafold.data.parsers.fasta import parse_fasta, write_fasta, FastaRecord
from gammafold.data.parsers.pdb import parse_pdb
from gammafold.data.vocabulary import VOCABULARY

logger = logging.getLogger(__name__)


# Standard amino acids
STANDARD_AA = set(VOCABULARY.AMINO_ACIDS.keys())


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    min_length: int = 50
    max_length: int = 1024
    max_pdb_resolution: float = 3.0
    remove_non_standard: bool = True
    remove_fragments: bool = True
    max_x_fraction: float = 0.1  # Max fraction of unknown residues


def filter_sequence(
    sequence: str,
    config: Optional[FilterConfig] = None
) -> Tuple[bool, str]:
    """
    Filter a single sequence.
    
    Args:
        sequence: Amino acid sequence
        config: Filter configuration
        
    Returns:
        Tuple of (passed, reason)
    """
    config = config or FilterConfig()
    sequence = sequence.upper().strip()
    
    # Length filter
    if len(sequence) < config.min_length:
        return False, f"too_short ({len(sequence)} < {config.min_length})"
    
    if len(sequence) > config.max_length:
        return False, f"too_long ({len(sequence)} > {config.max_length})"
    
    # Non-standard amino acid check
    if config.remove_non_standard:
        non_standard = set(sequence) - STANDARD_AA - {'X', 'U', 'O'}
        if non_standard:
            return False, f"non_standard_aa ({non_standard})"
    
    # Unknown residue fraction
    x_count = sequence.count('X')
    if x_count / len(sequence) > config.max_x_fraction:
        return False, f"too_many_unknowns ({x_count / len(sequence):.2%})"
    
    return True, "passed"


def filter_fasta(
    input_path: str,
    output_path: str,
    config: Optional[FilterConfig] = None,
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    Filter a FASTA file.
    
    Args:
        input_path: Input FASTA file
        output_path: Output FASTA file
        config: Filter configuration
        progress_callback: Optional progress callback
        
    Returns:
        Statistics dictionary
    """
    config = config or FilterConfig()
    records = parse_fasta(input_path)
    
    passed = []
    stats = defaultdict(int)
    stats['total'] = len(records)
    
    for i, record in enumerate(records):
        ok, reason = filter_sequence(record.sequence, config)
        stats[reason] += 1
        
        if ok:
            passed.append(record)
        
        if progress_callback and (i + 1) % 1000 == 0:
            progress_callback(i + 1, len(records))
    
    stats['passed'] = len(passed)
    
    # Write filtered sequences
    write_fasta(passed, output_path)
    logger.info(f"Filtered {len(records)} -> {len(passed)} sequences")
    
    return dict(stats)


def filter_pdb_directory(
    input_dir: str,
    output_dir: str,
    config: Optional[FilterConfig] = None
) -> dict:
    """
    Filter PDB files in a directory.
    
    Args:
        input_dir: Input directory with PDB files
        output_dir: Output directory for filtered PDBs
        config: Filter configuration
        
    Returns:
        Statistics dictionary
    """
    config = config or FilterConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(int)
    
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb') or f.endswith('.ent')]
    stats['total'] = len(pdb_files)
    
    for pdb_file in pdb_files:
        input_path = os.path.join(input_dir, pdb_file)
        
        try:
            structure = parse_pdb(input_path)
            
            # Resolution filter
            if structure.resolution and structure.resolution > config.max_pdb_resolution:
                stats['resolution_too_low'] += 1
                continue
            
            # Check chains
            valid_chains = 0
            for chain in structure.chains.values():
                ok, reason = filter_sequence(chain.sequence, config)
                if ok:
                    valid_chains += 1
            
            if valid_chains == 0:
                stats['no_valid_chains'] += 1
                continue
            
            # Copy to output
            output_path = os.path.join(output_dir, pdb_file)
            import shutil
            shutil.copy(input_path, output_path)
            stats['passed'] += 1
            
        except Exception as e:
            logger.warning(f"Failed to process {pdb_file}: {e}")
            stats['parse_error'] += 1
    
    logger.info(f"Filtered {stats['total']} -> {stats['passed']} PDB files")
    return dict(stats)


def compute_sequence_hash(sequence: str, k: int = 6) -> str:
    """Compute k-mer hash for sequence clustering."""
    sequence = sequence.upper()
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmers.add(sequence[i:i+k])
    
    # Hash the sorted k-mers
    kmer_str = ''.join(sorted(kmers))
    return hashlib.md5(kmer_str.encode()).hexdigest()[:16]


def cluster_sequences_simple(
    records: List[FastaRecord],
    similarity_threshold: float = 0.9
) -> List[FastaRecord]:
    """
    Simple sequence clustering to remove redundancy.
    
    Uses a greedy approach: keep first sequence, skip similar ones.
    For production, use tools like CD-HIT or MMseqs2.
    
    Args:
        records: List of FASTA records
        similarity_threshold: Minimum similarity to cluster (0-1)
        
    Returns:
        Representative sequences (one per cluster)
    """
    if not records:
        return []
    
    # Sort by length (longer first, as representatives)
    records = sorted(records, key=lambda r: len(r.sequence), reverse=True)
    
    representatives = []
    seen_hashes: Set[str] = set()
    
    for record in records:
        seq_hash = compute_sequence_hash(record.sequence)
        
        # Simple hash-based deduplication
        if seq_hash not in seen_hashes:
            representatives.append(record)
            seen_hashes.add(seq_hash)
    
    logger.info(f"Clustered {len(records)} -> {len(representatives)} sequences")
    return representatives


def remove_redundancy(
    input_path: str,
    output_path: str,
    similarity_threshold: float = 0.9
) -> dict:
    """
    Remove redundant sequences from a FASTA file.
    
    Args:
        input_path: Input FASTA file
        output_path: Output FASTA file
        similarity_threshold: Similarity threshold for clustering
        
    Returns:
        Statistics dictionary
    """
    records = parse_fasta(input_path)
    original_count = len(records)
    
    representatives = cluster_sequences_simple(records, similarity_threshold)
    
    write_fasta(representatives, output_path)
    
    return {
        'original': original_count,
        'clustered': len(representatives),
        'removed': original_count - len(representatives)
    }


def prepare_dataset(
    sequence_files: List[str],
    pdb_dir: Optional[str],
    output_dir: str,
    config: Optional[FilterConfig] = None
) -> dict:
    """
    Full data preparation pipeline.
    
    Args:
        sequence_files: List of input FASTA files
        pdb_dir: Directory with PDB files (optional)
        output_dir: Output directory
        config: Filter configuration
        
    Returns:
        Statistics dictionary
    """
    config = config or FilterConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Process sequences
    all_records = []
    for fasta_file in sequence_files:
        logger.info(f"Processing {fasta_file}")
        records = parse_fasta(fasta_file)
        
        filtered = []
        for record in records:
            ok, _ = filter_sequence(record.sequence, config)
            if ok:
                filtered.append(record)
        
        all_records.extend(filtered)
    
    # Remove redundancy
    representatives = cluster_sequences_simple(all_records)
    
    # Write output
    output_fasta = os.path.join(output_dir, "sequences.fasta")
    write_fasta(representatives, output_fasta)
    
    stats['sequences'] = {
        'original': len(all_records),
        'filtered': len(representatives),
        'output': output_fasta
    }
    
    # Process PDBs if provided
    if pdb_dir and os.path.exists(pdb_dir):
        pdb_output = os.path.join(output_dir, "structures")
        pdb_stats = filter_pdb_directory(pdb_dir, pdb_output, config)
        stats['structures'] = pdb_stats
    
    logger.info(f"Data preparation complete: {stats}")
    return stats
