"""
Train/validation/test split utilities for GammaFold.

Supports:
- Random splits
- Cluster-based splits (no sequence overlap)
- Saved/loaded split indices
"""

import os
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

from gammafold.data.parsers.fasta import parse_fasta, write_fasta, FastaRecord

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    cluster_based: bool = False
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class SplitInfo:
    """Information about a data split."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    config: Dict[str, Any]
    total_samples: int
    
    def save(self, path: str):
        """Save split info to JSON."""
        data = {
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'config': self.config,
            'total_samples': self.total_samples
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved split info to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SplitInfo':
        """Load split info from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def random_split(
    n_samples: int,
    config: Optional[SplitConfig] = None
) -> SplitInfo:
    """
    Create random train/val/test split indices.
    
    Args:
        n_samples: Total number of samples
        config: Split configuration
        
    Returns:
        SplitInfo with indices for each split
    """
    config = config or SplitConfig()
    
    # Generate shuffled indices
    indices = list(range(n_samples))
    random.seed(config.seed)
    random.shuffle(indices)
    
    # Compute split points
    n_train = int(n_samples * config.train_ratio)
    n_val = int(n_samples * config.val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    logger.info(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return SplitInfo(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=asdict(config),
        total_samples=n_samples
    )


def sequence_to_cluster_id(sequence: str, k: int = 6) -> str:
    """
    Assign a cluster ID based on k-mer composition.
    Simple approach - for production use MMseqs2 clusters.
    """
    sequence = sequence.upper()
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    
    # Use first few k-mers as cluster signature
    signature = ''.join(sorted(set(kmers[:100])))
    return hashlib.md5(signature.encode()).hexdigest()[:8]


def cluster_based_split(
    records: List[FastaRecord],
    config: Optional[SplitConfig] = None
) -> SplitInfo:
    """
    Create cluster-based split to avoid sequence overlap.
    
    Sequences in the same cluster go to the same split.
    
    Args:
        records: List of FASTA records
        config: Split configuration
        
    Returns:
        SplitInfo with indices for each split
    """
    config = config or SplitConfig()
    
    # Assign cluster IDs
    clusters: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        cluster_id = sequence_to_cluster_id(record.sequence)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    # Shuffle clusters
    cluster_ids = list(clusters.keys())
    random.seed(config.seed)
    random.shuffle(cluster_ids)
    
    # Assign clusters to splits
    n_clusters = len(cluster_ids)
    n_train_clusters = int(n_clusters * config.train_ratio)
    n_val_clusters = int(n_clusters * config.val_ratio)
    
    train_clusters = cluster_ids[:n_train_clusters]
    val_clusters = cluster_ids[n_train_clusters:n_train_clusters + n_val_clusters]
    test_clusters = cluster_ids[n_train_clusters + n_val_clusters:]
    
    # Collect indices
    train_indices = []
    for cid in train_clusters:
        train_indices.extend(clusters[cid])
    
    val_indices = []
    for cid in val_clusters:
        val_indices.extend(clusters[cid])
    
    test_indices = []
    for cid in test_clusters:
        test_indices.extend(clusters[cid])
    
    logger.info(f"Cluster-based split: {len(clusters)} clusters -> "
                f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return SplitInfo(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=asdict(config),
        total_samples=len(records)
    )


def split_fasta(
    input_path: str,
    output_dir: str,
    config: Optional[SplitConfig] = None
) -> SplitInfo:
    """
    Split a FASTA file into train/val/test sets.
    
    Args:
        input_path: Input FASTA file
        output_dir: Output directory for split files
        config: Split configuration
        
    Returns:
        SplitInfo object
    """
    config = config or SplitConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Parse records
    records = parse_fasta(input_path)
    logger.info(f"Loaded {len(records)} sequences from {input_path}")
    
    # Create split
    if config.cluster_based:
        split_info = cluster_based_split(records, config)
    else:
        split_info = random_split(len(records), config)
    
    # Write split files
    for split_name, indices in [
        ('train', split_info.train_indices),
        ('val', split_info.val_indices),
        ('test', split_info.test_indices)
    ]:
        split_records = [records[i] for i in indices]
        output_path = os.path.join(output_dir, f"{split_name}.fasta")
        write_fasta(split_records, output_path)
        logger.info(f"Wrote {len(split_records)} sequences to {output_path}")
    
    # Save split info
    split_info_path = os.path.join(output_dir, "split_info.json")
    split_info.save(split_info_path)
    
    return split_info


def split_pdb_directory(
    pdb_dir: str,
    output_dir: str,
    config: Optional[SplitConfig] = None,
    split_info: Optional[SplitInfo] = None
) -> Dict[str, List[str]]:
    """
    Split PDB files into train/val/test directories.
    
    Args:
        pdb_dir: Directory with PDB files
        output_dir: Output directory
        config: Split configuration
        split_info: Optional existing split info to use
        
    Returns:
        Dictionary mapping split names to file lists
    """
    import shutil
    
    config = config or SplitConfig()
    
    # Get PDB files
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    if split_info is None:
        split_info = random_split(len(pdb_files), config)
    
    result = {'train': [], 'val': [], 'test': []}
    
    for split_name, indices in [
        ('train', split_info.train_indices),
        ('val', split_info.val_indices),
        ('test', split_info.test_indices)
    ]:
        split_dir = os.path.join(output_dir, split_name)
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        
        for idx in indices:
            if idx < len(pdb_files):
                src = os.path.join(pdb_dir, pdb_files[idx])
                dst = os.path.join(split_dir, pdb_files[idx])
                shutil.copy(src, dst)
                result[split_name].append(dst)
    
    return result
