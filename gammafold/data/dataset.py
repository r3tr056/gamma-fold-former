"""
PyTorch datasets for GammaFold.

Supports:
- Sequence-only datasets (FASTA)
- Structure datasets (PDB with coordinates)
- Multi-modal datasets (sequence + structure + SS)

OPTIMIZED:
- LRU cache with configurable max size
- Lazy sequence loading option
- PDB structure caching
"""

import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
from functools import lru_cache
import numpy as np

from gammafold.data.tokenizer import GammaFoldTokenizer
from gammafold.data.parsers.fasta import parse_fasta, iter_fasta, FastaRecord
from gammafold.data.parsers.pdb import parse_pdb, PDBStructure

logger = logging.getLogger(__name__)


class LRUCache:
    """
    LRU cache with configurable maximum size.
    
    Evicts least recently used items when full.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict = {}
        self._access_order: List = []
    
    def get(self, key):
        """Get item, return None if not present."""
        if key in self._cache:
            # Move to end (most recent)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        """Put item, evicting if necessary."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def __contains__(self, key):
        return key in self._cache
    
    def __len__(self):
        return len(self._cache)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()


class ProteinSequenceDataset(Dataset):
    """
    Dataset for sequence-only training (MLM pretraining).
    
    Loads sequences from FASTA files and tokenizes them.
    
    Args:
        fasta_paths: Path(s) to FASTA file(s)
        tokenizer: GammaFoldTokenizer instance
        max_length: Maximum sequence length
        min_length: Minimum sequence length (filter short sequences)
        cache_tokenized: Cache tokenized sequences
        max_cache_size: Maximum number of cached items (0 = unlimited, -1 = no cache)
        lazy_load: If True, don't load all sequences into memory
    """
    
    def __init__(
        self,
        fasta_paths: Union[str, List[str]],
        tokenizer: GammaFoldTokenizer,
        max_length: int = 1024,
        min_length: int = 10,
        cache_tokenized: bool = True,
        max_cache_size: int = 10000,
        lazy_load: bool = False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.cache_tokenized = cache_tokenized
        self.lazy_load = lazy_load
        
        # Normalize paths
        if isinstance(fasta_paths, str):
            fasta_paths = [fasta_paths]
        self.fasta_paths = fasta_paths
        
        # Initialize cache
        if cache_tokenized and max_cache_size != -1:
            self._cache = LRUCache(max_cache_size) if max_cache_size > 0 else {}
        else:
            self._cache = None
        
        if lazy_load:
            # Build index only (file offset, length)
            self._build_lazy_index()
        else:
            # Load all sequences into memory
            self._load_sequences()
    
    def _load_sequences(self):
        """Load all sequences into memory."""
        self.sequences = []
        self.ids = []
        
        for path in self.fasta_paths:
            logger.info(f"Loading FASTA: {path}")
            records = parse_fasta(path)
            for record in records:
                if self.min_length <= len(record.sequence) <= self.max_length:
                    self.sequences.append(record.sequence)
                    self.ids.append(record.id)
        
        logger.info(f"Loaded {len(self.sequences)} sequences")
    
    def _build_lazy_index(self):
        """Build index for lazy loading without loading sequences."""
        self.index = []  # List of (file_path, file_offset, seq_id)
        
        for path in self.fasta_paths:
            logger.info(f"Indexing FASTA: {path}")
            with open(path, 'r') as f:
                current_id = None
                seq_start = None
                seq_len = 0
                
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('>'):
                        # Save previous sequence if valid
                        if current_id is not None and self.min_length <= seq_len <= self.max_length:
                            self.index.append((path, seq_start, current_id, seq_len))
                        
                        # Start new sequence
                        parts = line[1:].split(None, 1)
                        current_id = parts[0] if parts else "unknown"
                        seq_start = offset
                        seq_len = 0
                    else:
                        seq_len += len(line)
                
                # Handle last sequence
                if current_id is not None and self.min_length <= seq_len <= self.max_length:
                    self.index.append((path, seq_start, current_id, seq_len))
        
        logger.info(f"Indexed {len(self.index)} sequences")
    
    def _load_sequence_at_index(self, idx: int) -> str:
        """Load a single sequence from disk."""
        path, offset, seq_id, seq_len = self.index[idx]
        
        sequence_lines = []
        with open(path, 'r') as f:
            f.seek(offset)
            f.readline()  # Skip header
            
            chars_read = 0
            while chars_read < seq_len:
                line = f.readline().strip()
                if not line or line.startswith('>'):
                    break
                sequence_lines.append(line.upper())
                chars_read += len(line)
        
        return ''.join(sequence_lines)
    
    def __len__(self) -> int:
        if self.lazy_load:
            return len(self.index)
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self._cache is not None:
            cached = self._cache.get(idx) if isinstance(self._cache, LRUCache) else self._cache.get(idx)
            if cached is not None:
                return cached
        
        # Get sequence
        if self.lazy_load:
            sequence = self._load_sequence_at_index(idx)
        else:
            sequence = self.sequences[idx]
        
        encoded = self.tokenizer.encode(sequence)
        
        result = {
            'token_ids': encoded.token_ids,
            'properties': encoded.properties,
            'attention_mask': encoded.attention_mask,
            'ss_tokens': encoded.ss_tokens,
        }
        
        # Cache
        if self._cache is not None:
            if isinstance(self._cache, LRUCache):
                self._cache.put(idx, result)
            else:
                self._cache[idx] = result
        
        return result


class ProteinStructureDataset(Dataset):
    """
    Dataset for structure prediction training.
    
    Loads structures from PDB files and provides:
    - Tokenized sequences
    - Cα coordinates
    - Secondary structure labels
    - B-factors (as confidence proxy)
    
    OPTIMIZED: Caches parsed PDB structures with LRU eviction.
    
    Args:
        pdb_dir: Directory containing PDB files
        tokenizer: GammaFoldTokenizer instance
        structure_tokenizer: Optional VQ-VAE structure tokenizer
        max_length: Maximum sequence length
        file_list: Optional list of specific PDB files to load
        max_cache_size: Maximum cached PDB structures
    """
    
    def __init__(
        self,
        pdb_dir: str,
        tokenizer: GammaFoldTokenizer,
        structure_tokenizer=None,
        max_length: int = 1024,
        file_list: Optional[List[str]] = None,
        max_cache_size: int = 500
    ):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.tokenizer = tokenizer
        self.structure_tokenizer = structure_tokenizer
        self.max_length = max_length
        
        # PDB structure cache
        self._pdb_cache = LRUCache(max_cache_size)
        
        # Find all PDB files
        if file_list is not None:
            self.pdb_files = file_list
        else:
            self.pdb_files = []
            for f in os.listdir(pdb_dir):
                if f.endswith('.pdb') or f.endswith('.ent'):
                    self.pdb_files.append(f)
        
        logger.info(f"Found {len(self.pdb_files)} PDB files")
        
        # Index: (file, chain_id) tuples
        self._build_index()
    
    def _build_index(self):
        """Build index of (file, chain) pairs."""
        self.index = []
        
        for pdb_file in self.pdb_files:
            pdb_path = os.path.join(self.pdb_dir, pdb_file)
            try:
                structure = parse_pdb(pdb_path)
                # Cache the parsed structure
                self._pdb_cache.put(pdb_file, structure)
                
                for chain_id, chain in structure.chains.items():
                    if len(chain.sequence) <= self.max_length:
                        self.index.append((pdb_file, chain_id))
            except Exception as e:
                logger.warning(f"Failed to parse {pdb_file}: {e}")
        
        logger.info(f"Indexed {len(self.index)} chains")
    
    def _get_structure(self, pdb_file: str) -> PDBStructure:
        """Get structure from cache or parse from disk."""
        cached = self._pdb_cache.get(pdb_file)
        if cached is not None:
            return cached
        
        pdb_path = os.path.join(self.pdb_dir, pdb_file)
        structure = parse_pdb(pdb_path)
        self._pdb_cache.put(pdb_file, structure)
        return structure
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pdb_file, chain_id = self.index[idx]
        
        # Get structure from cache
        structure = self._get_structure(pdb_file)
        chain = structure.get_chain(chain_id)
        
        # Tokenize sequence with secondary structure
        encoded = self.tokenizer.encode(
            chain.sequence,
            secondary_structure=chain.secondary_structure
        )
        
        # Get coordinates
        coords = torch.tensor(chain.ca_coords, dtype=torch.float32)
        seq_len = len(chain.sequence)
        
        # Pad coordinates to max_length using F.pad for efficiency
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            coords = torch.nn.functional.pad(coords, (0, 0, 0, pad_len), value=0.0)
        
        # Create coordinate mask (1 for real coordinates)
        coord_mask = torch.zeros(self.max_length)
        coord_mask[:seq_len] = 1.0
        
        result = {
            'token_ids': encoded.token_ids,
            'properties': encoded.properties,
            'attention_mask': encoded.attention_mask,
            'ss_tokens': encoded.ss_tokens,
            'coords': coords,
            'coord_mask': coord_mask,
        }
        
        # Optional: generate structure tokens
        if self.structure_tokenizer is not None:
            with torch.no_grad():
                struct_tokens = self.structure_tokenizer.tokenize(
                    coords.unsqueeze(0)
                ).squeeze(0)
            result['structure_tokens'] = struct_tokens
        
        return result
    
    def clear_cache(self):
        """Clear the PDB structure cache."""
        self._pdb_cache.clear()


class MultiModalProteinDataset(Dataset):
    """
    Combined dataset supporting both sequence-only and structure data.
    
    Useful for training where some samples have structure and others don't.
    
    Args:
        sequence_dataset: Sequence-only dataset
        structure_dataset: Optional structure dataset
        structure_prob: Probability of sampling from structure dataset
    """
    
    def __init__(
        self,
        sequence_dataset: ProteinSequenceDataset,
        structure_dataset: Optional[ProteinStructureDataset] = None,
        structure_prob: float = 0.5
    ):
        super().__init__()
        self.sequence_dataset = sequence_dataset
        self.structure_dataset = structure_dataset
        self.structure_prob = structure_prob
        
        # Total length
        self.seq_len = len(sequence_dataset)
        self.struct_len = len(structure_dataset) if structure_dataset else 0
    
    def __len__(self) -> int:
        return self.seq_len + self.struct_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.seq_len:
            # Sequence-only sample
            item = self.sequence_dataset[idx]
            item['has_structure'] = torch.tensor(0)
        else:
            # Structure sample
            struct_idx = idx - self.seq_len
            item = self.structure_dataset[struct_idx]
            item['has_structure'] = torch.tensor(1)
        
        return item


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create DataLoader with typical settings.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
