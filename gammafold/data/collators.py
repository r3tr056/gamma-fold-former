"""
Data collators for GammaFold training.

Handles batching and padding of variable-length protein sequences.
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DataCollator:
    """
    Collates and pads batches of protein data.
    
    Handles:
    - Token ID padding
    - Property padding
    - Attention mask creation
    - Structure token padding
    """
    
    pad_token_id: int = 0
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8  # For tensor core efficiency
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched and padded tensors
        """
        # Get max length in batch
        lengths = [len(f['token_ids']) for f in features]
        max_len = max(lengths)
        
        # Apply max_length limit
        if self.max_length:
            max_len = min(max_len, self.max_length)
        
        # Pad to multiple for efficiency
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                       self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        batch_size = len(features)
        
        # Initialize tensors
        batch = {
            'token_ids': torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long),
            'attention_mask': torch.zeros((batch_size, max_len), dtype=torch.long),
        }
        
        # Handle properties
        if 'properties' in features[0]:
            num_props = features[0]['properties'].shape[-1]
            batch['properties'] = torch.zeros((batch_size, max_len, num_props), dtype=torch.float)
        
        # Handle secondary structure tokens
        if 'ss_tokens' in features[0]:
            batch['ss_tokens'] = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Handle structure tokens
        if 'structure_tokens' in features[0]:
            batch['structure_tokens'] = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        # Handle coordinates
        if 'coords' in features[0]:
            batch['coords'] = torch.zeros((batch_size, max_len, 3), dtype=torch.float)
            batch['coord_mask'] = torch.zeros((batch_size, max_len), dtype=torch.float)
        
        # Fill tensors
        for i, f in enumerate(features):
            seq_len = min(len(f['token_ids']), max_len)
            
            # Token IDs
            if isinstance(f['token_ids'], torch.Tensor):
                batch['token_ids'][i, :seq_len] = f['token_ids'][:seq_len]
            else:
                batch['token_ids'][i, :seq_len] = torch.tensor(f['token_ids'][:seq_len])
            
            # Attention mask
            batch['attention_mask'][i, :seq_len] = 1
            
            # Properties
            if 'properties' in f:
                props = f['properties']
                if isinstance(props, torch.Tensor):
                    batch['properties'][i, :seq_len] = props[:seq_len]
                else:
                    batch['properties'][i, :seq_len] = torch.tensor(props[:seq_len])
            
            # SS tokens
            if 'ss_tokens' in f:
                ss = f['ss_tokens']
                if isinstance(ss, torch.Tensor):
                    batch['ss_tokens'][i, :seq_len] = ss[:seq_len]
                else:
                    batch['ss_tokens'][i, :seq_len] = torch.tensor(ss[:seq_len])
            
            # Structure tokens
            if 'structure_tokens' in f:
                st = f['structure_tokens']
                if isinstance(st, torch.Tensor):
                    batch['structure_tokens'][i, :seq_len] = st[:seq_len]
                else:
                    batch['structure_tokens'][i, :seq_len] = torch.tensor(st[:seq_len])
            
            # Coordinates
            if 'coords' in f:
                coords = f['coords']
                if isinstance(coords, torch.Tensor):
                    batch['coords'][i, :seq_len] = coords[:seq_len]
                    batch['coord_mask'][i, :seq_len] = 1.0
                else:
                    batch['coords'][i, :seq_len] = torch.tensor(coords[:seq_len])
                    batch['coord_mask'][i, :seq_len] = 1.0
        
        return batch


@dataclass 
class DataCollatorForMLM(DataCollator):
    """
    Data collator with MLM masking.
    
    Additionally creates masked token IDs and MLM labels.
    """
    
    mask_token_id: int = 4  # <MASK>
    mlm_probability: float = 0.15
    replace_probability: float = 0.8
    random_probability: float = 0.1
    vocab_size: int = 36
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First do standard collation
        batch = super().__call__(features)
        
        # Create MLM targets
        batch['mlm_labels'] = batch['token_ids'].clone()
        
        # Create masking probability matrix
        probability_matrix = torch.full(batch['token_ids'].shape, self.mlm_probability)
        
        # Don't mask padding
        probability_matrix[batch['attention_mask'] == 0] = 0
        
        # Don't mask special tokens (0=PAD, 1=CLS, 2=SEP, etc.)
        special_tokens_mask = batch['token_ids'] < 5
        probability_matrix[special_tokens_mask] = 0
        
        # Sample masking positions
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set non-masked positions to -100 (ignore in loss)
        batch['mlm_labels'][~masked_indices] = -100
        
        # 80% replace with MASK
        indices_replaced = torch.bernoulli(
            torch.full(batch['token_ids'].shape, self.replace_probability)
        ).bool() & masked_indices
        batch['token_ids'][indices_replaced] = self.mask_token_id
        
        # 10% replace with random token
        indices_random = torch.bernoulli(
            torch.full(batch['token_ids'].shape, self.random_probability / (1 - self.replace_probability))
        ).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(5, self.vocab_size, batch['token_ids'].shape)
        batch['token_ids'][indices_random] = random_tokens[indices_random]
        
        # 10% keep original (implicit - no change needed)
        
        return batch


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn=None,
    **kwargs
):
    """
    Create a DataLoader with sensible defaults.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collation function
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    if collate_fn is None:
        collate_fn = DataCollator()
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        **kwargs
    )
