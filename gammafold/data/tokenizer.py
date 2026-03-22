"""
GammaFold Protein Tokenizer.

Rich multi-channel tokenizer that produces:
- Token IDs for embeddings
- Physicochemical property vectors (24 dimensions)
- Secondary structure tokens
- Attention masks

OPTIMIZED:
- Pre-allocated numpy arrays
- Vectorized property lookup
- Uses vocabulary constants instead of magic numbers
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from gammafold.data.vocabulary import ProteinVocabulary, VOCABULARY
from gammafold.data.properties import PhysicochemicalEncoder, PHYSICOCHEMICAL_ENCODER


@dataclass
class TokenizedProtein:
    """Container for tokenized protein data."""
    token_ids: torch.Tensor           # [seq_len] Token indices
    properties: torch.Tensor          # [seq_len, 24] Physicochemical features
    attention_mask: torch.Tensor      # [seq_len] 1 for real, 0 for padding
    ss_tokens: Optional[torch.Tensor] # [seq_len] Secondary structure tokens
    length: int                       # Original sequence length (without padding)
    sequence: str                     # Original sequence string


class GammaFoldTokenizer:
    """
    Multi-channel tokenizer for protein sequences.
    
    Produces multiple data channels:
    1. Token IDs for each amino acid + special tokens
    2. Physicochemical property vectors (24 dimensions)
    3. Secondary structure tokens (optional)
    4. Attention masks
    
    Args:
        max_length: Maximum sequence length (default 1024)
        add_special_tokens: Add <sos> and <eos> tokens
        pad_to_max: Always pad to max_length
    """
    
    # Constants from vocabulary for token ranges
    _AMINO_ACID_MIN = 1   # First amino acid token ID
    _AMINO_ACID_MAX = 20  # Last standard amino acid token ID
    
    def __init__(
        self,
        max_length: int = 1024,
        add_special_tokens: bool = True,
        pad_to_max: bool = True
    ):
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.pad_to_max = pad_to_max
        
        # Initialize vocabulary
        self.vocab = VOCABULARY
        
        # Initialize property encoder
        self.property_encoder = PHYSICOCHEMICAL_ENCODER
        
        # Build unified token lookup
        self._build_token_lookup()
        
        # Pre-compute property array for special tokens (zeros)
        self._zero_properties = np.zeros(self.num_properties, dtype=np.float32)
        
        # Build vectorized property lookup array
        self._build_property_lookup()
    
    def _build_token_lookup(self):
        """Build combined lookup table."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add all tokens
        for tok, idx in self.vocab.AMINO_ACIDS.items():
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            
        for tok, idx in self.vocab.EXTENDED_AA.items():
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            
        for tok, idx in self.vocab.SPECIAL.items():
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
    
    def _build_property_lookup(self):
        """Build fast property lookup array indexed by token ID."""
        # Create array large enough for all tokens
        max_token_id = max(self.token_to_id.values()) + 1
        self._property_array = np.zeros(
            (max_token_id, self.num_properties), 
            dtype=np.float32
        )
        
        # Fill in properties for amino acids
        for aa, idx in self.vocab.AMINO_ACIDS.items():
            self._property_array[idx] = self.property_encoder.encode(aa)
        
        for aa, idx in self.vocab.EXTENDED_AA.items():
            self._property_array[idx] = self.property_encoder.encode(aa)
        
        # Special tokens keep zeros
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab.VOCAB_SIZE
    
    @property
    def ss_vocab_size(self) -> int:
        """Return secondary structure vocabulary size."""
        return self.vocab.SS_VOCAB_SIZE
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab.SPECIAL['<pad>']
    
    @property
    def sos_token_id(self) -> int:
        return self.vocab.SPECIAL['<sos>']
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab.SPECIAL['<eos>']
    
    @property
    def mask_token_id(self) -> int:
        return self.vocab.SPECIAL['<mask>']
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab.SPECIAL['<unk>']
    
    @property
    def num_properties(self) -> int:
        """Return number of physicochemical properties."""
        return self.property_encoder.num_properties
    
    def encode(
        self,
        sequence: str,
        secondary_structure: Optional[str] = None,
        return_tensors: bool = True
    ) -> Union[TokenizedProtein, Dict]:
        """
        Encode a protein sequence into multiple channels.
        
        Args:
            sequence: Amino acid sequence string
            secondary_structure: Optional DSSP secondary structure string
            return_tensors: If True, return tensors; else return numpy arrays
            
        Returns:
            TokenizedProtein object with all encoded channels
        """
        # Clean sequence
        sequence = self._clean_sequence(sequence)
        original_seq = sequence
        seq_len = len(sequence)
        
        # Calculate total length with special tokens
        total_len = seq_len
        if self.add_special_tokens:
            total_len += 2  # <sos> and <eos>
        
        # Truncate if needed
        if total_len > self.max_length:
            seq_len = self.max_length - 2 if self.add_special_tokens else self.max_length
            sequence = sequence[:seq_len]
            if secondary_structure:
                secondary_structure = secondary_structure[:seq_len]
            total_len = self.max_length
        
        # Calculate target length for padding
        target_len = self.max_length if self.pad_to_max else total_len
        
        # Pre-allocate arrays
        token_ids = np.full(target_len, self.pad_token_id, dtype=np.int64)
        properties = np.zeros((target_len, self.num_properties), dtype=np.float32)
        ss_token_ids = np.full(target_len, self.vocab.SECONDARY_STRUCTURE['<ss_pad>'], dtype=np.int64)
        attention_mask = np.zeros(target_len, dtype=np.int64)
        
        # Position index
        pos = 0
        
        # Add start token
        if self.add_special_tokens:
            token_ids[pos] = self.sos_token_id
            attention_mask[pos] = 1
            pos += 1
        
        # Encode sequence - vectorized property lookup
        for i, aa in enumerate(sequence):
            aa_upper = aa.upper()
            
            # Token ID
            token_id = self.token_to_id.get(aa_upper, self.unk_token_id)
            token_ids[pos] = token_id
            
            # Properties - vectorized lookup
            if token_id < len(self._property_array):
                properties[pos] = self._property_array[token_id]
            
            # Secondary structure
            if secondary_structure and i < len(secondary_structure):
                ss = secondary_structure[i]
                ss_token_ids[pos] = self.vocab.SECONDARY_STRUCTURE.get(
                    ss, self.vocab.SECONDARY_STRUCTURE['<ss_unk>']
                )
            else:
                ss_token_ids[pos] = self.vocab.SECONDARY_STRUCTURE['<ss_unk>']
            
            attention_mask[pos] = 1
            pos += 1
        
        # Add end token
        if self.add_special_tokens:
            token_ids[pos] = self.eos_token_id
            attention_mask[pos] = 1
            pos += 1
        
        # Record actual length
        actual_length = pos
        
        if return_tensors:
            return TokenizedProtein(
                token_ids=torch.from_numpy(token_ids),
                properties=torch.from_numpy(properties),
                attention_mask=torch.from_numpy(attention_mask),
                ss_tokens=torch.from_numpy(ss_token_ids),
                length=actual_length,
                sequence=original_seq
            )
        else:
            return {
                'token_ids': token_ids,
                'properties': properties,
                'attention_mask': attention_mask,
                'ss_tokens': ss_token_ids,
                'length': actual_length,
                'sequence': original_seq
            }
    
    def batch_encode(
        self,
        sequences: List[str],
        secondary_structures: Optional[List[str]] = None,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            secondary_structures: Optional list of DSSP strings
            return_tensors: If True, return stacked tensors
            
        Returns:
            Dictionary with batched tensors
        """
        if secondary_structures is None:
            secondary_structures = [None] * len(sequences)
        
        batch_size = len(sequences)
        target_len = self.max_length if self.pad_to_max else max(
            len(s) + (2 if self.add_special_tokens else 0) for s in sequences
        )
        
        # Pre-allocate batch arrays
        batch_token_ids = np.full((batch_size, target_len), self.pad_token_id, dtype=np.int64)
        batch_properties = np.zeros((batch_size, target_len, self.num_properties), dtype=np.float32)
        batch_attention = np.zeros((batch_size, target_len), dtype=np.int64)
        batch_ss = np.full((batch_size, target_len), self.vocab.SECONDARY_STRUCTURE['<ss_pad>'], dtype=np.int64)
        batch_lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i, (seq, ss) in enumerate(zip(sequences, secondary_structures)):
            encoded = self.encode(seq, ss, return_tensors=False)
            length = encoded['length']
            
            batch_token_ids[i, :target_len] = encoded['token_ids'][:target_len]
            batch_properties[i, :target_len] = encoded['properties'][:target_len]
            batch_attention[i, :target_len] = encoded['attention_mask'][:target_len]
            batch_ss[i, :target_len] = encoded['ss_tokens'][:target_len]
            batch_lengths[i] = length
        
        return {
            'token_ids': torch.from_numpy(batch_token_ids),
            'properties': torch.from_numpy(batch_properties),
            'attention_mask': torch.from_numpy(batch_attention),
            'ss_tokens': torch.from_numpy(batch_ss),
            'lengths': torch.from_numpy(batch_lengths)
        }
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to sequence string.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: If True, skip special tokens
            
        Returns:
            Decoded sequence string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, '?')
            
            if skip_special_tokens and token in self.vocab.SPECIAL:
                continue
            
            tokens.append(token)
        
        return ''.join(tokens)
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate sequence."""
        # Remove whitespace and newlines
        sequence = ''.join(sequence.split())
        # Convert to uppercase
        sequence = sequence.upper()
        return sequence
    
    def create_mlm_targets(
        self,
        token_ids: torch.Tensor,
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1
    ) -> tuple:
        """
        Create masked language modeling targets.
        
        Following BERT/ESM-2:
        - 15% of tokens are selected for prediction
        - Of those: 80% replaced with <mask>, 10% random, 10% unchanged
        
        Args:
            token_ids: Input token IDs [batch, seq_len] or [seq_len]
            mask_prob: Probability of selecting a token for masking
            mask_token_prob: Probability of replacing with <mask>
            random_token_prob: Probability of replacing with random token
            
        Returns:
            Tuple of (masked_token_ids, labels)
        """
        original_shape = token_ids.shape
        if len(original_shape) == 1:
            token_ids = token_ids.unsqueeze(0)
        
        # Clone input for modification
        masked_ids = token_ids.clone()
        labels = token_ids.clone()
        
        # Create mask for amino acid positions only (using class constants)
        can_mask = (token_ids >= self._AMINO_ACID_MIN) & (token_ids <= self._AMINO_ACID_MAX)
        
        # Random selection of positions to mask
        rand_mask = torch.rand_like(token_ids.float()) < mask_prob
        mask_positions = rand_mask & can_mask
        
        # Set labels: -100 for positions we don't predict
        labels[~mask_positions] = -100
        
        # Determine how to handle each masked position
        rand_vals = torch.rand_like(token_ids.float())
        
        # 80% of time, replace with <mask>
        mask_with_mask_token = mask_positions & (rand_vals < mask_token_prob)
        masked_ids[mask_with_mask_token] = self.mask_token_id
        
        # 10% of time, replace with random amino acid
        mask_with_random = mask_positions & (rand_vals >= mask_token_prob) & \
                          (rand_vals < mask_token_prob + random_token_prob)
        random_tokens = torch.randint(
            self._AMINO_ACID_MIN, 
            self._AMINO_ACID_MAX + 1, 
            token_ids.shape, 
            device=token_ids.device
        )
        masked_ids[mask_with_random] = random_tokens[mask_with_random]
        
        # 10% of time, keep original (no change needed)
        
        if len(original_shape) == 1:
            masked_ids = masked_ids.squeeze(0)
            labels = labels.squeeze(0)
        
        return masked_ids, labels
