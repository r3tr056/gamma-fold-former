"""
Multi-channel embedding layer for GammaFold.

Combines multiple input channels into a unified representation:
- Sequence token embeddings (learnable)
- Physicochemical property projection
- Structure token embeddings (VQ-VAE codes)
- Secondary structure embeddings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiChannelEmbedding(nn.Module):
    """
    Multi-channel protein embedding layer.
    
    Combines:
    1. Sequence token embeddings (learnable)
    2. Physicochemical property projection
    3. Structure token embeddings (optional, VQ-VAE codes)
    4. Secondary structure embeddings (optional)
    
    Args:
        vocab_size: Sequence vocabulary size (default 36)
        embed_dim: Output embedding dimension
        num_properties: Number of physicochemical properties (default 24)
        structure_vocab_size: VQ-VAE codebook size (default 4096)
        ss_vocab_size: Secondary structure vocab size (default 10)
        dropout: Dropout rate
        use_structure: Enable structure token input
        use_secondary_structure: Enable secondary structure input
    """
    
    def __init__(
        self,
        vocab_size: int = 36,
        embed_dim: int = 512,
        num_properties: int = 24,
        structure_vocab_size: int = 4096,
        ss_vocab_size: int = 10,
        dropout: float = 0.1,
        use_structure: bool = True,
        use_secondary_structure: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_structure = use_structure
        self.use_ss = use_secondary_structure
        
        # Channel dimension allocation (percentages of embed_dim)
        # Sequence: 55%, Properties: 20%, Structure: 15%, SS: 10%
        self.seq_dim = int(embed_dim * 0.55)
        self.prop_dim = int(embed_dim * 0.20)
        self.struct_dim = int(embed_dim * 0.15) if use_structure else 0
        self.ss_dim = int(embed_dim * 0.10) if use_secondary_structure else 0
        
        # Adjust to exactly match embed_dim
        remaining = embed_dim - self.seq_dim - self.prop_dim - self.struct_dim - self.ss_dim
        self.seq_dim += remaining
        
        # Channel 1: Sequence token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.seq_dim,
            padding_idx=0
        )
        
        # Channel 2: Physicochemical property projection
        # Multi-layer projection to learn from raw properties
        self.property_proj = nn.Sequential(
            nn.Linear(num_properties, self.prop_dim * 2),
            nn.LayerNorm(self.prop_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.prop_dim * 2, self.prop_dim),
            nn.LayerNorm(self.prop_dim)
        )
        
        # Channel 3: Structure tokens (VQ-VAE codes)
        if use_structure:
            self.structure_embedding = nn.Embedding(
                num_embeddings=structure_vocab_size,
                embedding_dim=self.struct_dim,
                padding_idx=0
            )
            # Learned indicator for when structure is available
            self.structure_mask_embedding = nn.Parameter(
                torch.zeros(1, 1, self.struct_dim)
            )
        
        # Channel 4: Secondary structure embeddings
        if use_secondary_structure:
            self.ss_embedding = nn.Embedding(
                num_embeddings=ss_vocab_size,
                embedding_dim=self.ss_dim,
                padding_idx=0
            )
            self.ss_mask_embedding = nn.Parameter(
                torch.zeros(1, 1, self.ss_dim)
            )
        
        # Final fusion: concatenate all channels and project
        fusion_input_dim = self.seq_dim + self.prop_dim
        if use_structure:
            fusion_input_dim += self.struct_dim
        if use_secondary_structure:
            fusion_input_dim += self.ss_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.seq_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            nn.init.zeros_(self.token_embedding.weight[self.token_embedding.padding_idx])
        
        # Structure embeddings
        if self.use_structure:
            nn.init.normal_(self.structure_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.structure_embedding.weight[0])  # padding
        
        # Secondary structure embeddings
        if self.use_ss:
            nn.init.normal_(self.ss_embedding.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.ss_embedding.weight[0])  # padding
        
        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        properties: torch.Tensor,
        structure_tokens: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None,
        structure_mask: Optional[torch.Tensor] = None,
        ss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining all input channels.
        
        Args:
            token_ids: [B, L] Amino acid token IDs
            properties: [B, L, num_properties] Physicochemical properties
            structure_tokens: [B, L] Optional VQ-VAE structure codes
            ss_tokens: [B, L] Optional secondary structure labels
            structure_mask: [B, L] Optional mask for structure availability
            ss_mask: [B, L] Optional mask for SS availability
        
        Returns:
            embeddings: [B, L, embed_dim] Fused embeddings
        """
        batch_size, seq_len = token_ids.shape
        
        # Channel 1: Sequence embeddings
        seq_emb = self.token_embedding(token_ids) * self.scale  # [B, L, seq_dim]
        
        # Channel 2: Property embeddings
        prop_emb = self.property_proj(properties)  # [B, L, prop_dim]
        
        # Collect all channels
        channels = [seq_emb, prop_emb]
        
        # Channel 3: Structure (if enabled)
        if self.use_structure:
            if structure_tokens is not None:
                struct_emb = self.structure_embedding(structure_tokens)  # [B, L, struct_dim]
                
                # Apply mask if provided (use learned mask embedding where unavailable)
                if structure_mask is not None:
                    mask = structure_mask.unsqueeze(-1).float()
                    struct_emb = struct_emb * mask + \
                                self.structure_mask_embedding.expand(batch_size, seq_len, -1) * (1 - mask)
            else:
                # No structure available, use mask embedding
                struct_emb = self.structure_mask_embedding.expand(batch_size, seq_len, -1)
            
            channels.append(struct_emb)
        
        # Channel 4: Secondary structure (if enabled)
        if self.use_ss:
            if ss_tokens is not None:
                ss_emb = self.ss_embedding(ss_tokens)  # [B, L, ss_dim]
                
                if ss_mask is not None:
                    mask = ss_mask.unsqueeze(-1).float()
                    ss_emb = ss_emb * mask + \
                            self.ss_mask_embedding.expand(batch_size, seq_len, -1) * (1 - mask)
            else:
                ss_emb = self.ss_mask_embedding.expand(batch_size, seq_len, -1)
            
            channels.append(ss_emb)
        
        # Concatenate and fuse
        combined = torch.cat(channels, dim=-1)  # [B, L, total_dim]
        fused = self.fusion(combined)  # [B, L, embed_dim]
        
        return self.dropout(fused)
    
    def get_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get only the token embeddings (for visualization/analysis)."""
        return self.token_embedding(token_ids) * self.scale
    
    def get_channel_dims(self) -> dict:
        """Return dimensions allocated to each channel."""
        return {
            'sequence': self.seq_dim,
            'properties': self.prop_dim,
            'structure': self.struct_dim if self.use_structure else 0,
            'secondary_structure': self.ss_dim if self.use_ss else 0,
            'total': self.embed_dim
        }


class MultiStreamEmbedding(nn.Module):
    """
    Multi-stream embedding layer for cross-attention architecture.
    
    Unlike MultiChannelEmbedding which fuses all channels early,
    this layer keeps streams separate for cross-modal attention.
    
    Args:
        vocab_size: Sequence vocabulary size
        seq_dim: Sequence stream dimension
        prop_dim: Properties stream dimension
        struct_dim: Structure stream dimension
        ss_dim: Secondary structure stream dimension
        num_properties: Number of physicochemical properties
        structure_vocab_size: VQ-VAE codebook size
        ss_vocab_size: Secondary structure vocab size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int = 36,
        seq_dim: int = 384,
        prop_dim: int = 128,
        struct_dim: int = 192,
        ss_dim: int = 64,
        num_properties: int = 24,
        structure_vocab_size: int = 4096,
        ss_vocab_size: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.prop_dim = prop_dim
        self.struct_dim = struct_dim
        self.ss_dim = ss_dim
        
        # Stream 1: Sequence token embeddings
        self.seq_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=seq_dim,
            padding_idx=0
        )
        self.seq_layer_norm = nn.LayerNorm(seq_dim)
        
        # Stream 2: Property encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(num_properties, prop_dim * 2),
            nn.LayerNorm(prop_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(prop_dim * 2, prop_dim),
            nn.LayerNorm(prop_dim)
        )
        
        # Stream 3: Structure token embeddings
        self.struct_embedding = nn.Embedding(
            num_embeddings=structure_vocab_size,
            embedding_dim=struct_dim,
            padding_idx=0
        )
        self.struct_layer_norm = nn.LayerNorm(struct_dim)
        # Learned embedding for when structure is unavailable
        self.struct_unavailable = nn.Parameter(torch.zeros(1, 1, struct_dim))
        
        # Stream 4: Secondary structure embeddings
        self.ss_embedding = nn.Embedding(
            num_embeddings=ss_vocab_size,
            embedding_dim=ss_dim,
            padding_idx=0
        )
        self.ss_layer_norm = nn.LayerNorm(ss_dim)
        # Learned embedding for when SS is unavailable
        self.ss_unavailable = nn.Parameter(torch.zeros(1, 1, ss_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(seq_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.seq_embedding.weight, std=0.02)
        nn.init.zeros_(self.seq_embedding.weight[0])
        
        nn.init.normal_(self.struct_embedding.weight, std=0.02)
        nn.init.zeros_(self.struct_embedding.weight[0])
        
        nn.init.normal_(self.ss_embedding.weight, std=0.02)
        nn.init.zeros_(self.ss_embedding.weight[0])
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        properties: torch.Tensor,
        structure_tokens: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None,
        structure_mask: Optional[torch.Tensor] = None,
        ss_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Embed inputs into separate streams.
        
        Args:
            token_ids: [B, L] Amino acid token IDs
            properties: [B, L, num_properties] Physicochemical properties
            structure_tokens: [B, L] Optional VQ-VAE structure codes
            ss_tokens: [B, L] Optional secondary structure labels
            structure_mask: [B, L] Optional mask for structure availability
            ss_mask: [B, L] Optional mask for SS availability
        
        Returns:
            Tuple of (seq_emb, prop_emb, struct_emb, ss_emb)
            - seq_emb: [B, L, seq_dim]
            - prop_emb: [B, L, prop_dim]
            - struct_emb: [B, L, struct_dim]
            - ss_emb: [B, L, ss_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Stream 1: Sequence
        seq_emb = self.seq_embedding(token_ids) * self.scale
        seq_emb = self.seq_layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        
        # Stream 2: Properties
        prop_emb = self.prop_encoder(properties)
        prop_emb = self.dropout(prop_emb)
        
        # Stream 3: Structure
        if structure_tokens is not None:
            struct_emb = self.struct_embedding(structure_tokens)
            struct_emb = self.struct_layer_norm(struct_emb)
            
            # Apply mask if provided
            if structure_mask is not None:
                mask = structure_mask.unsqueeze(-1).float()
                unavailable = self.struct_unavailable.expand(batch_size, seq_len, -1)
                struct_emb = struct_emb * mask + unavailable * (1 - mask)
        else:
            # No structure available
            struct_emb = self.struct_unavailable.expand(batch_size, seq_len, -1)
        struct_emb = self.dropout(struct_emb)
        
        # Stream 4: Secondary Structure
        if ss_tokens is not None:
            ss_emb = self.ss_embedding(ss_tokens)
            ss_emb = self.ss_layer_norm(ss_emb)
            
            if ss_mask is not None:
                mask = ss_mask.unsqueeze(-1).float()
                unavailable = self.ss_unavailable.expand(batch_size, seq_len, -1)
                ss_emb = ss_emb * mask + unavailable * (1 - mask)
        else:
            ss_emb = self.ss_unavailable.expand(batch_size, seq_len, -1)
        ss_emb = self.dropout(ss_emb)
        
        return seq_emb, prop_emb, struct_emb, ss_emb
    
    def get_stream_dims(self) -> dict:
        """Return dimensions of each stream."""
        return {
            'seq': self.seq_dim,
            'prop': self.prop_dim,
            'struct': self.struct_dim,
            'ss': self.ss_dim,
            'total': self.seq_dim + self.prop_dim + self.struct_dim + self.ss_dim
        }
