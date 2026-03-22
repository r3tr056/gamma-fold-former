"""
GammaFold Former: Main Transformer Model for Protein Modeling.

A bidirectional transformer that processes protein sequences through:
1. Multi-channel input embedding (sequence + properties + structure + SS)
2. Stack of transformer blocks with RoPE attention
3. Task-specific output heads (MLM, structure prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from gammafold.models.embeddings import MultiChannelEmbedding
from gammafold.models.attention import TransformerBlock


@dataclass
class GammaFoldConfig:
    """Configuration for GammaFold Former model."""
    
    # Vocabulary
    vocab_size: int = 36
    ss_vocab_size: int = 10
    structure_vocab_size: int = 4096
    num_properties: int = 24
    
    # Model dimensions
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    ffn_dim: int = 2048  # embed_dim * 4
    
    # Sequence
    max_seq_len: int = 1024
    
    # Regularization
    dropout: float = 0.1
    
    # Features
    use_rope: bool = True
    use_structure: bool = True
    use_secondary_structure: bool = True
    
    # Output
    num_coord_dims: int = 3  # Cα xyz
    
    @classmethod
    def small(cls):
        """Small model (~30M params)."""
        return cls(
            embed_dim=320,
            num_heads=8,
            num_layers=6,
            ffn_dim=1280
        )
    
    @classmethod
    def medium(cls):
        """Medium model (~150M params)."""
        return cls(
            embed_dim=640,
            num_heads=10,
            num_layers=12,
            ffn_dim=2560
        )
    
    @classmethod
    def large(cls):
        """Large model (~650M params)."""
        return cls(
            embed_dim=1280,
            num_heads=20,
            num_layers=24,
            ffn_dim=5120
        )


class MLMHead(nn.Module):
    """Masked Language Modeling prediction head."""
    
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class StructurePredictionHead(nn.Module):
    """
    Structure prediction head for Cα coordinate regression.
    
    Predicts coordinates in a frame-invariant manner.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_coords: int = 3):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_coords)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
        Returns:
            coords: [batch, seq_len, 3]
        """
        return self.proj(hidden_states)


class DistanceMatrixHead(nn.Module):
    """
    Pairwise distance prediction head.
    
    Predicts binned distance distribution for each residue pair.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_bins: int = 64,
        min_dist: float = 2.0,
        max_dist: float = 22.0
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        # Outer product of features
        self.pair_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_bins)
        )
        
        # Bin edges for distance
        self.register_buffer(
            'bin_edges',
            torch.linspace(min_dist, max_dist, num_bins + 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
        Returns:
            distance_logits: [batch, seq_len, seq_len, num_bins]
        """
        batch, seq_len, dim = hidden_states.shape
        
        # Create pairwise features
        # [B, L, D] -> [B, L, 1, D] + [B, 1, L, D] -> [B, L, L, 2D]
        left = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
        right = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pair_features = torch.cat([left, right], dim=-1)
        
        # Project and predict
        pair_embed = self.pair_proj(pair_features)
        logits = self.output(pair_embed)
        
        return logits
    
    def compute_expected_distance(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected distance from logits."""
        probs = F.softmax(logits, dim=-1)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        expected = (probs * bin_centers).sum(dim=-1)
        return expected


class GammaFoldFormer(nn.Module):
    """
    GammaFold Former: Protein Sequence and Structure Model.
    
    A bidirectional transformer for protein modeling that:
    1. Takes multi-channel input (sequence, properties, structure, SS)
    2. Processes through transformer blocks with RoPE attention
    3. Outputs MLM predictions and/or structure predictions
    
    Args:
        config: GammaFoldConfig with model parameters
    """
    
    def __init__(self, config: GammaFoldConfig):
        super().__init__()
        self.config = config
        
        # Multi-channel embedding layer
        self.embedding = MultiChannelEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_properties=config.num_properties,
            structure_vocab_size=config.structure_vocab_size,
            ss_vocab_size=config.ss_vocab_size,
            dropout=config.dropout,
            use_structure=config.use_structure,
            use_secondary_structure=config.use_secondary_structure
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(config.embed_dim)
        
        # Output heads
        self.mlm_head = MLMHead(config.embed_dim, config.vocab_size)
        self.structure_head = StructurePredictionHead(
            config.embed_dim,
            hidden_dim=config.embed_dim // 2,
            num_coords=config.num_coord_dims
        )
        self.distance_head = DistanceMatrixHead(
            config.embed_dim,
            num_bins=64
        )
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Scale down output projections
        for layer in self.layers:
            nn.init.normal_(layer.attention.out_proj.weight, std=0.02 / (2 * len(self.layers)) ** 0.5)
    
    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = enable
    
    def forward(
        self,
        token_ids: torch.Tensor,
        properties: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structure_tokens: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        coord_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            token_ids: [B, L] Amino acid token IDs
            properties: [B, L, 24] Physicochemical properties
            attention_mask: [B, L] Attention mask (1=attend, 0=ignore)
            structure_tokens: [B, L] Optional VQ-VAE structure tokens
            ss_tokens: [B, L] Optional secondary structure tokens
            labels: [B, L] Optional MLM labels (-100 = ignore)
            coords: [B, L, 3] Optional ground truth coordinates
            coord_mask: [B, L] Optional coordinate validity mask
            return_dict: Return as dict (default) or tuple
            output_hidden_states: Return all hidden states
            output_attentions: Return attention weights
            
        Returns:
            Dict containing:
                - last_hidden_state: [B, L, D] Final hidden states
                - mlm_logits: [B, L, vocab_size] MLM predictions
                - coord_pred: [B, L, 3] Structure predictions
                - distance_logits: [B, L, L, 64] Distance predictions
                - loss: Total loss (if labels/coords provided)
                - mlm_loss: MLM loss component
                - coord_loss: Coordinate loss component
                - hidden_states: Optional list of all hidden states
                - attentions: Optional list of attention weights
        """
        _batch_size, seq_len = token_ids.shape  # noqa: F841
        
        # Embed inputs
        hidden_states = self.embedding(
            token_ids=token_ids,
            properties=properties,
            structure_tokens=structure_tokens,
            ss_tokens=ss_tokens
        )
        
        # Store hidden states and attentions if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Pass through transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, attn = torch_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False
                )
            else:
                hidden_states, attn = layer(
                    hidden_states,
                    attention_mask,
                    return_attention=output_attentions
                )
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            if output_attentions and attn is not None:
                all_attentions.append(attn)
        
        # Final layer norm
        hidden_states = self.final_ln(hidden_states)
        
        # MLM predictions
        mlm_logits = self.mlm_head(hidden_states)
        
        # Structure predictions
        coord_pred = self.structure_head(hidden_states)
        
        # Distance predictions (only for shorter sequences due to memory)
        distance_logits = None
        if seq_len <= 512:
            distance_logits = self.distance_head(hidden_states)
        
        # Compute losses if targets provided
        mlm_loss = None
        coord_loss = None
        total_loss = None
        
        if labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss = mlm_loss
        
        if coords is not None and coord_mask is not None:
            # Masked MSE loss for coordinates
            mask = coord_mask.unsqueeze(-1)  # [B, L, 1]
            coord_diff = (coord_pred - coords) ** 2
            coord_loss = (coord_diff * mask).sum() / (mask.sum() * 3 + 1e-8)
            
            if total_loss is not None:
                total_loss = total_loss + coord_loss
            else:
                total_loss = coord_loss
        
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'mlm_logits': mlm_logits,
                'coord_pred': coord_pred,
                'distance_logits': distance_logits,
                'loss': total_loss,
                'mlm_loss': mlm_loss,
                'coord_loss': coord_loss,
                'hidden_states': all_hidden_states,
                'attentions': all_attentions
            }
        
        return (hidden_states, mlm_logits, coord_pred, distance_logits, total_loss)
    
    def predict_structure(
        self,
        token_ids: torch.Tensor,
        properties: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for structure prediction only.
        
        Returns:
            coords: [B, L, 3] Predicted Cα coordinates
        """
        with torch.no_grad():
            outputs = self.forward(
                token_ids=token_ids,
                properties=properties,
                attention_mask=attention_mask,
                ss_tokens=ss_tokens,
                return_dict=True
            )
        return outputs['coord_pred']
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def num_parameters_millions(self) -> float:
        """Return parameters in millions."""
        return self.num_parameters / 1e6


def create_model(size: str = 'small', **kwargs) -> GammaFoldFormer:
    """
    Create a GammaFold Former model.
    
    Args:
        size: 'small', 'medium', or 'large'
        **kwargs: Override config parameters
        
    Returns:
        GammaFoldFormer model
    """
    if size == 'small':
        config = GammaFoldConfig.small()
    elif size == 'medium':
        config = GammaFoldConfig.medium()
    elif size == 'large':
        config = GammaFoldConfig.large()
    else:
        config = GammaFoldConfig()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return GammaFoldFormer(config)


# ============================================================================
# Multi-Modal Cross-Attention Architecture
# ============================================================================

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal cross-attention GammaFold."""
    
    # Vocabulary
    vocab_size: int = 36
    ss_vocab_size: int = 10
    structure_vocab_size: int = 4096
    num_properties: int = 24
    
    # Stream dimensions (separate dimensions for each modality)
    seq_dim: int = 384        # Sequence stream (largest)
    prop_dim: int = 128       # Properties stream
    struct_dim: int = 192     # Structure stream
    ss_dim: int = 64          # Secondary structure stream
    output_dim: int = 512     # After fusion
    
    # Architecture
    num_fusion_blocks: int = 6    # Cross-attention blocks
    num_post_layers: int = 4      # Post-fusion transformer layers
    num_heads: int = 8
    ffn_mult: int = 4
    
    # Sequence
    max_seq_len: int = 1024
    
    # Regularization
    dropout: float = 0.1
    
    # Output
    num_coord_dims: int = 3
    
    @classmethod
    def small(cls):
        """Small model (~25M params)."""
        return cls(
            seq_dim=256,
            prop_dim=96,
            struct_dim=128,
            ss_dim=48,
            output_dim=384,
            num_fusion_blocks=4,
            num_post_layers=2
        )
    
    @classmethod
    def medium(cls):
        """Medium model (~100M params)."""
        return cls(
            seq_dim=384,
            prop_dim=128,
            struct_dim=192,
            ss_dim=64,
            output_dim=512,
            num_fusion_blocks=6,
            num_post_layers=4
        )
    
    @classmethod
    def large(cls):
        """Large model (~400M params)."""
        return cls(
            seq_dim=576,    # divisible by 12
            prop_dim=192,   # divisible by 12
            struct_dim=288, # divisible by 12
            ss_dim=96,      # divisible by 12
            output_dim=768,
            num_fusion_blocks=8,
            num_post_layers=6,
            num_heads=12
        )


class GammaFoldMultiModal(nn.Module):
    """
    GammaFold with Multi-Modal Cross-Attention.
    
    Architecture:
    1. MultiStreamEmbedding - Separate embeddings for each modality
    2. CrossModalFusionBlocks - Cross-attention between modalities
    3. GatedFusion - Combine streams into unified representation
    4. Post-fusion transformer layers
    5. Task-specific output heads
    
    This architecture enables learning rich correlations between:
    - Sequence ↔ Properties (amino acid context affects property importance)
    - Sequence ↔ Structure (sequence patterns predict 3D geometry)
    - Properties ↔ Structure (physicochemical constraints)
    - All ↔ Secondary Structure (local structure patterns)
    
    Args:
        config: MultiModalConfig with model parameters
    """
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Import here to avoid circular imports
        from gammafold.models.embeddings import MultiStreamEmbedding
        from gammafold.models.cross_attention import MultiStreamEncoder
        from gammafold.models.attention import TransformerBlock
        
        # Multi-stream embedding
        self.embedding = MultiStreamEmbedding(
            vocab_size=config.vocab_size,
            seq_dim=config.seq_dim,
            prop_dim=config.prop_dim,
            struct_dim=config.struct_dim,
            ss_dim=config.ss_dim,
            num_properties=config.num_properties,
            structure_vocab_size=config.structure_vocab_size,
            ss_vocab_size=config.ss_vocab_size,
            dropout=config.dropout
        )
        
        # Cross-modal encoder
        self.cross_modal_encoder = MultiStreamEncoder(
            seq_dim=config.seq_dim,
            prop_dim=config.prop_dim,
            struct_dim=config.struct_dim,
            ss_dim=config.ss_dim,
            output_dim=config.output_dim,
            num_fusion_blocks=config.num_fusion_blocks,
            num_heads=config.num_heads,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout
        )
        
        # Post-fusion transformer layers
        self.post_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.output_dim,
                num_heads=config.num_heads,
                ffn_dim=config.output_dim * config.ffn_mult,
                dropout=config.dropout,
                use_rope=True,
                max_seq_len=config.max_seq_len
            )
            for _ in range(config.num_post_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(config.output_dim)
        
        # Output heads
        self.mlm_head = MLMHead(config.output_dim, config.vocab_size)
        self.structure_head = StructurePredictionHead(
            config.output_dim,
            hidden_dim=config.output_dim // 2,
            num_coords=config.num_coord_dims
        )
        self.distance_head = DistanceMatrixHead(
            config.output_dim,
            num_bins=64
        )
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing."""
        self.gradient_checkpointing = enable
    
    def forward(
        self,
        token_ids: torch.Tensor,
        properties: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structure_tokens: Optional[torch.Tensor] = None,
        ss_tokens: Optional[torch.Tensor] = None,
        structure_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        coord_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_stream_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-modal attention.
        
        Args:
            token_ids: [B, L] Amino acid token IDs
            properties: [B, L, 24] Physicochemical properties
            attention_mask: [B, L] Attention mask
            structure_tokens: [B, L] Optional VQ-VAE structure tokens
            ss_tokens: [B, L] Optional secondary structure tokens
            structure_mask: [B, L] Mask for structure availability
            labels: [B, L] Optional MLM labels
            coords: [B, L, 3] Optional ground truth coordinates
            coord_mask: [B, L] Coordinate validity mask
            return_dict: Return as dict
            output_stream_states: Return individual stream states
            
        Returns:
            Dict with predictions, losses, and optional stream states
        """
        _batch_size, seq_len = token_ids.shape  # noqa: F841
        
        # Embed into separate streams
        seq_emb, prop_emb, struct_emb, ss_emb = self.embedding(
            token_ids=token_ids,
            properties=properties,
            structure_tokens=structure_tokens,
            ss_tokens=ss_tokens,
            structure_mask=structure_mask
        )
        
        # Cross-modal attention
        hidden_states, stream_outputs = self.cross_modal_encoder(
            seq=seq_emb,
            prop=prop_emb,
            struct=struct_emb,
            ss=ss_emb,
            mask=attention_mask,
            struct_mask=structure_mask
        )
        
        # Post-fusion transformer layers
        for layer in self.post_layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, _ = torch_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    False,
                    use_reentrant=False
                )
            else:
                hidden_states, _ = layer(hidden_states, attention_mask)
        
        # Final layer norm
        hidden_states = self.final_ln(hidden_states)
        
        # Predictions
        mlm_logits = self.mlm_head(hidden_states)
        coord_pred = self.structure_head(hidden_states)
        
        distance_logits = None
        if seq_len <= 512:
            distance_logits = self.distance_head(hidden_states)
        
        # Losses
        mlm_loss = None
        coord_loss = None
        total_loss = None
        
        if labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss = mlm_loss
        
        if coords is not None and coord_mask is not None:
            mask = coord_mask.unsqueeze(-1)
            coord_diff = (coord_pred - coords) ** 2
            coord_loss = (coord_diff * mask).sum() / (mask.sum() * 3 + 1e-8)
            
            if total_loss is not None:
                total_loss = total_loss + coord_loss
            else:
                total_loss = coord_loss
        
        if return_dict:
            result = {
                'last_hidden_state': hidden_states,
                'mlm_logits': mlm_logits,
                'coord_pred': coord_pred,
                'distance_logits': distance_logits,
                'loss': total_loss,
                'mlm_loss': mlm_loss,
                'coord_loss': coord_loss,
            }
            if output_stream_states:
                result['stream_outputs'] = stream_outputs
            return result
        
        return (hidden_states, mlm_logits, coord_pred, distance_logits, total_loss)
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def num_parameters_millions(self) -> float:
        return self.num_parameters / 1e6


def create_multimodal_model(size: str = 'medium', **kwargs) -> GammaFoldMultiModal:
    """
    Create a GammaFold Multi-Modal model with cross-attention.
    
    Args:
        size: 'small', 'medium', or 'large'
        **kwargs: Override config parameters
        
    Returns:
        GammaFoldMultiModal model
    """
    if size == 'small':
        config = MultiModalConfig.small()
    elif size == 'medium':
        config = MultiModalConfig.medium()
    elif size == 'large':
        config = MultiModalConfig.large()
    else:
        config = MultiModalConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return GammaFoldMultiModal(config)
