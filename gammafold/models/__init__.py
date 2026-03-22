"""GammaFold Models."""

from gammafold.models.embeddings import MultiChannelEmbedding, MultiStreamEmbedding
from gammafold.models.attention import (
    RotaryPositionEmbedding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock
)
from gammafold.models.cross_attention import (
    CrossModalAttention,
    CrossModalFusionBlock,
    GatedFusion,
    MultiStreamEncoder
)
from gammafold.models.gamma_fold import (
    # Original model
    GammaFoldConfig,
    GammaFoldFormer,
    create_model,
    # Multi-modal model
    MultiModalConfig,
    GammaFoldMultiModal,
    create_multimodal_model,
    # Heads
    MLMHead,
    StructurePredictionHead,
    DistanceMatrixHead,
)
from gammafold.models.structure_tokenizer import StructureTokenizer

__all__ = [
    # Embeddings
    'MultiChannelEmbedding',
    'MultiStreamEmbedding',
    # Attention
    'RotaryPositionEmbedding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    # Cross-attention
    'CrossModalAttention',
    'CrossModalFusionBlock',
    'GatedFusion',
    'MultiStreamEncoder',
    # Original model
    'GammaFoldConfig',
    'GammaFoldFormer',
    'create_model',
    # Multi-modal model
    'MultiModalConfig',
    'GammaFoldMultiModal',
    'create_multimodal_model',
    # Heads
    'MLMHead',
    'StructurePredictionHead',
    'DistanceMatrixHead',
    # Structure
    'StructureTokenizer',
]
