"""
Rotary Position Embeddings (RoPE) and Attention for GammaFold.

RoPE encodes position information by rotating query and key vectors,
enabling length generalization and relative position awareness.

Includes Flash Attention support for 2-4x speedup on compatible GPUs.

Reference: RoFormer (Su et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# Check for Flash Attention / SDPA availability
_FLASH_ATTENTION_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')
_FLASH_ATTENTION_ENABLED = _FLASH_ATTENTION_AVAILABLE


def is_flash_attention_available() -> bool:
    """Check if Flash Attention (SDPA) is available."""
    return _FLASH_ATTENTION_AVAILABLE


def set_flash_attention_enabled(enabled: bool):
    """Enable or disable Flash Attention globally."""
    global _FLASH_ATTENTION_ENABLED
    if enabled and not _FLASH_ATTENTION_AVAILABLE:
        raise RuntimeError("Flash Attention not available in this PyTorch version")
    _FLASH_ATTENTION_ENABLED = enabled


def get_flash_attention_enabled() -> bool:
    """Check if Flash Attention is currently enabled."""
    return _FLASH_ATTENTION_ENABLED


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotation to query and key vectors based on position,
    encoding relative position information in the attention mechanism.
    
    Args:
        dim: Dimension per attention head
        max_seq_len: Maximum sequence length
        base: Base for computing rotation frequencies (default 10000)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for max length
        self._precompute_cos_sin(max_seq_len)
    
    def _precompute_cos_sin(self, seq_len: int):
        """Precompute cos and sin values for positions."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim/2]
        
        # Duplicate for full dim: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin: [1, seq_len, 1, dim]
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(2)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(2)
        
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            seq_len: Optional sequence length override
            
        Returns:
            Rotated q and k tensors
        """
        if seq_len is None:
            seq_len = q.shape[1]
        
        # Ensure we have precomputed values for this length
        if seq_len > self.cos_cached.shape[1]:
            self._precompute_cos_sin(seq_len)
        
        # Get cos/sin for the sequence length
        cos = self.cos_cached[:, :seq_len]  # [1, seq_len, 1, dim]
        sin = self.sin_cached[:, :seq_len]
        
        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embeddings.
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        use_rope: Whether to use rotary position embeddings
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        
        # QKV projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary position embedding
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len
            )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional Flash Attention.
        
        Uses PyTorch's scaled_dot_product_attention (SDPA) when available
        for 2-4x speedup. Falls back to standard attention when:
        - Flash Attention is not available
        - Attention weights are requested (return_attention=True)
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional mask [batch, seq_len] where 1=attend, 0=ignore
            return_attention: Whether to return attention weights (disables Flash Attention)
            
        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: Optional [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [B, L, H, D/H]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings
        if self.use_rope:
            q, k = self.rope(q, k, seq_len)
        
        # Transpose for attention: [B, H, L, D/H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use Flash Attention (SDPA) when possible
        use_flash = _FLASH_ATTENTION_ENABLED and not return_attention
        
        if use_flash:
            # Prepare attention mask for SDPA
            # SDPA expects: None, or [B, L], [B, 1, L], [B, 1, 1, L], or [B, H, L, L]
            attn_mask = None
            if attention_mask is not None:
                # Convert from [B, L] (1=attend) to [B, 1, 1, L] boolean
                attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
                # Expand to [B, 1, L, L] for key dimension
                attn_mask = attn_mask.expand(-1, -1, seq_len, -1)
            
            # Use scaled_dot_product_attention (Flash Attention)
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_probs = None
        else:
            # Standard attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
            
            # Apply attention mask
            if attention_mask is not None:
                # Expand mask: [B, L] -> [B, 1, 1, L]
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            # Softmax and dropout
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            output = torch.matmul(attn_probs, v)  # [B, H, L, D/H]
        
        # Reshape back: [B, L, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_probs
        return output, None


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.
    
    SwiGLU provides better performance than standard FFN.
    
    Args:
        embed_dim: Input/output dimension
        hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim * 4
        
        # SwiGLU: gate and up projections
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        SwiGLU(x) = (xW) * silu(xV) * W2
        """
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return self.dropout(output)


class TransformerBlock(nn.Module):
    """
    Single Transformer block with Pre-LayerNorm.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → + → LayerNorm → FFN → +
        └──────────────────────────────────────┘ └────────────────────┘
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden dimension
        dropout: Dropout rate
        use_rope: Use rotary position embeddings
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len
        )
        
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim or embed_dim * 4,
            dropout=dropout
        )
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, embed_dim]
            attention_mask: Optional mask [batch, seq_len]
            return_attention: Whether to return attention weights
        """
        # Pre-norm attention with residual
        residual = x
        x = self.ln1(x)
        attn_output, attn_weights = self.attention(x, attention_mask, return_attention)
        x = residual + self.dropout(attn_output)
        
        # Pre-norm FFN with residual
        residual = x
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = residual + ffn_output
        
        return x, attn_weights
