"""
Cross-Modal Attention for Multi-Stream GammaFold Architecture.

Enables learning correlations between different input modalities:
- Sequence ↔ Properties
- Sequence ↔ Structure  
- Properties ↔ Structure
- All ↔ Secondary Structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class CrossModalAttention(nn.Module):
    """
    Cross-attention between two modality streams.
    
    Query comes from one modality, Key/Value from another.
    This allows one stream to "attend to" information from another.
    
    Args:
        query_dim: Dimension of query stream
        kv_dim: Dimension of key-value stream
        num_heads: Number of attention heads
        dropout: Attention dropout
    """
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection from query stream
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        
        # Key, Value projections from kv stream (project to query_dim for compatibility)
        self.k_proj = nn.Linear(kv_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, query_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, L, query_dim] from one modality
            key_value: [B, L, kv_dim] from another modality
            attention_mask: [B, L] where 1=attend, 0=ignore
            
        Returns:
            output: [B, L, query_dim]
            attention_weights: Optional [B, H, L, L]
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head: [B, H, L, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        output = torch.matmul(attn_probs, v)  # [B, H, L, D]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.query_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attn_probs
        return output, None


class CrossModalFusionBlock(nn.Module):
    """
    One block of multi-stream cross-modal processing.
    
    Each block:
    1. Self-attention on each stream independently
    2. Cross-attention between streams
    3. Feed-forward on each stream
    
    Args:
        seq_dim: Sequence stream dimension
        prop_dim: Properties stream dimension
        struct_dim: Structure stream dimension
        ss_dim: Secondary structure stream dimension
        num_heads: Attention heads
        ffn_mult: FFN hidden dim multiplier
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        seq_dim: int = 384,
        prop_dim: int = 128,
        struct_dim: int = 192,
        ss_dim: int = 64,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.prop_dim = prop_dim
        self.struct_dim = struct_dim
        self.ss_dim = ss_dim
        
        # Helper to compute valid number of heads for a dimension
        def get_valid_heads(dim: int, desired: int) -> int:
            """Return largest valid head count <= desired that divides dim."""
            for h in range(desired, 0, -1):
                if dim % h == 0:
                    return h
            return 1
        
        # Compute head counts that divide each dimension
        seq_heads = get_valid_heads(seq_dim, num_heads)
        prop_heads = get_valid_heads(prop_dim, max(1, num_heads // 2))
        struct_heads = get_valid_heads(struct_dim, max(1, num_heads // 2))
        ss_heads = get_valid_heads(ss_dim, max(1, num_heads // 4))
        
        # ===== Self-Attention per stream =====
        self.seq_self_attn = nn.MultiheadAttention(
            seq_dim, seq_heads, dropout=dropout, batch_first=True
        )
        self.prop_self_attn = nn.MultiheadAttention(
            prop_dim, prop_heads, dropout=dropout, batch_first=True
        )
        self.struct_self_attn = nn.MultiheadAttention(
            struct_dim, struct_heads, dropout=dropout, batch_first=True
        )
        self.ss_self_attn = nn.MultiheadAttention(
            ss_dim, ss_heads, dropout=dropout, batch_first=True
        )
        
        # ===== Cross-Attention: Sequence attends to others =====
        self.seq_cross_prop = CrossModalAttention(seq_dim, prop_dim, seq_heads, dropout)
        self.seq_cross_struct = CrossModalAttention(seq_dim, struct_dim, seq_heads, dropout)
        self.seq_cross_ss = CrossModalAttention(seq_dim, ss_dim, seq_heads, dropout)
        
        # ===== Cross-Attention: Structure attends to sequence and properties =====
        self.struct_cross_seq = CrossModalAttention(struct_dim, seq_dim, struct_heads, dropout)
        self.struct_cross_prop = CrossModalAttention(struct_dim, prop_dim, struct_heads, dropout)
        
        # ===== Cross-Attention: Properties attends to sequence =====
        self.prop_cross_seq = CrossModalAttention(prop_dim, seq_dim, prop_heads, dropout)
        
        # ===== Cross-Attention: SS attends to sequence =====
        self.ss_cross_seq = CrossModalAttention(ss_dim, seq_dim, ss_heads, dropout)
        
        # ===== Gating for cross-attention fusion =====
        self.seq_gate = nn.Sequential(
            nn.Linear(seq_dim * 4, seq_dim),
            nn.Sigmoid()
        )
        self.struct_gate = nn.Sequential(
            nn.Linear(struct_dim * 3, struct_dim),
            nn.Sigmoid()
        )
        self.prop_gate = nn.Sequential(
            nn.Linear(prop_dim * 2, prop_dim),
            nn.Sigmoid()
        )
        self.ss_gate = nn.Sequential(
            nn.Linear(ss_dim * 2, ss_dim),
            nn.Sigmoid()
        )
        
        # ===== LayerNorms =====
        self.seq_ln1 = nn.LayerNorm(seq_dim)
        self.seq_ln2 = nn.LayerNorm(seq_dim)
        self.seq_ln3 = nn.LayerNorm(seq_dim)
        
        self.prop_ln1 = nn.LayerNorm(prop_dim)
        self.prop_ln2 = nn.LayerNorm(prop_dim)
        self.prop_ln3 = nn.LayerNorm(prop_dim)
        
        self.struct_ln1 = nn.LayerNorm(struct_dim)
        self.struct_ln2 = nn.LayerNorm(struct_dim)
        self.struct_ln3 = nn.LayerNorm(struct_dim)
        
        self.ss_ln1 = nn.LayerNorm(ss_dim)
        self.ss_ln2 = nn.LayerNorm(ss_dim)
        self.ss_ln3 = nn.LayerNorm(ss_dim)
        
        # ===== Feed-Forward Networks =====
        self.seq_ffn = self._make_ffn(seq_dim, ffn_mult, dropout)
        self.prop_ffn = self._make_ffn(prop_dim, ffn_mult, dropout)
        self.struct_ffn = self._make_ffn(struct_dim, ffn_mult, dropout)
        self.ss_ffn = self._make_ffn(ss_dim, ffn_mult, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def _make_ffn(self, dim: int, mult: int, dropout: float) -> nn.Module:
        """Create SwiGLU FFN."""
        hidden = dim * mult
        return nn.Sequential(
            nn.Linear(dim, hidden * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )
    
    def forward(
        self,
        seq: torch.Tensor,
        prop: torch.Tensor,
        struct: torch.Tensor,
        ss: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process all streams with self and cross attention.
        
        Args:
            seq: [B, L, seq_dim] Sequence stream
            prop: [B, L, prop_dim] Properties stream
            struct: [B, L, struct_dim] Structure stream
            ss: [B, L, ss_dim] Secondary structure stream
            mask: [B, L] Attention mask
            
        Returns:
            Updated (seq, prop, struct, ss) streams
        """
        # Convert mask for nn.MultiheadAttention (inverted: True = ignore)
        attn_mask = ~mask.bool() if mask is not None else None
        
        # ===== 1. Self-Attention per stream =====
        # Sequence
        seq_normed = self.seq_ln1(seq)
        seq_attn, _ = self.seq_self_attn(
            seq_normed, seq_normed, seq_normed,
            key_padding_mask=attn_mask
        )
        seq = seq + self.dropout(seq_attn)
        
        # Properties
        prop_normed = self.prop_ln1(prop)
        prop_attn, _ = self.prop_self_attn(
            prop_normed, prop_normed, prop_normed,
            key_padding_mask=attn_mask
        )
        prop = prop + self.dropout(prop_attn)
        
        # Structure
        struct_normed = self.struct_ln1(struct)
        struct_attn, _ = self.struct_self_attn(
            struct_normed, struct_normed, struct_normed,
            key_padding_mask=attn_mask
        )
        struct = struct + self.dropout(struct_attn)
        
        # Secondary Structure
        ss_normed = self.ss_ln1(ss)
        ss_attn, _ = self.ss_self_attn(
            ss_normed, ss_normed, ss_normed,
            key_padding_mask=attn_mask
        )
        ss = ss + self.dropout(ss_attn)
        
        # ===== 2. Cross-Attention =====
        # Sequence cross-attends to all others
        seq_normed = self.seq_ln2(seq)
        seq_from_prop, _ = self.seq_cross_prop(seq_normed, prop, mask)
        seq_from_struct, _ = self.seq_cross_struct(seq_normed, struct, mask)
        seq_from_ss, _ = self.seq_cross_ss(seq_normed, ss, mask)
        
        # Gated fusion for sequence
        seq_cross = torch.cat([seq_normed, seq_from_prop, seq_from_struct, seq_from_ss], dim=-1)
        seq_gate = self.seq_gate(seq_cross)
        seq_fused = seq_from_prop + seq_from_struct + seq_from_ss
        seq = seq + self.dropout(seq_gate * seq_fused)
        
        # Structure cross-attends to sequence and properties
        struct_normed = self.struct_ln2(struct)
        struct_from_seq, _ = self.struct_cross_seq(struct_normed, seq, mask)
        struct_from_prop, _ = self.struct_cross_prop(struct_normed, prop, mask)
        
        struct_cross = torch.cat([struct_normed, struct_from_seq, struct_from_prop], dim=-1)
        struct_gate = self.struct_gate(struct_cross)
        struct_fused = struct_from_seq + struct_from_prop
        struct = struct + self.dropout(struct_gate * struct_fused)
        
        # Properties cross-attends to sequence
        prop_normed = self.prop_ln2(prop)
        prop_from_seq, _ = self.prop_cross_seq(prop_normed, seq, mask)
        
        prop_cross = torch.cat([prop_normed, prop_from_seq], dim=-1)
        prop_gate = self.prop_gate(prop_cross)
        prop = prop + self.dropout(prop_gate * prop_from_seq)
        
        # SS cross-attends to sequence
        ss_normed = self.ss_ln2(ss)
        ss_from_seq, _ = self.ss_cross_seq(ss_normed, seq, mask)
        
        ss_cross = torch.cat([ss_normed, ss_from_seq], dim=-1)
        ss_gate = self.ss_gate(ss_cross)
        ss = ss + self.dropout(ss_gate * ss_from_seq)
        
        # ===== 3. Feed-Forward Networks =====
        seq = seq + self.dropout(self.seq_ffn(self.seq_ln3(seq)))
        prop = prop + self.dropout(self.prop_ffn(self.prop_ln3(prop)))
        struct = struct + self.dropout(self.struct_ffn(self.struct_ln3(struct)))
        ss = ss + self.dropout(self.ss_ffn(self.ss_ln3(ss)))
        
        return seq, prop, struct, ss


class SwiGLU(nn.Module):
    """SwiGLU activation: splits input and applies silu gating."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class GatedFusion(nn.Module):
    """
    Gated combination of all modality streams into unified representation.
    
    Uses learned attention weights to determine contribution of each modality
    at each position.
    
    Args:
        seq_dim: Sequence stream dimension
        prop_dim: Properties stream dimension
        struct_dim: Structure stream dimension
        ss_dim: Secondary structure stream dimension
        output_dim: Final output dimension
    """
    
    def __init__(
        self,
        seq_dim: int,
        prop_dim: int,
        struct_dim: int,
        ss_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        # Project each stream to output dimension
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.prop_proj = nn.Linear(prop_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.ss_proj = nn.Linear(ss_dim, output_dim)
        
        # Gate network: learns importance of each modality
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 4)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        seq: torch.Tensor,
        prop: torch.Tensor,
        struct: torch.Tensor,
        ss: torch.Tensor,
        struct_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine all streams with learned gating.
        
        Args:
            seq: [B, L, seq_dim]
            prop: [B, L, prop_dim]
            struct: [B, L, struct_dim]
            ss: [B, L, ss_dim]
            struct_mask: [B, L] Optional mask for positions without structure
            
        Returns:
            output: [B, L, output_dim] Fused representation
        """
        # Project all to output dimension
        s = self.seq_proj(seq)
        p = self.prop_proj(prop)
        st = self.struct_proj(struct)
        ss_out = self.ss_proj(ss)
        
        # Stack projections: [B, L, 4, D]
        stacked = torch.stack([s, p, st, ss_out], dim=2)
        
        # Compute gating weights from sum (more efficient than concat)
        # Use mean pooled features for gate computation
        gate_input = stacked.mean(dim=2)  # [B, L, D] - aggregate of all streams
        
        # Alternative: use separate gate input computation for better expressiveness
        # This uses a reshape instead of concat (same memory but clearer)
        gate_input = stacked.view(*stacked.shape[:2], -1)  # [B, L, 4*D]
        gates = self.gate_net(gate_input)  # [B, L, 4]
        
        # Apply structure mask if provided (downweight structure where unavailable)
        if struct_mask is not None:
            # Reduce structure gate where mask is 0
            gates = gates.clone()
            gates[..., 2] = gates[..., 2] * struct_mask
        
        # Softmax over modalities
        gates = F.softmax(gates, dim=-1)  # [B, L, 4]
        
        # Weighted combination: [B, L, 4, D] * [B, L, 4, 1] -> [B, L, 4, D] -> [B, L, D]
        output = (stacked * gates.unsqueeze(-1)).sum(dim=2)
        
        return self.layer_norm(output)


class MultiStreamEncoder(nn.Module):
    """
    Complete multi-stream encoder with cross-modal attention.
    
    Processes multiple modality streams through N fusion blocks,
    then combines them with gated fusion.
    
    Args:
        config: Configuration dictionary or object with stream dimensions
    """
    
    def __init__(
        self,
        seq_dim: int = 384,
        prop_dim: int = 128,
        struct_dim: int = 192,
        ss_dim: int = 64,
        output_dim: int = 512,
        num_fusion_blocks: int = 6,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.prop_dim = prop_dim
        self.struct_dim = struct_dim
        self.ss_dim = ss_dim
        self.output_dim = output_dim
        
        # Stack of fusion blocks
        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(
                seq_dim=seq_dim,
                prop_dim=prop_dim,
                struct_dim=struct_dim,
                ss_dim=ss_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dropout=dropout
            )
            for _ in range(num_fusion_blocks)
        ])
        
        # Final fusion
        self.gated_fusion = GatedFusion(
            seq_dim=seq_dim,
            prop_dim=prop_dim,
            struct_dim=struct_dim,
            ss_dim=ss_dim,
            output_dim=output_dim
        )
    
    def forward(
        self,
        seq: torch.Tensor,
        prop: torch.Tensor,
        struct: torch.Tensor,
        ss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        struct_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process all streams through fusion blocks.
        
        Returns:
            output: [B, L, output_dim] Final fused representation
            stream_outputs: Dict with final state of each stream
        """
        # Process through fusion blocks
        for block in self.fusion_blocks:
            seq, prop, struct, ss = block(seq, prop, struct, ss, mask)
        
        # Gated fusion
        output = self.gated_fusion(seq, prop, struct, ss, struct_mask)
        
        return output, {
            'seq': seq,
            'prop': prop,
            'struct': struct,
            'ss': ss
        }
