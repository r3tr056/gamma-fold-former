"""
VQ-VAE Structure Tokenizer for GammaFold.

Discretizes 3D protein structure into tokens using:
- SE(3)-aware local geometry encoding
- Vector quantization with learnable codebook
- Rotation-invariant feature extraction

OPTIMIZED: Vectorized feature extraction (no Python loops)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StructureTokenizer(nn.Module):
    """
    VQ-VAE based tokenizer for 3D protein structure.
    
    Encodes local 3D geometry around each residue into discrete tokens
    from a learned codebook. Uses rotation-invariant features.
    
    Architecture:
    1. Extract local geometry features (distances, angles) - VECTORIZED
    2. Encode to continuous latent space
    3. Quantize to nearest codebook vector
    4. Decode back to geometry (for training)
    
    Args:
        coord_dim: Coordinate dimensions (3 for Cα only)
        hidden_dim: Encoder/decoder hidden dimension
        codebook_size: Number of codes in VQ codebook (default 4096)
        codebook_dim: Dimension of each code vector
        window_size: Local context window for feature extraction
        commitment_cost: Commitment loss weight for VQ
    """
    
    def __init__(
        self,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        codebook_size: int = 4096,
        codebook_dim: int = 64,
        window_size: int = 9,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.window_size = window_size
        self.commitment_cost = commitment_cost
        
        # Input dimension: local coordinates + pairwise distances + angles
        # Distances: window_size pairs, Angles: window_size-2 triplets
        self.feature_dim = self._compute_feature_dim()
        
        # Encoder: local geometry → continuous latent
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_dim)
        )
        
        # Vector Quantization codebook
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        # Initialize codebook with uniform distribution
        nn.init.uniform_(
            self.codebook.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size
        )
        
        # Decoder: quantized latent → local geometry
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, coord_dim)  # Predict local offset
        )
        
        # EMA updates for codebook (optional, for training stability)
        self.register_buffer('codebook_usage', torch.zeros(codebook_size))
    
    def _compute_feature_dim(self) -> int:
        """Compute input feature dimension."""
        # Pairwise distances within window
        num_distances = self.window_size
        # Local angles (between consecutive triplets)
        num_angles = max(0, self.window_size - 2)
        # Dihedral angles
        num_dihedrals = max(0, self.window_size - 3)
        # Distance to center
        center_distances = self.window_size
        
        return num_distances + num_angles + num_dihedrals + center_distances
    
    def extract_features(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract rotation-invariant local features from coordinates.
        VECTORIZED implementation - no Python loops over sequence positions.
        
        Args:
            coords: [B, L, 3] Cα coordinates
            mask: [B, L] Optional validity mask (1=valid, 0=invalid)
            
        Returns:
            features: [B, L, feature_dim] Rotation-invariant features
        """
        B, L, D = coords.shape
        device = coords.device
        pad = self.window_size // 2
        
        # Pad coordinates for windowing using replicate padding
        # [B, L, D] -> [B, D, L] -> pad -> [B, D, L+2*pad] -> [B, L+2*pad, D]
        coords_padded = F.pad(
            coords.transpose(1, 2),
            (pad, pad),
            mode='replicate'
        ).transpose(1, 2)  # [B, L + 2*pad, D]
        
        # Create sliding windows: [B, L, window_size, D]
        # Use unfold to get all windows at once
        windows = coords_padded.unfold(1, self.window_size, 1)  # [B, L, D, W]
        windows = windows.permute(0, 1, 3, 2)  # [B, L, W, D]
        
        # Get center coordinates for each position
        center_idx = self.window_size // 2
        centers = windows[:, :, center_idx:center_idx+1, :]  # [B, L, 1, D]
        
        # Feature 1: Distances to center (vectorized)
        centered = windows - centers  # [B, L, W, D]
        distances_to_center = torch.norm(centered, dim=-1)  # [B, L, W]
        
        # Feature 2: Consecutive pairwise distances (vectorized)
        # Distance between position i and i+1 in window
        window_diffs = windows[:, :, 1:, :] - windows[:, :, :-1, :]  # [B, L, W-1, D]
        pairwise_dists = torch.norm(window_diffs, dim=-1)  # [B, L, W-1]
        # Pad to window_size
        pairwise_dists = F.pad(pairwise_dists, (0, 1), value=0.0)  # [B, L, W]
        
        # Feature 3: Bond angles (cosine of angle at middle point of triplets)
        # For triplets (i, i+1, i+2), compute angle at i+1
        if self.window_size >= 3:
            v1 = windows[:, :, :-2, :] - windows[:, :, 1:-1, :]  # [B, L, W-2, D]
            v2 = windows[:, :, 2:, :] - windows[:, :, 1:-1, :]   # [B, L, W-2, D]
            
            # Normalize vectors
            v1_norm = F.normalize(v1, dim=-1)
            v2_norm = F.normalize(v2, dim=-1)
            
            # Cosine of angles
            angles = (v1_norm * v2_norm).sum(dim=-1)  # [B, L, W-2]
        else:
            angles = torch.zeros(B, L, 0, device=device)
        
        # Feature 4: Dihedral angles (for quadruplets)
        if self.window_size >= 4:
            # Points: p1, p2, p3, p4
            p1 = windows[:, :, :-3, :]  # [B, L, W-3, D]
            p2 = windows[:, :, 1:-2, :]
            p3 = windows[:, :, 2:-1, :]
            p4 = windows[:, :, 3:, :]
            
            # Vectors along the backbone
            b1 = p2 - p1  # [B, L, W-3, D]
            b2 = p3 - p2
            b3 = p4 - p3
            
            # Normal vectors to planes
            n1 = torch.cross(b1, b2, dim=-1)  # [B, L, W-3, D]
            n2 = torch.cross(b2, b3, dim=-1)
            
            # Normalize
            n1 = F.normalize(n1, dim=-1)
            n2 = F.normalize(n2, dim=-1)
            
            # Cosine of dihedral angle
            dihedrals = (n1 * n2).sum(dim=-1)  # [B, L, W-3]
        else:
            dihedrals = torch.zeros(B, L, 0, device=device)
        
        # Concatenate all features
        features = torch.cat([
            distances_to_center,  # [B, L, W]
            pairwise_dists,       # [B, L, W]
            angles,               # [B, L, W-2]
            dihedrals             # [B, L, W-3]
        ], dim=-1)  # [B, L, feature_dim]
        
        # Handle NaN (from degenerate geometry like zero-length vectors)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply mask if provided (zero out features for invalid positions)
        if mask is not None:
            features = features * mask.unsqueeze(-1)
        
        return features
    
    def encode(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode 3D coordinates to continuous latent space.
        
        Args:
            coords: [B, L, 3] Cα coordinates
            mask: [B, L] Optional validity mask
            
        Returns:
            z: [B, L, codebook_dim] Continuous latent vectors
        """
        features = self.extract_features(coords, mask)
        z = self.encoder(features)
        return z
    
    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization: map continuous latent to discrete codes.
        
        Uses efficient distance computation with torch.cdist.
        
        Args:
            z: [B, L, codebook_dim] Continuous latent vectors
            
        Returns:
            z_q: [B, L, codebook_dim] Quantized vectors
            indices: [B, L] Codebook indices
            vq_loss: Scalar VQ loss
        """
        B, L, D = z.shape
        
        # Flatten for distance computation
        z_flat = z.reshape(-1, D)  # [B*L, D]
        
        # Compute distances using torch.cdist (more efficient)
        distances = torch.cdist(z_flat, self.codebook.weight, p=2).pow(2)  # [B*L, codebook_size]
        
        # Find nearest codebook vector
        indices = distances.argmin(dim=-1)  # [B*L]
        indices = indices.reshape(B, L)  # [B, L]
        
        # Get quantized vectors
        z_q = self.codebook(indices)  # [B, L, D]
        
        # VQ losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Update codebook usage (for monitoring)
        if self.training:
            with torch.no_grad():
                unique_indices = indices.unique()
                self.codebook_usage[unique_indices] += 1
        
        return z_q, indices, vq_loss
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to local geometry.
        
        Args:
            z_q: [B, L, codebook_dim] Quantized latent vectors
            
        Returns:
            coords_pred: [B, L, 3] Predicted coordinate offsets
        """
        return self.decoder(z_q)
    
    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → quantize → decode.
        
        Args:
            coords: [B, L, 3] Cα coordinates
            mask: [B, L] Optional validity mask
            
        Returns:
            tokens: [B, L] Structure tokens (codebook indices)
            z_q: [B, L, codebook_dim] Quantized latent
            coords_pred: [B, L, 3] Reconstructed coordinates
            vq_loss: Scalar VQ loss
        """
        # Encode
        z = self.encode(coords, mask)
        
        # Quantize
        z_q, tokens, vq_loss = self.quantize(z)
        
        # Decode
        coords_pred = self.decode(z_q)
        
        return tokens, z_q, coords_pred, vq_loss
    
    def tokenize(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tokenize coordinates (encode + quantize only).
        
        Args:
            coords: [B, L, 3] Cα coordinates
            mask: [B, L] Optional validity mask
            
        Returns:
            tokens: [B, L] Structure tokens
        """
        z = self.encode(coords, mask)
        _, tokens, _ = self.quantize(z)
        return tokens
    
    def get_codebook_utilization(self) -> float:
        """Return fraction of codebook that has been used."""
        return (self.codebook_usage > 0).float().mean().item()
    
    def reset_usage_stats(self):
        """Reset codebook usage statistics."""
        self.codebook_usage.zero_()
