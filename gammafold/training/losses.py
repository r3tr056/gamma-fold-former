"""
Loss functions for GammaFold training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Container for loss components."""
    total: torch.Tensor
    mlm: Optional[torch.Tensor] = None
    structure: Optional[torch.Tensor] = None
    distance: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of scalar values."""
        result = {'loss': self.total.item()}
        if self.mlm is not None:
            result['mlm_loss'] = self.mlm.item()
        if self.structure is not None:
            result['structure_loss'] = self.structure.item()
        if self.distance is not None:
            result['distance_loss'] = self.distance.item()
        return result


class MLMLoss(nn.Module):
    """
    Masked Language Modeling loss.
    
    Uses cross-entropy loss on masked positions.
    """
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MLM loss.
        
        Args:
            logits: [B, L, vocab_size] Predicted logits
            labels: [B, L] Target labels (-100 for non-masked positions)
            
        Returns:
            Scalar loss value
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        return self.criterion(logits_flat, labels_flat)


class StructureLoss(nn.Module):
    """
    Structure prediction loss.
    
    Uses MSE loss on predicted coordinates with masking.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute structure loss.
        
        Args:
            pred_coords: [B, L, 3] Predicted Cα coordinates
            target_coords: [B, L, 3] Target Cα coordinates
            mask: [B, L] Optional mask (1=valid, 0=ignore)
            
        Returns:
            Scalar loss value
        """
        # Compute squared differences
        sq_diff = (pred_coords - target_coords).pow(2).sum(dim=-1)  # [B, L]
        
        if mask is not None:
            # Apply mask
            sq_diff = sq_diff * mask
            if self.reduction == 'mean':
                loss = sq_diff.sum() / (mask.sum() + 1e-8)
            else:
                loss = sq_diff.sum()
        else:
            if self.reduction == 'mean':
                loss = sq_diff.mean()
            else:
                loss = sq_diff.sum()
        
        return loss


class DistanceMatrixLoss(nn.Module):
    """
    Distance matrix prediction loss.
    
    Uses MSE on predicted vs actual pairwise distances.
    """
    
    def __init__(self, max_dist: float = 20.0):
        super().__init__()
        self.max_dist = max_dist
    
    def forward(
        self,
        pred_dist: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distance matrix loss.
        
        Args:
            pred_dist: [B, L, L] Predicted distances
            coords: [B, L, 3] Target Cα coordinates
            mask: [B, L] Optional mask
            
        Returns:
            Scalar loss value
        """
        # Compute target distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, L, L, 3]
        target_dist = diff.pow(2).sum(dim=-1).sqrt()  # [B, L, L]
        
        # Clamp for numerical stability
        target_dist = target_dist.clamp(max=self.max_dist)
        
        # Compute loss
        sq_diff = (pred_dist - target_dist).pow(2)
        
        if mask is not None:
            # Create 2D mask
            mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, L, L]
            sq_diff = sq_diff * mask_2d
            loss = sq_diff.sum() / (mask_2d.sum() + 1e-8)
        else:
            loss = sq_diff.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for GammaFold training.
    
    Combines MLM, structure, and optional distance losses.
    """
    
    def __init__(
        self,
        mlm_weight: float = 1.0,
        structure_weight: float = 1.0,
        distance_weight: float = 0.1,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.mlm_weight = mlm_weight
        self.structure_weight = structure_weight
        self.distance_weight = distance_weight
        
        self.mlm_loss = MLMLoss(label_smoothing=label_smoothing)
        self.structure_loss = StructureLoss()
        self.distance_loss = DistanceMatrixLoss()
    
    def forward(
        self,
        mlm_logits: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        pred_coords: Optional[torch.Tensor] = None,
        target_coords: Optional[torch.Tensor] = None,
        pred_dist: Optional[torch.Tensor] = None,
        coord_mask: Optional[torch.Tensor] = None
    ) -> LossOutput:
        """
        Compute combined loss.
        
        Args:
            mlm_logits: MLM prediction logits
            mlm_labels: MLM target labels
            pred_coords: Predicted coordinates
            target_coords: Target coordinates
            pred_dist: Predicted distance matrix
            coord_mask: Coordinate validity mask
            
        Returns:
            LossOutput with total and component losses
        """
        total_loss = torch.tensor(0.0, device=self._get_device(mlm_logits, pred_coords))
        mlm_loss_val = None
        structure_loss_val = None
        distance_loss_val = None
        
        # MLM loss
        if mlm_logits is not None and mlm_labels is not None:
            mlm_loss_val = self.mlm_loss(mlm_logits, mlm_labels)
            total_loss = total_loss + self.mlm_weight * mlm_loss_val
        
        # Structure loss
        if pred_coords is not None and target_coords is not None:
            structure_loss_val = self.structure_loss(pred_coords, target_coords, coord_mask)
            total_loss = total_loss + self.structure_weight * structure_loss_val
        
        # Distance loss
        if pred_dist is not None and target_coords is not None:
            distance_loss_val = self.distance_loss(pred_dist, target_coords, coord_mask)
            total_loss = total_loss + self.distance_weight * distance_loss_val
        
        return LossOutput(
            total=total_loss,
            mlm=mlm_loss_val,
            structure=structure_loss_val,
            distance=distance_loss_val
        )
    
    def _get_device(self, *tensors):
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device('cpu')


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute prediction accuracy, ignoring specified indices.
    
    Args:
        logits: [B, L, V] Prediction logits
        labels: [B, L] Target labels
        ignore_index: Index to ignore
        
    Returns:
        Accuracy (0-1)
    """
    predictions = logits.argmax(dim=-1)  # [B, L]
    mask = labels != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == labels) & mask
    return correct.sum().float() / mask.sum().float()
