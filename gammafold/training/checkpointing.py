"""
Checkpointing utilities for GammaFold training.

Features:
- Model checkpointing with metadata
- Best model tracking
- Resume training from checkpoint
- Checkpoint management (cleanup old checkpoints)
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    step: int
    epoch: int
    loss: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    learning_rate: float
    timestamp: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save/load checkpoints
    - Keep best N checkpoints
    - Track training history
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        best_metric: str = "val_loss",
        mode: str = "min"
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            best_metric: Metric to track for best checkpoint
            mode: "min" or "max" for best metric selection
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.mode = mode
        
        self.checkpoints: List[str] = []
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_path: Optional[str] = None
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Scan directory for existing checkpoints."""
        if not self.checkpoint_dir.exists():
            return
        
        for f in sorted(self.checkpoint_dir.glob("checkpoint_*.pt")):
            self.checkpoints.append(str(f))
        
        # Load best checkpoint info
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            self.best_path = str(best_path)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """
        Save a checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
        }
        
        # Create metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            loss=metrics.get('loss', 0.0),
            val_loss=metrics.get('val_loss'),
            val_accuracy=metrics.get('val_accuracy'),
            learning_rate=metrics.get('learning_rate', 0.0),
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            training_config=training_config
        )
        checkpoint['metadata'] = metadata.to_dict()
        
        # Save checkpoint
        filename = f"checkpoint_{step:08d}.pt"
        checkpoint_path = str(self.checkpoint_dir / filename)
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Check if this is best
        metric_value = metrics.get(self.best_metric)
        if metric_value is not None:
            is_best = (self.mode == "min" and metric_value < self.best_value) or \
                      (self.mode == "max" and metric_value > self.best_value)
            
            if is_best:
                self.best_value = metric_value
                self._save_best(checkpoint_path)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return checkpoint_path
    
    def _save_best(self, checkpoint_path: str):
        """Copy checkpoint as best."""
        best_path = str(self.checkpoint_dir / "best.pt")
        shutil.copy(checkpoint_path, best_path)
        self.best_path = best_path
        logger.info(f"New best checkpoint: {self.best_metric}={self.best_value:.4f}")
    
    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)
                logger.debug(f"Removed old checkpoint: {oldest}")
    
    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler=None,
        map_location: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Returns:
            Checkpoint metadata and state
        """
        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'metadata': checkpoint.get('metadata', {})
        }
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        scaler=None,
        map_location: str = "cpu"
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        if self.best_path is None:
            raise ValueError("No best checkpoint available")
        return self.load(self.best_path, model, optimizer, scheduler, scaler, map_location)
    
    def get_latest_path(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1]
        return None


def save_model_for_inference(
    model: nn.Module,
    save_path: str,
    model_config: Dict[str, Any]
):
    """
    Save model for inference only (no optimizer state).
    
    Args:
        model: Trained model
        save_path: Output path
        model_config: Model configuration
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }, save_path)
    logger.info(f"Saved inference model: {save_path}")


def load_model_for_inference(
    model_class,
    checkpoint_path: str,
    device: str = "cpu"
) -> nn.Module:
    """
    Load model for inference.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to checkpoint
        device: Target device
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config and create model
    config = checkpoint.get('model_config', {})
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
