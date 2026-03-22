"""
Callback system for GammaFold training.

Provides extensible hooks for training events:
- on_train_begin / on_train_end
- on_epoch_begin / on_epoch_end
- on_step_begin / on_step_end
- on_validation_begin / on_validation_end
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """
    Base class for training callbacks.
    
    Override methods to add custom behavior at training events.
    """
    
    def on_train_begin(self, trainer, **kwargs):
        """Called when training starts."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called when training ends."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, trainer, step: int, **kwargs):
        """Called at the start of each training step."""
        pass
    
    def on_step_end(self, trainer, step: int, loss: float, metrics: Dict[str, float], **kwargs):
        """Called at the end of each training step."""
        pass
    
    def on_validation_begin(self, trainer, **kwargs):
        """Called before validation."""
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called after validation."""
        pass
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str, **kwargs):
        """Called when a checkpoint is saved."""
        pass
    
    def on_exception(self, trainer, exception: Exception, **kwargs):
        """Called when an exception occurs during training."""
        pass


class CallbackHandler:
    """
    Manages and dispatches callbacks.
    """
    
    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: TrainingCallback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: TrainingCallback):
        """Remove a callback."""
        self.callbacks.remove(callback)
    
    def fire(self, event: str, *args, **kwargs):
        """Fire an event to all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback {callback.__class__.__name__}.{event} failed: {e}")


class LoggingCallback(TrainingCallback):
    """
    Logs training progress to console.
    """
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
    
    def on_step_end(self, trainer, step: int, loss: float, metrics: Dict[str, float], **kwargs):
        if step % self.log_every == 0:
            lr = metrics.get('learning_rate', 0)
            logger.info(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
    
    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        loss = metrics.get('loss', 0)
        acc = metrics.get('accuracy', 0)
        logger.info(f"Validation: loss={loss:.4f}, accuracy={acc:.4f}")


class GradientMonitorCallback(TrainingCallback):
    """
    Monitors gradient statistics during training.
    """
    
    def __init__(self, log_every: int = 500):
        self.log_every = log_every
    
    def on_step_end(self, trainer, step: int, loss: float, metrics: Dict[str, float], **kwargs):
        if step % self.log_every != 0:
            return
        
        total_norm = 0.0
        max_norm = 0.0
        
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
        
        total_norm = total_norm ** 0.5
        logger.info(f"Gradient norm: total={total_norm:.4f}, max={max_norm:.4f}")


class LRSchedulerCallback(TrainingCallback):
    """
    Custom learning rate scheduling callback.
    """
    
    def __init__(self, warmup_steps: int = 1000, decay_factor: float = 0.1):
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
    
    def on_step_end(self, trainer, step: int, **kwargs):
        # Custom LR logic can be added here
        pass


class ModelEMACallback(TrainingCallback):
    """
    Exponential Moving Average of model weights.
    
    Maintains an EMA copy of the model for better generalization.
    """
    
    def __init__(self, decay: float = 0.999, start_step: int = 0):
        self.decay = decay
        self.start_step = start_step
        self.ema_model = None
    
    def on_train_begin(self, trainer, **kwargs):
        import copy
        self.ema_model = copy.deepcopy(trainer.model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
    
    def on_step_end(self, trainer, step: int, **kwargs):
        if step < self.start_step:
            return
        
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), trainer.model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    
    def get_ema_model(self):
        """Get the EMA model."""
        return self.ema_model


class MemoryMonitorCallback(TrainingCallback):
    """
    Monitors GPU memory usage.
    """
    
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
    
    def on_step_end(self, trainer, step: int, **kwargs):
        if step % self.log_every != 0:
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        except:
            pass


# Import torch for EMA callback
import torch


def create_default_callbacks(
    log_every: int = 100,
    monitor_gradients: bool = False,
    monitor_memory: bool = False
) -> List[TrainingCallback]:
    """
    Create a default set of callbacks.
    
    Args:
        log_every: Logging frequency
        monitor_gradients: Enable gradient monitoring
        monitor_memory: Enable memory monitoring
        
    Returns:
        List of callbacks
    """
    callbacks = [LoggingCallback(log_every)]
    
    if monitor_gradients:
        callbacks.append(GradientMonitorCallback(log_every * 5))
    
    if monitor_memory:
        callbacks.append(MemoryMonitorCallback(log_every))
    
    return callbacks
