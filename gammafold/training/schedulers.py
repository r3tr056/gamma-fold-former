"""
Learning rate schedulers for GammaFold training.
"""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine decay.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear warmup followed by linear decay.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = max(self.min_lr_ratio, 1 - progress * (1 - self.min_lr_ratio))
        
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmupConstantScheduler(_LRScheduler):
    """
    Linear warmup followed by constant LR.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            scale = 1.0
        
        return [base_lr * scale for base_lr in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> _LRScheduler:
    """
    Get a learning rate scheduler by name.
    
    Args:
        name: Scheduler name ("cosine", "linear", "constant")
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR ratio for decay
        
    Returns:
        Learning rate scheduler
    """
    schedulers = {
        'cosine': WarmupCosineScheduler,
        'linear': WarmupLinearScheduler,
        'constant': WarmupConstantScheduler,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Choose from: {list(schedulers.keys())}")
    
    if name == 'constant':
        return schedulers[name](optimizer, warmup_steps)
    else:
        return schedulers[name](optimizer, warmup_steps, total_steps, min_lr_ratio)


def get_linear_warmup_lambda(warmup_steps: int):
    """Get lambda function for linear warmup with LambdaLR."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.
    
    Compatible with HuggingFace Transformers API.
    
    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch
        
    Returns:
        LambdaLR with cosine schedule
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
