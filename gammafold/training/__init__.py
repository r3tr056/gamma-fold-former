"""
GammaFold Training Module.

Provides training infrastructure:
- Trainer: Main training loop
- TrainingConfig: Configuration
- Schedulers: LR schedules
- Losses: Combined loss functions
- Checkpointing: Model checkpoints
- Metrics: Tracking and logging
- Callbacks: Training hooks
"""

from gammafold.training.config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from gammafold.training.trainer import Trainer
from gammafold.training.schedulers import (
    get_scheduler,
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    WarmupConstantScheduler,
    get_cosine_schedule_with_warmup,
)
from gammafold.training.losses import (
    CombinedLoss,
    MLMLoss,
    StructureLoss,
    DistanceMatrixLoss,
    LossOutput,
    compute_accuracy,
)
from gammafold.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    is_main_process,
    get_world_size,
    get_rank,
)
from gammafold.training.checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
    save_model_for_inference,
    load_model_for_inference,
)
from gammafold.training.metrics import (
    MetricTracker,
    MetricLogger,
    EarlyStopping,
)
from gammafold.training.callbacks import (
    TrainingCallback,
    CallbackHandler,
    LoggingCallback,
    GradientMonitorCallback,
    ModelEMACallback,
    MemoryMonitorCallback,
    create_default_callbacks,
)

__all__ = [
    # Config
    'TrainingConfig',
    'ModelConfig', 
    'DataConfig',
    'OptimizerConfig',
    'SchedulerConfig',
    
    # Trainer
    'Trainer',
    
    # Schedulers
    'get_scheduler',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler',
    'WarmupConstantScheduler',
    'get_cosine_schedule_with_warmup',
    
    # Losses
    'CombinedLoss',
    'MLMLoss',
    'StructureLoss',
    'DistanceMatrixLoss',
    'LossOutput',
    'compute_accuracy',
    
    # Distributed
    'setup_distributed',
    'cleanup_distributed',
    'wrap_model_ddp',
    'is_main_process',
    'get_world_size',
    'get_rank',
    
    # Checkpointing
    'CheckpointManager',
    'CheckpointMetadata',
    'save_model_for_inference',
    'load_model_for_inference',
    
    # Metrics
    'MetricTracker',
    'MetricLogger',
    'EarlyStopping',
    
    # Callbacks
    'TrainingCallback',
    'CallbackHandler',
    'LoggingCallback',
    'GradientMonitorCallback',
    'ModelEMACallback',
    'MemoryMonitorCallback',
    'create_default_callbacks',
]

