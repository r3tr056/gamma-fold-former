"""
Training configuration for GammaFold.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "original"  # "original" or "multimodal"
    model_size: str = "small"     # "small", "medium", "large"
    
    # Override specific model parameters
    embed_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    dropout: Optional[float] = None


@dataclass
class DataConfig:
    """Data configuration."""
    train_data: str = "data/train"
    val_data: Optional[str] = "data/val"
    test_data: Optional[str] = None
    
    max_seq_len: int = 512
    min_seq_len: int = 10
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Gradient settings
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0  # Alternative: fraction of total steps
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of max LR


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Training parameters
    max_steps: int = 100000
    max_epochs: Optional[int] = None  # If set, takes precedence over max_steps
    eval_steps: int = 1000
    save_steps: int = 5000
    log_steps: int = 100
    
    # Mixed precision
    mixed_precision: bool = True
    bf16: bool = False  # Use bfloat16 instead of float16
    
    # Checkpointing
    output_dir: str = "outputs"
    save_total_limit: int = 3
    resume_from: Optional[str] = None
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Logging
    logging_dir: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "gammafold"
    wandb_run_name: Optional[str] = None
    
    # Loss weights
    mlm_loss_weight: float = 1.0
    structure_loss_weight: float = 1.0
    distance_loss_weight: float = 0.1
    
    # Misc
    seed: int = 42
    deterministic: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'optimizer': asdict(self.optimizer),
            'scheduler': asdict(self.scheduler),
            'max_steps': self.max_steps,
            'max_epochs': self.max_epochs,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'log_steps': self.log_steps,
            'mixed_precision': self.mixed_precision,
            'bf16': self.bf16,
            'output_dir': self.output_dir,
            'save_total_limit': self.save_total_limit,
            'resume_from': self.resume_from,
            'distributed': self.distributed,
            'logging_dir': self.logging_dir,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'mlm_loss_weight': self.mlm_loss_weight,
            'structure_loss_weight': self.structure_loss_weight,
            'distance_loss_weight': self.distance_loss_weight,
            'seed': self.seed,
        }
    
    def save(self, path: str):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        model = ModelConfig(**data.pop('model', {}))
        data_cfg = DataConfig(**data.pop('data', {}))
        optimizer = OptimizerConfig(**data.pop('optimizer', {}))
        scheduler = SchedulerConfig(**data.pop('scheduler', {}))
        
        return cls(
            model=model,
            data=data_cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            **data
        )
    
    @classmethod
    def small(cls) -> 'TrainingConfig':
        """Config for small model training."""
        return cls(
            model=ModelConfig(model_size="small"),
            data=DataConfig(batch_size=64, max_seq_len=256),
            optimizer=OptimizerConfig(learning_rate=3e-4),
            max_steps=50000,
        )
    
    @classmethod
    def medium(cls) -> 'TrainingConfig':
        """Config for medium model training."""
        return cls(
            model=ModelConfig(model_size="medium"),
            data=DataConfig(batch_size=32, max_seq_len=512),
            optimizer=OptimizerConfig(learning_rate=1e-4),
            max_steps=100000,
        )
    
    @classmethod
    def large(cls) -> 'TrainingConfig':
        """Config for large model training."""
        return cls(
            model=ModelConfig(model_size="large"),
            data=DataConfig(batch_size=16, max_seq_len=512),
            optimizer=OptimizerConfig(
                learning_rate=5e-5,
                gradient_accumulation_steps=4
            ),
            max_steps=200000,
        )
