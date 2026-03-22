"""
Metrics tracking and logging for GammaFold training.

Features:
- Metric accumulation and averaging
- TensorBoard integration
- WandB integration
- CSV logging
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class MetricTracker:
    """
    Tracks and accumulates metrics during training.
    
    Features:
    - Running average computation
    - Step-wise and epoch-wise tracking
    - Metric history
    """
    
    _values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _step_values: Dict[str, float] = field(default_factory=dict)
    _history: Dict[str, List[tuple]] = field(default_factory=lambda: defaultdict(list))
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number for history
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._values[name].append(value)
                self._step_values[name] = value
                if step is not None:
                    self._history[name].append((step, value))
    
    def get(self, name: str) -> float:
        """Get last value for a metric."""
        return self._step_values.get(name, 0.0)
    
    def get_average(self, name: str) -> float:
        """Get running average for a metric."""
        values = self._values.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all running averages."""
        return {name: self.get_average(name) for name in self._values}
    
    def reset(self):
        """Reset accumulated values (keep history)."""
        self._values = defaultdict(list)
        self._step_values = {}
    
    def get_history(self, name: str) -> List[tuple]:
        """Get history for a metric as (step, value) pairs."""
        return self._history.get(name, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            'averages': self.get_all_averages(),
            'last_values': dict(self._step_values),
            'history': {k: list(v) for k, v in self._history.items()}
        }


class MetricLogger:
    """
    Logs metrics to various backends.
    
    Supports:
    - Console logging
    - TensorBoard
    - Weights & Biases
    - CSV files
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        use_csv: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = MetricTracker()
        self.start_time = time.time()
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
                logger.info(f"TensorBoard logging to {self.log_dir / 'tensorboard'}")
            except ImportError:
                logger.warning("TensorBoard not available")
        
        # WandB
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "gammafold",
                    name=wandb_run_name,
                    config=wandb_config or {},
                    dir=str(self.log_dir)
                )
                logger.info(f"W&B logging enabled: {wandb_project}")
            except ImportError:
                logger.warning("W&B not available")
        
        # CSV
        self.csv_path = None
        self.csv_writer = None
        self.csv_file = None
        if use_csv:
            self.csv_path = str(self.log_dir / "metrics.csv")
            self._csv_headers_written = False
    
    def log(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Metric name -> value
            step: Current step
            prefix: Optional prefix for metric names
        """
        # Apply prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Update tracker
        self.tracker.update(metrics, step)
        
        # TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        # WandB
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)
        
        # CSV
        if self.csv_path:
            self._log_csv(metrics, step)
    
    def _log_csv(self, metrics: Dict[str, float], step: int):
        """Log to CSV file."""
        row = {'step': step, 'time': time.time() - self.start_time, **metrics}
        
        mode = 'a' if self._csv_headers_written else 'w'
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_headers_written:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow(row)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.tb_writer:
            self.tb_writer.add_hparams(params, {})
        
        # Save to file
        params_path = self.log_dir / "hyperparams.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
    
    def log_model_summary(self, model, input_shape: Optional[tuple] = None):
        """Log model architecture summary."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': model.__class__.__name__,
        }
        
        summary_path = self.log_dir / "model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Model: {total_params/1e6:.2f}M parameters ({trainable_params/1e6:.2f}M trainable)")
    
    def close(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        # Save final metrics
        metrics_path = self.log_dir / "final_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.tracker.to_dict(), f, indent=2)


class EarlyStopping:
    """
    Early stopping handler.
    
    Stops training if monitored metric doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True
    ):
        """
        Args:
            patience: Number of checks without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
            restore_best: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_state = None
        self.should_stop = False
    
    def __call__(self, value: float, model=None) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            model: Optional model to save best state
            
        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
            if model is not None and self.restore_best:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} checks without improvement")
        
        return self.should_stop
    
    def restore(self, model):
        """Restore best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info("Restored best model weights")
