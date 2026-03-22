"""
Main Trainer class for GammaFold.

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Checkpointing
- TensorBoard/WandB logging
- Distributed training support
"""

import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from gammafold.training.config import TrainingConfig
from gammafold.training.schedulers import get_scheduler
from gammafold.training.losses import CombinedLoss, compute_accuracy
from gammafold.models.gamma_fold import create_model, create_multimodal_model

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer for GammaFold models.
    
    Args:
        config: Training configuration
        model: Optional pre-created model (if None, creates from config)
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup
        self._setup_seed()
        self._setup_device()
        self._setup_output_dir()
        
        # Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._create_model()
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_fn()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = None
        self._setup_logging()
    
    def _setup_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_device(self):
        """Setup training device."""
        if torch.cuda.is_available():
            if self.config.local_rank >= 0:
                self.device = torch.device(f'cuda:{self.config.local_rank}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Training on device: {self.device}")
    
    def _setup_output_dir(self):
        """Create output directory."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, 'config.yaml')
        self.config.save(config_path)
    
    def _setup_logging(self):
        """Setup TensorBoard or WandB logging."""
        log_dir = self.config.logging_dir or os.path.join(self.config.output_dir, 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping logging setup")
        
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.to_dict()
                )
                logger.info(f"WandB logging enabled: {self.config.wandb_project}")
            except ImportError:
                logger.warning("WandB not available")
    
    def _create_model(self) -> nn.Module:
        """Create model from config."""
        model_cfg = self.config.model
        
        if model_cfg.model_type == "multimodal":
            model = create_multimodal_model(model_cfg.model_size)
        else:
            model = create_model(model_cfg.model_size)
        
        model = model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Created {model_cfg.model_type} model ({model_cfg.model_size}): {num_params/1e6:.1f}M parameters")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_cfg = self.config.optimizer
        
        # Separate weight decay for different param groups
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': opt_cfg.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            params,
            lr=opt_cfg.learning_rate,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_cfg = self.config.scheduler
        
        # Calculate warmup steps
        if sched_cfg.warmup_ratio > 0:
            warmup_steps = int(self.config.max_steps * sched_cfg.warmup_ratio)
        else:
            warmup_steps = sched_cfg.warmup_steps
        
        return get_scheduler(
            name=sched_cfg.scheduler,
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.config.max_steps,
            min_lr_ratio=sched_cfg.min_lr_ratio
        )
    
    def _create_loss_fn(self) -> CombinedLoss:
        """Create loss function."""
        return CombinedLoss(
            mlm_weight=self.config.mlm_loss_weight,
            structure_weight=self.config.structure_loss_weight,
            distance_weight=self.config.distance_loss_weight
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training statistics
        """
        logger.info("Starting training...")
        
        if self.train_dataloader is None:
            raise ValueError("No training dataloader provided")
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        
        total_loss = 0.0
        step_times = []
        
        while self.global_step < self.config.max_steps:
            step_start = time.time()
            
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            # Training step
            loss, metrics = self.train_step(batch)
            total_loss += loss
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Logging
            if self.global_step % self.config.log_steps == 0:
                avg_loss = total_loss / self.config.log_steps
                avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                lr = self.scheduler.get_last_lr()[0]
                
                logger.info(
                    f"Step {self.global_step}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Time: {avg_time*1000:.1f}ms"
                )
                
                self._log_metrics({
                    'train/loss': avg_loss,
                    'train/learning_rate': lr,
                    'train/step_time': avg_time,
                    **{f'train/{k}': v for k, v in metrics.items()}
                })
                
                total_loss = 0.0
            
            # Evaluation
            if self.val_dataloader and self.global_step % self.config.eval_steps == 0:
                val_metrics = self.evaluate()
                self._log_metrics({f'val/{k}': v for k, v in val_metrics.items()})
                
                # Save best model
                if val_metrics.get('loss', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best')
                
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f'step_{self.global_step}')
        
        # Final save
        self.save_checkpoint('final')
        
        logger.info("Training complete!")
        return {'final_step': self.global_step, 'best_val_loss': self.best_val_loss}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Tuple of (loss value, metrics dict)
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Create MLM targets
        from gammafold.data.tokenizer import GammaFoldTokenizer
        tokenizer = GammaFoldTokenizer()
        masked_ids, mlm_labels = tokenizer.create_mlm_targets(batch['token_ids'])
        batch['token_ids'] = masked_ids
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.mixed_precision):
            outputs = self.model(
                token_ids=batch['token_ids'],
                properties=batch['properties'],
                attention_mask=batch.get('attention_mask'),
                ss_tokens=batch.get('ss_tokens'),
                structure_tokens=batch.get('structure_tokens'),
            )
            
            # Compute loss
            loss_output = self.loss_fn(
                mlm_logits=outputs.get('mlm_logits'),
                mlm_labels=mlm_labels,
                pred_coords=outputs.get('coords'),
                target_coords=batch.get('coords'),
                coord_mask=batch.get('coord_mask')
            )
            
            loss = loss_output.total / self.config.optimizer.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (self.global_step + 1) % self.config.optimizer.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.config.optimizer.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.gradient_clip
                )
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Compute metrics
        metrics = loss_output.to_dict()
        if outputs.get('mlm_logits') is not None:
            metrics['accuracy'] = compute_accuracy(outputs['mlm_logits'], mlm_labels)
        
        return loss.item() * self.config.optimizer.gradient_accumulation_steps, metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Metrics dictionary
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Create MLM targets
            from gammafold.data.tokenizer import GammaFoldTokenizer
            tokenizer = GammaFoldTokenizer()
            masked_ids, mlm_labels = tokenizer.create_mlm_targets(batch['token_ids'])
            batch['token_ids'] = masked_ids
            
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    token_ids=batch['token_ids'],
                    properties=batch['properties'],
                    attention_mask=batch.get('attention_mask'),
                    ss_tokens=batch.get('ss_tokens'),
                )
                
                loss_output = self.loss_fn(
                    mlm_logits=outputs.get('mlm_logits'),
                    mlm_labels=mlm_labels,
                )
            
            total_loss += loss_output.total.item()
            if outputs.get('mlm_logits') is not None:
                total_accuracy += compute_accuracy(outputs['mlm_logits'], mlm_labels)
            n_batches += 1
        
        return {
            'loss': total_loss / max(1, n_batches),
            'accuracy': total_accuracy / max(1, n_batches)
        }
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'{name}.pt')
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Resumed from step {self.global_step}")
    
    def _cleanup_checkpoints(self, checkpoint_dir: str):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith('step_')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(checkpoint_dir, oldest))
            logger.debug(f"Removed old checkpoint: {oldest}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard and WandB."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except:
                pass
