"""
Training CLI for GammaFold.

Usage:
    gammafold train --config configs/small.yaml
    gammafold train --model original --size small --data data/train
    gammafold train --smoke-test
"""

import logging
import argparse
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_train(args: argparse.Namespace) -> int:
    """
    Run training command.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    import torch
    from gammafold.training.config import TrainingConfig, ModelConfig, DataConfig, OptimizerConfig
    from gammafold.training.trainer import Trainer
    from gammafold.data.dataset import ProteinSequenceDataset, create_dataloader
    from gammafold.data.tokenizer import GammaFoldTokenizer
    
    try:
        # Load or create config
        if args.config:
            logger.info(f"Loading config from {args.config}")
            config = TrainingConfig.from_yaml(args.config)
        else:
            config = TrainingConfig(
                model=ModelConfig(model_type=args.model, model_size=args.size),
                data=DataConfig(train_data=args.data, batch_size=args.batch_size),
                optimizer=OptimizerConfig(learning_rate=args.lr),
                max_steps=args.steps,
                output_dir=args.output
            )
        
        # Smoke test mode
        if args.smoke_test:
            logger.info("Running smoke test (10 steps)")
            config.max_steps = 10
            config.save_steps = 5
            config.eval_steps = 5
            config.log_steps = 1
        
        # Resume
        if args.resume:
            config.resume_from = args.resume
        
        # Create tokenizer and datasets
        tokenizer = GammaFoldTokenizer(max_length=config.data.max_seq_len)
        
        # Check if training data exists, otherwise use dummy data for smoke test
        import os
        if os.path.exists(config.data.train_data):
            train_dataset = ProteinSequenceDataset(
                fasta_paths=config.data.train_data,
                tokenizer=tokenizer,
                max_length=config.data.max_seq_len,
                min_length=config.data.min_seq_len
            )
        elif args.smoke_test:
            # Create dummy dataset for smoke test
            logger.warning("Training data not found, using dummy data for smoke test")
            train_dataset = DummyDataset(tokenizer, size=100, max_length=config.data.max_seq_len)
        else:
            raise FileNotFoundError(f"Training data not found: {config.data.train_data}")
        
        train_dataloader = create_dataloader(
            train_dataset,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )
        
        # Validation data
        val_dataloader = None
        if config.data.val_data and os.path.exists(config.data.val_data):
            val_dataset = ProteinSequenceDataset(
                fasta_paths=config.data.val_data,
                tokenizer=tokenizer,
                max_length=config.data.max_seq_len
            )
            val_dataloader = create_dataloader(
                val_dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers
            )
        
        # Create trainer
        trainer = Trainer(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
        
        # Log training info
        logger.info(f"Model: {config.model.model_type} ({config.model.model_size})")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batch size: {config.data.batch_size}")
        logger.info(f"Max steps: {config.max_steps}")
        logger.info(f"Device: {trainer.device}")
        
        # Train
        results = trainer.train()
        
        logger.info(f"Training complete! Results: {results}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for smoke testing."""
    
    def __init__(self, tokenizer, size=100, max_length=256):
        self.tokenizer = tokenizer
        self.size = size
        self.max_length = max_length
        
        # Generate random sequences
        import random
        aa = 'ACDEFGHIKLMNPQRSTVWY'
        self.sequences = []
        for _ in range(size):
            length = random.randint(50, max_length)
            seq = ''.join(random.choices(aa, k=length))
            self.sequences.append(seq)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.sequences[idx])
        return {
            'token_ids': encoded.token_ids,
            'properties': encoded.properties,
            'attention_mask': encoded.attention_mask,
            'ss_tokens': encoded.ss_tokens,
        }



def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train GammaFold model')
    parser.add_argument('--config', '-c', help='Config YAML file')
    parser.add_argument('--model', choices=['original', 'multimodal'], default='original')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], default='small')
    parser.add_argument('--data', '-d', default='data/train')
    parser.add_argument('--output', '-o', default='outputs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--smoke-test', action='store_true')
    
    args = parser.parse_args()
    return run_train(args)


if __name__ == '__main__':
    exit(main())
