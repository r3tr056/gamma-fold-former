"""
Distributed training utilities for GammaFold.
"""

import os
import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = 'nccl',
    init_method: str = 'env://'
) -> tuple:
    """
    Initialize distributed training.
    
    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method
        
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    # Get distributed info from environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False
) -> DDP:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: Model to wrap
        local_rank: Local GPU rank
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model
    """
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters
    )


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute mean across processes.
    
    Args:
        tensor: Tensor to reduce
        
    Returns:
        Reduced tensor (mean)
    """
    if not dist.is_initialized():
        return tensor
    
    cloned = tensor.clone()
    dist.all_reduce(cloned, op=dist.ReduceOp.SUM)
    return cloned / get_world_size()


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list
