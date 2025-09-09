# utils/memory.py

import gc
import torch
import logging
import psutil
import os

logger = logging.getLogger(__name__)

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_gpu_memory_reserved():
    """Get reserved GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0

def get_gpu_memory_free():
    """Get free GPU memory in MB"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        return total - allocated
    return 0

def get_system_memory_usage():
    """Get system RAM usage"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / 1024 / 1024 / 1024,  # GB
        'available': memory.available / 1024 / 1024 / 1024,  # GB
        'percent': memory.percent,
        'used': memory.used / 1024 / 1024 / 1024,  # GB
    }

def print_memory_usage(prefix=""):
    """Print current memory usage"""
    if torch.cuda.is_available():
        allocated = get_gpu_memory_usage()
        reserved = get_gpu_memory_reserved()
        free = get_gpu_memory_free()
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Free: {free:.2f} MB")
        
        # Print for all GPUs if multiple
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                logger.info(f"  GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    # System memory
    sys_mem = get_system_memory_usage()
    logger.info(f"{prefix}System RAM - Used: {sys_mem['used']:.2f} GB / {sys_mem['total']:.2f} GB ({sys_mem['percent']:.1f}%)")

def aggressive_memory_cleanup():
    """Aggressively clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # For all GPUs
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def optimize_memory_allocation():
    """Optimize CUDA memory allocation"""
    if torch.cuda.is_available():
        # Set memory fraction for PyTorch
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable memory efficient attention if available
        if hasattr(torch.cuda, 'memory_efficient_attention'):
            torch.cuda.memory_efficient_attention.enable()
        
        # Set CUDA memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logger.info("Memory optimization settings applied")

def check_memory_requirements(model_size_gb: float, batch_size: int, sequence_length: int):
    """Estimate memory requirements for training"""
    # Rough estimation formula
    # Model parameters + Gradients + Optimizer states + Activations
    
    # Model and gradients: ~2x model size
    model_memory = model_size_gb * 2
    
    # Optimizer states (Adam): ~2x model size
    optimizer_memory = model_size_gb * 2
    
    # Activations (rough estimate)
    activation_memory = (batch_size * sequence_length * 4096 * 4) / (1024**3)  # Assuming hidden size ~4096
    
    total_memory = model_memory + optimizer_memory + activation_memory
    
    logger.info(f"Estimated memory requirements:")
    logger.info(f"  Model + Gradients: {model_memory:.2f} GB")
    logger.info(f"  Optimizer states: {optimizer_memory:.2f} GB")
    logger.info(f"  Activations: {activation_memory:.2f} GB")
    logger.info(f"  Total: {total_memory:.2f} GB")
    
    # With DeepSpeed ZeRO-3
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    per_gpu_with_zero3 = total_memory / num_gpus
    logger.info(f"  With ZeRO-3 ({num_gpus} GPUs): {per_gpu_with_zero3:.2f} GB per GPU")
    
    return total_memory

def monitor_memory_during_training(step: int, phase: str = ""):
    """Monitor memory usage during training"""
    if torch.cuda.is_available():
        allocated = get_gpu_memory_usage()
        reserved = get_gpu_memory_reserved()
        
        # Log if memory usage is high
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        usage_percent = (allocated / total_memory) * 100
        
        if usage_percent > 90:
            logger.warning(f"High GPU memory usage at step {step} ({phase}): {usage_percent:.1f}%")
        
        # Track p