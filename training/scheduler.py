# training/scheduler.py

import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, ExponentialLR

def create_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler"""
    
    scheduler_type = config.lr_scheduler_type
    
    if scheduler_type == "cosine_warmup":
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            progress = float(current_step - config.warmup_steps) / \
                      float(max(1, num_training_steps - config.warmup_steps))
            return max(config.min_lr / config.learning_rate,
                      0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "linear_warmup":
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps,
            eta_min=config.min_lr
        )
        
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        
    elif scheduler_type == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=config.gamma)
        
    else:
        scheduler = None
    
    return scheduler