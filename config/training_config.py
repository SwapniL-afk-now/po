# config/training_config.py

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TrainingConfig:
    """Centralized training configuration"""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    num_responses_per_query: int = 3
    max_new_tokens: int = 1024
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # AGRPO Algorithm
    alpha: float = 1.0
    sigma_threshold: float = 0.1
    sigma_min: float = 0.1
    g_max: float = 3.0
    f_max: float = 3.0
    tau: float = 0.2
    epsilon: float = 1e-8
    kl_penalty: float = 0.0
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning Rate Scheduler
    lr_scheduler_type: str = "cosine_warmup"
    warmup_steps: int = 50
    min_lr: float = 1e-6
    t_max: int = 100
    step_size: int = 30
    gamma: float = 0.9
    
    # Training
    num_epochs: int = 2
    per_device_train_batch_size: int = 3  # Per GPU batch size
    gradient_accumulation_steps: int = 1
    checkpoint_every_n_steps: int = 10
    max_checkpoints_to_keep: int = 2
    eval_every_n_steps: int = 50
    log_every_n_steps: int = 1
    
    # Dataset
    dataset_name: str = "openai/gsm8k"
    dataset_split: str = "train"
    sample_size: int = 1000
    num_workers: int = 4
    
    # Paths
    output_dir: str = "outputs/agrpo_deepspeed"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # System
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Resume
    resume_from_checkpoint: bool = True
    checkpoint_path: str = None
    
    # Logging
    verbose: bool = True
    wandb_project: str = "agrpo-training"
    wandb_entity: str = None
    use_wandb: bool = False
    
    # Distributed Training
    local_rank: int = -1
    world_size: int = 1
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.log_dir), exist_ok=True)
    
    @property
    def effective_batch_size(self):
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * self.world_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
