# models/lora_config.py

from peft import LoraConfig, TaskType

def create_lora_config(config):
    """Create LoRA configuration"""
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {config.lora_r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.target_modules}")
    
    return lora_config