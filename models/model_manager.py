# models/model_manager.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel
from typing import Optional, Tuple, List
import os

class ModelManager:
    """Manages model initialization and loading"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.ref_model = None
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """Initialize tokenizer"""
        print(f"Loading tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # For batch generation
        self.tokenizer = tokenizer
        return tokenizer
    
    def setup_model(self, checkpoint_path: Optional[str] = None) -> AutoModelForCausalLM:
        """Setup model with LoRA adapters"""
        print(f"Loading base model: {self.config.model_name}")
        
        # Load base model with appropriate dtype
        if self.config.bf16:
            torch_dtype = torch.bfloat16
        elif self.config.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing if needed
        if self.config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Load or create LoRA adapter
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading LoRA adapter from: {checkpoint_path}")
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            print("Creating new LoRA adapter")
            from models.lora_config import create_lora_config
            lora_config = create_lora_config(self.config)
            model = get_peft_model(base_model, lora_config)
        
        # Print model info
        self._print_model_info(model)
        
        self.model = model
        return model
    
    def setup_reference_model(self, device: torch.device) -> AutoModelForCausalLM:
        """Setup reference model for KL divergence calculation"""
        print("Loading reference model for KL divergence...")
        
        if self.config.bf16:
            torch_dtype = torch.bfloat16
        elif self.config.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        
        ref_model.eval()
        
        # Disable gradients for reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        self.ref_model = ref_model
        return ref_model
    
    def _print_model_info(self, model):
        """Print model parameter information"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        percentage = 100 * trainable_params / all_params if all_params > 0 else 0
        
        print(f"\nModel Information:")
        print(f"  Total parameters: {all_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {percentage:.2f}%")
    
    def save_model(self, save_path: str, unwrap: bool = False):
        """Save model checkpoint"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        if unwrap and hasattr(self.model, 'module'):
            # For DeepSpeed distributed training
            self.model.module.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        # Also save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to: {save_path}")
    
    @property
    def im_end_token_id(self):
        """Get im_end token id for Qwen models"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        return self.tokenizer.convert_tokens_to_ids("<|im_end|>")