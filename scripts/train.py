"""
Main training script for AGRPO with DeepSpeed ZeRO-3
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from training.trainer import AGRPOTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AGRPO Training with DeepSpeed")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Model name or path")
    parser.add_argument("--num_responses", type=int, default=3,
                       help="Number of responses per query")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum new tokens to generate")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    
    # Dataset arguments
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Number of samples to use from dataset")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/agrpo_deepspeed",
                       help="Output directory")
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=10,
                       help="Save checkpoint every N steps")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false",
                       help="Don't use FP16 training")
    parser.set_defaults(fp16=True)
    
    # Resume arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Specific checkpoint to resume from")
    
    # Logging arguments
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="agrpo-training",
                       help="Weights & Biases project name")
    
    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, 
                       default="config/deepspeed_config.json",
                       help="DeepSpeed configuration file")
    
    args = parser.parse_args()
    return args

def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_name=args.model_name,
        num_responses_per_query=args.num_responses,
        max_new_tokens=args.max_new_tokens,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        seed=args.seed,
        fp16=args.fp16,
        resume_from_checkpoint=args.resume,
        checkpoint_path=args.checkpoint_path,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        local_rank=args.local_rank,
    )
    
    # Log configuration
    if config.local_rank in [-1, 0]:
        print("\n" + "="*80)
        print("AGRPO Training Configuration")
        print("="*80)
        print(json.dumps(config.to_dict(), indent=2))
        print("="*80 + "\n")
    
    # Create trainer
    trainer = AGRPOTrainer(config)
    
    # Start training
    trainer.train()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()