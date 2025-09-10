# /kaggle/working/main.py

import os
import subprocess
import torch
import sys

def main():
    """
    Configures the environment and launches the AGRPO training script using accelerate.
    This script replaces the need for 'scripts/launch.sh'.
    """
    # ==========================================================================
    # 1. CONFIGURE ENVIRONMENT
    # ==========================================================================
    print("--> Configuring environment variables...")
    
    # Set cache directories for Hugging Face
    cache_dir = "/kaggle/working/cache"
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    # Disable tokenizer parallelism to avoid deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set NCCL timeout to avoid timeouts during checkpoint saving
    os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour timeout
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    
    # Optimize NCCL settings for better performance
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if not available

    # Ensure the project directory is in the Python path
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    print(f"--> Environment configured. Cache directory: {cache_dir}")

    # ==========================================================================
    # 2. DEFINE TRAINING ARGUMENTS
    # ==========================================================================
    print("--> Defining training arguments...")
    
    # Dynamically determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("!!! WARNING: No GPUs detected. Training will be very slow. !!!")
        num_gpus = 1  # Fallback for CPU execution if needed
    
    print(f"--> Found {num_gpus} GPUs.")
    
    # Calculate optimal batch size and gradient accumulation for multi-GPU
    # Target effective batch size: 8
    # With 2 GPUs: batch_size=1, grad_accum=4 -> effective=8
    # With 1 GPU: batch_size=1, grad_accum=8 -> effective=8
    
    if num_gpus == 2:
        batch_size = 1
        gradient_accumulation_steps = 1
    else:
        batch_size = 1
        gradient_accumulation_steps = 8

    # Training parameters
    args = {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "output_dir": "/kaggle/working/outputs/agrpo_deepspeed",
        "batch_size": str(batch_size),
        "gradient_accumulation_steps": str(gradient_accumulation_steps),
        "num_epochs": "2",  # Reduced for testing
        "learning_rate": "1e-4",
        "sample_size": "100",  # Start with smaller dataset for testing
        "checkpoint_every_n_steps": "1",  # Save less frequently
        "num_responses": "3",  # Reduce responses per query for memory
        "max_new_tokens": "1024",  # Reduce max tokens for memory
    }

    print(f"--> Training configuration:")
    print(f"    Batch size per GPU: {batch_size}")
    print(f"    Gradient accumulation: {gradient_accumulation_steps}")
    print(f"    Effective batch size: {batch_size * gradient_accumulation_steps * num_gpus}")

    # ==========================================================================
    # 3. CONSTRUCT AND LAUNCH THE ACCELERATE COMMAND
    # ==========================================================================
    
    # Path to the training script
    training_script = "scripts/train.py"
    deepspeed_config = "config/deepspeed_config.json"

    # Base command for accelerate
    cmd = [
        'accelerate', 'launch',
        '--num_processes', str(num_gpus),
        '--use_deepspeed',
        '--deepspeed_config_file', deepspeed_config,
        '--zero3_init_flag', 'true',
        '--gradient_accumulation_steps', str(gradient_accumulation_steps),
        training_script
    ]
    
    # Add training arguments to the command
    for key, value in args.items():
        cmd.extend([f'--{key}', value])
    
    # Add boolean flags
    cmd.append('--fp16')
    cmd.append('--verbose')
    cmd.append('--no-resume')  # Don't resume for testing

    print("\n" + "="*80)
    print("Constructed command to execute:")
    # Print the command in a readable format
    print(' \\\n    '.join(cmd))
    print("="*80 + "\n")

    print("--> Launching training...")
    try:
        # Execute the command
        subprocess.run(cmd, check=True)
        print("\n--> Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERROR: Training script failed with exit code {e.returncode} !!!")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\n!!! ERROR: 'accelerate' command not found. Make sure it's installed and in your PATH. !!!")
        sys.exit(1)


if __name__ == "__main__":
    main()
