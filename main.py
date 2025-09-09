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

    # Ensure the project directory is in the Python path
    # This helps with imports like 'from config.training_config import TrainingConfig'
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
        num_gpus = 1 # Fallback for CPU execution if needed, though DeepSpeed may fail.

    print(f"--> Found {num_gpus} GPUs.")

    # Training parameters
    args = {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "output_dir": "/kaggle/working/outputs/agrpo_deepspeed",
        "batch_size": 3,
        "gradient_accumulation_steps": 1,
        "num_epochs": 4,
        "learning_rate": 1e-4,
        "sample_size": 1000,
        "checkpoint_every_n_steps": 1,
    }

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
        training_script
    ]
    
    # Add training arguments to the command
    cmd.extend([f'--{key}={value}' for key, value in args.items()])
    
    # Add boolean flags
    cmd.append('--fp16')
    cmd.append('--verbose')

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
