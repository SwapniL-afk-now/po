# utils/checkpoints.py

import os
import torch
import shutil
from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.checkpoint_dir = os.path.join(config.output_dir, config.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        model, 
        optimizer, 
        scheduler, 
        metadata: Dict,
        step: int
    ):
        """Save a training checkpoint - simplified for LoRA adapters"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        
        # For DeepSpeed with LoRA, we save the LoRA adapters separately
        # to avoid the NCCL timeout issues with full model saves
        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save LoRA adapters only (much smaller than full model)
            try:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'save_pretrained'):
                    unwrapped_model.save_pretrained(checkpoint_path)
                    logger.info(f"LoRA adapters saved to {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save model adapters: {e}")
            
            # Save optimizer and scheduler states locally (not with DeepSpeed)
            try:
                torch.save({
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }, os.path.join(checkpoint_path, "training_state.pt"))
            except Exception as e:
                logger.error(f"Failed to save optimizer state: {e}")
            
            # Save metadata
            torch.save(metadata, os.path.join(checkpoint_path, "metadata.pt"))
            
            logger.info(f"Checkpoint saved at step {step}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
        
        # Ensure all processes wait for checkpoint to complete
        self.accelerator.wait_for_everyone()
    
    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint"""
        checkpoints = self._get_checkpoint_list()
        if not checkpoints:
            return None
        
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
        
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, "metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location='cpu')
            metadata['checkpoint_path'] = checkpoint_path
            return metadata
        
        return None
    
    def _get_checkpoint_list(self) -> List[str]:
        """Get sorted list of checkpoints"""
        checkpoints = []
        if os.path.exists(self.checkpoint_dir):
            for item in os.listdir(self.checkpoint_dir):
                if item.startswith("checkpoint-"):
                    try:
                        step = int(item.split("-")[-1])
                        checkpoints.append((step, item))
                    except ValueError:
                        continue
        
        checkpoints.sort(key=lambda x: x[0])
        return [cp[1] for cp in checkpoints]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoints = self._get_checkpoint_list()
        
        if len(checkpoints) > self.config.max_checkpoints_to_keep:
            checkpoints_to_remove = checkpoints[:-self.config.max_checkpoints_to_keep]
            
            for checkpoint in checkpoints_to_remove:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint}")
