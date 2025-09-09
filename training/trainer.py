# # training/trainer.py

# import os
# import torch
# from datetime import datetime
# from accelerate import Accelerator
# from accelerate.utils import set_seed, DeepSpeedPlugin
# from typing import Dict, Any, Optional
# import logging
# from tqdm import tqdm

# from models.model_manager import ModelManager
# from algorithms.agrpo import AGRPOAlgorithm
# from algorithms.rewards import RewardExtractor
# from training.batch_processor import BatchProcessor
# from training.scheduler import create_scheduler
# from utils.metrics import MetricsTracker
# from utils.checkpoints import CheckpointManager
# from utils.logging import setup_logging

# logger = logging.getLogger(__name__)

# class AGRPOTrainer:
#     """Main trainer for AGRPO with DeepSpeed"""
    
#     def __init__(self, config):
#         self.config = config
#         setup_logging(config)
        
#         # Initialize DeepSpeed plugin for ZeRO-3
#         deepspeed_plugin = DeepSpeedPlugin(
#             zero_stage=3,
#             gradient_accumulation_steps=config.gradient_accumulation_steps,
#             gradient_clipping=config.max_grad_norm,
#             offload_optimizer_device="cpu",
#             offload_param_device="cpu",
#             zero3_init_flag=True,
#             zero3_save_16bit_model=True,
#         )
        
#         # Initialize accelerator
#         self.accelerator = Accelerator(
#             gradient_accumulation_steps=config.gradient_accumulation_steps,
#             mixed_precision='fp16' if config.fp16 else 'no',
#             deepspeed_plugin=deepspeed_plugin,
#             log_with="wandb" if config.use_wandb else None,
#             project_dir=config.output_dir,
#         )
        
#         # Set seed
#         set_seed(config.seed)
        
#         # Update config with distributed info
#         config.local_rank = self.accelerator.local_process_index
#         config.world_size = self.accelerator.num_processes
        
#         self.is_main_process = self.accelerator.is_main_process
#         self.device = self.accelerator.device
        
#         logger.info(f"Initialized trainer on device: {self.device}")
#         logger.info(f"Number of processes: {self.accelerator.num_processes}")
#         logger.info(f"DeepSpeed ZeRO-3 enabled")
        
#         # Initialize components
#         self._initialize_components()
    
#     def _initialize_components(self):
#         """Initialize all training components"""
#         # Model manager
#         self.model_manager = ModelManager(self.config)
        
#         # Tokenizer
#         self.tokenizer = self.model_manager.setup_tokenizer()
        
#         # Data
#         from data.dataset import create_dataloader
#         from data.prompts import Prompts
#         self.prompts = Prompts()
#         self.dataloader, self.dataset = create_dataloader(
#             self.config, self.tokenizer, self.prompts
#         )
        
#         # Models
#         checkpoint_path = self._find_checkpoint()
#         self.model = self.model_manager.setup_model(checkpoint_path)
#         self.ref_model = self.model_manager.setup_reference_model(self.device)
        
#         # Algorithms
#         self.agrpo_algo = AGRPOAlgorithm(self.config)
#         self.reward_extractor = RewardExtractor(debug=self.config.verbose)
        
#         # Batch processor
#         self.batch_processor = BatchProcessor(
#             self.config,
#             self.model,
#             self.ref_model,
#             self.tokenizer,
#             self.agrpo_algo,
#             self.reward_extractor,
#             self.accelerator
#         )
        
#         # Optimizer
#         self.optimizer = torch.optim.AdamW(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=self.config.learning_rate,
#             betas=(self.config.adam_beta1, self.config.adam_beta2),
#             eps=self.config.adam_epsilon,
#             weight_decay=self.config.weight_decay
#         )
        
#         # Learning rate scheduler
#         total_steps = len(self.dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
#         self.scheduler = create_scheduler(self.optimizer, self.config, total_steps)
        
#         # Prepare with accelerator
#         self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
#             self.model, self.optimizer, self.dataloader, self.scheduler
#         )
        
#         # Metrics and checkpoints
#         self.metrics_tracker = MetricsTracker(self.config)
#         self.checkpoint_manager = CheckpointManager(self.config, self.accelerator)
        
#         # Training state
#         self.global_step = 0
#         self.start_epoch = 0
#         self.total_reward = 0
#         self.correct_count = 0
#         self.total_responses = 0
        
#         # Load checkpoint if exists
#         self._load_checkpoint()
    
#     def _find_checkpoint(self) -> Optional[str]:
#         """Find latest checkpoint"""
#         if self.config.checkpoint_path:
#             return self.config.checkpoint_path
        
#         checkpoint_dir = os.path.join(self.config.output_dir, self.config.checkpoint_dir)
#         if os.path.exists(checkpoint_dir):
#             checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
#             if checkpoints:
#                 latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
#                 return os.path.join(checkpoint_dir, latest)
#         return None
    
#     def _load_checkpoint(self):
#         """Load training state from checkpoint"""
#         checkpoint = self.checkpoint_manager.load_latest_checkpoint()
#         if checkpoint:
#             self.global_step = checkpoint.get('global_step', 0)
#             self.start_epoch = checkpoint.get('epoch', 0)
#             self.total_reward = checkpoint.get('total_reward', 0)
#             self.correct_count = checkpoint.get('correct_count', 0)
#             self.total_responses = checkpoint.get('total_responses', 0)
            
#             # Load AGRPO state
#             if 'agrpo_state' in checkpoint:
#                 self.agrpo_algo.load_state_dict(checkpoint['agrpo_state'])
            
#             logger.info(f"Resumed from checkpoint at step {self.global_step}")
    
#     def train(self):
#         """Main training loop"""
#         logger.info("="*80)
#         logger.info("Starting AGRPO Training with DeepSpeed ZeRO-3")
#         logger.info(f"Total epochs: {self.config.num_epochs}")
#         logger.info(f"Batch size per device: {self.config.per_device_train_batch_size}")
#         logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
#         logger.info(f"Effective batch size: {self.config.effective_batch_size}")
#         logger.info("="*80)
        
#         for epoch in range(self.start_epoch, self.config.num_epochs):
#             self._train_epoch(epoch)
        
#         self._save_final_model()
#         self._print_summary()
    
#     def _train_epoch(self, epoch: int):
#         """Train for one epoch"""
#         logger.info(f"\n{'='*60}")
#         logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
#         logger.info(f"{'='*60}")
        
#         self.model.train()
#         epoch_metrics = {
#             'total_reward': 0,
#             'correct_count': 0,
#             'total_responses': 0,
#             'total_loss': 0,
#             'steps': 0
#         }
        
#         progress_bar = tqdm(
#             self.dataloader,
#             desc=f"Epoch {epoch + 1}",
#             disable=not self.is_main_process
#         )
        
#         for batch_idx, batch in enumerate(progress_bar):
#             # Process batch
#             batch_metrics = self.batch_processor.process_batch(
#                 batch, 
#                 self.global_step,
#                 self.optimizer,
#                 self.scheduler
#             )
            
#             # Update metrics
#             epoch_metrics['total_reward'] += batch_metrics['total_reward']
#             epoch_metrics['correct_count'] += batch_metrics['correct_count']
#             epoch_metrics['total_responses'] += batch_metrics['total_responses']
#             epoch_metrics['total_loss'] += batch_metrics.get('loss', 0)
#             epoch_metrics['steps'] += 1
            
#             self.total_reward += batch_metrics['total_reward']
#             self.correct_count += batch_metrics['correct_count']
#             self.total_responses += batch_metrics['total_responses']
            
#             # Update progress bar
#             if self.is_main_process:
#                 accuracy = batch_metrics['correct_count'] / batch_metrics['total_responses'] if batch_metrics['total_responses'] > 0 else 0
#                 progress_bar.set_postfix({
#                     'acc': f"{accuracy*100:.1f}%",
#                     'reward': f"{batch_metrics['avg_reward']:.3f}",
#                     'loss': f"{batch_metrics.get('loss', 0):.4f}"
#                 })
            
#             # Log metrics
#             if self.global_step % self.config.log_every_n_steps == 0:
#                 self._log_metrics(batch_metrics, epoch)
            
#             # Save checkpoint
#             if self.global_step % self.config.checkpoint_every_n_steps == 0:
#                 self._save_checkpoint(epoch, batch_idx)
            
#             self.global_step += 1
        
#         # Epoch summary
#         self._log_epoch_summary(epoch, epoch_metrics)
    
#     def _log_metrics(self, batch_metrics: Dict, epoch: int):
#         """Log training metrics"""
#         if self.is_main_process:
#             metrics = {
#                 'epoch': epoch,
#                 'step': self.global_step,
#                 'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate,
#                 **batch_metrics
#             }
            
#             self.metrics_tracker.add_step_metrics(**metrics)
            
#             if self.config.use_wandb and self.accelerator.is_initialized:
#                 self.accelerator.log(metrics, step=self.global_step)
    
#     def _save_checkpoint(self, epoch: int, batch_idx: int):
#         """Save training checkpoint"""
#         if self.is_main_process:
#             checkpoint = {
#                 'epoch': epoch,
#                 'batch_idx': batch_idx,
#                 'global_step': self.global_step,
#                 'total_reward': self.total_reward,
#                 'correct_count': self.correct_count,
#                 'total_responses': self.total_responses,
#                 'agrpo_state': self.agrpo_algo.get_state_dict(),
#             }
            
#             self.checkpoint_manager.save_checkpoint(
#                 self.model,
#                 self.optimizer,
#                 self.scheduler,
#                 checkpoint,
#                 self.global_step
#             )
            
#             # Save metrics
#             self.metrics_tracker.save_metrics()
#             self.metrics_tracker.plot_training_curves()
    
#     def _log_epoch_summary(self, epoch: int, metrics: Dict):
#         """Log epoch summary"""
#         if self.is_main_process:
#             accuracy = metrics['correct_count'] / metrics['total_responses'] if metrics['total_responses'] > 0 else 0
#             avg_loss = metrics['total_loss'] / metrics['steps'] if metrics['steps'] > 0 else 0
            
#             logger.info(f"\nEpoch {epoch + 1} Summary:")
#             logger.info(f"  Total responses: {metrics['total_responses']}")
#             logger.info(f"  Correct: {metrics['correct_count']}")
#             logger.info(f"  Accuracy: {accuracy*100:.2f}%")
#             logger.info(f"  Average reward: {metrics['total_reward']/metrics['total_responses']:.4f}")
#             logger.info(f"  Average loss: {avg_loss:.4f}")
    
#     def _save_final_model(self):
#         """Save final model"""
#         if self.is_main_process:
#             final_path = os.path.join(self.config.output_dir, "final_model")
#             self.model_manager.save_model(final_path, unwrap=True)
            
#             # Save final metrics
#             self.metrics_tracker.save_metrics()
#             self.metrics_tracker.plot_training_curves(
#                 save_path=os.path.join(self.config.output_dir, "final_training_curves.png")
#             )
    
#     def _print_summary(self):
#         """Print final training summary"""
#         if self.is_main_process:
#             logger.info("\n" + "="*80)
#             logger.info(" " * 30 + "TRAINING COMPLETED")
#             logger.info("="*80)
            
#             if self.total_responses > 0:
#                 accuracy = self.correct_count / self.total_responses
#                 avg_reward = self.total_reward / self.total_responses
                
#                 logger.info(f"Total steps: {self.global_step}")
#                 logger.info(f"Total responses: {self.total_responses}")
#                 logger.info(f"Correct answers: {self.correct_count}")
#                 logger.info(f"Overall accuracy: {accuracy*100:.2f}%")
#                 logger.info(f"Average reward: {avg_reward:.4f}")



# training/trainer.py

import os
import torch
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

from models.model_manager import ModelManager
from algorithms.agrpo import AGRPOAlgorithm
from algorithms.rewards import RewardExtractor
from training.batch_processor import BatchProcessor
from training.scheduler import create_scheduler
from utils.metrics import MetricsTracker
from utils.checkpoints import CheckpointManager
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

class AGRPOTrainer:
    """Main trainer for AGRPO with DeepSpeed"""
    
    def __init__(self, config):
        self.config = config
        setup_logging(config)
        
        # Initialize DeepSpeed plugin for ZeRO-3
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=3,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_clipping=config.max_grad_norm,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
        )
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision='fp16' if config.fp16 else 'no',
            deepspeed_plugin=deepspeed_plugin,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir,
        )
        
        # Set seed
        set_seed(config.seed)
        
        # Update config with distributed info
        config.local_rank = self.accelerator.local_process_index
        config.world_size = self.accelerator.num_processes
        
        self.is_main_process = self.accelerator.is_main_process
        self.device = self.accelerator.device
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Number of processes: {self.accelerator.num_processes}")
        logger.info(f"DeepSpeed ZeRO-3 enabled")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all training components"""
        # Model manager
        self.model_manager = ModelManager(self.config)
        
        # Tokenizer
        self.tokenizer = self.model_manager.setup_tokenizer()
        
        # Data
        from data.dataset import create_dataloader
        from data.prompts import Prompts
        self.prompts = Prompts()
        self.dataloader, self.dataset = create_dataloader(
            self.config, self.tokenizer, self.prompts
        )
        
        # Models
        checkpoint_path = self._find_checkpoint()
        self.model = self.model_manager.setup_model(checkpoint_path)
        # self.ref_model = self.model_manager.setup_reference_model(self.device) ### <--- THIS LINE HAS BEEN REMOVED
        
        # Algorithms
        self.agrpo_algo = AGRPOAlgorithm(self.config)
        self.reward_extractor = RewardExtractor(debug=self.config.verbose)
        
        # Batch processor
        self.batch_processor = BatchProcessor(
            self.config,
            self.model,
            # self.ref_model, ### <--- THIS ARGUMENT HAS BEEN REMOVED
            self.tokenizer,
            self.agrpo_algo,
            self.reward_extractor,
            self.accelerator
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = create_scheduler(self.optimizer, self.config, total_steps)
        
        # Prepare with accelerator
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.scheduler
        )
        
        # Metrics and checkpoints
        self.metrics_tracker = MetricsTracker(self.config)
        self.checkpoint_manager = CheckpointManager(self.config, self.accelerator)
        
        # Training state
        self.global_step = 0
        self.start_epoch = 0
        self.total_reward = 0
        self.correct_count = 0
        self.total_responses = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint"""
        if self.config.checkpoint_path:
            return self.config.checkpoint_path
        
        checkpoint_dir = os.path.join(self.config.output_dir, self.config.checkpoint_dir)
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                return os.path.join(checkpoint_dir, latest)
        return None
    
    def _load_checkpoint(self):
        """Load training state from checkpoint"""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self.global_step = checkpoint.get('global_step', 0)
            self.start_epoch = checkpoint.get('epoch', 0)
            self.total_reward = checkpoint.get('total_reward', 0)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_responses = checkpoint.get('total_responses', 0)
            
            # Load AGRPO state
            if 'agrpo_state' in checkpoint:
                self.agrpo_algo.load_state_dict(checkpoint['agrpo_state'])
            
            logger.info(f"Resumed from checkpoint at step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        logger.info("="*80)
        logger.info("Starting AGRPO Training with DeepSpeed ZeRO-3")
        logger.info(f"Total epochs: {self.config.num_epochs}")
        logger.info(f"Batch size per device: {self.config.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")
        logger.info("="*80)
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self._train_epoch(epoch)
        
        self._save_final_model()
        self._print_summary()
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
        logger.info(f"{'='*60}")
        
        self.model.train()
        epoch_metrics = {
            'total_reward': 0,
            'correct_count': 0,
            'total_responses': 0,
            'total_loss': 0,
            'steps': 0
        }
        
        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not self.is_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch
            batch_metrics = self.batch_processor.process_batch(
                batch, 
                self.global_step,
                self.optimizer,
                self.scheduler
            )
            
            # Update metrics
            epoch_metrics['total_reward'] += batch_metrics['total_reward']
            epoch_metrics['correct_count'] += batch_metrics['correct_count']
            epoch_metrics['total_responses'] += batch_metrics['total_responses']
            epoch_metrics['total_loss'] += batch_metrics.get('loss', 0)
            epoch_metrics['steps'] += 1
            
            self.total_reward += batch_metrics['total_reward']
            self.correct_count += batch_metrics['correct_count']
            self.total_responses += batch_metrics['total_responses']
            
            # Update progress bar
            if self.is_main_process:
                accuracy = batch_metrics['correct_count'] / batch_metrics['total_responses'] if batch_metrics['total_responses'] > 0 else 0
                progress_bar.set_postfix({
                    'acc': f"{accuracy*100:.1f}%",
                    'reward': f"{batch_metrics['avg_reward']:.3f}",
                    'loss': f"{batch_metrics.get('loss', 0):.4f}"
                })
            
            # Log metrics
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(batch_metrics, epoch)
            
            # Save checkpoint
            if self.global_step % self.config.checkpoint_every_n_steps == 0:
                self._save_checkpoint(epoch, batch_idx)
            
            self.global_step += 1
        
        # Epoch summary
        self._log_epoch_summary(epoch, epoch_metrics)
    
    def _log_metrics(self, batch_metrics: Dict, epoch: int):
        """Log training metrics"""
        if self.is_main_process:
            metrics = {
                'epoch': epoch,
                'step': self.global_step,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate,
                **batch_metrics
            }
            
            self.metrics_tracker.add_step_metrics(**metrics)
            
            if self.config.use_wandb and self.accelerator.is_initialized:
                self.accelerator.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, epoch: int, batch_idx: int):
        """Save training checkpoint"""
        if self.is_main_process:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'global_step': self.global_step,
                'total_reward': self.total_reward,
                'correct_count': self.correct_count,
                'total_responses': self.total_responses,
                'agrpo_state': self.agrpo_algo.get_state_dict(),
            }
            
            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                checkpoint,
                self.global_step
            )
            
            # Save metrics
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.plot_training_curves()
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict):
        """Log epoch summary"""
        if self.is_main_process:
            accuracy = metrics['correct_count'] / metrics['total_responses'] if metrics['total_responses'] > 0 else 0
            avg_loss = metrics['total_loss'] / metrics['steps'] if metrics['steps'] > 0 else 0
            
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Total responses: {metrics['total_responses']}")
            logger.info(f"  Correct: {metrics['correct_count']}")
            logger.info(f"  Accuracy: {accuracy*100:.2f}%")
            logger.info(f"  Average reward: {metrics['total_reward']/metrics['total_responses']:.4f}")
            logger.info(f"  Average loss: {avg_loss:.4f}")
    
    def _save_final_model(self):
        """Save final model"""
        if self.is_main_process:
            final_path = os.path.join(self.config.output_dir, "final_model")
            self.model_manager.save_model(final_path, unwrap=True)
            
            # Save final metrics
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.plot_training_curves(
                save_path=os.path.join(self.config.output_dir, "final_training_curves.png")
            )
    
    def _print_summary(self):
        """Print final training summary"""
        if self.is_main_process:
            logger.info("\n" + "="*80)
            logger.info(" " * 30 + "TRAINING COMPLETED")
            logger.info("="*80)
            
            if self.total_responses > 0:
                accuracy = self.correct_count / self.total_responses
                avg_reward = self.total_reward / self.total_responses
                
                logger.info(f"Total steps: {self.global_step}")
                logger.info(f"Total responses: {self.total_responses}")
                logger.info(f"Correct answers: {self.correct_count}")
                logger.info(f"Overall accuracy: {accuracy*100:.2f}%")
                logger.info(f"Average reward: {avg_reward:.4f}")
