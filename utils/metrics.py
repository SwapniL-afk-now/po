# utils/metrics.py

import os
import json
import pickle
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_dir = os.path.join(config.output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.metrics = {
            'steps': [],
            'epochs': [],
            'rewards': [],
            'accuracies': [],
            'losses': [],
            'policy_losses': [],
            'kl_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'mu_B_history': [],
            'sigma_B_history': [],
            'timestamps': [],
        }
        
        self.load_metrics()
    
    def add_step_metrics(self, **kwargs):
        """Add metrics for a training step"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Add timestamp
        self.metrics['timestamps'].append(datetime.now().isoformat())
    
    def save_metrics(self):
        """Save metrics to disk"""
        # Save as pickle
        pickle_path = os.path.join(self.metrics_dir, "metrics.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Save as JSON for readability
        json_path = os.path.join(self.metrics_dir, "metrics.json")
        json_metrics = {k: v for k, v in self.metrics.items() 
                       if k != 'timestamps'}  # Exclude timestamps from JSON
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        logger.debug(f"Metrics saved to {self.metrics_dir}")
    
    def load_metrics(self):
        """Load existing metrics if available"""
        pickle_path = os.path.join(self.metrics_dir, "metrics.pkl")
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    loaded_metrics = pickle.load(f)
                self.metrics.update(loaded_metrics)
                logger.info(f"Loaded existing metrics from {pickle_path}")
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
    
    def plot_training_curves(self, save_path: str = None):
        """Generate comprehensive training plots"""
        if save_path is None:
            save_path = os.path.join(
                self.metrics_dir, 
                f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('AGRPO Training Metrics (DeepSpeed ZeRO-3)', fontsize=16, fontweight='bold')
        
        # Plot rewards
        if self.metrics['rewards']:
            ax = axes[0, 0]
            ax.plot(self.metrics['steps'][:len(self.metrics['rewards'])], 
                   self.metrics['rewards'], label='Average Reward', alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Reward')
            ax.set_title('Average Reward per Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot accuracy
        if self.metrics['accuracies']:
            ax = axes[0, 1]
            accuracies_percent = [a * 100 for a in self.metrics['accuracies']]
            ax.plot(self.metrics['steps'][:len(accuracies_percent)], 
                   accuracies_percent, label='Accuracy', color='green', alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Answer Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot losses
        if self.metrics['losses']:
            ax = axes[0, 2]
            ax.plot(self.metrics['steps'][:len(self.metrics['losses'])], 
                   self.metrics['losses'], label='Total Loss', color='red', alpha=0.8)
            if self.metrics['policy_losses']:
                ax.plot(self.metrics['steps'][:len(self.metrics['policy_losses'])], 
                       self.metrics['policy_losses'], label='Policy Loss', 
                       color='orange', alpha=0.6, linestyle='--')
            if self.metrics['kl_losses']:
                ax.plot(self.metrics['steps'][:len(self.metrics['kl_losses'])], 
                       self.metrics['kl_losses'], label='KL Loss', 
                       color='purple', alpha=0.6, linestyle='--')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot learning rate
        if self.metrics['learning_rates']:
            ax = axes[1, 0]
            ax.plot(self.metrics['steps'][:len(self.metrics['learning_rates'])], 
                   self.metrics['learning_rates'], label='Learning Rate', 
                   color='blue', alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot EMA baselines
        if self.metrics['mu_B_history'] and self.metrics['sigma_B_history']:
            ax = axes[1, 1]
            ax.plot(self.metrics['steps'][:len(self.metrics['mu_B_history'])], 
                   self.metrics['mu_B_history'], label='μ_B (Mean)', 
                   color='cyan', alpha=0.8)
            ax2 = ax.twinx()
            ax2.plot(self.metrics['steps'][:len(self.metrics['sigma_B_history'])], 
                    self.metrics['sigma_B_history'], label='σ_B (Std)', 
                    color='magenta', alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('μ_B', color='cyan')
            ax2.set_ylabel('σ_B', color='magenta')
            ax.set_title('EMA Baselines')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Plot gradient norms if available
        if self.metrics.get('gradient_norms'):
            ax = axes[1, 2]
            ax.plot(self.metrics['steps'][:len(self.metrics['gradient_norms'])], 
                   self.metrics['gradient_norms'], label='Gradient Norm', 
                   color='brown', alpha=0.8)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_path}")
        return save_path