# algorithms/agrpo.py

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

class AGRPOAlgorithm:
    """Adaptive Group Relative Policy Optimization Algorithm"""
    
    def __init__(self, config):
        self.config = config
        
        # EMA baselines
        self.mu_B = 0.0  # Mean baseline
        self.sigma_B = 1.0  # Standard deviation baseline
        
        # Algorithm parameters
        self.alpha = config.alpha
        self.sigma_threshold = config.sigma_threshold
        self.sigma_min = config.sigma_min
        self.g_max = config.g_max
        self.f_max = config.f_max
        self.tau = config.tau
        self.epsilon = config.epsilon
        self.kl_penalty = config.kl_penalty
    
    def compute_advantages(
        self, 
        rewards: List[float], 
        log_probs_old: List[float],
        device: torch.device
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute AGRPO advantages for a group of responses
        
        Args:
            rewards: List of rewards for each response
            log_probs_old: List of log probabilities under old policy
            device: Device to place tensors on
            
        Returns:
            advantages: Tensor of advantages
            group_quality: Group quality metric G_g
            relative_weight: Relative weight w_rel
        """
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        log_probs_tensor = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        
        # Group statistics
        mu_g = torch.mean(rewards_tensor)
        sigma_g = torch.std(rewards_tensor) + self.epsilon
        
        # Relative advantages within group
        A_rel = (rewards_tensor - mu_g) / sigma_g
        
        # Surprisal-based adjustment
        surprisal = -log_probs_tensor
        f_mean = torch.mean(surprisal)
        f_std = torch.std(surprisal) + self.epsilon
        f_standardized = (surprisal - f_mean) / f_std
        f_safe = torch.clamp(f_standardized, min=-self.f_max, max=self.f_max)
        
        # Group quality compared to baseline
        G_g = (mu_g - self.mu_B) / max(self.sigma_B, self.sigma_min)
        G_g = torch.clamp(G_g, min=-self.g_max, max=self.g_max)
        
        # Relative weight based on group diversity
        w_rel = max(0.2, min(1.0, sigma_g.item() / self.sigma_threshold))
        
        # Combined advantages
        advantages = w_rel * A_rel + self.alpha * G_g * f_safe
        
        # Update EMA baselines
        self._update_baselines(mu_g.item(), sigma_g.item())
        
        return advantages, G_g.item(), w_rel
    
    def _update_baselines(self, mu_g: float, sigma_g: float):
        """Update exponential moving average baselines"""
        self.mu_B = self.tau * self.mu_B + (1 - self.tau) * mu_g
        self.sigma_B = self.tau * self.sigma_B + (1 - self.tau) * sigma_g
    
    def compute_policy_loss(
        self,
        ratios: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clipped policy gradient loss
        
        Args:
            ratios: Importance sampling ratios
            advantages: Advantage values
            
        Returns:
            Policy loss (scalar tensor)
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)
        
        # Clip ratios for stability
        clipped_ratios = torch.clamp(ratios, 0.8, 1.2)
        
        # Policy gradient loss
        policy_loss = -torch.mean(clipped_ratios * advantages)
        
        return policy_loss
    
    def compute_kl_divergence(
        self,
        log_probs_old: torch.Tensor,
        log_probs_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between old and new policies
        
        Args:
            log_probs_old: Log probabilities under old policy
            log_probs_new: Log probabilities under new policy
            
        Returns:
            KL divergence (scalar tensor)
        """
        return torch.mean(log_probs_new - log_probs_old)
    
    def compute_total_loss(
        self,
        policy_loss: torch.Tensor,
        kl_div: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total loss with KL penalty
        
        Args:
            policy_loss: Policy gradient loss
            kl_div: KL divergence
            
        Returns:
            Total loss
        """
        return policy_loss + self.kl_penalty * kl_div
    
    def get_state_dict(self) -> Dict:
        """Get algorithm state for checkpointing"""
        return {
            'mu_B': self.mu_B,
            'sigma_B': self.sigma_B
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load algorithm state from checkpoint"""
        self.mu_B = state_dict.get('mu_B', 0.0)
        self.sigma_B = state_dict.get('sigma_B', 1.0)