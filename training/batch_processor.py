# training/batch_processor.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing for AGRPO training"""
    
    def __init__(self, config, model, ref_model, tokenizer, agrpo_algo, reward_extractor, accelerator):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.agrpo_algo = agrpo_algo
        self.reward_extractor = reward_extractor
        self.accelerator = accelerator
        self.device = accelerator.device
        
        # Get special tokens
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    def process_batch(self, batch: Dict, global_step: int, optimizer, scheduler) -> Dict[str, Any]:
        """Process a single training batch"""
        
        # Generate responses
        responses_data = self._generate_responses(batch)
        
        # Compute losses and gradients
        loss_data = self._compute_losses(batch, responses_data)
        
        # Optimizer step
        if self.accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler:
                scheduler.step()
        
        # Compute metrics
        metrics = self._compute_metrics(responses_data, loss_data)
        
        return metrics
    
    def _generate_responses(self, batch: Dict) -> Dict:
        """Generate multiple responses per query"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        batch_size = input_ids.shape[0]
        num_responses = self.config.num_responses_per_query
        
        # Expand inputs for multiple responses
        expanded_input_ids = input_ids.repeat_interleave(num_responses, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_responses, dim=0)
        
        # Generate responses
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.im_end_token_id,
                use_cache=True,
            )
        self.model.train()
        
        # Process generated texts
        all_rewards = []
        all_responses = []
        all_extracted_answers = []
        
        for i in range(batch_size):
            example_rewards = []
            example_responses = []
            example_answers = []
            
            for j in range(num_responses):
                idx = i * num_responses + j
                input_length = input_ids[i].shape[0]
                
                # Decode generated text
                gen_text = self.tokenizer.decode(
                    generated_ids[idx][input_length:],
                    skip_special_tokens=False
                )
                
                # Truncate at end token
                if '<|im_end|>' in gen_text:
                    gen_text = gen_text.split('<|im_end|>')[0]
                
                # Compute reward
                reward, info = self.reward_extractor.compute_reward(
                    gen_text, batch['ground_truths'][i]
                )
                
                example_rewards.append(reward)
                example_responses.append(gen_text)
                example_answers.append(info)
            
            all_rewards.append(example_rewards)
            all_responses.append(example_responses)
            all_extracted_answers.append(example_answers)
        
        return {
            'generated_ids': generated_ids,
            'input_ids': input_ids,
            'rewards': all_rewards,
            'responses': all_responses,
            'extracted_answers': all_extracted_answers,
        }
    
    def _compute_losses(self, batch: Dict, responses_data: Dict) -> Dict:
        """Compute AGRPO losses"""
        total_policy_loss = 0
        total_kl_loss = 0
        num_examples = len(responses_data['rewards'])
        
        with self.accelerator.accumulate(self.model):
            for i in range(num_examples):
                # Get log probabilities for this example's responses
                log_probs_old = []
                log_probs_new = []
                
                for j in range(self.config.num_responses_per_query):
                    idx = i * self.config.num_responses_per_query + j
                    input_length = responses_data['input_ids'][i].shape[0]
                    
                    # Compute log prob under old policy
                    with torch.no_grad():
                        outputs_old = self.ref_model(
                            input_ids=responses_data['generated_ids'][idx].unsqueeze(0),
                            attention_mask=torch.ones_like(
                                responses_data['generated_ids'][idx].unsqueeze(0)
                            ).to(self.device)
                        )
                        logits_old = outputs_old.logits[0, input_length-1:-1]
                        gen_tokens = responses_data['generated_ids'][idx][input_length:]
                        
                        log_probs = F.log_softmax(logits_old, dim=-1)
                        token_log_probs = log_probs[range(len(gen_tokens)), gen_tokens]
                        log_prob_old = token_log_probs.sum()
                        log_probs_old.append(log_prob_old.item())
                    
                    # Compute log prob under new policy
                    outputs_new = self.model(
                        input_ids=responses_data['generated_ids'][idx].unsqueeze(0),
                        attention_mask=torch.ones_like(
                            responses_data['generated_ids'][idx].unsqueeze(0)
                        ).to(self.device)
                    )
                    logits_new = outputs_new.logits[0, input_length-1:-1]
                    
                    log_probs = F.log_softmax(logits_new, dim=-1)
                    token_log_probs = log_probs[range(len(gen_tokens)), gen_tokens]
                    log_prob_new = token_log_probs.sum()
                    log_probs_new.append(log_prob_new)
                
                # Compute advantages
                advantages, group_quality, w_rel = self.agrpo_algo.compute_advantages(
                    responses_data['rewards'][i],
                    log_probs_old,
                    self.device
                )
                
                # Stack log probs
                log_probs_old_tensor = torch.tensor(log_probs_old, device=self.device)
                log_probs_new_tensor = torch.stack(log_probs_new)
                
                # Compute ratios
                ratios = torch.exp(log_probs_new_tensor - log_probs_old_tensor)
                
                # Compute losses
                policy_loss = self.agrpo_algo.compute_policy_loss(ratios, advantages)
                kl_div = self.agrpo_algo.compute_kl_divergence(
                    log_probs_old_tensor, log_probs_new_tensor
                )
                
                total_loss = self.agrpo_algo.compute_total_loss(policy_loss, kl_div)
                
                # Scale by gradient accumulation
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(scaled_loss)
                
                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_div.item()
        
        return {
            'loss': (total_policy_loss + self.config.kl_penalty * total_kl_loss) / num_examples,
            'policy_loss': total_policy_loss / num_examples,
            'kl_loss': total_kl_loss / num_examples,
        }
    
    def _compute_metrics(self, responses_data: Dict, loss_data: Dict) -> Dict:
        """Compute batch metrics"""
        all_rewards_flat = [r for rewards in responses_data['rewards'] for r in rewards]
        
        total_reward = sum(all_rewards_flat)
        correct_count = sum(1 for r in all_rewards_flat if r == 1.0)
        total_responses = len(all_rewards_flat)
        
        return {
            'total_reward': total_reward,
            'correct_count': correct_count,
            'total_responses': total_responses,
            'avg_reward': total_reward / total_responses if total_responses > 0 else 0,
            'accuracy': correct_count / total_responses if total_responses > 0 else 0,
            **loss_data
        }