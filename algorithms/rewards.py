# algorithms/rewards.py

import re
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class RewardExtractor:
    """Extract answers and compute rewards from model outputs"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Answer extraction patterns
        self.answer_tag_patterns = [
            r'<answer>(.*?)</answer>',
            r'\[answer\](.*?)\[/answer\]',
            r'<answer>(.*?)$',  # Unclosed tag at end
        ]
        
        self.answer_phrase_patterns = [
            r'[Tt]he answer is[:\s]+(-?\d+\.?\d*)',
            r'[Ff]inal answer[:\s]+(-?\d+\.?\d*)',
            r'[Aa]nswer[:\s]+(-?\d+\.?\d*)',
            r'equals?[:\s]+(-?\d+\.?\d*)',
            r'[Ss]olution[:\s]+(-?\d+\.?\d*)',
        ]
    
    def extract_answer(self, generated_text: str) -> Tuple[Optional[str], str]:
        """
        Extract numerical answer from generated text
        
        Args:
            generated_text: Model's generated response
            
        Returns:
            extracted_answer: The extracted numerical answer (or None)
            extraction_method: Method used for extraction
        """
        # Try answer tags first
        for pattern in self.answer_tag_patterns:
            match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                # Extract number from answer text
                number_match = re.search(r'-?\d+\.?\d*', answer_text)
                if number_match:
                    answer = self._clean_number(number_match.group(0))
                    if self.debug:
                        logger.info(f"Extracted '{answer}' using answer tags")
                    return answer, "answer_tags"
        
        # Try answer phrases
        for pattern in self.answer_phrase_patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE)
            if match:
                answer = self._clean_number(match.group(1))
                if self.debug:
                    logger.info(f"Extracted '{answer}' using answer phrase")
                return answer, "answer_phrase"
        
        # Fallback: look for last number in the text
        last_part = generated_text[-300:] if len(generated_text) > 300 else generated_text
        numbers = re.findall(r'-?\d+\.?\d*', last_part)
        if numbers:
            answer = self._clean_number(numbers[-1])
            if self.debug:
                logger.info(f"Extracted '{answer}' using fallback (last number)")
            return answer, "fallback"
        
        if self.debug:
            logger.warning("No answer extracted from generated text")
        return None, "no_extraction"
    
    def _clean_number(self, number_str: str) -> str:
        """Clean and normalize a number string"""
        # Remove unnecessary decimal points
        if '.' in number_str:
            try:
                num = float(number_str)
                if num == int(num):
                    return str(int(num))
                return number_str
            except ValueError:
                return number_str
        return number_str
    
    def compute_reward(
        self,
        generated_text: str,
        ground_truth: str
    ) -> Tuple[float, str]:
        """
        Compute reward by comparing extracted answer with ground truth
        
        Args:
            generated_text: Model's generated response
            ground_truth: Correct answer
            
        Returns:
            reward: 1.0 if correct, -1.0 if incorrect
            info: Information about extraction and answer
        """
        extracted_answer, method = self.extract_answer(generated_text)
        
        if extracted_answer is None:
            return -1.0, f"No answer found (method: {method})"
        
        # Compare with ground truth
        is_correct = extracted_answer == ground_truth
        reward = 1.0 if is_correct else -1.0
        
        info = f"{extracted_answer} ({method}) {'✓' if is_correct else '✗'}"
        
        return reward, info
    
    def batch_compute_rewards(
        self,
        generated_texts: List[str],
        ground_truths: List[str]
    ) -> Tuple[List[float], List[str]]:
        """
        Compute rewards for a batch of generated texts
        
        Args:
            generated_texts: List of model responses
            ground_truths: List of correct answers
            
        Returns:
            rewards: List of rewards
            infos: List of extraction information
        """
        rewards = []
        infos = []
        
        for gen_text, ground_truth in zip(generated_texts, ground_truths):
            reward, info = self.compute_reward(gen_text, ground_truth)
            rewards.append(reward)
            infos.append(info)
        
        return rewards, infos