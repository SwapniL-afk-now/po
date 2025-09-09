# data/dataset.py

import re
from typing import Dict, List, Optional
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

class GSM8KDataset(Dataset):
    """GSM8K dataset for AGRPO training"""
    
    def __init__(self, config, tokenizer, prompts):
        self.config = config
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.data = self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print(f"Loading {self.config.dataset_name} dataset...")
        full_dataset = load_dataset(
            self.config.dataset_name, 
            "main", 
            split=self.config.dataset_split
        )
        
        # Sample if needed
        if self.config.sample_size and self.config.sample_size < len(full_dataset):
            np.random.seed(self.config.seed)
            indices = np.random.choice(len(full_dataset), size=self.config.sample_size, replace=False)
            dataset = full_dataset.select(indices)
            print(f"Sampled {len(dataset)} examples from {len(full_dataset)}")
        else:
            dataset = full_dataset
        
        # Prepare examples
        prepared_data = []
        for example in dataset:
            query = self._format_prompt(example['question'])
            ground_truth = self._extract_answer(example['answer'])
            
            if ground_truth is not None:
                prepared_data.append({
                    'query': query,
                    'ground_truth': ground_truth,
                    'question': example['question'],
                    'full_answer': example['answer']
                })
        
        print(f"Prepared {len(prepared_data)} examples")
        return prepared_data
    
    def _format_prompt(self, question: str) -> str:
        """Format question into model prompt"""
        prompt = f"<|im_start|>system\n{self.prompts.system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt
    
    def _extract_answer(self, answer_text: str) -> Optional[str]:
        """Extract numerical answer from GSM8K format"""
        if "####" in answer_text:
            final_answer = answer_text.split("####")[-1].strip()
            numbers = re.findall(r"-?\d+\.?\d*", final_answer)
            if numbers:
                answer = numbers[0]
                # Clean up decimal points
                if '.' in answer:
                    if answer.endswith('.0'):
                        answer = answer[:-2]
                    else:
                        try:
                            if float(answer) == int(float(answer)):
                                answer = str(int(float(answer)))
                        except:
                            pass
                return answer
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict], tokenizer, max_length: int = 1024):
    """Custom collate function for batching"""
    queries = [item['query'] for item in batch]
    ground_truths = [item['ground_truth'] for item in batch]
    questions = [item['question'] for item in batch]
    
    # Tokenize queries
    inputs = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'ground_truths': ground_truths,
        'queries': queries,
        'questions': questions
    }


def create_dataloader(config, tokenizer, prompts):
    """Create dataloader for training"""
    dataset = GSM8KDataset(config, tokenizer, prompts)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader, dataset