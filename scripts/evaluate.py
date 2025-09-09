# scripts/evaluate.py

"""
Evaluation script for trained AGRPO models
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.training_config import TrainingConfig
from models.model_manager import ModelManager
from algorithms.rewards import RewardExtractor
from data.dataset import create_dataloader
from data.prompts import Prompts

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate AGRPO trained model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Base model name")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k",
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    return parser.parse_args()

def evaluate_model(model, tokenizer, dataloader, reward_extractor, device, im_end_token_id):
    """Evaluate model on dataset"""
    
    model.eval()
    
    total_correct = 0
    total_samples = 0
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate responses
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                temperature=0.1,  # Lower temperature for evaluation
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_token_id,
            )
            
            # Process each sample
            for i in range(len(batch['ground_truths'])):
                input_length = input_ids[i].shape[0]
                gen_text = tokenizer.decode(
                    generated_ids[i][input_length:],
                    skip_special_tokens=False
                )
                
                if '<|im_end|>' in gen_text:
                    gen_text = gen_text.split('<|im_end|>')[0]
                
                # Compute reward
                reward, info = reward_extractor.compute_reward(
                    gen_text, batch['ground_truths'][i]
                )
                
                is_correct = reward == 1.0
                if is_correct:
                    total_correct += 1
                total_samples += 1
                
                # Store result
                all_results.append({
                    'question': batch['questions'][i],
                    'ground_truth': batch['ground_truths'][i],
                    'generated_answer': info.split()[0],  # Extract just the answer
                    'is_correct': is_correct,
                    'full_response': gen_text[:500],  # Truncate for storage
                })
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'results': all_results
    }

def main():
    """Main evaluation function"""
    args = parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Create minimal config
    config = TrainingConfig(
        model_name=args.base_model,
        dataset_name=args.dataset,
        dataset_split=args.split,
        per_device_train_batch_size=args.batch_size,
        sample_size=args.max_samples,
    )
    
    # Initialize components
    model_manager = ModelManager(config)
    tokenizer = model_manager.setup_tokenizer()
    model = model_manager.setup_model(checkpoint_path=args.model_path)
    model = model.to(args.device)
    
    # Create dataloader
    prompts = Prompts()
    dataloader, dataset = create_dataloader(config, tokenizer, prompts)
    
    # Initialize reward extractor
    reward_extractor = RewardExtractor(debug=False)
    
    # Get special tokens
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Run evaluation
    print(f"Evaluating on {len(dataset)} samples...")
    results = evaluate_model(
        model, tokenizer, dataloader, reward_extractor, 
        args.device, im_end_token_id
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Correct: {results['total_correct']}/{results['total_samples']}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
