"""
OPRO-based optimizer for Ava's prompt optimization.

This module implements an OPRO optimizer specifically for optimizing Ava's
identify_partial prompts to improve performance on cancel_not_for_all and 
partial_or_full classification tasks.

Key components adapted from OPRO:
- Meta-prompt generation for prompt optimization
- Candidate prompt generation using LLM
- Duplicate detection using MD5 hashing
- Top-k prompt tracking and selection
- Comprehensive logging and result storage
"""

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import yaml
import re
from collections import Counter
from os.path import join
from src.utils.open_ai_util import OpenAIClient
from src.utils.data_util import load_json_file, save_to_json
from src.utils.eval_prompt_util import substitute_prompt_variables, parse_model_response, compute_binary_metrics


def evaluate_single_instruction(args_tuple):
    """
    Evaluate a single instruction in a separate process for parallel processing.
    
    Args:
        args_tuple: Tuple containing (instruction, train_data, api_key, scorer_model, 
                   initial_prompt_response_format, save_folder, step_num)
    
    Returns:
        Dict containing all evaluation results and metadata
    """
    instruction, train_data, api_key, scorer_model, initial_prompt_response_format, save_folder, step_num = args_tuple
    
    try:
        # Initialize OpenAI client for evaluation
        eval_client = OpenAIClient(api_key=api_key)
        
        # Create full prompt by combining body with response format
        full_prompt = instruction + "\n\n" + initial_prompt_response_format
        
        # Prepare evaluation data
        results = []
        prediction_errors = []
        
        # Process each test case
        for element_id, element in tqdm(train_data.items(), desc=f"Evaluating {instruction[:30]}..."):
            # Extract required data
            ava_data = element.get("Ava", {})
            flight_booking_legs = ava_data.get("flight_booking_legs", "")
            chat_history = ava_data.get("chat_history", "")
            
            # Ground truth labels
            gt_cancel_not_for_all = element.get("cancel_not_for_all")
            gt_partial_or_full = element.get("partial_or_full")
            
            # Skip if missing required data
            if gt_cancel_not_for_all is None:
                continue
            
            # Substitute variables in full prompt
            filled_prompt = substitute_prompt_variables(full_prompt, flight_booking_legs, chat_history)
            
            # Call OpenAI API
            try:
                response = eval_client.call_openai_with_retry(
                    prompt=filled_prompt,
                    model=scorer_model,
                    temperature=0,
                    max_tokens=2048
                )
                
                # Parse response
                parsed_response = parse_model_response(response)
                
                if parsed_response:
                    pred_cancel_not_for_all = parsed_response.get("cancel_not_for_all_passengers")
                    pred_partial_or_full = parsed_response.get("partial_or_full")
                    
                    # Store result
                    results.append({
                        'element_id': element_id,
                        'gt_cancel_not_for_all': gt_cancel_not_for_all,
                        'pred_cancel_not_for_all': pred_cancel_not_for_all,
                        'gt_partial_or_full': gt_partial_or_full,
                        'pred_partial_or_full': pred_partial_or_full,
                        'raw_response': response,
                        'parsed_response': parsed_response
                    })
                else:
                    prediction_errors.append({
                        'element_id': element_id,
                        'error': 'Failed to parse JSON response',
                        'raw_response': response
                    })
                    
            except Exception as e:
                prediction_errors.append({
                    'element_id': element_id,
                    'error': str(e),
                    'raw_response': None
                })
        
        # Compute metrics for cancel_not_for_all
        cancel_not_for_all_results = [r for r in results if r['pred_cancel_not_for_all'] is not None]
        
        if cancel_not_for_all_results:
            cancel_gt = [r['gt_cancel_not_for_all'] for r in cancel_not_for_all_results]
            cancel_pred = [r['pred_cancel_not_for_all'] for r in cancel_not_for_all_results]
            cancel_metrics = compute_binary_metrics(cancel_gt, cancel_pred)
        else:
            cancel_metrics = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'balanced_accuracy': 0.0, 'adjusted_balanced_accuracy': 0.0
            }
        
        # Compute metrics for partial_or_full (ignore null)
        partial_results = [
            r for r in results 
            if r['gt_partial_or_full'] != "null" and r['pred_partial_or_full'] is not None
        ]
        
        if partial_results:
            # Convert to binary: partial=True, full=False
            partial_gt = [r['gt_partial_or_full'].lower() == "partial" for r in partial_results]
            partial_pred = [r['pred_partial_or_full'].lower() == "partial" for r in partial_results]
            partial_metrics = compute_binary_metrics(partial_gt, partial_pred)
        else:
            partial_metrics = {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'balanced_accuracy': 0.0, 'adjusted_balanced_accuracy': 0.0
            }
        
        # Get adjusted balanced accuracy scores
        cancel_adj_b_acc = cancel_metrics.get('adjusted_balanced_accuracy', 0.0)
        partial_adj_b_acc = partial_metrics.get('adjusted_balanced_accuracy', 0.0)
        
        # Combined score using Harmonic Mean (penalizes imbalanced performance)
        if cancel_adj_b_acc > 0 and partial_adj_b_acc > 0:
            combined_score = 2 * (cancel_adj_b_acc * partial_adj_b_acc) / (cancel_adj_b_acc + partial_adj_b_acc)
        else:
            combined_score = 0.0
        
        # Compile results
        evaluation_results = {
            'total_cases': len(train_data),
            'successful_predictions': len(results),
            'prediction_errors': len(prediction_errors),
            'cancel_not_for_all_metrics': cancel_metrics,
            'partial_or_full_metrics': partial_metrics,
            'detailed_results': results,
            'errors': prediction_errors,
            'combined_score': combined_score
        }
        
        # Generate filename for saving
        instruction_hash = hashlib.md5(instruction.encode()).hexdigest()
        filename = f"{instruction_hash}"
        result_file_path = os.path.join(save_folder, f"{filename}.json")
        
        # Return all results for aggregation
        return {
            'instruction': instruction,
            'combined_score': combined_score,
            'cancel_adj_b_acc': cancel_adj_b_acc,
            'partial_adj_b_acc': partial_adj_b_acc,
            'evaluation_results': evaluation_results,
            'result_file_path': result_file_path,
            'step_num': step_num,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'instruction': instruction,
            'combined_score': 0.0,
            'cancel_adj_b_acc': 0.0,
            'partial_adj_b_acc': 0.0,
            'evaluation_results': {},
            'result_file_path': None,
            'step_num': step_num,
            'success': False,
            'error': str(e)
        }


            

class AvaOproOptimizer:
    """OPRO optimizer for Ava's prompt optimization."""
    
    def __init__(
        self,
        train_data_path: str = "data/processed/logs/04222025-08182025/ground_truth/verified_ground_truth_balance_train.json",
        initial_prompt_file: str = "prompts/original/identify_partial.yaml",
        initial_prompt_key: str = "initial_prompt",
        api_key: Optional[str] = None,
        save_folder: str = "results/gpt-5-verified/",
        num_search_steps: int = 10,
        num_generated_instructions_in_each_step: int = 4,
        optimizer_model: str = "gpt-4o",
        scorer_model: str = "gpt-4o-mini",
        optimizer_temperature: float = 1.0,
        train_ratio: float = 1.0,
        num_examples: int = 3,
        max_processes: int = None,
        verbose: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the OPRO optimizer for Ava prompts.
        
        Args:
            train_data_path: Path to training data JSON
            initial_prompt_file: Path to YAML file with initial prompt
            initial_prompt_key: Key in YAML file for prompt to optimize
            api_key: OpenAI API key
            save_folder: Directory to save results
            max_processes: Maximum number of parallel processes (None = CPU count)
            verbose: Whether to print detailed progress
            random_seed: Random seed for reproducibility
        """
        self.train_data_path = train_data_path
        self.initial_prompt_file = initial_prompt_file
        self.initial_prompt_key = initial_prompt_key
        self.initial_prompt_body_key = initial_prompt_key + "_body"
        self.initial_prompt_response_format_key = initial_prompt_key + "_response_format"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.save_folder = join(save_folder, f"scorer_{scorer_model}", f"optimizer_{optimizer_model}", f"train_ratio_{train_ratio}", f"num_search_steps_{num_search_steps}", f"num_gen_inst_{num_generated_instructions_in_each_step}_num_exp_{num_examples}_opt_temperature_{optimizer_temperature}")
        self.num_search_steps=num_search_steps
        self.num_generated_instructions_in_each_step=num_generated_instructions_in_each_step
        self.optimizer_model=optimizer_model
        self.scorer_model=scorer_model
        self.optimizer_temperature=optimizer_temperature
        self.train_ratio=train_ratio
        self.num_examples=num_examples
        self.max_processes = max_processes
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        self._set_random_seeds(random_seed)
        
        # Initialize OpenAI client
        self.client = OpenAIClient(api_key=self.api_key)
        
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        
        # Load initial prompt parts
        self.initial_prompt_body, self.initial_prompt_response_format = self._load_initial_prompt_parts()
        
        # For backward compatibility, store full prompt (body only for optimization)
        self.initial_prompt = self.initial_prompt_body
        
        # OPRO configuration
        self.old_instruction_score_threshold = 0.5
        self.max_num_instructions = 10
        self.num_score_buckets = 100
        
        # Tracking variables (similar to OPRO)
        self.old_instructions_and_scores = []  # (prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step)
        self.meta_prompts = []  # (meta_prompt, step)
        self.old_instruction_md5_hashstrings_set = set()
        self.step_statistics = []  # Step-level statistics for comprehensive tracking
        
    def _load_initial_prompt_parts(self) -> Tuple[str, str]:
        """Load initial prompt body and response format from YAML file."""
        try:
            with open(self.initial_prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Initial prompt file not found: {self.initial_prompt_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {self.initial_prompt_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading initial prompt file {self.initial_prompt_file}: {e}")
        
        if not isinstance(prompt_data, dict):
            raise ValueError(f"YAML file {self.initial_prompt_file} must contain a dictionary at root level")
        
        # Check if both keys exist
        if self.initial_prompt_body_key not in prompt_data:
            raise ValueError(f"Key '{self.initial_prompt_body_key}' not found in {self.initial_prompt_file}")
        
        if self.initial_prompt_response_format_key not in prompt_data:
            raise ValueError(f"Key '{self.initial_prompt_response_format_key}' not found in {self.initial_prompt_file}")
        
        prompt_body = prompt_data[self.initial_prompt_body_key].strip()
        response_format = prompt_data[self.initial_prompt_response_format_key].strip()
        
        return prompt_body, response_format
    
    def _create_full_prompt(self, prompt_body: str) -> str:
        """Combine prompt body with response format to create full prompt."""
        return f"{prompt_body.strip()}\n\n{self.initial_prompt_response_format.strip()}"
    
    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
        if self.verbose:
            print(f"🎲 Random seeds set to {seed} for reproducibility")
    
    def _instruction_to_filename(self, instruction: str, md5_hashing: bool = True) -> str:
        """Convert instruction to filename (adapted from OPRO)."""
        if md5_hashing:
            m = hashlib.md5()
            m.update(instruction.encode("utf-8"))
            filename = m.hexdigest()
        else:
            # Remove punctuation and line breaks
            filename = instruction.replace("\n", "").replace("\r", "")
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = filename if filename else "NO_INSTRUCTION"
        return filename
    
    def _polish_prompt(self, prompt: str) -> str:
        """Polish prompt to standardize format."""
        prompt = prompt.strip()
        if prompt and not prompt.endswith('.'):
            prompt += '.'
        return prompt
    
    def _bucketize_score(self, score: float, n_buckets: int = 20) -> int:
        """Convert float score to bucket for meta-prompt display."""
        return round(score * n_buckets)
    
    def _gen_prompt_score_pairs_substr(
        self,
        old_prompts_and_scores: List[Tuple[str, float, float, float, int]],
        score_threshold: float = 0.1,
        max_num_prompts: int = 20,
        return_str_only: bool = False
    ) -> str:
        """Generate substring with prompt-score pairs for meta-prompt."""
        # Sort by combined score first, then by word count (fewer words is better)
        def sort_key(x):
            prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step = x
            word_count = len(prompt.split())
            return (-combined_score, word_count)  # Higher score first, then fewer words
        
        sorted_prompts = sorted(old_prompts_and_scores, key=sort_key)[:max_num_prompts]
        
        prompts_in_meta_prompt = []
        prompt_score_str = ""
        
        for prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step in sorted_prompts:
            if combined_score >= score_threshold:
                prompts_in_meta_prompt.append((prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step))
                combined_score_display = self._bucketize_score(combined_score, self.num_score_buckets)
                cancel_adj_b_acc_display = self._bucketize_score(cancel_adj_b_acc, self.num_score_buckets)
                partial_adj_b_acc_display = self._bucketize_score(partial_adj_b_acc, self.num_score_buckets)
                word_count = len(prompt.split())
                prompt_score_str += f"\ntext:\n{prompt}\ncombined_score:\n{combined_score_display}\ncancel_adj_b_acc:\n{cancel_adj_b_acc_display}\npartial_adj_b_acc:\n{partial_adj_b_acc_display}\nwords:\n{word_count}\n"
        
        if return_str_only:
            return prompt_score_str
        else:
            return prompt_score_str, prompts_in_meta_prompt
    
    def _sample_few_shot_examples(self, train_data: Dict, num_examples: int = 3) -> str:
        """Sample few-shot examples from training data."""
        if num_examples == 0 or not train_data:
            return ""
        
        # Sample random examples
        sample_keys = np.random.choice(list(train_data.keys()), 
                                      min(num_examples, len(train_data)), 
                                      replace=False)
        
        examples_str = "\n\nHere are some examples from the training data:\n"
        for i, key in enumerate(sample_keys):
            element = train_data[key]
            ava_data = element.get("Ava", {})
            flight_booking_legs = ava_data.get("flight_booking_legs", "")
            chat_history = ava_data.get("chat_history", "")
            cancel_not_for_all = element.get("cancel_not_for_all", False)
            partial_or_full = element.get("partial_or_full", "FULL")
            
            examples_str += f"\nExample {i+1}:\n"
            examples_str += f"Flight booking legs: {flight_booking_legs}\n"
            examples_str += f"Chat history: {chat_history}\n"
            examples_str += f"Ground truth - cancel_not_for_all_passengers: {cancel_not_for_all}\n"
            examples_str += f"Ground truth - partial_or_full: {partial_or_full}\n"
        
        return examples_str
    
    def _generate_meta_prompt(
        self,
        old_prompts_and_scores: List[Tuple[str, float, int]],
        train_data: Dict,
        num_examples: int = 3,
    ) -> str:
        """Generate meta-prompt for prompt optimization."""
        
        prompt_score_str, _ = self._gen_prompt_score_pairs_substr(
            old_prompts_and_scores,
            self.old_instruction_score_threshold,
            self.max_num_instructions
        )
        
        # Add few-shot examples
        examples_str = self._sample_few_shot_examples(train_data, num_examples)
        
        meta_prompt = f"""Your task is to generate prompts for a classification task. The prompts will be used to classify flight cancellation conversations into two categories:

1. cancel_not_for_all_passengers: Whether the user wants to cancel for fewer than all passengers
2. partial_or_full: Whether the user wants partial cancellation (specific legs) or full cancellation

Below are some previous prompts with their scores. Each prompt has three scores:
- combined_score: Harmonic mean of cancel_adj_b_acc and partial_adj_b_acc (0 to {self.num_score_buckets}, higher is better)
- cancel_adj_b_acc: Adjusted balanced accuracy for cancel_not_for_all_passengers classification (0 to {self.num_score_buckets}, higher is better)  
- partial_adj_b_acc: Adjusted balanced accuracy for partial_or_full classification (0 to {self.num_score_buckets}, higher is better)
- words: Word count (fewer is better when scores are equal)
{prompt_score_str}{examples_str}

Your task is to generate a new prompt that will achieve a higher score than the previous ones while being concise.

Requirements for the new prompt:
- It should help the AI assistant accurately classify cancel_not_for_all_passengers and partial_or_full
- Keep it concise while maintaining accuracy

Generate a new concise prompt that will improve classification accuracy. Output your prompt between <PROMPT> and </PROMPT> tags.

<PROMPT>
"""
        
        return meta_prompt
    
    def _parse_generated_prompts(self, raw_outputs: List[str]) -> List[str]:
        """Extract prompts from LLM outputs."""
        prompts = []
        
        for output in raw_outputs:
            # Look for content between <PROMPT> and </PROMPT> tags
            pattern = r'<PROMPT>(.*?)</PROMPT>'
            matches = re.findall(pattern, output, re.DOTALL)
            
            for match in matches:
                prompt = match.strip()
                if prompt:
                    prompts.append(prompt)
        
        return prompts
    
    def _filter_generated_prompts(self, generated_prompts: List[str], step: int) -> List[str]:
        """Filter generated prompts (remove duplicates, too long, etc.)."""
        filtered_prompts = []
        
        for prompt in generated_prompts:
            # Skip if too long
            if len(prompt) > 1024:
                if self.verbose:
                    print(f"Step {step}: Prompt too long ({len(prompt)} chars), skipped")
                continue
            
            # Skip if contains problematic content
            if any(tag in prompt.upper() for tag in ['<PROMPT>', '</PROMPT>', 'PROMPT>']):
                if self.verbose:
                    print(f"Step {step}: Prompt contains tags, skipped")
                continue
            
            # Check for duplicates using MD5
            prompt_md5 = self._instruction_to_filename(prompt, md5_hashing=True)
            if prompt_md5 not in self.old_instruction_md5_hashstrings_set:
                filtered_prompts.append(prompt)
                self.old_instruction_md5_hashstrings_set.add(prompt_md5)
            else:
                if self.verbose:
                    print(f"Step {step}: Duplicate prompt detected, skipped")
        
        return list(set(filtered_prompts))  # Remove any remaining duplicates
    
    def _evaluate_prompt_performance(
        self,
        prompt_body: str,
        train_data: Dict,
        model: str = "gpt-4o",
        temperature: float = 0
    ) -> Tuple[float, float, float, Dict]:
        """
        Evaluate a single prompt body and return combined score using harmonic mean.
        
        Returns:
            Tuple of (combined_score, cancel_adj_b_acc, partial_adj_b_acc, evaluation_results)
            where combined_score is the harmonic mean of the two adjusted balanced accuracies.
        """
        
        try:
            
            # Initialize OpenAI client for evaluation
            eval_client = OpenAIClient(api_key=self.api_key)
            
            # Create full prompt by combining body with response format
            full_prompt = self._create_full_prompt(prompt_body)
            
            # Prepare evaluation data
            results = []
            prediction_errors = []
            
            # Process each test case
            for element_id, element in tqdm(train_data.items(), desc="Evaluating prompt"):
                # Extract required data
                ava_data = element.get("Ava", {})
                flight_booking_legs = ava_data.get("flight_booking_legs", "")
                chat_history = ava_data.get("chat_history", "")
                
                # Ground truth labels
                gt_cancel_not_for_all = element.get("cancel_not_for_all")
                gt_partial_or_full = element.get("partial_or_full")
                
                # Skip if missing required data
                if gt_cancel_not_for_all is None:
                    continue
                
                # Substitute variables in full prompt
                filled_prompt = substitute_prompt_variables(full_prompt, flight_booking_legs, chat_history)
                
                # Call OpenAI API
                try:
                    response = eval_client.call_openai_with_retry(
                        prompt=filled_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=1024
                    )
                    
                    # Parse response
                    parsed_response = parse_model_response(response)
                    
                    if parsed_response:
                        pred_cancel_not_for_all = parsed_response.get("cancel_not_for_all_passengers")
                        pred_partial_or_full = parsed_response.get("partial_or_full")
                        
                        # Store result
                        results.append({
                            'element_id': element_id,
                            'gt_cancel_not_for_all': gt_cancel_not_for_all,
                            'pred_cancel_not_for_all': pred_cancel_not_for_all,
                            'gt_partial_or_full': gt_partial_or_full,
                            'pred_partial_or_full': pred_partial_or_full,
                            'raw_response': response,
                            'parsed_response': parsed_response
                        })
                    else:
                        prediction_errors.append({
                            'element_id': element_id,
                            'error': 'Failed to parse JSON response',
                            'raw_response': response
                        })
                        
                except Exception as e:
                    prediction_errors.append({
                        'element_id': element_id,
                        'error': str(e),
                        'raw_response': None
                    })
            
            # Compute metrics for cancel_not_for_all
            cancel_not_for_all_results = [r for r in results if r['pred_cancel_not_for_all'] is not None]
            
            if cancel_not_for_all_results:
                cancel_gt = [r['gt_cancel_not_for_all'] for r in cancel_not_for_all_results]
                cancel_pred = [r['pred_cancel_not_for_all'] for r in cancel_not_for_all_results]
                cancel_metrics = compute_binary_metrics(cancel_gt, cancel_pred)
            else:
                cancel_metrics = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                    'balanced_accuracy': 0.0, 'adjusted_balanced_accuracy': 0.0
                }
            
            # Compute metrics for partial_or_full (ignore null)
            partial_results = [
                r for r in results 
                if r['gt_partial_or_full'] != "null" and r['pred_partial_or_full'] is not None
            ]
            
            if partial_results:
                # Convert to binary: partial=True, full=False
                partial_gt = [r['gt_partial_or_full'].lower() == "partial" for r in partial_results]
                partial_pred = [r['pred_partial_or_full'].lower() == "partial" for r in partial_results]
                partial_metrics = compute_binary_metrics(partial_gt, partial_pred)
            else:
                partial_metrics = {
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                    'balanced_accuracy': 0.0, 'adjusted_balanced_accuracy': 0.0
                }
            
            # Get adjusted balanced accuracy scores as the optimization objective
            cancel_adj_b_acc = cancel_metrics.get('adjusted_balanced_accuracy', 0.0)
            partial_adj_b_acc = partial_metrics.get('adjusted_balanced_accuracy', 0.0)
            
            # Combined score using Harmonic Mean (penalizes imbalanced performance)
            if cancel_adj_b_acc > 0 and partial_adj_b_acc > 0:
                combined_score = 2 * (cancel_adj_b_acc * partial_adj_b_acc) / (cancel_adj_b_acc + partial_adj_b_acc)
            else:
                # Handle edge case where one score is 0 or negative
                combined_score = 0.0
            
            # Compile results
            evaluation_results = {
                'total_cases': len(train_data),
                'successful_predictions': len(results),
                'prediction_errors': len(prediction_errors),
                'cancel_not_for_all_metrics': cancel_metrics,
                'partial_or_full_metrics': partial_metrics,
                'detailed_results': results,
                'errors': prediction_errors,
                'combined_score': combined_score
            }
            
            return combined_score, cancel_adj_b_acc, partial_adj_b_acc, evaluation_results
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating prompt: {e}")
            return 0.0, 0.0, 0.0, {}
    
    def _evaluate_prompt_on_subset(
        self,
        prompt: str,
        train_data_subset: Dict,
        model: str = "gpt-4o"
    ) -> Tuple[float, float, float]:
        """Evaluate prompt on a subset of training data for faster iteration."""
        # This is a simplified version for faster evaluation during optimization
        # We could implement a faster evaluation here if needed
        combined_score, cancel_adj_b_acc, partial_adj_b_acc, _ = self._evaluate_prompt_performance(prompt, train_data_subset, model)
        return combined_score, cancel_adj_b_acc, partial_adj_b_acc
    
    def optimize(
        self,
        
    ) -> Dict[str, Any]:
        """
        Run OPRO optimization for Ava prompts following OPRO's structure.
        
        Args:
            num_search_steps: Number of optimization steps
            num_generated_instructions_in_each_step: Number of candidate prompts per step
            optimizer_model: Model for generating new prompts
            scorer_model: Model for evaluating prompts
            optimizer_temperature: Temperature for prompt generation
            train_ratio: Fraction of training data to use
            num_examples: Number of examples to include in meta-prompt
            
        Returns:
            Dictionary with optimization results
        """
        
        if self.verbose:
            print(f"🚀 Starting Ava OPRO optimization")
            print(f"📊 Steps: {self.num_search_steps}, Instructions per step: {self.num_generated_instructions_in_each_step}")
            print(f"🤖 Optimizer: {self.optimizer_model}, Scorer: {self.scorer_model}")
            print(f"📈 Train ratio: {self.train_ratio}, Examples in meta-prompt: {self.num_examples}")
        
        # Load and sample training data
        full_train_data = load_json_file(self.train_data_path)
        if not full_train_data:
            raise ValueError(f"No training data found in {self.train_data_path}")
            
        if self.train_ratio < 1.0:
            sample_size = max(1, int(len(full_train_data) * self.train_ratio))  # Ensure at least 1 sample
            sample_size = min(sample_size, len(full_train_data))  # Don't exceed available data
            train_keys = np.random.choice(list(full_train_data.keys()), sample_size, replace=False)
            train_data = {k: full_train_data[k] for k in train_keys}
        else:
            train_data = full_train_data
        
        if self.verbose:
            print(f"📁 Training data: {len(train_data)}/{len(full_train_data)} samples")
        
        # ============== Evaluating initial instructions ===============
        if self.verbose:
            print(f"\n============== Evaluating initial instructions ===============")
        
        print(f'Computing the score of "{self.initial_prompt[:50]}..." by prompting')
        
        initial_combined_score, initial_cancel_adj_b_acc, initial_partial_adj_b_acc, initial_results = self._evaluate_prompt_performance(
            self.initial_prompt, train_data, self.scorer_model
        )
        
        print(f"instruction: {self.initial_prompt[:50]}..., combined_score: {initial_combined_score:.3f}, cancel_adj_b_acc: {initial_cancel_adj_b_acc:.3f}, partial_adj_b_acc: {initial_partial_adj_b_acc:.3f}")
        
        # Save initial results
        initial_filename = self._instruction_to_filename(self.initial_prompt)
        result_file_path = os.path.join(self.save_folder, f"{initial_filename}.json")
        with open(result_file_path, 'w') as f:
            json.dump(initial_results, f, indent=2)
        print(f'Saving results of "{self.initial_prompt[:50]}..." to {result_file_path}')
        
        # Add initial prompt to tracking
        initial_word_count = len(self.initial_prompt.split(" "))
        self.old_instructions_and_scores.append((self.initial_prompt, initial_combined_score, initial_cancel_adj_b_acc, initial_partial_adj_b_acc, -1))
        
        # Add initial prompt MD5 to hash set for duplicate detection (following OPRO pattern)
        initial_prompt_md5 = self._instruction_to_filename(self.initial_prompt, md5_hashing=True)
        self.old_instruction_md5_hashstrings_set.add(initial_prompt_md5)
        
        # ============== Evolution ===============
        for i_step in tqdm(range(self.num_search_steps)):
            print(f"\n================== Step {i_step} =====================")
            
            print(f"current optimizer_temperature: {self.optimizer_temperature}")
            
            # Generate meta-prompt
            meta_prompt = self._generate_meta_prompt(
                self.old_instructions_and_scores,
                train_data,
                self.num_examples,
            )
            
            print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
            self.meta_prompts.append((meta_prompt, i_step))
            
            # Generate new instructions
            remaining_num_instructions_to_generate = self.num_generated_instructions_in_each_step
            generated_instructions_raw = []
            
            while remaining_num_instructions_to_generate > 0:
                optimizer_llm_input_text = meta_prompt
                # Generate instructions
                print(f"current temperature: {self.optimizer_temperature}")
                
                try:
                    raw_outputs = [
                        self.client.call_openai_with_retry(
                            prompt=optimizer_llm_input_text,
                            model=self.optimizer_model,
                            temperature=self.optimizer_temperature,
                            max_tokens=1024
                        )
                        for _ in range(min(remaining_num_instructions_to_generate, 4))
                    ]
                    
                    # Parse generated prompts
                    new_prompts = self._parse_generated_prompts(raw_outputs)
                    generated_instructions_raw.extend(new_prompts)
                    remaining_num_instructions_to_generate -= len(raw_outputs)
                    
                except Exception as e:
                    print(f"Error generating prompts: {e}")
                    break
            
            generated_instructions_raw = [self._polish_prompt(prompt) for prompt in generated_instructions_raw]
            print(f"\ninitially generated instructions: {[p[:50] + '...' for p in generated_instructions_raw]}\n")
            
            # Filter generated instructions first (more efficient to filter before duplicate checking)
            filtered_instructions = []
            for instruction in generated_instructions_raw:
                if len(instruction.split(" ")) > 1024:
                    print(f"Step {i_step}, instruction: {instruction[:30]}..., too long, skipped")
                    continue
                if any(tag in instruction.upper() for tag in ['<PROMPT>', '</PROMPT>', 'PROMPT>']):
                    print(f"Step {i_step}, instruction: {instruction[:30]}..., contains tags, skipped")
                    continue
                filtered_instructions.append(instruction)
            
            # Step-level tracking for metrics (defined early to collect results from both new and duplicate instructions)
            step_metrics = {
                'md5_hashes': [],
                'instructions': [],
                'combined_scores': [],
                'cancel_precision': [], 'cancel_recall': [], 'cancel_f1': [], 'cancel_accuracy': [], 'cancel_balanced_accuracy': [], 'cancel_adj_balanced_accuracy': [],
                'partial_precision': [], 'partial_recall': [], 'partial_f1': [], 'partial_accuracy': [], 'partial_balanced_accuracy': [], 'partial_adj_balanced_accuracy': [],
                'word_counts': []
            }
            
            # Do not evaluate old instructions again (check duplicates after filtering)
            to_evaluate_instructions = []
            for ins in filtered_instructions:
                ins_md5_hashstring = self._instruction_to_filename(ins, md5_hashing=True)
                if ins_md5_hashstring not in self.old_instruction_md5_hashstrings_set:
                    to_evaluate_instructions.append(ins)
                    # Note: MD5 hash will be added to set only AFTER successful evaluation and file saving
                else:
                    print(f"already evaluated '{ins[:30]}...' previously")
                    # Load previously saved results and add to step metrics
                    try:
                        filename = self._instruction_to_filename(ins)
                        result_file_path = os.path.join(self.save_folder, f"{filename}.json")
                        with open(result_file_path, 'r') as f:
                            results = json.load(f)
                        
                        cancel_metrics = results.get('cancel_not_for_all_metrics', {})
                        partial_metrics = results.get('partial_or_full_metrics', {})
                        
                        # Calculate combined score
                        cancel_adj_b_acc = cancel_metrics.get('adjusted_balanced_accuracy', 0.0)
                        partial_adj_b_acc = partial_metrics.get('adjusted_balanced_accuracy', 0.0)
                        if cancel_adj_b_acc > 0 and partial_adj_b_acc > 0:
                            combined_score = 2 * (cancel_adj_b_acc * partial_adj_b_acc) / (cancel_adj_b_acc + partial_adj_b_acc)
                        else:
                            combined_score = 0.0
                        
                        # Add to step metrics
                        word_count = len(ins.split())
                        step_metrics['md5_hashes'].append(ins_md5_hashstring)
                        step_metrics['instructions'].append(ins)
                        step_metrics['combined_scores'].append(combined_score)
                        step_metrics['word_counts'].append(word_count)
                        
                        # Cancel metrics
                        step_metrics['cancel_precision'].append(cancel_metrics.get('precision', 0.0))
                        step_metrics['cancel_recall'].append(cancel_metrics.get('recall', 0.0))
                        step_metrics['cancel_f1'].append(cancel_metrics.get('f1', 0.0))
                        step_metrics['cancel_accuracy'].append(cancel_metrics.get('accuracy', 0.0))
                        step_metrics['cancel_balanced_accuracy'].append(cancel_metrics.get('balanced_accuracy', 0.0))
                        step_metrics['cancel_adj_balanced_accuracy'].append(cancel_adj_b_acc)
                        
                        # Partial metrics
                        step_metrics['partial_precision'].append(partial_metrics.get('precision', 0.0))
                        step_metrics['partial_recall'].append(partial_metrics.get('recall', 0.0))
                        step_metrics['partial_f1'].append(partial_metrics.get('f1', 0.0))
                        step_metrics['partial_accuracy'].append(partial_metrics.get('accuracy', 0.0))
                        step_metrics['partial_balanced_accuracy'].append(partial_metrics.get('balanced_accuracy', 0.0))
                        step_metrics['partial_adj_balanced_accuracy'].append(partial_adj_b_acc)
                        
                        print(f"reused results: combined_score: {combined_score:.3f}, cancel_adj_b_acc: {cancel_adj_b_acc:.3f}, partial_adj_b_acc: {partial_adj_b_acc:.3f}")
                        
                    except Exception as e:
                        print(f"Warning: Could not load results for duplicate instruction: {e}")
            
            # Remove any remaining duplicates
            to_evaluate_instructions = list(set(to_evaluate_instructions))
                
            print(f"\nto-evaluate generated instructions: {[p[:50] + '...' for p in to_evaluate_instructions]}\n")
            
            # Evaluate newly generated instructions in parallel
            if to_evaluate_instructions:
                print(f"\n🚀 Starting parallel evaluation of {len(to_evaluate_instructions)} instructions...")
                
                # Prepare arguments for parallel processing
                eval_args = []
                for instruction in to_evaluate_instructions:
                    args_tuple = (
                        instruction,
                        train_data, 
                        self.api_key,
                        self.scorer_model,
                        self.initial_prompt_response_format,
                        self.save_folder,
                        i_step
                    )
                    eval_args.append(args_tuple)
                
                # Use ProcessPoolExecutor for parallel evaluation
                max_processes = self.max_processes or mp.cpu_count()
                num_processes = min(len(to_evaluate_instructions), max_processes)
                print(f"🔧 Using {num_processes} parallel processes (max available: {mp.cpu_count()})")
                
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    # Submit all evaluation tasks
                    future_to_instruction = {
                        executor.submit(evaluate_single_instruction, args): args[0] 
                        for args in eval_args
                    }
                    
                    # Collect results as they complete
                    evaluation_results = []
                    for future in tqdm(as_completed(future_to_instruction), 
                                     total=len(future_to_instruction), 
                                     desc="Evaluating instructions"):
                        try:
                            result = future.result()
                            evaluation_results.append(result)
                            
                            if result['success']:
                                instruction = result['instruction']
                                combined_score = result['combined_score']
                                cancel_adj_b_acc = result['cancel_adj_b_acc']
                                partial_adj_b_acc = result['partial_adj_b_acc']
                                print(f"✅ {instruction[:50]}... → Combined: {combined_score:.3f}")
                            else:
                                print(f"❌ {result['instruction'][:50]}... → Error: {result['error']}")
                                
                        except Exception as e:
                            print(f"❌ Future execution failed: {e}")
                
                # Process results and save to files
                print(f"\n📁 Saving {len(evaluation_results)} evaluation results...")
                for result in evaluation_results:
                    if result['success']:
                        instruction = result['instruction']
                        combined_score = result['combined_score']
                        cancel_adj_b_acc = result['cancel_adj_b_acc']
                        partial_adj_b_acc = result['partial_adj_b_acc']
                        results = result['evaluation_results']
                        result_file_path = result['result_file_path']
                        
                        print(f"Step {i_step}, instruction: {instruction[:50]}..., combined_score: {combined_score:.3f}, cancel_adj_b_acc: {cancel_adj_b_acc:.3f}, partial_adj_b_acc: {partial_adj_b_acc:.3f}")
                        
                        # Save results to individual JSON file
                        with open(result_file_path, 'w') as f:
                            json.dump(results, f, indent=2)
                        print(f"saving results to {result_file_path}")
                        
                        # Extract all metrics for step-level tracking
                        md5_hash = self._instruction_to_filename(instruction, md5_hashing=True)
                        
                        # Only add to MD5 set after successful evaluation and file save
                        self.old_instruction_md5_hashstrings_set.add(md5_hash)
                        word_count = len(instruction.split())
                        
                        cancel_metrics = results.get('cancel_not_for_all_metrics', {})
                        partial_metrics = results.get('partial_or_full_metrics', {})
                        
                        # Store step metrics
                        step_metrics['md5_hashes'].append(md5_hash)
                        step_metrics['instructions'].append(instruction)
                        step_metrics['combined_scores'].append(combined_score)
                        step_metrics['word_counts'].append(word_count)
                        
                        # Cancel metrics
                        step_metrics['cancel_precision'].append(cancel_metrics.get('precision', 0.0))
                        step_metrics['cancel_recall'].append(cancel_metrics.get('recall', 0.0))
                        step_metrics['cancel_f1'].append(cancel_metrics.get('f1', 0.0))
                        step_metrics['cancel_accuracy'].append(cancel_metrics.get('accuracy', 0.0))
                        step_metrics['cancel_balanced_accuracy'].append(cancel_metrics.get('balanced_accuracy', 0.0))
                        step_metrics['cancel_adj_balanced_accuracy'].append(cancel_metrics.get('adjusted_balanced_accuracy', 0.0))
                        
                        # Partial metrics
                        step_metrics['partial_precision'].append(partial_metrics.get('precision', 0.0))
                        step_metrics['partial_recall'].append(partial_metrics.get('recall', 0.0))
                        step_metrics['partial_f1'].append(partial_metrics.get('f1', 0.0))
                        step_metrics['partial_accuracy'].append(partial_metrics.get('accuracy', 0.0))
                        step_metrics['partial_balanced_accuracy'].append(partial_metrics.get('balanced_accuracy', 0.0))
                        step_metrics['partial_adj_balanced_accuracy'].append(partial_metrics.get('adjusted_balanced_accuracy', 0.0))
                        
                        # Update global tracking
                        self.old_instructions_and_scores.append((instruction, combined_score, cancel_adj_b_acc, partial_adj_b_acc, i_step))
                
                print(f"✅ Parallel evaluation completed for step {i_step}")
            
            # Compute step-level statistics (mean and std for each metric)
            if step_metrics['md5_hashes']:
                step_stats = {
                    'step': i_step,
                    'num_instructions': len(step_metrics['md5_hashes']),
                    'md5_hashes': step_metrics['md5_hashes'],
                }
                
                # Compute mean and std for each metric
                for metric_name in ['combined_scores', 'word_counts',
                                   'cancel_precision', 'cancel_recall', 'cancel_f1', 'cancel_accuracy', 'cancel_balanced_accuracy', 'cancel_adj_balanced_accuracy',
                                   'partial_precision', 'partial_recall', 'partial_f1', 'partial_accuracy', 'partial_balanced_accuracy', 'partial_adj_balanced_accuracy']:
                    values = step_metrics[metric_name]
                    if values:
                        step_stats[f'{metric_name}_mean'] = np.mean(values)
                        step_stats[f'{metric_name}_std'] = np.std(values) if len(values) > 1 else 0.0
                        step_stats[f'{metric_name}_values'] = values
                    else:
                        step_stats[f'{metric_name}_mean'] = 0.0
                        step_stats[f'{metric_name}_std'] = 0.0
                        step_stats[f'{metric_name}_values'] = []
                
                # Store step statistics
                self.step_statistics.append(step_stats)
                
                # Print step summary
                print(f"\n📊 Step {i_step} Summary:")
                print(f"  Instructions evaluated: {len(step_metrics['md5_hashes'])}")
                print(f"  Combined score: {step_stats['combined_scores_mean']:.3f} ± {step_stats['combined_scores_std']:.3f}")
                print(f"  Cancel adj b-acc: {step_stats['cancel_adj_balanced_accuracy_mean']:.3f} ± {step_stats['cancel_adj_balanced_accuracy_std']:.3f}")
                print(f"  Partial adj b-acc: {step_stats['partial_adj_balanced_accuracy_mean']:.3f} ± {step_stats['partial_adj_balanced_accuracy_std']:.3f}")
                print(f"  Word count: {step_stats['word_counts_mean']:.1f} ± {step_stats['word_counts_std']:.1f}")
            
            # Save intermediate results
            self._save_results()
        
        # Final save
        final_results = self._save_results()
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"💾 Results saved to: {self.save_folder}")
            
            # Show final best prompts with sorting by score then word count
            def sort_key(x):
                prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step = x
                word_count = len(prompt.split())
                return (-combined_score, word_count)  # Higher score first, then fewer words
            
            best_prompts = sorted(self.old_instructions_and_scores, key=sort_key)[:5]
            print(f"\n🏆 Top 5 final prompts (sorted by score, then word count):")
            for i, (prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step) in enumerate(best_prompts):
                word_count = len(prompt.split())
                print(f"  {i+1}. Combined: {combined_score:.3f}, Cancel Adj B-Acc: {cancel_adj_b_acc:.3f}, Partial Adj B-Acc: {partial_adj_b_acc:.3f}, Words: {word_count}, Step: {step}")
                print(f"     {prompt[:150]}...")
                print()
        
        return final_results
    
    def _save_results(self) -> Dict[str, Any]:
        """Save optimization results to files."""
        results_dict = {
            'meta_prompts': self.meta_prompts,
            'old_instructions_and_scores': self.old_instructions_and_scores,
            'step_statistics': self.step_statistics,  # Include step-level statistics
            'config': {
                'train_data_path': self.train_data_path,
                'initial_prompt_file': self.initial_prompt_file,
                'initial_prompt_key': self.initial_prompt_key,
                'save_folder': self.save_folder
            }
        }
        
        # Save as pickle
        with open(os.path.join(self.save_folder, "optimization_results.pkl"), 'wb') as f:
            pickle.dump(results_dict, f)
        
        # Save as JSON (without detailed results to avoid size issues)
        # Sort by combined_score (higher better) then word_count (lower better)
        def sort_key_for_json(x):
            prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step = x
            word_count = len(prompt.split())
            return (-combined_score, word_count)  # Negative score for descending order
        
        sorted_prompts = sorted(self.old_instructions_and_scores, key=sort_key_for_json)
        
        json_results = {
            'prompts_and_scores': [
                {
                    'prompt': prompt, 
                    'combined_score': combined_score, 
                    'cancel_adj_b_acc': cancel_adj_b_acc, 
                    'partial_adj_b_acc': partial_adj_b_acc, 
                    'word_count': len(prompt.split()),
                    'step': step
                }
                for prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step in sorted_prompts
            ],
            'step_statistics': self.step_statistics,  # Include step-level statistics
            'config': results_dict['config']
        }
        
        with open(os.path.join(self.save_folder, "optimization_results.json"), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return results_dict
    
    def get_best_prompts(self, top_k: int = 5) -> List[Tuple[str, float, float, float, int]]:
        """Get the top-k best prompts sorted by score, then word count."""
        def sort_key(x):
            prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step = x
            word_count = len(prompt.split())
            return (-combined_score, word_count)  # Higher score first, then fewer words
        
        return sorted(self.old_instructions_and_scores, key=sort_key)[:top_k]
    
    def save_best_prompt(self, output_file: str = None) -> str:
        """Save the best prompt to a file."""
        if not self.old_instructions_and_scores:
            raise ValueError("No prompts have been evaluated yet")
        
        # Get best prompt using the same sorting logic
        def sort_key(x):
            prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step = x
            word_count = len(prompt.split())
            return (-combined_score, word_count)  # Higher score first, then fewer words
        
        best_prompt, best_combined_score, best_cancel_adj_b_acc, best_partial_adj_b_acc, best_step = sorted(self.old_instructions_and_scores, key=sort_key)[0]
        best_word_count = len(best_prompt.split())
        
        if output_file is None:
            output_file = os.path.join(self.save_folder, "best_prompt.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"# Best Prompt (Combined: {best_combined_score:.3f}, Cancel Adj B-Acc: {best_cancel_adj_b_acc:.3f}, Partial Adj B-Acc: {best_partial_adj_b_acc:.3f}, Words: {best_word_count}, Step: {best_step})\n\n")
            f.write(best_prompt)
        
        return best_prompt


def main():
    """Example usage of the Ava OPRO optimizer."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run OPRO optimization for Ava\'s prompt')
    parser.add_argument('--train_data_path', type=str, 
                      default="data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth_balance_train.json",
                      help='Path to training data JSON file')
    parser.add_argument('--initial_prompt_file', type=str, 
                      default="prompts/original/identify_partial.yaml",
                      help='Path to YAML file with initial prompt')
    parser.add_argument('--initial_prompt_key', type=str, 
                      default="initial_prompt",
                      help='Key in YAML file for prompt to optimize')
    parser.add_argument('--num_search_steps', type=int, default=10,
                      help='Number of optimization steps (default: 10)')
    parser.add_argument('--num_generated_instructions_in_each_step', type=int, default=4,
                      help='Number of candidate prompts to generate per step (default: 4)')
    parser.add_argument('--optimizer_model', type=str, default="gpt-4.1",
                      help='Model for generating new prompts (default: gpt-4.1)')
    parser.add_argument('--scorer_model', type=str, default="gpt-4o-mini",
                      help='Model for evaluating prompts (default: gpt-4o-mini)')
    parser.add_argument('--optimizer_temperature', type=float, default=1.0,
                      help='Temperature for prompt generation (default: 1.0)')
    parser.add_argument('--train_ratio', type=float, default=0.5,
                      help='Fraction of training data to use (default: 0.25)')
    parser.add_argument('--num_examples', type=int, default=2,
                      help='Number of examples to include in meta-prompt (default: 2)')
    parser.add_argument('--max_processes', type=int, default=None,
                      help='Maximum number of parallel processes (default: CPU count)')
    parser.add_argument('--save_folder', type=str, default="results/gpt-5-verified/",
                      help='Folder to save optimization results (default: results/gpt-5-verified/)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                      help='Enable verbose output (default: True)')
    
    args = parser.parse_args()
    
    optimizer = AvaOproOptimizer(
        train_data_path=args.train_data_path,
        initial_prompt_file=args.initial_prompt_file,
        initial_prompt_key=args.initial_prompt_key,
        save_folder=args.save_folder,
        num_search_steps=args.num_search_steps,
        num_generated_instructions_in_each_step=args.num_generated_instructions_in_each_step,
        optimizer_model=args.optimizer_model,
        scorer_model=args.scorer_model,
        optimizer_temperature=args.optimizer_temperature,
        train_ratio=args.train_ratio,
        num_examples=args.num_examples,
        max_processes=args.max_processes,
        verbose=args.verbose,
        random_seed=args.random_seed
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Get best prompts
    best_prompts = optimizer.get_best_prompts(top_k=3)
    print("\n🏆 Best prompts:")
    for i, (prompt, combined_score, cancel_adj_b_acc, partial_adj_b_acc, step) in enumerate(best_prompts):
        print(f"{i+1}. Combined: {combined_score:.3f}, Cancel Adj B-Acc: {cancel_adj_b_acc:.3f}, Partial Adj B-Acc: {partial_adj_b_acc:.3f} (Step {step})")
        print(f"   {prompt[:200]}...")
    
    # Save best prompt
    best_prompt = optimizer.save_best_prompt()
    print(f"\n💾 Best prompt saved")


if __name__ == "__main__":
    main()
