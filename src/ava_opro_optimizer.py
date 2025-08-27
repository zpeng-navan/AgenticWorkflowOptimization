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
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import yaml
import re
from collections import Counter

from src.utils.open_ai_util import OpenAIClient
from src.utils.data_util import load_json_file, save_to_json
from src.utils.eval_prompt_util import substitute_prompt_variables, parse_model_response, compute_binary_metrics
            

class AvaOproOptimizer:
    """OPRO optimizer for Ava's prompt optimization."""
    
    def __init__(
        self,
        train_data_path: str = "data/processed/logs/04222025-08182025/ground_truth/verified_ground_truth_balance_train.json",
        initial_prompt_file: str = "prompts/original/identify_partial.yaml",
        initial_prompt_key: str = "initial_prompt",
        api_key: Optional[str] = None,
        save_folder: str = "results/ava_opro_optimization",
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
            verbose: Whether to print detailed progress
            random_seed: Random seed for reproducibility
        """
        self.train_data_path = train_data_path
        self.initial_prompt_file = initial_prompt_file
        self.initial_prompt_key = initial_prompt_key
        self.initial_prompt_body_key = initial_prompt_key + "_body"
        self.initial_prompt_response_format_key = initial_prompt_key + "_response_format"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.save_folder = save_folder
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
        self.num_score_buckets = 20
        
        # Tracking variables (similar to OPRO)
        self.old_instructions_and_scores = []  # (prompt, combined_score, cancel_f1, partial_f1, step)
        self.old_instructions_and_scores_raw = []  # All generated including skipped
        self.meta_prompts = []  # (meta_prompt, step)
        self.instruction_score_dict = {}  # {prompt: (combined_score, cancel_f1, partial_f1)}
        self.old_instruction_md5_hashstrings_set = set()
        self.detailed_results_by_instruction = {}
        
    def _load_initial_prompt_parts(self) -> Tuple[str, str]:
        """Load initial prompt body and response format from YAML file."""
        with open(self.initial_prompt_file, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
        
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
            prompt, combined_score, cancel_f1, partial_f1, step = x
            word_count = len(prompt.split())
            return (combined_score, -word_count)  # Higher score first, then fewer words
        
        sorted_prompts = sorted(old_prompts_and_scores, key=sort_key)[-max_num_prompts:]
        
        prompts_in_meta_prompt = []
        prompt_score_str = ""
        
        for prompt, combined_score, cancel_f1, partial_f1, step in sorted_prompts:
            if combined_score >= score_threshold:
                prompts_in_meta_prompt.append((prompt, combined_score, cancel_f1, partial_f1, step))
                combined_score_display = self._bucketize_score(combined_score, self.num_score_buckets)
                cancel_f1_display = self._bucketize_score(cancel_f1, self.num_score_buckets)
                partial_f1_display = self._bucketize_score(partial_f1, self.num_score_buckets)
                word_count = len(prompt.split())
                prompt_score_str += f"\ntext:\n{prompt}\ncombined_score:\n{combined_score_display}\ncancel_f1:\n{cancel_f1_display}\npartial_f1:\n{partial_f1_display}\nwords:\n{word_count}\n"
        
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
        optimizer_llm_name: str = "gpt-4o"
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
- combined_score: Average of cancel_f1 and partial_f1 (0 to {self.num_score_buckets}, higher is better)
- cancel_f1: F1 score for cancel_not_for_all_passengers classification (0 to {self.num_score_buckets}, higher is better)  
- partial_f1: F1 score for partial_or_full classification (0 to {self.num_score_buckets}, higher is better)
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
    ) -> Tuple[float, Dict]:
        """Evaluate a single prompt body and return combined score."""
        
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
                cancel_metrics = {'f1': 0.0}
            
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
                partial_metrics = {'f1': 0.0}
            
            # Combine F1 scores as the optimization objective
            cancel_f1 = cancel_metrics.get('f1', 0.0)
            partial_f1 = partial_metrics.get('f1', 0.0)
            
            # Combined score (equal weight to both tasks)
            combined_score = (cancel_f1 + partial_f1) / 2.0
            
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
            
            return combined_score, cancel_f1, partial_f1, evaluation_results
            
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
        combined_score, cancel_f1, partial_f1, _ = self._evaluate_prompt_performance(prompt, train_data_subset, model)
        return combined_score, cancel_f1, partial_f1
    
    def optimize(
        self,
        num_search_steps: int = 10,
        num_generated_instructions_in_each_step: int = 4,
        optimizer_model: str = "gpt-4o",
        scorer_model: str = "gpt-4o-mini",
        optimizer_temperature: float = 1.0,
        train_ratio: float = 1.0,
        num_examples: int = 3,
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
            print(f"📊 Steps: {num_search_steps}, Instructions per step: {num_generated_instructions_in_each_step}")
            print(f"🤖 Optimizer: {optimizer_model}, Scorer: {scorer_model}")
            print(f"📈 Train ratio: {train_ratio}, Examples in meta-prompt: {num_examples}")
        
        # Load and sample training data
        full_train_data = load_json_file(self.train_data_path)
        if train_ratio < 1.0:
            sample_size = int(len(full_train_data) * train_ratio)
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
        
        initial_combined_score, initial_cancel_f1, initial_partial_f1, initial_results = self._evaluate_prompt_performance(
            self.initial_prompt, train_data, scorer_model
        )
        
        print(f"instruction: {self.initial_prompt[:50]}..., combined_score: {initial_combined_score:.3f}, cancel_f1: {initial_cancel_f1:.3f}, partial_f1: {initial_partial_f1:.3f}")
        
        # Save initial results
        initial_filename = self._instruction_to_filename(self.initial_prompt)
        result_file_path = os.path.join(self.save_folder, f"{initial_filename}.json")
        with open(result_file_path, 'w') as f:
            json.dump(initial_results, f, indent=2)
        print(f'Saving results of "{self.initial_prompt[:50]}..." to {result_file_path}')
        
        # Add initial prompt to tracking
        initial_word_count = len(self.initial_prompt.split(" "))
        self.old_instructions_and_scores.append((self.initial_prompt, initial_combined_score, initial_cancel_f1, initial_partial_f1, initial_word_count))
        self.old_instructions_and_scores_raw.append((self.initial_prompt, initial_combined_score, initial_cancel_f1, initial_partial_f1, initial_word_count))
        self.instruction_score_dict[self.initial_prompt] = (initial_combined_score, initial_cancel_f1, initial_partial_f1)
        self.detailed_results_by_instruction[self.initial_prompt] = initial_results
        
        # Add initial prompt MD5 to hash set for duplicate detection (following OPRO pattern)
        initial_prompt_md5 = self._instruction_to_filename(self.initial_prompt, md5_hashing=True)
        self.old_instruction_md5_hashstrings_set.add(initial_prompt_md5)
        
        prev_saved_instructions = {self.initial_prompt}
        
        # ============== Evolution ===============
        for i_step in range(num_search_steps):
            print(f"\n================== Step {i_step} =====================")
            if not i_step % 10:
                print(f"old_instructions_and_scores: {[(p[:30], f'C:{combined:.2f}', f'Step:{st}') for p, combined, cancel, partial, st in self.old_instructions_and_scores]}")
            
            print(f"current optimizer_temperature: {optimizer_temperature}")
            
            # Generate meta-prompt
            meta_prompt = self._generate_meta_prompt(
                self.old_instructions_and_scores,
                train_data,
                num_examples,
                optimizer_model
            )
            
            print(f"\nmeta_prompt: \n\n{meta_prompt}\n")
            self.meta_prompts.append((meta_prompt, i_step))
            
            # Generate new instructions
            remaining_num_instructions_to_generate = num_generated_instructions_in_each_step
            generated_instructions_raw = []
            
            while remaining_num_instructions_to_generate > 0:
                optimizer_llm_input_text = meta_prompt
                # Generate instructions
                print(f"current temperature: {optimizer_temperature}")
                
                try:
                    raw_outputs = [
                        self.client.call_openai_with_retry(
                            prompt=optimizer_llm_input_text,
                            model=optimizer_model,
                            temperature=optimizer_temperature,
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
            
            # Do not evaluate old instructions again
            generated_instructions = []
            for ins in generated_instructions_raw:
                ins_md5_hashstring = self._instruction_to_filename(ins, md5_hashing=True)
                if ins_md5_hashstring not in self.old_instruction_md5_hashstrings_set:
                    generated_instructions.append(ins)
                    self.old_instruction_md5_hashstrings_set.add(ins_md5_hashstring)
                else:
                    print(f"already evaluated '{ins[:30]}...' previously")
            
            generated_instructions = list(set(generated_instructions))
            
            # Filter generated instructions
            to_evaluate_instructions = []
            for instruction in generated_instructions:
                if len(instruction.split(" ")) > 1024:
                    print(f"Step {i_step}, instruction: {instruction[:30]}..., too long, skipped")
                    continue
                if any(tag in instruction.upper() for tag in ['<PROMPT>', '</PROMPT>', 'PROMPT>']):
                    print(f"Step {i_step}, instruction: {instruction[:30]}..., contains tags, skipped")
                    continue
                to_evaluate_instructions.append(instruction)
                
            print(f"\nto-evaluate generated instructions: {[p[:50] + '...' for p in to_evaluate_instructions]}\n")
            
            # Evaluate newly generated instructions on the training set
            for instruction in to_evaluate_instructions:
                if instruction not in prev_saved_instructions:
                    print(f'Computing the score of "{instruction[:50]}..." by prompting')
                    
                    combined_score, cancel_f1, partial_f1, results = self._evaluate_prompt_performance(
                        instruction, train_data, scorer_model
                    )
                    
                    prev_saved_instructions.add(instruction)
                else:
                    # Load previously saved results
                    filename = self._instruction_to_filename(instruction)
                    result_file_path = os.path.join(self.save_folder, f"{filename}.json")
                    with open(result_file_path, 'r') as f:
                        results = json.load(f)
                    cancel_metrics = results.get('cancel_not_for_all_metrics', {})
                    partial_metrics = results.get('partial_or_full_metrics', {})
                    cancel_f1 = cancel_metrics.get('f1', 0.0)
                    partial_f1 = partial_metrics.get('f1', 0.0)
                    combined_score = (cancel_f1 + partial_f1) / 2.0
                    print(f'reading previously saved "{instruction[:50]}..." information')
                
                print(f"Step {i_step}, instruction: {instruction[:50]}..., combined_score: {combined_score:.3f}, cancel_f1: {cancel_f1:.3f}, partial_f1: {partial_f1:.3f}")
                
                # Save results
                filename = self._instruction_to_filename(instruction)
                result_file_path = os.path.join(self.save_folder, f"{filename}.json")
                with open(result_file_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"saving results to {result_file_path}")
                
                # Update tracking
                self.detailed_results_by_instruction[instruction] = results
                self.old_instructions_and_scores.append((instruction, combined_score, cancel_f1, partial_f1, i_step))
                self.instruction_score_dict[instruction] = (combined_score, cancel_f1, partial_f1)
            
            # Record all generated instructions
            for instruction in generated_instructions_raw:
                if instruction in self.instruction_score_dict:
                    combined_score, cancel_f1, partial_f1 = self.instruction_score_dict[instruction]
                else:
                    combined_score, cancel_f1, partial_f1 = np.nan, np.nan, np.nan
                self.old_instructions_and_scores_raw.append((instruction, combined_score, cancel_f1, partial_f1, i_step))
            
            # Save intermediate results
            self._save_results()
        
        # Final save
        final_results = self._save_results()
        
        if self.verbose:
            print(f"\n✅ Optimization complete!")
            print(f"💾 Results saved to: {self.save_folder}")
            
            # Show final best prompts with sorting by score then word count
            def sort_key(x):
                prompt, combined_score, cancel_f1, partial_f1, step = x
                word_count = len(prompt.split())
                return (combined_score, -word_count)
            
            best_prompts = sorted(self.old_instructions_and_scores, key=sort_key)[-5:]
            print(f"\n🏆 Top 5 final prompts (sorted by score, then word count):")
            for i, (prompt, combined_score, cancel_f1, partial_f1, step) in enumerate(best_prompts):
                word_count = len(prompt.split())
                print(f"  {i+1}. Combined: {combined_score:.3f}, Cancel F1: {cancel_f1:.3f}, Partial F1: {partial_f1:.3f}, Words: {word_count}, Step: {step}")
                print(f"     {prompt[:150]}...")
                print()
        
        return final_results
    
    def _save_results(self) -> Dict[str, Any]:
        """Save optimization results to files."""
        results_dict = {
            'meta_prompts': self.meta_prompts,
            'old_instructions_and_scores': self.old_instructions_and_scores,
            'old_instructions_and_scores_raw': self.old_instructions_and_scores_raw,
            'instruction_score_dict': self.instruction_score_dict,
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
            prompt, combined_score, cancel_f1, partial_f1, step = x
            word_count = len(prompt.split())
            return (-combined_score, word_count)  # Negative score for descending order
        
        sorted_prompts = sorted(self.old_instructions_and_scores, key=sort_key_for_json)
        
        json_results = {
            'prompts_and_scores': [
                {
                    'prompt': prompt, 
                    'combined_score': combined_score, 
                    'cancel_f1': cancel_f1, 
                    'partial_f1': partial_f1, 
                    'word_count': len(prompt.split()),
                    'step': step
                }
                for prompt, combined_score, cancel_f1, partial_f1, step in sorted_prompts
            ],
            'config': results_dict['config']
        }
        
        with open(os.path.join(self.save_folder, "optimization_results.json"), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return results_dict
    
    def get_best_prompts(self, top_k: int = 5) -> List[Tuple[str, float, float, float, int]]:
        """Get the top-k best prompts sorted by score, then word count."""
        def sort_key(x):
            prompt, combined_score, cancel_f1, partial_f1, step = x
            word_count = len(prompt.split())
            return (combined_score, -word_count)  # Higher score first, then fewer words
        
        return sorted(self.old_instructions_and_scores, key=sort_key)[-top_k:]
    
    def save_best_prompt(self, output_file: str = None) -> str:
        """Save the best prompt to a file."""
        if not self.old_instructions_and_scores:
            raise ValueError("No prompts have been evaluated yet")
        
        # Get best prompt using the same sorting logic
        def sort_key(x):
            prompt, combined_score, cancel_f1, partial_f1, step = x
            word_count = len(prompt.split())
            return (combined_score, -word_count)  # Higher score first, then fewer words
        
        best_prompt, best_combined_score, best_cancel_f1, best_partial_f1, best_step = max(self.old_instructions_and_scores, key=sort_key)
        best_word_count = len(best_prompt.split())
        
        if output_file is None:
            output_file = os.path.join(self.save_folder, "best_prompt.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"# Best Prompt (Combined: {best_combined_score:.3f}, Cancel F1: {best_cancel_f1:.3f}, Partial F1: {best_partial_f1:.3f}, Words: {best_word_count}, Step: {best_step})\n\n")
            f.write(best_prompt)
        
        return best_prompt


def main():
    """Example usage of the Ava OPRO optimizer."""
    
    optimizer = AvaOproOptimizer(
        save_folder="results/ava_opro_optimization",
        verbose=True,
        random_seed=42
    )
    
    # Run optimization
    results = optimizer.optimize(
        num_search_steps=2,  # Small number for testing
        num_generated_instructions_in_each_step=4,
        train_ratio=0.04,  # Use small subset for testing
        num_examples=2,  # Small number of examples for testing
    )
    
    # Get best prompts
    best_prompts = optimizer.get_best_prompts(top_k=3)
    print("\n🏆 Best prompts:")
    for i, (prompt, combined_score, cancel_f1, partial_f1, step) in enumerate(best_prompts):
        print(f"{i+1}. Combined: {combined_score:.3f}, Cancel F1: {cancel_f1:.3f}, Partial F1: {partial_f1:.3f} (Step {step})")
        print(f"   {prompt[:200]}...")
    
    # Save best prompt
    best_prompt = optimizer.save_best_prompt()
    print(f"\n💾 Best prompt saved")


if __name__ == "__main__":
    main()
