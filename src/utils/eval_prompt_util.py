"""
Evaluate prompt performance using OpenAI API against test dataset.

This module loads prompts from YAML files, runs them against test data,
and computes classification metrics (precision, recall, F1) for:
- cancel_not_for_all_passengers (True=positive, False=negative)  
- partial_or_full (partial=positive, full=negative, ignore null)
"""

import json
import yaml
import os
import argparse
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, balanced_accuracy_score
import pandas as pd
from tqdm import tqdm

from src.utils.open_ai_util import OpenAIClient
from src.utils.data_util import load_json_file, extract_json_from_string


def load_prompt_from_yaml(prompt_file_path: str, prompt_name: str) -> str:
    """
    Load prompt template from YAML file.
    
    Args:
        prompt_file_path: Path to YAML file containing prompts
        prompt_name: Name of the prompt key in the YAML file
        
    Returns:
        Prompt template string
    """
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_data = yaml.safe_load(f)
    
    if prompt_name not in prompt_data:
        raise ValueError(f"Prompt '{prompt_name}' not found in {prompt_file_path}")
    
    return prompt_data[prompt_name]


def substitute_prompt_variables(prompt_template: str, flight_booking_legs: str, chat_history: str) -> str:
    """
    Substitute variables in prompt template.
    
    Args:
        prompt_template: Template with ${variable} placeholders
        flight_booking_legs: Flight booking legs data
        chat_history: Chat history data
        
    Returns:
        Prompt with variables substituted
    """
    # Handle both ${var} and {var} formats
    prompt = prompt_template.replace("${flight_booking_legs}", flight_booking_legs)
    prompt = prompt.replace("${chat_history}", chat_history)
    prompt = prompt.replace("{flight_booking_legs}", flight_booking_legs)
    prompt = prompt.replace("{chat_history}", chat_history)
    
    return prompt


def parse_model_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse model JSON response and extract labels.
    
    Args:
        response: Raw model response text
        
    Returns:
        Parsed response dict or None if parsing failed
    """
    if not response:
        return None
    
    # Try to extract JSON
    json_data = extract_json_from_string(response)
    
    if json_data and isinstance(json_data, dict):
        return json_data
    
    # Fallback: try direct JSON parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None


def compute_binary_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
    """
    Compute precision, recall, F1, accuracy for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=True
    )
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    adjusted_balanced_accuracy = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'adjusted_balanced_accuracy': adjusted_balanced_accuracy
    }


def evaluate_prompt(
    prompt_file_path: str = "prompts/original/identify_partial.yaml",
    prompt_name: str = "initial_prompt",
    test_data_path: str = "data/processed/logs/04222025-08182025/ground_truth/verified_ground_truth_balance_test.json",
    model: str = "gpt-4o",
    temperature: float = 0,
    api_key: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate prompt performance against test dataset.
    
    Args:
        prompt_file_path: Path to YAML file with prompts
        prompt_name: Name of prompt to use from YAML file
        test_data_path: Path to test data JSON file
        model: OpenAI model to use
        temperature: Temperature for OpenAI API (0=deterministic, 1=random)
        api_key: OpenAI API key (if None, will use environment variable)
        verbose: Whether to print progress and detailed results
        
    Returns:
        Dictionary with evaluation results and metrics
    """
    # Load API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
    
    # Initialize OpenAI client
    client = OpenAIClient(api_key=api_key)
    
    # Load prompt template
    if verbose:
        print(f"Loading prompt '{prompt_name}' from {prompt_file_path}")
    prompt_template = load_prompt_from_yaml(prompt_file_path, prompt_name)
    
    # Load test data
    if verbose:
        print(f"Loading test data from {test_data_path}")
    test_data = load_json_file(test_data_path)
    
    # Prepare evaluation data
    results = []
    prediction_errors = []
    thought_word_counts = []  # Track thought word counts
    latencies = []  # Track API call latencies in milliseconds
    input_word_counts = []  # Track input word counts
    output_word_counts = []  # Track output word counts
    
    if verbose:
        print(f"Evaluating {len(test_data)} test cases...")
    
    # Process each test case
    debug_count = 0
    for element_id, element in tqdm(test_data.items(), disable=not verbose):
        # Extract required data
        ava_data = element.get("Ava", {})
        flight_booking_legs = ava_data.get("flight_booking_legs", "")
        chat_history = ava_data.get("chat_history", "")
        
        # Ground truth labels
        gt_cancel_not_for_all = element.get("cancel_not_for_all")
        gt_partial_or_full = element.get("partial_or_full")
        
        # Skip if missing required data
        if gt_cancel_not_for_all is None:
            raise ValueError(f"Missing cancel_not_for_all label for {element_id}")
        
        # Substitute variables in prompt
        prompt = substitute_prompt_variables(prompt_template, flight_booking_legs, chat_history)
        
        # Count input words
        input_words = len(prompt.split())
        input_word_counts.append(input_words)
        
        # Call OpenAI API
        try:
            start_time = time.time()
            response = client.call_openai_with_retry(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=2048
            )
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            # Count output words
            output_words = len(response.split()) if response else 0
            output_word_counts.append(output_words)
            
            # Parse response
            parsed_response = parse_model_response(response)
            
            if parsed_response:
                pred_cancel_not_for_all = parsed_response.get("cancel_not_for_all_passengers")
                pred_partial_or_full = parsed_response.get("partial_or_full")
                
                # Count thought words if present
                thought_words = 0
                if "thought" in parsed_response and parsed_response["thought"]:
                    thought_text = str(parsed_response["thought"])
                    thought_words = len(thought_text.split())
                    thought_word_counts.append(thought_words)
                
                # Store result
                results.append({
                    'element_id': element_id,
                    'gt_cancel_not_for_all': gt_cancel_not_for_all,
                    'pred_cancel_not_for_all': pred_cancel_not_for_all,
                    'gt_partial_or_full': gt_partial_or_full,
                    'pred_partial_or_full': pred_partial_or_full,
                    'thought_words': thought_words,
                    'input_words': input_words,
                    'output_words': output_words,
                    'latency': latency,
                    'raw_response': response,
                    'parsed_response': parsed_response
                })
            else:
                prediction_errors.append({
                    'element_id': element_id,
                    'error': 'Failed to parse JSON response',
                    'input_words': input_words,
                    'output_words': output_words,
                    'latency': latency,
                    'raw_response': response
                })
                
        except Exception as e:
            prediction_errors.append({
                'element_id': element_id,
                'error': str(e),
                'input_words': input_words,
                'output_words': None,  # No output words for failed API calls
                'latency': None,  # No latency measurement for failed calls (would be in ms)
                'raw_response': None
            })
        # debug_count += 1
        # if debug_count > 10:
        #     break
    
    if verbose:
        print(f"Successfully processed: {len(results)}/{len(test_data)} cases")
        if prediction_errors:
            print(f"Prediction errors: {len(prediction_errors)}")
    
    # Compute metrics for cancel_not_for_all
    cancel_not_for_all_results = [r for r in results if r['pred_cancel_not_for_all'] is not None]
    
    if cancel_not_for_all_results:
        cancel_gt = [r['gt_cancel_not_for_all'] for r in cancel_not_for_all_results]
        cancel_pred = [r['pred_cancel_not_for_all'] for r in cancel_not_for_all_results]
        cancel_metrics = compute_binary_metrics(cancel_gt, cancel_pred)
    else:
        cancel_metrics = {}
    
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
        partial_metrics = {}
    
    # Calculate average thought words, latency, input words, and output words
    avg_thought_words = np.mean(thought_word_counts) if thought_word_counts else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    avg_input_words = np.mean(input_word_counts) if input_word_counts else 0.0
    avg_output_words = np.mean(output_word_counts) if output_word_counts else 0.0
    
    # Compile final results
    evaluation_results = {
        'total_cases': len(test_data),
        'successful_predictions': len(results),
        'prediction_errors': len(prediction_errors),
        'cancel_not_for_all_metrics': cancel_metrics,
        'partial_or_full_metrics': partial_metrics,
        'avg_thought_words': float(avg_thought_words),
        'thought_word_counts': thought_word_counts,
        'avg_input_words': float(avg_input_words),
        'input_word_counts': input_word_counts,
        'avg_output_words': float(avg_output_words),
        'output_word_counts': output_word_counts,
        'avg_latency': float(avg_latency),
        'latencies': latencies,
        'detailed_results': results,
        'errors': prediction_errors,
        'config': {
            'prompt_file_path': prompt_file_path,
            'prompt_name': prompt_name,
            'test_data_path': test_data_path,
            'model': model,
            'temperature': temperature
        }
    }
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total test cases: {len(test_data)}")
        print(f"Successful predictions: {len(results)}")
        print(f"Prediction errors: {len(prediction_errors)}")
        
        print(f"\n📊 CANCEL_NOT_FOR_ALL_PASSENGERS METRICS:")
        print(f"Cases evaluated: {len(cancel_not_for_all_results)}")
        if cancel_metrics:
            print(f"Accuracy: {cancel_metrics['accuracy']:.3f}")
            print(f"Precision: {cancel_metrics['precision']:.3f}")
            print(f"Recall: {cancel_metrics['recall']:.3f}")
            print(f"F1-Score: {cancel_metrics['f1']:.3f}")
        
        print(f"\n📊 PARTIAL_OR_FULL METRICS:")
        print(f"Cases evaluated: {len(partial_results)} (excluding null)")
        if partial_metrics:
            print(f"Accuracy: {partial_metrics['accuracy']:.3f}")
            print(f"Precision: {partial_metrics['precision']:.3f}")
            print(f"Recall: {partial_metrics['recall']:.3f}")
            print(f"F1-Score: {partial_metrics['f1']:.3f}")
        
        print(f"\n💭 THOUGHT WORDS METRICS:")
        print(f"Responses with thought: {len(thought_word_counts)}")
        print(f"Average thought words: {avg_thought_words:.1f}")
        
        print(f"\n📝 INPUT/OUTPUT WORDS METRICS:")
        print(f"Total API calls: {len(input_word_counts)}")
        print(f"Average input words: {avg_input_words:.1f}")
        print(f"Average output words: {avg_output_words:.1f}")
        if input_word_counts:
            print(f"Min input words: {min(input_word_counts)}")
            print(f"Max input words: {max(input_word_counts)}")
        if output_word_counts:
            print(f"Min output words: {min(output_word_counts)}")
            print(f"Max output words: {max(output_word_counts)}")
        
        print(f"\n⏱️ LATENCY METRICS:")
        print(f"Successful API calls: {len(latencies)}")
        print(f"Average latency: {avg_latency:.1f}ms")
        if latencies:
            print(f"Min latency: {min(latencies):.1f}ms")
            print(f"Max latency: {max(latencies):.1f}ms")
    
    return evaluation_results


def compute_metrics_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation for metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries from multiple runs
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric keys
    metric_keys = metrics_list[0].keys()
    
    statistics = {}
    for key in metric_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if values:
            statistics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
    
    return statistics


def evaluate_prompt_multiple_runs(
    prompt_file_path: str = "prompts/original/identify_partial.yaml",
    prompt_name: str = "initial_prompt", 
    test_data_path: str = "data/processed/logs/04222025-08182025/ground_truth/verified_ground_truth_balance_test.json",
    model: str = "gpt-4o",
    temperature: float = 0,
    api_key: Optional[str] = None,
    run_num: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate prompt performance multiple times and compute statistics.
    
    Args:
        prompt_file_path: Path to YAML file with prompts
        prompt_name: Name of prompt to use from YAML file
        test_data_path: Path to test data JSON file
        model: OpenAI model to use
        temperature: Temperature for OpenAI API (0=deterministic, 1=random)
        api_key: OpenAI API key (if None, will use environment variable)
        run_num: Number of times to run the evaluation
        verbose: Whether to print progress and detailed results
        
    Returns:
        Dictionary with evaluation results, individual runs, and statistics
    """
    if run_num == 1:
        # Single run - return original results
        return evaluate_prompt(
            prompt_file_path=prompt_file_path,
            prompt_name=prompt_name,
            test_data_path=test_data_path,
            model=model,
            temperature=temperature,
            api_key=api_key,
            verbose=verbose
        )
    
    # Multiple runs
    if verbose:
        print(f"🔄 Running evaluation {run_num} times...")
    
    all_runs = []
    cancel_metrics_list = []
    partial_metrics_list = []
    avg_thought_words_list = []
    avg_input_words_list = []
    avg_output_words_list = []
    avg_latency_list = []
    
    for run_idx in tqdm(range(run_num)):
        if verbose:
            print(f"\n--- Run {run_idx + 1}/{run_num} ---")
        
        # Run single evaluation
        run_results = evaluate_prompt(
            prompt_file_path=prompt_file_path,
            prompt_name=prompt_name,
            test_data_path=test_data_path,
            model=model,
            temperature=temperature,
            api_key=api_key,
            verbose=verbose
        )
        
        # Store results
        all_runs.append(run_results)
        
        # Collect metrics for statistics
        if run_results.get('cancel_not_for_all_metrics'):
            cancel_metrics_list.append(run_results['cancel_not_for_all_metrics'])
        
        if run_results.get('partial_or_full_metrics'):
            partial_metrics_list.append(run_results['partial_or_full_metrics'])
        
        if 'avg_thought_words' in run_results:
            avg_thought_words_list.append(run_results['avg_thought_words'])
        
        if 'avg_input_words' in run_results:
            avg_input_words_list.append(run_results['avg_input_words'])
        
        if 'avg_output_words' in run_results:
            avg_output_words_list.append(run_results['avg_output_words'])
        
        if 'avg_latency' in run_results:
            avg_latency_list.append(run_results['avg_latency'])
    
    # Compute statistics
    cancel_statistics = compute_metrics_statistics(cancel_metrics_list)
    partial_statistics = compute_metrics_statistics(partial_metrics_list)
    
    # Compute combined adjusted balanced accuracy using harmonic mean
    combined_adj_b_acc_statistics = {}
    if (cancel_metrics_list and partial_metrics_list and 
        len(cancel_metrics_list) == len(partial_metrics_list)):
        
        combined_values = []
        for cancel_metrics, partial_metrics in zip(cancel_metrics_list, partial_metrics_list):
            if ('adjusted_balanced_accuracy' in cancel_metrics and 
                'adjusted_balanced_accuracy' in partial_metrics):
                
                cancel_adj_b_acc = cancel_metrics['adjusted_balanced_accuracy']
                partial_adj_b_acc = partial_metrics['adjusted_balanced_accuracy']
                
                # Compute harmonic mean
                if cancel_adj_b_acc > 0 and partial_adj_b_acc > 0:
                    harmonic_mean = 2 * (cancel_adj_b_acc * partial_adj_b_acc) / (cancel_adj_b_acc + partial_adj_b_acc)
                else:
                    harmonic_mean = 0.0
                
                combined_values.append(harmonic_mean)
        
        if combined_values:
            combined_adj_b_acc_statistics = {
                'combined_adjusted_balanced_accuracy': {
                    'mean': float(np.mean(combined_values)),
                    'std': float(np.std(combined_values)),
                    'values': combined_values
                }
            }
    
    # Compute avg_thought_words statistics
    thought_statistics = {}
    if avg_thought_words_list:
        thought_statistics = {
            'avg_thought_words': {
                'mean': float(np.mean(avg_thought_words_list)),
                'std': float(np.std(avg_thought_words_list)),
                'values': avg_thought_words_list
            }
        }
    
    # Compute avg_input_words statistics
    input_words_statistics = {}
    if avg_input_words_list:
        input_words_statistics = {
            'avg_input_words': {
                'mean': float(np.mean(avg_input_words_list)),
                'std': float(np.std(avg_input_words_list)),
                'values': avg_input_words_list
            }
        }
    
    # Compute avg_output_words statistics
    output_words_statistics = {}
    if avg_output_words_list:
        output_words_statistics = {
            'avg_output_words': {
                'mean': float(np.mean(avg_output_words_list)),
                'std': float(np.std(avg_output_words_list)),
                'values': avg_output_words_list
            }
        }
    
    # Compute avg_latency statistics
    latency_statistics = {}
    if avg_latency_list:
        latency_statistics = {
            'avg_latency': {
                'mean': float(np.mean(avg_latency_list)),
                'std': float(np.std(avg_latency_list)),
                'values': avg_latency_list
            }
        }
    
    # Compile aggregated results
    aggregated_results = {
        'run_num': run_num,
        'individual_runs': all_runs,
        'cancel_not_for_all_statistics': cancel_statistics,
        'partial_or_full_statistics': partial_statistics,
        'combined_adjusted_balanced_accuracy_statistics': combined_adj_b_acc_statistics,
        'thought_statistics': thought_statistics,
        'input_words_statistics': input_words_statistics,
        'output_words_statistics': output_words_statistics,
        'latency_statistics': latency_statistics,
        'config': {
            'prompt_file_path': prompt_file_path,
            'prompt_name': prompt_name,
            'test_data_path': test_data_path,
            'model': model,
            'temperature': temperature,
            'run_num': run_num
        }
    }
    
    # Print aggregated summary
    if verbose:
        print("\n" + "="*80)
        print("AGGREGATED EVALUATION RESULTS")
        print("="*80)
        print(f"Number of runs: {run_num}")
        
        print(f"\n📊 CANCEL_NOT_FOR_ALL_PASSENGERS STATISTICS:")
        if cancel_statistics:
            for metric_name, stats in cancel_statistics.items():
                print(f"{metric_name.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        print(f"\n📊 PARTIAL_OR_FULL STATISTICS:")
        if partial_statistics:
            for metric_name, stats in partial_statistics.items():
                print(f"{metric_name.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        print(f"\n📊 COMBINED ADJUSTED BALANCED ACCURACY STATISTICS:")
        if combined_adj_b_acc_statistics:
            for metric_name, stats in combined_adj_b_acc_statistics.items():
                print(f"{metric_name.upper().replace('_', ' ')}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        print(f"\n💭 THOUGHT WORDS STATISTICS:")
        if thought_statistics:
            for metric_name, stats in thought_statistics.items():
                print(f"{metric_name.upper().replace('_', ' ')}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        
        print(f"\n📝 INPUT/OUTPUT WORDS STATISTICS:")
        if input_words_statistics:
            for metric_name, stats in input_words_statistics.items():
                print(f"{metric_name.upper().replace('_', ' ')}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        if output_words_statistics:
            for metric_name, stats in output_words_statistics.items():
                print(f"{metric_name.upper().replace('_', ' ')}: {stats['mean']:.1f} ± {stats['std']:.1f}")
        
        print(f"\n⏱️ LATENCY STATISTICS:")
        if latency_statistics:
            for metric_name, stats in latency_statistics.items():
                print(f"{metric_name.upper().replace('_', ' ')}: {stats['mean']:.1f}ms ± {stats['std']:.1f}ms")
    
    return aggregated_results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate prompt performance using OpenAI API against test dataset"
    )
    
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default="prompts/original/identify_partial.yaml",
        help="Path to YAML file containing prompts (default: prompts/original/identify_partial.yaml)"
    )
    
    parser.add_argument(
        "--prompt_name", 
        type=str,
        default="initial_prompt",
        help="Name of the prompt to use from YAML file (default: initial_prompt)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for OpenAI API - controls randomness (0=deterministic, 1=random) (default: 0.0)"
    )
    
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/processed/logs/04222025-08182025/ground_truth/verified_ground_truth_balance_test.json",
        help="Path to test data JSON file"
    )

    parser.add_argument(
        "--data_source",
        type=str,
        default="gpt-5-verified",
        help="Data source (default: gpt-5-verified)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--run_num",
        type=int,
        default=1,
        help="Number of times to run the evaluation for computing mean and std (default: 1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and results"
    )
    
    args = parser.parse_args()
    
    # Generate output filename automatically
    prompt_dir = os.path.dirname(args.prompt_file_path)
    output_filename = f"evaluation_results_{args.prompt_name}_{args.model}_{args.temperature}_runs{args.run_num}.json"
    output_path = os.path.join(prompt_dir, args.data_source, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run evaluation
    print(f"🚀 Evaluating prompt '{args.prompt_name}' from {args.prompt_file_path}")
    print(f"📊 Model: {args.model}, Temperature: {args.temperature}")
    if args.run_num > 1:
        print(f"🔄 Number of runs: {args.run_num} (will compute mean ± std)")
    print(f"📁 Test data: {args.test_data_path}")
    print(f"💾 Output will be saved to: {output_path}")
    
    results = evaluate_prompt_multiple_runs(
        prompt_file_path=args.prompt_file_path,
        prompt_name=args.prompt_name,
        test_data_path=args.test_data_path,
        model=args.model,
        temperature=args.temperature,
        run_num=args.run_num,
        verbose=args.verbose
    )
    
    # Save results
    save_evaluation_results(results, output_path)
    print(f"\n💾 Results saved to {output_path}")