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
        
        # Call OpenAI API
        try:
            response = client.call_openai_with_retry(
                prompt=prompt,
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
    
    # Compile final results
    evaluation_results = {
        'total_cases': len(test_data),
        'successful_predictions': len(results),
        'prediction_errors': len(prediction_errors),
        'cancel_not_for_all_metrics': cancel_metrics,
        'partial_or_full_metrics': partial_metrics,
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
    
    return evaluation_results


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
        "--verbose",
        action="store_true",
        help="Print detailed progress and results"
    )
    
    args = parser.parse_args()
    
    # Generate output filename automatically
    prompt_dir = os.path.dirname(args.prompt_file_path)
    output_filename = f"evaluation_results_{args.prompt_name}_{args.model}_{args.temperature}.json"
    output_path = os.path.join(prompt_dir, args.data_source, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Run evaluation
    print(f"🚀 Evaluating prompt '{args.prompt_name}' from {args.prompt_file_path}")
    print(f"📊 Model: {args.model}, Temperature: {args.temperature}")
    print(f"📁 Test data: {args.test_data_path}")
    print(f"💾 Output will be saved to: {output_path}")
    
    results = evaluate_prompt(
        prompt_file_path=args.prompt_file_path,
        prompt_name=args.prompt_name,
        test_data_path=args.test_data_path,
        model=args.model,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Save results
    save_evaluation_results(results, output_path)
    print(f"\n💾 Results saved to {output_path}")