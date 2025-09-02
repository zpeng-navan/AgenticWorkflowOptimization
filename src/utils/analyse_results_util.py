"""
Analysis utility for evaluation results visualization.

This module provides functions to analyze and visualize evaluation results
from different models and prompts, creating comparative bar graphs.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.patches as mpatches

# Define custom colors based on LaTeX specification
CUSTOM_BLUE = (20/255, 81/255, 124/255)  # RGB(20, 81, 124)
CUSTOM_RED = (216/255, 56/255, 58/255)   # RGB(216, 56, 58)
CUSTOM_BLACK = (0, 0, 0)  # RGB(0, 0, 0) - Standard black
API_COST_PER_MILLION_TOKENS = {
    "gpt-3.5-turbo": {"input": 0.50 , "output": 1.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00}
}

def load_evaluation_results(results_dir: str, prompt_name: str) -> Dict[str, Dict]:
    """
    Load all evaluation results for a given prompt name.
    
    Args:
        results_dir: Directory containing the JSON result files
        prompt_name: Name of the prompt (e.g., 'initial_prompt', 'initial_prompt_simple')
    
    Returns:
        Dictionary mapping model names to their evaluation results
    """
    results = {}
    
    # Define model mapping from display name to file name
    model_mapping = {
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4o-mini': 'gpt-4o-mini', 
        'gpt-4o': 'gpt-4o',
        'o3-mini': 'o3-mini',
        'o3': 'o3',
        'gpt-4.1-mini': 'gpt-4.1-mini',
        'gpt-4.1': 'gpt-4.1',
        'gpt-5-mini': 'gpt-5-mini',
        'gpt-5': 'gpt-5'
    }
    
    for model_key, model_file in model_mapping.items():
        filename = f"evaluation_results_{prompt_name}_{model_file}_0.0_runs5.json"
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[model_key] = data
                print(f"✅ Loaded results for {model_key}")
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
        else:
            print(f"⚠️ File not found: {filepath}")
    
    return results


def extract_metric_values(results: Dict[str, Dict], metric_name: str) -> Tuple[List[float], List[float]]:
    """
    Extract metric values for both labels from results.
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        metric_name: Name of the metric to extract (e.g., 'f1', 'accuracy', 'precision')
    
    Returns:
        Tuple of (cancel_not_for_all_values, partial_or_full_values)
    """
    # Model display order as specified
    model_order = [
        'gpt-3.5-turbo',  # 3.5-turbo
        'gpt-4o-mini',    # 4o-mini  
        'gpt-4o',         # 4o
        'o3-mini',        # o3-mini
        'o3',             # o3
        'gpt-4.1-mini',   # 4.1-mini
        'gpt-4.1',        # 4.1
        'gpt-5-mini',     # 5-mini
        'gpt-5'           # 5
    ]
    
    cancel_values = []
    partial_values = []
    
    for model in model_order:
        if model in results:
            data = results[model]
            
            # Extract cancel_not_for_all metric
            if 'cancel_not_for_all_statistics' in data and metric_name in data['cancel_not_for_all_statistics']:
                cancel_mean = data['cancel_not_for_all_statistics'][metric_name]['mean']
                cancel_values.append(cancel_mean)
            else:
                cancel_values.append(0.0)  # Default value if missing
                print(f"⚠️ Missing cancel_not_for_all {metric_name} for {model}")
            
            # Extract partial_or_full metric  
            if 'partial_or_full_statistics' in data and metric_name in data['partial_or_full_statistics']:
                partial_mean = data['partial_or_full_statistics'][metric_name]['mean']
                partial_values.append(partial_mean)
            else:
                partial_values.append(0.0)  # Default value if missing
                print(f"⚠️ Missing partial_or_full {metric_name} for {model}")
        else:
            # Model not found, add zero values
            cancel_values.append(0.0)
            partial_values.append(0.0)
            print(f"⚠️ Model {model} not found in results")
    
    return cancel_values, partial_values


def two_labels_diff_model(metric_name: str, prompt_name: str, 
                         results_dir: str = "prompts/original/gpt-5-verified",
                         output_dir: str = "imgs/baselines") -> None:
    """
    Create a bar graph comparing two labels across different models for a given metric.
    
    Args:
        metric_name: Name of the metric to compare (e.g., 'balanced_accuracy', 'f1', 'precision')
        prompt_name: Name of the prompt ('initial_prompt' or 'initial_prompt_simple') 
        results_dir: Directory containing the JSON result files
        output_dir: Directory to save the output graphs
    """
    print(f"🚀 Creating bar graph for {metric_name} with {prompt_name}")
    
    # Load evaluation results
    results = load_evaluation_results(results_dir, prompt_name)
    
    if not results:
        print("❌ No results loaded. Cannot create graph.")
        return
    
    # Extract metric values
    cancel_values, partial_values = extract_metric_values(results, metric_name)
    
    # Model labels for display (shortened versions)
    model_labels = [
        'GPT-3.5-Turbo',
        'GPT-4o-Mini', 
        'GPT-4o',
        'o3-Mini',
        'o3',
        'GPT-4.1-Mini',
        'GPT-4.1',
        'GPT-5-Mini',
        'GPT-5'
    ]
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_labels))  # Label locations
    width = 0.35  # Width of bars
    
    # Create bars
    bars1 = ax.bar(x - width/2, cancel_values, width, label='cancel_not_for_all', 
                   color=CUSTOM_BLUE, alpha=0.8)
    bars2 = ax.bar(x + width/2, partial_values, width, label='partial_or_full', 
                   color=CUSTOM_RED, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name.replace("_", " ").title()} vs Models with {prompt_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if value is non-zero
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Improve layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename - note: corrected the filename format
    filename_base = f"{metric_name}_vs_models_{prompt_name}"
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved PNG: {png_path}")
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"{filename_base}.pdf") 
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved PDF: {pdf_path}")
    
    # Show the plot
    plt.show()
    plt.close()


def extract_latency_or_cost_values(results: Dict[str, Dict], latency_or_cost: str) -> List[float]:
    """
    Extract latency or cost values for all models.
    
    Args:
        results: Dictionary mapping model names to their evaluation results
        latency_or_cost: Either 'latency' or 'cost'
    
    Returns:
        List of latency (ms) or cost (USD per 1M calls) values
    """
    # Model display order as specified
    model_order = [
        'gpt-3.5-turbo',  # 3.5-turbo
        'gpt-4o-mini',    # 4o-mini  
        'gpt-4o',         # 4o
        'o3-mini',        # o3-mini
        'o3',             # o3
        'gpt-4.1-mini',   # 4.1-mini
        'gpt-4.1',        # 4.1
        'gpt-5-mini',     # 5-mini
        'gpt-5'           # 5
    ]
    
    values = []
    
    for model in model_order:
        if model in results:
            data = results[model]
            
            if latency_or_cost == "latency":
                # Extract latency
                if 'latency_statistics' in data and 'avg_latency' in data['latency_statistics']:
                    latency_ms = data['latency_statistics']['avg_latency']['mean']
                    values.append(latency_ms)
                else:
                    values.append(0.0)
                    print(f"⚠️ Missing latency for {model}")
            
            elif latency_or_cost == "cost":
                # Calculate cost per 1M calls
                if ('input_words_statistics' in data and 'output_words_statistics' in data and
                    model in API_COST_PER_MILLION_TOKENS):
                    
                    avg_input_words = data['input_words_statistics']['avg_input_words']['mean']
                    avg_output_words = data['output_words_statistics']['avg_output_words']['mean']
                    
                    input_cost_per_1M = API_COST_PER_MILLION_TOKENS[model]['input']  # USD per 1M input tokens
                    output_cost_per_1M = API_COST_PER_MILLION_TOKENS[model]['output']  # USD per 1M output tokens
                    
                    # Cost for 1M calls = avg_input_words * input_cost_per_1M + avg_output_words * output_cost_per_1M
                    total_cost = avg_input_words * input_cost_per_1M + avg_output_words * output_cost_per_1M
                    values.append(total_cost)
                else:
                    values.append(0.0)
                    print(f"⚠️ Missing cost data for {model}")
            else:
                values.append(0.0)
                print(f"⚠️ Unknown latency_or_cost option: {latency_or_cost}")
        else:
            # Model not found, add zero value
            values.append(0.0)
            print(f"⚠️ Model {model} not found in results")
    
    return values


def two_labels_diff_model_one_more(metric_name: str, 
                                 latency_or_cost: str = "latency",
                                 prompt_name: str = "initial_prompt", 
                                 results_dir: str = "prompts/original/gpt-5-verified",
                                 output_dir: str = "imgs/baselines") -> None:
    """
    Create a bar graph with line chart overlay showing metrics vs models with latency or cost.
    
    Args:
        metric_name: Name of the metric to compare (e.g., 'balanced_accuracy', 'f1', 'precision')
        latency_or_cost: Either 'latency' or 'cost' to determine the line chart data
        prompt_name: Name of the prompt ('initial_prompt' or 'initial_prompt_simple') 
        results_dir: Directory containing the JSON result files
        output_dir: Directory to save the output graphs
    """
    print(f"🚀 Creating combined bar+line graph for {metric_name} with {latency_or_cost} overlay using {prompt_name}")
    
    # Load evaluation results
    results = load_evaluation_results(results_dir, prompt_name)
    
    if not results:
        print("❌ No results loaded. Cannot create graph.")
        return
    
    # Extract metric values
    cancel_values, partial_values = extract_metric_values(results, metric_name)
    
    # Extract latency or cost values
    secondary_values = extract_latency_or_cost_values(results, latency_or_cost)
    
    # Model labels for display (shortened versions)
    model_labels = [
        'GPT-3.5-Turbo',
        'GPT-4o-Mini', 
        'GPT-4o',
        'o3-Mini',
        'o3',
        'GPT-4.1-Mini',
        'GPT-4.1',
        'GPT-5-Mini',
        'GPT-5'
    ]
    
    # Create the combined bar and line graph
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_labels))  # Label locations
    width = 0.35  # Width of bars
    
    # Create bars on primary y-axis
    bars1 = ax1.bar(x - width/2, cancel_values, width, label='cancel_not_for_all', 
                    color=CUSTOM_BLUE, alpha=0.8)
    bars2 = ax1.bar(x + width/2, partial_values, width, label='partial_or_full', 
                    color=CUSTOM_RED, alpha=0.8)
    
    # Customize primary axis (bars)
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if value is non-zero
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Create secondary y-axis for line chart
    ax2 = ax1.twinx()
    
    # Determine line color and label
    line_color = CUSTOM_BLACK
    if latency_or_cost == "latency":
        line_label = 'Avg Latency (ms)'
        y2_label = 'Latency (ms)'
    else:  # cost
        line_label = 'Cost per 1M calls (USD)'
        y2_label = 'Cost (USD per 1M calls)'
    
    # Create line chart on secondary y-axis
    line = ax2.plot(x, secondary_values, color=line_color, marker='o', linewidth=2, 
                    markersize=6, label=line_label)
    ax2.set_ylabel(y2_label, fontsize=12, fontweight='bold', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    
    # Add value labels on line points
    for i, value in enumerate(secondary_values):
        if value > 0:
            if latency_or_cost == "latency":
                label_text = f'{value:.1f}ms'
            else:  # cost
                label_text = f'${value:.2f}'
            ax2.annotate(label_text,
                        xy=(x[i], value),
                        xytext=(0, 10),  # 10 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color=line_color)
    
    # Set title
    title_suffix = "Latency" if latency_or_cost == "latency" else "Cost"
    ax1.set_title(f'{metric_name.replace("_", " ").title()} vs Models with {title_suffix} ({prompt_name.replace("_", " ").title()})', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Create combined legend
    bars_legend = [bars1, bars2]
    bars_labels = ['cancel_not_for_all', 'partial_or_full']
    line_legend = line
    line_labels = [line_label]
    
    # Combine legends
    all_handles = bars_legend + line_legend
    all_labels = bars_labels + line_labels
    ax1.legend(all_handles, all_labels, loc='upper left', fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename_base = f"{metric_name}_vs_models_{prompt_name}_{latency_or_cost}"
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved PNG: {png_path}")
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"{filename_base}.pdf") 
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved PDF: {pdf_path}")
    
    # Show the plot
    plt.show()
    plt.close()


def create_all_metric_comparisons(prompt_name: str, 
                                results_dir: str = "prompts/original/gpt-5-verified",
                                output_dir: str = "imgs/baselines") -> None:
    """
    Create bar graphs for all available metrics for a given prompt.
    
    Args:
        prompt_name: Name of the prompt ('initial_prompt' or 'initial_prompt_simple')
        results_dir: Directory containing the JSON result files  
        output_dir: Directory to save the output graphs
    """
    metrics = [
        'precision', 'recall', 'f1', 'accuracy', 
        'balanced_accuracy', 'adjusted_balanced_accuracy'
    ]
    
    print(f"🎯 Creating all metric comparisons for {prompt_name}")
    
    for metric in metrics:
        print(f"\n📊 Processing {metric}...")
        two_labels_diff_model(metric, prompt_name, results_dir, output_dir)
    
    print(f"\n✅ All metric comparisons completed for {prompt_name}")


if __name__ == "__main__":
    # Example usage
    print("📊 Analysis Utility for Evaluation Results")
    print("=" * 50)
    
    # Create a single comparison
    # two_labels_diff_model("balanced_accuracy", "initial_prompt_simple")
    
    # Test the new combined function
    two_labels_diff_model_one_more("adjusted_balanced_accuracy", "cost", "initial_prompt")
    two_labels_diff_model_one_more("adjusted_balanced_accuracy", "cost", "initial_prompt_simple")
    two_labels_diff_model_one_more("adjusted_balanced_accuracy", "latency", "initial_prompt")
    two_labels_diff_model_one_more("adjusted_balanced_accuracy", "latency", "initial_prompt_simple")
    
    # Uncomment to create all metric comparisons
    # create_all_metric_comparisons("initial_prompt")
    # create_all_metric_comparisons("initial_prompt_simple")
