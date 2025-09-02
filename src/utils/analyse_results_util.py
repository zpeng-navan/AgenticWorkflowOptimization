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


def metric_cost_latency(metric_name: str,
                       prompt_name: str = "initial_prompt",
                       results_dir: str = "prompts/original/gpt-5-verified", 
                       output_dir: str = "imgs/baselines") -> None:
    """
    Create a triple-axis graph showing:
    1. Average metric of both labels (bars)
    2. Cost per 1M calls (line)
    3. Latency (line)
    
    Args:
        metric_name: Name of the metric to plot
        prompt_name: Name of the prompt ('initial_prompt' or 'initial_prompt_simple')
        results_dir: Directory containing the JSON result files
        output_dir: Directory to save the output graphs
    """
    print(f"🚀 Creating triple-axis graph for {metric_name} with cost and latency using {prompt_name}")
    
    # Load results
    results = load_evaluation_results(results_dir, prompt_name)
    if not results:
        print(f"❌ No results found for {prompt_name}")
        return
    
    # Extract metric values for both labels
    cancel_values, partial_values = extract_metric_values(results, metric_name)
    
    # Calculate average of both labels
    avg_metric_values = [(c + p) / 2 for c, p in zip(cancel_values, partial_values)]
    
    # Extract cost and latency values
    cost_values = extract_latency_or_cost_values(results, "cost")
    latency_values = extract_latency_or_cost_values(results, "latency")
    
    # Model order
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o3-mini", "o3", 
              "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5"]
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create bars for average metric (primary y-axis)
    x = np.arange(len(models))
    bars = ax1.bar(x, avg_metric_values, color=CUSTOM_BLUE, alpha=0.7, 
                   label=f'Avg {metric_name.replace("_", " ").title()}', width=0.6)
    
    # Customize primary y-axis
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Average {metric_name.replace("_", " ").title()}', 
                   fontsize=12, fontweight='bold', color=CUSTOM_BLUE)
    ax1.tick_params(axis='y', labelcolor=CUSTOM_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
    
    add_value_labels_bars(bars)
    
    # Create secondary y-axis for cost
    ax2 = ax1.twinx()
    line_cost = ax2.plot(x, cost_values, color=CUSTOM_RED, marker='o', linewidth=2, 
                        markersize=6, label='Cost per 1M calls (USD)', linestyle='-')
    ax2.set_ylabel('Cost (USD per 1M calls)', fontsize=12, fontweight='bold', color=CUSTOM_RED)
    ax2.tick_params(axis='y', labelcolor=CUSTOM_RED)
    
    # Create tertiary y-axis for latency
    ax3 = ax1.twinx()
    # Offset the third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    line_latency = ax3.plot(x, latency_values, color=CUSTOM_BLACK, marker='s', linewidth=2,
                           markersize=6, label='Avg Latency (ms)', linestyle='--')
    ax3.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold', color=CUSTOM_BLACK)
    ax3.tick_params(axis='y', labelcolor=CUSTOM_BLACK)
    
    # Add value labels for cost line
    for i, value in enumerate(cost_values):
        ax2.annotate(f'${value:.2f}',
                    xy=(i, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, color=CUSTOM_RED, fontweight='bold')
    
    # Add value labels for latency line
    for i, value in enumerate(latency_values):
        ax3.annotate(f'{value:.1f}ms',
                    xy=(i, value),
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=8, color=CUSTOM_BLACK, fontweight='bold')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
              loc='upper left', bbox_to_anchor=(0.005, 0.98))
    
    # Set title
    title = f'{metric_name.replace("_", " ").title()} (Avg), Cost & Latency vs Models ({prompt_name.replace("_", " ").title()})'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{metric_name}_vs_models_{prompt_name}_cost_latency"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"💾 Saved PNG: {png_path}")
    print(f"💾 Saved PDF: {pdf_path}")
    
    plt.show()
    plt.close()


def metric_cost_latency_two_prompt(metric_name: str,
                                  results_dir: str = "prompts/original/gpt-5-verified", 
                                  output_dir: str = "imgs/baselines") -> None:
    """
    Create a comprehensive graph comparing both initial_prompt and initial_prompt_simple showing:
    1. Average metric of both labels (grouped bars)
    2. Cost per 1M calls for both prompts (lines)
    3. Latency for both prompts (lines)
    
    Colors:
    - Blue: initial_prompt
    - Red: initial_prompt_simple
    
    Args:
        metric_name: Name of the metric to plot
        results_dir: Directory containing the JSON result files
        output_dir: Directory to save the output graphs
    """
    print(f"🚀 Creating comparison graph for {metric_name} with cost and latency for both prompts")
    
    # Load results for both prompts
    results_initial = load_evaluation_results(results_dir, "initial_prompt")
    results_simple = load_evaluation_results(results_dir, "initial_prompt_simple")
    
    if not results_initial or not results_simple:
        print(f"❌ Missing results for one or both prompts")
        return
    
    # Extract data for initial_prompt
    cancel_values_init, partial_values_init = extract_metric_values(results_initial, metric_name)
    avg_metric_values_init = [(c + p) / 2 for c, p in zip(cancel_values_init, partial_values_init)]
    cost_values_init = extract_latency_or_cost_values(results_initial, "cost")
    latency_values_init = extract_latency_or_cost_values(results_initial, "latency")
    
    # Extract data for initial_prompt_simple
    cancel_values_simple, partial_values_simple = extract_metric_values(results_simple, metric_name)
    avg_metric_values_simple = [(c + p) / 2 for c, p in zip(cancel_values_simple, partial_values_simple)]
    cost_values_simple = extract_latency_or_cost_values(results_simple, "cost")
    latency_values_simple = extract_latency_or_cost_values(results_simple, "latency")
    
    # Model order
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o3-mini", "o3", 
              "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5"]
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Create line charts for average metric (primary y-axis)
    x = np.arange(len(models))
    
    # Round symbols for metric values
    line_metric_init = ax1.plot(x, avg_metric_values_init, color=CUSTOM_BLUE, marker='o', 
                               linewidth=2, markersize=8, linestyle='-',
                               label=f'Initial Prompt - Avg {metric_name.replace("_", " ").title()}')
    line_metric_simple = ax1.plot(x, avg_metric_values_simple, color=CUSTOM_RED, marker='o',
                                 linewidth=2, markersize=8, linestyle='-',
                                 label=f'Initial Prompt Simple - Avg {metric_name.replace("_", " ").title()}')
    
    # Customize primary y-axis
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Average {metric_name.replace("_", " ").title()}', 
                   fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels for metric lines
    for i, (value_init, value_simple) in enumerate(zip(avg_metric_values_init, avg_metric_values_simple)):
        ax1.annotate(f'{value_init:.3f}',
                    xy=(i, value_init),
                    xytext=(-8, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, color=CUSTOM_BLUE, fontweight='bold')
        ax1.annotate(f'{value_simple:.3f}',
                    xy=(i, value_simple),
                    xytext=(8, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, color=CUSTOM_RED, fontweight='bold')
    
    # Create secondary y-axis for cost
    ax2 = ax1.twinx()
    # Square symbols for cost values
    line_cost_init = ax2.plot(x, cost_values_init, color=CUSTOM_BLUE, marker='s', 
                             linewidth=2, markersize=6, linestyle='-',
                             label='Initial Prompt - Cost per 1M calls (USD)')
    line_cost_simple = ax2.plot(x, cost_values_simple, color=CUSTOM_RED, marker='s', 
                               linewidth=2, markersize=6, linestyle='-',
                               label='Initial Prompt Simple - Cost per 1M calls (USD)')
    ax2.set_ylabel('Cost (USD per 1M calls)', fontsize=12, fontweight='bold')
    
    # Create tertiary y-axis for latency
    ax3 = ax1.twinx()
    # Offset the third y-axis
    ax3.spines['right'].set_position(('outward', 60))
    # Triangle symbols for latency values
    line_latency_init = ax3.plot(x, latency_values_init, color=CUSTOM_BLUE, marker='^', 
                                linewidth=2, markersize=6, linestyle='-',
                                label='Initial Prompt - Avg Latency (ms)')
    line_latency_simple = ax3.plot(x, latency_values_simple, color=CUSTOM_RED, marker='^',
                                  linewidth=2, markersize=6, linestyle='-',
                                  label='Initial Prompt Simple - Avg Latency (ms)')
    ax3.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    
    # Add value labels for cost lines
    for i, (value_init, value_simple) in enumerate(zip(cost_values_init, cost_values_simple)):
        ax2.annotate(f'${value_init:.2f}',
                    xy=(i, value_init),
                    xytext=(-8, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, color=CUSTOM_BLUE, fontweight='bold')
        ax2.annotate(f'${value_simple:.2f}',
                    xy=(i, value_simple),
                    xytext=(8, 12),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, color=CUSTOM_RED, fontweight='bold')
    
    # Add value labels for latency lines
    for i, (value_init, value_simple) in enumerate(zip(latency_values_init, latency_values_simple)):
        ax3.annotate(f'{value_init:.1f}ms',
                    xy=(i, value_init),
                    xytext=(-8, -18),
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=7, color=CUSTOM_BLUE, fontweight='bold')
        ax3.annotate(f'{value_simple:.1f}ms',
                    xy=(i, value_simple),
                    xytext=(8, -18),
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=7, color=CUSTOM_RED, fontweight='bold')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
              loc='upper left', bbox_to_anchor=(0.005, 0.98), fontsize=9)
    
    # Set title
    title = f'{metric_name.replace("_", " ").title()} (Avg), Cost & Latency Comparison: Initial Prompt vs Initial Prompt Simple'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{metric_name}_vs_models_two_prompts_cost_latency"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"💾 Saved PNG: {png_path}")
    print(f"💾 Saved PDF: {pdf_path}")
    
    plt.show()
    plt.close()


def metric_cost_latency_two_prompt_grouped_bars(metric_name: str,
                                               results_dir: str = "prompts/original/gpt-5-verified", 
                                               output_dir: str = "imgs/baselines") -> None:
    """
    Create a grouped bar chart comparing both initial_prompt and initial_prompt_simple showing:
    - 3 groups per model: avg metric, cost, latency
    - 2 bars per group: initial_prompt (blue) vs initial_prompt_simple (red)
    - Different hatch patterns for each metric type
    
    Args:
        metric_name: Name of the metric to plot
        results_dir: Directory containing the JSON result files
        output_dir: Directory to save the output graphs
    """
    print(f"🚀 Creating grouped bar chart for {metric_name} with cost and latency for both prompts")
    
    # Load results for both prompts
    results_initial = load_evaluation_results(results_dir, "initial_prompt")
    results_simple = load_evaluation_results(results_dir, "initial_prompt_simple")
    
    if not results_initial or not results_simple:
        print(f"❌ Missing results for one or both prompts")
        return
    
    # Extract data for initial_prompt
    cancel_values_init, partial_values_init = extract_metric_values(results_initial, metric_name)
    avg_metric_values_init = [(c + p) / 2 for c, p in zip(cancel_values_init, partial_values_init)]
    cost_values_init = extract_latency_or_cost_values(results_initial, "cost")
    latency_values_init = extract_latency_or_cost_values(results_initial, "latency")
    
    # Extract data for initial_prompt_simple
    cancel_values_simple, partial_values_simple = extract_metric_values(results_simple, metric_name)
    avg_metric_values_simple = [(c + p) / 2 for c, p in zip(cancel_values_simple, partial_values_simple)]
    cost_values_simple = extract_latency_or_cost_values(results_simple, "cost")
    latency_values_simple = extract_latency_or_cost_values(results_simple, "latency")
    
    # Normalize only cost and latency to 0-1 range (metric values are already 0-1)
    def normalize_values(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [1.0] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    # Metric values are already 0-1, no normalization needed
    norm_metric_init = avg_metric_values_init
    norm_metric_simple = avg_metric_values_simple
    
    # Normalize cost and latency separately (they have different scales)
    norm_cost_init = normalize_values(cost_values_init + cost_values_simple)[:len(cost_values_init)]
    norm_cost_simple = normalize_values(cost_values_init + cost_values_simple)[len(cost_values_init):]
    
    norm_latency_init = normalize_values(latency_values_init + latency_values_simple)[:len(latency_values_init)]
    norm_latency_simple = normalize_values(latency_values_init + latency_values_simple)[len(latency_values_init):]
    
    # Model order
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o3-mini", "o3", 
              "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Set up positions for grouped bars
    n_models = len(models)
    n_groups = 3  # metric, cost, latency
    group_width = 0.8
    bar_width = group_width / (n_groups * 2)  # 2 bars per group
    group_spacing = 0.1
    
    # Calculate positions
    model_positions = np.arange(n_models)
    
    # Define hatch patterns for different metrics
    metric_hatch = ''      # No pattern for metric (solid)
    cost_hatch = '///'     # Diagonal lines for cost
    latency_hatch = '...'  # Dots for latency
    
    # Plot bars for each model
    for i, model in enumerate(models):
        base_pos = model_positions[i] - group_width/2
        
        # Group 1: Average Metric
        pos1_init = base_pos + 0 * (group_width / n_groups) + bar_width/2
        pos1_simple = pos1_init + bar_width
        
        # Group 2: Cost  
        pos2_init = base_pos + 1 * (group_width / n_groups) + bar_width/2
        pos2_simple = pos2_init + bar_width
        
        # Group 3: Latency
        pos3_init = base_pos + 2 * (group_width / n_groups) + bar_width/2
        pos3_simple = pos3_init + bar_width
        
        # Plot metric bars
        ax.bar(pos1_init, norm_metric_init[i], bar_width, 
               color=CUSTOM_BLUE, hatch=metric_hatch, alpha=0.8,
               label='Initial Prompt - Metric' if i == 0 else '')
        ax.bar(pos1_simple, norm_metric_simple[i], bar_width,
               color=CUSTOM_RED, hatch=metric_hatch, alpha=0.8,
               label='Initial Prompt Simple - Metric' if i == 0 else '')
        
        # Plot cost bars
        ax.bar(pos2_init, norm_cost_init[i], bar_width,
               color=CUSTOM_BLUE, hatch=cost_hatch, alpha=0.8,
               label='Initial Prompt - Cost' if i == 0 else '')
        ax.bar(pos2_simple, norm_cost_simple[i], bar_width,
               color=CUSTOM_RED, hatch=cost_hatch, alpha=0.8,
               label='Initial Prompt Simple - Cost' if i == 0 else '')
        
        # Plot latency bars
        ax.bar(pos3_init, norm_latency_init[i], bar_width,
               color=CUSTOM_BLUE, hatch=latency_hatch, alpha=0.8,
               label='Initial Prompt - Latency' if i == 0 else '')
        ax.bar(pos3_simple, norm_latency_simple[i], bar_width,
               color=CUSTOM_RED, hatch=latency_hatch, alpha=0.8,
               label='Initial Prompt Simple - Latency' if i == 0 else '')
        
        # Add value labels on bars with original values
        def add_value_label(pos, norm_val, orig_val, metric_type):
            if metric_type == 'metric':
                label_text = f'{orig_val:.3f}'
            elif metric_type == 'cost':
                label_text = f'${orig_val:.2f}'
            else:  # latency
                label_text = f'{orig_val:.1f}ms'
            
            ax.annotate(label_text, xy=(pos, norm_val), xytext=(0, 3),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=9, rotation=90, fontweight='bold')
        
        # Add labels with original values
        add_value_label(pos1_init, norm_metric_init[i], avg_metric_values_init[i], 'metric')
        add_value_label(pos1_simple, norm_metric_simple[i], avg_metric_values_simple[i], 'metric')
        add_value_label(pos2_init, norm_cost_init[i], cost_values_init[i], 'cost')
        add_value_label(pos2_simple, norm_cost_simple[i], cost_values_simple[i], 'cost')
        add_value_label(pos3_init, norm_latency_init[i], latency_values_init[i], 'latency')
        add_value_label(pos3_simple, norm_latency_simple[i], latency_values_simple[i], 'latency')
    
    # Customize axes
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Values (Metrics: Original 0-1 Scale, Cost & Latency: Normalized 0-1)', fontsize=14, fontweight='bold')
    ax.set_title(f'Grouped Comparison: {metric_name.replace("_", " ").title()}, Cost & Latency\n(Initial Prompt vs Initial Prompt Simple)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Add group separators
    for i in range(1, n_models):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create legend in upper left corner
    ax.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{metric_name}_vs_models_two_prompts_grouped_bars"
    
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"💾 Saved PNG: {png_path}")
    print(f"💾 Saved PDF: {pdf_path}")
    
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
    two_labels_diff_model("adjusted_balanced_accuracy", "initial_prompt")
    two_labels_diff_model("adjusted_balanced_accuracy", "initial_prompt_simple")
    
    # Test the new triple-axis function
    metric_cost_latency("adjusted_balanced_accuracy", "initial_prompt")
    metric_cost_latency("adjusted_balanced_accuracy", "initial_prompt_simple")
    
    # Test the new grouped bar chart function
    metric_cost_latency_two_prompt_grouped_bars("adjusted_balanced_accuracy")
