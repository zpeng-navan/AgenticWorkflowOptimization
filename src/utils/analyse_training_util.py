#!/usr/bin/env python3
"""
Training Analysis Utilities

This module provides functions to analyze and visualize training results from OPRO optimization.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def draw_training(
    results_dir: str,
    metric_name: str
) -> None:
    """
    Draw line chart showing metric progression during training steps.
    
    Args:
        results_dir: Path to results directory containing optimization_results.json
        metric_name: Name of the metric to plot (e.g., 'combined_scores', 'cancel_adj_balanced_accuracy')
    
    The function will:
    1. Load optimization_results.json from results_dir
    2. Extract step_statistics 
    3. Plot metric_name_mean with metric_name_std as shaded area
    4. Save as both PNG and PDF
    """
    
    # Load optimization results
    json_file_path = os.path.join(results_dir, "optimization_results.json")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"optimization_results.json not found in {results_dir}")
    
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract step statistics
    step_statistics = results.get('step_statistics', [])
    
    if not step_statistics:
        raise ValueError("No step_statistics found in optimization_results.json")
    
    # Extract data for plotting
    steps = []
    means = []
    stds = []
    
    mean_key = f"{metric_name}_mean"
    std_key = f"{metric_name}_std"
    
    for step_stat in step_statistics:
        if mean_key in step_stat and std_key in step_stat:
            steps.append(step_stat['step'])
            means.append(step_stat[mean_key])
            stds.append(step_stat[std_key])
    
    if not steps:
        raise ValueError(f"No data found for metric '{metric_name}' in step_statistics")
    
    # Convert to numpy arrays for easier manipulation
    steps = np.array(steps)
    means = np.array(means)
    stds = np.array(stds)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the main line
    plt.plot(steps, means, 'b-', linewidth=2, label=f'{metric_name} (mean)')
    
    # Add shaded area for standard deviation
    plt.fill_between(steps, 
                     means - stds, 
                     means + stds, 
                     alpha=0.3, 
                     color='blue', 
                     label=f'{metric_name} (±std)')
    
    # Customize the plot
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(f'{metric_name} Value', fontsize=12)
    plt.title(f'{metric_name.replace("_", " ").title()} vs Training Step', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set integer ticks for x-axis if reasonable number of steps
    if len(steps) <= 20:
        plt.xticks(steps)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plots
    output_filename = f"{metric_name}_vs_step"
    
    # Save PNG
    png_path = os.path.join(results_dir, f"{output_filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved PNG: {png_path}")
    
    # Save PDF  
    pdf_path = os.path.join(results_dir, f"{output_filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    # Close the figure to free memory
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw training metric visualization from OPRO optimization results")
    parser.add_argument("results_dir", 
                       help="Path to results directory containing optimization_results.json")
    parser.add_argument("metric_name", 
                       help="Name of metric to plot (e.g., 'combined_scores', 'cancel_adj_balanced_accuracy')")
    
    args = parser.parse_args()
    
    # Draw training plot with provided arguments
    try:
        draw_training(args.results_dir, args.metric_name)
        print(f"✅ Training plot for '{args.metric_name}' created successfully!")
        print(f"📁 Saved to: {args.results_dir}")
    except Exception as e:
        print(f"❌ Error creating training plot: {e}")
    