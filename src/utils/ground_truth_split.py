#!/usr/bin/env python3
"""
Ground truth split utility to create balanced datasets and train/dev/test splits.

This utility processes ground_truth.json to:
1. Create a balanced dataset by sampling equal numbers of sessions for each label
2. Split the balanced dataset into train/dev/test sets
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter


def load_ground_truth(file_path: str) -> Dict[str, Any]:
    """Load ground truth JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def analyze_distribution(data: Dict[str, Any], name: str) -> None:
    """Analyze and display the distribution of labels in the dataset."""
    print(f"\n📊 {name} Distribution")
    print("-" * 50)
    print(f"Total sessions: {len(data)}")
    
    # Count cancel_not_for_all
    cancel_not_for_all_counts = Counter()
    partial_or_full_counts = Counter()
    
    for channel_id, session_data in data.items():
        # Count cancel_not_for_all
        cancel_value = str(session_data.get("cancel_not_for_all", "null")).lower()
        cancel_not_for_all_counts[cancel_value] += 1
        
        # Count partial_or_full
        partial_value = str(session_data.get("partial_or_full", "null")).lower()
        partial_or_full_counts[partial_value] += 1
    
    print(f"\ncancel_not_for_all distribution:")
    for value, count in sorted(cancel_not_for_all_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")
    
    print(f"\npartial_or_full distribution:")
    for value, count in sorted(partial_or_full_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")


def create_balanced_dataset(ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a balanced dataset by sampling equal numbers for each label.
    
    Args:
        ground_truth_data: Original ground truth dataset
        
    Returns:
        Balanced dataset
    """
    print("🎯 Creating Balanced Dataset")
    print("=" * 50)
    
    # Separate sessions by labels
    cancel_true_sessions = {}
    cancel_false_sessions = {}
    partial_partial_sessions = {}
    partial_full_sessions = {}
    
    for channel_id, session_data in ground_truth_data.items():
        cancel_value = str(session_data.get("cancel_not_for_all", "null")).lower()
        partial_value = str(session_data.get("partial_or_full", "null")).lower()
        
        # Group by cancel_not_for_all
        if cancel_value == "true":
            cancel_true_sessions[channel_id] = session_data
        elif cancel_value == "false":
            cancel_false_sessions[channel_id] = session_data
        
        # Group by partial_or_full (exclude null)
        if partial_value == "partial":
            partial_partial_sessions[channel_id] = session_data
        elif partial_value == "full":
            partial_full_sessions[channel_id] = session_data
    
    print(f"📋 Original session counts:")
    print(f"  cancel_not_for_all=true: {len(cancel_true_sessions)}")
    print(f"  cancel_not_for_all=false: {len(cancel_false_sessions)}")
    print(f"  partial_or_full=partial: {len(partial_partial_sessions)}")
    print(f"  partial_or_full=full: {len(partial_full_sessions)}")
    
    # Balance cancel_not_for_all (need 186 of each)
    target_cancel_count = len(cancel_true_sessions)  # 186
    
    # Balance partial_or_full (need 71 of each)
    target_partial_count = len(partial_partial_sessions)  # 71
    
    print(f"\n🎯 Balancing targets:")
    print(f"  cancel_not_for_all: {target_cancel_count} each for true/false")
    print(f"  partial_or_full: {target_partial_count} each for partial/full")
    
    # Sample sessions for balanced dataset
    balanced_data = {}
    
    # Add all cancel_not_for_all=true sessions (186)
    balanced_data.update(cancel_true_sessions)
    
    # Randomly sample 186 cancel_not_for_all=false sessions
    if len(cancel_false_sessions) >= target_cancel_count:
        sampled_cancel_false = dict(random.sample(list(cancel_false_sessions.items()), target_cancel_count))
        # Add only those not already in balanced_data to avoid duplicates
        for channel_id, session_data in sampled_cancel_false.items():
            if channel_id not in balanced_data:
                balanced_data[channel_id] = session_data
    else:
        print(f"⚠️  Not enough cancel_not_for_all=false sessions ({len(cancel_false_sessions)} < {target_cancel_count})")
        # Add all available
        for channel_id, session_data in cancel_false_sessions.items():
            if channel_id not in balanced_data:
                balanced_data[channel_id] = session_data
    
    # Add all partial_or_full=partial sessions (71)
    for channel_id, session_data in partial_partial_sessions.items():
        if channel_id not in balanced_data:
            balanced_data[channel_id] = session_data
    
    # Randomly sample 71 partial_or_full=full sessions
    if len(partial_full_sessions) >= target_partial_count:
        sampled_partial_full = dict(random.sample(list(partial_full_sessions.items()), target_partial_count))
        # Add only those not already in balanced_data to avoid duplicates
        for channel_id, session_data in sampled_partial_full.items():
            if channel_id not in balanced_data:
                balanced_data[channel_id] = session_data
    else:
        print(f"⚠️  Not enough partial_or_full=full sessions ({len(partial_full_sessions)} < {target_partial_count})")
        # Add all available
        for channel_id, session_data in partial_full_sessions.items():
            if channel_id not in balanced_data:
                balanced_data[channel_id] = session_data
    
    print(f"\n✅ Balanced dataset created with {len(balanced_data)} sessions")
    
    return balanced_data


def split_dataset(data: Dict[str, Any], train_ratio: float = 0.5, dev_ratio: float = 0.2, test_ratio: float = 0.3) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split dataset into train/dev/test sets.
    
    Args:
        data: Dataset to split
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        test_ratio: Proportion for test set
        
    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    print(f"\n🔀 Splitting Dataset")
    print("=" * 30)
    
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️  Warning: Ratios sum to {total_ratio:.3f}, not 1.0")
    
    # Convert to list for shuffling
    items = list(data.items())
    random.shuffle(items)
    
    total_count = len(items)
    train_count = int(total_count * train_ratio)
    dev_count = int(total_count * dev_ratio)
    test_count = total_count - train_count - dev_count  # Remaining goes to test
    
    print(f"Total sessions: {total_count}")
    print(f"Train: {train_count} ({train_count/total_count*100:.1f}%)")
    print(f"Dev: {dev_count} ({dev_count/total_count*100:.1f}%)")
    print(f"Test: {test_count} ({test_count/total_count*100:.1f}%)")
    
    # Split the data
    train_items = items[:train_count]
    dev_items = items[train_count:train_count + dev_count]
    test_items = items[train_count + dev_count:]
    
    train_data = dict(train_items)
    dev_data = dict(dev_items)
    test_data = dict(test_items)
    
    return train_data, dev_data, test_data


def save_dataset(data: Dict[str, Any], file_path: str, name: str) -> None:
    """Save dataset to JSON file."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 {name} saved to: {file_path}")


def main():
    """Main function to create balanced dataset and splits."""
    print("🚀 Ground Truth Balancing and Splitting")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    input_file = base_dir / "data/processed/logs/04222025-08182025/ground_truth/ground_truth.json"
    output_dir = base_dir / "data/processed/logs/04222025-08182025/ground_truth"
    
    balance_file = output_dir / "ground_truth_balance.json"
    train_file = output_dir / "ground_truth_balance_train.json"
    dev_file = output_dir / "ground_truth_balance_dev.json"
    test_file = output_dir / "ground_truth_balance_test.json"
    
    # Load original ground truth data
    print(f"📁 Loading ground truth data from: {input_file}")
    ground_truth_data = load_ground_truth(str(input_file))
    
    if not ground_truth_data:
        print("❌ Failed to load ground truth data")
        return
    
    # Analyze original distribution
    analyze_distribution(ground_truth_data, "Original Dataset")
    
    # Create balanced dataset
    balanced_data = create_balanced_dataset(ground_truth_data)
    
    # Analyze balanced distribution
    analyze_distribution(balanced_data, "Balanced Dataset")
    
    # Save balanced dataset
    save_dataset(balanced_data, str(balance_file), "Balanced dataset")
    
    # Split into train/dev/test
    train_data, dev_data, test_data = split_dataset(balanced_data, 0.5, 0.2, 0.3)
    
    # Save split datasets
    save_dataset(train_data, str(train_file), "Training set")
    save_dataset(dev_data, str(dev_file), "Development set")
    save_dataset(test_data, str(test_file), "Test set")
    
    # Analyze split distributions
    analyze_distribution(train_data, "Training Set")
    analyze_distribution(dev_data, "Development Set")
    analyze_distribution(test_data, "Test Set")
    
    print(f"\n🎉 All datasets created successfully!")
    print(f"📊 Final summary:")
    print(f"  Balanced dataset: {len(balanced_data)} sessions")
    print(f"  Training set: {len(train_data)} sessions")
    print(f"  Development set: {len(dev_data)} sessions")
    print(f"  Test set: {len(test_data)} sessions")


if __name__ == "__main__":
    main()
