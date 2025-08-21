#!/usr/bin/env python3
"""
Ground truth utility to combine chat history and IP conversation data.

This utility combines three JSON files:
1. chat_history_success.json
2. chat_history_cancel_not_for_all_passengers=true.json  
3. ip_conversation_2025-04-22_2025-08-18.json

And creates a unified ground truth dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def combine_chat_history_files(success_file: str, cancel_not_for_all_file: str) -> Dict[str, Any]:
    """
    Combine chat history files, checking for overlaps.
    
    Args:
        success_file: Path to chat_history_success.json
        cancel_not_for_all_file: Path to chat_history_cancel_not_for_all_passengers=true.json
        
    Returns:
        Combined dictionary with overlap handling
    """
    print("📁 Loading chat history files...")
    
    success_data = load_json_file(success_file)
    cancel_not_for_all_data = load_json_file(cancel_not_for_all_file)
    
    print(f"✅ Loaded {len(success_data)} records from chat_history_success.json")
    print(f"✅ Loaded {len(cancel_not_for_all_data)} records from chat_history_cancel_not_for_all_passengers=true.json")
    
    # Check for overlapping channel_ids
    success_channels = set(success_data.keys())
    cancel_not_for_all_channels = set(cancel_not_for_all_data.keys())
    overlapping_channels = success_channels.intersection(cancel_not_for_all_channels)
    
    if overlapping_channels:
        print(f"⚠️  Found {len(overlapping_channels)} overlapping channel_ids:")
        for channel_id in sorted(overlapping_channels):
            print(f"   - {channel_id}")
        
        # Drop overlapping channels from success_data
        for channel_id in overlapping_channels:
            del success_data[channel_id]
        
        print(f"🗑️  Dropped {len(overlapping_channels)} overlapping channel_ids from chat_history_success.json")
    else:
        print("✅ No overlapping channel_ids found")
    
    # Combine the dictionaries
    combined_data = {**success_data, **cancel_not_for_all_data}
    
    print(f"📊 Combined total: {len(combined_data)} unique channel_ids")
    
    return combined_data


def merge_with_ip_conversation(ground_truth_data: Dict[str, Any], ip_conversation_file: str) -> Dict[str, Any]:
    """
    Merge ground truth data with IP conversation data.
    
    Args:
        ground_truth_data: Combined chat history data
        ip_conversation_file: Path to IP conversation JSON file
        
    Returns:
        Updated ground truth data with Ava field added
    """
    print("\n🔄 Processing IP conversation data...")
    
    ip_conversation_data = load_json_file(ip_conversation_file)
    print(f"✅ Loaded {len(ip_conversation_data)} channel_ids from IP conversation data")
    
    channels_to_remove = []
    channels_processed = 0
    
    for channel_id in list(ground_truth_data.keys()):
        if channel_id not in ip_conversation_data:
            # Channel not found in IP conversation data
            channels_to_remove.append(channel_id)
        elif len(ip_conversation_data[channel_id]) != 1:
            # Channel has more than one conversation pair
            channels_to_remove.append(channel_id)
        else:
            # Channel has exactly one conversation pair - add Ava data
            ground_truth_data[channel_id]["Ava"] = ip_conversation_data[channel_id][0]
            channels_processed += 1
    
    # Remove channels that don't meet criteria
    for channel_id in channels_to_remove:
        del ground_truth_data[channel_id]
    
    print(f"✅ Successfully processed {channels_processed} channels")
    print(f"🗑️  Removed {len(channels_to_remove)} channels that didn't meet criteria:")
    
    # Count reasons for removal
    not_found_count = 0
    multiple_pairs_count = 0
    
    for channel_id in channels_to_remove:
        if channel_id not in ip_conversation_data:
            not_found_count += 1
        else:
            multiple_pairs_count += 1
    
    print(f"   - {not_found_count} channels not found in IP conversation data")
    print(f"   - {multiple_pairs_count} channels with multiple conversation pairs")
    
    return ground_truth_data


def analyze_ground_truth_statistics(ground_truth_data: Dict[str, Any]) -> None:
    """
    Analyze and display statistics for the ground truth data.
    
    Args:
        ground_truth_data: Final ground truth dataset
    """
    print(f"\n📊 Ground Truth Dataset Statistics")
    print("=" * 50)
    
    print(f"Total records in ground truth: {len(ground_truth_data)}")
    
    # Analyze partial_or_full distribution from main level
    partial_or_full_counts = Counter()
    cancel_not_for_all_counts = Counter()
    
    for channel_id, data in ground_truth_data.items():
        # Get partial_or_full from main data level (convert to lowercase)
        partial_or_full = str(data["partial_or_full"]).lower()
        partial_or_full_counts[partial_or_full] += 1
        
        # Get cancel_not_for_all from main data level (convert to lowercase)
        # Note: The main level has "cancel_not_for_all" not "cancel_not_for_all_passengers"
        cancel_not_for_all = str(data["cancel_not_for_all"]).lower()
        cancel_not_for_all_counts[cancel_not_for_all] += 1
    
    print(f"\n📈 partial_or_full Distribution (main level):")
    print("-" * 40)
    for value, count in sorted(partial_or_full_counts.items()):
        percentage = (count / len(ground_truth_data)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📈 cancel_not_for_all Distribution (main level):")
    print("-" * 45)
    for value, count in sorted(cancel_not_for_all_counts.items()):
        percentage = (count / len(ground_truth_data)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")


def create_ground_truth(
    success_file: str,
    cancel_not_for_all_file: str, 
    ip_conversation_file: str,
    output_file: str
) -> Dict[str, Any]:
    """
    Main function to create ground truth dataset.
    
    Args:
        success_file: Path to chat_history_success.json
        cancel_not_for_all_file: Path to chat_history_cancel_not_for_all_passengers=true.json
        ip_conversation_file: Path to IP conversation JSON file
        output_file: Path for output ground truth JSON file
        
    Returns:
        Final ground truth dataset
    """
    print("🚀 Creating Ground Truth Dataset")
    print("=" * 50)
    
    # Step 1: Combine chat history files
    ground_truth_data = combine_chat_history_files(success_file, cancel_not_for_all_file)
    
    # Step 2: Merge with IP conversation data
    ground_truth_data = merge_with_ip_conversation(ground_truth_data, ip_conversation_file)
    
    # Step 3: Save ground truth dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Ground truth dataset saved to: {output_file}")
    
    # Step 4: Analyze statistics
    analyze_ground_truth_statistics(ground_truth_data)
    
    return ground_truth_data


def main():
    """Main function to run the ground truth creation."""
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    
    success_file = base_dir / "data/processed/logs/04222025-08182025/ava_chat_history_api_raw_data/chat_history_success.json"
    cancel_not_for_all_file = base_dir / "data/processed/logs/04222025-08182025/ava_chat_history_api_raw_data/chat_history_cancel_not_for_all_passengers=true.json"
    ip_conversation_file = base_dir / "data/processed/logs/04222025-08182025/ip_conversation/ip_conversation_2025-04-22_2025-08-18.json"
    output_file = base_dir / "data/processed/logs/04222025-08182025/ground_truth/ground_truth.json"
    
    # Create ground truth dataset
    result = create_ground_truth(
        str(success_file),
        str(cancel_not_for_all_file),
        str(ip_conversation_file),
        str(output_file)
    )
    
    return result


if __name__ == "__main__":
    main()
