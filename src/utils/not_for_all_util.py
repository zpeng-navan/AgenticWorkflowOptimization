#!/usr/bin/env python3
"""
Utility to convert chat_history_cancel_not_for_all_passengers=true.csv to JSON format.
"""

import csv
import json
import os
from pathlib import Path


def csv_to_json(csv_file_path, output_json_path):
    """
    Convert CSV file to JSON with the specified format.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_json_path (str): Path to the output JSON file
    """
    result = {}
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        # Use csv.reader to properly handle quoted fields with newlines
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            channel_sid = row.get('CHANNEL_SID', '').strip()
            if not channel_sid:
                continue
                
            ava_part_body = row.get('AVA_PART_BODY', '').strip()
            agent_part_body = row.get('AGENT_PART_BODY', '').strip()
            
            # Create the JSON structure for this channel
            result[channel_sid] = {
                "ava_part_body": ava_part_body,
                "agent_part_body": agent_part_body,
                "cancel_not_for_all": True,
                "partial_or_full": "null"
            }
    
    # Write the result to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(result, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(result)} records from CSV to JSON")
    print(f"Output saved to: {output_json_path}")
    
    return result


def main():
    """Main function to run the conversion."""
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    csv_file = base_dir / "data/raw/logs/04222025-08182025/ava_chat_history_api_raw_data/chat_history_cancel_not_for_all_passengers=true.csv"
    json_file = base_dir / "data/processed/logs/04222025-08182025/ava_chat_history_api_raw_data/chat_history_cancel_not_for_all_passengers=true.json"
    
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    # Convert CSV to JSON
    result = csv_to_json(str(csv_file), str(json_file))
    
    # Print sample of the result
    if result:
        sample_key = next(iter(result.keys()))
        print(f"\nSample output for channel {sample_key}:")
        print(json.dumps({sample_key: result[sample_key]}, indent=2))


if __name__ == "__main__":
    main()
