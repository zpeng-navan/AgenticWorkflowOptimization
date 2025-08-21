#!/usr/bin/env python3
"""
Utility to combine and process IP conversation files from data/raw/logs/04222025-08182025/ip_conversation
into a single structured JSON file.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def extract_flight_legs_and_chat_history(prompt_message: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract flight booking legs and chat history from prompt message.
    
    Args:
        prompt_message: The prompt message starting with 'prompt:'
        
    Returns:
        Tuple of (flight_booking_legs, chat_history) or (None, None) if not found
    """
    try:
        # Remove 'prompt:' prefix and get the content
        content = prompt_message[7:].strip() if prompt_message.startswith('prompt:') else prompt_message
        
        # Extract flight booking legs
        legs_pattern = r'### Flight booking legs ###\s*(\[.*?\])'
        legs_match = re.search(legs_pattern, content, re.DOTALL)
        flight_booking_legs = legs_match.group(1).strip() if legs_match else None
        
        # Extract chat history
        chat_pattern = r'### Chat history ###\s*(\[.*?\])'
        chat_match = re.search(chat_pattern, content, re.DOTALL)
        chat_history = chat_match.group(1).strip() if chat_match else None
        
        return flight_booking_legs, chat_history
        
    except Exception as e:
        print(f"Error extracting from prompt: {e}")
        return None, None


def parse_completion_response(completion_message: str) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    """
    Parse the completion message to extract response fields.
    
    Args:
        completion_message: The completion message starting with 'completion:'
        
    Returns:
        Tuple of (partial_or_full, cancel_not_for_all_passengers, thought) or (None, None, None) if parsing fails
    """
    try:
        # Remove 'completion:' prefix and get the JSON content
        json_content = completion_message[11:].strip() if completion_message.startswith('completion:') else completion_message
        
        # Parse the JSON response
        response_data = json.loads(json_content)
        
        partial_or_full = response_data.get('partial_or_full')
        cancel_not_for_all_passengers = response_data.get('cancel_not_for_all_passengers')
        thought = response_data.get('thought')
        
        return partial_or_full, cancel_not_for_all_passengers, thought
        
    except Exception as e:
        print(f"Error parsing completion JSON: {e}")
        return None, None, None


def load_ip_conversation_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load all ip_conversation JSON files from the input directory.
    
    Args:
        input_dir: Directory containing ip_conversation JSON files
        
    Returns:
        List of all records from all files
    """
    all_records = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return all_records
    
    # Get all JSON files with ip_conversation prefix
    json_files = sorted([f for f in input_path.iterdir() if f.name.startswith('ip_conversation_') and f.name.endswith('.json')])
    
    print(f"Found {len(json_files)} ip_conversation files to process")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                    print(f"Loaded {len(data)} records from {json_file.name}")
                else:
                    print(f"Warning: {json_file.name} does not contain a list")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"Total records loaded: {len(all_records)}")
    return all_records


def group_and_pair_messages(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by CHANNEL_ID and pair consecutive prompt/completion messages.
    
    Args:
        records: List of all records from all files
        
    Returns:
        Dictionary with CHANNEL_ID as key and list of paired message dicts as value
    """
    # Group by CHANNEL_ID
    grouped = {}
    for record in records:
        channel_id = record.get('CHANNEL_ID')
        if channel_id:
            if channel_id not in grouped:
                grouped[channel_id] = []
            grouped[channel_id].append(record)
    
    print(f"Grouped records into {len(grouped)} unique CHANNEL_IDs")
    
    # Sort each group by timestamp and pair messages
    result = {}
    total_pairs = 0
    
    for channel_id, channel_records in grouped.items():
        # Sort by timestamp
        channel_records.sort(key=lambda x: x.get('timestamp', 0))
        
        paired_messages = []
        i = 0
        
        while i < len(channel_records) - 1:
            current_record = channel_records[i]
            next_record = channel_records[i + 1]
            
            current_message = current_record.get('message', '')
            next_message = next_record.get('message', '')
            
            # Check if current is prompt and next is completion
            if current_message.startswith('prompt:') and next_message.startswith('completion:'):
                # Extract data from prompt
                flight_legs, chat_history = extract_flight_legs_and_chat_history(current_message)
                
                # Parse completion response
                partial_or_full, cancel_not_for_all_passengers, thought = parse_completion_response(next_message)
                
                # Only add if we successfully extracted all required data
                if all(x is not None for x in [flight_legs, chat_history, partial_or_full, cancel_not_for_all_passengers, thought]):
                    paired_dict = {
                        "flight_booking_legs": flight_legs,
                        "chat_history": chat_history,
                        "partial_or_full": partial_or_full,
                        "cancel_not_for_all_passengers": cancel_not_for_all_passengers,
                        "thought": thought,
                        "prompt_timestamp": current_record.get('timestamp'),
                        "completion_timestamp": next_record.get('timestamp')
                    }
                    paired_messages.append(paired_dict)
                    total_pairs += 1
                
                # Skip the next record since we've processed it
                i += 2
            else:
                # Move to next record
                i += 1
        
        # Only add channel if it has valid pairs
        if paired_messages:
            # Sort by prompt_timestamp to ensure chronological order
            paired_messages.sort(key=lambda x: x.get('prompt_timestamp', 0))
            result[channel_id] = paired_messages
    
    print(f"Created {total_pairs} valid prompt-completion pairs across {len(result)} CHANNEL_IDs")
    return result


def combine_ip_conversations(input_dir: str, output_file: str) -> Dict[str, Any]:
    """
    Main function to combine all ip_conversation files into a single structured JSON.
    
    Args:
        input_dir: Directory containing ip_conversation JSON files
        output_file: Output JSON file path
        
    Returns:
        Dictionary with combined and structured data
    """
    # Load all records from JSON files
    all_records = load_ip_conversation_files(input_dir)
    
    if not all_records:
        print("No records found to process")
        return {}
    
    # Group and pair messages
    structured_data = group_and_pair_messages(all_records)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    print(f"Combined data saved to: {output_file}")
    print(f"Final structure contains {len(structured_data)} CHANNEL_IDs with valid pairs")
    
    # Print sample statistics
    total_pairs = sum(len(pairs) for pairs in structured_data.values())
    print(f"Total prompt-completion pairs: {total_pairs}")
    
    if structured_data:
        sample_channel = next(iter(structured_data.keys()))
        sample_data = structured_data[sample_channel][0] if structured_data[sample_channel] else {}
        print(f"\nSample data structure for CHANNEL_ID {sample_channel}:")
        print(json.dumps({sample_channel: [sample_data]}, indent=2))
    
    return structured_data


def main():
    """Main function to run the IP conversation combination."""
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    input_dir = base_dir / "data/raw/logs/04222025-08182025/ip_conversation"
    output_file = base_dir / "data/processed/logs/04222025-08182025/ip_conversation/ip_conversation_2025-04-22_2025-08-18.json"
    
    # Combine IP conversations
    result = combine_ip_conversations(str(input_dir), str(output_file))
    
    return result


if __name__ == "__main__":
    main()
