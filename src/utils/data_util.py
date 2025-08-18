"""
This module is used to process the data.
"""

import json
import re
import os
import pandas as pd

def load_json_file(file_path):
    """
    Load a JSON file and return the data.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def write_to_file(file_path, content):
    """
    Write content to a file.
    """
    with open(file_path, "w") as f:
        f.write(content)

def save_to_json(data, file_path, indent=2):
    """
    Save data to a JSON file, creating directories if they don't exist.
    
    Args:
        data: The data to save (dict, list, or any JSON-serializable object)
        file_path (str): The path where to save the JSON file
        indent (int): Number of spaces for JSON indentation (default: 2)
    
    Example:
        data = {"session_1": {"status": "completed"}, "session_2": {"status": "pending"}}
        save_to_json(data, "output/results.json")
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write JSON data to file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def extract_json_from_string(text, prefix=None):
    """
    Extract JSON content from a string that may have a prefix.
    
    Args:
        text (str): The input text containing JSON
        prefix (str, optional): The prefix to remove before parsing JSON.
                               If None, will try to auto-detect common prefixes.
    
    Returns:
        dict: Parsed JSON object, or None if no valid JSON found
    
    Example:
        text = 'completion: {"key": "value", "number": 42}'
        result = extract_json_from_string(text)
        # Returns: {"key": "value", "number": 42}
    """
    # If no prefix specified, try to auto-detect common prefixes
    if prefix is None:
        # Common prefixes found in logs
        common_prefixes = ['completion:', 'response:', 'prompt:', 'result:']
        for p in common_prefixes:
            if text.strip().startswith(p):
                prefix = p
                break
    
    # Remove the prefix if found
    if prefix:
        # Find the prefix and remove it
        prefix_index = text.find(prefix)
        if prefix_index != -1:
            text = text[prefix_index + len(prefix):].strip()
    
    # Try to find JSON content using regex
    # Look for content starting with { and ending with }
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try to clean up common issues
            # Remove any trailing non-JSON content
            try:
                # Find the last } and cut there
                last_brace = json_str.rfind('}')
                if last_brace != -1:
                    cleaned_json = json_str[:last_brace + 1]
                    return json.loads(cleaned_json)
            except json.JSONDecodeError:
                pass
    
    return None

def extract_completion_json(text):
    """
    Convenience function to extract JSON from completion strings.
    
    Args:
        text (str): Text starting with "completion: " followed by JSON
    
    Returns:
        dict: Parsed JSON object, or None if no valid JSON found
    """
    return extract_json_from_string(text, prefix="completion:")

def extract_section_content(text, section_headers):
    """
    Extract content between specified section headers from text.
    
    Args:
        text (str): The input text to parse
        section_headers (list): List of section header names to extract (without ### markers)
    
    Returns:
        dict: Dictionary mapping section header names to their content
    
    Example:
        text = '''
        ### Flight booking legs ###
        [{"index":0,"label":"AUS -> SFO"}]
        
        ### Chat history ###
        [user: Hello
        agent: Hi Aaron]
        '''
        
        result = extract_section_content(text, ["Flight booking legs", "Chat history"])
        # Returns: {
        #     "Flight booking legs": '[{"index":0,"label":"AUS -> SFO"}]',
        #     "Chat history": '[user: Hello\nagent: Hi Aaron]'
        # }
    """
    result = {}
    
    for header in section_headers:
        # Create pattern to match the section header
        pattern = rf"###\s*{re.escape(header)}\s*###\s*\n(.*?)(?=\n###|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            result[header] = content
        else:
            result[header] = None
    
    return result

def extract_flight_and_chat_sections(text):
    """
    Convenience function to extract Flight booking legs and Chat history sections.
    
    Args:
        text (str): The input text to parse
    
    Returns:
        tuple: (flight_booking_legs_content, chat_history_content)
    """
    sections = extract_section_content(text, ["Flight booking legs", "Chat history"])
    return sections.get("Flight booking legs"), sections.get("Chat history")

def load_csv_file(file_path):
    """
    Load a CSV file and return the data.
    """
    return pd.read_csv(file_path)

    
