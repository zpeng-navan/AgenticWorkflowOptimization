"""
This module is used to process the data.
"""

import json

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