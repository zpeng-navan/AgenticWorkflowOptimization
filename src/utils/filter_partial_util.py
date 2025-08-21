"""
This module is used to filter the partial label from the success log.
All the success log have label "partial_or_full" and "cancel_not_for_all_passengers=False" in the json file.
If the session has visited node "confirm leg" (2omiktmpy-lyyxtjce), then the label "partial_or_full" should be "PARTIAL", 
otherwise, the label "partial_or_full" should be "FULL".
"""

import os
from os.path import join
from src.utils.data_util import load_json_file, write_to_file, extract_section_content, extract_completion_json, save_to_json, load_csv_file
import time
from datetime import datetime
import ast
import collections
from collections import defaultdict
# python load the .env file
from dotenv import load_dotenv
load_dotenv()

# get the data directory
cwd = os.getcwd()
data_dir = join(cwd, "data")

raw_log_dir = join(data_dir, "raw", "logs")
processed_log_dir = join(data_dir, "processed", "logs")

raw_success_file = join(raw_log_dir, "04222025-08182025", "ava_chat_history_api_raw_data", "chat_history_success.csv")
success_file = join(processed_log_dir, "04222025-08182025", "ava_chat_history_api_raw_data", "chat_history_success.json")
os.makedirs(os.path.dirname(success_file), exist_ok=True)
data = load_csv_file(raw_success_file)

def is_partial(row):
    """row["VISITED_NODES_ARR"] is a list of dict, if any dict contains key-value pair of "node_id": 2omiktmpy-lyyxtjce
    then return True, otherwise return False

    Args:
        row (_type_): _description_
    """
    visited_nodes = ast.literal_eval(row["VISITED_NODES_ARR"])
    for node in visited_nodes:
        if node["node_id"] == "2omiktmpy-lyyxtjce":
            return True
    return False

partial_num = 0
label_data = defaultdict(list)
for index, row in data.iterrows():
    channel_sid = row["CHANNEL_SID"]
    label_data[channel_sid] = defaultdict()
    label_data[channel_sid]["ava_part_body"] = row["AVA_PART_BODY"]
    label_data[channel_sid]["cancel_not_for_all"] = False
    label_data[channel_sid]["partial_or_full"] = "FULL"
    if is_partial(row):
        label_data[channel_sid]["partial_or_full"] = "PARTIAL"
        partial_num += 1
    save_to_json(label_data, success_file)

print(f"partial_num: {partial_num}")
print(f"total_num: {index + 1}")
print("done")