"""
This module is used to process the logs from New Relic.
"""

import os
from os.path import join
from src.utils.data_util import load_json_file, write_to_file
import time
from datetime import datetime

import collections
# python load the .env file
from dotenv import load_dotenv
load_dotenv()

# get the data directory
cwd = os.getcwd()
data_dir = join(cwd, "data")

raw_log_dir = join(data_dir, "raw", "logs")
processed_log_dir = join(data_dir, "processed", "logs")

"""
# important keys in json file
CHANNEL_ID:e1b3b0ff-6c0b-4114-b76f-a6e58d19dd61
channelId:e1b3b0ff-6c0b-4114-b76f-a6e58d19dd61
NODE_ID:2omiktmpy-lyyxtjce
NODE_NAME:confirm leg
PARENT_NODE_ID:qlhdz9uxo-lxp7y31z
PARENT_NODE_NAME:flight_cancellation
"""

def load_conversation_from_json(file_path):
    """
    Load a conversation from a JSON file.
    """
    data = load_json_file(file_path)
    data.sort(key=lambda x: x["timestamp"])
    # all the messages with the same channelId belong to the same session
    # extract to this format:
    # {f"{CHANNEL_ID}": [{"prompt": "...", "response": "...", "time": "...", "node_id": "...", "node_name": "..."}, ...]}
    extracted_data = collections.defaultdict(list)
    for item in data:
        item_dict = {"message": item["message"], "timestamp": item["timestamp"], "node_id": item["NODE_ID"], "node_name": item["NODE_NAME"]}
        extracted_data[item["CHANNEL_ID"]].append(item_dict)
    for key, value in extracted_data.items():
        value.sort(key=lambda x: x["timestamp"])
    return extracted_data

def merge_conversations(conversations: list[dict]):
    """
    Merge conversations. conversations is a list of dicts, each dict is a conversation consists of a list of interactions.
    Each interaction is a dict with the following keys:
    - prompt: str
    - response: str
    - time: str
    - node_id: str
    - node_name: str
    """
    merged_data = collections.defaultdict(list)
    for conversation in conversations:
        for key, value in conversation.items():
            merged_data[key].extend(value)
    for key, value in merged_data.items():
        value.sort(key=lambda x: x["timestamp"])
    return merged_data

def conversation_to_str(conversation: list[dict]):
    """
    Convert a conversation to a string in format:
    node_name, time: message\n\n
    """
    output_str = ""
    for interaction in conversation:
        output_str += f"{interaction['node_name']}, {datetime.fromtimestamp(interaction['timestamp'] / 1000)}: {interaction['message']}\n\n"
    return output_str

def print_conversation_by_session_id(conversation_dict: dict, session_id: str):
    """
    Print a conversation by session id.
    """
    if not session_id:
        print("Session id is empty")
        # print the first one conversation
        for session_id, conversation in conversation_dict.items():
            print(conversation_to_str(conversation))
            break
        return
    if session_id not in conversation_dict:
        print(f"Session {session_id} not found")
    print(conversation_to_str(conversation_dict[session_id]))

def save_conversation_to_file(conversation: list[dict], file_path: str):
    """
    Save a conversation to a file.
    """
    content = conversation_to_str(conversation)
    write_to_file(file_path, content)

def find_session_ends_with_human_agent(log_data: list[dict]):
    """
    Find the session ends with human agent.
    """
    session_ids = set()
    for interaction in log_data:
        session_ids.add(interaction["CHANNEL_ID"])
    return session_ids

def main():
    """
    Main function.
    """
    date = "06012025-12:00AM-06142025-12:00AM"
    flight_dispatcher_log_file = join(raw_log_dir, date, f"{os.getenv('flight_dispatcher')}.JSON")
    identify_partial_log_file = join(raw_log_dir, date, f"{os.getenv('identify_partial')}.JSON")
    confirm_leg_log_file = join(raw_log_dir, date, f"{os.getenv('confirm_leg')}.JSON")
    # human agent log file
    agent_log_file = join(raw_log_dir, date, f"{os.getenv('agent')}.JSON")
    # load the json file
    flight_dispatcher_data = load_conversation_from_json(flight_dispatcher_log_file)
    identify_partial_data = load_conversation_from_json(identify_partial_log_file)
    confirm_leg_data = load_conversation_from_json(confirm_leg_log_file)
    merged_data = merge_conversations([flight_dispatcher_data, identify_partial_data, confirm_leg_data])
    agent_data = load_json_file(agent_log_file)
    # find the session ends with human agent
    session_ids = find_session_ends_with_human_agent(agent_data)
    # print the conversation by session id
    for session_id in session_ids:
        print(f"Session {session_id}:")
        print_conversation_by_session_id(merged_data, session_id)
        print("-" * 100)


if __name__ == "__main__":
    main()
