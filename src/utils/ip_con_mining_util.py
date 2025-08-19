"""
identify partial conversation mining

This file is used to mine the identify partial conversation (120 days from current date) from the New Relic logs and save each day's log to a json file.
"""
import os
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from os.path import join
cwd = os.getcwd()

load_dotenv()
new_relic_api_key = os.getenv("NEW_RELIC_API_KEY")
account_id = os.getenv("NEW_RELIC_ACCOUNT_ID")
log_dir = join(cwd, "data", "raw", "logs", "identify_partial_conversation")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

from src.utils.relic_query_util import query_new_relic_all_results

def daterange(start_date, end_date):
    """Yield each date from start_date to end_date (exclusive of end_date)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

from tqdm import tqdm

import time

def mine_ip_conversations(
    start_date_str="2025-04-21 00:00:00",
    end_date_str="2025-04-25 00:00:00",
):
    """
    Mine identify partial conversations from New Relic logs for each day in the range.
    Logs progress and results to a log file in data/raw/logs/identify_partial_conversation.
    Query interval is set to 5 seconds. If the log is None, wait 10 seconds and retry, up to 10 times.
    
    Args:
        start_date_str: Start date string in format "YYYY-MM-DD HH:MM:SS"
        end_date_str: End date string in format "YYYY-MM-DD HH:MM:SS"  
    """
    # Parse date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

    log_file_path = os.path.join(log_dir, "mining_progress.log")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        date_list = list(daterange(start_date, end_date))
        for single_date in tqdm(date_list, desc="Mining IP Conversations"):
            
            # Split each day into 12-hour chunks: 00-12, 12-24
            chunks = [
                (0, 12),   # 00:00 - 12:00
                (12, 24)  # 12:00 - 24:00
            ]
            
            all_day_results = []
            
            for start_hour, end_hour in chunks:
                chunk_start = single_date.replace(hour=start_hour, minute=0, second=0)
                
                # Handle the case where end_hour is 24 (next day 00:00)
                if end_hour == 24:
                    chunk_end = single_date + timedelta(days=1)
                    chunk_end = chunk_end.replace(hour=0, minute=0, second=0)
                else:
                    chunk_end = single_date.replace(hour=end_hour, minute=0, second=0)
                
                chunk_start_str = chunk_start.strftime("%Y-%m-%d %H:%M:%S")
                chunk_end_str = chunk_end.strftime("%Y-%m-%d %H:%M:%S")
                
                query = (
                    "SELECT timestamp, CHANNEL_ID, message FROM Log "
                    "WHERE applicationName='ml-flow-svc' AND environment='prod' "
                    "AND NODE_ID='vsat41bgk-lyyx97j8' "
                    "AND (message LIKE 'prompt:%' OR message LIKE 'completion:%') "
                    f"SINCE '{chunk_start_str}' UNTIL '{chunk_end_str}' "
                    "ORDER BY timestamp ASC"
                )
                
                log_file.write(f"Querying logs for chunk {chunk_start_str} to {chunk_end_str} ...\n")
                log_file.flush()

                chunk_results = None
                attempt = 0
                while attempt < 10:
                    chunk_results = query_new_relic_all_results(query, batch_size=1000, timeout=300)
                    if chunk_results is not None:
                        break
                    attempt += 1
                    log_file.write(f"  Chunk {start_hour:02d}-{end_hour:02d} Attempt {attempt}: No logs returned, waiting 10 seconds before retry...\n")
                    log_file.flush()
                    time.sleep(10)
                
                if chunk_results:
                    # chunk_results is already a list of logs, not a dict
                    all_day_results.extend(chunk_results)
                    log_file.write(f"  Chunk {start_hour:02d}-{end_hour:02d}: Retrieved {len(chunk_results)} logs\n")
                else:
                    log_file.write(f"  Chunk {start_hour:02d}-{end_hour:02d}: Failed to retrieve logs after 10 attempts\n")
                
                log_file.flush()
                time.sleep(2)  # Short wait between chunks
            
            # Save all results for the day
            log_file.write(f"Total logs for {single_date.strftime('%Y-%m-%d')}: {len(all_day_results)}\n")
            
            # Save results to file
            file_name = f"ip_conversation_{single_date.strftime('%Y-%m-%d')}.json"
            file_path = os.path.join(log_dir, file_name)
            save_to_json(all_day_results, file_path)
            log_file.write(f"Saved {len(all_day_results) if all_day_results else 0} logs to {file_path}\n")
            log_file.flush()
            time.sleep(5)  # Wait 5 seconds between days

if __name__ == "__main__":
    mine_ip_conversations()
