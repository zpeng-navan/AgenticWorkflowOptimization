"""
This module is used to query the New Relic logs. It also provides a function to write the logs to a file.
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

def query_new_relic(query, log_file=None, timeout=120, retry_num=3, retry_interval=5):
    """
    Simplified NerdGraph query approach - returns all items up to specified limit
    
    Args:
        query: NRQL query string
        log_file: File handle for logging (if None, uses print)
        timeout: Timeout in seconds
        retry_num: Maximum number of retries for request errors
        retry_interval: Interval between retries in seconds
    """
    import time
    
    def log_message(msg):
        if log_file:
            log_file.write(f"{msg}\n")
            log_file.flush()
        else:
            print(msg)
    
    if not new_relic_api_key:
        raise ValueError("NEW_RELIC_API_KEY not found in environment variables")
    
    # Get account ID from environment if not provided
    if not account_id:
        raise ValueError("NEW_RELIC_ACCOUNT_ID not found in environment variables")
    
    # NerdGraph endpoint
    url = "https://api.newrelic.com/graphql"
    
    # Add LIMIT clause to query if not already present
    nrql_query = query.strip()
    
    # Convert account ID to integer for GraphQL
    try:
        account_id_int = int(account_id)
    except ValueError:
        log_message(f"Error: Account ID '{account_id}' is not a valid integer")
        return None
    
    # GraphQL query for NerdGraph - using integer account ID
    graphql_query = {
        "query": """
        query($accountId: Int!, $nrql: Nrql!) {
          actor {
            account(id: $accountId) {
              nrql(query: $nrql) {
                results
              }
            }
          }
        }
        """,
        "variables": {
            "accountId": account_id_int,  # Use account ID as integer
            "nrql": nrql_query
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "API-Key": new_relic_api_key
    }
    
    # Retry logic for any errors
    for attempt in range(retry_num + 1):  # +1 because we want retry_num actual retries after first attempt
        try:
            response = requests.post(url, headers=headers, json=graphql_query, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            if "errors" in result:
                error_messages = []
                for error in result["errors"]:
                    error_msg = error.get('message', 'Unknown error')
                    error_messages.append(error_msg)
                
                if attempt < retry_num:  # Retry if we haven't exhausted retries
                    log_message(f"GraphQL errors on attempt {attempt + 1}:")
                    for error_msg in error_messages:
                        log_message(f"  - {error_msg}")
                    log_message(f"Retrying in {retry_interval} seconds... ({attempt + 1}/{retry_num})")
                    time.sleep(retry_interval)
                    continue  # Try again
                else:
                    log_message(f"GraphQL errors after {retry_num + 1} attempts:")
                    for error_msg in error_messages:
                        log_message(f"  - {error_msg}")
                    return None
            
            # Extract logs data
            logs_data = result.get("data", {}).get("actor", {}).get("account", {}).get("nrql", {})
            
            if not logs_data:
                if attempt < retry_num:  # Retry if we haven't exhausted retries
                    log_message(f"No data returned from query on attempt {attempt + 1}")
                    log_message(f"Retrying in {retry_interval} seconds... ({attempt + 1}/{retry_num})")
                    time.sleep(retry_interval)
                    continue  # Try again
                else:
                    log_message(f"No data returned from query after {retry_num + 1} attempts")
                    return None
            
            logs = logs_data.get("results", [])
            
            log_message(f"Retrieved {len(logs)} log entries")
            
            return logs
            
        except requests.exceptions.RequestException as e:
            if attempt < retry_num:  # Only retry if we haven't exhausted retries
                log_message(f"Request error on attempt {attempt + 1}: {e}")
                if hasattr(e, 'response') and e.response:
                    log_message(f"Response status: {e.response.status_code}")
                    log_message(f"Response text: {e.response.text}")
                log_message(f"Retrying in {retry_interval} seconds... ({attempt + 1}/{retry_num})")
                time.sleep(retry_interval)
            else:
                log_message(f"Request error after {retry_num + 1} attempts: {e}")
                if hasattr(e, 'response') and e.response:
                    log_message(f"Response status: {e.response.status_code}")
                    log_message(f"Response text: {e.response.text}")
                return None
        except Exception as e:
            if attempt < retry_num:  # Retry for any other exception
                log_message(f"Error on attempt {attempt + 1}: {e}")
                log_message(f"Retrying in {retry_interval} seconds... ({attempt + 1}/{retry_num})")
                time.sleep(retry_interval)
            else:
                log_message(f"Error after {retry_num + 1} attempts: {e}")
                return None
    
    return None  # Should never reach here

def query_new_relic_all_results(query, log_file=None, batch_size=1000, timeout=60, retry_num=3, retry_interval=5):
    """
    Query New Relic and return ALL results using pagination if necessary
    
    Args:
        query: NRQL query string (should NOT include LIMIT or OFFSET)
        log_file: File handle for logging (if None, uses print)
        batch_size: Number of results per batch (max 10000)
        timeout: Timeout in seconds
        retry_num: Maximum number of retries for request errors
        retry_interval: Interval between retries in seconds
    
    Returns:
        List of all log entries combined from multiple batches if needed
    """
    
    def log_message(msg):
        if log_file:
            log_file.write(f"{msg}\n")
            log_file.flush()
        else:
            print(msg)
    
    all_logs = []
    offset = 0
    total_retrieved = 0
    
    log_message(f"Querying New Relic for all results (batch size: {batch_size})...")
    
    while True:
        # Create query with LIMIT and OFFSET for this batch
        batch_query = f"{query.strip()} LIMIT {batch_size} OFFSET {offset}"
        
        log_message(f"Fetching batch starting at offset {offset}...")
        
        # Query this batch
        batch_logs = query_new_relic(
            batch_query,
            log_file=log_file,
            timeout=timeout,
            retry_num=retry_num,
            retry_interval=retry_interval
        )
        
        # Handle different return values from query_new_relic
        if batch_logs is None:
            # Query failed - return None to indicate failure
            log_message(f"Batch query failed at offset {offset}")
            return None
        
        if len(batch_logs) == 0:
            # Query succeeded but no results - end of data
            log_message(f"No more results found at offset {offset}")
            break
            
        batch_count = len(batch_logs)
        
        log_message(f"Retrieved {batch_count} results in this batch")
        
        all_logs.extend(batch_logs)
        total_retrieved += batch_count
        
        # If we got fewer results than batch_size, we've reached the end
        if batch_count < batch_size:
            log_message(f"Reached end of results (got {batch_count} < {batch_size})")
            break
            
        # Move to next batch
        offset += batch_size
        
        # Safety check to prevent infinite loops
        if total_retrieved > 1000000:  # 1M results max
            log_message(f"WARNING: Reached safety limit of 1M results, stopping")
            break
    
    log_message(f"Total results retrieved: {total_retrieved}")
    
    return all_logs

def save_logs_to_file(logs_result, filename="new_relic_logs.json"):
    """
    Save logs to a JSON file
    """
    if not logs_result:
        print("No logs to save")
        return
    
    try:
        with open(filename, 'w+', encoding='utf-8') as f:
            json.dump(logs_result, f, indent=4, default=str)
        print(f"Logs saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving logs: {e}")
        return None

def main():
    """
    Main function to query New Relic logs with the specified query
    """
    query = "SELECT timestamp, CHANNEL_ID, message FROM Log WHERE applicationName='ml-flow-svc' AND environment='prod' AND NODE_ID='vsat41bgk-lyyx97j8' AND message LIKE 'prompt:%' OR message LIKE 'completion:%' SINCE '2025-04-22 00:00:00' UNTIL '2025-04-23 00:00:00'"
    
    print("=" * 80)
    print("NEW RELIC LOGS QUERY")
    print("=" * 80)
    
    logs_result = None
    
    print(f"\n🔍 Trying main query: {query}")
    try:
        logs_result = query_new_relic_all_results(query, log_file=None, batch_size=1000, timeout=60)
        if logs_result:
            print(f"✅ successful with query {query}!")
        else:
            print(f"⚠️ returned no logs with query {query}")
    except Exception as e:
        print(f"❌ failed: {e}")
    
    if logs_result:
        print("\n✅ Query successful!")
        print(f"Successful query: {query}")
        
        # Save logs to file
        filename = save_logs_to_file(logs_result, join(log_dir, "ml_flow_logs.json"))
        if filename:
            print(f"  Saved to: {filename}")
        
        
    return logs_result




if __name__ == "__main__":
    main()









