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

def query_new_relic(query, timeout=120):
    """
    Simplified NerdGraph query approach - returns all items up to specified limit
    
    Args:
        query: NRQL query string
        timeout: Timeout in seconds
    """
    
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
        print(f"Error: Account ID '{account_id}' is not a valid integer")
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
    
    try:
        response = requests.post(url, headers=headers, json=graphql_query, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if "errors" in result:
            print("GraphQL errors:")
            for error in result["errors"]:
                print(f"  - {error.get('message', 'Unknown error')}")
            return None
        
        # Extract logs data
        logs_data = result.get("data", {}).get("actor", {}).get("account", {}).get("nrql", {})
        
        if not logs_data:
            print("No data returned from query")
            return None
        
        logs = logs_data.get("results", [])
        
        print(f"Retrieved {len(logs)} log entries")
        
        return logs
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"Error querying logs: {e}")
        return None

def query_new_relic_all_results(query, batch_size=1000, timeout=60):
    """
    Query New Relic and return ALL results using pagination if necessary
    
    Args:
        query: NRQL query string (should NOT include LIMIT or OFFSET)
        batch_size: Number of results per batch (max 10000)
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with all results combined from multiple batches if needed
    """
    
    all_logs = []
    offset = 0
    total_retrieved = 0
    
    print(f"Querying New Relic for all results (batch size: {batch_size})...")
    
    while True:
        # Create query with LIMIT and OFFSET for this batch
        batch_query = f"{query.strip()} LIMIT {batch_size} OFFSET {offset}"
        
        print(f"Fetching batch starting at offset {offset}...")
        
        # Query this batch
        batch_logs = query_new_relic(
            batch_query, 
            timeout=timeout
        )
        
        if not batch_logs:
            print(f"No more results found at offset {offset}")
            break
            
        batch_count = len(batch_logs)
        
        print(f"Retrieved {batch_count} results in this batch")
        
        all_logs.extend(batch_logs)
        total_retrieved += batch_count
        
        # If we got fewer results than batch_size, we've reached the end
        if batch_count < batch_size:
            print(f"Reached end of results (got {batch_count} < {batch_size})")
            break
            
        # Move to next batch
        offset += batch_size
        
        # Safety check to prevent infinite loops
        if total_retrieved > 1000000:  # 1M results max
            print(f"WARNING: Reached safety limit of 1M results, stopping")
            break
    
    print(f"Total results retrieved: {total_retrieved}")
    
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
    query = "SELECT timestamp, CHANNEL_ID, message FROM Log WHERE applicationName='ml-flow-svc' AND environment='prod' AND NODE_ID='vsat41bgk-lyyx97j8' AND message LIKE 'prompt:%' OR message LIKE 'completion:%' SINCE '2025-04-21 00:00:00' UNTIL '2025-04-22 00:00:00'"
    
    print("=" * 80)
    print("NEW RELIC LOGS QUERY")
    print("=" * 80)
    
    logs_result = None
    
    print(f"\n🔍 Trying main query: {query}")
    try:
        logs_result = query_new_relic_all_results(query, batch_size=1000, timeout=60)
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









