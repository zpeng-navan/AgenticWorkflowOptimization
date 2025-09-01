"""
we assume that the conversation goes into exit after identify parital node should be labeled as:
cancel_not_for_all=True. 
To make the label more accurate, we need to verify the label.
"""

import os
import json
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from src.utils.data_util import load_json_file, save_to_json, extract_json_from_string
from tqdm import tqdm

# Load environment variables
load_dotenv()

def create_cancel_not_for_all_prompt(data_sections):
    """
    Create shared prompt template for cancel_not_for_all_passengers prediction.
    
    Args:
        data_sections (str): The data sections to include (flight booking legs, chat history, etc.)
    
    Returns:
        str: Formatted prompt template
    """
    return f"""Your name is Ava. You work for Navan, a company that builds a Corporate travel & expense management system. You are an assistant at the flight kiosk, where user wants to cancel their flight. Flight booking consists of legs. For example, if user has a one way flight, then there will be only one leg. If user books a round trip flight from New York to San Francisco, in that case such flight booking has 2 legs, first leg from New York to San Francisco, and second leg from San Francisco back to New York.

You are given the following pieces of information:
- legs of flight booking
- chat history between agent and user

### Your task ###
When analyzing chat history, check if the user explicitly mentions wanting to cancel the booking for fewer than all passengers in the reservation. Set the flag cancel_not_for_all_passengers to true if the cancellation is not for all passengers. Set the flag cancel_not_for_all_passengers to false if the cancellation applies to the entire passengers in booking.

{data_sections}

Please provide your response as a JSON object with the following format:
{{
  "reason": "Your detailed explanation here",
  "answer": true or false
}}

Make sure to return only valid JSON."""

def parse_llm_json_response(response_text):
    """
    Parse LLM response and extract the answer from JSON format.
    
    Args:
        response_text (str): The LLM response text
        
    Returns:
        bool or None: True/False if successfully parsed, None if failed
    """
    if not response_text:
        return None
    
    # Try to extract JSON from the response
    json_data = extract_json_from_string(response_text)
    
    if json_data and isinstance(json_data, dict):
        answer = json_data.get("answer")
        if isinstance(answer, bool):
            return answer
        # Handle string answers
        elif isinstance(answer, str):
            answer_lower = answer.lower().strip()
            if answer_lower == "true":
                return True
            elif answer_lower == "false":
                return False
    
    # Fallback: try direct JSON parsing if extract_json_from_string didn't work
    try:
        json_data = json.loads(response_text.strip())
        answer = json_data.get("answer")
        if isinstance(answer, bool):
            return answer
        elif isinstance(answer, str):
            answer_lower = answer.lower().strip()
            if answer_lower == "true":
                return True
            elif answer_lower == "false":
                return False
    except json.JSONDecodeError:
        pass
    
    return None

class OpenAIClient:
    """OpenAI client wrapper for LLM calls"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def get_completion(self, prompt, model="gpt-5", temperature=1.0, max_tokens=1000):
        """Get completion from OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

def verify_cancel_not_for_all_from_full_conversation(element, openai_client):
    """
    Based on the full conversation between user and agent, verify whether 
    cancel_not_for_all_passengers should be True or not using LLM.
    
    Args:
        element: Dictionary containing conversation data
        openai_client: OpenAI client instance
        
    Returns:
        bool: True if cancellation is not for all passengers
    """
    # Get the full conversation from both ava_part_body and agent_part_body
    ava_conversation = element.get("ava_part_body", "")
    agent_conversation = element.get("agent_part_body", "")
    
    # Combine conversations
    full_conversation = f"Ava Agent Conversation:\n{ava_conversation}\n\nHuman Agent Conversation:\n{agent_conversation}"
    
    # Get flight booking legs from Ava's data
    ava_data = element.get("Ava", {})
    flight_booking_legs = ava_data.get("flight_booking_legs", "")
    
    # Create data sections for the prompt (same format as second filter)
    data_sections = f"""### Flight booking legs ###
{flight_booking_legs}

### Chat history ###
{full_conversation}"""
    
    # Use shared prompt template
    prompt_template = create_cancel_not_for_all_prompt(data_sections)

    try:
        response = openai_client.get_completion(prompt_template, temperature=1.0)
        
        if response:
            answer = parse_llm_json_response(response)
            if answer is not None:
                return answer
                
        print(f"Warning: Could not parse LLM response for full conversation: {response}")
        return False
        
    except Exception as e:
        print(f"Error in full conversation verification: {e}")
        return False

def predict_cancel_not_for_all_with_llm(flight_booking_legs, chat_history, openai_client, num_calls=5):
    """
    Use LLM to predict cancel_not_for_all_passengers based on limited information
    with majority voting across multiple calls.
    
    Args:
        flight_booking_legs: Flight booking legs information
        chat_history: Chat history information  
        openai_client: OpenAI client instance
        num_calls: Number of LLM calls for majority voting (default: 5)
        
    Returns:
        bool: Final prediction based on majority voting
    """
    
    # Create data sections for the prompt
    data_sections = f"""### Flight booking legs ###
{flight_booking_legs}

### Chat history ###
{chat_history}"""
    
    # Use shared prompt template
    prompt_template = create_cancel_not_for_all_prompt(data_sections)

    predictions = []
    
    for i in range(num_calls):
        try:
            response = openai_client.get_completion(
                prompt_template, 
                temperature=1.0  # Small temperature for some variation
            )
            
            if response:
                answer = parse_llm_json_response(response)
                if answer is not None:
                    predictions.append(answer)
                else:
                    print(f"Warning: Could not parse JSON response in call {i+1}: {response}")
                    
        except Exception as e:
            print(f"Error in LLM call {i+1}: {e}")
    
    if not predictions:
        print("Warning: No valid predictions received from LLM")
        return False
    
    # Use majority voting
    prediction_counts = Counter(predictions)
    majority_prediction = prediction_counts.most_common(1)[0][0]
    
    print(f"LLM predictions: {predictions}")
    print(f"Majority prediction: {majority_prediction}")
    
    return majority_prediction

def process_ground_truth_verification():
    """
    Main function to process ground truth verification for cancel_not_for_all_passengers.
    """
    print("Starting cancel_not_for_all verification process...")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient()
    
    # Load ground truth data
    input_file = "data/processed/logs/04222025-08182025/ground_truth/ground_truth.json"
    output_file = "data/processed/logs/04222025-08182025/ground_truth/gpt-5-verified/verified_ground_truth.json"
    
    print(f"Loading data from {input_file}...")
    ground_truth_data = load_json_file(input_file)
    
    verified_data = {}
    to_be_verified_data = {}
    total_elements = len(ground_truth_data)
    processed_count = 0
    
    for element_id, element in tqdm(ground_truth_data.items()):
        processed_count += 1
        # print(f"\nProcessing element {processed_count}/{total_elements}: {element_id}")
        
        partial_or_full = element.get("partial_or_full")
        
        # Include elements where partial_or_full != "null" directly
        if partial_or_full != "null":
            verified_data[element_id] = element
            # print(f"✓ Added element (partial_or_full={partial_or_full})")
            continue
        else:
            to_be_verified_data[element_id] = element
    for element_id, element in tqdm(to_be_verified_data.items()):
        # For elements with partial_or_full = "null", do verification
        print("Verifying based on full conversation...")
        should_be_true = verify_cancel_not_for_all_from_full_conversation(element, openai_client)
        
        if not should_be_true:
            print("✗ Full conversation verification: cancel_not_for_all should be False, skipping")
            continue
            
        print("✓ Full conversation verification: cancel_not_for_all should be True")
        print("Running LLM prediction with majority voting...")
        
        # Get Ava's limited information
        ava_data = element.get("Ava", {})
        flight_booking_legs = ava_data.get("flight_booking_legs", "")
        chat_history = ava_data.get("chat_history", "")
        
        # Use LLM with majority voting
        llm_prediction = predict_cancel_not_for_all_with_llm(
            flight_booking_legs, 
            chat_history, 
            openai_client,
            num_calls=3
        )
        
        if llm_prediction:
            verified_data[element_id] = element
            print("✓ LLM prediction: True - Element qualified and added")
        else:
            print("✗ LLM prediction: False - Element not qualified, skipping")
    
    # Save verified data
    print(f"\nSaving {len(verified_data)} verified elements to {output_file}...")
    save_to_json(verified_data, output_file)
    
    print(f"\n🎉 Verification process complete!")
    print(f"Original to be verified elements: {len(to_be_verified_data)}")
    print(f"Dropped elements: {total_elements - len(verified_data)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    process_ground_truth_verification()