"""
This file implements the functions for calling openai api
"""

import hashlib
import numpy as np
import pandas as pd
import re
import time
from typing import Dict, List, Tuple, Union, Optional
import json
import os

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will fall back to system environment variables
    pass

# Import OpenAI with compatibility for both v0.x and v1.x
try:
    # Try new v1.x API first
    from openai import OpenAI
    OPENAI_V1 = True
except ImportError:
    # Fall back to old v0.x API
    import openai
    OPENAI_V1 = False

class OpenAIClient:
    """Handles OpenAI API calls with retry logic."""
    
    def __init__(self, api_key: str):
        """Initialize OpenAI client with API key."""
        self.api_key = api_key
        if OPENAI_V1:
            self.client = OpenAI(api_key=api_key)
        else:
            import openai
            openai.api_key = api_key
    
    def call_openai_with_retry(
        self,
        prompt: str, 
        model: str = "gpt-3.5-turbo", 
        temperature: float = 0,  # Changed from 0.8 to 0 for determinism
        max_tokens: int = 512,
        max_retries: int = 5,
        seed: Optional[int] = 42  # Added seed for reproducibility
    ) -> str:
        """
        Call OpenAI API with exponential backoff retry.
        
        Args:
            prompt: The input prompt text
            model: OpenAI model to use
            temperature: Controls randomness (0=deterministic, 1=random)  
            max_tokens: Maximum response length
            max_retries: Number of retry attempts
            seed: Random seed for reproducible results (v1.x API only)
        
        Returns:
            Generated text response
        """
        for attempt in range(max_retries):
            try:
                if OPENAI_V1:
                    # Use new v1.x API
                    params = {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    # Add seed if provided (for reproducibility)
                    if seed is not None:
                        params["seed"] = seed
                    
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message.content
                else:
                    # Use old v0.x API (seed not supported in older versions)
                    import openai
                    response = openai.ChatCompletion.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                # Handle both v0.x and v1.x error types
                error_name = type(e).__name__
                if any(error_type in error_name.lower() for error_type in ['ratelimit', 'timeout', 'api']):
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
                else:
                    raise e