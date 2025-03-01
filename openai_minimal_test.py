#!/usr/bin/env python
"""
Minimal OpenAI API test script - uses only the API key
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

def test_openai_connection():
    # Load environment variables from .env
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No OpenAI API key found. Please check your .env file")
        return False
    
    print(f"Using API key starting with: {api_key[:8]}...")
    
    # Initialize client with just the API key
    client = OpenAI(api_key=api_key)
    
    # Try the simplest possible API call - listing models
    print("Testing API connection by listing models...")
    try:
        models = client.models.list()
        print(f"Success! Found {len(models.data)} models available.")
        print(f"First few models:")
        for i, model in enumerate(models.data[:5]):
            print(f"  - {model.id}")
        
        print("\nAPI connection is working correctly!")
        return True
    except Exception as e:
        print(f"Error connecting to OpenAI API: {str(e)}")
        print("\nTroubleshooting suggestions:")
        print("1. Check that your API key is correct (no extra spaces or characters)")
        print("2. Ensure you have sufficient API credits")
        print("3. Check your internet connection")
        print("4. If you're using a VPN, try turning it off")
        print("5. Wait a few minutes and try again (rate limits can temporarily block access)")
        return False

if __name__ == "__main__":
    test_openai_connection() 