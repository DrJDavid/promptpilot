#!/usr/bin/env python
"""
Simple script to test Google Gemini API connection with a basic chat completion.
Just run this script to verify your API key works.
"""

import os
import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pathlib import Path
import re

def load_env_file():
    """Try to load API key from .env file if it exists"""
    env_path = Path('.env')
    if not env_path.exists():
        return False
    
    try:
        with open(env_path, 'r') as f:
            env_content = f.read()
            
        # Look for GEMINI_API_KEY
        match = re.search(r'GEMINI_API_KEY=([^\s]+)', env_content)
        if match:
            os.environ["GEMINI_API_KEY"] = match.group(1)
            return True
    except Exception as e:
        print(f"Error reading .env file: {e}")
    
    return False

def test_gemini():
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Try loading from .env file if not in environment
    if not api_key and load_env_file():
        api_key = os.environ.get("GEMINI_API_KEY")
        print("Loaded API key from .env file")
    
    if not api_key:
        print("❌ Error: No Gemini API key found in environment variables or .env file")
        print("Please set your API key with one of these methods:")
        print("  - In Git Bash: export GEMINI_API_KEY=your-key-here")
        print("  - In PowerShell: $env:GEMINI_API_KEY=\"your-key-here\"")
        print("  - In CMD: set GEMINI_API_KEY=your-key-here")
        print("  - In .env file: GEMINI_API_KEY=your-key-here")
        return
    
    print(f"Testing Gemini API connection with key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # List available models
        print("Fetching available models...")
        models = genai.list_models()
        gemini_models = [m for m in models if "gemini" in m.name.lower()]
        
        if not gemini_models:
            print("❌ No Gemini models available with your API key.")
            return

        print(f"✅ Found {len(gemini_models)} Gemini models:")
        for model in gemini_models:
            print(f"  - {model.name}")
        
        # Target model
        target_model_name = "gemini-2.0-flash-001"
        
        # Check if the target model is available
        target_model_available = any(m.name.endswith(target_model_name) for m in gemini_models)
        
        if target_model_available:
            model_name = target_model_name
            print(f"\nUsing requested model: {model_name}")
        else:
            # Fall back to another Gemini model if the targeted one isn't available
            fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
            for fallback in fallback_models:
                if any(fallback in m.name.lower() for m in gemini_models):
                    model_name = next(m.name for m in gemini_models if fallback in m.name.lower())
                    print(f"\n⚠️ Requested model '{target_model_name}' not available.")
                    print(f"Using fallback model: {model_name}")
                    break
            else:
                # If no fallbacks are available, use the first model in the list
                model_name = gemini_models[0].name
                print(f"\n⚠️ Requested model '{target_model_name}' not available.")
                print(f"Using available model: {model_name}")
        
        # Set up the model
        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Test with a simple prompt
        print("\nSending test request to Gemini API...")
        response = model.generate_content("Respond with 'Connection successful!' if you can read this.")
        
        print("\n✅ SUCCESS! Received response from Gemini:")
        print(f"Response: {response.text}")
        
        # Try starting a chat session
        print("\nYour Gemini API key is working! Let's try a quick chat:")
        
        # Initialize a chat session
        chat = model.start_chat(history=[])
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Ending chat. Goodbye!")
                break
                
            response = chat.send_message(user_input)
            print(f"\nGemini: {response.text}")
            print("\nType 'exit' to quit")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Connection failed. Check your API key and internet connection.")
        
if __name__ == "__main__":
    print("Simple Gemini API Connection Test")
    print("=================================")
    test_gemini() 