#!/usr/bin/env python
"""
Simple script to test OpenAI API connection with a basic chat completion.
Just run this script to verify your API key works.
"""

import os
from openai import OpenAI

def test_openai():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: No OpenAI API key found in environment variables")
        print("Please set your API key with: set OPENAI_API_KEY=your-key-here")
        return
    
    print(f"Testing OpenAI connection with key: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Make a simple chat completion call
        print("Sending test request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Respond with 'Connection successful!' if you can read this."}
            ],
            max_tokens=20
        )
        
        # Get the response
        result = response.choices[0].message.content
        print("\n✅ SUCCESS! Received response from OpenAI:")
        print(f"Response: {result}")
        
        # Try starting a simple chat
        print("\nYour OpenAI API key is working! Let's try a quick chat:")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Ending chat. Goodbye!")
                break
                
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            print(f"\nGPT-4o: {response.choices[0].message.content}")
            print("\nType 'exit' to quit")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Connection failed. Check your API key and internet connection.")

if __name__ == "__main__":
    print("Simple OpenAI API Connection Test")
    print("=================================")
    test_openai() 