#!/usr/bin/env python
"""
Simple script to test if the OpenAI API key is working properly.
This checks both embeddings and completions to ensure the key is valid.
"""

import os
import json
from openai import OpenAI
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_api():
    """Test the OpenAI API key with both embeddings and a simple completion."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Use the OpenAI API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("No OpenAI API key found in environment variables or .env file.")
        logger.error("Please set the OPENAI_API_KEY environment variable or add it to your .env file.")
        logger.error("You can get a key from https://platform.openai.com/api-keys")
        return False
    
    # Check key format (just informational)
    is_project_key = api_key.startswith("sk-proj-")
    
    logger.info(f"Testing OpenAI API key (masked): {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    logger.info(f"API key format: {'Project-scoped key' if is_project_key else 'Standard key'}")
    
    # Initialize the client with just the API key (simplest approach)
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False
    
    # Check API key validity by attempting to list models
    try:
        logger.info("Testing API key validity...")
        models = client.models.list()
        logger.info(f"✅ API key is valid. Available models: {len(models.data)}")
        logger.info(f"First model: {models.data[0].id}")
    except Exception as e:
        logger.error(f"❌ API key validation failed: {e}")
        logger.error("Your API key appears to be invalid or has insufficient permissions")
        return False
    
    # Test embeddings
    try:
        logger.info("Testing embeddings...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test of the OpenAI API key for embeddings."
        )
        embedding = response.data[0].embedding
        embedding_length = len(embedding)
        logger.info(f"✅ Embeddings test successful! Generated embedding of dimension: {embedding_length}")
        
        # Print a small sample of the embedding
        logger.info(f"Sample of embedding: {embedding[:5]}...")
    except Exception as e:
        logger.error(f"❌ Embeddings test failed: {e}")
        return False
    
    # Test completions
    try:
        logger.info("Testing completions...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'The OpenAI API key is working!' if you can read this."}
            ]
        )
        completion = response.choices[0].message.content
        logger.info(f"✅ Completions test successful! Response: {completion}")
    except Exception as e:
        logger.error(f"❌ Completions test failed: {e}")
        return False
    
    logger.info("✅ All tests passed! Your OpenAI API key is working correctly.")
    return True

if __name__ == "__main__":
    print("="*50)
    print("OpenAI API Key Tester")
    print("="*50)
    print("This script tests your OpenAI API key configuration.")
    print("You should see your API key in the .env file and ensure it's properly configured.")
    print("\nAPI Key Guidelines:")
    print("1. Standard API keys start with 'sk-' (recommended)")
    print("2. Project-scoped keys start with 'sk-proj-' but should work with minimal configuration")
    print("="*50 + "\n")
    
    test_openai_api() 