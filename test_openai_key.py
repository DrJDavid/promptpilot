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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_api():
    """Test the OpenAI API key with both embeddings and a simple completion."""
    # Use the OpenAI API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("No OpenAI API key found in environment variables. Please set the OPENAI_API_KEY environment variable.")
        logger.error("You can get a key from https://platform.openai.com/api-keys")
        return False
    
    # Check if this is a project-scoped key
    is_project_key = api_key.startswith("sk-proj-")
    
    logger.info(f"Testing OpenAI API key (masked): {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    logger.info(f"API key format: {'Project-scoped key' if is_project_key else 'Standard key'}")
    
    # For project-scoped keys, try to extract project ID from the key
    project_id = None
    openai_kwargs = {}
    
    if is_project_key:
        logger.warning("Project-scoped keys may not work correctly with the current setup.")
        logger.warning("Consider using a standard API key (starting with 'sk-' but not 'sk-proj-').")
        
        # Try to use environment variable for project ID if available
        project_id = os.environ.get("OPENAI_PROJECT_ID")
        if project_id:
            logger.info(f"Using project ID from environment: {project_id}")
            openai_kwargs["project"] = project_id
        else:
            # Try to extract project ID if available in the key
            match = re.search(r'proj-([a-zA-Z0-9]+)', api_key)
            if match:
                project_id = match.group(1)
                logger.info(f"Detected project ID from key: {project_id}")
                openai_kwargs["project"] = project_id
                logger.warning("Extracted project ID may not be correct.")
                logger.warning("If authentication fails, try setting OPENAI_PROJECT_ID environment variable.")
            else:
                logger.warning("Could not extract project ID from key. Using default.")
    
    # Initialize the client with appropriate parameters
    try:
        client = OpenAI(api_key=api_key, **openai_kwargs)
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
        if is_project_key:
            logger.error("For project-scoped keys, make sure you have the correct project ID")
            logger.error("You can set the OPENAI_PROJECT_ID environment variable with your project ID")
            logger.error("Or consider using a standard API key instead (recommended)")
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
    print("2. Project-scoped keys start with 'sk-proj-' and may require additional configuration")
    print("\nIf using a project-scoped key, you may need to set the OPENAI_PROJECT_ID environment variable")
    print("="*50 + "\n")
    
    test_openai_api() 