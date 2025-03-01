#!/usr/bin/env python
"""
Simple test script to verify OpenAI API connectivity.
Tests both GPT-4o text generation and text embeddings functionality.
"""

import os
import sys
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_api():
    """Test both GPT-4o text generation and embeddings functionality."""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found in environment variables.")
        logger.error("Please set the OPENAI_API_KEY environment variable in your .env file.")
        return False
    
    # Mask API key for display
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "********"
    logger.info(f"Using OpenAI API key (masked): {masked_key}")
    
    # Check key format
    is_project_key = api_key.startswith("sk-proj-")
    if is_project_key:
        logger.warning("You're using a project-scoped API key (starts with 'sk-proj-').")
        logger.warning("This type of key may cause issues with the standard OpenAI client.")
        logger.warning("Consider using a standard API key instead (starts with 'sk-' but not 'sk-proj-').")
    else:
        logger.info("Using standard API key format.")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
        logger.info("Successfully initialized OpenAI client.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False
    
    # Test 1: GPT-4o Text Generation
    logger.info("\n===== Testing GPT-4o Text Generation =====")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with 'OpenAI GPT-4o API is working!' if you can read this."}
            ],
            max_tokens=50
        )
        
        # Get and display the response
        completion = response.choices[0].message.content.strip()
        logger.info(f"GPT-4o Response: {completion}")
        logger.info("✅ GPT-4o test SUCCESSFUL!")
        
    except Exception as e:
        logger.error(f"❌ GPT-4o test FAILED: {e}")
        logger.error("Make sure your API key has access to GPT-4o.")
        text_generation_success = False
    else:
        text_generation_success = True
    
    # Test 2: Embeddings Generation
    logger.info("\n===== Testing Embeddings Generation =====")
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test of the OpenAI embeddings API."
        )
        
        # Get and display information about the embedding
        embedding = response.data[0].embedding
        embedding_length = len(embedding)
        logger.info(f"Generated embedding of dimension: {embedding_length}")
        logger.info(f"Sample of embedding (first 5 values): {embedding[:5]}...")
        logger.info("✅ Embeddings test SUCCESSFUL!")
        
    except Exception as e:
        logger.error(f"❌ Embeddings test FAILED: {e}")
        logger.error("Make sure your API key has access to the embeddings API.")
        embedding_success = False
    else:
        embedding_success = True
    
    # Summary
    logger.info("\n===== Test Summary =====")
    if text_generation_success and embedding_success:
        logger.info("✅ All tests PASSED! Your OpenAI API key is working correctly.")
        return True
    else:
        status = []
        if text_generation_success:
            status.append("GPT-4o: ✅")
        else:
            status.append("GPT-4o: ❌")
        
        if embedding_success:
            status.append("Embeddings: ✅")
        else:
            status.append("Embeddings: ❌")
        
        logger.warning(f"⚠️ Partial test results: {' | '.join(status)}")
        
        # Add helpful troubleshooting tips
        logger.info("\n===== Troubleshooting Tips =====")
        logger.info("1. Verify your API key is correct and active in your OpenAI account")
        logger.info("2. Check if your account has access to the models you're trying to use")
        logger.info("3. Ensure you have sufficient credits in your OpenAI account")
        logger.info("4. If using a project-scoped key (sk-proj-*), try a standard key instead")
        logger.info("5. Check OpenAI's status page for any ongoing service issues: https://status.openai.com/")
        
        return False
    
if __name__ == "__main__":
    print("\n" + "="*50)
    print("OpenAI API Connection Test")
    print("="*50)
    print("This script tests both GPT-4o and embeddings API functionality.")
    print("="*50 + "\n")
    
    success = test_openai_api()
    sys.exit(0 if success else 1) 