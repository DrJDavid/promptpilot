#!/usr/bin/env python
"""
Utility script to check OpenAI client initialization and embedding cache.
This script loads the repository analyzer and verifies if it can initialize the OpenAI client correctly.
"""

import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
import shutil

# Import the repository analyzer class
from core.analyze import RepositoryAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_openai_client():
    """Check if OpenAI client can be initialized correctly."""
    print("\n=== Checking OpenAI Client Initialization ===\n")
    
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No OpenAI API key found in environment variables")
        return False
    
    print(f"✅ Found API key: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        client = OpenAI(api_key=api_key)
        print("✅ Successfully initialized OpenAI client")
        
        # Try a simple embedding call
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="Hello, world!"
            )
            
            embedding = response.data[0].embedding
            print(f"✅ Successfully generated test embedding ({len(embedding)} dimensions)")
            print(f"   First 5 values: {embedding[:5]}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate test embedding: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        return False

def check_embedding_cache():
    """Check the embedding cache files."""
    print("\n=== Checking Embedding Cache Files ===\n")
    
    # Check main project cache
    cache_dir = os.path.join(os.getcwd(), ".promptpilot")
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    # Check cache files
    cache_path = os.path.join(cache_dir, "embeddings_cache.json")
    metadata_path = os.path.join(cache_dir, "embeddings_cache_metadata.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"✅ Found embeddings cache with {len(cache)} entries")
            
            # Sample an entry
            if cache:
                key = next(iter(cache))
                embedding = cache[key]
                print(f"   Sample entry for '{key}':")
                print(f"   Embedding dimensions: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
        except Exception as e:
            print(f"❌ Failed to load embeddings cache: {e}")
    else:
        print(f"❌ Embeddings cache not found: {cache_path}")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"✅ Found embeddings metadata with {len(metadata)} entries")
            
            # Sample an entry
            if metadata:
                key = next(iter(metadata))
                content_hash = metadata[key]
                print(f"   Sample metadata for '{key}': {content_hash[:8]}...")
        except Exception as e:
            print(f"❌ Failed to load embeddings metadata: {e}")
    else:
        print(f"❌ Embeddings metadata not found: {metadata_path}")

def initialize_repository_analyzer():
    """Try to initialize the repository analyzer."""
    print("\n=== Testing Repository Analyzer Initialization ===\n")
    
    try:
        # Initialize analyzer
        cache_dir = os.path.join(os.getcwd(), ".promptpilot")
        analyzer = RepositoryAnalyzer(cache_dir)
        
        print(f"✅ Successfully initialized repository analyzer")
        print(f"   OpenAI available: {analyzer.openai_available}")
        
        if analyzer.openai_available:
            print(f"   Embedding model: {analyzer.embedding_model}")
            print(f"   Client initialized: {analyzer.client is not None}")
            print(f"   Async client initialized: {analyzer.async_client is not None}")
        
        return analyzer
    except Exception as e:
        print(f"❌ Failed to initialize repository analyzer: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*50)
    print("OpenAI Client and Embedding Cache Check")
    print("="*50)
    
    # Check OpenAI client
    client_ok = check_openai_client()
    
    # Check embedding cache
    check_embedding_cache()
    
    # Initialize repository analyzer
    analyzer = initialize_repository_analyzer()
    
    print("\n" + "="*50)
    print("Check Complete")
    print("="*50)
    
    if client_ok and analyzer and analyzer.openai_available:
        print("✅ OpenAI client is working correctly")
        print("✅ Repository analyzer initialized successfully")
        print("The embedding functionality should now work correctly")
    else:
        print("❌ There are still issues with the OpenAI client or repository analyzer")
        print("Please check the logs above for details")
    
    print("="*50 + "\n") 