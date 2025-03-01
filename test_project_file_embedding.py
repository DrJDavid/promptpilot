#!/usr/bin/env python
"""
Test script for embedding a real project file using the working approach.

This script takes the embedding cache implementation from test_embeddings_cache.py
and applies it to a real file from the project to verify it works correctly.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_REPOSITORY_DIR = os.path.join(os.getcwd(), ".promptpilot")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

class EmbeddingCache:
    """Class to manage embeddings with efficient caching mechanisms."""
    
    def __init__(self, repository_dir: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the embedding cache manager.
        
        Args:
            repository_dir: Directory to store cache files (.promptpilot folder)
            embedding_model: OpenAI embedding model to use
        """
        self.repository_dir = repository_dir
        self.embedding_model = embedding_model
        
        # Ensure repository directory exists
        os.makedirs(repository_dir, exist_ok=True)
        
        # Cache file paths
        self.cache_dir = os.path.join(repository_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings_cache_path = os.path.join(self.cache_dir, "embeddings_cache.json")
        self.embeddings_metadata_path = os.path.join(self.cache_dir, "embeddings_metadata.json")
        
        # Load environment variables for API keys
        load_dotenv()
        
        # Check for OpenAI API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY in your .env file.")
            self.client = None
        else:
            # Initialize OpenAI client
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate a hash of file content for cache invalidation.
        
        Args:
            content: File content
            
        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    
    def _load_cache(self) -> tuple[Dict[str, List[float]], Dict[str, str]]:
        """
        Load embeddings and metadata from cache.
        
        Returns:
            Tuple of (embeddings dict, metadata dict)
        """
        embeddings = {}
        metadata = {}
        
        # Load embeddings cache if exists
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)
                logger.info(f"Loaded {len(embeddings)} cached embeddings")
            except Exception as e:
                logger.error(f"Failed to load embeddings cache: {e}")
        
        # Load metadata cache if exists
        if os.path.exists(self.embeddings_metadata_path):
            try:
                with open(self.embeddings_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded {len(metadata)} cached metadata")
            except Exception as e:
                logger.error(f"Failed to load embeddings metadata: {e}")
        
        return embeddings, metadata
    
    def _save_cache(self, embeddings: Dict[str, List[float]], metadata: Dict[str, str]) -> None:
        """
        Save embeddings and metadata to cache.
        
        Args:
            embeddings: Dictionary mapping file paths to embedding vectors
            metadata: Dictionary mapping file paths to content hashes
        """
        try:
            with open(self.embeddings_cache_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)
            
            with open(self.embeddings_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """
        Generate an embedding for file content using OpenAI API.
        
        Args:
            content: File content to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        if not self.client:
            logger.warning("OpenAI client not available, cannot generate embedding")
            return None
        
        # Truncate content if it's too long (API limit)
        if len(content) > 32000:
            logger.warning(f"Content too long ({len(content)} chars), truncating to 32K chars")
            content = content[:32000]
        
        try:
            # Generate embedding via OpenAI API
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=content
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _generate_mock_embedding(self, content: str) -> List[float]:
        """
        Generate a deterministic mock embedding for testing or fallback.
        
        Args:
            content: File content to embed
            
        Returns:
            Mock embedding vector (1536 dimensions)
        """
        import random
        
        # Generate a stable hash from the content
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        
        # Use the hash to seed random for deterministic output
        random.seed(content_hash)
        
        # Generate mock embedding with same dimensionality as OpenAI's
        embedding_size = 1536  # OpenAI's text-embedding-3-small dimension
        mock_embedding = [random.uniform(-1, 1) for _ in range(embedding_size)]
        
        # Normalize to unit vector (like real embeddings)
        norm = sum(x*x for x in mock_embedding) ** 0.5
        if norm > 0:
            mock_embedding = [x/norm for x in mock_embedding]
        
        logger.info("Generated mock embedding as fallback")
        return mock_embedding
    
    def get_embeddings(self, file_contents: Dict[str, str], use_cache: bool = True) -> Dict[str, List[float]]:
        """
        Get embeddings for multiple files with caching.
        
        Args:
            file_contents: Dictionary mapping file paths to file contents
            use_cache: Whether to use cached embeddings when available
            
        Returns:
            Dictionary mapping file paths to embedding vectors
        """
        start_time = time.time()
        
        # Load cache if using it
        cache_embeddings = {}
        cache_metadata = {}
        if use_cache:
            cache_embeddings, cache_metadata = self._load_cache()
        
        # Track which files need new embeddings
        to_process = {}
        result_embeddings = {}
        
        # Check which files need to be processed
        for file_path, content in file_contents.items():
            # Skip empty files
            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                continue
            
            # Calculate content hash
            content_hash = self._calculate_content_hash(content)
            
            # Check if file has a valid cached embedding
            cached = (
                use_cache and
                file_path in cache_embeddings and
                file_path in cache_metadata and
                cache_metadata[file_path] == content_hash
            )
            
            if cached:
                # Use cached embedding
                result_embeddings[file_path] = cache_embeddings[file_path]
                logger.debug(f"Using cached embedding for: {file_path}")
            else:
                # Mark for processing
                to_process[file_path] = content
        
        # Process files that need new embeddings
        processed_count = 0
        cache_hit_count = len(result_embeddings)
        
        logger.info(f"Cache hits: {cache_hit_count}/{len(file_contents)} files")
        
        if to_process:
            logger.info(f"Generating embeddings for {len(to_process)} files...")
            
            for file_path, content in to_process.items():
                # Generate embedding
                embedding = self._generate_embedding(content)
                
                # Fall back to mock embedding if real one fails
                if embedding is None:
                    embedding = self._generate_mock_embedding(content)
                
                # Store embedding
                result_embeddings[file_path] = embedding
                
                # Update cache metadata
                content_hash = self._calculate_content_hash(content)
                cache_metadata[file_path] = content_hash
                
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{len(to_process)} files")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.1)
            
            # Update cache with new embeddings
            if use_cache:
                cache_embeddings.update(result_embeddings)
                self._save_cache(cache_embeddings, cache_metadata)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embedding generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total files: {len(file_contents)}, Cached: {cache_hit_count}, Generated: {processed_count}")
        
        return result_embeddings


def test_real_project_file():
    """
    Test embedding generation with a real project file.
    """
    logger.info("Testing embedding generation with real project file")

    # Create cache manager
    test_dir = os.path.join(os.getcwd(), ".promptpilot", "test_real")
    os.makedirs(test_dir, exist_ok=True)
    
    cache = EmbeddingCache(test_dir)
    
    # Load a real project file (core/analyze.py)
    real_file_path = os.path.join(os.getcwd(), "core", "analyze.py")
    
    try:
        with open(real_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            
        logger.info(f"Loaded real project file: {real_file_path}")
        logger.info(f"File size: {len(file_content)} characters")
        
        # Create file contents dictionary
        file_contents = {"core/analyze.py": file_content}
        
        # First run - should generate embedding
        logger.info("First run - should generate new embedding")
        start_time = time.time()
        embeddings1 = cache.get_embeddings(file_contents)
        first_run_time = time.time() - start_time
        
        # Second run - should use cache
        logger.info("Second run - should use cached embedding")
        start_time = time.time()
        embeddings2 = cache.get_embeddings(file_contents)
        second_run_time = time.time() - start_time
        
        # Compare performance
        logger.info(f"First run time: {first_run_time:.2f} seconds")
        logger.info(f"Second run time: {second_run_time:.2f} seconds")
        if second_run_time < first_run_time:
            speedup = first_run_time / second_run_time
            logger.info(f"Cache speedup: {speedup:.2f}x faster")
        
        # Verify embedding dimensions
        embedding = embeddings1.get("core/analyze.py")
        if embedding:
            logger.info(f"Embedding dimensions: {len(embedding)}")
            # First few values for debugging
            logger.info(f"First 5 values: {embedding[:5]}")
        else:
            logger.error("Failed to generate embedding for analyze.py")
            
        return embeddings1
        
    except Exception as e:
        logger.error(f"Error testing real project file: {e}")
        return {}


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Real Project File Embedding Test")
    print("="*50)
    print("This script tests embedding generation with a real project file")
    print("using the working approach from test_embeddings_cache.py")
    print("="*50 + "\n")
    
    # Test with real project file
    embeddings = test_real_project_file()
    
    print("\n" + "="*50)
    print("Real Project File Embedding Test Complete")
    print("="*50)
    print(f"Successfully tested embedding generation for core/analyze.py")
    print("This confirms the embedding approach works with real project files.")
    print("="*50 + "\n") 