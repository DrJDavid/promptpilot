#!/usr/bin/env python
"""
Test script for implementing an efficient embedding cache.

This script demonstrates how to implement an embedding cache to
avoid regenerating embeddings for files that haven't changed,
improving performance of the analysis pipeline.
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
        Calculate a hash of file content to detect changes.
        
        Args:
            content: File content to hash
            
        Returns:
            SHA-256 hash of content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _load_cache(self) -> tuple[Dict[str, List[float]], Dict[str, str]]:
        """
        Load embeddings and metadata cache from disk.
        
        Returns:
            Tuple of (embeddings dict, metadata dict)
        """
        embeddings = {}
        metadata = {}
        
        # Load embeddings cache if it exists
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)
                logger.info(f"Loaded {len(embeddings)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
        
        # Load metadata cache if it exists
        if os.path.exists(self.embeddings_metadata_path):
            try:
                with open(self.embeddings_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(metadata)} cached files")
            except Exception as e:
                logger.warning(f"Failed to load embeddings metadata: {e}")
        
        return embeddings, metadata
    
    def _save_cache(self, embeddings: Dict[str, List[float]], metadata: Dict[str, str]) -> None:
        """
        Save embeddings and metadata cache to disk.
        
        Args:
            embeddings: Dictionary mapping file paths to embedding vectors
            metadata: Dictionary mapping file paths to content hashes
        """
        # Save embeddings cache
        with open(self.embeddings_cache_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        
        # Save metadata cache
        with open(self.embeddings_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        logger.info(f"Saved {len(embeddings)} embeddings to cache")
    
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


def test_embedding_cache(sample_files: Dict[str, str] = None):
    """
    Test the embedding cache implementation with sample files.
    
    Args:
        sample_files: Optional dictionary of sample files to test with
    """
    logger.info("Testing embedding cache implementation...")
    
    # Create a test repository directory
    test_dir = os.path.join(os.getcwd(), ".promptpilot", "test")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create cache manager
    cache = EmbeddingCache(test_dir)
    
    # Generate sample files if not provided
    if not sample_files:
        sample_files = {
            "file1.py": "def hello():\n    print('Hello, world!')\n\nhello()",
            "file2.py": "import math\n\ndef calculate_area(radius):\n    return math.pi * radius ** 2",
            "file3.txt": "This is a sample text file with some content."
        }
    
    # First run - should generate embeddings for all files
    logger.info("First run - should generate new embeddings for all files")
    start_time = time.time()
    embeddings1 = cache.get_embeddings(sample_files)
    first_run_time = time.time() - start_time
    
    # Second run with same files - should use cache
    logger.info("Second run - should use cached embeddings")
    start_time = time.time()
    embeddings2 = cache.get_embeddings(sample_files)
    second_run_time = time.time() - start_time
    
    # Compare performance
    logger.info(f"First run time: {first_run_time:.2f} seconds")
    logger.info(f"Second run time: {second_run_time:.2f} seconds")
    if second_run_time < first_run_time:
        speedup = first_run_time / second_run_time
        logger.info(f"Cache speedup: {speedup:.2f}x faster")
    
    # Modify one file and run again - should update just that file
    logger.info("Third run - modifying one file to test partial cache update")
    modified_files = sample_files.copy()
    modified_files["file2.py"] += "\n\nprint(calculate_area(5))"
    
    start_time = time.time()
    embeddings3 = cache.get_embeddings(modified_files)
    third_run_time = time.time() - start_time
    
    logger.info(f"Third run time (one modified file): {third_run_time:.2f} seconds")
    
    # Verify embeddings differ for modified file but remain same for others
    is_file1_same = embeddings1["file1.py"] == embeddings3["file1.py"]
    is_file2_different = embeddings1["file2.py"] != embeddings3["file2.py"]
    
    logger.info(f"Unmodified file has same embedding: {is_file1_same}")
    logger.info(f"Modified file has different embedding: {is_file2_different}")
    
    return embeddings3


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Embedding Cache Implementation Test")
    print("="*50)
    print("This script tests the implementation of an efficient embedding cache")
    print("that avoids regenerating embeddings for unchanged files.")
    print("="*50 + "\n")
    
    # Test the embedding cache
    embeddings = test_embedding_cache()
    
    print("\n" + "="*50)
    print("Embedding Cache Test Complete")
    print("="*50)
    print(f"Successfully generated/cached embeddings for {len(embeddings)} files")
    print("This cache implementation can be integrated into the main code")
    print("to improve performance of the analysis pipeline.")
    print("="*50 + "\n") 