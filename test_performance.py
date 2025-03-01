#!/usr/bin/env python3
"""
Test script for PromptPilot performance improvements.
"""

import os
import time
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("promptpilot.test")

def test_embedding_cache(repository_dir):
    """Test the embedding cache functionality."""
    from core.analyze import RepositoryAnalyzer
    import json
    
    logger.info("Testing embedding cache...")
    
    # Load repository data
    data_path = os.path.join(repository_dir, 'repository_data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
    
    # First run should create cache
    analyzer = RepositoryAnalyzer(repository_dir)
    logger.info("First run (should generate all embeddings):")
    start_time = time.time()
    embeddings1 = analyzer.generate_embeddings(repo_data)
    first_run_time = time.time() - start_time
    logger.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Second run should use cache
    logger.info("Second run (should use cache):")
    start_time = time.time()
    embeddings2 = analyzer.generate_embeddings(repo_data)
    second_run_time = time.time() - start_time
    logger.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Verify results
    if len(embeddings1) != len(embeddings2):
        logger.error(f"Embedding count mismatch: {len(embeddings1)} vs {len(embeddings2)}")
    else:
        logger.info(f"Embedding counts match: {len(embeddings1)}")
    
    # Check cache files exist
    cache_path = os.path.join(repository_dir, 'embeddings_cache.json')
    cache_metadata_path = os.path.join(repository_dir, 'embeddings_cache_metadata.json')
    
    if os.path.exists(cache_path) and os.path.exists(cache_metadata_path):
        logger.info("Cache files created successfully")
    else:
        logger.error("Cache files not found")
    
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    logger.info(f"Cache speedup: {speedup:.1f}x")
    
    return first_run_time, second_run_time

def test_large_file_handling(repo_path, output_dir=None):
    """Test the large file handling functionality."""
    from core.ingest import RepositoryIngestor
    import shutil
    
    logger.info("Testing large file handling...")
    
    # Create a temporary large file
    temp_dir = Path(repo_path) / "temp_test_dir"
    temp_dir.mkdir(exist_ok=True)
    
    large_file = temp_dir / "large_test_file.py"
    
    # Create a 1MB test file
    with open(large_file, 'w') as f:
        f.write("# Large test file\n")
        f.write("def test_function():\n")
        f.write("    print('This is a test')\n\n")
        
        # Add content to make it large
        for i in range(50000):
            f.write(f"# Line {i}: This is a test line to make the file larger\n")
    
    try:
        # Process the repository
        ingestor = RepositoryIngestor(repo_path, output_dir)
        repo_data = ingestor.process_repository()
        
        # Check if our large file was processed
        large_file_rel_path = os.path.relpath(large_file, repo_path)
        
        for file_entry in repo_data["files"]:
            if file_entry["metadata"]["path"] == large_file_rel_path:
                logger.info(f"Large file was processed: {large_file_rel_path}")
                logger.info(f"Content length: {len(file_entry['content'])} characters")
                return True
        
        logger.error(f"Large file was not processed: {large_file_rel_path}")
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

def test_async_embedding(repository_dir):
    """Test the async embedding functionality."""
    from core.analyze import RepositoryAnalyzer
    import json
    
    logger.info("Testing async embedding generation...")
    
    # Load repository data
    data_path = os.path.join(repository_dir, 'repository_data.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)
    
    # Delete cache to force regeneration
    cache_path = os.path.join(repository_dir, 'embeddings_cache.json')
    cache_metadata_path = os.path.join(repository_dir, 'embeddings_cache_metadata.json')
    
    if os.path.exists(cache_path):
        os.remove(cache_path)
    if os.path.exists(cache_metadata_path):
        os.remove(cache_metadata_path)
    
    analyzer = RepositoryAnalyzer(repository_dir)
    
    start_time = time.time()
    embeddings = analyzer.generate_embeddings(repo_data)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description="Test PromptPilot performance improvements")
    parser.add_argument("--repo-path", required=True, help="Path to repository")
    parser.add_argument("--output-dir", help="Output directory for processed data")
    parser.add_argument("--skip-cache-test", action="store_true", help="Skip embedding cache test")
    parser.add_argument("--skip-file-test", action="store_true", help="Skip large file test")
    parser.add_argument("--skip-async-test", action="store_true", help="Skip async embedding test")
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = os.path.join(args.repo_path, '.promptpilot')
    
    print("\n==== PROMPTPILOT PERFORMANCE TESTS ====\n")
    
    results = {}
    
    if not args.skip_cache_test:
        try:
            results["cache"] = test_embedding_cache(args.output_dir)
        except Exception as e:
            logger.error(f"Embedding cache test failed: {e}")
            results["cache"] = "FAILED"
    
    if not args.skip_file_test:
        try:
            results["large_file"] = test_large_file_handling(args.repo_path, args.output_dir)
        except Exception as e:
            logger.error(f"Large file test failed: {e}")
            results["large_file"] = "FAILED"
    
    if not args.skip_async_test:
        try:
            results["async"] = test_async_embedding(args.output_dir)
        except Exception as e:
            logger.error(f"Async embedding test failed: {e}")
            results["async"] = "FAILED"
    
    print("\n==== TEST RESULTS SUMMARY ====\n")
    
    if "cache" in results:
        if results["cache"] == "FAILED":
            print("❌ Embedding cache test: FAILED")
        else:
            first_run, second_run = results["cache"]
            speedup = first_run / second_run if second_run > 0 else float('inf')
            print(f"✅ Embedding cache test: PASSED - {speedup:.1f}x speedup")
    
    if "large_file" in results:
        if results["large_file"] == "FAILED" or not results["large_file"]:
            print("❌ Large file handling test: FAILED")
        else:
            print("✅ Large file handling test: PASSED")
    
    if "async" in results:
        if results["async"] == "FAILED":
            print("❌ Async embedding test: FAILED")
        else:
            print(f"✅ Async embedding test: PASSED - {results['async']:.2f} seconds")
    
    print("\nTests completed.")

if __name__ == "__main__":
    main() 