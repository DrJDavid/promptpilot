#!/usr/bin/env python
"""
Utility script to clear all embedding cache files.

This script deletes embedding cache files to force regeneration of embeddings
on the next analysis run. Useful when:
1. You want to ensure fresh embeddings
2. You're troubleshooting embedding issues
3. You've changed embedding models
"""

import os
import json
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_embedding_caches(repository_dir=None, dry_run=False):
    """
    Clear all embedding cache files in the repository.
    
    Args:
        repository_dir: Optional path to the repository directory (.promptpilot folder)
                       If None, will use default location.
        dry_run: If True, only report what would be deleted without actually deleting
    
    Returns:
        Number of files cleared
    """
    # Default repository directory is .promptpilot in current directory
    if repository_dir is None:
        repository_dir = os.path.join(os.getcwd(), ".promptpilot")
    
    # Check if directory exists
    if not os.path.exists(repository_dir):
        logger.warning(f"Repository directory {repository_dir} not found")
        return 0
    
    logger.info(f"Clearing embedding caches in {repository_dir}")
    
    # List of cache files to check for
    cache_files = [
        # Main embedding files
        os.path.join(repository_dir, "embeddings.json"),
        os.path.join(repository_dir, "embeddings_cache.json"),
        os.path.join(repository_dir, "embeddings_cache_metadata.json"),
        
        # Cache directory files (from newer implementation)
        os.path.join(repository_dir, "cache", "embeddings_cache.json"),
        os.path.join(repository_dir, "cache", "embeddings_metadata.json")
    ]
    
    # Track which files were cleared
    cleared_files = []
    
    # Process each potential cache file
    for file_path in cache_files:
        if os.path.exists(file_path):
            if dry_run:
                logger.info(f"Would delete: {file_path}")
                cleared_files.append(file_path)
            else:
                try:
                    # For JSON files, clear by replacing with empty object
                    # This preserves the file but empties the content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)
                    logger.info(f"Cleared cache file: {file_path}")
                    cleared_files.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to clear {file_path}: {e}")
    
    # Check for cache directory - may contain other cache files
    cache_dir = os.path.join(repository_dir, "cache")
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        if dry_run:
            logger.info(f"Would clear all files in: {cache_dir}")
        else:
            try:
                # Count files in the directory
                cache_file_count = len([f for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))])
                
                if cache_file_count > 0:
                    # Create directory if it doesn't exist (unlikely but possible)
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Clear all files in the directory
                    for filename in os.listdir(cache_dir):
                        file_path = os.path.join(cache_dir, filename)
                        if os.path.isfile(file_path):
                            try:
                                # For JSON files, clear by replacing with empty object
                                if file_path.endswith('.json'):
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        json.dump({}, f)
                                else:
                                    # For other files, delete them
                                    os.remove(file_path)
                                logger.info(f"Cleared cache file: {file_path}")
                                cleared_files.append(file_path)
                            except Exception as e:
                                logger.error(f"Failed to clear {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to clear cache directory: {e}")

    # Check for test directory caches
    test_dir = os.path.join(repository_dir, "test")
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        if dry_run:
            logger.info(f"Would clear all files in: {test_dir}")
        else:
            try:
                # Count files in the directory
                test_file_count = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
                
                if test_file_count > 0:
                    # Clear all files in the directory
                    for filename in os.listdir(test_dir):
                        file_path = os.path.join(test_dir, filename)
                        if os.path.isfile(file_path):
                            try:
                                # For JSON files, clear by replacing with empty object
                                if file_path.endswith('.json'):
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        json.dump({}, f)
                                else:
                                    # For other files, delete them
                                    os.remove(file_path)
                                logger.info(f"Cleared test cache file: {file_path}")
                                cleared_files.append(file_path)
                            except Exception as e:
                                logger.error(f"Failed to clear {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to clear test directory: {e}")
    
    # Report summary
    if cleared_files:
        logger.info(f"Successfully cleared {len(cleared_files)} cache files")
        return len(cleared_files)
    else:
        logger.info("No embedding cache files found to clear")
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear embedding cache files")
    parser.add_argument("--repository-dir", help="Path to the repository directory (.promptpilot folder)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PromptPilot Embedding Cache Cleaner")
    print("="*60)
    
    num_cleared = clear_embedding_caches(
        repository_dir=args.repository_dir,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print(f"\nDry run completed. {num_cleared} cache files would be cleared.")
    else:
        print(f"\nOperation completed. {num_cleared} cache files were cleared.")
    
    print("="*60 + "\n") 