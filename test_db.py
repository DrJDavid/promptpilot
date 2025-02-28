#!/usr/bin/env python3
"""
Test script for the PromptPilot ArangoDB integration.

This script tests the connection to ArangoDB and basic database operations.
"""

import os
import logging
from dotenv import load_dotenv
from core.enhanced_db import RepositoryDB, get_repository_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.test_db')

# Load environment variables
load_dotenv()

def test_db_connection():
    """Test the database connection."""
    logger.info("Testing database connection...")
    
    # Get database connection
    db = get_repository_db()
    
    if db.is_available():
        logger.info("Database connection successful!")
        logger.info(f"Connected to {db.host}:{db.port}, database: {db.db_name}")
        return True
    else:
        logger.error("Database connection failed!")
        logger.info("Make sure ArangoDB is running and credentials are correct in .env file.")
        logger.info("Required .env variables:")
        logger.info("  - ARANGO_HOST (default: localhost)")
        logger.info("  - ARANGO_PORT (default: 8529)")
        logger.info("  - ARANGO_DB (default: promptpilot)")
        logger.info("  - ARANGO_USER (default: root)")
        logger.info("  - ARANGO_PASSWORD")
        return False

def test_db_operations():
    """Test basic database operations."""
    logger.info("Testing basic database operations...")
    
    # Get database connection
    db = get_repository_db()
    
    if not db.is_available():
        logger.error("Database not available. Skipping operations test.")
        return False
    
    # Create test repository data
    repo_data = {
        'name': 'test-repo',
        'path': '/path/to/test-repo',
        'file_count': 10,
        'total_size_bytes': 1024,
        'file_types': {'.py': 5, '.md': 3, '.txt': 2},
        'processed_date': '2025-03-01'
    }
    
    # Store repository
    repo_id = db.store_repository(repo_data)
    
    if repo_id:
        logger.info(f"Repository stored with ID: {repo_id}")
        
        # Get repositories
        repos = db.get_repositories()
        logger.info(f"Found {len(repos)} repositories in database")
        
        # Get repository by path
        repo = db.get_repository_by_path(repo_data['path'])
        if repo:
            logger.info(f"Retrieved repository: {repo['name']}")
        else:
            logger.error("Failed to retrieve repository by path")
            return False
        
        return True
    else:
        logger.error("Failed to store repository")
        return False

if __name__ == "__main__":
    print("PromptPilot ArangoDB Test")
    print("========================")
    
    # Test database connection
    if test_db_connection():
        print("\n✅ Database connection test passed\n")
        
        # Test database operations
        if test_db_operations():
            print("\n✅ Database operations test passed\n")
            print("ArangoDB integration is working correctly!")
        else:
            print("\n❌ Database operations test failed\n")
    else:
        print("\n❌ Database connection test failed\n")
    
    print("\nTo run ArangoDB locally with Docker:")
    print("docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=password arangodb/arangodb:latest") 