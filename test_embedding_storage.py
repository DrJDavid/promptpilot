#!/usr/bin/env python
"""
Test script to verify if embedding storage works properly in Supabase.
This script attempts to update a single file record with a random embedding vector.
"""

import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("embedding_test")

def test_embedding_storage():
    """
    Test function to verify if basic embedding storage works in Supabase.
    Creates a random embedding vector and tries to update a file record.
    """
    from core.enhanced_db import RepositoryDB
    import numpy as np
    
    logger.info("Initializing embedding storage test")
    
    # Create a test embedding (1536 dimensions for OpenAI embeddings)
    test_embedding = list(np.random.rand(1536).astype(float))
    logger.info(f"Created test embedding with {len(test_embedding)} dimensions")
    
    # Initialize the DB
    db = RepositoryDB()
    logger.info("Initialized RepositoryDB")
    
    # Get a file ID from your database
    logger.info("Fetching a file from the database...")
    data, count = db.supabase.table('files').select("id").limit(1).execute()
    if not data or len(data) < 2 or not data[1]:
        logger.error("No files found in database")
        return
    
    file_id = data[1][0]['id']
    logger.info(f"Found file with ID: {file_id}")
    
    # Print the first few values of the embedding for verification
    logger.info(f"Sample of embedding vector: {test_embedding[:5]}...")
    
    # Attempt to store just the embedding
    try:
        logger.info(f"Attempting to update file {file_id} with test embedding")
        result = db.supabase.table('files').update({"embedding": test_embedding}).eq('id', file_id).execute()
        
        logger.info("Update completed")
        
        if result.data and len(result.data) > 1 and result.data[1]:
            logger.info("Successfully updated embedding")
            
            # Verify the update by checking the record
            verify_data, _ = db.supabase.table('files').select("id, embedding").eq('id', file_id).execute()
            if verify_data and len(verify_data) > 1 and verify_data[1]:
                embedding_stored = verify_data[1][0].get('embedding')
                if embedding_stored:
                    logger.info(f"Verified embedding is stored, first few values: {embedding_stored[:5]}")
                else:
                    logger.error("Embedding still appears to be NULL after update")
            else:
                logger.error("Could not verify if embedding was stored")
        else:
            logger.error(f"Failed to update embedding: {result}")
    except Exception as e:
        logger.error(f"Error updating embedding: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_embedding_storage() 