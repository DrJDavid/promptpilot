"""
Database utility functions for PromptPilot.

This module provides helper functions to integrate repository data
with the ArangoDB database.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from core.ingest import ingest_repository, RepositoryIngestor
from core.enhanced_db import get_repository_db, RepositoryDB
from core.ast_analyzer import ASTAnalyzer
from core.analyze import RepositoryAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.db_utils')

def ingest_and_store_repository(repo_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Ingest a repository and store it in the database.
    
    Args:
        repo_path: Path to the Git repository
        output_dir: Directory to store processed data
        
    Returns:
        Repository ID if successful, None otherwise
    """
    try:
        # Initialize DB
        db = get_repository_db()
        
        if not db.is_available():
            logger.warning("Database not available. Repository data will be saved to disk only.")
        
        # Ingest repository
        logger.info(f"Ingesting repository: {repo_path}")
        repo_data = ingest_repository(repo_path, output_dir)
        
        if not db.is_available():
            return None
        
        # Store repository in database
        logger.info("Storing repository in database...")
        repo_id = db.store_repository(repo_data)
        
        if repo_id:
            # Store files
            logger.info("Storing file data in database...")
            files_stored = db.store_files(repo_id, repo_data['files'])
            
            if files_stored:
                logger.info(f"Repository {repo_data['name']} successfully stored in database with ID: {repo_id}")
                return repo_id
            else:
                logger.error("Failed to store files in database")
                return None
        else:
            logger.error("Failed to store repository in database")
            return None
        
    except Exception as e:
        logger.error(f"Error ingesting and storing repository: {str(e)}")
        return None

def process_repository_with_ast(repo_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Process a repository with AST analysis and store all data in the database.
    
    Args:
        repo_path: Path to the Git repository
        output_dir: Directory to store processed data
        
    Returns:
        Repository ID if successful, None otherwise
    """
    try:
        # Initialize DB
        db = get_repository_db()
        
        if not db.is_available():
            logger.warning("Database not available. Repository data will be saved to disk only.")
        
        # Ingest repository
        logger.info(f"Ingesting repository: {repo_path}")
        ingestor = RepositoryIngestor(repo_path, output_dir)
        repo_data = ingestor.process_repository()
        
        # Determine output directory
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dot_dir = os.path.join(base_dir, '.promptpilot')
            repo_name = os.path.basename(os.path.abspath(repo_path))
            output_dir = os.path.join(dot_dir, repo_name)
        
        # Analyze AST
        logger.info("Analyzing AST...")
        ast_analyzer = ASTAnalyzer(output_dir)
        ast_data = ast_analyzer.analyze_repository()
        
        # Generate embeddings and analyze
        logger.info("Generating embeddings and analyzing repository...")
        analyzer = RepositoryAnalyzer(output_dir)
        embeddings, analysis = analyzer.analyze_repository()
        
        if not db.is_available():
            return None
        
        # Store repository in database
        logger.info("Storing repository in database...")
        repo_id = db.store_repository(repo_data)
        
        if repo_id:
            # Store files
            logger.info("Storing file data in database...")
            files_stored = db.store_files(repo_id, repo_data['files'])
            
            if files_stored:
                # Store AST data
                logger.info("Storing AST data in database...")
                ast_stored = db.store_ast_data(repo_id, ast_data)
                
                # Store analysis data
                logger.info("Storing analysis data in database...")
                analysis_stored = db.store_analysis_data(repo_id, embeddings, analysis)
                
                if ast_stored and analysis_stored:
                    logger.info(f"Repository {repo_data['name']} completely processed and stored in database")
                    return repo_id
                else:
                    logger.warning("Some data was not stored in the database")
                    return repo_id
            else:
                logger.error("Failed to store files in database")
                return None
        else:
            logger.error("Failed to store repository in database")
            return None
        
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        return None

def get_repository_from_db(repo_path: str) -> Optional[Dict[str, Any]]:
    """
    Get repository data from the database.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Repository data dictionary or None if not found
    """
    try:
        # Initialize DB
        db = get_repository_db()
        
        if not db.is_available():
            logger.warning("Database not available.")
            return None
        
        # Get repository
        return db.get_repository_by_path(repo_path)
        
    except Exception as e:
        logger.error(f"Error getting repository from database: {str(e)}")
        return None

def find_relevant_files(repo_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find files most relevant to a query using embeddings.
    
    Args:
        repo_id: Repository ID
        query_embedding: Query embedding vector
        top_k: Number of top results to return
        
    Returns:
        List of file documents with similarity scores
    """
    try:
        # Initialize DB
        db = get_repository_db()
        
        if not db.is_available():
            logger.warning("Database not available.")
            return []
        
        # Find relevant files
        return db.find_relevant_files(repo_id, query_embedding, top_k)
        
    except Exception as e:
        logger.error(f"Error finding relevant files: {str(e)}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database utilities for PromptPilot")
    parser.add_argument("repo_path", help="Path to the Git repository")
    parser.add_argument("--output", help="Directory to store processed data")
    parser.add_argument("--full", action="store_true", help="Run full processing (ingest, AST, analysis)")
    
    args = parser.parse_args()
    
    try:
        if args.full:
            repo_id = process_repository_with_ast(args.repo_path, args.output)
            if repo_id:
                print(f"Repository processed and stored with ID: {repo_id}")
        else:
            repo_id = ingest_and_store_repository(args.repo_path, args.output)
            if repo_id:
                print(f"Repository ingested and stored with ID: {repo_id}")
        
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise 