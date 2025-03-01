#!/usr/bin/env python
"""
Supabase Integration Demo for PromptPilot

This script demonstrates how to use the Supabase integration features
for repository processing, embedding generation, and similarity search.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.demo')

# Load environment variables
load_dotenv()

# Add the project directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from core.ingest import RepositoryIngestor, ingest_repository
from core.analyze import RepositoryAnalyzer
from core.enhanced_db import get_db, EnhancedDatabase


def check_supabase_config() -> bool:
    """Check if Supabase is configured correctly."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
        logger.error("Please add these to your .env file.")
        return False
    
    return True


def ingest_repo(repo_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest a repository using the RepositoryIngestor.
    
    Args:
        repo_path: Path to the Git repository
        output_dir: Directory to store processed data
        
    Returns:
        Repository data dictionary
    """
    logger.info(f"Ingesting repository at: {repo_path}")
    
    # Initialize the ingestor
    ingestor = RepositoryIngestor(repo_path, output_dir)
    
    # Process the repository
    repo_data = ingestor.process_repository()
    
    logger.info(f"Repository ingested: {repo_data['name']}")
    logger.info(f"Files processed: {repo_data['file_count']}")
    
    return repo_data


def generate_embeddings(repo_data_path: str, output_dir: Optional[str] = None) -> None:
    """
    Generate embeddings for a repository using the RepositoryAnalyzer.
    
    Args:
        repo_data_path: Path to the repository data JSON file
        output_dir: Directory to store analysis data
    """
    logger.info(f"Generating embeddings for repository data at: {repo_data_path}")
    
    # Initialize the analyzer with Supabase integration
    analyzer = RepositoryAnalyzer(repo_data_path, output_dir, use_supabase=True)
    
    # Generate embeddings
    analysis_results = analyzer.analyze_repository()
    
    logger.info(f"Embeddings generated for {analysis_results['file_count']} files")
    logger.info(f"Analysis results saved at: {output_dir}")


def process_with_enhanced_db(repo_data: Dict[str, Any]) -> Optional[str]:
    """
    Process a repository using the EnhancedDatabase.
    
    Args:
        repo_data: Repository data dictionary
        
    Returns:
        Repository ID or None if processing fails
    """
    logger.info(f"Processing repository with enhanced database: {repo_data['name']}")
    
    # Get the database instance
    db = get_db()
    
    # Process the repository with chunking for better results
    repo_id = db.process_repository_chunks(repo_data)
    
    if repo_id:
        logger.info(f"Repository processed successfully: {repo_id}")
    else:
        logger.error("Failed to process repository")
    
    return repo_id


def search_similar_code(query: str, repo_id: Optional[str] = None, top_k: int = 5) -> None:
    """
    Search for similar code using the EnhancedDatabase.
    
    Args:
        query: Search query
        repo_id: Repository ID (optional)
        top_k: Number of results to return
    """
    logger.info(f"Searching for code similar to: {query}")
    
    # Get the database instance
    db = get_db()
    
    # Search for similar code by text query
    results = db.search_similar_text(query, repo_id, top_k)
    
    logger.info(f"Found {len(results)} similar files")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.get('file_path', 'Unknown')} (similarity: {result.get('similarity', 0):.4f})")
        print(f"   Preview: {result.get('content', '')[:200]}...")
        print("   " + "-" * 50)


def run_full_pipeline(repo_path: str, query: Optional[str] = None) -> None:
    """
    Run the full pipeline: ingest, generate embeddings, and search.
    
    Args:
        repo_path: Path to the Git repository
        query: Search query (optional)
    """
    logger.info(f"Running full pipeline for repository: {repo_path}")
    
    # Ingest the repository
    repo_data = ingest_repo(repo_path)
    
    # Process with enhanced database
    repo_id = process_with_enhanced_db(repo_data)
    
    if repo_id and query:
        # Search for similar code
        search_similar_code(query, repo_id)
    
    logger.info("Pipeline completed successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Supabase Integration Demo for PromptPilot")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest repository command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a repository")
    ingest_parser.add_argument("repo_path", help="Path to the Git repository")
    ingest_parser.add_argument("--output", help="Directory to store processed data")
    
    # Generate embeddings command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for a repository")
    embed_parser.add_argument("data_path", help="Path to the repository data JSON file")
    embed_parser.add_argument("--output", help="Directory to store analysis data")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar code")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--repo-id", help="Repository ID (optional)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("repo_path", help="Path to the Git repository")
    pipeline_parser.add_argument("--query", help="Search query (optional)")
    
    # Test connection command
    subparsers.add_parser("test", help="Test Supabase connection")
    
    args = parser.parse_args()
    
    # Check if Supabase is configured
    if not check_supabase_config():
        sys.exit(1)
    
    # Run the appropriate command
    if args.command == "ingest":
        ingest_repo(args.repo_path, args.output)
    elif args.command == "embed":
        generate_embeddings(args.data_path, args.output)
    elif args.command == "search":
        search_similar_code(args.query, args.repo_id, args.top_k)
    elif args.command == "pipeline":
        run_full_pipeline(args.repo_path, args.query)
    elif args.command == "test":
        # Test Supabase connection
        try:
            db = get_db()
            print("Supabase connection successful!")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 