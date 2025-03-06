"""
Database utility functions for PromptPilot.

This module provides helper functions to integrate repository data
with Supabase database.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import time

from core.ingest import ingest_repository, RepositoryIngestor
from core.enhanced_db import get_db, EnhancedDatabase
from core.ast_analyzer import ASTAnalyzer
from core.analyze import RepositoryAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.db_utils')

# Load environment variables
load_dotenv()

def initialize_supabase() -> Optional[Client]:
    """
    Initialize Supabase client using environment variables.
    
    Returns:
        Supabase client or None if initialization fails
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
        return None
    
    try:
        client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

def store_repository_metadata(client: Client, repo_data: Dict[str, Any]) -> Optional[str]:
    """
    Store repository metadata in Supabase.
    
    Args:
        client: Supabase client
        repo_data: Repository data dictionary
        
    Returns:
        Repository ID or None if storage fails
    """
    if not client:
        return None
    
    try:
        # Generate repository ID if not present
        repo_id = repo_data.get('id', str(uuid.uuid4()))
        
        # Prepare repository record
        repo_record = {
            'id': repo_id,
            'name': repo_data.get('name', 'unnamed'),
            'path': repo_data.get('path', ''),
            'file_count': repo_data.get('file_count', 0),
            'total_size_bytes': repo_data.get('total_size_bytes', 0),
            'file_types': json.dumps(repo_data.get('file_types', {})),
            'processed_date': repo_data.get('processed_date', time.time()),
            'created_at': time.time()
        }
        
        # Insert repository record
        result = client.table('repositories').upsert(
            repo_record,
            on_conflict='id'
        ).execute()
        
        if result.data:
            logger.info(f"Repository metadata stored in Supabase: {repo_id}")
            return repo_id
        else:
            logger.warning("No data returned from repository insert operation")
            return repo_id
            
    except Exception as e:
        logger.error(f"Failed to store repository metadata: {e}")
        return None

def store_file_data(client: Client, repo_id: str, file_data: Dict[str, Any]) -> Optional[str]:
    """
    Store file data in Supabase.
    
    Args:
        client: Supabase client
        repo_id: Repository ID
        file_data: File data dictionary
        
    Returns:
        File ID or None if storage fails
    """
    if not client:
        return None
    
    try:
        # Generate file ID if not present
        file_id = file_data.get('id', str(uuid.uuid4()))
        metadata = file_data.get('metadata', {})
        
        # Prepare file record
        file_record = {
            'id': file_id,
            'repository_id': repo_id,
            'file_path': metadata.get('path', ''),
            'size_bytes': metadata.get('size_bytes', 0),
            'extension': metadata.get('extension', ''),
            'last_modified': metadata.get('last_modified', time.time()),
            'lines': file_data.get('lines', 0),
            'characters': file_data.get('characters', 0),
            'content': file_data.get('content', '')[:10000],  # Truncate content for database storage
            'content_url': file_data.get('content_url')
        }
        
        # Insert file record
        result = client.table('files').upsert(
            file_record,
            on_conflict='id'
        ).execute()
        
        if result.data:
            logger.debug(f"File data stored in Supabase: {file_record['file_path']}")
            return file_id
        else:
            logger.warning(f"No data returned from file insert operation: {file_record['file_path']}")
            return file_id
            
    except Exception as e:
        logger.error(f"Failed to store file data: {e}")
        return None

def search_similar_files(client: Client, query: str, repo_id: Optional[str] = None, 
                         top_k: int = 5, match_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Search for files similar to a query using pgvector.
    
    Args:
        client: Supabase client
        query: Query text
        repo_id: Repository ID (optional)
        top_k: Number of results to return
        match_threshold: Minimum similarity threshold
        
    Returns:
        List of similar files with scores and metadata
    """
    if not client:
        return []
    
    try:
        # Call RPC function for embedding search
        # Note: This assumes you've created the appropriate RPC function in Supabase
        params = {
            'query_text': query,
            'match_threshold': match_threshold,
            'match_count': top_k
        }
        
        if repo_id:
            params['repo_id'] = repo_id
        
        response = client.rpc(
            'search_files_by_text',
            params
        ).execute()
        
        if response.data:
            logger.info(f"Found {len(response.data)} similar files for query: '{query}'")
            return response.data
        else:
            logger.info(f"No similar files found for query: '{query}'")
            return []
            
    except Exception as e:
        logger.error(f"Failed to search similar files: {e}")
        return []

def upload_file_content(client: Client, content: str, file_path: str) -> Optional[str]:
    """
    Upload file content to Supabase Storage.
    
    Args:
        client: Supabase client
        content: File content
        file_path: File path (used for naming)
        
    Returns:
        Public URL of the uploaded content or None if upload fails
    """
    if not client:
        return None
    
    try:
        # Generate a unique filename based on path and timestamp
        safe_path = file_path.replace('/', '_').replace('\\', '_')
        timestamp = int(time.time())
        unique_filename = f"{safe_path}_{timestamp}.txt"
        
        # Upload file content to Supabase Storage
        result = client.storage.from_("file_contents").upload(
            unique_filename, 
            content.encode('utf-8'),
            {"content-type": "text/plain"}
        )
        
        # Get the public URL
        public_url = client.storage.from_("file_contents").get_public_url(unique_filename)
        
        logger.debug(f"Uploaded {file_path} to Supabase Storage")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to upload file content to Supabase Storage: {e}")
        return None

def download_file_content(client: Client, url: str) -> Optional[str]:
    """
    Download file content from Supabase Storage.
    
    Args:
        client: Supabase client
        url: Content URL
        
    Returns:
        File content as string or None if download fails
    """
    if not client:
        return None
    
    try:
        # Extract filename from URL
        filename = url.split('/')[-1]
        
        # Download content
        data = client.storage.from_("file_contents").download(filename)
        
        return data.decode('utf-8')
        
    except Exception as e:
        logger.error(f"Failed to download file content from Supabase Storage: {e}")
        return None

def get_repository_by_name(client: Client, repo_name: str) -> Optional[Dict[str, Any]]:
    """
    Get repository by name from Supabase.
    
    Args:
        client: Supabase client
        repo_name: Repository name
        
    Returns:
        Repository data or None if not found
    """
    if not client:
        return None
    
    try:
        response = client.table('repositories').select('*').eq('name', repo_name).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            logger.warning(f"Repository not found: {repo_name}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get repository by name: {e}")
        return None

def get_files_by_repository(client: Client, repo_id: str) -> List[Dict[str, Any]]:
    """
    Get all files for a repository from Supabase.
    
    Args:
        client: Supabase client
        repo_id: Repository ID
        
    Returns:
        List of file data
    """
    if not client:
        return []
    
    try:
        response = client.table('files').select('*').eq('repository_id', repo_id).execute()
        
        if response.data:
            return response.data
        else:
            logger.warning(f"No files found for repository: {repo_id}")
            return []
            
    except Exception as e:
        logger.error(f"Failed to get files by repository: {e}")
        return []

def create_tables_if_not_exist(client: Client) -> bool:
    """
    Create necessary tables in Supabase if they don't exist.
    Note: This should typically be done through migrations rather than in code.
    
    Args:
        client: Supabase client
        
    Returns:
        True if successful, False otherwise
    """
    if not client:
        return False
    
    try:
        # This function primarily serves as documentation for the expected schema
        # In practice, you should use Supabase migrations or SQL editor to create tables
        
        # Example of tables needed:
        # 1. repositories - stores repository metadata
        # 2. files - stores file data
        # 3. file_embeddings - stores file embeddings (created by pgvector extension)
        
        logger.info("Tables should be created through Supabase migrations or SQL editor")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def process_repository_data(client: Client, repo_data: Dict[str, Any]) -> Optional[str]:
    """
    Process and store complete repository data in Supabase.
    
    Args:
        client: Supabase client
        repo_data: Repository data dictionary with files
        
    Returns:
        Repository ID or None if processing fails
    """
    if not client:
        return None
    
    try:
        # Store repository metadata
        repo_id = store_repository_metadata(client, repo_data)
        
        if not repo_id:
            logger.error("Failed to store repository metadata")
            return None
        
        # Store file data
        for file_entry in repo_data.get('files', []):
            # Check if file has content
            if 'content' not in file_entry:
                continue
                
            # Upload content to storage if not already uploaded
            if not file_entry.get('content_url'):
                content_url = upload_file_content(
                    client, 
                    file_entry.get('content', ''), 
                    file_entry.get('metadata', {}).get('path', 'unknown')
                )
                file_entry['content_url'] = content_url
            
            # Store file record
            file_id = store_file_data(client, repo_id, file_entry)
            
            if not file_id:
                logger.warning(f"Failed to store file: {file_entry.get('metadata', {}).get('path', 'unknown')}")
        
        logger.info(f"Repository data processed successfully: {repo_id}")
        return repo_id
        
    except Exception as e:
        logger.error(f"Failed to process repository data: {e}")
        return None

# Make Supabase client available as a singleton
_supabase_client = None

def get_supabase_client() -> Optional[Client]:
    """
    Get or initialize Supabase client.
    
    Returns:
        Supabase client or None if initialization fails
    """
    global _supabase_client
    
    if _supabase_client is None:
        _supabase_client = initialize_supabase()
    
    return _supabase_client

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
        db = get_db()
        
        # Ingest repository
        logger.info(f"Ingesting repository: {repo_path}")
        repo_data = ingest_repository(repo_path, output_dir)

        # Calculate and add the content hash. Do this *before* storing in Supabase.
        for file_entry in repo_data['files']:
            content = file_entry['content']
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            file_entry['content_hash'] = content_hash
        
        # Store repository in database
        logger.info("Storing repository in database...")
        repo_id = db.process_repository(repo_data)
        
        if repo_id:
            logger.info(f"Repository {repo_data['name']} successfully stored in database with ID: {repo_id}")
            return repo_id
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
        db = get_db()
        
        # Ingest repository
        logger.info(f"Ingesting repository: {repo_path}")
        ingestor = RepositoryIngestor(repo_path, output_dir)
        repo_data = ingestor.process_repository()

        # Calculate and add the content hash. Do this *before* storing in Supabase.
        for file_entry in repo_data['files']:
            content = file_entry['content']
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            file_entry['content_hash'] = content_hash
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), ".promptpilot")
        
        # Analyze AST
        logger.info("Analyzing AST...")
        ast_analyzer = ASTAnalyzer(output_dir)
        ast_data = ast_analyzer.analyze_repository()
        
        # Generate embeddings and analyze
        logger.info("Generating embeddings and analyzing repository...")
        analyzer = RepositoryAnalyzer(output_dir, use_supabase=True)
        embeddings = analyzer.generate_embeddings()
        
        # Store repository in database
        logger.info("Storing repository in database...")
        repo_id = db.process_repository(repo_data)
        
        if repo_id:
            # Generate and store a text digest of the repository
            logger.info("Generating repository digest...")
            digest_text = ingestor.get_repository_digest()
            
            # Store the digest text in the database
            logger.info("Storing repository digest in database...")
            digest_success = db.store_repository_digest(repo_id, digest_text)
            
            if digest_success:
                logger.info(f"Repository digest stored successfully for repository {repo_id}")
            else:
                logger.warning(f"Failed to store repository digest for repository {repo_id}")
                
            logger.info(f"Repository {repo_data['name']} completely processed and stored in database")
            return repo_id
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
        db = get_db()
        
        # Get repository by path
        repo_data = None
        
        # First try to find by exact path
        repo_data = db.get_repository_by_path(repo_path)
        
        # If not found, try to find by name (basename)
        if not repo_data:
            repo_name = os.path.basename(os.path.normpath(repo_path))
            repo_data = db.get_repository_by_name(repo_name)
            
        return repo_data
        
    except Exception as e:
        logger.error(f"Error getting repository from database: {str(e)}")
        return None

def find_relevant_files(query: str, repo_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find files most relevant to a query using embeddings.
    
    Args:
        query: Text query
        repo_id: Repository ID (optional)
        top_k: Number of top results to return
        
    Returns:
        List of file documents with similarity scores
    """
    try:
        # Initialize DB
        db = get_db()
        
        # Find relevant files using text search
        return db.search_similar_text(query, repo_id, top_k)
        
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