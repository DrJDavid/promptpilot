"""
Enhanced database module with Supabase and pgvector integration.

This module provides advanced database functionality using Supabase's pgvector
extension for efficient vector similarity search and embedding storage.
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.enhanced_db')

# Load environment variables
load_dotenv()


class EnhancedDatabase:
    """Class for enhanced database operations with Supabase and pgvector."""
    
    def __init__(self):
        """Initialize the enhanced database connection."""
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
            raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized for enhanced database operations")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def store_repository(self, repo_data: Dict[str, Any]) -> str:
        """
        Store repository metadata in Supabase.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Repository ID
            
        Raises:
            ValueError: If repository data is invalid
        """
        if not repo_data or not isinstance(repo_data, dict):
            raise ValueError("Invalid repository data")
        
        # Generate repository ID if not present
        repo_id = repo_data.get('id', str(uuid.uuid4()))
        
        # Convert file_types to string if it's a dict
        file_types = repo_data.get('file_types', {})
        if isinstance(file_types, dict):
            file_types = json.dumps(file_types)
        
        # Prepare repository record
        repo_record = {
            'id': repo_id,
            'name': repo_data.get('name', 'unnamed'),
            'path': repo_data.get('path', ''),
            'file_count': repo_data.get('file_count', 0),
            'total_size_bytes': repo_data.get('total_size_bytes', 0),
            'file_types': file_types,
            'processed_date': repo_data.get('processed_date', datetime.now(timezone.utc).isoformat())
        }
        
        # Insert repository record
        try:
            result = self.client.table('repositories').upsert(
                repo_record,
                on_conflict='id'
            ).execute()
            
            if not result.data:
                logger.warning("No data returned from repository insert operation")
            
            logger.info(f"Repository metadata stored in Supabase: {repo_id}")
            return repo_id
            
        except Exception as e:
            logger.error(f"Failed to store repository metadata: {e}")
            raise
    
    def store_file(self, repo_id: str, file_data: Dict[str, Any]) -> str:
        """
        Store file data in Supabase.
        
        Args:
            repo_id: Repository ID
            file_data: File data dictionary
            
        Returns:
            File ID
            
        Raises:
            ValueError: If file data is invalid
        """
        if not repo_id or not file_data:
            raise ValueError("Invalid repository ID or file data")
        
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
            'content_url': file_data.get('content_url'),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Insert file record
        try:
            result = self.client.table('files').upsert(
                file_record,
                on_conflict='id'
            ).execute()
            
            if not result.data:
                logger.warning(f"No data returned from file insert operation: {file_record['file_path']}")
            
            logger.debug(f"File data stored in Supabase: {file_record['file_path']}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store file data: {e}")
            raise
    
    def store_embedding(self, file_id: str, content: str, embedding: List[float], 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content embedding in Supabase using pgvector.
        
        Args:
            file_id: File ID
            content: Text content
            embedding: Vector embedding
            metadata: Additional metadata
            
        Returns:
            Embedding ID
            
        Raises:
            ValueError: If embedding data is invalid
        """
        if not file_id or not embedding:
            raise ValueError("Invalid file ID or embedding")
        
        # Generate embedding ID
        embedding_id = str(uuid.uuid4())
        
        # Prepare embedding record
        embedding_record = {
            'id': embedding_id,
            'file_id': file_id,
            'content': content[:10000],  # Truncate content for database storage
            'embedding': embedding,
            'metadata': json.dumps(metadata) if metadata else None,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Insert embedding record
        try:
            result = self.client.table('file_embeddings').insert(
                embedding_record
            ).execute()
            
            if not result.data:
                logger.warning(f"No data returned from embedding insert operation for file {file_id}")
            
            logger.debug(f"Embedding stored in Supabase for file {file_id}")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            raise
    
    def search_similar_content(self, query_embedding: List[float], repo_id: Optional[str] = None,
                               top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            repo_id: Optional repository ID to limit search scope
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar content items with similarity scores
        """
        if not query_embedding:
            raise ValueError("Invalid query embedding")
        
        try:
            # Prepare RPC parameters
            params = {
                'query_embedding': query_embedding,
                'match_threshold': similarity_threshold,
                'match_count': top_k
            }
            
            if repo_id:
                params['repo_id'] = repo_id
            
            # Call the RPC function for similarity search
            result = self.client.rpc(
                'match_file_embeddings',
                params
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} similar content items")
                return result.data
            else:
                logger.info("No similar content found")
                return []
                
        except Exception as e:
            logger.error(f"Failed to search similar content: {e}")
            raise
    
    def search_similar_text(self, query_text: str, repo_id: Optional[str] = None,
                            top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar content using text query (generates embedding for you).
        
        Args:
            query_text: Query text
            repo_id: Optional repository ID to limit search scope
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar content items with similarity scores
        """
        if not query_text:
            raise ValueError("Invalid query text")
        
        try:
            # Prepare RPC parameters
            params = {
                'query_text': query_text,
                'match_threshold': similarity_threshold,
                'match_count': top_k
            }
            
            if repo_id:
                params['repo_id'] = repo_id
            
            # Call the RPC function for similarity search
            result = self.client.rpc(
                'search_files_by_text',
                params
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} similar content items for query: '{query_text}'")
                return result.data
            else:
                logger.info(f"No similar content found for query: '{query_text}'")
                return []
                
        except Exception as e:
            logger.error(f"Failed to search similar text: {e}")
            raise
    
    def get_repository_by_name(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """
        Get repository by name.
        
        Args:
            repo_name: Repository name
            
        Returns:
            Repository data or None if not found
        """
        if not repo_name:
            raise ValueError("Invalid repository name")
        
        try:
            result = self.client.table('repositories').select('*').eq('name', repo_name).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                logger.warning(f"Repository not found: {repo_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get repository by name: {e}")
            raise
    
    def get_files_by_repository(self, repo_id: str) -> List[Dict[str, Any]]:
        """
        Get all files for a repository.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            List of file data
        """
        if not repo_id:
            raise ValueError("Invalid repository ID")
        
        try:
            result = self.client.table('files').select('*').eq('repository_id', repo_id).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} files for repository {repo_id}")
                return result.data
            else:
                logger.warning(f"No files found for repository {repo_id}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get files by repository: {e}")
            raise
    
    def store_file_content(self, content: str, file_path: str) -> Optional[str]:
        """
        Store file content in Supabase Storage.
        
        Args:
            content: File content
            file_path: File path (used for naming)
            
        Returns:
            Public URL of the stored content or None if storage fails
        """
        if not content:
            return None
        
        try:
            # Generate a unique filename based on path and timestamp
            safe_path = file_path.replace('/', '_').replace('\\', '_')
            timestamp = int(time.time())
            unique_filename = f"{safe_path}_{timestamp}.txt"
            
            # Upload file content to Supabase Storage
            self.client.storage.from_("file_contents").upload(
                unique_filename, 
                content.encode('utf-8'),
                {"content-type": "text/plain"}
            )
            
            # Get public URL
            public_url = self.client.storage.from_("file_contents").get_public_url(unique_filename)
            
            logger.debug(f"File content stored in Supabase Storage: {unique_filename}")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to store file content: {e}")
            return None
    
    def retrieve_file_content(self, content_url: str) -> Optional[str]:
        """
        Retrieve file content from Supabase Storage.
        
        Args:
            content_url: URL of the content in Supabase Storage
            
        Returns:
            File content or None if retrieval fails
        """
        if not content_url:
            return None
        
        try:
            # Extract filename from URL
            filename = content_url.split('/')[-1]
            
            # Download content
            data = self.client.storage.from_("file_contents").download(filename)
            
            return data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to retrieve file content: {e}")
            return None
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for text using Supabase's built-in embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Vector embedding or None if generation fails
        """
        if not text:
            return None
        
        try:
            # Call Supabase Edge Function for embedding generation
            response = self.client.functions.invoke(
                "generate-embeddings",
                {"text": text}
            )
            
            if response and "embedding" in response:
                return response["embedding"]
            else:
                logger.warning("Embedding generation returned invalid response")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def process_repository(self, repo_data: Dict[str, Any]) -> Optional[str]:
        """
        Process complete repository data and store in Supabase.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Repository ID or None if processing fails
        """
        if not repo_data:
            raise ValueError("Invalid repository data")
        
        try:
            # Store repository metadata
            repo_id = self.store_repository(repo_data)
            
            # Process each file
            for file_entry in repo_data.get('files', []):
                # Check if file has content
                content = file_entry.get('content')
                if not content:
                    continue
                
                # Upload content to storage if not already uploaded
                if not file_entry.get('content_url'):
                    content_url = self.store_file_content(
                        content, 
                        file_entry.get('metadata', {}).get('path', 'unknown')
                    )
                    file_entry['content_url'] = content_url
                
                # Store file record
                file_id = self.store_file(repo_id, file_entry)
                
                # Generate and store embedding (if content is not too large)
                if len(content) <= 10000:  # Limit size for embedding generation
                    embedding = self.create_embedding(content)
                    if embedding:
                        self.store_embedding(
                            file_id, 
                            content, 
                            embedding,
                            file_entry.get('metadata')
                        )
            
            logger.info(f"Repository processed successfully: {repo_id}")
            return repo_id
            
        except Exception as e:
            logger.error(f"Failed to process repository: {e}")
            return None
    
    def process_repository_chunks(self, repo_data: Dict[str, Any], 
                                chunk_size: int = 5000) -> Optional[str]:
        """
        Process repository data with content chunking for better embedding.
        
        Args:
            repo_data: Repository data dictionary
            chunk_size: Maximum size of each content chunk
            
        Returns:
            Repository ID or None if processing fails
        """
        if not repo_data:
            raise ValueError("Invalid repository data")
        
        try:
            # Store repository metadata
            repo_id = self.store_repository(repo_data)
            
            # Process each file
            for file_entry in repo_data.get('files', []):
                # Check if file has content
                content = file_entry.get('content')
                if not content:
                    continue
                
                # Upload full content to storage if not already uploaded
                if not file_entry.get('content_url'):
                    content_url = self.store_file_content(
                        content, 
                        file_entry.get('metadata', {}).get('path', 'unknown')
                    )
                    file_entry['content_url'] = content_url
                
                # Store file record
                file_id = self.store_file(repo_id, file_entry)
                
                # Split content into chunks for embedding
                chunks = self._chunk_content(content, chunk_size)
                
                # Generate and store embeddings for each chunk
                for i, chunk in enumerate(chunks):
                    embedding = self.create_embedding(chunk)
                    if embedding:
                        # Add chunk metadata
                        chunk_metadata = dict(file_entry.get('metadata', {}))
                        chunk_metadata.update({
                            'chunk_index': i,
                            'chunk_count': len(chunks)
                        })
                        
                        self.store_embedding(
                            file_id,
                            chunk,
                            embedding,
                            chunk_metadata
                        )
            
            logger.info(f"Repository processed with chunking: {repo_id}")
            return repo_id
            
        except Exception as e:
            logger.error(f"Failed to process repository with chunking: {e}")
            return None
    
    def _chunk_content(self, content: str, max_length: int = 5000) -> List[str]:
        """
        Split content into chunks for embedding.
        
        Args:
            content: Text content
            max_length: Maximum length of each chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= max_length:
            return [content]
        
        # Split by lines for more natural chunks
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        
        # Handle zero magnitudes
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        # Calculate cosine similarity
        return dot_product / (magnitude_a * magnitude_b)


# Singleton instance for convenience
_db_instance = None

def get_db() -> EnhancedDatabase:
    """
    Get or create the EnhancedDatabase singleton instance.
    
    Returns:
        EnhancedDatabase instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = EnhancedDatabase()
    
    return _db_instance


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced database operations with Supabase and pgvector")
    parser.add_argument("action", choices=["test-connection", "search", "process"], help="Action to perform")
    parser.add_argument("--repo", help="Repository name or path")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--repo-id", help="Repository ID")
    
    args = parser.parse_args()
    
    try:
        db = EnhancedDatabase()
        
        if args.action == "test-connection":
            print("Connection to Supabase established successfully")
            
        elif args.action == "search" and args.query:
            results = db.search_similar_text(args.query, args.repo_id)
            
            print(f"\nSearch results for query: '{args.query}'")
            print("=" * 50)
            
            for i, result in enumerate(results):
                print(f"{i+1}. {result.get('file_path', 'Unknown')} (similarity: {result.get('similarity', 0):.4f})")
                print(f"   Preview: {result.get('content', '')[:100]}...")
                print()
                
        elif args.action == "process" and args.repo:
            # This would typically use repository data from ingest module
            print(f"To process a repository, use the ingest module and pass the result to EnhancedDatabase.process_repository()")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
