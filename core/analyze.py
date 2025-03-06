"""
Repository analysis module for PromptPilot.

This module handles generating embeddings for code files and analyzing the
code structure to identify key concepts and relationships between files.
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Callable, Coroutine
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import time
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import traceback

from openai import OpenAI, AsyncOpenAI  # Updated imports for newer OpenAI SDK

try:
    from postgrest.exceptions import APIError
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

from .enhanced_db import RepositoryDB
from .utils import timer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.analyze')

# Load environment variables
load_dotenv()

# Constants for embedding
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

class RepositoryAnalyzer:
    """Class to analyze a repository and generate embeddings for code files."""
    
    def __init__(self, 
                 repo_data_path: str,
                 output_dir: Optional[str] = None,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 chunk_size: int = 4000,
                 use_supabase: bool = True):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_data_path: Path to the repository data JSON file or directory
            output_dir: Output directory for analysis results
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum chunk size for embedding generation
            use_supabase: Whether to use Supabase for database operations
        """
        self.repo_data_path = repo_data_path
        
        # If repo_data_path is a directory, look for repository_data.json inside it
        if os.path.isdir(repo_data_path):
            self.repo_data_path = os.path.join(repo_data_path, "repository_data.json")
        
        # Get repository directory from repo_data_path
        self.repo_dir = os.path.dirname(self.repo_data_path)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.repo_dir
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define paths for embeddings
        self.file_embeddings_path = os.path.join(self.output_dir, "embeddings.json")
        self.embeddings_cache_path = os.path.join(self.output_dir, "embeddings_cache.json")
        self.embeddings_metadata_path = os.path.join(self.output_dir, "embeddings_metadata.json")
        
        # Set embedding model and chunk size
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        
        # Initialize OpenAI client if available
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_client = None
        self.async_openai_client = None
        self.openai_available = False
        if self.openai_api_key:
            try:
                # Initialize both sync and async clients
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.async_openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                self.openai_available = True
                logger.info(f"Using OpenAI embedding model: {embedding_model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI: {str(e)}")
                self.openai_available = False
        
        # Initialize Supabase client if available
        self.supabase = None
        self.use_supabase = use_supabase
        if use_supabase:
            try:
                from core.enhanced_db import get_db
                self.db = get_db()
                if not self.db or not self.db.is_available():
                    logger.warning("Supabase connection not available, using local embedding storage")
                    self.use_supabase = False
            except Exception as e:
                logger.error(f"Error initializing Supabase: {str(e)}")
                self.use_supabase = False

    def _get_file_content(self, file_entry: Dict[str, Any]) -> str:
        """
        Get the content of a file.
        
        Args:
            file_entry: Dictionary containing file information
            
        Returns:
            File content as a string
        """
        # Try to get content from content_url if available
        if file_entry.get('content_url') and self.supabase:
            try:
                # Extract filename from content_url
                url_parts = file_entry['content_url'].split('/')
                filename = url_parts[-1]
                
                # Download content from Supabase
                data = self.supabase.storage.from_("file_contents").download(filename)
                return data.decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to download content from Supabase: {str(e)}")
                # Fall back to content field
        
        # Use content field
        return file_entry.get('content', '')
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for text using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floating point values representing the embedding or empty list if generation fails
        """
        try:
            if not self.openai_available or not self.async_openai_client:
                logger.warning("OpenAI client not available for generating embeddings")
            return []
            
            # Use the new OpenAI API format
            response = await self.async_openai_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                return embedding
            else:
                logger.error("Empty response from OpenAI embeddings API")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _hash_content(self, content: str) -> str:
        """
        Generate a hash for content to detect changes.
        
        Args:
            content: File content
            
        Returns:
            Hash string
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _chunk_content(self, content: str, max_length: int = 4000) -> List[str]:
        """
        Split content into chunks for embedding.
        
        Args:
            content: File content
            max_length: Maximum length of each chunk
            
        Returns:
            List of content chunks
        """
        if len(content) <= max_length:
            return [content]
        
        # Split by lines first
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                # Add to current chunk
                current_chunk.append(line)
                current_length += line_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _store_embeddings_to_supabase(self, file_embeddings: Dict[str, Any]) -> bool:
        """
        Store file embeddings to Supabase database.
        
        Args:
            file_embeddings: Dictionary mapping file paths to their embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db or not self.use_supabase:
            logger.warning("Database connection not available, skipping Supabase storage")
            return False
            
        try:
            # Get repository ID from metadata
            repo_meta_path = os.path.join(self.repo_dir, "repository_metadata.json")
            if os.path.exists(repo_meta_path):
                with open(repo_meta_path, "r", encoding="utf-8") as f:
                    repo_metadata = json.load(f)
                    repo_id = repo_metadata.get("id")
                    
                    if not repo_id:
                        logger.warning("Repository ID not found in metadata, cannot store embeddings to Supabase")
                        return False
                    
                    # Extract just the embedding vectors for storage
                    embedding_vectors = {}
                    for file_path, embedding_data in file_embeddings.items():
                        embedding_vectors[file_path] = embedding_data["embedding"]
                    
                    # Store embeddings via database helper
                    success = self.db.store_file_embeddings(repo_id, embedding_vectors)
                    if success:
                        logger.info(f"Successfully stored {len(file_embeddings)} file embeddings to Supabase")
                    else:
                        logger.error("Failed to store file embeddings to Supabase")
                    
                    return success
            else:
                logger.warning(f"Repository metadata not found at {repo_meta_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing embeddings to Supabase: {e}")
            logger.error(traceback.format_exc())
            return False
    
    @timer
    def generate_file_embeddings(self) -> Dict[str, Any]:
        """
        Generate embeddings for files in the repository.
            
        Returns:
            Dictionary mapping file paths to their embeddings
        """
        try:
            # Load repository data
            if not os.path.exists(self.repo_data_path):
                logger.error(f"Repository data not found at {self.repo_data_path}")
                return {}
                
            with open(self.repo_data_path, "r", encoding="utf-8") as f:
                repo_data = json.load(f)
                
            if not repo_data.get("files"):
                logger.warning("No files found in repository data")
                return {}
                
            # Generate embeddings for each file with content
            file_embeddings = {}
            files_with_embeddings = 0
            
            for file_entry in repo_data["files"]:
                # Skip files without content
                if not file_entry.get("content"):
                    continue
            
                # Get and normalize file path
                file_path = file_entry["metadata"]["path"]
                normalized_path = file_path.replace('\\', '/')
            
                # Skip files that already have embeddings
                if normalized_path in file_embeddings:
                    continue
            
                # Generate embedding for file content
                content = file_entry["content"]
                embedding = asyncio.run(self._generate_embedding(content))
                
                if embedding and len(embedding) > 0:
                    file_embeddings[normalized_path] = {
                        "path": normalized_path,
                        "embedding": embedding,
                        "content_hash": self._hash_content(content)
                    }
                    files_with_embeddings += 1
            
            logger.info(f"Generated embeddings for {files_with_embeddings} files")
            
            # Save embeddings locally
            with open(self.file_embeddings_path, "w", encoding="utf-8") as f:
                json.dump(file_embeddings, f)
                
            logger.info(f"Saved file embeddings to {self.file_embeddings_path}")
            
            # Store in Supabase if available
            if self.use_supabase:
                success = self._store_embeddings_to_supabase(file_embeddings)
                if not success:
                    logger.warning("Failed to store embeddings in Supabase")
            
            return file_embeddings
            
        except Exception as e:
            logger.error(f"Error generating file embeddings: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def find_similar_files(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find files similar to a query using embeddings.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar files with scores and metadata
        """
        # First, try to use Supabase for similarity search
        if self.use_supabase and self.supabase:
            try:
                results = self._search_similar_supabase(query, top_k)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Supabase similarity search failed: {str(e)}")
        
        # Fall back to local similarity search
        return self._search_similar_local(query, top_k)
    
    async def _search_similar_supabase(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar files using Supabase's pgvector.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar files with scores and metadata
        """
        if not self.supabase:
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embedding(query)
            
            # Get repository ID
            embeddings_metadata_path = os.path.join(self.output_dir, 'embeddings_cache_metadata.json')
            repository_id = None
            
            if os.path.exists(embeddings_metadata_path):
                with open(embeddings_metadata_path, 'r', encoding='utf-8') as f:
                    cache_metadata = json.load(f)
                    repository_id = cache_metadata.get('repository_id')
            
            if not repository_id:
                # Try to find repository by name
                repo_name = repo_data.get('name', '')
                if repo_name:
                    response = self.supabase.table('repositories').select('id').eq('name', repo_name).execute()
                    if response.data:
                        repository_id = response.data[0].get('id')
            
            if not repository_id:
                logger.warning("Repository ID not found for similarity search")
                return []
            
            # Use RPC call to fetch similar files
            response = self.supabase.rpc(
                'match_file_embeddings',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.5,
                    'match_count': top_k,
                    'repo_id': repository_id
                }
            ).execute()
            
            # Process results
            results = []
            for item in response.data:
                results.append({
                    'file_path': item.get('file_path', ''),
                    'similarity': item.get('similarity', 0),
                    'content': item.get('content', ''),
                    'metadata': item.get('metadata', {})
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar files in Supabase: {e}")
            return []
    
    def _search_similar_local(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar files using local embeddings.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar files with scores and metadata
        """
        # Load embeddings
        embeddings_path = os.path.join(self.output_dir, 'embeddings_cache.json')
        
        if not os.path.exists(embeddings_path):
            logger.warning(f"Embeddings cache not found at {embeddings_path}")
            return []
        
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            # Generate embedding for the query
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            query_embedding = loop.run_until_complete(self._generate_embedding(query))
            loop.close()
            
            # Calculate similarity with each file
            similarities = []
            
            for file_path, file_data in embeddings_data.get('files', {}).items():
                for i, chunk_data in enumerate(file_data.get('chunks', [])):
                    chunk_embedding = chunk_data.get('embedding')
                    chunk_content = chunk_data.get('content')
                    
                    if not chunk_embedding or not chunk_content:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    
                    similarities.append({
                        'file_path': file_path,
                        'chunk_index': i,
                        'similarity': similarity,
                        'content': chunk_content,
                        'metadata': file_data.get('metadata', {})
                    })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to search similar files locally: {e}")
            return []
    
    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (0-1)
        """
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def analyze_repository(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze the repository and generate embeddings.
        
        Args:
            force_refresh: Whether to force regeneration of all data
            
        Returns:
            Dictionary containing analysis results
        """
        # Generate embeddings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embeddings_data = loop.run_until_complete(self.generate_file_embeddings())
        loop.close()
        
        # Perform additional analysis if needed
        # TODO: Add file similarity analysis, clustering, etc.
        
        # Save analysis results
        analysis_path = os.path.join(self.output_dir, 'repository_analysis.json')
        
        analysis_results = {
            'repository_name': repo_data.get('name', 'unknown'),
            'embedding_model': self.embedding_model,
            'file_count': len(embeddings_data),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f)
        
        logger.info(f"Repository analysis saved to {analysis_path}")
        
        return analysis_results


# Standalone function for simpler use cases
def analyze_repository(repo_data_path: str, output_dir: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Analyze a repository and generate embeddings.
    
    Args:
        repo_data_path: Path to the repository data JSON file
        output_dir: Directory to store analysis data
        force_refresh: Whether to force regeneration of all data
        
    Returns:
        Dictionary containing analysis results
    """
    analyzer = RepositoryAnalyzer(repo_data_path, output_dir)
    return analyzer.analyze_repository(force_refresh)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a repository for PromptPilot")
    parser.add_argument("repo_data_path", help="Path to the repository data JSON file")
    parser.add_argument("--output", help="Directory to store analysis data")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all data")
    parser.add_argument("--query", help="Search query for similar files")
    
    args = parser.parse_args()
    
    try:
        analyzer = RepositoryAnalyzer(args.repo_data_path, args.output)
        
        if args.query:
            # Find similar files
            similar_files = analyzer.find_similar_files(args.query)
            
            print(f"\nTop similar files for query: '{args.query}'")
            print("=" * 50)
            
            for i, result in enumerate(similar_files):
                print(f"{i+1}. {result['file_path']} (similarity: {result['similarity']:.4f})")
                print(f"   {result['metadata'].get('lines', 0)} lines, {result['metadata'].get('size_bytes', 0)/1024:.1f}KB")
                print(f"   Preview: {result['content'][:100]}...")
                print()
                
        else:
            # Run full analysis
            results = analyzer.analyze_repository(args.force)
            print(f"\nRepository analysis complete: {results['file_count']} files processed")
            
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise

