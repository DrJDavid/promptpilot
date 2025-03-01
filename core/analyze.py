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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.analyze')

# Load environment variables
load_dotenv()

try:
    import openai
except ImportError:
    openai = None
    logger.warning("OpenAI package not installed, will use simple embedding instead")


class RepositoryAnalyzer:
    """Class to analyze a repository and generate embeddings for code files."""
    
    def __init__(self, 
                 repo_data_path: str,
                 output_dir: Optional[str] = None,
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 4000,
                 use_supabase: bool = True):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_data_path: Path to the repository data JSON file
            output_dir: Directory to store analysis data
            embedding_model: Name of the embedding model to use
            chunk_size: Maximum number of tokens per chunk
            use_supabase: Whether to use Supabase for embedding generation and storage
        """
        self.repo_data_path = os.path.abspath(repo_data_path)
        
        # Load repository data
        with open(self.repo_data_path, 'r', encoding='utf-8') as f:
            self.repo_data = json.load(f)
        
        # Set output directory
        repo_dir = os.path.dirname(self.repo_data_path)
        self.output_dir = os.path.abspath(output_dir) if output_dir else repo_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set embedding model name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.openai_client = None
        self.use_supabase = use_supabase
        self.supabase = None
        
        # Initialize API clients
        if openai and os.environ.get("OPENAI_API_KEY") and not self.use_supabase:
            try:
                self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info(f"Using OpenAI embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
                self.openai_client = None
        
        # Initialize Supabase client if requested
        if self.use_supabase:
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            
            if supabase_url and supabase_key:
                try:
                    self.supabase = create_client(supabase_url, supabase_key)
                    logger.info("Supabase client initialized for embedding generation and storage")
                except Exception as e:
                    logger.error(f"Failed to initialize Supabase client: {e}")
                    self.supabase = None
            else:
                logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set, falling back to OpenAI")
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
    
    def _compute_simple_embedding(self, text: str) -> List[float]:
        """
        Compute a simple embedding for text using a hashing method.
        This is a fallback when OpenAI API is not available.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        # Create a hash from the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Use the hash to seed numpy's random number generator
        np.random.seed(int.from_bytes(hash_bytes[:4], byteorder='little'))
        
        # Generate a random vector of specific dimensions (e.g., 1536 for text-embedding-3-small)
        dimensions = 1536
        embedding = np.random.normal(0, 1, dimensions).tolist()
        
        return embedding
    
    async def _generate_embedding_openai(self, text: str) -> List[float]:
        """
        Generate an embedding using OpenAI's API.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        if not self.openai_client:
            return self._compute_simple_embedding(text)
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding generation failed: {str(e)}")
            return self._compute_simple_embedding(text)
    
    async def _generate_embedding_supabase(self, text: str) -> List[float]:
        """
        Generate an embedding using Supabase's pgvector.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        if not self.supabase:
            return await self._generate_embedding_openai(text)
        
        try:
            # Use Supabase's Edge Function to generate embedding
            response = await asyncio.to_thread(
                self.supabase.functions.invoke,
                "generate-embeddings",
                {"text": text}
            )
            
            if response and "embedding" in response:
                return response["embedding"]
            else:
                logger.warning("Supabase embedding generation returned invalid response")
                return await self._generate_embedding_openai(text)
                
        except Exception as e:
            logger.warning(f"Supabase embedding generation failed: {str(e)}")
            return await self._generate_embedding_openai(text)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        if self.use_supabase and self.supabase:
            return await self._generate_embedding_supabase(text)
        else:
            return await self._generate_embedding_openai(text)
    
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
    
    def _store_embeddings_to_supabase(self, embeddings_data: Dict[str, Any]) -> bool:
        """
        Store embeddings data to Supabase.
        
        Args:
            embeddings_data: Dictionary containing embedding information
            
        Returns:
            True if successful, False otherwise
        """
        if not self.supabase:
            return False
            
        try:
            # Prepare data for insertion
            repository_id = embeddings_data.get('repository_id', str(uuid.uuid4()))
            repository_name = self.repo_data.get('name', 'unnamed')
            
            # Store repository record
            repo_result = self.supabase.table('repositories').upsert({
                'id': repository_id,
                'name': repository_name,
                'path': self.repo_data.get('path', ''),
                'file_count': self.repo_data.get('file_count', 0),
                'processed_date': datetime.now().isoformat()
            }).execute()
            
            # Store file embeddings
            for file_path, file_data in embeddings_data.get('files', {}).items():
                for i, chunk_data in enumerate(file_data.get('chunks', [])):
                    # Prepare embedding record
                    embedding_record = {
                        'id': str(uuid.uuid4()),
                        'repository_id': repository_id,
                        'file_path': file_path,
                        'chunk_index': i,
                        'content': chunk_data.get('content', ''),
                        'embedding': chunk_data.get('embedding', []),
                        'content_hash': chunk_data.get('hash', ''),
                        'metadata': {
                            'file_size': file_data.get('metadata', {}).get('size_bytes', 0),
                            'extension': file_data.get('metadata', {}).get('extension', ''),
                            'lines': file_data.get('lines', 0)
                        }
                    }
                    
                    # Insert embedding record
                    self.supabase.table('file_embeddings').upsert(
                        embedding_record,
                        on_conflict='id'
                    ).execute()
            
            logger.info(f"Successfully stored embeddings for {repository_name} in Supabase")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings in Supabase: {e}")
            return False
    
    async def generate_embeddings(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate embeddings for all files in the repository.
        
        Args:
            force_refresh: Whether to force regeneration of all embeddings
            
        Returns:
            Dictionary containing embedding information
        """
        logger.info(f"Generating embeddings for repository: {self.repo_data.get('name', 'unknown')}")
        
        # Check if OpenAI API is available
        if not self.openai_client and not self.supabase:
            logger.warning("OpenAI API and Supabase not available, using simple embedding instead")
        
        # Try to load existing embeddings
        embeddings_path = os.path.join(self.output_dir, 'embeddings_cache.json')
        embeddings_metadata_path = os.path.join(self.output_dir, 'embeddings_cache_metadata.json')
        
        embeddings_data = {
            'repository_id': str(uuid.uuid4()),
            'model': self.embedding_model,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'files': {}
        }
        
        # Load existing cache if available
        if os.path.exists(embeddings_path) and os.path.exists(embeddings_metadata_path) and not force_refresh:
            try:
                with open(embeddings_path, 'r', encoding='utf-8') as f:
                    cached_embeddings = json.load(f)
                
                with open(embeddings_metadata_path, 'r', encoding='utf-8') as f:
                    cache_metadata = json.load(f)
                
                # Check if the cache is valid
                if cache_metadata.get('model') == self.embedding_model:
                    embeddings_data = cached_embeddings
                    logger.info(f"Loaded existing embeddings cache from {embeddings_path}")
                    
                    # Keep existing repository ID
                    if 'repository_id' in cache_metadata:
                        embeddings_data['repository_id'] = cache_metadata['repository_id']
            
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {str(e)}")
        
        # Get the list of files to process
        files_to_process = []
        content_hashes = {}
        
        for file_entry in self.repo_data.get('files', []):
            file_path = file_entry.get('metadata', {}).get('path', '')
            
            if not file_path:
                continue
            
            # Get file content
            content = self._get_file_content(file_entry)
            
            if not content:
                continue
            
            # Compute content hash
            content_hash = self._hash_content(content)
            content_hashes[file_path] = content_hash
            
            # Check if we need to process this file
            need_processing = True
            
            # Skip if the file is already in the cache and the content hasn't changed
            if file_path in embeddings_data.get('files', {}) and not force_refresh:
                cached_hash = embeddings_data['files'][file_path].get('hash')
                if cached_hash == content_hash:
                    need_processing = False
                    logger.debug(f"Skipping unchanged file: {file_path}")
            
            if need_processing:
                files_to_process.append((file_path, content, file_entry))
        
        # Generate embeddings for files that need processing
        if files_to_process:
            logger.info(f"Generating embeddings for {len(files_to_process)} files")
            
            # Create embedding tasks
            tasks = []
            
            for file_path, content, file_entry in files_to_process:
                # Split content into chunks
                chunks = self._chunk_content(content, self.chunk_size)
                
                # Create tasks for embedding generation
                for chunk in chunks:
                    tasks.append((file_path, chunk, content_hashes[file_path], file_entry))
            
            # Process tasks in parallel
            chunk_results = []
            
            async def process_chunk(task_data):
                file_path, chunk_content, content_hash, file_entry = task_data
                embedding = await self._generate_embedding(chunk_content)
                return {
                    'file_path': file_path,
                    'content': chunk_content,
                    'hash': content_hash,
                    'embedding': embedding,
                    'file_entry': file_entry
                }
            
            # Run tasks with progress bar
            async_tasks = [process_chunk(task) for task in tasks]
            for result in await tqdm_asyncio.gather(*async_tasks, desc="Generating embeddings"):
                chunk_results.append(result)
            
            # Organize results by file
            for result in chunk_results:
                file_path = result['file_path']
                file_entry = result['file_entry']
                
                # Initialize file entry if needed
                if file_path not in embeddings_data['files']:
                    embeddings_data['files'][file_path] = {
                        'metadata': file_entry.get('metadata', {}),
                        'hash': result['hash'],
                        'lines': file_entry.get('lines', 0),
                        'chunks': []
                    }
                
                # Add chunk data
                embeddings_data['files'][file_path]['chunks'].append({
                    'content': result['content'],
                    'embedding': result['embedding'],
                    'hash': result['hash']
                })
            
            # Save embeddings to disk
            with open(embeddings_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f)
            
            # Save metadata
            cache_metadata = {
                'model': self.embedding_model,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'file_count': len(embeddings_data['files']),
                'repository_id': embeddings_data.get('repository_id')
            }
            
            with open(embeddings_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_metadata, f)
            
            logger.info(f"Embeddings saved to {embeddings_path}")
        else:
            logger.info("No files need embedding updates")
        
        # Store embeddings in Supabase if enabled
        if self.use_supabase and self.supabase:
            self._store_embeddings_to_supabase(embeddings_data)
        
        return embeddings_data
    
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
                repo_name = self.repo_data.get('name', '')
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
        embeddings_data = loop.run_until_complete(self.generate_embeddings(force_refresh))
        loop.close()
        
        # Perform additional analysis if needed
        # TODO: Add file similarity analysis, clustering, etc.
        
        # Save analysis results
        analysis_path = os.path.join(self.output_dir, 'repository_analysis.json')
        
        analysis_results = {
            'repository_name': self.repo_data.get('name', 'unknown'),
            'embedding_model': self.embedding_model,
            'file_count': len(embeddings_data.get('files', {})),
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
