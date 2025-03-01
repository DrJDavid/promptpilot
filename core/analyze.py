"""
Analysis module for PromptPilot.

This module handles generating embeddings and analyzing code structure.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import openai
import hashlib
import asyncio
import re
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.analyze')

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class RepositoryAnalyzer:
    """Class to analyze repository content and generate embeddings."""
    
    def __init__(self, repository_dir: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the repository analyzer.
        
        Args:
            repository_dir: Directory containing repository data (.promptpilot folder)
            embedding_model: OpenAI embedding model to use
        """
        self.repository_dir = repository_dir
        self.embedding_model = embedding_model
        
        # Check if repository data exists
        self.data_path = os.path.join(repository_dir, 'repository_data.json')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Repository data not found at {self.data_path}")
        
        # Output paths
        self.embeddings_path = os.path.join(repository_dir, 'embeddings.json')
        self.analysis_path = os.path.join(repository_dir, 'analysis.json')
        
        # Initialize OpenAI client
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            logger.warning("Embeddings will not be generated")
            self.client = None
            self.openai_available = False
        else:
            # Initialize OpenAI client using the successful approach from our test
            try:
                from openai import OpenAI, AsyncOpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.async_client = AsyncOpenAI(api_key=self.api_key)
                self.openai_available = True
                logger.info(f"OpenAI client initialized with model: {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.openai_available = False
                logger.warning("Embeddings will use fallback generation")
    
    def _load_repository_data(self) -> Dict[str, Any]:
        """
        Load repository data from disk.
        
        Returns:
            Repository data dictionary
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Dictionary mapping file paths to embedding vectors
        """
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        
        logger.info(f"Embeddings saved to {self.embeddings_path}")
    
    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Save analysis results to disk.
        
        Args:
            analysis: Analysis results dictionary
        """
        with open(self.analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {self.analysis_path}")
    
    async def get_embedding_async(self, content: str, client) -> List[float]:
        """
        Get embedding from OpenAI API asynchronously.
        
        Args:
            content: Text content to embed
            client: Async OpenAI client
            
        Returns:
            Embedding vector
        """
        try:
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=content
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Async embedding API error: {e}")
            # Return None to allow fallback to mock embedding
            return None

    def _generate_simple_embedding(self, content: str) -> List[float]:
        """
        Generate a simple embedding based on hash of content (for fallback).
        This is used when OpenAI API is not available or fails.
        
        Args:
            content: Text content to embed
            
        Returns:
            Mock embedding vector (1536 dimensions)
        """
        logger.info("Generating mock embedding as fallback")
        
        # Use hash of content to seed a deterministic random embedding
        import hashlib
        import random
        
        # Generate a stable hash from the content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Use the hash to seed a random number generator for deterministic output
        random.seed(content_hash)
        
        # Generate 1536 values (OpenAI embedding size) between -1 and 1
        mock_embedding = [random.uniform(-1, 1) for _ in range(1536)]
        
        # Normalize to unit vector
        norm = sum(x*x for x in mock_embedding) ** 0.5
        if norm > 0:
            mock_embedding = [x/norm for x in mock_embedding]
            
        logger.warning("Using mock embedding. Similarity calculations will not be meaningful.")
        return mock_embedding
        
    async def _process_file_batch(self, files_batch, client):
        """Process a batch of files concurrently."""
        tasks = []
        for file_path, content, content_hash in files_batch:
            # Skip empty files
            if not content.strip():
                continue
                
            # Truncate content if too long
            if len(content) > 32000:
                logger.warning(f"Truncating {file_path} for embedding (too long)")
                content = content[:32000]
                
            # Create task
            task = asyncio.create_task(
                self.get_embedding_async(content, client)
            )
            tasks.append((file_path, content_hash, task))
        
        # Wait for all embeddings to complete
        results = []
        for file_path, content_hash, task in tasks:
            try:
                embedding = await task
                if embedding is not None:  # Only add successful embeddings
                    results.append((file_path, content_hash, embedding))
                else:
                    # If API fails, generate a simple embedding
                    try:
                        with open(os.path.join(os.path.dirname(self.repository_dir), file_path), 'r', encoding='utf-8') as f:
                            content = f.read()
                        simple_embedding = self._generate_simple_embedding(content)
                        results.append((file_path, content_hash, simple_embedding))
                        logger.warning(f"Using simple embedding for {file_path} due to API error")
                    except Exception as e:
                        logger.error(f"Could not generate simple embedding for {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        return results

    async def generate_embeddings_async(self, files_to_process: List[Tuple[str, str, str]]) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
        """
        Generate embeddings for files asynchronously.
        
        Args:
            files_to_process: List of (file_path, content, content_hash) tuples
                
        Returns:
            Tuple of (embeddings dict, cache metadata dict)
        """
        embeddings = {}
        cache_metadata = {}
        
        # Use the pre-initialized async client
        client = self.async_client
        
        # Process files in batches to control concurrency
        batch_size = 5  # Adjust based on rate limits
        batches = [files_to_process[i:i+batch_size] for i in range(0, len(files_to_process), batch_size)]
        
        for batch in tqdm(batches, desc="Generating embeddings (batched)"):
            batch_results = await self._process_file_batch(batch, client)
            
            # Process results
            for file_path, content_hash, embedding in batch_results:
                embeddings[file_path] = embedding
                cache_metadata[file_path] = content_hash
            
            # Add a small delay between batches to avoid rate limits
            await asyncio.sleep(0.5)
        
        return embeddings, cache_metadata
    
    def generate_embeddings(self, repo_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate embeddings for each file in the repository with caching.
        Now with async support for faster processing.
        
        Args:
            repo_data: Repository data dictionary
                
        Returns:
            Dictionary mapping file paths to embedding vectors
        """
        embeddings = {}
        
        # Check if OpenAI API is available
        if not self.openai_available:
            logger.warning("OpenAI API not available, using simple embeddings")
            # Generate simple embeddings for all files
            for file_entry in repo_data['files']:
                file_path = file_entry['metadata']['path']
                content = file_entry['content']
                embeddings[file_path] = self._generate_simple_embedding(content)
            return embeddings
            
        # Check for cache file
        cache_path = os.path.join(self.repository_dir, 'embeddings_cache.json')
        cache_metadata_path = os.path.join(self.repository_dir, 'embeddings_cache_metadata.json')
        
        # Try to load from cache
        cache_exists = os.path.exists(cache_path)
        cache_valid = False
        cache_metadata = {}
        
        if cache_exists:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    embeddings = json.load(f)
                    
                with open(cache_metadata_path, 'r', encoding='utf-8') as f:
                    cache_metadata = json.load(f)
                    
                logger.info(f"Loaded {len(embeddings)} embeddings from cache")
                cache_valid = True
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                embeddings = {}
                cache_metadata = {}
        
        # Prepare files to process
        files_to_process = []
        cache_hits = 0
        
        for file_entry in repo_data['files']:
            file_path = file_entry['metadata']['path']
            content = file_entry['content']
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Skip if in cache and content hasn't changed
            if cache_valid and file_path in embeddings and file_path in cache_metadata:
                if cache_metadata[file_path] == content_hash:
                    cache_hits += 1
                    continue
            
            # Add to processing list
            files_to_process.append((file_path, content, content_hash))
        
        if cache_hits > 0:
            logger.info(f"Using {cache_hits} cached embeddings (cache hit rate: {cache_hits/len(repo_data['files']):.1%})")
        
        # Process files that need new embeddings
        if files_to_process:
            logger.info(f"Generating embeddings for {len(files_to_process)} files")
            
            try:
                # Try to use async version
                # Get event loop or create one
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run async embedding generation
                new_embeddings, new_cache_metadata = loop.run_until_complete(
                    self.generate_embeddings_async(files_to_process)
                )
                
                # Update embeddings and cache metadata
                embeddings.update(new_embeddings)
                cache_metadata.update(new_cache_metadata)
                
                logger.info(f"Generated {len(new_embeddings)} embeddings asynchronously")
                
            except Exception as e:
                logger.warning(f"Async embedding generation failed: {e}. Falling back to sync version.")
                
                # Fall back to synchronous version using the pre-initialized client
                for file_path, content, content_hash in tqdm(files_to_process, desc="Generating embeddings"):
                    try:
                        # Skip empty files
                        if not content.strip():
                            continue
                        
                        # Truncate content if too long
                        if len(content) > 32000:
                            logger.warning(f"Truncating {file_path} for embedding (too long)")
                            content = content[:32000]
                        
                        # Generate embedding
                        response = self.client.embeddings.create(
                            input=content,
                            model=self.embedding_model
                        )
                        
                        # Extract embedding
                        embedding = response.data[0].embedding
                        embeddings[file_path] = embedding
                        
                        # Update cache metadata
                        cache_metadata[file_path] = content_hash
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"Error generating embedding for {file_path}: {str(e)}")
                        # Use simple embedding as fallback
                        embeddings[file_path] = self._generate_simple_embedding(content)
                        cache_metadata[file_path] = content_hash
        
        # Save updated cache
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f)
                
            with open(cache_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_metadata, f)
                
            logger.info(f"Saved {len(embeddings)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")
        
        return embeddings
    
    def analyze_file_similarities(self, embeddings: Dict[str, List[float]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze file similarities based on embeddings.
        
        Args:
            embeddings: Dictionary mapping file paths to embedding vectors
            
        Returns:
            Dictionary mapping file paths to lists of similar files
        """
        if not embeddings:
            return {}
        
        # Convert embeddings to numpy arrays for faster computation
        file_paths = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[path] for path in file_paths])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Build similarity dictionary
        similarities = {}
        for i, path in enumerate(file_paths):
            # Get similarity scores for this file
            scores = similarity_matrix[i]
            
            # Create list of (file_path, similarity_score) pairs, sorted by score
            similar_files = [
                {"path": file_paths[j], "similarity": float(scores[j])}
                for j in range(len(file_paths))
                if i != j  # Exclude self
            ]
            
            # Sort by similarity (descending)
            similar_files.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Keep top 10 similar files
            similarities[path] = similar_files[:10]
        
        return similarities
    
    def analyze_code_structure(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic code structure analysis.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Dictionary containing code structure information
        """
        # Initialize counters and collections
        imports = defaultdict(set)
        function_count = defaultdict(int)
        class_count = defaultdict(int)
        direct_dependencies = defaultdict(set)
        
        # Regular expressions would be more robust, but keeping it simple
        import_prefixes = [
            "import ", "from ", "require(", "using ", "#include ",
            "use ", "extern crate ", "const "
        ]
        
        function_prefixes = [
            "def ", "function ", "fn ", "func ", "void ", "int ", "string ",
            "public ", "private ", "protected ", "static ", "async "
        ]
        
        class_prefixes = [
            "class ", "interface ", "struct ", "enum ", "trait ", "type "
        ]
        
        # Process each file
        for file_entry in repo_data['files']:
            path = file_entry['metadata']['path']
            content = file_entry['content']
            ext = file_entry['metadata'].get('extension', '')
            
            # Process content line by line
            for line in content.splitlines():
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith(("#", "//", "/*", "*", "'")):
                    continue
                
                # Check for imports
                for prefix in import_prefixes:
                    if line.startswith(prefix):
                        imports[path].add(line)
                        
                        # Extract potential direct dependency
                        words = line.split()
                        if len(words) > 1:
                            potential_dep = words[1].strip('";,')
                            direct_dependencies[path].add(potential_dep)
                
                # Check for function definitions
                for prefix in function_prefixes:
                    if prefix in line and "(" in line and ")" in line:
                        function_count[path] += 1
                        break
                
                # Check for class definitions
                for prefix in class_prefixes:
                    if line.startswith(prefix) or f" {prefix}" in line:
                        class_count[path] += 1
                        break
        
        # Convert sets to lists for JSON serialization
        imports_dict = {k: list(v) for k, v in imports.items()}
        dependencies_dict = {k: list(v) for k, v in direct_dependencies.items()}
        
        # Create analysis dictionary
        analysis = {
            "imports": imports_dict,
            "function_count": function_count,
            "class_count": class_count,
            "direct_dependencies": dependencies_dict
        }
        
        return analysis
    
    def analyze_repository(self) -> Dict[str, Any]:
        """
        Analyze the repository and generate all necessary information.
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing repository in {self.repository_dir}")
        
        # Load repository data
        repo_data = self._load_repository_data()
        
        # Generate embeddings
        try:
            embeddings = self.generate_embeddings(repo_data)
            if not embeddings and self.openai_available:
                logger.warning("Failed to generate embeddings despite having API key.")
                logger.warning("Check your API key and connectivity.")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.warning("Proceeding with empty embeddings.")
            embeddings = {}
        
        # Analyze code structure
        try:
            structure_analysis = self.analyze_code_structure(repo_data)
        except Exception as e:
            logger.error(f"Error analyzing code structure: {e}")
            structure_analysis = {}
        
        # Analyze file similarities based on embeddings
        try:
            if embeddings:
                similarities = self.analyze_file_similarities(embeddings)
                logger.info(f"Generated similarity analysis for {len(similarities)} files")
            else:
                logger.warning("Skipping similarity analysis (no embeddings available)")
                similarities = {}
        except Exception as e:
            logger.error(f"Error analyzing file similarities: {e}")
            similarities = {}
        
        # Create main analysis object
        analysis = {
            "file_similarities": similarities,
            "code_structure": structure_analysis,
        }
        
        # Save analysis to disk
        self._save_analysis(analysis)
        
        return analysis
    
    def find_relevant_files(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find files most relevant to a query using embeddings.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing file paths and similarity scores
        """
        if not self.openai_available:
            logger.warning("OpenAI API key not available, using fallback for file search")
            return self._fallback_file_search(query, top_k)
        
        # Check if embeddings exist
        if not os.path.exists(self.embeddings_path):
            logger.warning("Embeddings not found, using fallback for file search")
            return self._fallback_file_search(query, top_k)
        
        try:
            # Load embeddings
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            
            if not embeddings:
                return self._fallback_file_search(query, top_k)
            
            # Generate query embedding
            client = openai.OpenAI()
            response = client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            query_embedding = response.data[0].embedding
            
            # Convert embeddings to numpy arrays
            file_paths = list(embeddings.keys())
            embedding_matrix = np.array([embeddings[path] for path in file_paths])
            query_vector = np.array(query_embedding)
            
            # Calculate similarities
            similarities = cosine_similarity([query_vector], embedding_matrix)[0]
            
            # Create similarity results
            results = [
                {"path": file_paths[i], "similarity": float(similarities[i])}
                for i in range(len(file_paths))
            ]
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding relevant files: {str(e)}")
            return self._fallback_file_search(query, top_k)
    
    def _fallback_file_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback method for finding relevant files when embeddings are not available.
        Uses simple keyword matching.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing file paths and match scores
        """
        logger.info("Using fallback file search (keyword matching)")
        
        # Load repository data
        repo_data = self._load_repository_data()
        
        # Normalize query
        query_terms = query.lower().split()
        
        # Calculate match scores for each file
        results = []
        for file_entry in repo_data['files']:
            path = file_entry['metadata']['path']
            content = file_entry['content'].lower()
            
            # Simple scoring: count occurrences of query terms in content
            score = sum(content.count(term) for term in query_terms)
            
            # Boost score for terms in file path
            path_lower = path.lower()
            path_score = sum(5 if term in path_lower else 0 for term in query_terms)
            
            total_score = score + path_score
            if total_score > 0:
                results.append({"path": path, "similarity": total_score})
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Normalize scores to 0-1 range
        if results:
            max_score = max(r["similarity"] for r in results)
            for r in results:
                r["similarity"] = r["similarity"] / max_score
        
        # Return top_k results
        return results[:top_k]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a repository for PromptPilot")
    parser.add_argument("repository_dir", help="Directory containing repository data (.promptpilot folder)")
    
    args = parser.parse_args()
    
    try:
        analyzer = RepositoryAnalyzer(args.repository_dir)
        analysis = analyzer.analyze_repository()
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {analyzer.analysis_path}")
        
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise
