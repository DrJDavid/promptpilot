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
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            logger.warning("Embeddings will not be generated")
        
        # Check if API key is set
        self.openai_available = bool(openai_api_key)
    
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
    
    def generate_embeddings(self, repo_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate embeddings for each file in the repository.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Dictionary mapping file paths to embedding vectors
        """
        if not self.openai_available:
            logger.warning("OpenAI API key not available, skipping embeddings generation")
            return {}
        
        embeddings = {}
        
        # Create client
        client = openai.OpenAI()
        
        # Process each file
        for file_entry in tqdm(repo_data['files'], desc="Generating embeddings"):
            file_path = file_entry['metadata']['path']
            
            try:
                # Get file content
                content = file_entry['content']
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Truncate content if too long (OpenAI has token limits)
                # text-embedding-3-small can handle 8191 tokens
                if len(content) > 32000:  # Rough character estimate for token limit
                    logger.warning(f"Truncating {file_path} for embedding (too long)")
                    content = content[:32000]
                
                # Generate embedding
                response = client.embeddings.create(
                    input=content,
                    model=self.embedding_model
                )
                
                # Extract embedding
                embedding = response.data[0].embedding
                embeddings[file_path] = embedding
                
                # Rate limiting (avoid hitting API limits)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embedding for {file_path}: {str(e)}")
        
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
        embeddings = self.generate_embeddings(repo_data)
        
        # Analyze file similarities
        similarities = self.analyze_file_similarities(embeddings)
        
        # Analyze code structure
        structure = self.analyze_code_structure(repo_data)
        
        # Combine all analysis results
        analysis = {
            "file_similarities": similarities,
            "code_structure": structure
        }
        
        # Save results
        self._save_embeddings(embeddings)
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
