"""
Enhanced prompt generator module for PromptPilot.

This module handles generating optimized prompts for code generation tasks,
using repository context and the Gemini API.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random
import traceback

import google.generativeai as genai
from dotenv import load_dotenv

# Import the database connection
from core.enhanced_db import get_db

# Try to import sgqlc, but don't fail if it's not working correctly
try:
    from sgqlc.operation import Operation
    from sgqlc.endpoint.http import HTTPEndpoint
    from core.schema import schema, Query
    GRAPHQL_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logging.warning(f"GraphQL functionality not available: {str(e)}")
    GRAPHQL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.prompt_generator')

# Load environment variables
load_dotenv()

# Default models
DEFAULT_MODEL = "gemini-1.5-flash"
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.2
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_TOP_K_FILES = 5
SYSTEM_PROMPT = "You are a helpful assistant."


class PromptGenerator:
    """Class to generate optimized prompts for code generation tasks."""
    
    def __init__(self, repository_dir: str, model: str = DEFAULT_MODEL):
        """
        Initialize the prompt generator.
        
        Args:
            repository_dir: Directory containing repository data (.promptpilot folder)
            model: Gemini model to use for prompt generation
        """
        self.repository_dir = repository_dir
        self.model_name = model
        
        # Get Supabase URL and key from environment
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not self.supabase_url:
            logger.warning("SUPABASE_URL environment variable not set.")
        if not self.supabase_key:
            logger.warning("SUPABASE_KEY environment variable not set.")
            
        # Initialize GraphQL endpoint if available
        self.endpoint = None
        if GRAPHQL_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                graphql_endpoint = f"{self.supabase_url}/graphql/v1"
                self.endpoint = HTTPEndpoint(
                    graphql_endpoint,
                    {"apikey": self.supabase_key, "Authorization": f"Bearer {self.supabase_key}"}
                )
                logger.info(f"GraphQL endpoint initialized at {graphql_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to initialize GraphQL endpoint: {e}")
                self.endpoint = None
        
        # Initialize Gemini
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. Gemini prompt enhancement will be disabled.")
            self.gemini_available = False
        else:
            self.gemini_available = True
            genai.configure(api_key=gemini_api_key)
            logger.info("Gemini API configured successfully")
    
    def _get_repo_id(self) -> Optional[str]:
        """
        Get repository ID from the database or local storage.
        
        Returns:
            Repository ID string if found, None otherwise
        """
        try:
            # Try to get repo ID from database
            db = get_db()
            if db and db.is_available():
                # Get repository name from directory path
                repo_name = os.path.basename(os.path.normpath(self.repository_dir.rstrip('/')))
                if repo_name == ".promptpilot":
                    # If the directory is the .promptpilot folder, use parent directory name
                    repo_name = os.path.basename(os.path.normpath(os.path.dirname(self.repository_dir)))
                
                # Query database for repository ID
                repositories = db.get_repositories()
                for repo in repositories:
                    if repo.get('name') == repo_name:
                        return repo.get('id')
            
            # Fall back to local storage
            # Try to get from repository_data.json
            repo_data_path = os.path.join(self.repository_dir, 'repository_data.json')
            if os.path.exists(repo_data_path):
                with open(repo_data_path, 'r') as f:
                    data = json.load(f)
                    return data.get('id')
                
            # Try repository_metadata.json as fallback
            repo_meta_path = os.path.join(self.repository_dir, 'repository_metadata.json')
            if os.path.exists(repo_meta_path):
                with open(repo_meta_path, 'r') as f:
                    data = json.load(f)
                    return data.get('id')
                
            return None
        except Exception as e:
            logger.error(f"Error getting repository ID: {e}")
            return None
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using available embedding service.
        
        Args:
            text: Text to generate embedding for
        
        Returns:
            List of float values representing the embedding
        """
        try:
            # For simplicity, we'll use the RepositoryAnalyzer's method
            # In a real implementation, you might want to extract this to a utility function
            from core.analyze import RepositoryAnalyzer
            
            # Create a temporary analyzer using the actual repository directory
            analyzer = RepositoryAnalyzer(self.repository_dir)
            embedding = await analyzer._generate_embedding(text)
            
            if not embedding:
                raise ValueError("Failed to generate embedding")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback (this is not ideal but prevents crashing)
            return [0.0] * 1536  # OpenAI embeddings are 1536 dimensions
    
    def find_relevant_contexts(self, task_description: str) -> Dict[str, Any]:
        """
        Find relevant contexts from repository based on task description.
        
        Args:
            task_description: Description of the task
        
        Returns:
            Dictionary of relevant contexts
        """
        logger.info(f"Finding relevant contexts for: {task_description}")
        
        # Check if GraphQL is available
        if GRAPHQL_AVAILABLE:
            try:
                # Get repository ID
                repo_id = self._get_repo_id()
                
                if not repo_id:
                    logger.warning(f"Repository ID not found for {self.repository_dir}")
                    return self._get_fallback_context()
                
                # Construct GraphQL query with search parameters
                op = Operation(Query)
                
                # Find relevant files
                files = op.repository(id=repo_id).files(
                    search=task_description,
                    similarity_threshold=self.similarity_threshold,
                    top_k=self.top_k
                )
                files.id()
                files.path()
                files.similarity()
                files.content()
                
                # Find relevant functions
                functions = op.repository(id=repo_id).functions(
                    search=task_description,
                    similarity_threshold=self.similarity_threshold,
                    top_k=self.top_k
                )
                functions.id()
                functions.name()
                functions.similarity()
                functions.docstring()
                functions.file_path()
                functions.signature()
                functions.start_line()
                functions.end_line()
                
                # Find relevant classes
                classes = op.repository(id=repo_id).classes(
                    search=task_description,
                    similarity_threshold=self.similarity_threshold,
                    top_k=self.top_k
                )
                classes.id()
                classes.name()
                classes.similarity()
                classes.docstring()
                classes.file_path()
                classes.start_line()
                classes.end_line()
                
                # Find relevant imports
                imports = op.repository(id=repo_id).imports(
                    search=task_description,
                    similarity_threshold=self.similarity_threshold,
                    top_k=self.top_k
                )
                imports.id()
                imports.statement()
                imports.file_path()
                
                # Execute query
                data = self.client.execute(op)
                
                # Create context dictionary
                context = {
                    "repository_name": data.repository.name if hasattr(data, 'repository') and hasattr(data.repository, 'name') else None,
                    "repo_id": repo_id,
                    "relevant_files": data.repository.files if hasattr(data, 'repository') and hasattr(data.repository, 'files') else [],
                    "relevant_functions": data.repository.functions if hasattr(data, 'repository') and hasattr(data.repository, 'functions') else [],
                    "relevant_classes": data.repository.classes if hasattr(data, 'repository') and hasattr(data.repository, 'classes') else [],
                    "relevant_imports": data.repository.imports if hasattr(data, 'repository') and hasattr(data.repository, 'imports') else []
                }
                
                return context
                
            except Exception as e:
                logger.error(f"Error finding contexts using GraphQL: {e}")
                return self._get_fallback_context()
        else:
            logger.info("GraphQL not available, using fallback context")
            return self._get_fallback_context()

    def _get_fallback_context(self) -> Dict[str, Any]:
        """
        Get a fallback context by reading files directly from the repository.
            
        Returns:
            Context dictionary with directly read files
        """
        try:
            # Get repository data path
            repo_data_path = os.path.join(self.repository_dir, 'repository_data.json')
            if not os.path.exists(repo_data_path):
                repo_data_path = os.path.join(self.repository_dir, 'repository_metadata.json')
                
            if not os.path.exists(repo_data_path):
                logger.error(f"Repository data not found at {repo_data_path}")
                return self._get_empty_context()
                
            # Load repository data
            with open(repo_data_path, 'r', encoding='utf-8') as f:
                repo_data = json.load(f)
        
            # Create context with files from repository
            context = {
                "repository_name": repo_data.get('name', 'unknown'),
                "repo_id": None,
                "relevant_files": [],
                "relevant_functions": [],
                "relevant_classes": [],
                "relevant_imports": []
            }
            
            # Get all files from repository data
            files = repo_data.get('files', [])
            if not files:
                logger.warning("No files found in repository data")
                return context
                
            # Sort files by extension and size to prioritize important files
            def file_importance(file):
                # Prioritize Python, JavaScript, and other code files
                ext = file.get('metadata', {}).get('extension', '').lower()
                size = file.get('metadata', {}).get('size_bytes', 0)
                
                # Score files based on extension and size
                ext_score = {
                    '.py': 100,
                    '.js': 90,
                    '.jsx': 90,
                    '.ts': 90,
                    '.tsx': 90,
                    '.java': 80,
                    '.c': 80,
                    '.cpp': 80,
                    '.h': 70,
                    '.rb': 70,
                    '.go': 70,
                    '.cs': 70,
                    '.php': 70,
                    '.html': 60,
                    '.css': 60,
                    '.md': 50,
                    '.json': 40,
                    '.yml': 40,
                    '.yaml': 40,
                    '.txt': 30,
                }.get(ext, 10)
                
                # Prefer medium-sized files (not too small, not too large)
                size_score = 50
                if size < 100:  # Too small
                    size_score = 10
                elif size > 100000:  # Too large
                    size_score = 20
                elif 1000 <= size <= 10000:  # Just right
                    size_score = 100
                
                return (ext_score, size_score)
            
            sorted_files = sorted(files, key=file_importance, reverse=True)
            
            # Take the top N most important files
            sample_size = min(15, len(sorted_files))
            sampled_files = sorted_files[:sample_size]
            
            # Add files to context
            for file_entry in sampled_files:
                metadata = file_entry.get('metadata', {})
                content = file_entry.get('content', '')
                
                file_info = {
                    "id": None,
                    "path": metadata.get('path', 'unknown'),
                    "similarity": 1.0,  # Dummy similarity
                    "content": content,  # Include actual content
                    "content_url": file_entry.get('content_url')
                }
                context["relevant_files"].append(file_info)
            
            logger.info(f"Using fallback context with {len(context['relevant_files'])} important files")
            return context
            
        except Exception as e:
            logger.error(f"Error creating fallback context: {e}")
            return self._get_empty_context()

    def _get_empty_context(self) -> Dict[str, Any]:
        """
        Get an empty context dictionary.
        
        Returns:
            Empty context dictionary
        """
        return {
            "repository_name": os.path.basename(self.repository_dir),
            "repo_id": None,
            "relevant_files": [],
            "relevant_functions": [],
            "relevant_classes": [],
            "relevant_imports": []
        }

    def _generate_prompt_template(self, contexts: Dict[str, Any], task_description: str,
                                max_tokens: int = 3250, max_context_files: int = 3,
                                skip_embedding_ranking: bool = False,
                                base_system_prompt: str = None) -> Dict[str, str]:
        """
        Generate a prompt template using the context and task description.
        
        Args:
            contexts: Dictionary containing context
            task_description: User-provided task description
            max_tokens: Maximum number of tokens for the prompt
            max_context_files: Maximum number of files to include in the context
            skip_embedding_ranking: Whether to skip embedding-based ranking
            base_system_prompt: Base system prompt to use
            
        Returns:
            Dictionary containing the prompt template
        """
        if base_system_prompt is None:
            base_system_prompt = "You are a helpful assistant."
        
        # Get repository name
        repository_name = contexts.get("repository_name", "")
        
        # Get the relevant files from the context
        files = contexts.get("relevant_files", [])
        
        # Rank files by similarity if embeddings available
        if not skip_embedding_ranking and any('embedding' in file for file in files):
            # Sort files by similarity
            files = sorted(files, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Limit the number of files
        files = files[:max_context_files]
        
        # Build prompt with repository and file information
        system_prompt = f"{base_system_prompt}\n\n"
        
        # Add repository information
        if repository_name:
            system_prompt += f"You are analyzing the '{repository_name}' repository. "
        
        system_prompt += f"The user has asked you to: {task_description}\n\n"
        
        # Add file information
        if files:
            system_prompt += "Here are some relevant files from the repository that may help you understand the codebase:\n\n"
            
            for file in files:
                file_path = file.get("path", "")
                content = file.get("content", "")
                
                if file_path and content:
                    # Add file path and content
                    system_prompt += f"File: {file_path}\n```\n{content}\n```\n\n"
        else:
            system_prompt += "No relevant files were found in the repository.\n\n"
        
        # Add final instructions
        system_prompt += "Using the context provided above, and your understanding of software development, please address the user's request."
        
        return {
            "system_prompt": system_prompt
        }
    
    async def enhance_prompt_with_gemini(self, task: str, context: Dict[str, Any]) -> str:
        """
        Enhance prompt template using Google's Gemini API.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            Enhanced prompt as a string
        """
        try:
            if not self.gemini_available:
                logger.warning("Gemini not available, using template as is")
                return None
                
            if not self.gemini_api_key:
                logger.warning("Gemini API key not found, skipping prompt enhancement")
                return None
                
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.gemini_api_key)
            
            # Generate prompt template
            template_dict = self._generate_prompt_template(context, task)
            template = template_dict.get("system_prompt", "")
            
            # Create prompt for Gemini
            gemini_prompt = f"""
            I have the following information about a code repository:
            
            {template}
            
            Based on the above information, could you help optimize this prompt to be more effective 
            for an AI assistant? Focus on highlighting the most important parts of the codebase 
            and organizing the information in a way that's easy to understand.
            
            Return only the optimized prompt without any additional comments.
            """
            
            # Get Gemini model
            model = genai.GenerativeModel('gemini-pro')
            
            # Generate response
            response = await model.generate_content_async(gemini_prompt)
            
            # Extract enhanced prompt
            if hasattr(response, 'text'):
                return response.text
            else:
                return template
                
        except Exception as e:
            logger.error(f"Error enhancing prompt with Gemini: {e}")
            return None
    
    async def generate_prompt(self, task: str) -> str:
        """
        Generate a prompt for a task.
        
        Args:
            task: Task description
            
        Returns:
            Generated prompt
        """
        try:
            # Get context for the task
            context = self.find_relevant_contexts(task)
            
            if not context.get("relevant_files"):
                logger.warning("No relevant files found for the task")
            
            # Check if we should try Gemini enhancement
            use_gemini = (
                hasattr(self, 'gemini_api_key') and 
                self.gemini_api_key and 
                os.environ.get('GEMINI_API_KEY')
            )
            
            # Generate prompt without Gemini
            if not use_gemini:
                # Generate prompt using template
                prompt_dict = self._generate_prompt_template(context, task)
                return prompt_dict.get("system_prompt", "")
                
            # Try to enhance with Gemini
            enhanced_prompt = await self.enhance_prompt_with_gemini(task, context)
            if enhanced_prompt:
                return enhanced_prompt
                
            # Fallback to template if Gemini fails
            prompt_dict = self._generate_prompt_template(context, task)
            return prompt_dict.get("system_prompt", "")
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            # Return basic prompt as fallback
            return f"You are a helpful assistant. The user has asked you to: {task}"

    def _generate_embeddings(self, contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add embeddings to the contexts for ranking.
        
        Args:
            contexts: Dictionary containing context for ranking
            
        Returns:
            Contexts with added embeddings
        """
        try:
            if not self.embedding_model:
                logger.warning("No embedding model provided, skipping embedding generation")
                return contexts
                
            # Create repository analyzer with the actual repository directory
            analyzer = RepositoryAnalyzer(self.repository_dir)
            
            # Process all file contents and generate embeddings
            for file in contexts.get("relevant_files", []):
                if not file.get("content"):
                    continue
                    
                file_content = file["content"]
                embedding = analyzer.generate_embedding(file_content)
                file["embedding"] = embedding
                
            return contexts
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return contexts

    def generate_enhanced_prompt(self, task_description: str, model: str = None, 
                              max_tokens: int = 3250, max_context_files: int = 3,
                              base_system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Generate an enhanced prompt using the repository context.
        
        Args:
            task_description: User-provided task description
            model: The model to generate the prompt for
            max_tokens: Maximum number of tokens for the prompt
            max_context_files: Maximum number of files to include in the context
            base_system_prompt: Base system prompt to use
            
        Returns:
            Generated prompt
        """
        # Find contexts from repository based on task description
        try:
            contexts = self.find_relevant_contexts(task_description)
        except Exception as e:
            logger.error(f"Error finding contexts: {e}")
            contexts = self._get_empty_context()
            
        # Generate embeddings for contexts when possible
        try:
            # Only generate embeddings if we have an embedding model
            if hasattr(self, 'embedding_model') and self.embedding_model:
                contexts = self._generate_embeddings(contexts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            
        # Skip embedding-based ranking if no embeddings available
        skip_embedding_ranking = not any(
            'embedding' in file for file in contexts.get('relevant_files', [])
        )
        
        # Generate enhanced prompt using the contexts
        prompt_template = self._generate_prompt_template(contexts, task_description, 
                                                        max_tokens=max_tokens,
                                                        max_context_files=max_context_files,
                                                        skip_embedding_ranking=skip_embedding_ranking,
                                                        base_system_prompt=base_system_prompt)
        
        # Replace system prompt
        system_prompt = prompt_template.get("system_prompt", "")
        
        # Create optimized prompt
        optimized_prompt = system_prompt
        
        return optimized_prompt


class RepositoryAnalyzer:
    """Simple Repository Analyzer that provides embedding functions."""
    
    def __init__(self, repository_dir):
        """
        Initialize the repository analyzer.
        
        Args:
            repository_dir: Path to the repository directory
        """
        self.repository_dir = repository_dir
        
    def generate_embedding(self, content: str) -> List[float]:
        """
        Generate a simple embedding for the provided content.
        This is a fallback method that creates a basic representation
        based on character frequencies.
        
        Args:
            content: The content to generate an embedding for
            
        Returns:
            A list of floats representing the embedding
        """
        # If content is empty, return a zero vector
        if not content or len(content) == 0:
            return [0.0] * 10
            
        # Create a simple embedding based on character frequencies
        # This is obviously not a proper embedding, but serves as a placeholder
        char_freq = {}
        for char in content:
            if char in char_freq:
                char_freq[char] += 1
            else:
                char_freq[char] = 1
                
        # Normalize by content length
        content_length = max(1, len(content))
        for char in char_freq:
            char_freq[char] /= content_length
            
        # Create fixed-size embedding from common ASCII characters
        embedding = []
        for ascii_val in range(32, 127):  # Basic ASCII range
            char = chr(ascii_val)
            embedding.append(char_freq.get(char, 0.0))
            
        # Normalize the embedding
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
            
        return embedding


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Generate an optimized prompt for a code generation task")
    parser.add_argument("repository_dir", help="Directory containing repository data (.promptpilot folder)")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL})")
    
    args = parser.parse_args()
    
    async def main():
        try:
            generator = PromptGenerator(args.repository_dir, model=args.model)
            prompt = await generator.generate_prompt(args.task)
            
            print("\n" + "="*80)
            print(prompt)
            print("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            traceback.print_exc()
            return 1
        
        return 0
    
    asyncio.run(main())
