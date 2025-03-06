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
    
    # Explicitly try to import SyncClient
    try:
        from sgqlc.endpoint.sync import SyncClient
        SYNC_CLIENT_AVAILABLE = True
    except ImportError:
        SYNC_CLIENT_AVAILABLE = False
        
    from core.schema import schema, Query
    GRAPHQL_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logging.warning(f"GraphQL functionality not available: {str(e)}")
    GRAPHQL_AVAILABLE = False
    SYNC_CLIENT_AVAILABLE = False

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
        
        # Set default parameters for similarity search
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        self.top_k = DEFAULT_TOP_K_FILES
        
        # Get Supabase URL and key from environment
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        if not self.supabase_url:
            logger.warning("SUPABASE_URL environment variable not set.")
        if not self.supabase_key:
            logger.warning("SUPABASE_KEY environment variable not set.")
            
        # Check if Gemini API key is available
        self.gemini_available = self.gemini_api_key is not None and len(self.gemini_api_key) > 0
        
        # Initialize GraphQL endpoint if available
        self.endpoint = None
        self.client = None
        
        if GRAPHQL_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                graphql_endpoint = f"{self.supabase_url}/graphql/v1"
                self.endpoint = HTTPEndpoint(
                    graphql_endpoint,
                    {"apikey": self.supabase_key, "Authorization": f"Bearer {self.supabase_key}"}
                )
                
                # Create a proper client with the execute method
                if SYNC_CLIENT_AVAILABLE:
                    try:
                        self.client = SyncClient(self.endpoint)
                        logger.info(f"GraphQL endpoint initialized with SyncClient at {graphql_endpoint}")
                    except Exception as client_err:
                        logger.warning(f"Failed to initialize SyncClient: {client_err}")
                        self.client = None
                else:
                    logger.warning("SyncClient not available, falling back to direct endpoint usage")
                    # Ensure the endpoint has an execute method
                    self.client = self.endpoint
                    
                # Verify the client has an execute method
                if not hasattr(self.client, 'execute'):
                    logger.warning("GraphQL client doesn't have execute method, falling back to direct HTTP requests")
                    
                    # Create a simple wrapper with an execute method
                    class SimpleClient:
                        def __init__(self, endpoint):
                            self.endpoint = endpoint
                            
                        def execute(self, operation, variables=None):
                            if variables:
                                return self.endpoint(operation, variables)
                            else:
                                return self.endpoint(operation)
                    
                    self.client = SimpleClient(self.endpoint)
                    logger.info("Created simple client wrapper for GraphQL endpoint")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize GraphQL endpoint: {e}")
                self.endpoint = None
                self.client = None
        
        # Initialize Gemini if API key available
        if self.gemini_available:
            try:
                genai.configure(api_key=self.gemini_api_key)
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API: {e}")
                self.gemini_available = False
    
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
        Generate embedding for text using an API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            # This is a simplified example - in a real application, you would 
            # call an embedding API like OpenAI's to generate the embedding
            
            # For now, just return a placeholder
            import numpy as np
            # Create a random but deterministic embedding
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(1536).tolist()  # 1536 dimensions for compatibility
            
            if not embedding:
                raise ValueError("Failed to generate embedding")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback (this is not ideal but prevents crashing)
            return [0.0] * 1536  # OpenAI embeddings are 1536 dimensions
    
    def find_relevant_contexts(self, task_description: str) -> Dict[str, Any]:
        """
        Find relevant contexts for a task description using GraphQL.
        
        Args:
            task_description: Task description to find contexts for
            
        Returns:
            Dictionary of relevant contexts
        """
        logger.info(f"Finding relevant contexts for: {task_description}")
        
        try:
            # Check if GraphQL is available
            if not GRAPHQL_AVAILABLE or not self.client:
                logger.warning("GraphQL not available, using fallback context")
                return self._get_fallback_context()
            
            # Try an introspection query first to understand the schema
            introspection_query = """
            {
                __schema {
                    queryType {
                        fields {
                            name
                            description
                        }
                    }
                }
            }
            """
            
            try:
                schema_result = self.client.execute(introspection_query)
                if isinstance(schema_result, dict) and 'data' in schema_result:
                    logger.info(f"GraphQL schema query type fields: {schema_result['data']['__schema']['queryType']['fields']}")
                else:
                    logger.warning(f"Unexpected schema introspection response format: {schema_result}")
            except Exception as schema_err:
                logger.error(f"Error during schema introspection: {schema_err}")
            
            # Get repository ID
            repo_id = self._get_repo_id()
            if not repo_id:
                logger.warning("Repository ID not found, using fallback context")
                return self._get_fallback_context()
            
            # Use Supabase REST API instead of GraphQL since we're having schema issues
            if self.supabase_url and self.supabase_key:
                try:
                    import httpx
                    # Construct URL for repositories endpoint
                    repo_url = f"{self.supabase_url}/rest/v1/repositories"
                    
                    # Set up headers for Supabase
                    headers = {
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # Query repositories
                    response = httpx.get(
                        repo_url,
                        headers=headers,
                        params={"select": "*"},
                        timeout=10.0  # 10 second timeout
                    )
                    
                    logger.info(f"Supabase REST API request: {repo_url} - Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        repositories = response.json()
                        logger.info(f"Found {len(repositories)} repositories via REST API")
                        
                        # Find our repository
                        repository = None
                        for repo in repositories:
                            if str(repo.get('id')) == str(repo_id):
                                repository = repo
                                break
                        
                        if repository:
                            # Query files, functions, etc. for this repository
                            # For now we'll use a simplified approach and just return the repository data
                            context = {
                                "repository_name": repository.get('name'),
                                "repo_id": repo_id,
                                "relevant_files": [],  # We'd need to query files separately
                                "relevant_functions": [],
                                "relevant_classes": [],
                                "relevant_imports": []
                            }
                            return context
                    else:
                        logger.error(f"Supabase REST API error: {response.status_code} - {response.text}")
                except Exception as rest_err:
                    logger.error(f"Error using Supabase REST API: {rest_err}")
                    logger.error(traceback.format_exc())
            
            # Fallback to original GraphQL attempt
            # Construct GraphQL query manually
            query = """
            query GetRelevantContexts($repoId: ID!, $search: String!, $threshold: Float!, $topK: Int!) {
              repositories(where: {id: {_eq: $repoId}}) {
                id
                name
                path
                file_count
                total_size_bytes
                file_types
                processed_date
                files(search: $search, similarityThreshold: $threshold, limit: $topK) {
                  id
                  path
                  content
                  similarity
                }
                functions(search: $search, similarityThreshold: $threshold, limit: $topK) {
                  id
                  name
                  docstring
                  signature
                  file_path
                  start_line
                  end_line
                  similarity
                }
                classes(search: $search, similarityThreshold: $threshold, limit: $topK) {
                  id
                  name
                  docstring
                  file_path
                  start_line
                  end_line
                  similarity
                }
                imports(limit: $topK) {
                  id
                  statement
                  file_path
                }
              }
            }
            """
            
            # Define variables for the query
            variables = {
                "repoId": repo_id,
                "search": task_description,
                "threshold": self.similarity_threshold,
                "topK": self.top_k
            }
            
            # Execute query
            try:
                if not self.client:
                    logger.error("GraphQL client not initialized")
                    return self._get_fallback_context()
                
                # Check if execute method exists and is callable
                if not hasattr(self.client, 'execute') or not callable(getattr(self.client, 'execute')):
                    logger.error("GraphQL client doesn't have a valid execute method")
                    return self._try_direct_rest_api(repo_id, task_description)
                    
                # Use a direct HTTP request approach
                data = self.client.execute(query, variables)
                
                # Check for valid response structure
                if not data or (isinstance(data, dict) and data.get('errors')):
                    if isinstance(data, dict) and data.get('errors'):
                        logger.error(f"GraphQL errors: {data.get('errors')}")
                        # If there's a schema error, try REST API
                        if any('Unknown field' in str(error.get('message', '')) for error in data.get('errors', [])):
                            logger.info("Schema mismatch detected, trying direct REST API approach")
                            return self._try_direct_rest_api(repo_id, task_description)
                    return self._get_fallback_context()
                
                # Extract repository data
                repository = None
                files = []
                functions = []
                classes = []
                imports = []
                
                if isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], dict):
                        if 'repositories' in data['data'] and data['data']['repositories']:
                            repository = data['data']['repositories'][0]  # Get first repository match
                        if 'files' in data['data']:
                            files = data['data']['files']
                        if 'functions' in data['data']:
                            functions = data['data']['functions']
                        if 'classes' in data['data']:
                            classes = data['data']['classes']
                        if 'imports' in data['data']:
                            imports = data['data']['imports']
                
                if not repository:
                    logger.warning(f"No repository found with ID {repo_id} in response, trying REST API")
                    return self._try_direct_rest_api(repo_id, task_description)
                
                # Create context dictionary
                context = {
                    "repository_name": repository.get('name'),
                    "repo_id": repo_id,
                    "relevant_files": files,
                    "relevant_functions": functions,
                    "relevant_classes": classes,
                    "relevant_imports": imports
                }
                
                return context
                
            except Exception as exec_err:
                logger.error(f"GraphQL execution error: {exec_err}")
                # Try REST API approach as fallback
                return self._try_direct_rest_api(repo_id, task_description)
                
        except Exception as e:
            logger.error(f"Error finding contexts using GraphQL: {e}")
            logger.error(traceback.format_exc())
            return self._get_fallback_context()

    def _try_direct_rest_api(self, repo_id: int, task_description: str) -> Dict[str, Any]:
        """
        Try to get repository data using direct REST API calls to Supabase.
        
        Args:
            repo_id: Repository ID
            task_description: Task description for potential similarity search
            
        Returns:
            Context dictionary
        """
        logger.info(f"Trying direct REST API approach for repository ID {repo_id}")
        
        try:
            # Initialize Supabase client if needed
            if not self.supabase:
                import os
                from supabase import create_client
                
                # Get Supabase credentials
                supabase_url = os.environ.get('SUPABASE_URL')
                supabase_key = os.environ.get('SUPABASE_KEY')
                
                if not supabase_url or not supabase_key:
                    logger.warning("Supabase credentials not available")
                    return self._get_fallback_context()
                
                # Initialize Supabase client
                self.supabase = create_client(supabase_url, supabase_key)
            
            # Get repository data
            repo_data = self.supabase.table('repositories').select('*').eq('id', repo_id).execute()
            if not repo_data.data:
                logger.warning(f"No repository found with ID {repo_id} using REST API")
                return self._get_fallback_context()
            
            repository = repo_data.data[0]
            
            # Get files (limit to avoid too much data)
            files_data = self.supabase.table('files').select('*').eq('repository_id', repo_id).limit(20).execute()
            files = files_data.data if files_data.data else []
            
            # Get functions
            functions_data = self.supabase.table('functions').select('*').eq('repository_id', repo_id).limit(10).execute()
            functions = functions_data.data if functions_data.data else []
            
            # Get classes
            classes_data = self.supabase.table('classes').select('*').eq('repository_id', repo_id).limit(10).execute()
            classes = classes_data.data if classes_data.data else []
            
            # Get imports
            imports_data = self.supabase.table('imports').select('*').eq('repository_id', repo_id).limit(10).execute()
            imports = imports_data.data if imports_data.data else []
            
            # Create context dictionary
            context = {
                "repository_name": repository.get('name'),
                "repo_id": repo_id,
                "relevant_files": files,
                "relevant_functions": functions,
                "relevant_classes": classes,
                "relevant_imports": imports
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error using REST API: {e}")
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
            repo_meta_path = os.path.join(self.repository_dir, 'repository_metadata.json')
            
            # Try to load repository metadata first (it's smaller)
            repo_meta = None
            if os.path.exists(repo_meta_path):
                try:
                    with open(repo_meta_path, 'r', encoding='utf-8') as f:
                        repo_meta = json.load(f)
                        logger.info(f"Loaded repository metadata from {repo_meta_path}")
                except Exception as e:
                    logger.error(f"Error loading repository metadata: {e}")
            
            # Try to load repository data
            repo_data = None
            if os.path.exists(repo_data_path):
                try:
                    with open(repo_data_path, 'r', encoding='utf-8') as f:
                        repo_data = json.load(f)
                        logger.info(f"Loaded repository data from {repo_data_path}")
                except Exception as e:
                    logger.error(f"Error loading repository data: {e}")
            
            # If neither file exists or could be loaded, return empty context
            if not repo_meta and not repo_data:
                logger.error(f"Repository data not found at {repo_data_path} or {repo_meta_path}")
                return self._get_empty_context()
                
            # Use metadata if data is not available
            data_source = repo_data if repo_data else repo_meta
        
            # Create context with files from repository
            context = {
                "repository_name": data_source.get('name', 'unknown'),
                "repo_id": data_source.get('id'),
                "relevant_files": [],
                "relevant_functions": [],
                "relevant_classes": [],
                "relevant_imports": []
            }
            
            # Get files from repository data
            files = data_source.get('files', [])
            if not files:
                logger.warning("No files found in repository data")
                return context
                
            # Sort files by extension and size to prioritize important files
            def file_importance(file):
                # Get extension and size, handling different data structures
                ext = None
                size = 0
                
                if 'extension' in file:
                    ext = file.get('extension', '').lower()
                elif 'metadata' in file and 'extension' in file['metadata']:
                    ext = file.get('metadata', {}).get('extension', '').lower()
                
                if 'size_bytes' in file:
                    size = file.get('size_bytes', 0)
                elif 'metadata' in file and 'size_bytes' in file['metadata']:
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
                    '.rs': 70,
                    '.php': 60,
                    '.html': 50,
                    '.css': 50,
                    '.md': 40,
                    '.json': 40,
                    '.txt': 30
                }.get(ext, 10)
                
                # Penalize very large files
                if size > 1000000:  # 1MB
                    ext_score *= 0.5
                
                return ext_score
            
            # Sort files by importance
            sorted_files = sorted(files, key=file_importance, reverse=True)
            
            # Take top files
            top_files = sorted_files[:5]
            
            # Add files to context
            for file in top_files:
                # Handle different data structures
                file_path = file.get('path')
                if not file_path and 'metadata' in file:
                    file_path = file.get('metadata', {}).get('path')
                
                content = file.get('content')
                if not content and 'content' in file:
                    content = file.get('content')
                
                # Only add file if it has a path
                if file_path:
                    context['relevant_files'].append({
                        'path': file_path,
                        'content': content if content else "Content not available",
                        'similarity': 1.0  # Default similarity for fallback
                    })
            
            logger.info(f"Created fallback context with {len(context['relevant_files'])} important files")
            return context
            
        except Exception as e:
            logger.error(f"Error creating fallback context: {e}")
            return self._get_empty_context()

    def _get_empty_context(self) -> Dict[str, Any]:
        """
        Get an empty context as a fallback.
        
        Returns:
            Empty context dictionary
        """
        return {
            "repository_name": "unknown",
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
        Generate a prompt template based on task description and context.
        
        Args:
            contexts: Context information
            task_description: Description of the task
            max_tokens: Maximum number of tokens to include in prompt
            max_context_files: Maximum number of files to include in context
            skip_embedding_ranking: Whether to skip embedding-based ranking
            base_system_prompt: Base system prompt to use
            
        Returns:
            Dictionary with system and user prompts
        """
        # If no base system prompt provided, use the default
        if not base_system_prompt:
            base_system_prompt = SYSTEM_PROMPT
            
        # Start building the context part of the prompt
        context_parts = []
        
        # Add repository name if available
        repo_name = contexts.get("repository_name", "unknown")
        context_parts.append(f"Repository: {repo_name}")
        
        # Add relevant files if available
        relevant_files = contexts.get("relevant_files", [])
        
        # If we have files and want to rank them by embedding similarity
        if relevant_files and not skip_embedding_ranking:
            try:
                # Get embeddings for files and task
                contexts_with_embeddings = self._generate_embeddings(contexts)
                relevant_files = contexts_with_embeddings.get("relevant_files", [])
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Continue with original files
        
        # Limit to max_context_files most relevant files
        included_files = relevant_files[:max_context_files]
        
        # Add file contents to context
        if included_files:
            context_parts.append("\nRelevant Files:")
            
            for i, file_info in enumerate(included_files):
                file_path = file_info.get("path", "unknown")
                file_content = file_info.get("content", "")
                
                if not file_content and "content_url" in file_info:
                    # If we have a content URL but no content, we could fetch it here
                    # For simplicity, we'll skip this for now
                    continue
                    
                # Cap file content length to avoid exceeding token limits
                if len(file_content) > 1000:
                    file_content = file_content[:1000] + "... [content truncated]"
                    
                context_parts.append(f"\nFile {i+1}: {file_path}\n```\n{file_content}\n```")
        
        # Add relevant functions if available
        relevant_functions = contexts.get("relevant_functions", [])
        if relevant_functions:
            context_parts.append("\nRelevant Functions:")
            
            for i, func in enumerate(relevant_functions[:5]):  # Limit to 5 functions
                func_name = func.get("name", "unknown")
                func_signature = func.get("signature", "")
                func_docstring = func.get("docstring", "")
                func_file = func.get("file_path", "")
                
                context_parts.append(f"\nFunction {i+1}: {func_name}")
                context_parts.append(f"File: {func_file}")
                if func_signature:
                    context_parts.append(f"Signature: {func_signature}")
                if func_docstring:
                    context_parts.append(f"Documentation: {func_docstring}")
        
        # Add relevant classes if available
        relevant_classes = contexts.get("relevant_classes", [])
        if relevant_classes:
            context_parts.append("\nRelevant Classes:")
            
            for i, cls in enumerate(relevant_classes[:3]):  # Limit to 3 classes
                cls_name = cls.get("name", "unknown")
                cls_docstring = cls.get("docstring", "")
                cls_file = cls.get("file_path", "")
                
                context_parts.append(f"\nClass {i+1}: {cls_name}")
                context_parts.append(f"File: {cls_file}")
                if cls_docstring:
                    context_parts.append(f"Documentation: {cls_docstring}")
        
        # Combine all context parts
        context_text = "\n".join(context_parts)
        
        # Create the system prompt with context and instructions
        system_prompt = f"""{base_system_prompt}

Here is the context information about the repository:

{context_text}

Task: {task_description}

Please use the provided context to complete the task. If the context doesn't provide sufficient information, explain what additional information would be helpful."""
        
        # Create a simple user prompt
        user_prompt = f"Help me with this task: {task_description}"
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
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
            model = genai.GenerativeModel('gemini-2.0-flash-001')
            
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
            Generated prompt as a string
        """
        try:
            # Find relevant contexts
            contexts = self.find_relevant_contexts(task)
            
            # If no context was found, use an empty context
            if not contexts:
                contexts = self._get_empty_context()
                
            # Try to enhance the prompt with Gemini
            if self.gemini_available:
                enhanced_prompt = await self.enhance_prompt_with_gemini(task, contexts)
                if enhanced_prompt:
                    return enhanced_prompt
            
            # Fall back to template-based prompt if Gemini enhancement failed
            template = self._generate_prompt_template(contexts, task)
            return template.get("system_prompt", "")
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            # Return a minimal prompt as fallback
            return f"""You are a helpful assistant. The user has asked you to help with the following task:

Task: {task}

Please provide your best assistance based on the task description."""
    
    def _generate_embeddings(self, contexts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for files and sort by similarity to task.
        
        Args:
            contexts: Context dictionary with files
            
        Returns:
            Updated context dictionary with files sorted by similarity
        """
        # This is a simplified placeholder for embedding generation
        # In a real implementation, you would generate embeddings and compute similarities
        
        # Just return the contexts as-is for now
        return contexts
        
    def generate_enhanced_prompt(self, task_description: str, model: str = None, 
                              max_tokens: int = 3250, max_context_files: int = 3,
                              base_system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Generate an enhanced prompt with Gemini (synchronous version).
        
        Args:
            task_description: Description of the task
            model: Model to use for prompt generation
            max_tokens: Maximum number of tokens to include in prompt
            max_context_files: Maximum number of files to include in context
            base_system_prompt: Base system prompt to use
            
        Returns:
            Enhanced prompt as a string
        """
        # Import asyncio
        import asyncio
        
        # Create event loop and run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async function in event loop
            return loop.run_until_complete(self.generate_prompt(task_description))
        finally:
            # Close loop
            loop.close()


class RepositoryAnalyzer:
    """Simple analyzer for repositories."""
    
    def __init__(self, repository_dir):
        """Initialize with repository directory."""
        self.repository_dir = repository_dir
        
    def generate_embedding(self, content: str) -> List[float]:
        """
        Generate a dummy embedding for content.
        This is a placeholder - in a real application, you would use
        an embedding model like OpenAI's to generate the embedding.
        
        Args:
            content: Text content to embed
            
        Returns:
            Embedding as a list of floats
        """
        import numpy as np
        
        # Create a random but deterministic embedding
        np.random.seed(hash(content) % 2**32)
        return np.random.randn(1536).tolist()  # 1536 dimensions like OpenAI embeddings


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