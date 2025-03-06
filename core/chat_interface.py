"""
Repository Chat Interface for PromptPilot.

This module handles the chat interface for interacting with repositories
using RAG (Retrieval Augmented Generation) with embeddings.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.chat')

try:
    from google.generativeai import GenerativeModel, configure as configure_genai
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google GenerativeAI package not available. Install with 'pip install google-generativeai'")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available. Install with 'pip install openai'")

from .enhanced_db import get_db

# System messages for different LLMs
SYSTEM_MESSAGES = {
    "gemini": """You are an AI assistant specializing in code analysis and understanding with advanced database querying capabilities. 
You're analyzing a repository that has been processed and its structure, code and other details are stored in a database which you can access.

You have the following capabilities to access and utilize repository information:
1. Code Search: You can search for relevant files, functions, classes and code snippets based on semantic meaning
2. Code Understanding: You can analyze and explain code from the repository in detail
3. Repository Structure Analysis: You can assess the file hierarchy and component relationships
4. Database Access: You can retrieve precise information about files, functions, classes and their relationships

QUERYING STRATEGY:
When working with the database, follow these steps:
1. Analyze the user's query to identify what specific repository information is needed
2. Break down complex queries into specific searchable entities (files, functions, classes, patterns)
3. For each entity:
   - Start with precise searches for exact matches (file names, function names)
   - If precise search fails, broaden to semantic searches
   - Consider related entities that might contain relevant information
4. When examining code, look for:
   - Dependencies between components
   - Function calls and data flow
   - Implementation patterns and architectural structures
5. Synthesize information from multiple sources before responding

RESPONSE GUIDELINES:
1. Always cite specific files, classes, and functions with their paths when providing information
2. Use code blocks with syntax highlighting for readability
3. When referencing implementation patterns, connect them to architectural principles
4. For complex functionalities, describe the relationships between components
5. When suggesting code changes, ensure they align with the existing architecture patterns
6. If information is incomplete, clearly state what is missing and suggest how to find it

PERFORMANCE CONSIDERATIONS:
1. Consider time complexity implications of suggested changes
2. Be aware of memory usage patterns in the codebase
3. Note potential concurrency issues in your suggestions
4. Highlight areas where performance optimizations might be possible

The current repository being analyzed is: {repo_name}
Path: {repo_path}
Repository ID in database: {repo_id}
""",
    
    "openai": """You are an AI assistant specializing in code analysis and understanding. 
You're analyzing a repository that has been processed and its structure, code and other details will be provided to you.

You have the ability to access information from the repository using the following capabilities:
1. Code Search: Relevant files and code snippets are automatically included in the context based on the user's query
2. Code Understanding: You can analyze and explain code that's included in the context
3. Repository Structure: The file hierarchy and organization is included in the initial context message

IMPORTANT INSTRUCTIONS:
- All search capabilities are automatically handled for you. The system will include relevant code snippets based on the user's query.
- When you need to reference a specific part of the code, use the context that was provided in the message.
- If you don't have enough information or the relevant code wasn't found in the provided context, clearly state this in your response.
- You do not need to search for or access files directly - this is all done for you behind the scenes.

When answering questions:
1. If the relevant code isn't found in the context provided, simply indicate this clearly
2. When referencing code, mention the file path and relevant line numbers if available
3. Use Markdown for code formatting for good readability
4. Be proactive about suggesting related areas of the codebase that might be relevant

The current repository being analyzed is: {repo_name}
Path: {repo_path}
Repository ID in database: {repo_id}
"""
}

PROMPT_TEMPLATES = {
    "default": """Based on the provided repository information, generate a detailed and effective prompt for the following task:

TASK: {task}

Consider:
- The repository structure and key files
- Important functions, classes, and their relationships
- The overall purpose and domain of the codebase
- Any specific requirements or constraints mentioned

The generated prompt should be detailed enough to provide clear context and instructions, but concise enough to be practical for use with an AI assistant.
""",
    "database_query": """You are analyzing a code repository stored in a database.

REPOSITORY: {repo_name}
QUERY: {query}

INSTRUCTIONS FOR DATABASE QUERYING:
1. Break down this query into specific searchable entities:
   - Files: Look for specific file names or paths
   - Functions: Identify key functions related to the query
   - Classes: Note relevant classes that might contain the information
   - Patterns: Consider code patterns that implement the queried functionality

2. For each entity, formulate both:
   - Exact match queries (for precise name matches)
   - Semantic similarity queries (for conceptual matches)

3. Consider relationships between components:
   - Parent-child relationships between classes
   - Function calls and dependencies
   - Data flow patterns across modules

4. Analyze the results with attention to:
   - Implementation patterns
   - Performance implications
   - Architecture considerations
   - Error handling techniques

Respond with a detailed analysis of the code, including specific file paths, function names, and relevant code snippets.
"""
}

class RepositoryChat:
    """
    Chat interface for interacting with repository data using RAG.
    """
    
    def __init__(self, repo_id: int, model_name: str = 'gemini-2.0-flash-001', temperature: float = 0.7):
        """
        Initialize the Repository Chat.
        
        Args:
            repo_id: The repository ID
            model_name: The model name to use
            temperature: Temperature for generation
        """
        self.repo_id = repo_id
        self.model_name = model_name
        self.temperature = temperature
        self.repo_name = None  # Will be set externally or derived from repo info
        self.digest_url = None  # Initialize digest_url attribute
        
        # Initialize the database client
        self.db = get_db()
        
        # Get repository information
        self.repo_info = self._get_repository_info()
        
        # Set repo_name from repository info if available
        if self.repo_info and 'name' in self.repo_info:
            self.repo_name = self.repo_info['name']
        else:
            # Fallback to a default name based on repo_id
            self.repo_name = f"Repository-{self.repo_id}"
            logger.warning(f"Repository name not found in info, using fallback: {self.repo_name}")
        
        # Try to get digest URL if available
        try:
            if hasattr(self.db, 'get_repository_digest_url'):
                self.digest_url = self.db.get_repository_digest_url(self.repo_id)
                logger.debug(f"Repository digest URL: {self.digest_url}")
        except Exception as e:
            logger.warning(f"Could not retrieve digest URL: {str(e)}")
            self.digest_url = None
            
        # Determine model provider
        self.model_provider = self._get_model_provider(model_name)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize history
        self.history = []
        
        # Initialize system message
        self.system_message = """You are an AI assistant analyzing a repository to provide insights and answers about the code. 
Be detailed and specific in your explanations, and always use the CONTEXT information provided to you.

IMPORTANT: Your answers must be based on the actual code context provided, not general knowledge.
When asked about the repository, refer to specific files, functions, classes and patterns found in the context.
If you don't see relevant information in the context, acknowledge this limitation and explain
what specific information you would need to better answer the question.

DO NOT make up information about the repository. Only use what's in the provided context."""
        
        # Initialize repository context
        self._initialize_repository_context()
        
        # Initialize cache for search results
        self.search_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 600  # Cache time-to-live in seconds (10 minutes)
    
    def _get_cached_or_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get search results from cache if available, otherwise perform a search.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # Create a cache key based on the query and top_k
        cache_key = f"{query}_{top_k}"
        
        # Check if we have a valid cached result
        current_time = time.time()
        if cache_key in self.search_cache and self.cache_expiry.get(cache_key, 0) > current_time:
            logger.debug(f"Cache hit for query: {query}")
            return self.search_cache[cache_key]
            
        # If no cache hit, perform the search
        logger.debug(f"Cache miss for query: {query}")
        results = self.search_code(query, top_k=top_k)
        
        # Cache the results
        self.search_cache[cache_key] = results
        self.cache_expiry[cache_key] = current_time + self.cache_ttl
        
        # Manage cache size - if it grows too large
        if len(self.search_cache) > 50:  # Limit to 50 cached queries
            # Remove the oldest entries
            oldest_keys = sorted(self.cache_expiry.keys(), key=lambda k: self.cache_expiry[k])[:10]
            for key in oldest_keys:
                self.search_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
        
        return results
    
    def clear_cache(self):
        """
        Clear the search cache.
        """
        self.search_cache = {}
        self.cache_expiry = {}
        logger.debug("Search cache cleared")
        
    def _get_repository_info(self) -> Dict[str, Any]:
        """Get repository information from the database."""
        try:
            logger.debug(f"Attempting to get repository info for ID: {self.repo_id}")
            repos = self.db.get_repositories()
            logger.debug(f"Found {len(repos)} repositories")
            
            for repo in repos:
                if repo.get('id') == self.repo_id:
                    logger.debug(f"Found matching repository: {repo.get('name', 'Unknown')}")
                    return repo
            
            # Try direct DB lookup if not found in local repositories
            try:
                logger.debug("Repository not found in local list, trying direct DB lookup")
                if hasattr(self.db, 'supabase'):
                    supabase_data = self.db.supabase.table('repositories').select('*').eq('id', self.repo_id).execute()
                    if supabase_data and hasattr(supabase_data, 'data') and len(supabase_data.data) > 0:
                        repo_data = supabase_data.data[0]
                        logger.debug(f"Found repository in database: {repo_data.get('name', 'Unknown')}")
                        return repo_data
            except Exception as db_err:
                logger.warning(f"Direct DB lookup failed: {str(db_err)}")
                
            logger.warning(f"Repository with ID {self.repo_id} not found")
            return {"name": f"Repository-{self.repo_id}", "id": self.repo_id}  # Return minimal info with ID
        except Exception as e:
            logger.error(f"Error getting repository info: {str(e)}")
            return {"name": f"Repository-{self.repo_id}", "id": self.repo_id}  # Return minimal info with ID
            
    def _get_model_provider(self, model_name: str) -> str:
        """Determine the model provider based on the model name."""
        if model_name.startswith('gemini'):
            if not GENAI_AVAILABLE:
                raise ImportError("Google GenerativeAI package is required for Gemini models")
            return "gemini"
        elif model_name.startswith('gpt'):
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package is required for GPT models")
            return "openai"
        else:
            # Default to gemini
            return "gemini"
            
    def _initialize_llm(self):
        """Initialize the language model based on the provider."""
        try:
            if self.model_provider == "gemini":
                # Check for both GOOGLE_API_KEY and GEMINI_API_KEY
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    api_key = os.environ.get("GEMINI_API_KEY")
                    
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for Gemini models")
                    
                configure_genai(api_key=api_key)
                
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
                
                self.llm = GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": self.temperature},
                    safety_settings=safety_settings
                )
                
            elif self.model_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
                    
                self.llm = OpenAI(api_key=api_key)
                
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
            
            logger.debug(f"Initialized LLM with provider {self.model_provider} and model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
            
    def search_code(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code in the repository related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        logger.debug(f"Searching code with query: '{query}', top_k={top_k}")
        
        # Classify the query intent to optimize search strategy
        query_intent = self._classify_query_intent(query)
        logger.debug(f"Query intent classified as: '{query_intent}'")
        
        # Expand the query with synonyms based on the intent
        expanded_queries = self._expand_query(query, query_intent)
        logger.debug(f"Expanded query: {expanded_queries}")
        
        # Track original parsed query for comparison
        original_parsed = self._parse_search_query(query)
        
        results = []
        
        # Adjust search strategy based on query intent
        if query_intent == 'file_lookup':
            # For file lookups, prioritize exact filename matches
            direct_matches = self._search_for_exact_file_matches(query)
            if direct_matches:
                logger.debug(f"Found {len(direct_matches)} exact file matches")
                results.extend(direct_matches)
                
            # If we already have enough exact matches, we might skip semantic search
            if len(results) >= top_k:
                logger.debug(f"Using only exact matches, skipping semantic search")
                return results[:top_k]
        
        # For function and class searches, we'll prioritize those specific elements
        prioritize_functions = query_intent == 'function_search'
        prioritize_classes = query_intent == 'class_search'
        
        # Generate embedding for the semantic query
        embedding = self._generate_embedding(query)
        
        if embedding:
            logger.debug(f"Generated embedding for semantic search")
            # Search for relevant files using the embedding with expanded query keywords
            semantic_results = self.db.find_relevant_files(self.repo_id, embedding, top_k=top_k*2)  # Get more results to filter
            logger.debug(f"Semantic search returned {len(semantic_results)} results")
            
            # Add content snippets if available
            for result in semantic_results:
                if result not in results:  # Avoid duplicates
                    try:
                        file_id = result.get('id')
                        path = result.get('path')
                        
                        # Try to get content snippet
                        content = self.db.get_file_content_by_path(self.repo_id, path)
                        if content:
                            # Create a snippet based on the query intent
                            if query_intent == 'function_search' or query_intent == 'class_search':
                                # For function/class searches, try to show more relevant code sections
                                relevant_lines = self._find_relevant_code_sections(content, expanded_queries, query_intent)
                                if relevant_lines:
                                    snippet = '\n'.join(relevant_lines)
                                    if len(snippet) > 800:
                                        snippet = snippet[:800] + "..."
                                else:
                                    # Default to first 10 lines if no specific sections found
                                    lines = content.split('\n')
                                    snippet = '\n'.join(lines[:min(10, len(lines))])
                                    if len(snippet) > 500:
                                        snippet = snippet[:500] + "..."
                            else:
                                # For other intents, use default snippet approach
                                lines = content.split('\n')
                                snippet = '\n'.join(lines[:min(10, len(lines))])
                                if len(snippet) > 500:
                                    snippet = snippet[:500] + "..."
                                    
                            result['snippet'] = snippet
                            
                            # Always check for specific code elements but prioritize based on intent
                            result['has_functions'] = self._has_relevant_functions(file_id, expanded_queries)
                            result['has_classes'] = self._has_relevant_classes(file_id, expanded_queries)
                                
                    except Exception as e:
                        logger.warning(f"Error getting content for file {file_id}: {str(e)}")
                    
                    # Add the query intent to the result for reference
                    result['query_intent'] = query_intent
                    results.append(result)
            
            # Define sorting key based on query intent
            def get_sorting_key(result):
                # Base relevance score
                relevance = result.get('similarity', 0)
                
                # Modify relevance based on intent and content
                if query_intent == 'file_lookup' and any(fp in result.get('path', '').lower() for fp in original_parsed['file_patterns']):
                    relevance += 0.5  # Boost for files that match the pattern
                    
                elif query_intent == 'function_search' and result.get('has_functions', False):
                    relevance += 0.4  # Boost for files containing relevant functions
                    
                elif query_intent == 'class_search' and result.get('has_classes', False):
                    relevance += 0.4  # Boost for files containing relevant classes
                    
                elif query_intent == 'architecture' and 'README' in result.get('path', ''):
                    relevance += 0.3  # Boost for README files in architecture queries
                    
                elif query_intent == 'dependency' and any(dep in result.get('path', '').lower() 
                                                      for dep in ['requirements', 'package', 'dependencies', 'setup']):
                    relevance += 0.3  # Boost for dependency-related files
                
                return relevance
            
            # Sort results based on intent-specific relevance
            results = sorted(results, key=get_sorting_key, reverse=True)
                               
            # Limit to top_k results
            results = results[:top_k]
                    
        return results
        
    def _find_relevant_code_sections(self, content: str, expanded_queries: Dict[str, List[str]], intent: str) -> List[str]:
        """
        Find the most relevant sections of code based on the query intent.
        
        Args:
            content: The file content
            expanded_queries: The expanded query terms
            intent: The query intent
            
        Returns:
            List of relevant code lines
        """
        lines = content.split('\n')
        relevant_sections = []
        
        # Focus on what we're looking for based on intent
        keywords = expanded_queries['keywords']
        
        if intent == 'function_search':
            # Look for function definitions
            function_names = expanded_queries['function_names']
            all_terms = function_names + keywords
            
            in_relevant_section = False
            section_start = 0
            current_section = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check if this line contains function definition
                is_function_def = any(f"def {fn}" in line_lower or f"function {fn}" in line_lower 
                                     for fn in function_names) if function_names else False
                
                # Or check if it contains relevant keywords
                has_keywords = any(kw.lower() in line_lower for kw in all_terms) if all_terms else False
                
                if is_function_def:
                    # We found a new function, add the previous section if it exists
                    if current_section:
                        relevant_sections.extend(current_section)
                        current_section = []
                    
                    # Start tracking a new section
                    in_relevant_section = True
                    section_start = i
                    current_section.append(line)
                elif in_relevant_section:
                    # Continue adding lines while in a relevant section
                    current_section.append(line)
                    
                    # Check if we've gone too far from the function definition
                    if i - section_start > 20:  # Limit section size
                        in_relevant_section = False
                        # Only add this section if it has keywords
                        if any(kw.lower() in '\n'.join(current_section).lower() for kw in all_terms):
                            relevant_sections.extend(current_section)
                        current_section = []
                elif has_keywords:
                    # Add context lines around lines with keywords
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    context_lines = lines[start_idx:end_idx]
                    relevant_sections.extend(context_lines)
        
        elif intent == 'class_search':
            # Look for class definitions
            class_names = expanded_queries['class_names']
            all_terms = class_names + keywords
            
            in_relevant_section = False
            section_start = 0
            current_section = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check if this line contains class definition
                is_class_def = any(f"class {cn}" in line_lower for cn in class_names) if class_names else False
                
                # Or check if it contains relevant keywords
                has_keywords = any(kw.lower() in line_lower for kw in all_terms) if all_terms else False
                
                if is_class_def:
                    # We found a new class, add the previous section if it exists
                    if current_section:
                        relevant_sections.extend(current_section)
                        current_section = []
                    
                    # Start tracking a new section
                    in_relevant_section = True
                    section_start = i
                    current_section.append(line)
                elif in_relevant_section:
                    # Continue adding lines while in a relevant section
                    current_section.append(line)
                    
                    # Check if we've gone too far from the class definition
                    if i - section_start > 30:  # Classes can be larger
                        in_relevant_section = False
                        # Only add this section if it has keywords
                        if any(kw.lower() in '\n'.join(current_section).lower() for kw in all_terms):
                            relevant_sections.extend(current_section)
                        current_section = []
                elif has_keywords:
                    # Add context lines around lines with keywords
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    context_lines = lines[start_idx:end_idx]
                    relevant_sections.extend(context_lines)
        
        else:
            # For other intents, just find sections with relevant keywords
            for i, line in enumerate(lines):
                if any(kw.lower() in line.lower() for kw in keywords):
                    # Add context lines around matching lines
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    context_lines = lines[start_idx:end_idx]
                    relevant_sections.extend(context_lines)
        
        # If we still have an active section, add it
        if current_section:
            relevant_sections.extend(current_section)
            
        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for line in relevant_sections:
            if line not in seen:
                seen.add(line)
                unique_sections.append(line)
                
        return unique_sections
        
    def _parse_search_query(self, query: str) -> Dict[str, List[str]]:
        """
        Parse a search query into components for more targeted searching.
        
        Args:
            query: The user's search query
            
        Returns:
            Dictionary with parsed query components
        """
        parsed = {
            'file_patterns': [],
            'function_names': [],
            'class_names': [],
            'keywords': []
        }
        
        # Look for file patterns (words with extensions)
        words = query.split()
        for word in words:
            # Check for file extensions
            if '.' in word and len(word) > 3 and not word.startswith(('http', 'www')):
                parsed['file_patterns'].append(word.strip(',.;:()[]{}'))
                
            # Check for potential function names (camelCase or snake_case patterns)
            elif ('_' in word or 
                  (any(c.islower() for c in word) and any(c.isupper() for c in word)) or
                  word.endswith('()')):
                clean_word = word.strip(',.;:()[]{}')
                if clean_word.endswith('()'):
                    clean_word = clean_word[:-2]
                parsed['function_names'].append(clean_word)
                
            # Check for potential class names (PascalCase)
            elif word and word[0].isupper() and not word.isupper() and len(word) > 1:
                parsed['class_names'].append(word.strip(',.;:()[]{}'))
                
            # Add as general keyword
            else:
                clean_word = word.strip(',.;:()[]{}')
                if clean_word and len(clean_word) > 2:
                    parsed['keywords'].append(clean_word)
        
        return parsed
        
    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of a user query to optimize search strategy and response format.
        
        Args:
            query: The user's search query
            
        Returns:
            Intent classification as one of: 'file_lookup', 'function_search', 'class_search',
            'architecture', 'performance', 'dependency', 'general'
        """
        query_lower = query.lower()
        parsed_query = self._parse_search_query(query)
        
        # File lookup intent patterns
        file_indicators = ["show me", "show file", "display", "view", "get the file", 
                          "what's in the file", "what is in", "content of", "open", 
                          "read", "find file", "locate file"]
        if any(indicator in query_lower for indicator in file_indicators) or parsed_query['file_patterns']:
            return 'file_lookup'
        
        # Function search intent patterns
        function_indicators = ["how does", "function", "method", "how to call", 
                              "implementation of", "code for", "how is implemented", 
                              "parameters", "arguments", "return value", "usage of"]
        if any(indicator in query_lower for indicator in function_indicators) or parsed_query['function_names']:
            return 'function_search'
        
        # Class search intent patterns
        class_indicators = ["class", "object", "instance", "inheritance", "subclass", 
                           "parent class", "derived class", "interface", "structure"]
        if any(indicator in query_lower for indicator in class_indicators) or parsed_query['class_names']:
            return 'class_search'
        
        # Architecture intent patterns
        architecture_indicators = ["architecture", "design", "pattern", "organization", 
                                  "structure", "system", "module", "component", "diagram", 
                                  "relationship", "flow", "hierarchy", "layering"]
        if any(indicator in query_lower for indicator in architecture_indicators):
            return 'architecture'
        
        # Performance intent patterns
        performance_indicators = ["performance", "speed", "optimization", "efficient", 
                                 "memory usage", "cpu usage", "bottleneck", "slow", 
                                 "fast", "resource", "complexity", "time complexity", 
                                 "space complexity", "benchmark", "profile"]
        if any(indicator in query_lower for indicator in performance_indicators):
            return 'performance'
        
        # Dependency intent patterns
        dependency_indicators = ["dependency", "import", "require", "uses", "used by", 
                               "depends on", "dependency chain", "library", "package", 
                               "module", "imported by", "connection", "linked to"]
        if any(indicator in query_lower for indicator in dependency_indicators):
            return 'dependency'
        
        # If no specific intent is detected, default to general
        return 'general'
    
    def _expand_query(self, query: str, intent: str) -> Dict[str, List[str]]:
        """
        Expand a search query with synonyms and related terms to improve search results.
        
        Args:
            query: The original search query
            intent: The classified intent of the query
            
        Returns:
            Dictionary with original parsed query components enriched with synonyms
        """
        # Get the base parsed query
        parsed_query = self._parse_search_query(query)
        
        # Define domain-specific synonym dictionary by intent
        synonyms = {
            'file_lookup': {
                'file': ['document', 'module', 'source', 'code', 'script'],
                'directory': ['folder', 'package', 'namespace'],
                'path': ['location', 'route', 'address'],
                'show': ['display', 'view', 'reveal', 'open'],
                'find': ['locate', 'search', 'discover'],
                'content': ['contents', 'data', 'information', 'text'],
            },
            'function_search': {
                'function': ['method', 'procedure', 'routine', 'subroutine', 'handler'],
                'call': ['invoke', 'execute', 'run'],
                'parameter': ['argument', 'input', 'param', 'arg'],
                'return': ['output', 'result', 'value'],
                'implement': ['define', 'code', 'write', 'create'],
                'usage': ['usage example', 'example', 'usage pattern', 'use case'],
            },
            'class_search': {
                'class': ['type', 'object', 'structure', 'entity'],
                'instance': ['object', 'instantiation'],
                'inherit': ['extend', 'derive', 'subclass'],
                'property': ['attribute', 'field', 'member', 'variable'],
                'method': ['function', 'operation', 'behavior'],
                'constructor': ['initializer', 'init', 'new', 'create'],
            },
            'architecture': {
                'architecture': ['design', 'structure', 'organization', 'layout'],
                'component': ['module', 'part', 'element', 'unit'],
                'system': ['application', 'program', 'software', 'app'],
                'interface': ['api', 'connection', 'contract', 'boundary'],
                'pattern': ['model', 'approach', 'template', 'paradigm'],
                'flow': ['process', 'sequence', 'pipeline', 'workflow'],
            },
            'performance': {
                'performance': ['speed', 'efficiency', 'throughput'],
                'optimize': ['improve', 'enhance', 'speed up', 'tune'],
                'bottleneck': ['constraint', 'limitation', 'choke point'],
                'resource': ['memory', 'cpu', 'processor', 'time', 'space'],
                'complexity': ['cost', 'efficiency', 'big-o', 'overhead'],
                'profile': ['benchmark', 'measure', 'analyze', 'assess'],
            },
            'dependency': {
                'dependency': ['import', 'require', 'reference', 'inclusion'],
                'import': ['include', 'require', 'load', 'reference'],
                'library': ['package', 'module', 'dependency', 'framework'],
                'uses': ['depends on', 'references', 'calls', 'needs'],
                'connection': ['relationship', 'link', 'association', 'coupling'],
                'circular': ['recursive', 'cyclical', 'loop', 'cycle'],
            },
            'general': {
                'code': ['source', 'implementation', 'script'],
                'error': ['bug', 'issue', 'problem', 'exception', 'fault'],
                'fix': ['resolve', 'solve', 'correct', 'handle'],
                'example': ['sample', 'instance', 'demo', 'illustration'],
                'works': ['functions', 'operates', 'behaves', 'runs'],
            }
        }
        
        # Common programming synonyms for all intents
        common_synonyms = {
            'variable': ['var', 'field', 'property', 'attribute'],
            'function': ['method', 'procedure', 'routine', 'subroutine'],
            'loop': ['iteration', 'cycle', 'repeat'],
            'condition': ['check', 'test', 'if statement', 'branch'],
            'array': ['list', 'collection', 'sequence', 'vector'],
            'object': ['instance', 'entity', 'class instance'],
            'string': ['text', 'char array', 'characters'],
            'number': ['integer', 'float', 'numeric', 'value'],
            'boolean': ['bool', 'flag', 'logical value', 'truth value'],
        }
        
        # Add common synonyms to all intent categories
        for intent_type in synonyms:
            for term, syn_list in common_synonyms.items():
                if term not in synonyms[intent_type]:
                    synonyms[intent_type][term] = syn_list
        
        # Function to expand keywords with synonyms while preserving the original weight
        def expand_with_synonyms(keywords: List[str], intent_type: str) -> List[str]:
            expanded = keywords.copy()  # Keep original keywords with full weight
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check if we have synonyms for this keyword
                for term, syn_list in synonyms.get(intent_type, {}).items():
                    if term == keyword_lower or keyword_lower in syn_list:
                        # Add all synonyms that aren't already in the list
                        for syn in [term] + syn_list:
                            if syn != keyword_lower and syn not in (k.lower() for k in expanded):
                                expanded.append(syn)
            return expanded
        
        # Expand each query component
        expanded_query = {
            'file_patterns': parsed_query['file_patterns'],  # File patterns usually shouldn't be expanded
            'function_names': parsed_query['function_names'],  # Function names should be exact
            'class_names': parsed_query['class_names'],  # Class names should be exact
            'keywords': expand_with_synonyms(parsed_query['keywords'], intent)
        }
        
        return expanded_query
        
    def _search_for_exact_file_matches(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for exact file matches based on the query.
        
        Args:
            query: The search query
            
        Returns:
            List of matching files
        """
        results = []
        
        # Extract potential file patterns from the query
        parsed = self._parse_search_query(query)
        file_patterns = parsed['file_patterns']
        
        if file_patterns:
            try:
                # Get all repository files
                all_files = self.db.get_repository_files(self.repo_id)
                
                # Find matches
                for pattern in file_patterns:
                    matches = [f for f in all_files if pattern.lower() in f.get('path', '').lower()]
                    for match in matches:
                        if match not in results:
                            results.append(match)
                            
            except Exception as e:
                logger.warning(f"Error searching for exact file matches: {str(e)}")
                
        return results
        
    def _has_relevant_functions(self, file_id: int, parsed_queries: Dict[str, List[str]]) -> bool:
        """
        Check if the file has functions relevant to the query.
        
        Args:
            file_id: The file ID
            parsed_queries: The parsed query components
            
        Returns:
            True if relevant functions are found
        """
        try:
            functions = self.db.get_file_functions(file_id)
            if not functions:
                return False
                
            # Check for function name matches
            for func in functions:
                func_name = func.get('name', '').lower()
                
                # Check against function names in query
                for name in parsed_queries['function_names']:
                    if name.lower() in func_name:
                        return True
                        
                # Check against keywords
                for keyword in parsed_queries['keywords']:
                    if keyword.lower() in func_name:
                        return True
                        
            return False
        except Exception as e:
            logger.debug(f"Error checking for relevant functions: {str(e)}")
            return False
            
    def _has_relevant_classes(self, file_id: int, parsed_queries: Dict[str, List[str]]) -> bool:
        """
        Check if the file has classes relevant to the query.
        
        Args:
            file_id: The file ID
            parsed_queries: The parsed query components
            
        Returns:
            True if relevant classes are found
        """
        try:
            classes = self.db.get_file_classes(file_id)
            if not classes:
                return False
                
            # Check for class name matches
            for cls in classes:
                cls_name = cls.get('name', '').lower()
                
                # Check against class names in query
                for name in parsed_queries['class_names']:
                    if name.lower() in cls_name:
                        return True
                        
                # Check against keywords
                for keyword in parsed_queries['keywords']:
                    if keyword.lower() in cls_name:
                        return True
                        
            return False
        except Exception as e:
            logger.debug(f"Error checking for relevant classes: {str(e)}")
            return False
        
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector or None if generation fails
        """
        try:
            # Check for OpenAI API key
            if "OPENAI_API_KEY" not in os.environ:
                logger.warning("OPENAI_API_KEY environment variable is required for embeddings")
                logger.info("Using fallback embedding method")
                # Fallback method - generate a deterministic but simple embedding using hash
                import hashlib
                import numpy as np
                
                # Create a deterministic seed from the text hash
                seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                
                # Generate a lower-dimensional embedding (similar to embeddings)
                return np.random.randn(1536).tolist()
                
            # Use OpenAI for embeddings
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
            
    def _get_context_for_query(self, query: str, max_results: int = 3) -> str:
        """
        Get repository context based on the query.
        
        Args:
            query: The search query
            max_results: Number of results to include
            
        Returns:
            Context string for the query
        """
        # Determine the intent of the query
        query_intent = self._classify_query_intent(query)
        
        # For specific intents, use specialized context generators
        if query_intent == 'function_search':
            return self._get_function_centric_context(query, max_results)
        elif query_intent == 'architecture':
            return self._get_architecture_context(query, max_results)
        elif query_intent == 'performance':
            return self._get_performance_context(query, max_results)
        elif query_intent == 'dependency' and 'database' in query.lower():
            return self._get_database_schema_context(query, max_results)
            
        # For repository-level queries, include the repository summary
        if any(term in query.lower() for term in ['repository', 'repo', 'codebase', 'project', 'structure']):
            context = self._get_repository_summary()
            context += "\n## Additional context\n\n"
        else:
            context = "Here are the most relevant code snippets for your query:\n\n"
        
        # Default context generation for other intents - use caching
        search_results = self._get_cached_or_search(query, top_k=max_results)
        
        if not search_results:
            if 'context' in locals() and 'Additional context' in context:
                # If we already have repository summary, just indicate no additional context found
                context += "No specific code snippets found for this query. Please refine your search if needed.\n\n"
                return context
            else:
                return "No relevant code found for this query."
                
        for result in search_results:
            file_path = result.get("path", "Unknown")
            snippet = result.get("snippet", "")
            
            context += f"## File: {file_path}\n\n"
            if snippet:
                context += f"```\n{snippet}\n```\n\n"
            else:
                context += "(No snippet available)\n\n"
                
        return context
    
    def _get_function_centric_context(self, query: str, max_results: int = 3) -> str:
        """
        Generate function-centric context for function-related queries.
        
        Args:
            query: The search query
            max_results: Maximum number of results to include
            
        Returns:
            Context string focused on functions
        """
        # Expand query with function-specific synonyms
        expanded_query = self._expand_query(query, 'function_search')
        
        # Search for relevant code
        search_results = self.search_code(query, top_k=max_results * 2)  # Get more results initially
        
        if not search_results:
            return "No relevant functions found for this query."
            
        context = "Here are the most relevant functions for your query:\n\n"
        added_functions = 0
        
        for result in search_results:
            if added_functions >= max_results:
                break
                
            file_id = result.get("id")
            file_path = result.get("path", "Unknown")
            
            # Try to get specific functions from this file
            try:
                functions = self.db.get_file_functions(file_id)
                relevant_functions = []
                
                # Filter for relevant functions based on the query
                for func in functions:
                    func_name = func.get("name", "")
                    
                    # Check if function matches any of our expanded query terms
                    if any(term.lower() in func_name.lower() for term in expanded_query['function_names'] + expanded_query['keywords']):
                        relevant_functions.append(func)
                
                if relevant_functions:
                    context += f"## File: {file_path}\n\n"
                    
                    for func in relevant_functions[:3]:  # Limit to 3 functions per file
                        func_name = func.get("name", "Unknown")
                        func_signature = func.get("signature", "")
                        func_body = func.get("body", "")
                        
                        context += f"### Function: {func_name}\n\n"
                        
                        if func_signature:
                            context += f"**Signature:**\n```\n{func_signature}\n```\n\n"
                            
                        if func_body:
                            # Limit function body to a reasonable size
                            if len(func_body) > 500:
                                func_body = func_body[:500] + "\n... (function continues)"
                                
                            context += f"**Implementation:**\n```\n{func_body}\n```\n\n"
                            
                        # Try to find callers and callees
                        try:
                            callers = self.db.get_function_callers(file_id, func_name)
                            callees = self.db.get_function_callees(file_id, func_name)
                            
                            if callers:
                                context += "**Called by:**\n"
                                for caller in callers[:5]:  # Limit to 5 callers
                                    caller_name = caller.get("caller_name", "Unknown")
                                    caller_file = caller.get("file_path", "Unknown")
                                    context += f"- `{caller_name}` in {caller_file}\n"
                                context += "\n"
                                
                            if callees:
                                context += "**Calls:**\n"
                                for callee in callees[:5]:  # Limit to 5 callees
                                    callee_name = callee.get("callee_name", "Unknown")
                                    callee_file = callee.get("file_path", "Unknown")
                                    context += f"- `{callee_name}` in {callee_file}\n"
                                context += "\n"
                                
                        except Exception as e:
                            logger.debug(f"Error retrieving function relationships: {str(e)}")
                        
                        added_functions += 1
                        
                        if added_functions >= max_results:
                            break
            
            except Exception as e:
                logger.debug(f"Error retrieving functions for file {file_id}: {str(e)}")
        
        if added_functions == 0:
            # Fallback to generic code snippets
            context = "Couldn't find specific functions matching your query. Here are some relevant code snippets:\n\n"
            
            for result in search_results[:max_results]:
                file_path = result.get("path", "Unknown")
                snippet = result.get("snippet", "")
                
                context += f"## File: {file_path}\n\n"
                if snippet:
                    context += f"```\n{snippet}\n```\n\n"
                else:
                    context += "(No snippet available)\n\n"
        
        return context
        
    def _get_architecture_context(self, query: str, max_results: int = 3) -> str:
        """
        Generate architecture-centric context for system design queries.
        
        Args:
            query: The search query
            max_results: Maximum number of results to include
            
        Returns:
            Context string focused on system architecture
        """
        # Expand query with architecture-specific synonyms
        expanded_query = self._expand_query(query, 'architecture')
        
        context = "# Repository Architecture Overview\n\n"
        
        # 1. Add repository structure information
        try:
            # Get directory structure
            dir_structure = self._get_repository_summary()
            context += "## Directory Structure\n\n"
            context += dir_structure + "\n\n"
        except Exception as e:
            logger.debug(f"Error getting repository structure: {str(e)}")
            context += "Could not retrieve directory structure.\n\n"
        
        # 2. Look for architecture-defining files (READMEs, architecture docs, etc.)
        architecture_files = []
        try:
            files = self.db.get_repository_files(self.repo_id)
            
            # Look for files that typically define architecture
            for file in files:
                path = file.get("path", "").lower()
                
                # Check for typical architecture documentation files
                if any(arch_file in path for arch_file in ["readme", "architecture", "design", "overview", 
                                                           "structure", "docs/", "documentation/", 
                                                           "spec", "blueprint", "architecture.md", 
                                                           "system", "components"]):
                    architecture_files.append(file)
                    
                # Also look for files with "main" or "app" in root directory (entry points)
                elif path.count("/") <= 1 and any(entry in path for entry in ["main.", "app.", "index.", "server.", "client."]):
                    architecture_files.append(file)
        except Exception as e:
            logger.debug(f"Error finding architecture files: {str(e)}")
        
        # Add README and architecture document content
        if architecture_files:
            context += "## Key Architecture Files\n\n"
            added_files = 0
            
            # Sort by relevance (READMEs and architecture docs first)
            architecture_files.sort(key=lambda x: 
                0 if "readme" in x.get("path", "").lower() else
                1 if "architecture" in x.get("path", "").lower() else
                2 if "design" in x.get("path", "").lower() else
                3
            )
            
            for file in architecture_files:
                if added_files >= max_results:
                    break
                    
                file_id = file.get("id")
                file_path = file.get("path", "")
                
                try:
                    file_data = self.db.get_file_content(file_id)
                    if file_data and "content" in file_data:
                        content = file_data["content"]
                        
                        # For long files, extract the most important sections
                        if len(content) > 1000:
                            lines = content.split("\n")
                            
                            # Look for headings and important sections
                            important_sections = []
                            current_section = []
                            in_architecture_section = False
                            
                            for line in lines:
                                # Check for headings
                                if line.startswith("#") or line.startswith("==") or line.startswith("--"):
                                    # If we were in an architecture section, add what we've collected
                                    if in_architecture_section and current_section:
                                        important_sections.extend(current_section)
                                        current_section = []
                                    
                                    # Check if this heading is about architecture
                                    lower_line = line.lower()
                                    if any(arch_term in lower_line for arch_term in ["architect", "design", "structure", 
                                                                                    "overview", "component", "system", 
                                                                                    "module", "organization"]):
                                        in_architecture_section = True
                                    else:
                                        in_architecture_section = False
                                
                                # Add lines if we're in an architecture section
                                if in_architecture_section:
                                    current_section.append(line)
                                
                                # Also add any import statements, as they define dependencies
                                elif "import " in line or "require" in line or "include" in line:
                                    important_sections.append(line)
                            
                            # Add any remaining architectural section
                            if in_architecture_section and current_section:
                                important_sections.extend(current_section)
                            
                            # If we found important sections, use those
                            if important_sections:
                                content = "\n".join(important_sections)
                            else:
                                # Otherwise just take the beginning of the file
                                content = "\n".join(lines[:50])
                        
                        context += f"### {file_path}\n\n"
                        context += f"```\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```\n\n"
                        added_files += 1
                except Exception as e:
                    logger.debug(f"Error getting content for file {file_id}: {str(e)}")
        
        # 3. Add information about key components and their relationships
        try:
            # Find main modules or components based on directory structure
            directories = set()
            for file in files:
                path = file.get("path", "")
                parts = path.split("/")
                if len(parts) > 1:
                    directories.add(parts[0])
            
            # Filter out common non-component directories
            non_components = [".git", ".github", "node_modules", "venv", "env", ".venv", "build", 
                             "dist", "tests", "test", "__pycache__"]
            components = [d for d in directories if d not in non_components and not d.startswith(".")]
            
            if components:
                context += "## Key Components\n\n"
                
                for component in components[:5]:  # Limit to top 5 components
                    # Count files in this component
                    component_files = [f for f in files if f.get("path", "").startswith(f"{component}/")]
                    
                    context += f"### {component}\n"
                    context += f"Contains {len(component_files)} files\n\n"
                    
                    # Try to determine what this component does based on its files
                    try:
                        # Get all file paths in this component
                        file_paths = [f.get("path") for f in component_files]
                        
                        # Check for typical patterns
                        if any("controller" in path.lower() for path in file_paths):
                            context += "- Contains controllers (handles user requests)\n"
                        if any("model" in path.lower() for path in file_paths):
                            context += "- Contains data models\n"
                        if any("view" in path.lower() for path in file_paths):
                            context += "- Contains views or UI components\n"
                        if any("service" in path.lower() for path in file_paths):
                            context += "- Contains services (business logic)\n"
                        if any("api" in path.lower() for path in file_paths):
                            context += "- Contains API definitions or implementations\n"
                        if any("util" in path.lower() or "helper" in path.lower() for path in file_paths):
                            context += "- Contains utilities or helper functions\n"
                        if any("test" in path.lower() for path in file_paths):
                            context += "- Contains tests\n"
                        if any("config" in path.lower() for path in file_paths):
                            context += "- Contains configuration files\n"
                            
                        context += "\n"
                    except Exception as e:
                        logger.debug(f"Error analyzing component {component}: {str(e)}")
            
        except Exception as e:
            logger.debug(f"Error analyzing components: {str(e)}")
        
        # 4. Finally, add results from semantic search for the query
        search_results = self.search_code(query, top_k=max_results)
        
        if search_results:
            context += "## Relevant Files for Your Architecture Query\n\n"
            
            for result in search_results:
                file_path = result.get("path", "Unknown")
                snippet = result.get("snippet", "")
                
                context += f"### {file_path}\n\n"
                if snippet:
                    context += f"```\n{snippet}\n```\n\n"
                else:
                    context += "(No snippet available)\n\n"
        
        return context
    
    def _get_repository_summary(self) -> str:
        """Generate a summary of the repository including statistics and structure."""
        summary = f"# Repository Summary: {self.repo_name}\n\n"
        
        # Get repository statistics
        try:
            function_count = self.db.get_function_count(self.repo_id)
            class_count = self.db.get_class_count(self.repo_id)
            import_count = self.db.get_import_count(self.repo_id)
            files = self.db.get_repository_files(self.repo_id)
            
            summary += f"## Statistics\n\n"
            summary += f"- **Files**: {len(files) if files else 'Unknown'}\n"
            summary += f"- **Functions**: {function_count if function_count else 'Unknown'}\n"
            summary += f"- **Classes**: {class_count if class_count else 'Unknown'}\n"
            summary += f"- **Imports**: {import_count if import_count else 'Unknown'}\n\n"
            
            # Add language distribution
            if files:
                languages = {}
                for file in files:
                    ext = os.path.splitext(file.get('path', ''))[1].lower()
                    if ext:
                        languages[ext] = languages.get(ext, 0) + 1
                
                if languages:
                    summary += f"## Language Distribution\n\n"
                    for ext, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(files)) * 100
                        summary += f"- {ext}: {count} files ({percentage:.1f}%)\n"
                    summary += "\n"
            
            # Get key files (those with most functions or classes)
            if files:
                try:
                    # First get file IDs
                    file_ids = [file.get('id') for file in files if file.get('id')]
                    
                    # Get function and class counts per file
                    file_stats = {}
                    for file_id in file_ids[:50]:  # Limit to avoid too many queries
                        file_path = next((f.get('path') for f in files if f.get('id') == file_id), "Unknown")
                        functions = self.db.get_file_functions(file_id)
                        classes = self.db.get_file_classes(file_id)
                        file_stats[file_path] = {
                            'functions': len(functions) if functions else 0,
                            'classes': len(classes) if classes else 0,
                            'complexity': (len(functions) if functions else 0) + (len(classes) if classes else 0) * 2  # Weight classes more
                        }
                    
                    # Sort by complexity
                    key_files = sorted(file_stats.items(), key=lambda x: x[1]['complexity'], reverse=True)[:10]
                    
                    if key_files:
                        summary += f"## Key Files\n\n"
                        for file_path, stats in key_files:
                            summary += f"- **{file_path}**: {stats['functions']} functions, {stats['classes']} classes\n"
                        summary += "\n"
                except Exception as e:
                    logger.warning(f"Error getting key files: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Error getting repository statistics: {str(e)}")
        
        # Include repository digest if available
        if self.digest_url:
            try:
                digest_path = self.db.download_repository_digest(self.repo_id)
                
                if digest_path and os.path.exists(digest_path):
                    with open(digest_path, 'r', encoding='utf-8') as f:
                        digest_content = f.read()
                    
                    excerpt = digest_content[:1000] + "..." if len(digest_content) > 1000 else digest_content
                    summary += f"\n## Repository Structure (from digest)\n\n{excerpt}\n"
            except Exception as e:
                logger.warning(f"Error including digest: {str(e)}")
                
        return summary

    def _initialize_repository_context(self):
        """Initialize the repository context with key information about the repository."""
        # Get repository summary
        summary = self._get_repository_summary()
        
        # Get repository structure (directory tree)
        try:
            # Fetch all files
            files = self.db.get_repository_files(self.repo_id)
            
            # Build directory structure
            dirs = {}
            for file in files:
                path = file.get('path', '')
                parts = path.split('/')
                
                # Handle Windows paths
                if '\\' in path:
                    parts = path.split('\\')
                
                current = dirs
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:  # file
                        if '__files__' not in current:
                            current['__files__'] = []
                        current['__files__'].append(part)
                    else:  # directory
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            
            # Format directory structure
            dir_structure = "## Repository Structure\n\n"
            
            def format_dir(d, prefix="", is_last=True, indent=""):
                result = ""
                items = list(d.items())
                
                # Sort items to put __files__ last
                items.sort(key=lambda x: "zzz" if x[0] == "__files__" else x[0])
                
                for i, (k, v) in enumerate(items):
                    is_last_item = i == len(items) - 1
                    
                    if k == "__files__":
                        for j, f in enumerate(sorted(v)):
                            file_is_last = (is_last_item and j == len(v) - 1)
                            if file_is_last:
                                result += f"{indent}└── {f}\n"
                            else:
                                result += f"{indent}├── {f}\n"
                    else:
                        if is_last_item:
                            result += f"{indent}└── {k}/\n"
                            result += format_dir(v, f"{k}/", True, f"{indent}    ")
                        else:
                            result += f"{indent}├── {k}/\n"
                            result += format_dir(v, f"{k}/", False, f"{indent}│   ")
                return result
            
            dir_structure += format_dir(dirs)
            
        except Exception as e:
            logger.warning(f"Error generating directory structure: {str(e)}")
            dir_structure = "## Repository Structure\n\nUnable to generate directory structure."
        
        # Get repository stats
        try:
            stats = "## Repository Statistics\n\n"
            stats += f"- Files: {len(files)}\n"
            
            # Count functions, classes, imports
            try:
                data, count = self.db.supabase.table('functions').select('count').eq('repository_id', self.repo_id).execute()
                if data and hasattr(data, 'data') and len(data.data) > 0:
                    function_count = data.data[0]['count']
                    stats += f"- Functions: {function_count}\n"
            except Exception as func_err:
                logger.warning(f"Error counting functions: {str(func_err)}")
            
            try:
                data, count = self.db.supabase.table('classes').select('count').eq('repository_id', self.repo_id).execute()
                if data and hasattr(data, 'data') and len(data.data) > 0:
                    class_count = data.data[0]['count']
                    stats += f"- Classes: {class_count}\n"
            except Exception as class_err:
                logger.warning(f"Error counting classes: {str(class_err)}")
            
            try:
                data, count = self.db.supabase.table('imports').select('count').eq('repository_id', self.repo_id).execute()
                if data and hasattr(data, 'data') and len(data.data) > 0:
                    import_count = data.data[0]['count']
                    stats += f"- Imports: {import_count}\n"
            except Exception as import_err:
                logger.warning(f"Error counting imports: {str(import_err)}")
            
        except Exception as e:
            logger.warning(f"Error generating repository stats: {str(e)}")
            stats = "## Repository Statistics\n\nUnable to generate repository statistics."
        
        # Get digest excerpt
        digest_excerpt = ""
        if self.digest_url:
            try:
                digest_path = self.db.download_repository_digest(self.repo_id)
                if digest_path:
                    with open(digest_path, 'r', encoding='utf-8') as f:
                        digest_content = f.read()
                        
                    # Get first 1000 chars
                    digest_excerpt = "## Repository Digest\n\n"
                    digest_excerpt += digest_content[:2000] + "...\n\n"
                    digest_excerpt += "(There's more in the full digest. Ask for specific information if needed.)"
            except Exception as e:
                logger.warning(f"Error including digest: {str(e)}")
        
        # Combine all information
        self.repository_context = f"{summary}\n\n{stats}\n\n{dir_structure}\n\n{digest_excerpt}"
        return self.repository_context

    def _get_detailed_search_results(self, query: str, max_results: int = 5) -> str:
        """
        Get detailed search results for a specific query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to include
            
        Returns:
            Formatted results string
        """
        # Search for relevant code
        results = self.search_code(query, top_k=max_results)
        
        if not results:
            return "No relevant code found for this specific search."
            
        context = "Here are the detailed search results relevant to your query:\n\n"
        
        for i, result in enumerate(results):
            file_path = result.get('path', 'Unknown')
            context += f"## File: {file_path}\n\n"
            
            # Add snippet with more context if available
            if 'snippet' in result:
                context += "```\n"
                context += result['snippet']
                context += "\n```\n\n"
                
                # Try to get full file content for more context
                try:
                    full_content = self.db.get_file_content_by_path(self.repo_id, file_path)
                    if full_content:
                        # Extract a slightly larger context from the full content
                        snippet = result['snippet']
                        start_pos = full_content.find(snippet[:min(50, len(snippet))])
                        
                        if start_pos >= 0:
                            # Get 10 lines before and after
                            content_lines = full_content.split('\n')
                            snippet_lines = snippet.split('\n')
                            
                            # Find approximate line number
                            line_count = 0
                            pos = 0
                            while pos < start_pos and pos >= 0:
                                pos = full_content.find('\n', pos + 1)
                                line_count += 1
                            
                            start_line = max(0, line_count - 5)
                            end_line = min(len(content_lines), line_count + len(snippet_lines) + 5)
                            
                            context += "**More context:**\n\n"
                            context += f"```\n# Lines {start_line+1}-{end_line} of {file_path}\n"
                            context += '\n'.join(content_lines[start_line:end_line])
                            context += "\n```\n\n"
                except Exception as e:
                    logger.warning(f"Error getting additional context: {str(e)}")
                
            # Add any relevant functions or classes
            try:
                file_id = result.get('id')
                if file_id:
                    # Get functions
                    functions = self.db.get_file_functions(file_id)
                    if functions:
                        context += f"### Functions in {file_path}:\n\n"
                        for func in functions[:5]:  # Limit to 5 functions per file
                            context += f"**{func.get('name', 'Unknown')}**: `{func.get('signature', '')}`\n\n"
                            if func.get('docstring'):
                                context += f"{func.get('docstring')}\n\n"
                            if func.get('body'):
                                context += "```python\n"
                                context += func.get('body')
                                context += "\n```\n\n"
                        
                    # Get classes
                    classes = self.db.get_file_classes(file_id)
                    if classes:
                        context += f"### Classes in {file_path}:\n\n"
                        for cls in classes[:3]:  # Limit to 3 classes per file
                            context += f"**{cls.get('name', 'Unknown')}**\n\n"
                            if cls.get('docstring'):
                                context += f"{cls.get('docstring')}\n\n"
                            if cls.get('body'):
                                context += "```python\n"
                                context += cls.get('body')
                                context += "\n```\n\n"
            except Exception as e:
                logger.warning(f"Error getting additional context: {str(e)}")
                
        return context

    def _get_performance_context(self, query: str, max_results: int = 3) -> str:
        """
        Generate performance-centric context for performance-related queries.
        
        Args:
            query: The search query
            max_results: Maximum number of results to include
            
        Returns:
            Context string focused on performance aspects
        """
        # Expand query with performance-specific synonyms
        expanded_query = self._expand_query(query, 'performance')
        
        # Performance-related keywords to look for
        perf_keywords = [
            "performance", "optimize", "optimization", "speed", "fast", "slow", "bottleneck",
            "memory", "cpu", "resource", "usage", "consumption", "leak", "profile", "benchmark",
            "complexity", "algorithm", "efficient", "inefficient", "latency", "throughput",
            "cache", "caching", "parallel", "concurrent", "async", "synchronous", "blocking",
            "non-blocking", "thread", "process", "lock", "mutex", "semaphore", "race condition",
            "deadlock", "starvation", "time complexity", "space complexity", "big-o", "O(n)",
            "O(1)", "O(n²)", "O(log n)", "O(n log n)", "expensive", "cheap", "cost"
        ]
        
        # Always include repository summary for performance queries
        context = self._get_repository_summary()
        context += "\n# Performance Analysis Context\n\n"
        
        # Search for relevant code with performance implications
        search_results = self.search_code(query, top_k=max_results * 2)  # Get more results initially
        
        if not search_results:
            context += "No specific code found with direct performance implications. The above repository summary may help you understand the structure and scale of the codebase.\n\n"
            return context
            
        # 1. Look for algorithmic complexity indicators
        context += "## Algorithmic Complexity Analysis\n\n"
        
        complexity_patterns = {
            "O(1)": "Constant time",
            "O\\(1\\)": "Constant time",
            "O(n)": "Linear time",
            "O\\(n\\)": "Linear time",
            "O(n²)": "Quadratic time",
            "O\\(n\\^2\\)": "Quadratic time",
            "O(n^2)": "Quadratic time",
            "O(n log n)": "Linearithmic time",
            "O\\(n log n\\)": "Linearithmic time",
            "O(log n)": "Logarithmic time",
            "O\\(log n\\)": "Logarithmic time",
            "O(2^n)": "Exponential time",
            "O\\(2\\^n\\)": "Exponential time"
        }
        
        # Count occurrences of complexity patterns
        complexity_found = False
        for result in search_results:
            if not isinstance(result, dict) or 'snippet' not in result:
                continue
                
            snippet = result.get('snippet', '')
            path = result.get('path', 'Unknown file')
            
            if not snippet:
                continue
                
            # Check for complexity patterns
            for pattern, description in complexity_patterns.items():
                if re.search(pattern, snippet, re.IGNORECASE):
                    context += f"Found **{description}** complexity in `{path}`:\n\n```\n{snippet}\n```\n\n"
                    complexity_found = True
        
        if not complexity_found:
            context += "No explicit time complexity annotations found in the codebase.\n\n"
            
        # 2. Analyze loops and potential bottlenecks
        context += "## Potential Performance Bottlenecks\n\n"
        
        bottlenecks_found = False
        for result in search_results:
            if not isinstance(result, dict) or 'snippet' not in result:
                continue
                
            snippet = result.get('snippet', '')
            path = result.get('path', 'Unknown file')
            
            if not snippet:
                continue
                
            # Look for nested loops
            nested_loop_pattern = r'for.*?\{.*?for.*?\{|while.*?\{.*?while.*?\{|for.*?\{.*?while.*?\{|while.*?\{.*?for.*?\{'
            if re.search(nested_loop_pattern, snippet, re.DOTALL):
                context += f"Potential bottleneck - nested loops in `{path}`:\n\n```\n{snippet}\n```\n\n"
                bottlenecks_found = True
                
            # Look for resource-intensive operations
            resource_patterns = [
                r'read_file', r'write_file', r'open\(.*?,.*(w|r)', r'serialize', r'deserialize',
                r'encode', r'decode', r'json\.loads', r'json\.dumps', r'pickle\.load', r'pickle\.dump',
                r'subprocess\.', r'os\.system', r'exec\(', r'eval\(', r'requests\.', r'urllib',
                r'download', r'upload', r'\.execute\(', r'query', r'cursor', r'\.commit\(',
            ]
            
            for pattern in resource_patterns:
                if re.search(pattern, snippet, re.IGNORECASE | re.DOTALL):
                    context += f"Potential bottleneck - resource-intensive operation in `{path}`:\n\n```\n{snippet}\n```\n\n"
                    bottlenecks_found = True
                    break
        
        if not bottlenecks_found:
            context += "No obvious performance bottlenecks identified in the examined code.\n\n"
            
        # 3. Check for caching mechanisms
        context += "## Caching and Optimization Techniques\n\n"
        
        optimization_found = False
        for result in search_results:
            if not isinstance(result, dict) or 'snippet' not in result:
                continue
                
            snippet = result.get('snippet', '')
            path = result.get('path', 'Unknown file')
            
            if not snippet:
                continue
                
            # Look for caching
            cache_patterns = [
                r'cache', r'memoize', r'memoization', r'lru_cache', r'cached_property',
                r'@cache', r'@lru_cache', r'@memoize', r'@cached_property'
            ]
            
            for pattern in cache_patterns:
                if re.search(pattern, snippet, re.IGNORECASE):
                    context += f"Found caching mechanism in `{path}`:\n\n```\n{snippet}\n```\n\n"
                    optimization_found = True
                    break
                    
            # Look for performance-related constants/parameters
            if re.search(r'BATCH_SIZE|CHUNK_SIZE|BUFFER_SIZE|TIMEOUT|THRESHOLD|MAX_|MIN_', snippet):
                context += f"Found performance-related configuration in `{path}`:\n\n```\n{snippet}\n```\n\n"
                optimization_found = True
        
        if not optimization_found:
            context += "No explicit caching or optimization mechanisms found in the examined code.\n\n"
            
        return context

    def _get_database_schema_context(self, query: str, max_results: int = 3) -> str:
        """
        Generate database-centric context for database schema related queries.
        
        Args:
            query: The search query
            max_results: Maximum number of results to include
            
        Returns:
            Context string focused on database schema
        """
        # Expand query with database-specific synonyms
        expanded_query = self._expand_query(query, 'dependency')
        
        # Database-related keywords to look for
        db_keywords = [
            "database", "db", "schema", "table", "column", "field", "index", "primary key", "foreign key",
            "relation", "query", "sql", "nosql", "orm", "migration", "model", "entity", "attribute",
            "datastore", "repository", "dao", "data access", "data layer", "persistence", "storage",
            "crud", "insert", "update", "delete", "select", "join", "transaction", "commit", "rollback"
        ]
        
        context = "# Database Schema Information\n\n"
        
        # 1. First look for database definition files
        schema_files = []
        try:
            files = self.db.get_repository_files(self.repo_id)
            
            # Look for files that typically define database schema
            for file in files:
                path = file.get("path", "").lower()
                
                # Check for typical schema files
                if any(schema_file in path for schema_file in [
                    "schema.", "models.", "entities.", "migrations", "database", 
                    ".sql", "orm.", "db.", "dao.", "repository.", "model."
                ]):
                    schema_files.append(file)
        except Exception as e:
            logger.debug(f"Error finding schema files: {str(e)}")
        
        # 2. Extract and analyze schema definition content
        if schema_files:
            context += "## Database Schema Definitions\n\n"
            
            # Sort schema files for relevance
            schema_files.sort(key=lambda x: 
                0 if "schema" in x.get("path", "").lower() else
                1 if "models" in x.get("path", "").lower() else
                2 if "migrations" in x.get("path", "").lower() else
                3 if "database" in x.get("path", "").lower() else
                4
            )
            
            tables_found = []
            columns_found = {}
            relationships_found = []
            
            for file in schema_files[:max_results]:
                file_id = file.get("id")
                file_path = file.get("path", "")
                
                try:
                    file_data = self.db.get_file_content(file_id)
                    if file_data and "content" in file_data:
                        content = file_data["content"]
                        
                        # Add the schema file content
                        context += f"### {file_path}\n\n"
                        
                        # Extract schema information
                        # 1. Look for table definitions
                        table_patterns = [
                            r"CREATE\s+TABLE\s+(\w+)",  # SQL
                            r"class\s+(\w+).*Model",  # ORM models
                            r"class\s+(\w+).*Entity",  # JPA/Entity
                            r"@Table\(.*name\s*=\s*[\"'](\w+)[\"']",  # JPA/Hibernate annotation
                            r"DbSet<(\w+)>",  # Entity Framework
                            r"Schema::create\([\"'](\w+)[\"']"  # Laravel
                        ]
                        
                        for pattern in table_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                table_name = match.group(1)
                                if table_name not in tables_found:
                                    tables_found.append(table_name)
                        
                        # 2. Look for column definitions
                        column_patterns = [
                            r"(\w+)\s+(\w+)\s*\(.*\)",  # SQL
                            r"@Column\(.*name\s*=\s*[\"'](\w+)[\"']",  # JPA/Hibernate annotation
                            r"(\w+)\s*=\s*(?:models|db)\.(\w+)Field",  # Django models
                            r"public\s+(\w+)\s+(\w+)\s*\{",  # Java/C# properties
                            r"(\w+)\s*:\s*(\w+)",  # TypeScript interfaces
                        ]
                        
                        for pattern in column_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                # Different patterns have different group meanings
                                if len(match.groups()) >= 2:
                                    col_name = match.group(1)
                                    col_type = match.group(2)
                                    
                                    # Try to associate columns with tables
                                    for table in tables_found:
                                        # Check if column is near table definition
                                        table_pos = content.find(table)
                                        col_pos = match.start()
                                        
                                        if 0 <= table_pos < col_pos and col_pos - table_pos < 1000:
                                            if table not in columns_found:
                                                columns_found[table] = []
                                            columns_found[table].append((col_name, col_type))
                                            break
                        
                        # 3. Look for relationship definitions
                        relationship_patterns = [
                            r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)",  # SQL
                            r"@ForeignKey\([\"'](\w+)[\"']\s*,\s*[\"'](\w+)[\"']\)",  # ORM annotation
                            r"@ManyToOne\(\)\s+(\w+)\s+(\w+)",  # JPA
                            r"@OneToMany\(\s*mappedBy\s*=\s*[\"'](\w+)[\"']\)",  # JPA
                            r"(\w+)_id\s*=\s*(?:models|db)\.ForeignKey\([\"'](\w+)[\"']",  # Django
                        ]
                        
                        for pattern in relationship_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                relationship = match.group(0)
                                relationships_found.append(relationship)
                        
                        # Show a snippet of the file (first 30 lines or 1000 chars)
                        lines = content.split('\n')
                        snippet = '\n'.join(lines[:min(30, len(lines))])
                        if len(snippet) > 1000:
                            snippet = snippet[:1000] + "\n..."
                            
                        context += f"```\n{snippet}\n```\n\n"
                except Exception as e:
                    logger.debug(f"Error getting content for file {file_id}: {str(e)}")
            
            # Add extracted schema information
            if tables_found:
                context += "## Tables Identified\n\n"
                for table in tables_found:
                    context += f"### {table}\n\n"
                    
                    if table in columns_found and columns_found[table]:
                        context += "| Column | Type |\n| ------ | ---- |\n"
                        for col_name, col_type in columns_found[table]:
                            context += f"| {col_name} | {col_type} |\n"
                        context += "\n"
                    else:
                        context += "No columns identified for this table.\n\n"
            
            if relationships_found:
                context += "## Relationships Identified\n\n"
                for relationship in relationships_found:
                    context += f"- {relationship}\n"
                context += "\n"
        
        # 3. Look for database query code
        query_files = []
        try:
            # Search for files with database queries
            search_results = self.search_code(query, top_k=max_results * 2)
            
            for result in search_results:
                file_id = result.get("id")
                file_path = result.get("path", "")
                snippet = result.get("snippet", "")
                
                # Check if the file contains database operations
                if snippet and any(kw in snippet.lower() for kw in db_keywords):
                    query_files.append(result)
        except Exception as e:
            logger.debug(f"Error finding query files: {str(e)}")
        
        if query_files:
            context += "## Database Operations and Queries\n\n"
            
            for file in query_files[:max_results]:
                file_path = file.get("path", "")
                snippet = file.get("snippet", "")
                
                context += f"### {file_path}\n\n"
                
                if snippet:
                    # Extract database operations from snippets
                    lines = snippet.split('\n')
                    db_operation_lines = []
                    
                    for line in lines:
                        if any(kw in line.lower() for kw in db_keywords):
                            db_operation_lines.append(line)
                    
                    if db_operation_lines:
                        context += "Database operations:\n```\n"
                        context += '\n'.join(db_operation_lines[:10])  # Limit to 10 operations
                        if len(db_operation_lines) > 10:
                            context += "\n... (more operations not shown)"
                        context += "\n```\n\n"
                    else:
                        context += f"```\n{snippet}\n```\n\n"
                else:
                    context += "(No snippet available)\n\n"
        
        # 4. Look for database configuration
        config_files = []
        try:
            # Look for files that might contain database configuration
            for file in files:
                path = file.get("path", "").lower()
                
                if any(config_file in path for config_file in [
                    "database.config", "db.config", "application.properties", "application.yml",
                    "settings.py", "config.py", "environment", ".env", "connection", "datasource"
                ]):
                    config_files.append(file)
        except Exception as e:
            logger.debug(f"Error finding config files: {str(e)}")
        
        if config_files:
            context += "## Database Configuration\n\n"
            
            for file in config_files[:max_results]:
                file_id = file.get("id")
                file_path = file.get("path", "")
                
                try:
                    file_data = self.db.get_file_content(file_id)
                    if file_data and "content" in file_data:
                        content = file_data["content"]
                        
                        context += f"### {file_path}\n\n"
                        
                        # Extract configuration settings
                        config_lines = []
                        lines = content.split('\n')
                        
                        for line in lines:
                            # Look for DB configuration but exclude passwords
                            if any(kw in line.lower() for kw in ["database", "db", "connection", "url", "host", "port", "user"]):
                                # Redact passwords
                                redacted_line = re.sub(r"(password|passwd|pwd)(\s*[:=]\s*)(['\"]\w+['\"])", r"\1\2'***REDACTED***'", line)
                                config_lines.append(redacted_line)
                        
                        if config_lines:
                            context += "Database configuration settings:\n```\n"
                            context += '\n'.join(config_lines)
                            context += "\n```\n\n"
                        else:
                            context += "No clear database configuration found in this file.\n\n"
                except Exception as e:
                    logger.debug(f"Error getting content for file {file_id}: {str(e)}")
        
        # If we haven't found much specific DB info, add general search results
        if not schema_files and not query_files and not config_files:
            context += "## General Database-Related Code\n\n"
            
            search_results = self.search_code("database schema model entity orm", top_k=max_results)
            
            for result in search_results:
                file_path = result.get("path", "Unknown")
                snippet = result.get("snippet", "")
                
                context += f"### {file_path}\n\n"
                if snippet:
                    context += f"```\n{snippet}\n```\n\n"
                else:
                    context += "(No snippet available)\n\n"
                    
        return context 

    def chat(self, user_input: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            AI response
        """
        # Check if the input is a disambiguation selection
        if self._is_disambiguation_selection(user_input):
            return self._handle_disambiguation_selection(user_input)
            
        # Check if this is a direct query about the repository itself
        repo_info_keywords = ['tell me about this repo', 'repository info', 'about the repo', 
                             'about this project', 'project info', 'codebase info', 
                             'what is this project', 'describe this repo', 'repository structure',
                             'repo structure', 'what does this repo do', 'give me an overview']
                             
        is_repo_info_query = any(keyword in user_input.lower() for keyword in repo_info_keywords)
        
        # Classify the query intent to optimize search and response strategy
        query_intent = self._classify_query_intent(user_input)
        search_query = user_input
        
        # Check for query ambiguity
        ambiguity_info = self._check_query_ambiguity(user_input)
        if ambiguity_info['is_ambiguous']:
            return self._generate_disambiguation_options(user_input, ambiguity_info)
        
        # These checks are now redundant with our intent classification but kept for backward compatibility
        file_indicators = ["show me", "show file", "display", "view", "get the file", "what's in the file", "what is in", "content of", "readme"]
        file_specific_request = query_intent == 'file_lookup' or any(indicator in user_input.lower() for indicator in file_indicators)
        
        is_direct_search = user_input.lower().startswith(("show me", "find", "search for", "locate", "where is", "how does"))
        
        # Get relevant context based on the query
        context = ""
        
        # For direct repository info queries, provide repository summary
        if is_repo_info_query:
            context = self._get_repository_summary()
        elif query_intent == 'file_lookup':
            # This is handled by the file_specific_request block below
            pass
        
        if file_specific_request:
            # Try to extract the filename
            import re
            
            file_pattern = None
            for indicator in file_indicators:
                if indicator in user_input.lower():
                    # Extract the part after the indicator
                    parts = user_input.lower().split(indicator, 1)
                    if len(parts) > 1:
                        potential_file = parts[1].strip().strip("'\" ")
                        if potential_file:
                            file_pattern = potential_file
                            break
            
            if file_pattern:
                # Try to find the file
                matching_files = self._search_for_exact_file_matches(file_pattern)
                
                if matching_files:
                    file_id = matching_files[0].get('id')
                    file_path = matching_files[0].get('path')
                    
                    if file_id:
                        try:
                            file_data = self.db.get_file_content(file_id)
                            if file_data and 'content' in file_data:
                                content = file_data['content']
                                
                                # Format response with file content
                                if len(content) > 4000:
                                    content = content[:2000] + "\n\n... (content truncated) ...\n\n" + content[-2000:]
                                
                                response = f"Here's the content of {file_path}:\n\n```\n{content}\n```"
                                
                                # Add to history
                                self.history.append({
                                    "role": "user",
                                    "content": user_input
                                })
                                self.history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                                return response
                        except Exception as e:
                            logger.error(f"Error getting file content: {str(e)}")
        
        # Standard context-based response
        if query_intent == 'function_explanation':
            context = self._get_function_centric_context(user_input)
        elif query_intent == 'architecture_overview':
            context = self._get_architecture_context(user_input)
        elif query_intent == 'performance_analysis':
            context = self._get_performance_context(user_input)
        elif query_intent == 'database_schema':
            context = self._get_database_schema_context(user_input)
        else:
            # Default search
            context = self._get_context_for_query(user_input)
            
        logger.debug(f"Generated context for query intent '{query_intent}'. Context length: {len(context)}")
        if not context or len(context) < 50:
            logger.warning(f"Context retrieval may have failed - very short or empty context: '{context}'")
        
        # Add the query to history
        self.history.append({
            "role": "user",
            "content": user_input
        })
        
        # Build the prompt with context
        messages = [
            {
                "role": "system",
                "content": f"{self.system_message}\n\nThe current repository being analyzed is: {self.repo_name}\n\nContext information:\n{context}"
            }
        ]
        
        # Add history to the messages
        for msg in self.history:
            messages.append(msg)
        
        # Generate response
        try:
            response = None
            
            if self.model_provider == "gemini":
                # Gemini requires a different message format
                gemini_messages = []
                
                # First, convert previous history to Gemini format
                history_messages = []
                for msg in self.history:
                    history_messages.append({
                        "role": "user" if msg["role"] == "user" else "model",
                        "parts": [{"text": msg["content"]}]
                    })
                
                # Add the current user's message with the system message and context
                system_and_query = {
                    "role": "user",
                    "parts": [{"text": f"{self.system_message}\n\nThe current repository being analyzed is: {self.repo_name}\n\nContext information:\n{context}\n\nUser query: {user_input}"}]
                }
                
                # Format final content for the model - system message should come first
                if history_messages:
                    # If we have history, add the current query with context
                    content = self.llm.generate_content(history_messages + [system_and_query])
                else:
                    # If this is the first message, just use system + query
                    content = self.llm.generate_content([system_and_query])
                    
                response = content.text
            elif self.model_provider == "openai":
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    temperature=self.temperature
                )
                response = completion.choices[0].message.content
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
            
            # Add the response to history
            self.history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
    
    def _check_query_ambiguity(self, query: str) -> Dict[str, Any]:
        """
        Check if a query is ambiguous and needs disambiguation.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with ambiguity information
        """
        result = {
            'is_ambiguous': False,
            'ambiguity_type': None,
            'options': []
        }
        
        # Parse the query
        parsed_query = self._parse_search_query(query)
        
        # Check for multiple conflicting query elements (e.g., multiple file patterns)
        if len(parsed_query['file_patterns']) > 1:
            result['is_ambiguous'] = True
            result['ambiguity_type'] = 'multiple_files'
            result['options'] = parsed_query['file_patterns']
            return result
            
        # Check for file pattern that matches multiple files
        if len(parsed_query['file_patterns']) == 1:
            file_pattern = parsed_query['file_patterns'][0]
            matching_files = self._search_for_exact_file_matches(file_pattern)
            
            if len(matching_files) > 3:  # More than 3 matching files is ambiguous
                result['is_ambiguous'] = True
                result['ambiguity_type'] = 'many_matching_files'
                result['options'] = [file['path'] for file in matching_files[:5]]  # Limit to 5 options
                return result
                
        # Check for function names that match multiple functions
        if len(parsed_query['function_names']) >= 1:
            function_name = parsed_query['function_names'][0]
            
            # Search for functions with this name across the repository
            matching_functions = []
            try:
                # Get all files
                files = self.db.get_repository_files(self.repo_id)
                
                # Check each file for matching functions
                for file in files[:50]:  # Limit to first 50 files for performance
                    file_id = file.get('id')
                    if not file_id:
                        continue
                        
                    functions = self.db.get_file_functions(file_id)
                    for func in functions:
                        if function_name.lower() in func.get('name', '').lower():
                            matching_functions.append({
                                'name': func.get('name'),
                                'file_path': file.get('path')
                            })
                            
                            # Limit to 5 matching functions
                            if len(matching_functions) >= 5:
                                break
                    
                    if len(matching_functions) >= 5:
                        break
                        
                if len(matching_functions) > 1:
                    result['is_ambiguous'] = True
                    result['ambiguity_type'] = 'multiple_functions'
                    result['options'] = [f"{func['name']} in {func['file_path']}" for func in matching_functions]
                    return result
                    
            except Exception as e:
                logger.debug(f"Error checking for function ambiguity: {str(e)}")
        
        # Check for class names that match multiple classes
        if len(parsed_query['class_names']) >= 1:
            class_name = parsed_query['class_names'][0]
            
            # Search for classes with this name across the repository
            matching_classes = []
            try:
                # Get all files
                files = self.db.get_repository_files(self.repo_id)
                
                # Check each file for matching classes
                for file in files[:50]:  # Limit to first 50 files for performance
                    file_id = file.get('id')
                    if not file_id:
                        continue
                        
                    classes = self.db.get_file_classes(file_id)
                    for cls in classes:
                        if class_name.lower() in cls.get('name', '').lower():
                            matching_classes.append({
                                'name': cls.get('name'),
                                'file_path': file.get('path')
                            })
                            
                            # Limit to 5 matching classes
                            if len(matching_classes) >= 5:
                                break
                    
                    if len(matching_classes) >= 5:
                        break
                        
                if len(matching_classes) > 1:
                    result['is_ambiguous'] = True
                    result['ambiguity_type'] = 'multiple_classes'
                    result['options'] = [f"{cls['name']} in {cls['file_path']}" for cls in matching_classes]
                    return result
                    
            except Exception as e:
                logger.debug(f"Error checking for class ambiguity: {str(e)}")
        
        # Check for vague queries (very short or too general)
        word_count = len(query.split())
        
        if word_count <= 2 and not any(parsed_query[key] for key in ['file_patterns', 'function_names', 'class_names']):
            # Query is too short and doesn't contain specific code elements
            search_results = self._get_cached_or_search(query, top_k=10)
            
            if len(search_results) >= 5:  # If we get many different results, it's probably ambiguous
                result['is_ambiguous'] = True
                result['ambiguity_type'] = 'vague_query'
                
                # Extract major topics from search results
                topics = []
                for res in search_results[:5]:
                    file_path = res.get('path', '')
                    parts = file_path.split('/')
                    if len(parts) > 1:
                        topics.append(parts[0])  # Use top-level directory as topic
                    else:
                        topics.append(file_path)  # Use filename as topic
                
                # Deduplicate topics
                topics = list(set(topics))
                
                # Suggest refinements for the query
                result['options'] = [
                    f"{query} in {topic}" for topic in topics
                ]
                
                # Add general refinement suggestions
                if query_has_functionality_terms(query):
                    result['options'].append(f"how does {query} work")
                    result['options'].append(f"show me the implementation of {query}")
                else:
                    result['options'].append(f"show files related to {query}")
                    result['options'].append(f"find functions that handle {query}")
                
                return result
        
        return result
    
    def _generate_disambiguation_options(self, query: str, ambiguity_info: Dict[str, Any]) -> str:
        """
        Generate a response with disambiguation options.
        
        Args:
            query: The original query
            ambiguity_info: Information about the ambiguity
            
        Returns:
            Response with disambiguation options
        """
        response = "I found multiple possibilities for your query. Could you please clarify which one you're interested in?\n\n"
        
        # Store the ambiguity info for later use when the user selects an option
        if not hasattr(self, 'disambiguation_state'):
            self.disambiguation_state = {}
            
        disambiguation_id = str(uuid.uuid4())
        self.disambiguation_state[disambiguation_id] = {
            'original_query': query,
            'ambiguity_info': ambiguity_info,
            'timestamp': time.time()
        }
        
        # Generate different messages based on ambiguity type
        ambiguity_type = ambiguity_info['ambiguity_type']
        options = ambiguity_info['options']
        
        if ambiguity_type == 'multiple_files':
            response += "I found multiple files that match your query. Which one would you like to know about?\n\n"
        elif ambiguity_type == 'many_matching_files':
            response += "I found many files matching that pattern. Here are some options:\n\n"
        elif ambiguity_type == 'multiple_functions':
            response += "I found multiple functions with that name. Which one are you referring to?\n\n"
        elif ambiguity_type == 'multiple_classes':
            response += "I found multiple classes with that name. Which one are you referring to?\n\n"
        elif ambiguity_type == 'vague_query':
            response += "Your query is a bit general. Here are some more specific options:\n\n"
        
        # List the options with selection numbers
        for i, option in enumerate(options, 1):
            response += f"{i}. {option}\n"
            
        # Add the disambiguation ID as a hidden reference
        response += f"\n(Reference ID: {disambiguation_id})\n"
        response += "\nPlease respond with the number of your selection, or rephrase your query for a different search."
        
        return response
    
    def _is_disambiguation_selection(self, user_input: str) -> bool:
        """
        Check if the user input is a disambiguation selection.
        
        Args:
            user_input: The user's input
            
        Returns:
            True if the input is a disambiguation selection
        """
        # Check if the input is just a number (option selection)
        if user_input.strip().isdigit():
            return True
            
        # Check for phrases like "option 1", "number 2", etc.
        option_patterns = [
            r"^option\s+(\d+)$",
            r"^(\d+)$",
            r"^number\s+(\d+)$",
            r"^select\s+(\d+)$",
            r"^choose\s+(\d+)$",
            r"^pick\s+(\d+)$"
        ]
        
        for pattern in option_patterns:
            if re.match(pattern, user_input.strip(), re.IGNORECASE):
                return True
                
        # Check if the input contains a reference ID
        if hasattr(self, 'disambiguation_state'):
            for disambiguation_id in self.disambiguation_state:
                if disambiguation_id in user_input:
                    return True
        
        return False
    
    def _handle_disambiguation_selection(self, user_input: str) -> str:
        """
        Handle a disambiguation selection from the user.
        
        Args:
            user_input: The user's selection
            
        Returns:
            Response for the selected option
        """
        if not hasattr(self, 'disambiguation_state') or not self.disambiguation_state:
            return "I'm sorry, but I don't have any disambiguation options pending. Could you please rephrase your question?"
            
        # Extract the selection number
        selection = None
        for pattern in [r"^(\d+)$", r"option\s+(\d+)", r"number\s+(\d+)", r"select\s+(\d+)", r"choose\s+(\d+)", r"pick\s+(\d+)"]:
            match = re.search(pattern, user_input.strip(), re.IGNORECASE)
            if match:
                selection = int(match.group(1))
                break
                
        # If we couldn't extract a selection number, look for a reference ID
        disambiguation_id = None
        if selection is None:
            for d_id in self.disambiguation_state:
                if d_id in user_input:
                    disambiguation_id = d_id
                    break
        else:
            # Find the most recent disambiguation state
            most_recent_time = 0
            for d_id, state in self.disambiguation_state.items():
                if state['timestamp'] > most_recent_time:
                    most_recent_time = state['timestamp']
                    disambiguation_id = d_id
        
        if disambiguation_id is None or disambiguation_id not in self.disambiguation_state:
            return "I'm sorry, but I couldn't find the corresponding disambiguation options. Could you please rephrase your question?"
            
        # Get the disambiguation state
        state = self.disambiguation_state[disambiguation_id]