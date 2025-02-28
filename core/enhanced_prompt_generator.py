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

import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.prompt_generator')

# Load environment variables
load_dotenv()

# Default models
DEFAULT_MODEL = "gemini-1.5-flash"
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.2


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
        
        # Check if repository data exists
        self.data_path = os.path.join(repository_dir, 'repository_data.json')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Repository data not found at {self.data_path}")
        
        # Check for analysis data
        self.analysis_path = os.path.join(repository_dir, 'analysis.json')
        self.has_analysis = os.path.exists(self.analysis_path)
        
        # Check for AST data
        self.ast_path = os.path.join(repository_dir, 'ast_data.json')
        self.has_ast = os.path.exists(self.ast_path)
        
        # Check for embeddings
        self.embeddings_path = os.path.join(repository_dir, 'embeddings.json')
        self.has_embeddings = os.path.exists(self.embeddings_path)
        
        # Initialize Gemini
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            logger.warning("Using fallback prompt generation mode")
        
        self.gemini_available = bool(gemini_api_key)
        
        if self.gemini_available:
            genai.configure(api_key=gemini_api_key)
    
    def _load_repository_data(self) -> Dict[str, Any]:
        """
        Load repository data from disk.
        
        Returns:
            Repository data dictionary
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_analysis_data(self) -> Optional[Dict[str, Any]]:
        """
        Load analysis data from disk if available.
        
        Returns:
            Analysis data dictionary or None if not available
        """
        if not self.has_analysis:
            return None
        
        try:
            with open(self.analysis_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading analysis data: {str(e)}")
            return None
    
    def _load_ast_data(self) -> Optional[Dict[str, Any]]:
        """
        Load AST data from disk if available.
        
        Returns:
            AST data dictionary or None if not available
        """
        if not self.has_ast:
            return None
        
        try:
            with open(self.ast_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading AST data: {str(e)}")
            return None
    
    def find_relevant_contexts(self, task: str) -> Dict[str, Any]:
        """
        Find relevant context information for a task.
        
        Args:
            task: Task description
            
        Returns:
            Dictionary containing relevant context information
        """
        # Load repository data
        repo_data = self._load_repository_data()
        analysis_data = self._load_analysis_data()
        ast_data = self._load_ast_data()
        
        # Initialize context
        context = {
            "repository_name": repo_data["name"],
            "file_count": repo_data["file_count"],
            "file_types": repo_data["file_types"],
            "relevant_files": [],
            "relevant_functions": [],
            "relevant_snippets": []
        }
        
        # Find relevant files
        if self.has_analysis and analysis_data:
            # Use analyzer to find relevant files
            try:
                from core.analyze import RepositoryAnalyzer
                analyzer = RepositoryAnalyzer(self.repository_dir)
                relevant_files = analyzer.find_relevant_files(task, top_k=8)
                context["relevant_files"] = relevant_files
            except ImportError:
                # Fallback to random selection
                logger.warning("Analyzer not available for finding relevant files")
                files = [f["metadata"]["path"] for f in repo_data["files"]]
                sample_size = min(8, len(files))
                context["relevant_files"] = [{"path": path, "similarity": 0.0} for path in random.sample(files, sample_size)]
        else:
            # Fallback to random selection
            files = [f["metadata"]["path"] for f in repo_data["files"]]
            sample_size = min(8, len(files))
            context["relevant_files"] = [{"path": path, "similarity": 0.0} for path in random.sample(files, sample_size)]
        
        # Find relevant functions if AST data is available
        if self.has_ast and ast_data and "functions" in ast_data:
            # Use AST data to find relevant functions
            functions = ast_data["functions"]
            
            # Simple keyword matching for now
            task_words = task.lower().split()
            
            # Score functions based on term matching
            scored_functions = []
            for func in functions:
                # Skip functions without name or path
                if "name" not in func or "path" not in func:
                    continue
                
                function_text = f"{func['name']} {func.get('signature', '')} {func.get('docstring', '')}"
                function_text = function_text.lower()
                
                # Calculate score based on keyword matches
                score = sum(3 if word in func['name'].lower() else
                           1 if word in function_text else 0
                           for word in task_words)
                
                if score > 0:
                    scored_functions.append({
                        "function": func,
                        "score": score
                    })
            
            # Sort by score
            scored_functions.sort(key=lambda x: x["score"], reverse=True)
            
            # Select top functions
            top_functions = [sf["function"] for sf in scored_functions[:5]]
            context["relevant_functions"] = top_functions
        
        # Extract snippets from relevant files
        for file_info in context["relevant_files"]:
            file_path = file_info["path"]
            
            # Find file in repo data
            for file_entry in repo_data["files"]:
                if file_entry["metadata"]["path"] == file_path:
                    # Extract a snippet (first ~50 lines or less)
                    content = file_entry["content"]
                    lines = content.splitlines()
                    snippet_lines = lines[:min(50, len(lines))]
                    snippet = "\n".join(snippet_lines)
                    
                    context["relevant_snippets"].append({
                        "path": file_path,
                        "snippet": snippet
                    })
                    break
        
        return context
    
    def generate_prompt_template(self, task: str, context: Dict[str, Any]) -> str:
        """
        Generate a prompt template for a task.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            Prompt template as a string
        """
        # Start with task description
        prompt = [
            "# Task\n",
            f"{task}\n\n",
            "# Repository Information\n",
            f"Repository name: {context['repository_name']}\n",
            f"Number of files: {context['file_count']}\n",
            "File types: " + ", ".join([f"{ext}: {count}" for ext, count in list(context['file_types'].items())[:5]]),
            "\n\n"
        ]
        
        # Add relevant files
        if context["relevant_files"]:
            prompt.append("# Relevant Files\n")
            for file_info in context["relevant_files"]:
                similarity = file_info.get("similarity", 0) * 100  # Convert to percentage
                prompt.append(f"- {file_info['path']} (relevance: {similarity:.1f}%)\n")
            prompt.append("\n")
        
        # Add relevant functions
        if context["relevant_functions"]:
            prompt.append("# Relevant Functions\n")
            for func in context["relevant_functions"]:
                prompt.append(f"## {func['name']} in {func['path']}\n")
                if "signature" in func:
                    prompt.append(f"Signature: `{func['signature']}`\n")
                if "docstring" in func and func["docstring"]:
                    prompt.append(f"Description: {func['docstring']}\n")
                prompt.append("\n")
            prompt.append("\n")
        
        # Add code snippets
        if context["relevant_snippets"]:
            prompt.append("# Code Snippets\n")
            for snippet_info in context["relevant_snippets"]:
                prompt.append(f"## From {snippet_info['path']}\n")
                prompt.append("```\n")
                prompt.append(snippet_info["snippet"])
                prompt.append("\n```\n\n")
        
        # Add instructions for the AI
        prompt.append("# Instructions\n")
        prompt.append("Based on the repository context above, provide a solution for the given task.\n")
        prompt.append("Make sure your solution is compatible with the existing codebase and follows its patterns and conventions.\n")
        prompt.append("If the provided context is insufficient, indicate what additional information would be helpful.\n")
        
        return "".join(prompt)
    
    def enhance_prompt_with_gemini(self, task: str, template: str) -> str:
        """
        Enhance a prompt template using Gemini.
        
        Args:
            task: Task description
            template: Prompt template
            
        Returns:
            Enhanced prompt as a string
        """
        if not self.gemini_available:
            logger.warning("Gemini not available, using template as is")
            return template
        
        try:
            # Setup Gemini
            generation_config = {
                "temperature": TEMPERATURE,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            # Create prompt for Gemini
            gemini_prompt = f"""
            You are an expert prompt engineer for code generation tasks. 
            Your goal is to optimize the following prompt to get the best code generation results.
            
            The original task is: {task}
            
            Here's a draft prompt that includes context from a repository:
            
            {template}
            
            Please analyze this prompt and improve it by:
            1. Removing any irrelevant or distracting information
            2. Better organizing the information to guide the model
            3. Adding clear instructions and expectations
            4. Focusing on the most relevant context for the task
            5. Making sure the prompt is clear, concise, and effective
            
            Return only the optimized prompt, ready to be used for code generation.
            """
            
            # Generate optimized prompt
            response = model.generate_content(gemini_prompt)
            
            # Extract response text
            if hasattr(response, 'text'):
                optimized_prompt = response.text
                return optimized_prompt
            else:
                logger.warning("Unexpected response format from Gemini, using template")
                return template
            
        except Exception as e:
            logger.error(f"Error enhancing prompt with Gemini: {str(e)}")
            return template
    
    def generate_prompt(self, task: str) -> str:
        """
        Generate an optimized prompt for a code generation task.
        
        Args:
            task: Task description
            
        Returns:
            Optimized prompt as a string
        """
        logger.info(f"Generating prompt for task: {task}")
        
        # Find relevant context
        context = self.find_relevant_contexts(task)
        
        # Generate prompt template
        template = self.generate_prompt_template(task, context)
        
        # Enhance template with Gemini
        optimized_prompt = self.enhance_prompt_with_gemini(task, template)
        
        return optimized_prompt


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate an optimized prompt for a code generation task")
    parser.add_argument("repository_dir", help="Directory containing repository data (.promptpilot folder)")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL})")
    
    args = parser.parse_args()
    
    try:
        generator = PromptGenerator(args.repository_dir, model=args.model)
        prompt = generator.generate_prompt(args.task)
        
        print("\n" + "=" * 80)
        print("GENERATED PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        raise
