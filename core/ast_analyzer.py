"""
AST (Abstract Syntax Tree) analyzer module for PromptPilot.

This module handles the analysis of code files to extract functions, 
classes, and imports - using regex-based parsing as a fallback for tree-sitter.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.ast_analyzer')

# Check for tree-sitter
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
    logger.info("Tree-sitter is available")
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter is not available, using regex-based fallback")

# Supported languages
SUPPORTED_LANGUAGES = {
    'python': {
        'extensions': ['.py'],
    },
    'javascript': {
        'extensions': ['.js', '.jsx'],
    },
    'typescript': {
        'extensions': ['.ts', '.tsx'],
    },
    'java': {
        'extensions': ['.java'],
    },
    'c': {
        'extensions': ['.c', '.h'],
    },
    'cpp': {
        'extensions': ['.cpp', '.hpp', '.cc', '.h'],
    },
}


class ASTAnalyzer:
    """Class to analyze code files and extract functions, classes, and imports."""
    
    def __init__(self, repository_dir: str):
        """
        Initialize the AST analyzer.
        
        Args:
            repository_dir: Directory containing repository data (.promptpilot folder)
        """
        self.repository_dir = repository_dir
        
        # Check if repository data exists
        self.data_path = os.path.join(repository_dir, 'repository_data.json')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Repository data not found at {self.data_path}")
        
        # Output path
        self.ast_data_path = os.path.join(repository_dir, 'ast_data.json')
        
        # Set up parsers directory (not actually used in fallback mode)
        self.parsers_dir = os.path.join(repository_dir, 'parsers')
        os.makedirs(self.parsers_dir, exist_ok=True)
        
        # For compatibility
        self.parsers = {}
    
    def _load_repository_data(self) -> Dict[str, Any]:
        """
        Load repository data from disk.
        
        Returns:
            Repository data dictionary
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_ast_data(self, ast_data: Dict[str, Any]) -> None:
        """
        Save AST data to disk.
        
        Args:
            ast_data: AST data dictionary
        """
        with open(self.ast_data_path, 'w', encoding='utf-8') as f:
            json.dump(ast_data, f, indent=2)
        
        logger.info(f"AST data saved to {self.ast_data_path}")
    
    def _get_language_for_file(self, file_path: str) -> Optional[str]:
        """
        Determine the programming language for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language identifier or None if unsupported
        """
        _, ext = os.path.splitext(file_path)
        
        for lang, lang_info in SUPPORTED_LANGUAGES.items():
            if ext in lang_info['extensions']:
                return lang
        
        return None
    
    def _analyze_ast(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """
        Analyze the file content to extract functions, classes, and imports.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            
        Returns:
            Dictionary containing functions, classes, and imports
        """
        # Always use the fallback method for simplicity
        result = {
            'path': file_path,
            'language': language,
            'functions': [],
            'classes': [],
            'imports': [],
        }
        
        try:
            # Use fallback analysis directly
            fallback_result = self._fallback_analyze(content, language)
            result.update(fallback_result)
                
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
        
        return result
    
    def _fallback_analyze(self, content: str, language: str) -> Dict[str, Any]:
        """
        Perform a basic analysis using regex patterns.
        Used when tree-sitter is unavailable or fails.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            Dictionary containing functions, classes, and imports
        """
        result = {
            'functions': [],
            'classes': [],
            'imports': [],
        }
        
        lines = content.splitlines()
        
        # Define patterns based on language
        function_patterns = []
        class_patterns = []
        import_patterns = []
        
        if language == 'python':
            function_patterns = [r'^\s*def\s+(\w+)\s*\(']
            class_patterns = [r'^\s*class\s+(\w+)']
            import_patterns = [r'^\s*import\s+', r'^\s*from\s+.*\s+import\s+']
        elif language in ('javascript', 'typescript'):
            function_patterns = [
                r'^\s*function\s+(\w+)\s*\(', 
                r'^\s*const\s+(\w+)\s*=\s*function',
                r'^\s*const\s+(\w+)\s*=\s*\(.*\)\s*=>'
            ]
            class_patterns = [r'^\s*class\s+(\w+)']
            import_patterns = [
                r'^\s*import\s+', 
                r'^\s*require\(', 
                r'^\s*export\s+'
            ]
        elif language in ('java', 'c', 'cpp'):
            function_patterns = [
                r'^\s*(public|private|protected|static|\s)*\s+[\w<>\[\]]+\s+(\w+)\s*\(',
                r'^\s*[\w<>\[\]]+\s+(\w+)\s*\('
            ]
            class_patterns = [
                r'^\s*(public|private|protected|static|\s)*\s+class\s+(\w+)',
                r'^\s*struct\s+(\w+)',
                r'^\s*enum\s+(\w+)'
            ]
            import_patterns = [
                r'^\s*import\s+', 
                r'^\s*#include\s+',
                r'^\s*using\s+'
            ]
        
        # Find functions
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith(('#', '//', '/*', '*')):
                continue
            
            # Check for functions
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    function_name = match.group(1) if len(match.groups()) >= 1 else "unknown"
                    
                    # Try to extract the full function
                    end_line = i
                    for j in range(i + 1, min(i + 50, len(lines))):
                        if j >= len(lines):
                            break
                        # Simple heuristic - indentation changes or empty line
                        if lines[j].strip() and (not lines[j].startswith(' ') or lines[j].startswith('}')):
                            end_line = j - 1
                            break
                        end_line = j
                    
                    function_body = '\n'.join(lines[i:end_line+1])
                    
                    # Try to extract docstring/comments (look at previous lines)
                    docstring = ""
                    for j in range(i - 1, max(0, i - 5), -1):
                        if lines[j].strip().startswith(('"""', "'''", "//", "/*", "*")):
                            docstring = lines[j].strip()
                            break
                    
                    result['functions'].append({
                        'name': function_name,
                        'signature': line.strip(),
                        'docstring': docstring,
                        'body': function_body,
                        'start_line': i + 1,
                        'end_line': end_line + 1
                    })
                    break
            
            # Check for classes
            for pattern in class_patterns:
                match = re.search(pattern, line)
                if match:
                    class_name = match.group(1) if len(match.groups()) >= 1 else "unknown"
                    
                    # Try to extract the class definition
                    end_line = i
                    brace_count = 0
                    for j in range(i, min(i + 200, len(lines))):
                        if j >= len(lines):
                            break
                            
                        # Count braces for languages with explicit blocks
                        if language in ('javascript', 'typescript', 'java', 'c', 'cpp'):
                            brace_count += lines[j].count('{') - lines[j].count('}')
                            if brace_count <= 0 and j > i:
                                end_line = j
                                break
                        else:
                            # For Python, use indentation
                            if j > i and lines[j].strip() and not lines[j].startswith(' '):
                                end_line = j - 1
                                break
                        end_line = j
                    
                    class_body = '\n'.join(lines[i:end_line+1])
                    
                    # Try to extract docstring (look at previous lines)
                    docstring = ""
                    for j in range(i - 1, max(0, i - 5), -1):
                        if lines[j].strip().startswith(('"""', "'''", "//", "/*", "*")):
                            docstring = lines[j].strip()
                            break
                    
                    result['classes'].append({
                        'name': class_name,
                        'docstring': docstring,
                        'body': class_body,
                        'start_line': i + 1,
                        'end_line': end_line + 1
                    })
                    break
            
            # Check for imports
            for pattern in import_patterns:
                if re.search(pattern, line):
                    result['imports'].append({
                        'statement': line.strip(),
                        'line': i + 1
                    })
                    break
        
        return result
    
    def analyze_repository(self) -> Dict[str, Any]:
        """
        Analyze the repository and generate AST data.
        
        Returns:
            Dictionary containing AST analysis results
        """
        logger.info(f"Analyzing repository code in {self.repository_dir}")
        
        # Load repository data
        repo_data = self._load_repository_data()
        
        # Initialize results
        all_functions = []
        all_classes = []
        all_imports = []
        
        # Process each file
        for file_entry in tqdm(repo_data['files'], desc="Analyzing code structures"):
            file_path = file_entry['metadata']['path']
            content = file_entry.get('content', '')
            
            # Get language for file
            language = self._get_language_for_file(file_path)
            
            if not language:
                # Skip unsupported files
                continue
            
            # Analyze file
            logger.debug(f"Analyzing {file_path} ({language})")
            result = self._analyze_ast(file_path, content, language)
            
            # Add path to each item
            for func in result['functions']:
                func['path'] = file_path
                all_functions.append(func)
            
            for cls in result['classes']:
                cls['path'] = file_path
                all_classes.append(cls)
            
            for imp in result['imports']:
                imp['path'] = file_path
                all_imports.append(imp)
        
        # Create AST data
        ast_data = {
            'functions': all_functions,
            'classes': all_classes,
            'imports': all_imports,
            'file_count': repo_data['file_count'],
            'function_count': len(all_functions),
            'class_count': len(all_classes),
            'import_count': len(all_imports)
        }
        
        # Save AST data locally
        self._save_ast_data(ast_data)
        
        # Try to store in database
        try:
            from core.enhanced_db import get_db
            db = get_db()
            
            if db and db.is_available():
                # Get repository ID
                repo_id = None
                repo_meta_path = os.path.join(self.repository_dir, 'repository_metadata.json')
                
                if os.path.exists(repo_meta_path):
                    with open(repo_meta_path, 'r') as f:
                        repo_meta = json.load(f)
                        repo_id = repo_meta.get('id')
                
                if not repo_id:
                    logger.warning("Repository ID not found, cannot store AST data in database")
                else:
                    # Store AST data in the database
                    logger.info(f"Storing AST data in database for repository {repo_id}")
                    try:
                        success = db.store_ast_data(repo_id, ast_data)
                        
                        if success:
                            logger.info("AST data successfully stored in database")
                        else:
                            logger.warning("Failed to store AST data in database")
                    except Exception as e:
                        logger.error(f"Error storing AST data: {str(e)}", exc_info=True)
                        logger.warning("Continuing with local AST data only")
            else:
                logger.warning("Database not available, AST data stored locally only")
        except Exception as e:
            logger.error(f"Error storing AST data in database: {e}")
            logger.info("AST data stored locally only")
        
        logger.info(f"Code analysis complete: {len(all_functions)} functions, {len(all_classes)} classes, {len(all_imports)} imports")
        
        return ast_data


class RepositoryASTAnalyzer(ASTAnalyzer):
    """
    Extension of ASTAnalyzer for use with the repository ingest process.
    This class maintains compatibility with the expected interface in ingest.py.
    """
    
    def __init__(self, repository_dir: str):
        """
        Initialize the repository AST analyzer.
        
        Args:
            repository_dir: Directory containing repository data (.promptpilot folder)
        """
        super().__init__(repository_dir)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the repository and generate AST data.
        This method is the expected interface for ingest.py.
        
        Returns:
            Dictionary containing AST analysis results
        """
        return self.analyze_repository()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze AST for a repository")
    parser.add_argument("repository_dir", help="Directory containing repository data (.promptpilot folder)")
    
    args = parser.parse_args()
    
    try:
        analyzer = ASTAnalyzer(args.repository_dir)
        ast_data = analyzer.analyze_repository()
        
        print("\nAST analysis complete!")
        print(f"Functions: {ast_data['function_count']}")
        print(f"Classes: {ast_data['class_count']}")
        print(f"Imports: {ast_data['import_count']}")
        print(f"Results saved to: {analyzer.ast_data_path}")
        
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise
