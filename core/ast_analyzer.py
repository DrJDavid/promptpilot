"""
AST (Abstract Syntax Tree) analyzer module for PromptPilot.

This module handles the Tree-sitter integration for generating and analyzing
AST representations of code files.
"""

import os
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import platform
import importlib.util
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.ast_analyzer')

# Try to import tree-sitter or mark as unavailable
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
    logger.info("Tree-sitter is available")
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter is not available (pip install tree-sitter)")


# Supported languages and their tree-sitter grammar repositories
SUPPORTED_LANGUAGES = {
    'python': {
        'extensions': ['.py'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-python',
        'parser_file': 'python.so',
    },
    'javascript': {
        'extensions': ['.js', '.jsx'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-javascript',
        'parser_file': 'javascript.so',
    },
    'typescript': {
        'extensions': ['.ts', '.tsx'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-typescript',
        'parser_file': 'typescript.so',
        'scope': 'typescript/src'
    },
    'java': {
        'extensions': ['.java'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-java',
        'parser_file': 'java.so',
    },
    'c': {
        'extensions': ['.c', '.h'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-c',
        'parser_file': 'c.so',
    },
    'cpp': {
        'extensions': ['.cpp', '.hpp', '.cc', '.h'],
        'repo': 'https://github.com/tree-sitter/tree-sitter-cpp',
        'parser_file': 'cpp.so',
    },
}


class ASTAnalyzer:
    """Class to analyze code files using Tree-sitter."""
    
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
        
        # Set up tree-sitter parsers directory
        self.parsers_dir = os.path.join(repository_dir, 'parsers')
        os.makedirs(self.parsers_dir, exist_ok=True)
        
        # Initialize parsers if tree-sitter is available
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._initialize_parsers()
    
    def _initialize_parsers(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        # Check if parsers already exist
        if all(os.path.exists(os.path.join(self.parsers_dir, lang_info['parser_file'])) 
               for lang_info in SUPPORTED_LANGUAGES.values()):
            # Load existing parsers
            for lang, lang_info in SUPPORTED_LANGUAGES.items():
                try:
                    parser_path = os.path.join(self.parsers_dir, lang_info['parser_file'])
                    Language.build_library(parser_path, [os.path.join(self.parsers_dir, lang)])
                    
                    self.parsers[lang] = Parser()
                    self.parsers[lang].set_language(Language(parser_path, lang))
                    logger.info(f"Loaded parser for {lang}")
                except Exception as e:
                    logger.error(f"Error loading parser for {lang}: {str(e)}")
            return
        
        # Build parsers
        logger.info("Building tree-sitter parsers (this may take a moment)...")
        
        for lang, lang_info in SUPPORTED_LANGUAGES.items():
            try:
                # Clone repository if needed
                lang_dir = os.path.join(self.parsers_dir, lang)
                if not os.path.exists(lang_dir):
                    logger.info(f"Cloning {lang} grammar repository...")
                    subprocess.run(
                        ['git', 'clone', lang_info['repo'], lang_dir],
                        check=True, capture_output=True
                    )
                
                # Build parser
                parser_path = os.path.join(self.parsers_dir, lang_info['parser_file'])
                scope = [os.path.join(lang_dir, lang_info.get('scope', ''))]
                Language.build_library(parser_path, scope)
                
                # Initialize parser
                self.parsers[lang] = Parser()
                self.parsers[lang].set_language(Language(parser_path, lang))
                logger.info(f"Built parser for {lang}")
                
            except Exception as e:
                logger.error(f"Error building parser for {lang}: {str(e)}")
    
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
    
    def _extract_text_from_node(self, code_bytes: bytes, node) -> str:
        """
        Extract text from a node in the AST.
        
        Args:
            code_bytes: Source code as bytes
            node: Tree-sitter node
            
        Returns:
            Text content of the node
        """
        start_byte = node.start_byte
        end_byte = node.end_byte
        return code_bytes[start_byte:end_byte].decode('utf-8')
    
    def _extract_functions_from_python(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract functions from a Python AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of function information dictionaries
        """
        functions = []
        query_string = """
        (function_definition
          name: (identifier) @function_name
          parameters: (parameters) @parameters
          body: (block) @body) @function_def
        """
        
        parser = self.parsers['python']
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        # Group captures by function
        functions_data = {}
        for capture in captures:
            node, label = capture
            function_id = node.id if label == 'function_def' else None
            
            if function_id is not None:
                if function_id not in functions_data:
                    functions_data[function_id] = {'node': node}
                functions_data[function_id]['full_node'] = node
            elif label == 'function_name':
                parent_id = node.parent.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['name_node'] = node
            elif label == 'parameters':
                parent_id = node.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['parameters_node'] = node
            elif label == 'body':
                parent_id = node.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['body_node'] = node
        
        # Extract function information
        for func_id, data in functions_data.items():
            if 'name_node' in data and 'parameters_node' in data:
                function_name = self._extract_text_from_node(code_bytes, data['name_node'])
                parameters = self._extract_text_from_node(code_bytes, data['parameters_node'])
                
                # Extract docstring if present
                docstring = ""
                if 'body_node' in data:
                    body_node = data['body_node']
                    expressions = [child for child in body_node.children if child.type == 'expression_statement']
                    if expressions and len(expressions) > 0:
                        first_expr = expressions[0]
                        if first_expr.children and first_expr.children[0].type == 'string':
                            docstring = self._extract_text_from_node(code_bytes, first_expr.children[0])
                            # Clean up quotes from docstring
                            docstring = docstring.strip('"\'').strip()
                
                # Extract full function text
                full_function = self._extract_text_from_node(code_bytes, data['full_node'])
                
                functions.append({
                    'name': function_name,
                    'signature': f"def {function_name}{parameters}:",
                    'docstring': docstring,
                    'body': full_function,
                    'start_line': data['full_node'].start_point[0] + 1,
                    'end_line': data['full_node'].end_point[0] + 1
                })
        
        return functions
    
    def _extract_classes_from_python(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract classes from a Python AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of class information dictionaries
        """
        classes = []
        query_string = """
        (class_definition
          name: (identifier) @class_name
          body: (block) @body) @class_def
        """
        
        parser = self.parsers['python']
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        # Group captures by class
        classes_data = {}
        for capture in captures:
            node, label = capture
            class_id = node.id if label == 'class_def' else None
            
            if class_id is not None:
                if class_id not in classes_data:
                    classes_data[class_id] = {'node': node}
                classes_data[class_id]['full_node'] = node
            elif label == 'class_name':
                parent_id = node.parent.id
                if parent_id in classes_data:
                    classes_data[parent_id]['name_node'] = node
            elif label == 'body':
                parent_id = node.parent.id
                if parent_id in classes_data:
                    classes_data[parent_id]['body_node'] = node
        
        # Extract class information
        for class_id, data in classes_data.items():
            if 'name_node' in data:
                class_name = self._extract_text_from_node(code_bytes, data['name_node'])
                
                # Extract docstring if present
                docstring = ""
                if 'body_node' in data:
                    body_node = data['body_node']
                    expressions = [child for child in body_node.children if child.type == 'expression_statement']
                    if expressions and len(expressions) > 0:
                        first_expr = expressions[0]
                        if first_expr.children and first_expr.children[0].type == 'string':
                            docstring = self._extract_text_from_node(code_bytes, first_expr.children[0])
                            # Clean up quotes from docstring
                            docstring = docstring.strip('"\'').strip()
                
                # Extract full class text
                full_class = self._extract_text_from_node(code_bytes, data['full_node'])
                
                classes.append({
                    'name': class_name,
                    'docstring': docstring,
                    'body': full_class,
                    'start_line': data['full_node'].start_point[0] + 1,
                    'end_line': data['full_node'].end_point[0] + 1
                })
        
        return classes
    
    def _extract_imports_from_python(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract imports from a Python AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of import information dictionaries
        """
        imports = []
        query_string = """
        (import_statement) @import
        (import_from_statement) @import_from
        """
        
        parser = self.parsers['python']
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        for capture in captures:
            node, _ = capture
            import_text = self._extract_text_from_node(code_bytes, node)
            
            imports.append({
                'statement': import_text,
                'line': node.start_point[0] + 1
            })
        
        return imports
    
    def _extract_functions_from_js_ts(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract functions from a JavaScript/TypeScript AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of function information dictionaries
        """
        functions = []
        query_string = """
        (function_declaration
          name: (identifier) @function_name
          parameters: (formal_parameters) @parameters
          body: (statement_block) @body) @function_decl
          
        (method_definition
          name: (property_identifier) @method_name
          parameters: (formal_parameters) @method_parameters
          body: (statement_block) @method_body) @method_def
          
        (arrow_function
          parameters: (formal_parameters) @arrow_params
          body: (_) @arrow_body) @arrow_func
        """
        
        lang = 'javascript'
        if 'typescript' in self.parsers:
            lang = 'typescript'
            
        parser = self.parsers[lang]
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        # Group captures by function
        functions_data = {}
        for capture in captures:
            node, label = capture
            
            if label in ('function_decl', 'method_def', 'arrow_func'):
                if node.id not in functions_data:
                    functions_data[node.id] = {'node': node, 'type': label}
            elif label in ('function_name', 'method_name'):
                parent_id = node.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['name_node'] = node
            elif label in ('parameters', 'method_parameters', 'arrow_params'):
                parent_id = node.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['parameters_node'] = node
            elif label in ('body', 'method_body', 'arrow_body'):
                parent_id = node.parent.id
                if parent_id in functions_data:
                    functions_data[parent_id]['body_node'] = node
        
        # Extract function information
        for func_id, data in functions_data.items():
            func_node = data['node']
            func_type = data.get('type', '')
            
            # Get function name if available
            function_name = "anonymous"
            if 'name_node' in data:
                function_name = self._extract_text_from_node(code_bytes, data['name_node'])
            elif func_type == 'arrow_func':
                # Try to get variable name for arrow functions
                parent = func_node.parent
                if parent and parent.type == 'variable_declarator':
                    grandparent = parent.parent
                    if grandparent and grandparent.type == 'lexical_declaration':
                        for child in grandparent.children:
                            if child.type == 'variable_declarator':
                                for subchild in child.children:
                                    if subchild.type == 'identifier':
                                        function_name = self._extract_text_from_node(code_bytes, subchild)
                                        break
            
            # Get parameters if available
            parameters = ""
            if 'parameters_node' in data:
                parameters = self._extract_text_from_node(code_bytes, data['parameters_node'])
            
            # Extract full function text
            full_function = self._extract_text_from_node(code_bytes, func_node)
            
            # Define signature based on function type
            signature = ""
            if func_type == 'function_decl':
                signature = f"function {function_name}{parameters}"
            elif func_type == 'method_def':
                signature = f"{function_name}{parameters}"
            elif func_type == 'arrow_func':
                signature = f"{function_name} = {parameters} =>"
            
            # Get JSDoc comments if available
            docstring = ""
            prev_sibling = func_node.prev_sibling
            if prev_sibling and prev_sibling.type == 'comment':
                docstring = self._extract_text_from_node(code_bytes, prev_sibling)
            
            functions.append({
                'name': function_name,
                'signature': signature,
                'docstring': docstring,
                'body': full_function,
                'start_line': func_node.start_point[0] + 1,
                'end_line': func_node.end_point[0] + 1
            })
        
        return functions
    
    def _extract_classes_from_js_ts(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract classes from a JavaScript/TypeScript AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of class information dictionaries
        """
        classes = []
        query_string = """
        (class_declaration
          name: (identifier) @class_name
          body: (class_body) @body) @class_decl
          
        (class
          name: (identifier) @class_name
          body: (class_body) @body) @class_expr
        """
        
        lang = 'javascript'
        if 'typescript' in self.parsers:
            lang = 'typescript'
            
        parser = self.parsers[lang]
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        # Group captures by class
        classes_data = {}
        for capture in captures:
            node, label = capture
            
            if label in ('class_decl', 'class_expr'):
                if node.id not in classes_data:
                    classes_data[node.id] = {'node': node}
            elif label == 'class_name':
                parent_id = node.parent.id
                if parent_id in classes_data:
                    classes_data[parent_id]['name_node'] = node
            elif label == 'body':
                parent_id = node.parent.id
                if parent_id in classes_data:
                    classes_data[parent_id]['body_node'] = node
        
        # Extract class information
        for class_id, data in classes_data.items():
            if 'name_node' in data:
                class_node = data['node']
                class_name = self._extract_text_from_node(code_bytes, data['name_node'])
                
                # Get JSDoc comments if available
                docstring = ""
                prev_sibling = class_node.prev_sibling
                if prev_sibling and prev_sibling.type == 'comment':
                    docstring = self._extract_text_from_node(code_bytes, prev_sibling)
                
                # Extract full class text
                full_class = self._extract_text_from_node(code_bytes, class_node)
                
                classes.append({
                    'name': class_name,
                    'docstring': docstring,
                    'body': full_class,
                    'start_line': class_node.start_point[0] + 1,
                    'end_line': class_node.end_point[0] + 1
                })
        
        return classes
    
    def _extract_imports_from_js_ts(self, code_bytes: bytes, root_node) -> List[Dict[str, Any]]:
        """
        Extract imports from a JavaScript/TypeScript AST.
        
        Args:
            code_bytes: Source code as bytes
            root_node: Root node of the AST
            
        Returns:
            List of import information dictionaries
        """
        imports = []
        query_string = """
        (import_statement) @import
        (import_declaration) @import_decl
        (export_statement) @export
        (require_call) @require
        """
        
        lang = 'javascript'
        if 'typescript' in self.parsers:
            lang = 'typescript'
            
        parser = self.parsers[lang]
        query = parser.language.query(query_string)
        captures = query.captures(root_node)
        
        for capture in captures:
            node, _ = capture
            import_text = self._extract_text_from_node(code_bytes, node)
            
            # For require calls, get the full statement
            if node.type == 'require_call':
                # Find parent statement
                parent = node.parent
                while parent and parent.type != 'expression_statement' and parent.type != 'variable_declaration':
                    parent = parent.parent
                
                if parent:
                    import_text = self._extract_text_from_node(code_bytes, parent)
            
            imports.append({
                'statement': import_text,
                'line': node.start_point[0] + 1
            })
        
        return imports
    
    def _analyze_ast(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """
        Analyze the AST of a file.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            
        Returns:
            Dictionary containing functions, classes, and imports
        """
        result = {
            'path': file_path,
            'language': language,
            'functions': [],
            'classes': [],
            'imports': [],
        }
        
        if not TREE_SITTER_AVAILABLE or language not in self.parsers:
            return result
        
        try:
            # Parse file content
            parser = self.parsers[language]
            code_bytes = content.encode('utf-8')
            tree = parser.parse(code_bytes)
            
            # Extract information based on language
            if language == 'python':
                result['functions'] = self._extract_functions_from_python(code_bytes, tree.root_node)
                result['classes'] = self._extract_classes_from_python(code_bytes, tree.root_node)
                result['imports'] = self._extract_imports_from_python(code_bytes, tree.root_node)
            elif language in ('javascript', 'typescript'):
                result['functions'] = self._extract_functions_from_js_ts(code_bytes, tree.root_node)
                result['classes'] = self._extract_classes_from_js_ts(code_bytes, tree.root_node)
                result['imports'] = self._extract_imports_from_js_ts(code_bytes, tree.root_node)
            # Add more language handlers here
            
            # Verify that extraction was successful
            if not result['functions'] and not result['classes'] and not result['imports']:
                # Fallback for unsupported language features
                logger.warning(f"No AST information extracted for {file_path}. Using fallback.")
                fallback_result = self._fallback_analyze(content, language)
                result.update(fallback_result)
                
        except Exception as e:
            logger.error(f"Error analyzing AST for {file_path}: {str(e)}")
            # Use fallback
            fallback_result = self._fallback_analyze(content, language)
            result.update(fallback_result)
        
        return result
    
    def _fallback_analyze(self, content: str, language: str) -> Dict[str, Any]:
        """
        Perform a basic analysis using regex-like pattern matching.
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
        logger.info(f"Analyzing repository AST in {self.repository_dir}")
        
        # Load repository data
        repo_data = self._load_repository_data()
        
        # Initialize results
        all_functions = []
        all_classes = []
        all_imports = []
        
        # Check if tree-sitter is available
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter is not available. Using fallback analysis.")
        
        # Process each file
        for file_entry in tqdm(repo_data['files'], desc="Analyzing AST"):
            file_path = file_entry['metadata']['path']
            content = file_entry['content']
            
            # Get language for file
            language = self._get_language_for_file(file_path)
            
            if not language:
                # Skip unsupported files
                continue
            
            # Analyze file
            logger.debug(f"Analyzing AST for {file_path} ({language})")
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
        
        # Save AST data
        self._save_ast_data(ast_data)
        
        logger.info(f"AST analysis complete: {len(all_functions)} functions, {len(all_classes)} classes, {len(all_imports)} imports")
        
        return ast_data


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
