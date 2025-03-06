"""
Repository ingestion module for PromptPilot.

This module handles extracting file contents and metadata from a Git repository.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
import logging
from collections import Counter, defaultdict
import uuid
import time
import traceback

# git package is provided by the gitpython library
# If you're seeing an import error, run: pip install gitpython
import git
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.ingest')

# File extensions to process, add more as needed
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', 
    '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', 
    '.sh', '.bash', '.html', '.css', '.scss', '.md', '.json', '.yml', '.yaml'
}

# Files to ignore
IGNORE_FILES = {
    '.git', '.gitignore', '.gitmodules', '.gitattributes',
    'node_modules', 'venv', '.venv', 'env', '.env',
    '__pycache__', '.pytest_cache', '.mypy_cache', '.coverage',
    '.DS_Store', 'Thumbs.db', '.idea', '.vscode'
}

# Maximum file size to process (in bytes)
MAX_FILE_SIZE = 500 * 1024  # 500KB


class RepositoryIngestor:
    """Class to ingest and process a Git repository."""
    
    def __init__(self, repo_path: str, output_dir: Optional[str] = None):
        """
        Initialize the repository ingestor.
        
        Args:
            repo_path: Path to the Git repository
            output_dir: Directory to store processed data
        """
        self.repo_path = os.path.abspath(repo_path)
        
        # Determine repository name from path
        self.repo_name = os.path.basename(self.repo_path)
        
        # Determine output directory
        if output_dir is None:
            cwd = os.getcwd()
            default_dir = os.path.join(cwd, '.promptpilot')
            self.output_dir = os.path.join(default_dir, self.repo_name)
        else:
            self.output_dir = os.path.abspath(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if supabase_url and supabase_key:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized for file storage")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.supabase = None
        else:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set, using local storage only")
            self.supabase = None
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Check if a file should be processed based on extension and ignore rules.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be processed, False otherwise
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            return False
        
        # Check if file is too large
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logger.debug(f"Skipping large file: {file_path} ({file_size/1024:.1f}KB)")
            return False
        
        # Check if file or its directory should be ignored
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part in IGNORE_FILES:
                return False
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        return ext.lower() in CODE_EXTENSIONS
    
    def list_repository_files(self) -> List[str]:
        """
        List all files in the repository that should be processed.
        
        Returns:
            List of file paths
        """
        files = []
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_FILES]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                if self.should_process_file(file_path):
                    files.append(file_path)
        
        return files
    
    def _chunk_large_file(self, file_path: str, max_chunk_size: int = 100 * 1024) -> Optional[str]:
        """
        Handle large files by chunking them.
        
        Args:
            file_path: Path to the file
            max_chunk_size: Maximum chunk size in bytes
            
        Returns:
            File content as a string or None if file is too large
        """
        # Get total file size
        file_size = os.path.getsize(file_path)
        
        # If file is still way too large, just read the first part
        if file_size > 2 * max_chunk_size:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(max_chunk_size)
                    return content + f"\n\n... (truncated, {file_size/1024:.1f}KB total)"
            except Exception as e:
                logger.warning(f"Error reading large file {file_path}: {str(e)}")
                return None
        
        # Otherwise, try to read the whole file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def extract_file_content(self, file_path: str) -> Optional[str]:
        """
        Extract content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as a string or None if extraction failed
        """
        try:
            # Handle large files
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return self._chunk_large_file(file_path)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            return content
            
        except UnicodeDecodeError:
            logger.warning(f"UnicodeDecodeError for {file_path}, skipping")
            return None
        except Exception as e:
            logger.warning(f"Failed to extract content from {file_path}: {str(e)}")
            return None
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            # Get relative path from repository root
            rel_path = os.path.relpath(file_path, self.repo_path)
            
            # Get file stats
            stats = os.stat(file_path)
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            
            return {
                'path': rel_path,
                'size_bytes': stats.st_size,
                'extension': ext.lower(),
                'last_modified': stats.st_mtime
            }
            
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_path}: {str(e)}")
            return {
                'path': file_path,
                'error': str(e)
            }
    
    def upload_to_supabase(self, file_content: str, file_path: str) -> Optional[str]:
        """
        Upload file content to Supabase Storage.
        
        Args:
            file_content: Content of the file as string
            file_path: Path of the file (used for naming)
            
        Returns:
            Public URL of the uploaded content or None if upload failed
        """
        if not self.supabase:
            return None
            
        try:
            # Generate a unique filename based on path and timestamp
            safe_path = file_path.replace('/', '_').replace('\\', '_')
            timestamp = int(time.time())
            unique_filename = f"{safe_path}_{timestamp}.txt"
            
            # Upload file content to Supabase Storage
            result = self.supabase.storage.from_("file_contents").upload(
                unique_filename, 
                file_content.encode('utf-8'),
                {"content-type": "text/plain"}
            )
            
            # Get the public URL
            public_url = self.supabase.storage.from_("file_contents").get_public_url(unique_filename)
            
            logger.debug(f"Uploaded {file_path} to Supabase Storage")
            return public_url
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to Supabase: {e}")
            return None
    
    def _process_repository(self) -> Dict[str, Any]:
        """
        Process the repository and extract all relevant information.
        
        Returns:
            Dictionary containing repository information
        """
        logger.info(f"Processing repository: {self.repo_name}")
        
        # Get list of files to process
        files = self.list_repository_files()
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        processed_files = []
        file_types = Counter()
        total_size = 0
        
        for file_path in tqdm(files, desc="Processing files"):
            # Extract content
            content = self.extract_file_content(file_path)
            
            if content is None:
                continue
            
            # Get metadata
            metadata = self.get_file_metadata(file_path)
            
            # Upload content to Supabase if available
            content_url = None
            if self.supabase:
                content_url = self.upload_to_supabase(content, metadata['path'])
            
            # Update statistics
            ext = metadata.get('extension', '')
            file_types[ext] += 1
            total_size += metadata.get('size_bytes', 0)
            
            # Create file entry
            file_entry = {
                'metadata': metadata,
                'content': content[:10000],  # Truncate content for database storage
                'content_url': content_url,  # Add content URL for access to full content
                'lines': len(content.splitlines()),
                'characters': len(content)
            }
            
            processed_files.append(file_entry)
        
        # Create repository summary
        repo_summary = {
            'name': self.repo_name,
            'path': self.repo_path,
            'file_count': len(processed_files),
            'total_size_bytes': total_size,
            'file_types': dict(file_types),
            'processed_date': os.path.getmtime(self.output_dir),
            'files': processed_files
        }
        
        # Extract metadata and save to disk
        metadata = {
            'name': repo_summary['name'],
            'path': repo_summary['path'],
            'file_count': repo_summary['file_count'],
            'total_size_bytes': repo_summary['total_size_bytes'],
            'file_types': repo_summary['file_types'],
            'processed_date': repo_summary['processed_date']
        }
        
        # Save to disk
        self.save_repository_data(metadata, repo_summary, self.output_dir)
        
        return repo_summary
    
    def save_repository_data(self, repository_metadata: Dict[str, Any], repository_data: Dict[str, Any], output_dir: str) -> Optional[int]:
        """
        Save repository data to the output directory and database.
        
        Args:
            repository_metadata: Repository metadata (name, path, etc.)
            repository_data: Repository data (files, etc.)
            output_dir: Output directory
            
        Returns:
            Optional repository ID if stored in the database, None otherwise
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store in database if available
        repo_id = None
        if self.supabase:
            try:
                # Store repository metadata in database using enhanced_db
                logger.info(f"Storing repository metadata for {repository_metadata['name']} in database")
                try:
                    from core.enhanced_db import RepositoryDB
                    db = RepositoryDB()
                    
                    if db and db.is_available():
                        repo_id = db.store_repository(repository_metadata)
                        
                        if repo_id:
                            logger.info(f"Repository stored with ID: {repo_id}")
                            # Add repository ID to metadata and data
                            repository_metadata['id'] = repo_id
                            repository_data['id'] = repo_id
                            
                            # Store files with repository ID (would need to add this method to RepositoryDB if needed)
                            # for file_entry in repository_data.get('files', []):
                            #     if 'content' in file_entry:
                            #         logger.debug(f"Storing file {file_entry['metadata']['path']} in database")
                            #         try:
                            #             file_id = db.store_file(repo_id, file_entry)
                            #             if file_id:
                            #                 file_entry['id'] = file_id
                            #         except Exception as file_err:
                            #             logger.error(f"Error storing file {file_entry['metadata']['path']}: {str(file_err)}")

                            # Store all files with repository ID
                            try:
                                files = repository_data.get('files', [])
                                if files:
                                    logger.info(f"Storing {len(files)} files in database for repository {repo_id}")
                                    success = db.store_files(repo_id, files)
                                    if success:
                                        logger.info(f"Successfully stored files for repository {repo_id}")
                                    else:
                                        logger.warning(f"Failed to store files for repository {repo_id}")
                            except Exception as file_err:
                                logger.error(f"Error storing files: {str(file_err)}")
                    else:
                        logger.error("Database not available")
                except ImportError:
                    logger.error("Enhanced database module not available")
                except Exception as db_err:
                    logger.error(f"Error with database operations: {str(db_err)}")
            except Exception as e:
                logger.error(f"Error storing repository data in database: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save metadata to JSON file
        metadata_path = os.path.join(output_dir, 'repository_metadata.json')
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(repository_metadata, f, indent=2)
                logger.info(f"Repository metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving repository metadata: {str(e)}")
        
        # Save repository data to JSON file
        data_path = os.path.join(output_dir, 'repository_data.json')
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(repository_data, f, indent=2)
                logger.info(f"Repository data saved to {data_path}")
        except Exception as e:
            logger.error(f"Error saving repository data: {str(e)}")
        
        # Save repository ID to a dedicated file for easier retrieval
        if repo_id:
            id_path = os.path.join(output_dir, 'repository_id.txt')
            try:
                with open(id_path, 'w', encoding='utf-8') as f:
                    f.write(str(repo_id))
                    logger.info(f"Repository ID saved to {id_path}")
            except Exception as e:
                logger.error(f"Error saving repository ID: {str(e)}")
        
        # Return success message
        logger.info(f"Repository data saved to {output_dir} with ID: {repo_id if repo_id else 'None'}")
        return repo_id
    
    def get_repository_digest(self) -> str:
        """
        Generate a text digest of the repository.
        
        Returns:
            Text digest as a string
        """
        # Load repository data
        full_data_path = os.path.join(self.output_dir, 'repository_data.json')
        
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(f"Repository data not found. Run process_repository() first.")
        
        with open(full_data_path, 'r', encoding='utf-8') as f:
            repo_data = json.load(f)
        
        # Generate digest
        digest = []
        
        # Add repository information
        digest.append(f"# Repository: {repo_data['name']}")
        digest.append(f"Files: {repo_data['file_count']}")
        digest.append(f"Total size: {repo_data['total_size_bytes'] / 1024:.1f}KB")
        digest.append("\n## File types:")
        
        for ext, count in repo_data['file_types'].items():
            ext_name = ext if ext else 'no extension'
            digest.append(f"- {ext_name}: {count} files")
        
        # Add summary of each file
        digest.append("\n## Files:")
        
        for file_entry in sorted(repo_data['files'], key=lambda f: f['metadata']['path']):
            path = file_entry['metadata']['path']
            lines = file_entry['lines']
            digest.append(f"\n### {path}")
            digest.append(f"Lines: {lines}")
            
            # Add URL if available
            if file_entry.get('content_url'):
                digest.append(f"Content URL: {file_entry['content_url']}")
            
            # Add first few lines of content as preview
            content = file_entry['content']
            preview_lines = content.splitlines()[:10]
            if preview_lines:
                digest.append("\nPreview:")
                digest.append("```")
                digest.extend(preview_lines)
                if len(content.splitlines()) > 10:
                    digest.append("... (truncated)")
                digest.append("```")
        
        return "\n".join(digest)

    def process(self, apply_ast_analysis=True) -> Dict[str, Any]:
        """
        Process the repository to extract all relevant data.
        
        Args:
            apply_ast_analysis: Whether to apply AST analysis to extract code structures
            
        Returns:
            Repository data
        """
        logger.info(f"Processing repository: {self.repo_path}")
        
        # Process repository
        repository_data = self._process_repository()
        
        # Extract metadata 
        metadata = {
            'name': repository_data.get('name', 'unknown'),
            'path': repository_data.get('path', ''),
            'file_count': repository_data.get('file_count', 0),
            'total_size_bytes': repository_data.get('total_size_bytes', 0),
            'file_types': repository_data.get('file_types', {}),
            'processed_date': repository_data.get('processed_date', time.time()),
        }
        
        # Apply AST analysis if requested
        if apply_ast_analysis:
            try:
                from core.ast_analyzer import RepositoryASTAnalyzer
                
                # Create AST analyzer and process
                ast_analyzer = RepositoryASTAnalyzer(self.output_dir)
                ast_result = ast_analyzer.analyze()
                
                # Merge AST results if successful
                if ast_result and isinstance(ast_result, dict):
                    # Include AST results in repository data
                    repository_data['functions'] = ast_result.get('functions', [])
                    repository_data['classes'] = ast_result.get('classes', [])
                    repository_data['imports'] = ast_result.get('imports', [])
                    logger.info(f"AST analysis complete: {len(ast_result.get('functions', []))} functions, "
                            f"{len(ast_result.get('classes', []))} classes, "
                            f"{len(ast_result.get('imports', []))} imports")
                    
                    # Also update metadata with function, class, and import counts
                    metadata['function_count'] = len(ast_result.get('functions', []))
                    metadata['class_count'] = len(ast_result.get('classes', []))
                    metadata['import_count'] = len(ast_result.get('imports', []))
            except Exception as e:
                logger.error(f"Error during AST analysis: {str(e)}")
                logger.warning("The repository was still processed, but AST analysis was incomplete.")
        
        # Save repository data and get ID
        repo_id = self.save_repository_data(metadata, repository_data, self.output_dir)
        
        # Add repo_id to both objects if available
        if repo_id:
            metadata['id'] = repo_id
            repository_data['id'] = repo_id
        
        return repository_data


# Standalone function for simpler use cases
def ingest_repository(repo_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest a repository and extract its content and metadata.
    
    Args:
        repo_path: Path to the Git repository
        output_dir: Directory to store processed data
        
    Returns:
        Dictionary containing repository information
    """
    ingestor = RepositoryIngestor(repo_path, output_dir)
    return ingestor.process()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest a Git repository for PromptPilot")
    parser.add_argument("repo_path", help="Path to the Git repository")
    parser.add_argument("--output", help="Directory to store processed data")
    
    args = parser.parse_args()
    
    try:
        ingestor = RepositoryIngestor(args.repo_path, args.output)
        repo_data = ingestor.process()
        
        # Generate and print digest
        digest = ingestor.get_repository_digest()
        print("\nRepository Digest:")
        print("=" * 50)
        print(digest)
        
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise
