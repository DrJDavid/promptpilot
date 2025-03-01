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
    
    def process_repository(self) -> Dict[str, Any]:
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
        
        # Save to disk
        self._save_repository_data(repo_summary)
        
        return repo_summary
    
    def _save_repository_data(self, repo_data: Dict[str, Any]) -> None:
        """
        Save repository data to disk.
        
        Args:
            repo_data: Repository data dictionary
        """
        # Save metadata separately from content to make it easier to read
        metadata_only = {
            'name': repo_data['name'],
            'path': repo_data['path'],
            'file_count': repo_data['file_count'],
            'total_size_bytes': repo_data['total_size_bytes'],
            'file_types': repo_data['file_types'],
            'processed_date': repo_data['processed_date'],
            'files': [
                {
                    'metadata': file_entry['metadata'],
                    'lines': file_entry['lines'],
                    'characters': file_entry['characters'],
                    'content_url': file_entry.get('content_url')
                }
                for file_entry in repo_data['files']
            ]
        }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'repository_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_only, f, indent=2)
        
        logger.info(f"Repository metadata saved to {metadata_path}")
        
        # Save full data (including content)
        full_data_path = os.path.join(self.output_dir, 'repository_data.json')
        with open(full_data_path, 'w', encoding='utf-8') as f:
            json.dump(repo_data, f)
        
        logger.info(f"Full repository data saved to {full_data_path}")
    
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
    return ingestor.process_repository()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest a Git repository for PromptPilot")
    parser.add_argument("repo_path", help="Path to the Git repository")
    parser.add_argument("--output", help="Directory to store processed data")
    
    args = parser.parse_args()
    
    try:
        ingestor = RepositoryIngestor(args.repo_path, args.output)
        repo_data = ingestor.process_repository()
        
        # Generate and print digest
        digest = ingestor.get_repository_digest()
        print("\nRepository Digest:")
        print("=" * 50)
        print(digest)
        
    except Exception as e:
        logger.error(f"Error processing repository: {str(e)}")
        raise
