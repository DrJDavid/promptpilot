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

# git package is provided by the gitpython library
# If you're seeing an import error, run: pip install gitpython
import git
from tqdm import tqdm

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
            output_dir: Directory to store processed data (defaults to .promptpilot in repo_path)
        """
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = os.path.basename(self.repo_path)
        
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path '{self.repo_path}' does not exist")
        
        if not os.path.isdir(os.path.join(self.repo_path, '.git')):
            raise ValueError(f"Path '{self.repo_path}' is not a Git repository")
        
        # Set up output directory
        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = os.path.join(self.repo_path, '.promptpilot')
            
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Repository ingestor initialized for {self.repo_name}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize Git repository
        self.repo = git.Repo(self.repo_path)
        
    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on extension and ignore rules.
        Updated to handle large files with chunking.
        
        Args:
            file_path: Path to the file
                
        Returns:
            True if the file should be processed, False otherwise
        """
        # Skip ignored files/directories
        for ignore in IGNORE_FILES:
            if ignore in file_path.split(os.sep):
                return False
        
        # Check if it's a directory
        if os.path.isdir(os.path.join(self.repo_path, file_path)):
            return False
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        return ext.lower() in CODE_EXTENSIONS
    
    def list_repository_files(self) -> List[str]:
        """
        List all files in the repository that should be processed.
        
        Returns:
            List of relative file paths
        """
        all_files = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
                
            # Process each file
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.repo_path)
                
                if self.should_process_file(rel_path):
                    all_files.append(rel_path)
        
        return all_files
    
    def _chunk_large_file(self, file_path: str, max_chunk_size: int = 100 * 1024) -> Optional[str]:
        """
        Process a large file by reading it in chunks.
        
        Args:
            file_path: Path to the file
            max_chunk_size: Maximum chunk size in bytes
            
        Returns:
            File content as a string, or None if the file couldn't be read
        """
        full_path = os.path.join(self.repo_path, file_path)
        
        try:
            file_size = os.path.getsize(full_path)
            
            # For text files, read in chunks
            chunks = []
            total_bytes_read = 0
            max_chunks = 10  # Limit the number of chunks to prevent excessive memory usage
            
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                while total_bytes_read < file_size and len(chunks) < max_chunks:
                    chunk = f.read(max_chunk_size)
                    if not chunk:
                        break
                        
                    chunks.append(chunk)
                    total_bytes_read += len(chunk.encode('utf-8'))
                    
                    # If we've read enough, break
                    if total_bytes_read >= 1000000:  # 1MB limit
                        chunks.append("\n... [file truncated due to size] ...")
                        break
            
            # Combine chunks
            content = ''.join(chunks)
            logger.info(f"Processed large file {file_path} ({total_bytes_read / 1024:.1f}KB of {file_size / 1024:.1f}KB)")
            
            return content
            
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 encoding
                chunks = []
                with open(full_path, 'r', encoding='latin-1', errors='replace') as f:
                    while len(chunks) < max_chunks:
                        chunk = f.read(max_chunk_size)
                        if not chunk:
                            break
                        chunks.append(chunk)
                return ''.join(chunks)
            except Exception as e:
                logger.warning(f"Failed to read large file {file_path}: {str(e)}")
                return None
        except Exception as e:
            logger.warning(f"Failed to read large file {file_path}: {str(e)}")
            return None
    
    def extract_file_content(self, file_path: str) -> Optional[str]:
        """
        Extract the content of a file.
        
        Args:
            file_path: Relative path to the file within the repository
            
        Returns:
            File content as a string, or None if the file couldn't be read
        """
        full_path = os.path.join(self.repo_path, file_path)
        
        # Check if it's a large file
        try:
            if os.path.getsize(full_path) > MAX_FILE_SIZE:
                return self._chunk_large_file(file_path)
        except (FileNotFoundError, OSError):
            return None
        
        # Handle normal-sized files as before
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Try with Latin-1 encoding as fallback
                with open(full_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {str(e)}")
                return None
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {str(e)}")
            return None
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a file.
        
        Args:
            file_path: Relative path to the file within the repository
            
        Returns:
            Dictionary of metadata
        """
        full_path = os.path.join(self.repo_path, file_path)
        
        try:
            # Get file stats
            stats = os.stat(full_path)
            
            # Get file extension
            _, ext = os.path.splitext(file_path)
            
            # Get Git history
            file_history = list(self.repo.iter_commits(paths=file_path, max_count=5))
            
            return {
                'path': file_path,
                'size_bytes': stats.st_size,
                'last_modified': stats.st_mtime,
                'extension': ext.lstrip('.').lower() if ext else '',
                'num_commits': len(file_history),
                'last_commit_hash': str(file_history[0]) if file_history else None,
                'last_commit_date': file_history[0].committed_datetime.isoformat() if file_history else None,
                'last_commit_author': file_history[0].author.name if file_history else None,
            }
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_path}: {str(e)}")
            return {
                'path': file_path,
                'error': str(e)
            }
    
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
            
            # Update statistics
            ext = metadata.get('extension', '')
            file_types[ext] += 1
            total_size += metadata.get('size_bytes', 0)
            
            # Create file entry
            file_entry = {
                'metadata': metadata,
                'content': content,
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
                    'characters': file_entry['characters']
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
