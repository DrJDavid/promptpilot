import os
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.db')

# Load environment variables
load_dotenv()

# Custom exception classes
class DatabaseError(Exception):
    pass

class StorageError(Exception):
    pass

class RepositoryDB:
    """Class for enhanced database operations with Supabase and pgvector."""

    def __init__(self):
        """Initialize the enhanced database connection."""
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
            raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized for enhanced database operations")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    def is_available(self) -> bool:
        return self.supabase is not None

    def store_repository(self, repo_data: Dict[str, Any]) -> bool:
        """Store repository metadata in Supabase."""
        if not self.supabase:
            return False
        try:
            repo_id = repo_data.get('id', str(uuid.uuid4()))
            repo_record = {
                'id': repo_id,
                'name': repo_data.get('name', 'unnamed'),
                'path': repo_data.get('path', ''),
                'file_count': repo_data.get('file_count', 0),
                'total_size_bytes': repo_data.get('total_size_bytes', 0),
                'file_types': json.dumps(repo_data.get('file_types', {})),
                'processed_date': repo_data.get('processed_date', datetime.now(timezone.utc).isoformat())
            }
            result = self.supabase.table('repositories').upsert(
                repo_record,
                on_conflict='id'
            ).execute()

            if result.data:
              logger.info(f"Repository metadata stored in Supabase: {repo_id}")
              return True
            else:
              logger.error(f"Failed to upsert repository data: {result.data}")
              return False

        except Exception as e:
            logger.error(f"Error storing repository: {e}")
            raise DatabaseError("Failed to store repository metadata") from e

    def store_files(self, repo_id: str, files: List[Dict[str, Any]]) -> bool:
      """Store file data, including content URLs from Supabase Storage."""
      if not self.supabase:
        return False

      try:
          files_to_upsert = []
          for file_entry in files:
            file_id = str(uuid.uuid4())
            file_doc = {
                'id': file_id,
                'repository_id': repo_id,
                'path': file_entry['metadata']['path'],
                'size_bytes': file_entry['metadata'].get('size_bytes', 0),
                'extension': file_entry['metadata'].get('extension', ''),
                'last_modified': file_entry['metadata'].get('last_modified', 0),
                'lines': file_entry.get('lines', 0),
                'characters': file_entry.get('characters', 0),
                'content': file_entry.get('content', ''),  # Truncated content
                'content_url': file_entry.get('content_url'),
                'content_hash': file_entry.get('content_hash')
            }
            files_to_upsert.append(file_doc)

          # Bulk upsert files
          data, count = self.supabase.table('files').upsert(files_to_upsert, on_conflict='repository_id, path').execute()

          logger.info(f"Stored {count} files for repository {repo_id}")  # type: ignore
          return True
      except Exception as e:
            logger.error(f"Error storing file data: {e}")
            raise DatabaseError("Failed to store file data") from e
    
    def store_file_embeddings(self, file_embeddings: Dict[str, List[float]]) -> bool:
      if not self.is_available():
        logger.warning("Supabase client not available. Skipping embedding storage.")
        return False
      
      #Use the path to create the link between the file_embeddings and the files.
      to_upsert = []
      
      for file_path, embedding in file_embeddings.items():
        file_data = self.get_file_by_path(file_path) #Grab the file ID, given the path

        if file_data:
            file_id = file_data.get('id')  # Extract the ID
            
            to_upsert.append({
                  'id': str(uuid.uuid4()), #Create the unique id for the embedding
                  'file_id': file_id,   # Use retrieved file_id
                  "embedding": embedding
            })
        else:
           logger.warning(f"No file_id located for path: {file_path}, skipping.")

      try:
          # Use 'path' as on_conflict.  If a file with this path exists, update it.
          data, count = self.supabase.table('file_embeddings').upsert(to_upsert, on_conflict='id').execute() # type: ignore
          logger.info(f"Upserted {count} file embeddings.")
          return True
      except Exception as e:
          logger.error(f"Error storing embeddings: {e}")
          return False

    def get_file_by_path(self, file_path:str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a file by its path
        """
        if not self.supabase:
            return None

        try:
            data, count = self.supabase.table('files').select("*").eq('path', file_path).execute()

            if data and count: # type: ignore
                return data[1][0]  # type: ignore
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting repository by path: {str(e)}")
        return None

    def store_ast_data(self, repo_id: str, ast_data: Dict[str, Any]) -> bool:
        if not self.is_available():
            logger.warning("Database not available. AST data not stored.")
            return False

        try:
            # Use helper method for files
            file_map = {file['path']: file['id'] for file in self.get_repository_files(repo_id)}

            # Prepare and upsert functions
            functions_to_upsert = [{
                'id': str(uuid.uuid4()),
                'repository_id': repo_id,
                'file_id': file_map[func['path']],
                'name': func.get('name', ''),
                'signature': func.get('signature', ''),
                'docstring': func.get('docstring', ''),
                'body': func.get('body', ''),
                'start_line': func.get('start_line', 0),
                'end_line': func.get('end_line', 0),
            } for func in ast_data.get('functions', []) if func.get('path') in file_map]
            
            if functions_to_upsert:
                self.supabase.table('functions').upsert(functions_to_upsert, on_conflict='repository_id, file_id, name, start_line').execute()

            # Prepare and upsert classes
            classes_to_upsert = [{
                'id': str(uuid.uuid4()),
                'repository_id': repo_id,
                'file_id': file_map[cls['path']],
                'name': cls.get('name', ''),
                'docstring': cls.get('docstring', ''),
                'body': cls.get('body', ''),
                'start_line': cls.get('start_line', 0),
                'end_line': cls.get('end_line', 0),
            } for cls in ast_data.get('classes', []) if cls.get('path') in file_map]
            if classes_to_upsert:
                self.supabase.table('classes').upsert(classes_to_upsert, on_conflict='repository_id, file_id, name, start_line').execute()

            # Prepare and upsert imports
            imports_to_upsert = [{
                'id': str(uuid.uuid4()),
                'repository_id': repo_id,
                'file_id': file_map[imp['path']],
                'statement': imp.get('statement', ''),
                'line': imp.get('line', 0),
            } for imp in ast_data.get('imports', []) if imp.get('path') in file_map]
            
            if imports_to_upsert:
                self.supabase.table('imports').upsert(imports_to_upsert, on_conflict='repository_id, file_id, statement, line').execute()

            logger.info(f"Stored AST data for repository {repo_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing AST data: {str(e)}")
            return False
    
    def find_relevant_files(self, repo_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.supabase:
            return []
        try:
            # Call the RPC.  Note the syntax.
            # Pass repo_id to the function.
            response = self.supabase.rpc('match_code_files', {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,  # Example threshold, adjust as needed
                'match_count': top_k,
                'repo_id': repo_id # Pass the repo_id
            }).execute()
            
            # .execute() gives (data, count).  We just want data.
            return response.data # type: ignore

        except Exception as e:
            logger.error(f"Error in find_relevant_files: {e}")
            return []
    
    #Added to help with testing
    def get_file_content_by_path(self, repo_id: int, file_path: str) -> Optional[str]:
        """Fetches the content_url of a file given its path and repo_id."""
        
        if not self.supabase:
            logger.warning("Supabase client not available.")
            return None

        try:
            data, count = self.supabase.table('files').select('content_url').match({'repository_id': repo_id, 'path': file_path}).execute()
            if data and count:  # type: ignore
                # Assuming the query returns a list of dicts, and we want the first match
                content_url = data[1][0]['content_url'] # type: ignore
                return content_url
            else:
                logger.warning(f"No file found with path '{file_path}' in repository {repo_id}.")
                return None
        except Exception as e:
            logger.error(f"Error fetching file content URL: {e}")
            return None
    
    def get_repositories(self) -> List[Dict[str, Any]]:
        """
        Get list of repositories in the database.

        Returns:
            List of repository documents
        """
        if not self.is_available():
            logger.warning("Database not available. Returning empty repository list.")
            return []

        try:
            data, count = self.supabase.table('repositories').select("*").execute()
            # data is a tuple: (rows, count).  The rows is a list of dicts.
            return data[1] #type: ignore

        except Exception as e:
            logger.error(f"Error getting repositories: {str(e)}")
            return []
    
    #Added to help with testing
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Retrieves all files from the database."""

        if not self.supabase:
            logger.warning("Supabase client not available. Returning empty file list.")
            return []
        
        try:
            data, count = self.supabase.table('files').select('*').execute()
            return data[1] # type: ignore
        
        except Exception as e:
            logger.error(f"Error retrieving files: {e}")
            return []
    
    def get_repository_files(self, repo_id: str) -> List[Dict[str, Any]]:
        """
        Get files for a repository.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            List of file documents
        """
        if not self.is_available():
            logger.warning("Database not available. Returning empty file list.")
            return []

        try:
        data, count = self.supabase.table('files').select("*").eq('repository_id', repo_id).execute()
        return data[1] # type: ignore

        except Exception as e:
            logger.error(f"Error getting repository files: {str(e)}")
            return []
    
    def get_file_functions(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Get functions for a file.
        
        Args:
            file_id: File ID
            
        Returns:
            List of function documents
        """
        if not self.is_available():
            logger.warning("Database not available. Returning empty function list.")
            return []

        try:
            data, count = self.supabase.table('functions').select("*").eq('file_id', file_id).execute()
            return data[1] # type: ignore

        except Exception as e:
            logger.error(f"Error getting file functions: {str(e)}")
            return []
    
    def get_file_classes(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Get classes for a file.
        
        Args:
            file_id: File ID
            
        Returns:
            List of class documents
        """
        if not self.is_available():
            logger.warning("Database not available. Returning empty class list.")
            return []

        try:
            data, count = self.supabase.table('classes').select("*").eq('file_id', file_id).execute()
            return data[1] # type: ignore

        except Exception as e:
            logger.error(f"Error getting file classes: {str(e)}")
            return []

# Singleton instance for convenience
_db_instance = None

def get_repository_db(self):
    """
    Get or create the EnhancedDatabase singleton instance.

    Returns:
        EnhancedDatabase instance
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = RepositoryDB()

    return _db_instance