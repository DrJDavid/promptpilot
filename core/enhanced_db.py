import json
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

    def store_repository(self, repo_data: Dict[str, Any]) -> int:
        """
        Store repository metadata in Supabase.
        
        Args:
            repo_data: Dictionary containing repository metadata
            
        Returns:
            Integer ID of the stored repository
        """
        if not self.supabase:
            logger.error("Supabase client not available")
            return 0
            
        try:
            # Generate or use a numeric ID for the repository
            repo_id = None
            
            # First, check if this repository already exists in the database
            repo_name = repo_data.get('name', 'unnamed')
            repo_path = repo_data.get('path', '')
            
            # Try to find by name and path first
            try:
                data, count = self.supabase.table('repositories').select("id").eq('name', repo_name).eq('path', repo_path).execute()
                if data and len(data) > 1 and len(data[1]) > 0:
                    repo_id = data[1][0]['id']
                    logger.info(f"Found existing repository ID: {repo_id}")
            except Exception as find_err:
                logger.warning(f"Error finding repository: {find_err}")
            
            # If not found, generate a new ID
            if not repo_id:
                # Create a deterministic integer from the name and path
                seed_str = f"{repo_name}:{repo_path}"
                repo_id = abs(hash(seed_str)) % (2**31)  # Ensure it fits in postgres integer
                logger.info(f"Generated numeric ID {repo_id} for repository {repo_name}")
                
            # Format timestamp properly for PostgreSQL
            if 'processed_date' in repo_data:
                # Check if it's a float timestamp
                processed_date = repo_data.get('processed_date')
                if isinstance(processed_date, (float, int)):
                    # Convert from Unix timestamp to ISO format
                    processed_date = datetime.fromtimestamp(processed_date, timezone.utc).isoformat()
                # Ensure it's a proper ISO format string
                elif isinstance(processed_date, str):
                    try:
                        # Try to parse and reformat to ensure valid ISO format
                        dt = datetime.fromisoformat(processed_date.replace('Z', '+00:00'))
                        processed_date = dt.isoformat()
                    except ValueError:
                        # If parsing fails, use current time
                        processed_date = datetime.now(timezone.utc).isoformat()
            else:
                processed_date = datetime.now(timezone.utc).isoformat()
                
            # Create the repository record
            repo_record = {
                'id': repo_id,
                'name': repo_name,
                'path': repo_path,
                'file_count': repo_data.get('file_count', 0),
                'total_size_bytes': repo_data.get('total_size_bytes', 0),
                'file_types': json.dumps(repo_data.get('file_types', {})),
                'processed_date': processed_date
            }
            
            # Upsert the repository record
            result = self.supabase.table('repositories').upsert(
                repo_record,
                on_conflict='id'
            ).execute()

            if result.data:
                logger.info(f"Repository metadata stored in Supabase with ID: {repo_id}")
                # Return the numeric ID for use in other functions
                return repo_id
            else:
                logger.error(f"Failed to upsert repository data: {result.data}")
                return 0

        except Exception as e:
            logger.error(f"Error storing repository: {e}")
            raise DatabaseError(f"Failed to store repository metadata: {e}") from e

    def store_files(self, repo_id: int, files: List[Dict[str, Any]]) -> bool:
        """
        Store file data, including content URLs from Supabase Storage.
        
        Args:
            repo_id: Integer ID of the repository
            files: List of file data dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self.supabase:
            logger.warning("Supabase client not available. Skipping file storage.")
            return False

        try:
            if not files:
                logger.warning(f"No files provided for repository {repo_id}")
                return True  # Nothing to store, but not an error

            files_to_upsert = []
            for file_entry in files:
                # Extract metadata, handle different structures
                metadata = file_entry.get('metadata', file_entry)
                
                # Format the last_modified timestamp properly for PostgreSQL
                last_modified = metadata.get('last_modified', datetime.now(timezone.utc).timestamp())
                
                # Convert timestamp to ISO format if it's a float/int (Unix timestamp)
                if isinstance(last_modified, (float, int)):
                    last_modified = datetime.fromtimestamp(last_modified, timezone.utc).isoformat()
                # Ensure it's a proper ISO format string if it's already a string
                elif isinstance(last_modified, str):
                    try:
                        # Try to parse and reformat to ensure valid ISO format
                        dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                        last_modified = dt.isoformat()
                    except ValueError:
                        # If parsing fails, use current time
                        last_modified = datetime.now(timezone.utc).isoformat()
                
                # Build the file record - don't specify ID, let the database generate it
                file_doc = {
                    'repository_id': repo_id,
                    'path': metadata.get('path', ''),
                    'size_bytes': metadata.get('size_bytes', 0),
                    'characters': metadata.get('characters', 0),
                    'lines': metadata.get('lines', 0),
                    'extension': metadata.get('extension', ''),
                    'last_modified': last_modified,
                    'content': file_entry.get('content', ''),
                    'content_hash': file_entry.get('content_hash', ''),
                    'content_url': file_entry.get('content_url', '')
                }
                
                # Only add if we have a valid path
                if file_doc['path']:
                    files_to_upsert.append(file_doc)
                else:
                    logger.warning(f"Skipping file without path: {file_entry}")

            # Bulk upsert in batches to avoid hitting request size limits
            batch_size = 100
            total_files = len(files_to_upsert)
            total_stored = 0
            
            for i in range(0, total_files, batch_size):
                batch = files_to_upsert[i:i+batch_size]
                
                # Select columns to update to avoid API errors
                data = self.supabase.table('files').upsert(
                    batch,
                    on_conflict='repository_id, path',
                    columns='"content","content_hash","size_bytes","characters","lines","content_url","path","last_modified","extension","repository_id"'
                ).execute()
                
                if data:
                    logger.info(f"Stored batch of {len(batch)} files for repository {repo_id} ({i+1}-{min(i+batch_size, total_files)} of {total_files})")
                    total_stored += len(batch)
                else:
                    logger.warning(f"Failed to store batch of {len(batch)} files")
            
            logger.info(f"Successfully stored {total_stored} of {total_files} files for repository {repo_id}")
            return total_stored > 0
            
        except Exception as e:
            logger.error(f"Error storing file data: {e}")
            return False

    def store_file_embeddings(self, repository_id: int, file_embeddings: Dict[str, List[float]]) -> bool:
        """
        Store file embeddings for a repository.
        
        Args:
            repository_id: Repository ID
            file_embeddings: Dictionary mapping file paths to embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Supabase client not available. Skipping embedding storage.")
            return False
        
        if not file_embeddings:
            logger.warning("No embeddings provided to store")
            return False
            
        successful_updates = 0
        
        # First, get all files for this repository to avoid repeated queries
        try:
            logger.info(f"Fetching files for repository {repository_id}")
            data = self.supabase.table('files').select("id,path").eq('repository_id', repository_id).execute()
            
            if not data or not hasattr(data, 'data') or len(data.data) == 0:
                logger.warning(f"No files found for repository {repository_id}")
                return False
            
            repository_files = data.data
            logger.info(f"Found {len(repository_files)} files in repository {repository_id}")
            
            # Create a mapping of file paths to file IDs for easier lookup
            file_path_map = {}
            for file in repository_files:
                path = file.get('path', '')
                
                # Handle different path formats
                normalized_path = path.replace('\\', '/')
                file_path_map[normalized_path] = file.get('id')
                
                # Also add the original path for Windows paths
                if '\\' in path:
                    file_path_map[path] = file.get('id')
            
            # Process each file embedding
            total_embeddings = len(file_embeddings)
            logger.info(f"Processing {total_embeddings} embeddings")
            
            for path, embedding in file_embeddings.items():
                # Normalize path
                normalized_path = path.replace('\\', '/')
                
                # Find matching file ID
                file_id = file_path_map.get(normalized_path)
                if not file_id:
                    logger.warning(f"No file ID found for path: {normalized_path}")
                    continue
                
                # Ensure embedding is a list of floats
                if not isinstance(embedding, list):
                    try:
                        # Try to convert to list if it's not already
                        embedding = list(embedding)
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid embedding format for file {file_id}: {type(embedding)}")
                        continue
                
                # Validate embedding
                if not embedding or not all(isinstance(x, (int, float)) for x in embedding):
                    logger.warning(f"Invalid embedding values for file {file_id}")
                    continue
                
                try:
                    # Update the file with embedding
                    logger.info(f"Updating embedding for file ID {file_id} ({normalized_path}), embedding length: {len(embedding)}")
                    result = self.supabase.table('files').update({
                        'embedding': embedding
                    }).eq('id', file_id).execute()
                    
                    if result and hasattr(result, 'data') and len(result.data) > 0:
                        successful_updates += 1
                        logger.info(f"Successfully updated embedding for file ID {file_id}")
                    else:
                        logger.warning(f"Failed to update embedding for file ID {file_id}")
                        
                except Exception as update_err:
                    logger.error(f"Error updating embedding for file {file_id}: {str(update_err)}")
            
            logger.info(f"Successfully stored embeddings for {successful_updates} of {total_embeddings} files")
            return successful_updates > 0
            
        except Exception as e:
            logger.error(f"Error storing file embeddings: {str(e)}", exc_info=True)
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
            # Ensure repo_id is an integer
            try:
                repo_id_int = int(repo_id)
                logger.info(f"Converted repository ID to integer: {repo_id_int}")
            except (ValueError, TypeError):
                logger.error(f"Repository ID {repo_id} is not a valid integer")
                return False
            
            # Check if tables exist and have the expected schema
            try:
                # Inspect the functions table
                logger.info("Checking functions table structure...")
                functions_info = self.supabase.table('functions').select('*').limit(1).execute()
                logger.info(f"Functions table exists with structure: {functions_info}")
            except Exception as schema_err:
                logger.error(f"Error checking table structure: {schema_err}")
                # Continue anyway, as the error might be just because the table is empty
            
            # Use helper method for files
            file_list = self.get_repository_files(repo_id_int)
            if not file_list:
                logger.error(f"No files found for repository {repo_id_int}. Cannot store AST data.")
                return False
                
            file_map = {file['path']: file['id'] for file in file_list}
            logger.info(f"Found {len(file_map)} files for repository {repo_id_int}")
            
            # First, delete existing AST data for this repository to avoid conflicts
            try:
                logger.info(f"Removing existing AST data for repository {repo_id_int}")
                self.supabase.table('functions').delete().eq('repository_id', repo_id_int).execute()
                self.supabase.table('classes').delete().eq('repository_id', repo_id_int).execute()
                self.supabase.table('imports').delete().eq('repository_id', repo_id_int).execute()
                logger.info(f"Successfully cleared existing AST data for repository {repo_id_int}")
            except Exception as clear_err:
                logger.warning(f"Failed to clear existing AST data: {clear_err}")
            
            # Prepare functions for insert
            functions_to_insert = []
            for func in ast_data.get('functions', []):
                path = func.get('path', '')
                if path in file_map:
                    # Generate a sequential numeric ID instead of UUID
                    # This is to work around potential type constraints in the database
                    func_id = len(functions_to_insert) + 1
                    
                    functions_to_insert.append({
                        'id': func_id,
                        'repository_id': repo_id_int,
                        'file_id': file_map[path],
                        'name': func.get('name', ''),
                        'signature': func.get('signature', ''),
                        'docstring': func.get('docstring', ''),
                        'body': func.get('body', ''),
                        'start_line': func.get('start_line', 0),
                        'end_line': func.get('end_line', 0),
                    })
                else:
                    logger.warning(f"No file ID found for path {path} - skipping function {func.get('name', 'unknown')}")
            
            logger.info(f"Prepared {len(functions_to_insert)} functions for insert")
            
            # Insert functions in small batches with detailed error catching
            if functions_to_insert:
                batch_size = 10  # Smaller batches for better error isolation
                for i in range(0, len(functions_to_insert), batch_size):
                    batch = functions_to_insert[i:i+batch_size]
                    try:
                        # Log the first item for debugging
                        if batch:
                            logger.info(f"Sample function record: {batch[0]}")
                            
                        result = self.supabase.table('functions').insert(batch).execute()
                        logger.info(f"Stored batch of {len(batch)} functions ({i+1}-{min(i+batch_size, len(functions_to_insert))} of {len(functions_to_insert)})")
                    except Exception as batch_err:
                        logger.error(f"Error storing batch of functions: {batch_err}")
                        # Continue with the next batch
                        continue
            
            # Prepare classes for insert using numeric IDs
            classes_to_insert = []
            for cls in ast_data.get('classes', []):
                path = cls.get('path', '')
                if path in file_map:
                    # Generate a sequential numeric ID
                    cls_id = len(classes_to_insert) + 1
                    
                    classes_to_insert.append({
                        'id': cls_id,
                        'repository_id': repo_id_int,
                        'file_id': file_map[path],
                        'name': cls.get('name', ''),
                        'docstring': cls.get('docstring', ''),
                        'body': cls.get('body', ''),
                        'start_line': cls.get('start_line', 0),
                        'end_line': cls.get('end_line', 0),
                    })
                else:
                    logger.warning(f"No file ID found for path {path} - skipping class {cls.get('name', 'unknown')}")
            
            logger.info(f"Prepared {len(classes_to_insert)} classes for insert")
            
            # Insert classes in batches
            if classes_to_insert:
                batch_size = 10  # Smaller batches
                for i in range(0, len(classes_to_insert), batch_size):
                    batch = classes_to_insert[i:i+batch_size]
                    try:
                        if batch:
                            logger.info(f"Sample class record: {batch[0]}")
                            
                        result = self.supabase.table('classes').insert(batch).execute()
                        logger.info(f"Stored batch of {len(batch)} classes ({i+1}-{min(i+batch_size, len(classes_to_insert))} of {len(classes_to_insert)})")
                    except Exception as batch_err:
                        logger.error(f"Error storing batch of classes: {batch_err}")
                        continue
            
            # Prepare imports for insert using numeric IDs
            imports_to_insert = []
            for imp in ast_data.get('imports', []):
                path = imp.get('path', '')
                if path in file_map:
                    # Generate a sequential numeric ID
                    imp_id = len(imports_to_insert) + 1
                    
                    imports_to_insert.append({
                        'id': imp_id,
                        'repository_id': repo_id_int,
                        'file_id': file_map[path],
                        'statement': imp.get('statement', ''),
                        'line': imp.get('line', 0),
                    })
                else:
                    logger.warning(f"No file ID found for path {path} - skipping import statement")
            
            logger.info(f"Prepared {len(imports_to_insert)} imports for insert")
            
            # Insert imports in batches
            if imports_to_insert:
                batch_size = 20  # Slightly larger batches for simpler records
                for i in range(0, len(imports_to_insert), batch_size):
                    batch = imports_to_insert[i:i+batch_size]
                    try:
                        if batch:
                            logger.info(f"Sample import record: {batch[0]}")
                            
                        result = self.supabase.table('imports').insert(batch).execute()
                        logger.info(f"Stored batch of {len(batch)} imports ({i+1}-{min(i+batch_size, len(imports_to_insert))} of {len(imports_to_insert)})")
                    except Exception as batch_err:
                        logger.error(f"Error storing batch of imports: {batch_err}")
                        continue

            stats = {
                'functions_prepared': len(functions_to_insert),
                'classes_prepared': len(classes_to_insert),
                'imports_prepared': len(imports_to_insert)
            }
            logger.info(f"AST data processing summary: {stats}")
            
            # Even if some batches failed, consider it success if we stored anything
            if len(functions_to_insert) > 0 or len(classes_to_insert) > 0 or len(imports_to_insert) > 0:
                logger.info(f"Successfully stored at least some AST data for repository {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing AST data: {str(e)}", exc_info=True)
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
    
    def get_file_content_by_path(self, repo_id: int, file_path: str) -> Optional[str]:
        """
        Fetches the content of a file given its path and repo_id.
        
        Args:
            repo_id: The repository ID
            file_path: The path of the file
            
        Returns:
            The file content as a string, or None if not found/error
        """
        
        if not self.supabase:
            logger.warning("Supabase client not available.")
            return None

        try:
            # First get the file record to get the content_url or ID
            data, count = self.supabase.table('files').select('*').match({'repository_id': repo_id, 'path': file_path}).execute()
            
            if not data or not count or not data[1] or len(data[1]) == 0:  # type: ignore
                logger.warning(f"No file found with path '{file_path}' in repository {repo_id}.")
                return None
            
            file_record = data[1][0]  # type: ignore
            file_id = file_record.get('id')
            content_url = file_record.get('content_url')
            content = ""
            
            if content_url:
                # Content URL is stored in the database - try to download the actual content
                try:
                    # Check if the content_url is a URL to storage 
                    if "storage" in content_url and "/object/public/" in content_url:
                        # Extract the bucket and filename from the URL
                        url_parts = content_url.split('/object/public/')
                        if len(url_parts) > 1:
                            bucket_and_file = url_parts[1].split('/', 1)
                            if len(bucket_and_file) > 1:
                                bucket_name = bucket_and_file[0]
                                file_name = bucket_and_file[1].split('?')[0]  # Remove query params
                                
                                # Download from storage
                                logger.info(f"Downloading file {file_name} from bucket {bucket_name}")
                                resp = self.supabase.storage.from_(bucket_name).download(file_name)
                                if resp:
                                    content = resp.decode('utf-8')
                                    logger.info(f"Successfully downloaded file content, size: {len(content)} chars")
                                else:
                                    logger.warning(f"Empty response when downloading {file_name}")
                                    # Fallback to content_url as content
                                    content = content_url
                            else:
                                # Fallback to content_url as content
                                content = content_url
                        else:
                            # Fallback to content_url as content
                            content = content_url
                    else:
                        # If not a storage URL, use the content_url field as the actual content
                        content = content_url
                except Exception as download_err:
                    logger.warning(f"Error downloading file content from URL: {str(download_err)}")
                    # Fallback to content_url as content
                    content = content_url
            elif file_id:
                # Try to get content from storage using file_id
                bucket_name = 'file_contents'
                object_name = f"{file_id}.txt"
                
                try:
                    # Attempt to get from storage bucket
                    resp = self.supabase.storage.from_(bucket_name).download(object_name)
                    if resp:
                        content = resp.decode('utf-8')
                        logger.info(f"Downloaded content from storage using file_id, size: {len(content)} chars")
                except Exception as storage_err:
                    logger.warning(f"Error retrieving file from storage: {str(storage_err)}")
            
            return content
        except Exception as e:
            logger.error(f"Error fetching file content: {e}")
            return None
    
    def get_file_content(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the content of a file by its ID.
        
        Args:
            file_id: The file ID
            
        Returns:
            Dictionary containing file content and metadata, or None if not found/error
        """
        if not self.supabase:
            logger.warning("Supabase client not available.")
            return None
        
        try:
            # First get the file record to get the content_url
            data, count = self.supabase.table('files').select('*').eq('id', file_id).execute()
            
            if not data or not count or not data[1] or len(data[1]) == 0:  # type: ignore
                logger.warning(f"No file found with ID {file_id}")
                return None
            
            file_record = data[1][0]  # type: ignore
            content_url = file_record.get('content_url')
            content = ""
            
            if content_url:
                # Content URL is stored in the database - try to download the actual content
                try:
                    # Check if the content_url is a URL to storage 
                    if "storage" in content_url and "/object/public/" in content_url:
                        # Extract the bucket and filename from the URL
                        url_parts = content_url.split('/object/public/')
                        if len(url_parts) > 1:
                            bucket_and_file = url_parts[1].split('/', 1)
                            if len(bucket_and_file) > 1:
                                bucket_name = bucket_and_file[0]
                                file_name = bucket_and_file[1].split('?')[0]  # Remove query params
                                
                                # Download from storage
                                logger.info(f"Downloading file {file_name} from bucket {bucket_name}")
                                resp = self.supabase.storage.from_(bucket_name).download(file_name)
                                if resp:
                                    content = resp.decode('utf-8')
                                    logger.info(f"Successfully downloaded file content, size: {len(content)} chars")
                                else:
                                    logger.warning(f"Empty response when downloading {file_name}")
                            else:
                                # Try using content_url directly as content as a fallback
                                content = content_url
                        else:
                            # Try using content_url directly as content as a fallback
                            content = content_url
                    else:
                        # If not a storage URL, use the content_url field as the actual content
                        content = content_url
                except Exception as download_err:
                    logger.warning(f"Error downloading file content from URL: {str(download_err)}")
                    # Use content_url as fallback
                    content = content_url
            else:
                # File content might be in storage bucket using file_id as name
                bucket_name = 'file_contents'
                object_name = f"{file_id}.txt"
                
                try:
                    # Attempt to get from storage bucket
                    resp = self.supabase.storage.from_(bucket_name).download(object_name)
                    if resp:
                        content = resp.decode('utf-8')
                        logger.info(f"Downloaded content from storage using file_id, size: {len(content)} chars")
                except Exception as storage_err:
                    logger.warning(f"Error retrieving file from storage: {str(storage_err)}")
                
            # Prepare result with content and metadata
            result = {
                "id": file_id,
                "path": file_record.get('path', 'Unknown'),
                "repository_id": file_record.get('repository_id'),
                "content": content,
                "metadata": {
                    "size": file_record.get('size', 0),
                    "extension": file_record.get('extension', ''),
                    "mime_type": file_record.get('mime_type', ''),
                    "path": file_record.get('path', 'Unknown')
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting file content: {str(e)}")
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

    def store_repository_archive(self, repo_id: int, repo_path: str, archive_name: str = None) -> Optional[str]:
        """
        Store the entire repository as a zip archive in Supabase storage.
        
        Args:
            repo_id: Integer ID of the repository
            repo_path: Path to the repository on disk
            archive_name: Optional name for the archive file
            
        Returns:
            URL to the stored archive file, or None if failed
        """
        if not self.is_available():
            logger.warning("Supabase client not available. Cannot store repository archive.")
            return None
            
        try:
            import tempfile
            import shutil
            import zipfile
            import os
            from pathlib import Path
            
            # Create a safe archive name if not provided
            if not archive_name:
                # Get the repository name from the path
                repo_name = os.path.basename(os.path.normpath(repo_path))
                timestamp = int(time.time())
                archive_name = f"{repo_name}_{timestamp}.zip"
            
            logger.info(f"Creating archive of repository {repo_path} as {archive_name}")
            
            # Create a temporary zip file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                temp_path = temp_file.name
            
            # Create a zip archive of the repository
            with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the repository and add files
                repo_abs_path = os.path.abspath(repo_path)
                for root, dirs, files in os.walk(repo_path):
                    # Skip .git directory and other hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for file in files:
                        # Skip hidden files
                        if file.startswith('.'):
                            continue
                            
                        file_path = os.path.join(root, file)
                        
                        # Skip the temp zip file itself if it's in the repo path
                        if os.path.samefile(file_path, temp_path):
                            continue
                            
                        # Make path relative to the repository root
                        rel_path = os.path.relpath(file_path, repo_abs_path)
                        
                        # Add file to zip
                        zipf.write(file_path, rel_path)
            
            # Upload the zip file to Supabase Storage
            bucket_name = "repository_archives"
            try:
                # Check if bucket exists, create if not
                buckets = self.supabase.storage.list_buckets()
                if not any(bucket.name == bucket_name for bucket in buckets):
                    self.supabase.storage.create_bucket(bucket_name)
                    logger.info(f"Created new storage bucket: {bucket_name}")
            except Exception as bucket_err:
                logger.warning(f"Error checking/creating bucket: {bucket_err}")
            
            # Upload the file
            with open(temp_path, 'rb') as f:
                response = self.supabase.storage.from_(bucket_name).upload(
                    archive_name,
                    f,
                    file_options={"content-type": "application/zip"}
                )
            
            # Get the URL for the uploaded file
            file_url = self.supabase.storage.from_(bucket_name).get_public_url(archive_name)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Update the repository record with the archive URL
            self.supabase.table('repositories').update(
                {"archive_url": file_url}
            ).eq('id', repo_id).execute()
            
            logger.info(f"Repository archive stored at {file_url}")
            return file_url
            
        except Exception as e:
            logger.error(f"Error storing repository archive: {str(e)}", exc_info=True)
            return None

    def store_repository_digest(self, repo_id: int, digest_text: str) -> bool:
        """
        Store a text digest of the repository in the database.
        
        Args:
            repo_id: Integer ID of the repository
            digest_text: Text digest of the repository
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Supabase client not available. Cannot store repository digest.")
            return False
        
        try:
            # Update the repository record with the digest
            result = self.supabase.table('repositories').update(
                {"digest": digest_text}
            ).eq('id', repo_id).execute()
            
            if result and hasattr(result, 'data') and result.data:
                logger.info(f"Repository digest stored for repository {repo_id}")
                return True
            else:
                logger.warning(f"Failed to store repository digest for repository {repo_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error storing repository digest: {str(e)}", exc_info=True)
            return False

    def generate_repository_digest_with_gitingest(self, repo_id: int, repo_path: str) -> Optional[str]:
        """
        Generate a repository digest using the gitingest command and store it.
        
        Args:
            repo_id: Integer ID of the repository
            repo_path: Path to the repository on disk
            
        Returns:
            Path to the digest file, or None if failed
        """
        import subprocess
        import os
        from pathlib import Path
        
        if not self.is_available():
            logger.warning("Supabase client not available. Cannot store repository digest.")
            return None
        
        try:
            logger.info(f"Generating repository digest using gitingest for {repo_path}")
            
            # Run gitingest command
            # This will create a digest.txt file in the current directory
            current_dir = os.getcwd()
            result = subprocess.run(
                ["gitingest", repo_path], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Check if digest.txt was created
            digest_path = os.path.join(current_dir, "digest.txt")
            if not os.path.exists(digest_path):
                logger.warning(f"Digest file not created at {digest_path}")
                return None
            
            # Upload digest to storage bucket instead of storing in the database directly
            try:
                # Create a unique filename
                repo_name = os.path.basename(os.path.normpath(repo_path))
                timestamp = int(time.time())
                digest_filename = f"{repo_name}_digest_{timestamp}.txt"
                
                # Define storage bucket name
                bucket_name = "repository_digests"
                
                # Ensure bucket exists
                try:
                    buckets = self.supabase.storage.list_buckets()
                    if not any(bucket.name == bucket_name for bucket in buckets):
                        self.supabase.storage.create_bucket(bucket_name)
                        logger.info(f"Created new storage bucket: {bucket_name}")
                except Exception as bucket_err:
                    logger.warning(f"Error checking/creating bucket: {bucket_err}")
                    
                # Upload the digest file
                with open(digest_path, 'rb') as f:
                    self.supabase.storage.from_(bucket_name).upload(
                        digest_filename,
                        f,
                        file_options={"content-type": "text/plain"}
                    )
                    
                # Get the public URL
                digest_url = self.supabase.storage.from_(bucket_name).get_public_url(digest_filename)
                
                # Update the repository record with the digest URL
                result = self.supabase.table('repositories').update(
                    {"digest_url": digest_url}
                ).eq('id', repo_id).execute()
                
                if result and hasattr(result, 'data') and result.data:
                    logger.info(f"Repository digest stored in bucket and URL saved for repository {repo_id}")
                else:
                    logger.warning(f"Failed to update repository with digest URL for repository {repo_id}")
                    logger.warning("The 'digest_url' column needs to be added to the 'repositories' table. Run the SQL migrations.")
                    
                return digest_path
                
            except Exception as upload_err:
                logger.error(f"Error uploading digest to storage: {upload_err}")
                # Continue anyway since we have the local file
                return digest_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running gitingest: {e.stdout} {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error generating repository digest: {str(e)}", exc_info=True)
            return None

    def get_repository_digest_url(self, repo_id: int) -> Optional[str]:
        """
        Get the URL for the repository digest from the database.
        
        Args:
            repo_id: Integer ID of the repository
            
        Returns:
            URL to the digest file or None if not found
        """
        if not self.is_available():
            logger.warning("Supabase client not available.")
            return None
            
        try:
            data, count = self.supabase.table('repositories').select('digest_url').eq('id', repo_id).execute()
            
            if data and len(data) > 1 and len(data[1]) > 0 and 'digest_url' in data[1][0]:
                digest_url = data[1][0]['digest_url']
                if digest_url:
                    logger.info(f"Found digest URL for repository {repo_id}: {digest_url}")
                    return digest_url
                else:
                    logger.warning(f"No digest URL found for repository {repo_id}")
                    return None
            else:
                logger.warning(f"No repository found with ID: {repo_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting repository digest URL: {str(e)}")
            return None
            
    def download_repository_digest(self, repo_id: int, output_path: Optional[str] = None) -> Optional[str]:
        """
        Download the repository digest from Supabase storage.
        
        Args:
            repo_id: Integer ID of the repository
            output_path: Optional path to save the downloaded digest
            
        Returns:
            Path to the saved digest file or None if download failed
        """
        import os
        import requests
        
        if not self.is_available():
            logger.warning("Supabase client not available.")
            return None
            
        try:
            # Get the digest URL
            digest_url = self.get_repository_digest_url(repo_id)
            if not digest_url:
                logger.warning(f"No digest URL found for repository {repo_id}")
                return None
                
            # Download the digest
            response = requests.get(digest_url)
            if response.status_code != 200:
                logger.error(f"Failed to download digest: HTTP {response.status_code}")
                return None
                
            # Determine where to save it
            if not output_path:
                output_path = os.path.join(os.getcwd(), f"repository_{repo_id}_digest.txt")
                
            # Save the digest content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            logger.info(f"Repository digest downloaded and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading repository digest: {str(e)}")
            return None

    def get_function_count(self, repo_id: int) -> Optional[int]:
        """
        Get the number of functions in a repository.
        
        Args:
            repo_id: The repository ID
            
        Returns:
            The number of functions, or None if not found/error
        """
        if not self.supabase:
            return None
            
        try:
            response = self.supabase.table("functions").select("count").eq("repository_id", repo_id).execute()
            if response and hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0]['count']
            return 0
        except Exception as e:
            logger.warning(f"Error getting function count: {str(e)}")
            return None
            
    def get_class_count(self, repo_id: int) -> Optional[int]:
        """
        Get the number of classes in a repository.
        
        Args:
            repo_id: The repository ID
            
        Returns:
            The number of classes, or None if not found/error
        """
        if not self.supabase:
            return None
            
        try:
            response = self.supabase.table("classes").select("count").eq("repository_id", repo_id).execute()
            if response and hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0]['count']
            return 0
        except Exception as e:
            logger.warning(f"Error getting class count: {str(e)}")
            return None
            
    def get_import_count(self, repo_id: int) -> Optional[int]:
        """
        Get the number of imports in a repository.
        
        Args:
            repo_id: The repository ID
            
        Returns:
            The number of imports, or None if not found/error
        """
        if not self.supabase:
            return None
            
        try:
            response = self.supabase.table("imports").select("count").eq("repository_id", repo_id).execute()
            if response and hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0]['count']
            return 0
        except Exception as e:
            logger.warning(f"Error getting import count: {str(e)}")
            return None

# Singleton instance for convenience
_db_instance = None

def get_db():
    """Get a singleton instance of the RepositoryDB."""
    try:
        db = RepositoryDB()
        if not db.is_available():
            logger.warning("Database connection not available")
            return None
        return db
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return None