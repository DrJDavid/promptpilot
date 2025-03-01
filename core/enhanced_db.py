"""
Database integration module for PromptPilot.

This module handles the ArangoDB connection for storing repository data,
analysis results, and code structure information.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dotenv import load_dotenv

# Try to import arango or mark as unavailable
try:
    from arango import ArangoClient
    from arango.exceptions import ServerConnectionError, DatabaseCreateError, CollectionCreateError
    ARANGO_AVAILABLE = True
    logging.info("ArangoDB client is available")
except ImportError:
    ARANGO_AVAILABLE = False
    logging.warning("ArangoDB client is not available (pip install python-arango)")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('promptpilot.db')

# Load environment variables
load_dotenv()

# Default connection settings
DEFAULT_HOST = os.environ.get('ARANGO_HOST', 'localhost')
DEFAULT_PORT = int(os.environ.get('ARANGO_PORT', 8529))
DEFAULT_DB_NAME = os.environ.get('ARANGO_DB', 'promptpilot')
DEFAULT_USER = os.environ.get('ARANGO_USER', 'root')
DEFAULT_PASSWORD = os.environ.get('ARANGO_PASSWORD', '')

# Collections to create
COLLECTIONS = {
    'repositories': {'edge': False, 'description': 'Repository metadata'},
    'files': {'edge': False, 'description': 'File content and metadata'},
    'functions': {'edge': False, 'description': 'Function definitions'},
    'classes': {'edge': False, 'description': 'Class definitions'},
    'imports': {'edge': False, 'description': 'Import statements'},
    'embeddings': {'edge': False, 'description': 'File and code embeddings'},
    'analysis': {'edge': False, 'description': 'Analysis results'},
    'contains': {'edge': True, 'description': 'Relationship between repositories and files'},
    'defines': {'edge': True, 'description': 'Relationship between files and functions/classes'},
    'imports_from': {'edge': True, 'description': 'Relationship between files for imports'},
    'similar_to': {'edge': True, 'description': 'Similarity relationships between files'}
}

# Graph definitions
GRAPHS = {
    'code_structure': {
        'edge_definitions': [
            {
                'collection': 'contains',
                'from': ['repositories'],
                'to': ['files']
            },
            {
                'collection': 'defines',
                'from': ['files'],
                'to': ['functions', 'classes']
            },
            {
                'collection': 'imports_from',
                'from': ['files'],
                'to': ['files']
            },
            {
                'collection': 'similar_to',
                'from': ['files'],
                'to': ['files']
            }
        ]
    }
}


class RepositoryDB:
    """Class to handle database operations for repositories."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, 
                 db_name: str = DEFAULT_DB_NAME, username: str = DEFAULT_USER, 
                 password: str = DEFAULT_PASSWORD):
        """
        Initialize the database connection.
        
        Args:
            host: ArangoDB host
            port: ArangoDB port
            db_name: Database name
            username: Username for authentication
            password: Password for authentication
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.username = username
        self.password = password
        
        # Check if ArangoDB client is available
        if not ARANGO_AVAILABLE:
            logger.warning("ArangoDB client is not available. Using fallback storage.")
            self.db = None
            self.client = None
            return
        
        # Initialize connection
        self.client = ArangoClient(hosts=f'http://{host}:{port}')
        
        try:
            # Connect to system database
            sys_db = self.client.db('_system', username=username, password=password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(db_name):
                sys_db.create_database(
                    name=db_name,
                    users=[{'username': username, 'password': password, 'active': True}]
                )
                logger.info(f"Created database: {db_name}")
            
            # Connect to the database
            self.db = self.client.db(db_name, username=username, password=password)
            logger.info(f"Connected to database: {db_name}")
            
            # Initialize collections and graphs
            self._initialize_collections()
            self._initialize_graphs()
            
        except ServerConnectionError as e:
            logger.error(f"Failed to connect to ArangoDB: {str(e)}")
            self.db = None
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.db = None
            self.client = None
    
    def _initialize_collections(self) -> None:
        """Initialize collections in the database."""
        if not self.db:
            return
        
        try:
            for name, config in COLLECTIONS.items():
                if not self.db.has_collection(name):
                    if config['edge']:
                        self.db.create_collection(name, edge=True)
                    else:
                        self.db.create_collection(name)
                    logger.info(f"Created collection: {name}")
            
            # Create indexes for improved query performance
            self._ensure_indexes()
            
        except Exception as e:
            logger.error(f"Error creating collections: {str(e)}")
    
    def _ensure_indexes(self) -> None:
        """Ensure all necessary indexes exist in the database."""
        if not self.db:
            return
            
        try:
            # Index for repositories collection
            repo_collection = self.db.collection('repositories')
            self._create_index_if_not_exists(repo_collection, ['path'], unique=True)
            
            # Indexes for files collection
            files_collection = self.db.collection('files')
            self._create_index_if_not_exists(files_collection, ['repo_id'])
            self._create_index_if_not_exists(files_collection, ['repo_id', 'path'])
            
            # Indexes for functions collection
            functions_collection = self.db.collection('functions')
            self._create_index_if_not_exists(functions_collection, ['repo_id'])
            self._create_index_if_not_exists(functions_collection, ['repo_id', 'file_path'])
            self._create_index_if_not_exists(functions_collection, ['name'])
            
            # Indexes for classes collection
            classes_collection = self.db.collection('classes')
            self._create_index_if_not_exists(classes_collection, ['repo_id'])
            self._create_index_if_not_exists(classes_collection, ['repo_id', 'file_path'])
            self._create_index_if_not_exists(classes_collection, ['name'])
            
            # Index for embeddings collection
            embeddings_collection = self.db.collection('embeddings')
            self._create_index_if_not_exists(embeddings_collection, ['repo_id'])
            self._create_index_if_not_exists(embeddings_collection, ['file_id'])
            
            logger.info("Database indexes verified")
            
        except Exception as e:
            logger.error(f"Error ensuring indexes: {str(e)}")
    
    def _create_index_if_not_exists(self, collection, fields, unique=False, sparse=False):
        """Create an index on a collection if it doesn't already exist."""
        try:
            # Check if index already exists
            existing_indexes = collection.indexes()
            
            for idx in existing_indexes:
                if idx['type'] == 'hash' and set(idx['fields']) == set(fields):
                    # Index already exists
                    return
            
            # Create the index
            collection.add_hash_index(fields=fields, unique=unique, sparse=sparse)
            logger.info(f"Created index on {collection.name}: {fields}")
            
        except Exception as e:
            logger.error(f"Error creating index on {collection.name}: {str(e)}")
    
    def _initialize_graphs(self) -> None:
        """Initialize graphs in the database."""
        if not self.db:
            return
        
        try:
            graph_service = self.db.graphs()
            
            for name, config in GRAPHS.items():
                if not graph_service.has(name):
                    graph = graph_service.create(name)
                    
                    # Add edge definitions
                    for edge_def in config['edge_definitions']:
                        graph.create_edge_definition(
                            edge_collection=edge_def['collection'],
                            from_vertex_collections=edge_def['from'],
                            to_vertex_collections=edge_def['to']
                        )
                    
                    logger.info(f"Created graph: {name}")
        except Exception as e:
            logger.error(f"Error creating graphs: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if the database is available."""
        return self.db is not None
    
    def store_repository(self, repo_data: Dict[str, Any]) -> Optional[str]:
        """
        Store repository data in the database.
        
        Args:
            repo_data: Repository data dictionary
            
        Returns:
            Repository ID or None if storage failed
        """
        if not self.is_available():
            logger.warning("Database not available. Repository data not stored.")
            return None
        
        try:
            # Create repository document
            repo_doc = {
                'name': repo_data['name'],
                'path': repo_data['path'],
                'file_count': repo_data['file_count'],
                'total_size_bytes': repo_data['total_size_bytes'],
                'file_types': repo_data['file_types'],
                'processed_date': repo_data['processed_date'],
                'timestamp': time.time()
            }
            
            # Check if repository already exists
            query = f"FOR doc IN repositories FILTER doc.path == '{repo_doc['path']}' RETURN doc"
            cursor = self.db.aql.execute(query)
            existing_docs = [doc for doc in cursor]
            
            if existing_docs:
                # Update existing repository
                repo_id = existing_docs[0]['_key']
                self.db.collection('repositories').update({
                    '_key': repo_id,
                    **repo_doc
                })
                logger.info(f"Updated repository in database: {repo_doc['name']}")
            else:
                # Insert new repository
                result = self.db.collection('repositories').insert(repo_doc)
                repo_id = result['_key']
                logger.info(f"Stored repository in database: {repo_doc['name']}")
            
            return repo_id
            
        except Exception as e:
            logger.error(f"Error storing repository data: {str(e)}")
            return None
    
    def store_files(self, repo_id: str, files: List[Dict[str, Any]]) -> bool:
        """
        Store file data in the database.
        
        Args:
            repo_id: Repository ID
            files: List of file data dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Database not available. File data not stored.")
            return False
        
        try:
            # First, clean up existing files for this repository
            self._cleanup_repository_data(repo_id)
            
            # Process each file
            file_ids = []
            
            for file_entry in files:
                # Create file document
                file_doc = {
                    'repo_id': repo_id,
                    'path': file_entry['metadata']['path'],
                    'size_bytes': file_entry['metadata'].get('size_bytes', 0),
                    'extension': file_entry['metadata'].get('extension', ''),
                    'last_modified': file_entry['metadata'].get('last_modified', 0),
                    'lines': file_entry.get('lines', 0),
                    'characters': file_entry.get('characters', 0),
                    'content': file_entry.get('content', ''),
                    'timestamp': time.time()
                }
                
                # Insert file
                result = self.db.collection('files').insert(file_doc)
                file_id = result['_key']
                file_ids.append(file_id)
                
                # Create relationship between repository and file
                edge_doc = {
                    '_from': f'repositories/{repo_id}',
                    '_to': f'files/{file_id}'
                }
                self.db.collection('contains').insert(edge_doc)
            
            logger.info(f"Stored {len(file_ids)} files for repository {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing file data: {str(e)}")
            return False
    
    def store_ast_data(self, repo_id: str, ast_data: Dict[str, Any]) -> bool:
        """
        Store AST data in the database.
        
        Args:
            repo_id: Repository ID
            ast_data: AST data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Database not available. AST data not stored.")
            return False
        
        try:
            # Get files for this repository
            file_map = self._get_file_map(repo_id)
            
            # Store functions
            for func in ast_data.get('functions', []):
                # Get file ID
                file_path = func.get('path', '')
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                # Create function document
                func_doc = {
                    'repo_id': repo_id,
                    'file_path': file_path,
                    'name': func.get('name', ''),
                    'signature': func.get('signature', ''),
                    'docstring': func.get('docstring', ''),
                    'body': func.get('body', ''),
                    'start_line': func.get('start_line', 0),
                    'end_line': func.get('end_line', 0),
                    'type': 'function',
                    'timestamp': time.time()
                }
                
                # Insert function
                result = self.db.collection('functions').insert(func_doc)
                func_id = result['_key']
                
                # Create relationship between file and function
                edge_doc = {
                    '_from': f'files/{file_id}',
                    '_to': f'functions/{func_id}'
                }
                self.db.collection('defines').insert(edge_doc)
            
            # Store classes
            for cls in ast_data.get('classes', []):
                # Get file ID
                file_path = cls.get('path', '')
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                # Create class document
                cls_doc = {
                    'repo_id': repo_id,
                    'file_path': file_path,
                    'name': cls.get('name', ''),
                    'docstring': cls.get('docstring', ''),
                    'body': cls.get('body', ''),
                    'start_line': cls.get('start_line', 0),
                    'end_line': cls.get('end_line', 0),
                    'type': 'class',
                    'timestamp': time.time()
                }
                
                # Insert class
                result = self.db.collection('classes').insert(cls_doc)
                cls_id = result['_key']
                
                # Create relationship between file and class
                edge_doc = {
                    '_from': f'files/{file_id}',
                    '_to': f'classes/{cls_id}'
                }
                self.db.collection('defines').insert(edge_doc)
            
            # Store imports
            for imp in ast_data.get('imports', []):
                # Get file ID
                file_path = imp.get('path', '')
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                # Create import document
                imp_doc = {
                    'repo_id': repo_id,
                    'file_path': file_path,
                    'statement': imp.get('statement', ''),
                    'line': imp.get('line', 0),
                    'timestamp': time.time()
                }
                
                # Insert import
                result = self.db.collection('imports').insert(imp_doc)
            
            logger.info(f"Stored AST data for repository {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing AST data: {str(e)}")
            return False
    
    def store_analysis_data(self, repo_id: str, embeddings: Dict[str, List[float]], 
                           analysis: Dict[str, Any]) -> bool:
        """
        Store analysis data in the database.
        
        Args:
            repo_id: Repository ID
            embeddings: Embeddings dictionary
            analysis: Analysis results dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Database not available. Analysis data not stored.")
            return False
        
        try:
            # Get files for this repository
            file_map = self._get_file_map(repo_id)
            file_map_by_path = {v: k for k, v in file_map.items()}
            
            # Store embeddings
            for file_path, embedding in embeddings.items():
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                # Create embedding document
                emb_doc = {
                    'repo_id': repo_id,
                    'file_id': file_id,
                    'file_path': file_path,
                    'embedding': embedding,
                    'timestamp': time.time()
                }
                
                # Insert embedding
                self.db.collection('embeddings').insert(emb_doc)
            
            # Store file similarities
            similarities = analysis.get('file_similarities', {})
            for file_path, similar_files in similarities.items():
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                # Create similarity edges
                for similar in similar_files:
                    similar_path = similar.get('path', '')
                    if similar_path not in file_map:
                        continue
                        
                    similar_id = file_map[similar_path]
                    similarity = similar.get('similarity', 0)
                    
                    # Create similarity edge
                    edge_doc = {
                        '_from': f'files/{file_id}',
                        '_to': f'files/{similar_id}',
                        'similarity': similarity
                    }
                    
                    # Check if edge already exists
                    query = f"""
                    FOR e IN similar_to 
                    FILTER e._from == 'files/{file_id}' AND e._to == 'files/{similar_id}'
                    RETURN e
                    """
                    cursor = self.db.aql.execute(query)
                    existing = [e for e in cursor]
                    
                    if not existing:
                        self.db.collection('similar_to').insert(edge_doc)
            
            # Store code structure
            code_structure = analysis.get('code_structure', {})
            
            # Store direct dependencies as edges
            dependencies = code_structure.get('direct_dependencies', {})
            for file_path, deps in dependencies.items():
                if file_path not in file_map:
                    continue
                    
                file_id = file_map[file_path]
                
                for dep in deps:
                    # Try to find the dependent file
                    for potential_path in file_map.keys():
                        if potential_path.endswith(f'/{dep}.py') or potential_path.endswith(f'/{dep}.js'):
                            dep_id = file_map[potential_path]
                            
                            # Create import edge
                            edge_doc = {
                                '_from': f'files/{file_id}',
                                '_to': f'files/{dep_id}'
                            }
                            
                            # Check if edge already exists
                            query = f"""
                            FOR e IN imports_from 
                            FILTER e._from == 'files/{file_id}' AND e._to == 'files/{dep_id}'
                            RETURN e
                            """
                            cursor = self.db.aql.execute(query)
                            existing = [e for e in cursor]
                            
                            if not existing:
                                self.db.collection('imports_from').insert(edge_doc)
            
            # Store analysis summary
            analysis_doc = {
                'repo_id': repo_id,
                'file_count': len(file_map),
                'embedding_count': len(embeddings),
                'function_count': sum(code_structure.get('function_count', {}).values()),
                'class_count': sum(code_structure.get('class_count', {}).values()),
                'import_count': len(code_structure.get('imports', {})),
                'timestamp': time.time()
            }
            
            self.db.collection('analysis').insert(analysis_doc)
            
            logger.info(f"Stored analysis data for repository {repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis data: {str(e)}")
            return False
    
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
            query = "FOR doc IN repositories RETURN doc"
            cursor = self.db.aql.execute(query)
            return [doc for doc in cursor]
            
        except Exception as e:
            logger.error(f"Error getting repositories: {str(e)}")
            return []
    
    def get_repository_by_path(self, repo_path: str) -> Optional[Dict[str, Any]]:
        """
        Get repository by path.
        
        Args:
            repo_path: Repository path
            
        Returns:
            Repository document or None if not found
        """
        if not self.is_available():
            logger.warning("Database not available. Repository not found.")
            return None
        
        try:
            query = f"FOR doc IN repositories FILTER doc.path == '{repo_path}' RETURN doc"
            cursor = self.db.aql.execute(query)
            docs = [doc for doc in cursor]
            
            if docs:
                return docs[0]
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error getting repository by path: {str(e)}")
            return None
    
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
            query = f"""
            FOR v, e IN 1..1 OUTBOUND 'repositories/{repo_id}' contains
            RETURN v
            """
            cursor = self.db.aql.execute(query)
            return [doc for doc in cursor]
            
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
            query = f"""
            FOR v, e IN 1..1 OUTBOUND 'files/{file_id}' defines
            FILTER v.type == 'function'
            RETURN v
            """
            cursor = self.db.aql.execute(query)
            return [doc for doc in cursor]
            
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
            query = f"""
            FOR v, e IN 1..1 OUTBOUND 'files/{file_id}' defines
            FILTER v.type == 'class'
            RETURN v
            """
            cursor = self.db.aql.execute(query)
            return [doc for doc in cursor]
            
        except Exception as e:
            logger.error(f"Error getting file classes: {str(e)}")
            return []
    
    def find_relevant_files(self, repo_id: str, embeddings: List[float], 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find files most relevant to a query using embeddings.
        
        Args:
            repo_id: Repository ID
            embeddings: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of file documents with similarity scores
        """
        if not self.is_available():
            logger.warning("Database not available. Returning empty relevant files list.")
            return []
        
        try:
            # Custom AQL function to compute cosine similarity
            # Note: This assumes ArangoDB has the right functions installed
            query = f"""
            FOR doc IN embeddings
            FILTER doc.repo_id == '{repo_id}'
            LET similarity = LENGTH(doc.embedding) == 0 ? 0 : 
                SUM(
                    FOR i IN 0..MIN(LENGTH(@embeddings), LENGTH(doc.embedding))-1
                    RETURN @embeddings[i] * doc.embedding[i]
                ) / (
                    SQRT(SUM(
                        FOR i IN 0..LENGTH(@embeddings)-1
                        RETURN @embeddings[i] * @embeddings[i]
                    )) * 
                    SQRT(SUM(
                        FOR i IN 0..LENGTH(doc.embedding)-1
                        RETURN doc.embedding[i] * doc.embedding[i]
                    ))
                )
            SORT similarity DESC
            LIMIT {top_k}
            LET file = DOCUMENT(CONCAT('files/', doc.file_id))
            RETURN {{
                file_id: doc.file_id,
                file_path: doc.file_path,
                similarity: similarity,
                file: file
            }}
            """
            
            cursor = self.db.aql.execute(query, bind_vars={'embeddings': embeddings})
            return [doc for doc in cursor]
            
        except Exception as e:
            logger.error(f"Error finding relevant files: {str(e)}")
            return []
    
    def _get_file_map(self, repo_id: str) -> Dict[str, str]:
        """
        Get map of file paths to IDs for a repository.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Dictionary mapping file paths to file IDs
        """
        file_map = {}
        
        if not self.is_available():
            return file_map
        
        try:
            query = f"""
            FOR v, e IN 1..1 OUTBOUND 'repositories/{repo_id}' contains
            RETURN {{path: v.path, id: v._key}}
            """
            cursor = self.db.aql.execute(query)
            
            for doc in cursor:
                file_map[doc['path']] = doc['id']
            
            return file_map
            
        except Exception as e:
            logger.error(f"Error getting file map: {str(e)}")
            return {}
    
    def _cleanup_repository_data(self, repo_id: str) -> None:
        """
        Clean up existing data for a repository.
        
        Args:
            repo_id: Repository ID
        """
        if not self.is_available():
            return
        
        try:
            # Get file IDs for this repository
            file_map = self._get_file_map(repo_id)
            file_ids = list(file_map.values())
            
            # Delete files and associated edges
            for file_id in file_ids:
                # Delete edges from files collection
                query = f"""
                FOR e IN contains
                FILTER e._to == 'files/{file_id}'
                REMOVE e IN contains
                """
                self.db.aql.execute(query)
                
                # Delete edges to functions/classes
                query = f"""
                FOR e IN defines
                FILTER e._from == 'files/{file_id}'
                REMOVE e IN defines
                """
                self.db.aql.execute(query)
                
                # Delete similarity edges
                query = f"""
                FOR e IN similar_to
                FILTER e._from == 'files/{file_id}' OR e._to == 'files/{file_id}'
                REMOVE e IN similar_to
                """
                self.db.aql.execute(query)
                
                # Delete import edges
                query = f"""
                FOR e IN imports_from
                FILTER e._from == 'files/{file_id}' OR e._to == 'files/{file_id}'
                REMOVE e IN imports_from
                """
                self.db.aql.execute(query)
                
                # Delete file
                self.db.collection('files').delete(file_id)
            
            # Delete functions for this repository
            query = f"""
            FOR doc IN functions
            FILTER doc.repo_id == '{repo_id}'
            REMOVE doc IN functions
            """
            self.db.aql.execute(query)
            
            # Delete classes for this repository
            query = f"""
            FOR doc IN classes
            FILTER doc.repo_id == '{repo_id}'
            REMOVE doc IN classes
            """
            self.db.aql.execute(query)
            
            # Delete imports for this repository
            query = f"""
            FOR doc IN imports
            FILTER doc.repo_id == '{repo_id}'
            REMOVE doc IN imports
            """
            self.db.aql.execute(query)
            
            # Delete embeddings for this repository
            query = f"""
            FOR doc IN embeddings
            FILTER doc.repo_id == '{repo_id}'
            REMOVE doc IN embeddings
            """
            self.db.aql.execute(query)
            
            # Delete analysis for this repository
            query = f"""
            FOR doc IN analysis
            FILTER doc.repo_id == '{repo_id}'
            REMOVE doc IN analysis
            """
            self.db.aql.execute(query)
            
            logger.info(f"Cleaned up existing data for repository {repo_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up repository data: {str(e)}")


# Standalone function for simple use cases
def get_repository_db(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, 
                     db_name: str = DEFAULT_DB_NAME, username: str = DEFAULT_USER, 
                     password: str = DEFAULT_PASSWORD) -> RepositoryDB:
    """
    Get a database connection for repositories.
    
    Args:
        host: ArangoDB host
        port: ArangoDB port
        db_name: Database name
        username: Username for authentication
        password: Password for authentication
        
    Returns:
        RepositoryDB instance
    """
    return RepositoryDB(host, port, db_name, username, password)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize database for PromptPilot")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"ArangoDB host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"ArangoDB port (default: {DEFAULT_PORT})")
    parser.add_argument("--db", default=DEFAULT_DB_NAME, help=f"Database name (default: {DEFAULT_DB_NAME})")
    parser.add_argument("--user", default=DEFAULT_USER, help=f"Username (default: {DEFAULT_USER})")
    parser.add_argument("--password", default='', help="Password (default: empty string)")
    
    args = parser.parse_args()
    
    try:
        # Initialize database
        db = RepositoryDB(args.host, args.port, args.db, args.user, args.password)
        
        if db.is_available():
            print(f"Database {args.db} initialized successfully.")
            print("Collections created:")
            for name in COLLECTIONS.keys():
                print(f"  - {name}")
            print("Graphs created:")
            for name in GRAPHS.keys():
                print(f"  - {name}")
        else:
            print("Failed to initialize database. Check connection settings and credentials.")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
