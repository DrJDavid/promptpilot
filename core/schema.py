"""
GraphQL schema implementation for PromptPilot.
This defines the GraphQL schema for querying repository data via Supabase PostgreSQL database.
"""

from sgqlc.types import Type, Field, list_of, String, Int, Float, ID, Boolean, non_null
from sgqlc.types.relay import Node, Connection, connection_args
from sgqlc.operation import Operation

# Base types
class File(Type):
    """File entity representing a file in a repository."""
    id = ID
    repository_id = ID
    path = String
    content = String
    size_bytes = Int
    extension = String
    last_modified = String  # timestamp
    lines = Int
    characters = Int
    content_url = String
    content_hash = String
    embedding = String  # Vector type, represented as string in GraphQL
    similarity = Float
    
class Function(Type):
    """Function entity representing a function in a repository."""
    id = ID
    file_id = ID
    repository_id = ID
    name = String
    signature = String
    docstring = String
    body = String
    start_line = Int
    end_line = Int
    similarity = Float
    
class Class(Type):
    """Class entity representing a class in a repository."""
    id = ID
    file_id = ID
    repository_id = ID
    name = String
    docstring = String
    body = String
    start_line = Int
    end_line = Int
    similarity = Float
    
class Import(Type):
    """Import entity representing an import statement in a repository."""
    id = ID
    file_id = ID
    repository_id = ID
    statement = String
    line = Int
    
class Repository(Type):
    """Repository entity containing files, functions, classes, and imports."""
    id = ID
    name = String
    path = String
    file_count = Int
    total_size_bytes = Int
    file_types = String  # JSON type, represented as string in GraphQL
    processed_date = String  # timestamp
    
    # Define resolvers for embeddings-based semantic search
    files = Field(list_of(File), args={
        'search': String,
        'similarityThreshold': Float,
        'limit': Int
    })
    
    functions = Field(list_of(Function), args={
        'search': String,
        'similarityThreshold': Float,
        'limit': Int
    })
    
    classes = Field(list_of(Class), args={
        'search': String,
        'similarityThreshold': Float,
        'limit': Int
    })
    
    imports = Field(list_of(Import), args={
        'limit': Int
    })
    
class Query(Type):
    """Root query type."""
    repository = Field(Repository, args={'id': ID})
    repositories = Field(list_of(Repository), args={
        'where': String,  # JSON argument for filtering
        'limit': Int,
        'offset': Int,
        'order': String  # JSON for ordering
    })
    
# Create schema
schema = {
    'query': Query
} 