from sgqlc.types import Schema, Type, Field, NonNull, Int, String, List, Float

# Define GraphQL types, mirroring your Supabase schema

class Function(Type):
    __typename__ = 'functions' # MUST match the table name in Supabase
    id = Field(NonNull(Int))
    name = Field(NonNull(String))
    signature = Field(String)
    docstring = Field(String)
    body = Field(String) # Might want to omit or truncate this
    start_line = Field(Int)
    end_line = Field(Int)
    file_id = Field(Int) # Add file_id and repo
    repository_id = Field(Int)

class Class(Type):
    __typename__ = 'classes'
    id = Field(NonNull(Int))
    name = Field(NonNull(String))
    docstring = Field(String)
    body = Field(String) # Might want to omit or truncate
    start_line = Field(Int)
    end_line = Field(Int)
    file_id = Field(Int) # Add file_id and repo
    repository_id = Field(Int)

class Import(Type):  # Added import type.
    __typename__ = 'imports'
    id = Field(NonNull(Int))
    statement = Field(NonNull(String))
    line = Field(Int)
    file_id = Field(Int)  # Add file_id and repo
    repository_id = Field(Int)


class File(Type):
    __typename__ = 'files'
    id = Field(NonNull(Int))
    path = Field(NonNull(String))
    repository_id = Field(NonNull(Int))
    content = Field(String)  # Probably don't fetch this, use content_url
    content_url = Field(String) #Definitely want this
    similarity = Field(Float) #for the query
    functions = Field(List(Function))  # Relationship to functions
    classes = Field(List(Class))      # Relationship to classes
    imports = Field(List(Import)) # Relationship to imports

class Repository(Type):
    __typename__ = 'repositories'
    id = Field(NonNull(Int))
    name = Field(NonNull(String))
    path = Field(String) # Keep path
    files = Field(List(File)) # Add relationship, so we can use nested queries

# Root Query type
class Query(Type):
    __typename__ = 'Query'
    # Search
    match_code_files = Field(
        List(File),
        args={
            'repo_id': Int,
            'query_embedding': NonNull(List(NonNull(Float))), # Add NonNull
            'match_threshold': Float,
            'match_count': Int
        }
    )
    # Add a way to query for a specific file
    file_by_path = Field(File, args={'repo_id': Int, 'path': String})
    #get a single repository
    repository = Field(Repository, args={'id': Int})
    #get all repositories.
    repositories = Field(List(Repository))

# Create the schema
schema = Schema(query=Query) 