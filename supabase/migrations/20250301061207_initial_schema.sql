-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create repositories table
CREATE TABLE repositories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    file_count INTEGER NOT NULL,
    total_size_bytes BIGINT NOT NULL,
    file_types JSONB,
    processed_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create files table
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    extension TEXT,
    last_modified TIMESTAMP WITH TIME ZONE,
    lines INTEGER,
    characters INTEGER,
    content TEXT,  -- Keep the truncated content for now
    content_url TEXT, -- URL for the file in Supabase Storage
    embedding VECTOR(1536), -- 1536 for OpenAI, 384 for gte-small
    content_hash TEXT,
    UNIQUE(repository_id, path)
);

-- Create functions table
CREATE TABLE functions (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    signature TEXT,
    docstring TEXT,
    body TEXT,
    start_line INTEGER,
    end_line INTEGER
);

-- Create classes table
CREATE TABLE classes (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    docstring TEXT,
    body TEXT,
    start_line INTEGER,
    end_line INTEGER
);

-- Create imports table
CREATE TABLE imports (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    line INTEGER
);

-- Create vector similarity search function
CREATE OR REPLACE FUNCTION match_code_files(
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_count INT,
    repo_id INTEGER
)
RETURNS TABLE (
    id INTEGER,
    repository_id INTEGER,
    path TEXT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        files.id,
        files.repository_id,
        files.path,
        files.content,
        1 - (files.embedding <=> query_embedding) AS similarity
    FROM files
    WHERE
        files.embedding IS NOT NULL
        AND (repo_id IS NULL OR files.repository_id = repo_id)
        AND 1 - (files.embedding <=> query_embedding) > match_threshold
    ORDER BY files.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create indexes (with names - for easier management)
CREATE INDEX repositories_path_idx ON repositories (path);

CREATE INDEX files_repo_id_idx ON files (repository_id);
CREATE INDEX files_repo_id_path_idx ON files (repository_id, path);
CREATE INDEX files_embedding_idx ON files USING hnsw (embedding vector_cosine_ops); -- HNSW index

CREATE INDEX functions_repo_id_idx ON functions (repository_id);
CREATE INDEX functions_repo_id_file_id_idx ON functions (repository_id, file_id);
CREATE INDEX functions_name_idx ON functions (name);

CREATE INDEX classes_repo_id_idx ON classes (repository_id);
CREATE INDEX classes_repo_id_file_id_idx ON classes (repository_id, file_id);
CREATE INDEX classes_name_idx ON classes (name);

CREATE INDEX imports_file_id_idx ON imports (file_id);
CREATE INDEX imports_repo_id_idx ON imports (repository_id);

-- Create the 'file_contents' storage bucket.  IMPORTANT: This requires the service role key.
INSERT INTO storage.buckets (id, name, public)
VALUES ('file_contents', 'file_contents', true); -- Set to 'true' for public access