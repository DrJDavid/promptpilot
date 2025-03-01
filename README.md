# PromptPilot with Supabase Integration

PromptPilot is a tool for analyzing code repositories and generating intelligent prompts for AI-assisted development. This version includes Supabase integration for efficient vector search capabilities.

## Features

- Repository ingestion and code extraction
- Code embedding generation and storage using Supabase pgvector
- Semantic code search using vector similarity
- AI-enhanced code understanding and prompt generation

## Setup

### Prerequisites

- Python 3.9+ installed
- A Supabase account (free tier available at [supabase.com](https://supabase.com))
- Git repository to analyze

### Environment Setup

1. Clone this repository
2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Supabase Configuration

1. Create a new Supabase project
2. Enable the pgvector extension via SQL Editor:
   ```sql
   -- Enable the pgvector extension
   create extension if not exists vector;
   ```
3. Create the necessary tables:
   ```sql
   -- Create repositories table
   create table repositories (
     id uuid primary key,
     name text not null,
     path text not null,
     file_count integer not null default 0,
     total_size_bytes bigint not null default 0,
     file_types jsonb,
     processed_date timestamp with time zone default now(),
     created_at timestamp with time zone default now()
   );

   -- Create files table
   create table files (
     id uuid primary key,
     repository_id uuid references repositories(id) on delete cascade,
     file_path text not null,
     size_bytes bigint not null default 0,
     extension text,
     last_modified timestamp with time zone,
     lines integer not null default 0,
     characters integer not null default 0,
     content text,
     content_url text,
     created_at timestamp with time zone default now()
   );

   -- Create file_embeddings table
   create table file_embeddings (
     id uuid primary key,
     file_id uuid references files(id) on delete cascade,
     content text,
     embedding vector(1536),
     metadata jsonb,
     created_at timestamp with time zone default now()
   );
   ```

4. Create a Storage bucket named "file_contents" with public read access

5. Create the following RPC functions for similarity search:
   ```sql
   -- Function to search file embeddings by vector similarity
   create or replace function match_file_embeddings(
     query_embedding vector(1536),
     match_threshold float,
     match_count int,
     repo_id uuid default null
   )
   returns table (
     id uuid,
     file_id uuid,
     file_path text,
     content text,
     similarity float,
     metadata jsonb
   )
   language plpgsql
   as $$
   begin
     return query
     select
       e.id,
       e.file_id,
       f.file_path,
       e.content,
       1 - (e.embedding <=> query_embedding) as similarity,
       e.metadata
     from
       file_embeddings e
     join
       files f on e.file_id = f.id
     where
       (repo_id is null or f.repository_id = repo_id) and
       (1 - (e.embedding <=> query_embedding)) > match_threshold
     order by
       e.embedding <=> query_embedding
     limit match_count;
   end;
   $$;

   -- Function to search file embeddings by text
   create or replace function search_files_by_text(
     query_text text,
     match_threshold float,
     match_count int,
     repo_id uuid default null
   )
   returns table (
     id uuid,
     file_path text,
     content text,
     similarity float,
     metadata jsonb
   )
   language plpgsql
   as $$
   declare
     query_embedding vector(1536);
   begin
     -- Call embedding generation (replace with your implementation)
     -- This is a placeholder - you need to implement embedding generation
     -- For example, through an Edge Function
     query_embedding := public.generate_embedding(query_text);
     
     return query
     select
       f.id,
       f.file_path,
       e.content,
       1 - (e.embedding <=> query_embedding) as similarity,
       e.metadata
     from
       file_embeddings e
     join
       files f on e.file_id = f.id
     where
       (repo_id is null or f.repository_id = repo_id) and
       (1 - (e.embedding <=> query_embedding)) > match_threshold
     order by
       e.embedding <=> query_embedding
     limit match_count;
   end;
   $$;
   ```

6. Create a Supabase Edge Function for embedding generation:
   - Go to Edge Functions in your Supabase dashboard
   - Create a new function named "generate-embeddings"
   - Implement the function to generate embeddings (example using gte-small):
   ```typescript
   import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
   import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

   const embeddingModel = 'gte-small'; // Supabase's default is gte-small
   
   serve(async (req) => {
     // Create a Supabase client with the service role key
     const supabaseClient = createClient(
       Deno.env.get('SUPABASE_URL') ?? '',
       Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
       { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
     )

     // Parse the request body
     const { text } = await req.json()
     
     if (!text) {
       return new Response(
         JSON.stringify({ error: 'Missing text parameter' }),
         { status: 400, headers: { 'Content-Type': 'application/json' } }
       )
     }

     try {
       // Generate embedding using pgvector's built-in functionality
       // This is a placeholder - implement according to your chosen approach
       // Could use Supabase's built-in embedding generation or call an external API
       
       // Example using custom SQL function (you'd need to create this)
       const { data, error } = await supabaseClient.rpc('generate_embedding', {
         input_text: text
       })
       
       if (error) throw error
       
       return new Response(
         JSON.stringify({ embedding: data }),
         { status: 200, headers: { 'Content-Type': 'application/json' } }
       )
     } catch (error) {
       return new Response(
         JSON.stringify({ error: error.message }),
         { status: 500, headers: { 'Content-Type': 'application/json' } }
       )
     }
   })
   ```

7. Create a `.env` file in the project root with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   OPENAI_API_KEY=your_openai_api_key  # Optional, for fallback embedding generation
   ```

## Usage

### Ingesting a Repository

```python
from core.ingest import RepositoryIngestor

# Initialize the repository ingestor
ingestor = RepositoryIngestor('/path/to/your/repo')

# Process the repository
repo_data = ingestor.process_repository()
```

### Generating Embeddings

```python
from core.analyze import RepositoryAnalyzer

# Initialize the repository analyzer with Supabase integration
analyzer = RepositoryAnalyzer('/path/to/repository_data.json', use_supabase=True)

# Generate embeddings
embeddings_data = analyzer.generate_embeddings()
```

### Searching for Similar Code

```python
from core.enhanced_db import get_db

# Get the database instance
db = get_db()

# Search for similar code by text query
results = db.search_similar_text("implement authentication function", top_k=5)

# Print results
for result in results:
    print(f"File: {result['file_path']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Content: {result['content'][:200]}...")
    print("-" * 50)
```

## Advanced Configuration

### Chunking Strategy

You can adjust the chunking strategy for large files in the `_chunk_content` method of the `EnhancedDatabase` class:

```python
# Process repository with custom chunk size
db.process_repository_chunks(repo_data, chunk_size=3000)
```

### Similarity Threshold

Adjust the similarity threshold in search functions to control the strictness of matches:

```python
# More strict matching
results = db.search_similar_text("query", similarity_threshold=0.8)

# More lenient matching
results = db.search_similar_text("query", similarity_threshold=0.5)
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
