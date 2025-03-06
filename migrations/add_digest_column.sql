-- Add digest column to repositories table
ALTER TABLE repositories 
ADD COLUMN IF NOT EXISTS digest TEXT;

-- Add indexes for AST data tables to fix the on_conflict issues
-- For functions table
CREATE INDEX IF NOT EXISTS functions_repository_id_idx ON functions(repository_id);
CREATE INDEX IF NOT EXISTS functions_file_id_idx ON functions(file_id);

-- For classes table
CREATE INDEX IF NOT EXISTS classes_repository_id_idx ON classes(repository_id);
CREATE INDEX IF NOT EXISTS classes_file_id_idx ON classes(file_id);

-- For imports table
CREATE INDEX IF NOT EXISTS imports_repository_id_idx ON imports(repository_id);
CREATE INDEX IF NOT EXISTS imports_file_id_idx ON imports(file_id);

-- Change column types if needed for AST data
ALTER TABLE functions 
ALTER COLUMN repository_id TYPE INTEGER USING repository_id::INTEGER;

ALTER TABLE classes
ALTER COLUMN repository_id TYPE INTEGER USING repository_id::INTEGER;

ALTER TABLE imports
ALTER COLUMN repository_id TYPE INTEGER USING repository_id::INTEGER; 