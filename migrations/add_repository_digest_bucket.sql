-- SQL migration to add support for digest storage in buckets
-- This migration assumes digest_url column already exists from previous migration

-- Verify that the digest_url column exists in repositories table
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'repositories' 
        AND column_name = 'digest_url'
    ) THEN
        ALTER TABLE repositories ADD COLUMN digest_url TEXT;
    END IF;
END $$;

-- Create a storage.buckets record if it doesn't exist
-- Note: This is typically handled by the Supabase client in code when
-- creating buckets, but adding it here for completeness
INSERT INTO storage.buckets (id, name, public)
SELECT 'repository_digests', 'repository_digests', true
WHERE NOT EXISTS (
    SELECT 1 FROM storage.buckets WHERE id = 'repository_digests'
);

-- Add comment about bucket policies
COMMENT ON TABLE storage.buckets IS 'Storage bucket for repository digests. Ensure appropriate policies are set for public access.'; 