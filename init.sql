-- Initialize database with extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create tables (existing schema will be created by app/db/init_db.py)
-- This file is for initial setup only

-- Log successful initialization
SELECT 'Database initialized successfully' as status;
