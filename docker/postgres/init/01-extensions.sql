-- Runs once, when the data volume is first created.
-- The Alembic migration also creates the extension, so a database provisioned
-- outside Compose still works; this just makes `recall migrate` cheaper and
-- surfaces a missing pgvector image early.
CREATE EXTENSION IF NOT EXISTS vector;
