-- Run this once in Supabase → SQL Editor → New query → Run.

-- Enables vector similarity search.
create extension if not exists vector;

-- Replaces the local Chroma 'meetings' collection: transcript + WhatsApp chunks.
-- Column is named 'ts', not 'timestamp' — 'timestamp' is a reserved keyword
-- and breaks parsing inside the match_chunks() function's RETURNS TABLE clause
-- below. The app still calls this field "timestamp" everywhere in Python.
create table if not exists chunks (
  id text primary key,
  text text not null,
  session text not null,
  ts text not null,
  embedding vector(768) not null
);

create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);

-- Nearest-neighbor search, called from the backend as supabase.rpc('match_chunks', ...).
-- Returns cosine distance so retrieval.py's relevance-threshold logic carries over
-- unchanged in shape (lower distance = closer), though the actual threshold number
-- will need recalibrating against this new embedding model.
create or replace function match_chunks(
  query_embedding vector(768),
  match_count int default 4
)
returns table (
  id text,
  text text,
  session text,
  ts text,
  distance float
)
language sql stable
as $$
  select
    id,
    text,
    session,
    ts,
    embedding <=> query_embedding as distance
  from chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Logs every /ask call for the monitoring dashboard.
create table if not exists queries (
  id bigserial primary key,
  question text not null,
  answer text not null,
  relevant boolean not null,
  sources jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);
