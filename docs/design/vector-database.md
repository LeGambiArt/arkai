# Design: vector database

## Scope

The vector database feature manages a local ChromaDB server and exposes
database/collection operations to RAG. ChromaDB is the only supported
vendor in the current configuration validator.

## Lifecycle and persistence

The CLI resolves the Chroma executable, port, and data directory from
Arkai configuration. It starts the server bound to loopback, records the
process ID and startup state, and polls Chroma's heartbeat endpoint
before reporting success. State files let later commands find the server
and port. A startup timeout kills the child and removes partial state.

The default database directory is in Arkai's data area;
`vectordb.database_dir` can override it. Database data persists when the
server stops.

## Status

The implementation currently warns that vector database commands are not
fully functional. Treat command availability and behavior as evolving;
RAG behavior depends on this service and should be tested against the
running Chroma version.

## Implementation map

- Server lifecycle and database operations: `arkai/vectordb.py`
- RAG client operations: `arkai/rag.py`
- Unit/BDD coverage: `tests/unit/test_vectordb.py`,
  `features/vectordb.feature`
- User instructions: [Vector database](../guides/vector-database.md)
