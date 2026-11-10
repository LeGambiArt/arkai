# Design: vector database

## Scope

The vector database feature runs a local vector database server as a
container and exposes database/collection operations to RAG. Two vendors
are supported: Qdrant (the default) and ChromaDB, selected with
`vectordb.vendor`. The server always runs under a container engine —
podman, Apple's `container` runtime, or docker — selected with
`vectordb.engine` (`auto` probes podman, then Apple container, then
docker). Both vendors are driven through their HTTP APIs; no Python
database package is required, which unblocked Python 3.14 support.

## Lifecycle and persistence

The CLI resolves the engine, image, host port, and data directory from
Arkai configuration. Before starting anything it verifies the engine's
VM/daemon is ready and raises an actionable error otherwise (e.g.
`podman machine start`, `container system start`, `colima start`). It
then starts a detached container named `arkai-vectordb` with the host
port bound to 127.0.0.1 and the data directory bind-mounted at the
vendor's persistent path (`/qdrant/storage` for Qdrant, `/chroma/chroma`
with `IS_PERSISTENT=TRUE` for ChromaDB). Readiness is polled on the
vendor's health endpoint (`/healthz` for Qdrant, `/api/v2/heartbeat`
for ChromaDB); a timeout stops and removes the container and cleans up
state.

The state file (`vectordb.state`) records `port`, `vendor`, `engine`,
`started_at`, and `databases`; there is no PID (the container engine
owns the process). `is_vectordb_running` checks the state file plus the
engine's container inspection. The default database directory is in
Arkai's data area, namespaced per vendor; `vectordb.database_dir` can
override it. Database data persists when the server stops.

## Vendor API abstraction

`arkai/vectordb.py` exposes vendor-neutral operations —
`list_collections`, `create_collection`, `drop_collection`,
`add_documents`, `search_documents` — dispatched on the running server's
vendor. Qdrant needs the vector dimension at collection creation time;
`rag ingest` passes the actual embedding dimension, and standalone
`vectordb initdb` derives it from the configured embedding model.
Search results are normalized to Chroma-style distances (lower is
better) so RAG scoring works unchanged for both vendors.

## Status

The implementation currently warns that vector database commands are not
fully functional. Treat command availability and behavior as evolving;
RAG behavior depends on this service and should be tested against the
running database version.

## Implementation map

- Container engine abstraction (run/stop/inspect, VM readiness errors):
  `arkai/container_engine.py`
- Server lifecycle, database operations, and vendor dispatch:
  `arkai/vectordb.py`
- RAG client operations: `arkai/rag.py`
- Unit/BDD coverage: `tests/unit/test_vectordb.py`,
  `tests/unit/test_container_engine.py`, `features/vectordb.feature`
- User instructions: [Vector database](../guides/vector-database.md)
