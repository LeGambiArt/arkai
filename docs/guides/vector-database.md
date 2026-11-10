# Vector database

Arkai runs its vector database server as a container under a container
engine. Two vendors are supported: **Qdrant** (the default) and
**ChromaDB**. Both are driven over their HTTP APIs, so no Python
database packages are installed — this keeps the `rag` extra free of the
ChromaDB dependency that blocked Python 3.14.

## Requirements

- A container engine: **podman**, Apple's **container** runtime
  (macOS 26+ on Apple silicon), or **docker**. With the default
  `vectordb.engine: auto`, Arkai probes podman first, then Apple's
  `container`, then docker, and uses the first one it finds.
- The engine must be running before `arkai vectordb start`:
  - podman on macOS: `podman machine start`
  - Apple container: `container system start`
  - docker: start Docker Desktop or run `colima start`

If the engine (or its VM) is not running, `arkai vectordb start` fails
with an error naming the engine and the exact command to fix it.

## Usage

```bash
arkai vectordb start
arkai vectordb status
arkai vectordb list
arkai vectordb initdb manuals
arkai vectordb drop manuals
arkai vectordb stop
```

The server listens on the configured host port (default `8082`, bound to
127.0.0.1) and persists data under Arkai's data directory
(`~/.local/share/arkai/qdrant` or `.../chromadb` by default; override
with `vectordb.database_dir`).

## Configuration

```yaml
vectordb:
  vendor: qdrant       # qdrant (default) or chromadb
  engine: auto         # auto, podman, apple, or docker
  port: 8082           # host port published for the container
  image: null          # override the container image (per-vendor default)
  database_dir: null   # override the data directory
```

- `vendor` selects the database. Qdrant collections are created with the
  embedding dimension of the configured `rag.embedding_model`
  (all-MiniLM-L6-v2, 384 dimensions, by default); ingesting into a
  missing collection creates it using the actual embedding dimension.
- `image` overrides the container image reference
  (`docker.io/qdrant/qdrant:latest` / `docker.io/chromadb/chroma:latest`
  by default).
- The container is named `arkai-vectordb`; stopping the server stops and
  removes it.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Container engine 'podman' is not ready` | `podman machine start` (macOS) |
| `Apple container services are not running` | `container system start` |
| `Docker is not ready` | Start Docker Desktop or `colima start` |
| `Port 8082 already in use` | Set `vectordb.port` or free the port |
| Container starts but times out | Check `podman logs arkai-vectordb` / `docker logs arkai-vectordb` |
| `No container engine found` | Install podman, Apple's container, or docker |

The current implementation warns that vector database commands are not
fully functional yet. See the [vector database
design](../design/vector-database.md) for lifecycle and API details. The
database is used by the [RAG guide](rag.md).
