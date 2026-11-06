# Vector database

Arkai currently integrates with ChromaDB. Install its `chroma` CLI
separately; the server uses the configured port (default `8082`) and
database directory (by default under Arkai's data directory).

```bash
arkai vectordb start
arkai vectordb status
arkai vectordb list
arkai vectordb initdb manuals
arkai vectordb stop
```

`vectordb drop NAME` removes a database. Use `arkai vectordb --help` for
the available commands. The database is used by the [RAG guide](rag.md).

The current implementation warns that vector database commands are not
fully functional yet. See the [vector database
design](../design/vector-database.md) for lifecycle and API details.
