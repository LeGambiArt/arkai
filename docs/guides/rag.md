# Ask questions over local documents

Arkai's optional retrieval-augmented generation workflow ingests
supported documents into a vector database, retrieves relevant passages,
and sends them to a running inference server. Install the optional
dependencies:

```bash
python -m pip install 'arkai[rag]'
```

From a source checkout, install with `python -m pip install -e '.[rag]'`.

Use a Python 3.10–3.14 environment. The `rag` extra no longer installs a
vector database package: the database server runs as a container
(Qdrant by default, ChromaDB opt-in) under podman, Apple's container
runtime, or docker. See the [vector database guide](vector-database.md)
for engine requirements.

Start and initialize the vector database, then ingest and ask:

```bash
arkai vectordb start
arkai vectordb initdb manuals
arkai rag ingest manuals docs/guide.pdf
arkai inference start
arkai rag ask manuals "What does the guide recommend?"
```

Ingesting into a collection that does not exist creates it, using the
embedding dimension of the configured embedding model (Qdrant needs the
dimension at creation time; ChromaDB infers it from the first ingest).

`rag ask` uses the configured embedding model and inference endpoint,
prints an answer plus numbered source passages, and does not start or
stop either service. Model-generated citations should be checked against
the listed sources. If an embedding model is not cached, Arkai warns
before downloading it.

`arkai rag query` performs retrieval without generation. `rag ask`
supports `--results`, `--max-context-tokens`, `--max-output-tokens`,
`--show-context`, `--inference-port`, and `--reasoning-effort`. Context
budgeting is approximate, not a strict model-token guarantee. See `arkai
rag --help` for exact options.

RAG ingestion uses the configured `rag.embedding_model` for both
ingestion and queries. A question with no retrieved passages does not
send a request to the inference server. For smaller context-window
models, reduce the output budget or context budget explicitly.

The RAG and vector database commands are still marked as not fully
functional in the implementation. Consult the [RAG
design](../design/rag.md) and [vector database
design](../design/vector-database.md) for current boundaries.
