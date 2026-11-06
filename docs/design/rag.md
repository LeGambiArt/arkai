# Design: retrieval-augmented generation

## Scope

RAG connects document ingestion, passage retrieval, and local inference.
It uses configured chunking and embedding settings, a running vector
database, and a running OpenAI-compatible inference endpoint. Arkai does
not claim that model-generated answers or citations are authoritative.

## Data flow

1. Ingestion detects a supported document format and dispatches it to a
   registered processor.
2. The processor extracts text and creates overlapping chunks using
   configured chunk size and overlap.
3. Arkai embeds chunks and stores their text and source metadata in the
   chosen Chroma collection.
4. A query embeds the question and retrieves the configured number of
   passages.
5. `rag ask` estimates a passage context budget, sends selected sources
   and the question to the running inference API, and prints the answer
   with numbered source labels.

The ask path does not start or stop the inference or vector database
services. If retrieval finds no passages, inference is not called. Token
budgeting is an estimate rather than a model tokenizer guarantee.

## Status and boundaries

The current CLI emits a warning that non-`ask` RAG operations are not
fully functional. The vector database integration also has known
unfinished commands; check the guides and tests before depending on it in
production.

## Implementation map

- RAG command, retrieval, and prompt assembly: `arkai/rag.py`
- Format processing and chunk creation: `arkai/document_processor.py`
- Embedding/model and inference calls: `arkai/rag.py`
- Unit/BDD coverage: `tests/unit/test_rag.py`,
  `tests/unit/test_document_processor.py`, `features/rag.feature`
- User instructions: [RAG](../guides/rag.md)
