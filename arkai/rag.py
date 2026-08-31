"""RAG (Retrieval-Augmented Generation) operations."""

import json
import os
from typing import Optional

import requests

from arkai import config, document_processor, model, utils, vectordb

# Configure sentence-transformers to cache models in arkai's models directory
# This must be set before importing SentenceTransformer
try:
    _models_dir = model.get_models_dir()
    _embeddings_dir = os.path.join(_models_dir, "embeddings")
    os.makedirs(_embeddings_dir, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = _embeddings_dir
except Exception:
    # If model dir setup fails, fall back to default cache
    pass


def cmd_rag_ingest(db_name: str, file_path: str, chunk_size: Optional[int] = None) -> None:
    """Ingest document into RAG vectordb.

    Args:
        db_name: Name of collection to ingest into
        file_path: Path to document file
        chunk_size: Override chunk size from config
    """
    if not os.path.exists(file_path):
        raise RuntimeError(f"File not found: {file_path}")

    if not vectordb.is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    cfg = config.load_config()
    if chunk_size is not None:
        cfg["rag"]["chunk_size"] = chunk_size

    if not config.validate_config(cfg):
        raise RuntimeError("Invalid configuration")

    format = document_processor.detect_format(file_path)
    if not format:
        raise RuntimeError(f"Unsupported file format for: {file_path}")

    processor = document_processor.get_processor(format)
    if processor is None:
        raise RuntimeError(f"No processor available for format: {format}")

    rag_chunk_size = config.get_config_value(cfg, "rag.chunk_size", 512)
    rag_chunk_overlap = config.get_config_value(cfg, "rag.chunk_overlap", 50)

    utils.info(f"Processing document: {file_path} (format: {format})")
    chunks = processor.process(file_path, rag_chunk_size, rag_chunk_overlap)
    utils.info(f"Generated {len(chunks)} chunks")

    vectordb_port = vectordb.get_vectordb_port()
    if vectordb_port is None:
        raise RuntimeError("Could not determine vectordb port")

    tenant = "default"
    database = "default"

    collection_id = _get_collection_id(vectordb_port, tenant, database, db_name)
    if collection_id is None:
        raise RuntimeError(f"Collection '{db_name}' not found")

    utils.info("Loading embedding model...")
    embedding_model = config.get_config_value(cfg, "rag.embedding_model")
    embedding_model_instance = _load_embedding_model(embedding_model)

    utils.info(f"Storing {len(chunks)} chunks in collection '{db_name}'...")

    total_words = 0
    for chunk_idx, chunk in enumerate(chunks):
        total_words += chunk.metadata.get("word_count", 0)

        chunk_id = f"{db_name}-{chunk_idx}"
        api_url = (
            f"http://127.0.0.1:{vectordb_port}/api/v2/tenants/{tenant}/"
            f"databases/{database}/collections/{collection_id}/add"
        )

        embedding = _embed_text(chunk.content, embedding_model_instance)

        payload = {
            "ids": [chunk_id],
            "documents": [chunk.content],
            "embeddings": [embedding],
            "metadatas": [chunk.metadata],
        }

        try:
            response = requests.post(api_url, json=payload, timeout=5)
            if response.status_code not in (200, 201):
                utils.warn(f"Failed to store chunk {chunk_idx}: {response.text}")
        except requests.RequestException as e:
            utils.warn(f"Error storing chunk {chunk_idx}: {e}")

        # Progress indicator with ANSI escape codes for in-place update
        progress = (chunk_idx + 1) / len(chunks)
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "─" * (bar_width - filled)
        percentage = int(progress * 100)
        print(
            f"\r[{bar}] {chunk_idx + 1}/{len(chunks)} ({percentage}%)",
            end="",
            flush=True,
        )

    print()  # Newline after progress bar complete
    utils.info(f"Ingestion complete: {len(chunks)} chunks, {total_words} words")


def cmd_rag_search(db_name: str, query: str, k: Optional[int] = None) -> None:
    """Search RAG vectordb for similar documents.

    Args:
        db_name: Name of collection to search
        query: Search query text
        k: Number of results to return
    """
    if not vectordb.is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    cfg = config.load_config()
    if not config.validate_config(cfg):
        raise RuntimeError("Invalid configuration")

    if k is None:
        k = config.get_config_value(cfg, "rag.max_search_results", 10)

    vectordb_port = vectordb.get_vectordb_port()
    if vectordb_port is None:
        raise RuntimeError("Could not determine vectordb port")

    tenant = "default"
    database = "default"

    collection_id = _get_collection_id(vectordb_port, tenant, database, db_name)
    if collection_id is None:
        raise RuntimeError(f"Collection '{db_name}' not found")

    utils.info("Loading embedding model...")
    embedding_model = config.get_config_value(cfg, "rag.embedding_model")
    embedding_model_instance = _load_embedding_model(embedding_model)
    query_embedding = _embed_text(query, embedding_model_instance)

    utils.info(f"Searching collection '{db_name}' for: {query}")

    api_url = (
        f"http://127.0.0.1:{vectordb_port}/api/v2/tenants/{tenant}/"
        f"databases/{database}/collections/{collection_id}/query"
    )
    payload = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }

    try:
        response = requests.post(api_url, json=payload, timeout=5)

        if response.status_code != 200:
            raise RuntimeError(f"Search failed: {response.text}")

        try:
            results = response.json()
        except json.JSONDecodeError:
            raise RuntimeError("Invalid response from vectordb")

        if not results or not results.get("documents"):
            utils.info("No results found")
            return

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        utils.info(f"Found {len(documents)} results:")
        utils.info("")

        for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
            score = 1.0 - (distance / 2.0) if distance else 0.0
            score = max(0.0, min(1.0, score))

            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            source = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"

            utils.info(f"  [{idx + 1}] Score: {score:.2f} | Source: {source}")
            utils.info(f"      {snippet}")
            utils.info("")

    except requests.RequestException as e:
        raise RuntimeError(f"Search error: {e}")


def _get_collection_id(
    port: int, tenant: str, database: str, collection_name: str
) -> Optional[str]:
    """Get collection ID by name from Chromadb V2 API.

    Args:
        port: Port of vectordb server
        tenant: Tenant name
        database: Database name
        collection_name: Name of collection to find

    Returns:
        Collection UUID or None if not found
    """
    api_url = f"http://127.0.0.1:{port}/api/v2/tenants/{tenant}/databases/{database}/collections"

    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code != 200:
            return None

        try:
            collections = response.json()
        except json.JSONDecodeError:
            return None

        if isinstance(collections, list):
            for collection in collections:
                if isinstance(collection, dict) and collection.get("name") == collection_name:
                    return collection.get("id")

        return None
    except requests.RequestException:
        return None


def _load_embedding_model(model_name: Optional[str] = None):
    """Load embedding model from sentence-transformers.

    Models are cached in ~/.local/share/arkai/models/embeddings/ for centralized management.
    Uses local cache only to avoid unnecessary downloads.

    Args:
        model_name: Model name to load. Defaults to 'all-MiniLM-L6-v2'

    Returns:
        Loaded SentenceTransformer model instance
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is required for embeddings. "
            "Install with: pip install sentence-transformers"
        )

    if model_name is None:
        model_name = "all-MiniLM-L6-v2"

    try:
        embeddings_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        if not embeddings_dir:
            embeddings_dir = os.path.join(model.get_models_dir(), "embeddings")

        model_instance = SentenceTransformer(
            model_name,
            cache_folder=embeddings_dir,
            local_files_only=True,
        )
        return model_instance
    except Exception:
        try:
            utils.info("Downloading embedding model (first use only)...")
            embeddings_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
            if not embeddings_dir:
                embeddings_dir = os.path.join(model.get_models_dir(), "embeddings")

            model_instance = SentenceTransformer(
                model_name,
                cache_folder=embeddings_dir,
            )
            return model_instance
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")


def _embed_text(text: str, model) -> list[float]:
    """Generate embedding for text using a pre-loaded model.

    Args:
        text: Text to embed
        model: Pre-loaded SentenceTransformer model instance

    Returns:
        Vector of floats representing the embedding
    """
    try:
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")
