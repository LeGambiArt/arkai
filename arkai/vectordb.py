"""Vector database service lifecycle management.

The vectordb server runs as a container under podman, Apple's container
runtime, or docker. Supported vendors are Qdrant (default) and ChromaDB;
both are driven through their HTTP APIs.
"""

import argparse
import os
import time
import uuid

import requests

from arkai import config, container_engine, utils

CONTAINER_NAME = "arkai-vectordb"

VENDOR_IMAGE_DEFAULTS = {
    "qdrant": "docker.io/qdrant/qdrant:latest",
    "chromadb": "docker.io/chromadb/chroma:latest",
}

# Port each vendor image listens on inside the container.
VENDOR_CONTAINER_PORTS = {"qdrant": 6333, "chromadb": 8000}

# Path inside the container where the vendor persists its data.
VENDOR_DATA_PATHS = {"qdrant": "/qdrant/storage", "chromadb": "/chroma/chroma"}

# HTTP endpoint that reports readiness once the server is up.
VENDOR_HEALTH_PATHS = {"qdrant": "/healthz", "chromadb": "/api/v2/heartbeat"}

# Environment variables required by each vendor image.
VENDOR_ENV = {"qdrant": {}, "chromadb": {"IS_PERSISTENT": "TRUE"}}

_CHROMA_API = "/api/v2/tenants/default/databases/default"


def exec_cmd(args: argparse.Namespace) -> None:
    """Select 'vectordb' command to execute."""
    match args.vectordb_cmd:
        case "start":
            cmd_vectordb_start(args.port)
        case "stop":
            cmd_vectordb_stop()
        case "status":
            cmd_vectordb_status()
        case "list":
            cmd_vectordb_list()
        case "initdb":
            cmd_vectordb_initdb(args.db_name)
        case "drop":
            cmd_vectordb_drop(args.db_name)


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    vectordb_parser = subparsers.add_parser("vectordb", help="Manage vector database server")
    vectordb_subparsers = vectordb_parser.add_subparsers(dest="vectordb_cmd", required=True)
    vectordb_start_parser = vectordb_subparsers.add_parser("start", help="Start vectordb server")
    vectordb_start_parser.add_argument("--port", type=int, help="Override port from config")
    vectordb_subparsers.add_parser("stop", help="Stop vectordb server")
    vectordb_subparsers.add_parser("status", help="Show vectordb server status")
    vectordb_subparsers.add_parser("list", help="List databases")
    vectordb_initdb_parser = vectordb_subparsers.add_parser(
        "initdb", help="Initialize new database"
    )
    vectordb_initdb_parser.add_argument("db_name", help="Database name")
    vectordb_drop_parser = vectordb_subparsers.add_parser("drop", help="Drop database")
    vectordb_drop_parser.add_argument("db_name", help="Database name")


def get_vectordb_pid_path() -> str:
    """Return path to the legacy vectordb server PID file (pre-container versions)."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "vectordb.pid")


def get_vectordb_state_path() -> str:
    """Return path to vectordb server state file (stores startup config)."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "vectordb.state")


def _read_state() -> dict:
    """Load the vectordb state file, returning {} when absent or invalid."""
    state_path = get_vectordb_state_path()
    if not os.path.exists(state_path):
        return {}
    try:
        state = utils.load_yaml(state_path)
    except Exception:
        return {}
    return state if isinstance(state, dict) else {}


def _vendor() -> str:
    """Return the vendor of the running (or configured) vectordb server."""
    return _read_state().get("vendor", "qdrant")


def _engine() -> str:
    """Return the engine of the running vectordb container."""
    engine = _read_state().get("engine")
    if engine in container_engine.ENGINES:
        return engine
    return container_engine.resolve_engine()


def is_vectordb_running() -> bool:
    """Check if vectordb server is running."""
    if not _read_state():
        return False
    return container_engine.container_running(_engine(), CONTAINER_NAME)


def get_vectordb_port() -> int | None:
    """Get port of running vectordb server from state file."""
    return _read_state().get("port")


def cmd_vectordb_start(port: int | None = None) -> None:
    """Start vectordb server as a container.

    Args:
        port: Override port from config
    """
    utils.warn("Vectordb commands are not fully functional yet.")
    cfg = config.load_config()

    if port is not None:
        cfg["vectordb"]["port"] = port

    if not config.validate_config(cfg):
        raise RuntimeError("Invalid configuration")

    if is_vectordb_running():
        utils.info("Vectordb server already running")
        return

    vectordb_port = config.get_config_value(cfg, "vectordb.port", 8082)
    if utils.is_port_in_use(vectordb_port):
        raise RuntimeError(f"Port {vectordb_port} already in use")

    vendor = config.get_config_value(cfg, "vectordb.vendor", "qdrant")
    engine = container_engine.resolve_engine(cfg)
    container_engine.ensure_engine_ready(engine)

    image = config.get_config_value(cfg, "vectordb.image") or VENDOR_IMAGE_DEFAULTS[vendor]

    database_dir = config.get_config_value(cfg, "vectordb.database_dir")
    if database_dir is None:
        data_home = utils.get_data_home()
        if data_home is None:
            raise RuntimeError("DATA_HOME not available")
        database_dir = os.path.join(data_home, vendor)

    database_dir = os.path.expanduser(database_dir)
    os.makedirs(database_dir, exist_ok=True)

    utils.info(f"Starting vectordb server ({vendor}) on port {vectordb_port}...")
    container_engine.run_container(
        engine,
        image=image,
        container_name=CONTAINER_NAME,
        port=VENDOR_CONTAINER_PORTS[vendor],
        host_port=vectordb_port,
        volumes=[(database_dir, VENDOR_DATA_PATHS[vendor])],
        env=VENDOR_ENV[vendor],
    )

    utils.info("Waiting for vectordb to be ready...")
    health_url = f"http://127.0.0.1:{vectordb_port}{VENDOR_HEALTH_PATHS[vendor]}"
    for _ in range(30):
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                state = {
                    "port": vectordb_port,
                    "vendor": vendor,
                    "engine": engine,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "databases": [],
                }
                utils.save_yaml(get_vectordb_state_path(), state)
                utils.info(
                    f"Vectordb server started on port {vectordb_port} ({vendor} via {engine})"
                )
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    # Server failed to become ready — remove the container and clean up state
    container_engine.stop_container(engine, CONTAINER_NAME)
    _remove_state_files()
    raise RuntimeError(
        f"Vectordb server failed to start (timeout); {container_engine.engine_logs_hint(engine)}"
    )


def _remove_state_files() -> None:
    """Remove state files, including stale ones from pre-container versions."""
    for path in (get_vectordb_pid_path(), get_vectordb_state_path()):
        if os.path.exists(path):
            os.remove(path)


def cmd_vectordb_stop() -> None:
    """Stop running vectordb server gracefully."""
    if not is_vectordb_running():
        _remove_state_files()
        utils.info("Vectordb server is not running")
        return

    engine = _engine()
    utils.info(f"Stopping vectordb container '{CONTAINER_NAME}' ({engine})...")
    container_engine.stop_container(engine, CONTAINER_NAME)
    _remove_state_files()
    utils.info("Vectordb server stopped")


def cmd_vectordb_status() -> None:
    """Show vectordb server status."""
    if not is_vectordb_running():
        utils.info("Vectordb server is not running")
        return

    state = _read_state()
    port = state.get("port", "unknown")
    vendor = state.get("vendor", "unknown")
    engine = state.get("engine", "unknown")
    started_at = state.get("started_at", "unknown")

    utils.info("Vectordb server status:")
    utils.info(f"  Container: {CONTAINER_NAME} (engine: {engine})")
    utils.info(f"  Port: {port}")
    utils.info(f"  Vendor: {vendor}")
    utils.info(f"  Started: {started_at}")

    if isinstance(state.get("databases"), list):
        databases = state["databases"]
        if databases:
            utils.info(f"  Databases: {', '.join(databases)}")
        else:
            utils.info("  Databases: (none)")


def cmd_vectordb_drop(db_name: str) -> None:
    """Drop (delete) a database from vectordb.

    Args:
        db_name: Name of database to drop
    """
    if not is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    if not _is_valid_collection_name(db_name):
        raise RuntimeError(
            f"Database name must contain only alphanumeric characters, "
            f"dots, underscores, and hyphens, and be 3-512 characters long, got: {db_name}"
        )

    port = get_vectordb_port()
    if port is None:
        raise RuntimeError("Could not determine vectordb port")

    utils.info(f"Dropping database '{db_name}'...")
    drop_collection(port, db_name)
    utils.info(f"Database '{db_name}' dropped successfully")


def cmd_vectordb_list() -> None:
    """List all databases in vectordb.

    Displays all available collections in the vectordb server.
    """
    if not is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    port = get_vectordb_port()
    if port is None:
        raise RuntimeError("Could not determine vectordb port")

    collections = list_collections(port)
    if not collections:
        utils.info("No databases found")
        return

    utils.info(f"Available databases ({len(collections)}):")
    utils.info("")
    for idx, collection in enumerate(collections, 1):
        name = collection.get("name", "unknown")
        coll_id = collection.get("id", "unknown")
        utils.info(f"  [{idx}] {name}")
        utils.info(f"      ID: {coll_id}")
    utils.info("")


def cmd_vectordb_initdb(db_name: str) -> None:
    """Initialize new database in vectordb.

    Args:
        db_name: Name of database to create
    """
    if not is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    if not _is_valid_collection_name(db_name):
        raise RuntimeError(
            f"Database name must contain only alphanumeric characters, "
            f"dots, underscores, and hyphens, and be 3-512 characters long, got: {db_name}"
        )

    port = get_vectordb_port()
    if port is None:
        raise RuntimeError("Could not determine vectordb port")

    utils.info(f"Initializing database '{db_name}' in vectordb...")

    if any(c.get("name") == db_name for c in list_collections(port)):
        utils.info(f"Database '{db_name}' already exists")
        return

    create_collection(port, db_name)
    utils.info(f"Database '{db_name}' created successfully")


def _is_valid_collection_name(name: str) -> bool:
    """Check if collection name is valid per Chromadb V2 API.

    Valid names: 3-512 characters from [a-zA-Z0-9._-], starting/ending with [a-zA-Z0-9].
    """
    if not name or len(name) < 3 or len(name) > 512:
        return False
    if not name[0].isalnum() or not name[-1].isalnum():
        return False
    for char in name:
        if not (char.isalnum() or char in "._-"):
            return False
    return True


def _api_request(method: str, url: str, **kwargs) -> requests.Response:
    """Issue a vectordb HTTP request, translating transport errors."""
    verbs = {
        "GET": requests.get,
        "POST": requests.post,
        "PUT": requests.put,
        "DELETE": requests.delete,
    }
    try:
        return verbs[method](url, timeout=5, **kwargs)
    except requests.RequestException as e:
        raise RuntimeError(f"Error communicating with vectordb server: {e}") from e


def _raise_http_error(action: str, response: requests.Response) -> None:
    """Raise a RuntimeError carrying the vectordb HTTP failure details."""
    error_msg = response.text.strip() if response.text else "(no response body)"
    raise RuntimeError(f"Failed to {action}: HTTP {response.status_code} - {error_msg}")


def _chroma_collection_id(port: int, collection_name: str) -> str:
    """Look up a ChromaDB collection UUID by name."""
    for collection in list_collections(port):
        if collection.get("name") == collection_name:
            coll_id = collection.get("id")
            if coll_id:
                return coll_id
    raise RuntimeError(f"Collection '{collection_name}' not found")


def _default_vector_size() -> int:
    """Determine the embedding dimension of the configured embedding model."""
    from arkai import rag  # local import: rag imports vectordb at module level

    cfg = config.load_config()
    model_name = config.get_config_value(cfg, "rag.embedding_model")
    try:
        model_instance = rag._load_embedding_model(model_name, quiet=True)
        # get_sentence_embedding_dimension was renamed in sentence-transformers v5
        get_dimension = (
            getattr(model_instance, "get_embedding_dimension", None)
            or model_instance.get_sentence_embedding_dimension
        )
        size = get_dimension()
    except Exception as e:
        raise RuntimeError(
            "Cannot determine vector size for qdrant collection; set "
            "rag.embedding_model or ingest a document first"
        ) from e
    if not size:
        raise RuntimeError(
            "Cannot determine vector size for qdrant collection; set "
            "rag.embedding_model or ingest a document first"
        )
    return size


def list_collections(port: int) -> list[dict]:
    """List collections in the running vectordb, regardless of vendor.

    Args:
        port: Port of vectordb server

    Returns:
        List of {"name": ..., "id": ...} entries (for Qdrant, id equals name).
    """
    if _vendor() == "chromadb":
        response = _api_request("GET", f"http://127.0.0.1:{port}{_CHROMA_API}/collections")
        if response.status_code != 200:
            _raise_http_error("list collections", response)
        try:
            collections = response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise RuntimeError("Invalid response from vectordb") from e
        if not isinstance(collections, list):
            raise RuntimeError("Unexpected response format from vectordb")
        return [c for c in collections if isinstance(c, dict)]

    response = _api_request("GET", f"http://127.0.0.1:{port}/collections")
    if response.status_code != 200:
        _raise_http_error("list collections", response)
    try:
        payload = response.json()
        entries = payload["result"]["collections"]
    except (requests.exceptions.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError("Invalid response from vectordb") from e
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected response format from vectordb")
    return [
        {"name": entry.get("name"), "id": entry.get("name")}
        for entry in entries
        if isinstance(entry, dict)
    ]


def _ensure_chroma_database(port: int) -> None:
    """Ensure ChromaDB's 'default' database exists, creating it if needed.

    ChromaDB refuses collection creation when the tenant database does
    not exist yet; creation is idempotent (409 when already present).
    """
    response = _api_request(
        "POST",
        f"http://127.0.0.1:{port}/api/v2/tenants/default/databases",
        json={"name": "default"},
    )
    if response.status_code not in (200, 201, 409):
        _raise_http_error("ensure default database", response)


def create_collection(port: int, name: str, vector_size: int | None = None) -> None:
    """Create a collection in the running vectordb.

    Args:
        port: Port of vectordb server
        name: Name of the collection to create
        vector_size: Embedding dimension. Required by Qdrant at creation
            time; when None, the configured embedding model's dimension is
            used.

    Raises:
        RuntimeError: If creation fails or the vector size cannot be determined.
    """
    if _vendor() == "chromadb":
        _ensure_chroma_database(port)
        response = _api_request(
            "POST",
            f"http://127.0.0.1:{port}{_CHROMA_API}/collections",
            json={"name": name},
        )
        if response.status_code not in (200, 201):
            _raise_http_error(f"create collection '{name}'", response)
        return

    if vector_size is None:
        vector_size = _default_vector_size()
    response = _api_request(
        "PUT",
        f"http://127.0.0.1:{port}/collections/{name}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
    )
    if response.status_code not in (200, 201):
        _raise_http_error(f"create collection '{name}'", response)


def drop_collection(port: int, name: str) -> None:
    """Drop a collection from the running vectordb.

    Args:
        port: Port of vectordb server
        name: Name of the collection to drop
    """
    if _vendor() == "chromadb":
        url = f"http://127.0.0.1:{port}{_CHROMA_API}/collections/{name}"
    else:
        url = f"http://127.0.0.1:{port}/collections/{name}"
    response = _api_request("DELETE", url)
    if response.status_code not in (200, 202, 204):
        _raise_http_error(f"drop collection '{name}'", response)


def add_documents(
    port: int,
    collection: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """Add documents with embeddings to a collection.

    Args:
        port: Port of vectordb server
        collection: Name of the target collection
        ids: Document IDs (mapped to deterministic UUIDs for Qdrant, which
            only accepts unsigned integers or UUIDs as point IDs)
        documents: Document texts
        embeddings: Embedding vectors, one per document
        metadatas: Metadata dicts, one per document
    """
    if _vendor() == "chromadb":
        collection_id = _chroma_collection_id(port, collection)
        response = _api_request(
            "POST",
            f"http://127.0.0.1:{port}{_CHROMA_API}/collections/{collection_id}/add",
            json={
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            },
        )
        if response.status_code not in (200, 201):
            _raise_http_error(f"add documents to '{collection}'", response)
        return

    points = [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"arkai:{collection}:{point_id}")),
            "vector": embedding,
            "payload": {"document": document, **(metadata or {})},
        }
        for point_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas)
    ]
    response = _api_request(
        "PUT",
        f"http://127.0.0.1:{port}/collections/{collection}/points",
        json={"points": points},
    )
    if response.status_code not in (200, 201):
        _raise_http_error(f"add documents to '{collection}'", response)


def search_documents(
    port: int, collection: str, query_embedding: list[float], k: int
) -> list[tuple[str, dict, float]]:
    """Search a collection for the k nearest documents to an embedding.

    Args:
        port: Port of vectordb server
        collection: Name of the collection to search
        query_embedding: Query vector
        k: Number of results to return

    Returns:
        List of (document, metadata, distance) tuples, nearest first.
        Distances are Chroma-style (lower is better); Qdrant cosine scores
        are converted with `1 - score`.
    """
    if _vendor() == "chromadb":
        collection_id = _chroma_collection_id(port, collection)
        response = _api_request(
            "POST",
            f"http://127.0.0.1:{port}{_CHROMA_API}/collections/{collection_id}/query",
            json={
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            },
        )
        if response.status_code != 200:
            _raise_http_error(f"search collection '{collection}'", response)
        try:
            data = response.json()
            documents = data["documents"][0]
            metadatas = data["metadatas"][0]
            distances = data["distances"][0]
            if (
                not isinstance(documents, list)
                or not isinstance(metadatas, list)
                or not isinstance(distances, list)
            ):
                raise TypeError("invalid result lists")
            if len(documents) != len(metadatas) or len(documents) != len(distances):
                raise ValueError("mismatched result lengths")
            return [
                (doc, meta if isinstance(meta, dict) else {}, distance)
                for doc, meta, distance in zip(documents, metadatas, distances)
                if isinstance(doc, str) and doc.strip()
            ]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RuntimeError("Invalid response from vectordb") from e

    response = _api_request(
        "POST",
        f"http://127.0.0.1:{port}/collections/{collection}/points/search",
        json={"vector": query_embedding, "limit": k, "with_payload": True},
    )
    if response.status_code != 200:
        _raise_http_error(f"search collection '{collection}'", response)
    results: list[tuple[str, dict, float]] = []
    try:
        for point in response.json()["result"]:
            if not isinstance(point, dict):
                continue
            payload = point.get("payload") or {}
            document = payload.get("document")
            if not isinstance(document, str) or not document.strip():
                continue
            metadata = {key: value for key, value in payload.items() if key != "document"}
            results.append((document, metadata, 1.0 - point["score"]))
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise RuntimeError("Invalid response from vectordb") from e
    return results
