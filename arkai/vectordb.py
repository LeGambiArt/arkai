"""Vector database service lifecycle management."""

import os
import signal
import subprocess
import time
from typing import Optional

import requests

from arkai import config, utils


def get_vectordb_pid_path() -> str:
    """Return path to vectordb server PID file."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "vectordb.pid")


def get_vectordb_state_path() -> str:
    """Return path to vectordb server state file (stores startup config)."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "vectordb.state")


def is_vectordb_running() -> bool:
    """Check if vectordb server is running."""
    pid_path = get_vectordb_pid_path()
    pid = utils.read_pid(pid_path)
    if pid is None:
        return False

    try:
        code, _, _ = utils.run_command(["kill", "-0", str(pid)])
        return code == 0
    except RuntimeError:
        return False


def get_vectordb_port() -> Optional[int]:
    """Get port of running vectordb server from state file."""
    state_path = get_vectordb_state_path()
    if not os.path.exists(state_path):
        return None
    try:
        state = utils.load_yaml(state_path)
        return state.get("port") if isinstance(state, dict) else None
    except Exception:
        return None


def cmd_vectordb_start(port: Optional[int] = None) -> None:
    """Start vectordb server (chromadb).

    Args:
        port: Override port from config
    """
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

    vectordb_path = config.get_config_value(cfg, "vectordb.path", "chroma")
    resolved_path = utils.resolve_binary(vectordb_path)

    database_dir = config.get_config_value(cfg, "vectordb.database_dir")
    if database_dir is None:
        data_home = utils.get_data_home()
        if data_home is None:
            raise RuntimeError("DATA_HOME not available")
        database_dir = os.path.join(data_home, "chromadb")

    database_dir = os.path.expanduser(database_dir)
    os.makedirs(database_dir, exist_ok=True)

    utils.info(f"Starting vectordb server on port {vectordb_port}...")

    # Disable SIGINT to increase chance of process and pid file are both created
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    process = subprocess.Popen(
        [
            resolved_path,
            "run",
            "--port",
            str(vectordb_port),
            "--host",
            "127.0.0.1",
            "--path",
            database_dir,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pid_path = get_vectordb_pid_path()
    utils.write_pid(pid_path, process.pid)

    # Restore SIGINT
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    utils.info("Waiting for vectordb to be ready...")
    for i in range(30):
        try:
            response = requests.get(
                f"http://127.0.0.1:{vectordb_port}/api/v2/heartbeat",
                timeout=1,
            )
            if response.status_code == 200:
                state_path = get_vectordb_state_path()
                state = {
                    "port": vectordb_port,
                    "vendor": "chromadb",
                    "pid": process.pid,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "databases": [],
                }
                utils.save_yaml(state_path, state)
                utils.info(f"Vectordb server started on port {vectordb_port}")
                return
        except (requests.RequestException, Exception):
            pass
        time.sleep(1)

    # Server failed to start—kill process and clean up files
    try:
        os.kill(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    if os.path.exists(pid_path):
        os.remove(pid_path)
    state_path = get_vectordb_state_path()
    if os.path.exists(state_path):
        os.remove(state_path)
    raise RuntimeError("Vectordb server failed to start (timeout)")


def cmd_vectordb_stop() -> None:
    """Stop running vectordb server gracefully."""
    pid_path = get_vectordb_pid_path()
    pid = utils.read_pid(pid_path)

    if pid is None:
        utils.info("Vectordb server is not running")
        return

    utils.info(f"Stopping vectordb server (PID: {pid})...")

    stopped = False
    try:
        utils.kill_process(pid)
        stopped = utils.wait_for_process_stop(pid, timeout_secs=10.0)
        if stopped:
            utils.info("Vectordb server stopped")
        else:
            utils.warn("Vectordb server did not stop cleanly")
    except RuntimeError as e:
        utils.warn(f"Error stopping vectordb: {e}")

    if stopped:
        if os.path.exists(pid_path):
            os.remove(pid_path)
        if os.path.exists(get_vectordb_state_path()):
            os.remove(get_vectordb_state_path())


def cmd_vectordb_status() -> None:
    """Show vectordb server status."""
    if not is_vectordb_running():
        utils.info("Vectordb server is not running")
        return

    pid_path = get_vectordb_pid_path()
    state_path = get_vectordb_state_path()
    pid = utils.read_pid(pid_path)

    if pid is None:
        utils.info("Vectordb server is not running")
        return

    try:
        state = utils.load_yaml(state_path) or {}
    except Exception:
        state = {}

    port = state.get("port", "unknown")
    vendor = state.get("vendor", "unknown")
    started_at = state.get("started_at", "unknown")

    utils.info("Vectordb server status:")
    utils.info(f"  PID: {pid}")
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

    tenant = "default"
    database = "default"

    utils.info(f"Dropping database '{db_name}'...")

    try:
        api_url = (
            f"http://127.0.0.1:{port}/api/v2/tenants/{tenant}/"
            f"databases/{database}/collections/{db_name}"
        )

        response = requests.delete(api_url, timeout=5)

        if response.status_code == 200:
            utils.info(f"Database '{db_name}' dropped successfully")
        else:
            error_msg = response.text.strip() if response.text else "(no response body)"
            raise RuntimeError(
                f"Failed to drop database: HTTP {response.status_code} - {error_msg}"
            )
    except requests.RequestException as e:
        raise RuntimeError(f"Error communicating with vectordb server: {e}")


def cmd_vectordb_list() -> None:
    """List all databases in vectordb.

    Displays all available collections in the vectordb server.
    """
    if not is_vectordb_running():
        raise RuntimeError("Vectordb server is not running. Start with 'arkai vectordb start'")

    port = get_vectordb_port()
    if port is None:
        raise RuntimeError("Could not determine vectordb port")

    tenant = "default"
    database = "default"

    try:
        api_url = (
            f"http://127.0.0.1:{port}/api/v2/tenants/{tenant}/databases/{database}/collections"
        )

        response = requests.get(api_url, timeout=5)

        if response.status_code != 200:
            error_msg = response.text.strip() if response.text else "(no response body)"
            raise RuntimeError(
                f"Failed to list databases: HTTP {response.status_code} - {error_msg}"
            )

        try:
            collections = response.json()
        except requests.exceptions.JSONDecodeError:
            raise RuntimeError("Invalid response from vectordb")

        if not isinstance(collections, list):
            raise RuntimeError("Unexpected response format from vectordb")

        if not collections:
            utils.info("No databases found")
            return

        utils.info(f"Available databases ({len(collections)}):")
        utils.info("")
        for idx, collection in enumerate(collections, 1):
            if isinstance(collection, dict):
                name = collection.get("name", "unknown")
                coll_id = collection.get("id", "unknown")
                utils.info(f"  [{idx}] {name}")
                utils.info(f"      ID: {coll_id}")
            else:
                utils.info(f"  [{idx}] {collection}")
        utils.info("")

    except requests.RequestException as e:
        raise RuntimeError(f"Error communicating with vectordb server: {e}")


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

    try:
        tenant = "default"
        database = "default"

        _ensure_database_exists(port, tenant, database)

        api_url = (
            f"http://127.0.0.1:{port}/api/v2/tenants/{tenant}/databases/{database}/collections"
        )
        payload = {"name": db_name}

        response = requests.post(api_url, json=payload, timeout=5)

        if response.status_code == 200:
            utils.info(f"Database '{db_name}' created successfully")
        elif response.status_code == 409:
            utils.info(f"Database '{db_name}' already exists")
        else:
            error_msg = response.text.strip() if response.text else "(no response body)"
            raise RuntimeError(
                f"Failed to create database: HTTP {response.status_code} - {error_msg}"
            )
    except requests.RequestException as e:
        raise RuntimeError(f"Error communicating with vectordb server: {e}")


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


def _get_collection_id_v2(
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
        except requests.exceptions.JSONDecodeError:
            return None

        if isinstance(collections, list):
            for collection in collections:
                if isinstance(collection, dict) and collection.get("name") == collection_name:
                    return collection.get("id")

        return None
    except requests.RequestException:
        return None


def _ensure_database_exists(port: int, tenant: str, database: str) -> None:
    """Ensure database exists, create if needed.

    Args:
        port: Port of vectordb server
        tenant: Tenant name
        database: Database name
    """
    api_url = f"http://127.0.0.1:{port}/api/v2/tenants/{tenant}/databases"
    payload = {"name": database}

    try:
        response = requests.post(api_url, json=payload, timeout=5)
        if response.status_code not in (200, 409):
            error_msg = response.text.strip() if response.text else "(no response body)"
            status = response.status_code
            utils.warn(f"Warning: could not ensure database exists: HTTP {status} - {error_msg}")
    except requests.RequestException as e:
        utils.warn(f"Warning: error ensuring database exists: {e}")
