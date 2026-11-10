"""Unit tests for vectordb service."""

from unittest.mock import MagicMock, patch

import pytest

from arkai import vectordb


def _state(vendor="qdrant", engine="podman", port=8082):
    return {
        "port": port,
        "vendor": vendor,
        "engine": engine,
        "started_at": "2026-01-01T00:00:00Z",
        "databases": [],
    }


class TestVectordbPaths:
    @patch("arkai.vectordb.utils")
    def test_get_vectordb_pid_path(self, mock_utils):
        """Test getting legacy vectordb PID path."""
        mock_utils.get_pid_dir.return_value = "/home/user/.local/state/arkai"
        pid_path = vectordb.get_vectordb_pid_path()
        assert pid_path == "/home/user/.local/state/arkai/vectordb.pid"
        assert ".local/state" in pid_path

    @patch("arkai.vectordb.utils")
    def test_get_vectordb_state_path(self, mock_utils):
        """Test getting vectordb state path."""
        mock_utils.get_pid_dir.return_value = "/home/user/.local/state/arkai"
        state_path = vectordb.get_vectordb_state_path()
        assert state_path == "/home/user/.local/state/arkai/vectordb.state"
        assert ".local/state" in state_path


class TestVectordbRunning:
    @patch("arkai.vectordb._read_state", return_value={})
    def test_not_running_without_state(self, _mock_state):
        """Test vectordb not running when state file missing."""
        assert not vectordb.is_vectordb_running()

    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb._read_state", return_value=_state())
    def test_running_when_container_running(self, _mock_state, mock_engine):
        """Test vectordb running when the container is up."""
        mock_engine.ENGINES = ("podman", "apple", "docker")
        mock_engine.container_running.return_value = True
        assert vectordb.is_vectordb_running()
        mock_engine.container_running.assert_called_once_with("podman", "arkai-vectordb")

    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb._read_state", return_value=_state())
    def test_not_running_when_container_stopped(self, _mock_state, mock_engine):
        """Test vectordb not running when the container is gone."""
        mock_engine.ENGINES = ("podman", "apple", "docker")
        mock_engine.container_running.return_value = False
        assert not vectordb.is_vectordb_running()

    @patch("arkai.vectordb.utils")
    @patch("os.path.exists")
    def test_get_vectordb_port(self, mock_exists, mock_utils):
        """Test getting vectordb port from state file."""
        mock_exists.return_value = True
        mock_utils.load_yaml.return_value = _state()
        assert vectordb.get_vectordb_port() == 8082

    @patch("os.path.exists")
    def test_get_vectordb_port_no_state(self, mock_exists):
        """Test getting port when state file missing."""
        mock_exists.return_value = False
        assert vectordb.get_vectordb_port() is None


class TestVectordbStart:
    @patch("arkai.vectordb.is_vectordb_running")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    def test_start_warns_about_incomplete_functionality(
        self, mock_config, mock_utils, mock_is_running
    ):
        """Test starting vectordb displays the incomplete functionality warning."""
        mock_is_running.return_value = True

        vectordb.cmd_vectordb_start()

        mock_utils.warn.assert_called_once_with("Vectordb commands are not fully functional yet.")

    @patch("arkai.vectordb.is_vectordb_running")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    def test_start_already_running(self, mock_config, mock_utils, mock_is_running):
        """Test starting when already running."""
        mock_is_running.return_value = True
        vectordb.cmd_vectordb_start()
        mock_utils.info.assert_called_with("Vectordb server already running")

    @patch("arkai.vectordb.is_vectordb_running")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    def test_start_port_conflict(self, mock_config, mock_utils, mock_is_running):
        """Test starting when port already in use."""
        mock_is_running.return_value = False
        mock_utils.is_port_in_use.return_value = True
        mock_config.load_config.return_value = {
            "vectordb": {"vendor": "qdrant", "port": 8082, "database_dir": None}
        }
        mock_config.validate_config.return_value = True
        mock_config.get_config_value.return_value = 8082

        with pytest.raises(RuntimeError, match="Port 8082 already in use"):
            vectordb.cmd_vectordb_start()

    def _configure(self, mock_config, mock_utils, tmp_path, vendor):
        mock_config.load_config.return_value = {
            "vectordb": {"vendor": vendor, "port": 8082, "database_dir": str(tmp_path)}
        }
        mock_config.validate_config.return_value = True
        values = {
            "vectordb.port": 8082,
            "vectordb.vendor": vendor,
            "vectordb.image": None,
            "vectordb.database_dir": str(tmp_path),
        }
        mock_config.get_config_value.side_effect = lambda cfg, key, default=None: values.get(
            key, default
        )
        mock_utils.is_port_in_use.return_value = False

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb.time")
    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    def test_start_qdrant_container(
        self, _mock_running, mock_config, mock_utils, mock_engine, mock_time, mock_requests
    ):
        """Test starting the default qdrant vendor under a container engine."""
        database_dir = _tmp_path()
        self._configure(mock_config, mock_utils, database_dir, "qdrant")
        mock_engine.resolve_engine.return_value = "podman"
        mock_requests.get.return_value = MagicMock(status_code=200)

        vectordb.cmd_vectordb_start()

        mock_engine.ensure_engine_ready.assert_called_once_with("podman")
        kwargs = mock_engine.run_container.call_args.kwargs
        assert mock_engine.run_container.call_args.args[0] == "podman"
        assert kwargs["image"] == "docker.io/qdrant/qdrant:latest"
        assert kwargs["container_name"] == "arkai-vectordb"
        assert kwargs["port"] == 6333
        assert kwargs["host_port"] == 8082
        assert kwargs["volumes"] == [(str(database_dir), "/qdrant/storage")]
        assert kwargs["env"] == {}
        assert "/healthz" in mock_requests.get.call_args.args[0]
        state = mock_utils.save_yaml.call_args.args[1]
        assert state["vendor"] == "qdrant"
        assert state["engine"] == "podman"
        assert "pid" not in state

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb.time")
    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    def test_start_chromadb_container(
        self, _mock_running, mock_config, mock_utils, mock_engine, mock_time, mock_requests
    ):
        """Test starting the chromadb vendor with its port, volume, and env."""
        database_dir = _tmp_path()
        self._configure(mock_config, mock_utils, database_dir, "chromadb")
        mock_engine.resolve_engine.return_value = "docker"
        mock_requests.get.return_value = MagicMock(status_code=200)

        vectordb.cmd_vectordb_start()

        kwargs = mock_engine.run_container.call_args.kwargs
        assert mock_engine.run_container.call_args.args[0] == "docker"
        assert kwargs["image"] == "docker.io/chromadb/chroma:latest"
        assert kwargs["port"] == 8000
        assert kwargs["volumes"] == [(str(database_dir), "/chroma/chroma")]
        assert kwargs["env"] == {"IS_PERSISTENT": "TRUE"}
        assert "/api/v2/heartbeat" in mock_requests.get.call_args.args[0]

    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    def test_start_propagates_engine_not_ready_error(
        self, _mock_running, mock_config, mock_utils, mock_engine
    ):
        """Test that a stopped engine VM surfaces its actionable error."""
        self._configure(mock_config, mock_utils, _tmp_path(), "qdrant")
        mock_engine.resolve_engine.return_value = "podman"
        mock_engine.ensure_engine_ready.side_effect = RuntimeError(
            "Container engine 'podman' is not ready: start it with 'podman machine start'"
        )

        with pytest.raises(RuntimeError, match="podman machine start"):
            vectordb.cmd_vectordb_start()
        mock_engine.run_container.assert_not_called()

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb.time")
    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.config")
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    def test_start_timeout_stops_container(
        self, _mock_running, mock_config, mock_utils, mock_engine, mock_time, mock_requests
    ):
        """Test that a container that never becomes ready is stopped again."""
        self._configure(mock_config, mock_utils, _tmp_path(), "qdrant")
        mock_engine.resolve_engine.return_value = "podman"
        mock_requests.get.side_effect = vectordb.requests.ConnectionError("refused")

        with pytest.raises(RuntimeError, match="timeout"):
            vectordb.cmd_vectordb_start()

        mock_engine.stop_container.assert_called_once_with("podman", "arkai-vectordb")


def _tmp_path():
    """Return a per-test scratch directory (never touches real DATA_HOME)."""
    import tempfile

    return tempfile.mkdtemp(prefix="arkai-test-vectordb-")


class TestVectordbStop:
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    @patch("arkai.vectordb.utils")
    def test_stop_when_not_running(self, mock_utils, _mock_running):
        """Test stopping when nothing runs reports and cleans stale files."""
        vectordb.cmd_vectordb_stop()
        mock_utils.info.assert_called_with("Vectordb server is not running")

    @patch("os.remove")
    @patch("os.path.exists", return_value=False)
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.container_engine")
    @patch("arkai.vectordb._read_state", return_value=_state())
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_stop_removes_container(
        self, _mock_running, _mock_state, mock_engine, mock_utils, _exists, _rm
    ):
        """Test stopping removes the container and state files."""
        mock_engine.ENGINES = ("podman", "apple", "docker")
        vectordb.cmd_vectordb_stop()
        mock_engine.stop_container.assert_called_once_with("podman", "arkai-vectordb")
        mock_utils.info.assert_called_with("Vectordb server stopped")


class TestVectordbInitdb:
    @patch("arkai.vectordb.is_vectordb_running", return_value=False)
    def test_initdb_not_running(self, _mock_running):
        """Test initdb when vectordb not running."""
        with pytest.raises(RuntimeError, match="not running"):
            vectordb.cmd_vectordb_initdb("test_db")

    @patch("arkai.vectordb.get_vectordb_port", return_value=8082)
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_initdb_invalid_name(self, _mock_running, _mock_port):
        """Test initdb with invalid database name."""
        with pytest.raises(RuntimeError, match="Database name must contain"):
            vectordb.cmd_vectordb_initdb("test-db!")

    @patch("arkai.vectordb.create_collection")
    @patch("arkai.vectordb.list_collections", return_value=[])
    @patch("arkai.vectordb.get_vectordb_port", return_value=8082)
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_initdb_creates_collection(self, _mock_running, _mock_port, mock_list, mock_create):
        """Test initdb creates the collection through the vendor dispatch."""
        vectordb.cmd_vectordb_initdb("test-db")
        mock_create.assert_called_once_with(8082, "test-db")

    @patch("arkai.vectordb.create_collection")
    @patch("arkai.vectordb.list_collections", return_value=[{"name": "test-db", "id": "x"}])
    @patch("arkai.vectordb.get_vectordb_port", return_value=8082)
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_initdb_existing_collection(self, _mock_running, _mock_port, mock_list, mock_create):
        """Test initdb is a no-op when the collection already exists."""
        vectordb.cmd_vectordb_initdb("test-db")
        mock_create.assert_not_called()

    @patch("arkai.vectordb.drop_collection")
    @patch("arkai.vectordb.get_vectordb_port", return_value=8082)
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_drop_deletes_collection(self, _mock_running, _mock_port, mock_drop):
        """Test drop goes through the vendor dispatch."""
        vectordb.cmd_vectordb_drop("test-db")
        mock_drop.assert_called_once_with(8082, "test-db")

    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.list_collections")
    @patch("arkai.vectordb.get_vectordb_port", return_value=8082)
    @patch("arkai.vectordb.is_vectordb_running", return_value=True)
    def test_list_prints_collections(
        self, _mock_running, _mock_port, mock_list, mock_utils, capsys
    ):
        """Test list displays collection names."""
        mock_list.return_value = [{"name": "manuals", "id": "abc"}]
        vectordb.cmd_vectordb_list()
        printed = " ".join(str(c.args[0]) for c in mock_utils.info.call_args_list)
        assert "manuals" in printed
        assert "Available databases (1)" in printed


def _response(status_code=200, json_data=None, text=""):
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = json_data if json_data is not None else {}
    return response


class TestQdrantRest:
    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_list_collections(self, _mock_vendor, mock_requests):
        """Test qdrant collection listing normalizes to name/id entries."""
        mock_requests.get.return_value = _response(
            json_data={"result": {"collections": [{"name": "manuals"}]}}
        )
        assert vectordb.list_collections(8082) == [{"name": "manuals", "id": "manuals"}]
        assert mock_requests.get.call_args.args[0] == "http://127.0.0.1:8082/collections"

    @patch("arkai.vectordb._default_vector_size", return_value=384)
    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_create_collection_with_dimension(self, _mock_vendor, mock_requests, _mock_size):
        """Test qdrant collection creation sends the vector size and distance."""
        mock_requests.put.return_value = _response()
        vectordb.create_collection(8082, "manuals")
        call = mock_requests.put.call_args
        assert call.args[0] == "http://127.0.0.1:8082/collections/manuals"
        assert call.kwargs["json"] == {"vectors": {"size": 384, "distance": "Cosine"}}

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_create_collection_reports_http_error(self, _mock_vendor, mock_requests):
        """Test qdrant collection creation failure mentions the collection."""
        mock_requests.put.return_value = _response(status_code=400, text="bad request")
        with pytest.raises(RuntimeError, match="create collection 'manuals'"):
            vectordb.create_collection(8082, "manuals", vector_size=384)

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_drop_collection(self, _mock_vendor, mock_requests):
        """Test qdrant collection deletion uses DELETE and accepts 200/202/204."""
        for status in (200, 202, 204):
            mock_requests.delete.return_value = _response(status_code=status)
            vectordb.drop_collection(8082, "manuals")
        call = mock_requests.delete.call_args
        assert call.args == ("http://127.0.0.1:8082/collections/manuals",)

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_add_documents_maps_ids_to_uuids(self, _mock_vendor, mock_requests):
        """Test qdrant upsert uses valid UUID point ids and document payloads."""
        mock_requests.put.return_value = _response()
        vectordb.add_documents(
            8082, "manuals", ["manuals-0"], ["hello"], [[0.1, 0.2]], [{"source": "a.md"}]
        )
        call = mock_requests.put.call_args
        assert call.args[0] == "http://127.0.0.1:8082/collections/manuals/points"
        point = call.kwargs["json"]["points"][0]
        import uuid as uuid_module

        uuid_module.UUID(point["id"])  # raises if not a valid UUID
        assert point["vector"] == [0.1, 0.2]
        assert point["payload"] == {"document": "hello", "source": "a.md"}

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_search_documents_converts_scores(self, _mock_vendor, mock_requests):
        """Test qdrant search maps cosine scores to chroma-style distances."""
        mock_requests.post.return_value = _response(
            json_data={
                "result": [
                    {"score": 0.75, "payload": {"document": "hello", "source": "a.md"}},
                    {"score": 0.5, "payload": {"document": "world"}},
                ]
            }
        )
        results = vectordb.search_documents(8082, "manuals", [0.1, 0.2], 2)
        assert results[0] == ("hello", {"source": "a.md"}, pytest.approx(0.25))
        assert results[1] == ("world", {}, pytest.approx(0.5))
        call = mock_requests.post.call_args
        assert call.args[0] == "http://127.0.0.1:8082/collections/manuals/points/search"
        assert call.kwargs["json"] == {"vector": [0.1, 0.2], "limit": 2, "with_payload": True}

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="qdrant")
    def test_search_documents_skips_empty_documents(self, _mock_vendor, mock_requests):
        """Test qdrant search drops points without usable document payloads."""
        mock_requests.post.return_value = _response(
            json_data={"result": [{"score": 0.9, "payload": {"source": "a.md"}}]}
        )
        assert vectordb.search_documents(8082, "manuals", [0.1], 1) == []


class TestChromaRest:
    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_list_collections(self, _mock_vendor, mock_requests):
        """Test chromadb collection listing passes entries through."""
        mock_requests.get.return_value = _response(json_data=[{"name": "manuals", "id": "uuid-1"}])
        assert vectordb.list_collections(8082) == [{"name": "manuals", "id": "uuid-1"}]

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_create_collection_ensures_default_database(self, _mock_vendor, mock_requests):
        """Test chromadb collection creation ensures the default database exists.

        ChromaDB rejects collection creation with HTTP 400 when the
        tenant's 'default' database has not been created yet.
        """
        mock_requests.post.return_value = _response(status_code=201)
        vectordb.create_collection(8082, "test")
        first_call = mock_requests.post.call_args_list[0]
        assert first_call.args[0] == "http://127.0.0.1:8082/api/v2/tenants/default/databases"
        assert first_call.kwargs["json"] == {"name": "default"}

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_create_collection_tolerates_existing_database(self, _mock_vendor, mock_requests):
        """Test chromadb collection creation proceeds when the database exists (409)."""
        mock_requests.post.side_effect = [_response(status_code=409), _response(status_code=201)]
        vectordb.create_collection(8082, "test")
        assert mock_requests.post.call_count == 2

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_add_documents_resolves_collection_id(self, _mock_vendor, mock_requests):
        """Test chromadb add looks up the collection UUID first."""
        mock_requests.get.return_value = _response(json_data=[{"name": "manuals", "id": "uuid-1"}])
        mock_requests.post.return_value = _response(status_code=201)
        vectordb.add_documents(8082, "manuals", ["m-0"], ["hello"], [[0.1]], [{"source": "a.md"}])
        add_call = mock_requests.post.call_args
        assert add_call.args[0] == (
            "http://127.0.0.1:8082/api/v2/tenants/default/databases/default/collections/uuid-1/add"
        )
        assert add_call.kwargs["json"]["ids"] == ["m-0"]

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_add_documents_unknown_collection(self, _mock_vendor, mock_requests):
        """Test chromadb add fails clearly when the collection is missing."""
        mock_requests.get.return_value = _response(json_data=[{"name": "other"}])
        with pytest.raises(RuntimeError, match="Collection 'manuals' not found"):
            vectordb.add_documents(8082, "manuals", ["m-0"], ["hello"], [[0.1]], [{}])

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_search_documents_parses_nested_lists(self, _mock_vendor, mock_requests):
        """Test chromadb search parses the nested documents/metadatas/distances."""
        mock_requests.get.return_value = _response(json_data=[{"name": "manuals", "id": "uuid-1"}])
        mock_requests.post.return_value = _response(
            json_data={
                "documents": [["hello", "world"]],
                "metadatas": [[{"source": "a.md"}, {}]],
                "distances": [[0.1, 0.4]],
            }
        )
        results = vectordb.search_documents(8082, "manuals", [0.1], 2)
        assert results == [("hello", {"source": "a.md"}, 0.1), ("world", {}, 0.4)]

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb._vendor", return_value="chromadb")
    def test_search_documents_rejects_mismatched_lists(self, _mock_vendor, mock_requests):
        """Test chromadb search rejects malformed response payloads."""
        mock_requests.get.return_value = _response(json_data=[{"name": "manuals", "id": "uuid-1"}])
        mock_requests.post.return_value = _response(
            json_data={
                "documents": [["hello"]],
                "metadatas": [[]],
                "distances": [[]],
            }
        )
        with pytest.raises(RuntimeError, match="Invalid response from vectordb"):
            vectordb.search_documents(8082, "manuals", [0.1], 1)
