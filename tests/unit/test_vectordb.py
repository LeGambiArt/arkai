"""Unit tests for vectordb service."""

from unittest.mock import MagicMock, patch

from arkai import vectordb


class TestVectordbPaths:
    def test_get_vectordb_pid_path(self):
        """Test getting vectordb PID path."""
        pid_path = vectordb.get_vectordb_pid_path()
        assert pid_path.endswith("vectordb.pid")
        assert ".local/state" in pid_path

    def test_get_vectordb_state_path(self):
        """Test getting vectordb state path."""
        state_path = vectordb.get_vectordb_state_path()
        assert state_path.endswith("vectordb.state")
        assert ".local/state" in state_path


class TestVectordbStatus:
    @patch("arkai.vectordb.utils")
    def test_is_vectordb_running_no_pid_file(self, mock_utils):
        """Test vectordb not running when PID file missing."""
        mock_utils.read_pid.return_value = None
        assert not vectordb.is_vectordb_running()

    @patch("arkai.vectordb.utils")
    def test_is_vectordb_running_process_exists(self, mock_utils):
        """Test vectordb running when process exists."""
        mock_utils.read_pid.return_value = 12345
        mock_utils.run_command.return_value = (0, "", "")
        assert vectordb.is_vectordb_running()

    @patch("arkai.vectordb.utils")
    def test_is_vectordb_running_process_gone(self, mock_utils):
        """Test vectordb not running when process gone."""
        mock_utils.read_pid.return_value = 12345
        mock_utils.run_command.return_value = (1, "", "")
        assert not vectordb.is_vectordb_running()

    @patch("os.path.exists")
    @patch("arkai.vectordb.utils")
    def test_get_vectordb_port(self, mock_utils, mock_exists):
        """Test getting vectordb port from state file."""
        mock_exists.return_value = True
        mock_utils.load_yaml.return_value = {"port": 8082}
        port = vectordb.get_vectordb_port()
        assert port == 8082

    @patch("os.path.exists")
    @patch("arkai.vectordb.utils")
    def test_get_vectordb_port_no_state(self, mock_utils, mock_exists):
        """Test getting port when state file missing."""
        mock_exists.return_value = False
        port = vectordb.get_vectordb_port()
        assert port is None


class TestVectordbStart:
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
            "vectordb": {"port": 8082, "path": "chroma", "database_dir": None}
        }
        mock_config.validate_config.return_value = True
        mock_config.get_config_value.return_value = 8082

        try:
            vectordb.cmd_vectordb_start()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Port 8082 already in use" in str(e)


class TestVectordbInitdb:
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.get_vectordb_port")
    @patch("arkai.vectordb.is_vectordb_running")
    def test_initdb_not_running(self, mock_is_running, mock_get_port, mock_utils):
        """Test initdb when vectordb not running."""
        mock_is_running.return_value = False

        try:
            vectordb.cmd_vectordb_initdb("test_db")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not running" in str(e)

    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.get_vectordb_port")
    @patch("arkai.vectordb.is_vectordb_running")
    def test_initdb_invalid_name(self, mock_is_running, mock_get_port, mock_utils):
        """Test initdb with invalid database name."""
        mock_is_running.return_value = True
        mock_get_port.return_value = 8082

        try:
            vectordb.cmd_vectordb_initdb("test-db!")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Database name must contain" in str(e)

    @patch("arkai.vectordb.requests")
    @patch("arkai.vectordb.utils")
    @patch("arkai.vectordb.get_vectordb_port")
    @patch("arkai.vectordb.is_vectordb_running")
    def test_initdb_valid_name(self, mock_is_running, mock_get_port, mock_utils, mock_requests):
        """Test initdb with valid database name."""
        mock_is_running.return_value = True
        mock_get_port.return_value = 8082

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        vectordb.cmd_vectordb_initdb("test-db")
        mock_utils.info.assert_called()
        assert mock_requests.post.call_count == 2
