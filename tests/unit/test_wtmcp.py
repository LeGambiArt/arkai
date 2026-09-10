"""Unit tests for wtmcp service lifecycle."""

import signal
import sys
from unittest.mock import MagicMock, patch

from arkai import wtmcp, wtmcp_supervisor


class TestWtmcpSupervisor:
    """Tests for the wtmcp supervisor process."""

    @patch("arkai.wtmcp_supervisor.signal.signal")
    @patch("arkai.wtmcp_supervisor.subprocess.Popen")
    def test_supervisor_tracks_child_and_forwards_termination(
        self, mock_popen, mock_signal, tmp_path
    ):
        """Supervisor records its child PID and forwards termination signals."""
        child = MagicMock(pid=12345)
        child.poll.return_value = None
        child.wait.return_value = 0
        mock_popen.return_value = child
        handlers = {}
        mock_signal.side_effect = lambda sig, handler: handlers.setdefault(sig, handler)
        child_pid_path = tmp_path / "wtmcp-8080.server.pid"

        result = wtmcp_supervisor.run(["wtmcp", "serve"], str(child_pid_path))

        assert result == 0
        mock_popen.assert_called_once_with(["wtmcp", "serve"])
        assert not child_pid_path.exists()
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        child.terminate.assert_called_once()


class TestWtmcpStart:
    """Tests for wtmcp server startup."""

    @patch("arkai.wtmcp.time.sleep")
    @patch("arkai.wtmcp.subprocess.Popen")
    @patch("arkai.wtmcp.is_wtmcp_running", side_effect=[False, True])
    @patch("arkai.wtmcp.utils.write_pid")
    @patch("arkai.wtmcp.utils.save_yaml")
    @patch("arkai.wtmcp.utils.resolve_binary", return_value="/usr/local/bin/wtmcp")
    @patch("arkai.wtmcp.utils.is_port_in_use", return_value=False)
    @patch("arkai.wtmcp.utils.get_pid_dir")
    @patch("arkai.wtmcp.config.load_config")
    def test_start_launches_supervisor(
        self,
        mock_load_config,
        mock_get_pid_dir,
        mock_port_in_use,
        mock_resolve_binary,
        mock_save_yaml,
        mock_write_pid,
        mock_is_running,
        mock_popen,
        mock_sleep,
        monkeypatch,
        tmp_path,
    ):
        """Start wtmcp under the internal supervisor rather than directly."""
        monkeypatch.chdir(tmp_path)
        mock_load_config.return_value = {"wtmcp": {"path": "wtmcp", "port": 8080}}
        mock_get_pid_dir.return_value = str(tmp_path)
        mock_popen.return_value = MagicMock(pid=9876)

        wtmcp.cmd_wtmcp_start()

        mock_popen.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "arkai.wtmcp_supervisor",
                "--child-pid-path",
                str(tmp_path / "wtmcp-8080.server.pid"),
                "--",
                "/usr/local/bin/wtmcp",
                "serve",
                "--port",
                "8080",
                "--transport",
                "streamable-http",
            ]
        )
        mock_write_pid.assert_called_once_with(str(tmp_path / "wtmcp-8080.pid"), 9876)
        mock_sleep.assert_called_once_with(0.5)
