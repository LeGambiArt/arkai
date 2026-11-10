"""Unit tests for the container engine abstraction."""

import subprocess
from unittest.mock import patch

import pytest

from arkai import container_engine


def _cfg(engine="auto"):
    return {"vectordb": {"engine": engine}}


def _run_result(returncode=0, stdout="", stderr=""):
    """Build a fake CompletedProcess for subprocess.run."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestResolveEngine:
    @patch("arkai.container_engine._engine_available")
    def test_auto_prefers_podman_then_apple_then_docker(self, mock_available):
        """Test auto detection order is podman, apple, docker."""
        mock_available.side_effect = lambda engine: engine != "podman"
        assert container_engine.resolve_engine(_cfg()) == "apple"

        mock_available.side_effect = lambda engine: engine not in ("podman", "apple")
        assert container_engine.resolve_engine(_cfg()) == "docker"

        mock_available.side_effect = lambda engine: True
        assert container_engine.resolve_engine(_cfg()) == "podman"

    @patch("arkai.container_engine._engine_available", return_value=False)
    def test_auto_with_no_engine_raises(self, _mock_available):
        """Test auto detection failure lists all engines."""
        with pytest.raises(RuntimeError, match="podman.*container.*docker"):
            container_engine.resolve_engine(_cfg())

    @patch("arkai.container_engine._engine_available", return_value=True)
    def test_explicit_engine(self, _mock_available):
        """Test an explicit engine name is returned as-is."""
        assert container_engine.resolve_engine(_cfg("docker")) == "docker"
        assert container_engine.resolve_engine(_cfg("apple")) == "apple"

    @patch("arkai.container_engine._engine_available", return_value=False)
    def test_explicit_engine_missing_binary(self, _mock_available):
        """Test an explicit engine without its binary raises with a hint."""
        with pytest.raises(RuntimeError, match="vectordb.engine.*auto"):
            container_engine.resolve_engine(_cfg("podman"))

    def test_unknown_engine_rejected(self):
        """Test an unknown engine name is rejected."""
        with pytest.raises(RuntimeError, match="vectordb.engine must be"):
            container_engine.resolve_engine(_cfg("containerd"))

    @patch("shutil.which", return_value=None)
    def test_apple_skipped_off_macos(self, _mock_which, monkeypatch):
        """Test apple engine is never available off macOS."""
        monkeypatch.setattr(container_engine, "_IS_MACOS", False)
        assert not container_engine._engine_available("apple")

    @patch("shutil.which", return_value="/usr/bin/container")
    def test_apple_available_on_macos(self, _mock_which, monkeypatch):
        """Test apple engine is available on macOS with the binary."""
        monkeypatch.setattr(container_engine, "_IS_MACOS", True)
        assert container_engine._engine_available("apple")


class TestEngineSystemReady:
    @patch("arkai.container_engine.subprocess.run")
    def test_podman_info_exit_zero(self, mock_run):
        mock_run.return_value = _run_result(0)
        assert container_engine.engine_system_ready("podman")
        assert mock_run.call_args.args[0] == ["podman", "info"]

    @patch("arkai.container_engine.subprocess.run")
    def test_docker_info_failure(self, mock_run):
        mock_run.return_value = _run_result(1, stderr="cannot connect to daemon")
        assert not container_engine.engine_system_ready("docker")

    @patch("arkai.container_engine.subprocess.run")
    def test_apple_system_status(self, mock_run):
        mock_run.return_value = _run_result(0)
        assert container_engine.engine_system_ready("apple")
        assert mock_run.call_args.args[0] == ["container", "system", "status"]

    @patch("arkai.container_engine.subprocess.run")
    def test_timeout_is_not_ready(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["podman", "info"], timeout=30)
        assert not container_engine.engine_system_ready("podman")


class TestEnsureEngineReady:
    @patch("arkai.container_engine.engine_system_ready", return_value=True)
    def test_ready_engine_does_not_raise(self, _mock_ready):
        container_engine.ensure_engine_ready("podman")

    @pytest.mark.parametrize(
        "engine, hint",
        [
            ("podman", "podman machine start"),
            ("apple", "container system start"),
            ("docker", "colima start"),
        ],
    )
    @patch("arkai.container_engine.engine_system_ready", return_value=False)
    @patch("arkai.container_engine.subprocess.run")
    def test_not_ready_errors_include_remedy(self, mock_run, _mock_ready, engine, hint):
        """Test each engine's failure mode names its actionable remedy."""
        mock_run.return_value = _run_result(1, stderr="boom")
        with pytest.raises(RuntimeError) as error:
            container_engine.ensure_engine_ready(engine)
        assert engine in str(error.value)
        assert hint in str(error.value)
        assert "boom" in str(error.value)


class TestRunContainer:
    @patch("arkai.container_engine._run")
    def test_run_builds_shared_flag_syntax(self, mock_run):
        """Test run uses the common -d --name -v -p -e flag surface."""
        mock_run.return_value = _run_result(0, stdout="container-id")
        container_engine.run_container(
            "podman",
            image="docker.io/qdrant/qdrant:latest",
            container_name="arkai-vectordb",
            port=6333,
            host_port=8082,
            volumes=[("/tmp/qdrant", "/qdrant/storage")],
            env={"IS_PERSISTENT": "TRUE"},
        )
        assert mock_run.call_args.args[0] == [
            "podman",
            "run",
            "-d",
            "--name",
            "arkai-vectordb",
            "-v",
            "/tmp/qdrant:/qdrant/storage",
            "-p",
            "127.0.0.1:8082:6333",
            "-e",
            "IS_PERSISTENT=TRUE",
            "docker.io/qdrant/qdrant:latest",
        ]

    @patch("arkai.container_engine.engine_system_ready", return_value=True)
    @patch("arkai.container_engine._run")
    def test_run_failure_surfaces_stderr(self, mock_run, _mock_ready):
        """Test a failed run reports exit code and engine stderr."""
        mock_run.return_value = _run_result(125, stderr="image pull failed: no network")
        with pytest.raises(RuntimeError) as error:
            container_engine.run_container(
                "docker",
                image="img",
                container_name="arkai-vectordb",
                port=6333,
                host_port=8082,
                volumes=[],
            )
        assert "docker" in str(error.value)
        assert "125" in str(error.value)
        assert "image pull failed" in str(error.value)

    @patch("arkai.container_engine.engine_system_ready", return_value=False)
    @patch("arkai.container_engine._run")
    def test_run_failure_with_dead_vm_names_remedy(self, mock_run, _mock_ready):
        """Test a failed run caused by a stopped VM names the remedy."""
        mock_run.return_value = _run_result(1, stderr="vm not running")
        with pytest.raises(RuntimeError, match="podman machine start"):
            container_engine.run_container(
                "podman",
                image="img",
                container_name="arkai-vectordb",
                port=6333,
                host_port=8082,
                volumes=[],
            )


class TestStopContainer:
    @patch("arkai.container_engine._run")
    def test_stop_and_remove(self, mock_run):
        """Test stop runs stop + rm for podman/docker."""
        mock_run.return_value = _run_result(0)
        container_engine.stop_container("podman", "arkai-vectordb")
        assert [c.args[0] for c in mock_run.call_args_list] == [
            ["podman", "stop", "arkai-vectordb"],
            ["podman", "rm", "arkai-vectordb"],
        ]

    @patch("arkai.container_engine._run")
    def test_stop_apple_uses_delete(self, mock_run):
        """Test stop uses container delete for apple."""
        mock_run.return_value = _run_result(0)
        container_engine.stop_container("apple", "arkai-vectordb")
        assert [c.args[0] for c in mock_run.call_args_list] == [
            ["container", "stop", "arkai-vectordb"],
            ["container", "delete", "arkai-vectordb"],
        ]

    @patch("arkai.container_engine._run")
    def test_stop_tolerates_missing_container(self, mock_run):
        """Test stop is silent when the container no longer exists."""
        mock_run.return_value = _run_result(1, stderr="Error: no such container")
        container_engine.stop_container("docker", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_stop_reports_other_failures(self, mock_run):
        """Test stop surfaces failures unrelated to missing containers."""
        mock_run.return_value = _run_result(1, stderr="permission denied")
        with pytest.raises(RuntimeError, match="permission denied"):
            container_engine.stop_container("docker", "arkai-vectordb")


class TestContainerRunning:
    @patch("arkai.container_engine._run")
    def test_podman_inspect_true(self, mock_run):
        mock_run.return_value = _run_result(0, stdout="true\n")
        assert container_engine.container_running("podman", "arkai-vectordb")
        assert mock_run.call_args.args[0] == [
            "podman",
            "inspect",
            "-f",
            "{{.State.Running}}",
            "arkai-vectordb",
        ]

    @patch("arkai.container_engine._run")
    def test_docker_inspect_stopped(self, mock_run):
        mock_run.return_value = _run_result(0, stdout="false\n")
        assert not container_engine.container_running("docker", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_inspect_missing_container(self, mock_run):
        mock_run.return_value = _run_result(1, stderr="no such object")
        assert not container_engine.container_running("docker", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_apple_list_matches_id(self, mock_run):
        mock_run.return_value = _run_result(
            0,
            stdout='[{"id": "arkai-vectordb", "state": "running"}]',
        )
        assert container_engine.container_running("apple", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_apple_list_wrapped_payload(self, mock_run):
        """Test apple list handles both raw and wrapped JSON shapes."""
        mock_run.return_value = _run_result(
            0, stdout='{"containers": [{"containerID": "arkai-vectordb"}]}'
        )
        assert container_engine.container_running("apple", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_apple_list_other_containers(self, mock_run):
        mock_run.return_value = _run_result(0, stdout='[{"id": "other"}]')
        assert not container_engine.container_running("apple", "arkai-vectordb")

    @patch("arkai.container_engine._run")
    def test_apple_list_invalid_json(self, mock_run):
        mock_run.return_value = _run_result(0, stdout="not json")
        assert not container_engine.container_running("apple", "arkai-vectordb")
