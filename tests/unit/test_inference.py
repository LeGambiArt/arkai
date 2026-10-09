"""Tests for inference server model discovery."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import inference


def test_get_inference_model_returns_server_model_id() -> None:
    """The model ID comes from the server's OpenAI-compatible model list."""
    response = MagicMock()
    response.json.return_value = {"data": [{"id": "/models/running.gguf"}]}
    with patch.object(inference.requests, "get", return_value=response) as get:
        assert inference.get_inference_model(9090) == "/models/running.gguf"

    get.assert_called_once_with("http://127.0.0.1:9090/v1/models", timeout=5)


@pytest.mark.parametrize(
    "response",
    ["not json", '{"data":[]}', '{"data":[{"id":null}]}'],
)
def test_get_inference_model_rejects_invalid_server_response(response: str) -> None:
    """Malformed model listings fail instead of silently using configured state."""
    server_response = MagicMock()
    server_response.json.side_effect = ValueError(response)
    with patch.object(inference.requests, "get", return_value=server_response):
        with pytest.raises(RuntimeError, match="invalid model"):
            inference.get_inference_model()


def test_get_inference_model_reports_server_failure() -> None:
    """A failed model query identifies the affected server port."""
    with patch.object(
        inference.requests,
        "get",
        side_effect=inference.requests.ConnectionError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="port 8081: connection refused"):
            inference.get_inference_model()


def test_health_check_uses_requests_and_accepts_successful_response() -> None:
    """The inference health check uses the requests client directly."""
    response = MagicMock()
    with patch.object(inference.requests, "get", return_value=response) as get:
        assert inference._is_inference_server_healthy(9090) is True

    get.assert_called_once_with("http://127.0.0.1:9090/v1/models", timeout=5)
    response.raise_for_status.assert_called_once_with()


def test_health_check_returns_false_for_request_failure() -> None:
    """Unavailable or unsuccessful inference servers are reported as unhealthy."""
    with patch.object(
        inference.requests,
        "get",
        side_effect=inference.requests.ConnectionError("connection refused"),
    ):
        assert inference._is_inference_server_healthy(9090) is False


def test_inference_cli_registers_backend_override() -> None:
    """The inference start command accepts a backend override."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    inference.ingest_cli_options(subparsers)

    args = parser.parse_args(["inference", "start", "--backend", "mlx"])
    assert args.backend == "mlx"


def test_inference_log_path_is_separate_for_explicit_ports(tmp_path: Path) -> None:
    """Each explicitly selected server port gets its own diagnostic log."""
    with patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)):
        assert inference.get_inference_log_path() == str(tmp_path / "inference.log")
        assert inference.get_inference_log_path(9090) == str(tmp_path / "inference-9090.log")


def test_inference_start_displays_and_preserves_backend_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Show backend output while starting and retain it in the diagnostic log."""
    cfg = {
        "inference": {
            "model": "hf:owner/model:Q4_K_M",
            "port": 9090,
            "gpu_layers": -1,
            "context_size": 2048,
            "startup_timeout": 1,
            "backend": "llama-cpp",
        }
    }
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None
    backend = MagicMock(name="llama-cpp")
    backend.name = "llama-cpp"
    backend.build_command.return_value = ["llama-server", "--port", "9090"]

    def become_healthy(*_args: object) -> bool:
        with (tmp_path / "inference-9090.log").open("a", encoding="utf-8") as log_file:
            log_file.write("Downloading model: 50%\n")
        return True

    with (
        patch.object(inference.config, "load_config", return_value=cfg),
        patch.object(inference.config, "validate_config", return_value=True),
        patch.object(
            inference.config,
            "resolve_model",
            return_value=(cfg["inference"]["model"], {}),
        ),
        patch.object(
            inference.config,
            "get_config_value",
            side_effect=lambda c, key, default=None: c.get("inference", {}).get(
                key.split(".")[-1], default
            ),
        ),
        patch.object(inference, "get_backend", return_value=backend),
        patch.object(inference, "is_inference_running", return_value=False),
        patch.object(inference, "_is_inference_server_healthy", side_effect=become_healthy),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "detect_gpu", return_value="cpu"),
        patch.object(inference.utils, "is_port_in_use", return_value=False),
        patch.object(inference.subprocess, "Popen", return_value=process) as popen,
        patch.object(inference.signal, "signal"),
    ):
        inference.cmd_inference_start(model="hf:owner/model:Q4_K_M", port=9090)

    kwargs = popen.call_args.kwargs
    assert kwargs["stdout"] is not inference.subprocess.DEVNULL
    assert kwargs["stderr"] == inference.subprocess.STDOUT
    log_path = tmp_path / "inference-9090.log"
    assert log_path.exists()
    assert "Downloading model: 50%" in capsys.readouterr().out
    assert "Downloading model: 50%" in log_path.read_text(encoding="utf-8")
