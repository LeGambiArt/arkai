"""Tests for inference server model discovery."""

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
