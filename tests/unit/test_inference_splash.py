"""Tests for the minimal Splash inference-server backend."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import config, inference
from arkai.inference_backend import get_backend, get_backend_names
from arkai.inference_splash import SplashBackend


def test_splash_is_registered() -> None:
    """Backend discovery and static configuration validation include Splash."""
    assert "splash" in get_backend_names()
    assert get_backend("splash").name == "splash"
    assert config.validate_config({"inference": {"backend": "splash", "model": "hf:org/model"}})


@pytest.mark.parametrize("model", ["hf:org/model", "hf:org/model:Q4_K_M"])
def test_splash_command(model: str) -> None:
    """Translate only the common model, port, and context settings into native flags."""
    with patch(
        "arkai.inference_splash.utils.resolve_binary", return_value="/bin/splash"
    ) as resolve:
        command = SplashBackend().build_command("splash", model, 9090, -1, 4096)
    assert command == [
        "/bin/splash",
        "serve",
        "--model",
        model[3:],
        "--host",
        "127.0.0.1",
        "--port",
        "9090",
        "--max-context",
        "4096",
    ]
    resolve.assert_called_once_with("splash", search_path=True)


def test_splash_respects_configured_executable() -> None:
    """Use the existing inference.path option, without adding installation discovery."""
    with patch(
        "arkai.inference_splash.utils.resolve_binary", return_value="/custom/splash"
    ) as resolve:
        command = SplashBackend().build_command(
            "/custom/splash", "hf:org/model", 8081, -1, 65536, path_is_configured=True
        )
    assert command[0] == "/custom/splash"
    resolve.assert_called_once_with("/custom/splash", search_path=False)


@pytest.mark.parametrize("model", ["model.gguf", "ollama:model:latest"])
def test_splash_rejects_non_huggingface_references(model: str) -> None:
    """Do not pretend that Splash accepts local target files or Ollama references."""
    with pytest.raises(RuntimeError, match="Hugging Face"):
        SplashBackend().build_command("splash", model, 8081, -1, 65536)


def test_splash_rejects_server_sampling_defaults() -> None:
    """Sampling has no native server flag and must not be silently discarded."""
    with pytest.raises(RuntimeError, match="sampling"):
        SplashBackend().build_command(
            "splash", "hf:org/model", 8081, -1, 65536, sampling={"temperature": 0.7}
        )


@pytest.mark.parametrize("system,machine", [("Linux", "arm64"), ("Darwin", "x86_64")])
def test_splash_requires_apple_silicon(system: str, machine: str) -> None:
    """Fail before launching a macOS-only executable on unsupported platforms."""
    with (
        patch("arkai.inference_splash.platform.system", return_value=system),
        patch("arkai.inference_splash.platform.machine", return_value=machine),
        pytest.raises(RuntimeError, match="Apple Silicon"),
    ):
        SplashBackend().check_environment()


def test_splash_readiness_endpoint() -> None:
    """Splash readiness remains public when its generation API is authenticated."""
    with patch.object(inference.requests, "get") as get:
        assert inference._is_inference_server_healthy(9090, "/ready")
    get.assert_called_once_with("http://127.0.0.1:9090/ready", timeout=5)


def test_existing_server_query_does_not_require_a_backend() -> None:
    """Accessing a running inference service neither selects nor launches a backend."""
    response = MagicMock()
    response.json.return_value = {"data": [{"id": "incoai/Qwen3.8-27B-Splash"}]}
    with (
        patch.object(inference.requests, "get", return_value=response),
        patch.object(inference, "get_backend") as backend,
        patch.object(inference.subprocess, "Popen") as popen,
    ):
        assert inference.get_inference_model(9090) == "incoai/Qwen3.8-27B-Splash"
    backend.assert_not_called()
    popen.assert_not_called()


def test_manager_starts_splash_with_existing_lifecycle(tmp_path: Path) -> None:
    """Select the native executable and readiness probe without changing process handling."""
    cfg = {
        "inference": {
            "backend": "splash",
            "model": "hf:org/model",
            "port": 9090,
            "startup_timeout": 1,
        }
    }
    process = MagicMock(pid=1234)
    process.poll.return_value = None
    with (
        patch.dict(os.environ, {"HOME": str(tmp_path)}),
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference, "is_inference_running", return_value=False),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "detect_gpu", return_value="metal"),
        patch.object(inference.utils, "is_port_in_use", return_value=False),
        patch.object(inference.utils, "resolve_binary", return_value="/bin/splash") as resolve,
        patch.object(SplashBackend, "check_environment"),
        patch.object(inference, "_is_inference_server_healthy", return_value=True) as healthy,
        patch.object(inference.subprocess, "Popen", return_value=process) as popen,
        patch.object(inference.signal, "signal"),
    ):
        inference.cmd_inference_start()
    resolve.assert_called_once_with("splash", search_path=True)
    assert popen.call_args.args[0][:4] == ["/bin/splash", "serve", "--model", "org/model"]
    assert popen.call_args.kwargs["env"]["HF_HUB_CACHE"] == str(
        tmp_path / ".local/share/arkai/models/splash/hub"
    )
    healthy.assert_called_once_with(9090, "/ready")
    assert inference.utils.load_yaml(str(tmp_path / "inference.state"))["backend"] == "splash"
