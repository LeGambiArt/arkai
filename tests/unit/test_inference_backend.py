"""Tests for inference backend registration and command construction."""

from pathlib import Path
from unittest.mock import patch

import pytest

from arkai.inference_backend import get_backend, get_backend_names, register_backend
from arkai.inference_llama_cpp import LlamaCppBackend


def test_llama_cpp_backend_builds_command_for_local_model(tmp_path: Path) -> None:
    """The llama.cpp backend translates common settings into server flags."""
    model = tmp_path / "models" / "model.gguf"
    model.parent.mkdir()
    model.touch()

    with patch("arkai.inference_llama_cpp.utils.resolve_binary", return_value="/bin/llama"):
        with patch("arkai.inference_llama_cpp.utils.get_data_home", return_value=str(tmp_path)):
            command = LlamaCppBackend().build_command("llama-server", "model.gguf", 9090, 12, 4096)

    assert command == [
        "/bin/llama",
        "--port",
        "9090",
        "--host",
        "127.0.0.1",
        "--model",
        str(model),
        "--n-gpu-layers",
        "12",
        "--ctx-size",
        "4096",
    ]


def test_llama_cpp_backend_builds_command_for_huggingface_model() -> None:
    """The llama.cpp backend preserves Hugging Face model references."""
    with patch("arkai.inference_llama_cpp.utils.resolve_binary", return_value="/bin/llama"):
        command = LlamaCppBackend().build_command("llama-server", "hf:org/model", 8081, -1, 65536)

    assert command == [
        "/bin/llama",
        "--port",
        "8081",
        "--host",
        "127.0.0.1",
        "-hf",
        "org/model",
        "--n-gpu-layers",
        "-1",
        "--ctx-size",
        "65536",
    ]


def test_backend_registry_contains_llama_cpp() -> None:
    """The built-in llama.cpp backend is available through the registry."""
    assert "llama-cpp" in get_backend_names()
    assert get_backend("llama-cpp").name == "llama-cpp"


def test_backend_registry_rejects_unknown_backend() -> None:
    """Unknown backend names include available alternatives in the error."""
    with pytest.raises(RuntimeError, match="not supported.*llama-cpp"):
        get_backend("missing")


def test_custom_backend_can_be_registered() -> None:
    """The registry accepts additional OpenAI-compatible backend implementations."""

    class FakeBackend:
        name = "fake"

        def build_command(
            self, path: str, model: str, port: int, gpu_layers: int, context_size: int
        ) -> list[str]:
            return [path, model, str(port), str(gpu_layers), str(context_size)]

    backend = FakeBackend()
    register_backend(backend)
    try:
        assert get_backend("fake") is backend
    finally:
        from arkai import inference_backend

        inference_backend._BACKENDS.pop("fake", None)
