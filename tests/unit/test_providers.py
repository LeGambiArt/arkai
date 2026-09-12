"""Unit tests for model providers."""

import hashlib
import json
from unittest.mock import patch

import pytest

from arkai import providers, utils


class FakeResponse:
    """Small requests response test double."""

    def __init__(self, payload=None, content=b"", status_code=200, error=None):
        self.payload = payload
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(content))}
        self.error = error

    def raise_for_status(self):
        """Provide the requests response API used by the provider."""
        if self.error:
            raise self.error

    def json(self):
        """Return the configured JSON payload."""
        return self.payload

    def iter_content(self, chunk_size):
        """Yield the configured content."""
        yield self.content


def digest(content):
    """Return an Ollama-style SHA-256 digest."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_model_reference_requires_explicit_provider():
    """Unqualified downloads are rejected."""
    with pytest.raises(ValueError, match="must include a provider"):
        providers.ModelReference.parse("owner/model")


def test_model_reference_rejects_unknown_provider():
    """Unknown provider names produce an actionable error."""
    with pytest.raises(ValueError, match="Supported providers"):
        providers.ModelReference.parse("unknown:model")


def test_huggingface_download_materializes_lfs_files(tmp_path, monkeypatch):
    """Hugging Face repositories have their Git-LFS model files downloaded."""
    monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
    commands = []
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 4\n"

    def run_command(command, **kwargs):
        commands.append(command)
        if "clone" in command:
            repository_dir = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
            (repository_dir / ".git").mkdir(parents=True)
            (repository_dir / "model.safetensors").write_bytes(pointer)
        elif command[-2:] == ["lfs", "pull"]:
            repository_dir = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
            (repository_dir / "model.safetensors").write_bytes(b"real")
        return 0, "", ""

    with (
        patch.object(providers.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(providers.utils, "run_command", side_effect=run_command),
    ):
        result = providers.download_model("hf:org/model")

    assert result.joinpath("model.safetensors").read_bytes() == b"real"
    assert commands[1][-2:] == ["lfs", "install"]
    assert commands[2][-2:] == ["lfs", "pull"]
    assert commands[1][0:2] == ["/usr/bin/git", "-C"]
    repository_path = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
    assert commands[1][2] == str(repository_path)


def test_huggingface_lfs_install_failure_is_reported(tmp_path, monkeypatch):
    """A failed repository Git-LFS initialization stops model materialization."""
    monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
    repository_dir = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
    (repository_dir / ".git").mkdir(parents=True)
    (repository_dir / "model.safetensors").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 4\n"
    )

    def run_command(command, **kwargs):
        if command[-2:] == ["lfs", "install"]:
            return 1, "", "git lfs install failed"
        raise AssertionError(f"git lfs pull should not run: {command}")

    with (
        patch.object(providers.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(providers.utils, "run_command", side_effect=run_command),
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize Git LFS"):
            providers.resolve_model("hf:org/model")


def test_huggingface_resolve_repairs_existing_lfs_cache(tmp_path, monkeypatch):
    """Resolving a cached model also materializes old Git-LFS pointers."""
    monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
    repository_dir = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
    (repository_dir / ".git").mkdir(parents=True)
    (repository_dir / "model.safetensors").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 4\n"
    )

    def run_command(command, **kwargs):
        (repository_dir / "model.safetensors").write_bytes(b"real")
        return 0, "", ""

    with (
        patch.object(providers.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(providers.utils, "run_command", side_effect=run_command),
    ):
        result = providers.resolve_model("hf:org/model")

    assert result == repository_dir
    assert (repository_dir / "model.safetensors").read_bytes() == b"real"


def test_huggingface_lfs_requires_git_lfs(tmp_path, monkeypatch):
    """A missing Git-LFS installation produces an actionable error."""
    monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
    repository_dir = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
    (repository_dir / ".git").mkdir(parents=True)
    (repository_dir / "model.safetensors").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 4\n"
    )

    with patch.object(
        providers.shutil,
        "which",
        side_effect=lambda name: "/usr/bin/git" if name == "git" else None,
    ):
        with pytest.raises(RuntimeError, match="Install Git LFS"):
            providers.resolve_model("hf:org/model")


def test_ollama_provider_reports_missing_model_tag():
    """A missing registry model produces an actionable error."""
    response = FakeResponse(
        status_code=404,
        error=providers.requests.HTTPError("not found"),
    )
    with patch.object(providers.requests, "get", return_value=response):
        with pytest.raises(RuntimeError, match="Ollama model or tag not found"):
            providers._get_json("https://registry.ollama.ai/v2/library/missing/manifests/latest")


def test_ollama_provider_downloads_and_verifies_model_layer(tmp_path, monkeypatch):
    """The Ollama provider stores a verified model layer as GGUF."""
    monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
    config = b'{"format":"gguf"}'
    model_blob = b"GGUF model data"
    config_digest = digest(config)
    model_digest = digest(model_blob)
    manifest = {
        "config": {"digest": config_digest},
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}],
    }

    def get(url, **kwargs):
        if url.endswith("/manifests/latest"):
            return FakeResponse(payload=manifest)
        if url.endswith(config_digest):
            return FakeResponse(content=config)
        if url.endswith(model_digest):
            return FakeResponse(content=model_blob)
        raise AssertionError(url)

    reference = providers.ModelReference.parse("ollama:llama3")
    with (
        patch.object(providers.requests, "get", side_effect=get),
        patch.object(providers.utils, "info") as info,
    ):
        result = providers.PROVIDERS["ollama"].download(reference)

    assert result.read_bytes() == model_blob
    assert json.loads((result.parent / "manifest.json").read_text()) == manifest
    assert providers.PROVIDERS["ollama"].list_models() == [("llama3:latest", "15.0B")]
    messages = [call.args[0] for call in info.call_args_list]
    assert "Resolving Ollama manifest for llama3" in messages
    assert any(message.startswith("Downloading Ollama model weights") for message in messages)
    assert any(message.startswith("Verified Ollama model weights") for message in messages)
