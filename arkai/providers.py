"""Model provider implementations and provider reference parsing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from arkai import utils


@dataclass(frozen=True)
class ModelReference:
    """A model identifier qualified with its provider name."""

    provider: str
    identifier: str

    @classmethod
    def parse(cls, value: str) -> ModelReference:
        """Parse a required ``provider:identifier`` model reference."""
        if ":" not in value:
            raise ValueError(
                "Model reference must include a provider, for example "
                "hf:owner/model or ollama:model:tag"
            )
        provider, identifier = value.split(":", 1)
        if provider not in PROVIDERS:
            supported = ", ".join(sorted(PROVIDERS))
            raise ValueError(
                f"Unknown model provider '{provider}'. Supported providers: {supported}"
            )
        if not identifier:
            raise ValueError(f"Model identifier is missing after provider '{provider}:'")
        return cls(provider, identifier)


class ModelProvider:
    """Base interface implemented by model sources."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def cache_dir(self) -> Path:
        """Return this provider's cache directory."""
        return Path(utils.get_data_home() or "") / "models" / "providers" / self.name

    def download(self, reference: ModelReference) -> Path:
        """Download a model and return its local path."""
        raise NotImplementedError

    def list_models(self) -> list[tuple[str, str]]:
        """Return cached identifiers and human-readable sizes."""
        raise NotImplementedError

    def resolve(self, reference: ModelReference) -> Path:
        """Resolve a downloaded model to a local path."""
        raise NotImplementedError

    def remove(self, reference: ModelReference) -> Path:
        """Remove a downloaded model and return its cache path."""
        raise NotImplementedError


class GitModelProvider(ModelProvider):
    """Provider for model repositories hosted by a Git-compatible service."""

    host: str

    def __init__(self, name: str, host: str) -> None:
        super().__init__(name)
        self.host = host

    def _repository_dir(self, identifier: str) -> Path:
        """Return and validate the repository cache path."""
        parts = identifier.split("/")
        if any(not part or part in {".", ".."} for part in parts):
            raise ValueError("Model identifier contains an invalid path")
        return self.cache_dir.joinpath(*parts)

    def download(self, reference: ModelReference) -> Path:
        """Clone or update a model repository using Git."""
        git_path = shutil.which("git")
        if git_path is None:
            raise RuntimeError("git command not found; install Git to download models")

        repository_dir = self._repository_dir(reference.identifier)
        repository_dir.parent.mkdir(parents=True, exist_ok=True)
        remote_identifier = reference.identifier
        repository_url = f"https://{self.host}/{remote_identifier}.git"
        git_env = os.environ.copy()
        git_env["GIT_TERMINAL_PROMPT"] = "0"

        try:
            if (repository_dir / ".git").exists():
                command = [git_path, "-C", str(repository_dir), "pull", "--ff-only"]
                action = "Updating"
            else:
                command = [git_path, "clone", "--depth", "1", repository_url, str(repository_dir)]
                action = "Cloning"
            utils.info(
                f"{action} {reference.provider}:{reference.identifier} from {repository_url}"
            )
            code, _, stderr = utils.run_command(command, capture=False, timeout=None, env=git_env)
        except RuntimeError:
            if repository_dir.exists() and not (repository_dir / ".git").exists():
                shutil.rmtree(repository_dir)
            raise

        if code != 0:
            if not (repository_dir / ".git").exists():
                shutil.rmtree(repository_dir, ignore_errors=True)
            error_text = stderr or "Git returned a non-zero exit status without an error message."
            raise RuntimeError(f"Failed to download {reference.identifier} with Git:\n{error_text}")
        self._materialize_lfs_files(repository_dir, git_path, git_env, reference.identifier)
        utils.info(f"Git transfer complete: {reference.provider}:{reference.identifier}")
        return repository_dir

    def list_models(self) -> list[tuple[str, str]]:
        """List Git repositories in this provider's cache."""
        if not self.cache_dir.is_dir():
            return []
        result = []
        for git_dir in self.cache_dir.rglob(".git"):
            repository_dir = git_dir.parent
            identifier = repository_dir.relative_to(self.cache_dir).as_posix()
            result.append((identifier, _format_size(_directory_size(repository_dir, True))))
        return sorted(result)

    def resolve(self, reference: ModelReference) -> Path:
        """Resolve a cloned repository directory."""
        repository_dir = self._repository_dir(reference.identifier)
        if not (repository_dir / ".git").exists():
            raise RuntimeError(f"Model not downloaded: {reference.provider}:{reference.identifier}")
        git_path = shutil.which("git")
        if git_path is None:
            raise RuntimeError("git command not found; install Git to use the cached model")
        self._materialize_lfs_files(
            repository_dir,
            git_path,
            {**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            reference.identifier,
        )
        return repository_dir

    def remove(self, reference: ModelReference) -> Path:
        """Remove a cloned repository from the provider cache."""
        repository_dir = self._repository_dir(reference.identifier)
        if not repository_dir.is_dir():
            raise RuntimeError(f"Model not found: {reference.provider}:{reference.identifier}")
        shutil.rmtree(repository_dir)
        return repository_dir

    def _materialize_lfs_files(
        self,
        repository_dir: Path,
        git_path: str,
        git_env: dict[str, str],
        identifier: str,
    ) -> None:
        """Download Git-LFS objects when a repository contains pointer files."""
        if not _contains_lfs_pointer(repository_dir):
            return

        if shutil.which("git-lfs") is None:
            raise RuntimeError(
                f"Model {identifier} contains Git-LFS files. Install Git LFS and retry: "
                "https://git-lfs.com/"
            )

        install_code, _, install_stderr = utils.run_command(
            [git_path, "-C", str(repository_dir), "lfs", "install"],
            capture=False,
            timeout=None,
            env=git_env,
        )
        if install_code != 0:
            error_text = install_stderr or "Git LFS initialization failed."
            raise RuntimeError(f"Failed to initialize Git LFS for {identifier}: {error_text}")

        expected_size = _lfs_pointer_size(repository_dir)
        if expected_size:
            utils.info(
                f"Downloading Git-LFS files for hf:{identifier} "
                f"(expected total: {_format_size(expected_size)}; this may take several minutes)"
            )
        else:
            utils.info(
                f"Downloading Git-LFS files for hf:{identifier} (this may take several minutes)"
            )
        code, _, stderr = utils.run_command(
            [git_path, "-C", str(repository_dir), "lfs", "pull"],
            capture=False,
            timeout=None,
            env=git_env,
        )
        if code != 0 or _contains_lfs_pointer(repository_dir):
            error_text = stderr or "Git LFS did not materialize all model files."
            raise RuntimeError(f"Failed to download Git-LFS files for {identifier}: {error_text}")


def _contains_lfs_pointer(repository_dir: Path) -> bool:
    """Return whether a checked-out repository contains a Git-LFS pointer file."""
    pointer_header = b"version https://git-lfs.github.com/spec/v1\n"
    for path in repository_dir.rglob("*"):
        if path.is_dir() or ".git" in path.parts:
            continue
        try:
            with path.open("rb") as model_file:
                if model_file.read(len(pointer_header)) == pointer_header:
                    return True
        except OSError:
            continue
    return False


def _lfs_pointer_size(repository_dir: Path) -> int:
    """Return the total expected size declared by Git-LFS pointer files."""
    pointer_header = b"version https://git-lfs.github.com/spec/v1\n"
    total_size = 0
    for path in repository_dir.rglob("*"):
        if path.is_dir() or ".git" in path.parts:
            continue
        try:
            with path.open("rb") as model_file:
                if model_file.read(len(pointer_header)) != pointer_header:
                    continue
                for line in model_file:
                    if line.startswith(b"size "):
                        try:
                            total_size += int(line.split()[1])
                        except (IndexError, ValueError):
                            pass
                        break
        except OSError:
            continue
    return total_size


class OllamaProvider(ModelProvider):
    """Provider for public models in the Ollama container registry."""

    registry = "https://registry.ollama.ai"

    def __init__(self) -> None:
        super().__init__("ollama")

    def _name_and_tag(self, identifier: str) -> tuple[str, str]:
        """Split an Ollama model name into registry name and tag."""
        name, separator, tag = identifier.rpartition(":")
        if not separator:
            name, tag = identifier, "latest"
        if not name or not tag or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("Ollama model must use the name[:tag] format")
        return name, tag

    def _registry_path(self, name: str) -> str:
        """Return the OCI path, adding the library namespace when needed."""
        return name if "/" in name else f"library/{name}"

    def download(self, reference: ModelReference) -> Path:
        """Download an Ollama manifest and its verified model blobs."""
        name, tag = self._name_and_tag(reference.identifier)
        model_dir = self.cache_dir / name / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        registry_path = self._registry_path(name)
        manifest_url = f"{self.registry}/v2/{registry_path}/manifests/{tag}"
        headers = {
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json, "
                "application/vnd.oci.image.manifest.v1+json"
            )
        }
        utils.info(f"Resolving Ollama manifest for {reference.identifier}")
        manifest = _get_json(manifest_url, headers=headers)
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        config = manifest.get("config", {})
        if config.get("digest"):
            self._download_blob(registry_path, config["digest"], model_dir, "configuration")

        model_blob = self._find_model_layer(manifest)
        blob_path = self._download_blob(
            registry_path, model_blob["digest"], model_dir, "model weights"
        )
        model_path = model_dir / "model.gguf"
        shutil.copyfile(blob_path, model_path)
        utils.info(f"Ollama model artifact ready: {model_path}")
        return model_path

    def _find_model_layer(self, manifest: dict) -> dict:
        """Find the model weight layer in an Ollama manifest."""
        for layer in manifest.get("layers", []):
            if "model" in layer.get("mediaType", ""):
                return layer
        raise RuntimeError("Ollama manifest does not contain a model layer")

    def _download_blob(self, registry_path: str, digest: str, model_dir: Path, label: str) -> Path:
        """Download and verify one content-addressed registry blob."""
        blob_path = model_dir / "blobs" / digest.replace(":", "-")
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        expected = digest.split(":", 1)[-1]
        if blob_path.exists() and _sha256(blob_path) == expected:
            utils.info(f"Using cached Ollama {label}: {digest}")
            return blob_path

        url = f"{self.registry}/v2/{registry_path}/blobs/{digest}"
        utils.info(f"Downloading Ollama {label}: {digest}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"Failed downloading Ollama {label}: {error}") from error

        total = int(response.headers.get("Content-Length", 0))
        completed = 0
        last_report = time.monotonic()
        with blob_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    completed += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 1:
                        _report_transfer(label, completed, total)
                        last_report = now
        _report_transfer(label, completed, total)
        utils.info(f"Verifying Ollama {label}: {digest}")
        if _sha256(blob_path) != expected:
            blob_path.unlink(missing_ok=True)
            raise RuntimeError(f"Digest verification failed for Ollama blob {digest}")
        utils.info(f"Verified Ollama {label}: {digest}")
        return blob_path

    def list_models(self) -> list[tuple[str, str]]:
        """List Ollama artifacts downloaded into the provider cache."""
        if not self.cache_dir.is_dir():
            return []
        result = []
        for model_path in self.cache_dir.rglob("model.gguf"):
            cache_parts = model_path.parent.relative_to(self.cache_dir).parts
            identifier = f"{'/'.join(cache_parts[:-1])}:{cache_parts[-1]}"
            result.append((identifier, _format_size(model_path.stat().st_size)))
        return sorted(result)

    def resolve(self, reference: ModelReference) -> Path:
        """Resolve a downloaded Ollama model to its GGUF artifact."""
        name, tag = self._name_and_tag(reference.identifier)
        model_path = self.cache_dir / name / tag / "model.gguf"
        if not model_path.exists():
            raise RuntimeError(f"Model not downloaded: {reference.provider}:{reference.identifier}")
        return model_path

    def remove(self, reference: ModelReference) -> Path:
        """Remove an Ollama model directory from the provider cache."""
        name, tag = self._name_and_tag(reference.identifier)
        model_dir = self.cache_dir / name / tag
        if not model_dir.is_dir():
            raise RuntimeError(f"Model not found: {reference.provider}:{reference.identifier}")
        shutil.rmtree(model_dir)
        return model_dir


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    """Fetch and decode a JSON response."""
    response = requests.get(url, headers=headers, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model or tag not found: {url}. "
                "Check the model name and tag in the Ollama registry."
            ) from error
        raise RuntimeError(
            f"Ollama registry request failed ({response.status_code}): {url}"
        ) from error
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON response from {url}")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(directory: Path, exclude_git: bool = False) -> int:
    """Return the total size of files below a directory."""
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not (exclude_git and ".git" in path.parts)
    )


def _format_size(size: int) -> str:
    """Format a byte count using a compact human-readable unit."""
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _report_transfer(label: str, completed: int, total: int) -> None:
    """Report a completed or in-progress provider transfer."""
    if total:
        utils.progress(f"  {label}: {_format_size(completed)} / {_format_size(total)}")
    else:
        utils.progress(f"  {label}: {_format_size(completed)} downloaded")


PROVIDERS: dict[str, ModelProvider] = {
    "hf": GitModelProvider("huggingface", "huggingface.co"),
    "ollama": OllamaProvider(),
}


def get_provider(reference: ModelReference) -> ModelProvider:
    """Return the provider registered for a model reference."""
    return PROVIDERS[reference.provider]


def download_model(value: str) -> Path:
    """Parse and download an explicitly qualified model reference."""
    reference = ModelReference.parse(value)
    return get_provider(reference).download(reference)


def list_provider_models() -> list[tuple[str, str, str]]:
    """Return provider, identifier, and size for every cached model."""
    models = []
    for provider_name, provider in PROVIDERS.items():
        models.extend(
            (provider_name, identifier, size) for identifier, size in provider.list_models()
        )
    return sorted(models)


def resolve_model(value: str) -> Path:
    """Resolve an explicitly qualified model reference from provider caches."""
    reference = ModelReference.parse(value)
    return get_provider(reference).resolve(reference)


def remove_model(value: str) -> Path:
    """Remove a provider-qualified model from its local cache."""
    reference = ModelReference.parse(value)
    return get_provider(reference).remove(reference)
