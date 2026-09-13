"""Backend registry for OpenAI-compatible inference servers."""

from typing import Protocol


class InferenceBackend(Protocol):
    """Command builder for an OpenAI-compatible inference server."""

    name: str

    def check_environment(self) -> None:
        """Validate that dependencies required by the backend are available."""

    def build_command(
        self,
        path: str,
        model: str,
        port: int,
        gpu_layers: int,
        context_size: int,
        path_is_configured: bool = False,
    ) -> list[str]:
        """Build the command used to start the inference server."""


_BACKENDS: dict[str, InferenceBackend] = {}
_BUILTINS_LOADED = False


def register_backend(backend: InferenceBackend) -> None:
    """Register an inference backend by its name."""
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> InferenceBackend:
    """Return a registered backend or raise an actionable error."""
    _load_builtin_backends()
    try:
        return _BACKENDS[name]
    except KeyError as error:
        available = ", ".join(sorted(_BACKENDS))
        raise RuntimeError(
            f"Inference backend '{name}' is not supported. Available: {available}"
        ) from error


def get_backend_names() -> set[str]:
    """Return the names of all registered inference backends."""
    _load_builtin_backends()
    return set(_BACKENDS)


def _load_builtin_backends() -> None:
    """Load bundled backends once."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from arkai.inference_llama_cpp import LlamaCppBackend
    from arkai.inference_mlx import MlxBackend

    register_backend(LlamaCppBackend())
    register_backend(MlxBackend())
    _BUILTINS_LOADED = True
