"""MLX-LM inference backend."""

import importlib

from arkai import providers, utils


class MlxBackend:
    """Build commands for the MLX-LM OpenAI-compatible server."""

    name = "mlx"

    def check_environment(self) -> None:
        """Ensure MLX-LM can be imported in the active Python environment."""
        self._check_mlx_lm_installed()

    def build_command(
        self,
        path: str,
        model: str,
        port: int,
        gpu_layers: int,
        context_size: int,
        path_is_configured: bool = False,
    ) -> list[str]:
        """Build an ``mlx_lm.server`` command from common inference settings.

        MLX-LM selects GPU execution automatically and does not expose
        llama.cpp's GPU-layer or context-size server flags.
        """
        if model.startswith("hf:"):
            model = str(providers.resolve_model(model))
        elif model.startswith("ollama:"):
            raise RuntimeError("MLX-LM requires a Hugging Face MLX model, not an Ollama model")

        server_path = utils.resolve_binary(path, search_path=not path_is_configured)
        return [
            server_path,
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    @staticmethod
    def _check_mlx_lm_installed() -> None:
        """Ensure MLX-LM can be imported in the active Python environment."""
        try:
            importlib.import_module("mlx_lm")
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "MLX-LM is not installed in the active Python environment. "
                "Install it with: pip install -e '.[mlx]'"
            ) from error
