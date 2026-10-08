"""Splash inference-server command construction."""

import platform

from arkai import utils
from arkai.inference_backend import SamplingSettings


class SplashBackend:
    """Start Splash using its own model installation and hardware validation."""

    name = "splash"

    def check_environment(self) -> None:
        """Require Apple Silicon macOS; Splash checks its remaining hardware requirements."""
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Splash requires macOS on Apple Silicon")

    def build_command(
        self,
        path: str,
        model: str,
        port: int,
        gpu_layers: int,
        context_size: int,
        path_is_configured: bool = False,
        sampling: SamplingSettings | None = None,
    ) -> list[str]:
        """Translate common settings; Splash selects GPU execution automatically."""
        if not model.startswith("hf:") or not model[3:]:
            raise RuntimeError("Splash requires a Hugging Face reference: hf:owner/model[:variant]")
        if sampling:
            raise RuntimeError(
                "Splash has no server sampling flags; remove inference/profile sampling defaults"
            )
        binary = utils.resolve_binary(path, search_path=not path_is_configured)
        return [
            binary,
            "serve",
            "--model",
            model[3:],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-context",
            str(context_size),
        ]
