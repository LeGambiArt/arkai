"""llama.cpp inference backend."""

import os
from pathlib import Path

from arkai import providers, utils
from arkai.inference_backend import SamplingSettings


class LlamaCppBackend:
    """Build commands for the llama.cpp ``llama-server`` executable."""

    name = "llama-cpp"

    def check_environment(self) -> None:
        """Validate llama.cpp backend dependencies at process launch time."""

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
        """Build a llama-server command from common inference settings."""
        command = ["--port", str(port), "--host", "127.0.0.1"]
        if model.startswith("hf:"):
            reference = providers.ModelReference.parse(model)
            cached_model = self._cached_model_path(reference)
            if cached_model is not None:
                command.extend(["--model", str(cached_model)])
            else:
                repository = reference.identifier
                if reference.quantization:
                    repository = f"{repository}:{reference.quantization}"
                command.extend(["-hf", repository])
        elif model.startswith("ollama:"):
            command.extend(["--model", str(providers.resolve_model(model))])
        else:
            data_home = utils.get_data_home()
            model_path = os.path.join(data_home, "models", model) if data_home else None
            if model_path is None or not os.path.exists(model_path):
                raise RuntimeError(f"Model not found: {model_path}")
            command.extend(["--model", model_path])

        server_path = utils.resolve_binary(path)
        command.insert(0, server_path)

        command.extend(
            [
                "--n-gpu-layers",
                str(gpu_layers),
                "--ctx-size",
                str(context_size),
            ]
        )
        option_names = {
            "temperature": "--temp",
            "top_p": "--top-p",
            "top_k": "--top-k",
            "min_p": "--min-p",
            "presence_penalty": "--presence-penalty",
            "frequency_penalty": "--frequency-penalty",
            "repetition_penalty": "--repeat-penalty",
        }
        for setting, option in option_names.items():
            if sampling and setting in sampling:
                command.extend([option, str(sampling[setting])])
        return command

    @staticmethod
    def _cached_model_path(reference: providers.ModelReference) -> Path | None:
        """Return a cached GGUF matching a reference, if Arkai has one."""
        try:
            repository = providers.get_provider(reference).resolve(reference)
        except RuntimeError:
            return None

        if not reference.quantization:
            return None
        candidates = sorted(
            path
            for path in repository.glob(f"*{reference.quantization}*.gguf")
            if path.is_file() and path.stat().st_size > 1024
        )
        return candidates[0] if candidates else None
