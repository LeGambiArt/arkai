"""llama.cpp inference backend."""

import os

from arkai import providers, utils


class LlamaCppBackend:
    """Build commands for the llama.cpp ``llama-server`` executable."""

    name = "llama-cpp"

    def build_command(
        self,
        path: str,
        model: str,
        port: int,
        gpu_layers: int,
        context_size: int,
    ) -> list[str]:
        """Build a llama-server command from common inference settings."""
        command = ["--port", str(port), "--host", "127.0.0.1"]
        if model.startswith("hf:"):
            reference = providers.ModelReference.parse(model)
            command.extend(["-hf", reference.identifier])
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
        return command
