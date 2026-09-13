"""Model lifecycle management: download, list, remove, convert, benchmark."""

import argparse
import os
from pathlib import Path

from arkai import providers, utils


def exec_cmd(args: dict | None = None) -> None:
    """Select 'model' command to execute."""
    match args.model_cmd:  # ty: ignore[unresolved-attribute]
        case "list":
            cmd_model_list()
        case "download":
            cmd_model_download(args.model_ref)  # ty: ignore[unresolved-attribute]
        case "remove":
            cmd_model_remove(args.model_name)  # ty: ignore[unresolved-attribute]
        case "convert":
            cmd_model_convert(
                args.model,  # ty: ignore[unresolved-attribute]
                args.quantization,  # ty: ignore[unresolved-attribute]
                args.output,  # ty: ignore[unresolved-attribute]
            )


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    model_parser = subparsers.add_parser("model", help="Manage models")
    model_subparsers = model_parser.add_subparsers(dest="model_cmd", required=True)
    download_parser = model_subparsers.add_parser("download", help="Download model from a provider")
    download_parser.add_argument(
        "model_ref",
        help="Provider-qualified model (hf:owner/model or ollama:model:tag)",
    )
    model_subparsers.add_parser("list", help="List available models")
    remove_parser = model_subparsers.add_parser("remove", help="Remove model")
    remove_parser.add_argument("model_name", help="Model file name")
    convert_parser = model_subparsers.add_parser(
        "convert", help="Convert a provider model to GGUF format"
    )
    convert_parser.add_argument("model", help="Provider-qualified model ID or local path")
    convert_parser.add_argument(
        "-q", "--quantization", default="Q6_K", help="Quantization level (default: Q6_K)"
    )
    convert_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: ~/.local/share/arkai/models/MODEL-QUANTIZATION.gguf)",
    )


def get_models_dir() -> str:
    """Return models directory path."""
    data_home = utils.get_data_home()
    if data_home is None:
        raise RuntimeError("DATA_HOME not available")
    return os.path.join(data_home, "models")


def cmd_model_download(model_ref: str) -> None:
    """Download a provider-qualified model into the Arkai model cache."""
    path = providers.download_model(model_ref)
    utils.info(f"Downloaded {model_ref} to {path}")


def cmd_model_list() -> None:
    """List local GGUF files and models cached by every provider.

    Displays two categories:
    1. Local GGUF files in ~/.local/share/arkai/models/
    2. Provider models downloaded into Arkai's model cache
    """
    models_dir = get_models_dir()
    gguf_files = []

    # Collect local GGUF files
    if os.path.exists(models_dir):
        gguf_files = sorted(Path(models_dir).glob("*.gguf"))

    provider_models = providers.list_provider_models()

    # If neither found, inform user
    if not gguf_files and not provider_models:
        utils.info("No models found")
        return

    # Display local GGUF models
    if gguf_files:
        utils.info("Local GGUF models:")
        for filepath in gguf_files:
            utils.info(f"  {filepath.name}")
    else:
        utils.info("Local GGUF models: none")

    if provider_models:
        if gguf_files:
            utils.info("")
        utils.info("Provider models:")
        for provider, identifier, _ in provider_models:
            utils.info(f"  {provider}:{identifier}")
    else:
        if gguf_files:
            utils.info("\nProvider models: none")


def cmd_model_remove(model_name: str) -> None:
    """Remove a local model file or provider-cached model.

    Args:
        model_name: Local model filename or provider reference
            (e.g., 'model.gguf' or 'hf:org/model')
    """
    if ":" in model_name:
        providers.remove_model(model_name)
        utils.info(f"Removed {model_name}")
        return

    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_name}")

    os.remove(model_path)
    utils.info(f"Removed {model_name}")


def cmd_model_convert(model: str, quantization: str = "Q6_K", output: str | None = None) -> None:
    """Convert a provider model to GGUF format when conversion is required.

    Args:
        model: Provider model ID (e.g., 'hf:apple/DiffuCoder-7B') or path
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q6_K, etc.)
        output: Optional output file path (defaults to
                ~/.local/share/arkai/models/MODEL-QUANTIZATION.gguf)
    """
    if model.startswith("ollama:"):
        raise RuntimeError(
            "Ollama models are already downloaded as GGUF and do not need conversion"
        )

    # Find arkai-convert script using importlib.resources for packaging
    convert_script: str | None = None
    try:
        from importlib.resources import files

        arkai_files = files("arkai")
        # Access parent directory - may not be available in all typing scenarios
        # so we handle the AttributeError at runtime
        if hasattr(arkai_files, "parent"):
            pkg_parent = getattr(arkai_files, "parent")
            convert_script = str(pkg_parent.joinpath("bin", "arkai-convert"))
        else:
            raise AttributeError("parent not found")
    except (ImportError, TypeError, AttributeError):
        # Fallback to direct path search
        script_dir = os.path.dirname(os.path.abspath(__file__))
        convert_script = os.path.join(script_dir, "..", "bin", "arkai-convert")

    if not os.path.exists(convert_script):
        raise RuntimeError(f"Conversion script not found: {convert_script}")

    # Build command
    cmd = [convert_script, model, "-q", quantization]
    if output:
        cmd.extend(["-o", output])

    try:
        code, stdout, stderr = utils.run_command(cmd, capture=False, timeout=None)
    except RuntimeError:
        raise RuntimeError(f"Conversion script not found: {convert_script}")

    if code != 0:
        error_msg = stderr or stdout or f"conversion script exited with status {code}"
        raise RuntimeError(f"Conversion failed:\n{error_msg}")

    # Extract output path from stdout (last line)
    output_path = stdout.strip().split("\n")[-1] if stdout else "unknown"
    utils.info(f"Conversion successful: {output_path}")
