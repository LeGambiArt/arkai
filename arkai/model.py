"""Model lifecycle management: download, list, remove, convert, benchmark."""

import argparse
import os
from pathlib import Path

from arkai import config, model_splash, providers, utils
from arkai.inference_backend import get_backend_names
from arkai.model_store import ModelEntry


def exec_cmd(args: argparse.Namespace) -> None:
    """Select 'model' command to execute."""
    match args.model_cmd:
        case "list":
            cmd_model_list(args.backend)
        case "download":
            cmd_model_download(args.model_ref, args.backend)
        case "remove":
            cmd_model_remove(args.model_name, args.backend, args.installation_id)
        case "resolve":
            cmd_model_resolve(args.model_ref, args.backend, args.installation_id)
        case "prune":
            cmd_model_prune(args.dry_run, args.backend)
        case "convert":
            cmd_model_convert(
                args.model,
                args.quantization,
                args.output,
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
        help=("Provider-qualified model (hf:owner/model[:quantization] or ollama:model:tag)"),
    )
    list_parser = model_subparsers.add_parser("list", help="List available models")
    remove_parser = model_subparsers.add_parser("remove", help="Remove model")
    remove_parser.add_argument("model_name", help="Model file name")
    resolve_parser = model_subparsers.add_parser(
        "resolve", help="Show the path of an installed model"
    )
    resolve_parser.add_argument("model_ref", help="Provider reference or local model filename")
    for store_parser in (download_parser, list_parser, remove_parser, resolve_parser):
        store_parser.add_argument(
            "--backend",
            choices=sorted(get_backend_names()),
            help="Override configured backend for model storage; list defaults to all stores",
        )
    for selection_parser in (remove_parser, resolve_parser):
        selection_parser.add_argument(
            "--id", dest="installation_id", help="Splash installation ID shown by list"
        )
    prune_parser = model_subparsers.add_parser(
        "prune", help="Collect unreferenced Arkai-owned Splash cache data"
    )
    prune_parser.add_argument("--backend", choices=["splash"], help="Override configured backend")
    prune_parser.add_argument(
        "--dry-run", action="store_true", help="Preview eligible paths without deleting them"
    )
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


def cmd_model_download(model_ref: str, backend: str | None = None) -> None:
    """Download a provider-qualified model into the Arkai model cache."""
    cfg = config.load_config()
    selected = backend or config.get_config_value(cfg, "inference.backend", "llama-cpp")
    if selected == "splash":
        path = model_splash.SplashModelStore(cfg).download(model_ref)
    else:
        path = providers.download_model(model_ref)
    utils.info(f"Downloaded {model_ref} to {path}")


def cmd_model_list(backend: str | None = None) -> None:
    """List local/provider caches and native Splash installations without requiring startup."""
    splash_models = []
    if backend is None or backend == "splash":
        try:
            splash_models = model_splash.SplashModelStore(config.load_config()).list_models()
        except (RuntimeError, ValueError) as error:
            if backend == "splash":
                raise
            utils.warn(str(error))
    if backend == "splash":
        if splash_models:
            _show_splash_models(splash_models)
        else:
            utils.info("No models found")
        return
    models_dir = get_models_dir()
    gguf_files = []

    # Collect local GGUF files
    if os.path.exists(models_dir):
        gguf_files = sorted(Path(models_dir).glob("*.gguf"))

    provider_models = providers.list_provider_models()

    # If neither found, inform user
    if not gguf_files and not provider_models and not splash_models:
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
    if splash_models:
        utils.info("")
        _show_splash_models(splash_models)


def _show_splash_models(entries: list[ModelEntry]) -> None:
    """Expose native identities, complete/incomplete state, and shared dependency sizes."""
    utils.info("Splash models:")
    for entry in entries:
        utils.info(f"  {entry.reference} [{entry.installation_id}]")
        utils.info(f"    Path: {entry.path}; {'complete' if entry.complete else 'incomplete'}")
        utils.info(f"    Revision: {entry.revision or 'unknown'}")
        utils.info(f"    Source bytes: {entry.owned_bytes} exclusive, {entry.shared_bytes} shared")
        for role, source in entry.sources.items():
            utils.info(f"    {role}: {source.get('repo')}@{source.get('revision') or 'local'}")


def cmd_model_resolve(
    model_ref: str, backend: str | None = None, installation_id: str | None = None
) -> None:
    """Print a locally installed model path, without starting inference or downloading it."""
    cfg = config.load_config()
    selected = backend or config.get_config_value(cfg, "inference.backend", "llama-cpp")
    if selected == "splash":
        path = model_splash.SplashModelStore(cfg).resolve(model_ref, installation_id)
    elif installation_id is not None:
        raise RuntimeError("--id is only supported for Splash model installations")
    elif ":" in model_ref:
        path = providers.resolve_model(model_ref)
    else:
        if Path(model_ref).name != model_ref:
            raise RuntimeError("Use a local model filename from the Arkai models directory")
        path = Path(get_models_dir()) / model_ref
        if not path.is_file():
            raise RuntimeError(f"Model not found: {model_ref}")
    utils.info(str(path))


def cmd_model_prune(dry_run: bool = False, backend: str | None = None) -> None:
    """Preview or collect unreferenced data in the default Arkai-owned Splash caches."""
    cfg = config.load_config()
    selected = backend or config.get_config_value(cfg, "inference.backend", "llama-cpp")
    if selected != "splash":
        raise RuntimeError(
            "Cache pruning is supported for Splash; configure it or use --backend splash"
        )
    result = model_splash.SplashModelStore(cfg).prune(dry_run)
    for path in result["paths"]:
        utils.info(f"{'Would remove' if dry_run else 'Removed'} {path}")
    if not result["paths"]:
        utils.info("No unreferenced Splash cache data found")


def cmd_model_remove(
    model_name: str, backend: str | None = None, installation_id: str | None = None
) -> None:
    """Remove a local model file or provider-cached model.

    Args:
        model_name: Local model filename or provider reference
            (e.g., 'model.gguf' or 'hf:org/model')
    """
    cfg = config.load_config()
    selected = backend or config.get_config_value(cfg, "inference.backend", "llama-cpp")
    if selected == "splash":
        model_splash.SplashModelStore(cfg).remove(model_name, installation_id)
        utils.info(
            f"Removed {model_name}; shared cache data retained for model prune --backend splash"
        )
        return
    if installation_id is not None:
        raise RuntimeError("--id is only supported for Splash model installations")
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
            "Ollama models are already downloaded as GGUF or native MLX artifacts "
            "and do not need conversion"
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
