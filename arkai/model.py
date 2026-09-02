"""Model lifecycle management: download, list, remove, convert, benchmark."""

import os
from pathlib import Path
from typing import Optional

from arkai import utils


def get_models_dir() -> str:
    """Return models directory path."""
    data_home = utils.get_data_home()
    if data_home is None:
        raise RuntimeError("DATA_HOME not available")
    return os.path.join(data_home, "models")


def cmd_model_download(hf_repo: str) -> None:
    """Download model from HuggingFace using hf CLI.

    Downloads to HuggingFace cache. Use 'arkai model convert' to convert to GGUF format.

    Args:
        hf_repo: HuggingFace repo path (e.g., 'ibm-granite/granite-4.1-8b-instruct-GGUF')

    Raises:
        RuntimeError: If hf CLI is not found, network fails, or download fails
    """
    try:
        code, _, stderr = utils.run_command(
            ["hf", "download", hf_repo, "--repo-type", "model"],
            capture=False,
            timeout=None,
        )
    except RuntimeError as e:
        if "Command not found" in str(e):
            raise RuntimeError("hf command not found; install huggingface-hub CLI")
        raise

    if code != 0:
        raise _handle_download_error(hf_repo, stderr)

    utils.info(f"Downloaded {hf_repo} to HuggingFace cache")


def _handle_download_error(hf_repo: str, stderr: str) -> RuntimeError:
    """Convert download error output to user-friendly error message.

    Args:
        hf_repo: HuggingFace repo that failed to download
        stderr: Error message from hf command

    Returns:
        RuntimeError with actionable error message
    """
    error_lower = stderr.lower()

    if "network" in error_lower or "connection" in error_lower or "timeout" in error_lower:
        return RuntimeError(
            f"Network error downloading {hf_repo}.\n"
            "Check your internet connection and try again.\n"
            f"If the error persists, try: hf download {hf_repo} --repo-type model --force-download"
        )

    if "authentication" in error_lower or "401" in error_lower or "forbidden" in error_lower:
        return RuntimeError(
            f"Authentication error accessing {hf_repo}.\n"
            "The model may require HuggingFace login:\n"
            "  1. Visit https://huggingface.co/{repo}/tree/main\n"
            "  2. Accept the model license (if required)\n"
            "  3. Run 'huggingface-cli login' or set HF_TOKEN"
        )

    if "not found" in error_lower or "404" in error_lower:
        return RuntimeError(
            f"Model {hf_repo} not found on HuggingFace.\n"
            "Check the repo ID is correct: https://huggingface.co/{hf_repo}\n"
            "Common issues:\n"
            "  - Typo in repo name (use owner/model-name format)\n"
            "  - Private repo without access\n"
            "  - Repo has been deleted"
        )

    if "disk" in error_lower or "space" in error_lower:
        return RuntimeError(
            f"Insufficient disk space to download {hf_repo}.\n"
            "Free up space on your system or configure HF_HOME to a different location"
        )

    if "reconstruction" in error_lower or "cas" in error_lower:
        return RuntimeError(
            f"File integrity error downloading {hf_repo}.\n"
            "The download was corrupted. Try again:\n"
            f"  hf download {hf_repo} --repo-type model --force-download\n"
            "If the error persists, the model may have upload issues on HuggingFace"
        )

    return RuntimeError(
        f"Failed to download {hf_repo}\n"
        f"Error: {stderr}\n"
        "Try again with: hf download {hf_repo} --repo-type model --force-download"
    )


def cmd_model_list() -> None:
    """List local GGUF models and HuggingFace cached models.

    Displays two categories:
    1. Local GGUF files in ~/.local/share/arkai/models/
    2. HuggingFace cached models (found via 'hf cache ls')
    """
    models_dir = get_models_dir()
    gguf_files = []

    # Collect local GGUF files
    if os.path.exists(models_dir):
        gguf_files = sorted(Path(models_dir).glob("*.gguf"))

    # Collect HuggingFace cached models
    hf_models = _get_huggingface_cached_models()

    # If neither found, inform user
    if not gguf_files and not hf_models:
        utils.info("No models found")
        return

    # Display local GGUF models
    if gguf_files:
        utils.info("Local GGUF models:")
        for filepath in gguf_files:
            utils.info(f"  {filepath.name}")
    else:
        utils.info("Local GGUF models: none")

    # Display HuggingFace cached models
    if hf_models:
        if gguf_files:
            utils.info("")
        utils.info("HuggingFace cached models:")
        for repo_id, _ in hf_models:
            utils.info(f"  {repo_id}")
    else:
        if not gguf_files:
            # Already printed "No models found" above
            pass
        else:
            utils.info("\nHuggingFace cached models: none")


def _get_huggingface_cached_models() -> list:
    """Get list of HuggingFace cached models.

    Returns:
        List of tuples (repo_id, size_string) for each cached model.
        Returns empty list if 'hf' command unavailable or no models cached.
    """
    try:
        code, stdout, stderr = utils.run_command(["hf", "cache", "ls"])
    except RuntimeError:
        # hf command not available
        return []

    if code != 0:
        # hf cache ls failed
        return []

    models = []
    # Parse output: each line after header is a cached model
    # Format: id<space>size<space>last_accessed<space>last_modified<space>refs
    lines = stdout.strip().split("\n")

    # Skip header line and summary lines
    for line in lines:
        if not line.strip():
            continue
        # Skip lines that don't look like model entries (contain "Found" or "Warning")
        if line.startswith("Found") or line.startswith("Warning"):
            continue
        # Skip the header line (starts with "ID") and separator line (dashes)
        if line.startswith("ID") or line.startswith("-"):
            continue

        # Parse model line: id<space>size<space>...
        # Split on whitespace and take first two non-empty parts
        parts = line.split()
        if len(parts) >= 2:
            full_id = parts[0].strip()
            size = parts[1].strip()
            # Only include if id_type is "model"
            if "/" in full_id:
                id_type, repo_id = full_id.split("/", 1)
                if id_type == "model":
                    models.append((repo_id, size))

    return models


def cmd_model_remove(model_name: str) -> None:
    """Remove a model file.

    Args:
        model_name: Name of model file (e.g., 'model.gguf')
    """
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_name}")

    os.remove(model_path)
    utils.info(f"Removed {model_name}")


def cmd_model_convert(model: str, quantization: str = "Q6_K", output: Optional[str] = None) -> None:
    """Convert HuggingFace model to GGUF format.

    Args:
        model: HuggingFace model ID (e.g., 'apple/DiffuCoder-7B'),
               model name from cache (e.g., 'DiffuCoder-7B'), or path
        quantization: Quantization level (Q4_K_M, Q5_K_M, Q6_K, etc.)
        output: Optional output file path (defaults to
                ~/.local/share/arkai/models/MODEL-QUANTIZATION.gguf)
    """
    # Find arkai-convert script using importlib.resources for packaging
    convert_script: Optional[str] = None
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
        error_msg = stderr if stderr else stdout
        raise RuntimeError(f"Conversion failed:\n{error_msg}")

    # Extract output path from stdout (last line)
    output_path = stdout.strip().split("\n")[-1] if stdout else "unknown"
    utils.info(f"Conversion successful: {output_path}")
