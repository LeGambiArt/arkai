"""Inference server lifecycle management."""

import argparse
import os
import signal
import subprocess
import time

import requests

from arkai import config, providers, utils


def exec_cmd(args: dict | None = None) -> None:
    """Select 'inference' command to execute."""
    match args.inference_cmd:  # ty: ignore[unresolved-attribute]
        case "start":
            cmd_inference_start(
                args.model,  # ty: ignore[unresolved-attribute]
                args.gpu_layers,  # ty: ignore[unresolved-attribute]
                args.context,  # ty: ignore[unresolved-attribute]
                args.port,  # ty: ignore[unresolved-attribute]
            )
        case "stop":
            cmd_inference_stop()
        case "status":
            cmd_inference_status()


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    inference_parser = subparsers.add_parser("inference", help="Manage inference engine server")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_cmd", required=True)
    start_parser = inference_subparsers.add_parser("start", help="Start inference server")
    start_parser.add_argument("-m", "--model", help="Override model from config")
    start_parser.add_argument("--gpu-layers", type=int, help="Override GPU layers")
    start_parser.add_argument("--context", type=int, help="Override context size")
    start_parser.add_argument("--port", type=int, help="Override port from config")
    inference_subparsers.add_parser("stop", help="Stop inference server")
    inference_subparsers.add_parser("status", help="Show inference server status")


def get_inference_pid_path(port: int | None = None) -> str:
    """Return path to an inference server PID file.

    The default instance retains the historical path. Explicit ports use separate
    files so multiple inference servers can run concurrently.
    """
    pid_dir = utils.get_pid_dir()
    filename = "inference.pid" if port is None else f"inference-{port}.pid"
    return os.path.join(pid_dir, filename)


def get_inference_state_path(port: int | None = None) -> str:
    """Return path to an inference server state file."""
    pid_dir = utils.get_pid_dir()
    filename = "inference.state" if port is None else f"inference-{port}.state"
    return os.path.join(pid_dir, filename)


def is_inference_running(port: int | None = None) -> bool:
    """Check if the inference server for ``port`` is running."""
    pid_path = get_inference_pid_path(port)
    pid = utils.read_pid(pid_path)
    if pid is None:
        return False

    return utils.is_process_running(pid)


def get_inference_model(port: int | None = None) -> str:
    """Return the model ID reported by the running inference server.

    Args:
        port: Inference server port. Defaults to 8081.

    Returns:
        The model ID from the server's OpenAI-compatible model listing.

    Raises:
        RuntimeError: If the server cannot be queried or returns an invalid response.
    """
    server_port = port or 8081
    url = f"http://127.0.0.1:{server_port}/v1/models"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        model_id = response.json()["data"][0]["id"]
    except requests.RequestException as error:
        raise RuntimeError(
            f"Unable to determine the running inference model on port {server_port}: {error}"
        ) from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Inference server returned an invalid model list on port {server_port}"
        ) from error

    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Inference server returned an invalid model ID on port {server_port}")
    return model_id


def _is_inference_server_healthy(port: int) -> bool:
    """Return whether the inference server responds successfully to a health check."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def cmd_inference_start(
    model: str | None = None,
    gpu_layers: int | None = None,
    context_size: int | None = None,
    port: int | None = None,
) -> None:
    """Start inference server (llama-server).

    Args:
        model: Override model from config
        gpu_layers: Override GPU layers from config
        context_size: Override context size from config
        port: Override port from config

    Raises:
        RuntimeError: If no model specified and none available in config
    """
    # Load config
    cfg = config.load_config()
    # A port supplied on the CLI identifies a separate instance. Otherwise use
    # the default PID/state paths, even when the config changes the port.
    instance_port = port

    # Apply CLI overrides before validation so they can satisfy required fields
    if model:
        cfg["inference"]["model"] = model
    if gpu_layers is not None:
        cfg["inference"]["gpu_layers"] = gpu_layers
    if context_size is not None:
        cfg["inference"]["context_size"] = context_size
    if port is not None:
        cfg["inference"]["port"] = port

    if not config.validate_config(cfg, require_model=True):
        raise RuntimeError("Invalid configuration")

    # Check only the requested instance, allowing explicit ports to coexist.
    if is_inference_running(instance_port):
        utils.info("Inference server already running")
        return

    # Detect GPU
    gpu_type = utils.detect_gpu()
    utils.info(f"Detected GPU: {gpu_type}")

    # Check port availability
    port = config.get_config_value(cfg, "inference.port", 8081)
    if utils.is_port_in_use(port):
        raise RuntimeError(f"Port {port} already in use")

    # Resolve model path
    model = config.get_config_value(cfg, "inference.model")

    model_path: str | None = None
    hf_model: str | None = None
    if model.startswith("hf:"):
        reference = providers.ModelReference.parse(model)
        hf_model = reference.identifier
    elif model.startswith("ollama:"):
        model_path = str(providers.resolve_model(model))
    elif model:
        data_home = utils.get_data_home()
        model_path = os.path.join(data_home, "models", model)  # ty: ignore[no-matching-overload]
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")

    # Start llama-server
    utils.info(f"Starting inference server on port {port}...")

    # Resolve llama-server binary path
    llama_bin = config.get_config_value(cfg, "inference.path", "llama-server")
    llama_server_path = utils.resolve_binary(llama_bin)

    cmd = [llama_server_path, "--port", str(port), "--host", "127.0.0.1"]

    if model_path:
        cmd.extend(["--model", model_path])
    elif hf_model:
        cmd.extend(["-hf", hf_model])

    gpu_layers_val = config.get_config_value(cfg, "inference.gpu_layers", -1)
    context_size_val = config.get_config_value(cfg, "inference.context_size", 65536)

    cmd.extend(
        [
            "--n-gpu-layers",
            str(gpu_layers_val),
            "--ctx-size",
            str(context_size_val),
        ]
    )

    # Disable SIGINT to ensure process and PID file are both created
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        # Start in background
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Write PID and state
        pid_path = get_inference_pid_path(instance_port)
        utils.write_pid(pid_path, proc.pid)
    finally:
        # Always restore SIGINT
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Save engine state (config used at startup)
    state = {
        "model": model,
        "gpu_layers": gpu_layers_val,
        "context_size": context_size_val,
        "port": port,
    }
    utils.save_yaml(get_inference_state_path(instance_port), state)

    # Wait for server to be ready
    utils.info("Waiting for inference server...")
    startup_timeout = config.get_config_value(cfg, "inference.startup_timeout", 600)
    startup_error = "Inference server failed to start"
    for _ in range(startup_timeout):
        if proc.poll() is not None:
            startup_error = (
                f"Inference server exited before becoming ready (code {proc.returncode})"
            )
            break
        if _is_inference_server_healthy(port):
            utils.info(f"Inference server ready on port {port}")
            return
        time.sleep(1)

    # Kill the process that failed to become ready and clean up its files
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    if os.path.exists(pid_path):
        os.remove(pid_path)
    state_path = get_inference_state_path(instance_port)
    if os.path.exists(state_path):
        os.remove(state_path)
    raise RuntimeError(startup_error)


def cmd_inference_stop(port: int | None = None) -> None:
    """Stop the inference server for ``port``."""
    pid_path = get_inference_pid_path(port)
    pid = utils.read_pid(pid_path)

    if pid is None:
        utils.info("Inference server not running")
        return

    utils.info(f"Stopping inference server (PID {pid})...")
    utils.kill_process(pid)

    if not utils.wait_for_process_stop(pid):
        raise RuntimeError(
            f"Inference server (PID {pid}) did not stop after SIGKILL; PID file preserved"
        )

    if os.path.exists(pid_path):
        os.remove(pid_path)

    state_path = get_inference_state_path(port)
    if os.path.exists(state_path):
        os.remove(state_path)

    utils.info("Inference server stopped")


def cmd_inference_status() -> None:
    """Show inference server status and health."""
    pid_path = get_inference_pid_path()
    pid = utils.read_pid(pid_path)

    utils.info("=== Engine Status ===")
    if pid is not None and is_inference_running():
        utils.info(f"Inference: running (PID {pid})")

        # Load state saved at startup
        state_path = get_inference_state_path()
        try:
            state = utils.load_yaml(state_path)
            port = state.get("port", 8081)

            # Check health
            if _is_inference_server_healthy(port):
                utils.info("Health: healthy")
            else:
                utils.info("Health: unresponsive")

            # Show startup config
            utils.info(f"Model: {state.get('model')}")
            utils.info(f"GPU layers: {state.get('gpu_layers')}")
            utils.info(f"Context: {state.get('context_size')}")
            utils.info(f"Port: {port}")

            gpu_type = utils.detect_gpu()
            utils.info(f"GPU: {gpu_type}")
        except FileNotFoundError:
            utils.info("Health: unknown (state file missing)")
    else:
        utils.info("Inference: stopped")
