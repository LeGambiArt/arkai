"""Inference server lifecycle management."""

import os
import subprocess
import time
from typing import Optional

from aitool import config, utils


def get_inference_pid_path() -> str:
    """Return path to inference server PID file."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "inference.pid")


def get_inference_state_path() -> str:
    """Return path to inference server state file (stores startup config)."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, "inference.state")


def is_inference_running() -> bool:
    """Check if inference server is running."""
    pid_path = get_inference_pid_path()
    pid = utils.read_pid(pid_path)
    if pid is None:
        return False

    # Check if process still exists
    try:
        code, _, _ = utils.run_command(["kill", "-0", str(pid)])
        return code == 0
    except RuntimeError:
        return False


def cmd_engine_start(
    model: Optional[str] = None,
    gpu_layers: Optional[int] = None,
    context_size: Optional[int] = None,
    port: Optional[int] = None,
) -> None:
    """Start inference server (llama-server).

    Args:
        model: Override model from config
        gpu_layers: Override GPU layers from config
        context_size: Override context size from config
        port: Override port from config
    """
    # Load config
    cfg = config.load_config()
    if not config.validate_config(cfg):
        raise RuntimeError("Invalid configuration")

    # Override with CLI args if provided
    if model:
        cfg["inference"]["model"] = model
    if gpu_layers is not None:
        cfg["inference"]["gpu_layers"] = gpu_layers
    if context_size is not None:
        cfg["inference"]["context_size"] = context_size
    if port is not None:
        cfg["inference"]["port"] = port

    # Check if already running
    if is_inference_running():
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
    model_file = config.get_config_value(cfg, "inference.model")
    hf_model = config.get_config_value(cfg, "inference.hf")

    model_path: Optional[str] = None
    if model_file:
        data_home = utils.get_data_home()
        model_path = os.path.join(data_home, "models", model_file)  # ty: ignore[no-matching-overload]
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

    # Start in background
    proc = subprocess.Popen(  # ty: ignore[no-matching-overload]
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write PID and state
    pid_path = get_inference_pid_path()
    utils.write_pid(pid_path, proc.pid)

    # Save engine state (config used at startup)
    state = {
        "model": model_file or hf_model,
        "gpu_layers": gpu_layers_val,
        "context_size": context_size_val,
        "port": port,
    }
    utils.save_yaml(get_inference_state_path(), state)

    # Wait for server to be ready
    utils.info("Waiting for inference server...")
    max_retries = 30
    for i in range(max_retries):
        try:
            code, _, _ = utils.run_command(["curl", "-sf", f"http://127.0.0.1:{port}/v1/models"])
            if code == 0:
                utils.info(f"Inference server ready on port {port}")
                return
        except RuntimeError:
            pass
        time.sleep(1)

    raise RuntimeError("Inference server failed to start")


def cmd_engine_stop() -> None:
    """Stop inference server."""
    pid_path = get_inference_pid_path()
    pid = utils.read_pid(pid_path)

    if pid is None:
        utils.info("Inference server not running")
        return

    utils.info(f"Stopping inference server (PID {pid})...")
    utils.kill_process(pid)

    # Poll for process termination (up to 10 seconds)
    for _ in range(20):
        try:
            code, _, _ = utils.run_command(["kill", "-0", str(pid)])
            if code != 0:
                break
        except RuntimeError:
            break
        time.sleep(0.5)

    if os.path.exists(pid_path):
        os.remove(pid_path)

    # Clean up state file
    state_path = get_inference_state_path()
    if os.path.exists(state_path):
        os.remove(state_path)

    utils.info("Inference server stopped")


def cmd_engine_status() -> None:
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
            try:
                code, _, _ = utils.run_command(
                    ["curl", "-sf", f"http://127.0.0.1:{port}/v1/models"]
                )
                if code == 0:
                    utils.info("Health: healthy")
                else:
                    utils.info("Health: unresponsive")
            except RuntimeError:
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
