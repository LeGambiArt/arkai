"""Inference server lifecycle management."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from arkai import config, utils
from arkai.inference_backend import get_backend
from arkai.inference_strata import StrataBackend


def exec_cmd(args: dict | None = None) -> None:
    """Select 'inference' command to execute."""
    match args.inference_cmd:  # ty: ignore[unresolved-attribute]
        case "start":
            cmd_inference_start(
                args.model,  # ty: ignore[unresolved-attribute]
                args.gpu_layers,  # ty: ignore[unresolved-attribute]
                args.context,  # ty: ignore[unresolved-attribute]
                args.port,  # ty: ignore[unresolved-attribute]
                backend=args.backend,  # ty: ignore[unresolved-attribute]
                allow_provisioning=args.allow_provisioning,  # ty: ignore[unresolved-attribute]
            )
        case "stop":
            cmd_inference_stop(args.port)  # ty: ignore[unresolved-attribute]
        case "status":
            cmd_inference_status(args.port)  # ty: ignore[unresolved-attribute]


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
    start_parser.add_argument("--backend", help="Override inference backend from config")
    start_parser.add_argument(
        "--allow-provisioning",
        action="store_true",
        help="Allow Strata to download or install model and engine files",
    )
    stop_parser = inference_subparsers.add_parser("stop", help="Stop inference server")
    stop_parser.add_argument("--port", type=int, help="Stop the explicitly selected server port")
    status_parser = inference_subparsers.add_parser("status", help="Show inference server status")
    status_parser.add_argument("--port", type=int, help="Show the explicitly selected server port")


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


def get_inference_log_path(port: int | None = None) -> str:
    """Return the file containing stdout and stderr from an inference server."""
    pid_dir = utils.get_pid_dir()
    filename = "inference.log" if port is None else f"inference-{port}.log"
    return os.path.join(pid_dir, filename)


def is_inference_running(port: int | None = None) -> bool:
    """Check if the inference server for ``port`` is running."""
    state_path = get_inference_state_path(port)
    try:
        state = utils.load_yaml(state_path)
    except FileNotFoundError:
        state = {}
    if state.get("backend") == "strata":
        profile = state.get("strata_profile")
        strata_path = state.get("strata_path", "strata")
        if not isinstance(profile, str):
            return False
        try:
            code, output, error = utils.run_command(
                StrataBackend().build_status_command(strata_path), timeout=30
            )
            if code != 0:
                raise RuntimeError(error.strip() or "Strata status command failed")
            if StrataBackend.profile_is_running(output, profile):
                return True
        except RuntimeError:
            pass
        return get_live_strata_state(port) is not None

    pid_path = get_inference_pid_path(port)
    pid = utils.read_pid(pid_path)
    if pid is not None and utils.is_process_running(pid):
        return True
    return get_live_strata_state(port) is not None


def get_inference_model(port: int | None = None) -> str:
    """Return the active model ID, using Arkai's Strata selection when needed.

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
        model_list = response.json()["data"]
    except requests.RequestException as error:
        raise RuntimeError(
            f"Unable to determine the running inference model on port {server_port}: {error}"
        ) from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Inference server returned an invalid model list on port {server_port}"
        ) from error

    if model_list == []:
        # Strata serves an OpenAI-compatible API but currently returns an empty
        # /v1/models list. The loaded Strata profile is the authoritative model.
        model_id = _strata_fallback_model(server_port)
        if model_id:
            return model_id
        raise RuntimeError(f"Inference server returned an invalid model list on port {server_port}")

    try:
        model_id = model_list[0]["id"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"Inference server returned an invalid model list on port {server_port}"
        ) from error

    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"Inference server returned an invalid model ID on port {server_port}")
    return model_id


def _strata_fallback_model(port: int) -> str | None:
    """Return the model selected for a Strata service whose model list is empty."""
    state_ports = [port, None] if port != 8081 else [None, port]
    for state_port in state_ports:
        try:
            state = utils.load_yaml(get_inference_state_path(state_port))
        except FileNotFoundError:
            continue
        if (
            state.get("backend") == "strata"
            and state.get("port", 8081) == port
            and isinstance(state.get("model"), str)
            and state["model"]
        ):
            return state["model"]

    live_state = get_live_strata_state(port)
    if live_state and isinstance(live_state.get("model"), str):
        return live_state["model"]

    cfg = config.load_config()
    if config.get_config_value(cfg, "inference.backend", "llama-cpp") != "strata":
        return None
    try:
        model, _sampling = config.resolve_model(cfg)
    except RuntimeError:
        return None
    return model or None


def _is_inference_server_healthy(port: int, path: str = "/v1/models") -> bool:
    """Return whether the inference server responds successfully to a health check."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}{path}", timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def is_inference_server_available(port: int) -> bool:
    """Check whether any compatible inference server is serving on ``port``.

    Unlike :func:`is_inference_running`, this does not require Arkai to own the
    process or have a PID file for it.
    """
    return _is_inference_server_healthy(port)


def _display_startup_output(log_file) -> None:
    """Forward newly written server output while Arkai waits for readiness."""
    output = log_file.read()
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()


def _load_inference_state(port: int | None = None) -> dict:
    """Load state for one inference instance, returning an empty mapping if absent."""
    try:
        return utils.load_yaml(get_inference_state_path(port))
    except FileNotFoundError:
        return {}


def _strata_cli_path(state: dict | None = None) -> str | None:
    """Resolve the Strata CLI used to discover live profiles without Arkai state."""
    if isinstance(state, dict) and isinstance(state.get("strata_path"), str):
        return state["strata_path"]

    cfg = config.load_config()
    inference_cfg = cfg.get("inference", {})
    if isinstance(inference_cfg, dict):
        configured = inference_cfg.get("path")
        if configured and inference_cfg.get("backend") == "strata":
            return str(configured)

    from arkai import tools

    managed = tools.get_managed_strata_binary()
    if managed is not None:
        return str(managed)
    if isinstance(inference_cfg, dict) and inference_cfg.get("backend") == "strata":
        return "strata"
    return None


def get_live_strata_state(port: int | None = None) -> dict | None:
    """Discover a ready Strata profile by port, even after Arkai tracking was detached."""
    saved_state = _load_inference_state(port)
    try:
        cfg = config.load_config()
        server_port = (
            port or saved_state.get("port") or config.get_config_value(cfg, "inference.port", 8081)
        )
        strata_path = _strata_cli_path(saved_state)
        if strata_path is None:
            return None
        backend = StrataBackend()
        code, output, _error = utils.run_command(
            backend.build_status_command(strata_path), timeout=30
        )
        if code != 0:
            return None
        launch = backend.running_launch_for_port(output, server_port)
    except (RuntimeError, OSError, TypeError, ValueError):
        return None
    if launch is None:
        return None

    strata_config: dict = {}
    config_path = launch.get("config")
    if isinstance(config_path, str):
        try:
            loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                strata_config = loaded
        except (OSError, ValueError):
            pass

    saved_model = saved_state.get("model")
    if (
        saved_state.get("backend") == "strata"
        and saved_state.get("port", server_port) == server_port
        and isinstance(saved_model, str)
        and saved_model
    ):
        model = saved_model
    else:
        model = strata_config.get("model_name") or launch.get("name") or "strata"

    context_size = saved_state.get("context_size")
    if not isinstance(context_size, int):
        args = strata_config.get("args", [])
        if isinstance(args, list):
            for index, argument in enumerate(args[:-1]):
                if argument == "--max-context":
                    try:
                        context_size = int(args[index + 1])
                    except (TypeError, ValueError):
                        context_size = None
                    break

    return {
        "backend": "strata",
        "model": model,
        "gpu_layers": "managed by Strata",
        "context_size": context_size,
        "port": server_port,
        "log": strata_config.get("log", get_inference_log_path(port)),
        "strata_path": strata_path,
        "strata_profile": launch.get("profile"),
        "strata_owned": False,
    }


def _run_strata_command(command: list[str], description: str, timeout: int | None = 30) -> str:
    """Run a Strata control command and return stdout or raise its reported error."""
    try:
        code, output, error = utils.run_command(command, timeout=timeout)
    except RuntimeError as error:
        raise RuntimeError(f"Unable to {description}: {error}") from error
    if code != 0:
        detail = error.strip() or output.strip() or f"exit code {code}"
        raise RuntimeError(f"Unable to {description}: {detail}")
    return output


def _strata_response_error(response: dict, description: str) -> str:
    """Format an error object returned by Strata's JSON CLI interface."""
    error = response.get("error")
    if isinstance(error, dict):
        detail = error.get("message")
    else:
        detail = error
    return f"Strata {description} failed" + (f": {detail}" if detail else "")


def _stop_strata_profile(backend: StrataBackend, path: str, profile: str) -> None:
    """Stop one known Strata launch and verify that exact profile is gone."""
    _run_strata_command(
        backend.build_stop_command(path, profile), f"stop Strata profile {profile}", timeout=180
    )
    status = _run_strata_command(
        backend.build_status_command(path), "check Strata service status", timeout=30
    )
    if backend.profile_is_running(status, profile):
        raise RuntimeError(f"Strata profile {profile} is still running; Arkai state was preserved")


def _cmd_inference_start_strata(
    backend: StrataBackend,
    command: list[str],
    model: str,
    port: int,
    gpu_layers: int,
    context_size: int,
    startup_timeout: int,
    instance_port: int | None,
    allow_provisioning: bool,
) -> None:
    """Plan and launch Strata, tracking only the specific service it reports."""
    plan_command = backend.build_plan_command(command)
    plan_output = _run_strata_command(plan_command, "check the Strata launch plan", timeout=120)
    plan_response = backend.parse_json_output(plan_output, "checking the launch plan")
    if plan_response.get("ok") is not True or not isinstance(plan_response.get("plan"), dict):
        raise RuntimeError(_strata_response_error(plan_response, "launch plan"))

    plan = plan_response["plan"]
    download = plan.get("download")
    if not isinstance(download, dict):
        raise RuntimeError("Strata's launch plan did not report provisioning requirements")
    amount = download.get("gb")
    needs_provisioning = (
        not isinstance(amount, (int, float))
        or amount > 0
        or download.get("engine_usable") is not True
        or download.get("engine_missing") is True
    )
    warnings = plan.get("warnings", [])
    if isinstance(warnings, list) and any(
        isinstance(warning, str) and "older than this model needs" in warning
        for warning in warnings
    ):
        needs_provisioning = True

    if needs_provisioning and not allow_provisioning:
        download_text = download.get("text")
        details = f" {download_text}" if isinstance(download_text, str) else ""
        raise RuntimeError(
            "Strata needs to download or update model/engine files."
            f"{details} Re-run with `arkai inference start --allow-provisioning` to allow this."
        )

    if allow_provisioning:
        # Strata's own CLI uses --yes for the corresponding consent.
        command = [*command, "--yes"]

    log_path = get_inference_log_path(instance_port)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Plan command: {shlex.join(plan_command)}\n")
        log_file.write(plan_output + "\n\n")
        log_file.write(f"Launch command: {shlex.join(command)}\n\n")
        log_file.flush()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        log_file.write(stdout)
        if stderr:
            log_file.write("\n--- stderr ---\n")
            log_file.write(stderr)

    if process.returncode != 0:
        try:
            response = backend.parse_json_output(stdout, "starting the service")
        except RuntimeError:
            response = {}
        if response:
            raise RuntimeError(_strata_response_error(response, "launch"))
        detail = stderr.strip() or stdout.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"Strata launch failed: {detail}. See log: {log_path}")

    response = backend.parse_json_output(stdout, "starting the service")
    if response.get("ok") is not True:
        raise RuntimeError(_strata_response_error(response, "launch"))
    if response.get("serving", True) is not True:
        message = response.get("message") or "Strata did not start an inference service"
        raise RuntimeError(str(message))

    profile = response.get("profile")
    reused = response.get("reused", False)
    if (
        not isinstance(profile, str)
        or not profile.startswith("launch-")
        or not profile.removeprefix("launch-")
        or not isinstance(reused, bool)
    ):
        raise RuntimeError(
            "Strata started but returned invalid service ownership information. "
            f"Check `strata launch --status`; Arkai log: {log_path}"
        )

    state = {
        "backend": "strata",
        "model": model,
        "gpu_layers": "managed by Strata" if gpu_layers == -1 else gpu_layers,
        "context_size": context_size,
        "port": port,
        "log": log_path,
        "strata_path": command[0],
        "strata_profile": profile,
        "strata_owned": not reused,
    }
    utils.save_yaml(get_inference_state_path(instance_port), state)

    ownership = "reused" if reused else "started by Arkai"
    utils.info(f"Strata launch returned profile {profile} ({ownership})")
    utils.info(f"Waiting for Strata API on port {port}...")
    for _ in range(startup_timeout):
        if _is_inference_server_healthy(port):
            utils.info(f"Strata inference server ready on port {port}")
            return
        time.sleep(1)

    if not reused:
        try:
            _stop_strata_profile(backend, command[0], profile)
        except RuntimeError as error:
            raise RuntimeError(
                f"Strata API did not become ready on port {port}; failed to stop its "
                f"Arkai-owned profile ({error}). See log: {log_path}"
            ) from error
    state_path = get_inference_state_path(instance_port)
    if os.path.exists(state_path):
        os.remove(state_path)
    raise RuntimeError(f"Strata API did not become ready on port {port}. See log: {log_path}")


def cmd_inference_start(
    model: str | None = None,
    gpu_layers: int | None = None,
    context_size: int | None = None,
    port: int | None = None,
    backend: str | None = None,
    allow_provisioning: bool = False,
) -> None:
    """Start the configured inference server.

    Args:
        model: Override model from config
        gpu_layers: Override GPU layers from config
        context_size: Override context size from config
        port: Override port from config
        backend: Override inference backend from config
        allow_provisioning: Allow Strata to download or update model and engine files

    Raises:
        RuntimeError: If no model specified and none available in config
    """
    # Load config
    cfg = config.load_config()
    # A port supplied on the CLI identifies a separate instance. Otherwise use
    # the default PID/state paths, even when the config changes the port.
    instance_port = port

    # Apply CLI overrides before validation so they can satisfy required fields.
    if model:
        cfg["inference"]["model"] = model
    if gpu_layers is not None:
        cfg["inference"]["gpu_layers"] = gpu_layers
    if context_size is not None:
        cfg["inference"]["context_size"] = context_size
    if port is not None:
        cfg["inference"]["port"] = port
    if backend is not None:
        cfg["inference"]["backend"] = backend

    if not config.validate_config(cfg, require_model=True):
        raise RuntimeError("Invalid configuration")

    port = config.get_config_value(cfg, "inference.port", 8081)
    model, sampling = config.resolve_model(cfg)
    context_size_val = config.get_config_value(cfg, "inference.context_size", 65536)
    gpu_layers_val = config.get_config_value(cfg, "inference.gpu_layers", -1)
    selected_backend = get_backend(config.get_config_value(cfg, "inference.backend", "llama-cpp"))

    # Check only the requested instance, allowing explicit ports to coexist. A
    # running Strata service is reused only when its profile/settings match; a
    # different service on the port is never silently replaced.
    if is_inference_running(instance_port):
        state = _load_inference_state(instance_port)
        if state.get("backend") == "strata":
            if (
                selected_backend.name == "strata"
                and state.get("model") == model
                and state.get("context_size") == context_size_val
                and state.get("port") == port
            ):
                utils.info("Strata inference service already running")
                return
            raise RuntimeError(
                f"A different Arkai-managed Strata profile is already running on port {port}; "
                "stop it explicitly before changing its settings"
            )
        if selected_backend.name == "strata":
            if state:
                raise RuntimeError(
                    f"Another Arkai-managed inference backend is already running on port {port}"
                )
            if get_live_strata_state(port) is None:
                raise RuntimeError(f"Another inference service is already running on port {port}")
            # An untracked Strata launch may be the same service the CLI is about
            # to reuse. Let Strata confirm or refuse without stopping it.
        else:
            utils.info("Inference server already running")
            return

    # Strata's own launcher safely detects matching services and refuses to
    # replace different or non-Strata services. Other backends keep Arkai's
    # existing port check.
    if selected_backend.name != "strata" and utils.is_port_in_use(port):
        raise RuntimeError(f"Port {port} already in use")

    if selected_backend.name == "strata":
        utils.info("Strata will select a supported GPU automatically")
    else:
        gpu_type = utils.detect_gpu()
        utils.info(f"Detected GPU: {gpu_type}")

    # Build the backend-specific server command. The manager owns lifecycle and
    # readiness handling; Strata delegates service lifecycle to its own CLI.
    configured_path = cfg.get("inference", {}).get("path")
    selected_backend.check_environment()
    default_path = {"mlx": "mlx_lm.server", "splash": "splash", "strata": "strata"}.get(
        selected_backend.name, "llama-server"
    )
    if selected_backend.name == "strata" and configured_path is None:
        from arkai import tools

        managed_strata = tools.get_managed_strata_binary()
        if managed_strata is not None:
            default_path = str(managed_strata)
    cmd = selected_backend.build_command(
        configured_path if configured_path is not None else default_path,
        model,
        port,
        gpu_layers_val,
        context_size_val,
        path_is_configured=configured_path is not None,
        sampling=sampling,
    )

    startup_timeout = config.get_config_value(cfg, "inference.startup_timeout", 600)
    if selected_backend.name == "strata":
        if not isinstance(selected_backend, StrataBackend):
            raise RuntimeError(
                "The registered Strata backend does not implement service lifecycle control"
            )
        _cmd_inference_start_strata(
            selected_backend,
            cmd,
            model,
            port,
            gpu_layers_val,
            context_size_val,
            startup_timeout,
            instance_port,
            allow_provisioning,
        )
        return

    if selected_backend.name == "splash":
        from arkai import model_splash

        cmd.extend(model_splash.serve_arguments(cfg))
    utils.info(f"Starting inference server on port {port}...")
    log_path = get_inference_log_path(instance_port)
    utils.info(f"Inference server logs: {log_path}")

    # Disable SIGINT to ensure process and PID file are both created
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        # Keep backend diagnostics available after the CLI detaches from the server.
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Command: {shlex.join(cmd)}\n\n")
            log_file.flush()
            startup_log_offset = log_file.tell()
            process_options: dict = {"stdout": log_file, "stderr": subprocess.STDOUT}
            if selected_backend.name == "splash":
                process_options["env"] = model_splash.splash_environment(cfg, create=True)
            proc = subprocess.Popen(cmd, **process_options)

        # Write PID and state
        pid_path = get_inference_pid_path(instance_port)
        utils.write_pid(pid_path, proc.pid)
    finally:
        # Always restore SIGINT
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Save engine state (config used at startup)
    state = {
        "backend": selected_backend.name,
        "model": model,
        "gpu_layers": gpu_layers_val,
        "context_size": context_size_val,
        "port": port,
        "log": log_path,
    }
    utils.save_yaml(get_inference_state_path(instance_port), state)

    # Wait for server to be ready
    utils.info("Waiting for inference server...")
    startup_error = "Inference server failed to start"
    health_path = "/ready" if selected_backend.name == "splash" else "/v1/models"
    with open(log_path, encoding="utf-8", errors="replace") as startup_log:
        startup_log.seek(startup_log_offset)
        utils.info("Server output during startup:")
        for _ in range(startup_timeout):
            _display_startup_output(startup_log)
            if proc.poll() is not None:
                _display_startup_output(startup_log)
                startup_error = (
                    f"Inference server exited before becoming ready (code {proc.returncode}). "
                    "See the server log for details"
                )
                break
            if _is_inference_server_healthy(port, health_path):
                _display_startup_output(startup_log)
                utils.info(f"Inference server ready on port {port}")
                return
            time.sleep(1)
        _display_startup_output(startup_log)

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
    raise RuntimeError(f"{startup_error}. See server log: {log_path}")


def cmd_inference_stop(port: int | None = None) -> None:
    """Stop the inference server for ``port``."""
    pid_path = get_inference_pid_path(port)
    state_path = get_inference_state_path(port)
    state = _load_inference_state(port)

    if state.get("backend") == "strata":
        profile = state.get("strata_profile")
        strata_path = state.get("strata_path")
        if not isinstance(profile, str) or not isinstance(strata_path, str):
            raise RuntimeError(
                "Strata state is missing its profile or CLI path; refusing an unsafe stop"
            )

        if state.get("strata_owned") is not True:
            if os.path.exists(state_path):
                os.remove(state_path)
            utils.info(f"Detached from reused Strata profile {profile}; it was left running")
            return

        _stop_strata_profile(StrataBackend(), strata_path, profile)
        if os.path.exists(state_path):
            os.remove(state_path)
        if os.path.exists(pid_path):
            os.remove(pid_path)
        utils.info(f"Arkai-owned Strata profile {profile} stopped")
        return

    if not state:
        live_state = get_live_strata_state(port)
        if live_state is not None:
            utils.info(
                f"Strata profile {live_state.get('strata_profile')} is running but not owned "
                "by Arkai; "
                "leaving it running"
            )
            return

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

    if os.path.exists(state_path):
        os.remove(state_path)

    utils.info("Inference server stopped")


def cmd_inference_status(port: int | None = None) -> None:
    """Show inference server status and health."""
    pid_path = get_inference_pid_path(port)
    pid = utils.read_pid(pid_path)
    state = _load_inference_state(port)
    if state.get("backend") != "strata":
        state = get_live_strata_state(port) or state

    utils.info("=== Engine Status ===")
    if state.get("backend") == "strata":
        server_port = state.get("port", 8081)
        running = is_inference_running(port)
        utils.info(
            f"Inference: running (Strata profile {state.get('strata_profile')})"
            if running
            else "Inference: stopped (Strata profile is not running)"
        )
        health = "healthy" if _is_inference_server_healthy(server_port) else "unresponsive"
        utils.info(f"Health: {health}")
        utils.info("Backend: strata")
        utils.info(f"Model: {state.get('model')}")
        utils.info(f"GPU layers: {state.get('gpu_layers')}")
        utils.info(f"Context: {state.get('context_size')}")
        utils.info(f"Port: {server_port}")
        ownership = "Arkai" if state.get("strata_owned") else "reused; not owned by Arkai"
        utils.info(f"Ownership: {ownership}")
        utils.info(f"Log: {state.get('log', get_inference_log_path(port))}")
        utils.info("GPU: selected by Strata")
        return

    if pid is not None and is_inference_running(port):
        utils.info(f"Inference: running (PID {pid})")

        # Load state saved at startup
        state_path = get_inference_state_path(port)
        try:
            state = utils.load_yaml(state_path)
            server_port = state.get("port", 8081)

            # Check health
            if _is_inference_server_healthy(server_port):
                utils.info("Health: healthy")
            else:
                utils.info("Health: unresponsive")

            # Show startup config
            utils.info(f"Backend: {state.get('backend', 'unknown')}")
            utils.info(f"Model: {state.get('model')}")
            utils.info(f"GPU layers: {state.get('gpu_layers')}")
            utils.info(f"Context: {state.get('context_size')}")
            utils.info(f"Port: {server_port}")
            utils.info(f"Log: {state.get('log', get_inference_log_path(port))}")

            gpu_type = utils.detect_gpu()
            utils.info(f"GPU: {gpu_type}")
        except FileNotFoundError:
            utils.info("Health: unknown (state file missing)")
    else:
        utils.info("Inference: stopped")
