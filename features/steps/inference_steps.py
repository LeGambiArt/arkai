"""Step definitions for inference management features."""

import json
import subprocess
import sys
from io import StringIO
from unittest.mock import patch

from behave import given, then, when

from arkai import inference, tools, utils
from features.environment import setup_open_mock


@given("port {port:d} is in use")  # ty: ignore[call-non-callable]
def step_port_in_use(context, port):
    """Mock the port as in use so the test is independent of real port state."""
    if hasattr(context, "port_in_use_patch") and context.port_in_use_patch:
        context.port_in_use_patch.stop()
    context.port_in_use_patch = patch.object(utils, "is_port_in_use", return_value=True)
    context.port_in_use_patch.start()


@given("MLX-LM is not importable")  # ty: ignore[call-non-callable]
def step_mlx_lm_not_importable(context):
    """Make the MLX-LM package import fail for the startup preflight."""
    context.mlx_import_patch = patch(
        "arkai.inference_mlx.importlib.import_module",
        side_effect=ModuleNotFoundError("No module named 'mlx_lm'"),
    )
    context.mlx_import_patch.start()


@given("Arkai-managed Strata is installed for inference")  # ty: ignore[call-non-callable]
def step_managed_strata_installed_for_inference(context):
    """Set up deterministic Strata CLI responses without invoking a real service."""
    context.strata_test_enabled = True
    context.strata_binary = tools.get_strata_binary()
    context.strata_download_gb = 0
    context.strata_launch_reused = False
    context.strata_profile_running = False
    context.strata_commands = []
    context.strata_launch_command = None


@given("the Strata launch plan requires model downloads")  # ty: ignore[call-non-callable]
def step_strata_model_requires_download(context):
    context.strata_download_gb = 65


@given("Strata reports an identical profile already running")  # ty: ignore[call-non-callable]
def step_strata_profile_is_reused(context):
    context.strata_launch_reused = True
    context.strata_profile_running = True


@given("the inference server is not running")  # ty: ignore[call-non-callable]
def step_server_not_running(context):
    """Mock the inference server as not running."""
    # Patch is_inference_running to return True for this scenario
    context.is_running_patch = patch.object(inference, "is_inference_running", return_value=False)
    context.is_running_patch.start()
    context.is_running_mock = context.is_running_patch

    # Mark inference as not running in context for open() mock
    context.inference_running = False
    # Recreate open mock to reflect new state
    setup_open_mock(context)


@given("the inference server is running")  # ty: ignore[call-non-callable]
def step_server_running(context):
    """Mock the inference server as running."""
    # Patch is_inference_running to return True for this scenario
    context.is_running_patch = patch.object(inference, "is_inference_running", return_value=True)
    context.is_running_patch.start()
    context.is_running_mock = context.is_running_patch

    # Mark inference as running in context for open() mock
    context.inference_running = True
    # Recreate open mock to reflect new state
    setup_open_mock(context)

    # Add inference.pid to existing files since the server is running
    pid_path = inference.get_inference_pid_path()
    context.existing_files.add(pid_path)


@then("the inference PID file does not exist")  # ty: ignore[call-non-callable]
def step_pid_file_not_exist(context):
    """Assert that the inference PID file does not exist."""
    pid_path = inference.get_inference_pid_path()
    assert pid_path not in context.existing_files, f"PID file should not exist: {pid_path}"


@then("the inference PID file exists")  # ty: ignore[call-non-callable]
def step_pid_file_exists(context):
    """Assert that the inference PID file exists."""
    pid_path = inference.get_inference_pid_path()
    assert pid_path in context.existing_files, f"PID file should exist: {pid_path}"


@when('I run "arkai inference {cmd}"')  # ty: ignore[call-non-callable]
def step_run_arkai_inference(context, cmd):
    """Run arkai inference command and capture output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    try:
        parts = cmd.split()
        if parts[0] == "start":
            # Parse start command args
            model = None
            gpu_layers = None
            context_size = None
            port = None
            backend = None
            allow_provisioning = False

            i = 1
            while i < len(parts):
                if parts[i] == "--model" and i + 1 < len(parts):
                    model = parts[i + 1]
                    i += 2
                elif parts[i] == "--gpu-layers" and i + 1 < len(parts):
                    gpu_layers = int(parts[i + 1])
                    i += 2
                elif parts[i] == "--context" and i + 1 < len(parts):
                    context_size = int(parts[i + 1])
                    i += 2
                elif parts[i] == "--port" and i + 1 < len(parts):
                    port = int(parts[i + 1])
                    i += 2
                elif parts[i] == "--backend" and i + 1 < len(parts):
                    backend = parts[i + 1]
                    i += 2
                elif parts[i] == "--allow-provisioning":
                    allow_provisioning = True
                    i += 1
                else:
                    i += 1

            if getattr(context, "strata_test_enabled", False):
                _run_strata_backend(
                    context,
                    "start",
                    model=model,
                    gpu_layers=gpu_layers,
                    context_size=context_size,
                    port=port,
                    backend=backend,
                    allow_provisioning=allow_provisioning,
                )
            else:
                inference.cmd_inference_start(
                    model,
                    gpu_layers,
                    context_size,
                    port,
                    backend=backend,
                    allow_provisioning=allow_provisioning,
                )
        elif parts[0] == "stop":
            if getattr(context, "strata_test_enabled", False):
                _run_strata_backend(context, "stop")
            else:
                inference.cmd_inference_stop()
            # After stop, mark inference as not running
            context.inference_running = False
            # Recreate open mock to reflect new state
            setup_open_mock(context)
            # Remove PID file from existing files since it's stopped
            pid_path = inference.get_inference_pid_path()
            context.existing_files.discard(pid_path)
            if hasattr(context, "is_running_patch") and context.is_running_patch:
                context.is_running_patch.stop()
                context.is_running_patch = None
        elif parts[0] == "status":
            inference.cmd_inference_status()
        else:
            context.exit_code = 1
            stderr_capture.write(f"Unknown command: {parts[0]}")
    except SystemExit as e:
        context.exit_code = e.code if e.code else 1
    except Exception as e:
        context.exit_code = 1
        stderr_capture.write(str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        context.stdout = stdout_capture.getvalue()
        context.stderr = stderr_capture.getvalue()


def _run_strata_backend(context, action: str, **overrides) -> None:
    """Run the Strata backend with a managed binary and fake CLI/service responses."""

    def run_command(command, timeout=30, **_kwargs):
        context.strata_commands.append(command)
        if "--plan" in command:
            plan = {
                "ok": True,
                "plan": {
                    "download": {
                        "gb": context.strata_download_gb,
                        "engine_usable": True,
                        "engine_missing": False,
                        "text": f"About {context.strata_download_gb} GB to download",
                    },
                    "warnings": [],
                },
            }
            return 0, json.dumps(plan), ""
        if "--stop" in command:
            context.strata_profile_running = False
            return 0, "Stopped the profile", ""
        if "--status" in command:
            running = []
            if context.strata_profile_running:
                running.append({"profile": "launch-q2-0-4k", "state": "ready", "port": 9090})
            return 0, json.dumps({"ok": True, "running": running}), ""
        raise AssertionError(f"Unexpected Strata control command: {command} (timeout={timeout})")

    def launch(command, **_kwargs):
        context.strata_launch_command = command
        context.strata_profile_running = True
        response = {
            "ok": True,
            "serving": True,
            "profile": "launch-q2-0-4k",
            "reused": context.strata_launch_reused,
            "urls": ["http://127.0.0.1:9090"],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    with (
        patch.object(tools, "_is_managed_strata_installed", return_value=True),
        patch.object(utils, "resolve_binary", side_effect=lambda path, **_kwargs: str(path)),
        patch.object(utils, "run_command", side_effect=run_command),
        patch.object(inference.subprocess, "run", side_effect=launch),
        patch.object(inference, "_is_inference_server_healthy", return_value=True),
        patch.object(inference, "is_inference_running", return_value=False)
        if action == "start"
        else patch.object(inference, "is_inference_running"),
    ):
        if action == "start":
            inference.cmd_inference_start(**overrides)
        else:
            inference.cmd_inference_stop()


@then("the Strata launch used Arkai's managed executable")  # ty: ignore[call-non-callable]
def step_strata_used_managed_executable(context):
    assert context.strata_launch_command
    assert context.strata_launch_command[0] == str(context.strata_binary)


@then("Strata received download approval")  # ty: ignore[call-non-callable]
def step_strata_received_download_approval(context):
    assert context.strata_launch_command
    assert "--yes" in context.strata_launch_command


@then("Arkai records the Strata profile as owned")  # ty: ignore[call-non-callable]
def step_strata_profile_owned(context):
    assert context.config_data["strata_profile"] == "launch-q2-0-4k"
    assert context.config_data["strata_owned"] is True


@then("Arkai records the Strata profile as reused")  # ty: ignore[call-non-callable]
def step_strata_profile_reused(context):
    assert context.config_data["strata_profile"] == "launch-q2-0-4k"
    assert context.config_data["strata_owned"] is False


@then("no Strata launch was started")  # ty: ignore[call-non-callable]
def step_no_strata_launch_started(context):
    assert context.strata_launch_command is None


@then("Arkai did not stop the reused Strata profile")  # ty: ignore[call-non-callable]
def step_reused_strata_profile_not_stopped(context):
    assert context.strata_profile_running is True
    assert not any("--stop" in command for command in context.strata_commands)
