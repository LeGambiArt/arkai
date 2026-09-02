"""Step definitions for engine management features."""

import sys
from io import StringIO
from unittest.mock import patch

from behave import given, then, when

from arkai import engine, utils
from features.environment import setup_open_mock


@given("port {port:d} is in use")  # ty: ignore[call-non-callable]
def step_port_in_use(context, port):
    """Mock the port as in use so the test is independent of real port state."""
    if hasattr(context, "port_in_use_patch") and context.port_in_use_patch:
        context.port_in_use_patch.stop()
    context.port_in_use_patch = patch.object(utils, "is_port_in_use", return_value=True)
    context.port_in_use_patch.start()


@given("the inference server is not running")  # ty: ignore[call-non-callable]
def step_server_not_running(context):
    """Mock the inference server as not running."""
    # Patch is_inference_running to return True for this scenario
    context.is_running_patch = patch.object(engine, "is_inference_running", return_value=False)
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
    context.is_running_patch = patch.object(engine, "is_inference_running", return_value=True)
    context.is_running_patch.start()
    context.is_running_mock = context.is_running_patch

    # Mark inference as running in context for open() mock
    context.inference_running = True
    # Recreate open mock to reflect new state
    setup_open_mock(context)

    # Add inference.pid to existing files since the server is running
    pid_path = engine.get_inference_pid_path()
    context.existing_files.add(pid_path)


@then("the inference PID file does not exist")  # ty: ignore[call-non-callable]
def step_pid_file_not_exist(context):
    """Assert that the inference PID file does not exist."""
    pid_path = engine.get_inference_pid_path()
    assert pid_path not in context.existing_files, f"PID file should not exist: {pid_path}"


@then("the inference PID file exists")  # ty: ignore[call-non-callable]
def step_pid_file_exists(context):
    """Assert that the inference PID file exists."""
    pid_path = engine.get_inference_pid_path()
    assert pid_path in context.existing_files, f"PID file should exist: {pid_path}"


@when('I run "arkai inference {cmd}"')  # ty: ignore[call-non-callable]
def step_run_arkai_engine(context, cmd):
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
                else:
                    i += 1

            engine.cmd_engine_start(model, gpu_layers, context_size, port)
        elif parts[0] == "stop":
            engine.cmd_engine_stop()
            # After stop, mark inference as not running
            context.inference_running = False
            # Recreate open mock to reflect new state
            setup_open_mock(context)
            # Remove PID file from existing files since it's stopped
            pid_path = engine.get_inference_pid_path()
            context.existing_files.discard(pid_path)
            if hasattr(context, "is_running_patch") and context.is_running_patch:
                context.is_running_patch.stop()
                context.is_running_patch = None
        elif parts[0] == "status":
            engine.cmd_engine_status()
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
