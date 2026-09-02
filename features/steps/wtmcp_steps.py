"""Step definitions for wtmcp plugin management features."""

import contextlib
import sys
from io import StringIO

from behave import given, then, when

from arkai import utils


@given("a valid .arkai.yaml file with plugins {plugins}")  # ty: ignore[call-non-callable]
def step_valid_config_with_plugins(context, plugins):
    """Create a valid config file with plugins."""
    context.config_file = ".arkai.yaml"
    plugin_list = [p.strip() for p in plugins.split(",")]
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
        "wtmcp": {"plugins": plugin_list},
    }
    context.existing_files.add(context.config_file)


@when('I run "arkai wtmcp {cmd}"')  # ty: ignore[call-non-callable]
def step_run_arkai_wtmcp(context, cmd):
    """Run arkai wtmcp command and capture output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    # Mock wtmcp check output with discovered plugins
    mock_wtmcp_output = """wtmcp 0.1.8
discovered plugins: 12
  - google-drive v0.1.0 (/path/to/plugin)
  - jira v0.2.0 (/path/to/plugin)
  - google-docs v0.3.0 (/path/to/plugin)
  - google-gmail v0.1.0 (/path/to/plugin)
  - github v0.1.0 (/path/to/plugin)
  - workspace v0.1.0 (/path/to/plugin)
  - terminal v0.1.0 (/path/to/plugin)

tool discovery: progressive
"""

    try:
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from arkai import utils as arkai_utils
        from arkai import wtmcp

        parts = cmd.split()

        patches = [
            mock_patch("subprocess.run"),
            mock_patch("subprocess.Popen"),
            mock_patch.object(arkai_utils, "resolve_binary", return_value="/usr/bin/wtmcp"),
            mock_patch.object(arkai_utils, "is_port_in_use", return_value=False),
            mock_patch("arkai.wtmcp.time.sleep"),
        ]
        with contextlib.ExitStack() as stack:
            mock_run, mock_popen, *_ = [stack.enter_context(p) for p in patches]

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_wtmcp_output
            mock_run.return_value = mock_result

            mock_proc = MagicMock()
            mock_proc.pid = 9998
            mock_popen.return_value = mock_proc

            if parts[0] == "list":
                port = None
                if len(parts) > 1 and parts[1] == "--port" and len(parts) > 2:
                    port = int(parts[2])
                wtmcp.cmd_wtmcp_list(port)
            elif parts[0] == "enable" and len(parts) > 1:
                plugin_name = parts[1]
                wtmcp.cmd_wtmcp_enable(plugin_name)
            elif parts[0] == "disable" and len(parts) > 1:
                plugin_name = parts[1]
                wtmcp.cmd_wtmcp_disable(plugin_name)
            elif parts[0] == "start":
                path = None
                port = None
                enable_plugins = []
                disable_plugins = []

                i = 1
                while i < len(parts):
                    if parts[i] == "--path" and i + 1 < len(parts):
                        path = parts[i + 1]
                        i += 2
                    elif parts[i] == "--port" and i + 1 < len(parts):
                        port = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "--enable" and i + 1 < len(parts):
                        enable_plugins.append(parts[i + 1])
                        i += 2
                    elif parts[i] == "--disable" and i + 1 < len(parts):
                        disable_plugins.append(parts[i + 1])
                        i += 2
                    else:
                        i += 1

                wtmcp.cmd_wtmcp_start(
                    path,
                    port,
                    enable_plugins if enable_plugins else None,
                    disable_plugins if disable_plugins else None,
                )
                # Add the PID file with proper path
                if port is not None:
                    pid_path = wtmcp.get_wtmcp_pid_path(port)
                    context.existing_files.add(pid_path)
            elif parts[0] == "stop":
                port = None
                if len(parts) > 1 and parts[1] == "--port" and len(parts) > 2:
                    port = int(parts[2])
                wtmcp.cmd_wtmcp_stop(port)
                # Remove PID file from existing files with proper path
                if port is not None:
                    pid_path = wtmcp.get_wtmcp_pid_path(port)
                    context.existing_files.discard(pid_path)
            elif parts[0] == "status":
                port = None
                if len(parts) > 1 and parts[1] == "--port" and len(parts) > 2:
                    port = int(parts[2])
                wtmcp.cmd_wtmcp_status(port)
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


@then("the project config has plugin {plugin}")  # ty: ignore[call-non-callable]
def step_check_plugin_enabled(context, plugin):
    """Check that plugin is in project config."""
    config_file = context.config_file
    config = utils.load_yaml(config_file)
    plugins = config.get("wtmcp", {}).get("plugins", [])
    plugin = plugin.strip('"')  # Remove quotes from feature syntax
    assert plugin in plugins, f"Plugin '{plugin}' not found in config. Plugins: {plugins}"


@then("the project config does not have plugin {plugin}")  # ty: ignore[call-non-callable]
def step_check_plugin_disabled(context, plugin):
    """Check that plugin is not in project config."""
    config_file = context.config_file
    config = utils.load_yaml(config_file)
    plugins = config.get("wtmcp", {}).get("plugins", [])
    plugin = plugin.strip('"')  # Remove quotes from feature syntax
    assert plugin not in plugins, f"Plugin '{plugin}' found in config but should be removed"


@given("a valid .arkai.yaml file with invalid wtmcp path")  # ty: ignore[call-non-callable]
def step_valid_config_with_invalid_wtmcp(context):
    """Create config with invalid wtmcp path."""
    from arkai import utils as arkai_utils

    context.config_file = ".arkai.yaml"
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
        "wtmcp": {"plugins": ["workspace"], "path": "/nonexistent/wtmcp"},
    }
    arkai_utils.save_yaml(context.config_file, config_data)


@given("the wtmcp server is running")  # ty: ignore[call-non-callable]
def step_wtmcp_server_running(context):
    """Mock wtmcp server as running."""
    from unittest.mock import patch

    from arkai import wtmcp as wtmcp_module

    # Mock is_wtmcp_running to return True for default port
    default_port = 8080
    context.wtmcp_running_patch = patch.object(wtmcp_module, "is_wtmcp_running", return_value=True)
    context.wtmcp_running_patch.start()

    # Add full PID file path to existing files
    pid_path = wtmcp_module.get_wtmcp_pid_path(default_port)
    context.existing_files.add(pid_path)


@then("the wtmcp server is running")  # ty: ignore[call-non-callable]
def step_check_wtmcp_running(context):
    """Check that wtmcp server is running."""
    from arkai import wtmcp as wtmcp_module

    # Check for any running instances (tests may use different ports)
    assert wtmcp_module.is_wtmcp_running(None), "wtmcp server is not running"


@then("the wtmcp server is not running")  # ty: ignore[call-non-callable]
def step_check_wtmcp_not_running(context):
    """Check that wtmcp server is not running."""
    from arkai import wtmcp as wtmcp_module

    # Check that no instances are running
    assert not wtmcp_module.is_wtmcp_running(None), "wtmcp server is still running"
