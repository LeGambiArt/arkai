"""Step definitions for agent execution."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

from behave import given, then, when

from aitool import agent, engine, utils, wtmcp


@given("a valid .aitool.yaml file with mcp disabled")  # ty: ignore[call-non-callable]
def step_valid_config_mcp_disabled(context):
    """Create a valid config file with agent.mcp set to false."""
    config_data = {
        "agent": {"name": "opencode", "mcp": False},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(".aitool.yaml", config_data)


def _run_agent_in_tty(context, no_mcp: bool = False) -> None:
    """Run agent with a mocked TTY and capture output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.wait.return_value = 0

    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(engine, "is_inference_running", return_value=True),
        patch.object(wtmcp, "is_wtmcp_running", return_value=False),
        patch.object(wtmcp, "cmd_wtmcp_start") as mock_wtmcp_start,
        patch.object(utils, "resolve_binary", return_value="/usr/bin/fake-agent"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        try:
            agent.cmd_agent(no_mcp=no_mcp)
        except SystemExit as e:
            context.exit_code = e.code if e.code else 1
        except Exception as e:
            context.exit_code = 1
            stderr_capture.write(str(e))
        finally:
            context.wtmcp_start_called = mock_wtmcp_start.called

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    context.stdout = stdout_capture.getvalue()
    context.stderr = stderr_capture.getvalue()


@when('I run "aitool agent" with "--no-mcp" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_no_mcp_flag(context):
    """Run aitool agent with --no-mcp in a TTY."""
    _run_agent_in_tty(context, no_mcp=True)


@when('I run "aitool agent" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_tty(context):
    """Run aitool agent in a TTY."""
    _run_agent_in_tty(context, no_mcp=False)


@then("wtmcp was not started")  # ty: ignore[call-non-callable]
def step_wtmcp_not_started(context):
    """Assert that wtmcp.cmd_wtmcp_start was not called."""
    assert not context.wtmcp_start_called, (
        "wtmcp.cmd_wtmcp_start was called but should not have been"
    )


@when('I run "aitool agent" with stdin piped')  # ty: ignore[call-non-callable]
def step_run_agent_piped(context):
    """Run aitool agent with stdin piped (non-interactive)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    try:
        # Mock service checks and Popen to prevent actual service/agent launches
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with (
            patch.object(engine, "is_inference_running", return_value=True),
            patch.object(wtmcp, "is_wtmcp_running", return_value=True),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            agent.cmd_agent()
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
