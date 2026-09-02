"""Step definitions for agent execution."""

import sys
from io import StringIO
from typing import Optional
from unittest.mock import MagicMock, patch

from behave import given, then, when

from arkai import agent, engine, utils, wtmcp


@given("a .arkai.yaml file with no model configured")  # ty: ignore[call-non-callable]
def step_config_no_model(context):
    """Create a config file that omits inference.model and inference.hf."""
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {},
    }
    utils.save_yaml(".arkai.yaml", config_data)


@given("a valid .arkai.yaml file with mcp disabled")  # ty: ignore[call-non-callable]
def step_valid_config_mcp_disabled(context):
    """Create a valid config file with agent.mcp set to false."""
    config_data = {
        "agent": {"name": "opencode", "mcp": False},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(".arkai.yaml", config_data)


def _run_agent_in_tty(
    context,
    no_mcp: bool = False,
    no_sandbox: bool = False,
    model: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    """Run agent with a mocked TTY and capture output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.wait.return_value = 0

    popen_cmd: list = []

    def capture_popen(cmd, **kwargs):
        popen_cmd.extend(cmd)
        return mock_proc

    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(engine, "is_inference_running", return_value=True),
        patch.object(wtmcp, "is_wtmcp_running", return_value=False),
        patch.object(wtmcp, "cmd_wtmcp_start") as mock_wtmcp_start,
        patch.object(utils, "resolve_binary", return_value="/usr/bin/fake-agent"),
        patch("subprocess.Popen", side_effect=capture_popen),
    ):
        try:
            agent.cmd_agent(
                no_mcp=no_mcp, no_sandbox=no_sandbox, model=model, agent_name=agent_name
            )
        except SystemExit as e:
            context.exit_code = e.code if e.code else 1
        except Exception as e:
            context.exit_code = 1
            stderr_capture.write(str(e))
        finally:
            context.wtmcp_start_called = mock_wtmcp_start.called
            # sandbox was used if the command starts with the arapuca prefix ("fake-agent run ...")
            context.sandbox_used = len(popen_cmd) >= 2 and popen_cmd[1] == "run"

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    context.stdout = stdout_capture.getvalue()
    context.stderr = stderr_capture.getvalue()


@given("a valid .arkai.yaml file with sandbox disabled")  # ty: ignore[call-non-callable]
def step_valid_config_sandbox_disabled(context):
    """Create a valid config file with sandbox.disable set to true."""
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
        "sandbox": {"disable": True},
    }
    utils.save_yaml(".arkai.yaml", config_data)


@when('I run "arkai agent start" with "-m test-model.gguf" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_with_model_flag(context):
    """Run arkai agent start with -m model override in a TTY."""
    _run_agent_in_tty(context, model="test-model.gguf")


@when(  # ty: ignore[call-non-callable]
    'I run "arkai agent start -m ibm-granite/granite-4.1-8b-GGUF'
    ' -a opencode --no-sandbox --no-mcp" in a TTY'
)
def step_run_agent_granite_no_sandbox_no_mcp(context):
    """Run arkai agent start with HF model, agent, no-sandbox, and no-mcp flags in a TTY."""
    _run_agent_in_tty(
        context,
        model="ibm-granite/granite-4.1-8b-GGUF",
        agent_name="opencode",
        no_sandbox=True,
        no_mcp=True,
    )


@when('I run "arkai agent start" with "--no-mcp" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_no_mcp_flag(context):
    """Run arkai agent start with --no-mcp in a TTY."""
    _run_agent_in_tty(context, no_mcp=True)


@when('I run "arkai agent start" with "--no-sandbox" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_no_sandbox_flag(context):
    """Run arkai agent start with --no-sandbox in a TTY."""
    _run_agent_in_tty(context, no_sandbox=True)


@when('I run "arkai agent start" in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_tty(context):
    """Run arkai agent start in a TTY."""
    _run_agent_in_tty(context)


@then("wtmcp was not started")  # ty: ignore[call-non-callable]
def step_wtmcp_not_started(context):
    """Assert that wtmcp.cmd_wtmcp_start was not called."""
    assert not context.wtmcp_start_called, (
        "wtmcp.cmd_wtmcp_start was called but should not have been"
    )


@then("the agent was not sandboxed")  # ty: ignore[call-non-callable]
def step_agent_not_sandboxed(context):
    """Assert that the agent was launched without arapuca."""
    assert not context.sandbox_used, "Agent was launched in sandbox but should not have been"


@when('I run "arkai agent start" with stdin piped')  # ty: ignore[call-non-callable]
def step_run_agent_piped(context):
    """Run arkai agent start with stdin piped (non-interactive)."""
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
            patch.object(sys.stdin, "isatty", return_value=False),
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
