"""Step definitions for hf: model prefix feature."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

from behave import given, when

from arkai import agent, engine, utils, wtmcp


@given("subprocess calls are mocked for inference start")  # ty: ignore[call-non-callable]
def step_mock_inference_subprocess(context):
    """Mock Popen and run_command so the inference start loop completes instantly."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345

    context.inference_popen_patch = patch("subprocess.Popen", return_value=mock_proc)
    context.inference_run_command_patch = patch.object(
        utils, "run_command", return_value=(0, "", "")
    )
    context.inference_popen_patch.start()
    context.inference_run_command_patch.start()


@when(  # ty: ignore[call-non-callable]
    'I run "arkai agent start" with "-m hf:ibm-granite/granite-4.1-8b-GGUF" in a TTY'
)
def step_run_agent_hf_model(context):
    """Run agent start with hf: prefix model in a TTY."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture, stderr_capture = StringIO(), StringIO()
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
        patch.object(utils, "resolve_binary", return_value="/usr/bin/fake-agent"),
        patch("subprocess.Popen", return_value=mock_proc),
    ):
        try:
            agent.cmd_agent(
                model="hf:ibm-granite/granite-4.1-8b-GGUF",
                no_sandbox=True,
                no_mcp=True,
            )
        except SystemExit as e:
            context.exit_code = e.code if e.code else 1
        except Exception as e:
            context.exit_code = 1
            stderr_capture.write(str(e))

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    context.stdout = stdout_capture.getvalue()
    context.stderr = stderr_capture.getvalue()
