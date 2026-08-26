"""Step definitions for agent execution."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

from behave import when

from aitool import agent, engine, wtmcp


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
