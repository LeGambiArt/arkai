"""Step definitions for agent execution."""

import json
import os
import pty
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from behave import given, then, when

from arkai import agent, agent_opencode, agent_session, inference, utils, wtmcp

_PI_NODE_AVAILABLE = shutil.which("node") is not None
_POSIX_PTY_AVAILABLE = os.name == "posix"


@when("I cancel the pi agent installation")  # ty: ignore[call-non-callable]
def step_cancel_pi_installation(context):
    """Decline the Pi installation warning."""
    with (
        patch("builtins.input", return_value="n"),
        patch.object(agent.utils, "run_command") as run_command,
    ):
        agent.cmd_agent_install("pi")
    context.pi_install_run = run_command.called


@then("the pi agent installation was not run")  # ty: ignore[call-non-callable]
def step_pi_installation_not_run(context):
    """Assert that no Pi installation subprocess was started."""
    assert not context.pi_install_run


@given("a .arkai.yaml file with no model configured")  # ty: ignore[call-non-callable]
def step_config_no_model(context):
    """Create a config file that omits inference.model."""
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
    model: str | None = None,
    agent_name: str | None = None,
    no_start_inference: bool = False,
    strata_empty_model_list: bool = False,
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

    config_dir = "/tmp/mock-opencode-sessions"
    config_path = f"{config_dir}/opencode-config.json"
    session_record = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": None,
        "agent": "opencode",
        "working_directory": os.getcwd(),
        "native_session_id": None,
        "launch_options": {},
        "status": "starting",
    }

    popen_cmd: list = []

    if strata_empty_model_list:
        model_response = MagicMock()
        model_response.json.return_value = {"object": "list", "data": []}
        state_path = inference.get_inference_state_path()
        strata_state = {"backend": "strata", "model": "q2_0", "port": 9090}

        def load_strata_state(path):
            if str(path) == state_path:
                return strata_state
            raise FileNotFoundError(path)

        model_lookup_patch = patch.object(inference.requests, "get", return_value=model_response)
        state_load_patch = patch.object(utils, "load_yaml", side_effect=load_strata_state)
    else:
        model_lookup_patch = patch.object(
            inference, "get_inference_model", return_value="test.gguf"
        )
        state_load_patch = nullcontext()

    def capture_popen(cmd, **kwargs):
        popen_cmd.extend(cmd)
        return mock_proc

    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(inference, "is_inference_running", return_value=True),
        model_lookup_patch,
        state_load_patch,
        patch.object(wtmcp, "is_wtmcp_running", return_value=False),
        patch.object(wtmcp, "cmd_wtmcp_start") as mock_wtmcp_start,
        patch.object(utils, "resolve_binary", return_value="/usr/bin/fake-agent"),
        patch("subprocess.Popen", side_effect=capture_popen),
        patch.object(agent_opencode.os.path, "expanduser", return_value=config_dir),
        patch.object(
            agent_opencode.tempfile, "mkstemp", return_value=(999, config_path)
        ) as mkstemp,
        patch.object(agent_opencode.os, "close") as close,
        patch.object(agent_opencode.Path, "write_text") as write_text,
        patch.object(agent_opencode.Path, "chmod"),
        patch.object(agent_session, "create_session", return_value=session_record),
        patch.object(agent_session, "save_session"),
        patch.object(agent_session, "get_session_dir", return_value=Path(config_dir)),
    ):
        try:
            agent.cmd_agent(
                no_mcp=no_mcp,
                no_sandbox=no_sandbox,
                model=model,
                agent_name=agent_name,
                no_start_inference=no_start_inference,
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
            context.agent_config_text = write_text.call_args.args[0]

            mkstemp.assert_called_once_with(
                prefix="opencode-", suffix=".json", dir=agent_opencode.Path(config_dir)
            )
            close.assert_called_once_with(999)
            write_text.assert_called_once()

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


@when('I run "arkai agent start" against Strata with an empty model list in a TTY')  # ty: ignore[call-non-callable]
def step_run_agent_with_strata_empty_models(context):
    """Run an agent against Strata's valid but empty ``/v1/models`` response."""
    _run_agent_in_tty(
        context,
        no_mcp=True,
        no_sandbox=True,
        agent_name="opencode",
        no_start_inference=True,
        strata_empty_model_list=True,
    )


@then('the agent config uses Strata model "{model}"')  # ty: ignore[call-non-callable]
def step_agent_config_uses_strata_model(context, model):
    """The empty model listing falls back to the model selected for Strata."""
    assert model in context.agent_config_text


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


@given('a saved Arkai agent session named "{name}"')  # ty: ignore[call-non-callable]
def step_saved_arkai_session(context, name):
    """Provide a saved session record without touching the user's session store."""
    context.saved_agent_session = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": name,
        "agent": "opencode",
        "working_directory": os.getcwd(),
        "created_at": "2026-10-22T00:00:00+00:00",
        "updated_at": "2026-10-22T00:00:00+00:00",
        "status": "paused",
        "native_session_id": None,
        "launch_options": {
            "no_mcp": True,
            "no_sandbox": True,
            "cloud_services": [],
            "environment_keys": [],
        },
    }


@when("I list Arkai agent sessions")  # ty: ignore[call-non-callable]
def step_list_arkai_sessions(context):
    """Run the public list option through argparse without starting services."""
    infrastructure_calls = []
    with (
        patch.object(
            agent_session,
            "list_sessions",
            return_value=[context.saved_agent_session],
        ),
        patch.object(
            agent,
            "_agent_context",
            side_effect=lambda **_: infrastructure_calls.append(1),
        ),
    ):
        context.run_command(["arkai", "agent", "start", "--list-session"])
    context.infrastructure_calls = infrastructure_calls


@then("listing does not start agent infrastructure")  # ty: ignore[call-non-callable]
def step_listing_does_not_start_infrastructure(context):
    """The list flag must exit before agent or inference initialization."""
    assert not context.infrastructure_calls


@when('I resume the Arkai session "{name}" in a TTY')  # ty: ignore[call-non-callable]
def step_resume_arkai_session(context, name):
    """Run a named resume through the CLI while faking only native execution."""

    @contextmanager
    def fake_agent_context(**kwargs):
        yield agent.AgentContext(
            agent_name=kwargs["agent_name"],
            workdir=kwargs["sandbox_cwd"],
        )

    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(agent_session, "find_session", return_value=context.saved_agent_session),
        patch.object(agent_session, "save_session"),
        patch.object(agent, "_agent_context", side_effect=fake_agent_context),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        context.run_command(["arkai", "agent", "start", "--session", name])
    context.resumed_session_call = dispatch.call_args


@then("the named Arkai session was resumed")  # ty: ignore[call-non-callable]
def step_named_arkai_session_resumed(context):
    """The CLI resolves the name and dispatches the existing session as a resume."""
    assert context.resumed_session_call is not None
    assert context.resumed_session_call.kwargs["resume_session"] is True
    assert (
        context.resumed_session_call.kwargs["session"]["name"]
        == context.saved_agent_session["name"]
    )


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
            patch.object(inference, "is_inference_running", return_value=True),
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


@given("a fake Pi agent and local inference server for prompt testing")  # ty: ignore[call-non-callable]
def step_prepare_pi_prompt_test(context):
    """Prepare isolated files and a local inference endpoint for the prompt subprocess."""
    if not _PI_NODE_AVAILABLE or not _POSIX_PTY_AVAILABLE:
        context.scenario.skip("Pi prompt behavior requires Node.js and a POSIX pseudo-terminal")

    # This scenario uses real temporary files and child processes rather than the
    # in-process filesystem/service mocks installed by features/environment.py.
    for patch_name in (
        "exists_patch",
        "open_patch",
        "makedirs_patch",
        "mkdir_patch",
        "path_mkdir_patch",
    ):
        fixture_patch = getattr(context, patch_name, None)
        if fixture_patch:
            fixture_patch.stop()
            setattr(context, patch_name, None)

    temporary_directory = tempfile.TemporaryDirectory(prefix="arkai-pi-prompt-")
    context.pi_prompt_tempdir = temporary_directory
    root = Path(temporary_directory.name)
    home = root / "home"
    workdir = root / "work"
    home.mkdir()
    workdir.mkdir()

    agent_path = root / "fake-pi.js"
    agent_path.write_text(
        "#!/usr/bin/env node\n"
        "const args = process.argv.slice(2);\n"
        "console.error('MCP: Project servers blocked: wtmcp (blocked: "
        "project server approval required)');\n"
        "const answer = 'The speed of light in a vacuum is exactly **299,792,458 "
        "meters per second** (approximately $3 \\u00d7 10^8 m/s).';\n"
        "if (process.stdin.isTTY || !args.includes('--system-prompt')) "
        "process.stdout.write('<|thought');\n"
        "else { process.stdout.write(answer.slice(0, -1)); "
        "setTimeout(() => process.stdout.write(answer.slice(-1)), 20); }\n",
        encoding="utf-8",
    )
    agent_path.chmod(0o755)

    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            response = json.dumps({"data": [{"id": "test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    inference_server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    server_thread = threading.Thread(target=inference_server.serve_forever, daemon=True)
    server_thread.start()

    pid_dir = home / ".local" / "state" / "arkai"
    pid_dir.mkdir(parents=True)
    (pid_dir / "inference.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (workdir / ".arkai.yaml").write_text(
        json.dumps(
            {
                "agent": {"name": "pi", "path": str(agent_path), "mcp": True},
                "inference": {
                    "model": "test.gguf",
                    "port": inference_server.server_address[1],
                },
            }
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["XDG_CONFIG_HOME"] = str(home / ".config")
    repository_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (repository_root, environment.get("PYTHONPATH")) if value
    )

    context.pi_prompt_agent = agent_path
    context.pi_prompt_environment = environment
    context.pi_prompt_inference_server = inference_server
    context.pi_prompt_server_thread = server_thread
    context.pi_prompt_workdir = workdir
    context.pi_prompt_answer = (
        "The speed of light in a vacuum is exactly **299,792,458 meters per second** "
        "(approximately $3 × 10^8 m/s)."
    )


@when("I run a Pi prompt from a TTY with MCP disabled")  # ty: ignore[call-non-callable]
def step_run_pi_prompt_from_tty(context):
    """Run the real Arkai prompt CLI around a fake Pi process in a pseudo-terminal."""
    master_fd, slave_fd = pty.openpty()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from arkai.cli import main; main()",
                "agent",
                "prompt",
                "-a",
                "pi",
                "-I",
                "-M",
                "--no-sandbox",
                "What is the speed of light?",
            ],
            cwd=context.pi_prompt_workdir,
            env=context.pi_prompt_environment,
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        context.pi_prompt_exit_code = result.returncode
        context.pi_prompt_output = result.stdout
    finally:
        os.close(master_fd)
        os.close(slave_fd)
        context.pi_prompt_inference_server.shutdown()
        context.pi_prompt_server_thread.join(timeout=5)
        context.pi_prompt_inference_server.server_close()
        context.pi_prompt_tempdir.cleanup()


@then("the Pi prompt exits successfully")  # ty: ignore[call-non-callable]
def step_pi_prompt_exits_successfully(context):
    """Assert the Pi prompt subprocess completed without an error."""
    assert context.pi_prompt_exit_code == 0, context.pi_prompt_output


@then("the prompt output contains the complete final answer")  # ty: ignore[call-non-callable]
def step_prompt_output_contains_complete_answer(context):
    """Assert Arkai captured and printed Pi's full response, not a partial thought token."""
    assert context.pi_prompt_answer in context.pi_prompt_output
    assert "<|thought" not in context.pi_prompt_output


@then("the blocked project MCP warning is not displayed")  # ty: ignore[call-non-callable]
def step_project_mcp_warning_is_not_displayed(context):
    """Assert the no-MCP prompt suppresses Pi's expected project-server warning."""
    assert "MCP: Project servers blocked:" not in context.pi_prompt_output
