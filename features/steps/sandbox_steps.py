"""Step implementations for sandbox feature tests."""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

from behave import given, when


def _run_arkai(context, argv):
    """Run an arkai CLI command in-process with captured output."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 0

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            sys.argv = ["arkai"] + argv
            from arkai import cli

            cli.main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 1

    context.stdout = stdout_buf.getvalue()
    context.stderr = stderr_buf.getvalue()
    context.exit_code = exit_code


@given("a sandbox profile without a path and a custom root path")  # ty: ignore[call-non-callable]
def step_profile_without_path_and_custom_root(context):
    """Configure a profile that should inherit the root Arapuca executable."""
    context.config_data = {
        "sandbox": {
            "path": "/custom/arapuca",
            "profiles": {
                "inherited": {
                    "memory_mb": 1024,
                    "cpus": 1,
                    "pids": 64,
                    "timeout": 0,
                }
            },
        }
    }


@when('I run "arkai sandbox {cmd}"')  # ty: ignore[call-non-callable]
def step_run_sandbox_command(context, cmd):
    """Run an arkai sandbox command and capture output."""
    _run_arkai(context, ["sandbox"] + cmd.split())


@given("I create a sandbox profile {name} with cpus={cpus} and memory={memory}")  # ty: ignore[call-non-callable]
def step_create_profile(context, name, cpus, memory):
    """Create a sandbox profile with specified parameters."""
    name = name.strip("'\"")
    _run_arkai(context, ["sandbox", "create", name, "--cpus", cpus, "--memory", memory])
    if context.exit_code != 0:
        raise AssertionError(f"Failed to create profile: {context.stderr}")


@given('I run "arkai sandbox {cmd}"')  # ty: ignore[call-non-callable]
def step_given_run_sandbox_command(context, cmd):
    """Given step: run an arkai sandbox command (used for setup)."""
    _run_arkai(context, ["sandbox"] + cmd.split())
