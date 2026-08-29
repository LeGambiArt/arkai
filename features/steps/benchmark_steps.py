"""Step definitions for benchmark feature tests."""

from behave import when

from arkai import utils


@when('I run benchmark command "{cmd}"')  # ty: ignore[call-non-callable]
def step_run_benchmark_command(context, cmd):
    """Run a benchmark command."""
    parts = cmd.split()
    code, stdout, stderr = utils.run_command(["arkai"] + parts)
    context.exit_code = code
    context.stdout = stdout
    context.stderr = stderr
