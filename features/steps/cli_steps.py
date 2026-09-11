"""Step definitions for top-level CLI behavior."""

import shlex

from behave import then, when


@when('I invoke "{command}"')  # ty: ignore[call-non-callable]
def step_run_cli_command(context, command):
    """Run an arkai CLI command through the Behave command harness."""
    context.run_command(shlex.split(command))


@then('the error does not contain "{text}"')  # ty: ignore[call-non-callable]
def step_check_error_not_contains(context, text):
    """Assert that stderr does not contain the specified text."""
    assert text not in context.stderr, f"'{text}' unexpectedly found in stderr:\n{context.stderr}"
