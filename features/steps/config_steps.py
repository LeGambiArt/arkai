import os

from behave import given, then, when

from arkai import utils


@given("a valid .arkai.yaml file")  # ty: ignore[call-non-callable]
def step_valid_config(context):
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }


@given("a valid config file at {filepath}")  # ty: ignore[call-non-callable]
def step_valid_config_at_path(context, filepath):
    context.config_file = filepath
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }


@given("an invalid .arkai.yaml file (missing required fields)")  # ty: ignore[call-non-callable]
def step_invalid_config(context):
    context.config_file = ".arkai.yaml"
    # Missing required inference.model (and inference.hf)
    context.config_data = {"agent": {"name": "opencode"}}


@given("no .arkai.yaml file exists")  # ty: ignore[call-non-callable]
def step_no_config(context):
    context.config_file = None
    context.config_data = None
    context.existing_files.discard(".arkai.yaml")


@when('I run "arkai config {cmd}"')  # ty: ignore[call-non-callable]
def step_run_config(context, cmd):
    parts = cmd.split()
    context.run_command(["arkai", "config"] + parts)


@then("the exit code is {code:d}")  # ty: ignore[call-non-callable]
def step_check_exit_code(context, code):
    assert context.exit_code == code, (
        f"Expected exit {code}, got {context.exit_code}\nstderr: {context.stderr}"
    )


@then('the output contains "{text}"')  # ty: ignore[call-non-callable]
def step_check_output(context, text):
    assert text in context.stdout, f"'{text}' not found in stdout:\n{context.stdout}"


@then('the error contains "{text}"')  # ty: ignore[call-non-callable]
def step_check_error(context, text):
    assert text in context.stderr, f"'{text}' not found in stderr:\n{context.stderr}"


@then('the output does not contain "{text}"')  # ty: ignore[call-non-callable]
def step_check_output_not_contains(context, text):
    assert text not in context.stdout, f"'{text}' unexpectedly found in stdout:\n{context.stdout}"


@then("the file .arkai.yaml exists with default values")  # ty: ignore[call-non-callable]
def step_check_config_created(context):
    assert os.path.exists(".arkai.yaml"), ".arkai.yaml was not created"
    cfg = utils.load_yaml(".arkai.yaml")
    assert "agent" in cfg, "Missing agent key"
    assert "inference" in cfg, "Missing inference key"
