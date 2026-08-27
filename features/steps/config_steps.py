import os
from pathlib import Path

from behave import given, then, when

from arkai import utils


@given("a valid .arkai.yaml file")  # ty: ignore[call-non-callable]
def step_valid_config(context):
    context.config_file = ".arkai.yaml"
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(context.config_file, config_data)


@given("a valid config file at {filepath}")  # ty: ignore[call-non-callable]
def step_valid_config_at_path(context, filepath):
    context.config_file = filepath
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    utils.save_yaml(filepath, config_data)


@given("an invalid .arkai.yaml file (missing required fields)")  # ty: ignore[call-non-callable]
def step_invalid_config(context):
    context.config_file = ".arkai.yaml"
    # Missing required agent.name
    config_data = {"inference": {"model": "test.gguf"}}
    utils.save_yaml(context.config_file, config_data)


@given("no .arkai.yaml file exists")  # ty: ignore[call-non-callable]
def step_no_config(context):
    if os.path.exists(".arkai.yaml"):
        os.remove(".arkai.yaml")


@given("an existing .arkai.yaml file")  # ty: ignore[call-non-callable]
def step_existing_config(context):
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(".arkai.yaml", config_data)


@when('I run "arkai config {cmd}"')  # ty: ignore[call-non-callable]
def step_run_config(context, cmd):
    parts = cmd.split()
    code, stdout, stderr = utils.run_command(["arkai", "config"] + parts)
    context.exit_code = code
    context.stdout = stdout
    context.stderr = stderr


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
