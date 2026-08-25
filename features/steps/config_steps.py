from behave import given, when, then
import os
from pathlib import Path
from aitool import utils


@given("a valid .aitool.yaml file")
def step_valid_config(context):
    context.config_file = ".aitool.yaml"
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(context.config_file, config_data)


@given("a valid config file at {filepath}")
def step_valid_config_at_path(context, filepath):
    context.config_file = filepath
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    utils.save_yaml(filepath, config_data)


@given("an invalid .aitool.yaml file (missing required fields)")
def step_invalid_config(context):
    context.config_file = ".aitool.yaml"
    # Missing required agent.name
    config_data = {"inference": {"model": "test.gguf"}}
    utils.save_yaml(context.config_file, config_data)


@given("no .aitool.yaml file exists")
def step_no_config(context):
    if os.path.exists(".aitool.yaml"):
        os.remove(".aitool.yaml")


@given("an existing .aitool.yaml file")
def step_existing_config(context):
    config_data = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }
    utils.save_yaml(".aitool.yaml", config_data)


@when('I run "aitool config {cmd}"')
def step_run_config(context, cmd):
    parts = cmd.split()
    code, stdout, stderr = utils.run_command(["aitool", "config"] + parts)
    context.exit_code = code
    context.stdout = stdout
    context.stderr = stderr


@then("the exit code is {code:d}")
def step_check_exit_code(context, code):
    assert (
        context.exit_code == code
    ), f"Expected exit {code}, got {context.exit_code}\nstderr: {context.stderr}"


@then('the output contains "{text}"')
def step_check_output(context, text):
    assert (
        text in context.stdout
    ), f"'{text}' not found in stdout:\n{context.stdout}"


@then('the error contains "{text}"')
def step_check_error(context, text):
    assert (
        text in context.stderr
    ), f"'{text}' not found in stderr:\n{context.stderr}"


@then("the file .aitool.yaml exists with default values")
def step_check_config_created(context):
    assert os.path.exists(".aitool.yaml"), ".aitool.yaml was not created"
    cfg = utils.load_yaml(".aitool.yaml")
    assert "agent" in cfg, "Missing agent key"
    assert "inference" in cfg, "Missing inference key"
