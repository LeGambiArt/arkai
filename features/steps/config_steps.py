import os
from unittest.mock import patch

from behave import given, then, when

from arkai import utils


@given("a valid .arkai.yaml file")  # ty: ignore[call-non-callable]
def step_valid_config(context):
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"backend": "llama-cpp", "model": "test.gguf"},
    }


@given("a valid config file at {filepath}")  # ty: ignore[call-non-callable]
def step_valid_config_at_path(context, filepath):
    context.config_file = filepath
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"backend": "llama-cpp", "model": "test.gguf"},
    }


@given("an invalid .arkai.yaml file (missing required fields)")  # ty: ignore[call-non-callable]
def step_invalid_config(context):
    context.config_file = ".arkai.yaml"
    # Missing required inference.model
    context.config_data = {"agent": {"name": "opencode"}}


@given("no .arkai.yaml file exists")  # ty: ignore[call-non-callable]
def step_no_config(context):
    context.config_file = None
    context.config_data = None
    context.existing_files.discard(".arkai.yaml")


@given("a valid .arkai.yaml file with inference context_size of {value}")  # ty: ignore[call-non-callable]
def step_valid_config_with_context_size(context, value):
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {
            "backend": "llama-cpp",
            "model": "test.gguf",
            "context_size": value,
        },
    }


@given('a valid .arkai.yaml file with inference backend "{backend}"')  # ty: ignore[call-non-callable]
def step_valid_config_with_backend(context, backend):
    """Create a config with an explicit inference backend."""
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {"backend": backend, "model": "test.gguf"},
    }


@given("a valid Strata inference config")  # ty: ignore[call-non-callable]
def step_valid_strata_inference_config(context):
    """Configure a Strata catalog model on a dedicated local test port."""
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {
            "backend": "strata",
            "model": "q2_0",
            "port": 9090,
            "context_size": 4096,
            "gpu_layers": -1,
            "startup_timeout": 1,
        },
    }


@given("a valid .arkai.yaml file with inference profiles")  # ty: ignore[call-non-callable]
def step_valid_config_with_multiple_models(context):
    """Create a config containing named model profiles and sampling settings."""
    context.config_file = ".arkai.yaml"
    context.config_data = {
        "agent": {"name": "opencode"},
        "inference": {
            "backend": "llama-cpp",
            "model": "balanced",
            "profiles": {
                "fast": {"model": "fast.gguf", "temperature": 0.2},
                "balanced": {"model": "balanced.gguf", "top_p": 0.9},
            },
        },
    }


@given('the config editor sets "{key}" to {value} and saves')  # ty: ignore[call-non-callable]
def step_config_editor_sets_and_saves(context, key, value):
    """Install a scripted editor that applies one change and saves."""
    from arkai import config_editor
    from arkai.config_editor import nested_set

    def script(target, merged, user, project, target_path):
        try:
            value_ = int(value)
        except ValueError:
            value_ = value.strip('"')
        nested_set(target, key, value_)
        return True

    patch_obj = patch.object(config_editor, "run_editor", side_effect=script)
    patch_obj.start()
    context.editor_start_config = (
        dict(context.config_data) if context.config_data is not None else None
    )
    context.add_cleanup(patch_obj.stop)


@given('the config editor adds inference profile "{name}" with model "{model}" and saves')  # ty: ignore[call-non-callable]
def step_config_editor_adds_profile_and_saves(context, name, model):
    """Install a scripted editor that creates one profile and saves."""
    from arkai import config_editor
    from arkai.config_editor import nested_set

    def script(target, merged, user, project, target_path):
        nested_set(target, f"inference.profiles.{name}", {"model": model})
        return True

    patch_obj = patch.object(config_editor, "run_editor", side_effect=script)
    patch_obj.start()
    context.editor_start_config = (
        dict(context.config_data) if context.config_data is not None else None
    )
    context.add_cleanup(patch_obj.stop)


@given(
    'the config editor adds cloud custom provider "{provider_id}" with base URL "{url}" and saves'
)  # ty: ignore[call-non-callable]
def step_config_editor_adds_cloud_provider_and_saves(context, provider_id, url):
    """Install a scripted editor that creates one custom cloud provider."""
    from arkai import config_editor
    from arkai.config_editor import nested_set

    def script(target, merged, user, project, target_path):
        nested_set(
            target,
            f"agent.cloud.custom.providers.{provider_id}",
            {"base_url": url, "default_model": "org/model-id"},
        )
        return True

    patch_obj = patch.object(config_editor, "run_editor", side_effect=script)
    patch_obj.start()
    context.editor_start_config = (
        dict(context.config_data) if context.config_data is not None else None
    )
    context.add_cleanup(patch_obj.stop)


@then('the file .arkai.yaml has cloud custom provider "{provider_id}" with base URL "{url}"')  # ty: ignore[call-non-callable]
def step_check_config_cloud_custom_provider(context, provider_id, url):
    assert context.config_data is not None, ".arkai.yaml was not created"
    provider = context.config_data["agent"]["cloud"]["custom"]["providers"][provider_id]
    assert provider["base_url"] == url


@given('the config editor sets custom key "{key}" to "{raw_value}" and saves')  # ty: ignore[call-non-callable]
def step_config_editor_sets_custom_key_and_saves(context, key, raw_value):
    """Install a scripted editor that adds one custom key and saves."""
    from arkai import config_editor
    from arkai.config_editor import nested_set, parse_custom_value

    ok, _message, value = parse_custom_value(raw_value)
    assert ok

    def script(target, merged, user, project, target_path):
        nested_set(target, key, value)
        return True

    patch_obj = patch.object(config_editor, "run_editor", side_effect=script)
    patch_obj.start()
    context.editor_start_config = (
        dict(context.config_data) if context.config_data is not None else None
    )
    context.add_cleanup(patch_obj.stop)


@then('the file .arkai.yaml has custom key "{key}" with value "{raw_value}"')  # ty: ignore[call-non-callable]
def step_check_config_custom_key(context, key, raw_value):
    from arkai.config_editor import nested_get, parse_custom_value

    assert context.config_data is not None, ".arkai.yaml was not created"
    ok, _message, expected = parse_custom_value(raw_value)
    assert ok
    assert nested_get(context.config_data, key) == expected


@given("the config editor aborts without saving")  # ty: ignore[call-non-callable]
def step_config_editor_aborts(context):
    """Install a scripted editor that quits without saving."""
    from arkai import config_editor

    patch_obj = patch.object(config_editor, "run_editor", return_value=False)
    patch_obj.start()
    context.editor_start_config = (
        dict(context.config_data) if context.config_data is not None else None
    )
    context.add_cleanup(patch_obj.stop)


@when('I run "arkai config {cmd}"')  # ty: ignore[call-non-callable]
def step_run_config(context, cmd):
    parts = cmd.split()
    context.run_command(["arkai", "config"] + parts)


@then("the exit code is {code:d}")  # ty: ignore[call-non-callable]
def step_check_exit_code(context, code):
    assert context.exit_code == code, (
        f"Expected exit {code}, got {context.exit_code}\nstderr: {context.stderr}"
    )


@then("the exit code is not {code:d}")  # ty: ignore[call-non-callable]
def step_check_exit_code_not(context, code):
    assert context.exit_code != code, (
        f"Expected exit not {code}, got {context.exit_code}\nstderr: {context.stderr}"
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
    assert cfg["inference"].get("backend") == "llama-cpp", "Missing default inference backend"


@then("the file .arkai.yaml is unchanged")  # ty: ignore[call-non-callable]
def step_check_config_unchanged(context):
    data = context.config_data
    assert data is not None, ".arkai.yaml content was lost"
    assert data == context.editor_start_config, (
        f".arkai.yaml changed: {context.editor_start_config} -> {data}"
    )


@then("the file .arkai.yaml has inference port {port:d}")  # ty: ignore[call-non-callable]
def step_check_config_inference_port(context, port):
    assert context.config_data is not None, ".arkai.yaml was not created"
    assert context.config_data["inference"]["port"] == port


@then('the file .arkai.yaml has agent name "{name}"')  # ty: ignore[call-non-callable]
def step_check_config_agent_name(context, name):
    assert context.config_data is not None, ".arkai.yaml was not created"
    assert context.config_data["agent"]["name"] == name


@then('the file {path} has agent name "{name}"')  # ty: ignore[call-non-callable]
def step_check_file_agent_name(context, path, name):
    if path == "<config_home>/arkai.yaml":
        path = os.path.join(utils.get_config_home(), "arkai.yaml")
    assert str(path) in context.existing_files, f"{path} was not created"
    cfg = utils.load_yaml(path)
    assert cfg["agent"]["name"] == name


@then("the file .arkai.yaml does not exist")  # ty: ignore[call-non-callable]
def step_check_config_not_created(context):
    assert ".arkai.yaml" not in context.existing_files, ".arkai.yaml was unexpectedly created"


@then('the file .arkai.yaml has inference profile "{name}" with model "{model}"')  # ty: ignore[call-non-callable]
def step_check_config_inference_profile(context, name, model):
    assert context.config_data is not None, ".arkai.yaml was not created"
    profile = context.config_data["inference"]["profiles"][name]
    assert profile["model"] == model


@then('the file .arkai.yaml has inference model "{model}"')  # ty: ignore[call-non-callable]
def step_check_config_inference_model(context, model):
    assert context.config_data is not None, ".arkai.yaml was not created"
    assert context.config_data["inference"]["model"] == model
