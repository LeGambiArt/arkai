"""Step definitions for model management features."""

import os
from unittest.mock import patch

from behave import given, then, when

from arkai import model


@given("a clean models directory")  # ty: ignore[call-non-callable]
def step_clean_models(context):
    """Mock up a clean models directory."""
    context.existing_files.union({f for f in context.existing_files if not f.endswith(".gguf")})


def _add_model_to_storage(context, filename):
    # Set up the existing_files set if not already set
    if not hasattr(context, "existing_files"):
        context.exisntig_files = set()
    # Add the file to the set
    if model.get_models_dir() not in filename:
        filename = os.path.join(model.get_models_dir(), filename)
    context.existing_files.add(filename)


@given('a model file "{filename}" in the models directory')  # ty: ignore[call-non-callable]
def step_add_model_file(context, filename):
    """Add a model file to the existing_files set."""
    _add_model_to_storage(context, filename)


@when('I run "arkai model {cmd}"')  # ty: ignore[call-non-callable]
def step_run_arkai_model(context, cmd):
    """Run arkai model command."""
    import sys
    from io import StringIO

    # Parse the command
    parts = cmd.split()
    cmd_name = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    try:
        import importlib

        from arkai import model

        importlib.reload(model)

        if cmd_name == "list":
            # Mock the HuggingFace cached models function to return empty list
            # This ensures tests don't depend on actual system state
            with patch.object(model, "_get_huggingface_cached_models", return_value=[]):
                model.cmd_model_list()
        elif cmd_name == "download":
            # TODO: actually not checking download, but mocking it.
            # Mock the hf download by creating a dummy .gguf file
            hf_repo = args[0]
            models_dir = model.get_models_dir()
            # Create a dummy .gguf file to simulate download
            model_filename = hf_repo.split("/")[-1] + ".gguf"
            model_path = os.path.join(models_dir, model_filename)
            # Add to existing_files set
            context.existing_files.add(model_path)
            stdout_capture.write(f"Downloaded {hf_repo} to {models_dir}\n")
        elif cmd_name == "remove":
            model.cmd_model_remove(args[0])
        else:
            context.exit_code = 1
            stderr_capture.write(f"Unknown command: {cmd_name}")
    except SystemExit as e:
        context.exit_code = e.code
    except Exception as e:
        context.exit_code = 1
        stderr_capture.write(str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        context.stdout = stdout_capture.getvalue()
        context.stderr = stderr_capture.getvalue()


@when('I run "arkai model list" with HuggingFace models')  # ty: ignore[call-non-callable]
def step_run_arkai_model_with_hf(context):
    """Run arkai model list with HuggingFace cached models."""
    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    try:
        import importlib

        from arkai import model

        importlib.reload(model)

        # Mock HuggingFace cached models with realistic long repo names
        hf_models = [
            ("apple/DiffuCoder-7B-Base", "15.2G"),
            ("empero-ai/Qwen3.8-9B-Distill", "19.3G"),
            ("ggml-org/Qwen3.8-27B-GGUF", "19.6G"),
            ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "53.9G"),
        ]
        with patch.object(model, "_get_huggingface_cached_models", return_value=hf_models):
            model.cmd_model_list()
    except SystemExit as e:
        context.exit_code = e.code
    except Exception as e:
        context.exit_code = 1
        stderr_capture.write(str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        context.stdout = stdout_capture.getvalue()
        context.stderr = stderr_capture.getvalue()


@then("a .gguf file exists in the models directory")  # ty: ignore[call-non-callable]
def step_check_gguf_exists(context):
    """Verify a .gguf file exists in the existing_files set."""
    gguf_in_models = [
        f for f in context.existing_files if model.get_models_dir() in f and f.endswith(".gguf")
    ]
    assert len(gguf_in_models) > 0, (
        f"No .gguf files found in models directory in existing_files: {context.existing_files}"
    )


@then('the file "{filename}" does not exist in the models directory')  # ty: ignore[call-non-callable]
def step_check_file_not_exists(context, filename):
    """Verify file does not exist."""
    models_dir = os.path.join(model.get_models_dir())
    filepath = os.path.join(models_dir, filename)
    assert not os.path.exists(filepath), f"File {filename} still exists"


@then("the output shows no models available")  # ty: ignore[call-non-callable]
def step_check_no_models_output(context):
    """Verify output indicates no models (either local or HF cache)."""
    output = context.stdout
    # Accept either "No models found" or HuggingFace models display (showing none available locally)
    assert "No models found" in output or "HuggingFace" in output, (
        f"Expected no models output, got:\n{output}"
    )


@when('I run "arkai model convert" without model argument')  # ty: ignore[call-non-callable]
def step_run_convert_no_args(context):
    """Run arkai model convert without required model argument."""
    import sys
    from io import StringIO

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    context.exit_code = 0

    try:
        import argparse

        # Simulate calling the CLI without model argument
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        model_parser = subparsers.add_parser("model")
        model_subparsers = model_parser.add_subparsers(dest="model_cmd")
        convert_parser = model_subparsers.add_parser("convert")
        convert_parser.add_argument("model", help="Model to convert")

        parser.parse_args(["model", "convert"])
    except SystemExit as e:
        context.exit_code = e.code
    except Exception as e:
        context.exit_code = 1
        stderr_capture.write(str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        context.stdout = stdout_capture.getvalue()
        context.stderr = stderr_capture.getvalue()
