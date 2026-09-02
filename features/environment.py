"""Behave test fixtures and setup."""

import fnmatch
import io
import os
import re
from pathlib import Path
from unittest.mock import patch

import yaml

from arkai import config, model, utils


def run_command_helper(context, cmd_args):
    """Run a command and set context with result.

    Uses the mocked patches to execute commands with proper environment setup.
    Sets context.exit_code, context.stdout, and context.stderr.
    """
    code, stdout, stderr = utils.run_command(cmd_args)
    context.exit_code = code
    context.stdout = stdout
    context.stderr = stderr


def setup_open_mock(context):
    """Create and start the open() mock with current context state.

    This should be called whenever context.inference_running or context.config_data changes.
    """
    # Stop existing patch if present
    if hasattr(context, "open_patch") and context.open_patch:
        context.open_patch.stop()

    def mock_open(filename, mode="r", *args, **kwargs):
        filepath = str(filename)
        contents = None
        if filepath.endswith(".arkai.yaml"):
            if "w" in mode or "a" in mode:
                # Write mode: capture data written to context.config_data
                class ConfigFileWriter(io.StringIO):
                    def close(self):
                        content = self.getvalue()
                        if content:
                            try:
                                context.config_data = yaml.safe_load(content)
                            except Exception:
                                pass
                        context.existing_files.add(filepath)
                        super().close()

                return ConfigFileWriter()
            else:
                # Read mode: return config_data as YAML
                if context.config_data is not None:
                    yaml_content = yaml.dump(context.config_data)
                    return io.StringIO(yaml_content)
                raise FileNotFoundError(f"Config file not found: {filename}")
        if filepath.endswith("inference.pid"):
            contents = "9999" if context.inference_running else None
        if filepath.endswith("vectordb.pid"):
            contents = "9997" if context.vectordb_running else None
        if re.search(r"wtmcp-\d+.pid", filepath) is not None:
            contents = "9998" if context.wtmcp_should_be_running else None
        if filepath == getattr(context, "test_doc_path", None) and hasattr(
            context, "test_doc_content"
        ):
            contents = context.test_doc_content
        if "w" in mode or "a" in mode:  # creating a file shold always succeed
            context.existing_files.add(filepath)
            return io.StringIO(contents)
        if contents:  # read case, only if data is to be returned
            return io.StringIO(contents)
        raise FileNotFoundError(f"File ({str(filename)}) is not mocked for mode '{mode}'.")

    context.open_patch = patch("builtins.open", side_effect=mock_open)
    context.open_patch.start()


def _clean_patches(context):
    """Clean patches."""
    # Stop any lingering patches from previous scenarios
    patches_to_clean = [
        "port_in_use_patch",
        "port_in_use_patch_utils",
        "port_in_use_patch_engine",
        "is_running_patch",
        "wtmcp_running_patch",
        "vectordb_running_patch",
        "inference_popen_patch",
        "inference_run_command_patch",
        "wait_stop_patch",
        "getenv_patch",
        "load_yaml_patch",
        "save_yaml_patch",
        "get_config_value_patch",
        "load_config_patch",
        "exists_patch",
        "open_patch",
        "run_command_patch",
        "makedirs_patch",
        "mkdir_patch",
        "path_mkdir_patch",
        "remove_patch",
        "glob_patch",
        "listdir_patch",
    ]
    for patch_name in patches_to_clean:
        if hasattr(context, patch_name):
            patch_obj = getattr(context, patch_name)
            if patch_obj:
                try:
                    patch_obj.stop()
                except Exception:
                    pass


def before_scenario(context, scenario):
    """Set up test environment before each scenario."""
    _clean_patches(context)

    # Mock os.getenv for HOME and XDG_CONFIG_HOME
    original_getenv = os.getenv

    def mock_getenv(key, default=None):
        if key == "HOME":
            return "/Home/User"
        elif key == "XDG_CONFIG_HOME":
            return "/Home/User/.config"
        return original_getenv(key, default)

    context.getenv_patch = patch.object(os, "getenv", side_effect=mock_getenv)
    context.getenv_patch.start()

    # Initialize config_data for load_yaml mock
    context.config_data = None

    # Track services running state
    context.inference_running = False
    context.wtmcp_running = False
    context.vectordb_running = False

    # Track existing files
    context.existing_files = {
        ".arkai.yaml",
        model.get_models_dir(),
        utils.get_pid_dir(),
    }

    # Mock utils.load_yaml to return context.config_data
    def mock_load_yaml(filepath):
        if context.config_data is not None:
            return context.config_data
        raise FileNotFoundError(f"Config file not found: {filepath}")

    context.load_yaml_patch = patch.object(utils, "load_yaml", side_effect=mock_load_yaml)
    context.load_yaml_patch.start()

    # Mock utils.save_yaml to store data in context.config_data
    def mock_save_yaml(filepath, data):
        context.config_data = data
        context.existing_files.add(str(filepath))

    context.save_yaml_patch = patch.object(utils, "save_yaml", side_effect=mock_save_yaml)
    context.save_yaml_patch.start()

    # Mock config.get_config_value to work with context.config_data
    def mock_get_config_value(cfg, key_path, default=None):
        if cfg is None or context.config_data is None:
            return default
        # Navigate through nested keys (e.g., "inference.model" -> cfg["inference"]["model"])
        keys = key_path.split(".")
        value = context.config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    context.get_config_value_patch = patch.object(
        config, "get_config_value", side_effect=mock_get_config_value
    )
    context.get_config_value_patch.start()

    # Mock config.load_config to return context.config_data
    def mock_load_config():
        return context.config_data if context.config_data is not None else {}

    context.load_config_patch = patch.object(config, "load_config", side_effect=mock_load_config)
    context.load_config_patch.start()

    # Mock os.path.exists to check context.existing_files
    def mock_exists(path):
        return str(path) in context.existing_files

    context.exists_patch = patch.object(os.path, "exists", side_effect=mock_exists)
    context.exists_patch.start()

    # Set up the open() mock with current context state
    setup_open_mock(context)

    # Mock os.makedirs to always succeed
    context.makedirs_patch = patch.object(os, "makedirs", return_value=None)
    context.makedirs_patch.start()

    # Mock os.mkdir to always succeed
    context.mkdir_patch = patch.object(os, "mkdir", return_value=None)
    context.mkdir_patch.start()

    # Mock Path.mkdir to always succeed — in Python 3.9, pathlib uses _NormalAccessor
    # which captures the original os.mkdir reference before the mock above is applied,
    # so Path.mkdir() must be patched directly to prevent real directory creation.
    context.path_mkdir_patch = patch.object(Path, "mkdir", return_value=None)
    context.path_mkdir_patch.start()

    # Mock os.remove to always succeed and update existing_files set
    def mock_remove(path):
        context.existing_files.discard(str(path))

    context.remove_patch = patch.object(os, "remove", side_effect=mock_remove)
    context.remove_patch.start()

    # Mock Path.glob to return files from existing_files matching the pattern
    def mock_glob(self, pattern):
        dir_path = str(self)
        matching = [
            f
            for f in context.existing_files
            if dir_path in f and fnmatch.fnmatch(os.path.basename(f), pattern)
        ]
        return [Path(f) for f in matching]

    context.glob_patch = patch.object(Path, "glob", mock_glob)
    context.glob_patch.start()

    # Mock os.listdir to return files from existing_files in the given directory
    def mock_listdir(path):
        path_str = str(path)
        # Add trailing separator if not present
        if not path_str.endswith(os.sep):
            path_str += os.sep
        # Find all files in existing_files that are direct children of this directory
        files = []
        for f in context.existing_files:
            if f.startswith(path_str):
                # Get the relative path and check if it's a direct child
                relative = f[len(path_str) :]
                # Only include if it doesn't contain another separator (direct child)
                if os.sep not in relative and "/" not in relative:
                    files.append(relative)
        return files

    context.listdir_patch = patch.object(os, "listdir", side_effect=mock_listdir)
    context.listdir_patch.start()

    # Initialize wtmcp state for dynamic mocking
    context.wtmcp_should_be_running = False
    context.wtmcp_port = 9998

    # Initialize vectordb state for dynamic mocking
    context.vectordb_should_be_running = False
    context.vectordb_port = 9997

    # Mock is_port_in_use with side_effect to read context state
    def mock_is_port_in_use(port):
        if context.wtmcp_should_be_running and port == context.wtmcp_port:
            return True
        if context.vectordb_should_be_running and port == context.vectordb_port:
            return True
        return False

    context.port_in_use_patch = patch.object(
        utils, "is_port_in_use", side_effect=mock_is_port_in_use
    )
    context.port_in_use_patch.start()

    # Mock wtmcp.is_wtmcp_running with side_effect to read context state
    from arkai import wtmcp

    def mock_is_wtmcp_running(port=None):
        if port is None:
            return context.wtmcp_should_be_running
        return context.wtmcp_should_be_running and port == context.wtmcp_port

    context.wtmcp_running_patch = patch.object(
        wtmcp, "is_wtmcp_running", side_effect=mock_is_wtmcp_running
    )
    context.wtmcp_running_patch.start()

    # Mock vectordb.is_vectordb_running with side_effect to read context state
    from arkai import vectordb

    def mock_is_vectordb_running():
        return context.vectordb_should_be_running

    context.vectordb_running_patch = patch.object(
        vectordb, "is_vectordb_running", side_effect=mock_is_vectordb_running
    )
    context.vectordb_running_patch.start()

    # Initialize models_dir to None; steps can set it as needed
    context.models_dir = None
    context.original_get_data_home = None

    # Mock utils.run_command to avoid spawning subprocesses
    original_run_command = utils.run_command

    def mock_run_command(cmd_args):
        if len(cmd_args) >= 1 and cmd_args[0] == "arkai":
            import sys

            from arkai.cli import main

            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                sys.argv = cmd_args
                main()
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 0
            except Exception as e:
                exit_code = 1
                sys.stderr.write(f"Error: {e}\n")
            finally:
                stdout_content = sys.stdout.getvalue()
                stderr_content = sys.stderr.getvalue()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            return exit_code, stdout_content, stderr_content
        return original_run_command(cmd_args)

    context.run_command_patch = patch.object(utils, "run_command", side_effect=mock_run_command)
    context.run_command_patch.start()

    # Attach run_command helper to context
    context.run_command = lambda cmd_args: run_command_helper(context, cmd_args)


def after_scenario(context, scenario):
    """Clean up after each scenario."""
    _clean_patches(context)

    # Restore original get_data_home if mocked
    if context.original_get_data_home is not None:
        utils.get_data_home = context.original_get_data_home
