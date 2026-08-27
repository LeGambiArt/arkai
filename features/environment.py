"""Behave test fixtures and setup."""

import os
import shutil
import tempfile
from unittest.mock import patch

from arkai import engine, utils


def before_scenario(context, scenario):
    """Set up test environment before each scenario."""
    # Stop any lingering patches from previous scenarios
    patches_to_clean = [
        "port_in_use_patch",
        "port_in_use_patch_utils",
        "port_in_use_patch_engine",
        "is_running_patch",
        "wtmcp_running_patch",
        "inference_popen_patch",
        "inference_run_command_patch",
    ]
    for patch_name in patches_to_clean:
        if hasattr(context, patch_name):
            patch_obj = getattr(context, patch_name)
            if patch_obj:
                try:
                    patch_obj.stop()
                except Exception:
                    pass

    # Clean up any leftover PID and state files from previous test
    pid_path = engine.get_inference_pid_path()
    if os.path.exists(pid_path):
        os.remove(pid_path)

    state_path = engine.get_inference_state_path()
    if os.path.exists(state_path):
        os.remove(state_path)

    # Clean up all wtmcp PID and state files (port-specific)
    pid_dir = utils.get_pid_dir()
    if os.path.exists(pid_dir):
        for filename in os.listdir(pid_dir):
            if filename.startswith("wtmcp-") and (
                filename.endswith(".pid") or filename.endswith(".state")
            ):
                filepath = os.path.join(pid_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)

    # Mock is_port_in_use to False by default so tests don't depend on real port state
    context.port_in_use_patch = patch.object(utils, "is_port_in_use", return_value=False)
    context.port_in_use_patch.start()

    # Create temporary working directory
    context.temp_dir = tempfile.mkdtemp()
    context.original_dir = os.getcwd()
    os.chdir(context.temp_dir)

    # Create temporary config directory
    context.config_dir = os.path.join(context.temp_dir, ".arkai")
    os.makedirs(context.config_dir, exist_ok=True)

    # Mock get_config_home to return temp directory for sandbox config
    context.get_config_home_patch = patch.object(
        utils, "get_config_home", return_value=context.config_dir
    )
    context.get_config_home_patch.start()

    # Set XDG_CONFIG_HOME so subprocess calls (e.g., in sandbox BDD tests) use the temp dir
    context._original_xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = context.config_dir

    # Initialize models_dir to None; steps can set it as needed
    context.models_dir = None
    context.original_get_data_home = None


def after_scenario(context, scenario):
    """Clean up after each scenario."""
    # Stop any running patches
    patches_to_stop = [
        "port_in_use_patch",
        "is_running_patch",
        "wtmcp_running_patch",
        "get_config_home_patch",
        "inference_popen_patch",
        "inference_run_command_patch",
    ]
    for patch_name in patches_to_stop:
        if hasattr(context, patch_name):
            patch_obj = getattr(context, patch_name)
            if patch_obj:
                try:
                    patch_obj.stop()
                except Exception:
                    pass

    # Clean up PID and state files
    pid_path = engine.get_inference_pid_path()
    if os.path.exists(pid_path):
        os.remove(pid_path)

    state_path = engine.get_inference_state_path()
    if os.path.exists(state_path):
        os.remove(state_path)

    # Clean up all wtmcp files (port-specific)
    pid_dir = utils.get_pid_dir()
    if os.path.exists(pid_dir):
        for filename in os.listdir(pid_dir):
            if filename.startswith("wtmcp-") and (
                filename.endswith(".pid") or filename.endswith(".state")
            ):
                filepath = os.path.join(pid_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)

    # Restore original get_data_home if mocked
    if context.original_get_data_home is not None:
        utils.get_data_home = context.original_get_data_home

    # Return to original directory
    os.chdir(context.original_dir)

    # Remove temporary directory
    if os.path.exists(context.temp_dir):
        shutil.rmtree(context.temp_dir)
