"""Behave test fixtures and setup."""

import os
import shutil
import tempfile

from aitool import utils


def before_scenario(context, scenario):
    """Set up test environment before each scenario."""
    # Create temporary working directory
    context.temp_dir = tempfile.mkdtemp()
    context.original_dir = os.getcwd()
    os.chdir(context.temp_dir)

    # Create temporary config directory
    context.config_dir = os.path.join(context.temp_dir, ".aitool")
    os.makedirs(context.config_dir, exist_ok=True)

    # Initialize models_dir to None; steps can set it as needed
    context.models_dir = None
    context.original_get_data_home = None


def after_scenario(context, scenario):
    """Clean up after each scenario."""
    # Restore original get_data_home if mocked
    if context.original_get_data_home is not None:
        utils.get_data_home = context.original_get_data_home

    # Return to original directory
    os.chdir(context.original_dir)

    # Remove temporary directory
    if os.path.exists(context.temp_dir):
        shutil.rmtree(context.temp_dir)
