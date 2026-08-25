"""Behave test fixtures and setup."""

import os
import shutil
import tempfile


def before_scenario(context, scenario):
    """Set up test environment before each scenario."""
    # Create temporary working directory
    context.temp_dir = tempfile.mkdtemp()
    context.original_dir = os.getcwd()
    os.chdir(context.temp_dir)

    # Create temporary config directory
    context.config_dir = os.path.join(context.temp_dir, ".aitool")
    os.makedirs(context.config_dir, exist_ok=True)


def after_scenario(context, scenario):
    """Clean up after each scenario."""
    # Return to original directory
    os.chdir(context.original_dir)

    # Remove temporary directory
    if os.path.exists(context.temp_dir):
        shutil.rmtree(context.temp_dir)
