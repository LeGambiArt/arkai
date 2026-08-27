"""Configuration loading, merging, and validation."""

import copy
import os
import sys
from typing import Any, Optional

from arkai import utils

# Defaults
DEFAULTS = {
    "agent": {
        "name": "opencode",
    },
    "inference": {
        "backend": "llama-cpp",
        "port": 8081,
        "gpu_layers": -1,
        "context_size": 65536,
    },
    "wtmcp": {
        "port": 8080,
        "path": "wtmcp",
    },
    "sandbox": {
        "enabled": True,
        "path": "arapuca",
        "memory_mb": 2048,
        "cpus": 2,
        "pids": 256,
        "timeout": 0,
        "active_profile": None,
        "profiles": {},
    },
}

VALID_AGENTS = {"opencode", "crush", "claude"}


def load_config(project_dir: str = ".") -> dict:
    """Load configuration from user and project, merge, return combined config.

    Precedence: defaults < user config < project config
    """
    # Start with defaults (deep copy to prevent mutation of module-level DEFAULTS)
    config = copy.deepcopy(DEFAULTS)

    # Load user config if exists
    user_config_path = os.path.join(utils.get_config_home(), "arkai.yaml")
    if os.path.exists(user_config_path):
        user_config = utils.load_yaml(user_config_path) or {}
        config = utils.merge_configs(config, user_config)

    # Load project config if exists
    project_config_path = os.path.join(project_dir, ".arkai.yaml")
    if os.path.exists(project_config_path):
        project_config = utils.load_yaml(project_config_path) or {}
        config = utils.merge_configs(config, project_config)

    return config


def validate_config(config: dict) -> bool:
    """Validate config schema, required fields, mutual exclusivity.

    Returns True if valid, False if any errors found (errors printed to stderr).
    """
    valid = True

    # Check agent name
    agent_name = get_config_value(config, "agent.name")
    if not agent_name:
        utils.error("agent.name is required")
        valid = False
    elif agent_name not in VALID_AGENTS:
        utils.error(f"agent.name must be one of {VALID_AGENTS}, got {agent_name}")
        valid = False

    # Check inference config
    model = get_config_value(config, "inference.model")
    hf = get_config_value(config, "inference.hf")

    if model and hf:
        utils.error("inference.model and inference.hf are mutually exclusive")
        valid = False
    elif not model and not hf:
        utils.error("inference.model or inference.hf is required")
        valid = False

    # Check port ranges
    inference_port = get_config_value(config, "inference.port", 8081)
    wtmcp_port = get_config_value(config, "wtmcp.port", 8080)

    if not (1024 <= inference_port <= 65535):
        utils.error(f"inference.port must be 1024-65535, got {inference_port}")
        valid = False
    if not (1024 <= wtmcp_port <= 65535):
        utils.error(f"wtmcp.port must be 1024-65535, got {wtmcp_port}")
        valid = False

    # Check numeric fields
    gpu_layers = get_config_value(config, "inference.gpu_layers", -1)
    if gpu_layers != -1 and not isinstance(gpu_layers, int):
        utils.error(f"inference.gpu_layers must be integer or -1, got {gpu_layers}")
        valid = False

    context_size = get_config_value(config, "inference.context_size", 65536)
    if not isinstance(context_size, int) or context_size <= 0:
        utils.error(f"inference.context_size must be positive integer, got {context_size}")
        valid = False

    # Validate sandbox volumes at root level
    root_volumes = get_config_value(config, "sandbox.volume", [])
    if root_volumes:
        err = utils.validate_volumes(root_volumes)
        if err:
            utils.error(f"sandbox.volume: {err}")
            valid = False

    # Validate sandbox environment at root level
    root_env = get_config_value(config, "sandbox.environment")
    if root_env is not None:
        err = utils.validate_environment(root_env)
        if err:
            utils.error(f"sandbox.environment: {err}")
            valid = False

    # Validate volumes and environment in each sandbox profile
    profiles = get_config_value(config, "sandbox.profiles", {})
    if profiles:
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            profile_vols = profile.get("volume", [])
            if profile_vols:
                err = utils.validate_volumes(profile_vols)
                if err:
                    utils.error(f"sandbox.profiles.{profile_name}.volume: {err}")
                    valid = False
            profile_env = profile.get("environment")
            if profile_env is not None:
                err = utils.validate_environment(profile_env)
                if err:
                    utils.error(f"sandbox.profiles.{profile_name}.environment: {err}")
                    valid = False

    return valid


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """Get nested config value using dot notation (e.g., 'inference.port')."""
    parts = key.split(".")
    value = config
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return default
    return value if value is not None else default


def cmd_config_validate(config_file: Optional[str] = None) -> None:
    """Validate configuration file.

    Args:
        config_file: Optional path to config file (default: .arkai.yaml)
    """
    if config_file is None:
        config_file = ".arkai.yaml"

    cfg = utils.load_yaml(config_file)
    if not validate_config(cfg):
        sys.exit(2)
    utils.info(f"Configuration valid: {config_file}")


def cmd_config_init() -> None:
    """Initialize a new configuration file in current directory.

    Fails if .arkai.yaml already exists.
    """
    config_file = ".arkai.yaml"

    if os.path.exists(config_file):
        raise RuntimeError(f"Configuration file already exists: {config_file}")

    # Create default config
    default_config = {
        "agent": {
            "name": "opencode",
        },
        "inference": {
            "backend": "llama-cpp",
            "port": 8081,
            "gpu_layers": -1,
            "context_size": 65536,
        },
        "wtmcp": {
            "port": 8080,
        },
        "sandbox": {
            "enabled": True,
            "memory_mb": 2048,
            "cpus": 200,
            "pids": 256,
            "timeout": 0,
        },
    }

    utils.save_yaml(config_file, default_config)
    utils.info(f"Configuration initialized: {config_file}")
