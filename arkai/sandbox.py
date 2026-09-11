"""Sandbox profile management: CRUD operations for arapuca sandbox configurations."""

import argparse
import os
import sys

from arkai import config as config_module
from arkai import utils


def exec_cmd(args: dict | None = None) -> None:
    """Select 'sandbox' command to execute."""
    match args.sandbox_cmd:  # ty: ignore[unresolved-attribute]
        case "create":
            create_env = (
                {
                    k: v
                    for e in args.environment  # ty: ignore[unresolved-attribute]
                    for k, _, v in [e.partition("=")]
                }
                if args.environment  # ty: ignore[unresolved-attribute]
                else None
            )
            cmd_sandbox_create(
                args.name,  # ty: ignore[unresolved-attribute]
                args.from_profile,  # ty: ignore[unresolved-attribute]
                args.path,  # ty: ignore[unresolved-attribute]
                args.memory,  # ty: ignore[unresolved-attribute]
                args.cpus,  # ty: ignore[unresolved-attribute]
                args.pids,  # ty: ignore[unresolved-attribute]
                args.timeout,  # ty: ignore[unresolved-attribute]
                args.volumes,  # ty: ignore[unresolved-attribute]
                create_env,
            )
        case "delete":
            cmd_sandbox_delete(args.name)  # ty: ignore[unresolved-attribute]
        case "show":
            cmd_sandbox_show(args.profile_name)  # ty: ignore[unresolved-attribute]
        case "list":
            cmd_sandbox_list()
        case "set-default":
            cmd_sandbox_set_default(args.name)  # ty: ignore[unresolved-attribute]
        case "active":
            cmd_sandbox_active(args.profile_name)  # ty: ignore[unresolved-attribute]


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    sandbox_parser = subparsers.add_parser("sandbox", help="Manage sandbox profiles")
    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_cmd", required=True)
    # list subcommand
    list_parser = sandbox_subparsers.add_parser("list", help="List sandbox profiles")
    list_parser = list_parser  # noqa: F841 - keep reference for consistency
    # show subcommand
    show_parser = sandbox_subparsers.add_parser("show", help="Show sandbox profile details")
    show_parser.add_argument("profile_name", help="Profile name to show (or 'default'/'active')")
    # create subcommand
    create_parser = sandbox_subparsers.add_parser("create", help="Create sandbox profile")
    create_parser.add_argument("name", help="Profile name")
    create_parser.add_argument("--from", dest="from_profile", help="Base profile to copy from")
    create_parser.add_argument("--path", help="Path to arapuca binary")
    create_parser.add_argument("--memory", type=int, help="Memory in MB")
    create_parser.add_argument("--cpus", type=int, help="CPU count")
    create_parser.add_argument("--pids", type=int, help="PID limit")
    create_parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    create_parser.add_argument(
        "-v",
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume in the sandbox (format: /path or /path:ro)",
    )
    create_parser.add_argument(
        "-e",
        "--env",
        action="append",
        dest="environment",
        metavar="KEY=VALUE",
        help="Set an environment variable in the sandbox (KEY=VALUE). Can be used multiple times",
    )
    # delete subcommand
    delete_parser = sandbox_subparsers.add_parser("delete", help="Delete sandbox profile")
    delete_parser.add_argument("name", help="Profile name")
    # set-default subcommand
    setdef_parser = sandbox_subparsers.add_parser("set-default", help="Set profile as default")
    setdef_parser.add_argument("name", help="Profile name")
    # active subcommand
    active_parser = sandbox_subparsers.add_parser("active", help="Set active profile")
    active_parser.add_argument("profile_name", nargs="?", help="Profile name (or empty to clear)")


def _deduplicate_volumes(volumes: list) -> list:
    """Deduplicate volumes with validation.

    First pass: exact match deduplication (path + flag together).
    Second pass: validate deduplicated list for path conflicts via utils.

    Args:
        volumes: List of volume strings in format "path" or "path:flag"

    Returns:
        Deduplicated volume list

    Raises:
        RuntimeError: If same path appears with different flags
    """
    if not volumes:
        return []

    # First pass: deduplicate exact matches
    unique_volumes: list = []
    seen_exact: set = set()
    for vol in volumes:
        if vol not in seen_exact:
            unique_volumes.append(vol)
            seen_exact.add(vol)

    # Second pass: validate for path conflicts
    err = utils.validate_volumes(unique_volumes)
    if err:
        raise RuntimeError(err)

    return unique_volumes


def _is_valid_profile_name(name: str) -> bool:
    """Check if profile name is a valid identifier.

    Args:
        name: Profile name to validate

    Returns:
        True if valid (alphanumeric + underscore, no spaces), False otherwise.
        Rejects reserved names: "default" and "active"
    """
    if not name:
        return False
    if name.lower() in ("default", "active"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _get_default_profile(cfg: dict) -> dict:
    """Extract default sandbox settings from config root.

    Args:
        cfg: Loaded configuration

    Returns:
        Dict with keys: path, memory_mb, cpus, pids, timeout, volume
    """
    sandbox = cfg.get("sandbox", {})
    result = {
        "path": sandbox.get("path", "arapuca"),
        "memory_mb": sandbox.get("memory_mb", 2048),
        "cpus": sandbox.get("cpus", 2),
        "pids": sandbox.get("pids", 256),
        "timeout": sandbox.get("timeout", 0),
        "volume": sandbox.get("volume", []),
    }
    if "environment" in sandbox:
        result["environment"] = sandbox["environment"]
    return result


def _get_profile(cfg: dict, name: str) -> dict | None:
    """Retrieve a profile by name from config.

    Args:
        cfg: Loaded configuration
        name: Profile name to retrieve

    Returns:
        Profile dict if found, None otherwise
    """
    profiles = cfg.get("sandbox", {}).get("profiles", {})
    profile = profiles.get(name)
    return profile.copy() if profile else None


def _profile_exists(cfg: dict, name: str) -> bool:
    """Check if a profile exists in config.

    Args:
        cfg: Loaded configuration
        name: Profile name to check

    Returns:
        True if profile exists, False otherwise
    """
    profiles = cfg.get("sandbox", {}).get("profiles", {})
    return name in profiles


def cmd_sandbox_list() -> None:
    """List all sandbox profiles with basic info."""
    cfg = config_module.load_config()

    print("\n=== Sandbox Profiles ===\n")

    # Show active profile indicator
    active = config_module.get_config_value(cfg, "sandbox.active_profile")
    if active:
        print(f"Active Profile: {active}\n")
    else:
        print("Active Profile: None (using defaults)\n")

    # Show all profiles
    profiles = cfg.get("sandbox", {}).get("profiles", {})
    if profiles:
        print("Available Profiles:")
        for name in sorted(profiles.keys()):
            profile = profiles[name]
            marker = " ✓" if name == active else ""
            cpus = profile.get("cpus", 2)
            memory = profile.get("memory_mb", 2048)
            print(f"  {name}{marker:2} (cpus={cpus}, memory={memory}mb)")
    else:
        print("No custom profiles defined")

    print()


def cmd_sandbox_show(profile_name: str) -> None:
    """Show details for a specific profile.

    Args:
        profile_name: Profile name to show, or "default"/"active"

    Raises:
        RuntimeError: If profile not found
    """
    cfg = config_module.load_config()

    print()

    if profile_name == "default":
        profile = _get_default_profile(cfg)
        print("Default Settings (sandbox root):")
    elif profile_name == "active":
        active_name = config_module.get_config_value(cfg, "sandbox.active_profile")
        if active_name:
            profile = _get_profile(cfg, active_name)
            if not profile:
                utils.error(f"Active profile '{active_name}' not found in config")
                sys.exit(1)
            print(f"Active Profile: {active_name}")
        else:
            profile = _get_default_profile(cfg)
            print("Active Profile: None (using defaults)")
    else:
        if not _profile_exists(cfg, profile_name):
            utils.error(f"Profile '{profile_name}' not found")
            sys.exit(1)
        profile = _get_profile(cfg, profile_name)
        if not profile:
            utils.error(f"Profile '{profile_name}' not found")
            sys.exit(1)
        print(f"Profile: {profile_name}")

    print(f"  path:      {profile['path']}")
    print(f"  memory_mb: {profile['memory_mb']}")
    print(f"  cpus:      {profile['cpus']}")
    print(f"  pids:      {profile['pids']}")
    print(f"  timeout:   {profile['timeout']}")
    volumes = profile.get("volume", [])
    if volumes:
        print("  volume:")
        for vol in volumes:
            print(f"    - {vol}")
    environment = profile.get("environment")
    if environment:
        print("  environment:")
        for key, val in environment.items():
            print(f"    {key}={val}")
    print()


def cmd_sandbox_create(
    name: str,
    from_profile: str | None = None,
    path: str | None = None,
    memory: int | None = None,
    cpus: int | None = None,
    pids: int | None = None,
    timeout: int | None = None,
    volume: list | None = None,
    environment: dict | None = None,
) -> None:
    """Create a new sandbox profile.

    Args:
        name: Profile name (must be valid identifier)
        from_profile: Base profile to copy from (optional)
        path: Arapuca binary path (optional)
        memory: Memory in MB (optional)
        cpus: CPU count (optional)
        pids: PID limit (optional)
        timeout: Timeout in seconds (optional)
        volume: List of volume mount strings (optional)
        environment: Dict of environment variables to set in the sandbox (optional)

    Raises:
        RuntimeError: If name is invalid, profile exists, or base profile not found
    """
    if not _is_valid_profile_name(name):
        utils.error(f"Invalid profile name '{name}': must be alphanumeric + underscore")
        sys.exit(1)

    cfg = config_module.load_config()

    if _profile_exists(cfg, name):
        utils.error(f"Profile '{name}' already exists")
        sys.exit(1)

    # Start with base profile or defaults
    if from_profile:
        base = _get_profile(cfg, from_profile)
        if not base:
            utils.error(f"Base profile '{from_profile}' not found")
            sys.exit(1)
        new_profile = base.copy()
    else:
        new_profile = _get_default_profile(cfg)

    # Apply overrides
    if path is not None:
        new_profile["path"] = path
    if memory is not None:
        new_profile["memory_mb"] = memory
    if cpus is not None:
        new_profile["cpus"] = cpus
    if pids is not None:
        new_profile["pids"] = pids
    if timeout is not None:
        new_profile["timeout"] = timeout
    if volume is not None:
        try:
            new_profile["volume"] = _deduplicate_volumes(volume)
        except RuntimeError as e:
            utils.error(str(e))
            sys.exit(1)
    if environment is not None:
        err = utils.validate_environment(environment)
        if err:
            utils.error(f"environment: {err}")
            sys.exit(1)
        new_profile["environment"] = environment

    # Add to config and save
    if "sandbox" not in cfg:
        cfg["sandbox"] = {}
    if "profiles" not in cfg["sandbox"]:
        cfg["sandbox"]["profiles"] = {}

    cfg["sandbox"]["profiles"][name] = new_profile

    # Determine which config file to save to
    config_home = utils.get_config_home()
    config_file = os.path.join(config_home, "arkai.yaml")
    os.makedirs(config_home, exist_ok=True)

    utils.save_yaml(config_file, cfg)
    print(f"✓ Created sandbox profile '{name}'")


def cmd_sandbox_delete(name: str) -> None:
    """Delete a sandbox profile.

    Args:
        name: Profile name to delete

    Raises:
        RuntimeError: If profile not found
    """
    cfg = config_module.load_config()

    if not _profile_exists(cfg, name):
        utils.error(f"Profile '{name}' not found")
        sys.exit(1)

    # If this is the active profile, clear it
    active = config_module.get_config_value(cfg, "sandbox.active_profile")
    if active == name:
        if "sandbox" in cfg:
            cfg["sandbox"]["active_profile"] = None

    # Remove the profile
    if "sandbox" in cfg and "profiles" in cfg["sandbox"]:
        del cfg["sandbox"]["profiles"][name]

    # Save config
    config_home = utils.get_config_home()
    config_file = os.path.join(config_home, "arkai.yaml")
    utils.save_yaml(config_file, cfg)
    print(f"✓ Deleted sandbox profile '{name}'")


def cmd_sandbox_set_default(name: str) -> None:
    """Promote a profile by copying its parameters into the default sandbox group.

    Args:
        name: Profile name to set as default

    Raises:
        RuntimeError: If profile not found
    """
    cfg = config_module.load_config()

    if not _profile_exists(cfg, name):
        utils.error(f"Profile '{name}' not found")
        sys.exit(1)

    profile = _get_profile(cfg, name)
    if not profile:
        utils.error(f"Profile '{name}' not found")
        sys.exit(1)

    # Copy all profile parameters to sandbox root
    if "sandbox" not in cfg:
        cfg["sandbox"] = {}

    cfg["sandbox"]["path"] = profile["path"]
    cfg["sandbox"]["memory_mb"] = profile["memory_mb"]
    cfg["sandbox"]["cpus"] = profile["cpus"]
    cfg["sandbox"]["pids"] = profile["pids"]
    cfg["sandbox"]["timeout"] = profile["timeout"]
    if "volume" in profile:
        cfg["sandbox"]["volume"] = profile["volume"]
    if "environment" in profile:
        cfg["sandbox"]["environment"] = profile["environment"]

    # Save config
    config_home = utils.get_config_home()
    config_file = os.path.join(config_home, "arkai.yaml")
    utils.save_yaml(config_file, cfg)
    print(f"✓ Set '{name}' as default sandbox settings")


def cmd_sandbox_active(profile_name: str | None) -> None:
    """Set the active sandbox profile for agent runs.

    Args:
        profile_name: Profile name to activate, or None to clear

    Raises:
        RuntimeError: If profile not found
    """
    cfg = config_module.load_config()

    # Allow clearing the active profile with empty string or "none"
    if profile_name is None or profile_name == "" or profile_name.lower() == "none":
        if "sandbox" in cfg:
            cfg["sandbox"]["active_profile"] = None
        config_home = utils.get_config_home()
        config_file = os.path.join(config_home, "arkai.yaml")
        utils.save_yaml(config_file, cfg)
        print("✓ Cleared active sandbox profile (using defaults)")
        return

    # Verify profile exists
    if not _profile_exists(cfg, profile_name):
        utils.error(f"Profile '{profile_name}' not found")
        sys.exit(1)

    # Set active profile
    if "sandbox" not in cfg:
        cfg["sandbox"] = {}
    cfg["sandbox"]["active_profile"] = profile_name

    # Save config
    config_home = utils.get_config_home()
    config_file = os.path.join(config_home, "arkai.yaml")
    utils.save_yaml(config_file, cfg)
    print(f"✓ Set active sandbox profile to '{profile_name}'")
