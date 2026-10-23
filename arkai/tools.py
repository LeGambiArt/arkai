"""Manage Arkai-owned installations of supported third-party tools."""

import argparse
import os
import platform
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

from arkai import config, utils

STRATA_INSTALLER_URL = "https://github.com/Niko1221/Strata/releases/latest/download/install.sh"
SUPPORTED_TOOLS = ("strata",)
_MANAGED_ENTRIES = ("bin", "data", "config", "cache", "state", "home", "tmp")
_INSTALL_COMMANDS = (
    "sh",
    "env",
    "curl",
    "tar",
    "cut",
    "uname",
    "awk",
    "head",
    "rm",
    "mkdir",
    "chmod",
    "mv",
    "ln",
    "ls",
    "sed",
    "mktemp",
)
_OPTIONAL_INSTALL_COMMANDS = (
    "sha256sum",
    "shasum",
    "nvidia-smi",
    "lspci",
    "system_profiler",
    "grep",
    "python3",
    "python",
    "bash",
)


def exec_cmd(args: argparse.Namespace) -> None:
    """Select the ``tools`` command to execute."""
    match args.tools_cmd:
        case "list":
            cmd_tools_list()
        case "install":
            cmd_tools_install(args.tool_name)
        case "update":
            cmd_tools_update(args.tool_name)
        case "remove":
            cmd_tools_remove(args.tool_name)


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Register the third-party tool management commands."""
    tools_parser = subparsers.add_parser("tools", help="Manage third-party tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_cmd", required=True)
    tools_subparsers.add_parser("list", help="List Arkai-managed tools and versions")

    install_parser = tools_subparsers.add_parser("install", help="Install a supported tool")
    install_parser.add_argument(
        "tool_name",
        nargs="?",
        help="Tool to install (omit to list supported tools)",
    )

    for action in ("update", "remove"):
        action_parser = tools_subparsers.add_parser(
            action,
            help=f"{action.title()} a supported tool",
        )
        action_parser.add_argument("tool_name", help="Tool to select")


def get_strata_root() -> Path:
    """Return the private Arkai-managed Strata installation directory."""
    return Path(utils.get_pid_dir()) / "tools" / "strata"


def get_strata_binary() -> Path:
    """Return the stable entry point used by the official Strata installer."""
    return get_strata_root() / "bin" / "strata"


def _validate_tool(tool_name: str) -> None:
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(
            f"Tool '{tool_name}' is not supported. Supported tools: {', '.join(SUPPORTED_TOOLS)}"
        )


def _is_managed_strata_installed() -> bool:
    """Return whether the stable Strata executable points inside Arkai's install root."""
    root = get_strata_root()
    binary = get_strata_binary()
    if root.is_symlink():
        return False
    try:
        resolved_root = root.resolve(strict=True)
        resolved_binary = binary.resolve(strict=True)
        resolved_binary.relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    return binary.is_file() and os.access(binary, os.X_OK)


def get_managed_strata_binary() -> Path | None:
    """Return Arkai's Strata executable when its managed install is valid."""
    return get_strata_binary() if _is_managed_strata_installed() else None


def _ensure_install_root() -> Path:
    """Create private Strata directories and reject symlinks that could redirect writes."""
    root = get_strata_root()
    if root.is_symlink():
        raise RuntimeError(f"Refusing to use a symlink as the Strata install root: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)

    for entry in _MANAGED_ENTRIES:
        path = root / entry
        if path.is_symlink():
            raise RuntimeError(f"Refusing to use a symlink in the Strata install directory: {path}")
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Strata install path is not a directory: {path}")
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)

    binary = get_strata_binary()
    if binary.exists() and not _is_managed_strata_installed():
        raise RuntimeError(
            f"Refusing to replace an unmanaged Strata executable at {binary}; "
            "move it aside or configure a different installation"
        )
    return root


def _host_tool_paths(command_names: tuple[str, ...]) -> dict[str, str]:
    """Resolve host utilities that will be made available to the sandbox."""
    result: dict[str, str] = {}
    missing = []
    for name in command_names:
        path = shutil.which(name)
        if path is None:
            missing.append(name)
        else:
            result[name] = str(Path(path).resolve())
    if missing:
        raise RuntimeError(f"Required host command(s) not found: {', '.join(missing)}")
    return result


def _sandbox_paths(tool_paths: dict[str, str]) -> tuple[list[str], str]:
    """Build read-only utility mounts and a restricted PATH for sandboxed commands."""
    command_directories = {str(Path(path).parent) for path in tool_paths.values()}
    directories = set(command_directories)
    # Dynamic system tools and curl need their shared libraries and TLS trust roots.
    common_runtime_paths = (
        "/etc/ssl",
        "/etc/ssl/cert.pem",
        "/etc/resolv.conf",
        "/etc/hosts",
    )
    system_paths = common_runtime_paths
    if platform.system() == "Linux":
        system_paths += (
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/etc/ca-certificates",
            "/etc/pki",
            "/etc/nsswitch.conf",
        )
    for candidate in system_paths:
        if Path(candidate).exists():
            directories.add(str(Path(candidate).resolve()))

    ordered_directories = sorted(directories)
    volumes = [f"{directory}:ro" for directory in ordered_directories]
    return volumes, os.pathsep.join(sorted(command_directories))


def _sandbox_environment(root: Path, path: str) -> dict[str, str]:
    """Return child-only paths without overriding Arapuca-managed variables."""
    env = {
        "PATH": path,
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_STATE_HOME": str(root / "state"),
        "XDG_DATA_HOME": str(root / "data"),
        "STRATA_DATA_DIR": str(root / "data"),
        "STRATA_BIN_DIR": str(root / "bin"),
    }
    return env


def _build_sandbox_prefix(
    root: Path,
    command_names: tuple[str, ...],
    *,
    allow_network: bool,
    read_only_root: bool = False,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Create an Arapuca command prefix for one external Strata operation."""
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Strata tool management requires macOS or Linux and Arapuca")

    from arkai.agent import _build_sandbox_cmd, _resolve_sandbox_profile

    tool_paths = _host_tool_paths(command_names)
    if "curl" in command_names and "sh" in command_names:
        for optional_tool in _OPTIONAL_INSTALL_COMMANDS:
            path = shutil.which(optional_tool)
            if path is not None:
                tool_paths[optional_tool] = str(Path(path).resolve())
        if not ("sha256sum" in tool_paths or "shasum" in tool_paths):
            raise RuntimeError("Strata's installer requires sha256sum or shasum")

    volumes, path = _sandbox_paths(tool_paths)
    environment = _sandbox_environment(root, path)
    loaded_cfg = config.load_config()
    profile = _resolve_sandbox_profile(loaded_cfg)
    cfg = {
        "inference": loaded_cfg.get("inference", {}),
        "sandbox": {
            "path": profile.get("path", "arapuca"),
            "memory_mb": profile.get("memory_mb", 2048),
            "cpus": profile.get("cpus", 2),
            "pids": profile.get("pids", 256),
            "timeout": profile.get("timeout", 0),
            "active_profile": None,
            "profiles": {},
            "volume": [],
            "environment": {},
        },
    }

    if read_only_root:
        volumes.append(f"{root}:ro")
        workdir = None
    else:
        workdir = str(root)

    prefix = _build_sandbox_cmd(
        cfg,
        workdir,
        None,
        None,
        cli_volumes=volumes,
        cli_environment={},
        allow_network=allow_network,
        allow_inference=False,
    )
    return prefix, tool_paths, environment


def _run_sandboxed(
    prefix: list[str],
    command: list[str],
    *,
    env_binary: str | None = None,
    environment: dict[str, str] | None = None,
    capture: bool = False,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Execute a tool command only after Arapuca's ``--`` separator."""
    if environment:
        if env_binary is None:
            raise ValueError("env_binary is required when setting sandboxed environment values")
        assignments = [f"{key}={value}" for key, value in environment.items()]
        command = [env_binary, *assignments, *command]
    code, stdout, stderr = utils.run_command(
        [*prefix, *command],
        capture=capture,
        timeout=timeout,
    )
    if code != 0:
        detail = (stderr or "").strip() or (stdout or "").strip() or f"exit code {code}"
        raise RuntimeError(f"Sandboxed command failed ({' '.join(command)}): {detail}")
    return code, stdout, stderr


def _download_installer(installer: Path) -> None:
    """Fetch the small official installer script with the project's requests dependency."""
    temporary = installer.with_name(f"{installer.name}.download")
    size_limit = 256 * 1024
    total = 0
    try:
        with requests.get(STRATA_INSTALLER_URL, stream=True, timeout=30) as response:
            response.raise_for_status()
            if urlparse(response.url).scheme != "https":
                raise RuntimeError("Strata's installer download redirected away from HTTPS")
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > size_limit:
                        raise RuntimeError(
                            "Strata's official installer script is unexpectedly large"
                        )
                    output.write(chunk)
        if total == 0:
            raise RuntimeError("Strata's official installer download was empty")
        with temporary.open("rb") as script:
            if not script.read(32).startswith(b"#!/bin/sh"):
                raise RuntimeError("Downloaded Strata installer is not a POSIX shell script")
        temporary.chmod(0o600)
        temporary.replace(installer)
    except requests.RequestException as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download Strata's official installer: {error}") from error
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _confirm(action: str, tool_name: str, location: Path) -> bool:
    """Ask before making a network-backed or destructive tool change."""
    if action in {"install", "update"}:
        utils.warn(
            f"This will {action} {tool_name} using its official installer inside Arapuca. "
            f"Files will be kept under {location}."
        )
    else:
        utils.warn(f"This will remove the Arkai-managed {tool_name} installation at {location}.")
    try:
        answer = input("Proceed? [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in {"y", "yes"}


def _install_or_update_strata(action: str) -> None:
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Strata tool management requires macOS or Linux and Arapuca")
    root = get_strata_root()
    if not _confirm(action, "strata", root):
        utils.info(f"Strata {action} cancelled.")
        return
    root = _ensure_install_root()

    installer = root / "tmp" / "install.sh"
    prefix, tools, environment = _build_sandbox_prefix(
        root,
        _INSTALL_COMMANDS,
        allow_network=True,
    )
    utils.info("Downloading Strata's official installer...")
    _download_installer(installer)

    utils.info(f"Running Strata's official installer inside Arapuca ({action})...")
    _run_sandboxed(
        prefix,
        [tools["sh"], str(installer)],
        env_binary=tools["env"],
        environment=environment,
    )
    if not _is_managed_strata_installed():
        raise RuntimeError(
            f"Strata's installer completed without creating an executable at {get_strata_binary()}"
        )

    version = _strata_version()
    utils.info(f"Strata {action} complete ({version}): {get_strata_binary()}")
    utils.info("Arkai will use this executable automatically for inference.backend: strata.")
    utils.info("Set inference.path to override it with another Strata installation.")


def _strata_version() -> str:
    """Read the installed Strata version by executing its CLI inside Arapuca."""
    if not _is_managed_strata_installed():
        raise RuntimeError("Arkai-managed Strata is not installed")
    root = get_strata_root()
    prefix, tools, environment = _build_sandbox_prefix(
        root,
        ("env", "python3"),
        allow_network=False,
        read_only_root=True,
    )
    _, stdout, stderr = _run_sandboxed(
        prefix,
        [str(get_strata_binary()), "--version"],
        env_binary=tools["env"],
        environment=environment,
        capture=True,
        timeout=30,
    )
    version = stdout.strip() or stderr.strip()
    return version.splitlines()[0] if version else "version unavailable"


def cmd_tools_list() -> None:
    """List Arkai-managed tools and their reported versions."""
    installed = [
        name for name in SUPPORTED_TOOLS if name == "strata" and _is_managed_strata_installed()
    ]
    if not installed:
        utils.info("No Arkai-managed tools installed")
        return

    utils.info("Arkai-managed tools:")
    for name in installed:
        version = _strata_version()
        utils.info(f"  {name}: {version}")


def cmd_tools_install(tool_name: str | None = None) -> None:
    """Install a supported third-party tool, or list supported installers."""
    if tool_name is None:
        utils.info("Tools supported for installation: " + ", ".join(SUPPORTED_TOOLS))
        return
    _validate_tool(tool_name)
    if _is_managed_strata_installed():
        raise RuntimeError("Strata is already installed; use 'arkai tools update strata'")
    _install_or_update_strata("install")


def cmd_tools_update(tool_name: str) -> None:
    """Update a supported Arkai-managed tool."""
    _validate_tool(tool_name)
    if not _is_managed_strata_installed():
        raise RuntimeError(
            "Arkai-managed Strata is not installed; use 'arkai tools install strata'"
        )
    _install_or_update_strata("update")


def cmd_tools_remove(tool_name: str) -> None:
    """Remove only the Arkai-managed installation of a supported tool."""
    _validate_tool(tool_name)
    if not _is_managed_strata_installed():
        raise RuntimeError("Arkai-managed Strata is not installed")

    root = get_strata_root()
    if not _confirm("remove", tool_name, root):
        utils.info("Strata removal cancelled.")
        return

    prefix, tools, _ = _build_sandbox_prefix(
        root,
        ("rm",),
        allow_network=False,
    )
    entries = list(_MANAGED_ENTRIES)
    _run_sandboxed(prefix, [tools["rm"], "-rf", *entries])
    try:
        root.rmdir()
    except OSError as error:
        remaining = ", ".join(sorted(path.name for path in root.iterdir()))
        raise RuntimeError(
            f"Strata files were removed, but unexpected files remain in {root}: {remaining}"
        ) from error
    utils.info("Removed Arkai-managed Strata")
