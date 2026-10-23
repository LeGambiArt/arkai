"""Manage Arkai-owned installations of supported third-party tools."""

import argparse
import json
import os
import platform
import shutil
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import requests

from arkai import config, utils

STRATA_INSTALLER_URL = "https://github.com/Niko1221/Strata/releases/latest/download/install.sh"
SUPPORTED_TOOLS = ("strata", "pi", "omp")
_TOOL_DISPLAY_NAMES = {"strata": "Strata", "pi": "Pi", "omp": "OMP"}
_MANAGED_ENTRIES = ("bin", "data", "config", "cache", "state", "home", "tmp")
_PI_MANAGED_ENTRIES = ("core", "agent", "npm-cache")
_OMP_MANAGED_ENTRIES = ("bin", "agent", "tmp")
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
    "sh",
    "git",
    "gzip",
    "tar",
    "curl",
    "wget",
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


def _get_tool_module(tool_name: str) -> ModuleType:
    """Load the implementation module for a supported agent tool lazily."""
    module_name = {"pi": "arkai.agent_pi", "omp": "arkai.agent_omp"}.get(tool_name)
    if module_name is None:
        raise ValueError(f"Tool does not have an agent installer: {tool_name}")
    return __import__(module_name, fromlist=["get_install_dir"])


def _tool_root(tool_name: str) -> Path:
    if tool_name == "strata":
        return get_strata_root()
    return _get_tool_module(tool_name).get_install_dir()


def _tool_binary(tool_name: str) -> Path:
    if tool_name == "strata":
        return get_strata_binary()
    if tool_name == "pi":
        module = _get_tool_module(tool_name)
        return module.get_core_dir() / "bin" / "pi"
    if tool_name == "omp":
        module = _get_tool_module(tool_name)
        return module.get_core_dir() / "omp"
    raise ValueError(f"Tool is not supported: {tool_name}")


def _validate_tool(tool_name: str) -> None:
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(
            f"Tool '{tool_name}' is not supported. Supported tools: {', '.join(SUPPORTED_TOOLS)}"
        )


def _is_managed_executable(binary: Path, root: Path) -> bool:
    """Return whether an executable points inside an Arkai-owned install root."""
    if root.is_symlink():
        return False
    try:
        resolved_root = root.resolve(strict=True)
        resolved_binary = binary.resolve(strict=True)
        resolved_binary.relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    return binary.is_file() and os.access(binary, os.X_OK)


def _is_managed_strata_installed() -> bool:
    """Return whether the stable Strata executable points inside Arkai's install root."""
    return _is_managed_executable(get_strata_binary(), get_strata_root())


def _is_tool_installed(tool_name: str) -> bool:
    if tool_name not in SUPPORTED_TOOLS:
        return False
    return _is_managed_executable(_tool_binary(tool_name), _tool_root(tool_name))


def get_managed_strata_binary() -> Path | None:
    """Return Arkai's Strata executable when its managed install is valid."""
    return get_strata_binary() if _is_managed_strata_installed() else None


def _ensure_managed_root(root: Path, entries: tuple[str, ...]) -> Path:
    """Create a private tool root and reject symlinks that could redirect writes."""
    if root.is_symlink():
        raise RuntimeError(f"Refusing to use a symlink as a managed tool root: {root}")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)

    for entry in entries:
        path = root / entry
        if path.is_symlink():
            raise RuntimeError(f"Refusing to use a symlink in the managed tool directory: {path}")
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Managed tool path is not a directory: {path}")
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)
    return root


def _ensure_install_root() -> Path:
    """Create private Strata directories and reject symlinks that could redirect writes."""
    root = _ensure_managed_root(get_strata_root(), _MANAGED_ENTRIES)

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
    runtime_roots: set[Path] = set()

    node_path = tool_paths.get("node")
    if node_path:
        node_root = Path(node_path).parent.parent
        if (
            node_root not in {Path("/"), Path("/usr"), Path("/usr/local"), Path("/opt")}
            and (node_root / "lib").is_dir()
            and (node_root / "bin" / "node").exists()
        ):
            runtime_roots.add(node_root)

    npm_path = tool_paths.get("npm")
    if npm_path:
        for parent in Path(npm_path).parents:
            package_json = parent / "package.json"
            try:
                package = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(package, dict) and package.get("name") == "npm":
                runtime_roots.add(parent)
                break

    normalized_roots: list[Path] = []
    for root in sorted(runtime_roots, key=lambda path: len(path.parts)):
        if not any(root.is_relative_to(parent) for parent in normalized_roots):
            normalized_roots.append(root)

    directories = {
        directory
        for directory in command_directories
        if not any(Path(directory).is_relative_to(root) for root in normalized_roots)
    }
    directories.update(str(root) for root in normalized_roots)
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


def _strata_environment(root: Path) -> dict[str, str]:
    """Return Strata-specific child environment without overriding Arapuca values."""
    return {
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_STATE_HOME": str(root / "state"),
        "XDG_DATA_HOME": str(root / "data"),
        "STRATA_DATA_DIR": str(root / "data"),
        "STRATA_BIN_DIR": str(root / "bin"),
    }


def _sandbox_environment(root: Path, path: str) -> dict[str, str]:
    """Legacy test/compatibility helper for the Strata child environment."""
    return {"PATH": path, **_strata_environment(root)}


def _build_sandbox_prefix(
    root: Path,
    command_names: tuple[str, ...],
    *,
    allow_network: bool,
    read_only_root: bool = False,
    child_environment: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Create an Arapuca command prefix for one managed tool operation."""
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Strata tool management requires macOS or Linux and Arapuca")

    from arkai.agent import _build_sandbox_cmd, _resolve_sandbox_profile

    tool_paths = _host_tool_paths(command_names)
    for optional_tool in _OPTIONAL_INSTALL_COMMANDS:
        path = shutil.which(optional_tool)
        if path is not None:
            tool_paths[optional_tool] = str(Path(path).resolve())
    if "curl" in command_names and "sh" in command_names:
        if not ("sha256sum" in tool_paths or "shasum" in tool_paths):
            raise RuntimeError("The selected install script requires sha256sum or shasum")

    volumes, path = _sandbox_paths(tool_paths)
    environment = {"PATH": path, **(child_environment or {})}
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


def run_sandboxed_tool(
    root: Path,
    command_names: tuple[str, ...],
    command: list[str],
    *,
    allow_network: bool,
    child_environment: dict[str, str] | None = None,
    read_only_root: bool = False,
    capture: bool = False,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Run an external tool with its writable scope confined to an Arapuca root."""
    selected_commands = tuple(dict.fromkeys(("env", *command_names)))
    prefix, tool_paths, environment = _build_sandbox_prefix(
        root,
        selected_commands,
        allow_network=allow_network,
        read_only_root=read_only_root,
        child_environment=child_environment,
    )
    if command and Path(command[0]).name in tool_paths:
        command = [tool_paths[Path(command[0]).name], *command[1:]]
    return _run_sandboxed(
        prefix,
        command,
        env_binary=tool_paths["env"],
        environment=environment,
        capture=capture,
        timeout=timeout,
    )


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


def _report_command_output(stdout: str, stderr: str) -> None:
    """Show captured tool output while preserving its stdout/stderr distinction."""
    if stdout.strip():
        utils.info(stdout.rstrip())
    if stderr.strip():
        utils.warn(stderr.rstrip())


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
    display_name = _TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    if action in {"install", "update"}:
        utils.warn(
            f"This will {action} {display_name} inside Arapuca using its supported installer. "
            f"Files will be kept under {location}."
        )
    else:
        utils.warn(f"This will remove the Arkai-managed {display_name} installation at {location}.")
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
        child_environment=_strata_environment(root),
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
        child_environment=_strata_environment(root),
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


def _pi_package_json() -> Path:
    module = _get_tool_module("pi")
    return module.get_core_dir() / "lib" / "node_modules" / module.PI_CORE_PACKAGE / "package.json"


def _pi_version() -> str:
    """Read Pi's version from the package installed in Arkai's private npm prefix."""
    package_json = _pi_package_json()
    try:
        package = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read the Arkai-managed Pi version at {package_json}"
        ) from error
    version = package.get("version") if isinstance(package, dict) else None
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"Arkai-managed Pi package has no version: {package_json}")
    return version


def _omp_version() -> str:
    """Read OMP's version through an Arapuca-sandboxed CLI invocation."""
    root = _tool_root("omp")
    _, stdout, stderr = run_sandboxed_tool(
        root,
        (),
        [str(_tool_binary("omp")), "--version"],
        allow_network=False,
        read_only_root=True,
        capture=True,
        timeout=30,
    )
    version = stdout.strip() or stderr.strip()
    return version.splitlines()[0] if version else "version unavailable"


def _tool_version(tool_name: str) -> str:
    match tool_name:
        case "strata":
            return _strata_version()
        case "pi":
            return _pi_version()
        case "omp":
            return _omp_version()
        case _:
            raise ValueError(f"Tool is not supported: {tool_name}")


def _install_or_update_pi(action: str) -> None:
    """Install or update Pi and Arkai's supported extensions inside Arapuca."""
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Pi tool management requires macOS or Linux and Arapuca")
    module = _get_tool_module("pi")
    root = module.get_install_dir()
    if not _confirm(action, "pi", root):
        utils.info(f"Pi {action} cancelled.")
        return

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm was not found. Install Node.js and npm, then retry.")
    module._resolve_node_path()
    root = _ensure_managed_root(root, _PI_MANAGED_ENTRIES)
    core_dir = module.get_core_dir()
    agent_dir = module.get_agent_dir()
    environment = {
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "npm_config_cache": str(root / "npm-cache"),
    }

    verb = "Installing" if action == "install" else "Updating"
    utils.info(f"{verb} {module.PI_CORE_PACKAGE} inside Arapuca...")
    _, stdout, stderr = run_sandboxed_tool(
        root,
        ("npm", "node"),
        [
            npm_path,
            "install",
            "--prefix",
            str(core_dir),
            "-g",
            "--ignore-scripts",
            module.PI_CORE_PACKAGE,
        ],
        allow_network=True,
        child_environment=environment,
        capture=True,
        timeout=None,
    )
    _report_command_output(stdout, stderr)

    pi_path = core_dir / "bin" / "pi"
    if action == "install":
        packages_to_install = module.PI_PACKAGES
        for package in packages_to_install:
            utils.info(f"Installing {package} inside Arapuca...")
            _, stdout, stderr = run_sandboxed_tool(
                root,
                ("node", "npm"),
                [str(pi_path), "install", f"npm:{package}"],
                allow_network=True,
                child_environment=environment,
                capture=True,
                timeout=None,
            )
            _report_command_output(stdout, stderr)
    else:
        utils.info("Checking Arkai-installed Pi extensions for updates inside Arapuca...")
        updates = module._check_for_package_updates(str(pi_path), str(agent_dir))
        available_sources = {update.get("source") for update in updates if isinstance(update, dict)}
        packages_to_update = [
            package for package in module.PI_PACKAGES if f"npm:{package}" in available_sources
        ]
        if not packages_to_update:
            utils.info("Arkai-installed Pi extensions are already up to date.")
        for package in packages_to_update:
            utils.info(f"Updating {package} inside Arapuca...")
            _, stdout, stderr = run_sandboxed_tool(
                root,
                ("node", "npm"),
                [str(pi_path), "update", "--extension", f"npm:{package}"],
                allow_network=True,
                child_environment=environment,
                capture=True,
                timeout=None,
            )
            _report_command_output(stdout, stderr)

    if not _is_tool_installed("pi"):
        raise RuntimeError(f"Pi installer completed without creating an executable at {pi_path}")
    utils.info(f"Pi {action} complete ({_pi_version()}): {core_dir}")


def _install_or_update_omp(action: str) -> None:
    """Install or update OMP's verified binary inside Arapuca."""
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("OMP tool management requires macOS or Linux and Arapuca")
    module = _get_tool_module("omp")
    root = module.get_install_dir()
    if not _confirm(action, "omp", root):
        utils.info(f"OMP {action} cancelled.")
        return

    root = _ensure_managed_root(root, _OMP_MANAGED_ENTRIES)
    core_dir = module.get_core_dir()
    installer = root / "tmp" / "install.sh"
    environment = {"PI_INSTALL_DIR": str(core_dir)}
    prefix, tools, sandbox_environment = _build_sandbox_prefix(
        root,
        _INSTALL_COMMANDS,
        allow_network=True,
        child_environment=environment,
    )

    utils.info("Downloading the official OMP installer inside Arapuca...")
    _run_sandboxed(
        prefix,
        [tools["curl"], "-fsSL", module.OMP_INSTALLER_URL, "-o", str(installer)],
        env_binary=tools["env"],
        environment=sandbox_environment,
        timeout=None,
    )
    utils.info(f"Running the OMP installer inside Arapuca ({action})...")
    _run_sandboxed(
        prefix,
        [tools["sh"], str(installer), "--binary"],
        env_binary=tools["env"],
        environment=sandbox_environment,
        timeout=None,
    )

    if not _is_tool_installed("omp"):
        raise RuntimeError(
            "OMP installer completed without creating an executable at "
            f"{module.get_core_dir() / 'omp'}"
        )
    utils.info(f"OMP {action} complete ({_omp_version()}): {module.get_core_dir() / 'omp'}")


def _install_or_update_tool(tool_name: str, action: str) -> None:
    if tool_name == "strata":
        _install_or_update_strata(action)
    elif tool_name == "pi":
        _install_or_update_pi(action)
    elif tool_name == "omp":
        _install_or_update_omp(action)
    else:
        raise ValueError(f"Tool is not supported: {tool_name}")


def _managed_entries(tool_name: str) -> tuple[str, ...]:
    if tool_name == "strata":
        return _MANAGED_ENTRIES
    if tool_name == "pi":
        return _PI_MANAGED_ENTRIES
    if tool_name == "omp":
        return _OMP_MANAGED_ENTRIES
    raise ValueError(f"Tool is not supported: {tool_name}")


def cmd_tools_list() -> None:
    """List Arkai-managed tools and their reported versions."""
    installed = [name for name in SUPPORTED_TOOLS if _is_tool_installed(name)]
    if not installed:
        utils.info("No Arkai-managed tools installed")
        return

    utils.info("Arkai-managed tools:")
    for name in installed:
        version = _tool_version(name)
        utils.info(f"  {name}: {version}")


def cmd_tools_install(tool_name: str | None = None) -> None:
    """Install a supported third-party tool, or list supported installers."""
    if tool_name is None:
        utils.info("Tools supported for installation: " + ", ".join(SUPPORTED_TOOLS))
        return
    _validate_tool(tool_name)
    if _is_tool_installed(tool_name):
        raise RuntimeError(
            f"{tool_name} is already installed; use 'arkai tools update {tool_name}'"
        )
    _install_or_update_tool(tool_name, "install")


def cmd_tools_update(tool_name: str) -> None:
    """Update a supported Arkai-managed tool."""
    _validate_tool(tool_name)
    if not _is_tool_installed(tool_name):
        raise RuntimeError(
            f"Arkai-managed {tool_name} is not installed; use 'arkai tools install {tool_name}'"
        )
    _install_or_update_tool(tool_name, "update")


def cmd_tools_remove(tool_name: str) -> None:
    """Remove only the Arkai-managed installation of a supported tool."""
    _validate_tool(tool_name)
    if not _is_tool_installed(tool_name):
        raise RuntimeError(f"Arkai-managed {tool_name} is not installed")

    root = _tool_root(tool_name)
    if not _confirm("remove", tool_name, root):
        utils.info(f"{_TOOL_DISPLAY_NAMES[tool_name]} removal cancelled.")
        return

    entries = _managed_entries(tool_name)
    run_sandboxed_tool(
        root,
        ("rm",),
        [shutil.which("rm") or "rm", "-rf", *entries],
        allow_network=False,
    )
    try:
        root.rmdir()
    except OSError as error:
        remaining = ", ".join(sorted(path.name for path in root.iterdir()))
        raise RuntimeError(
            f"{_TOOL_DISPLAY_NAMES[tool_name]} files were removed, but unexpected files remain "
            f"in {root}: {remaining}"
        ) from error
    utils.info(f"Removed Arkai-managed {_TOOL_DISPLAY_NAMES[tool_name]}")
