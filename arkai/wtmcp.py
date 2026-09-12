"""wtmcp plugin management and server lifecycle."""

import argparse
import os
import subprocess
import sys
import time

from arkai import config, utils


def exec_cmd(args: dict | None) -> None:
    """Select 'wtmcp' command to execute."""
    match args.wtmcp_cmd:  # ty: ignore[unresolved-attribute]
        case "start":
            cmd_wtmcp_start(
                args.path,  # ty: ignore[unresolved-attribute]
                args.port,  # ty: ignore[unresolved-attribute]
                args.enable_plugins,  # ty: ignore[unresolved-attribute]
                args.disable_plugins,  # ty: ignore[unresolved-attribute]
                args.wtmcp_config,  # ty: ignore[unresolved-attribute]
            )
        case "stop":
            cmd_wtmcp_stop(args.port)  # ty: ignore[unresolved-attribute]
        case "status":
            cmd_wtmcp_status(args.port)  # ty: ignore[unresolved-attribute]
        case "list":
            cmd_wtmcp_list(args.port)  # ty: ignore[unresolved-attribute]
        case "enable":
            cmd_wtmcp_enable(args.plugin)  # ty: ignore[unresolved-attribute]
        case "disable":
            cmd_wtmcp_disable(args.plugin)  # ty: ignore[unresolved-attribute]


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    wtmcp_parser = subparsers.add_parser("wtmcp", help="Manage wtmcp plugins and server")
    wtmcp_subparsers = wtmcp_parser.add_subparsers(dest="wtmcp_cmd", required=True)
    # status subcommand
    status_parser = wtmcp_subparsers.add_parser("status", help="Show wtmcp server status")
    status_parser.add_argument(
        "--port", type=int, help="Port to show status for (or all if not specified)"
    )
    # start subcommand
    start_parser = wtmcp_subparsers.add_parser("start", help="Start wtmcp server")
    start_parser.add_argument("--path", help="Override wtmcp binary path from config")
    start_parser.add_argument("--port", type=int, help="Override port from config")
    start_parser.add_argument(
        "--wtmcp-config",
        dest="wtmcp_config",
        help="Override base wtmcp config file path (default: ~/.config/wtmcp/config.yaml)",
    )
    start_parser.add_argument(
        "--enable",
        action="append",
        dest="enable_plugins",
        help="Enable plugin (can be used multiple times)",
    )
    start_parser.add_argument(
        "--disable",
        action="append",
        dest="disable_plugins",
        help="Disable plugin (can be used multiple times)",
    )
    # stop subcommand
    stop_parser = wtmcp_subparsers.add_parser("stop", help="Stop wtmcp server")
    stop_parser.add_argument("--port", type=int, help="Port of instance to stop")
    # list subcommand
    list_parser = wtmcp_subparsers.add_parser("list", help="List available plugins")
    list_parser.add_argument(
        "--port", type=int, help="Port of running instance to show plugins for"
    )
    # enable subcommand
    enable_parser = wtmcp_subparsers.add_parser("enable", help="Enable a plugin")
    enable_parser.add_argument("plugin", help="Plugin name")
    # disable subcommand
    disable_parser = wtmcp_subparsers.add_parser("disable", help="Disable a plugin")
    disable_parser.add_argument("plugin", help="Plugin name")


def cmd_wtmcp_list(port: int | None = None) -> None:
    """List available wtmcp plugins and show which are enabled.

    Args:
        port: Port of running instance, or None to find the only running instance

    Raises:
        RuntimeError: If no port specified and no or multiple instances are running
    """
    # Get wtmcp binary path
    cfg = config.load_config()
    wtmcp_bin = config.get_config_value(cfg, "wtmcp.path", "wtmcp")
    try:
        wtmcp_path = utils.resolve_binary(wtmcp_bin)
    except RuntimeError:
        raise RuntimeError("wtmcp binary not found. Configure wtmcp.path in .arkai.yaml")

    # Determine which instance to show
    if port is None:
        # Find running instances
        running_ports = _get_running_instances()
        if not running_ports:
            raise RuntimeError("No running wtmcp instances")
        elif len(running_ports) > 1:
            raise RuntimeError(
                f"Multiple wtmcp instances running on ports: {', '.join(map(str, running_ports))}. "
                f"Specify --port to select which one"
            )
        else:
            port = running_ports[0]

    # Load state for the instance to get enabled plugins
    state_path = get_wtmcp_state_path(port)
    try:
        state = utils.load_yaml(state_path)
        enabled_plugins = state.get("effective_plugins", [])
    except FileNotFoundError:
        raise RuntimeError(f"No running wtmcp instance found on port {port}")

    # Run wtmcp check to get available plugins
    try:
        result = subprocess.run(
            [wtmcp_path, "check"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse output to extract discovered plugins
            plugins_list = []
            in_plugins_section = False
            for line in result.stdout.split("\n"):
                if "discovered plugins:" in line:
                    in_plugins_section = True
                    continue
                if in_plugins_section:
                    line = line.strip()
                    if not line or line.startswith("tool discovery:"):
                        break
                    if line.startswith("- "):
                        # Extract plugin name and version (format: "- name vX.Y.Z")
                        plugin_info = line[2:].split()[0:2]  # Get name and version
                        if plugin_info:
                            plugins_list.append(" ".join(plugin_info))

            utils.info(f"=== wtmcp Plugins (port {port}) ===\n")

            # Show discovered plugins with status (sorted by name)
            if plugins_list:
                plugins_list_sorted = sorted(plugins_list, key=lambda x: x.split()[0])
                for plugin_info in plugins_list_sorted:
                    # Format: "name vX.Y.Z"
                    parts = plugin_info.split()
                    plugin_name = parts[0]
                    version = parts[1] if len(parts) > 1 else ""

                    is_enabled = plugin_name in enabled_plugins

                    # Discovered plugins: green if enabled, light grey if disabled
                    status_emoji = "🟢" if is_enabled else "⚪"
                    version_str = f" {version}" if version else ""
                    utils.info(f"{status_emoji} {plugin_name}{version_str}")

                # Show enabled plugins that are NOT discovered (sorted by name)
                discovered_names = [p.split()[0] for p in plugins_list] if plugins_list else []
                not_discovered = sorted([p for p in enabled_plugins if p not in discovered_names])
                for plugin_name in not_discovered:
                    utils.info(f"🔴 {plugin_name} (not discovered)")
            else:
                utils.info("No plugins discovered")
        else:
            utils.info("Failed to list plugins from wtmcp")
    except subprocess.TimeoutExpired:
        utils.info("wtmcp check timed out")
    except Exception as e:
        utils.info(f"Error listing plugins: {e}")


def cmd_wtmcp_enable(plugin_name: str) -> None:
    """Enable a wtmcp plugin in project config.

    Args:
        plugin_name: Name of the plugin to enable

    Raises:
        RuntimeError: If config file not found
    """
    project_config_path = ".arkai.yaml"
    if not os.path.exists(project_config_path):
        raise RuntimeError(
            "Project config not found. Create .arkai.yaml first with 'arkai config init'"
        )

    project_config = utils.load_yaml(project_config_path)
    plugins = project_config.get("wtmcp", {}).get("plugins")

    # Handle None (not present) vs empty list (explicit opt-out)
    if plugins is None:
        # Plugins not in config, start with empty list
        plugins = []
    elif not isinstance(plugins, list):
        plugins = []

    if plugin_name in plugins:
        utils.info(f"Plugin '{plugin_name}' is already enabled")
        return

    plugins.append(plugin_name)
    if "wtmcp" not in project_config:
        project_config["wtmcp"] = {}
    project_config["wtmcp"]["plugins"] = plugins

    utils.save_yaml(project_config_path, project_config)
    utils.info(f"Plugin '{plugin_name}' enabled in .arkai.yaml")


def cmd_wtmcp_disable(plugin_name: str) -> None:
    """Disable a wtmcp plugin in project config.

    Args:
        plugin_name: Name of the plugin to disable

    Raises:
        RuntimeError: If config file not found
    """
    project_config_path = ".arkai.yaml"
    if not os.path.exists(project_config_path):
        raise RuntimeError(
            "Project config not found. Create .arkai.yaml first with 'arkai config init'"
        )

    project_config = utils.load_yaml(project_config_path)
    plugins = project_config.get("wtmcp", {}).get("plugins")

    # Handle None (not present) vs empty list (explicit opt-out)
    if plugins is None or not isinstance(plugins, list):
        plugins = []

    if plugin_name not in plugins:
        utils.info(f"Plugin '{plugin_name}' is not enabled")
        return

    plugins.remove(plugin_name)
    if "wtmcp" not in project_config:
        project_config["wtmcp"] = {}
    project_config["wtmcp"]["plugins"] = plugins

    utils.save_yaml(project_config_path, project_config)
    utils.info(f"Plugin '{plugin_name}' disabled in .arkai.yaml")


def get_wtmcp_pid_path(port: int) -> str:
    """Return path to wtmcp server PID file for given port."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, f"wtmcp-{port}.pid")


def get_wtmcp_state_path(port: int) -> str:
    """Return path to wtmcp server state file for given port."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, f"wtmcp-{port}.state")


def get_wtmcp_server_pid_path(port: int) -> str:
    """Return path to the PID file written by the wtmcp supervisor."""
    pid_dir = utils.get_pid_dir()
    return os.path.join(pid_dir, f"wtmcp-{port}.server.pid")


def _is_process_running(pid: int | None) -> bool:
    """Return whether a PID identifies a currently running process."""
    if pid is None:
        return False
    return utils.is_process_running(pid)


def is_wtmcp_running(port: int | None = None) -> bool:
    """Check if wtmcp server is running on a specific port or any port.

    Args:
        port: Port to check, or None to check if any instance is running

    Returns:
        True if running, False otherwise
    """
    if port is not None:
        # Check specific port
        pid_path = get_wtmcp_pid_path(port)
        pid = utils.read_pid(pid_path)
        if pid is None:
            return False

        if not _is_process_running(pid):
            return False

        state_path = get_wtmcp_state_path(port)
        if not os.path.exists(state_path):
            return True
        try:
            state = utils.load_yaml(state_path)
        except (FileNotFoundError, RuntimeError):
            return False

        supervisor_pid = state.get("supervisor_pid")
        if supervisor_pid is None:
            return True
        server_pid = utils.read_pid(get_wtmcp_server_pid_path(port))
        return _is_process_running(supervisor_pid) and _is_process_running(server_pid)
    else:
        # Check if any instance is running
        pid_dir = utils.get_pid_dir()
        if not os.path.exists(pid_dir):
            return False

        for filename in os.listdir(pid_dir):
            if filename.startswith("wtmcp-") and filename.endswith(".pid"):
                pid_path = os.path.join(pid_dir, filename)
                pid = utils.read_pid(pid_path)
                if pid is not None:
                    if _is_process_running(pid):
                        return True
        return False


def cmd_wtmcp_start(
    path: str | None = None,
    port: int | None = None,
    enable_plugins: list | None = None,
    disable_plugins: list | None = None,
    wtmcp_config: str | None = None,
) -> None:
    """Start wtmcp server with project configuration.

    Args:
        path: Override wtmcp binary path from config
        port: Override wtmcp port from config
        enable_plugins: List of plugins to enable (overrides config)
        disable_plugins: List of plugins to disable (overrides config)
        wtmcp_config: Override base wtmcp config file path

    Raises:
        RuntimeError: If wtmcp binary not found or server fails to start
    """
    # Get wtmcp binary path
    cfg = config.load_config()
    if path is None:
        wtmcp_bin = config.get_config_value(cfg, "wtmcp.path", "wtmcp")
    else:
        wtmcp_bin = path
    try:
        wtmcp_path = utils.resolve_binary(wtmcp_bin)
    except RuntimeError:
        raise RuntimeError("wtmcp binary not found. Configure wtmcp.path in .arkai.yaml")

    # Get port from args or config
    if port is None:
        port = config.get_config_value(cfg, "wtmcp.port", 8080)

    # Check if already running on this port
    if is_wtmcp_running(port):
        utils.info(f"wtmcp server already running on port {port}")
        return

    # Let wtmcp use its own configuration home unless explicitly overridden.
    workdir = config.get_config_value(cfg, "wtmcp.workdir", None)
    if workdir:
        workdir = os.path.expanduser(workdir)

    # Build effective plugin list
    project_config_path = ".arkai.yaml"
    configured_plugins: list = []
    if os.path.exists(project_config_path):
        try:
            project_config = utils.load_yaml(project_config_path)
            plugins = project_config.get("wtmcp", {}).get("plugins")
            if plugins and isinstance(plugins, list):
                configured_plugins = plugins.copy()
        except Exception:
            pass

    # Apply enable/disable overrides
    effective_plugins = configured_plugins.copy()
    if enable_plugins:
        for plugin in enable_plugins:
            if plugin not in effective_plugins:
                effective_plugins.append(plugin)

    if disable_plugins:
        effective_plugins = [p for p in effective_plugins if p not in disable_plugins]

    # Log the effective plugin list
    if enable_plugins or disable_plugins:
        if effective_plugins:
            utils.info(f"Effective plugins: {', '.join(sorted(effective_plugins))}")
        else:
            utils.info("No plugins enabled")

    utils.info(f"Starting wtmcp server on port {port}...")

    # Check port availability
    if utils.is_port_in_use(port):
        raise RuntimeError(f"Port {port} already in use")

    # Get project config file path (if it exists)
    project_config_path = ".arkai.yaml"
    config_file = (
        os.path.abspath(project_config_path) if os.path.exists(project_config_path) else None
    )

    # Determine whether a custom config file needs to be generated.
    # Priority for the base config path: CLI arg > arkai config > default wtmcp config location.
    # If wtmcp_config is explicitly provided or arkai has plugins to inject, we must write a
    # merged config so wtmcp picks up the arkai-managed mcp-servers. Otherwise we let wtmcp
    # use its own defaults (env.d, credentials, etc.) untouched.
    explicit_config = wtmcp_config or config.get_config_value(cfg, "wtmcp.config")

    if explicit_config or effective_plugins:
        if explicit_config:
            base_wtmcp_config_path = os.path.expanduser(explicit_config)
        else:
            base_wtmcp_config_path = os.path.expanduser(
                os.path.join(utils.get_config_home(), "wtmcp", "config.yaml")
            )

        merged: dict = {}
        if os.path.exists(base_wtmcp_config_path):
            try:
                merged = utils.load_yaml(base_wtmcp_config_path) or {}
            except Exception:
                utils.warn(f"Could not load base wtmcp config from {base_wtmcp_config_path}")

        if "mcp-servers" not in merged:
            merged["mcp-servers"] = {}
        for plugin_name in effective_plugins:
            if plugin_name not in merged["mcp-servers"]:
                merged["mcp-servers"][plugin_name] = {"command": f"uvx {plugin_name}"}

        pid_dir = utils.get_pid_dir()
        generated_config_path: str | None = os.path.join(pid_dir, f"wtmcp-{port}.config.yaml")
        utils.save_yaml(generated_config_path, merged)
    else:
        base_wtmcp_config_path = None
        generated_config_path = None

    # Start wtmcp server in background
    cmd = [
        wtmcp_path,
        "serve",
        "--port",
        str(port),
        "--transport",
        "streamable-http",
    ]
    if generated_config_path:
        cmd.extend(["--config", generated_config_path])
    if workdir:
        cmd.extend(["--workdir", workdir])

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "arkai.wtmcp_supervisor",
                "--child-pid-path",
                get_wtmcp_server_pid_path(port),
                "--",
            ]
            + cmd
        )
    except OSError as e:
        raise RuntimeError(f"Failed to start wtmcp server: {e}") from e

    # Write PID
    pid_path = get_wtmcp_pid_path(port)
    utils.write_pid(pid_path, proc.pid)

    # Save server state with full context
    state = {
        "port": port,
        "workdir": workdir,
        "wtmcp_path": wtmcp_path,
        "supervisor_pid": proc.pid,
        "wtmcp_pid_path": get_wtmcp_server_pid_path(port),
        "config_file": config_file,
        "base_wtmcp_config": base_wtmcp_config_path,
        "wtmcp_config_file": generated_config_path,
        "startup_dir": os.getcwd(),
        "enable_plugins": enable_plugins or [],
        "disable_plugins": disable_plugins or [],
        "effective_plugins": effective_plugins,
    }
    utils.save_yaml(get_wtmcp_state_path(port), state)

    # Brief wait to check if process starts successfully
    time.sleep(0.5)
    if not is_wtmcp_running(port):
        utils.kill_process(proc.pid)
        utils.wait_for_process_stop(proc.pid)
        server_pid = utils.read_pid(get_wtmcp_server_pid_path(port))
        if server_pid is not None and _is_process_running(server_pid):
            utils.kill_process(server_pid)
            utils.wait_for_process_stop(server_pid)
        pid_path_cleanup = get_wtmcp_pid_path(port)
        if os.path.exists(pid_path_cleanup):
            os.remove(pid_path_cleanup)
        if generated_config_path and os.path.exists(generated_config_path):
            os.remove(generated_config_path)
        server_pid_path = get_wtmcp_server_pid_path(port)
        if os.path.exists(server_pid_path):
            os.remove(server_pid_path)
        state_path = get_wtmcp_state_path(port)
        if os.path.exists(state_path):
            os.remove(state_path)
        raise RuntimeError("wtmcp server failed to start")

    utils.info(f"wtmcp server started on port {port}")


def cmd_wtmcp_stop(port: int | None = None) -> None:
    """Stop wtmcp server on a specific port or the only running instance.

    Args:
        port: Port of the instance to stop, or None to stop the only instance if one is running

    Raises:
        RuntimeError: If no port specified and multiple instances are running
    """
    # Determine which port to stop
    if port is None:
        # Find running instances
        running_ports = _get_running_instances()
        if not running_ports:
            utils.info("wtmcp server not running")
            return
        elif len(running_ports) == 1:
            port = running_ports[0]
        else:
            raise RuntimeError(
                f"Multiple wtmcp instances running on ports: {', '.join(map(str, running_ports))}. "
                f"Specify --port to select which one to stop."
            )

    pid_path = get_wtmcp_pid_path(port)
    supervisor_pid = utils.read_pid(pid_path)

    if supervisor_pid is None:
        utils.info(f"wtmcp server not running on port {port}")
        return

    server_pid = utils.read_pid(get_wtmcp_server_pid_path(port))
    utils.info(f"Stopping wtmcp server on port {port} (supervisor PID {supervisor_pid})...")
    utils.kill_process(supervisor_pid)

    if not utils.wait_for_process_stop(supervisor_pid):
        raise RuntimeError(
            f"wtmcp supervisor (PID {supervisor_pid}) did not stop after SIGKILL; "
            "PID file preserved"
        )

    if server_pid is not None and _is_process_running(server_pid):
        utils.kill_process(server_pid)
        if not utils.wait_for_process_stop(server_pid):
            raise RuntimeError(
                f"wtmcp server (PID {server_pid}) did not stop after SIGKILL; PID file preserved"
            )

    if os.path.exists(pid_path):
        os.remove(pid_path)
    server_pid_path = get_wtmcp_server_pid_path(port)
    if os.path.exists(server_pid_path):
        os.remove(server_pid_path)

    state_path = get_wtmcp_state_path(port)
    if os.path.exists(state_path):
        try:
            state = utils.load_yaml(state_path)
            wtmcp_config_file = state.get("wtmcp_config_file")
            if wtmcp_config_file and os.path.exists(wtmcp_config_file):
                os.remove(wtmcp_config_file)
        except Exception:
            pass
        os.remove(state_path)

    utils.info("wtmcp server stopped")


def _get_running_instances() -> list:
    """Get list of ports with running wtmcp instances."""
    running_ports = []
    pid_dir = utils.get_pid_dir()
    for filename in os.listdir(pid_dir) or []:
        if filename.startswith("wtmcp-") and filename.endswith(".pid"):
            try:
                port = int(filename[6:-4])  # Extract port from "wtmcp-<port>.pid"
                if is_wtmcp_running(port):
                    running_ports.append(port)
            except (ValueError, IndexError):
                pass

    return sorted(running_ports)


def cmd_wtmcp_status(port: int | None = None) -> None:
    """Show wtmcp server status.

    Args:
        port: Port to show status for, or None to show status for all instances
    """
    if port is not None:
        # Show status for specific port
        pid_path = get_wtmcp_pid_path(port)
        pid = utils.read_pid(pid_path)

        utils.info("=== wtmcp Server Status ===")

        if pid is not None and is_wtmcp_running(port):
            utils.info(f"Status: running (supervisor PID {pid})")

            # Load state saved at startup
            state_path = get_wtmcp_state_path(port)
            try:
                state = utils.load_yaml(state_path)
                port = state.get("port", 8080)
                workdir = state.get("workdir")
                config_file = state.get("config_file")
                startup_dir = state.get("startup_dir")
                server_pid = utils.read_pid(get_wtmcp_server_pid_path(port))

                utils.info(f"Port: {port}")
                if workdir:
                    utils.info(f"Workdir: {workdir}")
                if config_file:
                    utils.info(f"Config: {config_file}")
                else:
                    utils.info("Config: (none)")
                utils.info(f"Startup dir: {startup_dir}")
                if server_pid is not None:
                    utils.info(f"Server PID: {server_pid}")
            except FileNotFoundError:
                utils.info("Status: running (state file missing)")
        else:
            utils.info("Status: stopped")
    else:
        # Show status for all instances
        running_ports = _get_running_instances()

        utils.info("=== wtmcp Server Status ===")

        if not running_ports:
            utils.info("Status: no instances running")
            return

        utils.info(f"Status: {len(running_ports)} instance(s) running")
        for p in running_ports:
            pid_path = get_wtmcp_pid_path(p)
            supervisor_pid = utils.read_pid(pid_path)
            server_pid = utils.read_pid(get_wtmcp_server_pid_path(p))
            utils.info(f"  Port {p} (supervisor PID {supervisor_pid}, server PID {server_pid})")

            # Load state
            state_path = get_wtmcp_state_path(p)
            try:
                state = utils.load_yaml(state_path)
                config_file = state.get("config_file")
                startup_dir = state.get("startup_dir")
                if config_file:
                    utils.info(f"    Config: {config_file}")
                utils.info(f"    Started from: {startup_dir}")
            except FileNotFoundError:
                pass
