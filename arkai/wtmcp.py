"""wtmcp plugin management and server lifecycle."""

import os
import subprocess
import time
from typing import Optional

from arkai import config, utils


def cmd_wtmcp_list(port: Optional[int] = None) -> None:
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


def is_wtmcp_running(port: Optional[int] = None) -> bool:
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

        # Check if process still exists
        try:
            code, _, _ = utils.run_command(["kill", "-0", str(pid)])
            return code == 0
        except RuntimeError:
            return False
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
                    try:
                        code, _, _ = utils.run_command(["kill", "-0", str(pid)])
                        if code == 0:
                            return True
                    except RuntimeError:
                        pass
        return False


def cmd_wtmcp_start(
    path: Optional[str] = None,
    port: Optional[int] = None,
    enable_plugins: Optional[list] = None,
    disable_plugins: Optional[list] = None,
) -> None:
    """Start wtmcp server with project configuration.

    Args:
        path: Override wtmcp binary path from config
        port: Override wtmcp port from config
        enable_plugins: List of plugins to enable (overrides config)
        disable_plugins: List of plugins to disable (overrides config)

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

    # Get workdir from config
    workdir = config.get_config_value(cfg, "wtmcp.workdir")
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

    # Create wtmcp configuration with effective plugins
    wtmcp_config = {"mcp-servers": {}}
    for plugin_name in effective_plugins:
        wtmcp_config["mcp-servers"][plugin_name] = {"command": f"uvx {plugin_name}"}

    # Write temporary wtmcp config file
    pid_dir = utils.get_pid_dir()
    wtmcp_config_path = os.path.join(pid_dir, f"wtmcp-{port}.config.yaml")
    utils.save_yaml(wtmcp_config_path, wtmcp_config)

    # Start wtmcp server in background
    cmd = [
        wtmcp_path,
        "serve",
        "--port",
        str(port),
        "--config",
        wtmcp_config_path,
        "--transport",
        "streamable-http",
    ]
    if workdir:
        cmd.extend(["--workdir", workdir])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write PID
    pid_path = get_wtmcp_pid_path(port)
    utils.write_pid(pid_path, proc.pid)

    # Save server state with full context
    state = {
        "port": port,
        "workdir": workdir,
        "wtmcp_path": wtmcp_path,
        "config_file": config_file,
        "wtmcp_config_file": wtmcp_config_path,
        "startup_dir": os.getcwd(),
        "enable_plugins": enable_plugins or [],
        "disable_plugins": disable_plugins or [],
        "effective_plugins": effective_plugins,
    }
    utils.save_yaml(get_wtmcp_state_path(port), state)

    # Brief wait to check if process starts successfully
    time.sleep(0.5)
    if not is_wtmcp_running(port):
        raise RuntimeError("wtmcp server failed to start")

    utils.info(f"wtmcp server started on port {port}")


def cmd_wtmcp_stop(port: Optional[int] = None) -> None:
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
    pid = utils.read_pid(pid_path)

    if pid is None:
        utils.info(f"wtmcp server not running on port {port}")
        return

    utils.info(f"Stopping wtmcp server on port {port} (PID {pid})...")
    utils.kill_process(pid)

    # Poll for process termination (up to 10 seconds)
    for _ in range(20):
        try:
            code, _, _ = utils.run_command(["kill", "-0", str(pid)])
            if code != 0:
                break
        except RuntimeError:
            break
        time.sleep(0.5)

    if os.path.exists(pid_path):
        os.remove(pid_path)

    # Clean up state file
    state_path = get_wtmcp_state_path(port)
    if os.path.exists(state_path):
        # Load state to get wtmcp config file path
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
    if not os.path.exists(pid_dir):
        return running_ports

    for filename in os.listdir(pid_dir):
        if filename.startswith("wtmcp-") and filename.endswith(".pid"):
            try:
                port = int(filename[6:-4])  # Extract port from "wtmcp-<port>.pid"
                if is_wtmcp_running(port):
                    running_ports.append(port)
            except (ValueError, IndexError):
                pass

    return sorted(running_ports)


def cmd_wtmcp_status(port: Optional[int] = None) -> None:
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
            utils.info(f"Status: running (PID {pid})")

            # Load state saved at startup
            state_path = get_wtmcp_state_path(port)
            try:
                state = utils.load_yaml(state_path)
                port = state.get("port", 8080)
                workdir = state.get("workdir")
                config_file = state.get("config_file")
                startup_dir = state.get("startup_dir")

                utils.info(f"Port: {port}")
                if workdir:
                    utils.info(f"Workdir: {workdir}")
                if config_file:
                    utils.info(f"Config: {config_file}")
                else:
                    utils.info("Config: (none)")
                utils.info(f"Startup dir: {startup_dir}")
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
            pid = utils.read_pid(pid_path)
            utils.info(f"  Port {p} (PID {pid})")

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
