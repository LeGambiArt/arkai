"""Shared utilities: paths, processes, GPU detection, YAML, errors."""

import os
import sys
import platform
import signal
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml


def get_config_home() -> str:
    """Return config directory path (~/.aitool or XDG_CONFIG_HOME/aitool)."""
    xdg = os.getenv("XDG_CONFIG_HOME")
    if xdg:
        return os.path.join(xdg, "aitool")

    home = os.getenv("HOME")
    if not home:
        error("HOME not set", 3)

    # macOS: ~/.config/aitool
    return os.path.join(home, ".config", "aitool")


def get_data_home() -> str:
    """Return data directory path (~/.local/share/aitool, never XDG_DATA_HOME)."""
    home = os.getenv("HOME")
    if not home:
        error("HOME not set", 3)

    return os.path.join(home, ".local", "share", "aitool")


def get_pid_dir() -> str:
    """Return PID directory path (~/.local/state/aitool)."""
    home = os.getenv("HOME")
    if not home:
        error("HOME not set", 3)

    return os.path.join(home, ".local", "state", "aitool")


def detect_gpu() -> str:
    """Detect GPU type: metal (arm64 macOS), cuda, rocm, or cpu."""
    system = platform.system()

    if system == "Darwin":
        # Check for arm64 (Apple Silicon)
        code, stdout, _ = run_command(["sysctl", "-n", "hw.optional.arm64"])
        if code == 0 and stdout.strip() == "1":
            return "metal"

    elif system == "Linux":
        # Check NVIDIA CUDA
        if shutil.which("nvidia-smi"):
            code, _, _ = run_command(["nvidia-smi"])
            if code == 0:
                return "cuda"

        # Check AMD ROCm
        if shutil.which("rocm-smi"):
            code, _, _ = run_command(["rocm-smi"])
            if code == 0:
                return "rocm"

    return "cpu"


def is_port_in_use(port: int) -> bool:
    """Check if port is already bound."""
    code, _, _ = run_command(["lsof", "-i", f":{port}"], capture=False)
    return code == 0


def run_command(cmd: list, capture: bool = True) -> Tuple[int, str, str]:
    """Run subprocess command, return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        error(f"Command timeout: {' '.join(cmd)}", 1)
    except FileNotFoundError:
        error(f"Command not found: {cmd[0]}", 3)


def load_yaml(path: str) -> dict:
    """Load and parse YAML file; error if not found or invalid."""
    if not os.path.exists(path):
        error(f"Config file not found: {path}", 2)

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data or {}
    except yaml.YAMLError as e:
        error(f"Invalid YAML in {path}: {e}", 2)


def save_yaml(path: str, data: dict) -> None:
    """Write data to YAML file, creating parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base; override wins."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def read_pid(path: str) -> Optional[int]:
    """Read PID from file; return None if not found or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return None


def write_pid(path: str, pid: int) -> None:
    """Write PID to file, creating parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(str(pid))


def kill_process(pid: int) -> bool:
    """Kill process by PID; return True if successful."""
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, OSError):
        return False


def error(msg: str, code: int = 1) -> None:
    """Print error to stderr and exit with code."""
    print(f"[{timestamp()}] Error: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    """Print warning to stderr."""
    print(f"[{timestamp()}] Warning: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    """Print info message to stdout."""
    print(f"[{timestamp()}] {msg}")


def timestamp() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def resolve_binary(binary_path: str) -> str:
    """Resolve binary path: expand tilde, return absolute path.

    If path is absolute or relative with directory separators, use as-is.
    If path is simple name, try to find in PATH.
    Error if not found.

    Args:
        binary_path: Path to binary (e.g., "llama-server", "/usr/bin/llama-server", "~/custom/bin/llama")

    Returns:
        Absolute path to binary

    Raises:
        SystemExit if binary not found
    """
    # Expand tilde
    expanded = os.path.expanduser(binary_path)

    # If already absolute, verify it exists
    if os.path.isabs(expanded):
        if os.path.exists(expanded) and os.access(expanded, os.X_OK):
            return expanded
        error(f"Binary not found or not executable: {expanded}", 3)

    # If contains directory separator, treat as relative path
    if os.sep in expanded:
        abs_path = os.path.abspath(expanded)
        if os.path.exists(abs_path) and os.access(abs_path, os.X_OK):
            return abs_path
        error(f"Binary not found or not executable: {abs_path}", 3)

    # Simple name: search in PATH
    found = shutil.which(expanded)
    if found:
        return found

    error(f"Binary not found in PATH: {expanded}", 3)
