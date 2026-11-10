"""Container engine abstraction for running vectordb servers as containers.

Supports podman, Apple's `container` runtime, and docker. All engines share
the `run -d --name -p -v -e` flag surface; differences are confined to
readiness probes and container inspection.
"""

import shutil
import subprocess
import sys

from arkai import config, utils

ENGINES = ("podman", "apple", "docker")
ENGINE_BINARIES = {"podman": "podman", "apple": "container", "docker": "docker"}

_NOT_READY_HINTS = {
    "apple": ("Apple container services are not running; start them with 'container system start'"),
    "podman": (
        "Podman is not ready (is the podman machine VM running?); "
        "start it with 'podman machine start'"
    ),
    "docker": (
        "Docker is not ready (is Docker Desktop or colima running?); "
        "start Docker Desktop or run 'colima start'"
    ),
}

_LOGS_HINTS = {
    "podman": "check container logs with 'podman logs arkai-vectordb'",
    "docker": "check container logs with 'docker logs arkai-vectordb'",
    "apple": "check services with 'container system status'",
}


_IS_MACOS = sys.platform == "darwin"


def _engine_available(engine: str) -> bool:
    """Check whether an engine's CLI binary is usable on this host."""
    if engine == "apple" and not _IS_MACOS:
        return False
    return shutil.which(ENGINE_BINARIES[engine]) is not None


def resolve_engine(cfg: dict | None = None) -> str:
    """Resolve the container engine to use.

    Reads vectordb.engine from the given config (default 'auto'). For 'auto',
    returns the first available engine in order podman, apple, docker.

    Args:
        cfg: Loaded configuration dict (None reads the on-disk config).

    Returns:
        Engine name in ENGINES.

    Raises:
        RuntimeError: If no engine is available or the configured engine is
            unknown/missing.
    """
    if cfg is None:
        cfg = config.load_config()
    engine = config.get_config_value(cfg, "vectordb.engine", "auto")
    if engine not in ("auto", *ENGINES):
        raise RuntimeError(
            f"vectordb.engine must be 'auto', 'podman', 'apple' or 'docker', got {engine}"
        )

    if engine == "auto":
        for candidate in ENGINES:
            if _engine_available(candidate):
                return candidate
        raise RuntimeError(
            "No container engine found (looked for: podman, container, docker). "
            "Install podman (https://podman.io), Apple's container runtime "
            "(macOS 26+, `brew install container`) or Docker "
            "(https://docker.com) and try again."
        )

    if not _engine_available(engine):
        if engine == "apple":
            raise RuntimeError(
                "Apple's container runtime is only available on macOS on Apple "
                "silicon (macOS 26 or newer); configure vectordb.engine to "
                "'podman' or 'docker' instead"
            )
        binary = ENGINE_BINARIES[engine]
        raise RuntimeError(
            f"Container engine '{engine}' not found (binary '{binary}' is not on "
            "PATH); install it or set vectordb.engine to 'auto'"
        )
    return engine


def engine_system_ready(engine: str) -> bool:
    """Check whether the engine's daemon/VM is running and accepting work."""
    binary = ENGINE_BINARIES[engine]
    if engine == "apple":
        argv = [binary, "system", "status"]
    else:
        argv = [binary, "info"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def ensure_engine_ready(engine: str) -> None:
    """Verify the engine VM/daemon is running; raise a clear error if not.

    Raises:
        RuntimeError: With an actionable remedy when the engine is not ready
            (e.g. 'podman machine start', 'container system start',
            'colima start').
    """
    if engine_system_ready(engine):
        return
    stderr = ""
    binary = ENGINE_BINARIES[engine]
    argv = [binary, "system", "status"] if engine == "apple" else [binary, "info"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        stderr = result.stderr or ""
    except (OSError, subprocess.SubprocessError):
        pass
    raise _not_ready_error(engine, stderr)


def _not_ready_error(engine: str, stderr: str | None = None) -> RuntimeError:
    """Build a clear error for an engine whose VM/daemon is not running."""
    message = f"Container engine '{engine}' is not ready: {_NOT_READY_HINTS[engine]}"
    detail = (stderr or "").strip()
    if detail:
        message += f"\nEngine reported: {detail[-500:]}"
    return RuntimeError(message)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    """Run an engine command, capturing output for diagnostics."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)


def run_container(
    engine: str,
    image: str,
    container_name: str,
    port: int,
    host_port: int,
    volumes: list[tuple[str, str]],
    env: dict[str, str] | None = None,
) -> None:
    """Start a detached container with the given image and port mapping.

    Args:
        engine: Engine name in ENGINES.
        image: Container image reference.
        container_name: Name for the container.
        port: Port the container listens on inside the sandbox.
        host_port: Host port to publish (bound to 127.0.0.1).
        volumes: List of (host_path, container_path) bind mounts.
        env: Environment variables for the container.

    Raises:
        RuntimeError: If the engine VM/daemon is not running, or the container
            fails to start (includes exit code and engine stderr).
    """
    binary = ENGINE_BINARIES[engine]
    argv = [binary, "run", "-d", "--name", container_name]
    for host_path, container_path in volumes:
        argv += ["-v", f"{host_path}:{container_path}"]
    argv += ["-p", f"127.0.0.1:{host_port}:{port}"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    argv.append(image)

    utils.info(f"Starting {engine} container {container_name} ({image})...")
    try:
        result = _run(argv)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"Failed to run {engine} command: {e}") from e

    if result.returncode != 0:
        # A dead VM/daemon is the most common cause; surface it explicitly.
        if not engine_system_ready(engine):
            raise _not_ready_error(engine, result.stderr)
        stderr = (result.stderr or "").strip()
        message = (
            f"Container engine '{engine}' failed to start container "
            f"(exit code {result.returncode})."
        )
        if stderr:
            message += f"\nEngine reported: {stderr[-500:]}"
        raise RuntimeError(message)


def _tolerates_missing(stderr: str) -> bool:
    """Check whether engine stderr indicates the container does not exist."""
    lowered = (stderr or "").lower()
    return "no such" in lowered or "not found" in lowered


def stop_container(engine: str, container_name: str) -> None:
    """Stop and remove a container; tolerate already-removed containers.

    Raises:
        RuntimeError: If stopping/removing fails for a reason other than the
            container not existing.
    """
    binary = ENGINE_BINARIES[engine]
    for action in ("stop", "rm" if engine != "apple" else "delete"):
        try:
            result = _run([binary, action, container_name])
        except (OSError, subprocess.SubprocessError) as e:
            raise RuntimeError(f"Failed to run {engine} command: {e}") from e
        if result.returncode != 0 and not _tolerates_missing(result.stderr):
            stderr = (result.stderr or "").strip()
            message = (
                f"Container engine '{engine}' failed to {action} container '{container_name}'."
            )
            if stderr:
                message += f"\nEngine reported: {stderr[-500:]}"
            raise RuntimeError(message)


def container_running(engine: str, container_name: str) -> bool:
    """Check whether a named container is currently running."""
    binary = ENGINE_BINARIES[engine]
    if engine == "apple":
        try:
            result = _run([binary, "list", "--format", "json"])
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            return False
        import json

        try:
            entries = json.loads(result.stdout or "[]")
        except ValueError:
            return False
        if isinstance(entries, dict):  # some versions wrap in {"containers": []}
            entries = entries.get("containers", [])
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in ("id", "containerID", "containerId"):
                if entry.get(key) == container_name:
                    return True
        return False

    try:
        result = _run([binary, "inspect", "-f", "{{.State.Running}}", container_name])
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def engine_logs_hint(engine: str) -> str:
    """Return a hint for diagnosing a container that failed to become ready."""
    return _LOGS_HINTS[engine]
