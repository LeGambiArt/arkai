"""Pi-specific paths, installation, configuration, and launch construction."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_install_command, run_launch_spec

PI_CORE_PACKAGE = "@earendil-works/pi-coding-agent"
PI_PACKAGES = ("pi-mcp-adapter", "pi-web-access", "pi-subagents")


def get_install_dir() -> Path:
    """Return the root directory for Arkai-managed Pi files."""
    return Path(os.path.expanduser("~/.local/state/arkai/pi"))


def get_core_dir() -> Path:
    """Return the directory containing Pi's locally installed executable."""
    return get_install_dir() / "core"


def get_agent_path(cfg: dict) -> str:
    """Resolve Pi's configured executable or Arkai-managed default."""
    configured_path = config.get_config_value(cfg, "agent.path")
    agent_path = configured_path or str(get_core_dir() / "bin" / "pi")
    return utils.resolve_binary(agent_path)


def get_agent_dir() -> Path:
    """Return the directory containing Pi's Arkai-managed runtime data."""
    return get_install_dir() / "agent"


def _resolve_node_path() -> str:
    """Resolve the Node executable used to run Pi."""
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("Node.js was not found in PATH; it is required to run Pi in a sandbox")
    return str(Path(node_path).resolve())


def get_runtime_volumes(agent_path: str, node_path: str) -> list[str]:
    """Return read-only volumes needed by Pi and its Node runtime in a sandbox."""
    bin_dir = Path(os.path.abspath(agent_path)).parent
    prefixes = [bin_dir.parent, Path(node_path).resolve().parent.parent]
    volumes: list[str] = []
    for prefix in prefixes:
        if prefix.exists():
            volume = f"{prefix}:ro"
            if volume not in volumes:
                volumes.append(volume)
    return volumes


def _create_node_path_preload(
    agent_dir: str, node_path: str, enable_osc52_fallback: bool = False
) -> Path:
    """Create a per-launch Node preload that exposes Node tools to Pi's children."""
    descriptor, preload_name = tempfile.mkstemp(
        prefix="arkai-pi-env-", suffix=".cjs", dir=agent_dir
    )
    preload_path = Path(preload_name)
    node_bin = str(Path(node_path).parent)
    preload = (
        f"const nodeBin = {json.dumps(node_bin)};\n"
        f"const pathSeparator = {json.dumps(os.pathsep)};\n"
        'process.env.PATH = nodeBin + pathSeparator + (process.env.PATH || "");\n'
    )
    if enable_osc52_fallback:
        preload += (
            "if (!process.env.SSH_CONNECTION && !process.env.SSH_CLIENT && "
            "!process.env.MOSH_CONNECTION) {\n"
            '    process.env.MOSH_CONNECTION = "arkai-local";\n'
            "}\n"
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as preload_file:
            preload_file.write(preload)
    except Exception:
        preload_path.unlink(missing_ok=True)
        raise
    return preload_path


def _is_arkai_managed_pi(agent_path: str) -> bool:
    """Return whether the selected Pi executable is Arkai's managed install."""
    try:
        return Path(agent_path).resolve() == (get_core_dir() / "bin" / "pi").resolve()
    except (OSError, RuntimeError):
        return False


def _check_for_package_updates(agent_path: str, agent_dir: str) -> list[dict[str, str]]:
    """Use Pi's own package manager to check Arkai-installed extensions for updates."""
    package_root = Path(agent_path).resolve().parents[2]
    entrypoint = package_root / "dist" / "index.js"
    if not entrypoint.is_file():
        raise RuntimeError(f"Pi package manager entry point not found: {entrypoint}")

    package_names = json.dumps(PI_PACKAGES)
    script = (
        f"const {{ DefaultPackageManager, SettingsManager }} = "
        f"await import({json.dumps(entrypoint.as_uri())});\n"
        "const agentDir = process.env.PI_CODING_AGENT_DIR;\n"
        "const settingsManager = SettingsManager.create(agentDir, agentDir);\n"
        "const packageManager = new DefaultPackageManager({\n"
        "  cwd: agentDir, agentDir, settingsManager,\n"
        "});\n"
        "const packageNames = new Set(" + package_names + ");\n"
        "const updates = await packageManager.checkForAvailableUpdates();\n"
        "console.log(JSON.stringify(updates.filter((update) =>\n"
        '  update.scope === "user" && update.type === "npm" &&\n'
        "  packageNames.has(update.displayName)\n"
        ")));\n"
    )
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir
    try:
        result = subprocess.run(
            [_resolve_node_path(), "--input-type=module", "-e", script],
            cwd=agent_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Pi package update check failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Pi package update check failed: {detail}")
    try:
        updates = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Pi package update check returned invalid JSON") from error
    if not isinstance(updates, list):
        raise RuntimeError("Pi package update check returned an unexpected result")
    package_names = set(PI_PACKAGES)
    return [
        update
        for update in updates
        if isinstance(update, dict)
        and isinstance(update.get("source"), str)
        and isinstance(update.get("displayName"), str)
        and update.get("scope") == "user"
        and update.get("type") == "npm"
        and update["displayName"] in package_names
    ]


def _maybe_update_arkai_packages(agent_path: str) -> None:
    """Ask before upgrading outdated extensions in Arkai's managed Pi installation."""
    if not _is_arkai_managed_pi(agent_path):
        return

    agent_dir = str(get_agent_dir())
    try:
        updates = _check_for_package_updates(agent_path, agent_dir)
    except RuntimeError as error:
        utils.warn(f"Could not check Arkai-installed Pi packages for updates: {error}")
        return
    if not updates:
        return

    package_list = "\n".join(f"- {update['displayName']}" for update in updates)
    utils.warn(f"Updates are available for Arkai-installed Pi packages:\n{package_list}")
    try:
        answer = input("Upgrade these packages before starting Pi? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer not in {"", "y", "yes"}:
        utils.info("Starting Pi without updating packages.")
        return

    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir
    for update in updates:
        utils.info(f"Updating {update['displayName']}...")
        try:
            result = subprocess.run(
                [agent_path, "update", "--extension", update["source"]],
                cwd=agent_dir,
                env=env,
                check=False,
            )
        except OSError as error:
            raise RuntimeError(f"Could not update {update['displayName']}: {error}") from error
        if result.returncode != 0:
            raise RuntimeError(
                f"Updating {update['displayName']} failed with exit code {result.returncode}"
            )
    utils.info("Pi package updates complete.")


def _prepare_agent_dir(cfg: dict, wtmcp_port: int | None) -> str:
    """Write Pi's model and optional MCP configuration files."""
    from arkai.agent import _get_agent_model_id, _get_model_name

    agent_dir = get_agent_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_id = _get_agent_model_id(cfg)
    model_name = _get_model_name(cfg)
    context_size = config.get_config_value(cfg, "inference.context_size", 65536)
    models = {
        "providers": {
            "local-llm": {
                "baseUrl": f"http://127.0.0.1:{inference_port}/v1",
                "api": "openai-completions",
                "apiKey": "local",
                "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                "models": [
                    {
                        "id": model_id,
                        "name": model_name,
                        "reasoning": False,
                        "contextWindow": context_size,
                        "maxTokens": min(8192, context_size),
                    }
                ],
            }
        }
    }
    Path(agent_dir, "models.json").write_text(json.dumps(models, indent=2))
    if wtmcp_port is not None:
        Path(agent_dir, "mcp.json").write_text(
            json.dumps(
                {"mcpServers": {"wtmcp": {"url": f"http://127.0.0.1:{wtmcp_port}/mcp"}}},
                indent=2,
            )
        )
    else:
        Path(agent_dir, "mcp.json").unlink(missing_ok=True)
    return str(agent_dir)


def build_launch_spec(
    agent_path: str,
    cfg: dict,
    wtmcp_port: int | None,
    use_sandbox: bool,
    workdir: str | None,
    sandbox_profile: str | None = None,
    cli_volumes: list | None = None,
    cli_environment: dict | None = None,
    prompt: str | None = None,
) -> AgentLaunchSpec:
    """Create Pi configuration and return its process launch specification."""
    from arkai import wtmcp
    from arkai.agent import _build_sandbox_cmd, _get_agent_model_id

    mcp_available = wtmcp_port is not None and wtmcp.is_wtmcp_running(wtmcp_port)
    agent_dir = _prepare_agent_dir(cfg, wtmcp_port if mcp_available else None)
    model_id = _get_agent_model_id(cfg)
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir

    agent_args = ["--model", f"local-llm/{model_id}"]
    theme = config.get_config_value(cfg, "agent.theme")
    if theme is not None:
        agent_args.extend(["--use-theme", theme])
    if mcp_available:
        agent_args.extend(["--mcp-config", str(Path(agent_dir, "mcp.json"))])
    if prompt is not None:
        agent_args.extend(["-p", prompt])

    cleanup_paths: list[Path] = []
    if use_sandbox:
        node_path = _resolve_node_path()
        runtime_volumes = list(cli_volumes or [])
        runtime_volumes.extend(get_runtime_volumes(agent_path, node_path))
        sandbox_environment = dict(cli_environment or {})
        sandbox_environment["PI_CODING_AGENT_DIR"] = agent_dir
        cmd = _build_sandbox_cmd(
            cfg,
            workdir,
            agent_dir,
            None,
            sandbox_profile,
            runtime_volumes,
            sandbox_environment,
            tty=prompt is None,
        )
        preload_path = _create_node_path_preload(
            agent_dir,
            node_path,
            enable_osc52_fallback=(
                prompt is None
                and platform.system() == "Darwin"
                and os.environ.get("TERM_PROGRAM") != "Apple_Terminal"
            ),
        )
        cleanup_paths.append(preload_path)
        cmd = cmd + [node_path, "--require", str(preload_path), str(Path(agent_path).resolve())]
    else:
        cmd = [agent_path]

    cmd.extend(agent_args)

    return AgentLaunchSpec(cmd, env, cleanup_paths)


def start(
    agent_path: str,
    cfg: dict,
    wtmcp_port: int | None,
    use_sandbox: bool,
    workdir: str | None,
    sandbox_profile: str | None = None,
    cli_volumes: list | None = None,
    cli_environment: dict | None = None,
    prompt: str | None = None,
    capture_stdout: bool = False,
) -> str | None:
    """Build and run Pi's launch specification."""
    if prompt is None and sys.stdin.isatty() and sys.stdout.isatty():
        _maybe_update_arkai_packages(agent_path)
    return run_launch_spec(
        build_launch_spec(
            agent_path,
            cfg,
            wtmcp_port,
            use_sandbox,
            workdir,
            sandbox_profile,
            cli_volumes,
            cli_environment,
            prompt,
        ),
        capture_stdout,
    )


def install() -> None:
    """Install Pi and its supported package extensions."""
    utils.warn("This will install:\n- pi.dev\n- pi-mcp-adapter\n- pi-web-access\n- pi-subagents")
    try:
        answer = input("Proceed with the installation? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in {"y", "yes"}:
        utils.info("Pi installation cancelled.")
        return

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm was not found. Install Node.js and npm, then retry.")
    core_dir = get_core_dir()
    agent_dir = get_agent_dir()
    utils.info(f"Installing {PI_CORE_PACKAGE}...")
    run_install_command(
        [npm_path, "install", "--prefix", str(core_dir), "-g", "--ignore-scripts", PI_CORE_PACKAGE]
    )
    pi_path = core_dir / "bin" / "pi"
    install_env = os.environ.copy()
    install_env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    for package in PI_PACKAGES:
        utils.info(f"Installing {package}...")
        run_install_command([str(pi_path), "install", f"npm:{package}"], env=install_env)
    utils.info(f"Pi installation complete: {core_dir}")
