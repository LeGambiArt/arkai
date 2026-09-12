"""Pi-specific paths, installation, configuration, and launch construction."""

import json
import os
import shutil
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec

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


def get_runtime_volumes(agent_path: str) -> list[str]:
    """Return read-only volumes needed by Pi's Node launcher in a sandbox."""
    bin_dir = Path(os.path.abspath(agent_path)).parent
    prefix = bin_dir.parent
    return [f"{prefix}:ro"] if prefix.exists() else []


def _prepare_agent_dir(cfg: dict, wtmcp_port: int | None) -> str:
    """Write Pi's model and optional MCP configuration files."""
    from arkai.agent import _get_model_name

    agent_dir = get_agent_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
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
                        "id": model_name,
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
    from arkai.agent import _build_sandbox_cmd, _get_model_name

    mcp_available = wtmcp_port is not None and wtmcp.is_wtmcp_running(wtmcp_port)
    agent_dir = _prepare_agent_dir(cfg, wtmcp_port if mcp_available else None)
    model_name = _get_model_name(cfg)
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir
    if use_sandbox:
        runtime_volumes = list(cli_volumes or [])
        runtime_volumes.extend(get_runtime_volumes(agent_path))
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
        ) + [agent_path]
    else:
        cmd = [agent_path]

    cmd.extend(["--model", f"local-llm/{model_name}"])
    if mcp_available:
        cmd.extend(["--mcp-config", str(Path(agent_dir, "mcp.json"))])
    if prompt is not None:
        cmd.extend(["-p", prompt])

    return AgentLaunchSpec(cmd, env)


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


def _run_install_command(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run an installation command and raise an actionable error on failure."""
    code, _, stderr = utils.run_command(command, timeout=None, env=env)
    if code != 0:
        detail = stderr.strip() or f"exit code {code}"
        raise RuntimeError(f"Installation command failed ({' '.join(command)}): {detail}")


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
    _run_install_command(
        [npm_path, "install", "--prefix", str(core_dir), "-g", "--ignore-scripts", PI_CORE_PACKAGE]
    )
    pi_path = core_dir / "bin" / "pi"
    install_env = os.environ.copy()
    install_env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    for package in PI_PACKAGES:
        utils.info(f"Installing {package}...")
        _run_install_command([str(pi_path), "install", f"npm:{package}"], env=install_env)
    utils.info(f"Pi installation complete: {core_dir}")
