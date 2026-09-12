"""Claude-specific paths, configuration, and launch command construction."""

import json
import os
import tempfile
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _get_config_dir() -> Path:
    """Return the temporary session directory used by Claude Code."""
    config_dir = Path(os.path.expanduser("~/.local/state/arkai/sessions"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_agent_path(cfg: dict) -> str:
    """Resolve the configured Claude Code executable."""
    return utils.resolve_binary(config.get_config_value(cfg, "agent.path") or "claude")


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
    """Create Claude configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _get_model_name

    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    context_size = config.get_config_value(cfg, "inference.context_size", 65536)
    model_name = _get_model_name(cfg)
    vertex_env_vars = [
        "CLAUDE_CODE_USE_VERTEX",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "GOOGLE_CLOUD_LOCATION",
        "ANTHROPIC_MODEL",
    ]
    anthropic_env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{inference_port}",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_API_KEY": "local",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_size),
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
    }
    command = [agent_path]
    if prompt is not None:
        command.extend(["-p", prompt, "--model", model_name])
    else:
        command.extend(["--model", model_name])

    cleanup_paths: list[Path] = []
    mcp_config: Path | None = None
    if wtmcp_port is not None:
        config_dir = _get_config_dir()
        file_descriptor, config_name = tempfile.mkstemp(
            prefix=".mcp-", suffix=".json", dir=config_dir
        )
        os.close(file_descriptor)
        mcp_config = Path(config_name)
        try:
            mcp_config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "wtmcp": {
                                "type": "url",
                                "url": f"http://127.0.0.1:{wtmcp_port}/mcp",
                            }
                        }
                    },
                    indent=2,
                )
            )
        except Exception:
            mcp_config.unlink(missing_ok=True)
            raise
        cleanup_paths.append(mcp_config)
        command.extend(["--mcp-config", str(mcp_config)])

    environment = os.environ.copy()
    if use_sandbox:
        sandbox_config_dir = str(_get_config_dir()) if mcp_config else None
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                sandbox_config_dir,
                wtmcp_port,
                sandbox_profile,
                cli_volumes,
                cli_environment,
            )
        except Exception:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
            raise
        unset_args = [arg for var in vertex_env_vars if var in environment for arg in ("-u", var)]
        env_pairs = [f"{key}={value}" for key, value in anthropic_env.items()]
        command = sandbox_prefix + ["env"] + unset_args + env_pairs + command
    else:
        for var in vertex_env_vars:
            environment.pop(var, None)
        environment.update(anthropic_env)

    return AgentLaunchSpec(command, environment, cleanup_paths)


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
    """Build and run Claude's launch specification."""
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
