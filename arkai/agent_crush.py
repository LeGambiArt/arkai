"""Crush-specific paths, configuration, and launch command construction."""

import json
import os
import tempfile
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _get_config_dir() -> Path:
    """Return the temporary session directory used by Crush."""
    config_dir = Path(os.path.expanduser("~/.local/state/arkai/sessions"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_agent_path(cfg: dict) -> str:
    """Resolve the configured Crush executable."""
    return utils.resolve_binary(config.get_config_value(cfg, "agent.path") or "crush")


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
    """Create Crush configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _get_agent_model_id

    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_name = _get_agent_model_id(cfg)
    sessions_dir = _get_config_dir()
    crush_config_dir = Path(tempfile.mkdtemp(prefix="crush-", dir=sessions_dir))
    config_file = crush_config_dir / "crush.json"
    config_data: dict = {
        "providers": {
            "local-llm": {
                "name": "Arkai Local Inference",
                "type": "openai-compat",
                "base_url": f"http://127.0.0.1:{inference_port}/v1",
                "api_key": "local",
                "discover_models": False,
                "models": [{"id": model_name, "name": "Arkai Local Inference"}],
            }
        },
        "models": {
            "large": {"model": model_name, "provider": "local-llm"},
            "small": {"model": model_name, "provider": "local-llm"},
        },
    }
    if wtmcp_port is not None:
        config_data["mcp"] = {
            "wtmcp": {"type": "http", "url": f"http://127.0.0.1:{wtmcp_port}/mcp"}
        }
    try:
        config_file.write_text(json.dumps(config_data, indent=2))
    except Exception:
        config_file.unlink(missing_ok=True)
        crush_config_dir.rmdir()
        raise

    environment = os.environ.copy()
    if use_sandbox:
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(crush_config_dir),
                wtmcp_port,
                sandbox_profile,
                cli_volumes,
                cli_environment,
                tty=prompt is None,
            )
        except Exception:
            config_file.unlink(missing_ok=True)
            crush_config_dir.rmdir()
            raise
        command = sandbox_prefix + [
            "env",
            f"CRUSH_GLOBAL_CONFIG={crush_config_dir}",
            f"CRUSH_GLOBAL_DATA={crush_config_dir}",
            agent_path,
        ]
    else:
        environment["CRUSH_GLOBAL_CONFIG"] = str(crush_config_dir)
        environment["CRUSH_GLOBAL_DATA"] = str(crush_config_dir)
        command = [agent_path]

    if prompt is not None:
        command.extend(["run", "--quiet", prompt])
    return AgentLaunchSpec(command, environment, [config_file, crush_config_dir])


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
    """Build and run Crush's launch specification."""
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
