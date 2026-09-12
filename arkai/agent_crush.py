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
    from arkai.agent import _build_sandbox_cmd, _get_model_name

    config_dir = _get_config_dir()
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_name = _get_model_name(cfg)
    config_data: dict = {
        "providers": {
            "local-llm": {
                "type": "llamacpp",
                "base_url": f"http://127.0.0.1:{inference_port}",
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
    file_descriptor, config_name = tempfile.mkstemp(
        prefix=".crush-", suffix=".json", dir=config_dir
    )
    os.close(file_descriptor)
    config_file = Path(config_name)
    try:
        config_file.write_text(json.dumps(config_data, indent=2))
    except Exception:
        config_file.unlink(missing_ok=True)
        raise

    environment = os.environ.copy()
    if use_sandbox:
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(config_dir),
                wtmcp_port,
                sandbox_profile,
                cli_volumes,
                cli_environment,
            )
        except Exception:
            config_file.unlink(missing_ok=True)
            raise
        command = sandbox_prefix + ["env", f"CRUSH_CONFIG={config_file}", agent_path]
    else:
        environment["CRUSH_CONFIG"] = str(config_file)
        command = [agent_path]

    if prompt is not None:
        command.extend(["run", prompt])
    else:
        command.extend(["--cwd", str(config_dir)])
    return AgentLaunchSpec(command, environment, [config_file])


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
