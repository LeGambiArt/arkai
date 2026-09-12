"""OpenCode-specific paths, configuration, and launch command construction."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _prepare_tui_config(config_dir: str, theme: str | None) -> str | None:
    """Copy the global OpenCode TUI config and apply an optional theme override."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    global_config_home = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    global_tui = global_config_home / "opencode" / "tui.json"
    tui_path = Path(config_dir) / "tui.json"

    if global_tui.exists():
        tui_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(global_tui, tui_path)
    elif theme is not None:
        tui_path.parent.mkdir(parents=True, exist_ok=True)
        tui_path.write_text(
            json.dumps({"$schema": "https://opencode.ai/tui.json", "theme": theme}, indent=2)
        )
    else:
        return None

    if theme is not None:
        try:
            tui_config = json.loads(tui_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid opencode TUI configuration: {tui_path}") from error
        if not isinstance(tui_config, dict):
            raise RuntimeError(f"Invalid opencode TUI configuration: {tui_path}")
        tui_config["theme"] = theme
        tui_path.write_text(json.dumps(tui_config, indent=2))

    return str(tui_path)


def _get_config_dir() -> Path:
    """Return the temporary session directory used by OpenCode."""
    config_dir = Path(os.path.expanduser("~/.local/state/arkai/sessions"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_agent_path(cfg: dict) -> str:
    """Resolve the configured OpenCode executable."""
    return utils.resolve_binary(config.get_config_value(cfg, "agent.path") or "opencode")


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
    """Create OpenCode configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _get_model_name

    config_dir = _get_config_dir()
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    inference_backend = config.get_config_value(cfg, "inference.backend", "llama-cpp")
    model_name = _get_model_name(cfg)
    config_data: dict = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "local-llm": {
                "name": f"Local LLM ({inference_backend})",
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": f"http://127.0.0.1:{inference_port}/v1"},
                "models": {model_name: {"name": model_name}},
            }
        },
        "model": f"local-llm/{model_name}",
    }
    if wtmcp_port is not None:
        config_data["mcp"] = {
            "wtmcp": {
                "type": "remote",
                "url": f"http://127.0.0.1:{wtmcp_port}/mcp",
                "oauth": False,
            }
        }
    file_descriptor, config_name = tempfile.mkstemp(
        prefix="opencode-", suffix=".json", dir=config_dir
    )
    os.close(file_descriptor)
    config_file = Path(config_name)
    try:
        config_file.write_text(json.dumps(config_data, indent=2))
    except Exception:
        config_file.unlink(missing_ok=True)
        raise

    try:
        tui_file = _prepare_tui_config(str(config_dir), config.get_config_value(cfg, "agent.theme"))
    except Exception:
        config_file.unlink(missing_ok=True)
        raise
    cleanup_paths = [config_file]
    if tui_file is not None:
        cleanup_paths.append(Path(tui_file))

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
                tty=prompt is None,
            )
        except Exception:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
            raise
        env_args = ["env", f"OPENCODE_CONFIG={config_file}"]
        if tui_file is not None:
            env_args.append(f"OPENCODE_TUI_CONFIG={tui_file}")
        command = sandbox_prefix + env_args + [agent_path]
    else:
        environment["OPENCODE_CONFIG"] = str(config_file)
        if tui_file is not None:
            environment["OPENCODE_TUI_CONFIG"] = tui_file
        command = [agent_path]

    if prompt is not None:
        command.extend(["run", prompt])
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
    """Build and run OpenCode's launch specification."""
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
