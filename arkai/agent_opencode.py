"""OpenCode-specific paths, configuration, and launch command construction."""

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

from arkai import cloud_auth, config, utils
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
    cloud_services: list[str] | None = None,
    cloud_api_key: str | None = None,
) -> AgentLaunchSpec:
    """Create OpenCode configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _get_agent_model_id, _get_model_name

    config_dir = _get_config_dir()
    cloud_services = list(dict.fromkeys(service.lower() for service in cloud_services or []))
    is_cloud = bool(cloud_services)
    config_data: dict = {"$schema": "https://opencode.ai/config.json"}
    if not is_cloud:
        inference_port = config.get_config_value(cfg, "inference.port", 8081)
        inference_backend = config.get_config_value(cfg, "inference.backend", "llama-cpp")
        model_id = _get_agent_model_id(cfg)
        model_name = _get_model_name(cfg)
        config_data.update(
            {
                "provider": {
                    "local-llm": {
                        "name": f"Local LLM ({inference_backend})",
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": f"http://127.0.0.1:{inference_port}/v1"},
                        "models": {model_id: {"name": model_name}},
                    }
                },
                "model": f"local-llm/{model_id}",
            }
        )
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
    sandbox_environment = dict(cli_environment or {})
    if is_cloud and cloud_api_key:
        environment["OPENAI_API_KEY"] = cloud_api_key
        sandbox_environment["OPENAI_API_KEY"] = cloud_api_key
    if use_sandbox:
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(config_dir),
                wtmcp_port,
                sandbox_profile,
                cli_volumes,
                sandbox_environment,
                tty=prompt is None,
                allow_network_listeners=True,
                allow_network=False,
                allow_inference=not is_cloud,
                extra_hosts=(
                    ["*.openai.com:443", "api.openai.com:443"]
                    if is_cloud and platform.system() == "Linux"
                    else None
                ),
            )
        except Exception:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
            raise
        env_args = ["env", f"OPENCODE_CONFIG={config_file}"]
        # OpenCode's project discovery walks above the mounted workspace. The
        # sandbox intentionally grants only metadata traversal there, which
        # Bun's discovery path cannot use reliably. Arkai also supplies the
        # authoritative per-run config, so do not load repository-controlled
        # config/plugins from outside the mounted project.
        env_args.append("OPENCODE_DISABLE_PROJECT_CONFIG=true")
        if tui_file is not None:
            env_args.append(f"OPENCODE_TUI_CONFIG={tui_file}")
        command = sandbox_prefix + env_args + [agent_path]
    else:
        environment["OPENCODE_CONFIG"] = str(config_file)
        if tui_file is not None:
            environment["OPENCODE_TUI_CONFIG"] = tui_file
        command = [agent_path]

    if prompt is not None:
        command.extend(["run", "--standalone", prompt])
    else:
        command.append("--standalone")
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
    cloud_services: list[str] | None = None,
) -> str | None:
    """Build and run OpenCode's launch specification."""
    cloud_api_key = (
        cloud_auth.ensure_openai_api_key("opencode", _get_config_dir(), agent_path)
        if cloud_services
        else None
    )
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
            cloud_services,
            cloud_api_key,
        ),
        capture_stdout,
    )
