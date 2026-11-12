"""OpenCode-specific paths, configuration, and launch command construction."""

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

from arkai import cloud_auth, config, utils
from arkai import cloud_provider as cloud_config_utils
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
    cloud_model: str | None = None,
    session_dir: str | None = None,
    resume_session: bool = False,
    cloud_config: dict | None = None,
    cloud_api_environment: dict[str, str] | None = None,
) -> AgentLaunchSpec:
    """Create OpenCode configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _get_agent_model_id, _get_model_name

    config_dir = _get_config_dir()
    session_data_home = Path(session_dir) if session_dir else None
    if session_data_home is not None:
        session_data_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        session_data_home.chmod(0o700)
    cloud_services = list(dict.fromkeys(service.lower() for service in cloud_services or []))
    is_cloud = bool(cloud_services)
    config_data: dict = {"$schema": "https://opencode.ai/config.json"}
    from arkai.agent import _cloud_provider

    cloud_provider = _cloud_provider(cloud_services)
    custom_provider_config = None
    custom_api_keys: dict[str, str] = {}
    custom_hosts: list[str] = []
    if cloud_provider == "custom":
        custom_provider_config = cloud_config or cloud_config_utils.custom_cloud_config_from_arkai(
            cfg
        )
        assert custom_provider_config is not None
        custom_api_keys = (
            dict(cloud_api_environment)
            if cloud_api_environment is not None
            else cloud_config_utils.custom_cloud_environment(custom_provider_config)
        )
        selected_model_reference = cloud_config_utils.select_custom_cloud_model(
            custom_provider_config, cloud_model
        )
        selected_provider, selected_model = cloud_config_utils.custom_cloud_model_provider(
            custom_provider_config, selected_model_reference
        )
        opencode_providers = {}
        for provider_id, provider in custom_provider_config["providers"].items():
            model_config = {}
            model_filter = (
                selected_model if provider_id == selected_provider["provider_id"] else None
            )
            for model in cloud_config_utils.custom_cloud_models(provider, model_filter):
                model_entry = {"name": model.get("name", model["id"])}
                limit = {"context": model["context_window"]}
                if "max_output_tokens" in model:
                    limit["output"] = model["max_output_tokens"]
                model_entry["limit"] = limit
                model_config[model["id"]] = model_entry

            native_protocols = {
                "openai": ("openai-completions", "openai-responses"),
                "anthropic": ("anthropic-messages",),
            }
            if provider["protocol"] in native_protocols.get(provider_id, ()):
                opencode_provider = {
                    "settings": {"baseURL": provider["base_url"]},
                }
                if provider.get("name") != "openai":
                    opencode_provider["name"] = provider["name"]
                if model_config:
                    opencode_provider["models"] = model_config
            else:
                opencode_provider = {
                    "name": provider["name"],
                    "package": (
                        "@opencode/ai/providers/openai-compatible"
                        if provider["protocol"] == "openai-completions"
                        else "@opencode/ai/providers/openai-compatible-responses"
                        if provider["protocol"] == "openai-responses"
                        else "@opencode/ai/providers/anthropic"
                    ),
                    "settings": {"baseURL": provider["base_url"]},
                    "models": model_config,
                }
            if provider.get("api_key_environment_name"):
                environment_name = provider["api_key_environment_name"]
                opencode_provider["env"] = [environment_name]
                api_key = cloud_config_utils.custom_cloud_resolved_api_key(
                    provider, custom_api_keys
                )
                if api_key:
                    custom_api_keys[environment_name] = api_key
            opencode_providers[provider_id] = opencode_provider
            if provider["host"] not in custom_hosts:
                custom_hosts.append(provider["host"])

        config_data["providers"] = opencode_providers
        config_data["model"] = selected_model_reference
    elif is_cloud and cloud_model is not None:
        config_data["model"] = cloud_model
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
    if cloud_provider == "custom":
        for native_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if native_key not in custom_api_keys:
                environment.pop(native_key, None)
                sandbox_environment.pop(native_key, None)
        for environment_name, api_key in custom_api_keys.items():
            environment[environment_name] = api_key
            sandbox_environment[environment_name] = api_key
    sandbox_volumes = list(cli_volumes or [])
    if cloud_provider == "vertex":
        sandbox_volumes, sandbox_environment = cloud_auth.vertex_sandbox_settings(
            sandbox_volumes, sandbox_environment
        )
    if use_sandbox:
        try:
            if session_data_home is not None:
                sandbox_volumes.append(str(session_data_home))
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(config_dir),
                wtmcp_port,
                sandbox_profile,
                sandbox_volumes,
                sandbox_environment,
                tty=prompt is None,
                allow_network_listeners=True,
                allow_network=False,
                allow_inference=not is_cloud,
                extra_hosts=(
                    custom_hosts
                    if cloud_provider == "custom" and platform.system() == "Linux"
                    else ["*.openai.com:443", "api.openai.com:443"]
                    if cloud_provider == "openai" and platform.system() == "Linux"
                    else [
                        "*-aiplatform.googleapis.com:443",
                        "aiplatform.googleapis.com:443",
                        "oauth2.googleapis.com:443",
                    ]
                    if cloud_provider == "vertex" and platform.system() == "Linux"
                    else None
                ),
            )
        except Exception:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
            raise
        env_args = ["env", f"OPENCODE_CONFIG={config_file}"]
        if session_data_home is not None:
            env_args.append(f"XDG_DATA_HOME={session_data_home}")
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
        if session_data_home is not None:
            environment["XDG_DATA_HOME"] = str(session_data_home)
        if tui_file is not None:
            environment["OPENCODE_TUI_CONFIG"] = tui_file
        command = [agent_path]

    if prompt is not None:
        command.extend(["run", "--standalone", prompt])
    else:
        if resume_session:
            command.append("--continue")
        command.append("--standalone")
    return AgentLaunchSpec(command, environment, cleanup_paths, working_directory=workdir)


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
    cloud_model: str | None = None,
    session_dir: str | None = None,
    resume_session: bool = False,
    native_session_id: str | None = None,
    session_record: dict | None = None,
    cloud_config: dict | None = None,
    cloud_api_environment: dict[str, str] | None = None,
) -> str | None:
    """Build and run OpenCode's launch specification."""
    cloud_api_key = (
        cloud_auth.ensure_openai_api_key("opencode", _get_config_dir(), agent_path)
        if cloud_services and any(service.lower() == "openai" for service in cloud_services)
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
            cloud_model,
            session_dir,
            resume_session,
            cloud_config,
            cloud_api_environment,
        ),
        capture_stdout,
    )
