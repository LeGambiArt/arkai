"""Crush-specific paths, configuration, and launch command construction."""

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from arkai import cloud_auth, config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _get_config_dir() -> Path:
    """Return the temporary session directory used by Crush."""
    config_dir = Path(os.path.expanduser("~/.local/state/arkai/sessions"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_agent_path(cfg: dict) -> str:
    """Resolve the configured Crush executable."""
    return utils.resolve_binary(config.get_config_value(cfg, "agent.path") or "crush")


def _cloud_model_reference(provider: str, model: str) -> tuple[str, str]:
    """Split a Crush provider/model reference and validate its cloud provider."""
    model_provider, separator, model_id = model.partition("/")
    if not separator:
        model_id = model
    else:
        if model_provider != provider:
            raise RuntimeError(
                "Crush model provider "
                f"{model_provider!r} does not match selected cloud provider {provider!r}"
            )
        if not model_id:
            raise RuntimeError("Crush cloud model must include a model ID after the provider")
    if provider == "vertexai":
        model_id = _vertex_model_id(model_id)
    return provider, model_id


def _vertex_model_id(model_id: str) -> str:
    """Convert Anthropic API date suffixes to Vertex's `@YYYYMMDD` form."""
    if not model_id.startswith("claude-"):
        return model_id
    prefix, separator, date = model_id.rpartition("-")
    if separator and len(date) == 8 and date.isdigit():
        return f"{prefix}@{date}"
    return model_id


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
    native_session_id: str | None = None,
) -> AgentLaunchSpec:
    """Create Crush configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _cloud_provider, _get_agent_model_id

    cloud_services = list(dict.fromkeys(service.lower() for service in cloud_services or []))
    cloud_provider = _cloud_provider(cloud_services)
    is_cloud = cloud_provider is not None
    provider_id = "openai" if cloud_provider == "openai" else "vertexai"
    cloud_model_id = (
        _cloud_model_reference(provider_id, cloud_model)[1] if is_cloud and cloud_model else None
    )
    sessions_dir = _get_config_dir()
    if session_dir:
        crush_config_dir = Path(session_dir) / "crush"
        crush_config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        crush_config_dir.chmod(0o700)
    else:
        crush_config_dir = Path(tempfile.mkdtemp(prefix="crush-", dir=sessions_dir))
    config_file = crush_config_dir / "crush.json"

    if is_cloud:
        config_data: dict = {
            "$schema": "https://charm.land/crush.json",
            "providers": {},
        }
        provider_config = {
            "id": provider_id,
            "name": "OpenAI" if cloud_provider == "openai" else "Google Vertex AI",
            "type": "openai" if cloud_provider == "openai" else "google-vertex",
        }
        if cloud_provider == "openai":
            provider_config["api_key"] = "$OPENAI_API_KEY"
        else:
            provider_config["models"] = [
                {
                    "id": "claude-sonnet-4-5@20250929",
                    "name": "Claude Sonnet 4.5",
                    "context_window": 200000,
                    "default_max_tokens": 50000,
                    "can_reason": True,
                    "supports_attachments": True,
                },
                {
                    "id": "claude-haiku-4-5@20251001",
                    "name": "Claude Haiku 4.5",
                    "context_window": 200000,
                    "default_max_tokens": 64000,
                    "can_reason": True,
                    "supports_attachments": True,
                },
            ]
        config_data["providers"][provider_id] = provider_config

        if cloud_model_id:
            provider_models = config_data["providers"][provider_id].setdefault("models", [])
            if not any(model["id"] == cloud_model_id for model in provider_models):
                provider_models.append({"id": cloud_model_id, "name": cloud_model_id})
            config_data["models"] = {
                "large": {"model": cloud_model_id, "provider": provider_id},
                "small": {"model": cloud_model_id, "provider": provider_id},
            }
        elif cloud_provider == "vertex":
            config_data["models"] = {
                "large": {
                    "model": "claude-sonnet-4-5@20250929",
                    "provider": provider_id,
                },
                "small": {
                    "model": "claude-haiku-4-5@20251001",
                    "provider": provider_id,
                },
            }
    else:
        inference_port = config.get_config_value(cfg, "inference.port", 8081)
        model_name = _get_agent_model_id(cfg)
        config_data = {
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
        if not session_dir:
            crush_config_dir.rmdir()
        raise

    environment = os.environ.copy()
    sandbox_environment = dict(cli_environment or {})
    sandbox_volumes = list(cli_volumes or [])
    if cloud_provider == "openai" and cloud_api_key:
        environment["OPENAI_API_KEY"] = cloud_api_key
        sandbox_environment["OPENAI_API_KEY"] = cloud_api_key
    if cloud_provider == "vertex":
        sandbox_volumes, sandbox_environment = cloud_auth.crush_vertex_sandbox_settings(
            sandbox_volumes, sandbox_environment
        )
        environment.update(sandbox_environment)

    if use_sandbox:
        try:
            extra_hosts = (
                ["*.openai.com:443", "api.openai.com:443"]
                if cloud_provider == "openai" and platform.system() == "Linux"
                else [
                    "*-aiplatform.googleapis.com:443",
                    "aiplatform.googleapis.com:443",
                    "oauth2.googleapis.com:443",
                ]
                if cloud_provider == "vertex" and platform.system() == "Linux"
                else None
            )
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(crush_config_dir),
                wtmcp_port,
                sandbox_profile,
                sandbox_volumes,
                sandbox_environment,
                tty=prompt is None,
                allow_network=False,
                allow_inference=not is_cloud,
                extra_hosts=extra_hosts,
            )
        except Exception:
            config_file.unlink(missing_ok=True)
            if not session_dir:
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

    if resume_session:
        command.extend(["--session", native_session_id]) if native_session_id else command.append(
            "--continue"
        )

    if prompt is not None:
        command.extend(["run", "--quiet", prompt])
    cleanup_paths = [config_file]
    if not session_dir:
        cleanup_paths.append(crush_config_dir)
    return AgentLaunchSpec(command, environment, cleanup_paths, working_directory=workdir)


def _native_session_ids(agent_path: str, session_dir: str, workdir: str | None) -> list[str]:
    """Read session IDs from the isolated Crush data directory."""
    crush_dir = Path(session_dir) / "crush"
    environment = os.environ.copy()
    environment["CRUSH_GLOBAL_CONFIG"] = str(crush_dir)
    environment["CRUSH_GLOBAL_DATA"] = str(crush_dir)
    try:
        result = subprocess.run(
            [agent_path, "session", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
            cwd=workdir,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    try:
        sessions = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(sessions, list):
        return []
    return [
        session_id
        for entry in sessions
        if isinstance(entry, dict)
        and isinstance((session_id := entry.get("uuid") or entry.get("id")), str)
        and session_id
    ]


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
) -> str | None:
    """Build and run Crush's launch specification."""
    cloud_api_key = (
        cloud_auth.ensure_openai_api_key("crush", _get_config_dir(), agent_path)
        if cloud_services and any(service.lower() == "openai" for service in cloud_services)
        else None
    )
    before_ids = (
        set(_native_session_ids(agent_path, session_dir, workdir))
        if session_dir and not resume_session
        else set()
    )
    result = run_launch_spec(
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
            native_session_id,
        ),
        capture_stdout,
    )
    if session_dir and session_record is not None and not resume_session:
        new_ids = set(_native_session_ids(agent_path, session_dir, workdir)) - before_ids
        if len(new_ids) == 1:
            session_record["native_session_id"] = new_ids.pop()
    return result
