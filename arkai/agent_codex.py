"""Codex CLI profile creation and launch command construction."""

import json
import os
import uuid
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _get_codex_home() -> Path:
    """Resolve Codex's config/auth directory without changing the user's setting."""
    configured_home = os.environ.get("CODEX_HOME")
    return Path(configured_home or "~/.codex").expanduser().resolve()


def _create_profile(cfg: dict, wtmcp_port: int | None, use_sandbox: bool) -> tuple[Path, str]:
    """Write a unique temporary profile that selects Arkai's local inference provider."""
    from arkai.agent import _get_agent_model_id

    codex_home = _get_codex_home()
    codex_home.mkdir(parents=True, exist_ok=True)
    profile_name = f"arkai-{uuid.uuid4().hex}"
    profile_path = codex_home / f"{profile_name}.config.toml"
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_id = _get_agent_model_id(cfg)

    profile_lines = [
        f"model = {json.dumps(model_id)}",
        'model_provider = "arkai"',
        # Codex adds web_search by default, which local inference servers do not expose.
        'web_search = "disabled"',
    ]
    if use_sandbox:
        # Arapuca is the outer sandbox; Codex's nested sandbox is not portable in that context.
        profile_lines.append('sandbox_mode = "danger-full-access"')
    profile_lines.extend(
        [
            "",
            "[model_providers.arkai]",
            'name = "Arkai local inference"',
            f'base_url = "http://127.0.0.1:{inference_port}/v1"',
            'wire_api = "responses"',
            "requires_openai_auth = false",
        ]
    )
    profile = "\n".join(profile_lines)
    if wtmcp_port is not None:
        profile += "\n\n[mcp_servers.wtmcp]\n"
        profile += f'url = "http://127.0.0.1:{wtmcp_port}/mcp"\n'

    try:
        file_descriptor = os.open(profile_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as profile_file:
            profile_file.write(profile + "\n")
    except Exception:
        profile_path.unlink(missing_ok=True)
        raise

    return profile_path, profile_name


def get_agent_path(cfg: dict) -> str:
    """Resolve the configured Codex CLI executable."""
    return utils.resolve_binary(config.get_config_value(cfg, "agent.path") or "codex")


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
    """Create a temporary Codex profile and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd

    profile_path, profile_name = _create_profile(cfg, wtmcp_port, use_sandbox)
    codex_home = profile_path.parent
    command = [agent_path, "--no-daemon", "--profile", profile_name]
    if prompt is not None:
        command.extend(["exec", prompt])

    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home)
    if use_sandbox:
        sandbox_environment = dict(cli_environment or {})
        sandbox_environment["CODEX_HOME"] = str(codex_home)
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(codex_home),
                wtmcp_port,
                sandbox_profile,
                cli_volumes,
                sandbox_environment,
                tty=prompt is None,
            )
        except Exception:
            profile_path.unlink(missing_ok=True)
            raise
        command = sandbox_prefix + command

    return AgentLaunchSpec(command, environment, [profile_path])


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
    """Build and run Codex's launch specification."""
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
