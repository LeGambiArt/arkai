"""Codex CLI profile creation and launch command construction."""

import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

from arkai import config, utils
from arkai.agent import AgentLaunchSpec, run_launch_spec


def _get_codex_home(session_dir: str | None = None) -> Path:
    """Resolve Codex's config/auth directory without changing the user's setting."""
    if session_dir:
        return Path(session_dir) / "codex-home"
    configured_home = os.environ.get("CODEX_HOME")
    return Path(configured_home or "~/.codex").expanduser().resolve()


def _prepare_codex_home(session_dir: str | None) -> tuple[Path, list[Path]]:
    """Create an isolated Codex session home while reusing user config and auth."""
    codex_home = _get_codex_home(session_dir)
    codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    codex_home.chmod(0o700)
    shared_paths: list[Path] = []
    if session_dir:
        global_home = _get_codex_home()
        for filename in ("auth.json", "config.toml"):
            global_path = global_home / filename
            if not global_path.is_file():
                continue
            source = global_path.resolve()
            link = codex_home / filename
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source)
            shared_paths.append(source)
    return codex_home, shared_paths


def _is_git_repository(path: str | None) -> bool:
    """Check whether a Codex working directory is inside a Git worktree."""
    workdir = Path(path or os.getcwd()).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _create_profile(
    cfg: dict,
    wtmcp_port: int | None,
    use_sandbox: bool,
    cloud_provider: str | None,
    cloud_model: str | None,
    codex_home: Path | None = None,
    profile_name: str | None = None,
) -> tuple[Path, str]:
    """Write a unique temporary profile for Arkai local inference or OpenAI cloud."""
    from arkai.agent import _get_agent_model_id

    codex_home = codex_home or _get_codex_home()
    codex_home.mkdir(parents=True, exist_ok=True)
    profile_name = profile_name or f"arkai-{uuid.uuid4().hex}"
    profile_path = codex_home / f"{profile_name}.config.toml"
    profile_lines: list[str] = []
    if cloud_provider == "openai":
        profile_lines.append('model_provider = "openai"')
        profile_lines.append('openai_base_url = "https://api.openai.com/v1"')
        if cloud_model:
            profile_lines.insert(0, f"model = {json.dumps(cloud_model)}")
    else:
        inference_port = config.get_config_value(cfg, "inference.port", 8081)
        model_id = _get_agent_model_id(cfg)
        profile_lines.extend(
            [
                f"model = {json.dumps(model_id)}",
                'model_provider = "arkai"',
                # Local inference servers do not expose Codex's hosted web-search tool.
                'web_search = "disabled"',
            ]
        )
    if use_sandbox:
        # Arapuca is the outer sandbox; Codex's nested sandbox is not portable in that context.
        profile_lines.append('sandbox_mode = "danger-full-access"')
    if cloud_provider is None:
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
    cloud_services: list[str] | None = None,
    cloud_model: str | None = None,
    session_dir: str | None = None,
    resume_session: bool = False,
    session_id: str | None = None,
) -> AgentLaunchSpec:
    """Create a temporary Codex profile and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd, _cloud_provider

    cloud_provider = _cloud_provider(cloud_services)
    if cloud_provider == "vertex":
        raise RuntimeError(
            "Codex does not support Vertex: its custom providers require the Responses API, "
            "which Vertex's OpenAI-compatible endpoint does not currently provide"
        )
    codex_home, shared_paths = _prepare_codex_home(session_dir)
    profile_name = f"arkai-{session_id.replace('-', '')}" if session_id else None
    profile_path, profile_name = _create_profile(
        cfg, wtmcp_port, use_sandbox, cloud_provider, cloud_model, codex_home, profile_name
    )
    command = [agent_path, "--no-daemon", "--profile", profile_name]
    if resume_session:
        command.extend(["resume", "--last"])
    if prompt is not None:
        command.append("exec")
        if not _is_git_repository(workdir):
            command.append("--skip-git-repo-check")
        command.append(prompt)

    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home)
    api_key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if cloud_provider == "openai" and api_key:
        environment["CODEX_API_KEY"] = api_key
    if use_sandbox:
        sandbox_environment = dict(cli_environment or {})
        sandbox_environment["CODEX_HOME"] = str(codex_home)
        sandbox_volumes = list(cli_volumes or [])
        mounted_paths = {volume.rsplit(":", 1)[0] for volume in sandbox_volumes}
        for shared_path in shared_paths:
            if str(shared_path) not in mounted_paths:
                sandbox_volumes.append(str(shared_path))
                mounted_paths.add(str(shared_path))
        for variable in ("CODEX_CA_CERTIFICATE", "SSL_CERT_FILE"):
            certificate = sandbox_environment.get(variable) or os.environ.get(variable)
            if not certificate:
                continue
            certificate_path = Path(certificate).expanduser().resolve()
            sandbox_environment[variable] = str(certificate_path)
            if certificate_path.is_file() and str(certificate_path) not in mounted_paths:
                sandbox_volumes.append(f"{certificate_path}:ro")
                mounted_paths.add(str(certificate_path))
        if cloud_provider == "openai":
            sandbox_api_key = (
                sandbox_environment.get("CODEX_API_KEY")
                or sandbox_environment.get("OPENAI_API_KEY")
                or os.environ.get("CODEX_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if sandbox_api_key:
                sandbox_environment["CODEX_API_KEY"] = sandbox_api_key
        extra_hosts = (
            ["*.openai.com:443", "api.openai.com:443"]
            if cloud_provider == "openai" and platform.system() == "Linux"
            else None
        )
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(codex_home),
                wtmcp_port,
                sandbox_profile,
                sandbox_volumes,
                sandbox_environment,
                tty=prompt is None,
                allow_inference=cloud_provider is None,
                extra_hosts=extra_hosts,
                allow_keychain=(platform.system() == "Darwin" and cloud_provider == "openai"),
            )
        except Exception:
            profile_path.unlink(missing_ok=True)
            raise
        command = sandbox_prefix + command

    return AgentLaunchSpec(command, environment, [profile_path], working_directory=workdir)


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
            cloud_services,
            cloud_model,
            session_dir,
            resume_session,
            session_record.get("id") if session_record else None,
        ),
        capture_stdout,
    )
