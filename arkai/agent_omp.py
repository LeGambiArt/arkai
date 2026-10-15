"""oh-my-pi-specific paths, installation, configuration, and launch construction."""

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

from arkai import cloud_auth, config, utils
from arkai.agent import AgentLaunchSpec, run_install_command, run_launch_spec

OMP_INSTALLER_URL = "https://omp.sh/install"
OMP_CONFIG_DIR = ".local/state/arkai/omp"


def get_install_dir() -> Path:
    """Return the root directory for Arkai-managed OMP files."""
    return Path(os.path.expanduser("~/.local/state/arkai/omp"))


def get_core_dir() -> Path:
    """Return the directory containing Arkai's OMP executable."""
    return get_install_dir() / "bin"


def get_agent_dir() -> Path:
    """Return the directory containing OMP's Arkai-managed configuration and data."""
    return get_install_dir() / "agent"


def get_global_agent_dir() -> Path:
    """Return OMP's user-level agent directory, including its stored credentials."""
    configured_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()
    config_root = os.environ.get("PI_CONFIG_DIR") or ".omp"
    config_root_path = Path(config_root).expanduser()
    if not config_root_path.is_absolute():
        config_root_path = Path.home() / config_root_path
    return (config_root_path / "agent").resolve()


def _write_project_mcp_config(workdir: str | None, port: int) -> Path | None:
    """Add a temporary project-level WTMCP entry without replacing user MCP config."""
    project_dir = Path(workdir or os.getcwd())
    candidates = [
        project_dir / ".omp" / "mcp.json",
        project_dir / ".omp" / ".mcp.json",
        project_dir / "mcp.json",
        project_dir / ".mcp.json",
    ]
    mcp_config = next((candidate for candidate in candidates if not candidate.exists()), None)
    if mcp_config is None:
        utils.warn("Skipping Arkai's WTMCP config; all OMP project MCP config paths already exist")
        return None
    mcp_config.parent.mkdir(parents=True, exist_ok=True)
    mcp_config.write_text(
        json.dumps(
            {"mcpServers": {"wtmcp": {"type": "http", "url": f"http://127.0.0.1:{port}/mcp"}}},
            indent=2,
        )
    )
    return mcp_config


def get_agent_path(cfg: dict) -> str:
    """Resolve OMP's configured executable or Arkai-managed default."""
    configured_path = config.get_config_value(cfg, "agent.path")
    agent_path = configured_path or str(get_core_dir() / "omp")
    return utils.resolve_binary(agent_path)


def _prepare_agent_dir(cfg: dict, wtmcp_port: int | None) -> Path:
    """Write OMP's Arkai-local model and optional MCP configuration."""
    from arkai.agent import _get_agent_model_id, _get_model_name

    agent_dir = get_agent_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)

    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_id = _get_agent_model_id(cfg)
    model_name = _get_model_name(cfg)
    context_size = config.get_config_value(cfg, "inference.context_size", 65536)

    models = {
        "providers": {
            "local-llm": {
                "baseUrl": f"http://127.0.0.1:{inference_port}/v1",
                "api": "openai-completions",
                "apiKey": "local",
                "models": [
                    {
                        "id": model_id,
                        "name": model_name,
                        "reasoning": False,
                        "contextWindow": context_size,
                        "maxTokens": min(8192, context_size),
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                    }
                ],
            }
        }
    }
    settings = {
        "modelRoles": {"default": f"local-llm/{model_id}"},
        "startup": {"quiet": True, "setupWizard": False},
    }

    utils.save_yaml(str(agent_dir / "models.yml"), models)
    utils.save_yaml(str(agent_dir / "config.yml"), settings)

    mcp_config = agent_dir / "mcp.json"
    if wtmcp_port is not None:
        mcp_config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "wtmcp": {
                            "type": "http",
                            "url": f"http://127.0.0.1:{wtmcp_port}/mcp",
                        }
                    }
                },
                indent=2,
            )
        )
    else:
        mcp_config.unlink(missing_ok=True)

    return agent_dir


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
) -> AgentLaunchSpec:
    """Create OMP configuration and return its process launch specification."""
    from arkai.agent import _build_sandbox_cmd

    cloud_services = list(dict.fromkeys(service.lower() for service in cloud_services or []))
    is_cloud = bool(cloud_services)
    from arkai.agent import _cloud_provider

    cloud_provider = _cloud_provider(cloud_services)
    if is_cloud:
        agent_dir = get_global_agent_dir()
        agent_dir.mkdir(parents=True, exist_ok=True)
        mcp_config = _write_project_mcp_config(workdir, wtmcp_port) if wtmcp_port else None
    else:
        agent_dir = _prepare_agent_dir(cfg, wtmcp_port)
        mcp_config = None

    environment = os.environ.copy()
    if not is_cloud:
        environment["PI_CONFIG_DIR"] = OMP_CONFIG_DIR
        environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    else:
        environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    environment["OMP_SKIP_SETUP"] = "1"

    command: list[str]
    if use_sandbox:
        resolved_agent_path = str(Path(agent_path).resolve())
        binary_dir = str(Path(resolved_agent_path).parent)
        volumes = list(cli_volumes or [])
        binary_volume = f"{binary_dir}:ro"
        if binary_volume not in volumes:
            volumes.append(binary_volume)
        sandbox_environment = dict(cli_environment or {})
        sandbox_environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
        sandbox_environment["OMP_SKIP_SETUP"] = "1"
        if cloud_provider == "vertex":
            volumes, sandbox_environment = cloud_auth.vertex_sandbox_settings(
                volumes, sandbox_environment
            )
        if is_cloud and "PI_CONFIG_DIR" in os.environ:
            sandbox_environment["PI_CONFIG_DIR"] = os.environ["PI_CONFIG_DIR"]
        try:
            sandbox_prefix = _build_sandbox_cmd(
                cfg,
                workdir,
                str(agent_dir),
                wtmcp_port,
                sandbox_profile,
                volumes,
                sandbox_environment,
                tty=prompt is None,
                allow_network=False,
                allow_inference=not is_cloud,
                extra_hosts=(
                    ["*.openai.com:443", "chatgpt.com:443", "*.chatgpt.com:443"]
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
            if is_cloud and mcp_config is not None:
                mcp_config.unlink(missing_ok=True)
            raise
        env_args = ["env"]
        if not is_cloud or "PI_CONFIG_DIR" in os.environ:
            config_root = OMP_CONFIG_DIR if not is_cloud else os.environ["PI_CONFIG_DIR"]
            env_args.append(f"PI_CONFIG_DIR={config_root}")
        env_args.extend(
            [
                f"PI_CODING_AGENT_DIR={agent_dir}",
                "OMP_SKIP_SETUP=1",
                resolved_agent_path,
            ]
        )
        command = sandbox_prefix + env_args
    else:
        command = [str(Path(agent_path).resolve())]

    command.append("--profile")
    command.append("default")
    if is_cloud:
        provider_name = "google-vertex" if cloud_provider == "vertex" else "openai"
        command.extend(["--provider", provider_name])
        if cloud_model is not None:
            command.extend(["--model", cloud_model])
    else:
        from arkai.agent import _get_agent_model_id

        command.extend(["--model", f"local-llm/{_get_agent_model_id(cfg)}"])
    if prompt is not None:
        command.extend(["-p", "--no-session", "--mode", "text", prompt])

    cleanup_paths = [mcp_config] if mcp_config is not None else []
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
    cloud_model: str | None = None,
) -> str | None:
    """Build and run OMP's launch specification."""
    if cloud_services and any(service.lower() == "openai" for service in cloud_services):
        cloud_auth.ensure_openai_api_key("omp", get_global_agent_dir(), agent_path)
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
        ),
        capture_stdout,
    )


def install() -> None:
    """Install OMP's verified prebuilt binary into Arkai's private directory."""
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError(
            f"Arkai's automatic OMP installer supports macOS and Linux, not {system or 'this OS'}. "
            "Install OMP using its official instructions, then set agent.path in Arkai config."
        )

    curl_path = shutil.which("curl")
    sh_path = shutil.which("sh")
    if curl_path is None or sh_path is None:
        missing = " and ".join(
            name for name, path in (("curl", curl_path), ("sh", sh_path)) if path is None
        )
        raise RuntimeError(f"{missing} was not found in PATH; install it and retry.")

    install_dir = get_core_dir()
    omp_path = install_dir / "omp"
    utils.warn(
        "This will download the official oh-my-pi installer and install its verified binary "
        f"under {install_dir}."
    )
    try:
        answer = input("Proceed with the installation? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in {"y", "yes"}:
        utils.info("OMP installation cancelled.")
        return

    with tempfile.TemporaryDirectory(prefix="arkai-omp-install-") as temp_dir:
        installer_path = Path(temp_dir) / "install.sh"
        utils.info("Downloading the official OMP installer...")
        run_install_command([curl_path, "-fsSL", OMP_INSTALLER_URL, "-o", str(installer_path)])
        install_env = os.environ.copy()
        install_env["PI_INSTALL_DIR"] = str(install_dir)
        utils.info("Installing the OMP binary...")
        run_install_command([sh_path, str(installer_path), "--binary"], env=install_env)

    if not omp_path.is_file() or not os.access(omp_path, os.X_OK):
        raise RuntimeError(f"OMP installer completed without creating an executable at {omp_path}")

    code, stdout, stderr = utils.run_command([str(omp_path), "--version"], timeout=30)
    if code != 0:
        detail = stderr.strip() or stdout.strip() or f"exit code {code}"
        raise RuntimeError(f"Installed OMP binary failed its version check: {detail}")
    version = stdout.strip() or stderr.strip() or "version unavailable"
    utils.info(f"OMP installation complete ({version}): {omp_path}")
