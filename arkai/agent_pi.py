"""Pi-specific paths, installation, configuration, and launch construction."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

from arkai import cloud_auth, config, utils
from arkai.agent import AgentLaunchSpec, run_install_command, run_launch_spec

PI_CORE_PACKAGE = "@earendil-works/pi-coding-agent"
PI_PACKAGES = ("pi-mcp-adapter", "pi-web-access", "pi-subagents")
VERTEX_ANTHROPIC_PROVIDER = "google-vertex-anthropic"
MODELS_DEV_API_URL = "https://models.dev/api.json"
VERTEX_MODEL_CACHE_SECONDS = 24 * 60 * 60
_PROMPT_SYSTEM_INSTRUCTION = (
    "You are an expert coding assistant. Use available tools only when needed. "
    "Answer factual questions directly and provide only the final answer."
)


def get_install_dir() -> Path:
    """Return the root directory for Arkai-managed Pi files."""
    return Path(os.path.expanduser("~/.local/state/arkai/pi"))


def get_core_dir() -> Path:
    """Return the directory containing Pi's locally installed executable."""
    return get_install_dir() / "core"


def get_agent_path(cfg: dict) -> str:
    """Resolve Pi's configured executable or Arkai-managed default."""
    configured_path = config.get_config_value(cfg, "agent.path")
    agent_path = configured_path or str(get_core_dir() / "bin" / "pi")
    return utils.resolve_binary(agent_path)


def get_agent_dir() -> Path:
    """Return the directory containing Pi's Arkai-managed runtime data."""
    return get_install_dir() / "agent"


def get_global_agent_dir() -> Path:
    """Return the user's Pi agent directory, including global auth and settings."""
    configured_dir = os.environ.get("PI_CODING_AGENT_DIR") or "~/.pi/agent"
    return Path(configured_dir).expanduser().resolve()


def _write_run_mcp_config(agent_dir: str, port: int) -> Path:
    """Write a temporary WTMCP config without replacing the user's global config."""
    descriptor, filename = tempfile.mkstemp(prefix="arkai-mcp-", suffix=".json", dir=agent_dir)
    path = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
            json.dump(
                {
                    "settings": {"projectServers": "allow"},
                    "mcpServers": {"wtmcp": {"url": f"http://127.0.0.1:{port}/mcp"}},
                },
                config_file,
                indent=2,
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _number(value: object, default: float = 0) -> float:
    """Return a finite model metadata number or its fallback."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value if value >= 0 else default


def _vertex_anthropic_models(provider_data: object) -> list[dict]:
    """Convert the Models.dev Vertex Anthropic catalog into Pi model entries."""
    provider_models = provider_data.get("models") if isinstance(provider_data, dict) else None
    if not isinstance(provider_models, dict):
        return []

    models = []
    for model in provider_models.values():
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        name = model.get("name")
        limit = model.get("limit")
        cost = model.get("cost")
        modalities = model.get("modalities")
        model_input = modalities.get("input") if isinstance(modalities, dict) else []
        if not isinstance(model_input, list):
            model_input = []
        if not isinstance(model_id, str) or not model_id or not isinstance(name, str) or not name:
            continue
        limit = limit if isinstance(limit, dict) else {}
        cost = cost if isinstance(cost, dict) else {}
        input_types = [value for value in model_input if value in {"text", "image"}]
        if "text" not in input_types:
            input_types.insert(0, "text")
        models.append(
            {
                "id": model_id,
                "name": name,
                "api": "anthropic-messages",
                "reasoning": bool(model.get("reasoning")),
                "input": input_types,
                "cost": {
                    "input": _number(cost.get("input")),
                    "output": _number(cost.get("output")),
                    "cacheRead": _number(cost.get("cache_read")),
                    "cacheWrite": _number(cost.get("cache_write")),
                },
                "contextWindow": max(1, int(_number(limit.get("context"), 200000))),
                "maxTokens": max(1, int(_number(limit.get("output"), 8192))),
            }
        )
    return sorted(models, key=lambda model: (model["name"].casefold(), model["id"]))


def _load_vertex_anthropic_models(agent_dir: Path) -> list[dict]:
    """Load the Vertex Anthropic catalog, refreshing its local cache once per day."""
    cache_dir = agent_dir / ".arkai"
    cache_path = cache_dir / "vertex-anthropic-models.json"
    cached_models: list[dict] = []
    try:
        cached_models = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cached_models, list):
            cached_models = []
    except (OSError, json.JSONDecodeError):
        pass

    try:
        cache_is_fresh = cache_path.is_file() and (
            time.time() - cache_path.stat().st_mtime < VERTEX_MODEL_CACHE_SECONDS
        )
    except OSError:
        cache_is_fresh = False
    if cache_is_fresh and cached_models:
        return cached_models

    try:
        response = requests.get(MODELS_DEV_API_URL, timeout=(5, 20))
        response.raise_for_status()
        catalog = response.json()
        models = _vertex_anthropic_models(
            catalog.get(VERTEX_ANTHROPIC_PROVIDER) if isinstance(catalog, dict) else None
        )
        if not models:
            raise RuntimeError("Models.dev returned no Vertex Anthropic chat models")
    except (requests.RequestException, ValueError, RuntimeError) as error:
        if cached_models:
            utils.warn(f"Could not refresh the Vertex Anthropic model catalog: {error}")
            return cached_models
        utils.warn(
            "Could not load Vertex Anthropic models from models.dev; "
            "Vertex Gemini models remain available"
        )
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix="vertex-models-", suffix=".json", dir=cache_dir)
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as cache_file:
            json.dump(models, cache_file, indent=2)
            cache_file.write("\n")
        os.replace(temporary_path, cache_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return models


def _merge_vertex_anthropic_models(agent_dir: Path, catalog_models: list[dict]) -> None:
    """Add catalog models to Pi's provider config without replacing user settings."""
    models_path = agent_dir / "models.json"
    try:
        models = json.loads(models_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        models = {"$schema": "https://pi.dev/schemas/models.schema.json", "providers": {}}
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not add Vertex Anthropic models: {models_path} is invalid JSON"
        ) from error
    if not isinstance(models, dict):
        raise RuntimeError(
            f"Could not add Vertex Anthropic models: {models_path} must contain an object"
        )

    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError(
            f"Could not add Vertex Anthropic models: invalid providers in {models_path}"
        )
    provider = providers.setdefault(VERTEX_ANTHROPIC_PROVIDER, {})
    if not isinstance(provider, dict):
        raise RuntimeError(
            f"Could not add Vertex Anthropic models: invalid provider config in {models_path}"
        )
    provider.setdefault("api", "anthropic-messages")
    provider.setdefault("baseUrl", "https://api.anthropic.com")
    provider_models = provider.setdefault("models", [])
    if not isinstance(provider_models, list):
        raise RuntimeError(
            f"Could not add Vertex Anthropic models: invalid model list in {models_path}"
        )

    configured_ids = {model.get("id") for model in provider_models if isinstance(model, dict)}
    additions = [model for model in catalog_models if model["id"] not in configured_ids]
    if not additions:
        return
    provider_models.extend(additions)
    write_path = models_path.resolve() if models_path.is_symlink() else models_path
    descriptor, filename = tempfile.mkstemp(
        prefix="arkai-models-", suffix=".json", dir=write_path.parent
    )
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as models_file:
            json.dump(models, models_file, indent=2)
            models_file.write("\n")
        os.replace(temporary_path, write_path)
        write_path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _ensure_vertex_anthropic_extension(agent_dir: Path, package_json: str) -> Path:
    """Install Arkai's Pi adapter for Anthropic models served by Vertex AI."""
    extension_source = Path(__file__).with_name("pi_vertex_anthropic.js")
    if not extension_source.is_file():
        raise RuntimeError(f"Pi Vertex Anthropic extension is missing: {extension_source}")
    extensions_dir = agent_dir / ".arkai"
    extensions_dir.mkdir(parents=True, exist_ok=True)
    extension_path = extensions_dir / "google-vertex-anthropic.js"
    source = extension_source.read_text(encoding="utf-8").replace(
        '"__ARKAI_PI_PACKAGE_JSON__"', json.dumps(package_json)
    )
    try:
        if extension_path.read_text(encoding="utf-8") == source:
            return extension_path
    except FileNotFoundError:
        pass

    descriptor, filename = tempfile.mkstemp(
        prefix="arkai-vertex-", suffix=".js", dir=extensions_dir
    )
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as extension_file:
            extension_file.write(source)
        os.replace(temporary_path, extension_path)
        extension_path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return extension_path


def _find_pi_package_json(agent_path: str) -> str:
    """Locate Pi's package manifest so the provider extension can load its dependencies."""
    resolved_path = Path(agent_path).resolve()
    start = resolved_path if resolved_path.is_dir() else resolved_path.parent
    for directory in (start, *start.parents):
        package_json = directory / "package.json"
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if package.get("name") == PI_CORE_PACKAGE:
            return str(package_json)
    raise RuntimeError(
        "Could not locate the Pi package to load the Vertex Anthropic provider extension"
    )


def ensure_openai_api_key(agent_path: str) -> None:
    """Use or securely save an OpenAI API key in Pi's global agent directory."""
    cloud_auth.ensure_openai_api_key("pi", get_global_agent_dir())


def _resolve_node_path() -> str:
    """Resolve the Node executable used to run Pi."""
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("Node.js was not found in PATH; it is required to run Pi in a sandbox")
    return str(Path(node_path).resolve())


def get_runtime_volumes(agent_path: str, node_path: str) -> list[str]:
    """Return read-only volumes needed by Pi and its Node runtime in a sandbox."""
    bin_dir = Path(os.path.abspath(agent_path)).parent
    prefixes = [bin_dir.parent, Path(node_path).resolve().parent.parent]
    volumes: list[str] = []
    for prefix in prefixes:
        if prefix.exists():
            volume = f"{prefix}:ro"
            if volume not in volumes:
                volumes.append(volume)
    return volumes


def _create_node_path_preload(
    agent_dir: str, node_path: str, enable_osc52_fallback: bool = False
) -> Path:
    """Create a per-launch Node preload that exposes Node tools to Pi's children."""
    descriptor, preload_name = tempfile.mkstemp(
        prefix="arkai-pi-env-", suffix=".cjs", dir=agent_dir
    )
    preload_path = Path(preload_name)
    node_bin = str(Path(node_path).parent)
    preload = (
        f"const nodeBin = {json.dumps(node_bin)};\n"
        f"const pathSeparator = {json.dumps(os.pathsep)};\n"
        'process.env.PATH = nodeBin + pathSeparator + (process.env.PATH || "");\n'
    )
    if enable_osc52_fallback:
        preload += (
            "if (!process.env.SSH_CONNECTION && !process.env.SSH_CLIENT && "
            "!process.env.MOSH_CONNECTION) {\n"
            '    process.env.MOSH_CONNECTION = "arkai-local";\n'
            "}\n"
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as preload_file:
            preload_file.write(preload)
    except Exception:
        preload_path.unlink(missing_ok=True)
        raise
    return preload_path


def _is_arkai_managed_pi(agent_path: str) -> bool:
    """Return whether the selected Pi executable is Arkai's managed install."""
    try:
        return Path(agent_path).resolve() == (get_core_dir() / "bin" / "pi").resolve()
    except (OSError, RuntimeError):
        return False


def _check_for_package_updates(agent_path: str, agent_dir: str) -> list[dict[str, str]]:
    """Use Pi's own package manager to check Arkai-installed extensions for updates."""
    package_root = Path(agent_path).resolve().parents[2]
    entrypoint = package_root / "dist" / "index.js"
    if not entrypoint.is_file():
        raise RuntimeError(f"Pi package manager entry point not found: {entrypoint}")

    package_names = json.dumps(PI_PACKAGES)
    script = (
        f"const {{ DefaultPackageManager, SettingsManager }} = "
        f"await import({json.dumps(entrypoint.as_uri())});\n"
        "const agentDir = process.env.PI_CODING_AGENT_DIR;\n"
        "const settingsManager = SettingsManager.create(agentDir, agentDir);\n"
        "const packageManager = new DefaultPackageManager({\n"
        "  cwd: agentDir, agentDir, settingsManager,\n"
        "});\n"
        "const packageNames = new Set(" + package_names + ");\n"
        "const updates = await packageManager.checkForAvailableUpdates();\n"
        "console.log(JSON.stringify(updates.filter((update) =>\n"
        '  update.scope === "user" && update.type === "npm" &&\n'
        "  packageNames.has(update.displayName)\n"
        ")));\n"
    )
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir
    try:
        result = subprocess.run(
            [_resolve_node_path(), "--input-type=module", "-e", script],
            cwd=agent_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Pi package update check failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Pi package update check failed: {detail}")
    try:
        updates = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Pi package update check returned invalid JSON") from error
    if not isinstance(updates, list):
        raise RuntimeError("Pi package update check returned an unexpected result")
    package_names = set(PI_PACKAGES)
    return [
        update
        for update in updates
        if isinstance(update, dict)
        and isinstance(update.get("source"), str)
        and isinstance(update.get("displayName"), str)
        and update.get("scope") == "user"
        and update.get("type") == "npm"
        and update["displayName"] in package_names
    ]


def _maybe_update_arkai_packages(agent_path: str) -> None:
    """Ask before upgrading outdated extensions in Arkai's managed Pi installation."""
    if not _is_arkai_managed_pi(agent_path):
        return

    agent_dir = str(get_agent_dir())
    try:
        updates = _check_for_package_updates(agent_path, agent_dir)
    except RuntimeError as error:
        utils.warn(f"Could not check Arkai-installed Pi packages for updates: {error}")
        return
    if not updates:
        return

    package_list = "\n".join(f"- {update['displayName']}" for update in updates)
    utils.warn(f"Updates are available for Arkai-installed Pi packages:\n{package_list}")
    try:
        answer = input("Upgrade these packages before starting Pi? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer not in {"", "y", "yes"}:
        utils.info("Starting Pi without updating packages.")
        return

    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = agent_dir
    for update in updates:
        utils.info(f"Updating {update['displayName']}...")
        try:
            result = subprocess.run(
                [agent_path, "update", "--extension", update["source"]],
                cwd=agent_dir,
                env=env,
                check=False,
            )
        except OSError as error:
            raise RuntimeError(f"Could not update {update['displayName']}: {error}") from error
        if result.returncode != 0:
            raise RuntimeError(
                f"Updating {update['displayName']} failed with exit code {result.returncode}"
            )
    utils.info("Pi package updates complete.")


def _prepare_agent_dir(cfg: dict, wtmcp_port: int | None) -> str:
    """Write Pi's model and optional MCP configuration files."""
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
                "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                "models": [
                    {
                        "id": model_id,
                        "name": model_name,
                        "reasoning": False,
                        "contextWindow": context_size,
                        "maxTokens": min(8192, context_size),
                    }
                ],
            }
        }
    }
    Path(agent_dir, "models.json").write_text(json.dumps(models, indent=2))
    if wtmcp_port is not None:
        Path(agent_dir, "mcp.json").write_text(
            json.dumps(
                {
                    "settings": {"projectServers": "allow"},
                    "mcpServers": {"wtmcp": {"url": f"http://127.0.0.1:{wtmcp_port}/mcp"}},
                },
                indent=2,
            )
        )
    else:
        Path(agent_dir, "mcp.json").unlink(missing_ok=True)
    return str(agent_dir)


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
    """Create Pi configuration and return its process launch specification."""
    from arkai import wtmcp
    from arkai.agent import _build_sandbox_cmd

    cloud_services = list(dict.fromkeys(service.lower() for service in cloud_services or []))
    is_cloud = bool(cloud_services)
    from arkai.agent import _cloud_provider

    cloud_provider = _cloud_provider(cloud_services)
    pi_package_json = _find_pi_package_json(agent_path) if cloud_provider == "vertex" else None
    vertex_extension_path: Path | None = None
    mcp_available = wtmcp_port is not None and wtmcp.is_wtmcp_running(wtmcp_port)
    if is_cloud:
        agent_dir = get_global_agent_dir()
        agent_dir.mkdir(parents=True, exist_ok=True)
        if cloud_provider == "vertex":
            assert pi_package_json is not None
            catalog_models = _load_vertex_anthropic_models(agent_dir)
            if catalog_models:
                _merge_vertex_anthropic_models(agent_dir, catalog_models)
                vertex_extension_path = _ensure_vertex_anthropic_extension(
                    agent_dir, pi_package_json
                )
        mcp_config = _write_run_mcp_config(str(agent_dir), wtmcp_port) if mcp_available else None
    else:
        agent_dir = Path(_prepare_agent_dir(cfg, wtmcp_port if mcp_available else None))
        mcp_config = Path(agent_dir, "mcp.json") if mcp_available else None
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    if pi_package_json is not None:
        env["ARKAI_PI_PACKAGE_JSON"] = pi_package_json
    if is_cloud:
        # Scope models without --provider, which Pi requires to accompany --model.
        # This leaves the provider's native authentication flow available.
        model_provider = VERTEX_ANTHROPIC_PROVIDER if cloud_provider == "vertex" else "openai"
        model_patterns = (
            "google-vertex/*,google-vertex-anthropic/*"
            if cloud_provider == "vertex"
            else f"{model_provider}/*"
        )
        agent_args = []
        if vertex_extension_path is not None:
            agent_args.extend(["--extension", str(vertex_extension_path)])
        agent_args.extend(["--models", model_patterns])
        if cloud_model is not None:
            agent_args.extend(["--model", cloud_model])
    else:
        from arkai.agent import _get_agent_model_id

        agent_args = ["--model", f"local-llm/{_get_agent_model_id(cfg)}"]
    theme = config.get_config_value(cfg, "agent.theme")
    if theme is not None:
        agent_args.extend(["--use-theme", theme])
    if mcp_available:
        agent_args.extend(["--mcp-config", str(mcp_config)])
    if prompt is not None:
        agent_args.extend(["--system-prompt", _PROMPT_SYSTEM_INSTRUCTION])
        agent_args.extend(["-p", prompt])

    cleanup_paths = [mcp_config] if is_cloud and mcp_config is not None else []
    try:
        if use_sandbox:
            node_path = _resolve_node_path()
            runtime_volumes = list(cli_volumes or [])
            runtime_volumes.extend(get_runtime_volumes(agent_path, node_path))
            if pi_package_json is not None:
                pi_package_dir = Path(pi_package_json).parent.resolve()
                package_is_mounted = False
                for volume in runtime_volumes:
                    source = volume.rsplit(":", 1)[0] if volume.endswith((":ro", ":rw")) else volume
                    try:
                        if pi_package_dir.is_relative_to(Path(source).resolve()):
                            package_is_mounted = True
                            break
                    except (OSError, RuntimeError):
                        continue
                if not package_is_mounted:
                    runtime_volumes.append(f"{pi_package_dir}:ro")
            sandbox_environment = dict(cli_environment or {})
            sandbox_environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
            if mcp_available:
                # Prevent project .mcp.json from replacing Arkai's host HTTP server
                # with a stdio WTMCP process running inside the agent sandbox.
                sandbox_environment["PI_MCP_CONFIG_MODE"] = "exclusive"
            if cloud_provider == "vertex":
                runtime_volumes, sandbox_environment = cloud_auth.vertex_sandbox_settings(
                    runtime_volumes, sandbox_environment
                )
                if pi_package_json is not None:
                    sandbox_environment["ARKAI_PI_PACKAGE_JSON"] = pi_package_json
            cmd = _build_sandbox_cmd(
                cfg,
                workdir,
                str(agent_dir),
                wtmcp_port if mcp_available else None,
                sandbox_profile,
                runtime_volumes,
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
            preload_path = _create_node_path_preload(
                str(agent_dir),
                node_path,
                enable_osc52_fallback=(
                    prompt is None
                    and platform.system() == "Darwin"
                    and os.environ.get("TERM_PROGRAM") != "Apple_Terminal"
                ),
            )
            cleanup_paths.append(preload_path)
            cmd = cmd + [node_path, "--require", str(preload_path), str(Path(agent_path).resolve())]
        else:
            cmd = [agent_path]
    except Exception:
        for path in cleanup_paths:
            path.unlink(missing_ok=True)
        raise

    cmd.extend(agent_args)

    suppressed_stderr_prefixes = (
        ("MCP: Project servers blocked:",) if prompt is not None and not mcp_available else ()
    )
    return AgentLaunchSpec(
        cmd,
        env,
        cleanup_paths,
        suppress_stderr_prefixes=suppressed_stderr_prefixes,
    )


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
    """Build and run Pi's launch specification."""
    if cloud_services and any(service.lower() == "openai" for service in cloud_services):
        ensure_openai_api_key(agent_path)
    if prompt is None and sys.stdin.isatty() and sys.stdout.isatty():
        _maybe_update_arkai_packages(agent_path)
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
    """Install Pi and its supported package extensions."""
    utils.warn("This will install:\n- pi.dev\n- pi-mcp-adapter\n- pi-web-access\n- pi-subagents")
    try:
        answer = input("Proceed with the installation? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in {"y", "yes"}:
        utils.info("Pi installation cancelled.")
        return

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm was not found. Install Node.js and npm, then retry.")
    core_dir = get_core_dir()
    agent_dir = get_agent_dir()
    utils.info(f"Installing {PI_CORE_PACKAGE}...")
    run_install_command(
        [npm_path, "install", "--prefix", str(core_dir), "-g", "--ignore-scripts", PI_CORE_PACKAGE]
    )
    pi_path = core_dir / "bin" / "pi"
    install_env = os.environ.copy()
    install_env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    for package in PI_PACKAGES:
        utils.info(f"Installing {package}...")
        run_install_command([str(pi_path), "install", f"npm:{package}"], env=install_env)
    utils.info(f"Pi installation complete: {core_dir}")
