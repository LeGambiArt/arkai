"""Agent and prompt execution."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from typing import Optional

from arkai import config, engine, utils


def _merge_volumes(profile_volumes: list, cli_volumes: Optional[list] = None) -> list:
    """Merge profile and CLI volumes with deduplication.

    Args:
        profile_volumes: Volumes from the resolved profile
        cli_volumes: Volumes from CLI arguments (optional)

    Returns:
        Merged and deduplicated volume list

    Raises:
        RuntimeError: If same path appears with different flags
    """
    from arkai import sandbox as sandbox_module

    merged = list(profile_volumes) if profile_volumes else []
    if cli_volumes:
        merged.extend(cli_volumes)

    return sandbox_module._deduplicate_volumes(merged)


def _merge_environment(profile_env: Optional[dict], cli_env: Optional[dict]) -> dict:
    """Merge profile and CLI environment variables; CLI values override profile.

    Args:
        profile_env: Environment variables from the resolved profile
        cli_env: Environment variables from CLI arguments (optional)

    Returns:
        Merged environment dict
    """
    merged = dict(profile_env) if profile_env else {}
    if cli_env:
        merged.update(cli_env)
    return merged


def _resolve_sandbox_profile(cfg: dict, profile_name: Optional[str] = None) -> dict:
    """Resolve active sandbox profile from CLI, config, or defaults.

    Priority: CLI --sandbox flag > config sandbox.active_profile > defaults

    Args:
        cfg: Loaded configuration
        profile_name: Profile name from --sandbox CLI flag (highest priority)

    Returns:
        Resolved profile dict with keys: path, memory_mb, cpus, pids, timeout

    Raises:
        RuntimeError: If specified profile does not exist
    """
    from arkai import sandbox as sandbox_module

    # Priority 1: CLI flag
    if profile_name:
        profile = sandbox_module._get_profile(cfg, profile_name)
        if not profile:
            raise RuntimeError(f"Sandbox profile not found: {profile_name}")
        return profile

    # Priority 2: Config active_profile
    active_profile_name = config.get_config_value(cfg, "sandbox.active_profile")
    if active_profile_name:
        profile = sandbox_module._get_profile(cfg, active_profile_name)
        if not profile:
            raise RuntimeError(
                f"Sandbox active_profile '{active_profile_name}' not found in config"
            )
        return profile

    # Priority 3: Defaults
    return sandbox_module._get_default_profile(cfg)


def _get_model_name(cfg: dict) -> str:
    """Extract model name from config."""
    model_file = config.get_config_value(cfg, "inference.model")
    hf_model = config.get_config_value(cfg, "inference.hf")

    if hf_model:
        return hf_model.split("/")[-1].replace("-GGUF", "").replace("-gguf", "")
    elif model_file:
        return os.path.basename(model_file).split(".")[0].split("-Q")[0]
    return "model"


def _start_wtmcp_server(cfg: dict) -> int:
    """Start wtmcp server and return port. Assumes inference server is running."""
    from arkai import wtmcp

    wtmcp_port = config.get_config_value(cfg, "wtmcp.port", 8080)

    if not wtmcp.is_wtmcp_running(wtmcp_port):
        try:
            wtmcp.cmd_wtmcp_start(port=wtmcp_port)
        except RuntimeError as e:
            utils.error(str(e), 1)
            sys.exit(1)

    return wtmcp_port


def _build_sandbox_cmd(
    cfg: dict,
    workdir: Optional[str],
    config_dir: Optional[str],
    wtmcp_port: Optional[int],
    sandbox_profile: Optional[str] = None,
    cli_volumes: Optional[list] = None,
    cli_environment: Optional[dict] = None,
) -> list:
    """Build the arapuca sandbox command prefix for an agent invocation.

    Args:
        cfg: Loaded configuration
        workdir: Directory to mount read-write and use as --cwd, or None to skip both.
            Defaults to the current directory at the call site; pass None only for --no-cwd.
        config_dir: Directory containing agent config files; mounted separately when not
            already covered by workdir. Pass None when the agent has no config file.
        wtmcp_port: wtmcp port to allow on Linux, or None if MCP is disabled

    Returns:
        List of command tokens ending with "--", ready for the agent command to be appended

    Raises:
        RuntimeError: If arapuca binary cannot be found
    """
    # Resolve the active sandbox profile
    profile = _resolve_sandbox_profile(cfg, sandbox_profile)

    arapuca_bin = profile.get("path", "arapuca")
    try:
        arapuca_path = utils.resolve_binary(arapuca_bin)
    except RuntimeError as e:
        raise RuntimeError(f"arapuca not found: {e}. Install arapuca or use --no-sandbox") from e

    memory = profile.get("memory_mb", 2048)
    cpus_value = profile.get("cpus", 2)
    pids = profile.get("pids", 256)
    timeout = profile.get("timeout", 0)
    inference_port = config.get_config_value(cfg, "inference.port", 8081)

    # Merge profile and CLI volumes
    profile_volumes = profile.get("volume", [])
    try:
        volumes = _merge_volumes(profile_volumes, cli_volumes)
    except RuntimeError as e:
        raise RuntimeError(f"Volume configuration error: {e}") from e

    # Merge profile and CLI environment variables
    profile_environment = profile.get("environment")
    environment = _merge_environment(profile_environment, cli_environment)

    cmd: list = [arapuca_path, "run"]

    if workdir:
        cmd += ["-v", f"{workdir}:rw"]

    # Mount config_dir separately when it is not already covered by the workdir mount
    if config_dir:
        config_dir_abs = os.path.abspath(config_dir)
        workdir_abs = os.path.abspath(workdir) if workdir else None
        if not workdir_abs or not config_dir_abs.startswith(workdir_abs + os.sep):
            cmd += ["-v", f"{config_dir_abs}:rw"]

    # Add profile and CLI volumes
    for vol in volumes:
        cmd += ["-v", vol]

    if platform.system() == "Linux":
        cmd += ["--allow-host", f"127.0.0.1:{inference_port}"]
        if wtmcp_port is not None:
            cmd += ["--allow-host", f"127.0.0.1:{wtmcp_port}"]
        cmd += ["--deny-network"]
    else:
        cmd += ["--seccomp", "baseline"]

    cmd += ["--memory", str(memory), "--cpus", str(cpus_value * 100), "--pids", str(pids)]

    if workdir:
        cmd += ["--cwd", workdir]

    term = os.environ.get("TERM")
    if term:
        cmd += ["--env", f"TERM={term}"]
    colorterm = os.environ.get("COLORTERM")
    if colorterm:
        cmd += ["--env", f"COLORTERM={colorterm}"]

    for env_key, env_val in environment.items():
        cmd += ["--env", f"{env_key}={env_val}"]

    if timeout and int(timeout) > 0:
        cmd += ["--timeout", str(timeout)]

    cmd += ["--"]
    return cmd


def _start_agent_opencode(
    agent_path: str,
    cfg: dict,
    wtmcp_port: Optional[int],
    use_sandbox: bool,
    workdir: Optional[str],
    sandbox_profile: Optional[str] = None,
    cli_volumes: Optional[list] = None,
    cli_environment: Optional[dict] = None,
) -> None:
    """Start opencode agent."""
    if use_sandbox:
        config_dir = os.path.join(workdir or os.getcwd(), ".arkai-session")
    else:
        config_dir = os.path.expanduser("~/.local/state/arkai/opencode")
    os.makedirs(config_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=config_dir,
        prefix="opencode-",
        suffix=".json",
        delete=False,
    ) as f:
        config_file = f.name
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
        json.dump(config_data, f, indent=2)

    if use_sandbox:
        sandbox_prefix = _build_sandbox_cmd(
            cfg, workdir, config_dir, wtmcp_port, sandbox_profile, cli_volumes, cli_environment
        )
        cmd = sandbox_prefix + ["env", f"OPENCODE_CONFIG={config_file}", agent_path]
        env = os.environ.copy()
    else:
        env = os.environ.copy()
        env["OPENCODE_CONFIG"] = config_file
        cmd = [agent_path]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=None,
            stderr=None,
            env=env,
        )
        proc.wait()
    except FileNotFoundError as e:
        utils.error(str(e), 3)
        sys.exit(3)
    finally:
        try:
            os.remove(config_file)
        except FileNotFoundError:
            pass
    # Don't exit - let the caller decide what to do
    # Infrastructure services stay running


def _start_agent_crush(
    agent_path: str,
    cfg: dict,
    wtmcp_port: Optional[int],
    use_sandbox: bool,
    workdir: Optional[str],
    sandbox_profile: Optional[str] = None,
    cli_volumes: Optional[list] = None,
    cli_environment: Optional[dict] = None,
) -> None:
    """Start crush agent."""
    if use_sandbox:
        config_dir = os.path.join(workdir or os.getcwd(), ".arkai-session")
    else:
        config_dir = os.path.expanduser("~/.local/state/arkai")
    os.makedirs(config_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=config_dir,
        prefix=".crush-",
        suffix=".json",
        delete=False,
    ) as f:
        config_file = f.name
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
        json.dump(config_data, f, indent=2)

    if use_sandbox:
        sandbox_prefix = _build_sandbox_cmd(
            cfg, workdir, config_dir, wtmcp_port, sandbox_profile, cli_volumes, cli_environment
        )
        cmd = sandbox_prefix + [
            "env",
            f"CRUSH_CONFIG={config_file}",
            agent_path,
            "--cwd",
            config_dir,
        ]
        env = os.environ.copy()
    else:
        env = os.environ.copy()
        env["CRUSH_CONFIG"] = config_file
        cmd = [agent_path, "--cwd", config_dir]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=None,
            stderr=None,
            env=env,
        )
        proc.wait()
    except FileNotFoundError as e:
        utils.error(str(e), 3)
        sys.exit(3)
    finally:
        try:
            os.remove(config_file)
        except FileNotFoundError:
            pass
    # Don't exit - let the caller decide what to do
    # Infrastructure services stay running


def _start_agent_claude(
    agent_path: str,
    cfg: dict,
    wtmcp_port: Optional[int],
    use_sandbox: bool,
    workdir: Optional[str],
    sandbox_profile: Optional[str] = None,
    cli_volumes: Optional[list] = None,
    cli_environment: Optional[dict] = None,
) -> None:
    """Start claude agent."""
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_name = _get_model_name(cfg)

    anthropic_env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{inference_port}",
        "ANTHROPIC_AUTH_TOKEN": "local",
        "ANTHROPIC_API_KEY": "local",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }

    mcp_config: Optional[str] = None
    agent_args = [agent_path, "--model", model_name]

    if wtmcp_port is not None:
        if use_sandbox:
            config_dir = os.path.join(workdir or os.getcwd(), ".arkai-session")
        else:
            config_dir = os.path.expanduser("~/.local/state/arkai")
        os.makedirs(config_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=config_dir,
            prefix=".mcp-",
            suffix=".json",
            delete=False,
        ) as f:
            mcp_config = f.name
            config_data = {
                "mcpServers": {
                    "wtmcp": {"type": "url", "url": f"http://127.0.0.1:{wtmcp_port}/mcp"}
                }
            }
            json.dump(config_data, f, indent=2)

        agent_args.extend(["--mcp-config", mcp_config])

    if use_sandbox:
        # config_dir is only relevant when an MCP config file was written
        sandbox_config_dir = (
            os.path.join(workdir or os.getcwd(), ".arkai-session") if mcp_config else None
        )
        sandbox_prefix = _build_sandbox_cmd(
            cfg,
            workdir,
            sandbox_config_dir,
            wtmcp_port,
            sandbox_profile,
            cli_volumes,
            cli_environment,
        )
        env_pairs = [f"{k}={v}" for k, v in anthropic_env.items()]
        cmd = sandbox_prefix + ["env"] + env_pairs + agent_args
        env = os.environ.copy()
    else:
        env = os.environ.copy()
        env.update(anthropic_env)
        cmd = agent_args

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=None,
            stderr=None,
            env=env,
        )
        proc.wait()
    except FileNotFoundError as e:
        utils.error(str(e), 3)
        sys.exit(3)
    finally:
        if mcp_config is not None:
            try:
                os.remove(mcp_config)
            except FileNotFoundError:
                pass
    # Don't exit - let the caller decide what to do
    # Infrastructure services stay running


def cmd_agent(
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    keep_inference: bool = False,
    keep_mcp: bool = False,
    no_mcp: bool = False,
    no_sandbox: bool = False,
    no_cwd: bool = False,
    sandbox_cwd: Optional[str] = None,
    sandbox_profile: Optional[str] = None,
    sandbox_volume: Optional[list] = None,
    sandbox_environment: Optional[dict] = None,
) -> None:
    """Start interactive agent session (requires TTY).

    Args:
        agent_name: Override agent from config
        model: Override model from config
        keep_inference: Keep inference server running after exit
        keep_mcp: Keep wtmcp server running after exit
        no_mcp: Skip wtmcp initialization regardless of config
        no_sandbox: Skip arapuca sandbox regardless of config
        no_cwd: Do not mount the current directory in the sandbox
        sandbox_cwd: Override the directory mounted and set as cwd in the sandbox
        sandbox_profile: Use specific sandbox profile for this run
        sandbox_volume: List of volumes to mount in the sandbox
        sandbox_environment: Dict of environment variables to set in the sandbox

    Raises:
        RuntimeError: If not in TTY or config invalid
    """
    # Check TTY
    if not sys.stdin.isatty():
        utils.error("agent requires a TTY (interactive terminal)", 1)
        sys.exit(1)

    # Load config
    cfg = config.load_config()

    # Override with CLI args before validation so CLI model satisfies the requirement
    if agent_name:
        cfg["agent"]["name"] = agent_name
    if model:
        if model.startswith("hf:"):
            cfg["inference"]["hf"] = model[3:]
            cfg["inference"].pop("model", None)
        else:
            cfg["inference"]["model"] = model

    config.validate_config(cfg)

    agent_name = config.get_config_value(cfg, "agent.name", "opencode")

    # Validate agent is supported
    if agent_name not in {"opencode", "crush", "claude"}:
        utils.error(f"Unsupported agent: {agent_name} (supported: opencode, crush, claude)", 2)
        sys.exit(2)

    # Determine whether to use MCP: config key agent.mcp (default True) and --no-mcp flag
    use_mcp = bool(config.get_config_value(cfg, "agent.mcp", True)) and not no_mcp

    # Determine whether to use sandbox: disabled by --no-sandbox or sandbox.disable config key
    use_sandbox = not no_sandbox and not bool(
        config.get_config_value(cfg, "sandbox.disable", False)
    )

    # Resolve sandbox working directory (ignored when sandbox is disabled)
    if no_cwd:
        workdir: Optional[str] = None
    elif sandbox_cwd:
        workdir = os.path.abspath(sandbox_cwd)
    else:
        workdir = os.getcwd()

    # Ensure inference is running
    if not engine.is_inference_running():
        try:
            engine.cmd_engine_start(model=model)
        except RuntimeError as e:
            utils.error(str(e), 1)
            sys.exit(1)

    # Start wtmcp if MCP is enabled
    wtmcp_port: Optional[int] = _start_wtmcp_server(cfg) if use_mcp else None

    # Resolve agent binary
    agent_bin = config.get_config_value(cfg, "agent.path", agent_name)
    try:
        agent_path_result = utils.resolve_binary(agent_bin)
    except RuntimeError as e:
        utils.error(str(e), 3)
        sys.exit(3)

    # Type guard: resolve_binary raises on error, so result is str
    assert agent_path_result is not None
    agent_path: str = agent_path_result

    # Start interactive agent
    try:
        if agent_name == "opencode":
            _start_agent_opencode(
                agent_path,
                cfg,
                wtmcp_port,
                use_sandbox,
                workdir,
                sandbox_profile,
                sandbox_volume,
                sandbox_environment,
            )
        elif agent_name == "crush":
            _start_agent_crush(
                agent_path,
                cfg,
                wtmcp_port,
                use_sandbox,
                workdir,
                sandbox_profile,
                sandbox_volume,
                sandbox_environment,
            )
        elif agent_name == "claude":
            _start_agent_claude(
                agent_path,
                cfg,
                wtmcp_port,
                use_sandbox,
                workdir,
                sandbox_profile,
                sandbox_volume,
                sandbox_environment,
            )
    finally:
        # Cleanup services unless user requested to keep them
        from arkai import wtmcp

        if wtmcp_port is not None:
            if keep_mcp and wtmcp.is_wtmcp_running(wtmcp_port):
                print(f"wtmcp still running on port {wtmcp_port}", file=sys.stderr)
            elif not keep_mcp and wtmcp.is_wtmcp_running(wtmcp_port):
                try:
                    wtmcp.cmd_wtmcp_stop(wtmcp_port)
                except Exception:
                    pass

        if keep_inference and engine.is_inference_running():
            inference_port = config.get_config_value(cfg, "inference.port", 8081)
            print(f"Inference server still running on port {inference_port}", file=sys.stderr)
        elif not keep_inference and engine.is_inference_running():
            try:
                engine.cmd_engine_stop()
            except Exception:
                pass
