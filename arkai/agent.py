"""Agent and prompt execution."""

import argparse
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from arkai import config, inference, utils


def exec_cmd(args: dict | None = None) -> None:
    """Select 'agent' command to execute."""
    agent_env = (
        {k: v for e in getattr(args, "environment", None) or [] for k, _, v in [e.partition("=")]}
        if getattr(args, "environment", None)
        else None
    )
    match args.agent_cmd:  # ty: ignore[unresolved-attribute]
        case "start":
            cmd_agent(
                args.agent,  # ty: ignore[unresolved-attribute]
                args.model,  # ty: ignore[unresolved-attribute]
                args.no_inference,  # ty: ignore[unresolved-attribute]
                args.no_mcp,  # ty: ignore[unresolved-attribute]
                args.no_sandbox,  # ty: ignore[unresolved-attribute]
                args.cwd,  # ty: ignore[unresolved-attribute]
                args.sandbox,  # ty: ignore[unresolved-attribute]
                args.volumes,  # ty: ignore[unresolved-attribute]
                agent_env,
                args.port,  # ty: ignore[unresolved-attribute]
                args.backend,  # ty: ignore[unresolved-attribute]
            )
        case "prompt":
            cmd_agent_prompt(
                args.prompt_text,  # ty: ignore[unresolved-attribute]
                args.agent,  # ty: ignore[unresolved-attribute]
                args.model,  # ty: ignore[unresolved-attribute]
                args.no_inference,  # ty: ignore[unresolved-attribute]
                args.no_mcp,  # ty: ignore[unresolved-attribute]
                args.no_sandbox,  # ty: ignore[unresolved-attribute]
                args.cwd,  # ty: ignore[unresolved-attribute]
                args.sandbox,  # ty: ignore[unresolved-attribute]
                args.volumes,  # ty: ignore[unresolved-attribute]
                agent_env,
                args.output,  # ty: ignore[unresolved-attribute]
                args.port,  # ty: ignore[unresolved-attribute]
                args.backend,  # ty: ignore[unresolved-attribute]
            )
        case "install":
            cmd_agent_install(args.agent_name)  # ty: ignore[unresolved-attribute]


def _add_agent_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by agent start and agent prompt.

    Args:
        parser: The argparse subparser to add arguments to
    """
    parser.add_argument("-a", "--agent", help="Override agent")
    parser.add_argument("-m", "--model", help="Override model")
    parser.add_argument("--port", type=int, help="Use a separate inference server port")
    parser.add_argument(
        "-I",
        "--no-inference",
        action="store_true",
        help="Do not start inference engine server",
    )
    parser.add_argument("-M", "--no-mcp", action="store_true", help="Skip wtmcp initialization")
    parser.add_argument("--no-sandbox", action="store_true", help="Skip arapuca sandbox")
    parser.add_argument(
        "-s",
        "--sandbox",
        metavar="PROFILE",
        help="Use specific sandbox profile for this run",
    )
    parser.add_argument(
        "-v",
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume in the sandbox (format: /path or /path:ro)",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        dest="environment",
        metavar="KEY=VALUE",
        help="Set an environment variable in the sandbox (KEY=VALUE). Can be used multiple times",
    )
    parser.add_argument(
        "--cwd",
        metavar="PATH",
        help="Override the directory mounted as cwd in the sandbox (defaults to os.getcwd())",
    )


def ingest_cli_options(subparsers: argparse._SubParsersAction) -> None:
    """Create command subparser.

    Args:
        parser: The argparse subparser to add arguments to
    """
    agent_parser = subparsers.add_parser("agent", help="Manage interactive agent")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_cmd", required=True)
    agent_start_parser = agent_subparsers.add_parser("start", help="Start interactive agent")
    _add_agent_common_args(agent_start_parser)
    agent_start_parser.add_argument("--backend", help="Override inference backend from config")
    agent_prompt_parser = agent_subparsers.add_parser(
        "prompt", help="Run agent with a prompt non-interactively"
    )
    _add_agent_common_args(agent_prompt_parser)
    agent_prompt_parser.add_argument("--backend", help="Override inference backend from config")
    agent_prompt_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Write agent output to file instead of stdout"
    )
    agent_prompt_parser.add_argument(
        "prompt_text", nargs="*", help="Prompt text (reads from stdin if not provided)"
    )
    agent_install_parser = agent_subparsers.add_parser("install", help="Install a supported agent")
    agent_install_parser.add_argument(
        "agent_name",
        nargs="?",
        help="Agent to install (omit to list supported agents)",
    )


@dataclass
class AgentContext:
    """Shared state for agent infrastructure setup and teardown."""

    cfg: dict = field(default_factory=dict)
    agent_name: str = ""
    agent_path: str = ""
    use_mcp: bool = False
    use_sandbox: bool = False
    workdir: str | None = None
    wtmcp_port: int | None = None
    no_start_inference: bool = False
    sandbox_profile: str | None = None
    sandbox_volume: list | None = None
    sandbox_environment: dict | None = None


@dataclass
class AgentLaunchSpec:
    """Command and resources prepared for one agent process."""

    command: list[str]
    environment: dict[str, str]
    cleanup_paths: list[Path] = field(default_factory=list)


def run_launch_spec(spec: AgentLaunchSpec, capture_stdout: bool = False) -> str | None:
    """Run an agent launch specification and clean up its temporary files."""
    try:
        process = subprocess.Popen(
            spec.command,
            stdin=sys.stdin,
            stdout=subprocess.PIPE if capture_stdout else None,
            stderr=None,
            env=spec.environment,
        )
        if capture_stdout:
            stdout_data, _ = process.communicate()
        else:
            process.wait()
            stdout_data = None
    except FileNotFoundError as error:
        utils.error(str(error), 3)
        sys.exit(3)
    finally:
        for path in spec.cleanup_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    if capture_stdout and stdout_data is not None:
        return stdout_data.decode("utf-8", errors="replace")
    return None


def _select_agent_name(agent_name: str | None) -> str:
    """Select the CLI agent override or the configured agent name."""
    if agent_name:
        return agent_name
    cfg = config.load_config()
    return config.get_config_value(cfg, "agent.name", "opencode")


@contextmanager
def _agent_context(
    agent_name: str,
    model: str | None = None,
    port: int | None = None,
    no_start_inference: bool = False,
    no_mcp: bool = False,
    no_sandbox: bool = False,
    sandbox_cwd: str | None = None,
    sandbox_profile: str | None = None,
    sandbox_volume: list | None = None,
    sandbox_environment: dict | None = None,
    backend: str | None = None,
) -> Iterator[AgentContext]:
    """Set up and tear down agent infrastructure.

    Loads config, starts inference and wtmcp servers, resolves agent binary,
    and yields an AgentContext. On exit, stops services unless keep flags are set.

    Args:
        agent_name: Selected agent name
        model: Override model from config
        port: Override inference server port from config. An explicit port allows a
            separate inference server to run alongside an existing one.
        no_start_inference: Do not start inference engine
        no_mcp: Skip wtmcp initialization regardless of config
        no_sandbox: Skip arapuca sandbox regardless of config
        sandbox_cwd: Override the directory mounted and set as cwd in the sandbox
        sandbox_profile: Use specific sandbox profile for this run
        sandbox_volume: List of volumes to mount in the sandbox
        sandbox_environment: Dict of environment variables to set in the sandbox
        backend: Override inference backend from config

    Yields:
        AgentContext with all infrastructure ready

    Raises:
        RuntimeError: If config invalid or services fail to start
    """
    cfg = config.load_config()

    cfg["agent"]["name"] = agent_name
    if model:
        cfg["inference"]["model"] = model
    if port is not None:
        cfg["inference"]["port"] = port
    if backend is not None:
        cfg["inference"]["backend"] = backend

    inference_running = (
        inference.is_inference_running(port)
        if port is not None
        else inference.is_inference_running()
    )
    require_model = not inference_running
    config.validate_config(cfg, require_model=require_model)

    if agent_name not in config.VALID_AGENTS:
        utils.error(
            f"Unsupported agent: {agent_name} "
            f"(supported: {', '.join(sorted(config.VALID_AGENTS))})",
            2,
        )
        sys.exit(2)

    use_mcp = bool(config.get_config_value(cfg, "agent.mcp", True)) and not no_mcp
    use_sandbox = not no_sandbox and not bool(
        config.get_config_value(cfg, "sandbox.disable", False)
    )

    launcher_modules = {
        "opencode": "arkai.agent_opencode",
        "crush": "arkai.agent_crush",
        "claude": "arkai.agent_claude",
        "pi": "arkai.agent_pi",
    }
    module = __import__(launcher_modules[agent_name], fromlist=["get_agent_path"])
    agent_path: str = module.get_agent_path(cfg)

    # Validate arapuca binary early, before starting any servers.
    if use_sandbox:
        sandbox_profile_obj = _resolve_sandbox_profile(cfg, sandbox_profile)
        arapuca_bin = sandbox_profile_obj.get("path", "arapuca")
        try:
            utils.resolve_binary(arapuca_bin)
        except RuntimeError as e:
            raise RuntimeError(
                f"arapuca not found: {e}. Install arapuca or use --no-sandbox"
            ) from e

    if sandbox_cwd:
        workdir = os.path.abspath(sandbox_cwd)
    else:
        workdir = os.getcwd()

    errors = []
    wtmcp_port, wtmcp_started = (None, False)
    engine_started: bool = False
    try:
        if not inference_running:
            if no_start_inference:
                raise RuntimeError("Inference engine is not running.")
            else:
                if port is None:
                    if backend is None:
                        inference.cmd_inference_start(model=model)  # may raise RuntimeError
                    else:
                        inference.cmd_inference_start(model=model, backend=backend)
                else:
                    if backend is None:
                        inference.cmd_inference_start(model=model, port=port)
                    else:
                        inference.cmd_inference_start(model=model, port=port, backend=backend)
                engine_started = True

        inference_port = config.get_config_value(cfg, "inference.port", 8081)
        active_model = inference.get_inference_model(inference_port)
        cfg.setdefault("inference", {})["model"] = active_model
        _sync_running_inference_backend(cfg, port)

        if use_mcp:
            # may raise RuntimeError
            wtmcp_port, wtmcp_started = _start_wtmcp_server(cfg)

        ctx = AgentContext(
            cfg=cfg,
            agent_name=agent_name,
            agent_path=agent_path,
            use_mcp=use_mcp,
            use_sandbox=use_sandbox,
            workdir=workdir,
            wtmcp_port=wtmcp_port,
            no_start_inference=no_start_inference,
            sandbox_profile=sandbox_profile,
            sandbox_volume=sandbox_volume,
            sandbox_environment=sandbox_environment,
        )

        yield ctx  # may raise from execution
    except Exception as ex:
        errors.append(ex)
    finally:
        from arkai import wtmcp

        if wtmcp_started and wtmcp.is_wtmcp_running(wtmcp_port):
            try:
                wtmcp.cmd_wtmcp_stop(wtmcp_port)
            except Exception as ex:
                errors.append(ex)

        if engine_started and inference.is_inference_running(port):
            try:
                if port is None:
                    inference.cmd_inference_stop()
                else:
                    inference.cmd_inference_stop(port)
            except Exception as ex:
                errors.append(ex)
        if errors:
            if len(errors) > 1:
                error_list = "\n- ".join(str(e) for e in errors)
                raise RuntimeError(rf"Multiple errors starting agent:\s{error_list}")
            else:
                raise errors[0] from None


def _merge_volumes(profile_volumes: list, cli_volumes: list | None = None) -> list:
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


def _merge_environment(profile_env: dict | None, cli_env: dict | None) -> dict:
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


def _resolve_sandbox_profile(cfg: dict, profile_name: str | None = None) -> dict:
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
    """Extract model name from config, with fallback for running inference service.

    Returns the model name from config, or a generic name if no model configured
    but an inference service is running.
    """
    model_file = config.get_config_value(cfg, "inference.model")

    if model_file and model_file.startswith(("hf:", "ollama:")):
        model_reference = model_file.split(":", 1)[1]
        model_name = model_reference.rsplit(":", 1)[0].split("/")[-1]
        return model_name.replace("-GGUF", "").replace("-gguf", "")
    elif model_file:
        return os.path.splitext(os.path.basename(model_file))[0]
    return "local-model"


def _get_agent_model_id(cfg: dict) -> str:
    """Return the model ID agent clients should send to the inference server."""
    inference_backend = config.get_config_value(cfg, "inference.backend", "llama-cpp")
    if inference_backend == "mlx":
        return "default_model"
    return _get_model_name(cfg)


def _sync_running_inference_backend(cfg: dict, port: int | None = None) -> None:
    """Use the backend recorded by the running inference instance when available."""
    state_path = inference.get_inference_state_path(port)
    try:
        state = utils.load_yaml(state_path)
    except FileNotFoundError:
        return

    backend = state.get("backend")
    if isinstance(backend, str) and backend:
        cfg.setdefault("inference", {})["backend"] = backend


def _start_wtmcp_server(cfg: dict) -> tuple[int, bool]:
    """Start wtmcp server and return port. Assumes inference server is running."""
    from arkai import wtmcp

    wtmcp_port = config.get_config_value(cfg, "wtmcp.port", 8080)

    wtmcp_started = False
    if not wtmcp.is_wtmcp_running(wtmcp_port):
        try:
            wtmcp.cmd_wtmcp_start(port=wtmcp_port)
            wtmcp_started = True
        except RuntimeError as e:
            utils.error(str(e), 1)
            sys.exit(1)

    return wtmcp_port, wtmcp_started


def _build_sandbox_cmd(
    cfg: dict,
    workdir: str | None,
    config_dir: str | None,
    wtmcp_port: int | None,
    sandbox_profile: str | None = None,
    cli_volumes: list | None = None,
    cli_environment: dict | None = None,
    tty: bool = False,
) -> list:
    """Build the arapuca sandbox command prefix for an agent invocation.

    Args:
        cfg: Loaded configuration
        workdir: Directory to mount read-write and use as --cwd.
            Defaults to the current directory at the call site.
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
    else:
        cmd += ["-v", f"{os.getcwd()}:ro"]

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

    if tty:
        cmd += ["--tty"]

    cmd += ["--"]
    return cmd


def _dispatch_agent(
    ctx: AgentContext,
    prompt: str | None = None,
    capture_stdout: bool = False,
) -> str | None:
    """Dispatch to the appropriate agent start function.

    Args:
        ctx: Agent infrastructure context
        prompt: If set, run non-interactively with this prompt
        capture_stdout: If True, capture and return stdout

    Returns:
        Captured stdout when capture_stdout is True, None otherwise
    """
    launcher_modules = {
        "opencode": "arkai.agent_opencode",
        "crush": "arkai.agent_crush",
        "claude": "arkai.agent_claude",
        "pi": "arkai.agent_pi",
    }
    module_name = launcher_modules.get(ctx.agent_name)
    if module_name:
        module = __import__(module_name, fromlist=["start"])
        return module.start(
            ctx.agent_path,
            ctx.cfg,
            ctx.wtmcp_port,
            ctx.use_sandbox,
            ctx.workdir,
            ctx.sandbox_profile,
            ctx.sandbox_volume,
            ctx.sandbox_environment,
            prompt,
            capture_stdout,
        )
    return None


def run_install_command(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run an installation command and raise an actionable error on failure."""
    code, _, stderr = utils.run_command(command, timeout=None, env=env)
    if code != 0:
        detail = stderr.strip() or f"exit code {code}"
        raise RuntimeError(f"Installation command failed ({' '.join(command)}): {detail}")


def cmd_agent_install(agent_name: str | None = None) -> None:
    """Install a supported agent, or list agents with an available installer.

    Args:
        agent_name: Agent to install. If omitted, print the installable agents.

    Raises:
        ValueError: If the requested agent has no installer.
    """
    installers = {"pi": "arkai.agent_pi"}
    if agent_name is None:
        utils.info("Agents supported for installation: " + ", ".join(installers))
        return

    module_name = installers.get(agent_name)
    if module_name is None:
        raise ValueError(
            f"Agent '{agent_name}' is not supported for installation. "
            f"Supported agents: {', '.join(installers)}"
        )

    module = __import__(module_name, fromlist=["install"])
    module.install()


def cmd_agent(
    agent_name: str | None = None,
    model: str | None = None,
    no_start_inference: bool = False,
    no_mcp: bool = False,
    no_sandbox: bool = False,
    sandbox_cwd: str | None = None,
    sandbox_profile: str | None = None,
    sandbox_volume: list | None = None,
    sandbox_environment: dict | None = None,
    port: int | None = None,
    backend: str | None = None,
) -> None:
    """Start interactive agent session (requires TTY).

    Args:
        agent_name: Override agent from config
        model: Override model from config
        port: Use a separate inference server on this port
        backend: Override inference backend from config
        no_start_inference: Do not start inference engine
        no_mcp: Skip wtmcp initialization regardless of config
        no_sandbox: Skip arapuca sandbox regardless of config
        sandbox_cwd: Override the directory mounted and set as cwd in the sandbox
        sandbox_profile: Use specific sandbox profile for this run
        sandbox_volume: List of volumes to mount in the sandbox
        sandbox_environment: Dict of environment variables to set in the sandbox

    Raises:
        RuntimeError: If not in TTY or config invalid
    """
    if not sys.stdin.isatty():
        utils.error("agent start requires a TTY (interactive terminal)", 1)
        sys.exit(1)

    selected_agent_name = _select_agent_name(agent_name)
    with _agent_context(
        agent_name=selected_agent_name,
        model=model,
        port=port,
        no_start_inference=no_start_inference,
        no_mcp=no_mcp,
        no_sandbox=no_sandbox,
        sandbox_cwd=sandbox_cwd,
        sandbox_profile=sandbox_profile,
        sandbox_volume=sandbox_volume,
        sandbox_environment=sandbox_environment,
        backend=backend,
    ) as ctx:
        _dispatch_agent(ctx)


def cmd_agent_prompt(
    prompt_args: list | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    no_start_inference: bool = False,
    no_mcp: bool = False,
    no_sandbox: bool = False,
    sandbox_cwd: str | None = None,
    sandbox_profile: str | None = None,
    sandbox_volume: list | None = None,
    sandbox_environment: dict | None = None,
    output_file: str | None = None,
    port: int | None = None,
    backend: str | None = None,
) -> None:
    """Run agent non-interactively with a prompt.

    The prompt is taken from CLI positional arguments if provided,
    otherwise read from stdin. Agent stdout is captured and either
    written to output_file or printed to stdout with markers on stderr.

    Args:
        prompt_args: Prompt text from CLI positional arguments
        agent_name: Override agent from config
        model: Override model from config
        port: Use a separate inference server on this port
        backend: Override inference backend from config
        no_start_inference: Do not start inference engine
        no_mcp: Skip wtmcp initialization regardless of config
        no_sandbox: Skip arapuca sandbox regardless of config
        sandbox_cwd: Override the directory mounted and set as cwd in the sandbox
        sandbox_profile: Use specific sandbox profile for this run
        sandbox_volume: List of volumes to mount in the sandbox
        sandbox_environment: Dict of environment variables to set in the sandbox
        output_file: Path to write agent output to; if None, print to stdout

    Raises:
        RuntimeError: If no prompt provided or config invalid
    """
    prompt: str | None = None

    if prompt_args:
        prompt = " ".join(prompt_args).strip()

    if not prompt:
        if sys.stdin.isatty():
            utils.error(
                "No prompt provided. Usage: arkai agent prompt 'your prompt'"
                " or echo 'prompt' | arkai agent prompt",
                1,
            )
            sys.exit(1)
        prompt = sys.stdin.read().strip()

    if not prompt:
        utils.error("Empty prompt", 1)
        sys.exit(1)

    selected_agent_name = _select_agent_name(agent_name)
    with _agent_context(
        agent_name=selected_agent_name,
        model=model,
        port=port,
        no_start_inference=no_start_inference,
        no_mcp=no_mcp,
        no_sandbox=no_sandbox,
        sandbox_cwd=sandbox_cwd,
        sandbox_profile=sandbox_profile,
        sandbox_volume=sandbox_volume,
        sandbox_environment=sandbox_environment,
        backend=backend,
    ) as ctx:
        result = _dispatch_agent(ctx, prompt, capture_stdout=True)

    answer = result.rstrip("\n") if result else ""

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(answer)
            f.write("\n")
        utils.info(f"Output written to {output_file}")
    else:
        print("====== Answer ======", file=sys.stderr)
        print(answer)
        print("====================", file=sys.stderr)
