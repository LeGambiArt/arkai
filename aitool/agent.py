"""Agent and prompt execution."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from typing import Optional

from aitool import config, engine, utils


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
    from aitool import wtmcp

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
    arapuca_bin = config.get_config_value(cfg, "sandbox.path", "arapuca")
    try:
        arapuca_path = utils.resolve_binary(arapuca_bin)
    except RuntimeError as e:
        raise RuntimeError(f"arapuca not found: {e}. Install arapuca or use --no-sandbox") from e

    memory = config.get_config_value(cfg, "sandbox.memory_mb", 2048)
    cpus = config.get_config_value(cfg, "sandbox.cpus", 200)
    pids = config.get_config_value(cfg, "sandbox.pids", 256)
    timeout = config.get_config_value(cfg, "sandbox.timeout", 0)
    inference_port = config.get_config_value(cfg, "inference.port", 8081)

    cmd: list = [arapuca_path, "run"]

    if workdir:
        cmd += ["-v", f"{workdir}:rw"]

    # Mount config_dir separately when it is not already covered by the workdir mount
    if config_dir:
        config_dir_abs = os.path.abspath(config_dir)
        workdir_abs = os.path.abspath(workdir) if workdir else None
        if not workdir_abs or not config_dir_abs.startswith(workdir_abs + os.sep):
            cmd += ["-v", f"{config_dir_abs}:rw"]

    if platform.system() == "Linux":
        cmd += ["--allow-host", f"127.0.0.1:{inference_port}"]
        if wtmcp_port is not None:
            cmd += ["--allow-host", f"127.0.0.1:{wtmcp_port}"]
        cmd += ["--deny-network"]
    else:
        cmd += ["--seccomp", "baseline"]

    cmd += ["--memory", str(memory), "--cpus", str(cpus), "--pids", str(pids)]

    if workdir:
        cmd += ["--cwd", workdir]

    term = os.environ.get("TERM")
    if term:
        cmd += ["--env", f"TERM={term}"]
    colorterm = os.environ.get("COLORTERM")
    if colorterm:
        cmd += ["--env", f"COLORTERM={colorterm}"]

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
) -> None:
    """Start opencode agent."""
    if use_sandbox:
        config_dir = os.path.join(workdir or os.getcwd(), ".aitool-session")
    else:
        config_dir = os.path.expanduser("~/.local/state/aitool/opencode")
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
        sandbox_prefix = _build_sandbox_cmd(cfg, workdir, config_dir, wtmcp_port)
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
) -> None:
    """Start crush agent."""
    if use_sandbox:
        config_dir = os.path.join(workdir or os.getcwd(), ".aitool-session")
    else:
        config_dir = os.path.expanduser("~/.local/state/aitool")
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
        sandbox_prefix = _build_sandbox_cmd(cfg, workdir, config_dir, wtmcp_port)
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
            config_dir = os.path.join(workdir or os.getcwd(), ".aitool-session")
        else:
            config_dir = os.path.expanduser("~/.local/state/aitool")
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
            os.path.join(workdir or os.getcwd(), ".aitool-session") if mcp_config else None
        )
        sandbox_prefix = _build_sandbox_cmd(cfg, workdir, sandbox_config_dir, wtmcp_port)
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

    Raises:
        RuntimeError: If not in TTY or config invalid
    """
    # Check TTY
    if not sys.stdin.isatty():
        utils.error("agent requires a TTY (interactive terminal)", 1)
        sys.exit(1)

    # Load config
    cfg = config.load_config()
    config.validate_config(cfg)

    # Override with CLI args
    if agent_name:
        cfg["agent"]["name"] = agent_name
    if model:
        cfg["inference"]["model"] = model

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
            engine.cmd_engine_start()
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
            _start_agent_opencode(agent_path, cfg, wtmcp_port, use_sandbox, workdir)
        elif agent_name == "crush":
            _start_agent_crush(agent_path, cfg, wtmcp_port, use_sandbox, workdir)
        elif agent_name == "claude":
            _start_agent_claude(agent_path, cfg, wtmcp_port, use_sandbox, workdir)
    finally:
        # Cleanup services unless user requested to keep them
        from aitool import wtmcp

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
