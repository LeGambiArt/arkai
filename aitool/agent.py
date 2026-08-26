"""Agent and prompt execution."""

import json
import os
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


def _start_agent_opencode(agent_path: str, cfg: dict, wtmcp_port: Optional[int]) -> None:
    """Start opencode agent."""
    # Generate unique config file for this execution
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

    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = config_file

    cmd = [agent_path]

    print("Starting opencode...", file=sys.stderr)
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
        # Clean up temp config file
        try:
            os.remove(config_file)
        except FileNotFoundError:
            pass
    # Don't exit - let the caller decide what to do
    # Infrastructure services stay running


def _start_agent_crush(agent_path: str, cfg: dict, wtmcp_port: Optional[int]) -> None:
    """Start crush agent."""
    # Generate unique config file for this execution
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

    env = os.environ.copy()
    env["CRUSH_CONFIG"] = config_file

    cmd = [agent_path, "--cwd", config_dir]

    print("Starting crush...", file=sys.stderr)
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
        # Clean up temp config file
        try:
            os.remove(config_file)
        except FileNotFoundError:
            pass
    # Don't exit - let the caller decide what to do
    # Infrastructure services stay running


def _start_agent_claude(agent_path: str, cfg: dict, wtmcp_port: Optional[int]) -> None:
    """Start claude agent."""
    inference_port = config.get_config_value(cfg, "inference.port", 8081)
    model_name = _get_model_name(cfg)

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{inference_port}"
    env["ANTHROPIC_AUTH_TOKEN"] = "local"
    env["ANTHROPIC_API_KEY"] = "local"
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"

    cmd = [agent_path, "--model", model_name]
    mcp_config: Optional[str] = None

    if wtmcp_port is not None:
        # Generate unique MCP config file for this execution
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

        cmd.extend(["--mcp-config", mcp_config])

    print("Starting claude...", file=sys.stderr)
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
) -> None:
    """Start interactive agent session (requires TTY).

    Args:
        agent_name: Override agent from config
        model: Override model from config
        keep_inference: Keep inference server running after exit
        keep_mcp: Keep wtmcp server running after exit
        no_mcp: Skip wtmcp initialization regardless of config

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
            _start_agent_opencode(agent_path, cfg, wtmcp_port)
        elif agent_name == "crush":
            _start_agent_crush(agent_path, cfg, wtmcp_port)
        elif agent_name == "claude":
            _start_agent_claude(agent_path, cfg, wtmcp_port)
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
