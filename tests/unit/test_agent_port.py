"""Tests for agent-specific inference server ports."""

from unittest.mock import patch

from arkai import agent, inference


def test_agent_start_passes_explicit_port_to_inference() -> None:
    """An explicit agent port starts a separate inference instance."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": False},
        "inference": {"model": "model.gguf", "port": 9090},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=False) as running,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.inference, "cmd_inference_stop"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context(model="other.gguf", port=9090):
            pass

    running.assert_any_call(9090)
    start.assert_called_once_with(model="other.gguf", port=9090)


def test_inference_paths_are_unique_for_explicit_ports() -> None:
    """Explicit inference ports must not share PID or state files."""
    assert inference.get_inference_pid_path() != inference.get_inference_pid_path(9090)
    assert inference.get_inference_state_path() != inference.get_inference_state_path(9090)


def test_configured_port_without_cli_override_uses_default_paths() -> None:
    """A configured port still uses the default instance bookkeeping paths."""
    assert inference.get_inference_pid_path() == inference.get_inference_pid_path(None)
    assert inference.get_inference_state_path() == inference.get_inference_state_path(None)


def test_agent_model_name_uses_huggingface_model_prefix() -> None:
    """The model name is derived from the repository in model: hf:<repo>."""
    assert agent._get_model_name({"inference": {"model": "hf:org/model-GGUF"}}) == "model"


def test_agent_model_name_preserves_model_details() -> None:
    """Local model names retain version, parameter-size, and quantization details."""
    assert (
        agent._get_model_name({"inference": {"model": "Qwen3.8-27B-Q6_K.gguf"}})
        == "Qwen3.8-27B-Q6_K"
    )


def test_agent_context_uses_model_reported_by_inference_server() -> None:
    """Agent launch configuration uses the model actually served by the engine."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": False},
        "inference": {"model": "configured.gguf", "port": 9090},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=True),
        patch.object(
            agent.inference, "get_inference_model", return_value="/models/active.gguf"
        ) as get_model,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context():
            assert cfg["inference"]["model"] == "/models/active.gguf"

    get_model.assert_called_once_with(9090)


def test_agent_cli_registers_port_for_start_and_prompt() -> None:
    """Both agent subcommands accept the inference port option."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    start_args = parser.parse_args(["agent", "start", "--port", "9090"])
    prompt_args = parser.parse_args(["agent", "prompt", "--port", "9091", "hello"])

    assert start_args.port == 9090
    assert prompt_args.port == 9091
    assert not hasattr(start_args, "theme")
    assert not hasattr(prompt_args, "theme")


def test_pi_agent_dispatches_to_pi_launcher() -> None:
    """Pi agent sessions are dispatched to the Pi launcher."""
    ctx = agent.AgentContext(agent_name="pi", agent_path="/bin/pi")
    with patch.object(agent, "_start_agent_pi", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    launcher.assert_called_once()
