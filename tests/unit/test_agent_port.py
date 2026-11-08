"""Tests for agent-specific inference server ports."""

from unittest.mock import patch

from arkai import agent, agent_codex, agent_crush, agent_omp, agent_opencode, agent_pi, inference


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
        patch.object(agent.inference, "is_inference_server_available", return_value=False),
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.inference, "cmd_inference_stop"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context("opencode", model="other.gguf", port=9090):
            pass

    running.assert_any_call(9090)
    start.assert_called_once_with(model="other.gguf", port=9090)


def test_agent_start_passes_quantized_model_to_inference() -> None:
    """Agent startup preserves the provider and quantization model reference."""
    model = "hf:empero-ai/Qwen3.8-35B-A3B-Distill-GGUF:Q4_K_M"
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": False},
        "inference": {"model": model},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=False),
        patch.object(agent.inference, "is_inference_server_available", return_value=False),
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.inference, "cmd_inference_stop"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context("opencode", model=model):
            pass

    start.assert_called_once_with(model=model)


def test_agent_context_reuses_healthy_untracked_inference_server() -> None:
    """A responsive inference server is reused even without Arkai's PID file."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": False},
        "inference": {"model": "saved-model.gguf", "port": 8081},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=False),
        patch.object(
            agent.inference, "is_inference_server_available", return_value=True
        ) as available,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model", return_value="active-model"),
        patch.object(agent.inference, "cmd_inference_stop") as stop,
        patch.object(agent, "_sync_running_inference_backend"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context("opencode", model="old-session-model") as context:
            assert context.cfg["inference"]["model"] == "active-model"

    available.assert_called_once_with(8081)
    start.assert_not_called()
    stop.assert_not_called()


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


def test_agent_model_id_preserves_running_server_id() -> None:
    """Server model IDs with dotted versions are not mistaken for file paths."""
    cfg = {
        "inference": {
            "backend": "splash",
            "model": "incoai/Qwen3.8-27B-Splash",
        }
    }

    assert agent._get_agent_model_id(cfg) == "incoai/Qwen3.8-27B-Splash"


def test_mlx_agent_uses_default_model_request_alias() -> None:
    """MLX-LM maps the default_model request ID to its configured model path."""
    cfg = {
        "inference": {
            "backend": "mlx",
            "model": "hf:mlx-community/Qwen3-4B-4bit",
        }
    }

    assert agent._get_agent_model_id(cfg) == "default_model"


def test_splash_agent_preserves_server_model_id() -> None:
    """Splash requires the full repository ID returned by its model endpoint."""
    assert (
        agent._get_agent_model_id(
            {"inference": {"backend": "splash", "model": "incoai/Qwen3.8-27B-Splash"}}
        )
        == "incoai/Qwen3.8-27B-Splash"
    )
    assert (
        agent._get_agent_model_id(
            {"inference": {"backend": "splash", "model": "hf:incoai/Qwen3.8-27B-Splash:Q4"}}
        )
        == "incoai/Qwen3.8-27B-Splash"
    )


def test_llama_cpp_agent_keeps_human_readable_model_request_id() -> None:
    """llama.cpp continues receiving Arkai's existing model request ID."""
    cfg = {"inference": {"model": "hf:org/model-GGUF"}}

    assert agent._get_agent_model_id(cfg) == "model"


def test_agent_uses_backend_from_running_inference_state(tmp_path, monkeypatch) -> None:
    """A running MLX instance overrides a stale backend in project configuration."""
    state_path = tmp_path / "inference.state"
    monkeypatch.setattr(agent.inference, "get_inference_state_path", lambda port: str(state_path))
    monkeypatch.setattr(agent.utils, "load_yaml", lambda path: {"backend": "mlx"})
    cfg = {"inference": {"backend": "llama-cpp"}}

    agent._sync_running_inference_backend(cfg)

    assert cfg["inference"]["backend"] == "mlx"


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
        with agent._agent_context("opencode"):
            assert cfg["inference"]["model"] == "/models/active.gguf"

    get_model.assert_called_once_with(9090)


def test_agent_context_continues_without_mcp_when_wtmcp_is_missing() -> None:
    """A missing optional wtmcp binary warns but does not block agent startup."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": True},
        "inference": {"model": "model.gguf", "port": 9090},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=True),
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch(
            "arkai.wtmcp.cmd_wtmcp_start",
            side_effect=RuntimeError("wtmcp binary not found. Configure wtmcp.path"),
        ),
        patch.object(agent.utils, "warn") as warn,
    ):
        with agent._agent_context("opencode") as context:
            assert context.use_mcp is False
            assert context.wtmcp_port is None

    warn.assert_called_once_with("wtmcp binary not found; starting agent without MCP.")


def test_agent_prompt_runs_without_mcp_when_wtmcp_is_missing(capsys) -> None:
    """Prompt mode also warns and dispatches normally when wtmcp is unavailable."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": True},
        "inference": {"model": "model.gguf", "port": 9090},
        "sandbox": {"disable": True},
    }

    def dispatch(context, prompt, capture_stdout):
        assert context.use_mcp is False
        assert context.wtmcp_port is None
        assert prompt == "hello"
        assert capture_stdout is True
        return "answer"

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=True),
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch(
            "arkai.wtmcp.cmd_wtmcp_start",
            side_effect=RuntimeError("wtmcp binary not found. Configure wtmcp.path"),
        ),
        patch.object(agent.utils, "warn") as warn,
        patch.object(agent, "_dispatch_agent", side_effect=dispatch),
    ):
        agent.cmd_agent_prompt(["hello"])

    warn.assert_called_once_with("wtmcp binary not found; starting agent without MCP.")
    assert "answer" in capsys.readouterr().out


def test_cloud_agent_skips_local_inference_engine() -> None:
    """An OpenAI Pi session does not query, start, or stop local inference."""
    cfg = {
        "agent": {"name": "pi", "path": "/agent", "mcp": False},
        "inference": {"port": 8081},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True) as validate,
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model") as get_model,
        patch.object(agent.inference, "cmd_inference_stop") as stop,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context(
            "pi", model="openai/cloud-only-model", cloud_services=["OPENAI"]
        ) as context:
            assert context.cloud_services == ["openai"]
            assert context.cloud_model == "openai/cloud-only-model"
            assert "model" not in cfg["inference"]

    validate.assert_called_once_with(cfg, require_model=False, warn_missing_model=False)
    running.assert_not_called()
    start.assert_not_called()
    get_model.assert_not_called()
    stop.assert_not_called()


def test_cloud_agent_skips_local_inference_for_opencode() -> None:
    """OpenCode cloud sessions do not query, start, or stop local inference."""
    cfg = {
        "agent": {
            "name": "opencode",
            "path": "/agent",
            "mcp": False,
            "mode": "cloud",
            "cloud": {"default": "custom"},
        },
        "inference": {"port": 8081},
        "sandbox": {"disable": True},
    }

    with patch.object(
        agent.config,
        "load_config",
        return_value=cfg,
    ):
        with (
            patch.object(agent.config, "validate_config", return_value=True),
            patch.object(agent.inference, "is_inference_running") as running,
            patch.object(agent.inference, "cmd_inference_start") as start,
            patch.object(agent.inference, "get_inference_model") as get_model,
            patch.object(agent.inference, "cmd_inference_stop") as stop,
            patch.object(agent.utils, "resolve_binary", return_value="/agent"),
        ):
            with agent._agent_context("opencode", cloud_services=["openai"]) as context:
                assert context.cloud_services == ["openai"]

    running.assert_not_called()
    start.assert_not_called()
    get_model.assert_not_called()
    stop.assert_not_called()


def test_cloud_agent_skips_local_inference_for_crush() -> None:
    """Crush cloud sessions do not require the local inference service."""
    cfg = {
        "agent": {"name": "crush", "path": "/agent", "mcp": False},
        "inference": {"port": 8081},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True) as validate,
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model") as get_model,
        patch.object(agent.inference, "cmd_inference_stop") as stop,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context(
            "crush", model="openai/gpt-cloud", cloud_services=["OPENAI"]
        ) as context:
            assert context.cloud_services == ["openai"]
            assert context.cloud_model == "openai/gpt-cloud"
            assert "model" not in cfg["inference"]

    validate.assert_called_once_with(cfg, require_model=False, warn_missing_model=False)
    running.assert_not_called()
    start.assert_not_called()
    get_model.assert_not_called()
    stop.assert_not_called()


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


def test_agent_cli_registers_backend_for_start_and_prompt() -> None:
    """Agent start and prompt commands accept an inference backend override."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    start_args = parser.parse_args(["agent", "start", "--backend", "mlx"])
    prompt_args = parser.parse_args(["agent", "prompt", "--backend", "mlx", "hello"])
    assert start_args.backend == "mlx"
    assert prompt_args.backend == "mlx"


def test_agent_cli_accepts_multiple_case_insensitive_cloud_services() -> None:
    """The start command normalizes cloud names and accepts multiple occurrences."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    args = parser.parse_args(["agent", "start", "--cloud", "OpenAI", "--cloud", "OPENAI"])

    assert args.cloud == ["openai", "openai"]


def test_agent_cli_accepts_vertex_cloud_provider() -> None:
    """The start command accepts Vertex as a cloud provider."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    args = parser.parse_args(["agent", "start", "--cloud", "Vertex"])

    assert args.cloud == ["vertex"]


def test_agent_cli_accepts_custom_cloud_provider() -> None:
    """The start and prompt commands accept the generic custom provider."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    start_args = parser.parse_args(["agent", "start", "--cloud", "Custom"])
    prompt_args = parser.parse_args(["agent", "prompt", "--cloud", "custom", "hello"])

    assert start_args.cloud == ["custom"]
    assert prompt_args.cloud == ["custom"]
    assert prompt_args.prompt_text == ["hello"]


def test_agent_prompt_cli_accepts_codex_openai_cloud() -> None:
    """Codex prompt mode accepts the supported OpenAI cloud provider."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    args = parser.parse_args(
        ["agent", "prompt", "--agent", "codex", "--cloud", "OpenAI", "Do the task"]
    )

    assert args.cloud == ["openai"]
    assert args.prompt_text == ["Do the task"]


def test_agent_cli_rejects_unknown_cloud_service() -> None:
    """Only supported cloud providers are accepted by the CLI parser."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    try:
        parser.parse_args(["agent", "start", "--cloud", "anthropic"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Expected an unknown cloud provider to be rejected")


def test_cloud_agent_rejects_multiple_distinct_providers() -> None:
    """A launch can select only one provider for its agent session."""
    cfg = {
        "agent": {"name": "pi", "path": "/agent", "mcp": False},
        "inference": {},
        "sandbox": {"disable": True},
    }
    with patch.object(agent.config, "load_config", return_value=cfg):
        try:
            with agent._agent_context("pi", cloud_services=["openai", "vertex"]):
                pass
        except RuntimeError as error:
            assert "Only one --cloud provider" in str(error)
        else:
            raise AssertionError("Expected multiple cloud providers to be rejected")


def test_codex_rejects_vertex_cloud_provider() -> None:
    """Codex fails clearly for Vertex because it only accepts Responses providers."""
    cfg = {
        "agent": {"name": "codex", "path": "/codex", "mcp": False},
        "inference": {},
        "sandbox": {"disable": True},
    }
    with patch.object(agent.config, "load_config", return_value=cfg):
        try:
            with agent._agent_context("codex", cloud_services=["vertex"]):
                pass
        except RuntimeError as error:
            assert "Codex does not support --cloud vertex" in str(error)
        else:
            raise AssertionError("Expected Codex to reject Vertex cloud mode")


def test_custom_cloud_config_default_skips_local_inference(
    monkeypatch,
) -> None:
    """The configured custom cloud is selected without starting inference."""
    cfg = {
        "agent": {
            "name": "opencode",
            "path": "/agent",
            "mcp": False,
            "mode": "cloud",
            "cloud": {
                "custom": {
                    "base_url": "https://gateway.example/v1",
                    "api_key": {"source": "environment", "name": "TEAM_MODEL_KEY"},
                    "default_provider": "gateway",
                    "providers": {"gateway": {"default_model": "org/model"}},
                },
            },
        },
        "inference": {},
        "sandbox": {"disable": True},
    }
    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "is_inference_server_available") as available,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        monkeypatch.setenv("TEAM_MODEL_KEY", "context-secret")
        with agent._agent_context("opencode") as context:
            assert context.cloud_model == "gateway/org/model"
            assert context.cloud_services == ["custom"]
            assert context.cloud_config is not None
            assert context.cloud_config["default_provider"] == "gateway"
            assert context.cloud_api_environment == {"TEAM_MODEL_KEY": "context-secret"}
            assert "context-secret" not in repr(context)
    assert context.cloud_api_environment == {}

    running.assert_not_called()
    available.assert_not_called()
    start.assert_not_called()


def test_cloud_mode_requires_a_default_cloud_service() -> None:
    """Cloud mode without a selected cloud service fails with a config hint."""
    cfg = {
        "agent": {"name": "opencode", "mode": "cloud"},
        "inference": {},
        "sandbox": {"disable": True},
    }
    with patch.object(agent.config, "load_config", return_value=cfg):
        try:
            with agent._agent_context("opencode"):
                pass
        except RuntimeError as error:
            assert "agent.cloud.default is not configured" in str(error)
        else:
            raise AssertionError("Expected cloud mode to require a default cloud service")


def test_cloud_mode_requires_default_when_multiple_configurations_exist() -> None:
    """Multiple configured cloud services require an explicit default selection."""
    cfg = {
        "agent": {
            "name": "opencode",
            "mode": "cloud",
            "cloud": {
                "openai": {},
                "custom": {
                    "base_url": "https://gateway.example/v1",
                    "providers": {"gateway": {"default_model": "org/model"}},
                },
            },
        },
        "inference": {},
        "sandbox": {"disable": True},
    }
    with patch.object(agent.config, "load_config", return_value=cfg):
        try:
            with agent._agent_context("opencode"):
                pass
        except RuntimeError as error:
            assert str(error) == "agent.mode is 'cloud' but agent.cloud.default is not configured"
        else:
            raise AssertionError("Expected multiple cloud services to require a default")


def test_cloud_mode_uses_explicit_default_when_multiple_configurations_exist() -> None:
    """An explicit cloud default resolves multiple configured services."""
    cfg = {
        "agent": {
            "name": "opencode",
            "path": "/agent",
            "mcp": False,
            "mode": "cloud",
            "cloud": {"default": "openai", "openai": {}, "custom": {}},
        },
        "inference": {},
        "sandbox": {"disable": True},
    }
    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context("opencode") as context:
            assert context.cloud_services == ["openai"]

    running.assert_not_called()
    start.assert_not_called()


def test_custom_cloud_requires_model_and_supported_agent() -> None:
    """Custom mode reports missing model configuration and unsupported agents."""
    base_config = {
        "agent": {
            "name": "opencode",
            "cloud": {
                "custom": {
                    "base_url": "https://gateway.example/v1",
                    "providers": {"gateway": {}},
                }
            },
        },
        "inference": {},
        "sandbox": {"disable": True},
    }
    with patch.object(agent.config, "load_config", return_value=base_config):
        try:
            with agent._agent_context("opencode", cloud_services=["custom"]):
                pass
        except RuntimeError as error:
            assert "default_model or an explicit --model" in str(error)
        else:
            raise AssertionError("Expected custom cloud to require a selected model")

    base_config["agent"]["cloud"]["custom"]["providers"]["gateway"]["default_model"] = "org/model"
    with patch.object(agent.config, "load_config", return_value=base_config):
        try:
            with agent._agent_context("crush", cloud_services=["custom"]):
                pass
        except RuntimeError as error:
            assert "opencode, pi, and omp" in str(error)
        else:
            raise AssertionError("Expected custom cloud to reject unsupported agents")


def test_custom_provider_alias_selection_is_shared_with_pi() -> None:
    cfg = {
        "agent": {
            "name": "pi",
            "mcp": False,
            "cloud": {
                "custom": {
                    "base_url": "https://gateway.example/v1",
                    "default_provider": "gateway",
                    "providers": {
                        "gateway": {"default_model": "org/model"},
                        "openai": {"models": [{"id": "gpt-model"}]},
                    },
                }
            },
        },
        "inference": {},
        "sandbox": {"disable": True},
    }
    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "is_inference_server_available") as available,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context(
            "pi", model="openai/gpt-model", cloud_services=["custom"]
        ) as context:
            assert context.cloud_model == "openai/gpt-model"
            assert context.cloud_config is not None
            assert set(context.cloud_config["providers"]) == {"gateway", "openai"}

    running.assert_not_called()
    available.assert_not_called()
    start.assert_not_called()


def test_agent_prompt_passes_backend_override_to_context() -> None:
    """The prompt command forwards its backend override to agent setup."""
    context = agent.AgentContext(agent_name="opencode")
    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context") as agent_context,
        patch.object(agent, "_dispatch_agent", return_value="answer"),
    ):
        agent_context.return_value.__enter__.return_value = context
        agent.cmd_agent_prompt(["hello"], backend="mlx")

    assert agent_context.call_args.kwargs["backend"] == "mlx"


def test_codex_cloud_context_skips_local_inference() -> None:
    """Codex OpenAI cloud runs bypass Arkai's local inference server."""
    cfg = {
        "agent": {"name": "codex", "path": "/codex", "mcp": False},
        "inference": {"model": "model.gguf"},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True) as validate,
        patch.object(agent.inference, "is_inference_running") as running,
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model") as get_model,
        patch.object(agent.utils, "resolve_binary", return_value="/codex"),
    ):
        with agent._agent_context(
            "codex", model="gpt-5.6-codex", cloud_services=["openai"]
        ) as context:
            assert context.cloud_services == ["openai"]
            assert context.cloud_model == "gpt-5.6-codex"

    validate.assert_called_once_with(cfg, require_model=False, warn_missing_model=False)
    running.assert_not_called()
    start.assert_not_called()
    get_model.assert_not_called()


def test_agent_prompt_passes_cloud_provider_to_context() -> None:
    """Prompt mode forwards the chosen cloud provider through setup."""
    context = agent.AgentContext(agent_name="codex")
    with (
        patch.object(agent, "_select_agent_name", return_value="codex"),
        patch.object(agent, "_agent_context") as agent_context,
        patch.object(agent, "_dispatch_agent", return_value="answer"),
    ):
        agent_context.return_value.__enter__.return_value = context
        agent.cmd_agent_prompt(["hello"], cloud_services=["openai"])

    assert agent_context.call_args.kwargs["cloud_services"] == ["openai"]


def test_agent_context_passes_backend_override_to_inference() -> None:
    """An agent backend override is applied when its inference server starts."""
    cfg = {
        "agent": {"name": "opencode", "path": "/agent", "mcp": False},
        "inference": {"model": "model.gguf"},
        "sandbox": {"disable": True},
    }

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=False),
        patch.object(agent.inference, "is_inference_server_available", return_value=False),
        patch.object(agent.inference, "cmd_inference_start") as start,
        patch.object(agent.inference, "get_inference_model", return_value="active.gguf"),
        patch.object(agent.inference, "cmd_inference_stop"),
        patch.object(agent.utils, "resolve_binary", return_value="/agent"),
    ):
        with agent._agent_context("opencode", backend="mlx"):
            pass

    start.assert_called_once_with(model=None, backend="mlx")


def test_pi_agent_dispatches_to_pi_launcher() -> None:
    """Pi agent sessions are dispatched to the Pi launcher."""
    ctx = agent.AgentContext(
        agent_name="pi",
        agent_path="/bin/pi",
        cloud_services=["openai"],
        cloud_model="openai/model-v2",
    )
    with patch.object(agent_pi, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_services"] == ["openai"]
    assert launcher.call_args.kwargs["cloud_model"] == "openai/model-v2"


def test_omp_agent_dispatches_to_omp_launcher() -> None:
    """OMP agent sessions are dispatched to the OMP launcher."""
    ctx = agent.AgentContext(
        agent_name="omp",
        agent_path="/bin/omp",
        cloud_services=["openai"],
        cloud_model="openai/model-v2",
    )
    with patch.object(agent_omp, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_services"] == ["openai"]
    assert launcher.call_args.kwargs["cloud_model"] == "openai/model-v2"


def test_opencode_agent_dispatches_cloud_services() -> None:
    """OpenCode receives the selected cloud providers from the shared dispatcher."""
    ctx = agent.AgentContext(
        agent_name="opencode",
        agent_path="/bin/opencode",
        cloud_services=["openai"],
        cloud_model="openai/model-v2",
    )
    with patch.object(agent_opencode, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_services"] == ["openai"]
    assert launcher.call_args.kwargs["cloud_model"] == "openai/model-v2"


def test_custom_cloud_dispatches_resolved_api_environment() -> None:
    """The dispatcher passes host-resolved custom credentials to the launcher."""
    resolved_environment = {"ARKAI_CUSTOM_API_KEY_GATEWAY_12345678": "keychain-secret"}
    ctx = agent.AgentContext(
        agent_name="opencode",
        agent_path="/bin/opencode",
        cloud_services=["custom"],
        cloud_config={"default_provider": "gateway", "providers": {}},
        cloud_api_environment=resolved_environment,
    )
    with patch.object(agent_opencode, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_api_environment"] == resolved_environment


def test_crush_agent_dispatches_cloud_services() -> None:
    """Crush receives selected cloud services and model from the shared dispatcher."""
    ctx = agent.AgentContext(
        agent_name="crush",
        agent_path="/bin/crush",
        cloud_services=["openai"],
        cloud_model="openai/model-v2",
    )
    with patch.object(agent_crush, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_services"] == ["openai"]
    assert launcher.call_args.kwargs["cloud_model"] == "openai/model-v2"


def test_codex_agent_dispatches_openai_cloud_provider() -> None:
    """Codex receives cloud selection and model from the shared dispatcher."""
    ctx = agent.AgentContext(
        agent_name="codex",
        agent_path="/bin/codex",
        cloud_services=["openai"],
        cloud_model="gpt-5.6-codex",
    )
    with patch.object(agent_codex, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.kwargs["cloud_services"] == ["openai"]
    assert launcher.call_args.kwargs["cloud_model"] == "gpt-5.6-codex"


def test_codex_agent_dispatches_to_codex_launcher() -> None:
    """The shared dispatcher sends Codex runs to the profile-aware launcher."""
    ctx = agent.AgentContext(agent_name="codex", agent_path="/bin/codex")
    with patch.object(agent_codex, "start", return_value=None) as launcher:
        agent._dispatch_agent(ctx)

    assert launcher.call_args.args[0] == "/bin/codex"
