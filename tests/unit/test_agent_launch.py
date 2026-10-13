"""Tests for agent launch specification construction and execution."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import agent, agent_claude, agent_crush, agent_omp, agent_opencode, agent_pi


def _cfg() -> dict:
    """Return a minimal configuration shared by launch-spec tests."""
    return {"inference": {"model": "model.gguf", "port": 8123}}


def test_opencode_builds_configuration_without_starting_process(
    tmp_path: Path, monkeypatch
) -> None:
    """OpenCode configuration is created before the generic process runner is called."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))

    with patch.object(agent.subprocess, "Popen") as popen:
        spec = agent_opencode.build_launch_spec("/agent", _cfg(), 8124, False, None)

    popen.assert_not_called()
    config_path = Path(spec.environment["OPENCODE_CONFIG"])
    assert json.loads(config_path.read_text())["model"] == "local-llm/model"
    config_path.unlink()


def test_opencode_uses_mlx_default_model_alias(tmp_path: Path, monkeypatch) -> None:
    """OpenCode sends MLX-LM's mapped default model ID instead of a HF short name."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))
    cfg = {"inference": {"backend": "mlx", "model": "hf:mlx-community/Qwen3-4B-4bit"}}

    spec = agent_opencode.build_launch_spec("/agent", cfg, None, False, None)

    config_path = Path(spec.environment["OPENCODE_CONFIG"])
    config_data = json.loads(config_path.read_text())
    assert config_data["model"] == "local-llm/default_model"
    assert "default_model" in config_data["provider"]["local-llm"]["models"]
    assert config_data["provider"]["local-llm"]["models"]["default_model"]["name"] == (
        "Qwen3-4B-4bit"
    )
    config_path.unlink()


def test_run_launch_spec_owns_popen_and_cleanup(tmp_path: Path) -> None:
    """The shared runner starts the process and removes module-created files."""
    temporary_dir = tmp_path / "session"
    temporary_dir.mkdir()
    temporary = temporary_dir / "agent.json"
    temporary.write_text("{}")
    process = MagicMock()
    process.communicate.return_value = (b"answer\n", None)
    process.returncode = 0
    spec = agent.AgentLaunchSpec(["/agent"], {"PATH": "/bin"}, [temporary, temporary_dir])

    with patch.object(agent.subprocess, "Popen", return_value=process) as popen:
        result = agent.run_launch_spec(spec, capture_stdout=True)

    assert result == "answer\n"
    popen.assert_called_once_with(
        ["/agent"],
        stdin=agent.sys.stdin,
        stdout=agent.subprocess.PIPE,
        stderr=None,
        env={"PATH": "/bin"},
    )
    assert not temporary.exists()
    assert not temporary_dir.exists()


def test_run_launch_spec_reports_nonzero_exit_and_cleans_up(tmp_path: Path) -> None:
    """A failed agent launch is reported after its temporary files are removed."""
    temporary = tmp_path / "agent.json"
    temporary.write_text("{}")
    process = MagicMock()
    process.wait.return_value = 127
    spec = agent.AgentLaunchSpec(["/agent"], {}, [temporary])

    with patch.object(agent.subprocess, "Popen", return_value=process):
        with pytest.raises(RuntimeError, match="exited with status 127"):
            agent.run_launch_spec(spec)

    assert not temporary.exists()


def test_run_launch_spec_cleans_up_when_popen_fails(tmp_path: Path) -> None:
    """Temporary configuration is removed when the executable cannot be started."""
    temporary = tmp_path / "agent.json"
    temporary.write_text("{}")
    spec = agent.AgentLaunchSpec(["/missing-agent"], {}, [temporary])

    with patch.object(agent.subprocess, "Popen", side_effect=FileNotFoundError("missing")):
        with pytest.raises(SystemExit) as error:
            agent.run_launch_spec(spec)

    assert error.value.code == 3
    assert not temporary.exists()


def test_opencode_interactive_sandbox_requests_tty(tmp_path: Path, monkeypatch) -> None:
    """Interactive OpenCode sessions allocate a terminal inside the sandbox."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))

    with patch.object(
        agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--tty", "--"]
    ) as build_sandbox:
        spec = agent_opencode.build_launch_spec("/agent", _cfg(), None, True, None)

    assert spec.command[:4] == ["arapuca", "run", "--tty", "--"]
    assert build_sandbox.call_args.kwargs["tty"] is True
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("prompt", "expected_tty"),
    [(None, True), ("do the task", False)],
)
def test_crush_sandbox_requests_tty_only_for_interactive_sessions(
    tmp_path: Path, monkeypatch, prompt: str | None, expected_tty: bool
) -> None:
    """Interactive Crush sessions allocate a TTY; prompt runs remain non-interactive."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    with patch.object(
        agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]
    ) as build_sandbox:
        spec = agent_crush.build_launch_spec("/agent", _cfg(), None, True, None, prompt=prompt)

    assert build_sandbox.call_args.kwargs["tty"] is expected_tty
    assert any(arg.startswith("CRUSH_GLOBAL_CONFIG=") for arg in spec.command)
    config_env = next(arg for arg in spec.command if arg.startswith("CRUSH_GLOBAL_CONFIG="))
    config_dir = Path(config_env.split("=", 1)[1])
    assert f"CRUSH_GLOBAL_DATA={config_dir}" in spec.command
    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_crush_selects_and_registers_local_inference(tmp_path: Path, monkeypatch) -> None:
    """Crush config and data roots both use Arkai's isolated local config."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    spec = agent_crush.build_launch_spec("/agent", _cfg(), None, False, None)

    crush_config_dir = Path(spec.environment["CRUSH_GLOBAL_CONFIG"])
    config_data = json.loads((crush_config_dir / "crush.json").read_text())
    provider = config_data["providers"]["local-llm"]
    assert provider["type"] == "openai-compat"
    assert provider["base_url"] == "http://127.0.0.1:8123/v1"
    assert provider["models"] == [{"id": "model", "name": "Arkai Local Inference"}]
    assert config_data["models"] == {
        "large": {"model": "model", "provider": "local-llm"},
        "small": {"model": "model", "provider": "local-llm"},
    }
    assert "CRUSH_CONFIG" not in spec.environment
    assert spec.environment["CRUSH_GLOBAL_DATA"] == str(crush_config_dir)
    assert spec.command == ["/agent"]

    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_crush_prompt_runs_quietly(tmp_path: Path, monkeypatch) -> None:
    """Non-interactive Crush runs suppress their status output."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    spec = agent_crush.build_launch_spec("/agent", _cfg(), None, False, None, prompt="do the task")

    assert spec.command == ["/agent", "run", "--quiet", "do the task"]
    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_omp_builds_local_model_and_mcp_configuration(tmp_path: Path, monkeypatch) -> None:
    """OMP uses Arkai's local inference endpoint and an isolated MCP config."""
    agent_dir = tmp_path / "omp-agent"
    omp_path = tmp_path / "bin" / "omp"
    monkeypatch.setattr(agent_omp, "get_agent_dir", lambda: agent_dir)
    cfg = {
        "inference": {
            "backend": "llama-cpp",
            "model": "/models/test.gguf",
            "port": 8123,
            "context_size": 32768,
        }
    }

    spec = agent_omp.build_launch_spec(str(omp_path), cfg, 8124, False, None)

    models = agent.utils.load_yaml(str(agent_dir / "models.yml"))
    provider = models["providers"]["local-llm"]
    assert provider["baseUrl"] == "http://127.0.0.1:8123/v1"
    assert provider["api"] == "openai-completions"
    assert provider["models"][0]["id"] == "test"
    assert provider["models"][0]["contextWindow"] == 32768
    settings = agent.utils.load_yaml(str(agent_dir / "config.yml"))
    assert settings["modelRoles"]["default"] == "local-llm/test"
    assert settings["startup"] == {"quiet": True, "setupWizard": False}
    mcp = json.loads((agent_dir / "mcp.json").read_text())
    assert mcp["mcpServers"]["wtmcp"] == {
        "type": "http",
        "url": "http://127.0.0.1:8124/mcp",
    }
    assert spec.environment["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert spec.environment["PI_CONFIG_DIR"] == agent_omp.OMP_CONFIG_DIR
    assert spec.environment["OMP_SKIP_SETUP"] == "1"
    assert spec.command == [
        str(omp_path.resolve()),
        "--profile",
        "default",
        "--model",
        "local-llm/test",
    ]


def test_omp_prompt_runs_without_session_and_cleans_stale_mcp(tmp_path: Path, monkeypatch) -> None:
    """OMP prompt mode is one-shot text output and removes stale MCP configuration."""
    agent_dir = tmp_path / "omp-agent"
    omp_path = tmp_path / "bin" / "omp"
    agent_dir.mkdir()
    (agent_dir / "mcp.json").write_text("{}")
    monkeypatch.setattr(agent_omp, "get_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "model.gguf", "port": 8123}}

    spec = agent_omp.build_launch_spec(str(omp_path), cfg, None, False, None, prompt="do the task")

    assert spec.command == [
        str(omp_path.resolve()),
        "--profile",
        "default",
        "--model",
        "local-llm/model",
        "-p",
        "--no-session",
        "--mode",
        "text",
        "do the task",
    ]
    assert not (agent_dir / "mcp.json").exists()


def test_omp_sandbox_mounts_binary_and_sets_agent_dir(tmp_path: Path, monkeypatch) -> None:
    """Sandboxed OMP receives its isolated config path and read-only executable mount."""
    agent_dir = tmp_path / "omp-agent"
    monkeypatch.setattr(agent_omp, "get_agent_dir", lambda: agent_dir)
    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_omp.build_launch_spec("/opt/omp/bin/omp", _cfg(), None, True, None)

    assert "/opt/omp/bin:ro" in build.call_args.args[5]
    assert spec.command[:8] == [
        "arapuca",
        "run",
        "--",
        "env",
        f"PI_CONFIG_DIR={agent_omp.OMP_CONFIG_DIR}",
        f"PI_CODING_AGENT_DIR={agent_dir}",
        "OMP_SKIP_SETUP=1",
        "/opt/omp/bin/omp",
    ]


def test_pi_cloud_uses_global_auth_and_openai_provider(tmp_path: Path, monkeypatch) -> None:
    """Pi cloud launch reads global credentials without replacing global models."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    auth_file = agent_dir / "auth.json"
    models_file = agent_dir / "models.json"
    settings_file = agent_dir / "settings.json"
    auth_file.write_text('{"openai":{"type":"api_key","key":"stored"}}')
    models_file.write_text('{"providers":{"custom":{}}}')
    settings_file.write_text('{"defaultProvider":"openai","defaultModel":"gpt-model"}')
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: agent_dir)

    with patch("arkai.wtmcp.is_wtmcp_running", return_value=False):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), None, False, str(tmp_path), cloud_services=["OPENAI"]
        )

    assert spec.command == ["/bin/pi", "--models", "openai/*"]
    assert spec.environment["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert auth_file.read_text() == '{"openai":{"type":"api_key","key":"stored"}}'
    assert models_file.read_text() == '{"providers":{"custom":{}}}'
    assert settings_file.read_text() == '{"defaultProvider":"openai","defaultModel":"gpt-model"}'


def test_pi_cloud_sandbox_mounts_global_agent_data_and_allows_openai_hosts(
    tmp_path: Path, monkeypatch
) -> None:
    """Pi cloud sandbox mounts its global auth directory and narrowly allows OpenAI."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    monkeypatch.setattr(agent_pi.platform, "system", lambda: "Linux")
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: agent_dir)
    monkeypatch.setattr(agent_pi, "_resolve_node_path", lambda: "/runtime/bin/node")
    monkeypatch.setattr(agent_pi, "get_runtime_volumes", lambda *_: [])
    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build,
    ):
        spec = agent_pi.build_launch_spec(
            "/runtime/bin/pi",
            _cfg(),
            None,
            True,
            str(tmp_path / "project"),
            cloud_services=["openai"],
        )

    assert build.call_args.args[2] == str(agent_dir)
    assert build.call_args.kwargs["allow_network"] is False
    assert build.call_args.kwargs["allow_inference"] is False
    assert build.call_args.kwargs["extra_hosts"] == [
        "*.openai.com:443",
        "chatgpt.com:443",
        "*.chatgpt.com:443",
    ]
    assert spec.command[-2:] == ["--models", "openai/*"]
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_omp_cloud_uses_global_auth_and_does_not_overwrite_local_settings(
    tmp_path: Path, monkeypatch
) -> None:
    """OMP cloud launch preserves the user's model/auth files and selects OpenAI."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    config_file = agent_dir / "config.yml"
    models_file = agent_dir / "models.yml"
    auth_file = agent_dir / "agent.db"
    config_file.write_text("modelRoles:\n  default: openai/gpt-model\n")
    models_file.write_text("providers: {}\n")
    auth_file.write_bytes(b"saved auth data")
    monkeypatch.setattr(agent_omp, "get_global_agent_dir", lambda: agent_dir)
    monkeypatch.delenv("PI_CONFIG_DIR", raising=False)

    spec = agent_omp.build_launch_spec(
        "/bin/omp", _cfg(), None, False, str(tmp_path), cloud_services=["OPENAI"]
    )

    assert spec.environment["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert "PI_CONFIG_DIR" not in spec.environment
    assert "--provider" in spec.command
    assert "openai" in spec.command
    assert "--model" not in spec.command
    assert config_file.read_text() == "modelRoles:\n  default: openai/gpt-model\n"
    assert models_file.read_text() == "providers: {}\n"
    assert auth_file.read_bytes() == b"saved auth data"


def test_omp_cloud_sandbox_mounts_global_agent_dir_and_wtmcp_config(
    tmp_path: Path, monkeypatch
) -> None:
    """OMP cloud sandbox mounts global data and creates only a temporary MCP overlay."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.setattr(agent_omp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(agent_omp, "get_global_agent_dir", lambda: agent_dir)
    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_omp.build_launch_spec(
            "/opt/omp/bin/omp", _cfg(), 8123, True, str(workdir), cloud_services=["openai"]
        )

    assert build.call_args.args[2] == str(agent_dir)
    assert build.call_args.kwargs["allow_inference"] is False
    assert build.call_args.kwargs["extra_hosts"] == [
        "*.openai.com:443",
        "chatgpt.com:443",
        "*.chatgpt.com:443",
    ]
    mcp_overlay = workdir / ".omp" / "mcp.json"
    assert json.loads(mcp_overlay.read_text())["mcpServers"]["wtmcp"]["url"] == (
        "http://127.0.0.1:8123/mcp"
    )
    assert mcp_overlay in spec.cleanup_paths
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_other_agents_only_build_launch_specs() -> None:
    """Each agent module exposes configuration and command construction separately."""
    assert hasattr(agent_claude, "build_launch_spec")
    assert hasattr(agent_crush, "build_launch_spec")
    assert hasattr(agent_omp, "build_launch_spec")
