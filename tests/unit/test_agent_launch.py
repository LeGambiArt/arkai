"""Tests for agent launch specification construction and execution."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import (
    agent,
    agent_claude,
    agent_codex,
    agent_crush,
    agent_omp,
    agent_opencode,
    agent_pi,
)


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
    assert spec.command == ["/agent", "--standalone"]
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


def test_opencode_cloud_uses_native_openai_provider_and_api_key(
    tmp_path: Path, monkeypatch
) -> None:
    """OpenCode cloud config leaves provider/model selection to OpenCode itself."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))

    spec = agent_opencode.build_launch_spec(
        "/agent", _cfg(), None, False, None, cloud_services=["OPENAI"], cloud_api_key="secret"
    )

    config_path = Path(spec.environment["OPENCODE_CONFIG"])
    config_data = json.loads(config_path.read_text())
    assert config_data == {"$schema": "https://opencode.ai/config.json"}
    assert spec.environment["OPENAI_API_KEY"] == "secret"
    assert spec.command == ["/agent", "--standalone"]
    config_path.unlink()


def test_opencode_cloud_model_overrides_agent_model(tmp_path: Path, monkeypatch) -> None:
    """A cloud model override is written to OpenCode config, not local inference config."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))

    spec = agent_opencode.build_launch_spec(
        "/agent",
        _cfg(),
        None,
        False,
        None,
        cloud_services=["openai"],
        cloud_model="openai/cloud-only-model",
    )

    config_path = Path(spec.environment["OPENCODE_CONFIG"])
    assert json.loads(config_path.read_text())["model"] == "openai/cloud-only-model"
    config_path.unlink()


def test_opencode_cloud_sandbox_allows_openai_only(tmp_path: Path, monkeypatch) -> None:
    """Cloud sandbox allows OpenAI endpoints but does not grant local inference access."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))
    monkeypatch.setattr(agent_opencode.platform, "system", lambda: "Linux")
    with patch.object(
        agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]
    ) as build_sandbox:
        spec = agent_opencode.build_launch_spec(
            "/agent",
            _cfg(),
            None,
            True,
            str(tmp_path),
            cloud_services=["openai"],
            cloud_api_key="secret",
        )

    assert build_sandbox.call_args.kwargs["allow_network"] is False
    assert build_sandbox.call_args.kwargs["allow_inference"] is False
    assert build_sandbox.call_args.kwargs["extra_hosts"] == [
        "*.openai.com:443",
        "api.openai.com:443",
    ]
    assert build_sandbox.call_args.args[6]["OPENAI_API_KEY"] == "secret"
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_opencode_vertex_sandbox_passes_adc_and_allows_vertex_hosts(
    tmp_path: Path, monkeypatch
) -> None:
    """OpenCode Vertex runs receive ADC settings and Vertex HTTPS access in the sandbox."""
    sessions = tmp_path / "sessions"
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}")
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))
    monkeypatch.setattr(agent_opencode.platform, "system", lambda: "Linux")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_opencode.build_launch_spec(
            "/agent", _cfg(), None, True, str(tmp_path), cloud_services=["vertex"]
        )

    assert build.call_args.kwargs["allow_inference"] is False
    assert build.call_args.kwargs["extra_hosts"] == [
        "*-aiplatform.googleapis.com:443",
        "aiplatform.googleapis.com:443",
        "oauth2.googleapis.com:443",
    ]
    assert build.call_args.args[6]["GOOGLE_CLOUD_PROJECT"] == "test-project"
    assert build.call_args.args[5] == [f"{credentials}:ro"]
    assert spec.environment["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials)
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


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
        stdin=agent.subprocess.DEVNULL,
        stdout=agent.subprocess.PIPE,
        stderr=None,
        env={"PATH": "/bin"},
    )
    assert not temporary.exists()
    assert not temporary_dir.exists()


def test_run_launch_spec_suppresses_selected_stderr_prefixes(capsys) -> None:
    """Selected child warnings are suppressed while other diagnostics are retained."""
    spec = agent.AgentLaunchSpec(
        ["/agent"],
        {"PATH": "/bin"},
        suppress_stderr_prefixes=("MCP: Project servers blocked:",),
    )
    process = MagicMock()
    process.communicate.return_value = (
        b"answer\n",
        b"MCP: Project servers blocked: wtmcp (blocked)\nOther diagnostic\n",
    )
    process.returncode = 0

    with patch.object(agent.subprocess, "Popen", return_value=process) as popen:
        result = agent.run_launch_spec(spec, capture_stdout=True)

    assert result == "answer\n"
    assert capsys.readouterr().err == "Other diagnostic\n"
    assert popen.call_args.kwargs["stderr"] == agent.subprocess.PIPE


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


def test_codex_builds_a_temporary_local_profile_without_touching_user_config(
    tmp_path: Path, monkeypatch
) -> None:
    """Codex local config uses a unique profile and preserves its base config."""
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    base_config = codex_home / "config.toml"
    base_config.write_text('approval_policy = "on-request"\n')
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    spec = agent_codex.build_launch_spec("/bin/codex", _cfg(), 8124, False, None)

    profile_path = spec.cleanup_paths[0]
    profile_name = profile_path.name.removesuffix(".config.toml")
    profile = profile_path.read_text()
    assert profile_name.startswith("arkai-")
    assert 'model = "model"' in profile
    assert 'model_provider = "arkai"' in profile
    assert 'web_search = "disabled"' in profile
    assert 'sandbox_mode = "danger-full-access"' not in profile
    assert 'base_url = "http://127.0.0.1:8123/v1"' in profile
    assert 'wire_api = "responses"' in profile
    assert "requires_openai_auth = false" in profile
    assert "[mcp_servers.wtmcp]" in profile
    assert 'url = "http://127.0.0.1:8124/mcp"' in profile
    assert base_config.read_text() == 'approval_policy = "on-request"\n'
    assert spec.command == ["/bin/codex", "--no-daemon", "--profile", profile_name]
    assert spec.environment["CODEX_HOME"] == str(codex_home.resolve())

    profile_path.unlink()


def test_codex_openai_cloud_uses_codex_openai_provider(tmp_path: Path, monkeypatch) -> None:
    """Codex OpenAI cloud runs select its built-in OpenAI provider and model."""
    codex_home = tmp_path / "codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    spec = agent_codex.build_launch_spec(
        "/bin/codex",
        _cfg(),
        None,
        False,
        None,
        cloud_services=["openai"],
        cloud_model="gpt-5.6-codex",
    )

    profile = spec.cleanup_paths[0].read_text()
    assert 'model = "gpt-5.6-codex"' in profile
    assert 'model_provider = "openai"' in profile
    assert 'openai_base_url = "https://api.openai.com/v1"' in profile
    assert 'model_provider = "arkai"' not in profile
    assert "[model_providers.arkai]" not in profile
    assert "127.0.0.1" not in profile
    assert spec.command[0:3] == ["/bin/codex", "--no-daemon", "--profile"]
    spec.cleanup_paths[0].unlink()


def test_codex_openai_cloud_sandbox_allows_provider_and_passes_api_key(
    tmp_path: Path, monkeypatch
) -> None:
    """Cloud Codex runs pass an available API key and allow OpenAI HTTPS access."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(agent_codex.platform, "system", lambda: "Linux")
    certificate = tmp_path / "ca.pem"
    certificate.write_text("test ca")
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_codex.build_launch_spec(
            "/bin/codex", _cfg(), None, True, str(tmp_path), cloud_services=["openai"]
        )

    assert build.call_args.kwargs["allow_inference"] is False
    assert build.call_args.kwargs["extra_hosts"] == ["*.openai.com:443", "api.openai.com:443"]
    assert build.call_args.args[6]["CODEX_API_KEY"] == "secret"
    assert build.call_args.args[6]["SSL_CERT_FILE"] == str(certificate.resolve())
    assert f"{certificate.resolve()}:ro" in build.call_args.args[5]
    spec.cleanup_paths[0].unlink()


def test_codex_openai_cloud_allows_mac_keychain_in_arakuca(tmp_path: Path, monkeypatch) -> None:
    """macOS Codex cloud profiles may access Keychain certificate/auth data."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(agent_codex.platform, "system", lambda: "Darwin")

    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_codex.build_launch_spec(
            "/bin/codex", _cfg(), None, True, str(tmp_path), cloud_services=["openai"]
        )

    assert build.call_args.kwargs["allow_keychain"] is True
    spec.cleanup_paths[0].unlink()


def test_codex_prompt_uses_exec_with_a_unique_profile(tmp_path: Path, monkeypatch) -> None:
    """Codex prompt mode uses exec while interactive mode leaves the TUI command default."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))

    spec = agent_codex.build_launch_spec(
        "/bin/codex", _cfg(), None, False, None, prompt="summarize this project"
    )

    profile_name = spec.cleanup_paths[0].name.removesuffix(".config.toml")
    assert spec.command == [
        "/bin/codex",
        "--no-daemon",
        "--profile",
        profile_name,
        "exec",
        "summarize this project",
    ]
    spec.cleanup_paths[0].unlink()


def test_codex_prompt_allows_a_non_git_working_directory(tmp_path: Path, monkeypatch) -> None:
    """Non-repository prompt runs opt out of Codex's Git repository startup check."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    workdir = tmp_path / "scratch"
    workdir.mkdir()

    spec = agent_codex.build_launch_spec(
        "/bin/codex", _cfg(), None, False, str(workdir), prompt="answer briefly"
    )

    assert "--skip-git-repo-check" in spec.command
    spec.cleanup_paths[0].unlink()


def test_codex_sandbox_mounts_codex_home_and_allocates_tty(tmp_path: Path, monkeypatch) -> None:
    """Codex's profile and auth home are visible to the sandboxed CLI."""
    codex_home = tmp_path / "codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_codex.build_launch_spec("/bin/codex", _cfg(), 8124, True, str(tmp_path))

    profile_name = spec.cleanup_paths[0].name.removesuffix(".config.toml")
    assert build.call_args.args[2] == str(codex_home.resolve())
    assert build.call_args.kwargs["tty"] is True
    assert build.call_args.args[6]["CODEX_HOME"] == str(codex_home.resolve())
    assert 'sandbox_mode = "danger-full-access"' in spec.cleanup_paths[0].read_text()
    assert spec.command == [
        "arapuca",
        "run",
        "--",
        "/bin/codex",
        "--no-daemon",
        "--profile",
        profile_name,
    ]
    spec.cleanup_paths[0].unlink()


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
    assert build_sandbox.call_args.kwargs["allow_network_listeners"] is True
    assert "OPENCODE_DISABLE_PROJECT_CONFIG=true" in spec.command
    assert spec.command[-1] == "--standalone"
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


def test_crush_openai_cloud_uses_native_provider_and_key(tmp_path: Path, monkeypatch) -> None:
    """Crush cloud runs configure OpenAI without registering local inference."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    spec = agent_crush.build_launch_spec(
        "/agent",
        _cfg(),
        None,
        False,
        None,
        cloud_services=["openai"],
        cloud_api_key="secret",
    )

    crush_config_dir = Path(spec.environment["CRUSH_GLOBAL_CONFIG"])
    config_data = json.loads((crush_config_dir / "crush.json").read_text())
    assert config_data["providers"]["openai"] == {
        "id": "openai",
        "name": "OpenAI",
        "type": "openai",
        "api_key": "$OPENAI_API_KEY",
    }
    assert "local-llm" not in config_data["providers"]
    assert "models" not in config_data
    assert spec.environment["OPENAI_API_KEY"] == "secret"
    assert spec.command == ["/agent"]
    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_crush_cloud_model_selects_large_and_small_models(tmp_path: Path, monkeypatch) -> None:
    """A cloud -m override becomes Crush's large and small model selection."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    spec = agent_crush.build_launch_spec(
        "/agent",
        _cfg(),
        None,
        False,
        None,
        cloud_services=["openai"],
        cloud_api_key="secret",
        cloud_model="openai/gpt-cloud",
    )

    crush_config_dir = Path(spec.environment["CRUSH_GLOBAL_CONFIG"])
    config_data = json.loads((crush_config_dir / "crush.json").read_text())
    assert config_data["models"] == {
        "large": {"model": "gpt-cloud", "provider": "openai"},
        "small": {"model": "gpt-cloud", "provider": "openai"},
    }
    assert config_data["providers"]["openai"]["models"] == [
        {"id": "gpt-cloud", "name": "gpt-cloud"}
    ]
    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_crush_vertex_cloud_passes_adc_and_provider_environment(
    tmp_path: Path, monkeypatch
) -> None:
    """Crush Vertex runs get native provider config, Vertex env aliases, and ADC mount."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}")
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)
    monkeypatch.setattr(agent_crush.platform, "system", lambda: "Linux")
    for name in (
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "GOOGLE_VERTEX_PROJECT",
        "GOOGLE_VERTEX_LOCATION",
        "GCLOUD_PROJECT",
        "VERTEX_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    with patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build:
        spec = agent_crush.build_launch_spec(
            "/agent", _cfg(), None, True, str(tmp_path), cloud_services=["vertex"]
        )

    crush_config_dir = Path(
        next(arg for arg in spec.command if arg.startswith("CRUSH_GLOBAL_CONFIG=")).split("=", 1)[1]
    )
    config_data = json.loads((crush_config_dir / "crush.json").read_text())
    assert config_data["providers"]["vertexai"]["type"] == "google-vertex"
    assert config_data["models"] == {
        "large": {"model": "claude-sonnet-4-5@20250929", "provider": "vertexai"},
        "small": {"model": "claude-haiku-4-5@20251001", "provider": "vertexai"},
    }
    assert {model["id"] for model in config_data["providers"]["vertexai"]["models"]} == {
        "claude-sonnet-4-5@20250929",
        "claude-haiku-4-5@20251001",
    }
    assert build.call_args.args[5] == [f"{credentials}:ro"]
    sandbox_env = build.call_args.args[6]
    assert sandbox_env["VERTEXAI_PROJECT"] == "test-project"
    assert sandbox_env["VERTEXAI_LOCATION"] == "us-central1"
    assert sandbox_env["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials)
    assert build.call_args.kwargs["allow_inference"] is False
    assert build.call_args.kwargs["extra_hosts"] == [
        "*-aiplatform.googleapis.com:443",
        "aiplatform.googleapis.com:443",
        "oauth2.googleapis.com:443",
    ]
    for path in spec.cleanup_paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def test_crush_normalizes_vertex_claude_api_date_model_ids(tmp_path: Path, monkeypatch) -> None:
    """Anthropic-style dated model IDs are converted to Vertex's `@` syntax."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(agent_crush, "_get_config_dir", lambda: sessions)

    spec = agent_crush.build_launch_spec(
        "/agent",
        _cfg(),
        None,
        False,
        None,
        cloud_services=["vertex"],
        cloud_model="vertexai/claude-haiku-4-5-20251001",
    )

    config_data = json.loads(
        (Path(spec.environment["CRUSH_GLOBAL_CONFIG"]) / "crush.json").read_text()
    )
    assert config_data["models"] == {
        "large": {"model": "claude-haiku-4-5@20251001", "provider": "vertexai"},
        "small": {"model": "claude-haiku-4-5@20251001", "provider": "vertexai"},
    }
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


def test_pi_cloud_model_override_is_passed_to_agent(tmp_path: Path, monkeypatch) -> None:
    """Pi receives an explicit provider model without changing the local model config."""
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: tmp_path / "pi-agent")
    with patch("arkai.wtmcp.is_wtmcp_running", return_value=False):
        spec = agent_pi.build_launch_spec(
            "/bin/pi",
            _cfg(),
            None,
            False,
            str(tmp_path),
            cloud_services=["openai"],
            cloud_model="openai/cloud-only-model",
        )

    assert spec.command == [
        "/bin/pi",
        "--models",
        "openai/*",
        "--model",
        "openai/cloud-only-model",
    ]


def test_pi_cloud_mcp_config_allows_project_servers(tmp_path: Path, monkeypatch) -> None:
    """Cloud Pi runs put the project-server policy in their selected MCP config."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: agent_dir)

    with patch("arkai.wtmcp.is_wtmcp_running", return_value=True):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), 8080, False, str(tmp_path), cloud_services=["openai"]
        )

    mcp_config_path = Path(spec.command[spec.command.index("--mcp-config") + 1])
    assert json.loads(mcp_config_path.read_text()) == {
        "settings": {"projectServers": "allow"},
        "mcpServers": {"wtmcp": {"url": "http://127.0.0.1:8080/mcp"}},
    }
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_pi_no_mcp_prompt_omits_config_and_suppresses_project_warning(
    tmp_path: Path, monkeypatch
) -> None:
    """No-MCP prompts neither pass Arkai MCP config nor show Pi's discovery warning."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)

    spec = agent_pi.build_launch_spec("/bin/pi", _cfg(), None, False, str(tmp_path), prompt="hello")

    assert "--mcp-config" not in spec.command
    assert spec.suppress_stderr_prefixes == ("MCP: Project servers blocked:",)
    assert not (agent_dir / "mcp.json").exists()


def test_pi_sandbox_allows_active_wtmcp_port(tmp_path: Path, monkeypatch) -> None:
    """Pi's sandbox configuration receives the active host MCP port."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    node_path = tmp_path / "node"
    node_path.touch()

    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=True),
        patch.object(agent_pi, "_resolve_node_path", return_value=str(node_path)),
        patch.object(agent.utils, "resolve_binary", return_value="/bin/arapuca"),
        patch.object(agent.platform, "system", return_value="Linux"),
    ):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), 8080, True, str(tmp_path), prompt="use MCP"
        )

    allow_hosts = [
        spec.command[index + 1]
        for index, argument in enumerate(spec.command[:-1])
        if argument == "--allow-host"
    ]
    assert allow_hosts == ["127.0.0.1:8123", "127.0.0.1:8080"]
    assert "PI_MCP_CONFIG_MODE=exclusive" in spec.command
    assert "--mcp-config" in spec.command
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_pi_sandbox_no_mcp_does_not_set_exclusive_mcp_mode(tmp_path: Path, monkeypatch) -> None:
    """No-MCP prompt launches leave Pi's normal MCP settings untouched."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    node_path = tmp_path / "node"
    node_path.touch()

    with (
        patch.object(agent_pi, "_resolve_node_path", return_value=str(node_path)),
        patch.object(
            agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]
        ) as build_sandbox,
    ):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), None, True, str(tmp_path), prompt="hello"
        )

    assert "PI_MCP_CONFIG_MODE" not in build_sandbox.call_args.args[6]
    assert "--mcp-config" not in spec.command
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_pi_vertex_cloud_registers_anthropic_catalog_and_scopes_picker(
    tmp_path: Path, monkeypatch
) -> None:
    """Pi Vertex launch exposes Anthropic model families through its native adapter."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: agent_dir)
    package_json = tmp_path / "pi-package" / "package.json"
    package_json.parent.mkdir()
    package_json.write_text('{"name":"@earendil-works/pi-coding-agent"}')
    monkeypatch.setattr(agent_pi, "_find_pi_package_json", lambda _: str(package_json))
    models_dev = {
        "google-vertex-anthropic": {
            "models": {
                "claude-haiku-4-5@20251001": {
                    "id": "claude-haiku-4-5@20251001",
                    "name": "Claude Haiku 4.5",
                    "reasoning": True,
                    "modalities": {"input": ["text", "image"]},
                    "limit": {"context": 200000, "output": 64000},
                    "cost": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25},
                },
                "claude-sonnet-5@default": {
                    "id": "claude-sonnet-5@default",
                    "name": "Claude Sonnet 5",
                    "reasoning": True,
                    "modalities": {"input": ["text"]},
                    "limit": {"context": 1000000, "output": 128000},
                    "cost": {"input": 2, "output": 10, "cache_read": 0.2, "cache_write": 2.5},
                },
            }
        }
    }
    response = MagicMock()
    response.json.return_value = models_dev
    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch.object(agent_pi.requests, "get", return_value=response),
    ):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), None, False, str(tmp_path), cloud_services=["vertex"]
        )

    assert spec.command == [
        "/bin/pi",
        "--extension",
        str(agent_dir / ".arkai" / "google-vertex-anthropic.js"),
        "--models",
        "google-vertex/*,google-vertex-anthropic/*",
    ]
    assert spec.environment["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert spec.environment["ARKAI_PI_PACKAGE_JSON"] == str(package_json)
    models = json.loads((agent_dir / "models.json").read_text())
    vertex_models = models["providers"]["google-vertex-anthropic"]["models"]
    assert [model["id"] for model in vertex_models] == [
        "claude-haiku-4-5@20251001",
        "claude-sonnet-5@default",
    ]
    assert all(model["api"] == "anthropic-messages" for model in vertex_models)
    assert (agent_dir / ".arkai" / "google-vertex-anthropic.js").is_file()


def test_pi_vertex_catalog_merge_preserves_existing_models(tmp_path: Path) -> None:
    """Refreshing the Vertex catalog preserves existing Pi model configuration."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    existing = {
        "$schema": "custom-schema",
        "providers": {
            "custom": {"models": [{"id": "custom-model"}]},
            "google-vertex-anthropic": {
                "api": "anthropic-messages",
                "models": [{"id": "existing-vertex-anthropic-model"}],
            },
        },
        "other": "keep-me",
    }
    models_path = agent_dir / "models.json"
    models_path.write_text(json.dumps(existing))

    catalog_models = [
        {"id": "claude-opus-5@default", "name": "Claude Opus 5", "api": "anthropic-messages"},
        {"id": "claude-sonnet-5@default", "name": "Claude Sonnet 5", "api": "anthropic-messages"},
    ]
    agent_pi._merge_vertex_anthropic_models(agent_dir, catalog_models)
    agent_pi._merge_vertex_anthropic_models(agent_dir, catalog_models)

    updated = json.loads(models_path.read_text())
    assert updated["$schema"] == "custom-schema"
    assert updated["other"] == "keep-me"
    assert updated["providers"]["custom"] == existing["providers"]["custom"]
    vertex_models = updated["providers"]["google-vertex-anthropic"]["models"]
    assert [model["id"] for model in vertex_models] == [
        "existing-vertex-anthropic-model",
        "claude-opus-5@default",
        "claude-sonnet-5@default",
    ]


def test_pi_vertex_sandbox_mounts_provider_runtime(tmp_path: Path, monkeypatch) -> None:
    """The sandbox mounts Pi's provider dependencies and Vertex ADC for extension calls."""
    agent_dir = tmp_path / "pi-agent"
    package_json = tmp_path / "pi-package" / "package.json"
    package_json.parent.mkdir()
    package_json.write_text('{"name":"@earendil-works/pi-coding-agent"}')
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}")
    monkeypatch.setattr(agent_pi, "get_global_agent_dir", lambda: agent_dir)
    monkeypatch.setattr(agent_pi, "_find_pi_package_json", lambda _: str(package_json))
    monkeypatch.setattr(agent_pi, "_resolve_node_path", lambda: "/runtime/bin/node")
    monkeypatch.setattr(agent_pi, "get_runtime_volumes", lambda *_: [])
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east5")
    response = MagicMock()
    response.json.return_value = {
        "google-vertex-anthropic": {
            "models": {
                "claude-haiku-4-5@20251001": {
                    "id": "claude-haiku-4-5@20251001",
                    "name": "Claude Haiku 4.5",
                    "reasoning": True,
                    "modalities": {"input": ["text"]},
                    "limit": {"context": 200000, "output": 64000},
                    "cost": {"input": 1, "output": 5},
                }
            }
        }
    }
    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch.object(agent_pi.requests, "get", return_value=response),
        patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]) as build,
    ):
        spec = agent_pi.build_launch_spec(
            "/bin/pi", _cfg(), None, True, str(tmp_path), cloud_services=["vertex"]
        )

    assert f"{package_json.parent}:ro" in build.call_args.args[5]
    assert f"{credentials}:ro" in build.call_args.args[5]
    assert build.call_args.args[6]["ARKAI_PI_PACKAGE_JSON"] == str(package_json)
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


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


def test_opencode_prompt_uses_private_server(tmp_path: Path, monkeypatch) -> None:
    """Prompt runs use a per-invocation private OpenCode server."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda _: str(sessions))

    spec = agent_opencode.build_launch_spec("/agent", _cfg(), None, False, None, prompt="hello")

    assert spec.command == ["/agent", "run", "--standalone", "hello"]
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


def test_omp_vertex_cloud_selects_google_vertex_provider(tmp_path: Path, monkeypatch) -> None:
    """OMP Vertex launches select OMP's native Google Vertex provider."""
    monkeypatch.setattr(agent_omp, "get_global_agent_dir", lambda: tmp_path / "omp-agent")

    spec = agent_omp.build_launch_spec(
        "/bin/omp", _cfg(), None, False, str(tmp_path), cloud_services=["vertex"]
    )

    assert spec.command[spec.command.index("--provider") + 1] == "google-vertex"
    assert "--model" not in spec.command


def test_omp_cloud_model_override_is_passed_to_agent(tmp_path: Path, monkeypatch) -> None:
    """OMP receives the requested cloud model alongside its native provider selection."""
    monkeypatch.setattr(agent_omp, "get_global_agent_dir", lambda: tmp_path / "omp-agent")

    spec = agent_omp.build_launch_spec(
        "/bin/omp",
        _cfg(),
        None,
        False,
        str(tmp_path),
        cloud_services=["vertex"],
        cloud_model="google-vertex/gemini-cloud-only",
    )

    assert spec.command[spec.command.index("--provider") + 1] == "google-vertex"
    assert spec.command[spec.command.index("--model") + 1] == "google-vertex/gemini-cloud-only"


def test_sandbox_mounts_read_write_paths_without_rw_suffix(tmp_path: Path, monkeypatch) -> None:
    """Arapuca defaults volume mounts to read-write; :rw breaks older CLIs."""
    workdir = str(tmp_path / "project")
    config_dir = str(tmp_path / "sessions")
    monkeypatch.setattr(agent.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(agent.utils, "resolve_binary", lambda _: "/arapuca")

    command = agent._build_sandbox_cmd(
        {
            "inference": {"port": 8123},
            "sandbox": {"volume": ["/extra:ro"]},
        },
        workdir,
        config_dir,
        None,
        allow_network_listeners=True,
    )

    mounts = [command[index + 1] for index, arg in enumerate(command[:-1]) if arg == "-v"]
    assert mounts == [workdir, config_dir, "/extra:ro"]
    assert f"{workdir}:rw" not in command
    assert f"{config_dir}:rw" not in command
    assert "--allow-network-listeners" in command


def test_other_agents_only_build_launch_specs() -> None:
    """Each agent module exposes configuration and command construction separately."""
    assert hasattr(agent_claude, "build_launch_spec")
    assert hasattr(agent_crush, "build_launch_spec")
    assert hasattr(agent_omp, "build_launch_spec")
