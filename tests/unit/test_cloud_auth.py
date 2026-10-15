"""Tests for OpenAI credentials used by cloud agent launches."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from arkai import cloud_auth


def test_vertex_sandbox_settings_mounts_explicit_adc_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    """Vertex sandbox settings pass Google variables and mount ADC credentials read-only."""
    credentials = tmp_path / "service-account.json"
    credentials.write_text("{}")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))

    volumes, environment = cloud_auth.vertex_sandbox_settings(["/extra:ro"], {})

    assert volumes == ["/extra:ro", f"{credentials}:ro"]
    assert environment["GOOGLE_CLOUD_PROJECT"] == "test-project"
    assert environment["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials)


def test_vertex_sandbox_settings_preserves_cli_auth_overrides(tmp_path: Path, monkeypatch) -> None:
    """Explicit sandbox environment values take precedence over the host environment."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "host-project")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(cloud_auth.Path, "home", lambda: tmp_path)

    volumes, environment = cloud_auth.vertex_sandbox_settings(
        environment={"GOOGLE_CLOUD_PROJECT": "sandbox-project"}
    )

    assert volumes == []
    assert environment["GOOGLE_CLOUD_PROJECT"] == "sandbox-project"


def test_crush_vertex_sandbox_settings_maps_google_variables_and_mounts_adc(
    tmp_path: Path, monkeypatch
) -> None:
    credentials = tmp_path / "adc.json"
    credentials.write_text("{}")
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

    volumes, environment = cloud_auth.crush_vertex_sandbox_settings()

    assert volumes == [f"{credentials}:ro"]
    assert environment["VERTEXAI_PROJECT"] == "test-project"
    assert environment["VERTEXAI_LOCATION"] == "us-central1"
    assert environment["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials)


def test_crush_reads_saved_openai_key(tmp_path: Path, monkeypatch) -> None:
    config_home = tmp_path / "config"
    config_file = config_home / "crush" / "crush.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"providers": {"openai": {"api_key": "saved"}}}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        key = cloud_auth.ensure_openai_api_key("crush", tmp_path / "session")

    assert key == "saved"
    prompt.assert_not_called()


def test_crush_saves_openai_key_in_global_config(tmp_path: Path, monkeypatch) -> None:
    config_home = tmp_path / "config"
    config_file = config_home / "crush" / "crush.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(json.dumps({"options": {"debug": True}}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("OPENAI_API_KEY", "crush-secret")

    key = cloud_auth.ensure_openai_api_key("crush", tmp_path / "session")

    assert key == "crush-secret"
    saved = json.loads(config_file.read_text())
    assert saved["options"] == {"debug": True}
    assert saved["providers"]["openai"] == {
        "id": "openai",
        "name": "OpenAI",
        "type": "openai",
        "api_key": "crush-secret",
    }
    assert config_file.stat().st_mode & 0o777 == 0o600


def test_environment_key_is_persisted_for_pi_sandbox_runs(tmp_path: Path, monkeypatch) -> None:
    """A host environment key is imported into Pi's global auth instead of passed through."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        key = cloud_auth.ensure_openai_api_key("pi", tmp_path / "pi-agent")

    assert key == "from-environment"
    prompt.assert_not_called()
    assert json.loads((tmp_path / "pi-agent" / "auth.json").read_text()) == {
        "openai": {"type": "api_key", "key": "from-environment"}
    }


def test_missing_key_requires_secure_interactive_prompt(tmp_path: Path, monkeypatch) -> None:
    """Do not fall back to an echoed input prompt when no controlling TTY exists."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with (
        patch.object(cloud_auth.getpass, "getpass", side_effect=cloud_auth.getpass.GetPassWarning),
        pytest.raises(RuntimeError, match="interactive terminal is required"),
    ):
        cloud_auth.ensure_openai_api_key("pi", tmp_path / "pi-agent")


def test_pi_global_auth_skips_prompt(tmp_path: Path, monkeypatch) -> None:
    """A Pi API key in its global auth file is reused as-is."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    auth_path = agent_dir / "auth.json"
    auth_path.write_text(json.dumps({"openai": {"type": "api_key", "key": "saved"}}))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        key = cloud_auth.ensure_openai_api_key("pi", agent_dir)

    assert key == "saved"
    prompt.assert_not_called()


def test_pi_prompt_saves_key_in_global_auth(tmp_path: Path, monkeypatch) -> None:
    """A prompted Pi key is saved to auth.json with private file permissions."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass", return_value="  secret  ") as prompt:
        key = cloud_auth.ensure_openai_api_key("pi", agent_dir)

    auth_path = agent_dir / "auth.json"
    assert prompt.call_count == 1
    assert key == "secret"
    assert json.loads(auth_path.read_text()) == {"openai": {"type": "api_key", "key": "secret"}}
    assert auth_path.stat().st_mode & 0o777 == 0o600
    with patch.object(cloud_auth.getpass, "getpass") as second_prompt:
        assert cloud_auth.ensure_openai_api_key("pi", agent_dir) == "secret"
    second_prompt.assert_not_called()


def test_omp_existing_database_key_skips_prompt(tmp_path: Path, monkeypatch) -> None:
    """OMP's active API key credential in agent.db is detected."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    with sqlite3.connect(agent_dir / "agent.db") as connection:
        connection.execute(
            "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY, provider TEXT, "
            "credential_type TEXT, data TEXT, disabled_cause TEXT)"
        )
        connection.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data) VALUES (?, ?, ?)",
            ("openai", "api_key", json.dumps({"key": "stored-in-omp"})),
        )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        key = cloud_auth.ensure_openai_api_key("omp", agent_dir)

    assert key == "stored-in-omp"
    prompt.assert_not_called()


def test_omp_prompt_persists_native_model_config(tmp_path: Path, monkeypatch) -> None:
    """A prompted key is saved in OMP's global models.yml provider configuration."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass", return_value="omp-secret") as prompt:
        key = cloud_auth.ensure_openai_api_key("omp", agent_dir)

    assert key == "omp-secret"
    assert yaml.safe_load((agent_dir / "models.yml").read_text()) == {
        "providers": {"openai": {"apiKey": "omp-secret"}}
    }
    assert (agent_dir / "models.yml").stat().st_mode & 0o777 == 0o600
    prompt.assert_called_once()
    with patch.object(cloud_auth.getpass, "getpass") as second_prompt:
        assert cloud_auth.ensure_openai_api_key("omp", agent_dir) == "omp-secret"
    second_prompt.assert_not_called()


def test_omp_environment_key_is_persisted_in_global_models_config(
    tmp_path: Path, monkeypatch
) -> None:
    """A host environment key is imported into OMP config rather than forwarded to the sandbox."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        first_key = cloud_auth.ensure_openai_api_key("omp", agent_dir)
        prompt.assert_not_called()
    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        next_key = cloud_auth.ensure_openai_api_key("omp", agent_dir)

    models_path = agent_dir / "models.yml"
    assert first_key == next_key == "environment-secret"
    assert yaml.safe_load(models_path.read_text()) == {
        "providers": {"openai": {"apiKey": "environment-secret"}}
    }
    assert models_path.stat().st_mode & 0o777 == 0o600
    prompt.assert_not_called()


def test_omp_resolved_global_key_is_persisted_to_models_config(tmp_path: Path, monkeypatch) -> None:
    """Keys resolved from OMP dotenv or auth storage remain available in its sandbox."""
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = cloud_auth.subprocess.CompletedProcess(
        args=["omp", "token", "openai"], returncode=0, stdout="resolved-key\n", stderr=""
    )

    with patch.object(cloud_auth.subprocess, "run", return_value=result) as run:
        key = cloud_auth.ensure_openai_api_key("omp", agent_dir, "/bin/omp")

    assert key == "resolved-key"
    run.assert_called_once()
    assert yaml.safe_load((agent_dir / "models.yml").read_text()) == {
        "providers": {"openai": {"apiKey": "resolved-key"}}
    }


def test_opencode_reads_native_auth_file(tmp_path: Path, monkeypatch) -> None:
    """OpenCode cloud launches reuse its saved OpenAI API key."""
    data_home = tmp_path / "data"
    auth_path = data_home / "opencode" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text('{"openai":{"type":"api","key":"saved"}}')
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass") as prompt:
        key = cloud_auth.ensure_openai_api_key("opencode", tmp_path)

    assert key == "saved"
    prompt.assert_not_called()


def test_opencode_saves_prompted_key_to_native_auth_file(tmp_path: Path, monkeypatch) -> None:
    """OpenCode API keys are saved with private permissions in its global auth file."""
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch.object(cloud_auth.getpass, "getpass", return_value="  secret  ") as prompt:
        key = cloud_auth.ensure_openai_api_key("opencode", tmp_path)

    auth_path = data_home / "opencode" / "auth.json"
    assert key == "secret"
    assert json.loads(auth_path.read_text()) == {"openai": {"type": "api", "key": "secret"}}
    assert auth_path.stat().st_mode & 0o777 == 0o600
    prompt.assert_called_once()
