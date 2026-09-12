"""Tests for opencode TUI configuration."""

import json
from unittest.mock import MagicMock, patch

from arkai import agent, agent_opencode


def test_prepare_opencode_tui_copies_global_config(tmp_path, monkeypatch) -> None:
    """A global tui.json is copied when no project theme is configured."""
    config_home = tmp_path / "config"
    global_tui = config_home / "opencode" / "tui.json"
    global_tui.parent.mkdir(parents=True)
    global_tui.write_text('{"$schema":"custom","theme":"catppuccin","keybinds":{}}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    target = agent_opencode._prepare_tui_config(str(tmp_path / "sessions"), None)

    assert target is not None
    assert json.loads((tmp_path / "sessions" / "tui.json").read_text()) == {
        "$schema": "custom",
        "theme": "catppuccin",
        "keybinds": {},
    }


def test_prepare_opencode_tui_overrides_global_theme(tmp_path, monkeypatch) -> None:
    """agent.theme overrides the theme from the copied global configuration."""
    config_home = tmp_path / "config"
    global_tui = config_home / "opencode" / "tui.json"
    global_tui.parent.mkdir(parents=True)
    global_tui.write_text('{"theme":"catppuccin","keybinds":{}}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    agent_opencode._prepare_tui_config(str(tmp_path / "sessions"), "orng")

    assert json.loads((tmp_path / "sessions" / "tui.json").read_text()) == {
        "theme": "orng",
        "keybinds": {},
    }


def test_prepare_opencode_tui_creates_config_for_theme(tmp_path, monkeypatch) -> None:
    """A configured theme creates tui.json when no global file exists."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    agent_opencode._prepare_tui_config(str(tmp_path / "sessions"), "orng")

    assert json.loads((tmp_path / "sessions" / "tui.json").read_text()) == {
        "$schema": "https://opencode.ai/tui.json",
        "theme": "orng",
    }


def test_prepare_opencode_tui_does_nothing_without_source_or_theme(tmp_path, monkeypatch) -> None:
    """No TUI file is created when neither global nor agent configuration exists."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    target = agent_opencode._prepare_tui_config(str(tmp_path / "sessions"), None)

    assert target is None
    assert not (tmp_path / "sessions" / "tui.json").exists()


def test_start_opencode_passes_tui_config_environment_variable(tmp_path, monkeypatch) -> None:
    """A configured theme is passed to opencode through OPENCODE_TUI_CONFIG."""
    config_home = tmp_path / "config"
    global_tui = config_home / "opencode" / "tui.json"
    global_tui.parent.mkdir(parents=True)
    global_tui.write_text('{"theme":"catppuccin"}')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda path: str(sessions_dir))
    process = MagicMock()

    with patch.object(agent.subprocess, "Popen", return_value=process) as popen:
        agent_opencode.start(
            "/agent",
            {"agent": {"theme": "orng"}, "inference": {"model": "model.gguf"}},
            None,
            False,
            None,
        )

    launch_env = popen.call_args.kwargs["env"]
    assert launch_env["OPENCODE_TUI_CONFIG"] == str(sessions_dir / "tui.json")
    assert list(sessions_dir.glob("opencode-*.json")) == []
    assert not (sessions_dir / "tui.json").exists()


def test_start_opencode_removes_config_when_creation_fails(tmp_path, monkeypatch) -> None:
    """A partially-created config is removed when config generation fails."""
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(agent_opencode.os.path, "expanduser", lambda path: str(sessions_dir))

    def fail_model_lookup(cfg: dict) -> str:
        raise RuntimeError("model lookup failed")

    monkeypatch.setattr(agent, "_get_model_name", fail_model_lookup)

    try:
        agent_opencode.start(
            "/agent",
            {"inference": {"model": "model.gguf"}},
            None,
            False,
            None,
        )
    except RuntimeError as error:
        assert str(error) == "model lookup failed"
    else:
        raise AssertionError("expected model lookup failure")

    assert list(sessions_dir.glob("opencode-*.json")) == []
