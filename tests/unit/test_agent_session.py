"""Tests for Arkai-managed interactive agent sessions."""

import argparse
import stat
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from arkai import agent, agent_session


def test_create_and_find_session_persists_private_metadata(tmp_path: Path, monkeypatch) -> None:
    """Session records and native storage are private and survive reloads."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()

    session = agent_session.create_session(
        "opencode",
        "feature work",
        str(workdir),
        {"model": "local-model", "environment_keys": ["TOKEN"]},
    )

    session_dir = sessions_dir / session["id"]
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((session_dir / "session.json").stat().st_mode) == 0o600
    assert (session_dir / "native").is_dir()
    assert agent_session.load_session(session["id"]) == session
    assert agent_session.find_session("feature work", working_directory=str(workdir)) == session
    assert agent_session.find_session(session["id"]) == session


def test_list_sessions_filters_by_agent_and_working_directory(tmp_path: Path, monkeypatch) -> None:
    """Session listing supports the filters used by --list-session."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    agent_session.create_session("opencode", "one", str(first_dir), {})
    agent_session.create_session("pi", "two", str(first_dir), {})
    agent_session.create_session("opencode", "three", str(second_dir), {})

    assert len(agent_session.list_sessions()) == 3
    assert [item["name"] for item in agent_session.list_sessions(agent="pi")] == ["two"]
    assert sorted(
        item["name"] for item in agent_session.list_sessions(working_directory=str(first_dir))
    ) == ["one", "two"]


def test_find_session_rejects_duplicate_project_names(tmp_path: Path, monkeypatch) -> None:
    """A duplicated session name is never resolved arbitrarily."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    agent_session.create_session("opencode", "same", str(workdir), {})
    agent_session.create_session("pi", "same", str(workdir), {})

    with pytest.raises(RuntimeError, match="ambiguous"):
        agent_session.find_session("same", working_directory=str(workdir))


def test_session_lookup_rejects_non_id_path_references(tmp_path: Path, monkeypatch) -> None:
    """Names cannot be interpreted as paths outside Arkai's session root."""
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: tmp_path / "agent-sessions")
    assert agent_session.load_session("../../outside") is None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    agent.ingest_cli_options(parser.add_subparsers(dest="command"))
    return parser


def test_agent_start_parses_session_modes_without_reassigning_sandbox_short_flag() -> None:
    """Session flags coexist with the existing -s/--sandbox option."""
    parser = _parser()

    assert parser.parse_args(["agent", "start", "--list-session"]).list_session is True
    assert parser.parse_args(["agent", "start", "--session"]).session_name == ""
    assert parser.parse_args(["agent", "start", "--session", "topic"]).session_name == "topic"
    sandbox_args = parser.parse_args(["agent", "start", "-s", "dev"])
    assert sandbox_args.sandbox == "dev"
    assert sandbox_args.session_name is None

    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "start", "--list-session", "--session"])


def test_list_session_dispatches_without_starting_agent(monkeypatch) -> None:
    """The listing option exits before TTY checks and infrastructure setup."""
    args = _parser().parse_args(["agent", "start", "--list-session", "--agent", "pi"])
    with (
        patch.object(agent, "cmd_agent_list_sessions") as list_sessions,
        patch.object(agent, "cmd_agent") as start_agent,
    ):
        agent.exec_cmd(args)

    list_sessions.assert_called_once_with("pi", None)
    start_agent.assert_not_called()


def test_named_session_resumes_saved_agent_and_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """A named session restores its original agent, options, and project path."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session(
        "pi",
        "topic",
        str(workdir),
        {
            "model": "saved-model",
            "no_start_inference": True,
            "no_mcp": False,
            "no_sandbox": False,
            "sandbox_profile": "dev",
            "sandbox_volume": ["/tmp/shared:ro"],
            "port": 9090,
            "backend": "mlx",
            "cloud_services": ["openai"],
            "environment_keys": ["TOKEN"],
        },
    )

    @contextmanager
    def fake_context(**kwargs):
        yield agent.AgentContext(agent_name=kwargs["agent_name"], workdir=kwargs["sandbox_cwd"])

    with (
        patch.object(agent, "_agent_context", side_effect=fake_context) as context,
        patch.object(agent, "_dispatch_agent") as dispatch,
        patch.object(agent.utils, "warn") as warn,
    ):
        agent.cmd_agent(session_name="topic")

    context.assert_called_once()
    assert context.call_args.kwargs["agent_name"] == "pi"
    assert context.call_args.kwargs["model"] is None
    assert context.call_args.kwargs["sandbox_cwd"] == str(workdir)
    assert context.call_args.kwargs["port"] is None
    assert context.call_args.kwargs["backend"] is None
    assert context.call_args.kwargs["cloud_services"] == ["openai"]
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["resume_session"] is True
    assert dispatch.call_args.kwargs["session"]["id"] == saved["id"]
    saved_after_resume = agent_session.load_session(saved["id"])
    assert saved_after_resume is not None
    assert saved_after_resume["status"] == "paused"
    warn.assert_called_once()


def test_named_session_creates_new_record_when_name_is_unknown(tmp_path: Path, monkeypatch) -> None:
    """A new name creates a private Arkai session record before launching."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)

    @contextmanager
    def fake_context(**kwargs):
        yield agent.AgentContext(agent_name=kwargs["agent_name"], workdir=kwargs["sandbox_cwd"])

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=fake_context),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(session_name="new-topic")

    session = dispatch.call_args.kwargs["session"]
    assert session["name"] == "new-topic"
    assert session["agent"] == "opencode"
    assert session["status"] == "paused"
    assert dispatch.call_args.kwargs["resume_session"] is False


def test_named_session_rejects_explicit_agent_mismatch(tmp_path: Path, monkeypatch) -> None:
    """An agent override cannot silently start a different agent for a saved name."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    agent_session.create_session("pi", "topic", str(workdir), {})

    with pytest.raises(RuntimeError, match="belongs to agent pi"):
        agent.cmd_agent(agent_name="opencode", session_name="topic")
