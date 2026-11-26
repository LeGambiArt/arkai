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
    assert [item["name"] for item in agent_session.list_sessions(agents=["pi"])] == ["two"]
    assert sorted(
        item["name"]
        for item in agent_session.list_sessions(
            agents=["pi", "opencode"], working_directory=str(first_dir)
        )
    ) == ["one", "two"]
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
    reset_args = parser.parse_args(["agent", "start", "-S"])
    assert reset_args.reset_session is True
    assert reset_args.session_name is None
    sandbox_args = parser.parse_args(["agent", "start", "-s", "dev"])
    assert sandbox_args.sandbox == "dev"
    assert sandbox_args.session_name is None

    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "start", "--list-session", "--session"])
    with pytest.raises(SystemExit):
        parser.parse_args(["agent", "start", "-S", "--session", "topic"])


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
    saved["agents_used"] = ["pi"]
    agent_session.save_session(saved)

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
    """A non-shared agent override cannot resume another agent's saved session."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("crush", "topic", str(workdir), {})
    saved["agents_used"] = ["crush"]
    agent_session.save_session(saved)

    with pytest.raises(RuntimeError, match="belongs to agent crush"):
        agent.cmd_agent(agent_name="codex", session_name=saved["id"])


def test_shared_agent_starts_fresh_when_only_a_non_shared_session_exists(
    tmp_path: Path, monkeypatch
) -> None:
    """A shared agent ignores another agent's records instead of resuming them."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("crush", "topic", str(workdir), {})
    saved["agents_used"] = ["crush"]
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(agent_name="opencode", session_name="topic")

    assert dispatch.call_args.kwargs["resume_session"] is False
    assert dispatch.call_args.kwargs["session"]["id"] != saved["id"]
    assert dispatch.call_args.kwargs["session"]["agent"] == "opencode"


def _fake_agent_context():
    @contextmanager
    def fake_context(**kwargs):
        yield agent.AgentContext(agent_name=kwargs["agent_name"], workdir=kwargs["sandbox_cwd"])

    return fake_context


def test_auto_resume_picks_newest_shared_session_for_directory(tmp_path: Path, monkeypatch) -> None:
    """A plain start resumes the directory's most recent shared-session record."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    older = agent_session.create_session("opencode", "older", str(workdir), {})
    older["agents_used"] = ["opencode"]
    agent_session.save_session(older)
    newest = agent_session.create_session("pi", "newer", str(workdir), {})
    newest["agents_used"] = ["pi"]
    newest["updated_at"] = "2026-02-01T00:00:00+00:00"
    agent_session.save_session(newest)

    with (
        patch.object(agent, "_select_agent_name", return_value="pi"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent()

    assert dispatch.call_args.kwargs["session"]["id"] == newest["id"]
    assert dispatch.call_args.kwargs["resume_session"] is True


def test_auto_resume_ignores_non_shared_sessions(tmp_path: Path, monkeypatch) -> None:
    """Records for agents outside the shared set never block a fresh start."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("crush", "work", str(workdir), {})

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent()

    assert dispatch.call_args.kwargs["resume_session"] is False
    assert dispatch.call_args.kwargs["session"]["id"] != saved["id"]


def test_auto_resume_creates_session_when_store_is_empty(tmp_path: Path, monkeypatch) -> None:
    """With no saved directory session a plain start creates a fresh record."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent()

    assert dispatch.call_args.kwargs["resume_session"] is False
    assert dispatch.call_args.kwargs["session"]["name"] is None


def test_auto_resume_reuses_record_with_requested_shared_agent(tmp_path: Path, monkeypatch) -> None:
    """A shared record is resumed by another shared agent without native resume."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("opencode", "shared", str(workdir), {})
    saved["agents_used"] = ["opencode"]
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(agent_name="pi")

    assert dispatch.call_args.kwargs["session"]["id"] == saved["id"]
    assert dispatch.call_args.kwargs["resume_session"] is False
    reloaded = agent_session.load_session(saved["id"])
    assert reloaded is not None
    assert reloaded["agent"] == "pi"
    assert reloaded["agents_used"] == ["opencode", "pi"]


def test_shared_agent_reuses_named_session_of_another_shared_agent(
    tmp_path: Path, monkeypatch
) -> None:
    """A named resume crosses shared agents, launching the requested agent."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("opencode", "topic", str(workdir), {})
    saved["agents_used"] = ["opencode"]
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()) as context,
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(agent_name="pi", session_name="topic")

    assert dispatch.call_args.kwargs["session"]["id"] == saved["id"]
    assert dispatch.call_args.kwargs["resume_session"] is False
    assert context.call_args.kwargs["agent_name"] == "pi"


def test_legacy_record_without_agents_used_resumes_natively(tmp_path: Path, monkeypatch) -> None:
    """Records saved before sharing resume natively with their recorded agent."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("opencode", "legacy", str(workdir), {})
    saved.pop("agents_used")
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(session_name="legacy")

    assert dispatch.call_args.kwargs["session"]["id"] == saved["id"]
    assert dispatch.call_args.kwargs["resume_session"] is True


def test_find_latest_session_returns_newest_matching_filter(tmp_path: Path, monkeypatch) -> None:
    """find_latest_session honors recency, agent sets, directories, and emptiness."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    other_dir = tmp_path / "other"
    workdir.mkdir()
    other_dir.mkdir()
    older = agent_session.create_session("opencode", "older", str(workdir), {})
    agent_session.create_session("crush", "crushed", str(workdir), {})
    newest = agent_session.create_session("pi", "newest", str(workdir), {})
    agent_session.create_session("pi", "elsewhere", str(other_dir), {})
    # Force deterministic ordering; create_session stamps identical timestamps.
    older["updated_at"] = "2026-01-01T00:00:00+00:00"
    agent_session.save_session(older)
    newest["updated_at"] = "2026-02-01T00:00:00+00:00"
    agent_session.save_session(newest)

    latest = agent_session.find_latest_session(
        list(agent_session.SHARED_SESSION_AGENTS), str(workdir)
    )
    assert latest is not None
    assert latest["id"] == newest["id"]
    only_opencode = agent_session.find_latest_session(["opencode"], str(workdir))
    assert only_opencode is not None
    assert only_opencode["id"] == older["id"]
    assert agent_session.find_latest_session(["crush"], str(other_dir)) is None
    assert agent_session.find_latest_session(None, str(workdir)) is not None


def test_first_run_of_new_record_records_agent_usage(tmp_path: Path, monkeypatch) -> None:
    """A brand-new record ends paused with exactly its launching agent in use."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent()

    session = dispatch.call_args.kwargs["session"]
    assert session["agents_used"] == ["opencode"]
    reloaded = agent_session.load_session(session["id"])
    assert reloaded is not None
    assert reloaded["agents_used"] == ["opencode"]
    assert reloaded["status"] == "paused"


def test_reset_session_deletes_directory_session_and_starts_fresh(
    tmp_path: Path, monkeypatch
) -> None:
    """-S deletes the directory's current session record and creates a new one."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("opencode", "work", str(workdir), {})
    saved["agents_used"] = ["opencode"]
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(reset_session=True)

    assert dispatch.call_args.kwargs["resume_session"] is False
    new_session = dispatch.call_args.kwargs["session"]
    assert new_session["id"] != saved["id"]
    assert agent_session.load_session(saved["id"]) is None
    assert not agent_session.get_session_dir(saved["id"]).exists()
    assert agent_session.load_session(new_session["id"]) is not None
    assert new_session["agents_used"] == ["opencode"]


def test_reset_session_without_directory_session_starts_fresh(tmp_path: Path, monkeypatch) -> None:
    """-S with no directory session just starts a new one."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(reset_session=True)

    assert dispatch.call_args.kwargs["resume_session"] is False
    assert dispatch.call_args.kwargs["session"]["id"] is not None
    assert len(agent_session.list_sessions()) == 1


def test_reset_session_only_targets_the_directory_current_session(
    tmp_path: Path, monkeypatch
) -> None:
    """-S deletes the newest record only; older directory sessions survive."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    older = agent_session.create_session("opencode", "older", str(workdir), {})
    older["agents_used"] = ["opencode"]
    older["updated_at"] = "2026-01-01T00:00:00+00:00"
    agent_session.save_session(older)
    newest = agent_session.create_session("pi", "newest", str(workdir), {})
    newest["agents_used"] = ["pi"]
    newest["updated_at"] = "2026-02-01T00:00:00+00:00"
    agent_session.save_session(newest)

    with (
        patch.object(agent, "_select_agent_name", return_value="opencode"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(reset_session=True)

    assert agent_session.load_session(newest["id"]) is None
    assert agent_session.load_session(older["id"]) is not None
    assert dispatch.call_args.kwargs["session"]["id"] not in {older["id"], newest["id"]}


def test_reset_session_ignores_non_shared_agent_directory_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    """-S with a non-shared agent keeps today's fresh start and deletes nothing."""
    sessions_dir = tmp_path / "agent-sessions"
    monkeypatch.setattr(agent_session, "get_sessions_dir", lambda: sessions_dir)
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(agent.sys.stdin, "isatty", lambda: True)
    saved = agent_session.create_session("crush", "work", str(workdir), {})
    saved["agents_used"] = ["crush"]
    agent_session.save_session(saved)

    with (
        patch.object(agent, "_select_agent_name", return_value="crush"),
        patch.object(agent, "_agent_context", side_effect=_fake_agent_context()),
        patch.object(agent, "_dispatch_agent") as dispatch,
    ):
        agent.cmd_agent(reset_session=True)

    assert dispatch.call_args.kwargs["resume_session"] is False
    assert agent_session.load_session(saved["id"]) is not None
