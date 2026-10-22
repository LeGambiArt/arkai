"""Persistent Arkai metadata for resumable interactive agent sessions."""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arkai import utils


def get_sessions_dir() -> Path:
    """Return the private root directory for Arkai-managed agent sessions."""
    return Path(utils.get_pid_dir()) / "agent-sessions"


def get_session_dir(session_id: str) -> Path:
    """Return the storage directory for one Arkai session."""
    return get_sessions_dir() / session_id


def _session_path(session_id: str) -> Path:
    return get_session_dir(session_id) / "session.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_session(session: dict[str, Any]) -> None:
    session_dir = get_session_dir(session["id"])
    session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    session_dir.chmod(0o700)
    descriptor, filename = tempfile.mkstemp(prefix="session-", suffix=".json", dir=session_dir)
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as session_file:
            json.dump(session, session_file, indent=2)
            session_file.write("\n")
        os.replace(temporary_path, _session_path(session["id"]))
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def create_session(
    agent: str,
    name: str | None,
    working_directory: str,
    launch_options: dict[str, Any],
) -> dict[str, Any]:
    """Create a new session record and its persistent native-data directory."""
    sessions_dir = get_sessions_dir()
    sessions_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    sessions_dir.chmod(0o700)

    session_id = str(uuid.uuid4())
    now = _now()
    session: dict[str, Any] = {
        "id": session_id,
        "name": name,
        "agent": agent,
        "working_directory": str(Path(working_directory).expanduser().resolve()),
        "created_at": now,
        "updated_at": now,
        "status": "starting",
        "native_session_id": session_id if agent == "claude" else None,
        "launch_options": launch_options,
    }
    _write_session(session)
    native_dir = get_session_dir(session_id) / "native"
    native_dir.mkdir(mode=0o700)
    return session


def save_session(session: dict[str, Any]) -> None:
    """Persist an updated session record."""
    session["updated_at"] = _now()
    _write_session(session)


def load_session(session_id: str) -> dict[str, Any] | None:
    """Load a session by its full Arkai ID."""
    try:
        normalized_id = str(uuid.UUID(session_id))
    except (AttributeError, TypeError, ValueError):
        return None
    try:
        data = json.loads(_session_path(normalized_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read Arkai session {session_id}: {error}") from error
    if not isinstance(data, dict) or data.get("id") != normalized_id:
        raise RuntimeError(f"Invalid Arkai session metadata: {_session_path(normalized_id)}")
    return data


def list_sessions(
    agent: str | None = None,
    working_directory: str | None = None,
) -> list[dict[str, Any]]:
    """List Arkai-managed sessions, newest first."""
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []
    cwd = str(Path(working_directory).expanduser().resolve()) if working_directory else None
    sessions = []
    for path in sessions_dir.glob("*/session.json"):
        session = load_session(path.parent.name)
        if session is None:
            continue
        if agent and session.get("agent") != agent:
            continue
        if cwd and session.get("working_directory") != cwd:
            continue
        sessions.append(session)
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def find_session(
    reference: str,
    agent: str | None = None,
    working_directory: str | None = None,
) -> dict[str, Any] | None:
    """Find a session by full Arkai ID or project-scoped name."""
    direct = load_session(reference)
    if direct is not None:
        return direct
    matches = [
        session
        for session in list_sessions(agent=agent, working_directory=working_directory)
        if session.get("name") == reference
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"Session name {reference!r} is ambiguous; use one of the full IDs shown by "
            "'arkai agent start --list-session'"
        )
    return matches[0] if matches else None
