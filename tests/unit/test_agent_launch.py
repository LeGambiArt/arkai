"""Tests for agent launch specification construction and execution."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import agent, agent_claude, agent_crush, agent_opencode


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


def test_run_launch_spec_owns_popen_and_cleanup(tmp_path: Path) -> None:
    """The shared runner starts the process and removes module-created files."""
    temporary = tmp_path / "agent.json"
    temporary.write_text("{}")
    process = MagicMock()
    process.communicate.return_value = (b"answer\n", None)
    spec = agent.AgentLaunchSpec(["/agent"], {"PATH": "/bin"}, [temporary])

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


def test_other_agents_only_build_launch_specs() -> None:
    """Each agent module exposes configuration and command construction separately."""
    assert hasattr(agent_claude, "build_launch_spec")
    assert hasattr(agent_crush, "build_launch_spec")
