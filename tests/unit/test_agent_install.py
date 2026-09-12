"""Tests for Pi installation and agent registration."""

import json
from pathlib import Path
from unittest.mock import patch

from arkai import agent, config


def test_pi_is_a_valid_agent() -> None:
    """Pi is accepted by configuration validation."""
    assert "pi" in config.VALID_AGENTS


def test_install_pi_declines_before_running_commands(capsys) -> None:
    """Declining the warning does not invoke npm or Pi."""
    with (
        patch("builtins.input", return_value="n"),
        patch.object(agent.utils, "run_command") as run_command,
    ):
        agent.cmd_agent_install("pi")

    run_command.assert_not_called()
    assert "cancelled" in capsys.readouterr().out.lower()


def test_install_pi_installs_core_and_packages() -> None:
    """The installer installs Pi and its supported package extensions."""
    commands = []

    def capture_command(command: list, **kwargs) -> tuple[int, str, str]:
        commands.append((command, kwargs))
        return 0, "", ""

    core_dir = Path("/arkai-state/pi/core")
    agent_dir = Path("/arkai-state/pi/agent")
    with (
        patch("builtins.input", return_value="yes"),
        patch.object(agent.shutil, "which", side_effect=lambda name: f"/bin/{name}"),
        patch.object(agent.utils, "run_command", side_effect=capture_command),
        patch.object(agent, "_get_pi_core_dir", return_value=core_dir),
        patch.object(agent, "_get_pi_agent_dir", return_value=agent_dir),
    ):
        agent.cmd_agent_install("pi")

    assert commands[0][0] == [
        "/bin/npm",
        "install",
        "--prefix",
        str(core_dir),
        "-g",
        "--ignore-scripts",
        agent.PI_CORE_PACKAGE,
    ]
    assert commands[0][1] == {"timeout": None, "env": None}
    assert [command for command, _ in commands[1:]] == [
        [str(core_dir / "bin" / "pi"), "install", "npm:pi-mcp-adapter"],
        [str(core_dir / "bin" / "pi"), "install", "npm:pi-web-access"],
        [str(core_dir / "bin" / "pi"), "install", "npm:pi-subagents"],
    ]
    for _, kwargs in commands[1:]:
        assert kwargs["env"]["PI_CODING_AGENT_DIR"] == str(agent_dir)


def test_pi_default_binary_is_arkai_local(tmp_path: Path, monkeypatch) -> None:
    """Pi uses the local arkai installation unless agent.path overrides it."""
    core_dir = tmp_path / "pi" / "core"
    monkeypatch.setattr(agent, "_get_pi_core_dir", lambda: core_dir)
    cfg = {"agent": {"name": "pi"}, "inference": {"model": "test.gguf"}}

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=True),
        patch.object(
            agent.utils, "resolve_binary", return_value=str(core_dir / "bin" / "pi")
        ) as resolve,
    ):
        with agent._agent_context(no_mcp=True, no_sandbox=True):
            pass

    resolve.assert_called_once_with(str(core_dir / "bin" / "pi"))


def test_install_pi_reports_each_package(capsys) -> None:
    """The installer reports progress for the core and extension packages."""
    with (
        patch("builtins.input", return_value="yes"),
        patch.object(agent.shutil, "which", side_effect=lambda name: f"/bin/{name}"),
        patch.object(agent.utils, "run_command", return_value=(0, "", "")),
    ):
        agent.cmd_agent_install("pi")

    output = capsys.readouterr().out
    assert agent.PI_CORE_PACKAGE in output
    for package in agent.PI_PACKAGES:
        assert package in output
    assert "installation complete" in output.lower()


def test_agent_cli_registers_install_subcommand() -> None:
    """The agent command exposes an install subcommand."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    agent.ingest_cli_options(subparsers)

    args = parser.parse_args(["agent", "install", "pi"])

    assert args.agent_cmd == "install"
    assert args.agent_name == "pi"


def test_agent_install_lists_supported_agents(capsys) -> None:
    """Omitting the install name lists supported installers."""
    agent.cmd_agent_install()

    assert "pi" in capsys.readouterr().out


def test_agent_install_rejects_unsupported_agent() -> None:
    """Unknown agent installers fail with supported alternatives."""
    import pytest

    with pytest.raises(ValueError, match="Supported agents: pi"):
        agent.cmd_agent_install("opencode")


def test_pi_runtime_volumes_include_node_global_paths(tmp_path: Path, monkeypatch) -> None:
    """Pi's Node installation is mounted read-only in the sandbox."""
    monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path / "packages"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pi_path = bin_dir / "pi"
    pi_path.touch()

    assert agent._get_pi_runtime_volumes(str(pi_path)) == [
        f"{tmp_path}:ro",
    ]


def test_pi_runtime_volumes_preserve_npm_bin_symlink_location(tmp_path: Path, monkeypatch) -> None:
    """A symlinked Pi executable still resolves its NVM prefix from bin."""
    monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path / "missing-packages"))
    prefix = tmp_path / "node"
    bin_dir = prefix / "bin"
    package_dir = prefix / "lib" / "node_modules"
    bin_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    target = package_dir / "pi.js"
    target.touch()
    pi_path = bin_dir / "pi"
    pi_path.symlink_to(target)

    assert agent._get_pi_runtime_volumes(str(pi_path)) == [f"{prefix}:ro"]


def test_pi_passes_mcp_config_when_wtmcp_is_running(tmp_path: Path, monkeypatch) -> None:
    """Pi receives its MCP configuration only when wtmcp is available."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent, "_get_pi_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf", "port": 8123}}

    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=True),
        patch.object(agent.subprocess, "Popen") as popen,
    ):
        agent._start_agent_pi("/bin/pi", cfg, 8080, False, None)

    command = popen.call_args.args[0]
    assert command == [
        "/bin/pi",
        "--model",
        "local-llm/test",
        "--mcp-config",
        str(agent_dir / "mcp.json"),
    ]
    assert json.loads((agent_dir / "mcp.json").read_text()) == {
        "mcpServers": {"wtmcp": {"url": "http://127.0.0.1:8080/mcp"}}
    }


def test_pi_omits_mcp_config_when_wtmcp_is_not_running(tmp_path: Path, monkeypatch) -> None:
    """Pi does not use stale MCP configuration when wtmcp is unavailable."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    (agent_dir / "mcp.json").write_text("{}")
    monkeypatch.setattr(agent, "_get_pi_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf"}}

    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch.object(agent.subprocess, "Popen") as popen,
    ):
        agent._start_agent_pi("/bin/pi", cfg, 8080, False, None)

    assert "--mcp-config" not in popen.call_args.args[0]
    assert not (agent_dir / "mcp.json").exists()


def test_pi_omits_mcp_config_when_mcp_is_disabled(tmp_path: Path, monkeypatch) -> None:
    """Pi starts without an MCP configuration when MCP is disabled."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent, "_get_pi_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf"}}

    with patch.object(agent.subprocess, "Popen") as popen:
        agent._start_agent_pi("/bin/pi", cfg, None, False, None)

    assert "--mcp-config" not in popen.call_args.args[0]
    assert not (agent_dir / "mcp.json").exists()
