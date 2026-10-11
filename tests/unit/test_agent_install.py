"""Tests for Pi installation and agent registration."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from arkai import agent, agent_pi, config


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
        patch.object(agent_pi.shutil, "which", side_effect=lambda name: f"/bin/{name}"),
        patch.object(agent.utils, "run_command", side_effect=capture_command),
        patch.object(agent_pi, "get_core_dir", return_value=core_dir),
        patch.object(agent_pi, "get_agent_dir", return_value=agent_dir),
    ):
        agent.cmd_agent_install("pi")

    assert commands[0][0] == [
        "/bin/npm",
        "install",
        "--prefix",
        str(core_dir),
        "-g",
        "--ignore-scripts",
        agent_pi.PI_CORE_PACKAGE,
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
    monkeypatch.setattr(agent_pi, "get_core_dir", lambda: core_dir)
    cfg = {"agent": {"name": "pi"}, "inference": {"model": "test.gguf"}}

    with (
        patch.object(agent.config, "load_config", return_value=cfg),
        patch.object(agent.config, "validate_config", return_value=True),
        patch.object(agent.inference, "is_inference_running", return_value=True),
        patch.object(agent.inference, "get_inference_model", return_value="test"),
        patch.object(
            agent.utils, "resolve_binary", return_value=str(core_dir / "bin" / "pi")
        ) as resolve,
    ):
        with agent._agent_context("pi", no_mcp=True, no_sandbox=True):
            pass

    resolve.assert_called_once_with(str(core_dir / "bin" / "pi"))


def test_install_pi_reports_each_package(capsys) -> None:
    """The installer reports progress for the core and extension packages."""
    with (
        patch("builtins.input", return_value="yes"),
        patch.object(agent_pi.shutil, "which", side_effect=lambda name: f"/bin/{name}"),
        patch.object(agent.utils, "run_command", return_value=(0, "", "")),
    ):
        agent.cmd_agent_install("pi")

    output = capsys.readouterr().out
    assert agent_pi.PI_CORE_PACKAGE in output
    for package in agent_pi.PI_PACKAGES:
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
    """Pi and its Node installation are mounted read-only in the sandbox."""
    monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path / "packages"))
    pi_prefix = tmp_path / "pi"
    bin_dir = pi_prefix / "bin"
    bin_dir.mkdir(parents=True)
    pi_path = bin_dir / "pi"
    pi_path.touch()
    node_prefix = tmp_path / "node-v24"
    node_bin = node_prefix / "bin"
    node_bin.mkdir(parents=True)
    node_path = node_bin / "node"
    node_path.touch()

    assert agent_pi.get_runtime_volumes(str(pi_path), str(node_path)) == [
        f"{pi_prefix}:ro",
        f"{node_prefix}:ro",
    ]


def test_pi_sandbox_requires_node_from_host_path(monkeypatch) -> None:
    """A missing Node executable produces an actionable sandbox launch error."""
    monkeypatch.setattr(agent_pi.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Node.js was not found in PATH"):
        agent_pi._resolve_node_path()


def test_pi_identifies_arkai_managed_binary(tmp_path: Path, monkeypatch) -> None:
    """Only Arkai's managed Pi binary is eligible for automatic update checks."""
    core_dir = tmp_path / "pi" / "core"
    pi_path = core_dir / "bin" / "pi"
    pi_path.parent.mkdir(parents=True)
    pi_path.write_text("#!/bin/sh\n")
    monkeypatch.setattr(agent_pi, "get_core_dir", lambda: core_dir)

    assert agent_pi._is_arkai_managed_pi(str(pi_path))
    assert not agent_pi._is_arkai_managed_pi("/usr/local/bin/pi")


def test_pi_update_check_uses_pi_manager_and_filters_to_arkai_packages(
    tmp_path: Path, monkeypatch
) -> None:
    """The preflight uses Pi's update checker but only reports Arkai extensions."""
    core_dir = tmp_path / "pi-core"
    package_root = core_dir / "lib" / "node_modules" / "@earendil-works" / "pi-coding-agent"
    pi_entrypoint = package_root / "dist" / "bundle" / "cli.js"
    pi_entrypoint.parent.mkdir(parents=True)
    pi_entrypoint.touch()
    pi_path = core_dir / "bin" / "pi"
    pi_path.parent.mkdir(parents=True)
    pi_path.symlink_to(pi_entrypoint)
    manager_entrypoint = package_root / "dist" / "index.js"
    manager_entrypoint.touch()
    agent_dir = tmp_path / "pi-agent"
    updates = [
        {
            "source": "npm:pi-mcp-adapter",
            "displayName": "pi-mcp-adapter",
            "scope": "user",
            "type": "npm",
        },
        {
            "source": "npm:pi-web-access",
            "displayName": "pi-web-access",
            "scope": "user",
            "type": "npm",
        },
        {
            "source": "npm:another-extension",
            "displayName": "another-extension",
            "scope": "user",
            "type": "npm",
        },
        {
            "source": "npm:pi-subagents",
            "displayName": "pi-subagents",
            "scope": "project",
            "type": "npm",
        },
    ]
    result = SimpleNamespace(returncode=0, stdout=json.dumps(updates), stderr="")

    with (
        patch.object(agent_pi, "_resolve_node_path", return_value="/node/bin/node"),
        patch.object(agent_pi.subprocess, "run", return_value=result) as run,
    ):
        found = agent_pi._check_for_package_updates(str(pi_path), str(agent_dir))

    assert [update["displayName"] for update in found] == ["pi-mcp-adapter", "pi-web-access"]
    command = run.call_args.args[0]
    assert command[:3] == ["/node/bin/node", "--input-type=module", "-e"]
    assert manager_entrypoint.as_uri() in command[3]
    assert "checkForAvailableUpdates" in command[3]
    assert run.call_args.kwargs["env"]["PI_CODING_AGENT_DIR"] == str(agent_dir)
    assert run.call_args.kwargs["cwd"] == str(agent_dir)


def test_pi_updates_only_reported_packages_after_confirmation(tmp_path: Path, monkeypatch) -> None:
    """Pi's updater runs for each outdated package only after user approval."""
    agent_dir = tmp_path / "pi-agent"
    updates = [
        {"source": "npm:pi-mcp-adapter", "displayName": "pi-mcp-adapter"},
        {"source": "npm:pi-web-access", "displayName": "pi-web-access"},
    ]
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    with (
        patch.object(agent_pi, "_is_arkai_managed_pi", return_value=True),
        patch.object(agent_pi, "_check_for_package_updates", return_value=updates),
        patch("builtins.input", return_value="") as prompt,
        patch.object(agent_pi.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run,
    ):
        agent_pi._maybe_update_arkai_packages("/managed/pi")

    assert [call.args[0] for call in run.call_args_list] == [
        ["/managed/pi", "update", "--extension", "npm:pi-mcp-adapter"],
        ["/managed/pi", "update", "--extension", "npm:pi-web-access"],
    ]
    prompt.assert_called_once_with("Upgrade these packages before starting Pi? [Y/n] ")
    assert all(call.kwargs["cwd"] == str(agent_dir) for call in run.call_args_list)


def test_pi_keeps_outdated_packages_when_update_is_declined(tmp_path: Path, monkeypatch) -> None:
    """Declining the Pi update prompt leaves installed packages unchanged."""
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: tmp_path / "pi-agent")
    with (
        patch.object(agent_pi, "_is_arkai_managed_pi", return_value=True),
        patch.object(
            agent_pi,
            "_check_for_package_updates",
            return_value=[{"source": "npm:pi-mcp-adapter", "displayName": "pi-mcp-adapter"}],
        ),
        patch("builtins.input", return_value="n"),
        patch.object(agent_pi.subprocess, "run") as run,
    ):
        agent_pi._maybe_update_arkai_packages("/managed/pi")

    run.assert_not_called()


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
    node_prefix = tmp_path / "node-runtime"
    node_bin = node_prefix / "bin"
    node_bin.mkdir(parents=True)
    node_path = node_bin / "node"
    node_path.touch()

    assert agent_pi.get_runtime_volumes(str(pi_path), str(node_path)) == [
        f"{prefix}:ro",
        f"{node_prefix}:ro",
    ]


def test_pi_passes_mcp_config_when_wtmcp_is_running(tmp_path: Path, monkeypatch) -> None:
    """Pi receives its MCP configuration only when wtmcp is available."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf", "port": 8123}}

    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=True),
        patch.object(agent.subprocess, "Popen") as popen,
    ):
        agent_pi.start("/bin/pi", cfg, 8080, False, None)

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


def test_pi_passes_configured_theme_to_cli(tmp_path: Path, monkeypatch) -> None:
    """Pi receives the configured theme through its theme CLI option."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    cfg = {"agent": {"theme": "dracula"}, "inference": {"model": "test.gguf"}}

    with patch.object(agent.subprocess, "Popen") as popen:
        agent_pi.start("/bin/pi", cfg, None, False, None)

    command = popen.call_args.args[0]
    assert command == [
        "/bin/pi",
        "--model",
        "local-llm/test",
        "--use-theme",
        "dracula",
    ]


@pytest.mark.parametrize(
    ("term_program", "expected_osc52"),
    [("iTerm.app", True), ("Apple_Terminal", False)],
)
def test_pi_interactive_sandbox_requests_tty(
    tmp_path: Path, monkeypatch, term_program: str, expected_osc52: bool
) -> None:
    """Interactive sandboxed Pi runs Node directly and allocate a terminal."""
    monkeypatch.setenv("TERM_PROGRAM", term_program)
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    pi_prefix = tmp_path / "pi-core"
    pi_bin = pi_prefix / "bin"
    pi_bin.mkdir(parents=True)
    pi_entrypoint = pi_prefix / "lib" / "cli.js"
    pi_entrypoint.parent.mkdir(parents=True)
    pi_entrypoint.write_text("#!/usr/bin/env node\n")
    pi_path = pi_bin / "pi"
    pi_path.symlink_to(pi_entrypoint)
    node_prefix = tmp_path / "node-runtime"
    node_bin = node_prefix / "bin"
    node_bin.mkdir(parents=True)
    node_path = node_bin / "node"
    node_path.touch()

    with (
        patch.object(
            agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--tty", "--"]
        ) as build_sandbox,
        patch.object(agent_pi, "_resolve_node_path", return_value=str(node_path)),
        patch.object(agent_pi.platform, "system", return_value="Darwin"),
    ):
        spec = agent_pi.build_launch_spec(
            str(pi_path),
            {"inference": {"model": "test.gguf"}},
            None,
            True,
            None,
        )

    assert spec.command[:4] == ["arapuca", "run", "--tty", "--"]
    assert spec.command[4:8] == [
        str(node_path),
        "--require",
        str(spec.cleanup_paths[0]),
        str(pi_entrypoint),
    ]
    assert spec.command[8:] == ["--model", "local-llm/test"]
    preload = spec.cleanup_paths[0].read_text()
    assert "process.env.PATH" in preload
    assert str(node_bin) in preload
    assert ('process.env.MOSH_CONNECTION = "arkai-local"' in preload) is expected_osc52
    assert build_sandbox.call_args.kwargs["tty"] is True
    assert build_sandbox.call_args.args[5] == [f"{pi_prefix}:ro", f"{node_prefix}:ro"]
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_pi_prompt_sandbox_does_not_enable_osc52_fallback(tmp_path: Path, monkeypatch) -> None:
    """Non-interactive Pi launches do not emit terminal clipboard sequences."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    pi_prefix = tmp_path / "pi-core"
    pi_bin = pi_prefix / "bin"
    pi_bin.mkdir(parents=True)
    pi_path = pi_bin / "pi"
    pi_path.write_text("#!/bin/sh\n")
    node_prefix = tmp_path / "node-runtime"
    node_bin = node_prefix / "bin"
    node_bin.mkdir(parents=True)
    node_path = node_bin / "node"
    node_path.touch()

    with (
        patch.object(agent, "_build_sandbox_cmd", return_value=["arapuca", "run", "--"]),
        patch.object(agent_pi, "_resolve_node_path", return_value=str(node_path)),
        patch.object(agent_pi.platform, "system", return_value="Darwin"),
    ):
        spec = agent_pi.build_launch_spec(
            str(pi_path),
            {"inference": {"model": "test.gguf"}},
            None,
            True,
            None,
            prompt="hello",
        )

    assert "-p" in spec.command
    assert "--tty" not in spec.command
    assert "MOSH_CONNECTION" not in spec.cleanup_paths[0].read_text()
    for path in spec.cleanup_paths:
        path.unlink(missing_ok=True)


def test_pi_omits_mcp_config_when_wtmcp_is_not_running(tmp_path: Path, monkeypatch) -> None:
    """Pi does not use stale MCP configuration when wtmcp is unavailable."""
    agent_dir = tmp_path / "pi-agent"
    agent_dir.mkdir()
    (agent_dir / "mcp.json").write_text("{}")
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf"}}

    with (
        patch("arkai.wtmcp.is_wtmcp_running", return_value=False),
        patch.object(agent.subprocess, "Popen") as popen,
    ):
        agent_pi.start("/bin/pi", cfg, 8080, False, None)

    assert "--mcp-config" not in popen.call_args.args[0]
    assert not (agent_dir / "mcp.json").exists()


def test_pi_omits_mcp_config_when_mcp_is_disabled(tmp_path: Path, monkeypatch) -> None:
    """Pi starts without an MCP configuration when MCP is disabled."""
    agent_dir = tmp_path / "pi-agent"
    monkeypatch.setattr(agent_pi, "get_agent_dir", lambda: agent_dir)
    cfg = {"inference": {"model": "test.gguf"}}

    with patch.object(agent.subprocess, "Popen") as popen:
        agent_pi.start("/bin/pi", cfg, None, False, None)

    assert "--mcp-config" not in popen.call_args.args[0]
    assert not (agent_dir / "mcp.json").exists()
