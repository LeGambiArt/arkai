"""Tests for third-party tool management."""

import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from arkai import tools


def _create_strata_install(root: Path, version: str = "0.2.3") -> Path:
    """Create the directory and symlinks used by Strata's official installer."""
    binary = root / "data" / "terminal" / version / "bin" / "strata"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    current = root / "data" / "terminal" / "current"
    current.symlink_to(root / "data" / "terminal" / version)
    stable = root / "bin" / "strata"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.symlink_to(current / "bin" / "strata")
    return stable


def _create_pi_install(root: Path, version: str = "1.2.3") -> Path:
    package_dir = root / "core" / "lib" / "node_modules" / "@earendil-works" / "pi-coding-agent"
    entrypoint = package_dir / "dist" / "bundle" / "cli.js"
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text("#!/usr/bin/env node\n")
    entrypoint.chmod(0o755)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package.json").write_text(
        json.dumps({"name": "@earendil-works/pi-coding-agent", "version": version})
    )
    binary = root / "core" / "bin" / "pi"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.symlink_to(entrypoint)
    (root / "agent").mkdir(parents=True, exist_ok=True)
    return binary


def _create_omp_install(root: Path, version: str = "omp test-version") -> Path:
    binary = root / "bin" / "omp"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    (root / "agent").mkdir(parents=True, exist_ok=True)
    return binary


def _fake_sandbox_builder(calls: list[dict]):
    def build(
        root,
        command_names,
        *,
        allow_network,
        read_only_root=False,
        child_environment=None,
    ):
        calls.append(
            {
                "root": root,
                "command_names": command_names,
                "allow_network": allow_network,
                "read_only_root": read_only_root,
            }
        )
        return (
            ["/usr/bin/arapuca", "run", "--"],
            {name: f"/usr/bin/{name}" for name in command_names},
            {"PATH": "/usr/bin", **(child_environment or {})},
        )

    return build


def _environment_args(root: Path) -> list[str]:
    return [
        f"{name}={value}" for name, value in tools._sandbox_environment(root, "/usr/bin").items()
    ]


def test_tools_parser_registers_lifecycle_commands() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    tools.ingest_cli_options(subparsers)

    assert parser.parse_args(["tools", "list"]).tools_cmd == "list"
    assert parser.parse_args(["tools", "install", "strata"]).tool_name == "strata"
    assert parser.parse_args(["tools", "update", "strata"]).tools_cmd == "update"
    assert parser.parse_args(["tools", "remove", "strata"]).tools_cmd == "remove"


def test_tools_install_without_name_lists_supported_tools(capsys) -> None:
    tools.cmd_tools_install()

    assert "strata" in capsys.readouterr().out


def test_managed_strata_binary_is_available_to_inference(tmp_path: Path, monkeypatch) -> None:
    """Only a verified Arkai-owned Strata installation is auto-discovered."""
    root = tmp_path / "strata"
    binary = _create_strata_install(root)
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)

    assert tools.get_managed_strata_binary() == binary


def test_managed_strata_binary_is_none_when_not_installed(tmp_path: Path, monkeypatch) -> None:
    """A missing managed install leaves normal PATH discovery available."""
    monkeypatch.setattr(tools, "get_strata_root", lambda: tmp_path / "strata")

    assert tools.get_managed_strata_binary() is None


def test_pi_version_comes_from_arkai_managed_package_metadata(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "pi"
    _create_pi_install(root, "1.2.3")
    from arkai import agent_pi

    monkeypatch.setattr(agent_pi, "get_install_dir", lambda: root)

    assert tools._is_tool_installed("pi")
    assert tools._tool_version("pi") == "1.2.3"


def test_omp_version_is_executed_inside_arapuca(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "omp"
    binary = _create_omp_install(root)
    from arkai import agent_omp

    monkeypatch.setattr(agent_omp, "get_install_dir", lambda: root)
    sandbox_calls: list[dict] = []

    with (
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools.utils, "run_command", return_value=(0, "omp 2.0\n", "")) as run,
    ):
        assert tools._tool_version("omp") == "omp 2.0"

    command = run.call_args.args[0]
    assert command[:2] == ["/usr/bin/arapuca", "run"]
    assert command[command.index("--") + 1 :] == [
        "/usr/bin/env",
        "PATH=/usr/bin",
        str(binary),
        "--version",
    ]
    assert sandbox_calls[0]["allow_network"] is False


def test_sandbox_path_contains_only_command_directories(tmp_path: Path, monkeypatch) -> None:
    command_dir = tmp_path / "bin"
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")

    volumes, path = tools._sandbox_paths({"curl": str(command_dir / "curl")})

    assert f"{command_dir}:ro" in volumes
    assert path == str(command_dir)


def test_sandbox_mounts_complete_npm_runtime_but_keeps_its_bin_on_path(
    tmp_path: Path, monkeypatch
) -> None:
    npm_root = tmp_path / "nvm" / "lib" / "node_modules" / "npm"
    npm_bin = npm_root / "bin"
    npm_bin.mkdir(parents=True)
    (npm_root / "package.json").write_text(json.dumps({"name": "npm"}))
    npm = npm_bin / "npm-cli.js"
    npm.touch()
    node_bin = tmp_path / "nvm" / "bin"
    node_bin.mkdir()
    (tmp_path / "nvm" / "lib").mkdir(exist_ok=True)
    node = node_bin / "node"
    node.touch()
    env_bin = Path("/usr/bin/env")
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")

    volumes, path = tools._sandbox_paths({"npm": str(npm), "node": str(node), "env": str(env_bin)})

    assert f"{tmp_path / 'nvm'}:ro" in volumes
    assert f"{npm_bin}:ro" not in volumes
    assert f"{node_bin}:ro" not in volumes
    assert str(npm_bin) in path
    assert str(node_bin) in path


def test_sandbox_environment_leaves_managed_variables_to_arapuca(tmp_path: Path) -> None:
    environment = tools._sandbox_environment(tmp_path, "/usr/bin")

    assert "HOME" not in environment
    assert "TMPDIR" not in environment
    assert environment["PATH"] == "/usr/bin"
    assert environment["STRATA_DATA_DIR"] == str(tmp_path / "data")


def test_download_installer_uses_requests_and_checks_https(tmp_path: Path) -> None:
    installer = tmp_path / "install.sh"
    script = b"#!/bin/sh\nprintf 'installer test\\n'\n"

    class FakeResponse:
        url = tools.STRATA_INSTALLER_URL

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield script

    with patch.object(tools.requests, "get", return_value=FakeResponse()) as get:
        tools._download_installer(installer)

    get.assert_called_once_with(tools.STRATA_INSTALLER_URL, stream=True, timeout=30)
    assert installer.read_bytes() == script
    assert installer.stat().st_mode & 0o777 == 0o600


def test_sandbox_prefix_does_not_inherit_profile_mounts_or_secrets() -> None:
    from arkai import agent

    cfg = {
        "inference": {"port": 8123},
        "sandbox": {
            "path": "arapuca",
            "active_profile": "agent",
            "profiles": {
                "agent": {
                    "path": "/custom/arapuca",
                    "memory_mb": 4096,
                    "cpus": 4,
                    "pids": 512,
                    "timeout": 90,
                    "volume": ["/private/credentials"],
                    "environment": {"TOKEN": "secret"},
                }
            },
            "volume": ["/home/user"],
            "environment": {"API_KEY": "secret"},
        },
    }

    with (
        patch.object(tools.config, "load_config", return_value=cfg),
        patch.object(tools, "_host_tool_paths", return_value={"rm": "/usr/bin/rm"}),
        patch.object(tools, "_sandbox_paths", return_value=([], "/usr/bin")),
        patch.object(agent, "_build_sandbox_cmd", return_value=["/arapuca", "run", "--"]) as build,
    ):
        tools._build_sandbox_prefix(Path("/install"), ("rm",), allow_network=False)

    sandbox_config = build.call_args.args[0]["sandbox"]
    assert sandbox_config["path"] == "/custom/arapuca"
    assert sandbox_config["memory_mb"] == 4096
    assert sandbox_config["volume"] == []
    assert sandbox_config["environment"] == {}
    assert sandbox_config["active_profile"] is None
    assert build.call_args.kwargs["cli_environment"] == {}


def test_sandboxed_command_handles_uncaptured_error_output() -> None:
    with patch.object(tools.utils, "run_command", return_value=(1, None, None)):
        with pytest.raises(RuntimeError, match="exit code 1"):
            tools._run_sandboxed(["/arapuca", "run", "--"], ["/bin/false"])


def test_tools_list_reports_managed_strata_version(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "strata"
    _create_strata_install(root)
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)
    sandbox_calls: list[dict] = []

    with (
        patch.object(tools, "_is_tool_installed", side_effect=lambda name: name == "strata"),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools.utils, "run_command", return_value=(0, "Strata 0.2.3\n", "")) as run,
    ):
        tools.cmd_tools_list()

    assert "strata: Strata 0.2.3" in capsys.readouterr().out
    assert sandbox_calls[0]["read_only_root"] is True
    command = run.call_args.args[0]
    assert command[:2] == ["/usr/bin/arapuca", "run"]
    assert command[command.index("--") + 1 :] == [
        "/usr/bin/env",
        *_environment_args(root),
        str(root / "bin" / "strata"),
        "--version",
    ]


def test_tools_list_reports_empty_managed_installations(capsys) -> None:
    with patch.object(tools, "_is_tool_installed", return_value=False):
        tools.cmd_tools_list()

    assert "No Arkai-managed tools installed" in capsys.readouterr().out


def test_tools_list_shows_versions_for_each_managed_tool(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from arkai import agent_omp, agent_pi

    strata_root = tmp_path / "strata"
    _create_strata_install(strata_root)
    pi_root = tmp_path / "pi"
    _create_pi_install(pi_root, "1.2.3")
    omp_root = tmp_path / "omp"
    _create_omp_install(omp_root)
    monkeypatch.setattr(tools, "get_strata_root", lambda: strata_root)
    monkeypatch.setattr(agent_pi, "get_install_dir", lambda: pi_root)
    monkeypatch.setattr(agent_omp, "get_install_dir", lambda: omp_root)

    with (
        patch.object(tools, "_strata_version", return_value="Strata 0.2.3"),
        patch.object(tools, "_omp_version", return_value="OMP 2.0"),
    ):
        tools.cmd_tools_list()

    output = capsys.readouterr().out
    assert "strata: Strata 0.2.3" in output
    assert "pi: 1.2.3" in output
    assert "omp: OMP 2.0" in output


def test_strata_install_downloads_and_runs_official_installer_in_sandbox(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "strata"
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")
    sandbox_calls: list[dict] = []
    commands: list[list[str]] = []

    def run_command(command: list[str], **kwargs) -> tuple[int, str, str]:
        commands.append(command)
        external_command = command[command.index("--") + 1 :]
        assignments = _environment_args(root)
        assert external_command[: 1 + len(assignments)] == ["/usr/bin/env", *assignments]
        child_command = external_command[1 + len(assignments) :]
        if child_command[0] == "/usr/bin/sh":
            _create_strata_install(root)
        elif child_command[0] == str(root / "bin" / "strata"):
            return 0, "Strata 0.2.3\n", ""
        return 0, "", ""

    with (
        patch("builtins.input", return_value="yes"),
        patch.object(
            tools,
            "_download_installer",
            side_effect=lambda path: path.write_text("#!/bin/sh\n# official installer\n"),
        ) as download,
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools.utils, "run_command", side_effect=run_command),
    ):
        tools.cmd_tools_install("strata")

    assert len(commands) == 2
    download.assert_called_once_with(root / "tmp" / "install.sh")
    assert all(command[:2] == ["/usr/bin/arapuca", "run"] for command in commands)
    assert all("--" in command for command in commands)
    assert commands[0][commands[0].index("--") + 1 :] == [
        "/usr/bin/env",
        *_environment_args(root),
        "/usr/bin/sh",
        str(root / "tmp" / "install.sh"),
    ]
    assert sandbox_calls[0]["allow_network"] is True
    assert sandbox_calls[-1]["read_only_root"] is True
    output = capsys.readouterr().out
    assert "Strata install complete (Strata 0.2.3)" in output
    assert str(root / "bin" / "strata") in output


def test_strata_install_decline_does_not_create_files_or_run_commands(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "strata"
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)

    with (
        patch("builtins.input", return_value="no"),
        patch.object(tools.utils, "run_command") as run,
    ):
        tools.cmd_tools_install("strata")

    assert not root.exists()
    run.assert_not_called()
    assert "cancelled" in capsys.readouterr().out.lower()


def test_strata_update_reuses_sandboxed_official_installer(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "strata"
    _create_strata_install(root, "0.2.2")
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")
    sandbox_calls: list[dict] = []

    def run_command(command: list[str], **kwargs) -> tuple[int, str, str]:
        external = command[command.index("--") + 1 :]
        assignments = _environment_args(root)
        assert external[: 1 + len(assignments)] == ["/usr/bin/env", *assignments]
        child_command = external[1 + len(assignments) :]
        if child_command[0] == "/usr/bin/sh":
            (root / "data" / "terminal" / "current").unlink()
            (root / "bin" / "strata").unlink()
            _create_strata_install(root, "0.2.3")
        elif child_command[0] == str(root / "bin" / "strata"):
            return 0, "Strata 0.2.3\n", ""
        return 0, "", ""

    with (
        patch("builtins.input", return_value="y"),
        patch.object(
            tools,
            "_download_installer",
            side_effect=lambda path: path.write_text("#!/bin/sh\n# official installer\n"),
        ),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools.utils, "run_command", side_effect=run_command),
    ):
        tools.cmd_tools_update("strata")

    assert [call["allow_network"] for call in sandbox_calls] == [True, False]
    assert tools._is_managed_strata_installed()


def test_pi_install_and_update_commands_run_inside_network_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    from arkai import agent_pi

    root = tmp_path / "pi"
    monkeypatch.setattr(agent_pi, "get_install_dir", lambda: root)
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        tools.shutil,
        "which",
        lambda name: {"npm": "/node/bin/npm", "node": "/node/bin/node"}.get(name, f"/bin/{name}"),
    )
    sandbox_calls: list[dict] = []
    commands: list[tuple[list[str], list[str], dict]] = []

    def run_sandboxed(prefix: list[str], command: list[str], **kwargs) -> tuple[int, str, str]:
        commands.append((prefix, command, kwargs))
        if command[0] == "/usr/bin/npm":
            _create_pi_install(root)
        return 0, "", ""

    with (
        patch("builtins.input", return_value="yes"),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools, "_run_sandboxed", side_effect=run_sandboxed),
    ):
        tools.cmd_tools_install("pi")

    assert len(commands) == 4
    assert commands[0][1] == [
        "/usr/bin/npm",
        "install",
        "--prefix",
        str(root / "core"),
        "-g",
        "--ignore-scripts",
        agent_pi.PI_CORE_PACKAGE,
    ]
    assert [command[1][1:] for command in commands[1:]] == [
        ["install", "npm:pi-mcp-adapter"],
        ["install", "npm:pi-web-access"],
        ["install", "npm:pi-subagents"],
    ]
    assert all(call["allow_network"] is True for call in sandbox_calls)
    assert all(prefix[:2] == ["/usr/bin/arapuca", "run"] for prefix, _, _ in commands)
    assert all("PI_CODING_AGENT_DIR" in kwargs["environment"] for _, _, kwargs in commands)
    assert tools._is_tool_installed("pi")


def test_pi_update_uses_pi_package_manager_for_outdated_extensions(
    tmp_path: Path, monkeypatch
) -> None:
    from arkai import agent_pi

    root = tmp_path / "pi"
    _create_pi_install(root)
    monkeypatch.setattr(agent_pi, "get_install_dir", lambda: root)
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        tools.shutil,
        "which",
        lambda name: {"npm": "/node/bin/npm", "node": "/node/bin/node"}.get(name, f"/bin/{name}"),
    )
    sandbox_calls: list[dict] = []
    commands: list[tuple[list[str], list[str], dict]] = []
    updates = [
        {"source": "npm:pi-mcp-adapter", "displayName": "pi-mcp-adapter"},
        {"source": "npm:pi-subagents", "displayName": "pi-subagents"},
        {"source": "npm:unmanaged-package", "displayName": "unmanaged-package"},
    ]

    def run_sandboxed(prefix: list[str], command: list[str], **kwargs) -> tuple[int, str, str]:
        commands.append((prefix, command, kwargs))
        return 0, "", ""

    with (
        patch("builtins.input", return_value="yes"),
        patch.object(agent_pi, "_check_for_package_updates", return_value=updates),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools, "_run_sandboxed", side_effect=run_sandboxed),
    ):
        tools.cmd_tools_update("pi")

    assert commands[0][1][0] == "/usr/bin/npm"
    assert [command[1] for command in commands[1:]] == [
        [str(root / "core" / "bin" / "pi"), "update", "--extension", "npm:pi-mcp-adapter"],
        [str(root / "core" / "bin" / "pi"), "update", "--extension", "npm:pi-subagents"],
    ]
    assert all(call["allow_network"] is True for call in sandbox_calls)
    assert all(prefix[:2] == ["/usr/bin/arapuca", "run"] for prefix, _, _ in commands)


def test_omp_install_downloads_and_runs_official_script_in_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    from arkai import agent_omp

    root = tmp_path / "omp"
    monkeypatch.setattr(agent_omp, "get_install_dir", lambda: root)
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")
    sandbox_calls: list[dict] = []
    commands: list[tuple[list[str], list[str], dict]] = []

    def run_sandboxed(prefix: list[str], command: list[str], **kwargs) -> tuple[int, str, str]:
        commands.append((prefix, command, kwargs))
        if command[0] == "/usr/bin/curl":
            Path(command[-1]).write_text("#!/bin/sh\n# official OMP installer\n")
        elif command[0] == "/usr/bin/sh":
            _create_omp_install(root)
        elif command[0] == str(root / "bin" / "omp"):
            return 0, "omp 2.0\n", ""
        return 0, "", ""

    with (
        patch("builtins.input", return_value="yes"),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools, "_run_sandboxed", side_effect=run_sandboxed),
    ):
        tools.cmd_tools_install("omp")

    assert commands[0][1] == [
        "/usr/bin/curl",
        "-fsSL",
        agent_omp.OMP_INSTALLER_URL,
        "-o",
        str(root / "tmp" / "install.sh"),
    ]
    assert commands[1][1] == ["/usr/bin/sh", str(root / "tmp" / "install.sh"), "--binary"]
    assert all(prefix[:2] == ["/usr/bin/arapuca", "run"] for prefix, _, _ in commands)
    assert sandbox_calls[0]["allow_network"] is True
    assert tools._is_tool_installed("omp")


def test_strata_remove_executes_rm_inside_sandbox(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "strata"
    _create_strata_install(root)
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)
    sandbox_calls: list[dict] = []

    def run_command(command: list[str], **kwargs) -> tuple[int, str, str]:
        external = command[command.index("--") + 1 :]
        assert external[:2] == ["/usr/bin/env", "PATH=/usr/bin"]
        assert external[2].endswith("/rm")
        assert external[3] == "-rf"
        for entry in external[4:]:
            path = root / entry
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        return 0, "", ""

    with (
        patch("builtins.input", return_value="yes"),
        patch.object(
            tools, "_build_sandbox_prefix", side_effect=_fake_sandbox_builder(sandbox_calls)
        ),
        patch.object(tools.utils, "run_command", side_effect=run_command) as run,
    ):
        tools.cmd_tools_remove("strata")

    assert sandbox_calls[0]["allow_network"] is False
    assert run.call_args.args[0][:2] == ["/usr/bin/arapuca", "run"]
    assert not root.exists()


def test_tools_commands_reject_unsupported_tool() -> None:
    with pytest.raises(ValueError, match="Supported tools: strata"):
        tools.cmd_tools_install("unknown")


def test_update_requires_an_arkai_managed_install() -> None:
    with patch.object(tools, "_is_tool_installed", return_value=False):
        with pytest.raises(RuntimeError, match="arkai tools install strata"):
            tools.cmd_tools_update("strata")
