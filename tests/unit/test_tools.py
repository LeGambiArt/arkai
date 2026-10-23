"""Tests for third-party tool management."""

import argparse
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


def _fake_sandbox_builder(calls: list[dict]):
    def build(root, command_names, *, allow_network, read_only_root=False):
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
            tools._sandbox_environment(root, "/usr/bin"),
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


def test_sandbox_path_contains_only_command_directories(tmp_path: Path, monkeypatch) -> None:
    command_dir = tmp_path / "bin"
    monkeypatch.setattr(tools.platform, "system", lambda: "Darwin")

    volumes, path = tools._sandbox_paths({"curl": str(command_dir / "curl")})

    assert f"{command_dir}:ro" in volumes
    assert path == str(command_dir)


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
    with patch.object(tools, "_is_managed_strata_installed", return_value=False):
        tools.cmd_tools_list()

    assert "No Arkai-managed tools installed" in capsys.readouterr().out


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


def test_strata_remove_executes_rm_inside_sandbox(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "strata"
    _create_strata_install(root)
    monkeypatch.setattr(tools, "get_strata_root", lambda: root)
    sandbox_calls: list[dict] = []

    def run_command(command: list[str], **kwargs) -> tuple[int, str, str]:
        external = command[command.index("--") + 1 :]
        assert external[:2] == ["/usr/bin/rm", "-rf"]
        for entry in external[2:]:
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
    with patch.object(tools, "_is_managed_strata_installed", return_value=False):
        with pytest.raises(RuntimeError, match="arkai tools install strata"):
            tools.cmd_tools_update("strata")
