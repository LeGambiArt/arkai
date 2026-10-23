"""Behavior steps for Arkai's third-party tool manager."""

import shlex
from pathlib import Path
from unittest.mock import patch

from behave import given, then, when

from arkai import agent_omp, agent_pi, tools


@given("no Arkai-managed tools are installed")  # ty: ignore[call-non-callable]
def step_no_managed_tools(context):
    context.tools_is_installed = False


@given("Arkai-managed Strata is installed")  # ty: ignore[call-non-callable]
def step_managed_strata_installed(context):
    context.tools_is_installed = True


@given("Arkai-managed Pi is installed")  # ty: ignore[call-non-callable]
def step_managed_pi_installed(context):
    context.tools_is_installed = True


@given("Arkai-managed OMP is installed")  # ty: ignore[call-non-callable]
def step_managed_omp_installed(context):
    context.tools_is_installed = True


@when('I run the tools command "{arguments}"')  # ty: ignore[call-non-callable]
def step_run_tools_command(context, arguments):
    _run_tools_command(context, arguments)


@when('I run the tools command with confirmation "{answer}" for "{arguments}"')  # ty: ignore[call-non-callable]
def step_run_tools_command_with_confirmation(context, answer, arguments):
    _run_tools_command(context, arguments, answer)


def _run_tools_command(context, arguments: str, answer: str = "yes") -> None:
    """Run the public CLI while replacing network and sandbox side effects."""
    if not hasattr(context, "tools_is_installed"):
        context.tools_is_installed = False
    context.tools_sandbox_calls = []
    context.tools_external_calls = []
    context.tools_installer_downloads = []
    context.tools_action = shlex.split(arguments)[0]
    root = tools.get_strata_root()

    def build_sandbox(
        root_path,
        command_names,
        *,
        allow_network,
        read_only_root=False,
        child_environment=None,
    ):
        context.tools_sandbox_calls.append(
            {
                "root": root_path,
                "command_names": command_names,
                "allow_network": allow_network,
                "read_only_root": read_only_root,
            }
        )
        return (
            ["/usr/bin/arapuca", "run", "--"],
            {name: f"/sandbox/bin/{name}" for name in command_names},
            {"PATH": "/sandbox/bin", **(child_environment or {})},
        )

    def run_sandboxed(prefix, command, **kwargs):
        context.tools_external_calls.append((prefix, command, kwargs))
        if context.tools_action in {"install", "update"}:
            context.tools_is_installed = True
        elif context.tools_action == "remove":
            context.tools_is_installed = False
        return 0, "", ""

    def download_installer(installer: Path) -> None:
        context.tools_installer_downloads.append(installer)

    with (
        patch.object(
            tools,
            "_is_managed_strata_installed",
            side_effect=lambda: context.tools_is_installed,
        ),
        patch.object(tools, "_ensure_install_root", return_value=root),
        patch.object(tools, "_ensure_managed_root", return_value=root),
        patch.object(tools, "_tool_root", return_value=root),
        patch.object(
            tools, "_is_tool_installed", side_effect=lambda _name: context.tools_is_installed
        ),
        patch.object(tools, "_build_sandbox_prefix", side_effect=build_sandbox),
        patch.object(tools, "_download_installer", side_effect=download_installer),
        patch.object(tools, "_run_sandboxed", side_effect=run_sandboxed),
        patch.object(tools, "_strata_version", return_value="Strata 0.2.3"),
        patch.object(tools, "_pi_version", return_value="1.2.3"),
        patch.object(tools, "_omp_version", return_value="omp 2.0"),
        patch.object(tools.shutil, "which", side_effect=lambda name: f"/sandbox/bin/{name}"),
        patch.object(agent_pi, "get_install_dir", return_value=root),
        patch.object(agent_omp, "get_install_dir", return_value=root),
        patch.object(agent_pi, "_resolve_node_path", return_value="/sandbox/bin/node"),
        patch.object(
            agent_pi,
            "_check_for_package_updates",
            return_value=[
                {"source": f"npm:{package}", "displayName": package}
                for package in agent_pi.PI_PACKAGES
            ],
        ),
        patch("builtins.input", return_value=answer),
        patch.object(Path, "rmdir", return_value=None),
    ):
        context.run_command(["arkai", "tools", *shlex.split(arguments)])


@then("no external tool was executed")  # ty: ignore[call-non-callable]
def step_no_tools_executed(context):
    assert not context.tools_external_calls


@then("the installer script bootstrap was requested")  # ty: ignore[call-non-callable]
def step_installer_download_requested(context):
    assert context.tools_installer_downloads == [tools.get_strata_root() / "tmp" / "install.sh"]


@then("all external tool commands ran inside Arapuca")  # ty: ignore[call-non-callable]
def step_tools_ran_inside_arapuca(context):
    assert context.tools_external_calls
    for prefix, _command, kwargs in context.tools_external_calls:
        assert prefix[:2] == ["/usr/bin/arapuca", "run"]
        assert prefix[-1] == "--"
        assert kwargs["env_binary"].endswith("/env")
        assert "HOME" not in kwargs["environment"]
        assert "TMPDIR" not in kwargs["environment"]


@then("the sandbox allowed network access")  # ty: ignore[call-non-callable]
def step_sandbox_network_access(context):
    assert context.tools_sandbox_calls
    assert context.tools_sandbox_calls[0]["allow_network"] is True


@then("Pi's supported extensions were updated")  # ty: ignore[call-non-callable]
def step_pi_extensions_updated(context):
    commands = [command for _prefix, command, _kwargs in context.tools_external_calls]
    updated = {
        command[3]
        for command in commands
        if len(command) >= 4 and command[1:3] == ["update", "--extension"]
    }
    assert updated == {f"npm:{package}" for package in tools._get_tool_module("pi").PI_PACKAGES}


@then("tool removal ran inside Arapuca without network access")  # ty: ignore[call-non-callable]
def step_tool_removal_sandboxed(context):
    assert context.tools_sandbox_calls[0]["allow_network"] is False
    prefix, command, _kwargs = context.tools_external_calls[0]
    assert prefix[0] == "/usr/bin/arapuca"
    assert prefix[-1] == "--"
    assert command[0].endswith("/rm")
