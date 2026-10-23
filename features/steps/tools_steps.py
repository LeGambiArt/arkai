"""Behavior steps for Arkai's third-party tool manager."""

import shlex
from pathlib import Path
from unittest.mock import patch

from behave import given, then, when

from arkai import tools


@given("no Arkai-managed tools are installed")  # ty: ignore[call-non-callable]
def step_no_managed_tools(context):
    context.tools_is_installed = False


@given("Arkai-managed Strata is installed")  # ty: ignore[call-non-callable]
def step_managed_strata_installed(context):
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
    root = tools.get_strata_root()

    def build_sandbox(root_path, command_names, *, allow_network, read_only_root=False):
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
            tools._sandbox_environment(root_path, "/sandbox/bin"),
        )

    def run_sandboxed(prefix, command, **kwargs):
        context.tools_external_calls.append((prefix, command, kwargs))
        executable = command[0]
        if executable.endswith("/sh"):
            context.tools_is_installed = True
        elif executable.endswith("/rm"):
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
        patch.object(tools, "_build_sandbox_prefix", side_effect=build_sandbox),
        patch.object(tools, "_download_installer", side_effect=download_installer),
        patch.object(tools, "_run_sandboxed", side_effect=run_sandboxed),
        patch.object(tools, "_strata_version", return_value="Strata 0.2.3"),
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


@then("the external installer ran inside Arapuca")  # ty: ignore[call-non-callable]
def step_installer_ran_inside_arapuca(context):
    assert context.tools_external_calls
    prefix, command, kwargs = context.tools_external_calls[0]
    assert prefix[:2] == ["/usr/bin/arapuca", "run"]
    assert prefix[-1] == "--"
    assert command[0].endswith("/sh")
    assert kwargs["env_binary"].endswith("/env")
    assert "HOME" not in kwargs["environment"]
    assert "TMPDIR" not in kwargs["environment"]


@then("the sandbox allowed network access")  # ty: ignore[call-non-callable]
def step_sandbox_network_access(context):
    assert context.tools_sandbox_calls
    assert context.tools_sandbox_calls[0]["allow_network"] is True


@then("tool removal ran inside Arapuca without network access")  # ty: ignore[call-non-callable]
def step_tool_removal_sandboxed(context):
    assert context.tools_sandbox_calls[0]["allow_network"] is False
    prefix, command, _kwargs = context.tools_external_calls[0]
    assert prefix[0] == "/usr/bin/arapuca"
    assert prefix[-1] == "--"
    assert command[0].endswith("/rm")
