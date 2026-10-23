"""Tests for the Strata inference service integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import config, inference, status
from arkai.inference_backend import get_backend, get_backend_names
from arkai.inference_strata import StrataBackend


def test_strata_is_registered_and_configurable() -> None:
    """Strata is a built-in backend and config validation recognizes it."""
    assert "strata" in get_backend_names()
    assert get_backend("strata").name == "strata"
    assert config.validate_config({"inference": {"backend": "strata", "model": "q2_0"}})


def test_strata_builds_local_launch_command() -> None:
    """Strata receives the Arkai model, context, port and loopback access only."""
    with patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"):
        command = StrataBackend().build_command("strata", "q2_0", 9090, -1, 65536)

    assert command == [
        "/bin/strata",
        "launch",
        "--model",
        "q2_0",
        "--context",
        "65536",
        "--access",
        "local",
        "--port",
        "9090",
        "--tune",
        "skip",
        "--no-tui",
        "--output",
        "json",
    ]


def test_strata_respects_configured_cli_path() -> None:
    """A configured CLI path is not searched for on PATH."""
    with patch(
        "arkai.inference_strata.utils.resolve_binary", return_value="/custom/strata"
    ) as resolve:
        command = StrataBackend().build_command(
            "/custom/strata", "q2_0", 8081, -1, 65536, path_is_configured=True
        )
    assert command[0] == "/custom/strata"
    resolve.assert_called_once_with("/custom/strata", search_path=False)


def test_strata_rejects_unsupported_gpu_layers_and_sampling() -> None:
    """Unsupported generic settings fail instead of being silently discarded."""
    backend = StrataBackend()
    with pytest.raises(RuntimeError, match="selects GPU execution automatically"):
        backend.build_command("strata", "q2_0", 8081, 8, 65536)
    with pytest.raises(RuntimeError, match="sampling defaults"):
        backend.build_command("strata", "q2_0", 8081, -1, 65536, sampling={"temperature": 0.7})


def test_strata_builds_safe_plan_status_and_profile_scoped_stop_commands() -> None:
    """Control commands are plan-only or scoped to one known launch profile."""
    backend = StrataBackend()
    command = ["/bin/strata", "launch", "--model", "q2_0"]
    assert backend.build_plan_command(command) == [
        "/bin/strata",
        "launch",
        "--plan",
        "--model",
        "q2_0",
    ]
    with patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"):
        assert backend.build_status_command("/bin/strata") == [
            "/bin/strata",
            "launch",
            "--status",
            "--output",
            "json",
        ]
        assert backend.build_stop_command("/bin/strata", "launch-my-model") == [
            "/bin/strata",
            "launch",
            "--stop",
            "--preset",
            "my-model",
        ]
        with pytest.raises(RuntimeError, match="Invalid Strata launch profile"):
            backend.build_stop_command("/bin/strata", "other-profile")


def test_strata_profile_status_requires_exact_profile() -> None:
    """Only ready/starting entries for the exact profile count as running."""
    status = json.dumps(
        {
            "ok": True,
            "running": [
                {"profile": "launch-one", "state": "ready"},
                {"profile": "launch-two", "state": "stale"},
            ],
        }
    )
    assert StrataBackend.profile_is_running(status, "launch-one")
    assert not StrataBackend.profile_is_running(status, "launch-two")
    assert not StrataBackend.profile_is_running(status, "launch-other")


def test_strata_profile_status_accepts_human_status_trailer() -> None:
    """Some Strata CLI versions append a human-readable status line after JSON."""
    status = (
        '{"schema_version":1,"ok":true,"running":'
        '[{"profile":"launch-one","state":"ready"}]}\n'
        "one: ready at http://127.0.0.1:8081\n"
    )

    assert StrataBackend.profile_is_running(status, "launch-one")


def test_empty_strata_model_list_uses_saved_profile_model(tmp_path: Path) -> None:
    """A loaded Strata profile supplies the model ID when its API list is empty."""
    response = MagicMock()
    response.json.return_value = {"object": "list", "data": []}
    inference.utils.save_yaml(
        str(tmp_path / "inference.state"),
        {"backend": "strata", "model": "q2_0", "port": 8081},
    )

    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.requests, "get", return_value=response),
    ):
        assert inference.get_inference_model(8081) == "q2_0"


def test_empty_strata_model_list_uses_configured_catalog_model(tmp_path: Path) -> None:
    """An externally started Strata service can fall back to Arkai's Strata config."""
    response = MagicMock()
    response.json.return_value = {"object": "list", "data": []}
    cfg = {"inference": {"backend": "strata", "model": "q2_0"}}

    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.requests, "get", return_value=response),
        patch.object(config, "load_config", return_value=cfg),
    ):
        assert inference.get_inference_model(8081) == "q2_0"


def test_empty_model_list_discovers_live_strata_after_arkai_state_was_detached(
    tmp_path: Path,
) -> None:
    """A live managed Strata profile remains usable after Arkai tracking is detached."""
    response = MagicMock()
    response.json.return_value = {"object": "list", "data": []}
    strata_config = tmp_path / "q2_0-128k.json"
    strata_config.write_text(
        json.dumps(
            {
                "model_name": "qwen3.8-flash-next-q2_0",
                "port": 8081,
                "args": ["--max-context", "131072"],
                "log": "/strata/serve.log",
            }
        ),
        encoding="utf-8",
    )
    status = (
        json.dumps(
            {
                "ok": True,
                "running": [
                    {
                        "name": "q2_0-128k",
                        "profile": "launch-q2_0-128k",
                        "state": "ready",
                        "port": 8081,
                        "config": str(strata_config),
                    }
                ],
            }
        )
        + "\nq2_0-128k: ready at http://127.0.0.1:8081\n"
    )

    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(config, "load_config", return_value={"inference": {"backend": "llama-cpp"}}),
        patch("arkai.tools.get_managed_strata_binary", return_value=Path("/tools/strata")),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/tools/strata"),
        patch.object(inference.utils, "run_command", return_value=(0, status, "")),
        patch.object(inference.requests, "get", return_value=response),
    ):
        assert inference.get_inference_model(8081) == "qwen3.8-flash-next-q2_0"
        live_state = inference.get_live_strata_state(8081)

    assert live_state is not None
    assert live_state["strata_profile"] == "launch-q2_0-128k"
    assert live_state["model"] == "qwen3.8-flash-next-q2_0"
    assert live_state["context_size"] == 131072
    assert live_state["strata_owned"] is False


def test_empty_non_strata_model_list_remains_invalid(tmp_path: Path) -> None:
    """Do not hide an empty model list from other OpenAI-compatible servers."""
    response = MagicMock()
    response.json.return_value = {"object": "list", "data": []}
    cfg = {"inference": {"backend": "llama-cpp"}}

    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.requests, "get", return_value=response),
        patch.object(config, "load_config", return_value=cfg),
        pytest.raises(RuntimeError, match="invalid model list"),
    ):
        inference.get_inference_model(8081)


def _config() -> dict:
    return {
        "inference": {
            "backend": "strata",
            "model": "q2_0",
            "port": 9090,
            "context_size": 65536,
            "gpu_layers": -1,
            "startup_timeout": 1,
        }
    }


def _plan(gb: float = 0, engine_usable: bool = True) -> str:
    return json.dumps(
        {
            "ok": True,
            "plan": {
                "download": {
                    "gb": gb,
                    "engine_usable": engine_usable,
                    "engine_missing": not engine_usable,
                    "text": f"About {gb} GB to download",
                },
                "warnings": [],
            },
        }
    )


def _launch_result(reused: bool = False) -> str:
    return json.dumps(
        {
            "ok": True,
            "serving": True,
            "profile": "launch-q2-0-64k",
            "reused": reused,
            "urls": ["http://127.0.0.1:9090"],
        }
    )


def test_strata_start_requires_opt_in_for_provisioning(tmp_path: Path) -> None:
    """A missing model or engine is not downloaded without the explicit flag."""
    cfg = _config()
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference, "is_inference_running", return_value=False),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "run_command", return_value=(0, _plan(65), "")),
        patch.object(inference.subprocess, "run") as launch,
    ):
        with pytest.raises(RuntimeError, match="arkai inference start --allow-provisioning"):
            inference.cmd_inference_start()

    launch.assert_not_called()
    assert not (tmp_path / "inference.state").exists()


@pytest.mark.parametrize("reused", [False, True])
def test_strata_start_records_profile_ownership(tmp_path: Path, reused: bool) -> None:
    """Arkai records whether Strata started or reused the returned profile."""
    cfg = _config()
    process = MagicMock(returncode=0, stdout=_launch_result(reused), stderr="")
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference, "is_inference_running", return_value=False),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "run_command", return_value=(0, _plan(), "")),
        patch.object(inference, "_is_inference_server_healthy", return_value=True),
        patch.object(inference.subprocess, "run", return_value=process) as launch,
    ):
        inference.cmd_inference_start()

    command = launch.call_args.args[0]
    assert command[0:2] == ["/bin/strata", "launch"]
    assert "--access" in command and command[command.index("--access") + 1] == "local"
    assert "--yes" not in command
    state = inference.utils.load_yaml(str(tmp_path / "inference.state"))
    assert state["strata_profile"] == "launch-q2-0-64k"
    assert state["strata_owned"] == (not reused)


def test_strata_allow_provisioning_is_forwarded_to_strata(tmp_path: Path) -> None:
    """The explicit approval is passed through to Strata's launch command."""
    cfg = _config()
    process = MagicMock(returncode=0, stdout=_launch_result(), stderr="")
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference, "is_inference_running", return_value=False),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "run_command", return_value=(0, _plan(65), "")),
        patch.object(inference, "_is_inference_server_healthy", return_value=True),
        patch.object(inference.subprocess, "run", return_value=process) as launch,
    ):
        inference.cmd_inference_start(allow_provisioning=True)

    assert "--yes" in launch.call_args.args[0]


def test_strata_start_reuses_matching_arkai_state_without_relaunch(tmp_path: Path) -> None:
    """A healthy profile already tracked by Arkai is not launched twice."""
    cfg = _config()
    inference.utils.save_yaml(
        str(tmp_path / "inference.state"),
        {
            "backend": "strata",
            "model": "q2_0",
            "context_size": 65536,
            "port": 9090,
            "strata_path": "/bin/strata",
            "strata_profile": "launch-q2-0-64k",
            "strata_owned": True,
        },
    )
    status = json.dumps({"ok": True, "running": [{"profile": "launch-q2-0-64k", "state": "ready"}]})
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "run_command", return_value=(0, status, "")),
        patch.object(inference.subprocess, "run") as launch,
    ):
        inference.cmd_inference_start()

    launch.assert_not_called()


def test_strata_does_not_replace_a_different_running_profile(tmp_path: Path) -> None:
    """A different Arkai-tracked Strata profile on the port is left untouched."""
    cfg = _config()
    inference.utils.save_yaml(
        str(tmp_path / "inference.state"),
        {
            "backend": "strata",
            "model": "iq3_xxs",
            "context_size": 65536,
            "port": 9090,
            "strata_path": "/bin/strata",
            "strata_profile": "launch-iq3-xxs-64k",
            "strata_owned": True,
        },
    )
    status = json.dumps(
        {"ok": True, "running": [{"profile": "launch-iq3-xxs-64k", "state": "ready"}]}
    )
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "run_command", return_value=(0, status, "")),
        patch.object(inference.subprocess, "run") as launch,
        pytest.raises(RuntimeError, match="different Arkai-managed Strata profile"),
    ):
        inference.cmd_inference_start()

    launch.assert_not_called()


def test_strata_start_does_not_restart_conflicting_service(tmp_path: Path) -> None:
    """A Strata CLI refusal for a port conflict is surfaced without a stop attempt."""
    cfg = _config()
    failed = json.dumps({"ok": False, "error": {"message": "another model is serving"}})
    with (
        patch.object(config, "load_config", return_value=cfg),
        patch.object(inference, "is_inference_running", return_value=False),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "run_command", return_value=(0, _plan(), "")),
        patch.object(
            inference.subprocess,
            "run",
            return_value=MagicMock(returncode=2, stdout=failed, stderr=""),
        ),
        pytest.raises(RuntimeError, match="another model is serving"),
    ):
        inference.cmd_inference_start()

    assert not (tmp_path / "inference.state").exists()


def test_strata_stop_detaches_reused_profile_without_stopping_it(tmp_path: Path) -> None:
    """Arkai detaches from a borrowed service without stopping it."""
    state_path = tmp_path / "inference.state"
    inference.utils.save_yaml(
        str(state_path),
        {
            "backend": "strata",
            "strata_profile": "launch-existing",
            "strata_path": "/bin/strata",
            "strata_owned": False,
        },
    )
    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(inference.utils, "run_command") as run_command,
    ):
        inference.cmd_inference_stop()

    run_command.assert_not_called()
    assert not state_path.exists()


def test_strata_stop_uses_profile_scope_and_verifies_status(tmp_path: Path) -> None:
    """Arkai stops only its recorded profile and clears state after confirmation."""
    state_path = tmp_path / "inference.state"
    inference.utils.save_yaml(
        str(state_path),
        {
            "backend": "strata",
            "strata_profile": "launch-owned",
            "strata_path": "/bin/strata",
            "strata_owned": True,
        },
    )
    with (
        patch.object(inference.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch("arkai.inference_strata.utils.resolve_binary", return_value="/bin/strata"),
        patch.object(
            inference.utils,
            "run_command",
            side_effect=[(0, "Stopped owned.", ""), (0, '{"ok":true,"running":[]}', "")],
        ) as run_command,
    ):
        inference.cmd_inference_stop()

    assert run_command.call_args_list[0].args[0] == [
        "/bin/strata",
        "launch",
        "--stop",
        "--preset",
        "owned",
    ]
    assert not state_path.exists()


def test_global_status_reports_strata_profile_and_ownership(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The aggregate Arkai status command does not report a Strata PID as None."""
    inference.utils.save_yaml(
        str(tmp_path / "inference.state"),
        {
            "backend": "strata",
            "model": "q2_0",
            "port": 9090,
            "strata_profile": "launch-q2-0-64k",
            "strata_owned": False,
        },
    )
    with (
        patch.object(status.utils, "get_pid_dir", return_value=str(tmp_path)),
        patch.object(status.inference_module, "is_inference_running", return_value=True),
        patch.object(status.wtmcp_module, "_get_running_instances", return_value=[]),
        patch.object(status.vectordb_module, "is_vectordb_running", return_value=False),
    ):
        status.cmd_status()

    output = capsys.readouterr().out
    assert "Strata profile launch-q2-0-64k; reused" in output
    assert "PID None" not in output
