"""Isolated tests for Splash model-store routing and bulk cache management."""

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arkai import config, inference, model, model_splash
from arkai.model_store import ModelEntry


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never inspect or mutate the user's real Splash installation in these tests."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "load_config", lambda: {"inference": {"backend": "llama-cpp"}})
    return tmp_path


def test_cache_environment_is_shared_and_does_not_mutate_parent(home: Path, monkeypatch) -> None:
    """Both native downloads and serving use Arkai-owned caches by default."""
    monkeypatch.setenv("HF_HUB_CACHE", "/external/hub")
    monkeypatch.setenv("SPLASH_WEIGHT_CACHE", "/external/weights")
    before = dict(os.environ)
    environment = model_splash.splash_environment({}, create=True)
    base = home / ".local/share/arkai/models/splash"
    assert environment["HF_HUB_CACHE"] == str(base / "hub")
    assert environment["SPLASH_WEIGHT_CACHE"] == str(base / "weights")
    assert (base / "hub").is_dir()
    assert (base / "weights").is_dir()
    assert dict(os.environ) == before


def test_inventory_does_not_create_caches_or_require_splash(home: Path) -> None:
    """An absent optional installation is an empty inventory, not a subprocess launch."""
    with patch.object(model_splash, "discover_installation") as discover:
        assert model_splash.SplashModelStore({}).list_models() == []
    discover.assert_not_called()
    assert not (home / ".local").exists()


def test_cache_overrides_are_absolute_and_read_only(home: Path, monkeypatch) -> None:
    """Discovery resolves custom paths without creating directories."""
    monkeypatch.chdir(home)
    cfg = {"inference": {"splash": {"hub_cache": "other/hub", "weight_cache": "other/weights"}}}
    environment = model_splash.splash_environment(cfg)
    assert environment["HF_HUB_CACHE"] == str(home / "other/hub")
    assert not (home / "other").exists()


def test_download_and_remove_use_configured_backend(home: Path, monkeypatch, capsys) -> None:
    """Native model management does not require --backend when Splash is configured."""
    cfg = {"inference": {"backend": "splash", "splash": {"hub_cache": str(home / "hub")}}}
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    store = MagicMock()
    store.download.return_value = home / "selection"
    with patch.object(model_splash, "SplashModelStore", return_value=store):
        model.cmd_model_download("hf:owner/model")
        model.cmd_model_remove("hf:owner/model", installation_id="id")
    store.download.assert_called_once_with("hf:owner/model")
    store.remove.assert_called_once_with("hf:owner/model", "id")
    assert "Downloaded" in capsys.readouterr().out


def test_explicit_legacy_backend_preserves_provider_management(home: Path, monkeypatch) -> None:
    """Users can opt out of a configured Splash backend without changing their config."""
    monkeypatch.setattr(config, "load_config", lambda: {"inference": {"backend": "splash"}})
    with patch.object(model.providers, "download_model", return_value=home) as download:
        model.cmd_model_download("hf:owner/model", backend="llama-cpp")
    download.assert_called_once_with("hf:owner/model")


def test_model_list_includes_native_and_existing_models(home: Path, monkeypatch, capsys) -> None:
    """Unfiltered listing includes native models regardless of the configured backend."""
    legacy = home / ".local/share/arkai/models/local.gguf"
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    entry = ModelEntry("hf:owner/splash-model", "installation", home / "selection", True)
    with (
        patch.object(model_splash.SplashModelStore, "list_models", return_value=[entry]),
        patch.object(
            model.providers, "list_provider_models", return_value=[("hf", "owner/git-model", "1G")]
        ),
    ):
        model.cmd_model_list()
    output = capsys.readouterr().out
    assert "local.gguf" in output
    assert "hf:owner/git-model" in output
    assert "hf:owner/splash-model" in output
    assert "hf:owner/git-model\n\nSplash models:" in output


def test_native_list_filter_does_not_inspect_legacy_cache(home: Path, capsys) -> None:
    """An explicit selector lists just the requested store."""
    with (
        patch.object(model_splash.SplashModelStore, "list_models", return_value=[]),
        patch.object(model.providers, "list_provider_models") as legacy,
    ):
        model.cmd_model_list(backend="splash")
    legacy.assert_not_called()
    assert "No models found" in capsys.readouterr().out


def test_native_prune_requires_owned_caches(home: Path) -> None:
    """A configured external cache must never be collected as Arkai-owned data."""
    store = model_splash.SplashModelStore(
        {"inference": {"splash": {"hub_cache": str(home / "external")}}}
    )
    with patch.object(store, "call", return_value={}) as call:
        store.prune(dry_run=True)
    assert call.call_args.kwargs["owned"] is False


def test_resolve_uses_native_installation_identity(home: Path) -> None:
    """Resolution is local and refuses ambiguity through the same versioned bridge."""
    store = model_splash.SplashModelStore({})
    with patch.object(store, "call", return_value={"path": str(home / "selection")}) as call:
        assert store.resolve("hf:owner/model", "id") == home / "selection"
    call.assert_called_once_with("resolve", model="owner/model", installation_id="id")


@pytest.fixture
def transport(home: Path, monkeypatch):
    """Use a fake child process while keeping the bridge transport validation real."""
    monkeypatch.setattr(
        model_splash,
        "discover_installation",
        lambda cfg: model_splash.SplashInstallation(home, Path(sys.executable)),
    )
    process = MagicMock(pid=12345, returncode=0)
    process.__enter__.return_value = process
    process.communicate.return_value = ('{"version":1,"path":"/selection"}', "")
    monkeypatch.setattr(model_splash.subprocess, "Popen", MagicMock(return_value=process))
    return process


def test_download_calls_native_installer_without_starting_inference(transport, monkeypatch) -> None:
    """Provision all dependencies rather than cloning a target-only Git repository."""
    monkeypatch.setattr("arkai.inference_splash.SplashBackend.check_environment", lambda self: None)
    assert model_splash.SplashModelStore({}).download("hf:owner/model") == Path("/selection")
    request = json.loads(transport.communicate.call_args.args[0])
    assert request["operation"] == "download"
    assert transport.communicate.call_args.kwargs["timeout"] is None


@pytest.mark.parametrize("output", ["invalid", "[]", '{"version":99}'])
def test_bridge_rejects_bad_protocol(transport, output: str) -> None:
    """Unknown response formats cannot become filesystem-management instructions."""
    transport.communicate.return_value = (output, "")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        model_splash.SplashModelStore({}).call("info", model="owner/model")


def test_bridge_redacts_credentials(transport, monkeypatch) -> None:
    """Errors may include Hub diagnostics but must not expose an inherited token."""
    monkeypatch.setenv("HF_TOKEN", "private-token")
    transport.returncode = 1
    transport.communicate.return_value = ('{"version":1,"error":"denied private-token"}', "")
    with pytest.raises(RuntimeError, match="denied \\[redacted\\]"):
        model_splash.SplashModelStore({}).call("info", model="owner/model")


def test_cancelled_download_kills_its_group(transport) -> None:
    """Cancellation leaves no installer or hardware-check descendants running."""
    transport.communicate.side_effect = KeyboardInterrupt
    with patch.object(model_splash.os, "killpg") as kill:
        with pytest.raises(KeyboardInterrupt):
            model_splash.SplashModelStore({}).call("download", model="owner/model")
    kill.assert_called_once_with(transport.pid, model_splash.signal.SIGKILL)


def test_inference_uses_download_cache_and_source_options(home: Path, monkeypatch) -> None:
    """Serving must use the same cache paths and selection options as model downloads."""
    cfg = {
        "inference": {
            "backend": "splash",
            "model": "hf:owner/model",
            "port": 9090,
            "startup_timeout": 1,
            "splash": {"revision": "abc", "language_only": True},
        }
    }
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    process = MagicMock(pid=12345)
    process.poll.return_value = None
    with (
        patch.object(inference, "is_inference_running", return_value=False),
        patch.object(inference.utils, "get_pid_dir", return_value=str(home / "state")),
        patch.object(inference.utils, "detect_gpu", return_value="metal"),
        patch.object(inference.utils, "is_port_in_use", return_value=False),
        patch.object(inference.utils, "resolve_binary", return_value="/bin/splash"),
        patch("arkai.inference_splash.SplashBackend.check_environment"),
        patch.object(inference, "_is_inference_server_healthy", return_value=True),
        patch.object(inference.subprocess, "Popen", return_value=process) as popen,
        patch.object(inference.signal, "signal"),
    ):
        inference.cmd_inference_start()
    command = popen.call_args.args[0]
    assert command[command.index("--revision") + 1] == "abc"
    assert "--language-only" in command
    expected = model_splash.splash_environment(cfg)
    assert popen.call_args.kwargs["env"]["HF_HUB_CACHE"] == expected["HF_HUB_CACHE"]
    assert popen.call_args.kwargs["env"]["SPLASH_WEIGHT_CACHE"] == expected["SPLASH_WEIGHT_CACHE"]


def test_model_cli_defaults_and_store_selection(home: Path, monkeypatch) -> None:
    """Download, resolution, removal, and pruning are exposed without mandatory overrides."""
    parser = argparse.ArgumentParser()
    model.ingest_cli_options(parser.add_subparsers(dest="command"))
    cfg = {"inference": {"backend": "splash"}}
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    store = MagicMock()
    store.download.return_value = store.resolve.return_value = home / "selection"
    store.list_models.return_value = []
    store.prune.return_value = {"paths": []}
    commands = [
        ["model", "download", "hf:owner/model"],
        ["model", "resolve", "hf:owner/model"],
        ["model", "remove", "hf:owner/model", "--id", "installation"],
        ["model", "prune", "--dry-run"],
        ["model", "list", "--backend", "splash"],
    ]
    with patch.object(model_splash, "SplashModelStore", return_value=store):
        for command in commands:
            model.exec_cmd(parser.parse_args(command))
    store.download.assert_called_once_with("hf:owner/model")
    store.resolve.assert_called_once_with("hf:owner/model", None)
    store.remove.assert_called_once_with("hf:owner/model", "installation")
    store.prune.assert_called_once_with(True)


def test_local_resolution_and_non_native_id_errors(home: Path, capsys) -> None:
    """Legacy stores retain normal local/provider resolution rather than using native metadata."""
    local = home / ".local/share/arkai/models/model.gguf"
    local.parent.mkdir(parents=True)
    local.touch()
    model.cmd_model_resolve("model.gguf")
    assert str(local) in capsys.readouterr().out
    with patch.object(model.providers, "resolve_model", return_value=local) as resolve:
        model.cmd_model_resolve("hf:owner/model")
    resolve.assert_called_once_with("hf:owner/model")
    for callback in (model.cmd_model_resolve, model.cmd_model_remove):
        with pytest.raises(RuntimeError, match="--id"):
            callback("hf:owner/model", installation_id="native-id")
    with pytest.raises(RuntimeError, match="filename"):
        model.cmd_model_resolve("../outside")
    with pytest.raises(RuntimeError, match="not found"):
        model.cmd_model_resolve("missing.gguf")
    with pytest.raises(RuntimeError, match="pruning"):
        model.cmd_model_prune(True)


def test_native_list_errors_do_not_hide_other_models(home: Path, capsys) -> None:
    """An optional unknown installation cannot break the legacy model inventory."""
    with patch.object(
        model_splash.SplashModelStore,
        "list_models",
        side_effect=RuntimeError("unsupported release"),
    ):
        model.cmd_model_list()
        assert "unsupported release" in capsys.readouterr().err
        with pytest.raises(RuntimeError, match="unsupported"):
            model.cmd_model_list("splash")


@pytest.mark.parametrize(
    "settings",
    [
        "invalid",
        {"unexpected": 1},
        {"language_only": "yes"},
        {"max_memory": "0G"},
        {"kv_format": "q4"},
        {"revision": ""},
        {"default_reasoning_effort": "bad"},
    ],
)
def test_invalid_splash_settings_are_rejected(settings) -> None:
    """Malformed source or cache settings cannot reach the native installer."""
    with pytest.raises(ValueError):
        model_splash.splash_options({"inference": {"splash": settings}})
    assert not config.validate_config(
        {"inference": {"backend": "splash", "model": "hf:owner/model", "splash": settings}}
    )


def test_cache_creation_failure_is_actionable(home: Path) -> None:
    """A file in place of a cache directory fails before the downloader runs."""
    file = home / "blocked"
    file.touch()
    with pytest.raises(RuntimeError, match="Splash cache"):
        model_splash.splash_environment(
            {"inference": {"splash": {"hub_cache": str(file)}}}, create=True
        )


@pytest.mark.parametrize(
    "reference",
    [
        "local.gguf",
        "ollama:owner/model",
        "hf:../model",
        "hf:owner/model:",
        "hf:owner-/model",
        "hf:owner/model.git",
    ],
)
def test_invalid_model_references_are_rejected(reference: str) -> None:
    """Model identifiers cannot escape a cache or be mistaken for another provider."""
    with pytest.raises(ValueError, match="Splash"):
        model_splash.splash_model_id(reference)


def test_discovery_is_relocatable_and_independent_of_other_backend_path(home: Path) -> None:
    """Listing Splash while llama.cpp is configured must not inspect llama.cpp's program root."""
    root = home / "package/libexec"
    (root / "install").mkdir(parents=True)
    (root / "install/models.py").touch()
    (root / "python/bin").mkdir(parents=True)
    (root / "python/bin/python3").touch()
    (root / "release.json").write_text('{"version":"1.1.0"}')
    with patch.object(
        model_splash.utils, "resolve_binary", return_value=str(home / "package/bin/splash")
    ) as resolve:
        installed = model_splash.discover_installation(
            {"inference": {"backend": "llama-cpp", "path": "/llama-server"}}
        )
    resolve.assert_called_once_with("splash")
    assert installed.root == root
    assert installed.python == root / "python/bin/python3"


@pytest.mark.parametrize("release", [None, "invalid", '{"version":"2.0.0"}', '{"version":"1.1.0"}'])
def test_discovery_fails_on_unsupported_or_broken_packages(home: Path, release) -> None:
    """Only the inspected installer format is allowed to perform model management."""
    if release is not None:
        (home / "release.json").write_text(release)
    with pytest.raises(RuntimeError, match="Splash"):
        model_splash.discover_installation(
            {"inference": {"splash": {"installation_root": str(home)}}}
        )


def test_discovery_reports_missing_runtime(home: Path) -> None:
    """An absent optional runtime gives an actionable model-management error."""
    with patch.object(model_splash.utils, "resolve_binary", side_effect=RuntimeError("missing")):
        with pytest.raises(RuntimeError, match="not installed"):
            model_splash.discover_installation({})


def test_native_inventory_protocol_validation(transport, home: Path) -> None:
    """Inventory must have validated fields and can be inspected without cache creation."""
    (home / "Library/Application Support/Splash/models").mkdir(parents=True)
    entry = {
        "reference": "hf:owner/model",
        "installation_id": "id",
        "path": "/selection",
        "complete": True,
    }
    transport.communicate.return_value = (json.dumps({"version": 1, "models": [entry]}), "")
    assert model_splash.SplashModelStore({}).list_models()[0].reference == "hf:owner/model"
    transport.communicate.return_value = ('{"version":1,"models":false}', "")
    with pytest.raises(RuntimeError, match="inventory"):
        model_splash.SplashModelStore({}).list_models()
    with pytest.raises(RuntimeError, match="inventory"):
        ModelEntry.from_mapping({**entry, "owned_bytes": -1})


def test_download_diagnostics_are_redacted(transport, monkeypatch, capsys) -> None:
    """Successful transfers must not print credentials from native diagnostics."""
    monkeypatch.setenv("HF_TOKEN", "private-token")
    transport.communicate.return_value = (
        '{"version":1,"path":"/selection"}',
        "fetched private-token",
    )
    model_splash.SplashModelStore({}).call("download", model="owner/model")
    assert "private-token" not in capsys.readouterr().out


def test_source_options_translate_local_draft_and_reasoning(home: Path) -> None:
    """Translate native settings, keeping cache and installation paths environment-only."""
    draft = home / "draft"
    draft.mkdir()
    options = {
        "draft_model": str(draft),
        "max_memory": "28G",
        "max_cache_disk": "0",
        "kv_format": "bf16",
        "default_reasoning_effort": "none",
    }
    args = model_splash.serve_arguments({"inference": {"splash": options}})
    assert args[args.index("--draft-model") + 1] == str(draft)
    assert args[args.index("--max-cache-disk") + 1] == "0"
    assert args[args.index("--default-reasoning-effort") + 1] == "none"
