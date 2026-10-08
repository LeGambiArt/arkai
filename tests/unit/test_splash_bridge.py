"""Offline fixtures for native Splash installation and shared-cache safety."""

import fcntl
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arkai import splash_bridge
from arkai.splash_bridge import SplashBridge


@pytest.fixture
def native(tmp_path, monkeypatch):
    """Replace the installer API, never reading the user's real installation."""
    models_root = tmp_path / "models"
    runtime = tmp_path / "runtime"
    models_root.mkdir()
    runtime.mkdir()

    def verify(path, **kwargs):
        record = json.loads((path / "model.json").read_text())
        for entry in record["files"].values():
            if not Path(entry["path"]).is_file():
                raise RuntimeError("incomplete")
        return record

    api = SimpleNamespace(
        paths=SimpleNamespace(MODELS=models_root, RUNTIME=runtime, BINARY=tmp_path / "engine"),
        models=SimpleNamespace(
            ModelError=RuntimeError,
            read_json=lambda path: json.loads(path.read_text()),
            parse_draft_model=lambda value: value,
            Selection=SimpleNamespace(
                of=MagicMock(return_value=SimpleNamespace(link=models_root / "selection"))
            ),
        ),
        assembly=SimpleNamespace(
            _well_formed=lambda r: r.get("version") == 1 and "files" in r,
            verify=verify,
            is_held=lambda path: False,
        ),
        hub=SimpleNamespace(pin_owner=lambda link: link.name),
        legacy=SimpleNamespace(verify=MagicMock()),
        upstream=SimpleNamespace(prepare=MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            scan_cache_dir=MagicMock(return_value=SimpleNamespace(warnings=[], repos=[]))
        ),
    )
    return SplashBridge(api), tmp_path


def selection(bridge, tmp_path, name="owner/model:Q4", draft="shared"):
    """Create a synthetic immutable assembly and its target/draft snapshot pins."""
    files = {}
    for role, repository in (("target", name.split(":")[0]), ("draft", f"owner/{draft}")):
        snapshot = (
            tmp_path
            / "hub"
            / ("models--" + repository.replace("/", "--"))
            / "snapshots"
            / ("a" * 40)
        )
        snapshot.mkdir(parents=True, exist_ok=True)
        weight = snapshot / "model.safetensors"
        weight.write_bytes(b"weights")
        files[f"{role}/model.safetensors"] = {"path": str(weight)}
    record = {
        "version": 1,
        "model": name,
        "vision_format": "none",
        "sources": {
            "target": {"repo": name.split(":")[0], "revision": "a" * 40},
            "draft": {"repo": f"owner/{draft}", "revision": "a" * 40},
        },
        "files": files,
    }
    assembly = bridge.models_root / ".resolved" / name.replace("/", "-")
    assembly.mkdir(parents=True)
    (assembly / "model.json").write_text(json.dumps(record))
    link = bridge.models_root / name
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(assembly, target_is_directory=True)
    for entry in files.values():
        snapshot = Path(entry["path"]).parent
        pin = snapshot.parent.parent / "refs/splash" / link.name / snapshot.name
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(snapshot.name)
    return link, record


def test_inventory_reports_shared_dependencies(native):
    """List installations created outside Arkai without double-counting shared files."""
    bridge, tmp = native
    first, _ = selection(bridge, tmp)
    selection(bridge, tmp, "owner/other:Q4")
    entry = next(item for item in bridge.list_models() if item["path"] == str(first))
    assert entry["reference"] == "hf:owner/model:Q4"
    assert entry["complete"]
    assert entry["shared_bytes"] == entry["owned_bytes"] == 7
    assert entry["sources"]["draft"]["repo"] == "owner/shared"


def test_hashed_selections_are_listed(native):
    """Source-option and revision selections remain visible with their actual model IDs."""
    bridge, tmp = native
    link, _ = selection(bridge, tmp)
    hashed = bridge.models_root / ".selections" / ("b" * 64)
    hashed.parent.mkdir()
    link.rename(hashed)
    assert bridge.list_models()[0]["reference"] == "hf:owner/model:Q4"


def test_incomplete_sources_are_visible_but_not_resolvable(native):
    """Interrupted installation data must not be presented as ready to serve."""
    bridge, tmp = native
    _, record = selection(bridge, tmp)
    Path(record["files"]["target/model.safetensors"]["path"]).unlink()
    assert bridge.list_models()[0]["complete"] is False
    with pytest.raises(RuntimeError, match="incomplete"):
        bridge.resolve("owner/model:Q4", None)


def test_resolve_complete_model_and_ambiguity(native):
    """Resolution is local, retains source files, and requires IDs for ambiguous references."""
    bridge, tmp = native
    link, _ = selection(bridge, tmp)
    assert bridge.dispatch({"operation": "resolve", "model": "owner/model:Q4"})["path"] == str(link)
    alternate = bridge.models_root / ".selections" / ("b" * 64)
    alternate.parent.mkdir()
    alternate.symlink_to(link.resolve())
    with pytest.raises(RuntimeError, match="--id"):
        bridge.resolve("owner/model:Q4", None)
    entry = next(item for item in bridge.list_models() if item["path"] == str(alternate))
    assert bridge.resolve("owner/model:Q4", entry["installation_id"])["path"] == str(alternate)


def test_unknown_schema_fails_closed(native):
    """Do not use unknown dependency metadata for filesystem mutations."""
    bridge, tmp = native
    link, record = selection(bridge, tmp)
    record["version"] = 99
    (link / "model.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="schema"):
        bridge.list_models()


def test_removal_retains_other_installations_and_shared_sources(native):
    """Only the chosen link and its own pins are removed, not target or draft files."""
    bridge, tmp = native
    link, record = selection(bridge, tmp)
    other, _ = selection(bridge, tmp, "owner/other:Q4")
    bridge.remove("owner/model:Q4", None)
    assert not link.is_symlink()
    assert other.is_symlink()
    assert all(Path(entry["path"]).is_file() for entry in record["files"].values())
    repository = Path(record["files"]["draft/model.safetensors"]["path"]).parent.parent.parent
    assert not (repository / "refs/splash" / link.name / ("a" * 40)).exists()
    assert (repository / "refs/splash" / other.name / ("a" * 40)).exists()


@pytest.mark.parametrize("lock_name", ["serve.lock", ".install.lock"])
def test_removal_refuses_live_server_or_installer(native, lock_name):
    """Honor native locks, including processes started outside Arkai."""
    bridge, tmp = native
    link, _ = selection(bridge, tmp)
    lock_path = (
        bridge.runtime / lock_name if lock_name == "serve.lock" else bridge.models_root / lock_name
    )
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH if lock_name == "serve.lock" else fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="running|busy"):
            bridge.remove("owner/model:Q4", None)
    assert link.is_symlink()


def test_removal_requires_explicit_installation_id(native):
    """Never guess between revisions/options of the same model reference."""
    bridge, tmp = native
    link, _ = selection(bridge, tmp)
    other = bridge.models_root / ".selections" / ("b" * 64)
    other.parent.mkdir()
    other.symlink_to(link.resolve())
    with pytest.raises(RuntimeError, match="--id"):
        bridge.remove("owner/model:Q4", None)
    entry = next(item for item in bridge.list_models() if item["path"] == str(other))
    bridge.remove("owner/model:Q4", entry["installation_id"])
    assert not other.is_symlink()
    assert link.is_symlink()


def test_nonselection_directories_are_not_removed(native):
    """Never treat arbitrary user folders as native model installations."""
    bridge, _ = native
    folder = bridge.models_root / "owner/model:Q4"
    folder.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="not found"):
        bridge.remove("owner/model:Q4", None)
    assert folder.is_dir()


def test_prune_preserves_live_and_unknown_prepared_entries(native):
    """Preview and execution collect only recognized entries with unused sources."""
    bridge, tmp = native
    _, record = selection(bridge, tmp)
    bridge.remove("owner/model:Q4", None)
    cache = tmp / "weights"
    cache.mkdir()
    live, dead, unknown = [cache / (key * 64) for key in ("a", "b", "c")]
    for directory in (live, dead, unknown):
        directory.mkdir()
        (directory / "weights").write_bytes(b"prepared")
        (directory / "sha256").write_text("d" * 64)
    (live / "source").write_text(
        record["files"]["target/model.safetensors"]["path"] + "\nlayer-0.bin\n"
    )
    (dead / "source").write_text(f"{tmp / 'missing'}\nlayer-0.bin\n")
    (unknown / "source").write_text("future-unknown-format\n")
    preview = bridge.prune(tmp / "hub", cache, True, True)
    assert str(dead) in preview["paths"]
    assert str(live) not in preview["paths"]
    assert str(unknown) not in preview["paths"]
    assert dead.exists()
    bridge.prune(tmp / "hub", cache, False, True)
    assert not dead.exists()
    assert live.exists() and unknown.exists()


def test_external_cache_pruning_is_refused(native):
    """Custom cache paths do not imply ownership of their existing contents."""
    bridge, tmp = native
    with pytest.raises(RuntimeError, match="external"):
        bridge.prune(tmp / "hub", tmp / "weights", True, False)


def test_prepared_weights_retained_while_other_models_installed(native):
    """Content sharing across source paths makes provenance-only deletion unsafe."""
    bridge, tmp = native
    selection(bridge, tmp)
    assert bridge.prune(tmp / "hub", tmp / "weights", True, True)["paths"] == []


def test_pin_validation_precedes_removal(native):
    """Refuse tampered pins without partially uninstalling a selection."""
    bridge, tmp = native
    link, record = selection(bridge, tmp)
    source = Path(record["files"]["target/model.safetensors"]["path"])
    pin = source.parent.parent.parent / "refs/splash" / link.name / ("a" * 40)
    pin.write_text("invalid")
    with pytest.raises(RuntimeError, match="invalid.*pin"):
        bridge.remove("owner/model:Q4", None)
    assert link.is_symlink()


def test_held_assembly_cannot_be_removed(native):
    """Protect a held assembly independently of the global serve lock."""
    bridge, tmp = native
    link, _ = selection(bridge, tmp)
    bridge.api.assembly.is_held = lambda path: True
    with pytest.raises(RuntimeError, match="running"):
        bridge.remove("owner/model:Q4", None)
    assert link.is_symlink()


def test_empty_inventory_does_not_create_native_directory(native):
    """Listing an absent store does not create locks or cache directories."""
    bridge, _ = native
    bridge.models_root.rmdir()
    assert bridge.dispatch({"operation": "list"}) == {"models": []}
    assert not bridge.models_root.exists()


def test_download_only_runs_installer_after_hardware_preflight(native):
    """No server starts and no installer runs on an unsupported device."""
    bridge, _ = native
    request = {
        "operation": "info",
        "model": "owner/model",
        "options": {"draft_model": "owner/draft", "language_only": True},
    }
    bridge.dispatch(request)
    bridge.api.upstream.prepare.assert_not_called()
    request["operation"] = "download"
    with patch.object(
        splash_bridge.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
    ):
        bridge.dispatch(request)
    bridge.api.upstream.prepare.assert_called_once()
    with patch.object(
        splash_bridge.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=1, stderr="unsupported GPU"),
    ):
        with pytest.raises(RuntimeError, match="unsupported GPU"):
            bridge.dispatch(request)
    assert bridge.api.upstream.prepare.call_count == 1


def test_dispatch_remove_and_prune(native):
    """Expose safe operations through the same versioned transport."""
    bridge, tmp = native
    selection(bridge, tmp)
    assert bridge.dispatch({"operation": "remove", "model": "owner/model:Q4"}) == {}
    assert bridge.dispatch(
        {
            "operation": "prune",
            "hub_cache": str(tmp / "hub"),
            "weight_cache": str(tmp / "weights"),
            "dry_run": True,
            "owned": True,
        }
    )["dry_run"]
    with pytest.raises(RuntimeError, match="Unknown"):
        bridge.dispatch({"operation": "unknown", "model": "owner/model"})


def test_load_api_verifies_package_and_import_root(tmp_path, monkeypatch):
    """Do not execute a mismatched installer API version or package root."""
    (tmp_path / "release.json").write_text('{"version":"1.1.0"}')
    monkeypatch.setattr(sys, "path", list(sys.path))
    paths = SimpleNamespace(ROOT=tmp_path)
    with patch.object(
        splash_bridge.importlib,
        "import_module",
        side_effect=lambda name: paths if name == "install.paths" else SimpleNamespace(),
    ):
        assert splash_bridge.load_api(tmp_path).paths.ROOT == tmp_path
        paths.ROOT = tmp_path / "other"
        with pytest.raises(RuntimeError, match="unexpected"):
            splash_bridge.load_api(tmp_path)
    (tmp_path / "release.json").write_text('{"version":"2.0.0"}')
    with pytest.raises(RuntimeError, match="1.1.0"):
        splash_bridge.load_api(tmp_path)


def test_bridge_main_outputs_json_and_redacts_keys(native, monkeypatch, capsys):
    """Structured diagnostics must not expose an inherited Hugging Face token."""
    bridge, tmp = native
    monkeypatch.setattr(splash_bridge, "load_api", lambda root: bridge.api)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"operation":"list"}'))
    assert splash_bridge.main([str(tmp)]) == 0
    assert json.loads(capsys.readouterr().out) == {"version": 1, "models": []}
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setattr(
        sys, "stdin", io.StringIO('{"operation":"private-token","model":"owner/model"}')
    )
    assert splash_bridge.main([str(tmp)]) == 1
    assert "private-token" not in json.loads(capsys.readouterr().out)["error"]


def test_legacy_package_inventory_and_removal(native):
    """Packed packages use their native snapshot rather than an Arkai Git clone."""
    bridge, tmp = native
    target = tmp / "hub/models--owner--legacy/snapshots" / ("c" * 40)
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"weights")
    (target / "manifest.json").write_text("{}")
    link = bridge.models_root / "owner/legacy"
    link.parent.mkdir()
    link.symlink_to(target)
    bridge.api.legacy.validate_manifest = lambda path: {"artifacts": [{"path": "weights.bin"}]}
    entry = bridge.list_models()[0]
    assert entry["reference"] == "hf:owner/legacy"
    assert entry["revision"] == "c" * 40
    bridge.remove("owner/legacy", None)
    assert not link.is_symlink()
    assert (target / "weights.bin").exists()


def test_external_assembly_and_unknown_selection_are_refused(native):
    """Unfamiliar symlinks cannot authorize removal of an external directory."""
    bridge, tmp = native
    link, record = selection(bridge, tmp)
    external = tmp / "external"
    external.mkdir()
    (external / "model.json").write_text(json.dumps(record))
    link.unlink()
    link.symlink_to(external)
    with pytest.raises(RuntimeError, match="Unexpected"):
        bridge.list_models()
    (external / "model.json").unlink()
    with pytest.raises(RuntimeError, match="Incomplete"):
        bridge.list_models()
    assert external.exists()


def test_hub_prune_respects_references_and_preview(native):
    """The Hub deletion strategy only sees unreferenced, unpinned commits."""
    bridge, tmp = native
    hub = tmp / "hub"
    hub.mkdir()
    snapshot = hub / "models--owner--model/snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    revisions = [
        SimpleNamespace(commit_hash="a" * 40, refs=[]),
        SimpleNamespace(commit_hash="b" * 40, refs=["splash/other-installation/commit"]),
    ]
    strategy = SimpleNamespace(snapshots={snapshot}, blobs=set(), repos=set(), execute=MagicMock())
    cache = SimpleNamespace(
        warnings=[],
        repos=[SimpleNamespace(revisions=revisions)],
        delete_revisions=MagicMock(return_value=strategy),
    )
    getattr(sys.modules["huggingface_hub"], "scan_cache_dir").return_value = cache
    assert bridge.prune(hub, tmp / "weights", True, True)["paths"] == [str(snapshot)]
    cache.delete_revisions.assert_called_with("a" * 40)
    strategy.execute.assert_not_called()
    bridge.prune(hub, tmp / "weights", False, True)
    strategy.execute.assert_called_once()


def test_hub_prune_refuses_unrecognized_and_external_paths(native):
    """Never prune unknown cache data or targets outside an owned cache root."""
    bridge, tmp = native
    hub = tmp / "hub"
    hub.mkdir()
    scanner = getattr(sys.modules["huggingface_hub"], "scan_cache_dir")
    scanner.return_value = SimpleNamespace(warnings=["unknown"], repos=[])
    with pytest.raises(RuntimeError, match="unrecognized"):
        bridge.prune(hub, tmp / "weights", True, True)
    strategy = SimpleNamespace(snapshots={tmp / "external"}, blobs=set(), repos=set())
    scanner.return_value = SimpleNamespace(
        warnings=[],
        repos=[SimpleNamespace(revisions=[SimpleNamespace(commit_hash="a" * 40, refs=[])])],
        delete_revisions=lambda *args: strategy,
    )
    with pytest.raises(RuntimeError, match="outside"):
        bridge.prune(hub, tmp / "weights", True, True)


def test_legacy_install_resolve_remove_and_prune_complete_lifecycle(native):
    """Exercise the complete native lifecycle, including main-only cached package reclamation."""
    bridge, tmp = native
    snapshot = tmp / "hub/models--owner--legacy/snapshots" / ("c" * 40)
    link = bridge.models_root / "owner/legacy"
    bridge.api.models.Selection.of.return_value = SimpleNamespace(link=link)

    def install(selection):
        snapshot.mkdir(parents=True)
        (snapshot / "weights.bin").write_bytes(b"weights")
        (snapshot / "manifest.json").write_text("{}")
        link.parent.mkdir()
        link.symlink_to(snapshot)
        pin = snapshot.parent.parent / "refs/splash" / link.name / snapshot.name
        pin.parent.mkdir(parents=True)
        pin.write_text(snapshot.name)

    bridge.api.upstream.prepare.side_effect = install
    bridge.api.legacy.validate_manifest = lambda path: {"artifacts": [{"path": "weights.bin"}]}
    with patch.object(
        splash_bridge.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
    ):
        assert bridge.dispatch({"operation": "download", "model": "owner/legacy"})["path"] == str(
            link
        )
    assert bridge.resolve("owner/legacy", None)["complete"]
    assert bridge.list_models()[0]["owned_bytes"] == 7
    bridge.remove("owner/legacy", None)
    assert not link.is_symlink()
    assert (snapshot / "weights.bin").exists()

    weights = tmp / "weights"
    prepared = weights / ("d" * 64)
    prepared.mkdir(parents=True)
    (prepared / "weights").write_bytes(b"prepared")
    (prepared / "sha256").write_text("e" * 64)
    (prepared / "source").write_text(str(snapshot / "weights.bin") + "\nlayer-0.bin\n")
    strategy = SimpleNamespace(
        snapshots={snapshot},
        blobs=set(),
        repos=set(),
        execute=MagicMock(side_effect=lambda: shutil.rmtree(snapshot)),
    )
    revision = SimpleNamespace(commit_hash=snapshot.name, refs=["main"])
    cache = SimpleNamespace(
        warnings=[],
        repos=[SimpleNamespace(revisions=[revision])],
        delete_revisions=MagicMock(return_value=strategy),
    )
    getattr(sys.modules["huggingface_hub"], "scan_cache_dir").return_value = cache
    preview = bridge.prune(tmp / "hub", weights, True, True)
    assert preview["paths"] == [str(snapshot), str(prepared)]
    assert snapshot.exists() and prepared.exists()
    result = bridge.prune(tmp / "hub", weights, False, True)
    assert result["paths"] == preview["paths"]
    assert not snapshot.exists() and not prepared.exists()
