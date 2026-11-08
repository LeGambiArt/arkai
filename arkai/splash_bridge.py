"""Isolated bridge to the verified Splash 1.1.0 installer and cache format.

Runs in Splash's Python. Only this module imports its private APIs. Standard
output is one JSON response; provisioning diagnostics use standard error.
"""

import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_api(root: Path) -> Any:
    """Load only the supported package layout in the isolated child process."""
    release = json.loads((root / "release.json").read_text())
    if release.get("version") != "1.1.0":
        raise RuntimeError("Only the Splash 1.1.0 installer interface is supported")
    sys.path.insert(0, str(root))
    api = SimpleNamespace(
        **{
            name: importlib.import_module(f"install.{name}")
            for name in ("paths", "models", "assembly", "hub", "legacy", "upstream")
        }
    )
    if Path(api.paths.ROOT).resolve() != root.resolve():
        raise RuntimeError("Splash installer loaded from an unexpected directory")
    return api


@contextmanager
def file_lock(path: Path, shared: bool = False, create: bool = True) -> Generator[None, None, None]:
    """Acquire a native lock without following a symlink or waiting indefinitely."""
    if not create and not path.exists():
        yield
        return
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(path), os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Splash is running or an installation is busy; stop it and retry"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _snapshot(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", parent.name):
            return parent
    return None


class SplashBridge:
    """Manage selections using the installer format and its locking conventions."""

    def __init__(self, api: Any) -> None:
        self.api = api
        self.models_root = Path(api.paths.MODELS).resolve()
        self.runtime = Path(api.paths.RUNTIME).resolve()

    def _links(self) -> list[Path]:
        if not self.models_root.is_dir():
            return []
        # Ordinary owner/model links and .selections/<hash>, never source caches.
        return sorted(
            link
            for parent in self.models_root.iterdir()
            if parent.is_dir() and not parent.is_symlink() and parent.name != ".resolved"
            for link in parent.iterdir()
            if link.is_symlink()
        )

    def _entry(self, link: Path) -> dict:
        target = link.resolve()
        complete = True
        if (link / "model.json").is_file():
            if not _inside(target, self.models_root / ".resolved"):
                raise RuntimeError(f"Unexpected Splash assembly location: {link}")
            record = self.api.models.read_json(link / "model.json")
            if not self.api.assembly._well_formed(record):
                raise RuntimeError(
                    f"Unknown Splash assembly schema at {link}; refusing unsafe operations"
                )
            model = record["model"]
            sources = record["sources"]
            files = [entry["path"] for entry in record["files"].values()]
            if any(not Path(value).is_absolute() or ".." in Path(value).parts for value in files):
                raise RuntimeError(f"Unknown Splash source-path schema at {link}")
            try:
                self.api.assembly.verify(link)
            except (self.api.models.ModelError, OSError):
                complete = False
        elif (link / "manifest.json").is_file():
            model = link.relative_to(self.models_root).as_posix()
            if model.startswith(".") or _snapshot(target) != target:
                raise RuntimeError(f"Unexpected Splash package location: {link}")
            manifest = self.api.legacy.validate_manifest(link / "manifest.json")
            files = [str(target / item["path"]) for item in manifest["artifacts"]]
            sources = {"target": {"repo": model, "revision": target.name}}
            try:
                self.api.legacy.verify(link, model, full=False)
            except (self.api.models.ModelError, OSError):
                complete = False
        else:
            raise RuntimeError(f"Incomplete Splash selection without recognizable metadata: {link}")
        return {
            "reference": "hf:" + model,
            "installation_id": hashlib.sha256(str(link).encode()).hexdigest(),
            "path": str(link),
            "complete": complete,
            "sources": sources,
            "files": files,
            "revision": sources.get("target", {}).get("revision"),
            "owned_bytes": 0,
            "shared_bytes": 0,
        }

    def list_models(self) -> list[dict]:
        """Read native selections without downloading, repairing, or hashing model weights."""
        entries = [self._entry(link) for link in self._links()]
        usage = Counter(
            path
            for entry in entries
            for path in {str(Path(value).resolve()) for value in entry["files"]}
        )
        for entry in entries:
            for path in {Path(value).resolve() for value in entry["files"]}:
                try:
                    size = path.stat().st_size
                except OSError:
                    entry["complete"] = False
                    continue
                entry["shared_bytes" if usage[str(path)] > 1 else "owned_bytes"] += size
        return entries

    def _select(self, model: str, installation_id: str | None) -> dict:
        matches = [
            entry
            for entry in self.list_models()
            if entry["reference"] == "hf:" + model
            and (installation_id is None or entry["installation_id"] == installation_id)
        ]
        if not matches:
            raise RuntimeError(f"Splash model not found: hf:{model}")
        if len(matches) != 1:
            raise RuntimeError(
                "Multiple Splash installations match; choose an installation with --id"
            )
        return matches[0]

    @contextmanager
    def _management_lock(self) -> Generator[None, None, None]:
        # Native servers retain a shared serve.lock, including externally started servers.
        with file_lock(self.runtime / "serve.lock"), file_lock(self.models_root / ".install.lock"):
            yield

    def resolve(self, model: str, installation_id: str | None) -> dict:
        """Resolve a complete selection locally, refusing incomplete or ambiguous data."""
        with file_lock(self.models_root / ".install.lock", shared=True, create=False):
            entry = self._select(model, installation_id)
            if not entry["complete"]:
                raise RuntimeError(f"Splash model is incomplete: hf:{model}; download it again")
            return entry

    def remove(self, model: str, installation_id: str | None) -> None:
        """Unlink one inactive installation and its pins without deleting shared sources."""
        with self._management_lock():
            entry = self._select(model, installation_id)
            link = Path(entry["path"])
            assembly = link.resolve()
            if self.api.assembly.is_held(assembly):
                raise RuntimeError("Splash model is running; stop the server before removal")
            owner = self.api.hub.pin_owner(link)
            pins = {
                snapshot.parent.parent / "refs/splash" / owner / snapshot.name
                for value in entry["files"]
                for snapshot in [_snapshot(Path(value))]
                if snapshot is not None
            }
            for pin in pins:
                if pin.is_symlink() or any(parent.is_symlink() for parent in list(pin.parents)[:3]):
                    raise RuntimeError(f"Refusing to remove a symlinked Splash cache pin: {pin}")
                if pin.exists() and pin.read_text().strip() != pin.name:
                    raise RuntimeError(f"Refusing to remove an invalid Splash cache pin: {pin}")
            link.unlink()  # Never recurse through a selection symlink.
            for pin in pins:
                pin.unlink(missing_ok=True)
            if _inside(assembly, self.models_root / ".resolved"):
                if not any(other.resolve() == assembly for other in self._links()):
                    shutil.rmtree(assembly)

    def _prepared_candidates(
        self, cache: Path, entries: list[dict], removed: list[Path]
    ) -> list[Path]:
        if not cache.is_dir() or entries:
            # Prepared keys can be shared by content across source paths. Provenance
            # cannot prove that another installation does not use an entry.
            return []
        candidates = []
        for directory in sorted(cache.iterdir()):
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or not re.fullmatch(r"[0-9a-f]{64}", directory.name)
            ):
                continue
            source, weights, digest = (
                directory / "source",
                directory / "weights",
                directory / "sha256",
            )
            if source.is_symlink() or not source.is_file() or source.stat().st_size > 16384:
                continue
            if (
                weights.is_symlink()
                or not weights.is_file()
                or digest.is_symlink()
                or not digest.is_file()
                or digest.stat().st_size != 64
                or not re.fullmatch(r"[0-9a-f]{64}", digest.read_text())
            ):
                continue
            lines = source.read_text().splitlines()
            if len(lines) == 2 and Path(lines[0]).is_absolute():
                origin = lines[0]
            elif (
                len(lines) == 4
                and lines[0] == "splash-prepared-weight-v1"
                and lines[3].startswith("source ")
            ):
                origin = lines[3][7:]
            else:
                continue  # Unrecognized cache data is never eligible for deletion.
            source_path = Path(origin)
            if not source_path.is_absolute():
                continue
            disappears = any(
                _inside(source_path.absolute(), path) or _inside(source_path.resolve(), path)
                for path in removed
            )
            if not source_path.exists() or disappears:
                candidates.append(directory)
        return candidates

    def prune(self, hub_cache: Path, weight_cache: Path, dry_run: bool, owned: bool) -> dict:
        """Collect only demonstrably unreferenced data in the default Arkai-owned caches."""
        if not owned:
            raise RuntimeError(
                "Refusing to prune external caches; "
                "only default Arkai-owned Splash caches are eligible"
            )
        with self._management_lock(), file_lock(weight_cache / "prepare.lock", create=not dry_run):
            entries = self.list_models()  # Unknown schemas fail closed before deletion.
            paths: list[Path] = []
            strategy = None
            if hub_cache.is_dir():
                from huggingface_hub import scan_cache_dir

                cache = scan_cache_dir(hub_cache)
                if cache.warnings:
                    raise RuntimeError(
                        "Splash Hub cache contains unrecognized entries; refusing unsafe pruning"
                    )
                protected = {
                    snapshot.name
                    for entry in entries
                    for value in entry["files"]
                    for snapshot in [_snapshot(Path(value))]
                    if snapshot is not None
                }
                revisions = [revision for repo in cache.repos for revision in repo.revisions]
                # Legacy package downloads create refs/main as cache bookkeeping,
                # not an installation pin. Preserve native pins and unknown refs;
                # an unused main-only snapshot in this owned cache is collectible.
                protected.update(
                    revision.commit_hash
                    for revision in revisions
                    if any(reference != "main" for reference in revision.refs)
                )
                unused = {
                    revision.commit_hash
                    for revision in revisions
                    if revision.commit_hash not in protected
                }
                if unused:
                    strategy = cache.delete_revisions(*sorted(unused))
                    if any(
                        not _inside(path.resolve(), hub_cache.resolve())
                        for path in strategy.snapshots | strategy.blobs | strategy.repos
                    ):
                        raise RuntimeError("Refusing to prune Hub data outside the owned cache")
                    paths.extend(sorted(strategy.snapshots | strategy.blobs | strategy.repos))
            prepared = self._prepared_candidates(weight_cache, entries, paths)
            if strategy is not None and not dry_run:
                strategy.execute()
            paths.extend(prepared)
            if not dry_run:
                for directory in prepared:
                    shutil.rmtree(directory)
            return {"paths": [str(path) for path in paths], "dry_run": dry_run}

    def dispatch(self, request: dict) -> dict:
        """Execute a validated request in Splash's own runtime."""
        operation = request["operation"]
        if operation == "list":
            with file_lock(self.models_root / ".install.lock", shared=True, create=False):
                return {"models": self.list_models()}
        if operation == "prune":
            return self.prune(
                Path(request["hub_cache"]),
                Path(request["weight_cache"]),
                request.get("dry_run", False),
                request.get("owned", False),
            )
        model = request["model"]
        if operation == "resolve":
            return self.resolve(model, request.get("installation_id"))
        if operation == "remove":
            self.remove(model, request.get("installation_id"))
            return {}
        options = request.get("options", {})
        draft = options.get("draft_model")
        if draft is not None:
            draft = self.api.models.parse_draft_model(draft)
        selection = self.api.models.Selection.of(
            self.models_root,
            model,
            revision=options.get("revision"),
            language_only=options.get("language_only", False),
            draft_model=draft,
        )
        if operation == "download":
            with file_lock(self.runtime / "serve.lock", shared=True):
                result = subprocess.run(
                    [str(self.api.paths.BINARY), "device-check"], capture_output=True, text=True
                )
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or "Splash hardware check failed")
                self.api.upstream.prepare(selection)
        elif operation != "info":
            raise RuntimeError(f"Unknown Splash bridge operation: {operation}")
        return {
            "path": str(selection.link),
            "models_root": str(self.models_root),
            "runtime": str(self.runtime),
        }


def main(argv: list[str] | None = None) -> int:
    """Emit a versioned JSON response, with installer diagnostics and secrets separated."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        with contextlib.redirect_stdout(sys.stderr):
            bridge = SplashBridge(load_api(Path(arguments[0])))
            response = bridge.dispatch(json.load(sys.stdin))
        response["version"] = 1
        code = 0
    except Exception as error:
        message = str(error)
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "SPLASH_API_KEY"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[redacted]")
        response = {"version": 1, "error": message}
        code = 1
    print(json.dumps(response))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
