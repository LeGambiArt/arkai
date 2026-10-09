"""Native Splash model management and shared download/serving cache paths."""

import hashlib
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arkai import utils
from arkai.model_store import ModelEntry

SUPPORTED_VERSION = "1.1.0"
_REPOSITORY = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
_VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_OPTIONS = {
    "max_memory",
    "max_cache_disk",
    "kv_format",
    "language_only",
    "revision",
    "draft_model",
    "default_reasoning_effort",
    "hub_cache",
    "weight_cache",
    "installation_root",
    "python_path",
}


def splash_options(cfg: dict) -> dict:
    """Validate backend options before provisioning or launching a subprocess."""
    settings = cfg.get("inference", {})
    options = settings.get("splash", {}) if isinstance(settings, dict) else {}
    if not isinstance(options, dict):
        raise ValueError("inference.splash must be a mapping")
    unknown = set(options) - _OPTIONS
    if unknown:
        raise ValueError(f"Unknown inference.splash options: {', '.join(sorted(unknown))}")
    for name, value in options.items():
        if name == "language_only":
            if type(value) is not bool:
                raise ValueError("inference.splash.language_only must be a boolean")
        elif name in {"max_memory", "max_cache_disk"}:
            text = str(value)
            match = re.fullmatch(r"([0-9]+)(?:[KMG](?:B|IB)?)?", text, re.IGNORECASE)
            if match is None:
                raise ValueError(f"inference.splash.{name} must be a size such as 28G")
            if int(match.group(1)) == 0 and (name != "max_cache_disk" or text != "0"):
                raise ValueError(f"inference.splash.{name} must be positive")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"inference.splash.{name} must be a nonempty string")
    if options.get("kv_format", "int8") not in {"int8", "bf16"}:
        raise ValueError("inference.splash.kv_format must be int8 or bf16")
    if options.get("default_reasoning_effort", "none") not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("inference.splash.default_reasoning_effort is invalid")
    return options


def splash_model_id(reference: str) -> str:
    """Translate an Arkai HF reference while preserving its selected variant."""
    if not isinstance(reference, str) or not reference.startswith("hf:"):
        raise ValueError("Splash requires a Hugging Face reference: hf:owner/model[:variant]")
    native = reference[3:]
    repository, separator, variant = native.partition(":")
    if (
        not _REPOSITORY.fullmatch(repository)
        or any(value in repository for value in ("..", "--"))
        or repository.endswith(".git")
        or separator
        and (not _VARIANT.fullmatch(variant) or ".." in variant)
    ):
        raise ValueError(
            "Splash model must name a valid Hugging Face repository and optional variant"
        )
    return native


def splash_environment(cfg: dict, create: bool = False) -> dict[str, str]:
    """Return a child environment using the same caches for download and inference."""
    from arkai.model import get_models_dir

    options = splash_options(cfg)
    base = Path(get_models_dir()) / "splash"
    environment = os.environ.copy()
    for setting, variable, directory in (
        ("hub_cache", "HF_HUB_CACHE", "hub"),
        ("weight_cache", "SPLASH_WEIGHT_CACHE", "weights"),
    ):
        path = Path(options.get(setting, str(base / directory))).expanduser().resolve()
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise RuntimeError(
                    f"Cannot create Splash cache {path}: {error}. Check permissions"
                ) from error
        environment[variable] = str(path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def serve_arguments(cfg: dict) -> list[str]:
    """Use identical native source-selection options for downloads and serving."""
    options = splash_options(cfg)
    arguments = []
    for name in (
        "max_memory",
        "max_cache_disk",
        "kv_format",
        "revision",
        "draft_model",
        "default_reasoning_effort",
    ):
        if name in options:
            value = options[name]
            if name == "draft_model":
                local = Path(value).expanduser()
                if local.is_dir():
                    value = str(local.resolve())
            arguments.extend(["--" + name.replace("_", "-"), str(value)])
    if options.get("language_only"):
        arguments.append("--language-only")
    return arguments


@dataclass(frozen=True)
class SplashInstallation:
    """Verified program root and the Python executable bundled with Splash."""

    root: Path
    python: Path


def discover_installation(cfg: dict) -> SplashInstallation:
    """Locate a supported package without assuming a particular Homebrew prefix."""
    options = splash_options(cfg)
    if "installation_root" in options:
        roots = [Path(options["installation_root"]).expanduser().resolve()]
    else:
        settings = cfg.get("inference", {})
        path = settings.get("path", "splash") if settings.get("backend") == "splash" else "splash"
        try:
            binary = Path(utils.resolve_binary(path)).resolve()
        except RuntimeError as error:
            raise RuntimeError(
                "Splash is not installed; install Splash 1.1.0 or configure "
                "inference.splash.installation_root"
            ) from error
        roots = [binary.parent.parent / "libexec", binary.parent.parent, binary.parent]
    for root in roots:
        release = root / "release.json"
        if not release.is_file():
            continue
        try:
            version = json.loads(release.read_text())["version"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid Splash release metadata: {release}") from error
        if version != SUPPORTED_VERSION:
            raise RuntimeError(
                f"Splash {version} has an unverified installer interface; "
                f"only {SUPPORTED_VERSION} is supported"
            )
        python = (
            Path(options.get("python_path", str(root / "python/bin/python3")))
            .expanduser()
            .resolve()
        )
        if not python.is_file() or not (root / "install/models.py").is_file():
            raise RuntimeError(f"Incomplete Splash installation at {root}; reinstall Splash")
        return SplashInstallation(root.resolve(), python)
    raise RuntimeError(
        "Cannot discover Splash 1.1.0; configure inference.splash.installation_root "
        "to the directory containing release.json and install/models.py"
    )


class SplashModelStore:
    """Manage native selections through a version-checked installer bridge."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def call(self, operation: str, **settings: Any) -> dict:
        """Run the bridge in Splash's Python without importing its dependencies into Arkai."""
        installed = discover_installation(self.cfg)
        environment = splash_environment(self.cfg, create=operation == "download")
        request = {"operation": operation, "options": splash_options(self.cfg), **settings}
        command = [
            str(installed.python),
            "-B",
            str(Path(__file__).with_name("splash_bridge.py")),
            str(installed.root),
        ]
        try:
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                start_new_session=True,
            ) as process:
                try:
                    stdout, stderr = process.communicate(
                        json.dumps(request),
                        timeout=None if operation == "download" else 60,
                    )
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
                returncode = process.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Splash model {operation} failed: {error}") from error
        try:
            response = json.loads(stdout)
            if not isinstance(response, dict) or response.get("version") != 1:
                raise ValueError("unknown bridge response")
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                "Splash bridge returned invalid JSON; check the installed version"
            ) from error
        for variable in ("SPLASH_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if environment.get(variable):
                stderr = stderr.replace(environment[variable], "[redacted]")
        if returncode or response.get("error"):
            message = str(response.get("error") or stderr or "operation failed")
            for variable in ("SPLASH_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                if environment.get(variable):
                    message = message.replace(environment[variable], "[redacted]")
            raise RuntimeError(f"Splash model {operation} failed: {message}")
        if stderr and operation == "download":
            utils.info(stderr.rstrip())
        return response

    def download(self, reference: str) -> Path:
        """Install target, draft, tokenizer, and applicable vision sources without serving."""
        from arkai.inference_splash import SplashBackend

        SplashBackend().check_environment()
        utils.info(
            f"Downloading Splash model and dependencies for {reference}; "
            "this may take several minutes"
        )
        return Path(self.call("download", model=splash_model_id(reference))["path"])

    def list_models(self) -> list[ModelEntry]:
        """List native model selections without depending on Splash's private APIs."""
        home = Path(os.getenv("HOME") or str(Path.home()))
        models_root = home / "Library" / "Application Support" / "Splash" / "models"
        if not models_root.is_dir():
            return []

        entries = []
        try:
            parents = sorted(models_root.iterdir())
            for parent in parents:
                if parent.is_symlink() or not parent.is_dir():
                    continue
                for selection in sorted(parent.iterdir()):
                    if not selection.is_symlink():
                        continue
                    try:
                        reference = _selection_reference(selection, models_root)
                        target = selection.resolve()
                    except (OSError, RuntimeError):
                        continue
                    if reference is None:
                        continue
                    entries.append(
                        ModelEntry(
                            reference=reference,
                            installation_id=hashlib.sha256(str(selection).encode()).hexdigest(),
                            path=selection,
                            # A version-independent inventory can confirm that a selection
                            # target exists, but cannot validate release-specific artifacts.
                            complete=None if target.is_dir() else False,
                            revision=_snapshot_revision(target),
                        )
                    )
        except OSError as error:
            raise RuntimeError(
                f"Cannot read Splash model inventory at {models_root}: {error}"
            ) from error
        return entries

    def resolve(self, reference: str, installation_id: str | None = None) -> Path:
        """Resolve a complete cached selection without fetching or repairing artifacts."""
        return Path(
            self.call("resolve", model=splash_model_id(reference), installation_id=installation_id)[
                "path"
            ]
        )

    def remove(self, reference: str, installation_id: str | None = None) -> None:
        """Uninstall an inactive selection, retaining shared dependencies for safe pruning."""
        self.call("remove", model=splash_model_id(reference), installation_id=installation_id)

    def prune(self, dry_run: bool = False) -> dict:
        """Collect unreferenced data only inside Arkai's default owned caches."""
        from arkai.model import get_models_dir

        environment = splash_environment(self.cfg)
        base = (Path(get_models_dir()) / "splash").resolve()
        owned = (
            Path(environment["HF_HUB_CACHE"]) == base / "hub"
            and Path(environment["SPLASH_WEIGHT_CACHE"]) == base / "weights"
        )
        return self.call(
            "prune",
            hub_cache=environment["HF_HUB_CACHE"],
            weight_cache=environment["SPLASH_WEIGHT_CACHE"],
            dry_run=dry_run,
            owned=owned,
        )


def _selection_reference(selection: Path, models_root: Path) -> str | None:
    """Derive a model reference from a native selection path or its metadata."""
    relative = selection.relative_to(models_root)
    if len(relative.parts) == 2 and relative.parts[0] != ".selections":
        candidate = f"hf:{relative.parts[0]}/{relative.parts[1]}"
    elif relative.parts[0] == ".selections":
        metadata = selection.resolve() / "model.json"
        try:
            record = json.loads(metadata.read_text(encoding="utf-8"))
            model = record.get("model") if isinstance(record, dict) else None
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(model, str):
            return None
        candidate = f"hf:{model}"
    else:
        return None
    try:
        return "hf:" + splash_model_id(candidate)
    except ValueError:
        return None


def _snapshot_revision(target: Path) -> str | None:
    """Return a Hub snapshot revision when the target path exposes one."""
    parts = target.parts
    for index, part in enumerate(parts[:-1]):
        revision = parts[index + 1]
        if part == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            return revision
    return None
