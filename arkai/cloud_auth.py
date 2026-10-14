"""Credential helpers for cloud-backed agent launches."""

import getpass
import json
import os
import sqlite3
import subprocess
import tempfile
import warnings
from pathlib import Path

import yaml


def _key_from_data(data: object) -> str | None:
    if isinstance(data, dict):
        key = data.get("key")
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _read_pi_key(agent_dir: Path) -> str | None:
    auth_path = agent_dir / "auth.json"
    try:
        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(auth_data, dict):
        return None
    return _key_from_data(auth_data.get("openai"))


def _get_opencode_auth_path() -> Path:
    """Return OpenCode's standard global authentication file."""
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "opencode" / "auth.json"


def _read_opencode_key() -> str | None:
    """Read an OpenAI API key saved by OpenCode's own auth flow."""
    try:
        auth_data = json.loads(_get_opencode_auth_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(auth_data, dict):
        return None
    return _key_from_data(auth_data.get("openai"))


def _read_omp_key(agent_dir: Path) -> str | None:
    database = agent_dir / "agent.db"
    if not database.is_file():
        return None
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=5)
        try:
            tables = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'auth_credentials'"
            ).fetchone()
            if tables is None:
                return None
            columns = {row[1] for row in connection.execute("PRAGMA table_info(auth_credentials)")}
            disabled_filter = (
                "AND disabled_cause IS NULL"
                if "disabled_cause" in columns
                else "AND disabled = 0"
                if "disabled" in columns
                else ""
            )
            order = "ORDER BY id DESC" if "id" in columns else ""
            row = connection.execute(
                "SELECT data FROM auth_credentials "
                "WHERE provider = 'openai' AND credential_type = 'api_key' "
                f"{disabled_filter} {order} LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        return _key_from_data(json.loads(row[0]))
    except (TypeError, json.JSONDecodeError):
        return None


def _read_model_config_key(agent_name: str, agent_dir: Path) -> str | None:
    model_paths = (
        [agent_dir / "models.json"]
        if agent_name == "pi"
        else [agent_dir / "models.yml", agent_dir / "models.yaml"]
    )
    for model_path in model_paths:
        try:
            if model_path.suffix == ".json":
                models = json.loads(model_path.read_text(encoding="utf-8"))
            else:
                models = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError):
            continue
        providers = models.get("providers") if isinstance(models, dict) else None
        provider = providers.get("openai") if isinstance(providers, dict) else None
        if isinstance(provider, dict):
            key = provider.get("apiKey")
            if isinstance(key, str) and key.strip():
                key = key.strip()
                if key.isidentifier() and key.upper() == key and key not in os.environ:
                    continue
                return key
    return None


def _write_pi_key(agent_dir: Path, key: str) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_path = agent_dir / "auth.json"
    try:
        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        auth_data = {}
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not save OpenAI credentials: {auth_path} is invalid JSON"
        ) from error
    if not isinstance(auth_data, dict):
        raise RuntimeError(f"Could not save OpenAI credentials: {auth_path} must contain an object")
    auth_data["openai"] = {"type": "api_key", "key": key}

    descriptor, filename = tempfile.mkstemp(prefix="auth-", suffix=".json", dir=agent_dir)
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as auth_file:
            json.dump(auth_data, auth_file, indent=2)
            auth_file.write("\n")
        os.replace(temporary_path, auth_path)
        auth_path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_opencode_key(key: str) -> None:
    """Persist an OpenCode API key in its standard auth file securely."""
    auth_path = _get_opencode_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        auth_data = {}
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not save OpenAI credentials: {auth_path} is invalid JSON"
        ) from error
    if not isinstance(auth_data, dict):
        raise RuntimeError(f"Could not save OpenAI credentials: {auth_path} must contain an object")
    auth_data["openai"] = {"type": "api", "key": key}

    descriptor, filename = tempfile.mkstemp(prefix="auth-", suffix=".json", dir=auth_path.parent)
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as auth_file:
            json.dump(auth_data, auth_file, indent=2)
            auth_file.write("\n")
        os.replace(temporary_path, auth_path)
        auth_path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_omp_cli_key(agent_path: str, agent_dir: Path) -> str | None:
    """Ask OMP's credential resolver about database, config, and dotenv credentials."""
    environment = os.environ.copy()
    environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    environment["OMP_SKIP_SETUP"] = "1"
    try:
        result = subprocess.run(
            [agent_path, "token", "openai"],
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    key = result.stdout.strip()
    return key if key and key != "OPENAI_API_KEY" else None


def _write_omp_models_key(agent_dir: Path, key: str) -> None:
    """Persist an OMP OpenAI key in its global models config with private permissions."""
    agent_dir.mkdir(parents=True, exist_ok=True)
    paths = [agent_dir / "models.yml", agent_dir / "models.yaml"]
    models_path = next((path for path in paths if path.exists()), paths[0])
    if models_path.exists():
        try:
            models = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise RuntimeError(f"Could not read OMP model configuration: {models_path}") from error
    else:
        models = {}
    if not isinstance(models, dict):
        raise RuntimeError(f"OMP model configuration must contain a YAML object: {models_path}")
    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError(f"Invalid providers section in OMP model configuration: {models_path}")
    provider = providers.setdefault("openai", {})
    if not isinstance(provider, dict):
        raise RuntimeError(f"Invalid OpenAI provider config in {models_path}")
    provider["apiKey"] = key

    descriptor, filename = tempfile.mkstemp(prefix=".arkai-models-", dir=agent_dir)
    temporary_path = Path(filename)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
            yaml.safe_dump(models, config_file, sort_keys=False)
        os.replace(temporary_path, models_path)
        models_path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_openai_api_key(agent_name: str, agent_dir: Path, agent_path: str | None = None) -> str:
    """Find or securely prompt for and persist the OpenAI API key."""
    if agent_name == "opencode":
        global_key = _read_opencode_key()
        if global_key:
            return global_key
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    elif agent_name == "pi":
        global_key = _read_pi_key(agent_dir)
        if global_key:
            if global_key.startswith("$"):
                variable = global_key[2:-1] if global_key.startswith("${") else global_key[1:]
                resolved = os.environ.get(variable, "").strip()
                if resolved:
                    _write_pi_key(agent_dir, resolved)
                    return resolved
            else:
                return global_key
        configured_key = _read_model_config_key(agent_name, agent_dir)
        if configured_key:
            if configured_key.startswith("$"):
                variable = (
                    configured_key[2:-1] if configured_key.startswith("${") else configured_key[1:]
                )
                resolved = os.environ.get(variable, "").strip()
                if resolved:
                    _write_pi_key(agent_dir, resolved)
                    return resolved
            else:
                return configured_key
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    else:
        global_key = _read_omp_key(agent_dir)
        if global_key:
            return global_key
        configured_key = _read_model_config_key(agent_name, agent_dir)
        if configured_key:
            resolved = os.environ.get(configured_key, "").strip()
            if resolved:
                _write_omp_models_key(agent_dir, resolved)
                return resolved
            if configured_key != "OPENAI_API_KEY":
                return configured_key
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key and agent_path:
            key = _read_omp_cli_key(agent_path, agent_dir) or ""
            if key:
                _write_omp_models_key(agent_dir, key)
                return key

    if not key:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass(f"OpenAI API key for {agent_name} (input hidden): ").strip()
        except getpass.GetPassWarning as error:
            raise RuntimeError(
                "An interactive terminal is required to enter an OpenAI API key"
            ) from error
        except (EOFError, KeyboardInterrupt) as error:
            raise RuntimeError("OpenAI API key is required for --cloud openai") from error
    if not key:
        raise RuntimeError("OpenAI API key is required for --cloud openai")

    if agent_name == "opencode":
        _write_opencode_key(key)
    elif agent_name == "pi":
        _write_pi_key(agent_dir, key)
    else:
        _write_omp_models_key(agent_dir, key)
    return key
