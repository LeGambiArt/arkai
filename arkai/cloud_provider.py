"""Configuration helpers for user-defined cloud model providers."""

import hashlib
import importlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from arkai import utils

_PROVIDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_CUSTOM_CLOUD_CONTEXT_WINDOW = 262144

_MODEL_METADATA_FIELDS = {
    "name",
    "context_window",
    "max_output_tokens",
    "reasoning",
    "supports_developer_role",
}
_SUPPORTED_CUSTOM_PROTOCOLS = {"openai-completions", "anthropic-messages"}
_KEYCHAIN_BUNDLE_SERVICE = "arkai/credential-cache"
_KEYCHAIN_BUNDLE_ACCOUNT = "v1"


class _InvalidKeychainBundle(RuntimeError):
    """A keychain bundle exists but does not have a supported format."""


def _load_system_keyring():
    """Load a supported OS keyring backend without exposing backend errors."""
    try:
        keyring = importlib.import_module("keyring")
    except ImportError:
        raise RuntimeError(
            "Keychain API-key sources require the optional keyring dependency; "
            "install Arkai with the 'keychain' extra"
        ) from None

    try:
        backend = keyring.get_keyring()
        backend_module = type(backend).__module__.lower()
        supported_backend_prefixes = (
            "keyring.backends.macos",
            "keyring.backends.windows",
            "keyring.backends.secretservice",
            "keyring.backends.kwallet",
        )
        if not backend_module.startswith(supported_backend_prefixes):
            raise RuntimeError("unsupported keyring backend")
    except Exception:
        raise RuntimeError(
            "Could not access the system keychain; check that a supported keychain is available"
        ) from None
    return keyring


class _KeychainCredentialCache:
    """Cache keyring lookups in memory for one provider-resolution operation."""

    def __init__(self, keyring):
        self._keyring = keyring
        self._bundle_checked = False
        self._bundle: dict[tuple[str, str], str] = {}
        self._legacy: dict[tuple[str, str], str | None] = {}
        self._lock = threading.RLock()

    def get_password(self, service: str, account: str) -> str | None:
        """Resolve a key from one bundled keychain item or the legacy item."""
        reference = (service, account)
        with self._lock:
            if not self._bundle_checked:
                self._load_bundle()
            bundled_key = self._bundle.get(reference)
            if bundled_key is not None:
                return bundled_key
            if reference not in self._legacy:
                self._legacy[reference] = self._keyring.get_password(service, account)
            return self._legacy[reference]

    def clear(self) -> None:
        """Drop cached secret references when provider resolution is complete."""
        with self._lock:
            self._bundle.clear()
            self._legacy.clear()

    def _load_bundle(self) -> None:
        self._bundle_checked = True
        raw_bundle = self._keyring.get_password(_KEYCHAIN_BUNDLE_SERVICE, _KEYCHAIN_BUNDLE_ACCOUNT)
        if raw_bundle is None:
            return
        try:
            bundle = json.loads(raw_bundle)
        except (TypeError, json.JSONDecodeError):
            raise _InvalidKeychainBundle(
                "Arkai's keychain credential bundle is invalid; replace it with a version 1 bundle"
            ) from None
        if (
            not isinstance(bundle, dict)
            or set(bundle) != {"version", "credentials"}
            or isinstance(bundle.get("version"), bool)
            or bundle.get("version") != 1
            or not isinstance(bundle.get("credentials"), list)
        ):
            raise _InvalidKeychainBundle(
                "Arkai's keychain credential bundle is invalid; replace it with a version 1 bundle"
            )

        credentials: dict[tuple[str, str], str] = {}
        for entry in bundle["credentials"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"service", "account", "key"}
                or not isinstance(entry.get("service"), str)
                or not entry["service"].strip()
                or not isinstance(entry.get("account"), str)
                or not entry["account"].strip()
                or not isinstance(entry.get("key"), str)
                or not entry["key"].strip()
            ):
                raise _InvalidKeychainBundle(
                    "Arkai's keychain credential bundle is invalid; replace it with "
                    "a version 1 bundle"
                )
            reference = (entry["service"].strip(), entry["account"].strip())
            if reference in credentials:
                raise _InvalidKeychainBundle(
                    "Arkai's keychain credential bundle contains duplicate entries"
                )
            credentials[reference] = entry["key"].strip()
        self._bundle = credentials


def _normalize_base_url(value: object, path: str) -> tuple[str, str]:
    """Validate an HTTPS endpoint and return its normalized URL and sandbox host."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} is required")
    base_url = value.strip().rstrip("/")
    try:
        parsed_url = urlsplit(base_url)
        port = parsed_url.port
    except ValueError as error:
        raise ValueError(f"{path} must be a valid HTTPS URL") from error
    if (
        parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(f"{path} must be HTTPS without credentials, query, or fragment")
    if port is not None and port == 0:
        raise ValueError(f"{path} port must be between 1 and 65535")
    return base_url, _url_host(parsed_url.hostname, port)


def _normalize_models(value: object, path: str, default_model: str | None) -> list[dict]:
    """Validate provider model metadata and add its default model if needed."""
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")

    models: list[dict] = []
    model_ids: set[str] = set()
    for index, entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_path} must be a mapping")
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{entry_path}.id is required")
        model_id = model_id.strip()
        if model_id in model_ids:
            raise ValueError(f"Duplicate custom cloud model ID: {model_id}")
        model_ids.add(model_id)
        model = {"id": model_id}
        for field in _MODEL_METADATA_FIELDS:
            if field not in entry:
                continue
            metadata = entry[field]
            if field == "name":
                if not isinstance(metadata, str) or not metadata.strip():
                    raise ValueError(f"{entry_path}.{field} must be a non-empty string")
                model[field] = metadata.strip()
            elif field in {"context_window", "max_output_tokens"}:
                parsed_size = utils.parse_token_size(metadata)
                if parsed_size is None:
                    raise ValueError(
                        f"{entry_path}.{field} must be a positive integer or size like 64k"
                    )
                model[field] = parsed_size
            elif not isinstance(metadata, bool):
                raise ValueError(f"{entry_path}.{field} must be a boolean")
            else:
                model[field] = metadata
        models.append(model)

    if default_model and default_model not in model_ids:
        models.append({"id": default_model, "name": default_model})
    return models


def _normalize_api_key(value: object, path: str) -> dict | None:
    """Validate an optional API-key source declaration."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")

    source = value.get("source")
    if source == "environment":
        if set(value) != {"source", "name"}:
            raise ValueError(f"{path} with source 'environment' must contain only source and name")
        name = value.get("name")
        if not isinstance(name, str) or not _ENVIRONMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"{path}.name must be an environment variable name")
        return {"source": source, "name": name}

    if source == "keychain":
        if set(value) != {"source", "service", "account"}:
            raise ValueError(
                f"{path} with source 'keychain' must contain only source, service, and account"
            )
        service = value.get("service")
        account = value.get("account")
        if not isinstance(service, str) or not service.strip():
            raise ValueError(f"{path}.service must be a non-empty string")
        if not isinstance(account, str) or not account.strip():
            raise ValueError(f"{path}.account must be a non-empty string")
        return {"source": source, "service": service.strip(), "account": account.strip()}

    raise ValueError(f"{path}.source must be 'environment' or 'keychain'")


def _api_key_environment_name(provider_id: str, api_key: dict | None) -> str | None:
    """Return the environment binding consumed by native agent configurations."""
    if api_key is None:
        return None
    if api_key["source"] == "environment":
        return api_key["name"]

    provider_token = re.sub(r"[^A-Za-z0-9_]", "_", provider_id).upper()
    provider_hash = hashlib.sha256(provider_id.encode("utf-8")).hexdigest()[:8].upper()
    return f"ARKAI_CUSTOM_API_KEY_{provider_token}_{provider_hash}"


def normalize_custom_cloud_config(value: object) -> dict:
    """Validate and normalize a map of user-defined OpenAI-compatible providers."""
    if not isinstance(value, dict):
        raise ValueError("agent.cloud.custom must be a mapping")

    if "api_key_env" in value:
        raise ValueError(
            "agent.cloud.custom.api_key_env was replaced by api_key.source; "
            "use api_key: {source: environment, name: VARIABLE}"
        )

    moved_fields = {"models", "default_model", "provider_id", "name"} & value.keys()
    if moved_fields:
        fields = ", ".join(sorted(moved_fields))
        raise ValueError(
            f"agent.cloud.custom.{fields} moved under agent.cloud.custom.providers.<id>"
        )

    default_base_url = value.get("base_url")
    default_protocol = value.get("protocol", "openai-completions")
    if not isinstance(default_protocol, str) or default_protocol not in _SUPPORTED_CUSTOM_PROTOCOLS:
        raise ValueError(
            "agent.cloud.custom.protocol must be 'openai-completions' or 'anthropic-messages'"
        )
    default_api_key = _normalize_api_key(value.get("api_key"), "agent.cloud.custom.api_key")

    provider_values = value.get("providers")
    if not isinstance(provider_values, dict) or not provider_values:
        raise ValueError("agent.cloud.custom.providers must be a non-empty mapping")

    providers: dict[str, dict] = {}
    for provider_id, provider_value in provider_values.items():
        provider_path = f"agent.cloud.custom.providers.{provider_id}"
        if not isinstance(provider_id, str) or not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
            raise ValueError("custom provider IDs must contain only letters, digits, '_' or '-'")
        if not isinstance(provider_value, dict):
            raise ValueError(f"{provider_path} must be a mapping")
        if "api_key_env" in provider_value:
            raise ValueError(
                f"{provider_path}.api_key_env was replaced by api_key.source; "
                "use api_key: {source: environment, name: VARIABLE}"
            )

        base_url, host = _normalize_base_url(
            provider_value.get("base_url", default_base_url), f"{provider_path}.base_url"
        )
        inferred_protocol = {
            "openai": "openai-completions",
            "anthropic": "anthropic-messages",
        }.get(provider_id, default_protocol)
        protocol = provider_value.get("protocol", inferred_protocol)
        if not isinstance(protocol, str) or protocol not in _SUPPORTED_CUSTOM_PROTOCOLS:
            raise ValueError(
                f"{provider_path}.protocol must be 'openai-completions' or 'anthropic-messages'"
            )
        api_key = _normalize_api_key(
            provider_value.get("api_key", default_api_key), f"{provider_path}.api_key"
        )
        name = provider_value.get("name", provider_id)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{provider_path}.name must be a non-empty string")
        default_model = provider_value.get("default_model")
        if default_model is not None and (
            not isinstance(default_model, str) or not default_model.strip()
        ):
            raise ValueError(f"{provider_path}.default_model must be a non-empty string")
        if isinstance(default_model, str):
            default_model = default_model.strip()

        providers[provider_id] = {
            "provider_id": provider_id,
            "name": name.strip(),
            "protocol": protocol,
            "base_url": base_url,
            "api_key": api_key,
            "api_key_environment_name": _api_key_environment_name(provider_id, api_key),
            "default_model": default_model,
            "models": _normalize_models(
                provider_value.get("models"), f"{provider_path}.models", default_model
            ),
            "host": host,
        }

    credential_bindings: dict[str, dict] = {}
    for provider in providers.values():
        environment_name = provider["api_key_environment_name"]
        if not environment_name:
            continue
        api_key = provider["api_key"]
        if (
            environment_name in credential_bindings
            and credential_bindings[environment_name] != api_key
        ):
            raise ValueError("Custom provider API keys must use unique environment bindings")
        credential_bindings[environment_name] = api_key

    default_provider = value.get("default_provider")
    if default_provider is None:
        if len(providers) != 1:
            raise ValueError(
                "agent.cloud.custom.default_provider is required with multiple providers"
            )
        default_provider = next(iter(providers))
    if not isinstance(default_provider, str) or default_provider not in providers:
        raise ValueError("agent.cloud.custom.default_provider must name a configured provider")

    return {
        "default_provider": default_provider,
        "providers": providers,
    }


def custom_cloud_config_from_arkai(cfg: dict) -> dict:
    """Read the custom provider settings from a loaded Arkai configuration."""
    agent = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    cloud = agent.get("cloud", {}) if isinstance(agent, dict) else {}
    settings = cloud.get("custom") if isinstance(cloud, dict) else None
    try:
        return normalize_custom_cloud_config(settings)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def custom_cloud_model_provider(config: dict, model_reference: str) -> tuple[dict, str]:
    """Resolve a provider-qualified model reference, defaulting unqualified IDs."""
    provider_id, separator, model_id = model_reference.partition("/")
    providers = config["providers"]
    if separator and provider_id in providers:
        if not model_id:
            raise RuntimeError("Custom cloud model reference must include a model ID")
        return providers[provider_id], model_id
    return providers[config["default_provider"]], model_reference


def select_custom_cloud_model(config: dict, override: str | None = None) -> str:
    """Choose a provider-qualified CLI override or the default provider's model."""
    if override:
        selected_provider, model = custom_cloud_model_provider(config, override)
    else:
        selected_provider = config["providers"][config["default_provider"]]
        model = selected_provider.get("default_model")
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(
            "--cloud custom requires the default provider's default_model or an explicit --model"
        )
    return custom_cloud_model_reference(selected_provider, model.strip())


def custom_cloud_models(provider: dict, selected_model: str | None = None) -> list[dict]:
    """Return configured model metadata and include a CLI-only model if necessary."""
    models = [dict(model) for model in provider.get("models", [])]
    known_ids = {model["id"] for model in models}
    if selected_model and selected_model not in known_ids:
        models.append({"id": selected_model, "name": selected_model})
    for model in models:
        # Cloud models commonly support larger windows than agent defaults; an
        # unconfigured context window must not silently truncate cloud sessions.
        model.setdefault("context_window", DEFAULT_CUSTOM_CLOUD_CONTEXT_WINDOW)
    return models


def custom_cloud_api_key(
    provider: dict, credential_cache: _KeychainCredentialCache | None = None
) -> str | None:
    """Resolve the optional API key from its configured source."""
    api_key = provider.get("api_key")
    if not api_key:
        return None

    if api_key["source"] == "environment":
        name = api_key["name"]
        key = os.environ.get(name, "").strip()
        if not key:
            raise RuntimeError(
                f"Environment variable {name} is required for custom provider "
                f"{provider['provider_id']!r}"
            )
        return key

    service = api_key["service"]
    account = api_key["account"]
    owns_cache = credential_cache is None
    if credential_cache is None:
        credential_cache = _KeychainCredentialCache(_load_system_keyring())
    try:
        key = credential_cache.get_password(service, account)
    except _InvalidKeychainBundle as error:
        raise RuntimeError(str(error)) from None
    except Exception:
        raise RuntimeError(
            f"Could not access the system keychain for custom provider "
            f"{provider['provider_id']!r}; check that a supported keychain is available"
        ) from None
    finally:
        if owns_cache:
            credential_cache.clear()
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError(
            f"No API key found in the system keychain for custom provider "
            f"{provider['provider_id']!r} (service {service!r}, account {account!r})"
        )
    return key.strip()


def discover_provider_models(provider: dict, api_key: str | None) -> list[dict] | None:
    """Fetch a provider's model listing from its OpenAI-style /models endpoint."""
    import requests

    try:
        response = requests.get(
            f"{provider['base_url'].rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=(5, 20),
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        listing = response.json()
    except ValueError:
        return None
    entries = listing.get("data") if isinstance(listing, dict) else None
    if not isinstance(entries, list):
        return None
    models = []
    for entry in entries:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            models.append({"id": model_id.strip(), "name": model_id.strip()})
    return models or None


def discover_anthropic_provider_models(provider: dict, api_key: str | None) -> list[dict]:
    """Combine a Messages gateway's listing with the Anthropic model catalog.

    Gateways commonly route Anthropic's stable model IDs over the Messages
    API without listing them, so the gateway's own listing is not the full
    routable set. The listing stays authoritative for IDs it names; catalog
    entries are appended for the rest.
    """
    discovered = discover_provider_models(provider, api_key) or []
    known_ids = {model["id"] for model in discovered}
    models = list(discovered)
    for model in anthropic_catalog_models():
        if model["id"] not in known_ids:
            models.append(model)
    return models


_ANTHROPIC_CATALOG_URL = "https://models.dev/api.json"
_ANTHROPIC_CATALOG_CACHE_SECONDS = 24 * 60 * 60
_ANTHROPIC_CATALOG_CACHE = Path(
    os.path.expanduser("~/.local/state/arkai/cache/anthropic-models.json")
)


def anthropic_catalog_models() -> list[dict]:
    """Return Claude model metadata from the models.dev catalog, cached for a day."""
    import time

    import requests

    try:
        cached = json.loads(_ANTHROPIC_CATALOG_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = None
    now = time.time()
    if (
        isinstance(cached, dict)
        and now - cached.get("fetched_at", 0) < _ANTHROPIC_CATALOG_CACHE_SECONDS
        and isinstance(cached.get("models"), list)
    ):
        return cached["models"]
    try:
        response = requests.get(_ANTHROPIC_CATALOG_URL, timeout=(5, 20))
        response.raise_for_status()
        catalog = response.json()
    except (requests.RequestException, ValueError):
        return cached["models"] if isinstance(cached, dict) else []
    provider_data = catalog.get("anthropic") if isinstance(catalog, dict) else None
    provider_models = provider_data.get("models") if isinstance(provider_data, dict) else None
    models = []
    if isinstance(provider_models, dict):
        for model_id, model in provider_models.items():
            if not isinstance(model, dict):
                continue
            entry: dict = {"id": model_id, "name": model.get("name", model_id)}
            limit = model.get("limit")
            if isinstance(limit, dict) and limit.get("context"):
                entry["context_window"] = limit["context"]
            if model.get("reasoning"):
                entry["reasoning"] = True
            models.append(entry)
    if models:
        try:
            _ANTHROPIC_CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _ANTHROPIC_CATALOG_CACHE.write_text(
                json.dumps({"fetched_at": now, "models": models}), encoding="utf-8"
            )
        except OSError:
            pass
    return models


def custom_cloud_environment(config: dict) -> dict[str, str]:
    """Return resolved credentials under the runtime names used by agent clients."""
    environment: dict[str, str] = {}
    uses_keychain = any(
        isinstance(provider.get("api_key"), dict)
        and provider["api_key"].get("source") == "keychain"
        for provider in config["providers"].values()
    )
    credential_cache = _KeychainCredentialCache(_load_system_keyring()) if uses_keychain else None
    try:
        for provider in config["providers"].values():
            variable = provider.get("api_key_environment_name")
            key = custom_cloud_api_key(provider, credential_cache)
            if not variable or not key:
                continue
            environment[variable] = key
        return environment
    finally:
        if credential_cache is not None:
            credential_cache.clear()


def custom_cloud_resolved_api_key(provider: dict, environment: dict[str, str]) -> str | None:
    """Return a provider's key from an already-resolved launch environment."""
    variable = provider.get("api_key_environment_name")
    if not variable:
        return None
    key = environment.get(variable)
    if not key:
        raise RuntimeError(
            f"Resolved API key is missing for custom provider {provider['provider_id']!r}"
        )
    return key


def custom_cloud_model_reference(provider: dict, model_id: str) -> str:
    """Return the provider-qualified model reference used by agent clients."""
    return f"{provider['provider_id']}/{model_id}"


def _url_host(hostname: str, port: int | None) -> str:
    """Format an exact hostname/port allowlist entry for the sandbox."""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return f"{hostname}:{port or 443}"


def create_private_agent_dir(
    source_dir: Path,
    state_dir: Path,
    prefix: str,
    excluded_entries: set[str] | None = None,
) -> tuple[Path, list[str]]:
    """Stage user agent settings in a private Arkai runtime directory."""
    state_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=state_dir))
    runtime_dir.chmod(0o700)
    read_only_volumes: list[str] = []
    excluded = excluded_entries or set()
    try:
        if source_dir.is_dir():
            for source in source_dir.iterdir():
                if source.name in excluded or source.name in {"session", "sessions"}:
                    continue
                destination = runtime_dir / source.name
                if source.is_symlink():
                    target = source.resolve()
                    if target.is_dir():
                        destination.symlink_to(target, target_is_directory=True)
                        volume = f"{target}:ro"
                        if volume not in read_only_volumes:
                            read_only_volumes.append(volume)
                    elif target.is_file():
                        shutil.copy2(target, destination)
                elif source.is_dir():
                    destination.symlink_to(source.resolve(), target_is_directory=True)
                    volume = f"{source_dir.resolve()}:ro"
                    if volume not in read_only_volumes:
                        read_only_volumes.append(volume)
                elif source.is_file():
                    shutil.copy2(source, destination)
        return runtime_dir, read_only_volumes
    except Exception:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        raise
