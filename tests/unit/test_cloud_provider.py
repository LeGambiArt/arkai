"""Tests for custom cloud provider configuration and credential resolution."""

import sys
from types import SimpleNamespace

import pytest

from arkai import cloud_provider


def _base_config(api_key: dict | None = None) -> dict:
    custom = {
        "base_url": "https://gateway.example/v1",
        "default_provider": "gateway",
        "providers": {"gateway": {"default_model": "org/model"}},
    }
    if api_key is not None:
        custom["api_key"] = api_key
    return custom


def test_normalizes_inherited_environment_and_provider_keychain_sources() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "api_key": {"source": "environment", "name": "TEAM_MODEL_KEY"},
            "default_provider": "gateway",
            "providers": {
                "gateway": {},
                "private": {
                    "api_key": {
                        "source": "keychain",
                        "service": "arkai/cloud",
                        "account": "private",
                    }
                },
            },
        }
    )

    assert config["providers"]["gateway"]["api_key"] == {
        "source": "environment",
        "name": "TEAM_MODEL_KEY",
    }
    assert config["providers"]["gateway"]["api_key_environment_name"] == "TEAM_MODEL_KEY"
    assert config["providers"]["private"]["api_key"] == {
        "source": "keychain",
        "service": "arkai/cloud",
        "account": "private",
    }
    assert config["providers"]["private"]["api_key_environment_name"].startswith(
        "ARKAI_CUSTOM_API_KEY_PRIVATE_"
    )


def test_environment_source_resolves_to_runtime_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEAM_MODEL_KEY", "environment-secret")
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "environment", "name": "TEAM_MODEL_KEY"})
    )

    assert cloud_provider.custom_cloud_environment(config) == {
        "TEAM_MODEL_KEY": "environment-secret"
    }


def test_environment_source_requires_variable_value(monkeypatch) -> None:
    monkeypatch.delenv("TEAM_MODEL_KEY", raising=False)
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "environment", "name": "TEAM_MODEL_KEY"})
    )

    with pytest.raises(RuntimeError, match="TEAM_MODEL_KEY is required"):
        cloud_provider.custom_cloud_environment(config)


@pytest.mark.parametrize(
    "backend_module",
    [
        "keyring.backends.macOS",
        "keyring.backends.Windows",
        "keyring.backends.SecretService",
        "keyring.backends.kwallet",
    ],
)
def test_keychain_source_resolves_through_native_keyring_backend(
    monkeypatch, backend_module: str
) -> None:
    backend_type = type("TestKeychain", (), {"__module__": backend_module})
    keyring = SimpleNamespace(
        get_keyring=lambda: backend_type(),
        get_password=lambda service, account: (
            "keychain-secret" if (service, account) == ("arkai/cloud", "gateway") else None
        ),
    )
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "keychain", "service": "arkai/cloud", "account": "gateway"})
    )
    environment_name = config["providers"]["gateway"]["api_key_environment_name"]

    assert cloud_provider.custom_cloud_environment(config) == {environment_name: "keychain-secret"}


def test_keychain_source_rejects_unsupported_backend(monkeypatch) -> None:
    backend_type = type("PlaintextKeyring", (), {"__module__": "keyrings.alt.file"})
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(get_keyring=lambda: backend_type(), get_password=lambda *_: "secret"),
    )
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "keychain", "service": "arkai/cloud", "account": "gateway"})
    )

    with pytest.raises(RuntimeError, match="Could not access the system keychain"):
        cloud_provider.custom_cloud_environment(config)


def test_keychain_source_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "keychain", "service": "arkai/cloud", "account": "gateway"})
    )

    with pytest.raises(RuntimeError, match="optional keyring dependency"):
        cloud_provider.custom_cloud_environment(config)


def test_keychain_source_reports_missing_item_without_exposing_secret(monkeypatch) -> None:
    backend_type = type("TestKeychain", (), {"__module__": "keyring.backends.Windows"})
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(get_keyring=lambda: backend_type(), get_password=lambda *_: None),
    )
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "keychain", "service": "arkai/cloud", "account": "gateway"})
    )

    with pytest.raises(RuntimeError, match="No API key found in the system keychain"):
        cloud_provider.custom_cloud_environment(config)


@pytest.mark.parametrize(
    "api_key",
    [
        {"source": "environment", "name": "not-valid"},
        {"source": "environment", "name": "TOKEN", "extra": "value"},
        {"source": "keychain", "service": "arkai", "account": ""},
        {"source": "keychain", "service": "arkai", "account": "gateway", "name": "TOKEN"},
        {"source": "unsupported", "name": "TOKEN"},
    ],
)
def test_rejects_invalid_api_key_source(api_key: dict) -> None:
    with pytest.raises(ValueError, match="api_key"):
        cloud_provider.normalize_custom_cloud_config(_base_config(api_key))


def test_rejects_legacy_api_key_env_configuration() -> None:
    config = _base_config()
    config["api_key_env"] = "TEAM_MODEL_KEY"

    with pytest.raises(ValueError, match="api_key_env was replaced"):
        cloud_provider.normalize_custom_cloud_config(config)


def test_shared_source_can_be_inherited_by_multiple_providers() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "api_key": {"source": "environment", "name": "TEAM_MODEL_KEY"},
            "providers": {"gateway": {}, "anthropic": {}},
            "default_provider": "gateway",
        }
    )

    assert config["providers"]["gateway"]["api_key_environment_name"] == "TEAM_MODEL_KEY"
    assert config["providers"]["anthropic"]["api_key_environment_name"] == "TEAM_MODEL_KEY"
