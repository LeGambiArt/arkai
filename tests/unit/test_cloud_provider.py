"""Tests for custom cloud provider configuration and credential resolution."""

import json
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


def test_openai_responses_protocol_is_supported() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "protocol": "openai-responses",
            "default_provider": "gateway",
            "providers": {"gateway": {"default_model": "org/model"}},
        }
    )
    assert config["providers"]["gateway"]["protocol"] == "openai-responses"


def test_openai_provider_id_infers_responses_protocol() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config()
        | {"providers": {"openai": {"default_model": "gpt-model"}}, "default_provider": "openai"}
    )
    assert config["providers"]["openai"]["protocol"] == "openai-responses"


def test_openai_provider_id_keeps_explicit_completions_protocol() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config()
        | {
            "providers": {
                "openai": {"default_model": "gpt-model", "protocol": "openai-completions"}
            },
            "default_provider": "openai",
        }
    )
    assert config["providers"]["openai"]["protocol"] == "openai-completions"


def test_model_protocol_overrides_provider_protocol() -> None:
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config()
        | {
            "protocol": "openai-completions",
            "providers": {
                "gateway": {
                    "default_model": "org/model",
                    "models": [
                        {"id": "org/model", "protocol": "openai-responses"},
                        {"id": "org/other"},
                    ],
                }
            },
        }
    )
    models = config["providers"]["gateway"]["models"]
    assert models[0]["protocol"] == "openai-responses"
    assert "protocol" not in models[1]
    provider = config["providers"]["gateway"]
    assert cloud_provider.custom_cloud_model_protocol(provider, models[0]) == "openai-responses"
    assert cloud_provider.custom_cloud_model_protocol(provider, models[1]) == "openai-completions"


def test_model_protocol_defaults_to_openai_completions_without_root_protocol() -> None:
    config = cloud_provider.normalize_custom_cloud_config(_base_config())
    provider = config["providers"]["gateway"]
    assert provider["protocol"] == "openai-completions"
    assert cloud_provider.custom_cloud_model_protocol(provider, provider["models"][0]) == (
        "openai-completions"
    )


def test_rejects_unknown_model_protocol() -> None:
    with pytest.raises(ValueError, match=r"models\[0\].protocol"):
        cloud_provider.normalize_custom_cloud_config(
            _base_config()
            | {
                "providers": {
                    "gateway": {
                        "default_model": "org/model",
                        "models": [{"id": "org/model", "protocol": "unsupported"}],
                    }
                }
            }
        )


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


@pytest.mark.parametrize(
    "backend_module",
    ["keyring.backends.macOS", "keyring.backends.SecretService"],
)
def test_keychain_bundle_resolves_multiple_provider_keys_with_one_keyring_read(
    monkeypatch, backend_module: str
) -> None:
    """A single OS-keyring bundle supplies all configured custom-provider keys."""
    backend_type = type("TestKeychain", (), {"__module__": backend_module})
    bundle_reference = (
        cloud_provider._KEYCHAIN_BUNDLE_SERVICE,
        cloud_provider._KEYCHAIN_BUNDLE_ACCOUNT,
    )
    bundle = json.dumps(
        {
            "version": 1,
            "credentials": [
                {"service": "arkai/cloud", "account": "gateway", "key": "gateway-secret"},
                {"service": "arkai/cloud", "account": "private", "key": "private-secret"},
            ],
        }
    )
    calls = []

    def get_password(service, account):
        calls.append((service, account))
        return bundle if (service, account) == bundle_reference else None

    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(get_keyring=lambda: backend_type(), get_password=get_password),
    )
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "default_provider": "gateway",
            "providers": {
                "gateway": {
                    "api_key": {
                        "source": "keychain",
                        "service": "arkai/cloud",
                        "account": "gateway",
                    }
                },
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

    environment = cloud_provider.custom_cloud_environment(config)

    assert environment == {
        config["providers"]["gateway"]["api_key_environment_name"]: "gateway-secret",
        config["providers"]["private"]["api_key_environment_name"]: "private-secret",
    }
    assert calls == [bundle_reference]


def test_keychain_cache_reads_duplicate_legacy_entry_once(monkeypatch) -> None:
    backend_type = type("TestKeychain", (), {"__module__": "keyring.backends.macos"})
    bundle_reference = (
        cloud_provider._KEYCHAIN_BUNDLE_SERVICE,
        cloud_provider._KEYCHAIN_BUNDLE_ACCOUNT,
    )
    legacy_reference = ("arkai/cloud", "shared")
    calls = []

    def get_password(service, account):
        calls.append((service, account))
        if (service, account) == legacy_reference:
            return "shared-secret"
        return None

    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(get_keyring=lambda: backend_type(), get_password=get_password),
    )
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "api_key": {"source": "keychain", "service": "arkai/cloud", "account": "shared"},
            "default_provider": "gateway",
            "providers": {"gateway": {}, "private": {}},
        }
    )

    environment = cloud_provider.custom_cloud_environment(config)

    assert set(environment.values()) == {"shared-secret"}
    assert calls == [bundle_reference, legacy_reference]


def test_keychain_bundle_rejects_invalid_json_without_exposing_contents(monkeypatch) -> None:
    backend_type = type("TestKeychain", (), {"__module__": "keyring.backends.macos"})
    secret = "do-not-include-this-secret"
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(
            get_keyring=lambda: backend_type(),
            get_password=lambda *_: secret,
        ),
    )
    config = cloud_provider.normalize_custom_cloud_config(
        _base_config({"source": "keychain", "service": "arkai/cloud", "account": "gateway"})
    )

    with pytest.raises(RuntimeError, match="credential bundle is invalid") as error:
        cloud_provider.custom_cloud_environment(config)

    assert secret not in str(error.value)


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


def test_custom_cloud_models_default_context_window() -> None:
    """Models without an explicit context window get a cloud-sized default."""
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "default_provider": "gateway",
            "providers": {
                "gateway": {
                    "default_model": "org/model-one",
                    "models": [
                        {"id": "org/model-one", "context_window": 65536},
                        {"id": "org/model-two"},
                    ],
                }
            },
        }
    )

    models = cloud_provider.custom_cloud_models(config["providers"]["gateway"])

    assert [model["context_window"] for model in models] == [65536, 262144]
    assert (
        cloud_provider.custom_cloud_models(config["providers"]["gateway"], "org/model-three")[-1][
            "context_window"
        ]
        == 262144
    )


def test_model_metadata_accepts_size_unit_strings() -> None:
    """Context and output limits accept K/M size units and normalize to token counts."""
    config = cloud_provider.normalize_custom_cloud_config(
        {
            "base_url": "https://gateway.example/v1",
            "default_provider": "gateway",
            "providers": {
                "gateway": {
                    "models": [
                        {
                            "id": "org/model-one",
                            "context_window": "64k",
                            "max_output_tokens": "8K",
                        },
                        {
                            "id": "org/model-two",
                            "context_window": "1M",
                            "max_output_tokens": " 1m ",
                        },
                    ],
                }
            },
        }
    )

    models = config["providers"]["gateway"]["models"]

    assert [(model["context_window"], model["max_output_tokens"]) for model in models] == [
        (65536, 8192),
        (1048576, 1048576),
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        {"context_window": "banana"},
        {"context_window": "1.5m"},
        {"context_window": True},
        {"context_window": -1},
        {"context_window": 0},
        {"max_output_tokens": "50x"},
        {"max_output_tokens": 3.5},
        {"max_output_tokens": False},
    ],
)
def test_rejects_invalid_model_size_metadata(metadata: dict) -> None:
    """Size-unit fields only accept positive integers or valid K/M strings."""
    with pytest.raises(ValueError, match="context_window|max_output_tokens"):
        cloud_provider.normalize_custom_cloud_config(
            {
                "base_url": "https://gateway.example/v1",
                "providers": {"gateway": {"models": [{"id": "org/model", **metadata}]}},
            }
        )
