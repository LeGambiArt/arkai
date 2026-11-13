# Design: configuration

## Scope

The configuration feature loads, layers, validates, and resolves Arkai's
YAML settings for CLI commands. It is intentionally a small Python
implementation rather than a separate schema service.

## Loading and precedence

`arkai.config.load_config` deep-copies built-in defaults, merges the user
file at `~/.config/arkai/arkai.yaml`, then merges `.arkai.yaml` in the
selected project directory. Merge behavior is provided by
`arkai.utils.merge_configs`. Command handlers apply their supported CLI
overrides after loading.

## Validation and resolution

Validation checks agent and backend names, model/profile shape, sampling
values, ports, RAG chunk settings, context size, and sandbox volume and
environment declarations. Custom cloud settings are validated by
`arkai.cloud_provider`. Validation can allow a missing model when a
healthy inference service is already available; otherwise local inference
startup requires a model.

`resolve_model` selects the configured profile and merges profile
sampling values over inference-level defaults. Commands that accept a
model override apply it before validation.

## Interactive field editor

The interactive field editor (`arkai/config_editor.py`) is schema-driven: a
`Field` registry (`FIELD_GROUPS`) lists every editable scalar key with its
kind, choices, and bounds. Collection editors for `inference.profiles`,
`sandbox.profiles`, and `agent.cloud.custom.providers` are declared via
`CollectionSpec` (title, prefix, member `Field`s, required member keys) in
`COLLECTIONS`; the field-list rows open a profile-list modal
(add/delete/edit), and a member modal per profile.
Profile-name validation reuses `arkai.sandbox._is_valid_profile_name` so the
TUI and the runtime share one rule; custom provider IDs are validated with
`arkai.cloud_provider._PROVIDER_ID_PATTERN`, protocols against the supported
protocol set, and `base_url` through `_normalize_base_url`; volume and
environment entries are validated with
`utils.validate_volumes`/`validate_environment`; member source tags reuse the
same layer precedence. Cloud provider members inherit the
`agent.cloud.custom` root defaults exactly as sandbox members inherit root
`sandbox` values, and the `api_key` member opens a source picker that writes
the same `{source, ...}` declaration the loader validates. A "Custom keys"
section lists every file key outside the managed registry; `a` adds a dotted
key with a YAML-parsed value stored without schema validation, so model- or
plugin-specific settings round-trip untouched. Keys inside profile
collections never appear there: each profile member modal lists the
profile's unmanaged keys (via `_profile_custom_keys`, which skips the
member keys and modal extras declared in `collection_extra_keys`) below its
member rows and offers the same add/edit/delete, with
`validate_profile_custom_key` rejecting member-key collisions. The curses UI is
isolated in `run_editor` behind a pure-logic/pure-UI split: `cmd_config_edit`
and `cmd_config_init --tui` both drive the same editor (edit loads an
existing target dict; init passes the seed dict and writes only on save),
and file writes happen only in those two command handlers. If
`validate_config` gains new scalar constraints, the registry entries must be
updated in the same change so the editor cannot write values validation
would reject.

## Evolution rules

- Keep defaults and validation in `arkai/config.py` aligned with the
  example configuration and the user-facing configuration guide. Every new
  configuration key must be documented with its default in
  `docs/arkai-full-config.yaml`.
- CLI options remain the source of truth for per-command overrides; avoid
  claiming that every option is a persistent YAML setting.
- Add or update unit tests for normalization and validation edge cases.
  Add Behave scenarios for user-visible CLI and configuration behavior.
- Unknown keys are not currently rejected by a strict external schema, so
  examples should use only keys consumed by the implementation.

## Implementation map

- Loading, defaults, validation, and model profile resolution:
  `arkai/config.py`
- Interactive field editor: `arkai/config_editor.py`
- YAML merge and filesystem paths: `arkai/utils.py`
- Unit coverage: `tests/unit/test_config.py`, `tests/unit/test_config_editor.py`
- CLI behavior: `features/config.feature`
- User instructions: [Configuration](../guides/configuration.md)
