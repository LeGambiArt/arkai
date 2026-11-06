# Design: model management

## Scope

Model management provides a common CLI for downloading, listing,
resolving, removing, converting, and (for supported stores) pruning
locally available models. The model reference and storage behavior depend
on the selected provider or inference backend.

## Storage paths

- Provider downloads are cached by `arkai.providers`; local GGUF files
  are stored under Arkai's model data directory.
- Splash selections use Splash's native metadata and installation layout.
  Arkai's bridge calls the supported installer interface and returns a
  validated inventory rather than treating native selections as GGUF
  files.
- Git/provider caches and Splash's Hub/weight caches are separate stores.

## Safety properties

Listing and resolution do not start inference or download a model. Splash
removal unlinks the selected installation and its pins while preserving
shared source files. Pruning is limited to recognized, unreferenced data
in the default Arkai-owned caches; custom or external caches are not
pruned. A dry-run is available to preview candidate paths.

Conversion is a separate GGUF operation for source models that require it,
it is not part of Splash's native install flow.

## Implementation map

- CLI and common model operations: `arkai/model.py`
- Provider cache: `arkai/providers.py`
- Splash store and inventory: `arkai/model_splash.py`,
  `arkai/model_store.py`, `arkai/splash_bridge.py`
- Unit/BDD coverage: `tests/unit/test_model.py`,
  `tests/unit/test_providers.py`, `tests/unit/test_splash_models.py`,
  `features/model.feature`
- User instructions: [Models and
inference](../guides/models-and-inference.md)
