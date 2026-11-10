# Contributing to Arkai

Thanks for helping improve Arkai. For substantial changes, open an issue
or discussion first so the expected behavior and scope can be agreed on.
Keep pull requests focused, describe the user-visible effect, and include
relevant tests and documentation updates.

## Development setup

Arkai requires Python 3.10+. From a checkout, create a virtual
environment and install the development and optional RAG dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,rag]'
```

The RAG extra includes heavyweight embedding dependencies. If you do not
need RAG, install `.[dev]` instead.

## Tests and checks

Run all configured checks before submitting a change:

```bash
./ci/check.sh
```

The script also supports running checks individually:

```bash
./ci/check.sh format
./ci/check.sh lint
./ci/check.sh type
./ci/check.sh unit
./ci/check.sh bdd
./ci/check.sh coverage
```

Add or update unit tests under `tests/unit/` for implementation
correctness, including invalid input, negative paths, and expected
failures. Changes to user-visible CLI behavior or configuration behavior
must also have Behave BDD coverage under `features/`. Keep those
scenarios focused on observable behavior, and keep unit tests for
lower-level logic and edge cases.

## Documentation

Keep the README focused on what Arkai does, installation, and first use.
Put advanced usage in `docs/guides/`, feature architecture and
implementation in a single-feature document under `docs/design/`, and
contributor workflow here.  Update examples and links when changing CLI
options or configuration behavior.  Every new configuration key must also
be added to [docs/arkai-full-config.yaml](docs/arkai-full-config.yaml),
with its default
and a brief description.

## Pull requests

- Explain the problem and the behavior changed.
- List the checks you ran and any that could not be run.
- Call out compatibility changes, external tool requirements, and
  security implications.
- Do not include generated files, caches, local credentials, or unrelated
  changes.

## Releases

Set the package version in `arkai/__init__.py` (`__version__`) and keep
`contrib/arkai.spec` (`Version:`) identical in the same change; the version
in `pyproject.toml` is derived from `arkai.__version__` at build time. Ensure
checks pass before publishing a GitHub release. Use `contrib/changelog` to
gather the release notes from the commits since the latest release.
`.github/workflows/release.yml` builds the package and publishes it to PyPI
when a release is published.
Successful publishing also depends on the matching PyPI trusted-publisher and
GitHub `pypi` environment configuration.
