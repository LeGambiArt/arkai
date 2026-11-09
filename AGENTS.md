- Never create commits unless the user explicitly asks for a commit in the conversation.
- Follow the repository's [contribution guidelines](CONTRIBUTING.md).
- Add or update unit tests under `tests/unit/` for correctness, including negative cases and expected failures.
- For any change to CLI options or configuration behavior, also add or update `behave` BDD scenarios under `features/` that exercise the user-visible behavior.
- Add every new configuration key, its default, and a brief description to [`docs/arkai-full-config.yaml`](docs/arkai-full-config.yaml).

## Release and version changes

- Gather the commit message body for the release notes by running `contrib/changelog`, which groups the commits since the latest release into Highlights, Bug Fixes, Documentation, and Internal Changes sections; review and adjust the result to describe the new features and fixes accurately.
- Keep the release version identical in `arkai/__init__.py` (`__version__`) and `contrib/arkai.spec` (`Version:`); update both in the same change. `pyproject.toml` derives the version from `arkai.__version__` at build time, so it needs no edit.
- Run `./ci/check.sh` and build both distributions with `uv build --python 3.14` before presenting the release for review. Report any checks that could not run rather than implying they passed.
- Propose the exact release commit message in the conversation. A human must explicitly approve that exact message before any release/version commit is created; general approval to prepare a release is not enough. After approval, create the release commit with `git commit -s`.
- The publishing workflow is `.github/workflows/release.yml`; publishing a GitHub Release triggers the PyPI upload. It uses Trusted Publishing (OIDC), so do not add a PyPI API token. The PyPI publisher and GitHub environment must match project `arkai`, owner `LeGambiArt`, repository `arkai`, workflow `release.yml`, and environment `pypi`.
