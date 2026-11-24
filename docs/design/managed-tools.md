# Design: managed tools

## Scope

Arkai's managed-tools feature installs and updates a limited set of
third-party tools under Arkai-owned directories. It currently supports
Pi, OMP, OpenCode, and Strata. It does not attempt to manage arbitrary
system packages or replace global installations.

## Installation model

The CLI delegates Pi, OMP, and OpenCode installation to their adapter
modules and uses Strata's official installer. Managed executables are
resolved from the Arkai-owned roots rather than from a modified global
`PATH`. Arkai uses OpenCode's v2 installer (`https://opencode.ai/v2/install`);
the plain `/install` URL serves the OpenCode 1 line only, which refuses
versions ≥ 2 and would pin old releases. OpenCode's installer hardcodes
`$HOME/.opencode/bin`, so Arkai redirects HOME into the managed root for the
installer and passes `--no-modify-path` to keep the redirected home's shell
configs untouched.
Install and update operations request confirmation and execute through the
configured sandbox with a constrained set of host utilities and runtime
paths. When the configured Arapuca executable cannot be resolved, Arkai
warns that the operation will run with no sandbox and executes the same
commands on the host, with their writable locations still pointed at the
Arkai-managed directories. Removal takes the same path: it warns and
runs `rm` on the host when Arapuca is unavailable, still limited to
Arkai-owned files.

Before writing, the manager verifies that managed roots and known entries
are not symlinks, uses private directory permissions, and refuses to
overwrite unmanaged executables in the managed location. Removal is
limited to Arkai-owned files and leaves separate installations and
non-owned data alone.

## Implementation map

- Registry, ownership checks, and Strata installer: `arkai/tools.py`
- Pi adapter/installer: `arkai/agent_pi.py`
- OMP adapter/installer: `arkai/agent_omp.py`
- OpenCode adapter: `arkai/agent_opencode.py`
- CLI behavior: `features/tools.feature`
- Unit coverage: `tests/unit/test_tools.py`,
  `tests/unit/test_agent_install.py`
- User instructions: [Managed tools](../guides/tools.md)
