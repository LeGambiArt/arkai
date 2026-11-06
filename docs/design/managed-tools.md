# Design: managed tools

## Scope

Arkai's managed-tools feature installs and updates a limited set of
third-party tools under Arkai-owned directories. It currently supports
Pi, OMP, and Strata. It does not attempt to manage arbitrary system
packages or replace global installations.

## Installation model

The CLI delegates Pi and OMP installation to their adapter modules and
uses Strata's official installer. Managed executables are resolved from
the Arkai-owned roots rather than from a modified global `PATH`. Install
and update operations request confirmation and execute through the
configured sandbox with a constrained set of host utilities and runtime
paths.

Before writing, the manager verifies that managed roots and known entries
are not symlinks, uses private directory permissions, and refuses to
overwrite unmanaged executables in the managed location. Removal is
limited to Arkai-owned files and leaves separate installations and
non-owned data alone.

## Implementation map

- Registry, ownership checks, and Strata installer: `arkai/tools.py`
- Pi adapter/installer: `arkai/agent_pi.py`
- OMP adapter/installer: `arkai/agent_omp.py`
- CLI behavior: `features/tools.feature`
- Unit coverage: `tests/unit/test_tools.py`,
  `tests/unit/test_agent_install.py`
- User instructions: [Managed tools](../guides/tools.md)
