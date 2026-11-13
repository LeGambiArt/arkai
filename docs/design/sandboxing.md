# Design: sandboxing

## Scope

Arkai uses Arapuca to wrap supported agent processes and apply a selected
resource profile, filesystem mounts, and environment. Arkai owns profile
management and launch argument construction; Arapuca enforces the
OS-level boundary.

## Profile resolution

Sandbox defaults and named profiles are read from layered Arkai YAML. A
profile supplies resource limits and optional mount/environment values.
Per-run CLI mounts and environment values are merged with the selected
profile; CLI values take precedence for duplicate environment names.
Volume paths are expanded and validated before launch.

Agent startup resolves the sandbox executable before starting services so
a missing runtime fails early. `--no-sandbox` bypasses the wrapper for
that run.

## Platform behavior and limitations

The documented Linux profile restricts filesystem visibility to mounted
paths and limits network access to required local services. Network
control is grant-based: `--allow-host` and `--allow-network-listeners`
each activate an isolated network namespace whose only egress is the
CONNECT proxy for granted hosts, and Arapuca rejects combining either
grant with `--deny-network`. Arkai therefore emits `--deny-network`
only for runs without any network grant. macOS uses Arapuca's baseline
profile; it does not provide the same network denial behavior, and
listener isolation has platform limitations. Do not treat the sandbox as
a stronger boundary than the host platform provides.

## Implementation map

- Profile CRUD and validation: `arkai/sandbox.py`
- Launch command and profile merge: `arkai/agent.py`
- Shared path/environment checks: `arkai/utils.py`
- Unit/BDD coverage: `tests/unit/test_sandbox.py`,
  `features/sandbox.feature`
- User instructions: [Sandboxing](../guides/sandboxing.md)
