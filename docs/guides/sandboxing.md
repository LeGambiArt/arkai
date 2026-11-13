# Sandboxing

Arkai wraps agent execution with `arapuca` by default. The current
sandbox integration uses filesystem mounts and resource limits; on Linux
it restricts network access to the local services Arkai needs, while
macOS uses Arapuca's baseline profile. Review the [sandbox
design](../design/sandboxing.md) for platform limitations before relying
on it as a security boundary.

On Linux, Arapuca's network rules are grant-based: every permitted
endpoint (`--allow-host`) or listener (`--allow-network-listeners`)
activates an isolated network namespace with no direct egress, and
HTTPS reaches only the granted hosts through a CONNECT proxy. Because
Arapuca rejects `--deny-network` combined with either grant, Arkai
requests deny mode only when a run has no network grants at all; a
granted run keeps the namespace isolation without the deny flag.

Sandboxed OpenCode requests Arapuca's network-listener capability for its
private local server. On Linux those listeners stay inside the isolated
network namespace; on macOS, Seatbelt cannot reliably restrict listeners
to loopback, so OpenCode may bind to a non-loopback interface. Use
sandboxed OpenCode only on trusted networks if this limitation is
acceptable. Sandboxed OpenCode also skips project-level OpenCode
configuration and rules; Arkai provides the per-run configuration instead.

Install `arapuca` separately and configure `sandbox.path` if it is not on
`PATH`. Agent launches accept `--sandbox PROFILE`, `--volume PATH`,
`--env KEY=VALUE`, and `--cwd PATH` overrides. `--no-sandbox` bypasses
the boundary for that run; use it only when you accept the risk.

Arkai supplies OMP with a private one-shot config overlay that disables
`spelling.typoDetection` and `spelling.autocomplete`, avoiding Apple's
system text-checking service without granting additional XPC access.

Sandbox profiles live under `sandbox.profiles` in the user or project
YAML configuration. They can define memory, CPU and process limits,
timeout, read-only or writable volume mounts, and environment values. A
profile inherits the root `sandbox.path` unless it sets its own path; other
omitted profile values use Arkai's built-in defaults. Profiles created with
`arkai sandbox create` start from root defaults unless `--from` is
supplied. Choose an active profile and inspect its effective settings
with:

```bash
arkai sandbox create restricted --memory 1024 --cpus 1 --pids 128
arkai sandbox active restricted
arkai sandbox list
arkai sandbox show active
arkai sandbox active
```

The final command clears the active profile and returns to root sandbox
settings. Other operations include `sandbox delete`, `sandbox
set-default`, and `sandbox show default`. See `arkai sandbox --help` for
all options.
